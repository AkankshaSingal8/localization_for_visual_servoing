#!/usr/bin/env bash
# FoundationPose (DINOv2 + SAM2 -> FP registration -> PBVS) visual
# servoing against a live xArm.
#
#   *** THIS SCRIPT WILL MOVE THE ROBOT ARM ***
#
# By default this is the full perception -> FP -> PBVS -> xArm pipeline.
# The old "smoke" behaviour (perception only, --no-robot) is still
# available via `SMOKE=1 ./smoke_fp.sh`. The filename is kept for
# continuity with PROGRESS.md / docs; despite the name it is no longer
# a pure smoke test.
#
# What will happen when you run this
# ----------------------------------
#   1. Preflight checks (below) run BEFORE Python starts so you fail
#      fast with a clear error instead of a silent robot.connect() miss.
#   2. Python opens the xArm at $ARM_IP (see env overrides).
#   3. Perception loop starts; the OpenCV window opens. First launch
#      takes ~20 s because CUDA kernels JIT-compile for FP.
#   4. Once FP has registered, the HUD shows "FP: REG" and a blue
#      crosshair tracks the object.
#   5. A startup calibration thread fires automatically (see smoke_ekf.sh
#      for details). Servo is AUTO-ENABLED at the end of that
#      calibration — no keypress required.
#   6. From that moment, the arm issues set_position() every VS_RATE
#      and approaches the target along robot +X.
#
# Keypresses in the OpenCV window
# -------------------------------
#   v   pause / resume servo (flip RobotController.enabled)
#   r   soft-reset the mask tracker  (keep anchor)
#   R   full reset of mask tracker + EKF
#   c   one-shot depth calibration from the current frame (assumes the
#       object is at --depth-cal-m metres; default 0.30). FP publishes
#       metric depth directly, but the DINOv2 fallback path still uses
#       the monocular depth scaler, so keep it calibrated.
#   q   quit
#
# Real-hardware pre-flight checklist — do every time
# --------------------------------------------------
#   [ ] xArm powered on, READY indicator lit, no error state.
#   [ ] At least ~50 mm of free workspace around the arm in every
#       direction. Startup calibration uses 8 mm; servo accumulates.
#   [ ] Physical e-stop within reach.
#   [ ] DISPLAY set so the OpenCV window can open (you need v/q access).
#   [ ] Reference image matches the actual object on the desk AND
#       FP_BOX matches the physical box dimensions (in metres, X Y Z).
#   [ ] The green mask is reliably on the target object. If unsure, run
#       once with SMOKE=1 first to verify perception, THEN re-run
#       without SMOKE to enable motion.
#
# Reference ↔ FP_BOX pairing (CRITICAL FOR FP)
# --------------------------------------------
#   FoundationPose estimates a 6-DoF pose using --fp-box as the object
#   extents. If REFERENCE points at object A but FP_BOX is object B's
#   dimensions, FP will either fail to register or lock onto a badly
#   scaled pose and the arm will track the wrong point in 3-D.
#
#   Known-good pairings:
#     - Cheez-It box:
#         REFERENCE=masked_objects/cheez_it_box.png
#         FP_BOX="0.19 0.06 0.22"
#     - Amazon Basics facial tissue box (aka input_image_transparent.png):
#         REFERENCE=experiments/input_image_transparent.png
#         FP_BOX="<measure it!>"   # typical: 0.23 0.11 0.11, verify!
#
# Environment overrides (all optional)
# ------------------------------------
#   REFERENCE=path/to/ref.png   Reference image for DINOv2 detection.
#                               Default: experiments/input_image_transparent.png
#                               (same object as smoke_ekf.sh).
#   FP_BOX="X Y Z"              Object extents in metres, space-separated.
#                               Default: "0.19 0.06 0.22" (Cheez-It).
#                               YOU MUST OVERRIDE THIS if REFERENCE is
#                               not the Cheez-It box — the script will
#                               warn if it detects a likely mismatch.
#   ARM_IP=192.168.1.XXX        Override xArm IP used by the preflight
#                               ping. MUST match ROBOT_IP in
#                               FoundationModel/negative_weighing.py
#                               (default 192.168.1.241).
#   DEPTH_SCALE=0.000123        Explicit depth scale for the DINOv2
#                               fallback path. Usually auto-loaded from
#                               experiments/depth_scale.json.
#   DEPTH_OFFSET=0.0            Additive depth offset in metres.
#   DEPTH_CAL_M=0.30            Distance (metres) for 'c' calibration.
#   SMOKE=1                     Force perception-only (adds --no-robot).
#                               No motion, no arm connection, no
#                               preflight ping.
#   SKIP_ARM_PRECHECK=1         Skip the ICMP ping to the xArm (xarm
#                               SDK import is still checked).

set -euo pipefail

source "$(dirname "$0")/_env.sh"

ARM_IP="${ARM_IP:-192.168.1.241}"
SMOKE="${SMOKE:-0}"
REFERENCE="${REFERENCE:-experiments/input_image_transparent.png}"
FP_BOX="${FP_BOX:-0.19 0.06 0.22}"

# ── Reference ↔ FP_BOX sanity check ───────────────────────────────────
# Cheap heuristic: if REFERENCE obviously isn't the Cheez-It file but
# FP_BOX is still the Cheez-It default, shout at the user. FP is too
# unforgiving of wrong extents to let this slide silently.
if [[ "$REFERENCE" != *"cheez_it"* && "$FP_BOX" == "0.19 0.06 0.22" ]]; then
    cat >&2 <<EOF

[config] WARNING: REFERENCE looks non-Cheez-It but FP_BOX is still the
                  Cheez-It default (0.19 0.06 0.22). FoundationPose
                  will produce a wrong-scale pose.

  REFERENCE = $REFERENCE
  FP_BOX    = $FP_BOX

  Measure the physical object's X/Y/Z in metres and re-run, e.g.:
      FP_BOX="0.23 0.11 0.11" ./experiments/smoke_fp.sh

  Proceeding in 3 s — Ctrl-C to abort.

EOF
    sleep 3
fi

# ── Preflight ──────────────────────────────────────────────────────────
if [ "$SMOKE" = "1" ]; then
    echo "[preflight] SMOKE=1 — skipping arm reachability + xarm-SDK"
    echo "            checks (no motion, --no-robot will be passed)."
else
    # 1. xArm reachable on the network.
    if [ "${SKIP_ARM_PRECHECK:-0}" = "1" ]; then
        echo "[preflight] SKIP_ARM_PRECHECK=1 — skipping ICMP check for $ARM_IP."
    else
        echo "[preflight] pinging xArm at $ARM_IP ..."
        if ping -c 1 -W 2 "$ARM_IP" >/dev/null 2>&1; then
            echo "[preflight] xArm is reachable."
        else
            cat >&2 <<EOF

[preflight] ERROR: cannot reach xArm at $ARM_IP.

  Possible causes:
    - xArm is powered off, in error state, or disconnected.
    - Wrong IP. ROBOT_IP in FoundationModel/negative_weighing.py is
      192.168.1.241. If that's wrong, fix it there or re-run with
      ARM_IP=<correct-ip>.
    - Network path blocks ICMP. Re-run with SKIP_ARM_PRECHECK=1.

  Aborting so the arm doesn't attempt to connect silently.

EOF
            exit 1
        fi
    fi

    # 2. xarm SDK importable in the active env. If not, robot.connect()
    #    would catch the ImportError and silently return False, which
    #    looks identical to a powered-off arm.
    if ! python -c "from xarm.wrapper import XArmAPI" 2>/dev/null; then
        cat >&2 <<'EOF'

[preflight] ERROR: xarm Python SDK is not importable in this env.

  Activate the foundationpose env and install:
    pip install xarm-python-sdk

  Or clone https://github.com/xArm-Developer/xArm-Python-SDK and run
  python setup.py install inside it.

EOF
        exit 1
    fi
    echo "[preflight] xarm SDK import OK."

    # 3. FoundationPose repo present. dinov2_servo.py will fail inside
    #    the CUDA-JIT path otherwise, which is much slower / noisier.
    if [ ! -d "$FOUNDATIONPOSE_ROOT" ]; then
        cat >&2 <<EOF

[preflight] ERROR: FOUNDATIONPOSE_ROOT=$FOUNDATIONPOSE_ROOT does not
                   exist. Clone FoundationPose there or export
                   FOUNDATIONPOSE_ROOT before running.

EOF
        exit 1
    fi
    echo "[preflight] FoundationPose repo found at $FOUNDATIONPOSE_ROOT."
fi

# ── Build the Python invocation ────────────────────────────────────────
# shellcheck disable=SC2206  # we WANT FP_BOX to word-split on spaces.
FP_BOX_ARR=( $FP_BOX )
if [ "${#FP_BOX_ARR[@]}" -ne 3 ]; then
    echo "[config] ERROR: FP_BOX must be 3 space-separated floats, got: '$FP_BOX'" >&2
    exit 1
fi

PY_ARGS=(
    FoundationModel/dinov2_servo.py
    --reference "$REFERENCE"
    --mode foundationpose
    --fp-box "${FP_BOX_ARR[@]}"
)

if [ "$SMOKE" = "1" ]; then
    PY_ARGS+=( --no-robot )
fi

DEPTH_STATE_FILE="experiments/depth_scale.json"

if [ -n "${DEPTH_SCALE:-}" ]; then
    echo "[env] DEPTH_SCALE=$DEPTH_SCALE — passing as --depth-scale (CLI override)."
    PY_ARGS+=( --depth-scale "$DEPTH_SCALE" )
elif [ -f "$DEPTH_STATE_FILE" ]; then
    SAVED_SCALE=$(python - <<PY
import json
try:
    with open("$DEPTH_STATE_FILE") as f:
        print(json.load(f)["scale"])
except Exception:
    pass
PY
)
    if [ -n "$SAVED_SCALE" ]; then
        echo "[env] Auto-loaded depth scale $SAVED_SCALE from $DEPTH_STATE_FILE."
        echo "      (delete that file or export DEPTH_SCALE=... to override)."
    else
        echo "[env] $DEPTH_STATE_FILE present but unreadable; Python will warn."
    fi
else
    cat <<WARN

[env] WARNING: no DEPTH_SCALE set and no saved calibration at
              $DEPTH_STATE_FILE. FP publishes its own metric depth
              from registered pose, so the main FP path is fine, but
              the DINOv2 fallback path will run UNCALIBRATED until
              you press 'c' or run experiments/calibrate_depth.sh.

WARN
fi

if [ -n "${DEPTH_OFFSET:-}" ]; then
    echo "[env] DEPTH_OFFSET=$DEPTH_OFFSET — passing as --depth-offset."
    PY_ARGS+=( --depth-offset "$DEPTH_OFFSET" )
fi

if [ -n "${DEPTH_CAL_M:-}" ]; then
    echo "[env] DEPTH_CAL_M=$DEPTH_CAL_M — passing as --depth-cal-m."
    PY_ARGS+=( --depth-cal-m "$DEPTH_CAL_M" )
fi

echo "[config] REFERENCE = $REFERENCE"
echo "[config] FP_BOX    = ${FP_BOX_ARR[*]}"
echo "[launch] python ${PY_ARGS[*]}"
exec python "${PY_ARGS[@]}"
