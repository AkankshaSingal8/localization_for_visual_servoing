#!/usr/bin/env bash
# EKF (DINOv2 + SAM2) visual servoing against a live xArm.
#
#   *** THIS SCRIPT WILL MOVE THE ROBOT ARM ***
#
# By default this is the full perception -> EKF -> PBVS -> xArm pipeline.
# The old "smoke" behaviour (perception only, --no-robot) is still
# available via `SMOKE=1 ./smoke_ekf.sh` — see the env overrides below.
# The filename is kept for continuity with PROGRESS.md / docs; despite
# the name it is no longer a pure smoke test.
#
# What will happen when you run this
# ----------------------------------
#   1. Preflight checks (below) run BEFORE Python starts so you fail
#      fast with a clear error instead of a silent robot.connect() miss.
#   2. Python opens the xArm at $ARM_IP (see env overrides).
#   3. Perception loop starts; the OpenCV window opens.
#   4. A startup calibration thread fires automatically and moves the
#      arm +8 mm in Y, measures optical flow, returns home, then repeats
#      in Z (~32 mm of total travel). Servo is AUTO-ENABLED at the end
#      of that calibration — no keypress required.
#   5. From that moment, the arm issues set_position() every VS_RATE
#      (0.3 s) and approaches the target at >= VS_APPROACH (3 mm/step,
#      ~10 mm/s) along robot +X.
#
# Keypresses in the OpenCV window
# -------------------------------
#   v   pause / resume servo (flip RobotController.enabled)
#   r   soft-reset the mask tracker  (keep anchor)
#   R   full reset of mask tracker + EKF
#   c   one-shot depth calibration from the current frame (assumes the
#       object is at --depth-cal-m metres; default 0.30)
#   q   quit
#
# Real-hardware pre-flight checklist — do every time
# --------------------------------------------------
#   [ ] xArm powered on, READY indicator lit, no error state.
#   [ ] At least ~50 mm of free workspace around the arm in every
#       direction. Startup calibration uses 8 mm; servo accumulates.
#   [ ] Physical e-stop within reach.
#   [ ] DISPLAY set so the OpenCV window can open (you need v/q access).
#   [ ] Reference image matches the actual object on the desk.
#   [ ] The green mask is reliably on the target object. If unsure, run
#       once with SMOKE=1 first to verify perception, THEN re-run
#       without SMOKE to enable motion. Every bad mask drives the arm
#       toward the wrong 3-D point until the next re-detection.
#
# Environment overrides (all optional)
# ------------------------------------
#   ARM_IP=192.168.1.XXX   Override xArm IP used by the preflight ping.
#                          MUST match ROBOT_IP in
#                          FoundationModel/negative_weighing.py (default
#                          192.168.1.241). This script only uses ARM_IP
#                          for the preflight — it does not pass it into
#                          Python.
#   DEPTH_SCALE=0.000123   Explicit depth scale, passed to Python as
#                          --depth-scale <value>. Overrides the saved
#                          calibration below. Usually you do NOT need
#                          to set this — just run calibrate_depth.sh
#                          (or press 'c' in-session) once, and the
#                          value is persisted to
#                          experiments/depth_scale.json and auto-loaded
#                          on every subsequent run.
#   DEPTH_OFFSET=0.0       Additive depth offset in metres, passed to
#                          Python as --depth-offset <value>. Default 0.
#                          Usually leave unset; set it only if you have
#                          a known systematic bias in the ZED depth.
#   DEPTH_CAL_M=0.30       Distance (metres) the object will be at when
#                          you press 'c' to calibrate depth in-session.
#                          Passed as --depth-cal-m. Must match the
#                          physical distance you place the object at
#                          during calibrate_depth.sh / the in-session
#                          'c' press.
#   SMOKE=1                Force perception-only (adds --no-robot). No
#                          arm motion, no arm connection attempt, no
#                          preflight. Useful for re-running without
#                          editing this file.
#   SKIP_ARM_PRECHECK=1    Skip the ICMP ping to the xArm. Use only if
#                          the arm is reachable through a gateway that
#                          blocks ICMP; xarm SDK import is still
#                          checked.
#   CAM_TO_ROBOT=zed_forward
#                          Camera-to-robot rotation preset (see
#                          CAM_ROT_PRESETS in FoundationModel/dinov2_servo.py).
#                          Default is 'zed_forward' for the ZED Mini
#                          eye-in-hand mount on the xArm; change to
#                          'identity' only if the camera axes are
#                          physically aligned with the robot base.
#                          Wrong preset = robot moves along a wrong
#                          axis (classic symptom: arm goes UP instead
#                          of FORWARD when the object is too far,
#                          because camera +Z is mapped to robot +Z).
#
# Reference image note
# --------------------
#   experiments/input_image_transparent.png is the current target
#   reference (Amazon Basics facial tissue box, 1024x768 RGBA). It is
#   byte-identical to masked_objects/amazon_tissue_box.png — either path
#   works, we use the experiments/ copy so it travels with the
#   experiment scripts.

set -euo pipefail

source "$(dirname "$0")/_env.sh"

ARM_IP="${ARM_IP:-192.168.1.241}"
SMOKE="${SMOKE:-0}"

if [[ $# -ge 1 ]]; then
    REFERENCE="$1"
fi
REFERENCE="${REFERENCE:-experiments/input_image_transparent.png}"
OBJ_NAME="$(basename "${REFERENCE%.*}")"
OUT_PREFIX="${OBJ_NAME}_ekf"

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
fi

# ── Build the Python invocation ────────────────────────────────────────
CAM_TO_ROBOT="${CAM_TO_ROBOT:-zed_forward}"

PY_ARGS=(
    FoundationModel/dinov2_servo.py
    --reference "$REFERENCE"
    --out-prefix "$OUT_PREFIX"
    --mode ekf
    --cam-to-robot "$CAM_TO_ROBOT"
)
echo "[env] CAM_TO_ROBOT=$CAM_TO_ROBOT (physical ZED-Mini eye-in-hand default)."

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
              $DEPTH_STATE_FILE — depth scaler will run UNCALIBRATED
              (default_scale=0.001). EKF Z and the PBVS approach
              distance will be meaningless until you press 'c' in
              the window or run experiments/calibrate_depth.sh first.

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

echo "[config] REFERENCE  = $REFERENCE"
echo "[config] OUT_PREFIX = $OUT_PREFIX"
echo "[launch] python ${PY_ARGS[*]}"
exec python "${PY_ARGS[@]}"
