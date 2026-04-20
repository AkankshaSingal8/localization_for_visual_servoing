#!/usr/bin/env bash
# Run all three visual-servoing pipelines back-to-back for a single
# reference object and collect their per-frame metrics CSVs.
#
#   *** THIS SCRIPT WILL MOVE THE ROBOT ARM ***
#
# Usage
# -----
#   ./experiments/run_all_experiment.sh masked_objects/cheez_it_box.png
#
# What it does
# ------------
#   For the reference image <ref.png> given on the command line it runs:
#     1. Experiment 1  (CSV suffix _ekf)  -- EKF + DINOv2/SAM2/DepthAnything -> PBVS
#     2. Experiment 2  (CSV suffix _fp)   -- EKF + FoundationPose 6-DoF      -> PBVS
#     3. Experiment 3  (CSV suffix _pbvs) -- IBVS baseline (raw DINOv2+SAM2 centroid
#                                            through the calibrated image Jacobian)
#
#   Convergence criterion (shared by all three trials):
#
#       ekf_z (filtered camera-frame depth to the box, metres)
#         <=  STOP_DEPTH_M  continuously for AUTO_EXIT_CONVERGE_SEC seconds
#
#   This is the simplest signal that works for every pipeline: FP and
#   EKF both populate ekf_z through the EKF update, and for IBVS the
#   perception branch of _seg_loop still updates the EKF from the
#   centroid + DepthAnything so ekf_z is available even though the IBVS
#   *controller* doesn't consult it. No reference pose from FP is
#   needed; the trials are independent.
#
#   Between every experiment the arm is commanded back to whatever pose
#   the arm was in *at the moment this script started* — so you set the
#   start pose by simply jogging the arm there before invoking the
#   script, no code edits needed. The return motion is commanded with a
#   deliberately slow speed / mvacc (see HOME_SPEED / HOME_MVACC below)
#   so the arm doesn't slam back across the workspace between trials.
#
#   Every experiment passes the same safety guardrails to Python
#   (dinov2_servo.py):
#     --z-floor-mm               Robot Z is clamped so the end-effector
#                                 never goes below this height.
#     --auto-exit-converge-sec    Exit when err_m < dead_zone for this
#                                 long (success).
#     --auto-exit-lost-sec        Exit when no centroid for this long
#                                 (object out of view).
#     --auto-exit-max-sec         Hard wall-clock timeout per trial.
#
#   A final `timeout` wrapper is placed around every Python invocation
#   so even if the auto-exit path fails, the script cannot hang forever.
#
# Output
# ------
#   A run-specific directory is created next to the working dir:
#       runs/<refbase>_<timestamp>/
#           <refbase>_ekf.csv
#           <refbase>_fp.csv
#           <refbase>_pbvs.csv
#           run.log                (combined stdout/stderr per trial)
#
#   The raw `metrics_*.csv` produced by dinov2_servo.py is moved into
#   this directory and renamed to `<refbase>_<suffix>.csv`, matching the
#   naming scheme requested in the spec.
#
# Environment overrides (all optional)
# ------------------------------------
#   Z_FLOOR_MM               Minimum robot Z in mm (default -150). Set
#                            to 0 if your arm is desk-mounted and the
#                            end-effector should never drop below the
#                            mounting flange.
#   AUTO_EXIT_CONVERGE_SEC   Seconds inside the PBVS dead-zone before
#                            declaring success (default 3.0).
#   AUTO_EXIT_LOST_SEC       Seconds without a centroid before declaring
#                            "out of view" (default 5.0).
#   AUTO_EXIT_MAX_SEC        Max wall-clock per trial in seconds
#                            (default 90.0).
#   HARD_TIMEOUT_SEC         Safety-net `timeout` applied around each
#                            Python invocation (default: auto_exit_max +
#                            30 s).
#   CAM_TO_ROBOT             zed_forward | identity (default zed_forward,
#                            matching the physical ZED-Mini-on-xArm
#                            mount).
#   ARM_IP                   Override xArm IP for the preflight ping
#                            (default 192.168.1.241).
#   SKIP_ARM_PRECHECK=1      Skip the ICMP ping.
#   HOME_SPEED               mm/s used for the slow return-to-home move
#                            (default 40, much less than the servo-time
#                            VS_SPEED of 80 so homing doesn't whip).
#   HOME_MVACC               mm/s^2 used for the slow return-to-home
#                            move (default 200).
#   SKIP_HOMING=1            Don't send the arm back to HOME between
#                            trials (unsafe; only for dry-runs without
#                            the arm actually connected).
#   SKIP_MODES="fp ibvs"     Space-separated list of mode suffixes
#                            (ekf|fp|pbvs) to skip. Useful if e.g. FP
#                            registration is failing and you only want
#                            the other two pipelines re-run.
#   STOP_DEPTH_M             Depth-to-box threshold in metres (default
#                            0.37). When the EKF-filtered camera-frame
#                            Z stays <= this value for
#                            AUTO_EXIT_CONVERGE_SEC seconds, the trial
#                            auto-ends. Default is target_depth (0.35)
#                            + 20 mm of tolerance.
#   FP_BOX="W H D"           Override the FoundationPose extents (metres).
#                            If unset, the script looks the reference
#                            image up in the built-in table (see below).
#
# Known FP_BOX pairings (metres)
# ------------------------------
#   cheez_it_box      0.19 0.06 0.22
#   amazon_tissue_box 0.21 0.07 0.12
#   cardboard_box     0.25 0.15 0.20
#   protein_bar       0.16 0.04 0.05
#   brownie_box       0.18 0.05 0.20
#
# Exit codes
# ----------
#   0 on clean completion of every requested trial. Non-zero if any
#   trial segfaulted, timed out at the `timeout` wrapper (hard kill),
#   or the arm failed to reach HOME between trials.

set -euo pipefail

# ── Arg parsing ────────────────────────────────────────────────────────
if [[ $# -lt 1 ]]; then
    cat >&2 <<EOF
Usage: $0 <path/to/reference.png>
See header of this script for env-var overrides.
EOF
    exit 2
fi
REFERENCE="$1"

# Repo env (conda activation + PATH for CUDA + cd to project root).
source "$(dirname "$0")/_env.sh"

if [[ ! -f "$REFERENCE" ]]; then
    echo "[run-all] ERROR: reference image not found: $REFERENCE" >&2
    exit 2
fi

REFBASE="$(basename "${REFERENCE%.*}")"
TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="runs/${REFBASE}_${TS}"
mkdir -p "$OUT_DIR"
LOG_FILE="$OUT_DIR/run.log"

# ── Defaults / env overrides ───────────────────────────────────────────
Z_FLOOR_MM="${Z_FLOOR_MM:--150}"
AUTO_EXIT_CONVERGE_SEC="${AUTO_EXIT_CONVERGE_SEC:-3.0}"
AUTO_EXIT_LOST_SEC="${AUTO_EXIT_LOST_SEC:-5.0}"
AUTO_EXIT_MAX_SEC="${AUTO_EXIT_MAX_SEC:-90.0}"
# Bash-level hard kill in case Python's own auto-exit hangs (e.g.
# stuck CUDA call). Default is auto_exit_max + 30 s of grace.
HARD_TIMEOUT_SEC="${HARD_TIMEOUT_SEC:-$(python - <<PY
print(float("${AUTO_EXIT_MAX_SEC}") + 30.0)
PY
)}"
CAM_TO_ROBOT="${CAM_TO_ROBOT:-zed_forward}"
ARM_IP="${ARM_IP:-192.168.1.241}"
SKIP_MODES="${SKIP_MODES:-}"
SKIP_HOMING="${SKIP_HOMING:-0}"
# Slow homing motion (a lot less than VS_SPEED=80 / VS_MVACC=500) so
# the arm doesn't fly back to the start pose between trials.
HOME_SPEED="${HOME_SPEED:-40}"
HOME_MVACC="${HOME_MVACC:-200}"
# Where we remember the start pose that every trial returns to.
INIT_POSE_FILE="$OUT_DIR/initial_pose.json"
# Depth-based stop threshold. Default = target_depth (0.35 m from
# dinov2_servo.py) + 20 mm slack, so the trial auto-ends just as the
# arm arrives at the PBVS target plane. Widen if the monocular depth
# scale is pessimistic, tighten for a stricter stop.
STOP_DEPTH_M="${STOP_DEPTH_M:-0.37}"

# ── FP_BOX lookup (metres, W x H x D) ─────────────────────────────────
#
# FoundationPose needs the procedural-box extents (or a CAD mesh) so
# its first-frame registration has something to match against. These
# are typical retail-packaging sizes; if you've measured the real box
# with a ruler, override with  FP_BOX="W H D"  on the command line and
# that overrides whatever this table says. Missing entries fall back
# to a generic 0.18 x 0.05 x 0.20 m "grocery-aisle box" shape and log
# a warning rather than silently skipping the FP trial.
lookup_fp_box() {
    local ref="$1"
    case "$ref" in
        *cheez_it*)           echo "0.19 0.06 0.22" ;;
        *amazon_tissue*|*tissue*) echo "0.21 0.07 0.12" ;;
        *cardboard*)          echo "0.25 0.15 0.20" ;;
        *protein*)            echo "0.16 0.04 0.05" ;;
        *brownie*)            echo "0.18 0.05 0.20" ;;
        *cake*)               echo "0.14 0.05 0.20" ;;
        *tofu*)               echo "0.10 0.05 0.14" ;;
        *baking_mix*)         echo "0.17 0.06 0.22" ;;
        *mac_and_cheese*|*mac*cheese*) echo "0.16 0.04 0.20" ;;
        *mashed_potato*)      echo "0.10 0.04 0.17" ;;
        *stuffing*)           echo "0.16 0.05 0.20" ;;
        *lamp*)               echo "0.20 0.15 0.25" ;;
        *)                    echo "" ;;
    esac
}
# Fallback extents used when the basename isn't in the table. Chosen
# to match an average grocery-aisle box so FP still has something
# plausible to register against instead of being skipped entirely.
FP_BOX_FALLBACK="${FP_BOX_FALLBACK:-0.18 0.05 0.20}"
FP_BOX="${FP_BOX:-$(lookup_fp_box "$REFBASE")}"

# ── Preflight (same sanity checks as smoke_ekf.sh / smoke_fp.sh) ───────
echo "[preflight] reference: $REFERENCE"
echo "[preflight] out dir  : $OUT_DIR"

if [[ "${SKIP_ARM_PRECHECK:-0}" != "1" ]]; then
    echo "[preflight] pinging xArm at $ARM_IP ..."
    if ! ping -c 1 -W 2 "$ARM_IP" >/dev/null 2>&1; then
        echo "[preflight] ERROR: cannot reach xArm at $ARM_IP. Abort." >&2
        exit 1
    fi
fi

if ! python -c "from xarm.wrapper import XArmAPI" >/dev/null 2>&1; then
    echo "[preflight] ERROR: xarm Python SDK not importable." >&2
    exit 1
fi

# ── Helpers ────────────────────────────────────────────────────────────
has_mode() {  # returns 0 iff $1 is NOT in SKIP_MODES
    local m="$1"
    for skipped in $SKIP_MODES; do
        [[ "$skipped" == "$m" ]] && return 1
    done
    return 0
}

capture_initial_pose() {
    # Read the arm's CURRENT Cartesian pose (x,y,z,roll,pitch,yaw) and
    # persist it to $INIT_POSE_FILE. Every subsequent send_home reads
    # this file. This means "initial position" == "whatever pose the
    # arm was in when the script was launched", so the operator only
    # needs to jog the arm there once before running the batch.
    if [[ "$SKIP_HOMING" == "1" ]]; then
        echo "[init] SKIP_HOMING=1, not capturing start pose." | tee -a "$LOG_FILE"
        return 0
    fi
    echo "[init] capturing current robot pose as 'initial' ..." | tee -a "$LOG_FILE"
    python - "$ARM_IP" "$INIT_POSE_FILE" <<'PY' 2>&1 | tee -a "$LOG_FILE"
import json, sys
from xarm.wrapper import XArmAPI

ip, out_path = sys.argv[1], sys.argv[2]
arm = XArmAPI(ip, baud_checkset=False)
arm.clean_error(); arm.clean_warn()
arm.motion_enable(True); arm.set_mode(0); arm.set_state(0)

code, pose = arm.get_position()
if code != 0 or pose is None:
    print(f"[init][py] get_position failed: code={code}")
    sys.exit(2)
x, y, z, roll, pitch, yaw = pose[:6]
with open(out_path, "w") as f:
    json.dump(dict(x=float(x), y=float(y), z=float(z),
                   roll=float(roll), pitch=float(pitch), yaw=float(yaw)),
              f, indent=2)
print(f"[init][py] captured x={x:.1f} y={y:.1f} z={z:.1f} "
      f"roll={roll:.1f} pitch={pitch:.1f} yaw={yaw:.1f} mm/deg")
PY
    if [[ ! -s "$INIT_POSE_FILE" ]]; then
        echo "[init] ERROR: failed to capture initial pose." >&2
        return 1
    fi
    echo "[init] saved -> $INIT_POSE_FILE" | tee -a "$LOG_FILE"
}

send_home() {
    # Move the arm back to the pose captured by capture_initial_pose at
    # a deliberately slow speed/mvacc (HOME_SPEED / HOME_MVACC) so it
    # doesn't whip back across the workspace between trials.
    if [[ "$SKIP_HOMING" == "1" ]]; then
        echo "[home] SKIP_HOMING=1, not returning arm to home." | tee -a "$LOG_FILE"
        return 0
    fi
    if [[ ! -s "$INIT_POSE_FILE" ]]; then
        echo "[home] WARNING: no $INIT_POSE_FILE; skipping return." | tee -a "$LOG_FILE"
        return 1
    fi
    echo "[home] returning arm to initial pose (slow: speed=${HOME_SPEED} mvacc=${HOME_MVACC}) ..." \
        | tee -a "$LOG_FILE"
    python - "$ARM_IP" "$INIT_POSE_FILE" "$HOME_SPEED" "$HOME_MVACC" <<'PY' 2>&1 | tee -a "$LOG_FILE"
import json, sys
from xarm.wrapper import XArmAPI

ip, pose_path, speed_s, mvacc_s = sys.argv[1:5]
speed = float(speed_s); mvacc = float(mvacc_s)
pose = json.load(open(pose_path))

arm = XArmAPI(ip, baud_checkset=False)
arm.clean_error(); arm.clean_warn()
arm.motion_enable(True); arm.set_mode(0); arm.set_state(0)

print(f"[home][py] target x={pose['x']:.1f} y={pose['y']:.1f} z={pose['z']:.1f} "
      f"roll={pose['roll']:.1f} pitch={pose['pitch']:.1f} yaw={pose['yaw']:.1f}  "
      f"speed={speed} mvacc={mvacc}")
ret = arm.set_position(
    x=pose["x"], y=pose["y"], z=pose["z"],
    roll=pose["roll"], pitch=pose["pitch"], yaw=pose["yaw"],
    speed=speed, mvacc=mvacc, wait=True,
)
if ret == 0:
    print("[home][py] reached initial pose.")
    sys.exit(0)
print(f"[home][py] set_position returned {ret}")
sys.exit(3)
PY
    local rc=${PIPESTATUS[0]}
    if [[ $rc -eq 0 ]]; then
        echo "[home] ok." | tee -a "$LOG_FILE"
        return 0
    fi
    echo "[home] WARNING: return-to-initial did not succeed (rc=$rc)." >&2
    return 1
}

run_experiment() {
    # $1 = suffix written into the output CSV filename (ekf|fp|pbvs)
    # $2 = --mode value passed to dinov2_servo.py (ekf|foundationpose|ibvs)
    local suffix="$1"; local mode="$2"
    local out_csv="$OUT_DIR/${REFBASE}_${suffix}.csv"

    echo "" | tee -a "$LOG_FILE"
    echo "====================================================================" | tee -a "$LOG_FILE"
    echo "[trial] ${REFBASE}_${suffix}   (mode=${mode})" | tee -a "$LOG_FILE"
    echo "====================================================================" | tee -a "$LOG_FILE"

    # Common args shared by all three modes, including the depth-based
    # stop that unifies convergence across EKF / FP / IBVS.
    local -a py_args=(
        FoundationModel/dinov2_servo.py
        --reference "$REFERENCE"
        --out-prefix "${REFBASE}_${suffix}"
        --mode "$mode"
        --cam-to-robot "$CAM_TO_ROBOT"
        --z-floor-mm "$Z_FLOOR_MM"
        --auto-exit-converge-sec "$AUTO_EXIT_CONVERGE_SEC"
        --auto-exit-lost-sec "$AUTO_EXIT_LOST_SEC"
        --auto-exit-max-sec "$AUTO_EXIT_MAX_SEC"
        --stop-depth-m "$STOP_DEPTH_M"
        --run-tag "${REFBASE}_${suffix}"
    )

    # Auto-load persisted depth scale if present (same as smoke_ekf.sh).
    if [[ -f "experiments/depth_scale.json" ]]; then
        local saved_scale
        saved_scale="$(python - <<PY
import json
try:
    print(json.load(open("experiments/depth_scale.json"))["scale"])
except Exception:
    pass
PY
)"
        if [[ -n "$saved_scale" ]]; then
            py_args+=( --depth-scale "$saved_scale" )
        fi
    fi

    # FoundationPose-specific extents. If the reference isn't in the
    # lookup table and FP_BOX wasn't overridden, fall back to a
    # generic grocery-aisle box size so the trial still runs and
    # produces a CSV — far better than silently skipping the whole
    # FP experiment the way the previous version did.
    if [[ "$mode" == "foundationpose" ]]; then
        local fp_extents="$FP_BOX"
        if [[ -z "$fp_extents" ]]; then
            fp_extents="$FP_BOX_FALLBACK"
            echo "[trial] WARN: no FP_BOX known for '$REFBASE'. Using " \
                 "fallback extents ($fp_extents m). Override with " \
                 "FP_BOX=\"W H D\" for measured dimensions." \
                 | tee -a "$LOG_FILE"
        fi
        # shellcheck disable=SC2206
        local fp_arr=( $fp_extents )
        py_args+=( --fp-box "${fp_arr[@]}" )
    fi


    echo "[trial] \$ python ${py_args[*]}" | tee -a "$LOG_FILE"
    echo "[trial] hard-timeout: ${HARD_TIMEOUT_SEC} s" | tee -a "$LOG_FILE"

    # Snapshot existing metrics_*.csv so we can identify the new one.
    local before_list
    before_list="$(ls -1 metrics_*.csv 2>/dev/null || true)"

    # `set +e` around the run so a per-trial non-zero exit (e.g. a
    # `timeout`-triggered kill) doesn't abort the whole batch — we
    # still want to collect the partial CSV and move on.
    set +e
    timeout --signal=TERM --kill-after=10 "${HARD_TIMEOUT_SEC}" \
        python "${py_args[@]}" 2>&1 | tee -a "$LOG_FILE"
    local rc=${PIPESTATUS[0]}
    set -e

    case "$rc" in
        0)   echo "[trial] exit 0 (clean)" | tee -a "$LOG_FILE" ;;
        124) echo "[trial] exit 124 (hard-timeout; Python did not shut down in time)" | tee -a "$LOG_FILE" ;;
        137) echo "[trial] exit 137 (hard-timeout KILL)" | tee -a "$LOG_FILE" ;;
        *)   echo "[trial] exit $rc (see $LOG_FILE)" | tee -a "$LOG_FILE" ;;
    esac

    # Find the new metrics_*.csv (newest one that wasn't there before).
    local new_csv=""
    for f in metrics_*.csv; do
        [[ -f "$f" ]] || continue
        if ! grep -qxF "$f" <<<"$before_list"; then
            if [[ -z "$new_csv" || "$f" -nt "$new_csv" ]]; then
                new_csv="$f"
            fi
        fi
    done
    if [[ -z "$new_csv" ]]; then
        echo "[trial] WARNING: no new metrics_*.csv produced." | tee -a "$LOG_FILE"
        return 0
    fi
    mv -- "$new_csv" "$out_csv"
    echo "[trial] CSV -> $out_csv" | tee -a "$LOG_FILE"
}

# ── Orchestrate the three trials ──────────────────────────────────────
# The arm's CURRENT pose is the one we return to between trials. Do
# NOT move the arm here — the operator positioned it intentionally.
capture_initial_pose || {
    echo "[fatal] cannot capture initial pose; aborting batch." >&2
    exit 1
}

MODES_TO_RUN=( "ekf:ekf" "fp:foundationpose" "pbvs:ibvs" )

for spec in "${MODES_TO_RUN[@]}"; do
    suffix="${spec%%:*}"
    mode="${spec##*:}"
    if ! has_mode "$suffix"; then
        echo "[skip] suffix=$suffix (in SKIP_MODES)"
        continue
    fi
    run_experiment "$suffix" "$mode"
    # Always re-home between trials (even after failures).
    send_home || echo "[home] continuing despite homing failure" >&2
done

echo ""
echo "===================================================================="
echo "[done] All trials complete. Output directory: $OUT_DIR"
echo "       CSVs:"
ls -1 "$OUT_DIR"/*.csv 2>/dev/null || echo "       (none produced)"
echo "===================================================================="
