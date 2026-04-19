#!/usr/bin/env bash
# Step 0.4: depth calibration (run once per camera/setup change).
#
# Press 'c' in the OpenCV window once the mask has locked on. The
# resulting scale is written to experiments/depth_scale.json and is
# automatically picked up by smoke_ekf.sh on subsequent runs — no
# copy-paste required. Override only if you want a specific value:
#
#     DEPTH_SCALE=<number> ./experiments/smoke_ekf.sh
#
# No robot motion here — this script always passes --no-robot.
#
# Physical setup
# --------------
#   - Place the target object (the same physical object your reference
#     image corresponds to) exactly DEPTH_CAL_M metres in front of the
#     ZED Mini front face. Default is 0.30 m. Use a tape measure; an
#     error here propagates 1:1 into every subsequent servo run.
#   - Make sure the object is fully visible in the OpenCV window and
#     the green mask has locked on (~3 s) BEFORE pressing 'c'.
#
# Keypresses in the window
# ------------------------
#   c   capture the current depth. The scale is:
#         - logged as "Depth calibrated at Z=... -> scale=..."
#         - saved to experiments/depth_scale.json
#       You do NOT need to copy the number anywhere.
#   q   quit.
#
# Environment overrides
# ---------------------
#   REFERENCE=path/to/ref.png
#                          Reference image to lock the mask onto.
#                          Defaults to experiments/input_image_transparent.png
#                          (Amazon Basics facial tissue box) to match
#                          smoke_ekf.sh. Change it ONLY if you are
#                          calibrating against a different physical
#                          object.
#   DEPTH_CAL_M=0.30       Distance (metres) the object is placed at
#                          during this calibration. Passed through as
#                          --depth-cal-m. If you re-run smoke_ekf.sh
#                          later with a non-default DEPTH_CAL_M, that
#                          affects only the in-session 'c' key, NOT the
#                          DEPTH_SCALE value you export from here —
#                          the scale is distance-independent once
#                          captured correctly.

set -euo pipefail

source "$(dirname "$0")/_env.sh"

REFERENCE="${REFERENCE:-experiments/input_image_transparent.png}"
DEPTH_CAL_M="${DEPTH_CAL_M:-0.30}"

echo ""
echo "===================================================================="
echo "DEPTH CALIBRATION"
echo "  Reference image : $REFERENCE"
echo "  Calibration dist: ${DEPTH_CAL_M} m"
echo ""
echo "  1. Place the target object exactly ${DEPTH_CAL_M} m in front of"
echo "     the ZED Mini."
echo "  2. Wait for the green mask to lock on (~3 s)."
echo "  3. Click the OpenCV window, then press 'c'."
echo "     -> scale auto-saved to experiments/depth_scale.json."
echo "  4. Press 'q' to quit."
echo ""
echo "  Then just run:"
echo "     ./experiments/smoke_ekf.sh"
echo "  (smoke_ekf.sh auto-loads the saved scale.)"
echo "===================================================================="
echo ""

exec python FoundationModel/dinov2_servo.py \
    --reference "$REFERENCE" \
    --mode ekf \
    --depth-cal-m "$DEPTH_CAL_M" \
    --no-robot
