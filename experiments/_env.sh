# Shared environment setup for experiment-day scripts.
# Source this from other scripts in experiments/ so each one activates
# the foundationpose env the same way.
#
# Usage from another bash script:
#   source "$(dirname "$0")/_env.sh"

# Activate the foundationpose conda env
source /home/akanksha/miniconda3/etc/profile.d/conda.sh
conda activate foundationpose

# Strip the global PYTHONPATH (set in ~/.bashrc) that would otherwise
# leak GroundingDINO/Depth-Anything-V2 into every python process.
unset PYTHONPATH

# CUDA toolkit path (nvcc lives here, not in /usr/local/cuda/bin).
export PATH=/usr/local/cuda-12.4/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.4

# FoundationPose repo root (used by dinov2_servo.py's foundationpose mode)
export FOUNDATIONPOSE_ROOT=$HOME/FoundationPose

# Project root. All subsequent commands should run from here.
PROJECT_ROOT=/home/akanksha/repo/localization_for_visual_servoing
cd "$PROJECT_ROOT"

echo "[env] conda env:     $CONDA_DEFAULT_ENV"
echo "[env] python:        $(which python)"
echo "[env] cwd:           $(pwd)"
echo "[env] DISPLAY:       ${DISPLAY:-<unset>}"
echo "[env] FOUNDATIONPOSE_ROOT: $FOUNDATIONPOSE_ROOT"

if [ -z "${DISPLAY:-}" ]; then
    echo ""
    echo "[env] WARNING: DISPLAY is unset. OpenCV windows will not open."
    echo "              Run this from a terminal on the workstation"
    echo "              desktop, or ssh with -X forwarding."
    echo ""
fi
