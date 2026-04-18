# Creating the `foundationpose` Conda Environment

Step-by-step recipe for creating a fresh, isolated conda environment for
this project on the GPU workstation. This doc is **surgical**: it only
creates one new named environment and installs packages into that env.
Nothing in it touches your system Python, your base conda env, or any
other env you already have.

---

## Safety invariants (read first)

- `conda create -n foundationpose ...` creates a *new named environment*.
  It does not modify any existing env.
- `conda activate foundationpose` only affects the current terminal.
  Opening a new terminal gets you back to your base env / shell.
- Every `pip install` in this doc runs **inside the activated
  foundationpose env** — packages go into
  `~/miniconda3/envs/foundationpose/lib/...`, not into your system
  Python or any other env.
- If anything goes wrong, a single command tears the whole env down
  (see "Rollback / cleanup" at the bottom).
- The only files outside the env that we touch: one optional line
  appended to `~/.bashrc` in the final step. That line is also easy to
  remove.

**Before you start**, list your existing envs so you can prove we didn't
break anything later:

```bash
conda env list > /tmp/envs_before.txt
cat /tmp/envs_before.txt
```

Save this output. At the end of setup we'll diff against it to confirm
only `foundationpose` was added.

---

## Step 1: Create and activate the env

```bash
# Python 3.9 is what FoundationPose requires. Do not use 3.10+.
conda create -n foundationpose python=3.9 -y

# Activate it (this terminal only; your other terminals are unaffected)
conda activate foundationpose

# Sanity check — these should point inside the new env:
python --version                  # -> Python 3.9.x
which python                      # -> .../envs/foundationpose/bin/python
which pip                         # -> .../envs/foundationpose/bin/pip
```

If `which python` does not contain `/envs/foundationpose/` the env
isn't active; stop and fix it before installing anything.

---

## Step 2: Clone FoundationPose (if not already cloned)

```bash
cd ~
[ -d FoundationPose ] || git clone https://github.com/NVlabs/FoundationPose.git
export FOUNDATIONPOSE_ROOT=$HOME/FoundationPose
cd $FOUNDATIONPOSE_ROOT

# Verify
ls estimater.py && echo "FoundationPose repo OK"
```

---

## Step 3: Install CUDA-matched PyTorch first

Everything else depends on PyTorch, and PyTorch needs to match your
driver's CUDA version. Check your driver:

```bash
nvidia-smi | grep "CUDA Version"
# Typical: "CUDA Version: 12.1", "11.8", "12.4", "12.8"
```

### Picking the right wheel

NVIDIA drivers are **forward compatible** with PyTorch CUDA builds
older than the driver. So a driver reporting CUDA 12.8 can run any
PyTorch cu11x/cu121/cu124/cu128 build. You do NOT need a PyTorch wheel
that exactly matches the driver's CUDA version.

For this project we pin to **PyTorch 2.1.0 on cu121** because
FoundationPose, kaolin 0.15.0, and pytorch3d were all tested against
that combination. Using a newer PyTorch tends to cascade-break the
FP dependency chain.

| Driver CUDA | Recommended wheel | Index URL |
|---|---|---|
| 11.8 | torch 2.1.0 cu118 | `https://download.pytorch.org/whl/cu118` |
| 12.1 | torch 2.1.0 cu121 | `https://download.pytorch.org/whl/cu121` |
| 12.4 | torch 2.1.0 cu121 | `https://download.pytorch.org/whl/cu121` |
| **12.8** | **torch 2.1.0 cu121** | `https://download.pytorch.org/whl/cu121` |

The reason cu121 is the pick even for a cu12.8 driver: PyTorch's cu124
and cu128 wheels only exist for PyTorch 2.4+, which is too new for
FoundationPose's pinned deps.

### Install command (for CUDA 12.8 driver)

```bash
pip install torch==2.1.0 torchvision==0.16.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Verify PyTorch sees the GPU
python -c "
import torch
print('CUDA available:', torch.cuda.is_available())
print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')
print('PyTorch version:', torch.__version__)
print('PyTorch CUDA version:', torch.version.cuda)
"
```

**You must see `CUDA available: True` before proceeding.** Expected
output (driver 12.8 + wheel cu121):
```
CUDA available: True
Device: <your GPU name>
PyTorch version: 2.1.0+cu121
PyTorch CUDA version: 12.1
```

The `12.1` PyTorch version is correct — your 12.8 driver runs the 12.1
build via forward compatibility.

### ⚠️ If your env is Python 3.11 (or anything other than 3.9)

FoundationPose's dependency chain (specifically `kaolin==0.15.0`) has
prebuilt wheels only for Python 3.8-3.10. On Python 3.11 you will hit
"no matching distribution" when `pip install kaolin==0.15.0` runs in
Step 4.

You have three options:

**Option A (recommended): recreate the env with Python 3.9**

Fastest and most reliable. Your existing Python 3.11 env is still there
in other projects; this only recreates *this* specific env.

```bash
conda deactivate
conda env remove -n foundationpose -y
conda create -n foundationpose python=3.9 -y
conda activate foundationpose
# Now restart from Step 3
```

**Option B: keep Python 3.11 and use newer kaolin**

Kaolin 0.16+ supports Python 3.11 but requires PyTorch 2.2+, which
means you also need to bump PyTorch. Try:

```bash
pip install torch==2.4.0 torchvision==0.19.0 \
    --index-url https://download.pytorch.org/whl/cu121
# In Step 4 below, use kaolin==0.17.0 instead of 0.15.0
```

This sometimes works, sometimes doesn't — FoundationPose was not tested
against these versions. If FP fails to build or produces wrong poses,
fall back to Option A.

**Option C: skip FoundationPose entirely**

If you're only running Methods 1 and 2 (IBVS and EKF-DINOv2), you can
stay on Python 3.11 and skip Steps 4-6 (FoundationPose deps and build).
You'll still need the project deps in Step 7. This is a valid starting
point to get experiments going while the FP env is sorted.

---

## Step 4: FoundationPose's Python deps

```bash
cd $FOUNDATIONPOSE_ROOT

# Eigen via conda (cleanest — installs into the active env only)
conda install conda-forge::eigen=3.4.0 -y

# FoundationPose's own requirements.txt
pip install -r requirements.txt

# nvdiffrast (NVIDIA's differentiable rasterizer; FP's core dep)
pip install git+https://github.com/NVlabs/nvdiffrast.git

# Kaolin and pytorch3d (both picky about PyTorch version)
pip install kaolin==0.15.0
pip install pytorch3d
```

If `kaolin` or `pytorch3d` fails: they have to match the installed
PyTorch version. Copy the exact error and we'll pick a matching wheel.

---

## Step 5: Build FoundationPose's C++/CUDA extensions

```bash
# Make sure nvcc is on PATH (FoundationPose's build script needs it)
export PATH=/usr/local/cuda/bin:$PATH
nvcc --version    # should print a CUDA version

cd $FOUNDATIONPOSE_ROOT
bash build_all_conda.sh
```

This takes 3-10 minutes. If it finishes without errors you're past the
hardest part.

---

## Step 6: FoundationPose model weights

Download both weight directories into `$FOUNDATIONPOSE_ROOT/weights/`:

- `2023-10-28-18-33-37` (refiner)
- `2024-01-11-20-02-45` (scorer)

The download links are in FoundationPose's README (Google Drive). If
you have `gdown` and the Drive URL:

```bash
pip install gdown
mkdir -p $FOUNDATIONPOSE_ROOT/weights
cd $FOUNDATIONPOSE_ROOT/weights
gdown --folder <drive-folder-url>
ls $FOUNDATIONPOSE_ROOT/weights
# Should list both dated directories
```

If Drive won't cooperate, download to your laptop and `scp` them over.

---

## Step 7: Our project's Python deps

All of this goes into the same `foundationpose` env:

```bash
# Still in the activated foundationpose env
pip install opencv-python pyyaml trimesh xarm-python-sdk

# Verify our project's imports work
python -c "
import cv2, yaml, trimesh
from xarm.wrapper import XArmAPI
print('Project deps OK')
"
```

---

## Step 8: Install pyzed (ZED SDK Python wrapper)

The Python wrapper must match the system-installed ZED SDK. First check
whether the SDK is installed:

```bash
ls /usr/local/zed/ 2>/dev/null && echo "ZED SDK installed" \
    || echo "ZED SDK NOT installed — download from stereolabs.com first"
```

If the SDK is present:

```bash
# Install the matching pyzed wheel
cd /usr/local/zed
python get_python_api.py
# Installs pyzed into the ACTIVE env only — safe.

# Verify
python -c "import pyzed.sl as sl; print('pyzed version:', sl.Camera().get_sdk_version())"
```

If the SDK is not installed, get it from stereolabs.com for your Ubuntu
version, install the `.run` file at system level, then come back to
this step.

---

## Step 9: Project path and smoke test

Clone the project if it isn't already on this machine, and test
end-to-end:

```bash
[ -d ~/localization_for_visual_servoing ] \
    || git clone <your-project-url> ~/localization_for_visual_servoing

cd ~/localization_for_visual_servoing

# Import-only smoke test
python -c "
import sys
sys.path.insert(0, 'EKF')
sys.path.insert(0, 'FoundationModel')
from ekf_servo import PoseEKF, PBVSController, CameraIntrinsics
from foundationpose_wrapper import FoundationPoseWrapper, make_box_mesh
print('Project imports OK')
"

# DINOv2+SAM2 mode smoke test (no arm, no camera — uses any cam index)
python FoundationModel/dinov2_servo.py \
    --reference masked_objects/cheez_it_box.png \
    --mode ekf --no-robot --no-pyzed
# Press 'q' in the OpenCV window to quit

# FoundationPose mode smoke test (needs ZED connected)
python FoundationModel/dinov2_servo.py \
    --reference masked_objects/cheez_it_box.png \
    --mode foundationpose \
    --fp-box 0.19 0.06 0.22 \
    --no-robot
# Press 'q'
```

---

## Step 10: Confirm no other envs were touched

```bash
conda env list > /tmp/envs_after.txt
diff /tmp/envs_before.txt /tmp/envs_after.txt
```

The only difference should be one new line for `foundationpose`. If
anything else changed, something went wrong and we should talk.

---

## Step 11 (optional): Make activation convenient

Append this to your `~/.bashrc` so `fp-env` activates the env from any
new terminal:

```bash
cat >> ~/.bashrc << 'EOF'

# --- FoundationPose project (added by SETUP_CONDA_ENV.md) ---
export FOUNDATIONPOSE_ROOT=$HOME/FoundationPose
alias fp-env='conda activate foundationpose'
# -----------------------------------------------------------
EOF

source ~/.bashrc
fp-env              # activates the env
```

This adds **one three-line block** to your `~/.bashrc` and nothing else.
Easy to remove later: just delete those three lines.

---

## Rollback / cleanup (if you ever need to start over)

The whole env can be destroyed with one command. It does not touch any
other env, your system Python, or anything outside
`~/miniconda3/envs/foundationpose/`.

```bash
conda deactivate                             # leave the env first
conda env remove -n foundationpose -y        # delete the whole env
```

If you also want to remove the FoundationPose repo:

```bash
rm -rf $FOUNDATIONPOSE_ROOT
```

And if you added the `~/.bashrc` block in Step 11, open the file and
delete the three-line block between the `---` comments.

After all three, `conda env list` should show exactly the same envs as
when you started.

---

## Common failure modes and what to do

### `conda create` hangs on "Solving environment"

Your base conda is old. Run `conda update conda -n base -y` and retry.
This only updates the base env's conda, doesn't touch packages.

### `torch.cuda.is_available()` returns False

The PyTorch wheel you installed doesn't match the system's CUDA driver.
Uninstall PyTorch and reinstall the right wheel:

```bash
pip uninstall torch torchvision -y
# For any CUDA 12.x driver, cu121 is the safe bet:
pip install torch==2.1.0 torchvision==0.16.0 \
    --index-url https://download.pytorch.org/whl/cu121
```

### `pip install kaolin==0.15.0` says "no matching distribution"

You're on Python 3.10+ and kaolin's prebuilt wheels only go up to 3.10.
Easiest fix is to recreate the env with Python 3.9:

```bash
conda deactivate
conda env remove -n foundationpose -y
conda create -n foundationpose python=3.9 -y
conda activate foundationpose
# Restart from Step 3
```

Alternatively, if you must keep Python 3.11, try `kaolin==0.17.0` with
`torch==2.4.0`. See the Python 3.11 section in Step 3 above.

### `nvdiffrast` build fails with "nvcc not found"

```bash
# Find nvcc
which nvcc || find /usr/local -name nvcc 2>/dev/null | head -5

# Add its directory to PATH
export PATH=/usr/local/cuda/bin:$PATH   # adjust if yours is elsewhere

# Reinstall
pip uninstall nvdiffrast -y
pip install git+https://github.com/NVlabs/nvdiffrast.git
```

### `build_all_conda.sh` fails on eigen paths

Eigen's header directory isn't where the script expects. Reinstall via
conda (inside the active env) and try again:

```bash
conda install conda-forge::eigen=3.4.0 -y --force-reinstall
bash build_all_conda.sh
```

### `pyzed` import fails with SDK version mismatch

Your ZED SDK (system) and pyzed (Python) versions don't match. Reinstall
pyzed from the SDK's installer:

```bash
cd /usr/local/zed
python get_python_api.py
```

### OpenGL / glfw errors on `nvdiffrast` import

On headless GPU boxes, use CUDA rasterization instead of OpenGL:

```python
# Inside Python, this is already what our wrapper does:
import nvdiffrast.torch as dr
ctx = dr.RasterizeCudaContext()    # not RasterizeGLContext
```

If FoundationPose itself is using the GL context somewhere, set this
environment variable before launching:

```bash
export NVDIFFRAST_USE_CUDA=1
```

---

## File layout at the end of setup

Your home directory should look like:

```
$HOME/
├── miniconda3/
│   └── envs/
│       ├── base/                    (untouched)
│       ├── foundationpose/          (new, our env)
│       └── <any other envs>/        (untouched)
├── FoundationPose/                  (new, the upstream repo)
│   ├── estimater.py
│   ├── run_demo.py
│   ├── weights/
│   │   ├── 2023-10-28-18-33-37/
│   │   └── 2024-01-11-20-02-45/
│   └── ...
└── localization_for_visual_servoing/  (our project)
    ├── EKF/
    ├── FoundationModel/
    ├── experiments/
    └── ...
```

No other location on disk is modified by this setup (other than the
optional `~/.bashrc` line in Step 11).
