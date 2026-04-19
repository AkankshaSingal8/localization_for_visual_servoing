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

### Pre-flight checks worth doing before Step 1

On the GPU workstation, these two gotchas are present and *will* bite
you during Step 4-5 unless you know about them:

```bash
# 1) Is there a global PYTHONPATH leaking into every python process?
#    (If this returns anything, it will pollute every `python` run in
#    the new env and can fake-install packages like groundingdino.)
echo "PYTHONPATH=$PYTHONPATH"

# 2) Does /usr/local/cuda/bin/nvcc exist? On some boxes /usr/local/cuda
#    is an empty directory and the real toolkits live in versioned
#    folders like /usr/local/cuda-12.4/, /usr/local/cuda-12.8/.
ls /usr/local/cuda/bin/nvcc 2>/dev/null || ls -d /usr/local/cuda-*/ 2>/dev/null
```

If `PYTHONPATH` is non-empty, plan to `unset PYTHONPATH` at the top of
every command you run in this env (until you strip it from `~/.bashrc`).
If `/usr/local/cuda/bin/nvcc` is missing, use the versioned directory
(e.g. `/usr/local/cuda-12.4/bin`) everywhere the doc says
`/usr/local/cuda/bin`.

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

This is the path that was verified end-to-end on the workstation
(driver 12.8, dual RTX 4090, Ubuntu). Steps 4 and 5 below have
Python-3.11-specific subsections that you must follow; the generic
commands will not all work on 3.11.

```bash
pip install torch==2.4.0 torchvision==0.19.0 \
    --index-url https://download.pytorch.org/whl/cu121
```

Verify:

```bash
python -c "
import torch
print('CUDA available:', torch.cuda.is_available())
print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')
print('PyTorch version:', torch.__version__)
print('PyTorch CUDA version:', torch.version.cuda)
"
# Expected:
#   CUDA available: True
#   Device: <your GPU>
#   PyTorch version: 2.4.0+cu121
#   PyTorch CUDA version: 12.1
```

If `CUDA available: False` here, fix that now — none of the later
steps will work.

**Option C: skip FoundationPose entirely**

If you're only running Methods 1 and 2 (IBVS and EKF-DINOv2), you can
stay on Python 3.11 and skip Steps 4-6 (FoundationPose deps and build).
You'll still need the project deps in Step 7. This is a valid starting
point to get experiments going while the FP env is sorted.

---

## Step 4: FoundationPose's Python deps

This is the step with the most failure modes. Read the Python-3.11
subsection below before running anything if you're on the py311 path.

### 4a. Eigen via conda

```bash
cd $FOUNDATIONPOSE_ROOT
conda install conda-forge::eigen=3.4.0 -y
```

### 4b. Downgrade setuptools so `pkg_resources` still works

Several packages in `requirements.txt` (`visdom` via `torchnet`, plus
more) have legacy `setup.py` files that `import pkg_resources` at the
top level. Setuptools 81+ removed that module entirely; on a fresh
Python 3.11 env you'll have setuptools 82.x, which will blow up
`pip install -r requirements.txt` with `ModuleNotFoundError: No module
named 'pkg_resources'`.

Pin setuptools to the last version that still ships `pkg_resources`,
and also constrain pip's build-isolation environments to respect that
pin:

```bash
pip install "setuptools<81" wheel
python -c "import pkg_resources; print('pkg_resources OK')"

# Used by every subsequent pip install -r in this step
echo "setuptools<81" > /tmp/pip_constraint.txt
echo "wheel"         >> /tmp/pip_constraint.txt
```

From here on, run `pip install` commands in this step with
`PIP_CONSTRAINT=/tmp/pip_constraint.txt` prepended so the build
isolation env also gets setuptools < 81.

### 4c. FoundationPose's own `requirements.txt` (filtered)

FP's `requirements.txt` pins `torch==2.0.0+cu118`, `torchvision==0.15.1+cu118`,
and `torchaudio==2.0.1+cu118`. Installing as-is will *downgrade* the
PyTorch we just installed in Step 3 and break cu12.x compatibility.
Filter those three lines (plus the cu118 `--extra-index-url`) out
first:

```bash
grep -vE '^(torch==|torchvision==|torchaudio==|--extra-index-url|# PyTorch)' \
    $FOUNDATIONPOSE_ROOT/requirements.txt > /tmp/fp_requirements_filtered.txt

PIP_CONSTRAINT=/tmp/pip_constraint.txt \
    pip install -r /tmp/fp_requirements_filtered.txt
```

This will intentionally downgrade numpy to 1.26.4 (needed by
`numba`, `scipy`, `scikit-learn`, and `open3d 0.18` on Python 3.11).
Verify afterwards that PyTorch is still 2.4.0+cu121:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# -> 2.4.0+cu121 True
```

### 4d. nvdiffrast

nvdiffrast's `setup.py` needs to import torch, so it cannot run inside
pip's default build-isolation environment. It also needs `nvcc` on
`PATH`. Use the versioned CUDA dir (see the pre-flight note):

```bash
export PATH=/usr/local/cuda-12.4/bin:$PATH    # adjust for your box
export CUDA_HOME=/usr/local/cuda-12.4
nvcc --version    # must succeed before proceeding

pip install --no-build-isolation git+https://github.com/NVlabs/nvdiffrast.git
```

### 4e. Kaolin (Python 3.11: use 0.17.0 from NVIDIA's wheel index)

PyPI only has `kaolin==0.1`. Real kaolin wheels live at NVIDIA's own
index, which is keyed by `torch_<ver>_cu<ver>`. For torch 2.4.0 + cu121:

```bash
PIP_CONSTRAINT=/tmp/pip_constraint.txt \
    pip install kaolin==0.17.0 \
        -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu121.html
```

(If you're on Python 3.9 / torch 2.1.0 / cu121, swap the URL to
`torch-2.1.0_cu121.html` and use `kaolin==0.15.0`.)

Kaolin will downgrade `jupyter-client` to 7.4.9, which pip will report
as conflicting with `ipykernel>=8.8.0`. This warning is cosmetic
(affects Jupyter UI only, not FP runtime) — ignore it.

### 4f. pytorch3d

For torch 2.4.0 + cu121 + Python 3.11, FAIR's prebuilt wheel index
(`https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py311_cu121_pyt240/download.html`)
has no matching wheel. Build from source instead — the dependencies
(`fvcore`, `iopath`) are already installed by `requirements.txt`:

```bash
export FORCE_CUDA=1
export TORCH_CUDA_ARCH_LIST="8.9"   # RTX 40xx; use your GPU's arch
pip install --no-build-isolation \
    "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```

This takes ~1 minute and produces `pytorch3d-0.7.8`.

If your GPU is different, look up its compute capability (e.g.
3090 = 8.6, A100 = 8.0, H100 = 9.0) and set `TORCH_CUDA_ARCH_LIST`
accordingly.

### 4g. Verify everything imports cleanly

```bash
python -c "
import torch, torchvision, numpy, scipy
import kaolin, pytorch3d
import nvdiffrast.torch as dr
import trimesh, cv2
print('torch:', torch.__version__)
print('numpy:', numpy.__version__)   # should be 1.26.4
print('kaolin:', kaolin.__version__)
print('pytorch3d:', pytorch3d.__version__)
print('ALL STEP-4 IMPORTS OK')
"
```

---

## Step 5: Build FoundationPose's C++/CUDA extensions

### 5a. Patch mycuda's C++ standard (required for PyTorch 2.2+)

`bundlesdf/mycuda/setup.py` hardcodes `-std=c++14`. PyTorch 2.2+ uses
C++17 in its headers and will bail with `#error You need C++17 to
compile PyTorch` when `common.cu` is compiled. Patch the two flag
lines:

```bash
sed -i 's/-std=c++14/-std=c++17/g' $FOUNDATIONPOSE_ROOT/bundlesdf/mycuda/setup.py
grep '\-std=c++' $FOUNDATIONPOSE_ROOT/bundlesdf/mycuda/setup.py   # verify
```

Skip this step only if you're on torch 2.1.x (Python 3.9 path).

### 5b. Ensure `nvcc` is on PATH

```bash
export PATH=/usr/local/cuda-12.4/bin:$PATH    # or whatever version exists
export CUDA_HOME=/usr/local/cuda-12.4
nvcc --version
```

### 5c. Run `build_all_conda.sh`, then manually fix the mycuda step

```bash
cd $FOUNDATIONPOSE_ROOT
export FORCE_CUDA=1
export TORCH_CUDA_ARCH_LIST="8.9"    # adjust for your GPU
bash build_all_conda.sh
```

`mycpp` will build cleanly (only harmless `printf` format warnings).
`mycuda` will fail with `ModuleNotFoundError: No module named 'torch'`
because the build script uses `pip install -e .` without
`--no-build-isolation`. Rebuild it by hand:

```bash
cd $FOUNDATIONPOSE_ROOT/bundlesdf/mycuda
rm -rf build *.egg-info *.so
pip install --no-build-isolation -e .
```

This takes ~1 minute and produces `common-0.0.0` (editable install)
plus `.so` files under `bundlesdf/mycuda/`.

### 5d. Verify both extensions import

Note: `common` depends on torch's `libc10.so`, so you must
`import torch` *before* `import common`:

```bash
cd $FOUNDATIONPOSE_ROOT
python -c "
import torch
import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), 'mycpp', 'build'))
import mycpp
import common
print('mycpp:', mycpp.__file__)
print('common:', common.__file__)
print('STEP 5 OK')
"
```

---

## Step 6: FoundationPose model weights

Download both weight directories into `$FOUNDATIONPOSE_ROOT/weights/`:

- `2023-10-28-18-33-37` (refiner)
- `2024-01-11-20-02-45` (scorer)

The download links are in FoundationPose's README (Google Drive). If
you have `gdown` and the Drive URL (the verified one is
`https://drive.google.com/drive/folders/1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i`):

```bash
pip install gdown
mkdir -p $FOUNDATIONPOSE_ROOT/weights
cd $FOUNDATIONPOSE_ROOT/weights
gdown --folder https://drive.google.com/drive/folders/1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i
```

gdown preserves the Drive folder's internal structure, so the weights
land at `weights/no_diffusion/2023-10-28-18-33-37/` and
`weights/no_diffusion/2024-01-11-20-02-45/`. FoundationPose's code
(see `learning/training/predict_pose_refine.py:100`) expects them
*one level up*, at `weights/<date>/`. Flatten:

```bash
cd $FOUNDATIONPOSE_ROOT/weights
mv no_diffusion/2023-10-28-18-33-37 . && \
mv no_diffusion/2024-01-11-20-02-45 . && \
rmdir no_diffusion

ls $FOUNDATIONPOSE_ROOT/weights
# 2023-10-28-18-33-37   2024-01-11-20-02-45
```

Expected sizes: refiner ~68 MB, scorer ~190 MB.

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
```

### ⚠️ pyzed 5.2 will upgrade numpy to 2.x — revert it

`pyzed-5.2`'s wheel metadata declares `numpy>=2.0`, so `get_python_api.py`
will upgrade numpy to 2.4.x. That breaks numba, scipy, scikit-learn,
kaolin, and cmeel-boost, all of which pin `numpy<2.0`. The cython-
generated pyzed C extension actually works fine against numpy 1.26.4
at runtime, so pin numpy back:

```bash
pip install "numpy==1.26.4"
```

`pip check` will complain forever that pyzed wants numpy>=2.0. Ignore
that warning.

### Verify

```bash
python -c "
import numpy, pyzed.sl as sl
print('numpy:', numpy.__version__)            # -> 1.26.4
print('pyzed SDK:', sl.Camera().get_sdk_version())
"
```

If the SDK is not installed, get it from stereolabs.com for your Ubuntu
version, install the `.run` file at system level, then come back to
this step.

---

## Step 9: Project path and smoke test

Clone the project if it isn't already on this machine, and test
end-to-end. On the workstation the project actually lives at
`/home/akanksha/repo/localization_for_visual_servoing`, not
`~/localization_for_visual_servoing` — substitute the correct path
in the commands below.

```bash
PROJECT_ROOT=$HOME/localization_for_visual_servoing   # or the actual path
[ -d "$PROJECT_ROOT" ] \
    || git clone <your-project-url> "$PROJECT_ROOT"

cd "$PROJECT_ROOT"

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
