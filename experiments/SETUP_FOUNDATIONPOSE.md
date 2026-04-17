# FoundationPose leg: setup on the GPU machine

This doc covers the steps needed on the GPU workstation to run the third
perception back-end (`--mode foundationpose`). FoundationPose requires
CUDA + nvdiffrast, so it only runs on the arm-day machine, not on the Mac
used for prototyping.

## 1. Clone NVLabs/FoundationPose

```bash
cd ~/
git clone https://github.com/NVlabs/FoundationPose.git
export FOUNDATIONPOSE_ROOT=$HOME/FoundationPose
```

Either use the Docker path (quickest, recommended by the authors) or the
conda path. Full instructions in the repo's own README; the key points
for us:

### Docker path (preferred)

```bash
cd $FOUNDATIONPOSE_ROOT/docker
docker pull wenbowen123/foundationpose
bash run_container.sh
# inside the container, once:
bash build_all.sh
```

### Conda path

```bash
conda create -n foundationpose python=3.9 -y
conda activate foundationpose
conda install conda-forge::eigen=3.4.0 -y
pip install -r $FOUNDATIONPOSE_ROOT/requirements.txt
pip install git+https://github.com/NVlabs/nvdiffrast.git
pip install kaolin==0.15.0
pip install pytorch3d
cd $FOUNDATIONPOSE_ROOT && bash build_all_conda.sh
```

## 2. Download the model weights

From the Google Drive links in the FoundationPose README, download both
checkpoints into `$FOUNDATIONPOSE_ROOT/weights/`:

- Refiner: `2023-10-28-18-33-37`
- Scorer:  `2024-01-11-20-02-45`

Verify:

```bash
ls $FOUNDATIONPOSE_ROOT/weights
# expect both directories above
```

## 3. Install the rest of our servoing deps in the same env

Still inside the FoundationPose conda env / container, install the
extras our pipeline needs that aren't already in FoundationPose's
requirements:

```bash
pip install opencv-python pyyaml xarm-python-sdk
# ZED SDK Python wrapper (pyzed) — install per Stereolabs instructions
# on the host; the container won't see the ZED directly.
```

If you're running inside the Docker container, you also need to
pass the ZED device through at `docker run` time (see StereoLabs docs).

## 4. Prepare an object mesh

Two options:

**(a) Procedural box mesh** — no CAD needed, just measure the box in
millimetres with a ruler:

```bash
# inside our repo
python FoundationModel/dinov2_servo.py \
    --reference masked_objects/cheez_it_box.png \
    --mode foundationpose \
    --fp-box 0.19 0.06 0.22 \        # width height depth in METRES
    --fp-repo-dir $FOUNDATIONPOSE_ROOT \
    --cam-to-robot zed_forward
```

**(b) Supplied CAD** — if you have an OBJ/PLY in metres:

```bash
python FoundationModel/dinov2_servo.py \
    --reference masked_objects/cheez_it_box.png \
    --mode foundationpose \
    --fp-mesh path/to/cheez_it.obj \
    --fp-repo-dir $FOUNDATIONPOSE_ROOT
```

The mesh origin is treated as the tracked point — for a procedural box,
that's the geometric centre.

## 5. First-run smoke test (no arm needed)

With the ZED plugged in but the arm disabled, verify the perception
back-end works:

```bash
python FoundationModel/dinov2_servo.py \
    --reference masked_objects/cheez_it_box.png \
    --mode foundationpose \
    --fp-box 0.19 0.06 0.22 \
    --fp-repo-dir $FOUNDATIONPOSE_ROOT \
    --no-robot
```

Place the box ~40 cm in front of the camera. In the OpenCV window you
should see the blue EKF crosshair locked onto the object and the HUD
reading stable `EKF: X=... Y=... Z=...` values. The first frame will
take a few seconds (FP register is slow); subsequent frames should
run at 3-10 Hz depending on your GPU.

If the register call fails with depth-validity errors, check that the
ZED stereo depth map is coming through — the servo script auto-enables
`DEPTH_MODE.NEURAL` when `--mode foundationpose` is set, but you still
need a working pyzed install.

## 6. Batch comparison run

Once all three perception back-ends work individually, the full
three-way comparison from a single trials file:

```yaml
# experiments/trials_threeway.yaml
common:
    cam_to_robot: zed_forward
    depth_cal_m: 0.30
    fp_repo_dir: /home/YOU/FoundationPose
trials:
    - tag: cheezit_ibvs_t1
      mode: ibvs
      reference: ../masked_objects/cheez_it_box.png
    - tag: cheezit_ekf_dinov2_t1
      mode: ekf
      reference: ../masked_objects/cheez_it_box.png
    - tag: cheezit_ekf_fp_t1
      mode: foundationpose
      reference: ../masked_objects/cheez_it_box.png
      fp_box: [0.19, 0.06, 0.22]
```

Run:

```bash
python experiments/run_batch.py \
    --trials experiments/trials_threeway.yaml \
    --csv-dir experiments/runs/threeway
python experiments/analyze_csvs.py experiments/runs/threeway \
    --tol-cm 1.0 --csv experiments/runs/threeway_summary.csv
```

## Common failure modes

- **`ImportError: cannot import name 'FoundationPose' from 'estimater'`**
  → `$FOUNDATIONPOSE_ROOT` is wrong or the repo is missing. `ls $FOUNDATIONPOSE_ROOT/estimater.py` must succeed.
- **`nvdiffrast` fails to import** → you're not inside the conda env
  / Docker container the build was done in.
- **`RasterizeCudaContext` crashes** → the GPU must be CUDA-capable and
  the driver/container must expose it. `nvidia-smi` inside the env
  should list your GPU.
- **First register takes >30 seconds** → expected on the first call
  while CUDA JIT-compiles kernels; subsequent frames are fast.
- **Tracking drifts after a few seconds** → reduce
  `--fp-redetect-interval` from 60 to 20, forcing more re-registration.
