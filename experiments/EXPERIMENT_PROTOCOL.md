# Experiment Day Protocol

Exact steps, commands, and what-to-do-with-your-hands for the three
experiments. Everything variable is auto-captured in the CSV via the
per-trial `run_tag`, so the only thing you need to track on paper is a
high-level session log ("session started 10:45, object placement chart
as drawn on whiteboard").

---

## Before you start (15 min, once per session)

### Step 0.1: Physical setup

- Power on the xArm, put it in a safe starting pose (roughly upright,
  hand pointing forward at chest height).
- Mount the ZED Mini on the end-effector using the bracket; plug in
  USB.
- Plug the arm into the network (ethernet, 192.168.1.241).
- Clear the workspace. Put masking tape on the table to mark three
  object positions labelled **P1** (centered, ~50 cm from home pose
  in the forward direction), **P2** (20° to the left of the optical
  axis at the same distance), **P3** (15° above the optical axis, so
  the object is on a small stand). Use a ruler; mark with tape.

### Step 0.2: Software

On the GPU workstation, open a terminal:

```bash
cd ~/localization_for_visual_servoing
conda activate foundationpose       # or source your venv
git pull                             # get the latest
export FOUNDATIONPOSE_ROOT=$HOME/FoundationPose
```

### Step 0.3: Smoke test each mode (no arm)

```bash
# Verify DINOv2+SAM2 pipeline loads and tracks
python FoundationModel/dinov2_servo.py \
    --reference masked_objects/cheez_it_box.png \
    --mode ekf --no-robot

# In the OpenCV window: a Cheez-It box should be highlighted in green.
# Press 'q' to quit.

# Verify FoundationPose loads (takes ~20 s first time due to CUDA JIT)
python FoundationModel/dinov2_servo.py \
    --reference masked_objects/cheez_it_box.png \
    --mode foundationpose \
    --fp-box 0.19 0.06 0.22 \
    --no-robot

# A blue crosshair should track the box. HUD shows "FP: REG". Press 'q'.
```

If either fails, fix it before continuing — experiments won't recover
gracefully.

### Step 0.4: Calibrate depth once (5 min)

```bash
python FoundationModel/dinov2_servo.py \
    --reference masked_objects/cheez_it_box.png \
    --mode ekf --no-robot
```

- Place a Cheez-It box exactly 30 cm in front of the camera (use a
  ruler or a tape marker on the table).
- In the OpenCV window, press `c`.
- Read the log line: `Depth calibrated at Z=0.300m (d_rel=X.XXXX)
  -> scale=Y.YYYYYY`.
- **Note this scale value** (the single number after `scale=`). Example:
  `0.001234`.
- Press `q` to quit.

### Step 0.5: Generate trial files

```bash
python experiments/generate_trials.py \
    --depth-scale 0.001234 \
    --fp-repo-dir $FOUNDATIONPOSE_ROOT
```

This writes three files into `experiments/`:
- `trials_main.yaml` (90 trials, Experiment 1)
- `trials_robustness.yaml` (36 trials, Experiment 2)
- `trials_qr_sweep.yaml` (16 trials, Experiment 3)

---

## Experiment 1: Main three-way comparison (~60 min)

### Step 1.1: Launch

```bash
cd experiments
python run_batch.py \
    --trials trials_main.yaml \
    --csv-dir runs/exp1_main
```

The runner prints the first trial's info and waits for you to press
ENTER.

### Step 1.2: Per-trial loop (repeat 90 times)

For each trial prompt:

1. **Read the tag**. The runner prints e.g. `cheezit_centered_far_ekf_t1`.
   The tag tells you: object = cheezit, pose = centered_far, mode =
   ekf, trial = 1.

2. **Physically set up the scene**:
   - Place the right object (cheezit / tissue / cardboard / protein /
     brownie) at the right tape-marked position (P1/P2/P3 for
     centered_far / off_axis_left / off_axis_high).
   - Move the robot to a consistent "home" pose. On the arm's teach
     pendant or via Python, send it back to a fixed start. A simple
     script:
     ```bash
     # In a second terminal
     python -c "
     from xarm.wrapper import XArmAPI
     a = XArmAPI('192.168.1.241'); a.motion_enable(True); a.set_mode(0); a.set_state(0)
     a.set_position(x=300, y=0, z=300, roll=180, pitch=0, yaw=0,
                    speed=80, mvacc=500, wait=True)
     "
     ```
   - You can save this as `experiments/home.py` for quick re-use.

3. **Press ENTER** in the batch-runner terminal. An OpenCV window
   pops up.

4. **Wait for perception to lock on**. You'll see the green mask and
   the blue crosshair settle on the object. This takes ~3 s.

5. **Press `v`** to enable the servo. The arm starts moving.

6. **Watch it converge** or fail. Typical run: 10-30 s.

7. **When done**, press `q` to end the trial. The CSV is automatically
   saved to `runs/exp1_main/<tag>.csv`.

The runner auto-advances to the next trial.

### What gets auto-logged

Every frame in the CSV has: timestamp, pipeline mode, run_tag, pixel
centroid, EKF-filtered 3D position, EKF uncertainty, depth estimate,
servo command (dx/dy/dz mm), 3D error, iteration time (ms), robot FK
position, and (for FP trials) raw FoundationPose translation.

**The only thing you might want to note on paper**: if a trial goes
clearly wrong (arm collided, object fell, perception totally lost),
press `q` immediately and write the tag in your log with "REDO". The
batch runner will still save the bad CSV but you can delete it later.

### Step 1.3: Mid-session analysis (optional, ~30 s)

After every ~15 trials, spot-check:

```bash
python analyze_csvs.py runs/exp1_main
```

You'll see a live table of everything done so far. If IBVS trials are
showing final errors above 5 cm consistently, something's off with
the calibration — pause, investigate.

---

## Experiment 2: Robustness (~25 min)

### Step 2.1: Launch

```bash
cd experiments
python run_batch.py \
    --trials trials_robustness.yaml \
    --csv-dir runs/exp2_robustness
```

### Step 2.2: Per-trial loop (repeat 36 times)

Tags look like `cheezit_occlusion_ekf_t1`. The **condition** (baseline
/ occlusion / dim_light / distractor) is the second field. Set up the
scene according to the condition:

- **baseline**: normal lighting, clean table, Cheez-It at position P1.
- **occlusion**: same as baseline, but during the trial (after the arm
  starts moving), briefly wave your hand across ~30% of the object for
  ~2 seconds, then remove. Do this around the 5-second mark.
- **dim_light**: turn off the main room lights, close blinds. Keep one
  side lamp on. Object at P1.
- **distractor**: place a second box of similar color 15 cm to the
  left of the target. Object (target) at P1.

Then same loop as Experiment 1: home robot, press ENTER, wait for
lock, press `v`, watch, press `q`.

### Robustness-specific notes

For the `occlusion` trials, press `v` first, count to 5, *then*
introduce the occlusion. This lets the EKF's uncertainty trace show a
clear spike when the object is hidden and recovery when it's back.

---

## Experiment 3: Q/R sensitivity sweep (~10 min)

### Step 3.1: Launch

```bash
cd experiments
python run_batch.py \
    --trials trials_qr_sweep.yaml \
    --csv-dir runs/exp3_qr_sweep
```

### Step 3.2: Per-trial loop (repeat 16 times)

Same object (Cheez-It), same pose (P1), every trial. The only thing
that changes is the EKF's Q and R parameters, which are encoded in the
tag. You don't need to change anything physically between trials —
just reset the robot home pose and press ENTER.

Tags look like `cheezit_q0p010_r6_t1` meaning Q_pos = 0.010, R_uv = 6.0.

---

## After all three experiments (no arm needed)

### Generate the paper's result tables

```bash
cd experiments

# Exp 1: main comparison
python analyze_csvs.py runs/exp1_main \
    --tols-cm 0.5,1.0,2.0 \
    --csv runs/summaries/exp1_main.csv

# Exp 2: robustness
python analyze_csvs.py runs/exp2_robustness \
    --tols-cm 0.5,1.0,2.0 \
    --csv runs/summaries/exp2_robustness.csv

# Exp 3: Q/R sweep (looser tol is more informative here)
python analyze_csvs.py runs/exp3_qr_sweep \
    --tols-cm 1.0,2.0,5.0 \
    --csv runs/summaries/exp3_qr_sweep.csv
```

Each stdout ends with an **aggregate pass-rate table** grouped by
pipeline (or, for Exp 3, effectively grouped by Q value via the
run_tag). That aggregate is what goes into the paper verbatim.

### Post-session file list

After the day, you should have:

```
experiments/runs/
├── exp1_main/                # 90 CSV files
│   ├── cheezit_centered_far_ibvs_t1.csv
│   ├── cheezit_centered_far_ibvs_t2.csv
│   ├── ...
├── exp2_robustness/          # 36 CSVs
├── exp3_qr_sweep/            # 16 CSVs
└── summaries/                # 3 summary CSVs ready for the paper
    ├── exp1_main.csv
    ├── exp2_robustness.csv
    └── exp3_qr_sweep.csv
```

142 individual CSVs plus 3 summaries. The per-trial CSVs are your raw
data; the summaries are the paper's tables. Nothing else is needed
from the session — no notebook, no photos (unless you want them for
the presentation), no manual readings.

---

## Troubleshooting mid-session

### Arm collides / unsafe motion

Press the physical e-stop. Don't try to fix in software. Investigate
what went wrong, then clean up the CSV (move the bad trial to
`runs/_failed/`) and continue.

### Perception loses the object mid-trial

Press `q` to end that trial. The CSV will record the partial attempt.
Mark the tag as "REDO" mentally, come back to it after the batch.

### FoundationPose register fails (log shows "register failed")

Place the object more squarely in front of the camera and try again.
FP needs a reasonable starting depth / view to lock onto.

### Depth calibration drifts (EKF errors seem systematically wrong)

Pause, re-run the depth calibration step (Step 0.4). Update the
`depth_scale` value in the trials YAML files:
```bash
sed -i 's/depth_scale: [0-9.]*/depth_scale: 0.001456/' experiments/trials_*.yaml
```
Then resume.

### Robot home pose drifts across trials

Make the home script a one-liner you run between trials:
```bash
alias home="python experiments/home.py"
```
Then typing `home` before each ENTER guarantees consistency.

---

## What to bring on session day

- This protocol document, printed or on a laptop beside the GPU box.
- Masking tape + ruler for object positions.
- A side lamp (for the dim_light condition).
- A second similar-colored box (for the distractor condition).
- A phone for the occlusion condition (easier than using your hand
  since the hand stays in camera view).
- An hour and a half of uninterrupted time.

Good luck. The framework is set up so that if each trial takes 30
seconds of arm time, the whole session is 70 minutes of arm time plus
~30 min of resets and troubleshooting. Plan for 2 hours total in the
lab.
