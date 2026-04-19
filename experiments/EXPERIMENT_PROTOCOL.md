# Experiment Day Protocol

Exact steps, commands, and what-to-do-with-your-hands for the
per-pipeline three-way evaluation (3 experiments x 5 objects = 15
trials, single start pose P1). Each experiment is one pipeline swept
over all five objects; run them back-to-back. Everything variable is
auto-captured in the CSV via the per-trial `run_tag`, so the only
thing you need to track on paper is a high-level session log
("session started 10:45, object placement chart as drawn on
whiteboard").

---

## Before you start (15 min, once per session)

### Step 0.1: Physical setup

- Power on the xArm, put it in a safe starting pose (roughly upright,
  hand pointing forward at chest height).
- Mount the ZED Mini on the end-effector using the bracket; plug in
  USB.
- Plug the arm into the network (ethernet, 192.168.1.241).
- Clear the workspace. Put masking tape on the table to mark the
  single object position **P1** (centered, ~50 cm from home pose in
  the forward direction). Use a ruler; mark with tape. All 15 trials
  use this same P1 location; only the object and pipeline change.

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

### Step 0.5: Generate trial file

```bash
python experiments/generate_trials.py \
    --depth-scale 0.001234 \
    --fp-repo-dir $FOUNDATIONPOSE_ROOT
```

This writes one file into `experiments/`:
- `trials_main.yaml` (15 trials: 3 experiments x 5 objects)

Trials are ordered by pipeline (Experiment 1 first, then 2, then 3),
so you run one pipeline to completion before moving to the next.
Within each experiment you cycle through the five objects in a fixed
order. Experiment 1 = Pipeline A (IBVS), Experiment 2 = Pipeline B
(EKF + DINOv2/SAM2/DepthAnything), Experiment 3 = Pipeline C (EKF +
FoundationPose).

---

## Per-pipeline three-way evaluation (~20 min)

### Step 1.1: Launch

```bash
cd experiments
python run_batch.py \
    --trials trials_main.yaml \
    --csv-dir runs/main
```

The runner prints the first trial's info and waits for you to press
ENTER.

### Step 1.2: Per-trial loop (repeat 15 times)

For each trial prompt:

1. **Read the tag**. The runner prints e.g. `cheezit_ekf`. The tag
   tells you: object = cheezit, pipeline = ekf. All trials are at
   start pose P1 and there are no repeats, so no extra suffix.

2. **Physically set up the scene**:
   - Place the right object (cheezit / tissue / cardboard / protein /
     brownie) at the tape-marked P1 position.
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
   saved to `runs/main/<tag>.csv`.

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

After each pipeline's 5-trial block, spot-check:

```bash
python analyze_csvs.py runs/main
```

You'll see a live table of everything done so far. If IBVS trials are
showing final errors above 5 cm consistently, something's off with
the calibration — pause, investigate before starting the next
pipeline.

---

## After all three experiments (no arm needed)

### Generate the paper's result tables

```bash
cd experiments

python analyze_csvs.py runs/main \
    --tols-cm 0.5,1.0,2.0 \
    --csv runs/summaries/main.csv
```

Stdout ends with an **aggregate pass-rate table** grouped by pipeline
— one row per experiment (1 / 2 / 3) summarizing that experiment's 5
per-object subcomponents. That aggregate is what goes into the paper
verbatim.

### Post-session file list

After the day, you should have:

```
experiments/runs/
├── main/                     # 15 CSV files (5 per experiment)
│   ├── cheezit_ibvs.csv      # Experiment 1 trials
│   ├── tissue_ibvs.csv
│   ├── cardboard_ibvs.csv
│   ├── protein_ibvs.csv
│   ├── brownie_ibvs.csv
│   ├── cheezit_ekf.csv       # Experiment 2 trials
│   ├── ...
│   ├── cheezit_foundationpose.csv    # Experiment 3 trials
│   └── ...
└── summaries/                # 1 summary CSV ready for the paper
    └── main.csv
```

15 individual CSVs plus 1 summary. The per-trial CSVs are your raw
data; the summary is the paper's table. Nothing else is needed from
the session — no notebook, no photos (unless you want them for the
presentation), no manual readings.

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
`depth_scale` value in the trials YAML file:
```bash
sed -i 's/depth_scale: [0-9.]*/depth_scale: 0.001456/' experiments/trials_main.yaml
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
- Masking tape + ruler for the single P1 object position.
- The five target objects (Cheez-It, tissue, cardboard shipping,
  protein bar, brownie mix).
- ~45 minutes of uninterrupted time.

Good luck. The framework is set up so that if each trial takes 30
seconds of arm time, the whole session is ~8 minutes of arm time
plus ~20 min of resets and troubleshooting. Plan for ~30 minutes
total in the lab.
