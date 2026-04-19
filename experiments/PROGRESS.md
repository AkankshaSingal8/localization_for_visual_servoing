# Progress Log

Running log of notable changes made to get the experiment scripts in this
folder working end-to-end. Additive log — newest entries on top. Each
entry should be self-contained so anyone picking up the repo later can
reproduce what was done and why.

---

## 2026-04-18 — stacked / multi-instance fix (pre-SAM2 CC split, v2)

### Motivation

During a stacked-box smoke test (`smoke_ekf_stacked.log`) the arm went
down instead of approaching the top box:

```
DINOv2 bbox: (392, 252, 749, 591)   ← 339 px tall, spans BOTH boxes
SAM2 mask:   area=90763px (9.8%)    ← one big blob covering both
Centroid:    (575, 440)              ← landed in the gap
```

### Why the first fix failed

The initial attempt added a **post-SAM2** connected-components filter
in `ReferenceModel.detect_in_scene`. The filter's job was to split
SAM2's output into CCs, rank them by DINOv2 mean-similarity, and keep
the best one (topmost tiebreak for identical stacked boxes).

That didn't move the needle because **SAM2 never returned two CCs**.
Tracing back: `similarity_to_bbox` ran a 15×15 elliptical CLOSE with
2 iterations, i.e. ~30-pixel gap bridging on the DINOv2 binary
similarity mask. Two stacked boxes with even a small gap were fused
into **one** connected component before `similarity_to_bbox` picked
its winner. The wide bbox (covering both boxes) was then handed to
SAM2, which segmented both instances as one mask. The post-SAM2
filter looked at one CC, shrugged, and passed it through. All the
downstream logging was quiet because `n_candidates == 1`.

### Change (v2)

The real fix has to happen *before* SAM2, inside
`similarity_to_bbox`, so the bbox sent to SAM2 is tight around just
one instance.

- **`FoundationModel/dinov2_match_segment.py`**
  - New public helper `select_best_cc_by_similarity(mask, sim_upscaled,
    min_area_frac=0.002, sim_tie_eps=0.02)`.
    - Splits `mask` into connected components.
    - Primary pick = highest mean `sim_upscaled` inside each CC.
    - Tiebreak = smallest `y_min` when runner-ups are within
      `sim_tie_eps` of the leader (handles identical stacked boxes).
    - Returns `{mask, bbox, mean_sim, area, y_min, n_candidates,
      tiebreak_used, other_mean_sims}` for structured logging.
  - `similarity_to_bbox`:
    - Morphology rewritten. Was: 15×15 CLOSE ×2 then 15×15 OPEN ×1.
      Now: 7×7 OPEN ×1 then 7×7 CLOSE ×1. DINOv2 patches are 14 px,
      so any hole *inside* a single instance is smaller than that —
      a 7 px CLOSE still closes them. A 7 px CLOSE *does not* bridge
      two physically separate instances.
    - The inline CC picker is replaced with a call to the new
      helper, so the topmost tiebreak now applies here too.
    - Emits a richer log line when multiple CCs survive:
      `DINOv2 picker: 2 CCs → mean_sim=0.920 (topmost-tiebreak);
      dropped sims=[0.9]`.

- **`FoundationModel/dinov2_servo.py`**
  - Post-SAM2 filter is kept as a safety net in case
    `similarity_to_bbox` ever returns a bbox that still contains
    multiple instances (e.g. objects truly touching at one pixel).
    Logs `SAM2 multi-CC filter: ...` when that path fires.

### Why this is safe for single-object scenes

- Normal single-box scene → one CC, no ties, `tiebreak_used=False`,
  behavior identical to the old code aside from a slightly tighter
  bbox (shrunken by reduced morph close) — which is strictly better.
- Two CCs with one clearly winning on DINOv2 sim → similarity pick
  fires, topmost-tiebreak stays off.
- Two CCs with equal sim (truly identical boxes) → topmost-tiebreak
  fires, we keep the reference box on top.

### Verification

Synthetic sanity test added during development (1280×720 frame,
background sim ~U[0.2, 0.5], two 100×200 high-sim blobs):

1. Top blob sim 0.92, bottom 0.90 → picker returns bbox height
   ≈ 132 px at y1 ≈ 285 (top blob only, primary-by-sim).
2. Both blobs sim 0.90 → picker returns same bbox via topmost
   tiebreak.
3. Single 200×200 blob → picker returns bbox height ≈ 232 px
   (unchanged vs pre-fix).

### Files touched

- `FoundationModel/dinov2_match_segment.py` (morph + picker rewrite +
  new helper `select_best_cc_by_similarity`)
- `FoundationModel/dinov2_servo.py` (import helper, post-SAM2 safety
  net — unchanged semantics)

### How to verify on hardware

1. Stack two similar boxes, reference on top.
2. `./experiments/smoke_ekf.sh` (or `SMOKE=1 ...` for perception only).
3. Expect in the log:
   `DINOv2 picker: 2 CCs → mean_sim=... (topmost-tiebreak)`.
4. `DINOv2 bbox: (...)` should now be ~single-box-height tall
   (~230 px, not ~340 px).
5. `SAM2 mask: area=` should drop back to ~single-box footprint
   (~55–60 K px vs the previous ~90 K).
6. Centroid should land *inside* the top box.
7. PBVS error should monotonically decrease.

---

## 2026-04-18 — depth calibration is now persisted and auto-loaded (no more copy-paste)

### Motivation

Previous flow required the operator to eyeball the log line
`Depth calibrated at Z=0.300m (d_rel=...) -> scale=0.000123`, copy
the scale number by hand, and re-invoke `smoke_ekf.sh` with
`DEPTH_SCALE=<number>`. That's error-prone (typo in 6 digits,
forgetting a leading zero, losing the value between sessions) and
adds a step every time.

### Change

- **`FoundationModel/dinov2_servo.py`**
  - New module-level constant `DEPTH_SCALE_STATE_PATH` pointing at
    `experiments/depth_scale.json` (resolved relative to the source
    file, so it's stable regardless of cwd).
  - New helpers `_save_depth_scale_state(...)` and
    `_load_depth_scale_state(...)`. The saver writes atomically
    (tmp + rename) and swallows I/O errors to a warning — a disk
    problem must never take down a live servo loop.
  - `DINOv2CameraStreamer._calibrate_depth` (triggered by the 'c'
    key) now calls `_save_depth_scale_state` right after the
    in-memory calibration succeeds. JSON payload includes scale,
    offset, the calibration distance, the raw `d_rel_median`
    sample, the reference image path, a timestamp, and a
    `"source"` tag.
  - `main()` startup precedence for the depth scaler is now:
      1. `--depth-scale <value>` on the CLI (explicit override),
      2. a previously saved `depth_scale.json` (auto-load),
      3. uncalibrated `default_scale=0.001` + warning.
    Whichever path is taken is logged.
  - Plumbed `cam_thread.reference_path = args.reference` so the
    saved JSON records which reference image produced the
    calibration.

- **`experiments/smoke_ekf.sh`**
  - `DEPTH_SCALE` env-var behaviour changed: it is now an explicit
    override on top of the auto-loaded value; unset is the normal
    case.
  - When unset, the script checks `experiments/depth_scale.json`
    and logs the value it will pick up (or a warning if neither
    exists).
  - Header env-vars table updated so the default story reads
    "calibrate once, run everywhere" instead of "calibrate, copy,
    paste, run".

- **`experiments/calibrate_depth.sh`**
  - Banner and header rewritten: no longer tells the operator to
    copy the number; instead tells them the scale was auto-saved
    and they can now just run `smoke_ekf.sh`.

### Verification

- `bash -n` on both experiment scripts — pass.
- `python -c "import ast; ast.parse(open(...).read())"` on
  `dinov2_servo.py` — pass.
- ReadLints on `dinov2_servo.py` — clean.
- Round-trip sanity on the JSON payload (save → load) — pass.

### New flow

```bash
# 1. Once per camera setup.
./experiments/calibrate_depth.sh
# Place object at 0.30 m, wait for mask, press 'c', press 'q'.
# experiments/depth_scale.json is now populated.

# 2. Every subsequent run: just this. Auto-calibrated.
./experiments/smoke_ekf.sh

# Want a specific value for one run only?
DEPTH_SCALE=0.000123 ./experiments/smoke_ekf.sh

# Re-calibrate in the middle of a session? Press 'c' again —
# the file gets overwritten, so the NEXT run picks up the fresh
# value automatically.
```

### Notes

- `experiments/depth_scale.json` is machine-specific (depends on
  the exact ZED Mini unit, its resolution, and the monocular-depth
  model loaded). It should be ignored by version control if this
  ever moves back to a git repo.
- Deleting the file is the "reset to uncalibrated" button.

---

## 2026-04-18 — calibration flow wired end-to-end (DEPTH_CAL_M + DEPTH_OFFSET env vars, calibrate_depth.sh reference fixed)

### Problem

After promoting `smoke_ekf.sh` to a motion script with a `DEPTH_SCALE`
pass-through, the documented calibration flow still broke in two
places:

1. `experiments/calibrate_depth.sh` was still hard-coded to
   `masked_objects/cheez_it_box.png`. Running it against a physical
   Amazon Basics tissue box never produced a valid mask lock, so the
   'c' key either did nothing useful or captured depth off the
   background — same failure mode as the original reference-image
   bug one entry below.
2. There was no way to calibrate at a distance other than 30 cm
   without editing Python. If the operator's bench only allowed 25 cm
   of working distance, they either mis-measured to make 30 cm fit or
   silently calibrated against the wrong Z.

### Fix

- `experiments/calibrate_depth.sh` rewritten:
  - Defaults `REFERENCE` to `experiments/input_image_transparent.png`
    so it matches `smoke_ekf.sh` out of the box. Can still be
    overridden via `REFERENCE=...` for other targets.
  - Accepts `DEPTH_CAL_M` env var (default 0.30) and forwards it as
    `--depth-cal-m`.
  - Stdout banner now prints the actual reference path and
    calibration distance, plus the exact follow-up command
    (`DEPTH_SCALE=<number> ./experiments/smoke_ekf.sh`), so the
    operator doesn't need to remember the glue.
  - Uses `exec python ...` so signals propagate cleanly.
- `experiments/smoke_ekf.sh`:
  - Added `DEPTH_CAL_M` env var → `--depth-cal-m` pass-through, so
    the in-session 'c' key calibrates at the right distance without
    editing anything.
  - Added `DEPTH_OFFSET` env var → `--depth-offset` pass-through for
    completeness (rarely used, but the Python CLI already supports
    it).
  - Header env-vars table expanded accordingly.

### Verification

- `bash -n` passes on both scripts.
- `chmod +x` preserved on both.
- `grep` confirms `--reference`, `--depth-scale`, `--depth-offset`,
  `--depth-cal-m`, and `--no-robot` all map to real argparse options
  in `FoundationModel/dinov2_servo.py`.

### Usage (full calibrated-motion flow)

```bash
# 1. Calibrate once per camera setup.
./experiments/calibrate_depth.sh
# -> watch for: "Depth calibrated at Z=0.300m (d_rel=...) -> scale=0.000123"
# -> copy the number after scale=

# 2. Run the servo with that scale.
DEPTH_SCALE=0.000123 ./experiments/smoke_ekf.sh

# Non-default calibration distance (same number goes into both steps):
DEPTH_CAL_M=0.25 ./experiments/calibrate_depth.sh
DEPTH_SCALE=<captured> DEPTH_CAL_M=0.25 ./experiments/smoke_ekf.sh
```

---

## 2026-04-18 — `smoke_ekf.sh` promoted to a full-motion script (no longer a smoke test)

### Context

`smoke_ekf.sh` was originally a perception-only check: it invoked
`FoundationModel/dinov2_servo.py` with `--no-robot` so the arm would
never move. Once the mask + reference-image issues were resolved we
wanted the same script to actually drive the xArm, so the `--no-robot`
flag was dropped. That by itself enables motion (all three gates —
`RobotController.connect()`, startup calibration, and the 'v' toggle —
work once the arm is reachable), but it left two real problems:

1. The header comment still described the script as a smoke test, so
   anyone reading the file got a dangerously wrong mental model: they
   would not expect the arm to move on launch.
2. If the arm was powered off or on the wrong network, `connect()`
   inside Python would swallow the failure and the servo loop would
   simply never enable. From the user's side that looked identical to
   a perception bug ("mask is fine but the arm won't move"), which had
   already burnt one debugging session.
3. `dinov2_servo.py` already accepts `--depth-scale`, but the script
   never forwarded it, so every run started with an uncalibrated depth
   scaler (`default_scale=0.001`). That made the PBVS approach
   distance meaningless until the operator remembered to press `c`.

### Fix

Rewrote `experiments/smoke_ekf.sh`:

- **Accurate header.** The top-of-file comment now states that the
  script will move the arm, enumerates what happens on launch (connect
  → ±8 mm Y/Z calibration → autonomous servo at ~10 mm/s along +X),
  lists every keybind the window reacts to, and carries an explicit
  pre-flight checklist (e-stop reachable, ~50 mm clearance, reference
  matches target, mask verified via `SMOKE=1` first if in doubt).

- **Fail-fast preflight.** Before invoking Python the script now:
  - Pings `ARM_IP` (default `192.168.1.241`) with `ping -c 1 -W 2`.
    On failure it prints the likely causes (powered off, wrong IP,
    ICMP-blocking gateway) and exits non-zero. This replaces the
    previous silent `Robot connect failed` from inside Python.
  - Runs `python -c "from xarm.wrapper import XArmAPI"` to confirm the
    xarm SDK is importable in the active env — otherwise
    `RobotController.connect()` would catch the `ImportError` and
    silently return `False`, again indistinguishable from an
    unreachable arm.
  - Both checks are skippable via env vars for the edge cases.

- **`DEPTH_SCALE` pass-through.** If `DEPTH_SCALE` is exported, the
  script appends `--depth-scale "$DEPTH_SCALE"` to the Python args and
  logs that it did so. If it is unset, it prints a loud warning
  explaining that the depth scaler is uncalibrated and how to obtain a
  value (`./calibrate_depth.sh`). This makes it trivial to go from
  calibration → servo without editing this file.

- **`SMOKE=1` escape hatch.** The perception-only behaviour is still
  one env-var away: `SMOKE=1 ./smoke_ekf.sh` re-adds `--no-robot`,
  skips both preflight checks, and never touches the arm. This
  preserves the original "quick vision check" workflow without the
  script lying about what it does by default.

- **`SKIP_ARM_PRECHECK=1`** for networks that block ICMP (still
  requires the xarm SDK import to succeed).

### Verification

- `bash -n experiments/smoke_ekf.sh` — syntax OK.
- `chmod +x` preserved.
- Manual read-through confirms: every existing arg (`--reference`,
  `--mode ekf`) is still passed, `--no-robot` is only added under
  `SMOKE=1`, and `--depth-scale` is only added when `DEPTH_SCALE` is a
  non-empty string.

### Usage

```bash
# Normal: arm WILL move. Calibrated depth.
DEPTH_SCALE=0.0001234 ./experiments/smoke_ekf.sh

# Perception only (old smoke-test behaviour).
SMOKE=1 ./experiments/smoke_ekf.sh

# Non-default arm IP.
ARM_IP=192.168.1.99 ./experiments/smoke_ekf.sh

# Arm on a network path that blocks ICMP.
SKIP_ARM_PRECHECK=1 ./experiments/smoke_ekf.sh
```

### Out of scope

- `calibrate_depth.sh` still references `masked_objects/cheez_it_box.png`;
  left as-is because depth calibration is supposed to use whatever
  rigid reference the operator physically places at 30 cm, not
  necessarily the servoing target. Swap manually if needed.
- `ROBOT_IP` inside `FoundationModel/negative_weighing.py` is still
  hard-coded; the `ARM_IP` override only affects the preflight ping,
  not the actual connect() call. Decoupling that properly is a
  follow-up.

---

## 2026-04-18 — root cause: `smoke_ekf.sh` was using the wrong reference image

### Symptom

After rounds 1 and 2 of mask-location guards below, the mask was still
sometimes landing on the wrong region and DINOv2 component mean
similarity was *uniformly* capped at `0.69–0.73` on every detection
across every run. No amount of threshold / color-weight tuning moved
those numbers.

### Root cause

`smoke_ekf.sh` was committed with:

```bash
python FoundationModel/dinov2_servo.py \
    --reference masked_objects/cheez_it_box.png \
    --mode ekf --no-robot
```

… but the physical target on the desk is an Amazon Basics facial tissue
box (orange top half, white bottom band, `facial tissue` / `160 2-PLY`
branding), not a Cheez-It box. The two are totally different products
with different colors, layouts, and surface text. DINOv2 was being fed
a reference of **the wrong object** and asked to find it in a scene
that didn't contain that object at all — so the best it could ever do
was "which blob in the scene is vaguely the most Cheez-It-ish,"
plateauing at ~0.70 similarity.

`masked_objects/` had both available all along:

```
masked_objects/amazon_tissue_box.png   830,382 bytes   ← actual target
masked_objects/cheez_it_box.png      1,001,250 bytes   ← what the script used
```

The user's canonical copy of the correct reference lives at
`experiments/input_image_transparent.png`, which is byte-identical to
`masked_objects/amazon_tissue_box.png`.

This is what rounds 1 and 2 were papering over: the topmost-mask bug
and the teleport-on-redetect bug are both real and the guards are
still correct defence-in-depth, but the reason DINOv2 was producing
garbage to begin with was the reference/target mismatch.

### Fix

One-line change in `experiments/smoke_ekf.sh`:

```bash
python FoundationModel/dinov2_servo.py \
    --reference experiments/input_image_transparent.png \
    --mode ekf --no-robot
```

No code changes in `FoundationModel/`. No threshold or color-weight
tuning applied yet — doing the minimal change first to isolate how
much of the remaining drift is reference-image vs pipeline-tuning.

### How to verify the fix

Re-run `./experiments/smoke_ekf.sh`. Expected:

- `Reference: 1024x768, alpha present, fg pixels: <large>` in the
  startup banner (landscape tissue box, not the portrait Cheez-It).
- DINOv2 component mean similarity (new `sim_mean=` field from round 2)
  should jump well above `0.80` on good detections. If it's still
  `~0.7x` with the correct reference, the reference image itself is
  the next problem (glare, scale mismatch, framing).
- Green mask should land on the tissue box from the first lock, not
  drift around the frame.
- Re-detection rejections from round-2 guards should be rare (they
  were firing every cycle before because DINOv2 was lost).

### Follow-up tuning knobs (only apply if still drifting with the
### correct reference)

The tissue box reference has ~60% uniform orange + ~30% plain white +
small text, which is still a weaker semantic match than, say, a
densely-textured box. If drift persists:

- `--color-weight 0.2` (currently `0.7`) — demotes the mean-pooled
  ResNet18 color vector (which matches any warm-colored surface) and
  lets DINOv2 patch semantics dominate.
- `--threshold-pct 96` (currently `93`) — shrinks the similarity
  hot-region before bbox extraction, reducing background contamination.
- Trim the white bottom band out of the alpha channel of
  `input_image_transparent.png` so `ref_fg_feat` and the foreground
  DINOv2 patches only encode the distinctive orange+text upper region.

---

## 2026-04-18 — `smoke_ekf.sh` mask was *still* not landing on the box (round 2)

### Symptom

After the `select_topmost=False` + `DINOV2_MASK_MAX_FRAC` fixes below
("round 1"), re-running `./smoke_ekf.sh` produced masks at a healthy
*size* (~8–9% of the frame rather than 23–46%) but the mask was still
painted on the wrong region of the image (upper-left of the frame rather
than on the target box).

### Evidence from the log

Run at 15:30:25 (terminal lines 7–1020, recording
`vs_dinov2_20260418_153025.mp4`):

| Re-detect | Bbox                   | DINOv2 sim | Mask area | Centroid     |
|----------:|------------------------|-----------:|----------:|--------------|
| #1        | `(751, 468, 1140, 720)`| `0.718`    | `8.8%`    | `(932, 609)` |
| #2        | `(747, 462, 1133, 720)`| `0.720`    | `8.9%`    | `(912, 606)` |
| #3        | `(402,  31,  576, 234)`| `0.692`    | `8.3%`    | `(476, 206)` |

Between re-detects #2 and #3 the anchor centroid jumped **≈ 450 px
diagonally** across the frame (from the lower-right quadrant to the
upper-left quadrant) in a single re-detection step. SAM2 happily
re-propagated from the new (wrong) anchor, locking the green mask at
`(453, 232)` for the rest of the run. The DINOv2 component mean
similarity was ≤ `0.72` on every single detection — well below the
`~0.80` we'd expect from a clean reference match.

### Root cause

Round 1 stopped SAM2 from segmenting *too large* a region, but did
nothing to stop the pipeline from accepting *any* small region DINOv2
happened to point at. With DINOv2 similarity in the 0.69–0.72 range the
93rd-percentile similarity threshold is essentially picking "the most
textured blob we can find" — and when the real box is briefly occluded
or the camera shifts, that blob can be the monitor, a poster, or a
corner of the robot base. Two missing guards:

1. **No confidence floor on the DINOv2 match itself.** The pipeline only
   checked the downstream SAM2 score. SAM2 will gladly segment a clean
   edge around whatever bbox you hand it with score `> 0.9`, even if the
   bbox is on the wrong object.
2. **No temporal consistency check on re-detections.** Once the anchor
   was locked in the lower-right (where the box was), the 30-frame
   re-detection still allowed a fresh DINOv2 hit in the upper-left to
   blindly overwrite it. Real tabletop targets cannot teleport 450 px
   in 1 second.

### Fix

Two additions in `FoundationModel/dinov2_servo.py`, plus a small plumbing
change in `FoundationModel/dinov2_match_segment.py` to surface the
DINOv2 confidence to the caller.

1. **Surface DINOv2 component mean similarity.** `similarity_to_bbox`
   already computes this internally (`best_score`) and prints it to
   stdout (`Selected component X with mean similarity 0.xxx`). It now
   also returns it as a 4th tuple element:

   ```python
   return (x1, y1, x2, y2), comp_mask, sim_upscaled, best_score
   ```

   The peak-fallback branch returns `-1.0` (no component found).
   `ReferenceModel.detect_in_scene` propagates this as a 6th return
   value `dinov2_sim_mean`.

2. **Two new acceptance guards in `run_dinov2_pipeline` (detection
   path).** New module-level constants:

   ```python
   DINOV2_SIM_MIN_REDETECT      = 0.78   # confidence floor for re-detect
   DINOV2_MAX_CENTROID_JUMP_PX  = 250    # anti-teleport guard
   ```

   Both guards apply **only when `tracker.anchor_locked`** — the first
   detection has nothing to compare against. On re-detection the
   pipeline now rejects a candidate when either:

   - `sim_mean < DINOV2_SIM_MIN_REDETECT` — weak DINOv2 confidence (or
     peak-fallback, which returns `-1.0`);
   - `‖new_centroid − tracker.prev_centroid‖ > 250 px` — centroid
     teleport.

   On reject we log the specific reason and **do not** call
   `tracker.update(None, None, None)`. Previously a reject would wipe
   `prev_logits` / `prev_centroid`, causing the next SAM2 propagation
   frame to lose its prior and re-acquire from scratch — exactly the
   scenario we're trying to avoid. Now the tracker simply keeps its
   existing (good) anchor and continues propagating.

Why not apply the guards on the very first detection? Because the
similarities observed on this reference are uniformly in the 0.69–0.72
range; a 0.78 floor would mean the system never locks on in the first
place. The first lock is best-effort; re-detections must earn the right
to overwrite it.

### How to verify the fix

Re-run `./experiments/smoke_ekf.sh` with the same scene. Expected
changes in the log:

- The first `--- Running DINOv2 detection ---` still produces a
  centroid and the line `DINOv2 detection done. Centroid: (...),
  sim_mean=0.7xx, anchor_locked: True` (new `sim_mean=` field).
- Subsequent re-detections that *would* have jumped across the frame
  now log one of:
  - `DINOv2 detection: rejected — weak DINOv2 similarity (0.692 < 0.78)
    — not confident enough to overwrite locked anchor`
  - `DINOv2 detection: rejected — centroid teleport (450px > 250px)
    from (912, 606) to (476, 206) — likely a spurious match on
    background`
- The green mask stays wherever the first good lock placed it; there
  are no centroid teleports during the run.

If the very first lock is on the wrong region, that's a reference-image
problem (see "If the mask is still wrong" below the round 1 section) —
neither guard will help because both are gated on `anchor_locked`.

### Tuning knobs

Both thresholds are module-level constants at the top of
`dinov2_servo.py`. If the scene genuinely has a fast-moving target, the
`250 px` jump limit can be increased. If a better reference image
raises typical DINOv2 similarity above `0.85`, the similarity floor can
be raised to `0.82` for stricter rejection.

---

## 2026-04-18 — `smoke_ekf.sh` mask was landing on the desk/wall instead of the target box (round 1)

### Symptom

After the sam2 import issue (see next entry) was resolved, `smoke_ekf.sh`
ran without errors and the OpenCV window opened, but the green SAM2 mask
was being painted on the background/desk instead of the Cheez-It box.
The EKF anchor locked at a bogus centroid and SAM2 propagation happily
tracked that bogus region frame-to-frame with `score ≈ 0.995`.

### Evidence from the log

Three tell-tale signatures in the same run
(`vs_dinov2_20260418_152044.mp4`, terminal lines 7–1019):

- Per-detection mask area ballooned: `SAM2 mask: score=0.992,
  area=212629px (23.1%)` on "good" re-detections and `area=424143px
  (46.0%)` on bad ones. A Cheez-It box at ~0.5 m fills ~10–15% of an
  HD720 frame, not 46%.
- Anchor centroid jumping between `(640, 620)` and `(605, 267)` across
  re-detections 30 frames apart. `y=267` in a 720-tall frame is the
  upper third — that's the wall above the desk, not the box.
- One re-detection logged `SAM2: best mask score = 0.229` (SAM2
  effectively refused the bbox prompt) followed by `Split into 36
  components, selected topmost` → `Refined topmost mask: score=0.988`.
  The high post-refinement score was deceptive — SAM2 was happily
  re-segmenting the wall via a point prompt placed on the wall.

### Root cause

`ReferenceModel.detect_in_scene` called `refine_with_sam2` with its
default `select_topmost=True`. The topmost-selection logic in
`dinov2_match_segment.py:_select_topmost_mask` picks the connected
component whose top edge has the smallest y-coordinate — useful for
isolating the top box in a **stacked** scene, but actively wrong for a
**single box on a desk**: it systematically picks the component *above*
the box (wall/ceiling/background) rather than the box itself. SAM2's
subsequent point-prompt re-segmentation then confirms that wrong region
with high score, so nothing downstream flags the failure.

Additionally, `run_dinov2_pipeline` accepted any SAM2 mask as long as
`sam_score > 0.5`, with no check on mask area. A 46%-of-frame mask is
physically impossible for the target object yet the pipeline still
locked on to it.

### Fix

Two surgical changes in `FoundationModel/dinov2_servo.py`:

1. **Disable `select_topmost` for single-object scenes.** In
   `ReferenceModel.detect_in_scene`:

   ```python
   sam_mask, sam_score, sam_logits = refine_with_sam2(
       scene_bgr, bbox, select_topmost=False, return_logits=True)
   ```

   This skips the stack-of-boxes heuristic entirely; SAM2's best
   bbox-prompted mask is kept as-is. If a future trial actually uses
   stacked objects, flip it back (or expose a CLI flag).

2. **Reject oversized masks in `run_dinov2_pipeline`.** Added a
   module-level constant:

   ```python
   DINOV2_MASK_MAX_FRAC = 0.35   # reject masks covering >35% of frame
   ```

   and guarded the detection-accept branch:

   ```python
   mask_frac = np.count_nonzero(sam_mask) / float(h * w)
   if sam_mask is not None and sam_score > 0.5 \
           and mask_frac < DINOV2_MASK_MAX_FRAC:
       ...accept mask...
   else:
       if sam_mask is not None and mask_frac >= DINOV2_MASK_MAX_FRAC:
           logger.info("rejected oversized mask (%.1f%%)", ...)
       tracker.update(None, None, None)
   ```

   This would have rejected both the 46% and 45.6% masks from the buggy
   run and forced the tracker to keep its previous state rather than
   jumping to the wall.

### How to verify the fix

Re-run `./experiments/smoke_ekf.sh`. Expected:

- `SAM2 mask: score=0.9xx, area=<5-15>%` (not 23% / 46%).
- No `Split into N components, selected topmost` log lines (that code
  path is gated behind `select_topmost=True` and is now off).
- Per-redetection centroid stable across re-detection cycles instead of
  swinging between `(640, 620)` and `(605, 267)`.
- Occasional `DINOv2 detection: rejected oversized mask (X%)` lines are
  **expected** now when SAM2 returns garbage; the tracker should keep
  its previous anchor instead of jumping.

### If the mask is still wrong after these fixes

The underlying DINOv2 similarity scores in the buggy run were
`0.69–0.74`, which is borderline. If the fixed run still drifts:

- Capture a fresh reference image of the actual box at roughly the
  expected servo distance/angle with matched lighting; replace
  `masked_objects/cheez_it_box.png`.
- Raise `DINOV2_THRESHOLD_PCT` from 93 → 96/97 to shrink the DINOv2
  bbox so SAM2 gets a tighter prompt.

---

## 2026-04-18 — `smoke_ekf.sh` was crashing with `ModuleNotFoundError: No module named 'sam2'`

### Symptom

Running `./smoke_ekf.sh` from `experiments/` opened the ZED camera, loaded
DINOv2, and then every perception frame raised:

```
File ".../FoundationModel/dinov2_match_segment.py", line 350, in _get_sam2
    from sam2.build_sam import build_sam2
ModuleNotFoundError: No module named 'sam2'
```

DINOv2 detection itself still produced bboxes (visible in the log as
`DINOv2 bbox: (...)`), but `refine_with_sam2` failed on every frame, so
no mask ever reached the EKF and the smoke test was effectively
non-functional.

### Root cause

Two independent problems stacked on top of each other:

1. **Wrong `THIRD_PARTY_ROOT` in every `FoundationModel/*.py` file.**
   Each file defined
   ```python
   THIRD_PARTY_ROOT = os.path.join(os.path.dirname(__file__), "third-party")
   ```
   which resolves to `FoundationModel/third-party/`. The actual cloned
   repos (`sam2/`, `GroundingDINO/`, `Depth-Anything-V2/`, `sam3/`) live
   at the project root under `localization_for_visual_servoing/third-party/`,
   one directory up. `sys.path.insert(0, THIRD_PARTY_ROOT + "/sam2")`
   was adding a non-existent path, so `import sam2` fell through to a
   system-wide search that found nothing.

2. **`sam2` runtime dep `hydra-core` was missing in the `foundationpose`
   conda env.** `sam2/__init__.py` imports `from hydra import
   initialize_config_module` at load time, so even with the path fix,
   the first `import sam2.build_sam` raised
   `ModuleNotFoundError: No module named 'hydra'`.

### Fix 1 — re-point `THIRD_PARTY_ROOT` one level up

Updated all eight files that define `THIRD_PARTY_ROOT` in
`FoundationModel/` to resolve to the project-root `third-party/`:

```python
THIRD_PARTY_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "third-party")
)
```

Files changed:

- `FoundationModel/dinov2_match_segment.py`
- `FoundationModel/dinov2_servo.py`
- `FoundationModel/negative_weighing.py`
- `FoundationModel/top_box_bias.py`
- `FoundationModel/sam2_tracking_method.py`
- `FoundationModel/horizontal_edge_detector.py`
- `FoundationModel/test_vs.py`
- `FoundationModel/test_vs_point.py`

Nothing else in those files was touched; the downstream
`sys.path.insert(0, os.path.join(THIRD_PARTY_ROOT, "sam2"))` and
`os.path.join(THIRD_PARTY_ROOT, "sam2/checkpoints/sam2.1_hiera_large.pt")`
references now resolve correctly.

### Fix 2 — install the missing runtime deps in `foundationpose`

Inside the activated `foundationpose` env:

```bash
pip install "hydra-core>=1.3.2" "iopath>=0.1.10"
```

`iopath` was already present; `hydra-core==1.3.2` was installed fresh.
`omegaconf` and `antlr4-python3-runtime` (hydra's transitive deps) were
already in the env so no other packages were touched.

### Verification

End-to-end smoke of the SAM2 load path inside `foundationpose`:

```bash
python -c "
import sys
sys.path.insert(0, '/home/akanksha/repo/localization_for_visual_servoing/third-party/sam2')
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import torch
m = build_sam2('configs/sam2.1/sam2.1_hiera_l.yaml',
               '/home/akanksha/repo/localization_for_visual_servoing/third-party/sam2/checkpoints/sam2.1_hiera_large.pt',
               device='cuda')
SAM2ImagePredictor(m)
print('SAM2 end-to-end load OK')
"
# -> SAM2 end-to-end load OK
```

After the two fixes, `./experiments/smoke_ekf.sh` runs without the
`ModuleNotFoundError` and enters the normal perception loop (green mask +
blue crosshair on the object in the OpenCV window).

### Known follow-up issues (not fixed here)

These appear in `smoke_ekf.sh` output but are orthogonal to the sam2
crash. They are logged here so the next person doesn't confuse them for
regressions:

- `Depth scaler is UNCALIBRATED (default_scale=0.001)` — expected on a
  fresh workstation. Run the calibration step from
  `EXPERIMENT_PROTOCOL.md` §0.4 (press `c` with the object at 30 cm) or
  pass `--depth-scale <value>` into `smoke_ekf.sh`.
- `Failed to read ZED intrinsics: 'pyzed.sl.CameraInformation' object
  has no attribute 'calibration_parameters'` — pyzed SDK API rename.
  Intrinsics fall back to hard-coded defaults; fine for the smoke test,
  will bias metric runs. Separate fix needed in `dinov2_servo.py`'s
  intrinsic-read path.
- `xFormers is not available (...)` warnings from DINOv2 — harmless;
  pure-PyTorch attention path is used.

---

## How to run `smoke_ekf.sh` (summary)

Starting from a shell on the workstation:

```bash
cd /home/akanksha/repo/localization_for_visual_servoing/experiments
./smoke_ekf.sh
```

`_env.sh` (sourced by `smoke_ekf.sh`) handles:

- activating the `foundationpose` conda env,
- unsetting any leaked global `PYTHONPATH`,
- putting the CUDA 12.4 toolkit on `PATH`,
- `cd`-ing into the project root,
- echoing a one-line env summary and warning if `DISPLAY` is unset.

Prerequisites (only need to be done once per machine):

1. **`foundationpose` conda env exists and has project deps** — follow
   `SETUP_CONDA_ENV.md` through Step 7.
2. **`third-party/` clones are present at the project root** with the
   layout:
   ```
   localization_for_visual_servoing/third-party/
   ├── sam2/
   │   ├── sam2/                             (the python package)
   │   └── checkpoints/sam2.1_hiera_large.pt (~900 MB)
   ├── GroundingDINO/
   │   └── weights/groundingdino_swint_ogc.pth
   └── Depth-Anything-V2/
       └── checkpoints/
   ```
3. **`hydra-core>=1.3.2`** is installed in the `foundationpose` env (see
   Fix 2 above).
4. **ZED Mini is plugged in and `DISPLAY` is set** (the OpenCV window
   needs a real display). The script prints a warning if not.

Expected behaviour:

1. `_env.sh` prints the env banner.
2. DINOv2 ViT-B/14 loads from torch hub cache (~1 s).
3. First SAM2 call triggers model build + load onto CUDA (~2-3 s the
   first frame, subfaster thereafter).
4. An OpenCV window opens at 1280×720 showing the ZED left feed, a
   green mask on the Cheez-It box, and a blue crosshair centroid.
5. HUD text shows EKF mode and uncertainty.
6. Press `q` in the window to exit cleanly. A recording
   (`vs_dinov2_<timestamp>.mp4`) and metrics CSV
   (`metrics_<timestamp>.csv`) are saved into the project root.

No robot motion happens — the script passes `--no-robot`.
