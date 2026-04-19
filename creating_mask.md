# Creating Object Masks — Methodology & Design Notes

This document explains `create_masks.py` in `localization_for_visual_servoing/`:
the pipeline that turns raw photos in `objects/` into tight, transparent‑background
PNG cutouts in `masked_objects/`.  It records *why* each design decision was made,
what assumptions the code bakes in, what the known limitations are, and which
kinds of new images it is expected to handle or fail on.

---

## 1. What the script does, in one sentence

For every image in `objects/*.jpeg`, run SAM2 with a centre‑biased prompt,
select and refine the resulting box mask, post‑process away residual backdrop
leaks, and save an RGBA PNG cropped to the box's bounding box (background fully
transparent) to `masked_objects/<name>.png`.

---

## 2. Starting point and problem statement

**Existing artefact.**  The `masked_objects/` folder already contained PNGs for
each object image, but they were not true cutouts:

- The alpha channel was either fully opaque or (where a mask existed) only
  loosely covered the box — visible cloth / floor showed through.
- The saved file size was small because a white background had been composited
  in, not a transparent one.  That background was always baked into the image
  data rather than left transparent.

**Goal.**  Produce a PNG per object with:

1. A *tight* foreground selection — pixels that actually belong to the box
   are opaque.
2. A *true* transparent background — anything that isn't the box has
   `alpha = 0`.
3. The image cropped to the bounding box of the mask so downstream code does
   not have to handle a giant empty canvas.

**Consumer.**  The masked PNGs feed into `FoundationModel/dinov2_match_segment.py`
and `sam2_tracking_method.py`, which both consume RGBA reference images where
the alpha channel defines which patches to match (see
`ReferenceCache.ref_mask_patches` in `sam2_tracking_method.py`).  That pipeline
expects pixel‑accurate alpha because it builds per‑patch DINOv2 feature
averages out of it — loose or leaky masks pollute the feature bank with
backdrop features and hurt detection performance.

---

## 3. Dataset and structural assumptions

Every decision in the script exploits one or more of these invariants in the
`objects/` dataset:

1. **A single box object per image.**  No clutter, no duplicates.  We never
   have to pick among multiple foreground candidates.
2. **The box is roughly centred horizontally.**  It is always placed near the
   image centre by the photographer.  The exact centre may not be on the box
   (e.g. small tofu box sits in the lower half) but the horizontal column
   through the image centre always intersects the box.
3. **Backgrounds are one of two types:**
   a. A dark draped cloth (commonly a black photography backdrop) covering the
      upper portion of the frame.
   b. A wooden tabletop (warm, desaturated, bright) covering the lower
      portion.
   Often both appear in the same frame — cloth above, wood below — with the
   box sitting on the wood in front of the cloth.
4. **The image corners and image borders are always backdrop.**  The box
   never touches the image frame.
5. **The box is the most prominent foreground object.**  SAM2's own scoring
   will usually rank a box‑covering mask highest among its candidates.

The script does not work (or works poorly) if any of these assumptions is
violated — see §8 (Limitations).

---

## 4. Why SAM2 over simpler alternatives

Choices considered:

| Option | Why rejected |
|---|---|
| HSV / GrabCut thresholding | The backgrounds vary (black cloth, warm wood, dark chair, etc.).  Single‑image thresholding works for some boxes and not others; GrabCut needs a seed mask nearly as good as what we want as output. |
| `rembg` (U2Net‑based) | Not installed on the target environment and requires an extra dependency.  We already had SAM2 checkpoints and a Python env configured. |
| Classical contour / edge detection | Boxes frequently have low contrast with the wooden table surface (light cardboard) or with the dark cloth (brownie box).  Edge maps are noisy and geometry‑dependent. |
| **SAM2 (chosen)** | Checkpoint `sam2.1_hiera_large.pt` already exists in `third-party/sam2/checkpoints/`.  The `semvs` conda env already imports it (see `FoundationModel/sam2_tracking_method.py`).  SAM2 excels at prompt‑driven object segmentation with weak prompts and is robust to background variation. |

Using SAM2 lets us lean on a strong prior trained on billions of mask–image
pairs and focus our custom logic on:

- *Prompting* SAM2 correctly so it segments the box rather than a sub‑region.
- *Selecting* the right candidate mask out of the 3 it returns.
- *Post‑processing* small residual errors that SAM2 itself cannot catch.

---

## 5. Pipeline overview

```
image (BGR)
   │
   ▼
┌──────────────────────────────────────────────┐
│  Try several prompt configs on SAM2          │   § 6.1
│     (wide / medium / tight)                  │
│  For each config:                            │
│     predict_single → 3 candidate masks       │
│     keep largest CC containing centre        │
│     drop degenerate (< 2% or > 92% area)     │
│     track best‑scoring valid candidate       │
└──────────────────────────────────────────────┘
   │
   ▼ best mask (still may have backdrop leaks)
┌──────────────────────────────────────────────┐
│  strip_backdrop_leaks                        │   § 6.2
│     find HSV‑dark‑grey connected components  │
│     drop CC‑inside‑mask portions that        │
│       (a) span mask boundary                 │
│       (b) reach the image border outside     │
│       (c) are <= 18 % of mask area           │
└──────────────────────────────────────────────┘
   │
   ▼
┌──────────────────────────────────────────────┐
│  refine_mask  (close gaps, fill holes)       │   § 6.3
└──────────────────────────────────────────────┘
   │
   ▼
┌──────────────────────────────────────────────┐
│  to_rgba  (attach alpha, crop to bbox)       │   § 6.4
└──────────────────────────────────────────────┘
   │
   ▼
masked_objects/<name>.png  (RGBA)
```

---

## 6. Stage‑by‑stage design

### 6.1 Prompting SAM2

SAM2's `SAM2ImagePredictor.predict` accepts three prompt types:
`point_coords + point_labels`, `box`, and `mask_input`.  It returns 3 candidate
masks with per‑mask quality scores.  Our job is to choose prompts that hint
*"the object is here and reaches roughly this extent"* without over‑specifying
its shape.

#### 6.1.1 Positive / negative points

`center_prompts(h, w, y_spread)` places:

- **5 positive points on the central vertical column** — at the image centre
  and at `cy ± y_spread·h/2` and `cy ± y_spread·h/4`.  Multiple points spread
  vertically force SAM2 to group the whole tall box into one mask instead of
  latching onto a single high‑contrast panel (we lost the entire
  brownie box early on when only one centre point was provided — SAM2 isolated
  just the green brownie photo because that was the most coherent colour
  region under a single point).
- **2 horizontal neighbour positives** at `(cx ± w/8, cy)` — gives SAM2 a
  small horizontal extent hint, reducing the chance it returns a thin vertical
  sliver.
- **4 corner negatives** at the image corners with an 8 px margin — these are
  *always* backdrop in this dataset (§3, assumption 4), and explicitly
  negative prompts improve SAM2's precision at the mask edge.

> **Design choice.**  Positives live on the central column, not in a small
> cross around the centre, because boxes in the dataset are usually *tall*
> (stuffing box, cheez‑it, TRUBAR, etc.).  A tight cross under‑constrains
> the vertical extent and SAM2 frequently truncated tall boxes.

#### 6.1.2 Box prompt

`center_box_prompt(h, w, x_frac, y_frac)` computes a loose bounding box that
excludes the outer margins of the image:

```python
[x_margin, y_margin, w - x_margin, h - y_margin]
```

The box prompt is strictly weaker than a pixel‑accurate bbox — it just tells
SAM2 *"look within this region"*.  Adding it increased per‑image mask scores
from ~0.77 avg (points only) to ~0.93 avg (points + box).  Intuition: SAM2's
box prompt embedding suppresses candidate masks that grow into the margins
(which are backdrop in this dataset).

#### 6.1.3 Multi‑config strategy

A single `(y_spread, x_frac, y_frac)` triple cannot cover the whole dataset:

- Images like `brownie_box.jpeg` or `cheez_it_box.jpeg` have the box
  occupying ~90 % of the image height.  They need a **wide** prompt so that
  positive points are distributed across the full box and the box prompt
  doesn't cut off the object's top or bottom.
- Images like `tofu_box.jpeg` have a small box in the lower half of the
  image, surrounded by cloth above and wood below.  Under a wide prompt, the
  upper positive points land on the cloth and SAM2 segments the cloth + box
  together.

We therefore try three prompt configurations (`PROMPT_CONFIGS`):

| name   | y_spread | x_frac | y_frac |
|--------|---------:|-------:|-------:|
| wide   | 0.50     | 0.15   | 0.08   |
| medium | 0.30     | 0.20   | 0.15   |
| tight  | 0.15     | 0.28   | 0.25   |

For each, SAM2 returns 3 candidate masks; for each candidate we:

1. Keep only the connected component containing the image centre (§ 6.1.4).
2. Reject degenerate candidates (`area < 2 %` of image — empty — or
   `> 92 %` — the background).
3. Track the candidate with the highest SAM2 score among those that passed.

The overall winner across all three configs is chosen.

> **Why use SAM2's score as the selection criterion?**  It is the most direct
> proxy for how well the mask matches SAM2's internal object prior, and on
> this dataset it correlates well with visual quality (the `wide` config
> wins on tall boxes, `medium` wins on small lower‑half boxes, etc.).  It is
> not perfect — see § 8.

#### 6.1.4 Largest‑component‑containing‑centre

Even SAM2's best mask occasionally returns two disjoint blobs (object + a
stray edge of cloth).  `largest_component_containing(mask, centre)` uses
`cv2.connectedComponentsWithStats`:

- If the centre pixel is inside a labelled CC → take that CC.
- Otherwise → take the largest non‑background CC (centre fell on background
  for small objects).

This leverages the "box is near centre" assumption (§3.2) in a soft way that
still works when the centre technically isn't *on* the box.

### 6.2 Stripping backdrop leaks

Even with a good prompt, SAM2 sometimes produces a mask that extends slightly
into the backdrop.  This happens when the backdrop colour is similar to a box
edge (low contrast) or when draped cloth creates folds that SAM2 treats as
part of the object.  The most visible failure was the tofu box: SAM2 pulled
~1.5 % of the image worth of black cloth into the mask directly above the
object.

`strip_backdrop_leaks` implements the following logic:

1. Compute a backdrop binary mask over the whole image:
   `(V < 65) & (S < 60)` in HSV.  Rationale:
   - `V < 65` = the pixel is dark (black cloth, wood shadow).  The wooden
     table itself is bright (`V ≈ 160–200`), so the low‑V threshold targets
     cloth and shadow.
   - `S < 60` = the pixel is mostly grey / black (saturation close to zero).
     A dark navy box panel has `S ≫ 60` on average and is excluded.
2. Apply a 7×7 ellipse morphological closing on the backdrop mask to bridge
   thin gaps (e.g. text overlays, fabric highlights) so a continuous drape
   plus any thin intrusion into the foreground form a single connected
   component.
3. Find connected components.  For each CC:
   - Skip if area < 500 px — irrelevant noise.
   - Compute `inside = CC ∩ mask` and `outside = CC \ mask`.  If either is
     too small, skip (the CC doesn't straddle the mask boundary and so
     isn't a leak candidate).
   - The outside portion must intersect the **image border** (top / bottom /
     left / right pixels).  This is the characteristic property of true
     backdrop — cloth and table always reach the frame edge.  A deep interior
     dark region of a box (e.g. a chocolate photo on the front panel) is
     surrounded by mask pixels and never touches the border.
   - The inside portion must be `<= 18 %` of the current mask area.  This
     was the critical safeguard against eroding the entire navy half of the
     TRUBAR protein‑bar box: that panel's dark navy pixels partially satisfy
     the HSV backdrop threshold and form a large cross‑boundary CC with the
     adjacent cloth, but the CC‑inside portion is ~21 % of the mask —
     above the threshold.  Real cloth leaks in this dataset are uniformly
     < 10 % of mask area.
4. Subtract the accumulated `to_drop` pixels from the mask, close small
   resulting nicks with a 5×5 closing, and re‑apply
   `largest_component_containing` in case the subtraction disconnected the
   mask.

> **Why these exact numbers?**  They were tuned against the 14‑image
> dataset.  Empirical HSV statistics on the tofu cloth showed `mean V=43,
> mean S=45`; TRUBAR's navy panel had `mean V=104, mean S=68` overall but
> individual dark‑navy pixels with `V<65 ∧ S<60` form ~60 % of the panel.
> The combination of `V<65`, `S<60`, border‑touch, and the 18 % size cap is
> what makes the rule specific enough to remove the tofu leak without
> eroding the navy panel.  See § 9 for how to re‑tune.

### 6.3 Morphological refinement (`refine_mask`)

Applied last, regardless of whether `strip_backdrop_leaks` removed anything:

- **7×7 close, 2 iterations** → seals 1–2 px gaps at the box edge.
- **7×7 open, 1 iteration** → removes 1–2 px speckles.
- **Contour fill** → any internal holes (e.g. small bright specular
  highlights that SAM2 excluded from the mask) are filled so the alpha
  channel is fully opaque over the box's visible extent.

This stage is pure bookkeeping; it never changes the overall shape of the
mask by more than a few pixels.

### 6.4 RGBA output and cropping (`to_rgba`)

The BGR image is converted to BGRA and the mask is written into the alpha
channel.  Then, unless `--no-crop` was passed, the result is cropped to the
tight bounding box of the mask with a 4 px padding on each side.

Why crop by default?
- It keeps saved file sizes small (~300–800 KB instead of several MB).
- It matches how `ReferenceCache` in `sam2_tracking_method.py` consumes the
  reference — it uses the full image dimensions as‑is.
- It provides a visually useful file that can be inspected in an image
  viewer.

Use `--no-crop` when the downstream consumer expects the mask to live in the
exact pixel coordinates of the original image.

---

## 7. Current results (14 images)

All scores above 0.89 except tofu (0.82) which is a legitimately harder case
because the box occupies only ~12 % of the frame.

| image              | cfg    | SAM score | mask % |
|--------------------|--------|----------:|-------:|
| amazon_tissue_box  | medium | 0.941     | 26.1   |
| baking_mix_box     | wide   | 0.895     | 35.6   |
| brownie_box        | wide   | 0.937     | 32.3   |
| cake_box           | wide   | 0.962     | 29.8   |
| cardboard_box      | wide   | 0.939     | 29.9   |
| cheez_it_box       | wide   | 0.912     | 28.9   |
| lamp_box           | wide   | 0.960     | 43.4   |
| mac_and_cheese_box | wide   | 0.948     | 29.0   |
| mashed_potatoes    | wide   | 0.896     | 29.1   |
| protein_bar        | wide   | 0.935     | 33.3   |
| protein_bar_2      | wide   | 0.959     | 32.1   |
| protein_bar_3      | wide   | 0.956     | 28.4   |
| stuffing_box       | wide   | 0.900     | 43.0   |
| tofu_box           | medium | 0.819     | 12.0   |

Visual inspection confirms:
- **Tight boundaries** on all boxes.
- **Clean alpha channel** — the previously baked‑in white background is gone.
- **Dark‑panel boxes preserved** — TRUBAR navy, brownie chocolate photo,
  and the dark top design of the cardboard shipping box are all retained.
- **Backdrop leaks removed** — the tofu box's ~1.5 % cloth protrusion is
  gone, leaving a small spur at the top that is actually the box's own
  top face in shadow (not backdrop).

---

## 8. Limitations and failure modes

### 8.1 Dataset‑dependent assumptions

- **Object must be centred** (§3.2).  A box photographed in the corner will
  fail `largest_component_containing` (no valid centre CC → falls back to
  largest CC, which may be the backdrop itself).  *Fix:* replace the
  centre‑based seed with an object detector bbox or let the user click.
- **One object per image.**  Two boxes in the same frame will end up in a
  single mask or only the one containing the centre will survive.  *Fix:*
  loop over detected seed points and write one PNG per instance.
- **Image corners must be backdrop.**  If a box has extended corners (long
  thin object filling the frame diagonally), the corner negative prompts
  will contradict SAM2's segmentation.  *Fix:* detect mask coverage and
  drop negative prompts when the mask reaches a corner.

### 8.2 Backdrop‑stripping heuristics

- **HSV thresholds are tuned for black cloth + wood floor.**  If a future
  image has a dark blue cloth, a bright white wall, or a glass tabletop,
  the thresholds `V<65, S<60` will either fire in unintended places or
  miss genuine leaks.  *Fix:* replace the HSV heuristic with an image‑specific
  colour model (e.g. fit a Gaussian over the 4 corner patches and define
  backdrop as pixels within 2σ of that distribution).
- **The 18 % cap is dataset‑specific.**  It was chosen to fit between tofu's
  leak (~13 % of mask) and TRUBAR's navy‑as‑backdrop false positive
  (~21 % of mask).  If a new image has a cloth leak that is, say, 25 % of
  the mask, this rule will refuse to strip it.  If a new box's dark panel
  is ~17 % of the mask, the rule will strip it by mistake.  *Fix:* add a
  colour‑coherence check (e.g. require the inside‑portion saturation
  distribution to match the outside‑portion) or use GrabCut with the SAM
  mask as a seed to get a proper per‑pixel colour model.
- **Small leaks that don't touch the image border go undetected.**  A
  box photographed on a very large cloth where the cloth's dark CC doesn't
  reach the frame (e.g. a tight close‑up) will not trigger the stripper.
  *Fix:* also flag CCs whose outside portion is "statistically backdrop‑like"
  even without frame contact.

### 8.3 SAM2 prompt strategy

- **Score‑based selection assumes SAM2's score reflects visual quality.**
  On the current dataset it does, but this is not guaranteed — SAM2's
  quality head occasionally prefers a sharper but semantically wrong mask.
  *Fix:* add a penalty proportional to the fraction of mask pixels that
  fall under the HSV backdrop threshold; prefer lower‑penalty masks when
  scores are within 0.05.
- **`PROMPT_CONFIGS` has only 3 entries.**  Very tall narrow objects
  (e.g. a slender protein bar standing upright) might prefer a config with
  `x_frac=0.35, y_spread=0.6`.  None of the current boxes look like that,
  but adding a fourth entry is cheap if needed.
- **No prompt covers *centre‑off‑object* cases.**  A box whose centre is in
  a window cut‑out (e.g. a box with a see‑through hole in the front) could
  put the positive point on the backdrop.  *Fix:* validate prompts by
  checking the pixel colour under each positive point before calling
  SAM2 — if a positive is too dark+desaturated, shift it by a small offset
  until it lands on a colourful pixel.

### 8.4 Output format

- **Cropped PNGs lose the absolute image coordinates.**  Any downstream
  consumer that needs to know where the box was in the original photo
  should use `--no-crop` (or we should also save the bbox in a sidecar
  JSON).  `ReferenceCache` in the existing code doesn't need this but a
  future localisation pipeline might.

---

## 9. Re‑tuning and extension

### Running on new images

```bash
conda run -n semvs python localization_for_visual_servoing/create_masks.py \
    --debug
```

Debug overlays land in `masked_objects/debug/` with the SAM2 score annotated
and the mask contour drawn in red.  This is the fastest way to diagnose a
bad result.

### Tuning knobs (most impactful first)

1. `PROMPT_CONFIGS` — add a new `(name, y_spread, x_frac, y_frac)` row if a
   new image class is poorly served by the three existing configurations.
2. `v_thresh, s_thresh` in `strip_backdrop_leaks` — raise them if the
   backdrop is lighter; lower them if dark box panels are being eroded.
3. `max_drop_frac` — raise it if known‑good leaks aren't being removed
   because they are larger than 18 % of the mask; lower it if large dark
   box panels are being eaten.
4. `refine_mask` kernel sizes — raise to 9×9 or 11×11 on very high
   resolution images where a 7×7 kernel is proportionally too small.

### Regression testing

There is no automated visual diff.  The fastest manual check is:

```bash
# Run with debug overlays on
conda run -n semvs python localization_for_visual_servoing/create_masks.py \
    --debug

# Then flip through masked_objects/debug/*.png in an image viewer
```

---

## 10. Code pointers

- `create_masks.py::predict_box_mask` — top‑level SAM2 prompting loop.
- `create_masks.py::strip_backdrop_leaks` — backdrop‑leak post‑processor.
- `create_masks.py::refine_mask` — morphological cleanup + hole fill.
- `create_masks.py::to_rgba` — RGBA packing + bbox crop.
- `third-party/sam2/checkpoints/sam2.1_hiera_large.pt` — SAM2 weights.
- `FoundationModel/sam2_tracking_method.py` — downstream consumer that
  reads the produced RGBA PNGs as reference images for the DINOv2 +
  SAM2 tracking servo.

---

## 11. Version history

- **v1** — single centre‑point prompt; failed on brownie_box (segmented
  only the green brownie photo).
- **v2** — centre cluster of positive points + corner negatives.  Worked
  for most images; failed on cardboard and brownie (complex dark designs).
- **v3** — added a loose box prompt alongside the points.  Average SAM2
  score jumped from 0.77 to 0.93.  Brownie fixed.  tofu still had cloth
  leak.
- **v4** — multi‑config prompting (wide / medium / tight) with
  score‑based selection.  Fixed tofu's score but left a visible cloth
  blob above the box.
- **v5** — added `strip_backdrop_leaks` with the border‑touching + 18 %
  cap rule; final current version.  Tofu cloth blob removed, TRUBAR
  navy preserved.
