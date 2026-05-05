# Dual-Glint Model: The Journey

Two IR LEDs, two corneal reflections, one gaze estimate. This doc traces every approach we
tried — what we expected, what broke, and what we kept.

---

## Hardware Context

- Head-mounted camera + two IR LEDs fixed relative to the eye
- LEDs are on either side of the eye — one right, one left
- PCCR = `pupil_center − glint_xy` (vector, in pixels)
- `swap_pccr=True` active throughout: camera is 90° rotated, so `(dx, dy)` axes are
  swapped — `dx` is the vertical axis in the image, `dy` is horizontal

```
swap_pccr ON  →  dx = vertical displacement (main signal)
                 dy = horizontal displacement
```

---

## Evolution Overview

```mermaid
flowchart TD
    A[Single GazeModel\n6-term poly, one glint\nmedian ~109px] --> B[DualGazeModel\ntwo 6-term models by LED side\nlabeled by switch_dx sweep]
    B --> C[Midpoint / Normalized-D\ntested, rejected\nworse than single]
    B --> D[TwoGlintGazeModel\n11-term joint poly\nboth PCCRs as features\nmedian ~45px]
    D --> E[SplitGlintModel v1\ntwo separate 6-term models\nselection: max|dx|\nper-glint]
    E --> F[SplitGlintModel v2\nselection: max|dy|]
    F --> G[SplitGlintModel v3\nselection: max magnitude\nhypot dx,dy]
    G --> H[SplitGlintModel v4\nalways right LED\nfallback left only\ncurrent default]
    H --> I[+Lorentzian LOO reweighting\nweighted least squares\ncurrent final]
```

---

## Stage 1 — Single GazeModel (baseline)

**What:** One `GazeModel` — 6-term polynomial `[1, dx, dy, dx·dy, dx², dy²]`.
Only the preferred-side glint used. Side assigned by `switch_dx` from a horizontal
sweep calibration: 5 dots at y=50%, sweep left→right, find where glint side flips.

**Why it works at all:** The PCCR is roughly monotonic with gaze angle. One glint is
enough to get a coarse estimate.

**Numbers (20260501_005039, 28 fixations):**
- `corr(dx, sx) = 0.896` — swap fix already in
- LOO median **109px**, mean 153px
- All samples `side=+1` — DualGazeModel's neg model never trained

**Failure mode:** Screen edges/corners. Only 28 points, polynomial extrapolates
badly at extremes. Corners blow up to 300–500px error.

---

## Stage 2 — DualGazeModel

**What:** Two independent `GazeModel` instances — `model_pos` (right LED, side=+1)
and `model_neg` (left LED, side=-1). At prediction: use whichever side is currently
visible. Shared `fallback` model for when sub-model is undertrained.

**Training change:** Emit two samples per fixation — one for each LED that was visible.
Doubles data. Both LED models now trained on correct per-LED geometry.

**Why it helps:** Each LED has a slightly different geometric relationship to the
cornea. Training each model on its own LED's data removes systematic cross-LED error.

**Failure mode:** Each sub-model only sees half the data. At prediction time you're
using only one PCCR vector — throwing away the other LED's information entirely.

---

## Stage 3 — Alternative Feature Tests (rejected)

Tested on `20260501_005039` (28 fixations):

| Method | mean | median | p75 | Verdict |
|--------|------|--------|-----|---------|
| Single preferred glint (baseline) | 153px | **109px** | 163px | ✓ kept |
| Glint midpoint `(g1+g2)/2` | 166px | 112px | 178px | ✗ worse |
| Midpoint ÷ D (normalised by interglint distance) | 171px | 111px | 181px | ✗ worse |
| 4D feature `[dx1, dy1, dx2, dy2]` | 178px | 150px | 237px | ✗ much worse |

**Why midpoint fails:** Midpoint normalisation by inter-glint distance D is a standard
technique for **remote** eye trackers where the user's head translates relative to
fixed LEDs — D changes with depth and compensates. Our setup is **head-mounted**:
LEDs are fixed relative to the eye camera. D barely changes (134–140px across all
fixations). Normalising by a constant is just a fixed scale factor the polynomial
already absorbs. No benefit.

**Why 4D feature fails:** At this stage we had only 28 samples. A 4D input space
with cross-terms → even more parameters to fit. Catastrophic overfit on LOO.

---

## Stage 4 — TwoGlintGazeModel (11-term joint poly)

**What:** Single model with 11-term polynomial using both PCCR vectors simultaneously
as features: `[1, dx1, dy1, dx2, dy2, dx1·dx2, dx1·dy2, dy1·dx2, dy1·dy2, dx1², dy2²]`.
Joint model trained only on frames where **both** glints were detected.

**Why it's better:** The cross-terms `dx1·dx2`, `dy1·dy2` capture the **differential
signal** between the two glints, which is a much stronger predictor of eye angle
than either PCCR alone. R² jumped from ~0.72 (DualGazeModel) to **~0.91**.

**Numbers (20260501_015920, 63 fixations):**
- LOO median **45px** (was 63px with DualGazeModel)
- Mean **64px** (was 105px)

**Why it was later replaced:** With only 20–68 samples in early recordings, the
11-term model overfits. LOO tail blows up: p90 = 753px on some recordings.
On LOO you remove one sample → fit on 19 → predict one point → catastrophic.
The separate 6-term models per LED are more robust with sparse data.

---

## Stage 5 — SplitGlintModel (two × 6-term models)

**What:** Two independent `GazeModel` instances:
- `model_right` — trained on `(dx1, dy1)` from the right LED
- `model_left` — trained on `(dx2, dy2)` from the left LED

Both trained **only on frames where both glints were present**.  
Prediction: run both models, pick one via a selection criterion.

Training filters applied:
- `|dx| < MIN_PCCR = 10px` → drop (glint near pupil centre = noise)
- Per-glint sigma filter: `|dy| > mean + 3σ` → drop
- Collection-time gate: `|dy1| > 120px OR |dy2| > 120px` → strip to single-glint

---

### Selection Criterion v1 — max |dx|

**Commit:** `c198f7c`

```python
if abs(dx1) >= abs(dx2):
    return model_right.predict(dx1, dy1)
else:
    return model_left.predict(dx2, dy2)
```

**Reasoning:** Larger |dx| = glint further from pupil vertically = higher SNR.
`|dx|` is the dominant axis (camera is 90° rotated, dx is the vertical displacement).

**Problem:** `|dx|` is coarse — doesn't account for `dy` contribution to signal quality.

---

### Selection Criterion v2 — max |dy|

**Commit:** `f529bfc`

```python
if abs(dy1) >= abs(dy2):
    return model_right.predict(dx1, dy1)
else:
    return model_left.predict(dx2, dy2)
```

**Reasoning:** `|dy|` showed better discrimination in practice for this dataset.

**Problem:** Still ignores the full 2D PCCR magnitude.

---

### Selection Criterion v3 — max magnitude (`hypot(dx, dy)`)

**Commit:** after `f529bfc`

```python
if (dx1**2 + dy1**2) >= (dx2**2 + dy2**2):
    return model_right.predict(dx1, dy1)
else:
    return model_left.predict(dx2, dy2)
```

**Reasoning:** Larger 2D magnitude = glint physically further from pupil = best SNR.

**Tested across 10 recordings:**

| rec | N | always-right | max-magnitude | oracle |
|-----|---|-------------|--------------|--------|
| 033307 | 20 | **132.8px** | 153.2px | 117px |
| 064810 | 47 | 90.5px | **88.9px** | 62px |
| 065115 | 21 | 219.9px | **154.1px** | 102px |

Mixed results. On some recordings it beats always-right; on others it's worse.
**Root cause:** Left LED model is under-trained and noisier (fewer clean samples,
larger magnitude variance: 34.8px std vs 12.2px for right). Any selection that
sometimes picks left adds noise on recordings where left model is weak.

**Flag kept:** `SplitGlintModel.USE_MAX_MAGNITUDE = False` — set True to restore.

---

### Selection Criterion v4 — Always Right, Fallback Left

**Current default** (`USE_MAX_MAGNITUDE = False`)

```python
if abs(dx1) >= MIN_PCCR:
    return model_right.predict(dx1, dy1)   # right LED always preferred
elif abs(dx2) >= MIN_PCCR:
    return model_left.predict(dx2, dy2)    # fallback: right near-zero
else:
    # both near-zero — use whichever has larger magnitude
    if (dx1**2 + dy1**2) >= (dx2**2 + dy2**2):
        return model_right.predict(dx1, dy1)
    return model_left.predict(dx2, dy2)
```

**Why this wins:**
- Right LED model is consistently better trained (more clean samples, lower variance)
- The few frames where left would genuinely be better (left glint much further from
  pupil) don't outweigh the many frames where switching to the weaker left model
  degrades accuracy
- `always-right` beats max-magnitude on 8/10 recordings in cross-validation

---

## Stage 6 — Lorentzian LOO Reweighting (current)

**What:** Before fitting, compute per-sample LOO residuals. Downweight samples with
high LOO error using a Lorentzian kernel:

```
w_i = 1 / (1 + (err_i / median_err)²)
```

Then fit with **weighted least squares** (scale rows by `√w_i`).

**Effect:** Samples in high-error regions don't just contribute less — their
influence on the polynomial shape is reduced proportionally. If a new calibration
point in that region gives a better prediction, it gains weight and the old
outlier's weight decays further.

**Numbers (88 samples, 87 after dropping off-screen outlier):**

| | mean | median | p75 | p90 |
|--|------|--------|-----|-----|
| Unweighted LOO | 43.5px | 28.0px | 46.5px | 83.0px |
| Weighted LOO | **31.6px** | **20.8px** | **31.6px** | **57.1px** |

67/87 samples improved. Mean −27%, p90 −31%.

**Weights thread through `SplitGlintModel`:**
- `fit(samples, weights=weights)` 
- `_sigma_filter(...)` returns `(kept, rejected, kept_weights)` — weights indexed correctly
- Each sub-model (`model_right`, `model_left`) gets its own weight subset

---

## What Didn't Work — Summary

| Approach | Why it failed |
|----------|--------------|
| Glint midpoint as PCCR | D constant in head-mounted rig — normalisation useless |
| Midpoint ÷ D | Same — D is a fixed scale, polynomial absorbs it |
| 4D joint feature (flat) | Too many params, too few samples → overfit on LOO tail |
| 11-term TwoGlintGazeModel | Same overfit problem at N<50; replaced by two 6-term |
| max\|dx\| selection | Ignores dy contribution to signal quality |
| max\|dy\| selection | Ignores dx contribution |
| max magnitude selection | Left model too weak on most recordings — hurts median |
| switch_dx sweep for labeling | switch_dx often ≈ 0 or not reliable; moved to always-right |

## What We Kept

- **SplitGlintModel** — two independent 6-term models, one per LED
- **Training:** both LEDs' samples used; per-glint sigma filter + collection gate
- **Selection:** always right LED (`model_right`), fallback to left only when `|dx1| < 10px`
- **Reweighting:** Lorentzian LOO weights → weighted least squares fit
- **Glint detection:** brightness-sorted, circularity-filtered, `min_dist_factor` gate
  to reject blobs inside the pupil area (tear film false positives)
