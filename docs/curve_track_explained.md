# curve_track — How It Works

## Overview

`curve_track` detects text lines on a book page by finding where ink is densest
across horizontal slices of the image. It works column-by-column, then connects
those density peaks across columns into continuous curves — one curve per text line.
The output is a set of thin curved polygon strips, one per detected line.

This is different from bounding-box approaches: instead of boxing individual
words, it tracks the *baseline* of each line across the whole page, so the
result follows the page's curvature naturally.

---

## Step 1 — Preprocessing

```
raw frame (BGR)
  → grayscale
  → CLAHE (contrast normalisation, tile-based)
  → Gaussian blur
  → adaptive threshold (BINARY_INV)  ← "bin_inv"
```

`bin_inv` is a binary image where **ink pixels are white (255)** and the white
page background is black. This inverts the usual convention so that summing
rows gives a count of ink pixels.

**Adaptive threshold** (not global) is used because book pages have uneven
illumination — brighter near the top, darker near the spine. A global threshold
would miss text in shadow regions. Adaptive threshold computes a local threshold
for each pixel based on its neighbourhood (`adaptive_block` × `adaptive_block`
pixels), then subtracts a constant `adaptive_c` to tune sensitivity.

---

## Step 2 — Strip Projection

The binarised image is divided into `strip_count` vertical strips (default 24),
each about `W / strip_count` pixels wide.

```
 strip 0   strip 1   strip 2  ...  strip 23
 |         |         |             |
 |  text   |  text   |  text       |
 |  line1  |  line1  |  line1      |
 |         |         |             |
 |  line2  |  line2  |  line2      |
 |         |         |             |
```

For each strip, the **horizontal projection profile** is computed: for every
row y, count the number of white (ink) pixels in that strip column.

```python
profile[y] = number of ink pixels in strip at row y
```

Where text lines cross a strip, the profile has a spike. Background rows have
near-zero counts.

The profile is then box-filtered (smoothed) with a window of `smooth_win`
samples to reduce noise from isolated ink dots.

---

## Step 3 — Peak Finding (NMS)

`_ct_find_peaks` finds the local maxima of the profile — each peak is a
candidate text-line centre for this strip.

Two filters are applied:

- **Min relative height** (`peak_min_height`): peak must be at least this
  fraction of the profile's maximum. Suppresses weak responses from noise or
  thin hairlines.
- **Min distance** (`peak_min_dist`): two peaks closer than this many pixels
  are de-duplicated (non-maximum suppression). Prevents double-detection of
  thick lines.

Result: a small list of y-coordinates, one per text line passing through this strip.

---

## Step 4 — Greedy Track Association

This is the core of `_ct_detect_tracks`. Strips are processed left to right.
A **track** is a list of `(x, y)` points — one point per strip where that
text line was detected.

For each new strip's peaks, the algorithm tries to extend existing tracks:

```
for each existing track:
    find the peak in this strip closest to track's last y
    if distance <= y_tol:
        extend track with this peak
    else:
        mark track as "missed this strip"

for any peaks not claimed by any track:
    start a new track
```

`track_max_gap` lets a track survive N consecutive strips with no matching peak (e.g. over a margin or illustration). Tracks shorter than `min_track_len` are discarded at the end.

Tracks are sorted by mean y so output index 0 = topmost line.

### Outlier filtering (three layers)

**Important:** adaptive threshold converts shadow/hand boundaries into thin
horizontal edge artefacts — same FWHM as real text — so simple shape tests
are not enough. Three layers work together:

**Layer 1 — FWHM filter (Pass 2 pre-filter)**
Compute each peak's FWHM (full-width at half-maximum in the profile). The
global median FWHM across all peaks in the frame is the reference line height.
Peaks wider than 2.5× the median are dropped — genuine wide blobs that
adaptive threshold didn't convert to thin edges.

**Layer 2 — Neighbour support (Pass 2 pre-filter)**
After FWHM filtering, a peak must also have at least one peak within `y_tol`
in the adjacent strip (s−1 or s+1). Removes single-strip spikes and trims the
boundary strips of any object that enters/exits the frame mid-page.

**Layer 3 — Inter-spacing post-check**
After track assembly, sort tracks by mean y and compute consecutive gaps.
Any pair of tracks closer than 55% of the median gap is flagged as a shadow
edge pair (adaptive threshold produces one peak at each edge of a shadow band,
creating two closely-spaced tracks). The shorter track of the pair is dropped.

```
real lines:   y=80, 160, 240   gaps=[80,80]  median=80  threshold=44
shadow pair:  y=294, 327       gap=33  →  33 < 44  →  drop shorter one
```

`min_track_len` remains the final gate — for partial-width hands/shadows that
survive all three layers, raise this value.

---

## Step 5 — Polynomial Fit → Polygon Strip

Each track is a sparse sequence of `(x, y)` samples (one per strip). To get
a smooth curve and a proper polygon:

1. **Fit a polynomial** (degree 1 if <5 points, degree 2 otherwise) through
   the track points using `np.polyfit`.
2. **Evaluate the polynomial** at every x from the track's left edge to its
   right edge → a dense smooth baseline curve `(x, y_centre)`.
3. **Offset up and down** by `half_h = peak_min_dist // 2` pixels to create
   a top edge and a bottom edge.
4. **Concatenate** top edge + reversed bottom edge → closed polygon (Quad).

```
top edge →  ___________
           /           \
bot edge ← ‾‾‾‾‾‾‾‾‾‾‾
```

The result is a thin curved strip polygon that tightly hugs each text line,
following page curvature.

---

## Page Split (`split_pages`)

When a book is open flat, the camera sees two pages side by side with a dark
spine/gutter in the middle. Without split handling, a single text line on the
left page and a different line at the same y on the right page could be
mistaken for one continuous line.

`split_pages=True` enables `_ct_find_page_split`, which detects the spine:

### How `_ct_find_page_split` works

1. **Vertical projection**: sum `bin_inv` column-wise → `vprof[x]` = total ink
   at column x across the full height.
2. **Smooth** with a wide kernel (≈ `W/50` pixels) to get a coarse density profile.
3. **Search near the centre** (±18% of image width around `W/2`) for the
   minimum — the spine gutter has almost no ink, so it's a deep valley.
4. **Validate**: the valley must be ≤ 72% of the median density of its
   neighbours. If the dip isn't deep enough, it's not a real spine and
   `None` is returned (no split).

```
ink density (vertical projection, smoothed):

  ████████████  ░░░░  ████████████
  left page     spine  right page
                  ↑
              valley_idx = split_x
```

### What happens after split is detected

The binarised image is cut at `split_x` (with a small padding to exclude the
dark spine itself). `_ct_detect_tracks` is run independently on the left half
and the right half, each with `strip_count / 2` strips. The x-coordinates from
the right half are offset by `split_x + pad` so they map back to the original
frame.

This means:
- Left page lines get polygon strips confined to the left half
- Right page lines get polygon strips confined to the right half
- No cross-page track associations possible

---

## Parameter Reference

| Param | Effect |
|-------|--------|
| `clahe_clip` | CLAHE contrast limit. Higher = more aggressive contrast stretch in dark areas. Too high → noise amplification. |
| `blur_ksize` | Pre-blur before adaptive threshold. Larger = smoother binarisation, fewer speckles. |
| `adaptive_block` | Neighbourhood size for adaptive threshold. Should be larger than the tallest letter. |
| `adaptive_c` | Subtracted from local mean before thresholding. Higher = only darker ink gets picked up. |
| `strip_count` | Number of vertical slices. More strips = finer x-resolution of the curve, but slower. |
| `peak_min_height` | Fraction of max profile value a peak must reach. Raise to suppress weak/partial lines. |
| `peak_min_dist` | Minimum gap (px) between two peaks in a strip. Roughly = inter-line gap. |
| `y_tol` | Max y drift (px) between adjacent strips for a track to continue. Raise for heavily curved pages. |
| `track_max_gap` | How many strips a track can skip (e.g. over a figure). |
| `smooth_win` | Profile smoothing window. Larger = less noise, but can merge adjacent lines. |
| `min_track_len` | Minimum strips a track must span to be kept. Raise to suppress short spurious detections outside the book. |
| `split_pages` | Auto-detect book spine and process each page half independently. |

---

## Data Flow Summary

```
BGR frame
  │
  ▼  grayscale + CLAHE + blur + adaptive threshold
bin_inv  (ink = white)
  │
  ├─ [if split_pages] _ct_find_page_split()
  │     vertical projection → valley near centre → split_x
  │
  ▼  _ct_detect_tracks() [per half if split]
  │   for each strip left→right:
  │     horizontal projection → smooth → NMS peaks
  │     greedy assign peaks to existing tracks (y_tol)
  │     start new tracks for unmatched peaks
  │   discard short tracks (min_track_len)
  │
  tracks[]  [(x,y), (x,y), …] per line
  │
  ▼  polyfit → evaluate → offset ±half_h
  │
  Quad polygons[]  (curved strip per line)
```
