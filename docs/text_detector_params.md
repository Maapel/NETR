# Text Detector — Parameter Reference

All parameters exposed in `tools/text_tuner.py`. Organised by method.

---

## Line Detection Method (`line_method`)

| Value | Description |
|-------|-------------|
| `mser_global` | MSER regions → group into lines using merge/docstrum/word_chain |
| `gap_scan` | White inter-line gaps → strip borders |
| `curve_track` | Strip projection peaks, follows page curvature |
| `local_patch` | Splits image into N×N patches, runs `mser_global` per patch |

---

## Shared Pre-processing (all methods)

| Parameter | What it does |
|-----------|-------------|
| `clahe_clip` | CLAHE contrast limit. 0 = off. Use 2–4 for dark or low-contrast images (book photos, dim lighting). Dramatically improves MSER region count on poorly-lit frames. |
| `clahe_tile` | CLAHE tile grid size (px). Smaller = more local contrast boost. 8 is a good default; reduce to 4 for very uneven lighting. |
| `blur_ksize` | Gaussian blur kernel before MSER (0 = off, must be odd). Smooths noise at the cost of fine detail. |
| `adaptive_thresh` | Enable local binarisation before MSER for non-uniform lighting. |
| `adaptive_block` | Block size for adaptive threshold (must be odd). Larger = smoother local regions; smaller = more local contrast. |
| `adaptive_c` | Constant subtracted from local mean in adaptive threshold. Higher = only keep very dark text; lower = picks up faint text too. |
| `binarize` | Otsu threshold after CLAHE+blur — cleaner binary input for MSER. |

---

## MSER Region Filtering (all MSER-based methods)

| Parameter | What it does |
|-----------|-------------|
| `min_area` | Minimum bounding-box area (px²) to keep an MSER region. Raise to discard tiny noise dots. |
| `max_area` | Maximum bounding-box area (px²). Lower to reject large non-text blobs (hands, page edges). |
| `min_aspect` | Min w/h ratio. 0.2 means height can be at most 5× width. Filters near-vertical strokes and serif noise. |
| `max_aspect` | Max w/h ratio. Filters very wide flat blobs that are unlikely to be single characters. |
| `max_line_angle` | Reject text lines whose orientation is more than this many degrees from horizontal. 45° allows tilted pages but rejects vertical noise. |
| `nms_iou` | IoU threshold for quad deduplication (NMS). Lower = more aggressive — overlapping quads merged. Mainly affects `local_patch` seam duplicates. |

---

## Line Grouping — `mser_global` (merge method)

| Parameter | What it does |
|-----------|-------------|
| `skew_method` | How to estimate dominant page skew. `projection` sweeps angles, picks highest histogram variance (accurate, ~10ms). `nn` takes median nearest-neighbour angle (fast, ~2ms, less robust). |
| `merge_y_overlap` | Y-band tolerance for grouping regions into the same line, as a fraction of median char height. 0.5 = regions whose centres are within ½ char height are on the same line. |
| `merge_x_gap` | Fixed pixel gap: adjacent regions this far apart horizontally are still merged into the same line. Lower bound; `x_gap_scale` may widen it further. |
| `x_gap_scale` | Scale-invariant word gap = max(`merge_x_gap`, `x_gap_scale` × median char height). Adapts automatically to zoom/DPI so word spaces are bridged at any resolution. |
| `min_line_len` | Minimum number of regions required to output a line. Raise to suppress single- or two-character noise lines. |

---

## Line Grouping — `mser_global` (docstrum method)

| Parameter | What it does |
|-----------|-------------|
| `docstrum_k` | Number of nearest neighbours used to estimate each region's local text direction. Higher = more robust on sparse text; lower = faster. |
| `docstrum_angle_tol` | Max angle deviation (°) between a region's smoothed orientation and a candidate edge direction. Smaller = stricter grouping (fewer false merges). |
| `docstrum_ht_ratio` | Height compatibility ratio: two regions only connect if their heights differ by less than this factor. Prevents linking capitals to sub/superscripts across lines. |
| `docstrum_max_dist` | Maximum kNN edge length in units of median char height. Raise for widely spaced text; lower to prevent inter-line connections. |
| `docstrum_min_len` | Minimum regions in a connected component before it becomes a line. Filters isolated noise blobs. |
| `chain_break_dist` | Break chain if gap between consecutive boxes exceeds N × median char height. 0 = off. Use 3–6 to split chains that jump across word space or gutter. |
| `chain_merge_dist` | Merge chain endpoints within N × median char height if angle is compatible. 0 = off. Use 3–8 to bridge word spaces and stitch same-line fragments. |
| `chain_merge_gap` | How far (in units of median char height) the direction ray from a chain's tail can reach to merge with an adjacent chain's hull. Raise to bridge larger word spaces. |

---

## `gap_scan` — Inter-line Gap Detection

Finds horizontal strips of bright (white) pixels between text lines and uses their edges as line borders. Works well for clean scans on white paper. Fails on camera images with shadows or dark backgrounds.

| Parameter | What it does |
|-----------|-------------|
| `gap_threshold` | Normalised row brightness (0–1) above which a row is a gap. 0.75 for clean scans; lower if gaps are not bright enough. |
| `gap_min_h` | Minimum consecutive bright-row height (px) to count as a real gap. Raise to ignore narrow bright stripes from descenders or noise. |

---

## `local_patch` — Patch Grid

Splits the image into an N×N grid with 10% overlap, runs `mser_global` independently on each patch with its own skew estimate. Good for books with page curl or two-page spreads where global skew estimation fails.

| Parameter | What it does |
|-----------|-------------|
| `patch_grid` | Split into N×N patches. Use 2–4 for books with curl or two-page spreads. |
| `skew_method` (per patch) | Applied independently per patch. `projection` accurate but slower; `nn` faster for many small patches. |
| `split_pages` | Process left and right page halves separately (`word_chain` grouping only). |

---

## `curve_track` — Strip Projection Peaks

Slices the image into vertical strips, computes a horizontal projection (row-sum of inverted binary image) per strip, finds peaks in each strip's projection, then links peaks across strips into tracks. Handles page curvature and tilted books naturally.

**Best choice for live book reading** — handles two-page spreads, page curl, and camera angle without requiring global skew correction.

| Parameter | What it does |
|-----------|-------------|
| `strip_count` | Number of vertical strips to slice into. More strips = finer curvature tracking; fewer = faster and more robust on noisy images. |
| `peak_min_height` | Min peak height as a fraction of the strip's max projection value. Raise to suppress weak/partial lines; lower to pick up faint text. |
| `peak_min_dist` | Minimum y-distance between two peaks in the same strip. Should be roughly the inter-line gap in pixels. Too small = double-detects one line; too large = merges adjacent lines. |
| `y_tol` | Max y-drift allowed between a track's last peak and a candidate peak in the next strip. Controls how much the tracked line centre can shift per strip — set to roughly the peak shift expected from page curvature. |
| `track_max_gap` | Max consecutive strips a track can be absent before it is considered dead. Raise for images where some strips have no text (margins, illustrations). |
| `smooth_win` | Box-filter kernel for smoothing the per-strip projection before peak finding. Larger = smoother, less noise; smaller = sharper peaks, more local. |
| `min_track_len` | Minimum strips a track must span to be kept. Raise to suppress short spurious tracks from noise or margin annotations. |
| `split_pages` | Process each page half separately (useful for two-page spreads). |
| `book_mask` | Detect the bright page region (bright background + dark text blobs) and discard any detected lines whose centre falls outside it. Prevents detections bleeding onto desk, hands, or background. |
| `stabilize_lines` | Optical flow stabilization: warp previous frame's tracks onto the current frame using LK sparse flow + RANSAC homography. Reduces jitter when the book or camera moves slightly. Off by default. |

---

## Spine Detection (`curve_track` only)

Detects the book spine/gutter to optionally split processing between left and right pages.

| Method | Description |
|--------|-------------|
| `column_sum` | Vertical projection valley — fast, works well for straight-on camera angle. |
| `rotated_proj` | Sweeps ±`spine_angle_range`° from vertical, finds the angle with the sharpest valley. Handles tilted books. **Default.** |
| `line_perp` | Spine perpendicular to the average tracked line direction — most accurate, auto-angle, no sweep needed. |
| `none` | No spine detection. |

| Parameter | What it does |
|-----------|-------------|
| `spine_search_frac` | ±fraction of image width to search for spine (centred on image mid). 0.35 = middle 70% of image. Increase if spine is off-centre. |
| `spine_valley_thresh` | Valley/neighbour density ratio threshold. Valley must be below this fraction of its neighbours. 0.85 = valley must be ≥15% below neighbours. Lower = stricter (fewer false positives). |
| `spine_angle_range` | `rotated_proj` only: ±degrees from vertical to search. Increase for heavily tilted books. |
| `spine_n_angles` | `rotated_proj` only: number of angles to test within ±`spine_angle_range`. More = finer search, slower. |
