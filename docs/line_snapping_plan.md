# Text Line Detection → Gaze Snapping Plan

## Context

The system (IoTitty / Gaze-Augmented Reading System) maps PCCR eye-tracker output
to a physical book page via homography:

```
eye cam → pupil-glint vector → homography → world cam (x, y) → nearest text line
```

`curve_track` already detects curved text line polygons on the world cam frame.
The gap is the **snapping layer**: given a gaze point in world cam coordinates,
reliably assign it to a line index and derive reading analytics.

Target metrics from the paper:
- Line detection accuracy ≥ 85%
- WPM estimation error ≤ 20%
- End-to-end latency ≤ 400 ms

---

## What We Have

| Component | Location | Output |
|-----------|----------|--------|
| `curve_track` detector | `compute/text_detector.py` | List of curved strip polygons + raw track points `[(x,y),…]` per line |
| `book_mask` | same | Filters polygons outside Canny page quad |
| `/text_lines/<n>` endpoint | `receiver.py` | JSON `{lines, w, h}` at ≤5 fps (200ms cache) |
| Gaze point on world cam | `receiver.py` `drawGazeOnWorldCanvas()` | `(gx, gy)` in canvas pixels, from `__lastStats.gaze_scene` |

The raw **track points** (`_last_debug["tracks"]`) are richer than the polygon output —
each track is `[(x0,y0), (x1,y1), …]` sampled across vertical strips, giving
the actual curve of the text baseline. This is the key input for snapping.

---

## Proposed Architecture

```
world cam frame
      │
      ▼
_detect_lines_curve_track()   ← already runs, 200ms cache
      │  tracks[]  (raw strip samples + fitted poly coefficients)
      │  book_quad  (Canny page boundary)
      ▼
LineSnapper.update(tracks, book_quad, gaze_xy)
      │
      ├─ snap_to_line()       → line_idx, snap_y (expected y on that line at gaze_x)
      ├─ stability_filter()   → debounced line_idx (hysteresis + dwell)
      ├─ x_progress()         → 0.0–1.0 position along the line
      ├─ regression_detect()  → bool
      └─ wpm_estimate()       → float
      ▼
/gaze_line  JSON endpoint   →  JS overlay (highlight current line, show analytics)
```

---

## Implementation Plan

### 1. Return track polylines from `/text_lines/<n>`

Currently the endpoint only returns polygon corners. Add the fitted track data:

```python
# In receiver.py _text_lines handler, extend result:
result = {
    "lines": [q.tolist() for q in quads],
    "tracks": [
        {"pts": t, "coeff": np.polyfit(...).tolist(), "deg": deg}
        for t in tracks
    ],
    "w": W, "h": H,
}
```

The JS stores `_lastTextLines.tracks` alongside polygons.

---

### 2. `LineSnapper` — `compute/line_snapper.py`

```python
class LineSnapper:
    def __init__(
        self,
        hysteresis_px: float = 8.0,    # must beat current line by this to switch
        dwell_frames: int = 3,          # frames gaze must stay on new line to commit
        regression_px: float = 30.0,   # x-drop on same line counts as regression
        wpm_smoothing: float = 0.15,    # EMA alpha for WPM
    ): ...

    def update(
        self,
        tracks: list[list[tuple[int,int]]],   # raw track points per line
        gaze_xy: tuple[float, float],
        frame_ts_ms: float,
    ) -> dict:
        """Returns snapping result dict."""
```

**Core snapping — `_snap_to_line()`:**

For each track, evaluate the fitted polynomial at `gaze_x` → `expected_y`.
Distance = `|gaze_y - expected_y|`. Guard: only consider lines whose
x-range contains `gaze_x` (±one strip width tolerance).

```python
def _curve_y_at(self, track_pts, gaze_x):
    xs = [p[0] for p in track_pts]
    ys = [p[1] for p in track_pts]
    if gaze_x < min(xs) - STRIP_TOL or gaze_x > max(xs) + STRIP_TOL:
        return None
    deg = min(2, len(xs) - 1)
    coeff = np.polyfit(xs, ys, deg)
    return float(np.polyval(coeff, gaze_x))
```

**Hysteresis + dwell:**

```
candidate = argmin(distances)
if candidate == current_line:
    reset dwell counter
elif distances[candidate] < distances[current_line] - hysteresis_px:
    dwell_counter += 1
    if dwell_counter >= dwell_frames:
        commit → current_line = candidate; reset counter
else:
    reset dwell counter (gaze didn't stay)
```

This prevents jitter at line boundaries — the new line must win by a
margin AND hold for multiple frames before committing.

**x-progress:**

```python
x_lo = min(p[0] for p in track)
x_hi = max(p[0] for p in track)
x_progress = (gaze_x - x_lo) / max(1, x_hi - x_lo)   # 0.0 = line start, 1.0 = end
```

**Regression detection:**

```python
if line_idx == prev_line_idx:
    if gaze_x < prev_gaze_x - regression_px:
        is_regression = True   # backward jump on same line
elif line_idx < prev_line_idx:
    is_regression = True       # jumped up a line
```

**WPM estimation:**

```python
# Estimate word count per line from its pixel width
line_width_px = x_hi - x_lo
# Average word width ≈ 5 × inter-line gap (peak_min_dist gives approx char height)
# Tune: ~6 chars/word × char_width_px, or calibrate from known text
words_per_line = line_width_px / avg_word_width_px

# Track time spent on each line
line_dwell_ms = current_ts - line_entry_ts
if line_dwell_ms > 500:   # ignore glances
    wpm_raw = (words_per_line / line_dwell_ms) * 60_000
    wpm_smoothed = alpha * wpm_raw + (1-alpha) * wpm_smoothed
```

`avg_word_width_px` is calibrated once at session start (known inter-line gap
→ estimate char height → word width ≈ 5 × char_height). Or let user set WPM
calibration mode (read a known passage).

---

### 3. `/gaze_line` endpoint in `receiver.py`

```python
# Returns:
{
  "line_idx":    2,          # 0-based, top-to-bottom
  "line_count":  18,         # total detected lines
  "snap_y":      341,        # expected y of that line at gaze_x (for overlay)
  "x_progress":  0.42,       # 0–1 left→right on current line
  "is_regression": false,
  "wpm":         220.5,
  "dwell_ms":    1200,       # ms on current line
  "gaze_xy":     [480, 338], # raw gaze in world cam px
}
```

Runs at world-cam frame rate but the snapper's `update()` call is cheap (<1ms)
since the heavy detection is already cached.

---

### 4. JS overlay in `receiver.py`

**Highlight current line:**

```js
// After drawTextLinesOnWorldCanvas(), add:
function drawLineHighlight() {
  if (!__lastGazeLine || !_lastTextLines) return;
  const idx = __lastGazeLine.line_idx;
  const poly = _lastTextLines.lines[idx];
  if (!poly) return;
  ctx.save();
  ctx.fillStyle = 'rgba(255, 255, 0, 0.18)';
  ctx.beginPath();
  // scale poly to canvas, fill the strip
  ...
  ctx.fill();
  ctx.restore();
}
```

**HUD overlay (optional):**

```
Line 3 / 18   WPM: 223   Progress: ██░░░░  ← regression indicator
```

**Poll rate:** `/gaze_line` fetched every frame (cheap endpoint, no detection work).

---

### 5. Calibration hook — `avg_word_width_px`

One-time calibration mode:
1. User reads a passage of known word count aloud or for a known duration
2. System records total lines traversed × words_per_line estimate
3. Fits `avg_word_width_px` to match measured WPM

Stored in `eye_settings.json` alongside existing calibration data.

---

## Data Flow Summary

```
World cam JPEG (every frame)
  → _detect_lines_curve_track()  [200ms cache]
      → tracks[], book_quad

Gaze point from PCCR pipeline
  → /stats → gaze_scene [x, y]

LineSnapper.update(tracks, gaze_xy, ts)
  → /gaze_line JSON  [every frame, <1ms]

JS:
  drawBitmap()                 ← world cam frame
  drawTextLinesOnWorldCanvas() ← yellow line outlines
  drawLineHighlight()          ← filled highlight on current line
  drawGazeOnWorldCanvas()      ← crosshair
  updateReadingHUD()           ← line index, WPM, regression flag
```

---

## Accuracy Improvements Over Naive Centroid Snapping

| Issue | Naive approach | This plan |
|-------|---------------|-----------|
| Curved page | Uses polygon centroid → wrong y on curved lines | Evaluates fitted polynomial at gaze_x → correct y |
| Jitter at line boundary | Snaps on every frame | Hysteresis + dwell filter |
| Partial-line gaze | Matches all lines | Guards on x-range before computing distance |
| Line count drift | Detects on every call | 200ms cache + stability across frames |
| WPM on dense/sparse text | Fixed word assumption | Width-based estimate, calibratable |

---

## Files to Create / Modify

| File | Change |
|------|--------|
| `compute/line_snapper.py` | New — `LineSnapper` class |
| `receiver.py` | Add `/gaze_line` endpoint, instantiate `LineSnapper`, extend `/text_lines` to include track data, add JS overlay |
| `eye_settings.json` | Add `avg_word_width_px` calibration field |

No changes needed to `compute/text_detector.py` — existing track output is sufficient.

---

## Open Questions

1. **Page homography vs pixel coords**: gaze_scene is in world cam pixel coords —
   does it need to be corrected for perspective before distance comparison, or is
   pixel-distance snapping accurate enough given the narrow depth-of-field of a
   flat book? (Likely fine for flat/near-flat books; revisit for curved pages.)

2. **Multi-column layout**: curve_track on a two-page spread with `split_pages=True`
   gives interleaved line indices. The snapper needs to know which page (left/right)
   the gaze is on before indexing.

3. **Line re-reading**: same line visited twice in a session = valid re-read or
   calibration drift? Needs a session-level state machine.
