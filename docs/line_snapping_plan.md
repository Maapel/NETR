# Line-Based Gaze Snapping — Implementation Notes

Branch: `feature/gaze-line-snap`  
Commit: `071c68d`

---

## Why

Gaze model calibrated at screen distance D_s. Book sits at depth D_b.
Parallax offset: `Δy ≈ gaze_y × (D_b/D_s − 1)` — zero at image centre, grows with
eccentricity. Snapping `gaze_y` to the nearest detected text baseline absorbs this
vertical error without needing ArUco on the book.

Horizontal error not corrected by snapping — depth affects x similarly, but line
snapping gives x_progress directly (relative position along the line), which is what
matters for reading analytics.

---

## Files

| File | Role |
|------|------|
| `compute/line_snapper.py` | `LineSnapper` class — all snapping logic |
| `receiver.py` | `GET /gaze_line` endpoint + JS overlay |

`compute/text_detector.py` unchanged — existing `_last_debug["tracks"]` output is
sufficient.

---

## LineSnapper (`compute/line_snapper.py`)

### Input

```python
snapper.update(
    tracks: list[list[tuple[int,int]]],  # curve_track raw strip samples per line
    gaze_xy: tuple[float, float],        # raw gaze in scene/world cam pixel coords
    ts_ms: float,                        # current time in milliseconds
) -> SnapResult
```

`tracks` comes from `detector._last_debug["tracks"]` after `detect_lines()`.  
Each track is `[(x0,y0), (x1,y1), …]` sorted left→right along the text baseline.

### SnapResult fields

| Field | Type | Notes |
|-------|------|-------|
| `line_idx` | `int \| None` | 0-based index, sorted top→bottom |
| `line_count` | `int` | total detected lines this frame |
| `snap_xy` | `(float, float) \| None` | gaze after vertical snap |
| `x_progress` | `float` | 0..1 left→right along locked line |
| `confidence` | `float` | 0..1; drops when gaze far from all lines |
| `is_regression` | `bool` | True on the frame line_idx decreases |
| `wpm` | `float` | EMA-smoothed reading speed estimate |
| `dwell_ms` | `float` | ms spent on current line since lock |
| `raw_gaze_xy` | `(float, float)` | input before snap |

### Three-layer snapping

**Layer 1 — Spatial**  
For each track, fit polynomial (degree = min(3, len(track)−1)) and evaluate at
`gaze_x` → `baseline_y`. Distance = `|gaze_y − baseline_y|`. Nearest track wins.  
If `gaze_x` is outside a track's x-range, use the closest endpoint y (no rejection —
gaze can be slightly outside the page).

**Layer 2 — Hysteresis (dwell)**  
`DWELL_FRAMES = 4`. Gaze must stay near a new line for 4 consecutive frames before
`locked_line` commits. Prevents jitter at line boundaries.

**Layer 3 — Velocity gate (saccade detection)**  
`SACCADE_PX_MS = 2.0`. If smoothed gaze moves >2 px/ms between consecutive frames,
skip the hysteresis update for that frame. Suppresses false line-switches during
inter-line saccades.

### Gaze smoothing

150ms rolling median window on raw gaze before any snapping.  
Deque of `(ts_ms, x, y)`, keep entries within `[now − 150ms, now]`.

### Confidence

```python
conf = 1 / (1 + (nearest_dist / SNAP_CONF_SCALE) ** 2)
# SNAP_CONF_SCALE = 80px → conf = 0.5 at 80px distance
```

### WPM estimation

Measures x_progress rate on the locked line:

```python
words = dx_progress × 10   # assume 10 words per full line width
wpm_inst = words / dt_minutes
wpm = 0.15 * wpm_inst + 0.85 * wpm_prev   # EMA
```

Crude but self-calibrating per session. Assumes roughly uniform word density.

### Tuning knobs (class constants)

| Constant | Default | Effect |
|----------|---------|--------|
| `SMOOTH_MS` | 150 | rolling median window |
| `DWELL_FRAMES` | 4 | frames to commit line switch |
| `SACCADE_PX_MS` | 2.0 | px/ms saccade threshold |
| `SNAP_CONF_SCALE` | 80 | px distance for conf=0.5 |
| `WPM_ALPHA` | 0.15 | EMA smoothing for WPM |

---

## `/gaze_line` endpoint (`receiver.py`)

`GET /gaze_line`

1. Gets latest gaze from `_engine_get_result()["gaze"]`
2. Decodes world cam latest frame → `detect_lines(bgr)` → `_last_debug["tracks"]`
3. Calls `snapper.update(tracks, gaze, ts_ms)` under `_line_snapper_lock`
4. Returns:

```json
{
  "ok": true,
  "line_idx": 2,
  "line_count": 14,
  "snap_xy": [480.0, 341.2],
  "x_progress": 0.42,
  "confidence": 0.96,
  "is_regression": false,
  "wpm": 218.4,
  "dwell_ms": 1340.0,
  "raw_gaze_xy": [481.0, 352.0]
}
```

`ok: false` if engine gaze unavailable or text detector not loaded.

World cam = `CAMS[2]` when `g_eye_cam == 1`, else `CAMS[1]`.

Note: `detect_lines()` here shares no cache with `/text_lines/<n>` — runs its own
detection on demand. Latency dominated by `curve_track` (~20–50ms). Could be unified
with the existing 200ms cache; left as future work.

---

## JS overlay

Checkbox: **"Line snap gaze"** (`id=line_snap_world`)

When checked:
- `_fetchGazeLine()` called each display loop frame (fire-and-forget, deduped)
- Raw gaze: **cyan crosshair** (unchanged)
- Snapped gaze: **yellow crosshair** + dashed yellow line from raw to snapped
- Progress bar: thin yellow line along `snap_y` height, filled to `x_progress`
- HUD text (`id=line_snap_hud`): `L3/14 conf:96% 218 WPM ↩` (↩ = regression)

---

## Data Flow

```
World cam frame (every display tick)
      │
      ▼
detect_lines(bgr)             → tracks[] (curve_track baselines)
      │
Engine latest result          → gaze_xy (scene px)
      │
LineSnapper.update()          → SnapResult
      │
/gaze_line JSON response
      │
JS: drawGazeOnWorldCanvas()   → cyan raw + yellow snapped overlays
    line_snap_hud              → L3/14 conf:96% 218 WPM
```

---

## Limitations / Open Issues

1. **Horizontal parallax** not corrected. Line snap only fixes vertical.
   Full correction needs page homography (fit perspective transform from
   detected line layout → flat grid). Not implemented yet.

2. **`/gaze_line` detection not cached** — runs `detect_lines()` every call,
   separate from the `/text_lines` 200ms cache. Should be unified.

3. **WPM assumes uniform word density.** Dense (small font) vs sparse (large font)
   lines give wrong estimates. Calibration mode (read known passage) not yet built.

4. **Two-page spread with `split_pages=True`**: line indices are interleaved.
   Snapper needs to filter to the page containing the gaze x before sorting.

5. **Saccade threshold (2 px/ms)** is fixed. Should adapt to per-session noise floor.
