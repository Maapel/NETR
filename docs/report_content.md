# Report Content — NETR Project

All text blocks, numbers, tables, design decisions, and plot references
needed to write the final report. Pull whatever you need.

---

## 1. Project Summary (one paragraph)

NETR (Non-invasive Eye Tracking Rig) is a head-mounted gaze tracking and
reading assistance system built from two ESP32-CAM modules (~$7 each), two
IR LEDs, and a Python compute stack on a laptop. One camera faces the eye;
one faces the scene (book/screen). The eye camera detects the pupil and
corneal glints to compute a PCCR (Pupil-Cornea Corneal Reflection) vector.
A polynomial model maps PCCR → screen coordinates. An adaptive calibration
system minimises calibration effort. A text line detector identifies which
line on a physical book the user is reading, and the nearest line to the
gaze is highlighted in a browser overlay. The system runs at 20 fps (eye)
and 12 fps (world) with ~75–100 ms end-to-end latency.

---

## 2. Hardware

### Cameras
- AI-Thinker ESP32-CAM, OV2640 sensor, up to 2 MP
- **cam1** (world/scene): MAC `c4:dd:57:ea:28:5c`, streams UDP port 5000, cmd port 5001
- **cam2** (eye): MAC `c4:dd:57:ea:3d:84`, streams UDP port 5002, cmd port 5003
- Eye camera physically rotated 90° → compensated in software via `swap_pccr`
- Two IR LEDs (850 nm) on either side of eye camera lens — produce corneal reflections

### Connectivity
- Discovery beacon: UDP port 5004
- OTA update: port 3232, password `esp32ota`
- JPEG frames fragmented into 1400-byte UDP packets
- 12-byte header per fragment: frame ID, fragment index, total fragments, timestamp

### FreeRTOS Tasks
| Task | Core | Purpose |
|------|------|---------|
| captureTask | 0 | Camera capture at target FPS |
| sendTask | 1 | UDP fragmentation + transmit |
| cmdTask | 1 | Receive settings commands |
| otaTask | 1 | Over-the-air firmware update |
| discoveryTask | 1 | Beacon broadcast + laptop IP discovery |
| timeSyncTask | 1 | Cristian's algorithm clock sync every 10s |
| ledTask | 1 | LED state machine |
| wifiTask | 1 | WiFi watchdog + reconnect |

### Camera Init Rule
Always init at `FRAMESIZE_UXGA` (max DMA buffer), then set working resolution.
After any `set_framesize` call, flush 3 frames to drain stale sensor FIFO.

### Clock Sync
Cristian's algorithm with local NTP server (`ntp_server.py`) on laptop.
Sync every 10s. Essential for correlating eye frames with world frames during
fixation aggregation.

---

## 3. Eye Processing Pipeline

### Pupil Detection (`PupilDetector`)
- Algorithm: `threshold` (default) — CLAHE + Gaussian blur + adaptive threshold + blob detection
- Identifies largest dark circular region
- Tuneable: `min_radius=15`, `max_radius=150`, `dark_percentile=1.8`, `morph_ksize=5`

### Glint Detection (`GlintDetector`)
Two-stage process:

**Stage 1 — Limbus estimation**
- 16 radial rays cast from pupil edge, each profiles intensity outward
- Iris-to-sclera brightness jump detected as limbus boundary
- Median of all ray detections = limbus radius
- Defines iris annulus = search region for glints (limbus circle minus pupil disk)

**Stage 2 — Adaptive thresholding within iris annulus**
- 99th-percentile brightness in annulus → adaptive threshold
- Blobs above threshold filtered by circularity + area
- `min_dist_factor = 0.3`: reject blobs within 0.3× pupil radius (tear film artefacts)

### PCCR Computation
```
PCCR = (dx, dy) = pupil_center − glint_xy
```

**swap_pccr=True** (camera rotated 90°):
```
dx_effective = dy_raw   ← horizontal in real world
dy_effective = dx_raw   ← vertical in real world
```

After swap: `corr(dx, screen_X) = 0.896` — dx cleanly tracks horizontal gaze.

**LED side labeling**  
With swap_pccr active, labeling uses `dy` axis (horizontal in real world):
- `side = +1` if `dy > switch_dx` (right LED visible)
- `side = −1` otherwise (left LED visible)

---

## 4. Gaze Model — Full Evolution

### Polynomial Basis (all models)
```
φ(dx, dy) = [1, dx, dy, dx·dy, dx², dy²]
```
Two independent least-squares fits:
```
X̂ = φ · A,   Ŷ = φ · B,   A, B ∈ ℝ⁶
```

### Stage 1 — Single GazeModel (baseline)
- One 6-term model, preferred-side glint only
- Side determined by horizontal sweep calibration (`switch_dx`)
- **Result (28 fixations, rec 20260501_005039):** LOO median **109 px**, mean 153 px
- Failure: corners extrapolate 300–500 px

### Stage 2 — DualGazeModel
- Two independent GazeModel instances (model_pos, model_neg)
- Both trained by emitting two samples per fixation
- Failure: each model sees half the data; other LED PCCR discarded at inference

### Stage 3 — Rejected Alternatives (tested on 28 fixations)
| Method | Mean | Median | p75 | Verdict |
|--------|------|--------|-----|---------|
| Single preferred glint | 153 | **109** | 163 | ✓ kept |
| Midpoint (g1+g2)/2 | 166 | 112 | 178 | ✗ |
| Midpoint ÷ D (normalised) | 171 | 111 | 181 | ✗ |
| 4D joint feature | 178 | 150 | 237 | ✗ |

**Why midpoint/D fails:** In remote trackers, D varies with head depth (compensates
parallax). In head-mounted rig, LEDs fixed relative to camera — D only varies
134–140 px across all fixations. Normalising by a near-constant is just a fixed
scale absorbed by the polynomial.

**Why 4D fails:** 28 samples × cross-terms over 4 inputs → catastrophic LOO overfit.

### Stage 4 — TwoGlintGazeModel (11-term joint)
```
φ₁₁ = [1, dx1, dy1, dx2, dy2, dx1·dx2, dx1·dy2, dy1·dx2, dy1·dy2, dx1², dy2²]
```
- Cross-terms capture differential signal between two glints
- R² improved from ~0.72 → ~0.91
- **Result (63 fixations, rec 20260501_015920):** LOO median **45 px** (was 63 px)
- **Why replaced:** At N < 50, 11-term overfits. p90 = 753 px on some recordings.
  LOO on 19 samples for 11 free parameters = catastrophic.

### Stage 5 — SplitGlintModel (two × 6-term)
- `model_right`: trained on (dx1, dy1) from right LED
- `model_left`:  trained on (dx2, dy2) from left LED
- Both trained only on frames with **both** glints detected

**Pre-training filters:**
1. `|dx| < 10 px` → drop (glint near pupil centre = noise)
2. Per-glint sigma filter: `|dy| > μ + 3σ` → drop
3. Collection gate: `|dy1| > 120 OR |dy2| > 120` → strip to single-glint

**Selection criterion evolution:**
| Version | Logic | Problem |
|---------|-------|---------|
| v1 | max \|dx\| | ignores dy |
| v2 | max \|dy\| | ignores dx |
| v3 | max magnitude √(dx²+dy²) | mixed results across 10 recordings |
| **v4 (current)** | **always-right, fallback left** | **wins 8/10 recordings** |

**Why always-right wins:** Left LED model consistently undertrained — fewer clean
samples, higher variance (σ = 34.8 px vs 12.2 px for right). Dynamic selection
sometimes picks the weaker model, degrading median.

### Stage 6 — Lorentzian LOO Reweighting (final, current)
Per-sample LOO residuals computed first. Downweight noisy samples:
```
w_i = 1 / (1 + (err_i / median_err)²)
```
Weighted least squares: scale design matrix rows by √w_i.

**Lorentzian vs Gaussian:** Heavier tails — doesn't aggressively penalise moderate
errors, only genuine outliers collapse toward zero influence.

**Result (88 samples, rec 20260505_040942, 87 after dropping off-screen outlier):**
| | Mean | Median | p75 | p90 |
|--|------|--------|-----|-----|
| Unweighted | 43.5 px | 28.0 px | 46.5 px | 83.0 px |
| Weighted   | **31.6 px** | **20.8 px** | **31.6 px** | **57.1 px** |

67/87 samples improved. Mean −27%, p90 −31%.

---

## 5. Calibration System

### Fixation Aggregation (per dot, 500 ms window)
1. **Homography sync:** For each eye frame, fetch world frame by timestamp.
   Detect ArUco markers → compute homography H → project stimulus (sx, sy)
   into world-cam space → get (X, Y). X, Y are model training targets.
2. **Blink filter:** Drop frames where pupil radius deviates > ±15% from median.
3. **IQR filter:** Keep frames where dx, dy both within [Q1−1.5×IQR, Q3+1.5×IQR].
4. **Mean:** Mean of surviving (dx, dy, X, Y) = single training sample.

**Note on X, Y vs sx, sy:**
- `sx, sy` = stimulus dot position on screen (ground truth)
- `X, Y` = same position projected through ArUco homography into world-cam pixels
- Model is trained to predict X, Y (not sx, sy directly)
- LOO error = ||predicted − X,Y|| (model quality)
- End-to-end error = ||predicted − sx,sy|| (includes homography mapping gap)

### Adaptive Calibration Point Spawning

**Why adaptive:** Fixed grid wastes samples on accurate central regions while
undersampling high-error corners where ArUco coverage is weakest.

**Phase 1 — Bootstrap (first 9 points): max-min distance**
- Pick next point that maximises minimum distance to all collected points
- Guarantees spatial screen coverage before any error signal exists

**Phase 2 — Error-guided: IDW scoring**
After each fixation, recompute LOO errors. Score candidates by:
```
score(c) = Σ(err_i / d(c,pᵢ)²) / Σ(1 / d(c,pᵢ)²)
```
Next point = candidate with highest score → biases toward high-error regions.
Pool resets when all grid slots visited.

---

## 6. Text Line Detection — curve_track

### Why curve_track over MSER-based methods
- MSER requires global skew estimation → fails on curved book pages
- curve_track needs no skew assumption — projection peaks naturally follow curvature
- Handles two-page spreads, tilted camera, page curl

### Pipeline
```
BGR → grayscale → CLAHE → blur → adaptive threshold → bin_inv
```
bin_inv: ink = white (255), background = black (inverted convention).
Adaptive threshold (neighbourhood-based) handles uneven illumination.

**Strip projection:** Slice image into N vertical strips (default 24).
Per strip: `profile[y] = Σ bin_inv(x, y)` — counts ink pixels per row.

**Peak tracking:** NMS finds peaks per strip. Greedy left-to-right association:
extend existing tracks within `y_tol` px, start new tracks for unmatched peaks.
Tracks killed after `track_max_gap` missed strips.

**Three-layer outlier rejection** (shadow/hand edges produce false peaks):
1. FWHM filter: peaks wider than 2.5× median FWHM dropped
2. Neighbour support: peak must match adjacent strip within y_tol
3. Spacing check: track pairs < 55% of median inter-track gap = shadow edge pair, shorter dropped

**Polygon output:** Fit degree-2 polynomial through track, evaluate densely,
offset ±(inter-line gap / 2) → curved polygon strip per line.

### Book Page Masking
- Threshold at 60th brightness percentile (page = bright, text = dark)
- Morphological close + open → fill text gaps, remove noise
- Largest contour → convex hull → approxPolyDP quad
- Discard line detections whose centre falls outside quad

### Active Line Detection
For each track segment, project gaze point onto segment, compute 2D distance.
Track with minimum distance = active (currently read) line.
Highlighted in browser overlay as bright yellow with glow.

### Spine Detection (for two-page splits)
`rotated_proj` (default): sweep ±20° from vertical, find sharpest projection valley.
Handles tilted books. Falls back to `column_sum` (vertical only) if rotated not needed.

### Temporal Stabilisation (optional, off by default)
LK sparse optical flow on `goodFeaturesToTrack` keypoints within page quad.
RANSAC homography → perspectiveTransform previous tracks onto current frame.
Reduces jitter when book moves between frames.

---

## 7. Receiver and Browser

- `receiver.py`: central hub — UDP reassembly, EyePipeline at 20 fps, world frame buffer
- Browser dashboard at `localhost:8080`
- Frame hash cache: skip `detect_lines` if world frame unchanged (~20 Hz poll vs ~12 fps world)

**Eye overlay:**
- Green circle = pupil
- Dashed ring = limbus
- Yellow dot + R = right LED glint
- Blue dot + L = left LED glint
- Arrow = PCCR vector

**World overlay:**
- Dashed blue outline = book page quad
- Dim green polygons = all detected text lines
- Bright yellow + glow = active line (nearest to gaze)
- Crosshair = gaze estimate

---

## 8. Evaluation Data

### Best recording: 20260505_040942
- Mode: saccade calibration
- Screen: 1462 × 872 px
- Total samples: 88 (76 two-glint, 12 single-glint)
- **LOO model error: median 25.2 px, mean 46.7 px, p90 82.5 px**
- Homography error: median 309 px (ArUco coverage ~60% of screen)
- End-to-end: median 297 px (bottleneck = homography, not model)

### 2nd recording: 20260505_053152
- Total: 116 samples (91 two-glint, 25 single-glint)
- **LOO model error: median 42.3 px, p90 131.8 px**
- Homography error: median 242 px (slightly better coverage)
- End-to-end: median 236 px

### Error by screen X zone (rec 20260505_053152)
| Screen X band | N | Median error |
|--------------|---|-------------|
| 0–300 px | 26 | 120 px |
| 300–600 px | 17 | 118 px |
| 600–900 px | 29 | 226 px |
| 900–1200 px | 34 | 359 px |
| 1200–1500 px | 10 | **466 px** |

Root cause: ArUco markers absent near right screen edge.
X,Y range only 127–932 px even though screen goes to 1345 px.

### Temporal stability
- Median frame-to-frame PCCR jump within fixation: **1.6 px**
- Within-fixation std: **2.1 px**
- Conclusion: 25 px LOO error is systematic (calibration), not random noise.
  No temporal smoothing needed on gaze signal.

### PCCR axis correlation
- `corr(dx, screen_X) = 0.896` (after swap_pccr)
- `corr(dy, screen_Y) = 0.754`
- dx is dominant axis, dy has more curvature/spread

### Eye overlay video stats (rec 20260505_040942, 2978 frames @ 20 fps)
- Pupil detected: ~100% of frames
- Glint detected: ~100% of frames
- Dual glint: ~100% of frames

---

## 9. System Latency (estimated)
| Stage | Latency |
|-------|---------|
| Eye camera capture (20 fps) | 50 ms |
| UDP transmission | 2–5 ms |
| Pupil + glint detection | 8–15 ms |
| PCCR + model inference | < 1 ms |
| Browser poll + render | 16–33 ms |
| **Total** | **~75–100 ms** |

---

## 10. Key Design Decisions

**Head-mounted vs remote:** Head-mounting eliminates head-movement compensation.
PCCR stable across large head poses — LED/camera geometry is fixed relative to eye.
Tradeoff: wearability, 90° rotation needing swap_pccr.

**Two cameras:** World camera essential — gaze model outputs world-cam coordinates;
without it, no way to identify what object is being looked at.

**Two LEDs vs one:** Differential signal between two glints lifts R² from ~0.72 → ~0.91.
Two separate 6-term models chosen over one 11-term joint model due to sample sparsity
(11 params, N < 100 → overfit).

**Always-right LED selection:** Empirical — wins 8/10 recordings in cross-validation.
Consistent asymmetry: right LED at slightly better physical angle to cornea.

**Adaptive calibration:** Phase 1 (max-min distance) = coverage. Phase 2 (IDW scoring)
= concentrates remaining budget on high-error corners (exactly where ArUco weakest).

**curve_track over MSER:** MSER global skew fails on curved pages. Projection-peak
tracking needs no global assumption. Three-layer rejection handles adaptive threshold
false edges from shadows/hands.

---

## 11. Limitations

1. **Homography coverage** — primary bottleneck. ArUco at screen corners → end-to-end drops ~300 px → ~25 px immediately.
2. **Session persistence** — model not persistent across sessions. Head rig drift requires recalibration.
3. **Left LED model quality** — consistently higher variance (σ = 34.8 vs 12.2 px for right). Better IR power/placement on left would enable reliable dynamic selection.
4. **Word-level detection** — current accuracy identifies line, not word. Word width ~30–60 px, LOO error ~25 px — borderline.
5. **Reading speed inference** — temporal sequence of active-line transitions could estimate reading speed / re-reading events (not yet implemented).

---

## 12. Plots — Relative Paths from Project Root

### Early analysis (recordings 20260501_005039 and 20260501_015920)
Full analysis in `plots/inference.md`

| Path | Description |
|------|-------------|
| `plots/inference/01_pccr_space.png` | PCCR feature space coloured by screen X/Y |
| `plots/inference/02_correlation.png` | Linear fit dx→screen_X, dy→screen_Y |
| `plots/inference/03_loo_arrows.png` | LOO predicted vs target arrows |
| `plots/inference/04_error_heatmap.png` | Spatial error heatmap |
| `plots/inference/05_dual_glint_sides.png` | Left/right glint side labeling |
| `plots/inference/06_frames_vs_error.png` | Frame count vs LOO error per fixation |
| `plots/inference/07_pccr_magnitude.png` | PCCR magnitude distribution |
| `plots/inference/08_swap_pccr_effect.png` | Before/after swap_pccr |
| `plots/inference/09_loo_015920_pos.png` | LOO positive-side model |
| `plots/inference/10_loo_015920_neg.png` | LOO negative-side model |
| `plots/inference/11_error_heatmap_015920.png` | Spatial error heatmap (63-fixation rec) |
| `plots/inference/12_loo_comparison.png` | Model comparison |

### Best recording: 20260505_040942 (88 samples, LOO med=25 px)
Full analysis in `plots/20260505_040942/inference.md`

| Path | Description |
|------|-------------|
| `plots/20260505_040942/01_loo_predicted_vs_target.png` | LOO predicted (cyan) vs homography target (white), error-coloured lines |
| `plots/20260505_040942/02_error_heatmap.png` | Spatial IDW-interpolated error heatmap |
| `plots/20260505_040942/03_error_cdf.png` | Error CDF: model (blue, med=25px) vs end-to-end (red) vs homography baseline (green) |
| `plots/20260505_040942/04_pccr_feature_space.png` | Right + left LED PCCR scatter, colour = stimulus X |
| `plots/20260505_040942/05_per_fixation_error.png` | Per-fixation bar chart sorted left→right |
| `plots/20260505_040942/eye_overlay.avi` | 2978-frame annotated eye video |

### 2nd recording: 20260505_053152 (116 samples, LOO med=42 px)
| Path | Description |
|------|-------------|
| `plots/20260505_053152/01_loo_predicted_vs_target.png` | Same as above |
| `plots/20260505_053152/02_error_heatmap.png` | |
| `plots/20260505_053152/03_error_cdf.png` | Shows same bottleneck pattern |
| `plots/20260505_053152/04_pccr_feature_space.png` | |
| `plots/20260505_053152/05_per_fixation_error.png` | |

### Data quality overview (20260504 batch)
| Path | Description |
|------|-------------|
| `plots/latest_recordings_20260504/01_screen_coverage_scatter.png` | Stimulus dot coverage across screen |
| `plots/latest_recordings_20260504/02_side_distribution.png` | Left vs right glint distribution |
| `plots/latest_recordings_20260504/03_sync_offsets.png` | Clock sync offsets |
| `plots/latest_recordings_20260504/04_quality_indicators.png` | Pupil/glint detection quality |
| `plots/latest_recordings_20260504/05_duration_vs_samples.png` | Session duration vs sample count |

### Other
| Path | Description |
|------|-------------|
| `plots/glint_annotation.png` | Annotated glint detection example |
| `plots/loo_per_point.png` | Per-point LOO error (older analysis) |

---

## 13. Source Docs (all in `docs/`)

| File | Contents |
|------|----------|
| `docs/dual_glint_journey.md` | Full gaze model evolution: Stage 1→6, failure table, Mermaid flowchart |
| `docs/curve_track_explained.md` | Detailed curve_track algorithm walkthrough with data flow diagram |
| `docs/text_detector_params.md` | All method descriptions + parameter reference from text_tuner.py |
| `docs/project_report.tex` | LaTeX source of full report |
| `docs/project_report.pdf` | Compiled 5-page two-column report |

---

## 14. Key Files (code)

| File | Role |
|------|------|
| `esp32cam-stream/src/main.cpp` | Firmware (both cameras) |
| `receiver.py` | UDP receiver, EyePipeline, HTTP server, text line detection |
| `calibration_server.py` | Calibration UI, fixation aggregation, adaptive spawning, model fit |
| `compute/gaze_model.py` | SplitGlintModel, GazeModel, Lorentzian LOO reweighting |
| `compute/eye_pipeline.py` | Pupil + glint + PCCR pipeline |
| `compute/glint_detector.py` | Limbus estimation, adaptive threshold, glint blob detection |
| `compute/pupil_detector.py` | Pupil detection algorithms |
| `compute/text_detector.py` | curve_track, book masking, optical flow stabilisation |
| `tools/text_tuner.py` | Browser UI for text detector parameter tuning |
| `eye_settings.json` | Persisted eye pipeline settings |
