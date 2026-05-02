"""
Line-based gaze snapper for book reading.

Takes raw gaze_xy (scene camera coords) + detected text line tracks and
snaps gaze vertically onto the nearest text baseline.

This cancels depth/parallax error in the vertical axis: the gaze model was
calibrated at screen distance D_s, but books sit at a different depth D_b,
causing a vertical shift proportional to eccentricity. Snapping to the nearest
detected line absorbs that vertical error.

Usage:
    snapper = LineSnapper(scene_w=1280, scene_h=720)
    result = snapper.update(tracks, gaze_xy, ts_ms)
    # result.snap_xy — vertically corrected gaze
    # result.line_idx — 0-based index of locked line
    # result.x_progress — 0..1 horizontal progress along line
    # result.confidence — 0..1
    # result.is_regression — True when gaze jumps backward
    # result.wpm — smoothed estimated reading speed
    # result.dwell_ms — time spent on current line
"""

from __future__ import annotations

import collections
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class SnapResult:
    line_idx: Optional[int]           # index into sorted (top→bottom) line list
    line_count: int                   # total detected lines
    snap_xy: Optional[tuple[float, float]]  # gaze after vertical snap
    x_progress: float                 # 0..1 along locked line (left→right)
    confidence: float                 # 0..1; low when gaze is far from any line
    is_regression: bool               # True when line_idx < previous locked line
    wpm: float                        # smoothed WPM estimate (0 if unknown)
    dwell_ms: float                   # ms spent on current line
    raw_gaze_xy: tuple[float, float]  # input before snap


class LineSnapper:
    """
    Three-layer snapping:
      1. Spatial: fit polynomial baseline per track, evaluate at gaze_x
      2. Hysteresis: require gaze to dwell `dwell_frames` near a new line
         before switching (prevents jitter between adjacent lines)
      3. Velocity gate: suppress snap during fast saccades
    """

    # ── Tuning knobs ──────────────────────────────────────────────────────────
    SMOOTH_MS        = 150.0   # rolling median window for raw gaze
    DWELL_FRAMES     = 4       # frames near new line before committing
    SACCADE_PX_MS    = 2.0     # px/ms threshold — above = saccade, no snap
    SNAP_CONF_SCALE  = 80.0    # px distance at which confidence = 0.5
    WPM_ALPHA        = 0.15    # EMA smoothing for WPM

    def __init__(self, scene_w: int = 1280, scene_h: int = 720):
        self.scene_w = scene_w
        self.scene_h = scene_h

        # Rolling gaze buffer for smoothing: deque of (ts_ms, x, y)
        self._smooth_buf: collections.deque = collections.deque(maxlen=64)

        # Hysteresis state
        self._locked_line: Optional[int] = None   # currently committed line idx
        self._candidate: Optional[int] = None     # line we are considering
        self._cand_count: int = 0                 # frames near candidate

        # WPM / dwell tracking
        self._line_enter_ts: float = 0.0          # ms when locked onto current line
        self._line_enter_x: float = 0.0           # x_progress at line entry
        self._wpm_smooth: float = 0.0
        self._prev_gaze: Optional[tuple[float, float, float]] = None  # (ts, x, y)
        self._prev_locked_line: Optional[int] = None

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, tracks: list[list[tuple[int, int]]],
               gaze_xy: tuple[float, float],
               ts_ms: float) -> SnapResult:
        """
        tracks: list of curve_track outputs — each is [(x0,y0),(x1,y1),...] sorted L→R
        gaze_xy: raw gaze in scene pixel coords
        ts_ms: current time in milliseconds
        """
        raw = gaze_xy

        # 1. Smooth raw gaze
        self._smooth_buf.append((ts_ms, gaze_xy[0], gaze_xy[1]))
        smooth_xy = self._rolling_median()

        # 2. Sort tracks top→bottom by median y
        sorted_tracks = sorted(tracks, key=lambda t: np.median([p[1] for p in t]))
        n = len(sorted_tracks)

        if n == 0 or smooth_xy is None:
            return SnapResult(None, 0, None, 0.0, 0.0, False, self._wpm_smooth, 0.0, raw)

        # 3. Find nearest baseline y at smooth_xy[0]
        gx, gy = smooth_xy
        baseline_ys = [self._curve_y_at(t, gx) for t in sorted_tracks]

        dists = []
        for i, by in enumerate(baseline_ys):
            if by is None:
                # Gaze x outside track range — use closest endpoint y
                xs = [p[0] for p in sorted_tracks[i]]
                ys = [p[1] for p in sorted_tracks[i]]
                if gx < xs[0]:
                    by = float(ys[0])
                else:
                    by = float(ys[-1])
                baseline_ys[i] = by
            dists.append(abs(gy - by))

        nearest_idx = int(np.argmin(dists))
        nearest_dist = dists[nearest_idx]

        # 4. Velocity gate: during saccades skip hysteresis update
        is_saccade = self._detect_saccade(smooth_xy, ts_ms)

        # 5. Hysteresis: commit to new line only after dwell_frames
        if not is_saccade:
            if nearest_idx == self._locked_line:
                self._candidate = None
                self._cand_count = 0
            elif nearest_idx == self._candidate:
                self._cand_count += 1
                if self._cand_count >= self.DWELL_FRAMES:
                    prev_locked = self._locked_line
                    self._locked_line = nearest_idx
                    self._candidate = None
                    self._cand_count = 0
                    self._on_line_switch(ts_ms, gx / max(self.scene_w, 1),
                                        nearest_idx, sorted_tracks[nearest_idx],
                                        baseline_ys[nearest_idx],
                                        (prev_locked is not None and nearest_idx < prev_locked))
            else:
                self._candidate = nearest_idx
                self._cand_count = 1

        if self._locked_line is None:
            # First lock: snap immediately without dwell
            self._locked_line = nearest_idx
            self._on_line_switch(ts_ms, gx / max(self.scene_w, 1),
                                 nearest_idx, sorted_tracks[nearest_idx],
                                 baseline_ys[nearest_idx], False)

        locked = self._locked_line
        locked_by = baseline_ys[locked] if locked < len(baseline_ys) else gy
        snap_xy = (gx, float(locked_by)) if locked_by is not None else (gx, gy)

        # 6. Confidence
        conf = 1.0 / (1.0 + (nearest_dist / self.SNAP_CONF_SCALE) ** 2)

        # 7. x_progress along locked line
        x_prog = self._x_progress(sorted_tracks[locked], gx)

        # 8. Regression check
        is_reg = (locked < (self._prev_locked_line or locked))

        # 9. Dwell
        dwell = ts_ms - self._line_enter_ts if self._line_enter_ts else 0.0

        # 10. WPM
        self._update_wpm(x_prog, ts_ms, sorted_tracks[locked])

        self._prev_locked_line = locked
        self._prev_gaze = (ts_ms, gx, gy)

        return SnapResult(
            line_idx=locked,
            line_count=n,
            snap_xy=snap_xy,
            x_progress=x_prog,
            confidence=float(conf),
            is_regression=is_reg,
            wpm=self._wpm_smooth,
            dwell_ms=float(dwell),
            raw_gaze_xy=raw,
        )

    def reset(self):
        """Call when scene changes (new page, camera switch)."""
        self._smooth_buf.clear()
        self._locked_line = None
        self._candidate = None
        self._cand_count = 0
        self._line_enter_ts = 0.0
        self._line_enter_x = 0.0
        self._wpm_smooth = 0.0
        self._prev_gaze = None
        self._prev_locked_line: Optional[int] = None

    # ── Internals ─────────────────────────────────────────────────────────────

    def _rolling_median(self) -> Optional[tuple[float, float]]:
        now = self._smooth_buf[-1][0] if self._smooth_buf else 0.0
        cutoff = now - self.SMOOTH_MS
        pts = [(x, y) for ts, x, y in self._smooth_buf if ts >= cutoff]
        if not pts:
            return None
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return (float(np.median(xs)), float(np.median(ys)))

    @staticmethod
    def _curve_y_at(track: list[tuple[int, int]], gaze_x: float) -> Optional[float]:
        """Polynomial-fit the track and evaluate at gaze_x. Returns None if x outside range."""
        if len(track) < 2:
            return None
        pts = np.array(track, dtype=float)
        xs, ys = pts[:, 0], pts[:, 1]
        if gaze_x < xs.min() or gaze_x > xs.max():
            return None
        deg = min(3, len(track) - 1)
        try:
            coeffs = np.polyfit(xs, ys, deg)
            return float(np.polyval(coeffs, gaze_x))
        except np.linalg.LinAlgError:
            return float(np.interp(gaze_x, xs, ys))

    @staticmethod
    def _x_progress(track: list[tuple[int, int]], gaze_x: float) -> float:
        """0..1 progress along track from leftmost to rightmost point."""
        if not track:
            return 0.0
        xs = [p[0] for p in track]
        x0, x1 = min(xs), max(xs)
        if x1 <= x0:
            return 0.0
        return float(np.clip((gaze_x - x0) / (x1 - x0), 0.0, 1.0))

    def _detect_saccade(self, smooth_xy: tuple[float, float], ts_ms: float) -> bool:
        if self._prev_gaze is None:
            return False
        pt0_ts, px, py = self._prev_gaze
        dt = ts_ms - pt0_ts
        if dt < 1e-3:
            return False
        dist = ((smooth_xy[0] - px) ** 2 + (smooth_xy[1] - py) ** 2) ** 0.5
        return (dist / dt) > self.SACCADE_PX_MS

    def _on_line_switch(self, ts_ms: float, x_prog: float,
                        line_idx: int, track, baseline_y, is_regression: bool):
        self._line_enter_ts = ts_ms
        self._line_enter_x = x_prog

    def _update_wpm(self, x_prog: float, ts_ms: float,
                    track: list[tuple[int, int]]):
        """Estimate WPM from x progress rate. Assumes ~5 chars/word, ~10 chars/line."""
        if self._line_enter_ts <= 0 or ts_ms <= self._line_enter_ts:
            return
        dt_min = (ts_ms - self._line_enter_ts) / 60000.0
        if dt_min < 1e-6:
            return
        # Fraction of line covered since entry
        dx = x_prog - self._line_enter_x
        if dx <= 0:
            return
        # Assume 10 words per full line width
        words = dx * 10.0
        inst_wpm = words / dt_min
        # EMA smooth
        if self._wpm_smooth == 0:
            self._wpm_smooth = inst_wpm
        else:
            self._wpm_smooth = (self.WPM_ALPHA * inst_wpm +
                                (1.0 - self.WPM_ALPHA) * self._wpm_smooth)
