"""
Line-based gaze snapper for book reading.

Goal: detect WHICH TEXT LINE the reader is looking at.
Book can be at ANY angle in the world cam — never assume lines are horizontal.

Snapping uses 2D distance from gaze point to each track curve (not just Y distance).
"Backward jump" regression detection projects gaze movement onto the track's local
tangent direction, so it works regardless of book rotation.

Usage:
    snapper = LineSnapper()
    result = snapper.update(tracks, gaze_xy, ts_ms)
    # result.line_idx  — which line (0 = topmost along reading direction)
    # result.is_new_line, result.is_regression, result.dwell_ms, result.confidence
"""

from __future__ import annotations

import collections
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class SnapResult:
    line_idx: Optional[int]     # index into sorted line list (None = no lines)
    line_count: int             # total detected lines
    snap_xy: Optional[tuple[float, float]]  # nearest point on locked baseline
    confidence: float           # 0..1; low = gaze far from all lines
    is_new_line: bool           # just switched to a different line
    is_regression: bool         # backward jump along reading direction
    dwell_ms: float             # ms on current line
    raw_gaze_xy: tuple[float, float]


class LineSnapper:
    """
    Snap gaze to nearest text line track.

    Two signals for line-change detection:
      1. Gaze y (2D curve distance) → which line is gaze nearest to
      2. Large backward jump along track tangent → end-of-line wrap or re-read

    Book can be at any angle — all geometry is 2D track-relative.
    """

    SMOOTH_MS       = 120.0   # rolling median window for raw gaze (ms)
    DWELL_FRAMES    = 4       # frames near new line before committing
    CONF_SCALE      = 60.0    # px; distance at which confidence = 0.5
    JUMP_FRAC       = 0.5     # backward jump > 50% of track length = line-end wrap
    SACCADE_PX_MS   = 1.5     # px/ms; above = saccade, skip hysteresis update

    def __init__(self):
        self._smooth_buf: collections.deque = collections.deque(maxlen=64)
        self._locked: Optional[int] = None
        self._candidate: Optional[int] = None
        self._cand_count: int = 0
        self._line_enter_ts: float = 0.0
        self._prev_gaze: Optional[tuple[float, float, float]] = None  # (ts, x, y)
        self._prev_proj: Optional[float] = None  # projected position along last track

    # ── Public ────────────────────────────────────────────────────────────────

    def update(self,
               tracks: list[list[tuple[int, int]]],
               gaze_xy: tuple[float, float],
               ts_ms: float) -> SnapResult:
        """
        tracks: curve_track output — list of [(x,y),...] polylines, one per line.
        gaze_xy: raw gaze in scene/world cam pixel coords.
        ts_ms: current time in milliseconds.
        """
        raw = gaze_xy
        self._smooth_buf.append((ts_ms, gaze_xy[0], gaze_xy[1]))
        smooth = self._rolling_median()

        if not tracks or smooth is None:
            return SnapResult(None, 0, None, 0.0, False, False, 0.0, raw)

        # Sort tracks by their mean position projected onto the dominant reading
        # direction. For a tilted book this gives correct top→bottom ordering.
        sorted_tracks = _sort_tracks(tracks)
        n = len(sorted_tracks)
        gx, gy = smooth

        # 2D distance from gaze to each track curve
        dists, snaps = zip(*[_dist_to_track(gx, gy, t) for t in sorted_tracks])
        nearest = int(np.argmin(dists))
        nearest_dist = dists[nearest]

        # Saccade gate
        is_saccade = self._is_saccade(smooth, ts_ms)

        # Hysteresis
        is_new = False
        if not is_saccade:
            if nearest == self._locked:
                self._candidate = None
                self._cand_count = 0
            elif nearest == self._candidate:
                self._cand_count += 1
                if self._cand_count >= self.DWELL_FRAMES:
                    self._locked = nearest
                    self._candidate = None
                    self._cand_count = 0
                    self._line_enter_ts = ts_ms
                    self._prev_proj = None
                    is_new = True
            else:
                self._candidate = nearest
                self._cand_count = 1

        # First lock
        if self._locked is None:
            self._locked = nearest
            self._line_enter_ts = ts_ms
            self._prev_proj = None
            is_new = True

        locked = self._locked
        snap_xy = snaps[locked] if locked < len(snaps) else (gx, gy)

        # Regression: large backward jump along track tangent direction.
        # NOT gated by saccade — a fast backward jump IS the signal we want.
        # Only gate: must be on same line (not a new-line commit this frame).
        is_regression = False
        track = sorted_tracks[locked]
        proj = _project_along_track(gx, gy, track)
        track_len = _track_length(track)
        if (not is_new and self._prev_proj is not None and track_len > 0):
            delta = proj - self._prev_proj
            # Backward jump > JUMP_FRAC of track length = re-read or line wrap
            if delta < -self.JUMP_FRAC * track_len:
                is_regression = True
        self._prev_proj = proj

        conf = 1.0 / (1.0 + (nearest_dist / self.CONF_SCALE) ** 2)
        dwell = ts_ms - self._line_enter_ts if self._line_enter_ts else 0.0
        self._prev_gaze = (ts_ms, gx, gy)

        return SnapResult(
            line_idx=locked,
            line_count=n,
            snap_xy=snap_xy,
            confidence=float(conf),
            is_new_line=is_new,
            is_regression=is_regression,
            dwell_ms=float(dwell),
            raw_gaze_xy=raw,
        )

    def reset(self):
        self._smooth_buf.clear()
        self._locked = None
        self._candidate = None
        self._cand_count = 0
        self._line_enter_ts = 0.0
        self._prev_gaze = None
        self._prev_proj = None

    # ── Internals ─────────────────────────────────────────────────────────────

    def _rolling_median(self) -> Optional[tuple[float, float]]:
        now = self._smooth_buf[-1][0]
        cutoff = now - self.SMOOTH_MS
        pts = [(x, y) for ts, x, y in self._smooth_buf if ts >= cutoff]
        if not pts:
            return None
        return (float(np.median([p[0] for p in pts])),
                float(np.median([p[1] for p in pts])))

    def _is_saccade(self, smooth: tuple[float, float], ts_ms: float) -> bool:
        if self._prev_gaze is None:
            return False
        pt0_ts, px, py = self._prev_gaze
        dt = ts_ms - pt0_ts
        if dt < 1e-3:
            return False
        dist = ((smooth[0] - px) ** 2 + (smooth[1] - py) ** 2) ** 0.5
        return (dist / dt) > self.SACCADE_PX_MS


# ── Module-level geometry helpers ─────────────────────────────────────────────

def _sort_tracks(tracks: list[list[tuple[int, int]]]) -> list[list[tuple[int, int]]]:
    """Sort tracks top→bottom along the dominant reading direction.
    Works for any book rotation — uses mean position projected onto the
    perpendicular of the average track tangent (= across-line direction)."""
    if len(tracks) <= 1:
        return list(tracks)

    # Estimate dominant line direction from all track vectors
    vecs = []
    for t in tracks:
        if len(t) >= 2:
            pts = np.array(t, dtype=float)
            v = pts[-1] - pts[0]
            norm = np.linalg.norm(v)
            if norm > 1:
                vecs.append(v / norm)
    if vecs:
        mean_v = np.mean(vecs, axis=0)
        mean_v /= (np.linalg.norm(mean_v) + 1e-9)
    else:
        mean_v = np.array([1.0, 0.0])

    # Perpendicular to reading direction = across-line axis
    perp = np.array([-mean_v[1], mean_v[0]])

    def _key(t):
        mean_pt = np.mean(t, axis=0)
        return float(mean_pt @ perp)

    return sorted(tracks, key=_key)


def _dist_to_track(gx: float, gy: float,
                   track: list[tuple[int, int]]) -> tuple[float, tuple[float, float]]:
    """2D distance from point (gx,gy) to the nearest segment of the track polyline.
    Returns (distance, nearest_point_on_track)."""
    if not track:
        return 1e9, (gx, gy)
    if len(track) == 1:
        tx, ty = float(track[0][0]), float(track[0][1])
        return float(((gx - tx) ** 2 + (gy - ty) ** 2) ** 0.5), (tx, ty)

    best_d = 1e9
    best_pt = (float(track[0][0]), float(track[0][1]))
    g = np.array([gx, gy])

    for i in range(len(track) - 1):
        a = np.array(track[i],   dtype=float)
        b = np.array(track[i+1], dtype=float)
        ab = b - a
        ab_len2 = float(ab @ ab)
        if ab_len2 < 1e-9:
            t = 0.0
        else:
            t = float(np.clip((g - a) @ ab / ab_len2, 0.0, 1.0))
        closest = a + t * ab
        d = float(np.linalg.norm(g - closest))
        if d < best_d:
            best_d = d
            best_pt = (float(closest[0]), float(closest[1]))

    return best_d, best_pt


def _project_along_track(gx: float, gy: float,
                         track: list[tuple[int, int]]) -> float:
    """Project gaze onto track, return cumulative arc-length position."""
    if len(track) < 2:
        return 0.0
    g = np.array([gx, gy])
    pts = [np.array(p, dtype=float) for p in track]

    best_s = 0.0
    best_d = 1e9
    s = 0.0
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        seg = b - a
        seg_len = float(np.linalg.norm(seg))
        if seg_len < 1e-9:
            s += seg_len
            continue
        t = float(np.clip((g - a) @ seg / (seg_len ** 2), 0.0, 1.0))
        closest = a + t * seg
        d = float(np.linalg.norm(g - closest))
        if d < best_d:
            best_d = d
            best_s = s + t * seg_len
        s += seg_len

    return best_s


def _track_length(track: list[tuple[int, int]]) -> float:
    if len(track) < 2:
        return 0.0
    pts = np.array(track, dtype=float)
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
