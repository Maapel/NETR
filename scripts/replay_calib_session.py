#!/usr/bin/env python3
"""
Rebuild gaze samples from a saved calibration session (saccade mode) and fit GazeModel.

Requires: marker_screen_positions.json, screen_events.jsonl, eye.jsonl,
          world_cam_raw.avi, world_raw_capture_ts_ms.txt, meta.json

Usage:
  python scripts/replay_calib_session.py recordings/20260101_120000
  python scripts/replay_calib_session.py recordings/... --fit-only-from samples.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from gaze_model import GazeModel
from netr import calib_geom


def _load_ts_ms(path: Path) -> np.ndarray:
    lines = path.read_text().splitlines()
    return np.array([float(x) for x in lines if x.strip()], dtype=np.float64)


def _nearest_frame_idx(ts_arr: np.ndarray, t: float) -> int:
    return int(np.argmin(np.abs(ts_arr - float(t))))


def _load_eye_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _iter_screen_events(path: Path):
    for line in path.read_text().splitlines():
        if line.strip():
            yield json.loads(line)


def _aggregate_fixation(
    synced: list[tuple[float, float, float, float, float]],
    sx: float,
    sy: float,
) -> dict | None:
    """Apply blink + IQR filters; return one sample dict or None."""
    if not synced:
        return None
    dxs = np.array([s[0] for s in synced], dtype=float)
    dys = np.array([s[1] for s in synced], dtype=float)
    Xs = np.array([s[2] for s in synced], dtype=float)
    Ys = np.array([s[3] for s in synced], dtype=float)
    radii = np.array([s[4] for s in synced], dtype=float)

    if not np.all(np.isnan(radii)):
        r_med = float(np.nanmedian(radii))
        if r_med > 0:
            blink_mask = np.abs(radii - r_med) <= 0.15 * r_med
            if not np.any(blink_mask):
                return None
            dxs = dxs[blink_mask]
            dys = dys[blink_mask]
            Xs = Xs[blink_mask]
            Ys = Ys[blink_mask]

    if dxs.std() == 0 and dys.std() == 0:
        return None

    def _iqr_mask(arr: np.ndarray) -> np.ndarray:
        q1, q3 = np.percentile(arr, [25, 75])
        iqr = q3 - q1
        return (arr >= q1 - 1.5 * iqr) & (arr <= q3 + 1.5 * iqr)

    mask = _iqr_mask(dxs) & _iqr_mask(dys)
    n_clean = int(mask.sum())
    avg_dx = float(dxs[mask].mean() if n_clean >= 2 else dxs.mean())
    avg_dy = float(dys[mask].mean() if n_clean >= 2 else dys.mean())
    avg_X = float(Xs[mask].mean() if n_clean >= 2 else Xs.mean())
    avg_Y = float(Ys[mask].mean() if n_clean >= 2 else Ys.mean())
    return {"dx": avg_dx, "dy": avg_dy, "X": avg_X, "Y": avg_Y, "sx": sx, "sy": sy}


def replay_saccade_session(d: Path) -> tuple[list[dict], tuple[int, int] | None]:
    meta = json.loads((d / "meta.json").read_text())
    mode = meta.get("mode", "saccade")
    if mode != "saccade":
        raise SystemExit(
            f"Session mode is {mode!r}. Only 'saccade' is supported for full replay "
            "(sweep needs browser/camera time alignment — use --fit-only-from samples.json)."
        )

    markers_path = d / "marker_screen_positions.json"
    if not markers_path.is_file():
        raise SystemExit(f"Missing {markers_path.name} — cannot compute homography offline.")

    markers_doc = json.loads(markers_path.read_text())
    dict_name = markers_doc.get("aruco_dict") or meta.get("aruco_dict", "4x4")
    screen_markers = {int(k): v for k, v in markers_doc["positions"].items()}

    detector = calib_geom.create_aruco_detector(dict_name)
    world_ts = _load_ts_ms(d / "world_raw_capture_ts_ms.txt")
    eyes = _load_eye_jsonl(d / "eye.jsonl")

    cap = cv2.VideoCapture(str(d / "world_cam_raw.avi"))
    if not cap.isOpened():
        raise SystemExit("Could not open world_cam_raw.avi")

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n_frames != len(world_ts):
        print(
            f"[replay] Warning: AVI frame count ({n_frames}) != timestamp lines ({len(world_ts)}); "
            "using min length for indexing.",
            file=sys.stderr,
        )
    usable = min(n_frames, len(world_ts))

    cached_H: np.ndarray | None = None
    samples: list[dict] = []
    scene_wh: tuple[int, int] | None = None

    for ev in _iter_screen_events(d / "screen_events.jsonl"):
        if ev.get("event") != "fixation":
            continue
        win = ev.get("capture_window_ms")
        if not win or len(win) != 2:
            continue
        lo, hi = float(win[0]), float(win[1])
        sx = float(ev["x"])
        sy = float(ev["y"])

        window_eyes = [e for e in eyes if lo <= float(e["ts"]) <= hi]
        synced: list[tuple[float, float, float, float, float]] = []

        for e in window_eyes:
            t = float(e["ts"])
            idx = _nearest_frame_idx(world_ts[:usable], t)
            idx = max(0, min(idx, usable - 1))
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            if scene_wh is None:
                scene_wh = (int(frame.shape[1]), int(frame.shape[0]))

            detected, _ = calib_geom.detect_aruco_corners(frame, detector)
            if detected:
                H_use = calib_geom.compute_homography_screen_to_scene(detected, screen_markers)
                if H_use is not None:
                    cached_H = H_use
            else:
                H_use = cached_H
            if H_use is None:
                continue
            X, Y = calib_geom.screen_to_scene(H_use, sx, sy)
            r = float(e.get("r", float("nan")))
            synced.append((float(e["dx"]), float(e["dy"]), X, Y, r))

        row = _aggregate_fixation(synced, sx, sy)
        if row:
            samples.append(row)

    cap.release()
    return samples, scene_wh


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("session_dir", type=Path, help="Path to recordings/<timestamp>/")
    ap.add_argument(
        "--out-model",
        type=Path,
        default=None,
        help="Output gaze model JSON (default: <session>/gaze_model_fitted.json)",
    )
    ap.add_argument(
        "--out-samples",
        type=Path,
        default=None,
        help="Output samples JSON (default: <session>/offline_samples.json)",
    )
    ap.add_argument(
        "--fit-only-from",
        type=Path,
        default=None,
        help="Skip video replay; fit only from this samples JSON (list of {dx,dy,X,Y}).",
    )
    args = ap.parse_args()
    d = args.session_dir.resolve()
    if not d.is_dir():
        raise SystemExit(f"Not a directory: {d}")

    out_model = args.out_model or (d / "gaze_model_fitted.json")
    out_samples = args.out_samples or (d / "offline_samples.json")

    if args.fit_only_from:
        samples = json.loads(Path(args.fit_only_from).read_text())
        if not isinstance(samples, list) or len(samples) < 6:
            raise SystemExit("Need list of at least 6 samples for fit.")
        scene_wh = None
        for s in samples:
            if "X" not in s or "Y" not in s or "dx" not in s or "dy" not in s:
                raise SystemExit("Each sample needs dx, dy, X, Y")
    else:
        samples, scene_wh = replay_saccade_session(d)

    out_samples.write_text(json.dumps(samples, indent=2))
    print(f"[replay] Wrote {len(samples)} samples → {out_samples}")

    if len(samples) < 6:
        raise SystemExit(f"Need at least 6 samples after replay, got {len(samples)}")

    model = GazeModel()
    diag = model.fit([{"dx": s["dx"], "dy": s["dy"], "X": s["X"], "Y": s["Y"]} for s in samples])
    sw, sh = (None, None)
    if scene_wh:
        sw, sh = scene_wh
    model.save(out_model, scene_width=sw, scene_height=sh)
    print(f"[replay] Wrote model → {out_model}")
    print(
        f"[replay] Fit: n={diag['n_samples']}  R²x={diag['r2_x']:.4f}  R²y={diag['r2_y']:.4f}"
    )


if __name__ == "__main__":
    main()
