"""
Smooth Pursuit Calibration Server — Phase 3 + 4.

Serves the calibration webapp at http://localhost:8090
WebSocket at ws://localhost:8090/ws

Workflow:
  1. Open http://localhost:8090 in browser (fullscreen recommended).
  2. Press START — a session folder under recordings/ opens; Lissajous or saccade runs.
  3. Eye + world streams should be running (receiver.py on port 8080, analysis on).
  4. Press STOP — session is finalized, dataset is collected & gaze model is trained.
  5. Model saved to gaze_model.json; raw videos + screen_events.jsonl in recordings/<ts>/.

Scene camera: reads from the receiver's MJPEG stream (cam1 or cam2)
Eye vectors: polled from receiver `/stats` (`pccr_vector`, `pccr_ts_ms`).

Ports: HTTP/WS 8090
"""

import asyncio
import json
import time
import threading
import pathlib
import struct
import urllib.request
import urllib.error
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

import cv2
import numpy as np

from gaze_model import GazeModel
from netr import calib_geom as _calib_geom

# ── Rig config ────────────────────────────────────────────────────────────────
import sys as _sys
_sys.path.insert(0, str(pathlib.Path(__file__).parent))
try:
    import rig_config as _rig_cfg
    _WORLD_CAM = _rig_cfg.world_cam()
    _EYE_CAM = _rig_cfg.eye_cam()
except Exception:
    _WORLD_CAM = 1
    _EYE_CAM = 2

# ── Config ────────────────────────────────────────────────────────────────────
RECEIVER_URL  = "http://localhost:8080"
ENGINE_URL    = "http://localhost:8081"
SCENE_CAM_ID  = _WORLD_CAM     # which cam to use for ArUco detection (world cam from rig_config)
EYE_CAM_ID    = _EYE_CAM      # eye stream cam id (rig_config)
from netr.calib_geom import ARUCO_IDS  # TL, TR, BL, BR order
SYNC_WINDOW_MS = 150        # max ms between target coord and eye vector
MODEL_PATH    = pathlib.Path(__file__).parent / "gaze_model.json"
DATASET_PATH  = pathlib.Path(__file__).parent / "calib_dataset.json"

# ── Global state ──────────────────────────────────────────────────────────────
_ws_clients: set = set()
_ws_lock = threading.Lock()

# Incoming target coords from browser: list of {"ts", "x", "y"}
_target_buf: list[dict] = []
_target_lock = threading.Lock()

# Eye vectors polled from receiver: list of {"ts", "dx", "dy"}
_eye_buf: list[dict] = []
_eye_lock = threading.Lock()

_calibrating = False
_calib_mode  = "sweep"   # "sweep" | "saccade"

# Verbose pipeline trace (toggle via web "Trace" button → /set_calib_trace)
_calib_trace_enabled = False
_trace_push_lock = threading.Lock()
_trace_push_appended = 0
_trace_push_dropped = 0
_trace_push_last_log = 0.0


def _calib_trace(fmt: str, *args):
    if not _calib_trace_enabled:
        return
    msg = fmt % args if args else fmt
    print(f"[calib_trace] {msg}", flush=True)


def _forward_receiver_calib_trace(on: bool):
    v = "1" if on else "0"
    try:
        urllib.request.urlopen(f"{RECEIVER_URL}/set?calib_trace={v}", timeout=1).close()
    except Exception as e:
        print(f"[calib] sync calib_trace to receiver failed: {e}", flush=True)


def _trace_reset_push_counters():
    global _trace_push_appended, _trace_push_dropped, _trace_push_last_log
    with _trace_push_lock:
        _trace_push_appended = 0
        _trace_push_dropped = 0
        _trace_push_last_log = 0.0


def _trace_record_push_eye(did_append: bool):
    """Aggregated /push_eye logging (hot path)."""
    global _trace_push_appended, _trace_push_dropped, _trace_push_last_log
    if not _calib_trace_enabled:
        return
    now = time.time()
    emit = False
    ap = dr = 0
    with _trace_push_lock:
        if did_append:
            _trace_push_appended += 1
        else:
            _trace_push_dropped += 1
        ap, dr = _trace_push_appended, _trace_push_dropped
        if not did_append and dr == 1 and ap == 0:
            emit = True
        elif ap > 0 and ap % 45 == 0:
            emit = True
        elif now - _trace_push_last_log >= 1.0 and (ap + dr) > 0:
            emit = True
        if emit:
            _trace_push_last_log = now
    if emit:
        with _pending_lock:
            pb = len(_pending_eye)
        _calib_trace(
            "push_eye summary appended=%d dropped_no_pending_target=%d pending_buf=%d",
            ap, dr, pb,
        )


# ── Debug state ───────────────────────────────────────────────────────────────
_debug = {
    "scene_cam_ok": False,
    "scene_cam_error": "",
    "aruco_detected": 0,          # number of ArUco markers found in last frame
    "aruco_ids": [],
    "aruco_expected_met": False,  # last frame had all ARUCO_IDS corners (see _detect_aruco_corners)
    "homography_ok": False,
    "eye_poll_ok": False,
    "eye_poll_error": "",
    "pccr_vector": None,
    "pccr_age_ms": None,          # ms since last valid pccr
    "target_buf_size": 0,
    "eye_buf_size": 0,
    "last_sync_tried": 0,
    "last_sync_matched": 0,
}
_debug_lock = threading.Lock()
_last_scene_jpeg: bytes | None = None   # annotated scene frame for /scene_frame
_last_eye_jpeg:   bytes | None = None   # analyzed eye frame from engine /frame
_last_pccr_ts: float = 0.0
_last_homography_debug_jpeg: bytes | None = None  # plane-overlay debug frame
HOMOGRAPHY_DEBUG_DIR = pathlib.Path(__file__).parent / "homography_debug"
HOMOGRAPHY_DEBUG_DIR.mkdir(exist_ok=True)
_last_homography_debug_save_ts: float = 0.0   # throttle: save at most 1/sec

# Saccade mode: directly collected fixation samples {dx, dy, X, Y, sx, sy}
_saccade_samples: list[dict] = []
_saccade_lock = threading.Lock()

# Push-based capture: frames received from receiver during the active capture window
_pending_eye: list[dict] = []   # [{ts, dx, dy}] for the current target
_pending_target: dict | None = None  # {x, y} of the target being captured
_pending_lock = threading.Lock()
FIXATE_MS = 300   # capture window duration (matches JS FIXATE_MS)

# Saccade: each fixation flushes the *previous* target in a background thread
# (_flush_pending_target does many HTTP fetches). STOP must wait for them or
# _saccade_samples will be undercounted ("only 0–few synced samples").
_flush_inflight = 0
_flush_inflight_cv = threading.Condition(threading.Lock())


def _start_async_flush_pending(eyes: list, target: dict):
    """Run _flush_pending_target in a thread; pair with _wait_async_flushes on stop."""

    def _run():
        global _flush_inflight
        try:
            _flush_pending_target(eyes, target)
        finally:
            with _flush_inflight_cv:
                _flush_inflight -= 1
                _flush_inflight_cv.notify_all()
            _calib_trace(
                "async flush done thread=%s target=(%.1f,%.1f) had_n_eyes=%d",
                threading.current_thread().name,
                target.get("x", 0), target.get("y", 0), len(eyes),
            )

    global _flush_inflight
    with _flush_inflight_cv:
        _flush_inflight += 1
        inf = _flush_inflight
    _calib_trace(
        "async flush START inflight=%d n_eyes=%d target=(%.1f,%.1f)",
        inf, len(eyes), target.get("x", 0), target.get("y", 0),
    )
    threading.Thread(target=_run, daemon=True).start()


def _wait_async_flushes(timeout_s: float = 120.0):
    """Block until all async fixation flushes finish (or timeout)."""
    deadline = time.time() + timeout_s
    with _flush_inflight_cv:
        _calib_trace("wait_async_flushes: entering inflight=%d", _flush_inflight)
        while _flush_inflight > 0:
            remaining = deadline - time.time()
            if remaining <= 0:
                print(f"[calib] WARNING: timed out after {timeout_s:.0f}s waiting for "
                      f"{_flush_inflight} pending flush thread(s) — sample count may be low")
                break
            _flush_inflight_cv.wait(timeout=remaining)
        _calib_trace("wait_async_flushes: done inflight=%d", _flush_inflight)

_model = GazeModel()                          # scene-space model (saved to disk)
_screen_model = GazeModel()                   # screen-space model (for live cursor)
SCREEN_MODEL_PATH = pathlib.Path(__file__).parent / "screen_model.json"
_model.load(MODEL_PATH)
_screen_model.load(SCREEN_MODEL_PATH)

# ── Recording ─────────────────────────────────────────────────────────────────
# Per calibration session (START → STOP): raw sensor videos + aligned timestamps.
#   world_cam_raw.avi + world_raw_capture_ts_ms.txt — ESP32 capture time (ms)
#   eye_cam_raw.avi   + eye_raw_capture_ts_ms.txt
#   screen_events.jsonl — stimulus; ts_browser_ms; fixations include capture_window_ms
#   eye.jsonl — PCCR keyed by ts = pccr_ts_ms (same camera timebase as eye video)
RECORDINGS_DIR = pathlib.Path(__file__).parent / "recordings"
_recording         = False
_rec_dir: pathlib.Path | None = None
_rec_eye_f         = None   # eye.jsonl
_rec_target_f      = None   # targets.jsonl (legacy mirror of stimulus rows)
_rec_fixation_f    = None   # fixations.jsonl
_rec_screen_f      = None   # screen_events.jsonl (canonical stimulus log)
_rec_frame_n       = 0
_rec_lock          = threading.Lock()
_rec_world_writer: "cv2.VideoWriter | None" = None   # world_cam_raw.avi
_rec_eye_writer:   "cv2.VideoWriter | None" = None   # eye_cam_raw.avi
_rec_homo_writer:  "cv2.VideoWriter | None" = None   # homography_debug.avi
_rec_world_ts_f    = None   # world_raw_capture_ts_ms.txt
_rec_eye_ts_f      = None   # eye_raw_capture_ts_ms.txt
_rec_homo_ts_f     = None   # homography_debug_capture_ts_ms.txt


def _receiver_stats_snapshot() -> dict:
    try:
        with urllib.request.urlopen(f"{RECEIVER_URL}/stats", timeout=0.5) as r:
            return json.loads(r.read().decode())
    except Exception:
        return {}


def _append_screen_event(obj: dict):
    """Append one JSON line to screen_events.jsonl during an active session."""
    global _rec_screen_f
    if not (_recording and _rec_screen_f):
        return
    try:
        _rec_screen_f.write(json.dumps(obj) + "\n")
        _rec_screen_f.flush()
    except Exception:
        pass


def _persist_marker_screen_positions():
    """Write ArUco marker centers in calibration browser pixels for offline homography."""
    global _rec_dir, _recording
    if not (_recording and _rec_dir):
        return
    with _screen_markers_lock:
        sm = {str(int(k)): [float(v[0]), float(v[1])] for k, v in _screen_markers.items()}
    try:
        with open(_rec_dir / "marker_screen_positions.json", "w") as f:
            json.dump({
                "aruco_dict": _aruco_dict_name,
                "positions": sm,
                "updated_server_ms": time.time() * 1000.0,
            }, f, indent=2)
    except Exception:
        pass


def _rec_start(screen_w: int, screen_h: int, mode: str):
    global _recording, _rec_dir, _rec_eye_f, _rec_target_f, _rec_fixation_f, _rec_screen_f
    global _rec_frame_n, _rec_world_writer, _rec_eye_writer, _rec_homo_writer
    global _rec_world_ts_f, _rec_eye_ts_f, _rec_homo_ts_f
    ts = time.strftime("%Y%m%d_%H%M%S")
    d  = RECORDINGS_DIR / ts
    d.mkdir(parents=True, exist_ok=True)
    stats = _receiver_stats_snapshot()
    meta = {
        "session_id": ts,
        "screen_w": screen_w,
        "screen_h": screen_h,
        "mode": mode,
        "aruco_dict": _aruco_dict_name,
        "scene_cam": SCENE_CAM_ID,
        "eye_cam": EYE_CAM_ID,
        "sync_offset_ms": stats.get("sync_offset_ms"),
        "timebases": {
            "stimulus": "browser_epoch_ms",
            "cameras_and_pccr": "camera_epoch_ms",
        },
        "outputs": {
            "world_video": "world_cam_raw.avi",
            "world_timestamps": "world_raw_capture_ts_ms.txt",
            "eye_video": "eye_cam_raw.avi",
            "eye_timestamps": "eye_raw_capture_ts_ms.txt",
            "eye_vectors": "eye.jsonl",
            "stimulus": "screen_events.jsonl",
            "stimulus_legacy": "targets.jsonl",
            "homography_debug_video": "homography_debug.avi",
            "homography_debug_timestamps": "homography_debug_capture_ts_ms.txt",
            "aggregated_fixations": "fixations.jsonl",
            "training_samples": "samples.json",
            "marker_screen_positions": "marker_screen_positions.json",
        },
        "started_wall": ts,
        "status": "in_progress",
    }
    _rec_dir        = d
    _rec_frame_n    = 0
    _rec_world_writer = None
    _rec_eye_writer   = None
    _rec_homo_writer  = None
    _rec_eye_f      = open(d / "eye.jsonl",      "w")
    _rec_target_f   = open(d / "targets.jsonl",  "w")
    _rec_fixation_f = open(d / "fixations.jsonl", "w")
    _rec_screen_f   = open(d / "screen_events.jsonl", "w")
    _rec_world_ts_f = open(d / "world_raw_capture_ts_ms.txt", "w")
    _rec_eye_ts_f   = open(d / "eye_raw_capture_ts_ms.txt",   "w")
    _rec_homo_ts_f  = open(d / "homography_debug_capture_ts_ms.txt", "w")
    with open(d / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    _recording = True
    print(f"[rec] Recording started → {d}")
    return str(d)


def _rec_stop() -> str:
    global _recording, _rec_dir
    global _rec_eye_f, _rec_target_f, _rec_fixation_f, _rec_screen_f
    global _rec_world_writer, _rec_eye_writer, _rec_homo_writer
    global _rec_world_ts_f, _rec_eye_ts_f, _rec_homo_ts_f
    _recording = False
    for w in (_rec_world_writer, _rec_eye_writer, _rec_homo_writer):
        try:
            if w: w.release()
        except Exception: pass
    _rec_world_writer = _rec_eye_writer = _rec_homo_writer = None
    for f in (_rec_eye_f, _rec_target_f, _rec_fixation_f, _rec_screen_f,
              _rec_world_ts_f, _rec_eye_ts_f, _rec_homo_ts_f):
        try:
            if f: f.close()
        except Exception: pass
    _rec_eye_f = _rec_target_f = _rec_fixation_f = _rec_screen_f = None
    _rec_world_ts_f = _rec_eye_ts_f = _rec_homo_ts_f = None
    d = _rec_dir
    path = str(d) if d else ""
    if d and (d / "meta.json").exists():
        try:
            stats = _receiver_stats_snapshot()
            with open(d / "meta.json") as f:
                meta = json.load(f)
            meta["status"] = "complete"
            meta["ended_wall"] = time.strftime("%Y%m%d_%H%M%S")
            meta["sync_offset_ms_end"] = stats.get("sync_offset_ms")
            with open(d / "meta.json", "w") as f:
                json.dump(meta, f, indent=2)
        except Exception:
            pass
    print(f"[rec] Recording saved → {path}")
    return path

def _set_calib_window(x: float, y: float, from_ms: float, until_ms: float):
    """Tell receiver to push eye frames for this target window. Fire-and-forget."""
    def _send():
        try:
            body = json.dumps({"x": x, "y": y,
                               "from_ms": from_ms, "until_ms": until_ms}).encode()
            req = urllib.request.Request(
                f"{RECEIVER_URL}/calib_window",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=0.2).close()
            _calib_trace(
                "calib_window POST ok xy=(%.1f,%.1f) from_ms=%.1f until_ms=%.1f span=%.0fms",
                x, y, from_ms, until_ms, until_ms - from_ms,
            )
        except Exception as e:
            _calib_trace("calib_window POST FAILED: %r", e)
    threading.Thread(target=_send, daemon=True).start()


def _clear_calib_window():
    """Tell receiver to stop capturing (window expired)."""
    def _send():
        try:
            req = urllib.request.Request(
                f"{RECEIVER_URL}/calib_window",
                data=b'{"clear":true}',
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=0.2).close()
            _calib_trace("calib_window POST clear ok")
        except Exception as e:
            _calib_trace("calib_window POST clear FAILED: %r", e)
    threading.Thread(target=_send, daemon=True).start()


def _fetch_world_frame_at(ts_ms: float) -> np.ndarray | None:
    """Fetch the world-cam JPEG closest to ts_ms from receiver and decode it."""
    url = f"{RECEIVER_URL}/closest_frame?cam={SCENE_CAM_ID}&ts_ms={ts_ms:.3f}"
    try:
        with urllib.request.urlopen(url, timeout=0.5) as r:
            data = r.read()
        arr = np.frombuffer(data, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None


def _snapshot_pending() -> tuple[list, dict | None]:
    """Atomically drain _pending_eye/_pending_target and return the snapshot."""
    global _pending_eye, _pending_target
    with _pending_lock:
        eyes          = list(_pending_eye)
        target        = _pending_target
        _pending_eye   = []
        _pending_target = None
    return eyes, target


def _flush_pending_target(eyes: list | None = None, target: dict | None = None):
    """For each captured eye frame, fetch the world frame at the same timestamp,
    run ArUco, compute scene_xy, then average all synced (dx,dy,X,Y) pairs.

    Pass eyes/target when caller has already snapshotted (fixation handler).
    Call with no args to snapshot now (stop handler).
    """
    if eyes is None or target is None:
        eyes, target = _snapshot_pending()
    if not eyes or not target:
        _calib_trace(
            "flush_pending SKIP: eyes=%d target=%s (empty snapshot or stop race)",
            len(eyes) if eyes else 0, "ok" if target else "None",
        )
        return
    import numpy as _np

    sx, sy = target["x"], target["y"]
    tss = [e["ts"] for e in eyes]
    _calib_trace(
        "flush_pending START target=(%.1f,%.1f) n_eyes=%d ts_eye[min..max]=[%.1f..%.1f]",
        sx, sy, len(eyes), min(tss), max(tss),
    )

    # Build per-sample synced tuples: fetch world frame at each eye timestamp
    synced: list[tuple] = []   # (dx, dy, X, Y, r|nan)
    # Cache the global homography once as fallback for frames where per-frame
    # ArUco fails (transient occlusion, motion blur).  If it's also None the
    # session never had a valid homography — samples will still be 0.
    with _homography_lock:
        cached_H = _homography
    _calib_trace("flush_pending cached_H=%s", "None" if cached_H is None else "present")

    miss_no_frame = miss_no_aruco = miss_no_homo = miss_no_cached = 0
    for e in eyes:
        frame = _fetch_world_frame_at(e["ts"])
        if frame is None:
            miss_no_frame += 1
            H_use = cached_H   # frame gone from buffer — use cached homography
        else:
            detected, _ = _detect_aruco_corners(frame)
            if not detected:
                miss_no_aruco += 1
                H_use = cached_H
            else:
                H_use = _compute_homography(detected)
                if H_use is None:
                    miss_no_homo += 1
                    H_use = cached_H
        if H_use is None:
            miss_no_cached += 1
            continue
        pt = np.array([[[sx, sy]]], dtype=np.float32)
        sc = cv2.perspectiveTransform(pt, H_use)[0][0]
        r  = e.get("r", float("nan"))
        synced.append((e["dx"], e["dy"], float(sc[0]), float(sc[1]), r))

    n_synced = len(synced)
    print(f"[calib] Target ({sx:.0f},{sy:.0f}) n={len(eyes)} synced={n_synced} "
          f"miss(no_frame={miss_no_frame} no_aruco={miss_no_aruco} "
          f"no_homo={miss_no_homo} no_cached={miss_no_cached})")
    if n_synced == 0:
        print(f"[calib] No synced samples for target ({sx:.0f},{sy:.0f}) — skipped")
        _calib_trace(
            "flush_pending n_synced=0 all_H_none=%s miss nf=%d na=%d nh=%d nc=%d",
            str(miss_no_cached == len(eyes)),
            miss_no_frame, miss_no_aruco, miss_no_homo, miss_no_cached,
        )
        return

    dxs = _np.array([s[0] for s in synced])
    dys = _np.array([s[1] for s in synced])
    Xs  = _np.array([s[2] for s in synced])
    Ys  = _np.array([s[3] for s in synced])

    # Radius pre-filter: drop blink/occlusion frames before IQR.
    # A biological pupil cannot jump >15% instantaneously — any frame that does
    # is an eyelid artifact (eyelid slicing the contour during a blink).
    radii = _np.array([s[4] for s in synced], dtype=float)
    if not _np.all(_np.isnan(radii)):
        r_med = _np.nanmedian(radii)
        if r_med > 0:
            blink_mask = _np.abs(radii - r_med) <= 0.15 * r_med
            n_blink = int((~blink_mask).sum())
            if n_blink:
                print(f"[calib] Blink filter: dropped {n_blink}/{len(synced)} frames "
                      f"(r_med={r_med:.1f}, threshold=±{0.15*r_med:.1f}px)")
            if not _np.any(blink_mask):
                print(f"[calib] All frames failed blink filter for ({sx:.0f},{sy:.0f}) — discarding target")
                _calib_trace("flush_pending DISCARD all blink filter fail n=%d", len(synced))
                return
            dxs = dxs[blink_mask]
            dys = dys[blink_mask]
            Xs  = Xs[blink_mask]
            Ys  = Ys[blink_mask]

    # Drop frozen (all identical) — tracking stuck
    if dxs.std() == 0 and dys.std() == 0:
        print(f"[calib] Skipping frozen target ({sx:.0f},{sy:.0f}) n={n_synced}")
        _calib_trace("flush_pending DISCARD frozen PCCR n=%d", n_synced)
        return

    # IQR outlier rejection on PCCR
    def _iqr_mask(arr):
        q1, q3 = _np.percentile(arr, [25, 75])
        iqr = q3 - q1
        return (arr >= q1 - 1.5*iqr) & (arr <= q3 + 1.5*iqr)
    mask = _iqr_mask(dxs) & _iqr_mask(dys)
    n_clean = int(mask.sum())
    avg_dx = float(dxs[mask].mean() if n_clean >= 2 else dxs.mean())
    avg_dy = float(dys[mask].mean() if n_clean >= 2 else dys.mean())
    avg_X  = float(Xs[mask].mean()  if n_clean >= 2 else Xs.mean())
    avg_Y  = float(Ys[mask].mean()  if n_clean >= 2 else Ys.mean())

    print(f"[calib] Target ({sx:.0f},{sy:.0f}) clean={n_clean} "
          f"dx={avg_dx:.2f}±{dxs.std():.2f} dy={avg_dy:.2f}±{dys.std():.2f} "
          f"X={avg_X:.1f} Y={avg_Y:.1f}")

    # Record fixation
    if _recording and _rec_fixation_f:
        try:
            _rec_fixation_f.write(json.dumps({
                "ts": eyes[-1]["ts"], "x": sx, "y": sy,
                "eye": {"dx": avg_dx, "dy": avg_dy, "n": n_synced,
                        "dx_std": float(dxs.std()), "dy_std": float(dys.std()),
                        "eye_ts": eyes[-1]["ts"]},
                "scene": {"X": avg_X, "Y": avg_Y},
            }) + "\n")
            _rec_fixation_f.flush()
        except Exception: pass

    # Add to calibration dataset — scene_xy is the temporally-matched average
    with _saccade_lock:
        _saccade_samples.append({"dx": avg_dx, "dy": avg_dy,
                                 "X": avg_X, "Y": avg_Y, "sx": sx, "sy": sy})
        n = len(_saccade_samples)
    _calib_trace(
        "flush_pending APPEND sample #%d clean=%d avg dx=%.3f dy=%.3f scene X=%.1f Y=%.1f",
        n, n_clean, avg_dx, avg_dy, avg_X, avg_Y,
    )
    diag = _refit_models()
    if diag:
        _broadcast(json.dumps({
            "type": "ready",
            "n": n,
            "r2_x": round(diag["r2_x"], 3),
            "r2_y": round(diag["r2_y"], 3),
        }))


def _refit_models():
    """Refit both models from current saccade samples. Called after each fixation."""
    with _saccade_lock:
        samples = list(_saccade_samples)
    if len(samples) < 6:
        return None
    try:
        diag = _model.fit(samples)
        _screen_model.fit([{"dx": s["dx"], "dy": s["dy"],
                            "X": s["sx"], "Y": s["sy"]} for s in samples])
        _model.save(MODEL_PATH)
        _screen_model.save(SCREEN_MODEL_PATH)
        return diag
    except Exception:
        return None

# ── ArUco helpers ─────────────────────────────────────────────────────────────
_aruco_dict_name = "4x4"
_aruco_lock      = threading.Lock()
_aruco_dict      = cv2.aruco.getPredefinedDictionary(_calib_geom.ARUCO_DICTS[_aruco_dict_name])
_aruco_detector  = _calib_geom.create_aruco_detector(_aruco_dict_name)


def _generate_marker_png(marker_id: int, size: int = 200) -> bytes:
    with _aruco_lock:
        d = _aruco_dict
    img = cv2.aruco.generateImageMarker(d, marker_id, size)
    _, buf = cv2.imencode(".png", img)
    return buf.tobytes()


_marker_pngs: dict[int, bytes] = {i: _generate_marker_png(i) for i in ARUCO_IDS}
print(f"[aruco] Markers ready: DICT_{_aruco_dict_name.upper()}_50 IDs {ARUCO_IDS}")


def _switch_aruco_dict(name: str):
    global _aruco_dict, _aruco_dict_name, _aruco_detector, _marker_pngs, _homography
    if name not in _calib_geom.ARUCO_DICTS:
        print(f"[aruco] Unknown dict '{name}', options: {list(_calib_geom.ARUCO_DICTS)}")
        return
    with _aruco_lock:
        _aruco_dict_name = name
        _aruco_dict      = cv2.aruco.getPredefinedDictionary(_calib_geom.ARUCO_DICTS[name])
        _aruco_detector  = _calib_geom.create_aruco_detector(name)
    _marker_pngs = {i: _generate_marker_png(i) for i in ARUCO_IDS}
    with _homography_lock:
        _homography = None   # invalidate — markers changed
    print(f"[aruco] Switched to DICT_{name.upper()}_50, regenerated markers")

# Cached homography (recomputed when scene frame updates)
_homography: np.ndarray | None = None
_homography_lock = threading.Lock()
# Actual marker screen positions sent from browser {id: [x,y], ...}
_screen_markers: dict = {}
_screen_markers_lock = threading.Lock()


def _detect_aruco_corners(frame_bgr: np.ndarray) -> tuple[dict[int, np.ndarray] | None, list[int]]:
    """Detect ArUco markers. Returns (dict id->center, list of all found ids). dict is None if <4 required found."""
    with _aruco_lock:
        det = _aruco_detector
    return _calib_geom.detect_aruco_corners(frame_bgr, det)


def _annotate_scene_frame(frame_bgr: np.ndarray, detected: dict | None, all_ids: list[int]) -> bytes:
    """Draw ArUco detection results onto frame, return JPEG bytes."""
    out = frame_bgr.copy()
    h, w = out.shape[:2]
    # Draw all detected markers
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    with _aruco_lock:
        det = _aruco_detector
    corners, ids, _ = det.detectMarkers(gray)
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(out, corners, ids)
    # Status overlay
    status_color = (0, 220, 0) if detected else (0, 80, 255)
    status_text  = f"ArUco: {len(all_ids)} found  IDs={all_ids}" if all_ids else "ArUco: none detected"
    cv2.rectangle(out, (0, 0), (w, 28), (0, 0, 0), -1)
    cv2.putText(out, status_text, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 1, cv2.LINE_AA)
    if detected:
        cv2.putText(out, "HOMOGRAPHY OK", (w - 180, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 0), 1, cv2.LINE_AA)
        for mid, center in detected.items():
            cx, cy = int(center[0]), int(center[1])
            cv2.circle(out, (cx, cy), 8, (0, 255, 255), -1)
            cv2.putText(out, f"ID{mid}", (cx + 10, cy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    else:
        need = set(ARUCO_IDS) - set(all_ids)
        cv2.putText(out, f"MISSING IDs: {sorted(need)}", (w - 240, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 80, 255), 1, cv2.LINE_AA)
    # Project active calibration target onto scene frame
    with _pending_lock:
        tgt = _pending_target
    if tgt is not None:
        scene_xy = _screen_to_scene(tgt["x"], tgt["y"])
        if scene_xy:
            tx, ty = int(scene_xy[0]), int(scene_xy[1])
            arm = 30
            cv2.circle(out, (tx, ty), 16, (0, 0, 0),   3)
            cv2.circle(out, (tx, ty), 16, (0, 0, 255),  2)
            cv2.line(out, (tx - arm, ty), (tx + arm, ty), (0, 0, 0),   2)
            cv2.line(out, (tx, ty - arm), (tx, ty + arm), (0, 0, 0),   2)
            cv2.line(out, (tx - arm, ty), (tx + arm, ty), (0, 0, 255), 1)
            cv2.line(out, (tx, ty - arm), (tx, ty + arm), (0, 0, 255), 1)
            cv2.circle(out, (tx, ty), 3, (255, 255, 255), -1)
            cv2.putText(out, f"({tgt['x']:.0f},{tgt['y']:.0f})",
                        (tx + 20, ty - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
    _, jpg = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return jpg.tobytes()


def _compute_homography(scene_corners: dict[int, np.ndarray]) -> np.ndarray | None:
    """
    Compute homography from screen space → scene camera space.
    Uses actual marker positions sent from browser (_screen_markers).
    ArUco IDs 0,1,2,3 = TL, TR, BL, BR.
    """
    with _screen_markers_lock:
        sm = dict(_screen_markers)
    H = _calib_geom.compute_homography_screen_to_scene(scene_corners, sm)
    if H is None:
        print(f"[homography] Missing screen marker positions: have {list(sm.keys())}, need {ARUCO_IDS}")
    return H


def _screen_to_scene(x: float, y: float) -> tuple[float, float] | None:
    with _homography_lock:
        H = _homography
    if H is None:
        return None
    pt = np.array([[[x, y]]], dtype=np.float32)
    res = cv2.perspectiveTransform(pt, H)
    return float(res[0][0][0]), float(res[0][0][1])


def _render_homography_debug(frame_bgr: np.ndarray, H: np.ndarray,
                              tgt: dict | None) -> np.ndarray:
    """Render homography verification overlay onto a copy of frame_bgr. Returns BGR."""
    out = frame_bgr.copy()

    with _screen_markers_lock:
        sm = dict(_screen_markers)

    # Screen corners ordered TL→TR→BR→BL (IDs 0,1,3,2)
    corner_order = [0, 1, 3, 2]
    if all(i in sm for i in corner_order):
        screen_quad = np.array([[sm[i]] for i in corner_order], dtype=np.float32)
        scene_quad  = cv2.perspectiveTransform(screen_quad, H)
        pts = scene_quad.reshape(-1, 1, 2).astype(np.int32)

        overlay = out.copy()
        cv2.fillPoly(overlay, [pts], (0, 200, 60))
        cv2.addWeighted(overlay, 0.30, out, 0.70, 0, out)
        cv2.polylines(out, [pts], isClosed=True, color=(0, 255, 80), thickness=2)

        labels = {0: "TL", 1: "TR", 2: "BL", 3: "BR"}
        for idx, sid in enumerate(corner_order):
            px = int(scene_quad[idx, 0, 0])
            py = int(scene_quad[idx, 0, 1])
            cv2.circle(out, (px, py), 7, (0, 255, 255), -1)
            cv2.putText(out, labels[sid], (px + 9, py - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

    if tgt is not None:
        pt = np.array([[[tgt["x"], tgt["y"]]]], dtype=np.float32)
        sc = cv2.perspectiveTransform(pt, H)[0][0]
        tx, ty = int(sc[0]), int(sc[1])
        # Thin outer ring — centre stays visible, easy to judge precision
        cv2.circle(out, (tx, ty), 16, (0, 0, 0),   3)   # black shadow
        cv2.circle(out, (tx, ty), 16, (0, 0, 255),  2)   # red ring
        # Fine crosshair extending well beyond the ring
        arm = 30
        cv2.line(out, (tx - arm, ty), (tx + arm, ty), (0, 0, 0),   2)
        cv2.line(out, (tx, ty - arm), (tx, ty + arm), (0, 0, 0),   2)
        cv2.line(out, (tx - arm, ty), (tx + arm, ty), (0, 0, 255), 1)
        cv2.line(out, (tx, ty - arm), (tx, ty + arm), (0, 0, 255), 1)
        # Small filled centre dot so the exact pixel is unambiguous
        cv2.circle(out, (tx, ty), 3, (255, 255, 255), -1)
        cv2.putText(out, f"({tgt['x']:.0f},{tgt['y']:.0f})",
                    (tx + 20, ty - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

    cv2.putText(out, f"homography check  {time.strftime('%H:%M:%S')}",
                (6, out.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
    return out


def _make_homography_debug(frame_bgr: np.ndarray, H: np.ndarray,
                            tgt: dict | None) -> bytes:
    out = _render_homography_debug(frame_bgr, H, tgt)
    _, jpg = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return jpg.tobytes()


# ── Calibration shutter video ─────────────────────────────────────────────────
_calib_video_buf:    list[np.ndarray] = []   # BGR frames for current window
_calib_video_tgt:    dict | None = None      # target being recorded
_calib_video_tgt_id: int = -1               # id() of _calib_video_tgt for change detection
_calib_video_lock    = threading.Lock()


def _write_calib_video(frames: list[np.ndarray], tgt: dict):
    """Write accumulated BGR frames to homography_debug/<ts>_sx<x>_sy<y>.avi."""
    if not frames:
        return
    h, w = frames[0].shape[:2]
    ts   = time.strftime("%Y%m%d_%H%M%S")
    path = HOMOGRAPHY_DEBUG_DIR / f"{ts}_sx{tgt['x']:.0f}_sy{tgt['y']:.0f}.avi"
    fps  = len(frames) / 0.3   # window is 300 ms — preserve real timing
    fps  = max(1.0, min(fps, 60.0))
    vw   = cv2.VideoWriter(str(path),
                           cv2.VideoWriter_fourcc(*"MJPG"),
                           fps, (w, h))
    for f in frames:
        vw.write(f)
    vw.release()
    print(f"[calib] Saved shutter video ({len(frames)} frames, {fps:.1f}fps) → {path.name}")


def _flush_calib_video_buf():
    """Flush any pending shutter video to disk (call at calibration stop)."""
    global _calib_video_buf, _calib_video_tgt, _calib_video_tgt_id
    buf = tgt = None
    with _calib_video_lock:
        if _calib_video_buf and _calib_video_tgt:
            buf = _calib_video_buf[:]
            tgt = _calib_video_tgt
            _calib_video_buf.clear()
            _calib_video_tgt    = None
            _calib_video_tgt_id = -1
    if buf and tgt:
        threading.Thread(target=_write_calib_video, args=(buf, tgt), daemon=True).start()


# ── Scene cam poller ──────────────────────────────────────────────────────────
def _scene_cam_thread():
    """Periodically grab a frame from the receiver MJPEG stream and update homography."""
    global _homography, _last_scene_jpeg
    global _rec_world_writer, _rec_homo_writer
    global _calib_video_tgt, _calib_video_tgt_id
    snap_url = f"{RECEIVER_URL}/jpeg/{SCENE_CAM_ID}?raw=1"
    prev_aruco_count = -1
    while True:
        try:
            capture_ms = time.time() * 1000.0
            with urllib.request.urlopen(snap_url, timeout=2) as resp:
                h = resp.headers.get("X-Capture-Ts-Ms")
                if h is not None:
                    try:
                        capture_ms = float(h)
                    except ValueError:
                        pass
                data = resp.read()
            arr = np.frombuffer(data, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is not None:
                h_px, w_px = frame.shape[:2]
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                mean_brightness = float(gray.mean())
                detected, all_ids = _detect_aruco_corners(frame)
                # Store annotated frame for /scene_frame endpoint
                _last_scene_jpeg = _annotate_scene_frame(frame, detected, all_ids)
                n = len(all_ids)
                with _debug_lock:
                    _debug["scene_cam_ok"] = True
                    _debug["scene_cam_error"] = ""
                    _debug["aruco_detected"] = n
                    _debug["aruco_ids"] = all_ids
                    _debug["aruco_expected_met"] = detected is not None
                    _debug["frame_size"] = [w_px, h_px]
                    _debug["brightness"] = round(mean_brightness, 1)
                if n != prev_aruco_count:
                    print(f"[scene] frame={w_px}×{h_px}  brightness={mean_brightness:.0f}/255"
                          f"  ArUco: {n} found IDs={all_ids}" +
                          (f"  MISSING={sorted(set(ARUCO_IDS)-set(all_ids))}" if n < 4 else "  → homography OK"))
                    prev_aruco_count = n
                # World cam recording — raw frames + ESP32 capture timestamps
                if _recording and _rec_dir:
                    with _rec_lock:
                        if _rec_world_writer is None:
                            _h, _w = frame.shape[:2]
                            _rec_world_writer = cv2.VideoWriter(
                                str(_rec_dir / "world_cam_raw.avi"),
                                cv2.VideoWriter_fourcc(*"MJPG"),
                                20.0, (_w, _h))
                        _rec_world_writer.write(frame)
                        if _rec_world_ts_f:
                            _rec_world_ts_f.write(f"{capture_ms:.3f}\n")

                if detected:
                    H = _compute_homography(detected)
                    if H is not None:
                        with _homography_lock:
                            _homography = H
                        with _debug_lock:
                            _debug["homography_ok"] = True
                        global _last_homography_debug_jpeg, _last_homography_debug_save_ts
                        with _pending_lock:
                            tgt = _pending_target
                        debug_bgr = _render_homography_debug(frame, H, tgt)
                        _, _jpg = cv2.imencode(".jpg", debug_bgr,
                                              [cv2.IMWRITE_JPEG_QUALITY, 90])
                        _last_homography_debug_jpeg = _jpg.tobytes()
                        # Accumulate into per-target shutter video (id() for change detection)
                        with _calib_video_lock:
                            cur_id = id(tgt) if tgt is not None else -1
                            if tgt is not None:
                                if cur_id != _calib_video_tgt_id:
                                    if _calib_video_buf and _calib_video_tgt:
                                        _prev = (_calib_video_buf[:], _calib_video_tgt)
                                        threading.Thread(
                                            target=_write_calib_video,
                                            args=_prev, daemon=True).start()
                                    _calib_video_buf.clear()
                                    _calib_video_tgt    = tgt
                                    _calib_video_tgt_id = cur_id
                                _calib_video_buf.append(debug_bgr.copy())
                            else:
                                now = time.time()
                                if now - _last_homography_debug_save_ts >= 1.0:
                                    _last_homography_debug_save_ts = now
                                    fname = HOMOGRAPHY_DEBUG_DIR / f"{time.strftime('%Y%m%d_%H%M%S')}.jpg"
                                    fname.write_bytes(_last_homography_debug_jpeg)
                        # Homography debug recording — same capture time as world frame
                        if _recording and _rec_dir:
                            with _rec_lock:
                                if _rec_homo_writer is None:
                                    _h, _w = debug_bgr.shape[:2]
                                    _rec_homo_writer = cv2.VideoWriter(
                                        str(_rec_dir / "homography_debug.avi"),
                                        cv2.VideoWriter_fourcc(*"MJPG"),
                                        20.0, (_w, _h))
                                _rec_homo_writer.write(debug_bgr)
                                if _rec_homo_ts_f:
                                    _rec_homo_ts_f.write(f"{capture_ms:.3f}\n")
                    else:
                        print("[scene] Waiting for marker positions from browser (resize the window or move slider)…")
                else:
                    with _homography_lock:
                        _homography = None
                    with _debug_lock:
                        _debug["homography_ok"] = False
            else:
                print(f"[scene] Frame decode failed (empty response from {snap_url})")
                with _debug_lock:
                    _debug["scene_cam_ok"] = False
                    _debug["scene_cam_error"] = "frame decode failed"
                    _debug["aruco_expected_met"] = False
        except Exception as e:
            err = str(e)
            with _debug_lock:
                _debug["scene_cam_ok"] = False
                _debug["scene_cam_error"] = err
                _debug["homography_ok"] = False
                _debug["aruco_expected_met"] = False
            print(f"[scene] Error fetching {snap_url}: {err}")
        time.sleep(0.05)  # ~20 fps

# ── Eye cam poller (receiver /jpeg — annotated preview + raw session video) ─
def _eye_cam_thread():
    """Serve annotated eye preview; during recording, also append raw eye AVI + ts."""
    global _last_eye_jpeg, _rec_eye_writer
    preview_url = f"{RECEIVER_URL}/jpeg/{EYE_CAM_ID}"
    raw_url = f"{RECEIVER_URL}/jpeg/{EYE_CAM_ID}?raw=1"
    while True:
        try:
            with urllib.request.urlopen(preview_url, timeout=1) as resp:
                data = resp.read()
            if data:
                _last_eye_jpeg = data
            if _recording and _rec_dir:
                cap_ms = time.time() * 1000.0
                with urllib.request.urlopen(raw_url, timeout=1) as resp2:
                    h2 = resp2.headers.get("X-Capture-Ts-Ms")
                    if h2 is not None:
                        try:
                            cap_ms = float(h2)
                        except ValueError:
                            pass
                    raw_data = resp2.read()
                arr = np.frombuffer(raw_data, np.uint8)
                frm = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frm is not None:
                    with _rec_lock:
                        if _rec_eye_writer is None:
                            _h, _w = frm.shape[:2]
                            _rec_eye_writer = cv2.VideoWriter(
                                str(_rec_dir / "eye_cam_raw.avi"),
                                cv2.VideoWriter_fourcc(*"MJPG"),
                                20.0, (_w, _h))
                        _rec_eye_writer.write(frm)
                        if _rec_eye_ts_f:
                            _rec_eye_ts_f.write(f"{cap_ms:.3f}\n")
        except Exception:
            pass
        time.sleep(0.05)

_last_screen_size = [0, 0]   # updated by websocket messages


# ── Eye vector poller ─────────────────────────────────────────────────────────
def _eye_poll_thread():
    """Poll receiver /stats for PCCR + pccr_ts_ms (camera capture time, same as eye video)."""
    global _last_pccr_ts
    stats_url = f"{RECEIVER_URL}/stats"
    prev_vec_ok = None
    _last_pccr_ts_seen: float = 0.0   # dedupe: same camera frame must not append twice

    while True:
        try:
            with urllib.request.urlopen(stats_url, timeout=1) as resp:
                d = json.loads(resp.read().decode())
            vec = d.get("pccr_vector")
            pccr_ts_ms = d.get("pccr_ts_ms")

            if vec and len(vec) == 2 and pccr_ts_ms is not None:
                pccr_ts_ms = float(pccr_ts_ms)
                _last_pccr_ts = pccr_ts_ms
                with _debug_lock:
                    _debug["eye_poll_ok"] = True
                    _debug["eye_poll_error"] = ""
                    _debug["pccr_vector"] = [round(vec[0], 3), round(vec[1], 3)]
                    _debug["pccr_age_ms"] = 0
                if prev_vec_ok is not True:
                    print(f"[eye] pccr_vector OK (camera ts_ms): {vec}")
                    prev_vec_ok = True

                if pccr_ts_ms > _last_pccr_ts_seen:
                    _last_pccr_ts_seen = pccr_ts_ms
                    entry = {"ts": pccr_ts_ms, "dx": vec[0], "dy": vec[1]}
                    with _eye_lock:
                        _eye_buf.append(entry)
                        cutoff = pccr_ts_ms - 10000
                        while _eye_buf and _eye_buf[0]["ts"] < cutoff:
                            _eye_buf.pop(0)
                    if _recording and _rec_eye_f:
                        try:
                            _rec_eye_f.write(json.dumps(entry) + "\n")
                            _rec_eye_f.flush()
                        except Exception:
                            pass
            else:
                poll_ts = time.time() * 1000
                age = poll_ts - _last_pccr_ts if _last_pccr_ts else None
                with _debug_lock:
                    _debug["pccr_vector"] = None
                    _debug["pccr_age_ms"] = round(age) if age else None
                if prev_vec_ok is not False:
                    print("[eye] pccr_vector is null — check receiver analysis=1 and eye cam.")
                    prev_vec_ok = False
        except Exception as e:
            err = str(e)
            with _debug_lock:
                _debug["eye_poll_ok"] = False
                _debug["eye_poll_error"] = err
            if prev_vec_ok is not False:
                print(f"[eye] Error polling {stats_url}: {err}")
                prev_vec_ok = False
        with _debug_lock:
            _debug["target_buf_size"] = len(_target_buf)
            _debug["eye_buf_size"] = len(_eye_buf)
        time.sleep(0.033)  # ~30 Hz poll; dedupe on pccr_ts_ms


# ── Temporal sync ─────────────────────────────────────────────────────────────
def _sync_and_build_dataset(screen_w: int, screen_h: int) -> list[dict]:
    """
    Match target coords with eye vectors within SYNC_WINDOW_MS.
    Map target screen coords → scene camera coords via homography.
    Returns list of {"dx","dy","X","Y"} training samples.
    """
    with _target_lock:
        targets = list(_target_buf)
    with _eye_lock:
        eyes = list(_eye_buf)

    if not targets or not eyes:
        return []

    print(f"[sync] targets={len(targets)}  eye_samples={len(eyes)}")
    if targets and eyes:
        t_range = (targets[0]["ts"], targets[-1]["ts"])
        e_range = (eyes[0]["ts"],   eyes[-1]["ts"])
        print(f"[sync] target ts range: {t_range[0]:.0f}–{t_range[1]:.0f}")
        print(f"[sync] eye    ts range: {e_range[0]:.0f}–{e_range[1]:.0f}")
        overlap_ms = min(t_range[1], e_range[1]) - max(t_range[0], e_range[0])
        print(f"[sync] ts overlap: {overlap_ms:.0f} ms  (need > 0 for matches)")

    with _homography_lock:
        has_H = _homography is not None
    print(f"[sync] homography={'OK' if has_H else 'MISSING'}")

    samples = []
    no_eye_match = 0
    no_homography = 0
    ei = 0
    for t in targets:
        best = None
        best_dt = float("inf")
        while ei < len(eyes) and eyes[ei]["ts"] < t["ts"] - SYNC_WINDOW_MS:
            ei += 1
        j = ei
        while j < len(eyes) and eyes[j]["ts"] <= t["ts"] + SYNC_WINDOW_MS:
            dt = abs(eyes[j]["ts"] - t["ts"])
            if dt < best_dt:
                best_dt = dt
                best = eyes[j]
            j += 1
        if best is None:
            no_eye_match += 1
            continue

        scene_xy = _screen_to_scene(t["x"], t["y"])
        if scene_xy is None:
            no_homography += 1
            continue
        X, Y = scene_xy
        samples.append({"dx": best["dx"], "dy": best["dy"], "X": X, "Y": Y})

    print(f"[sync] matched={len(samples)}  no_eye_match={no_eye_match}  no_homography={no_homography}")
    with _debug_lock:
        _debug["last_sync_tried"]   = len(targets)
        _debug["last_sync_matched"] = len(samples)
    return samples


# ── WebSocket (minimal, no external deps) ─────────────────────────────────────
# RFC 6455 subset: text frames only, no fragmentation, no extensions

def _ws_handshake(rfile, wfile, headers: dict) -> bool:
    import base64, hashlib
    key = headers.get("sec-websocket-key", "").strip()
    if not key:
        return False
    magic = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
    accept = base64.b64encode(hashlib.sha1((key + magic).encode()).digest()).decode()
    resp = (
        "HTTP/1.1 101 Switching Protocols\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        f"Sec-WebSocket-Accept: {accept}\r\n\r\n"
    )
    wfile.write(resp.encode())
    wfile.flush()
    return True


def _ws_recv_frame(rfile) -> str | None:
    """Read one WebSocket text frame. Returns text or None on error/close."""
    try:
        b0 = rfile.read(1)
        if not b0:
            return None
        b1 = rfile.read(1)
        if not b1:
            return None
        opcode = b0[0] & 0x0F
        if opcode == 0x8:   # close
            return None
        masked = (b1[0] & 0x80) != 0
        length = b1[0] & 0x7F
        if length == 126:
            length = struct.unpack(">H", rfile.read(2))[0]
        elif length == 127:
            length = struct.unpack(">Q", rfile.read(8))[0]
        mask = rfile.read(4) if masked else b"\x00\x00\x00\x00"
        data = bytearray(rfile.read(length))
        if masked:
            for i in range(len(data)):
                data[i] ^= mask[i % 4]
        return data.decode("utf-8", errors="replace")
    except Exception:
        return None


def _ws_send_frame(wfile, text: str):
    """Send one WebSocket text frame."""
    try:
        payload = text.encode("utf-8")
        n = len(payload)
        header = bytearray()
        header.append(0x81)  # FIN + text opcode
        if n < 126:
            header.append(n)
        elif n < 65536:
            header.append(126)
            header += struct.pack(">H", n)
        else:
            header.append(127)
            header += struct.pack(">Q", n)
        wfile.write(bytes(header) + payload)
        wfile.flush()
    except Exception:
        pass


def _broadcast(msg: str):
    with _ws_lock:
        clients = set(_ws_clients)
    for wfile in clients:
        _ws_send_frame(wfile, msg)


def _handle_ws(rfile, wfile):
    global _calibrating, _calib_mode, _pending_eye, _pending_target
    with _ws_lock:
        _ws_clients.add(wfile)
    try:
        while True:
            text = _ws_recv_frame(rfile)
            if text is None:
                break
            try:
                msg = json.loads(text)
            except json.JSONDecodeError:
                continue

            mtype = msg.get("type")

            if mtype == "screen_size":
                _last_screen_size[0] = msg.get("w", 0)
                _last_screen_size[1] = msg.get("h", 0)

            elif mtype == "set_dict":
                name = msg.get("dict", "4x4")
                _switch_aruco_dict(name)
                _ws_send_frame(wfile, json.dumps({"type": "dict_changed", "dict": name}))

            elif mtype == "marker_positions":
                # Browser sends actual pixel centers of the 4 ArUco markers
                positions = msg.get("positions", {})  # {id: [x, y]}
                with _screen_markers_lock:
                    _screen_markers.clear()
                    for k, v in positions.items():
                        _screen_markers[int(k)] = v
                print(f"[ws] marker positions updated: { {k: [round(v[0]),round(v[1])] for k,v in _screen_markers.items()} }")
                _persist_marker_screen_positions()

            elif mtype == "record_toggle":
                sw, sh = _last_screen_size[0], _last_screen_size[1]
                mode   = msg.get("mode", _calib_mode)
                if not _recording:
                    rec_path = _rec_start(sw, sh, mode)
                    _ws_send_frame(wfile, json.dumps({"type": "record_ack",
                                                      "recording": True, "path": rec_path}))
                else:
                    rec_path = _rec_stop()
                    _ws_send_frame(wfile, json.dumps({"type": "record_ack",
                                                      "recording": False, "path": rec_path}))

            elif mtype == "start":
                _calibrating = True
                _calib_mode  = msg.get("mode", "sweep")
                with _target_lock:
                    _target_buf.clear()
                with _eye_lock:
                    _eye_buf.clear()
                with _saccade_lock:
                    _saccade_samples.clear()
                with _pending_lock:
                    _pending_eye.clear()
                    _pending_target = None
                _clear_calib_window()
                _calib_trace("WS start mode=%s screen=%s", _calib_mode, _last_screen_size)
                # Fresh session folder for each calibration run
                if _recording:
                    _rec_stop()
                sw, sh = _last_screen_size[0], _last_screen_size[1]
                if sw <= 0 or sh <= 0:
                    sw, sh = 1920, 1080
                rec_path = _rec_start(sw, sh, _calib_mode)
                t0 = time.time() * 1000.0
                start_row = {
                    "event": "calib_start", "mode": _calib_mode,
                    "ts_browser_ms": t0, "ts_source": "server_wall_ms",
                    "screen_w": sw, "screen_h": sh,
                }
                _append_screen_event(start_row)
                try:
                    if _rec_target_f:
                        _rec_target_f.write(json.dumps({
                            "event": "calib_start", "ts": t0, "mode": _calib_mode,
                        }) + "\n")
                        _rec_target_f.flush()
                except Exception:
                    pass
                _persist_marker_screen_positions()
                _ws_send_frame(wfile, json.dumps({
                    "type": "ack",
                    "msg": f"calibration started ({_calib_mode})",
                    "session_path": rec_path,
                    "recording": True,
                }))

            elif mtype == "target":
                if _calibrating:
                    entry = {"ts": msg["ts"], "x": msg["x"], "y": msg["y"]}
                    with _target_lock:
                        _target_buf.append(entry)
                    if _recording:
                        _append_screen_event({
                            "event": "target", "mode": _calib_mode,
                            "ts_browser_ms": float(msg["ts"]),
                            "x": float(msg["x"]), "y": float(msg["y"]),
                        })
                    if _recording and _rec_target_f:
                        try:
                            _rec_target_f.write(json.dumps(entry) + "\n")
                            _rec_target_f.flush()
                        except Exception:
                            pass

            elif mtype == "fixation":
                # Saccade mode: eye has settled — eye is now fixated for FIXATE_MS.
                # 1. Flush pending samples from the PREVIOUS target (now complete).
                # 2. Open a new capture window for this target on the receiver.
                # Receiver will push eye frames during [fixation_ts, fixation_ts+FIXATE_MS].
                if _calibrating:
                    ts = msg["ts"]   # ms, same clock as receiver's time.time()*1000
                    x, y = msg["x"], msg["y"]
                    # Atomically swap: drain previous target+eyes AND install the new
                    # target in a single locked critical section.  Earlier code released
                    # the lock between snapshot (which set _pending_target=None) and
                    # re-assignment, letting /push_eye observe a None target and drop
                    # every sample — see `dropped_no_pending_target` in calib_trace.
                    _trace_reset_push_counters()
                    with _pending_lock:
                        prev_eyes = _pending_eye
                        prev_target = _pending_target
                        _pending_eye = []
                        _pending_target = {"x": x, "y": y}
                    srv_now = time.time() * 1000
                    _calib_trace(
                        "WS fixation ts=%.1f server_now=%.1f delta_ms=%.1f xy=(%.1f,%.1f) "
                        "prev_eyes=%d prev_target=%s will_async_flush=%s capture=[%.1f,%.1f]",
                        ts, srv_now, ts - srv_now, x, y,
                        len(prev_eyes), "yes" if prev_target else "no",
                        str(bool(prev_eyes and prev_target)),
                        ts, ts + FIXATE_MS,
                    )
                    _set_calib_window(x, y, from_ms=ts, until_ms=ts + FIXATE_MS)
                    if prev_eyes and prev_target:
                        _start_async_flush_pending(prev_eyes, prev_target)
                    if _recording:
                        _append_screen_event({
                            "event": "fixation", "mode": _calib_mode,
                            "ts_browser_ms": float(ts),
                            "x": float(x), "y": float(y),
                            "capture_window_ms": [float(ts), float(ts + FIXATE_MS)],
                        })
                    if _recording and _rec_target_f:
                        try:
                            _rec_target_f.write(json.dumps({
                                "ts": ts, "x": x, "y": y, "event": "fixation"
                            }) + "\n")
                            _rec_target_f.flush()
                        except Exception:
                            pass

            elif mtype == "stop":
                _calibrating = False
                _calib_trace("WS stop: sync flush last pending + wait async flushes")
                # Flush the last pending target (fixation handler only flushes
                # the *previous* target when a new one arrives, so the final
                # point is never flushed otherwise).
                _flush_pending_target()
                # Wait for background flushes from earlier fixations (each can
                # take seconds); otherwise STOP reports far too few samples.
                _wait_async_flushes()
                _flush_calib_video_buf()
                with _saccade_lock:
                    samples = list(_saccade_samples)
                _calib_trace("WS stop: n_saccade_samples=%d will_train=%s",
                             len(samples), str(len(samples) >= 6))
                # Save samples to recording
                if _recording and _rec_dir:
                    try:
                        with open(_rec_dir / "samples.json", "w") as f:
                            json.dump(samples, f)
                    except Exception:
                        pass
                if _recording:
                    t1 = time.time() * 1000.0
                    _append_screen_event({
                        "event": "calib_stop", "mode": _calib_mode,
                        "ts_browser_ms": t1, "ts_source": "server_wall_ms",
                    })
                    try:
                        if _rec_target_f:
                            _rec_target_f.write(json.dumps({
                                "event": "calib_stop", "ts": t1, "mode": _calib_mode,
                            }) + "\n")
                            _rec_target_f.flush()
                    except Exception:
                        pass
                    _persist_marker_screen_positions()
                session_path = str(_rec_dir) if _rec_dir else ""
                was_recording = _recording
                result = {"type": "result", "n_samples": len(samples),
                          "recording": was_recording, "session_path": session_path}
                if len(samples) >= 6:
                    try:
                        diag = _model.fit(samples)
                        _model.save(MODEL_PATH)
                        with open(DATASET_PATH, "w") as f:
                            json.dump(samples, f)
                        result.update({"ok": True, **diag})
                    except Exception as e:
                        result.update({"ok": False, "error": str(e)})
                else:
                    result.update({"ok": False, "error": f"Only {len(samples)} synced samples (need 6+). Check eye pipeline is running and homography is valid."})
                _ws_send_frame(wfile, json.dumps(result))
                if _recording:
                    _rec_stop()

            elif mtype == "status":
                with _homography_lock:
                    has_H = _homography is not None
                _ws_send_frame(wfile, json.dumps({
                    "type": "status",
                    "homography": has_H,
                    "model_trained": _model.trained,
                    "calibrating": _calibrating,
                }))

    finally:
        with _ws_lock:
            _ws_clients.discard(wfile)


# ── HTTP handler ──────────────────────────────────────────────────────────────
_HTML = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Gaze Calibration</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { background: #000; overflow: hidden; font-family: monospace; }
canvas { display: block; }
#hud {
  position: fixed; top: 12px; left: 50%; transform: translateX(-50%);
  display: flex; gap: 12px; align-items: center; z-index: 10;
}
button {
  padding: 8px 20px; font-size: 14px; cursor: pointer;
  border: 1px solid #555; border-radius: 4px; background: #111; color: #eee;
}
button:hover { background: #222; }
button:disabled { opacity: 0.4; cursor: default; }
button.mode { background: #1a1a1a; }
button.mode.active { background: #333; color: #ffcc00; border-color: #ffcc00; }
button.live-on    { background: #1a3320; color: #44ff88; border-color: #44ff88; }
button.paused-on  { background: #2a1a0a; color: #ffaa44; border-color: #ffaa44; }
button.rec-on     { background: #2a0a0a; color: #ff4444; border-color: #ff4444; }
@keyframes recblink { 0%,100%{opacity:1} 50%{opacity:0.4} }
button.rec-on     { animation: recblink 1.2s infinite; }
button.trace-on   { background: #1a1a33; color: #88ccff; border-color: #6699cc; }
#status {
  color: #888; font-size: 13px; min-width: 320px; text-align: center;
}
#reopen-debug {
  position: fixed; bottom: 30vh; left: 12px; z-index: 25;
  display: none;
  padding: 4px 10px; font-size: 11px; background: rgba(0,0,0,0.7);
  border: 1px solid #555; border-radius: 4px; color: #aaa; cursor: pointer;
}
#debug-panel {
  position: fixed; bottom: 30vh; left: 12px; z-index: 20;
  background: rgba(0,0,0,0.82); border: 1px solid #333; border-radius: 6px;
  padding: 10px 14px; font-size: 12px; color: #ccc; min-width: 280px;
  max-width: 340px;
}
#debug-panel h4 {
  color: #888; margin-bottom: 6px; font-size: 11px; letter-spacing: 1px;
  display: flex; justify-content: space-between; align-items: center;
}
#debug-panel h4 button {
  font-size: 11px; padding: 0 6px; line-height: 16px; border-radius: 3px;
  border: 1px solid #555; background: #222; color: #aaa; cursor: pointer;
}
#debug-panel .row { display: flex; justify-content: space-between; margin: 2px 0; }
#debug-panel .ok  { color: #44ff88; }
#debug-panel .err { color: #ff5050; }
#debug-panel .warn{ color: #ffcc00; }
#debug-panel .slider-row { margin-top: 8px; border-top: 1px solid #333; padding-top: 6px; }
#debug-panel .slider-row label { color: #888; font-size: 11px; display: block; margin-bottom: 3px; }
#debug-panel input[type=range] { width: 100%; accent-color: #ffcc00; }
.dict-btn { padding: 3px 10px; font-size: 11px; background: #1a1a1a; color: #aaa;
  border: 1px solid #444; border-radius: 3px; cursor: pointer; }
.dict-btn.active { background: #333; color: #ffcc00; border-color: #ffcc00; }
#cam-container {
  position: fixed; bottom: 40vh; right: 12px; z-index: 20;
  display: none; flex-direction: column; gap: 4px;
}
#scene-view {
  border: 1px solid #444; border-radius: 4px;
  width: 320px; display: block;
}
#cam-switch {
  display: flex; gap: 0;
}
#cam-switch button {
  flex: 1; padding: 4px 0; font-size: 11px; border-radius: 0;
  border: 1px solid #444; background: #1a1a1a; color: #888; cursor: pointer;
}
#cam-switch button.active {
  background: #333; color: #ffcc00; border-color: #ffcc00;
}
#cam-switch button:first-child { border-radius: 4px 0 0 4px; }
#cam-switch button:last-child  { border-radius: 0 4px 4px 0; }
.aruco-marker {
  position: fixed; z-index: 5;
  background: white;
  image-rendering: pixelated;
}
</style>
</head>
<body>
<div id="hud">
  <button id="btnSweep" class="mode active">SWEEP</button>
  <button id="btnSaccade" class="mode">SACCADE</button>
  <button id="btnStart">START</button>
  <button id="btnStop" disabled>STOP</button>
  <button id="btnLive" disabled>LIVE OFF</button>
  <button onclick="window.open('/viz','_blank')">📊 Viz</button>
  <button id="btnRecord" style="display:none" title="Manual session (optional)">⏺ Record</button>
  <button id="btnPause">⏸ Pause Streams</button>
  <button id="btnTrace" type="button">Trace OFF</button>
  <span id="status">Connecting…</span>
</div>
<div id="debug-panel">
  <h4>DEBUG <button id="close-debug">✕</button></h4>
  <div class="row"><span>Scene cam</span><span id="d-scene">…</span></div>
  <div class="row"><span>ArUco</span><span id="d-aruco">…</span></div>
  <div class="row"><span>Homography</span><span id="d-homo">…</span></div>
  <div class="row"><span>PCCR vector</span><span id="d-pccr">…</span></div>
  <div class="row"><span>Eye poll</span><span id="d-eye">…</span></div>
  <div class="row"><span>Target buf</span><span id="d-tbuf">…</span></div>
  <div class="row"><span>Eye buf</span><span id="d-ebuf">…</span></div>
  <div class="row"><span>Frame size</span><span id="d-frame">…</span></div>
  <div class="row"><span>Brightness</span><span id="d-bright">…</span></div>
  <div class="row"><span>Screen size</span><span id="d-screen">…</span></div>
  <div class="row"><span>Last sync</span><span id="d-sync">…</span></div>
  <div class="row"><span>Saccade pts</span><span id="d-sacc">…</span></div>
  <div class="row"><span>Pipeline trace</span><span id="d-trace">…</span></div>
  <div class="slider-row">
    <label>ArUco marker size: <span id="marker-size-val">20</span>% of screen</label>
    <input type="range" id="marker-size" min="8" max="35" value="20" step="1">
  </div>
  <div class="slider-row">
    <label>ArUco dictionary: <span id="d-dict">4x4</span></label>
    <div style="display:flex;gap:6px;flex-wrap:wrap;margin-top:4px">
      <button class="dict-btn active" data-dict="4x4">4×4</button>
      <button class="dict-btn" data-dict="5x5">5×5</button>
      <button class="dict-btn" data-dict="6x6">6×6</button>
      <button class="dict-btn" data-dict="7x7">7×7</button>
    </div>
  </div>
</div>
<button id="reopen-debug">⬛ DEBUG</button>
<div id="cam-container">
  <div id="cam-switch">
    <button id="btnWorld" class="active">🌍 World (ArUco)</button>
    <button id="btnEye">👁 Eye (PCCR)</button>
  </div>
  <img id="scene-view" alt="cam feed">
</div>
<canvas id="c"></canvas>
<img id="m0" class="aruco-marker" src="/marker/0" alt="">
<img id="m1" class="aruco-marker" src="/marker/1" alt="">
<img id="m2" class="aruco-marker" src="/marker/2" alt="">
<img id="m3" class="aruco-marker" src="/marker/3" alt="">
<script>
const canvas = document.getElementById('c');
const ctx = canvas.getContext('2d');
const statusEl  = document.getElementById('status');
const btnStart  = document.getElementById('btnStart');
const btnStop   = document.getElementById('btnStop');
const btnLive   = document.getElementById('btnLive');
const btnSweep  = document.getElementById('btnSweep');
const btnSaccade= document.getElementById('btnSaccade');
const btnPause  = document.getElementById('btnPause');
const btnRecord = document.getElementById('btnRecord');
const btnTrace  = document.getElementById('btnTrace');

let traceOn = false;
function syncTraceUi() {
  fetch('/set_calib_trace').then(r => r.json()).then(d => {
    traceOn = !!d.on;
    btnTrace.textContent = traceOn ? 'Trace ON' : 'Trace OFF';
    btnTrace.classList.toggle('trace-on', traceOn);
  }).catch(() => {});
}
btnTrace.onclick = () => {
  const v = traceOn ? 0 : 1;
  fetch(`/set_calib_trace?v=${v}`).then(r => r.json()).then(d => {
    if (d.ok) {
      traceOn = !!d.on;
      btnTrace.textContent = traceOn ? 'Trace ON' : 'Trace OFF';
      btnTrace.classList.toggle('trace-on', traceOn);
    }
  }).catch(() => {});
};

let recOn = false;
btnRecord.onclick = () => {
  ws.send(JSON.stringify({ type: 'record_toggle', mode }));
};
// Handle server response
function handleRecordAck(m) {
  recOn = m.recording;
  if (recOn) {
    btnRecord.textContent = '⏹ Stop Rec';
    btnRecord.classList.add('rec-on');
    statusEl.textContent = `Recording → ${m.path.split('/').pop()}`;
  } else {
    btnRecord.textContent = '⏺ Record';
    btnRecord.classList.remove('rec-on');
    statusEl.textContent = `Saved → ${m.path.split('/').pop()}`;
  }
}

let streamsPaused = false;
function setStreamsPaused(val) {
  streamsPaused = val;
  // Proxy through calibration server to avoid CORS (8090→8080)
  fetch(`/pause_receiver?v=${val ? 1 : 0}`).catch(() => {});
  if (val) {
    btnPause.textContent = '▶ Resume Streams';
    btnPause.classList.add('paused-on');
  } else {
    btnPause.textContent = '⏸ Pause Streams';
    btnPause.classList.remove('paused-on');
  }
}
// Auto-pause receiver MJPEG when calibration opens; resume on close
setStreamsPaused(true);
window.addEventListener('beforeunload', () => setStreamsPaused(false));
btnPause.onclick = () => setStreamsPaused(!streamsPaused);

let W, H;
function resize() {
  W = canvas.width  = window.innerWidth;
  H = canvas.height = window.innerHeight;
}
resize();
window.addEventListener('resize', resize);

let mode = 'sweep';  // 'sweep' | 'saccade'
btnSweep.onclick = () => {
  if (running) return;
  mode = 'sweep';
  btnSweep.classList.add('active');
  btnSaccade.classList.remove('active');
};
btnSaccade.onclick = () => {
  if (running) return;
  mode = 'saccade';
  btnSaccade.classList.add('active');
  btnSweep.classList.remove('active');
};

// ── WebSocket ────────────────────────────────────────────────────────────────
const ws = new WebSocket(`ws://${location.host}/ws`);
ws.onopen = () => {
  statusEl.textContent = 'Connected';
  ws.send(JSON.stringify({type:'screen_size', w:W, h:H}));
  ws.send(JSON.stringify({type:'status'}));
  syncTraceUi();
  // Send marker positions after WS is open
  setTimeout(updateMarkers, 100);
};
ws.onclose = () => statusEl.textContent = 'Disconnected';
ws.onmessage = e => {
  const m = JSON.parse(e.data);
  if (m.type === 'record_ack') { handleRecordAck(m); return; }
  if (m.type === 'ack') {
    statusEl.textContent = m.session_path
      ? `Calibrating…  session: ${m.session_path.split('/').pop()}`
      : 'Calibrating…';
    return;
  }
  if (m.type === 'status') {
    statusEl.textContent = `Homography: ${m.homography?'✓':'✗'}  Model: ${m.model_trained?'✓':'✗'}`;
    if (m.model_trained) btnLive.disabled = false;
  }
  if (m.type === 'ready') {
    btnLive.disabled = false;
    statusEl.textContent = `${m.n} pts — R²x=${m.r2_x}  R²y=${m.r2_y}`;
  }
  if (m.type === 'result') {
    btnStop.disabled  = false;
    btnStart.disabled = false;
    const folder = m.session_path ? m.session_path.split('/').pop() : '';
    const rec = m.recording && folder ? `  💾 ${folder}` : '';
    if (m.ok) {
      statusEl.textContent = `Done! n=${m.n_samples}  R²x=${m.r2_x.toFixed(3)}  R²y=${m.r2_y.toFixed(3)}${rec}`;
    } else {
      statusEl.textContent = `Failed: ${m.error}${rec}`;
    }
  }
};

// ── ArUco markers — server-generated PNGs placed as DOM elements ─────────────
// Markers are generated with cv2.aruco.generateImageMarker() server-side,
// guaranteed to match what OpenCV detects. Each img has white CSS padding
// to provide the required quiet zone.
let bgCanvas = null;
let markerPct = 20;  // % of min(W,H) — controlled by slider
const QUIET_PX = 16;  // white quiet-zone padding around each marker (px)
const EDGE_PX  = 4;   // gap from screen edge

function buildBackground() {
  bgCanvas = document.createElement('canvas');
  bgCanvas.width  = W;
  bgCanvas.height = H;
  const bctx = bgCanvas.getContext('2d');
  bctx.fillStyle = '#111';
  bctx.fillRect(0, 0, W, H);
}

function updateMarkers() {
  const S = Math.min(W, H) * markerPct / 100;  // marker display size in px
  // Each img element: size=S, padding=QUIET_PX on all sides (white bg)
  // Total element size = S + 2*QUIET_PX
  // Element placed EDGE_PX from screen edge
  // Marker center (for homography) = EDGE_PX + QUIET_PX + S/2 from that edge
  const M = EDGE_PX + QUIET_PX + S / 2;

  const corners = [
    { id: 0, left: EDGE_PX,       top:    EDGE_PX },       // TL
    { id: 1, right: EDGE_PX,      top:    EDGE_PX },       // TR
    { id: 2, left: EDGE_PX,       bottom: EDGE_PX },       // BL
    { id: 3, right: EDGE_PX,      bottom: EDGE_PX },       // BR
  ];
  const positions = {};
  for (const c of corners) {
    const el = document.getElementById(`m${c.id}`);
    el.style.width   = S + 'px';
    el.style.height  = S + 'px';
    el.style.padding = QUIET_PX + 'px';
    el.style.left = el.style.right = el.style.top = el.style.bottom = '';
    if (c.left   !== undefined) el.style.left   = c.left   + 'px';
    if (c.right  !== undefined) el.style.right  = c.right  + 'px';
    if (c.top    !== undefined) el.style.top    = c.top    + 'px';
    if (c.bottom !== undefined) el.style.bottom = c.bottom + 'px';
    // Compute center position in screen coords
    const cx = c.left  !== undefined ? M : W - M;
    const cy = c.top   !== undefined ? M : H - M;
    positions[c.id] = [cx, cy];
  }

  // Send actual marker centers to server for homography
  if (ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: 'marker_positions', positions }));
  }
}

buildBackground();
updateMarkers();
window.addEventListener('resize', () => { resize(); buildBackground(); updateMarkers(); });

// Marker size slider
document.getElementById('marker-size').addEventListener('input', function() {
  markerPct = Number(this.value);
  document.getElementById('marker-size-val').textContent = markerPct;
  updateMarkers();
});

// ── Collision detection — list of rects to avoid ─────────────────────────────
function getBlockedRects() {
  const S    = Math.min(W, H) * markerPct / 100;
  const mEnd = EDGE_PX + 2 * QUIET_PX + S;  // outer edge of marker+quietzone
  const R    = TARGET_R + 6;                 // dot radius + margin
  const rects = [
    // HUD bar at top
    { x: 0, y: 0, w: W, h: 70 },
    // ArUco markers (TL, TR, BL, BR)
    { x: 0,       y: 0,       w: mEnd, h: mEnd },
    { x: W - mEnd,y: 0,       w: mEnd, h: mEnd },
    { x: 0,       y: H - mEnd,w: mEnd, h: mEnd },
    { x: W - mEnd,y: H - mEnd,w: mEnd, h: mEnd },
    // Debug panel — bottom:30vh, left:12px, ~340px wide, ~300px tall
    { x: 0, y: H - H*0.30 - 300, w: 360, h: 300 + H*0.30 },
    // Scene view  — bottom:40vh, right:12px, 320px wide, ~240px tall
    { x: W - 336, y: H - H*0.40 - 240, w: 336, h: 240 + H*0.40 },
  ];
  return rects;
}

function isBlocked(x, y) {
  const R = TARGET_R + 6;
  for (const r of getBlockedRects()) {
    if (x + R > r.x && x - R < r.x + r.w &&
        y + R > r.y && y - R < r.y + r.h) return true;
  }
  return false;
}

function safeRandom(genFn, maxTries = 30) {
  for (let i = 0; i < maxTries; i++) {
    const p = genFn();
    if (!isBlocked(p.x, p.y)) return p;
  }
  return genFn(); // fallback: return last attempt unchecked
}

// ── Sweep mode (reading lines) ────────────────────────────────────────────────
const TARGET_R = 12;
const LINES    = 7;
const LINE_MS  = 5000;
const TOTAL_MS = LINES * LINE_MS;

function readingPos(t) {
  const pad = 40;
  const cycle   = t % TOTAL_MS;
  const lineIdx = Math.floor(cycle / LINE_MS);
  const lineT   = (cycle % LINE_MS) / LINE_MS;
  return {
    x: pad + lineT * (W - 2*pad),
    y: pad + (lineIdx / (LINES - 1)) * (H - 2*pad),
  };
}

// ── Saccade mode — stratified 9-zone grid, max-distance sequencing ───────────
const SETTLE_MS = 700;   // eye settling time after jump (ms)
const FIXATE_MS = 300;   // sample window after gated fixation (ms)
const ARUCO_POLL_MS = 80;  // /debug poll rate during saccade (homography + IDs)

// 3×3 zone grid — (row, col) for distance math
const ZONE_POS = [
  [0,0],[0,1],[0,2],
  [1,0],[1,1],[1,2],
  [2,0],[2,1],[2,2],
];
const CENTER_ZONE = 4;

function zoneDist(a, b) {
  const dr = ZONE_POS[a][0] - ZONE_POS[b][0];
  const dc = ZONE_POS[a][1] - ZONE_POS[b][1];
  return Math.sqrt(dr*dr + dc*dc);
}

// Build sequence: center first, then greedy max-distance through remaining 8 zones.
// After all 9 visited, regenerate (keeps cross-screen jumps across repeats).
function buildZoneSequence() {
  const seq = [CENTER_ZONE];
  const rem = [0,1,2,3,5,6,7,8];
  while (rem.length) {
    const last = seq[seq.length - 1];
    const maxD = Math.max(...rem.map(z => zoneDist(last, z)));
    const candidates = rem.filter(z => Math.abs(zoneDist(last, z) - maxD) < 0.01);
    const chosen = candidates[Math.floor(Math.random() * candidates.length)];
    seq.push(chosen);
    rem.splice(rem.indexOf(chosen), 1);
  }
  return seq;
}

function randomInZone(zoneIdx) {
  const pad   = 40;
  const cellW = (W - 2*pad) / 3;
  const cellH = (H - 2*pad) / 3;
  const inset = Math.min(cellW, cellH) * 0.12;
  const col   = ZONE_POS[zoneIdx][1];
  const row   = ZONE_POS[zoneIdx][0];
  if (zoneIdx === CENTER_ZONE) {
    const p = { x: W/2, y: H/2 };
    return isBlocked(p.x, p.y) ? safeRandom(() => ({
      x: pad + col*cellW + inset + Math.random()*(cellW - 2*inset),
      y: pad + row*cellH + inset + Math.random()*(cellH - 2*inset),
    })) : p;
  }
  return safeRandom(() => ({
    x: pad + col*cellW + inset + Math.random()*(cellW - 2*inset),
    y: pad + row*cellH + inset + Math.random()*(cellH - 2*inset),
  }));
}

let zoneSeq      = [];
let zoneSeqIdx   = 0;
let saccadePos   = { x: 0, y: 0 };
let saccadeStart = 0;
let saccadeSampled = false;
let saccadeCount = 0;
let fixationSentAt = null;   // performance.now() when fixation WS was sent
let lastArucoPoll  = 0;
let arucoGateOk    = false;

function pollArucoGate(now) {
  if (now - lastArucoPoll < ARUCO_POLL_MS) return;
  lastArucoPoll = now;
  fetch('/debug').then(r => r.json()).then(d => {
    const exp = d.expected_aruco_ids || [];
    const ids = d.aruco_ids || [];
    arucoGateOk = !!(d.homography_ok && d.aruco_expected_met &&
      exp.length > 0 && exp.every(id => ids.includes(id)));
  }).catch(() => { arucoGateOk = false; });
}

function nextSaccadePoint() {
  if (zoneSeqIdx >= zoneSeq.length) {
    // All 9 zones visited — generate a new sequence for the next round
    // (but skip re-centering: start from farthest zone from current position)
    zoneSeq    = buildZoneSequence().slice(1); // skip center on repeats
    zoneSeqIdx = 0;
  }
  const zone   = zoneSeq[zoneSeqIdx++];
  saccadePos   = randomInZone(zone);
  saccadeStart = performance.now();
  saccadeSampled = false;
  fixationSentAt = null;
}

function initSaccade() {
  zoneSeq    = buildZoneSequence();
  zoneSeqIdx = 0;
  saccadeCount = 0;
  arucoGateOk   = false;
  lastArucoPoll = 0;
  nextSaccadePoint();  // first point = center zone
}

function tickSaccade(now) {
  pollArucoGate(now);
  const dwell   = now - saccadeStart;
  const settled = dwell >= SETTLE_MS;

  if (settled && !saccadeSampled && arucoGateOk) {
    if (ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({
        type: 'fixation',
        ts:   performance.timeOrigin + now,
        x:    saccadePos.x,
        y:    saccadePos.y,
      }));
    }
    fixationSentAt = now;
    saccadeSampled = true;
    saccadeCount++;
    const round = Math.ceil(saccadeCount / 9);
    const inRound = ((saccadeCount - 1) % 9) + 1;
    statusEl.textContent = `Saccade: ${saccadeCount} pts  (round ${round}, point ${inRound}/9)`;
  } else if (settled && !saccadeSampled) {
    const nextIdx = saccadeCount + 1;
    const round = Math.ceil(nextIdx / 9);
    const inRound = ((nextIdx - 1) % 9) + 1;
    statusEl.textContent =
      `Saccade: ${saccadeCount} pts  (round ${round}, point ${inRound}/9) — waiting for 4 ArUco markers`;
  }

  if (saccadeSampled && fixationSentAt !== null && (now - fixationSentAt) >= FIXATE_MS) {
    nextSaccadePoint();
  }

  // Shrinking ring: large yellow → small green as eye should settle
  const progress = Math.min(dwell / SETTLE_MS, 1);
  const ringR    = TARGET_R + 18 * (1 - progress);
  const color    = settled ? '#44ff88' : '#ffcc00';

  ctx.beginPath();
  ctx.arc(saccadePos.x, saccadePos.y, ringR, 0, Math.PI * 2);
  ctx.strokeStyle = color;
  ctx.lineWidth   = 2;
  ctx.stroke();

  ctx.beginPath();
  ctx.arc(saccadePos.x, saccadePos.y, TARGET_R, 0, Math.PI * 2);
  ctx.fillStyle = color;
  ctx.fill();
}

// ── Live gaze cursor ──────────────────────────────────────────────────────────
let liveOn      = false;
let gazeCursor  = null;   // {x, y} in screen pixels, or null
let liveTimer   = null;

function startLive() {
  liveTimer = setInterval(() => {
    fetch('/live').then(r => r.json()).then(d => {
      gazeCursor = d.ok ? {x: d.x, y: d.y} : null;
    }).catch(() => { gazeCursor = null; });
  }, 80);  // ~12 Hz — fast enough to feel live, not spammy
}

function stopLive() {
  clearInterval(liveTimer);
  liveTimer  = null;
  gazeCursor = null;
}

btnLive.onclick = () => {
  liveOn = !liveOn;
  if (liveOn) {
    btnLive.textContent = 'LIVE ON';
    btnLive.classList.add('live-on');
    startLive();
  } else {
    btnLive.textContent = 'LIVE OFF';
    btnLive.classList.remove('live-on');
    stopLive();
  }
};

function drawGazeCursor(pos) {
  const R = 18;
  ctx.save();
  ctx.strokeStyle = 'rgba(255, 80, 80, 0.9)';
  ctx.lineWidth   = 2;
  // Circle
  ctx.beginPath();
  ctx.arc(pos.x, pos.y, R, 0, Math.PI * 2);
  ctx.stroke();
  // Crosshair lines
  ctx.beginPath();
  ctx.moveTo(pos.x - R - 6, pos.y); ctx.lineTo(pos.x + R + 6, pos.y);
  ctx.moveTo(pos.x, pos.y - R - 6); ctx.lineTo(pos.x, pos.y + R + 6);
  ctx.stroke();
  ctx.restore();
}

// ── Render loop ───────────────────────────────────────────────────────────────
let running = false;
let startTime = 0;

function render(now) {
  requestAnimationFrame(render);
  ctx.drawImage(bgCanvas, 0, 0);

  // Live gaze cursor — drawn regardless of running state
  if (liveOn && gazeCursor) drawGazeCursor(gazeCursor);

  if (!running) return;

  if (mode === 'saccade') {
    tickSaccade(now);
  } else {
    const elapsed = now - startTime;
    const pos = readingPos(elapsed);
    ctx.beginPath();
    ctx.arc(pos.x, pos.y, TARGET_R, 0, Math.PI * 2);
    ctx.fillStyle = '#ffcc00';
    ctx.fill();
    if (ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({type:'target', ts: performance.timeOrigin + now, x: pos.x, y: pos.y}));
    }
  }
}
requestAnimationFrame(render);

// ── Buttons ───────────────────────────────────────────────────────────────────
btnStart.onclick = () => {
  if (ws.readyState !== WebSocket.OPEN) { statusEl.textContent='Not connected'; return; }
  ws.send(JSON.stringify({type:'start', mode}));
  ws.send(JSON.stringify({type:'screen_size', w:W, h:H}));
  startTime    = performance.now();
  saccadeCount = 0;
  if (mode === 'saccade') initSaccade();
  running = true;
  btnStart.disabled  = true;
  btnStop.disabled   = false;
  btnSweep.disabled  = true;
  btnSaccade.disabled= true;
};
btnStop.onclick = () => {
  running = false;
  btnStop.disabled    = true;
  btnStart.disabled   = false;
  btnSweep.disabled   = false;
  btnSaccade.disabled = false;
  statusEl.textContent = 'Processing…';
  ws.send(JSON.stringify({type:'stop'}));
};

// ── Debug panel ───────────────────────────────────────────────────────────────
function cls(id, c) {
  const el = document.getElementById(id);
  el.className = c;
  return el;
}
function dbg(id, text, ok) {
  const el = cls(id, ok === true ? 'ok' : ok === false ? 'err' : 'warn');
  el.textContent = text;
}

function pollDebug() {
  fetch('/debug').then(r => r.json()).then(d => {
    dbg('d-scene', d.scene_cam_ok ? 'OK' : ('ERR: ' + d.scene_cam_error), d.scene_cam_ok);
    const exp = d.expected_aruco_ids || [];
    const ids = d.aruco_ids || [];
    const arucoOk = !!(d.aruco_expected_met && exp.length > 0 && exp.every(id => ids.includes(id)));
    dbg('d-aruco', `${d.aruco_detected} found  IDs=[${d.aruco_ids}]  need=[${exp}]`, arucoOk ? true : (d.aruco_detected > 0 ? null : false));
    dbg('d-homo', d.homography_ok ? 'OK' : 'MISSING', d.homography_ok);
    if (d.pccr_vector) {
      dbg('d-pccr', `[${d.pccr_vector[0]}, ${d.pccr_vector[1]}]`, true);
    } else {
      dbg('d-pccr', 'null — analysis on?', false);
    }
    dbg('d-eye', d.eye_poll_ok ? 'OK' : ('ERR: ' + d.eye_poll_error), d.eye_poll_ok);
    dbg('d-tbuf', d.target_buf_size, d.target_buf_size > 0 ? true : null);
    dbg('d-ebuf', d.eye_buf_size, d.eye_buf_size > 0 ? true : null);
    if (d.frame_size) {
      dbg('d-frame', `${d.frame_size[0]}×${d.frame_size[1]}`, true);
    } else {
      dbg('d-frame', 'no frame', false);
    }
    if (d.brightness !== null && d.brightness !== undefined) {
      const bOk = d.brightness > 30 && d.brightness < 220;
      dbg('d-bright', `${d.brightness}/255 ${bOk ? '' : '⚠ check lighting'}`, bOk ? true : null);
    } else {
      dbg('d-bright', '—', null);
    }
    const ss = d.screen_size;
    dbg('d-screen', ss && ss[0] ? `${ss[0]}×${ss[1]}` : 'not sent', ss && ss[0]);
    dbg('d-sync', `tried=${d.last_sync_tried} matched=${d.last_sync_matched}`,
        d.last_sync_matched > 0 ? true : (d.last_sync_tried > 0 ? false : null));
    dbg('d-sacc', d.saccade_samples, d.saccade_samples >= 6 ? true : (d.saccade_samples > 0 ? null : false));
    dbg('d-trace', d.calib_trace ? 'ON (terminal)' : 'OFF', d.calib_trace ? true : null);
    if (d.aruco_dict) document.getElementById('d-dict').textContent = d.aruco_dict;
  }).catch(() => {});
}
pollDebug();
setInterval(pollDebug, 800);

// ── Debug panel open/close ────────────────────────────────────────────────────
const debugPanel  = document.getElementById('debug-panel');
const reopenBtn   = document.getElementById('reopen-debug');
document.getElementById('close-debug').onclick = () => {
  debugPanel.style.display = 'none';
  reopenBtn.style.display  = 'block';
};
reopenBtn.onclick = () => {
  debugPanel.style.display = 'block';
  reopenBtn.style.display  = 'none';
};

// ── ArUco dictionary switcher ─────────────────────────────────────────────────
function switchDict(name) {
  fetch(`/set_dict?d=${name}`).then(r => r.json()).then(d => {
    if (!d.ok) return;
    document.getElementById('d-dict').textContent = name;
    document.querySelectorAll('.dict-btn').forEach(b => {
      b.classList.toggle('active', b.dataset.dict === name);
    });
    // Reload marker images with cache-busting so browser fetches regenerated PNGs
    const t = Date.now();
    [0,1,2,3].forEach(i => {
      document.getElementById(`m${i}`).src = `/marker/${i}?t=${t}`;
    });
    // Re-send marker positions after images reload (size unchanged)
    setTimeout(updateMarkers, 300);
  }).catch(() => {});
}
document.querySelectorAll('.dict-btn').forEach(btn => {
  btn.onclick = () => switchDict(btn.dataset.dict);
});

// Handle server-initiated dict change (via WS)
// (already handled inline in ws.onmessage below)

// ── Scene cam refresh ─────────────────────────────────────────────────────────
// ── Cam view toggle (world ↔ eye) ────────────────────────────────────────────
const sceneImg    = document.getElementById('scene-view');
const camContainer= document.getElementById('cam-container');
const btnWorld    = document.getElementById('btnWorld');
const btnEye      = document.getElementById('btnEye');

let camView = 'world';  // 'world' | 'eye'

function startCamStream(view) {
  camView = view;
  const url = view === 'eye' ? '/eye_stream' : '/scene_stream';
  sceneImg.src = url;
  camContainer.style.display = 'flex';
  btnWorld.classList.toggle('active', view === 'world');
  btnEye.classList.toggle('active', view === 'eye');
}

sceneImg.onerror = () => {
  setTimeout(() => startCamStream(camView), 1000);
};

startCamStream('world');

btnWorld.onclick = () => startCamStream('world');
btnEye.onclick   = () => startCamStream('eye');
</script>
</body>
</html>"""


_VIZ_HTML = """<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>Calibration Data Viz</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: #111; color: #ccc; font-family: monospace; padding: 16px; }
h2 { color: #fa0; margin-bottom: 12px; font-size: 15px; }
.grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
canvas { background: #1a1a1a; border: 1px solid #333; border-radius: 4px; display: block; width: 100%; }
.panel { }
.label { font-size: 12px; color: #888; margin-bottom: 6px; }
.stats { font-size: 12px; color: #aaa; margin-top: 8px; line-height: 1.7; }
.stats span { color: #ffcc00; }
.warn { color: #ff5050; }
</style></head>
<body>
<h2>Calibration Data Visualisation &nbsp;<a href="/viz" style="font-size:11px;color:#666" onclick="location.reload();return false">↻ refresh</a>
  &nbsp;<a href="/" style="font-size:11px;color:#666">← back</a></h2>
<div class="grid">
  <div class="panel">
    <div class="label">PCCR space (dx, dy) — colour = target X position</div>
    <canvas id="cPccr" height="400"></canvas>
    <div class="stats" id="statsPccr"></div>
  </div>
  <div class="panel">
    <div class="label">Scene space (X, Y) from homography — colour = target X</div>
    <canvas id="cScene" height="400"></canvas>
    <div class="stats" id="statsScene"></div>
  </div>
  <div class="panel">
    <div class="label">PCCR dx distribution per screen column (should spread left→right)</div>
    <canvas id="cDxX" height="300"></canvas>
  </div>
  <div class="panel">
    <div class="label">PCCR dy distribution per screen row (should spread top→bottom)</div>
    <canvas id="cDyY" height="300"></canvas>
  </div>
</div>
<div class="stats" id="statsGlobal" style="margin-top:14px"></div>
<script>
fetch('/viz/data').then(r=>r.json()).then(({samples, n, model_trained}) => {
  if (!n) { document.getElementById('statsGlobal').textContent = 'No samples found.'; return; }

  const dx = samples.map(s=>s.dx), dy = samples.map(s=>s.dy);
  const sx = samples.map(s=>s.sx ?? s.X), sy = samples.map(s=>s.sy ?? s.Y);
  const X  = samples.map(s=>s.X),  Y  = samples.map(s=>s.Y);

  function stats(arr, name) {
    const mn = Math.min(...arr), mx = Math.max(...arr);
    const mean = arr.reduce((a,b)=>a+b,0)/arr.length;
    const std  = Math.sqrt(arr.map(v=>(v-mean)**2).reduce((a,b)=>a+b,0)/arr.length);
    return {mn, mx, mean, std, range: mx-mn,
      html: `${name}: min=<span>${mn.toFixed(2)}</span> max=<span>${mx.toFixed(2)}</span> range=<span>${(mx-mn).toFixed(2)}</span> std=<span>${std.toFixed(2)}</span>`};
  }

  const sDx = stats(dx,'dx'), sDy = stats(dy,'dy');
  const sX  = stats(X,'X'),   sY  = stats(Y,'Y');

  const warn = (v, th, msg) => v < th ? `<span class="warn"> ⚠ ${msg}</span>` : '';

  document.getElementById('statsPccr').innerHTML =
    sDx.html + warn(sDx.range, 5, 'dx barely moves — pupil/glint not tracking?') + '<br>' +
    sDy.html + warn(sDy.range, 5, 'dy barely moves — pupil/glint not tracking?');
  document.getElementById('statsScene').innerHTML =
    sX.html  + warn(sX.range, 50, 'scene X barely varies — homography wrong?') + '<br>' +
    sY.html  + warn(sY.range, 50, 'scene Y barely varies — homography wrong?');
  document.getElementById('statsGlobal').innerHTML =
    `n=<span>${n}</span>  model_trained=<span>${model_trained}</span>` +
    warn(sDx.range < 5 || sDy.range < 5, 1, 'PCCR not varying — eye pipeline likely not tracking') +
    warn(sX.range < 50 || sY.range < 50, 1, 'Scene coords not varying — check homography');

  // colour by screen X position
  function hue(v, mn, mx) {
    const t = (v-mn)/(mx-mn||1);
    return `hsl(${Math.round(t*260)},90%,55%)`;
  }

  function scatter(id, xs, ys, colours, xLabel='', yLabel='') {
    const c = document.getElementById(id);
    const W = c.offsetWidth || 480, H = parseInt(c.getAttribute('height'));
    c.width = W; c.height = H;
    const ctx = c.getContext('2d');
    const pad = 36;
    const mnX=Math.min(...xs), mxX=Math.max(...xs);
    const mnY=Math.min(...ys), mxY=Math.max(...ys);
    const tx = v => pad + (v-mnX)/(mxX-mnX||1)*(W-2*pad);
    const ty = v => H-pad - (v-mnY)/(mxY-mnY||1)*(H-2*pad);
    // axes
    ctx.strokeStyle='#444'; ctx.lineWidth=1;
    ctx.strokeRect(pad, pad, W-2*pad, H-2*pad);
    // grid
    ctx.strokeStyle='#222'; ctx.setLineDash([3,3]);
    for(let i=1;i<4;i++){
      const gx=pad+(W-2*pad)*i/4, gy=pad+(H-2*pad)*i/4;
      ctx.beginPath(); ctx.moveTo(gx,pad); ctx.lineTo(gx,H-pad); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(pad,gy); ctx.lineTo(W-pad,gy); ctx.stroke();
    }
    ctx.setLineDash([]);
    // points
    xs.forEach((x,i) => {
      ctx.fillStyle = colours[i];
      ctx.beginPath(); ctx.arc(tx(x), ty(ys[i]), 4, 0, Math.PI*2); ctx.fill();
    });
    // axis labels
    ctx.fillStyle='#666'; ctx.font='10px monospace'; ctx.textAlign='center';
    ctx.fillText(mnX.toFixed(1), pad, H-4);
    ctx.fillText(mxX.toFixed(1), W-pad, H-4);
    ctx.fillText(xLabel, W/2, H-4);
    ctx.save(); ctx.translate(10, H/2); ctx.rotate(-Math.PI/2);
    ctx.fillText(yLabel, 0, 0); ctx.restore();
  }

  const colours = sx.map((v,i) => hue(v, Math.min(...sx), Math.max(...sx)));
  scatter('cPccr',  dx, dy, colours, 'dx', 'dy');
  scatter('cScene', X,  Y,  colours, 'X(scene)', 'Y(scene)');
  scatter('cDxX',   sx, dx, colours, 'screen X', 'dx');
  scatter('cDyY',   sy, dy, colours, 'screen Y', 'dy');

  // Correlation table
  if (data.corr && Object.keys(data.corr).length) {
    const c = data.corr;
    const bar = v => {
      const pct = Math.round(Math.abs(v)*100);
      const col = Math.abs(v) > 0.5 ? '#4c4' : Math.abs(v) > 0.25 ? '#aa4' : '#a44';
      return `<span style="display:inline-block;width:${pct}px;height:8px;background:${col};vertical-align:middle"></span>`;
    };
    document.getElementById('statsGlobal').innerHTML += `
      <table style="margin-top:10px;font-family:monospace;font-size:11px;color:#aaa;border-collapse:collapse">
        <tr><th style="padding:2px 8px;color:#888">Correlation</th><th>screen X</th><th>screen Y</th></tr>
        <tr><td style="padding:2px 8px">PCCR dx</td>
            <td>${c.dx_sx.toFixed(3)} ${bar(c.dx_sx)}</td>
            <td>${c.dx_sy.toFixed(3)} ${bar(c.dx_sy)}</td></tr>
        <tr><td style="padding:2px 8px">PCCR dy</td>
            <td>${c.dy_sx.toFixed(3)} ${bar(c.dy_sx)}</td>
            <td>${c.dy_sy.toFixed(3)} ${bar(c.dy_sy)}</td></tr>
      </table>
      <div style="margin-top:6px;font-size:10px;color:#666">
        Good: |r|&gt;0.6 for dx↔X and dy↔Y. If dy↔X is stronger, camera may be rotated ~90°.
        If all correlations near 0, PCCR is not encoding gaze (check glint detection + camera angle).
      </div>`;
  }

  // Residuals plot (predicted vs actual screen position)
  if (data.preds && data.preds.length === n) {
    const panel = document.createElement('div'); panel.className='panel';
    panel.innerHTML = '<div class="label">Residuals: predicted vs actual screen position</div>' +
                      '<canvas id="cResid" height="300"></canvas>' +
                      '<div class="stats" id="statsResid"></div>';
    document.querySelector('.grid').appendChild(panel);
    setTimeout(() => {
      const px_arr = data.preds.map(p=>p.px), py_arr = data.preds.map(p=>p.py);
      const ex = sx.map((v,i)=>v - px_arr[i]);
      const ey = sy.map((v,i)=>v - py_arr[i]);
      const dist = ex.map((v,i)=>Math.sqrt(v*v+ey[i]*ey[i]));
      const mae = dist.reduce((a,b)=>a+b,0)/dist.length;
      const max_err = Math.max(...dist);
      const colResid = dist.map(d => {
        const t = Math.min(d/300,1);
        return `hsl(${Math.round((1-t)*120)},80%,50%)`;
      });
      scatter('cResid', px_arr, py_arr, colResid, 'pred X', 'pred Y');
      // Draw actual positions too
      const c = document.getElementById('cResid');
      const ctx = c.getContext('2d');
      const W=c.width, H=c.height, pad=36;
      const mnX=Math.min(...px_arr), mxX=Math.max(...px_arr);
      const mnY=Math.min(...py_arr), mxY=Math.max(...py_arr);
      const tx = v => pad+(v-mnX)/(mxX-mnX||1)*(W-2*pad);
      const ty = v => H-pad-(v-mnY)/(mxY-mnY||1)*(H-2*pad);
      ctx.strokeStyle='rgba(255,255,255,0.3)'; ctx.lineWidth=1;
      px_arr.forEach((_,i) => {
        ctx.beginPath();
        ctx.moveTo(tx(px_arr[i]), ty(py_arr[i]));
        ctx.lineTo(tx(sx[i]), ty(sy[i]));
        ctx.stroke();
      });
      document.getElementById('statsResid').innerHTML =
        `MAE=<span>${mae.toFixed(1)}px</span>  max_err=<span>${max_err.toFixed(1)}px</span>  ` +
        (mae < 100 ? '<span style="color:#4c4">OK</span>' :
         mae < 200 ? '<span style="color:#aa4">moderate</span>' :
                     '<span style="color:#a44">poor — check PCCR correlations</span>');
    }, 50);
  }
});
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_): pass

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            body = _HTML.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        elif self.path == "/ws":
            headers = {k.lower(): v for k, v in self.headers.items()}
            if "upgrade" not in headers or headers["upgrade"].lower() != "websocket":
                self.send_response(400); self.end_headers(); return
            if not _ws_handshake(self.rfile, self.wfile, headers):
                self.send_response(400); self.end_headers(); return
            _handle_ws(self.rfile, self.wfile)

        elif self.path == "/model":
            # Return current model coefficients
            if _model.trained:
                body = json.dumps({"trained": True, "A": _model.A.tolist(), "B": _model.B.tolist()}).encode()
            else:
                body = json.dumps({"trained": False}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        elif self.path.startswith("/predict"):
            import urllib.parse
            params = dict(urllib.parse.parse_qsl(urllib.parse.urlparse(self.path).query))
            try:
                dx, dy = float(params["dx"]), float(params["dy"])
                X, Y = _model.predict(dx, dy)
                body = json.dumps({"X": X, "Y": Y}).encode()
                code = 200
            except Exception as e:
                body = json.dumps({"error": str(e)}).encode()
                code = 400
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        elif self.path.startswith("/set_dict"):
            import urllib.parse
            name = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query).get("d", ["4x4"])[0]
            _switch_aruco_dict(name)
            body = json.dumps({"ok": True, "dict": name}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        elif self.path.startswith("/marker/"):
            try:
                # Strip query string before parsing ID
                raw = self.path.split("/marker/")[1].split("?")[0].split(".")[0]
                mid = int(raw)
                png = _marker_pngs[mid]
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(png)))
                self.send_header("Cache-Control", "max-age=3600")
                self.end_headers()
                self.wfile.write(png)
            except (KeyError, ValueError):
                self.send_response(404); self.end_headers()

        elif self.path.startswith("/set_calib_trace"):
            import urllib.parse
            q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            global _calib_trace_enabled
            if "v" in q and len(q["v"]) > 0 and q["v"][0] != "":
                on = q["v"][0] not in ("0", "false", "False")
                _calib_trace_enabled = on
                _forward_receiver_calib_trace(on)
                print(f"[calib] pipeline trace logging -> {'ON' if on else 'OFF'}", flush=True)
                body = json.dumps({"ok": True, "on": on}).encode()
            else:
                body = json.dumps({"on": _calib_trace_enabled}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        elif self.path.startswith("/pause_receiver"):
            import urllib.parse
            val = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query).get("v", ["1"])[0]
            try:
                urllib.request.urlopen(f"{RECEIVER_URL}/set?pause_streams={val}", timeout=1).close()
                body = b'{"ok":true}'
            except Exception as e:
                body = json.dumps({"ok": False, "error": str(e)}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        elif self.path == "/debug":
            with _debug_lock:
                d = dict(_debug)
            d.setdefault("frame_size", None)
            d.setdefault("brightness", None)
            d.setdefault("aruco_expected_met", False)
            d["expected_aruco_ids"] = list(ARUCO_IDS)
            with _homography_lock:
                d["homography_ok"] = _homography is not None
            d["calibrating"]     = _calibrating
            d["aruco_dict"]      = _aruco_dict_name
            d["calib_mode"]      = _calib_mode
            d["saccade_samples"] = len(_saccade_samples)
            d["model_trained"]   = _model.trained
            d["screen_size"]     = _last_screen_size
            d["calib_trace"]     = _calib_trace_enabled
            body = json.dumps(d).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        elif self.path == "/scene_frame":
            jpg = _last_scene_jpeg
            if jpg is None:
                self.send_response(503); self.end_headers(); return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(jpg)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(jpg)

        elif self.path == "/homography_debug":
            jpg = _last_homography_debug_jpeg
            if jpg is None:
                self.send_response(503); self.end_headers(); return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(jpg)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(jpg)

        elif self.path in ("/scene_stream", "/eye_stream"):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            is_eye = self.path == "/eye_stream"
            try:
                last_sent = None
                while True:
                    jpg = _last_eye_jpeg if is_eye else _last_scene_jpeg
                    if jpg is not None and jpg is not last_sent:
                        hdr = (b"--frame\r\nContent-Type: image/jpeg\r\n"
                               b"Content-Length: " + str(len(jpg)).encode() + b"\r\n\r\n")
                        self.wfile.write(hdr + jpg + b"\r\n")
                        self.wfile.flush()
                        last_sent = jpg
                    else:
                        time.sleep(0.01)
            except (BrokenPipeError, ConnectionResetError):
                pass

        elif self.path == "/live":
            # Fetch current pccr_vector from receiver, predict screen position
            try:
                with urllib.request.urlopen(f"{RECEIVER_URL}/stats", timeout=1) as r:
                    d = json.loads(r.read())
                vec = d.get("pccr_vector")
                if vec and _screen_model.trained:
                    sx, sy = _screen_model.predict(vec[0], vec[1])
                    body = json.dumps({"ok": True, "x": sx, "y": sy,
                                       "dx": vec[0], "dy": vec[1]}).encode()
                else:
                    body = json.dumps({"ok": False,
                                       "reason": "no vector" if not vec else "model not ready"}).encode()
            except Exception as e:
                body = json.dumps({"ok": False, "reason": str(e)}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        elif self.path == "/viz" or self.path == "/viz/data":
            # Load samples from disk (latest calib_dataset.json)
            try:
                with open(DATASET_PATH) as f:
                    samples = json.load(f)
            except Exception:
                samples = []
            with _saccade_lock:
                live = list(_saccade_samples)
            if live:
                samples = live  # prefer in-memory if available

            if self.path == "/viz/data":
                # Compute correlations and model predictions server-side
                corr = {}
                preds = []
                if len(samples) >= 3:
                    import numpy as _np
                    _dx = _np.array([s["dx"] for s in samples])
                    _dy = _np.array([s["dy"] for s in samples])
                    _sx = _np.array([s.get("sx", s.get("X", 0)) for s in samples])
                    _sy = _np.array([s.get("sy", s.get("Y", 0)) for s in samples])
                    def _corr(a, b):
                        if a.std() == 0 or b.std() == 0: return 0.0
                        return float(_np.corrcoef(a, b)[0, 1])
                    corr = {
                        "dx_sx": round(_corr(_dx, _sx), 3),
                        "dy_sx": round(_corr(_dy, _sx), 3),
                        "dx_sy": round(_corr(_dx, _sy), 3),
                        "dy_sy": round(_corr(_dy, _sy), 3),
                    }
                    if _model.trained:
                        try:
                            for s in samples:
                                px, py = _model.predict(s["dx"], s["dy"])
                                preds.append({"px": round(px, 1), "py": round(py, 1)})
                        except Exception:
                            preds = []
                body = json.dumps({"samples": samples,
                                   "n": len(samples),
                                   "model_trained": _model.trained,
                                   "corr": corr,
                                   "preds": preds}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                body = _VIZ_HTML.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        else:
            self.send_response(404); self.end_headers()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body   = self.rfile.read(length)

        if self.path == "/push_eye":
            # Receiver pushes PCCR for a single eye frame during the capture window.
            # Body: {"ts": float_ms, "dx": float, "dy": float, "x": float, "y": float, "r": float}
            # ts is the camera's own capture timestamp (synced to laptop clock).
            try:
                d = json.loads(body)
                entry = {
                    "ts": float(d["ts"]),
                    "dx": float(d["dx"]),
                    "dy": float(d["dy"]),
                }
                if "r" in d:
                    entry["r"] = float(d["r"])
                # Also buffer for display/debug
                with _eye_lock:
                    _eye_buf.append(entry)
                    cutoff = entry["ts"] - 10000
                    while _eye_buf and _eye_buf[0]["ts"] < cutoff:
                        _eye_buf.pop(0)
                # eye.jsonl is written only from _eye_poll_thread (/stats) to avoid duplicate clocks.
                # Add to pending accumulator for current target
                appended = False
                with _pending_lock:
                    if _pending_target is not None:
                        _pending_eye.append(entry)
                        appended = True
                _trace_record_push_eye(appended)
                self.send_response(200)
                self.send_header("Content-Length", "0")
                self.end_headers()
            except Exception as e:
                _calib_trace("POST /push_eye bad JSON or field: %r", e)
                try: self.send_error(400)
                except BrokenPipeError: pass
        else:
            try: self.send_error(404)
            except BrokenPipeError: pass


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


def main():
    threading.Thread(target=_scene_cam_thread, daemon=True).start()
    threading.Thread(target=_eye_cam_thread,   daemon=True).start()
    threading.Thread(target=_eye_poll_thread,  daemon=True).start()

    server = ThreadedHTTPServer(("", 8090), Handler)
    print("Calibration server on http://localhost:8090")
    print(f"Scene cam: cam{SCENE_CAM_ID} (world cam from rig_config) via {RECEIVER_URL}")
    print(f"Model path: {MODEL_PATH}")
    if _model.trained:
        print("Existing gaze model loaded.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
