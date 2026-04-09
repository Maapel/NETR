"""
Smooth Pursuit Calibration Server — Phase 3 + 4.

Serves the calibration webapp at http://localhost:8090
WebSocket at ws://localhost:8090/ws

Workflow:
  1. Open http://localhost:8090 in browser (fullscreen recommended).
  2. Press START — Lissajous target begins moving over ArUco corners.
  3. The eye camera should already be running (receiver.py on port 8080).
  4. Press STOP — dataset is collected & gaze model is trained.
  5. Model saved to gaze_model.json in the project directory.

Scene camera: reads from the receiver's MJPEG stream (cam1 or cam2)
Eye vectors:  polled from receiver's /stats endpoint (live pccr_vector)

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

# ── Rig config ────────────────────────────────────────────────────────────────
import sys as _sys
_sys.path.insert(0, str(pathlib.Path(__file__).parent))
try:
    import rig_config as _rig_cfg
    _WORLD_CAM = _rig_cfg.world_cam()
except Exception:
    _WORLD_CAM = 1

# ── Config ────────────────────────────────────────────────────────────────────
RECEIVER_URL  = "http://localhost:8080"
ENGINE_URL    = "http://localhost:8081"
SCENE_CAM_ID  = _WORLD_CAM     # which cam to use for ArUco detection (world cam from rig_config)
ARUCO_DICT    = cv2.aruco.DICT_4X4_50
ARUCO_IDS     = [0, 1, 2, 3]   # TL, TR, BL, BR order
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

# ── Debug state ───────────────────────────────────────────────────────────────
_debug = {
    "scene_cam_ok": False,
    "scene_cam_error": "",
    "aruco_detected": 0,          # number of ArUco markers found in last frame
    "aruco_ids": [],
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

# Saccade mode: directly collected fixation samples {dx, dy, X, Y, sx, sy}
_saccade_samples: list[dict] = []
_saccade_lock = threading.Lock()

# Push-based capture: frames received from receiver during the active capture window
_pending_eye: list[dict] = []   # [{ts, dx, dy}] for the current target
_pending_target: dict | None = None  # {x, y} of the target being captured
_pending_lock = threading.Lock()
FIXATE_MS = 300   # capture window duration (matches JS FIXATE_MS)

_model = GazeModel()                          # scene-space model (saved to disk)
_screen_model = GazeModel()                   # screen-space model (for live cursor)
SCREEN_MODEL_PATH = pathlib.Path(__file__).parent / "screen_model.json"
_model.load(MODEL_PATH)
_screen_model.load(SCREEN_MODEL_PATH)

# ── Recording ─────────────────────────────────────────────────────────────────
RECORDINGS_DIR = pathlib.Path(__file__).parent / "recordings"
_recording       = False
_rec_dir: pathlib.Path | None = None
_rec_eye_f       = None   # open file handle for eye.jsonl
_rec_target_f    = None   # open file handle for targets.jsonl
_rec_fixation_f  = None   # open file handle for fixations.jsonl
_rec_frame_n     = 0      # total frames seen during recording
_rec_lock        = threading.Lock()


def _rec_start(screen_w: int, screen_h: int, mode: str):
    global _recording, _rec_dir, _rec_eye_f, _rec_target_f, _rec_fixation_f, _rec_frame_n
    ts = time.strftime("%Y%m%d_%H%M%S")
    d  = RECORDINGS_DIR / ts
    (d / "frames").mkdir(parents=True, exist_ok=True)
    _rec_dir      = d
    _rec_frame_n  = 0
    _rec_eye_f    = open(d / "eye.jsonl",      "w")
    _rec_target_f = open(d / "targets.jsonl",  "w")
    _rec_fixation_f = open(d / "fixations.jsonl", "w")
    with open(d / "meta.json", "w") as f:
        json.dump({
            "screen_w": screen_w, "screen_h": screen_h,
            "mode": mode, "aruco_dict": _aruco_dict_name,
            "scene_cam": SCENE_CAM_ID, "started": ts,
        }, f, indent=2)
    _recording = True
    print(f"[rec] Recording started → {d}")
    return str(d)


def _rec_stop() -> str:
    global _recording, _rec_dir, _rec_eye_f, _rec_target_f, _rec_fixation_f
    _recording = False
    for f in (_rec_eye_f, _rec_target_f, _rec_fixation_f):
        try: f.close()
        except Exception: pass
    _rec_eye_f = _rec_target_f = _rec_fixation_f = None
    path = str(_rec_dir) if _rec_dir else ""
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
        except Exception:
            pass
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
        except Exception:
            pass
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


def _flush_pending_target():
    """For each captured eye frame, fetch the world frame at the same timestamp,
    run ArUco, compute scene_xy, then average all synced (dx,dy,X,Y) pairs."""
    global _pending_eye, _pending_target
    with _pending_lock:
        eyes   = list(_pending_eye)
        target = _pending_target
        _pending_eye    = []
        _pending_target = None
    if not eyes or not target:
        return
    import numpy as _np

    sx, sy = target["x"], target["y"]

    # Build per-sample synced tuples: fetch world frame at each eye timestamp
    synced: list[tuple[float, float, float, float]] = []   # (dx, dy, X, Y)
    aruco_miss = 0
    for e in eyes:
        frame = _fetch_world_frame_at(e["ts"])
        if frame is None:
            aruco_miss += 1
            continue
        detected, _ = _detect_aruco_corners(frame)
        if not detected:
            aruco_miss += 1
            continue
        H = _compute_homography(detected)
        if H is None:
            aruco_miss += 1
            continue
        pt = np.array([[[sx, sy]]], dtype=np.float32)
        sc = cv2.perspectiveTransform(pt, H)[0][0]
        synced.append((e["dx"], e["dy"], float(sc[0]), float(sc[1])))

    n_synced = len(synced)
    print(f"[calib] Target ({sx:.0f},{sy:.0f}) n={len(eyes)} synced={n_synced} aruco_miss={aruco_miss}")
    if n_synced == 0:
        print(f"[calib] No synced samples for target ({sx:.0f},{sy:.0f}) — skipped")
        return

    dxs = _np.array([s[0] for s in synced])
    dys = _np.array([s[1] for s in synced])
    Xs  = _np.array([s[2] for s in synced])
    Ys  = _np.array([s[3] for s in synced])

    # Drop frozen (all identical) — tracking stuck
    if dxs.std() == 0 and dys.std() == 0:
        print(f"[calib] Skipping frozen target ({sx:.0f},{sy:.0f}) n={n_synced}")
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
ARUCO_DICTS = {
    "4x4":  cv2.aruco.DICT_4X4_50,
    "5x5":  cv2.aruco.DICT_5X5_50,
    "6x6":  cv2.aruco.DICT_6X6_50,
    "7x7":  cv2.aruco.DICT_7X7_50,
}
_aruco_dict_name = "4x4"
_aruco_dict      = cv2.aruco.getPredefinedDictionary(ARUCO_DICTS[_aruco_dict_name])
_aruco_lock      = threading.Lock()

def _make_params():
    p = cv2.aruco.DetectorParameters()
    p.adaptiveThreshWinSizeMin    = 3
    p.adaptiveThreshWinSizeMax    = 53
    p.adaptiveThreshWinSizeStep   = 4
    p.minMarkerPerimeterRate      = 0.01
    p.polygonalApproxAccuracyRate = 0.08
    return p

_aruco_detector = cv2.aruco.ArucoDetector(_aruco_dict, _make_params())


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
    if name not in ARUCO_DICTS:
        print(f"[aruco] Unknown dict '{name}', options: {list(ARUCO_DICTS)}")
        return
    with _aruco_lock:
        _aruco_dict_name = name
        _aruco_dict      = cv2.aruco.getPredefinedDictionary(ARUCO_DICTS[name])
        _aruco_detector  = cv2.aruco.ArucoDetector(_aruco_dict, _make_params())
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


_clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

def _detect_aruco_corners(frame_bgr: np.ndarray) -> tuple[dict[int, np.ndarray] | None, list[int]]:
    """Detect ArUco markers. Returns (dict id->center, list of all found ids). dict is None if <4 required found."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = _clahe.apply(gray)   # boost contrast before detection (handles dark/low-exposure frames)
    with _aruco_lock:
        det = _aruco_detector
    corners, ids, _ = det.detectMarkers(gray)
    all_ids = ids.flatten().tolist() if ids is not None else []
    if ids is None or len(ids) < 4:
        return None, all_ids
    result = {}
    for i, mid in enumerate(ids.flatten()):
        if mid in ARUCO_IDS:
            c = corners[i][0]
            result[int(mid)] = c.mean(axis=0)
    if len(result) < 4:
        return None, all_ids
    return result, all_ids


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
    if len(sm) < 4 or not all(i in sm for i in ARUCO_IDS):
        print(f"[homography] Missing screen marker positions: have {list(sm.keys())}, need {ARUCO_IDS}")
        return None
    screen_pts = np.array([sm[i] for i in ARUCO_IDS], dtype=np.float32)
    scene_pts  = np.array([scene_corners[i] for i in ARUCO_IDS], dtype=np.float32)
    H, _ = cv2.findHomography(screen_pts, scene_pts)
    return H


def _screen_to_scene(x: float, y: float) -> tuple[float, float] | None:
    with _homography_lock:
        H = _homography
    if H is None:
        return None
    pt = np.array([[[x, y]]], dtype=np.float32)
    res = cv2.perspectiveTransform(pt, H)
    return float(res[0][0][0]), float(res[0][0][1])


# ── Scene cam poller ──────────────────────────────────────────────────────────
def _scene_cam_thread():
    """Periodically grab a frame from the receiver MJPEG stream and update homography."""
    global _homography, _last_scene_jpeg, _rec_frame_n
    snap_url = f"{RECEIVER_URL}/jpeg/{SCENE_CAM_ID}"
    prev_aruco_count = -1
    while True:
        try:
            with urllib.request.urlopen(snap_url, timeout=2) as resp:
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
                # Record raw frame every ~7th poll (~3fps) to keep storage sane
                if _recording and _rec_dir:
                    with _rec_lock:
                        n = _rec_frame_n
                        _rec_frame_n += 1
                    if n % 7 == 0:
                        fpath = _rec_dir / "frames" / f"{n:06d}.jpg"
                        cv2.imwrite(str(fpath), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                        with open(_rec_dir / "frames_index.jsonl", "a") as fi:
                            fi.write(json.dumps({"n": n, "ts": time.time()*1000,
                                                 "file": f"frames/{n:06d}.jpg"}) + "\n")
                n = len(all_ids)
                with _debug_lock:
                    _debug["scene_cam_ok"] = True
                    _debug["scene_cam_error"] = ""
                    _debug["aruco_detected"] = n
                    _debug["aruco_ids"] = all_ids
                    _debug["frame_size"] = [w_px, h_px]
                    _debug["brightness"] = round(mean_brightness, 1)
                if n != prev_aruco_count:
                    print(f"[scene] frame={w_px}×{h_px}  brightness={mean_brightness:.0f}/255"
                          f"  ArUco: {n} found IDs={all_ids}" +
                          (f"  MISSING={sorted(set(ARUCO_IDS)-set(all_ids))}" if n < 4 else "  → homography OK"))
                    prev_aruco_count = n
                if detected:
                    H = _compute_homography(detected)
                    if H is not None:
                        with _homography_lock:
                            _homography = H
                        with _debug_lock:
                            _debug["homography_ok"] = True
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
        except Exception as e:
            err = str(e)
            with _debug_lock:
                _debug["scene_cam_ok"] = False
                _debug["scene_cam_error"] = err
                _debug["homography_ok"] = False
            print(f"[scene] Error fetching {snap_url}: {err}")
        time.sleep(0.05)  # ~20 fps

# ── Eye cam poller (engine /frame — has pupil/glint overlay) ─────────────────
def _eye_cam_thread():
    """Fetch analyzed eye frame from engine at ~20fps for /eye_stream."""
    global _last_eye_jpeg
    frame_url = f"{ENGINE_URL}/frame"
    while True:
        try:
            with urllib.request.urlopen(frame_url, timeout=1) as resp:
                data = resp.read()
            if data:
                _last_eye_jpeg = data
        except Exception:
            pass
        time.sleep(0.05)

_last_screen_size = [0, 0]   # updated by websocket messages


# ── Eye vector poller ─────────────────────────────────────────────────────────
def _eye_poll_thread():
    """Poll engine /result directly for PCCR with the engine's own processing timestamp.

    Using engine timestamp instead of poll time because:
    - Engine stamps each frame when processing finishes (time.time())
    - Calibration server and browser both use the same laptop wall clock
    - Poll time would add up to 33ms jitter per sample; engine ts is accurate to frame
    - Deduplication by engine ts prevents the same frame being counted twice
      (engine ~14fps, poll ~30Hz → each frame would otherwise appear ~2x)
    """
    global _last_pccr_ts
    result_url = f"{ENGINE_URL}/result"
    prev_vec_ok = None
    _last_engine_ts: float = 0.0   # track last seen engine frame ts for deduplication

    while True:
        try:
            with urllib.request.urlopen(result_url, timeout=1) as resp:
                d = json.loads(resp.read())
            vec = d.get("pccr_vector") if d.get("ready") else None
            # Engine stores ts in seconds; convert to ms for consistency with browser
            engine_ts_ms = d.get("ts", 0.0) * 1000

            if vec and len(vec) == 2 and engine_ts_ms > 0:
                _last_pccr_ts = engine_ts_ms
                with _debug_lock:
                    _debug["eye_poll_ok"] = True
                    _debug["eye_poll_error"] = ""
                    _debug["pccr_vector"] = [round(vec[0], 3), round(vec[1], 3)]
                    _debug["pccr_age_ms"] = 0
                if prev_vec_ok is not True:
                    print(f"[eye] pccr_vector OK (engine ts): {vec}")
                    prev_vec_ok = True

                # Only add to buffer if this is a new engine frame (dedup)
                if engine_ts_ms > _last_engine_ts:
                    _last_engine_ts = engine_ts_ms
                    entry = {"ts": engine_ts_ms, "dx": vec[0], "dy": vec[1]}
                    with _eye_lock:
                        _eye_buf.append(entry)
                        cutoff = engine_ts_ms - 10000
                        while _eye_buf and _eye_buf[0]["ts"] < cutoff:
                            _eye_buf.pop(0)
                    if _recording and _rec_eye_f:
                        try:
                            _rec_eye_f.write(json.dumps(entry) + "\n")
                            _rec_eye_f.flush()
                        except Exception: pass
            else:
                poll_ts = time.time() * 1000
                age = poll_ts - _last_pccr_ts if _last_pccr_ts else None
                with _debug_lock:
                    _debug["pccr_vector"] = None
                    _debug["pccr_age_ms"] = round(age) if age else None
                if prev_vec_ok is not False:
                    print(f"[eye] pccr_vector is null — engine ready={d.get('ready')}. "
                          "Check: engine running? analysis toggled on? eye cam correct?")
                    prev_vec_ok = False
        except Exception as e:
            err = str(e)
            with _debug_lock:
                _debug["eye_poll_ok"] = False
                _debug["eye_poll_error"] = err
            if prev_vec_ok is not False:
                print(f"[eye] Error polling {result_url}: {err}")
                prev_vec_ok = False
        with _debug_lock:
            _debug["target_buf_size"] = len(_target_buf)
            _debug["eye_buf_size"] = len(_eye_buf)
        time.sleep(0.033)  # ~30 Hz poll; dedup ensures each engine frame counted once


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
    global _calibrating, _calib_mode
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
                # Auto-start recording if not already recording
                if _recording and _rec_target_f:
                    try:
                        _rec_target_f.write(json.dumps({"event": "calib_start",
                                                         "ts": time.time()*1000,
                                                         "mode": _calib_mode}) + "\n")
                        _rec_target_f.flush()
                    except Exception: pass
                _ws_send_frame(wfile, json.dumps({"type": "ack", "msg": f"calibration started ({_calib_mode})"}))

            elif mtype == "target":
                if _calibrating:
                    entry = {"ts": msg["ts"], "x": msg["x"], "y": msg["y"]}
                    with _target_lock:
                        _target_buf.append(entry)
                    if _recording and _rec_target_f:
                        try:
                            _rec_target_f.write(json.dumps(entry) + "\n")
                            _rec_target_f.flush()
                        except Exception: pass

            elif mtype == "fixation":
                # Saccade mode: eye has settled — eye is now fixated for FIXATE_MS.
                # 1. Flush pending samples from the PREVIOUS target (now complete).
                # 2. Open a new capture window for this target on the receiver.
                # Receiver will push eye frames during [fixation_ts, fixation_ts+FIXATE_MS].
                if _calibrating:
                    ts = msg["ts"]   # ms, same clock as receiver's time.time()*1000
                    x, y = msg["x"], msg["y"]
                    # Flush previous target's accumulated frames → adds to dataset
                    threading.Thread(target=_flush_pending_target, daemon=True).start()
                    # Register new target and open capture window on receiver
                    with _pending_lock:
                        _pending_target = {"x": x, "y": y}
                    _set_calib_window(x, y, from_ms=ts, until_ms=ts + FIXATE_MS)
                    # Record fixation event
                    if _recording and _rec_target_f:
                        try:
                            _rec_target_f.write(json.dumps({
                                "ts": ts, "x": x, "y": y, "event": "fixation"
                            }) + "\n")
                            _rec_target_f.flush()
                        except Exception: pass

            elif mtype == "stop":
                _calibrating = False
                if _calib_mode == "saccade":
                    with _saccade_lock:
                        samples = list(_saccade_samples)
                else:
                    sw, sh = _last_screen_size
                    samples = _sync_and_build_dataset(sw, sh)
                # Save samples to recording
                if _recording and _rec_dir:
                    try:
                        with open(_rec_dir / "samples.json", "w") as f:
                            json.dump(samples, f)
                    except Exception: pass
                result = {"type": "result", "n_samples": len(samples),
                          "recording": _recording}
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
  <button id="btnRecord">⏺ Record</button>
  <button id="btnPause">⏸ Pause Streams</button>
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
  // Send marker positions after WS is open
  setTimeout(updateMarkers, 100);
};
ws.onclose = () => statusEl.textContent = 'Disconnected';
ws.onmessage = e => {
  const m = JSON.parse(e.data);
  if (m.type === 'record_ack') { handleRecordAck(m); return; }
  if (m.type === 'ack')    statusEl.textContent = 'Calibrating…';
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
    const rec = m.recording ? '  💾 recorded' : '';
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
const FIXATE_MS = 300;   // sample window after settling (ms)
const POINT_MS  = SETTLE_MS + FIXATE_MS;

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
}

function initSaccade() {
  zoneSeq    = buildZoneSequence();
  zoneSeqIdx = 0;
  saccadeCount = 0;
  nextSaccadePoint();  // first point = center zone
}

function tickSaccade(now) {
  const dwell   = now - saccadeStart;
  const settled = dwell >= SETTLE_MS;

  if (settled && !saccadeSampled) {
    if (ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({
        type: 'fixation',
        ts:   performance.timeOrigin + now,
        x:    saccadePos.x,
        y:    saccadePos.y,
      }));
    }
    saccadeSampled = true;
    saccadeCount++;
    const round = Math.ceil(saccadeCount / 9);
    const inRound = ((saccadeCount - 1) % 9) + 1;
    statusEl.textContent = `Saccade: ${saccadeCount} pts  (round ${round}, point ${inRound}/9)`;
  }

  if (dwell >= POINT_MS) nextSaccadePoint();

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
    const arucoOk = d.aruco_detected >= 4;
    dbg('d-aruco', `${d.aruco_detected} found  IDs=[${d.aruco_ids}]`, arucoOk ? true : (d.aruco_detected > 0 ? null : false));
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
            with _homography_lock:
                d["homography_ok"] = _homography is not None
            d["calibrating"]     = _calibrating
            d["aruco_dict"]      = _aruco_dict_name
            d["calib_mode"]      = _calib_mode
            d["saccade_samples"] = len(_saccade_samples)
            d["model_trained"]   = _model.trained
            d["screen_size"]     = _last_screen_size
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
            # Body: {"ts": float_ms, "dx": float, "dy": float, "x": float, "y": float}
            # ts is the camera's own capture timestamp (synced to laptop clock).
            try:
                d = json.loads(body)
                entry = {
                    "ts": float(d["ts"]),
                    "dx": float(d["dx"]),
                    "dy": float(d["dy"]),
                }
                # Also buffer for display/debug
                with _eye_lock:
                    _eye_buf.append(entry)
                    cutoff = entry["ts"] - 10000
                    while _eye_buf and _eye_buf[0]["ts"] < cutoff:
                        _eye_buf.pop(0)
                # Record raw eye stream
                if _recording and _rec_eye_f:
                    try:
                        _rec_eye_f.write(json.dumps(entry) + "\n")
                        _rec_eye_f.flush()
                    except Exception: pass
                # Add to pending accumulator for current target
                with _pending_lock:
                    if _pending_target is not None:
                        _pending_eye.append(entry)
                self.send_response(200)
                self.send_header("Content-Length", "0")
                self.end_headers()
            except Exception:
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
