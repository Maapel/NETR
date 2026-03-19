"""
Dual-camera UDP receiver — time-synced browser display.

Packet format (little-endian):
  [4B frame_id][2B chunk_idx][2B total_chunks][8B timestamp_us][JPEG chunk]

Ports:
  cam1 stream → 5000   cam1 cmd → 5001
  cam2 stream → 5002   cam2 cmd → 5003
  discovery   → 5004

Open http://localhost:8080 in your browser.
"""

import socket
import struct
import sys
import time
import threading
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

# ── Pupil detection (optional — requires cv2/numpy) ───────────────────────────
# Inlined from /home/maadhav/iot-software/pupil_detection.py — cv2/numpy only,
# no pygame dependency.
try:
    import cv2
    import numpy as np

    _clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    _GLINT_THRESH = 190
    _HOUGH_P1, _HOUGH_P2 = 50, 20
    _R_MIN, _R_MAX = 30, 160
    _SX, _SY = (0.25, 0.75), (0.10, 0.75)

    def _detect_pupil(gray, _unused):
        h, w = gray.shape
        _, glint = cv2.threshold(gray, _GLINT_THRESH, 255, cv2.THRESH_BINARY)
        glint = cv2.dilate(glint,
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
                           iterations=2)
        clean    = cv2.inpaint(gray, glint, inpaintRadius=9, flags=cv2.INPAINT_TELEA)
        enhanced = _clahe.apply(clean)
        blurred  = cv2.GaussianBlur(enhanced, (5, 5), 0)
        x0, x1 = int(w * _SX[0]), int(w * _SX[1])
        y0, y1 = int(h * _SY[0]), int(h * _SY[1])
        crop = blurred[y0:y1, x0:x1]
        circles = cv2.HoughCircles(crop, cv2.HOUGH_GRADIENT,
                                   dp=1.2, minDist=50,
                                   param1=_HOUGH_P1, param2=_HOUGH_P2,
                                   minRadius=_R_MIN, maxRadius=_R_MAX)
        debug = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        if circles is None:
            return None, None, debug
        circles = np.round(circles[0]).astype(int)
        circles[:, 0] += x0
        circles[:, 1] += y0
        cx, cy, r = circles[0]
        return (int(cx), int(cy)), int(r), debug

    def _draw_overlay(frame, center, radius):
        if center is None:
            return frame
        cx, cy = center
        r = radius if radius else 20
        cv2.circle(frame, (cx, cy), r, (0, 200, 255), 2)
        cv2.circle(frame, (cx, cy), 4, (0, 255,   0), -1)
        cv2.line(frame, (cx - r, cy), (cx + r, cy), (0, 255, 0), 1)
        cv2.line(frame, (cx, cy - r), (cx, cy + r), (0, 255, 0), 1)
        cv2.putText(frame, f"({cx},{cy})", (cx + r + 4, cy - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        return frame

    _PUPIL_OK = True
except ImportError:
    _PUPIL_OK = False

g_analysis_enabled = False   # toggled via /set?analysis=1|0

# ── Persistent settings ───────────────────────────────────────────────────────
import pathlib as _pathlib
_SETTINGS_FILE = _pathlib.Path(__file__).parent / "cam_settings.json"
_DEFAULT_SETTINGS = {
    "quality": 12, "fps": 30, "res": 5,
    "brightness": 0, "contrast": 0, "saturation": 0,
    "auto_exposure": 1, "exposure": 300, "auto_gain": 1,
    "gain": 0, "ae_level": 0, "hmirror": 0, "vflip": 0, "wb_mode": 0,
}
def _load_settings():
    try:
        with open(_SETTINGS_FILE) as f:
            saved = json.load(f)
        return {**_DEFAULT_SETTINGS, **saved}
    except (FileNotFoundError, json.JSONDecodeError):
        return dict(_DEFAULT_SETTINGS)
def _save_settings(s):
    with open(_SETTINGS_FILE, "w") as f:
        json.dump(s, f, indent=2)
g_settings = _load_settings()

def _apply_pupil_overlay(data: bytes) -> bytes:
    """Decode JPEG, run pupil detection, draw overlay, re-encode. Returns JPEG bytes."""
    if not _PUPIL_OK or not data:
        return data
    arr = np.frombuffer(data, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        return data
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    center, radius, _ = _detect_pupil(gray, 0)
    _draw_overlay(bgr, center, radius)
    _, enc = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return enc.tobytes()

import collections
import os
from datetime import datetime

RECORD_BUF_SECS = 30  # 30-seconds rolling buffer
RECORD_MAX_FRAMES = 30 * RECORD_BUF_SECS  # ~3600 frames at 30fps

HTTP_PORT      = 8080
DISCOVERY_PORT = 5004
TIMESYNC_PORT  = 5005
FRAME_TIMEOUT  = 0.25    # discard incomplete frames older than this (s)
SYNC_WINDOW_US = 200_000 # max |ts1-ts2| to call frames "paired" (200 ms)
RECENT_BUF_US  = 500_000 # how long to keep frames for pairing (500 ms)

HDR      = struct.Struct("<IHHq")
HDR_SIZE = HDR.size   # 16 bytes


# ── Per-camera state ───────────────────────────────────────────────────────────
class CamState:
    def __init__(self, cam_id: int):
        self.cam_id   = cam_id
        self.udp_port = 5000 + (cam_id - 1) * 2
        self.cmd_port = 5001 + (cam_id - 1) * 2
        self.ip       = ""

        # UDP reassembly buffer
        self._frames: dict = {}
        self._frames_lock  = threading.Lock()

        # Latest complete frame
        self.latest_frame: bytes = b""
        self.frame_lock  = threading.Lock()
        self.frame_event = threading.Event()

        # Recent frames kept for timestamp-based pairing: [(ts_us, jpeg), ...]
        self._recent: list[tuple[int, bytes]] = []
        self._recent_lock = threading.Lock()

        # Rolling 2-minute recording buffer: deque of (ts_us, jpeg_bytes)
        self.rec_buf: collections.deque = collections.deque(maxlen=RECORD_MAX_FRAMES)

        self.stats = {
            "fps": 0.0, "latency_ms": 0.0,
            "dropped": 0, "online": False, "last_ts_us": 0,
        }
        self._frame_count = 0
        self._fps_ts = time.monotonic()

    def send_cmd(self, cmd: bytes):
        if not self.ip:
            return
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.sendto(cmd, (self.ip, self.cmd_port))

    def _expire_old(self, now: float):
        expired = [fid for fid, f in self._frames.items()
                   if now - f["recv_time"] > FRAME_TIMEOUT]
        for fid in expired:
            self.stats["dropped"] += 1
            del self._frames[fid]

    def ingest(self, pkt: bytes):
        """Process one UDP packet. Returns completed JPEG bytes or None."""
        if len(pkt) < HDR_SIZE:
            return None
        frame_id, chunk_idx, total_chunks, ts_us = HDR.unpack_from(pkt)
        payload = pkt[HDR_SIZE:]
        now = time.monotonic()

        with self._frames_lock:
            self._expire_old(now)
            if frame_id not in self._frames:
                self._frames[frame_id] = {
                    "ts": ts_us, "total": total_chunks,
                    "chunks": {}, "recv_time": now,
                }
            entry = self._frames[frame_id]
            entry["chunks"][chunk_idx] = payload

            if len(entry["chunks"]) != entry["total"]:
                return None

            jpeg = b"".join(entry["chunks"][i] for i in range(entry["total"]))
            del self._frames[frame_id]

        # Update latest frame
        with self.frame_lock:
            self.latest_frame = jpeg
        self.frame_event.set()

        # Rolling 2-min recording buffer (always recording)
        self.rec_buf.append((ts_us, jpeg))

        # Store in recent buffer for pairing
        now_us = int(time.time() * 1e6)
        with self._recent_lock:
            self._recent.append((ts_us, jpeg))
            cutoff = now_us - RECENT_BUF_US
            while self._recent and self._recent[0][0] < cutoff:
                self._recent.pop(0)

        # Update stats
        self.stats["online"]     = True
        self.stats["last_ts_us"] = ts_us
        self.stats["latency_ms"] = (now_us - ts_us) / 1000.0
        self._frame_count += 1
        elapsed = time.monotonic() - self._fps_ts
        if elapsed >= 2.0:
            self.stats["fps"] = self._frame_count / elapsed
            print(
                f"cam{self.cam_id}  FPS:{self.stats['fps']:.1f}"
                f"  lat:{self.stats['latency_ms']:.0f}ms"
                f"  frame:{len(jpeg)}B  drop:{self.stats['dropped']}"
            )
            self._frame_count = 0
            self._fps_ts = time.monotonic()

        return jpeg

    def best_frame_near(self, ts_us: int) -> bytes | None:
        """Return the frame from our recent buffer closest to ts_us, or None."""
        with self._recent_lock:
            if not self._recent:
                return None
            best_ts, best_jpeg = min(self._recent, key=lambda f: abs(f[0] - ts_us))
            if abs(best_ts - ts_us) > SYNC_WINDOW_US:
                return None
            return best_jpeg


CAMS = {1: CamState(1), 2: CamState(2)}

# ── AVI reader cache (avoid reopening per frame) ─────────────────────────────
_avi_cache = {}      # key: (rec_name, cid) → {"cap": VideoCapture, "total": int}
_avi_cache_lock = threading.Lock()

def _get_avi(rec_name: str, cid: int):
    """Return (VideoCapture, total_frames) from cache, opening if needed."""
    key = (rec_name, cid)
    with _avi_cache_lock:
        if key in _avi_cache:
            entry = _avi_cache[key]
            return entry["cap"], entry["total"], entry["lock"]
    # Open outside global lock
    avi_path = _pathlib.Path(__file__).parent / "recordings" / rec_name / f"cam{cid}.avi"
    if not avi_path.exists():
        return None, 0, None
    cap = cv2.VideoCapture(str(avi_path))
    # Count frames reliably by seeking to end
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    lock = threading.Lock()
    with _avi_cache_lock:
        _avi_cache[key] = {"cap": cap, "total": total, "lock": lock}
    return cap, total, lock

# Global sync offset reported to browser
sync_offset_ms = 0.0


# ── HTTP server ────────────────────────────────────────────────────────────────
class MJPEGHandler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_GET(self):
        path = self.path.split("?")[0]
        if   path == "/":          self._index()
        elif path == "/stream/1":  self._mjpeg(CAMS[1])
        elif path == "/stream/2":  self._mjpeg(CAMS[2])
        elif path == "/jpeg/1":    self._jpeg(CAMS[1])
        elif path == "/jpeg/2":    self._jpeg(CAMS[2])
        elif path == "/stats":     self._stats()
        elif path == "/settings":  self._get_settings()
        elif path == "/record/save": self._record_save()
        elif path == "/player":      self._player()
        elif path == "/recordings":  self._list_recordings()
        elif path.startswith("/recordings/info/"): self._recording_info(path)
        elif path.startswith("/playback/"): self._playback_frame(path)
        elif self.path.startswith("/set?"): self._set_cmd(self.path[5:])
        else:
            try: self.send_error(404)
            except BrokenPipeError: pass

    # ── /stats ────────────────────────────────────────────────────────────────
    def _stats(self):
        body = json.dumps({
            "cam1": CAMS[1].stats,
            "cam2": CAMS[2].stats,
            "sync_offset_ms": round(sync_offset_ms, 1),
        }).encode()
        try:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except BrokenPipeError:
            pass

    # ── /settings (return saved settings as JSON) ──────────────────────────
    def _get_settings(self):
        try:
            body = json.dumps(g_settings).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except BrokenPipeError:
            pass

    # ── /record/save — dump rolling 2-min buffer to disk ────────────────────
    def _record_save(self):
        threading.Thread(target=self._do_save, daemon=True).start()
        body = json.dumps({"ok": True, "msg": "Saving..."}).encode()
        try:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except BrokenPipeError:
            pass

    @staticmethod
    def _do_save():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = _pathlib.Path(__file__).parent / "recordings" / ts
        out_dir.mkdir(parents=True, exist_ok=True)
        for cid, cam in CAMS.items():
            frames = list(cam.rec_buf)
            if not frames:
                print(f"cam{cid}: no frames to save")
                continue
            cam_dir = out_dir / f"cam{cid}"
            if _PUPIL_OK:
                # Save as MJPEG AVI
                first_jpg = frames[0][1]
                arr = np.frombuffer(first_jpg, np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                h, w = img.shape[:2]
                # Estimate FPS from timestamps
                dt = (frames[-1][0] - frames[0][0]) / 1e6
                fps = len(frames) / dt if dt > 0 else 30.0
                path = str(out_dir / f"cam{cid}.avi")
                writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"),
                                         fps, (w, h))
                for _, jpeg in frames:
                    a = np.frombuffer(jpeg, np.uint8)
                    f = cv2.imdecode(a, cv2.IMREAD_COLOR)
                    if f is not None:
                        writer.write(f)
                writer.release()
                print(f"cam{cid}: saved {len(frames)} frames → {path}  ({fps:.1f} fps)")
            else:
                # Fallback: save individual JPEGs
                cam_dir.mkdir(parents=True, exist_ok=True)
                for i, (ts_us, jpeg) in enumerate(frames):
                    (cam_dir / f"{i:05d}_{ts_us}.jpg").write_bytes(jpeg)
                print(f"cam{cid}: saved {len(frames)} frames → {cam_dir}/")

    # ── /recordings — list saved recordings ─────────────────────────────────
    def _list_recordings(self):
        rec_dir = _pathlib.Path(__file__).parent / "recordings"
        result = []
        if rec_dir.is_dir():
            for d in sorted(rec_dir.iterdir(), reverse=True):
                if not d.is_dir():
                    continue
                entry = {"name": d.name, "cams": {}}
                for cid in [1, 2]:
                    avi = d / f"cam{cid}.avi"
                    jpgdir = d / f"cam{cid}"
                    if avi.exists():
                        entry["cams"][str(cid)] = {"type": "avi", "path": str(avi)}
                    elif jpgdir.is_dir():
                        frames = sorted(jpgdir.glob("*.jpg"))
                        entry["cams"][str(cid)] = {"type": "jpg", "count": len(frames)}
                if entry["cams"]:
                    result.append(entry)
        body = json.dumps(result).encode()
        try:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except BrokenPipeError:
            pass

    # ── /recordings/info/<rec_name> — get frame counts without loading ────
    def _recording_info(self, path: str):
        rec_name = path.split("/")[-1]
        rec_dir = _pathlib.Path(__file__).parent / "recordings" / rec_name
        info = {}
        for cid in [1, 2]:
            avi = rec_dir / f"cam{cid}.avi"
            jpgdir = rec_dir / f"cam{cid}"
            if avi.exists() and _PUPIL_OK:
                cap, total, _ = _get_avi(rec_name, cid)
                info[str(cid)] = total if cap else 0
            elif jpgdir.is_dir():
                info[str(cid)] = len(list(jpgdir.glob("*.jpg")))
            else:
                info[str(cid)] = 0
        body = json.dumps(info).encode()
        try:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except BrokenPipeError:
            pass

    # ── /playback/<rec_name>/cam<n>/<frame_idx>?analysis=0|1 ─────────────
    def _playback_frame(self, path: str):
        import urllib.parse
        query = ""
        if "?" in self.path:
            query = self.path.split("?", 1)[1]
        params = dict(urllib.parse.parse_qsl(query))
        do_analysis = params.get("analysis", "0") != "0"

        parts = path.strip("/").split("/")  # playback / <rec> / cam<n> / <idx>
        if len(parts) != 4:
            try: self.send_error(400)
            except BrokenPipeError: pass
            return
        _, rec_name, cam_str, idx_str = parts
        try:
            cid = int(cam_str.replace("cam", ""))
            idx = int(idx_str)
        except ValueError:
            try: self.send_error(400)
            except BrokenPipeError: pass
            return

        rec_dir = _pathlib.Path(__file__).parent / "recordings" / rec_name
        jpeg = None
        total = 0

        # Try AVI first (cached reader)
        avi = rec_dir / f"cam{cid}.avi"
        if avi.exists() and _PUPIL_OK:
            cap, total, lock = _get_avi(rec_name, cid)
            if cap and lock:
                idx = max(0, min(idx, total - 1))
                with lock:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                if ret:
                    if do_analysis:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        gray = cv2.equalizeHist(gray)
                        center, radius, _ = _detect_pupil(gray, 0)
                        _draw_overlay(frame, center, radius)
                    _, enc = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    jpeg = enc.tobytes()
        else:
            # Fallback: JPEG directory
            jpgdir = rec_dir / f"cam{cid}"
            if jpgdir.is_dir():
                frames = sorted(jpgdir.glob("*.jpg"))
                total = len(frames)
                if 0 <= idx < total:
                    jpeg = frames[idx].read_bytes()
                    if do_analysis and _PUPIL_OK:
                        arr = np.frombuffer(jpeg, np.uint8)
                        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                        if bgr is not None:
                            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                            gray = cv2.equalizeHist(gray)
                            center, radius, _ = _detect_pupil(gray, 0)
                            _draw_overlay(bgr, center, radius)
                            _, enc = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
                            jpeg = enc.tobytes()

        if jpeg is None:
            try: self.send_error(404)
            except BrokenPipeError: pass
            return
        try:
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(jpeg)))
            self.send_header("X-Total-Frames", str(total))
            self.send_header("Access-Control-Expose-Headers", "X-Total-Frames")
            self.end_headers()
            self.wfile.write(jpeg)
        except BrokenPipeError:
            pass

    # ── /set?quality=X&fps=Y&cam=1&analysis=0|1&brightness=0&... ────────────
    # All sensor params mapped to firmware command prefixes
    _PARAM_MAP = {
        "quality":       ("q:",  0,  63),
        "fps":           ("f:",  1,  60),
        "res":           ("r:",  0,  13),
        "brightness":    ("br:", -2,  2),
        "contrast":      ("ct:", -2,  2),
        "saturation":    ("sa:", -2,  2),
        "auto_exposure": ("ae:",  0,  1),
        "exposure":      ("ev:",  0, 1200),
        "auto_gain":     ("ag:",  0,  1),
        "gain":          ("gv:",  0, 30),
        "ae_level":      ("al:", -2,  2),
        "hmirror":       ("hm:",  0,  1),
        "vflip":         ("vf:",  0,  1),
        "wb_mode":       ("wm:",  0,  4),
    }

    def _set_cmd(self, query: str):
        global g_analysis_enabled, g_settings
        import urllib.parse
        params = dict(urllib.parse.parse_qsl(query))
        cmds = []
        if "analysis" in params:
            g_analysis_enabled = params["analysis"] != "0"
        for key, (prefix, lo, hi) in self._PARAM_MAP.items():
            if key in params:
                val = max(lo, min(hi, int(params[key])))
                cmds.append(f"{prefix}{val}".encode())
                g_settings[key] = val

        target_ids = []
        if "cam" in params:
            try: target_ids = [int(params["cam"])]
            except ValueError: pass
        if not target_ids:
            target_ids = [1, 2]

        if cmds:
            for cid in target_ids:
                for cmd in cmds:
                    CAMS[cid].send_cmd(cmd)
                    time.sleep(0.06)  # ESP32 cmdTask polls every 50ms
            _save_settings(g_settings)

        try:
            body = json.dumps({"ok": bool(cmds or "analysis" in params)}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)
        except BrokenPipeError:
            pass

    # ── /jpeg/<n>  (single frame, for canvas-based sync display) ─────────────
    def _jpeg(self, cam: CamState):
        with cam.frame_lock:
            data = cam.latest_frame
        if not data:
            try: self.send_error(503)
            except BrokenPipeError: pass
            return
        if g_analysis_enabled:
            data = _apply_pupil_overlay(data)
        try:
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)
        except BrokenPipeError:
            pass

    # ── /stream/<n>  MJPEG (fallback) ─────────────────────────────────────────
    def _mjpeg(self, cam: CamState):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        try:
            while True:
                cam.frame_event.wait(timeout=2)
                cam.frame_event.clear()
                with cam.frame_lock:
                    data = cam.latest_frame
                if not data:
                    continue
                if g_analysis_enabled:
                    data = _apply_pupil_overlay(data)
                hdr = (b"--frame\r\nContent-Type: image/jpeg\r\n"
                       b"Content-Length: " + str(len(data)).encode() + b"\r\n\r\n")
                self.wfile.write(hdr + data + b"\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass

    # ── / (main page) ─────────────────────────────────────────────────────────
    def _index(self):
        html = r"""<!DOCTYPE html>
<html>
<head>
  <title>ESP32-CAM Dual Stream</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { background: #0d0d0d; color: #ddd; font-family: monospace;
           display: flex; flex-direction: column; align-items: center;
           padding: 14px; gap: 10px; }
    h2 { font-size: 15px; color: #7af; letter-spacing: 2px; margin-top: 4px; }

    .feeds { display: flex; gap: 10px; flex-wrap: wrap;
             justify-content: center; width: 100%; }
    .feed  { flex: 1; min-width: 260px; display: flex;
             flex-direction: column; align-items: center; gap: 5px; }
    .feed-label { font-size: 12px; font-weight: bold; color: #fa0; }
    canvas { width: 100%; border: 2px solid #333; border-radius: 3px;
             background: #111; display: block; }
    canvas.online  { border-color: #3a3; }
    canvas.offline { border-color: #622; }
    .feed-stats { font-size: 11px; color: #8d8; }

    .sync-bar { font-size: 12px; color: #adf;
                background: #1a1a2e; padding: 4px 14px; border-radius: 3px; }

    .ctrl { display: flex; gap: 14px; flex-wrap: wrap;
            justify-content: center; align-items: flex-end; }
    .ctrl-group { display: flex; flex-direction: column; gap: 3px; font-size: 11px; }
    select { background: #1e1e1e; color: #ddd; border: 1px solid #444;
             padding: 2px 4px; border-radius: 3px; }
    input[type=range] { width: 130px; }
    button { padding: 4px 14px; background: #2a5; color: #fff;
             border: none; border-radius: 3px; cursor: pointer; font-size: 11px; }
    button:active { background: #194; }
    #feedback { font-size: 11px; color: #fa0; min-height: 13px; }

  </style>
</head>
<body>
  <h2>ESP32-CAM  ·  DUAL STREAM</h2>

  <div class="feeds">
    <div class="feed">
      <div class="feed-label">CAM 1</div>
      <canvas id="c1" width="320" height="240"></canvas>
      <div class="feed-stats" id="s1">—</div>
    </div>
    <div class="feed">
      <div class="feed-label">CAM 2</div>
      <canvas id="c2" width="320" height="240"></canvas>
      <div class="feed-stats" id="s2">—</div>
    </div>
  </div>

  <div class="sync-bar" id="sync-info">sync: —</div>

  <div class="ctrl">
    <div class="ctrl-group">
      <span>Target</span>
      <select id="target">
        <option value="">Both</option>
        <option value="1">Cam 1</option>
        <option value="2">Cam 2</option>
      </select>
    </div>
    <div class="ctrl-group">
      <span>Quality (lower=better): <b id="qval">12</b></span>
      <input type="range" min="4" max="63" value="12" id="quality"
             oninput="document.getElementById('qval').textContent=this.value">
    </div>
    <div class="ctrl-group">
      <span>FPS: <b id="fval">30</b></span>
      <input type="range" min="1" max="30" value="30" id="fps"
             oninput="document.getElementById('fval').textContent=this.value">
    </div>
    <div class="ctrl-group">
      <span>Resolution</span>
      <select id="res">
        <option value="0">96x96</option>
        <option value="1">QQVGA 160x120</option>
        <option value="2">QCIF 176x144</option>
        <option value="3">HQVGA 240x176</option>
        <option value="4">240x240</option>
        <option value="5" selected>QVGA 320x240</option>
        <option value="6">CIF 400x296</option>
        <option value="7">HVGA 480x320</option>
        <option value="8">VGA 640x480</option>
        <option value="9">SVGA 800x600</option>
        <option value="10">XGA 1024x768</option>
        <option value="11">HD 1280x720</option>
        <option value="12">SXGA 1280x1024</option>
        <option value="13">UXGA 1600x1200</option>
      </select>
    </div>
    <div class="ctrl-group">
      <span>Brightness: <b id="brval">0</b></span>
      <input type="range" min="-2" max="2" value="0" id="brightness"
             oninput="document.getElementById('brval').textContent=this.value">
    </div>
    <div class="ctrl-group">
      <span>Contrast: <b id="ctval">0</b></span>
      <input type="range" min="-2" max="2" value="0" id="contrast"
             oninput="document.getElementById('ctval').textContent=this.value">
    </div>
    <div class="ctrl-group">
      <span>Saturation: <b id="saval">0</b></span>
      <input type="range" min="-2" max="2" value="0" id="saturation"
             oninput="document.getElementById('saval').textContent=this.value">
    </div>
    <div class="ctrl-group">
      <span>AE Level: <b id="alval">0</b></span>
      <input type="range" min="-2" max="2" value="0" id="ae_level"
             oninput="document.getElementById('alval').textContent=this.value">
    </div>
    <div class="ctrl-group">
      <span>Auto Exposure</span>
      <select id="auto_exposure">
        <option value="1" selected>On</option>
        <option value="0">Off</option>
      </select>
    </div>
    <div class="ctrl-group">
      <span>Exposure: <b id="evval">300</b></span>
      <input type="range" min="0" max="1200" value="300" id="exposure"
             oninput="document.getElementById('evval').textContent=this.value">
    </div>
    <div class="ctrl-group">
      <span>Auto Gain</span>
      <select id="auto_gain">
        <option value="1" selected>On</option>
        <option value="0">Off</option>
      </select>
    </div>
    <div class="ctrl-group">
      <span>Gain: <b id="gvval">0</b></span>
      <input type="range" min="0" max="30" value="0" id="gain"
             oninput="document.getElementById('gvval').textContent=this.value">
    </div>
    <div class="ctrl-group">
      <span>WB Mode</span>
      <select id="wb_mode">
        <option value="0" selected>Auto</option>
        <option value="1">Sunny</option>
        <option value="2">Cloudy</option>
        <option value="3">Office</option>
        <option value="4">Home</option>
      </select>
    </div>
    <div class="ctrl-group">
      <span>H-Mirror</span>
      <select id="hmirror">
        <option value="0" selected>Off</option>
        <option value="1">On</option>
      </select>
    </div>
    <div class="ctrl-group">
      <span>V-Flip</span>
      <select id="vflip">
        <option value="0" selected>Off</option>
        <option value="1">On</option>
      </select>
    </div>
    <button onclick="applySettings()">Apply</button>
    <button onclick="saveRecording()" style="background:#c33">Save 2min Buffer</button>
    <button onclick="window.open('/player','_blank')" style="background:#669">Player</button>
  </div>
  <div id="feedback"></div>

<script>
// ── Synchronized canvas display ───────────────────────────────────────────────
// Both frames are fetched simultaneously, decoded in parallel, then drawn in
// the same requestAnimationFrame callback — same vsync = true display sync.

const c1  = document.getElementById('c1');
const c2  = document.getElementById('c2');
const ctx1 = c1.getContext('2d');
const ctx2 = c2.getContext('2d');

let displayFps = 20;   // how often we poll for new frames (Hz)
let running    = true;

async function fetchBitmap(url) {
  const r = await fetch(url + '?t=' + Date.now());
  if (!r.ok) return null;
  const blob = await r.blob();
  return createImageBitmap(blob);
}

function drawBitmap(canvas, ctx, bmp) {
  if (!bmp) return;
  canvas.width  = bmp.width;
  canvas.height = bmp.height;
  ctx.drawImage(bmp, 0, 0);
  bmp.close();
}

async function loop() {
  if (!running) return;
  const start = performance.now();

  try {
    // Fetch both frames at the exact same time
    const [bmp1, bmp2] = await Promise.all([
      fetchBitmap('/jpeg/1'),
      fetchBitmap('/jpeg/2'),
    ]);

    // Draw both in one RAF — guaranteed same browser frame (vsync)
    requestAnimationFrame(() => {
      drawBitmap(c1, ctx1, bmp1);
      drawBitmap(c2, ctx2, bmp2);
    });
  } catch (_) {}

  const elapsed = performance.now() - start;
  const wait    = Math.max(0, 1000 / displayFps - elapsed);
  setTimeout(loop, wait);
}

loop();

// ── Stats polling ─────────────────────────────────────────────────────────────
function fmtCam(s) {
  if (!s.online) return 'offline';
  return `FPS ${s.fps.toFixed(1)}  lat ${s.latency_ms.toFixed(0)}ms  drop ${s.dropped}`;
}

setInterval(() => {
  fetch('/stats').then(r => r.json()).then(d => {
    document.getElementById('s1').textContent = fmtCam(d.cam1);
    document.getElementById('s2').textContent = fmtCam(d.cam2);
    c1.className = d.cam1.online ? 'online' : 'offline';
    c2.className = d.cam2.online ? 'online' : 'offline';

    const off = d.sync_offset_ms;
    const color = off < 20 ? '#5f5' : off < 50 ? '#fa0' : '#f55';
    document.getElementById('sync-info').innerHTML =
      `sync offset: <span style="color:${color}">${off.toFixed(1)} ms</span>` +
      `  &nbsp;(NTP-based timestamp delta between cameras)`;
  }).catch(() => {});
}, 1000);

// ── Controls ──────────────────────────────────────────────────────────────────
const ALL_KEYS = ['quality','fps','res','brightness','contrast','saturation',
  'ae_level','auto_exposure','exposure','auto_gain','gain','wb_mode','hmirror','vflip'];
const LABEL_MAP = {quality:'qval',fps:'fval',brightness:'brval',contrast:'ctval',
  saturation:'saval',ae_level:'alval',exposure:'evval',gain:'gvval'};

function applySettings() {
  const cam = document.getElementById('target').value;
  const fb  = document.getElementById('feedback');
  fb.textContent = 'Sending...';
  let qs = ALL_KEYS.map(k => k + '=' + document.getElementById(k).value).join('&');
  if (cam) qs += '&cam=' + cam;
  fetch('/set?' + qs).then(r => r.json()).then(d => {
    fb.textContent = d.ok ? 'Applied to ' + (cam ? 'cam'+cam : 'both') : 'Send failed';
  }).catch(e => { fb.textContent = 'Error: ' + e; });
}

// Load saved settings on page load
fetch('/settings').then(r => r.json()).then(s => {
  for (const k of ALL_KEYS) {
    const el = document.getElementById(k);
    if (el && s[k] !== undefined) {
      el.value = s[k];
      if (LABEL_MAP[k]) document.getElementById(LABEL_MAP[k]).textContent = s[k];
    }
  }
}).catch(() => {});

function saveRecording() {
  const fb = document.getElementById('feedback');
  fb.textContent = 'Saving 2-min buffer...';
  fetch('/record/save').then(r => r.json()).then(d => {
    fb.textContent = d.ok ? 'Saved to recordings/' : 'Save failed';
  }).catch(e => { fb.textContent = 'Error: ' + e; });
}


</script>
</body>
</html>"""
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(html.encode())


    # ── /player — dedicated recording playback page ─────────────────────────
    def _player(self):
        html = r"""<!DOCTYPE html>
<html>
<head>
  <title>NETR Player</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { background: #0a0a0a; color: #ddd; font-family: monospace;
           display: flex; flex-direction: column; height: 100vh; padding: 10px; }
    .top-bar { display: flex; gap: 10px; align-items: center; flex-wrap: wrap;
               padding: 8px 0; border-bottom: 1px solid #333; margin-bottom: 8px; }
    .top-bar h2 { font-size: 14px; color: #fa0; margin-right: 10px; }
    .top-bar select, .top-bar button, .top-bar label {
      font-size: 12px; font-family: monospace; }
    .top-bar select { background: #1e1e1e; color: #ddd; border: 1px solid #444;
                      padding: 3px 6px; border-radius: 3px; }
    .top-bar button { padding: 4px 12px; border: none; border-radius: 3px;
                      cursor: pointer; color: #fff; }
    .btn-load { background: #2a5; }
    .btn-play { background: #47a; min-width: 60px; }
    .btn-step { background: #555; }
    .btn-speed { background: #444; font-size: 11px !important; padding: 3px 8px !important; }
    .btn-speed.active { background: #f80; color: #000; }
    label { color: #adf; cursor: pointer; user-select: none; }
    label input { margin-right: 4px; }
    .info { font-size: 11px; color: #8d8; margin-left: auto; }

    .slider-row { padding: 4px 0 8px 0; }
    .slider-row input { width: 100%; }

    .viewer { flex: 1; display: flex; gap: 8px; min-height: 0; }
    .cam-panel { flex: 1; display: flex; flex-direction: column; align-items: center;
                 min-width: 0; }
    .cam-label { font-size: 11px; color: #fa0; margin-bottom: 4px; font-weight: bold; }
    .cam-panel canvas { width: 100%; flex: 1; object-fit: contain;
                        background: #111; border: 1px solid #333; border-radius: 3px; }

    .status-bar { font-size: 11px; color: #777; padding: 6px 0 0 0;
                  border-top: 1px solid #222; margin-top: 8px; text-align: center; }
    .kb { font-size: 10px; color: #555; margin-top: 4px; text-align: center; }
  </style>
</head>
<body>
  <div class="top-bar">
    <h2>NETR Player</h2>
    <select id="rec"><option value="">— select recording —</option></select>
    <button class="btn-load" onclick="loadRec()">Load</button>
    <button class="btn-step" onclick="step(-10)">-10</button>
    <button class="btn-step" onclick="step(-1)">&lt;</button>
    <button class="btn-play" onclick="togglePlay()" id="btn-play">Play</button>
    <button class="btn-step" onclick="step(1)">&gt;</button>
    <button class="btn-step" onclick="step(10)">+10</button>
    <span style="color:#555">|</span>
    <button class="btn-speed" data-speed="0.25">0.25x</button>
    <button class="btn-speed" data-speed="0.5">0.5x</button>
    <button class="btn-speed active" data-speed="1">1x</button>
    <button class="btn-speed" data-speed="2">2x</button>
    <label><input type="checkbox" id="analysis" checked> Analysis</label>
    <span class="info" id="info">No recording loaded</span>
  </div>

  <div class="slider-row">
    <input type="range" min="0" max="0" value="0" id="slider"
           oninput="seek(+this.value)">
  </div>

  <div class="viewer">
    <div class="cam-panel">
      <div class="cam-label">CAM 1</div>
      <canvas id="c1"></canvas>
    </div>
    <div class="cam-panel">
      <div class="cam-label">CAM 2</div>
      <canvas id="c2"></canvas>
    </div>
  </div>

  <div class="status-bar" id="status">—</div>
  <div class="kb">Space: play/pause &nbsp; Left/Right: step &nbsp; Shift+Left/Right: skip 10 &nbsp; A: toggle analysis &nbsp; 1-4: speed</div>

<script>
const c1 = document.getElementById('c1');
const c2 = document.getElementById('c2');
const ctx1 = c1.getContext('2d');
const ctx2 = c2.getContext('2d');

let rec = null;
let total = {1: 0, 2: 0};
let idx = 0;
let playing = false;
let speed = 1;
let rendering = false;

// Refresh recordings list
fetch('/recordings').then(r => r.json()).then(recs => {
  const sel = document.getElementById('rec');
  recs.forEach(r => {
    const cams = Object.keys(r.cams).map(c => 'cam'+c).join('+');
    sel.innerHTML += `<option value="${r.name}">${r.name} (${cams})</option>`;
  });
}).catch(() => {});

// Speed buttons
document.querySelectorAll('.btn-speed').forEach(btn => {
  btn.onclick = () => {
    document.querySelectorAll('.btn-speed').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    speed = parseFloat(btn.dataset.speed);
  };
});

async function loadRec() {
  const name = document.getElementById('rec').value;
  if (!name) return;
  rec = name;
  idx = 0;
  stop();
  document.getElementById('status').textContent = 'Loading...';
  try {
    const info = await fetch(`/recordings/info/${rec}`).then(r => r.json());
    total[1] = info['1'] || 0;
    total[2] = info['2'] || 0;
  } catch(_) { total[1] = 0; total[2] = 0; }
  const slider = document.getElementById('slider');
  const maxF = Math.max(total[1], total[2]);
  slider.max = Math.max(0, maxF - 1);
  slider.value = 0;
  document.getElementById('status').textContent =
    `${rec}: cam1=${total[1]} cam2=${total[2]} frames`;
  render();
}

async function render() {
  if (!rec || rendering) return;
  rendering = true;
  const ana = document.getElementById('analysis').checked ? '1' : '0';
  const maxF = Math.max(total[1], total[2]);
  document.getElementById('info').textContent =
    `Frame ${idx} / ${maxF - 1}`;
  document.getElementById('slider').value = idx;

  const fetches = [1, 2].map(cid => {
    if (idx >= total[cid] || total[cid] === 0) return Promise.resolve();
    // Clamp to last frame for shorter cam
    const fi = Math.min(idx, total[cid] - 1);
    return fetch(`/playback/${rec}/cam${cid}/${fi}?analysis=${ana}`)
      .then(r => { if (!r.ok) throw new Error(r.status); return r.blob(); })
      .then(b => createImageBitmap(b))
      .then(bmp => {
        const canvas = cid === 1 ? c1 : c2;
        const ctx = cid === 1 ? ctx1 : ctx2;
        canvas.width = bmp.width;
        canvas.height = bmp.height;
        ctx.drawImage(bmp, 0, 0);
        bmp.close();
      })
      .catch(e => {
        document.getElementById('status').textContent = `cam${cid} frame ${fi}: ${e}`;
      });
  });
  await Promise.all(fetches);
  rendering = false;
}

function step(delta) {
  if (!rec) return;
  const maxF = Math.max(total[1], total[2]);
  idx = Math.max(0, Math.min(idx + delta, maxF - 1));
  render();
}

function seek(val) {
  if (!rec) return;
  idx = val;
  render();
}

function stop() {
  playing = false;
  document.getElementById('btn-play').textContent = 'Play';
}

function togglePlay() {
  if (!rec) return;
  if (playing) { stop(); return; }
  playing = true;
  document.getElementById('btn-play').textContent = 'Pause';
  playLoop();
}

async function playLoop() {
  while (playing) {
    const maxF = Math.max(total[1], total[2]);
    if (idx >= maxF - 1) { stop(); return; }
    idx++;
    const t0 = performance.now();
    await render();
    const elapsed = performance.now() - t0;
    const target = (1000 / 30) / speed;
    const wait = Math.max(0, target - elapsed);
    if (wait > 0) await new Promise(r => setTimeout(r, wait));
  }
}

// Keyboard shortcuts
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'SELECT') return;
  switch(e.code) {
    case 'Space':      e.preventDefault(); togglePlay(); break;
    case 'ArrowLeft':  e.preventDefault(); step(e.shiftKey ? -10 : -1); break;
    case 'ArrowRight': e.preventDefault(); step(e.shiftKey ? 10 : 1); break;
    case 'KeyA':       document.getElementById('analysis').click(); break;
    case 'Digit1':     document.querySelectorAll('.btn-speed')[0].click(); break;
    case 'Digit2':     document.querySelectorAll('.btn-speed')[1].click(); break;
    case 'Digit3':     document.querySelectorAll('.btn-speed')[2].click(); break;
    case 'Digit4':     document.querySelectorAll('.btn-speed')[3].click(); break;
  }
});
</script>
</body>
</html>"""
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(html.encode())


class QuietHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address):
        import sys
        if issubclass(sys.exc_info()[0], BrokenPipeError):
            return
        super().handle_error(request, client_address)


# ── UDP receiver per camera ────────────────────────────────────────────────────
def udp_serve(cam: CamState):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
    sock.bind(("0.0.0.0", cam.udp_port))
    print(f"cam{cam.cam_id}: UDP :{cam.udp_port}")

    while True:
        pkt, _ = sock.recvfrom(65535)
        cam.ingest(pkt)


# ── Sync offset tracker ────────────────────────────────────────────────────────
def sync_tracker():
    """Periodically compute |ts1 - ts2| between the latest frames."""
    global sync_offset_ms
    while True:
        time.sleep(0.5)
        ts1 = CAMS[1].stats["last_ts_us"]
        ts2 = CAMS[2].stats["last_ts_us"]
        if ts1 and ts2:
            sync_offset_ms = abs(ts1 - ts2) / 1000.0


# ── Offline watchdog ──────────────────────────────────────────────────────────
def watchdog():
    last_fps = {1: 0.0, 2: 0.0}
    stale    = {1: 0,   2: 0}
    while True:
        time.sleep(1)
        for cid, cam in CAMS.items():
            fps = cam.stats["fps"]
            if fps == last_fps[cid]:
                stale[cid] += 1
            else:
                stale[cid] = 0
            last_fps[cid] = fps
            if stale[cid] >= 3 and cam.stats["online"]:
                cam.stats["online"] = False
                print(f"cam{cid}: offline")


# ── Beacon ────────────────────────────────────────────────────────────────────
def _own_ip() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]


def timesync_server():
    """
    NTP-style round-trip responder on UDP 5005.
    Receives: SYNC_REQ:<cam_id>:<T1_mono_us>
    Sends:    SYNC_RESP:<T1_mono_us>:<T2_us>:<T3_us>
    T2 = receive wall time, T3 = send wall time (both in Unix µs).
    ESP32 computes: corrected_time = T3 + (T4_mono - T1_mono) / 2
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", TIMESYNC_PORT))
    while True:
        try:
            data, addr = sock.recvfrom(128)
            t2 = int(time.time() * 1e6)          # record receive time immediately
            msg = data.decode(errors="ignore").strip()
            if msg.startswith("SYNC_REQ:"):
                # "SYNC_REQ:<cam_id>:<T1_mono>"
                parts = msg[9:].split(":", 1)
                if len(parts) == 2:
                    t1_mono = parts[1]
                    t3 = int(time.time() * 1e6)  # record send time just before sending
                    sock.sendto(f"SYNC_RESP:{t1_mono}:{t2}:{t3}".encode(), addr)
        except Exception:
            continue


def _push_settings(cid: int):
    """Send all saved settings to a camera on discovery."""
    param_map = MJPEGHandler._PARAM_MAP
    for key, (prefix, lo, hi) in param_map.items():
        if key in g_settings:
            val = max(lo, min(hi, int(g_settings[key])))
            CAMS[cid].send_cmd(f"{prefix}{val}".encode())
            time.sleep(0.02)  # small gap so ESP32 processes each command


def beacon_thread():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", DISCOVERY_PORT))
    sock.settimeout(1.0)

    own_ip = _own_ip()
    print(f"Beacon: LAPTOP:{own_ip}  UDP:{DISCOVERY_PORT}")

    while True:
        # Include current timestamp so ESP32s sync their clock from the beacon
        msg = f"LAPTOP:{own_ip}:{int(time.time() * 1e6):.0f}".encode()
        sock.sendto(msg, ("255.255.255.255", DISCOVERY_PORT))
        try:
            data, addr = sock.recvfrom(64)
            txt = data.decode(errors="ignore").strip()
            if txt.startswith("CAM:"):
                try:
                    cid = int(txt[4:])
                    if cid in CAMS and CAMS[cid].ip != addr[0]:
                        CAMS[cid].ip = addr[0]
                        print(f"cam{cid} discovered: {addr[0]}")
                        _push_settings(cid)
                except ValueError:
                    pass
        except socket.timeout:
            pass
        time.sleep(4)


if __name__ == "__main__":
    print(f"Browser → http://localhost:{HTTP_PORT}")
    threading.Thread(target=QuietHTTPServer(("0.0.0.0", HTTP_PORT), MJPEGHandler).serve_forever,
                     daemon=True).start()
    threading.Thread(target=beacon_thread,   daemon=True).start()
    threading.Thread(target=timesync_server, daemon=True).start()
    threading.Thread(target=sync_tracker,    daemon=True).start()
    threading.Thread(target=watchdog,     daemon=True).start()
    threading.Thread(target=udp_serve, args=(CAMS[1],), daemon=True).start()
    udp_serve(CAMS[2])
