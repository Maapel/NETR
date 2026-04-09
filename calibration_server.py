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
SCENE_CAM_ID  = _WORLD_CAM     # which cam to use for ArUco detection (world cam from rig_config)
ARUCO_DICT    = cv2.aruco.DICT_4X4_50
ARUCO_IDS     = [0, 1, 2, 3]   # TL, TR, BL, BR order
SYNC_WINDOW_MS = 20         # max ms between target coord and eye vector
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

# Saccade mode: directly collected fixation samples {dx, dy, X, Y, sx, sy}
_saccade_samples: list[dict] = []
_saccade_lock = threading.Lock()

_model = GazeModel()                          # scene-space model (saved to disk)
_screen_model = GazeModel()                   # screen-space model (for live cursor)
SCREEN_MODEL_PATH = pathlib.Path(__file__).parent / "screen_model.json"
_model.load(MODEL_PATH)
_screen_model.load(SCREEN_MODEL_PATH)

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
_aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
_aruco_params = cv2.aruco.DetectorParameters()
_aruco_detector = cv2.aruco.ArucoDetector(_aruco_dict, _aruco_params)

# Cached homography (recomputed when scene frame updates)
_homography: np.ndarray | None = None
_homography_lock = threading.Lock()


def _detect_aruco_corners(frame_bgr: np.ndarray) -> dict[int, np.ndarray] | None:
    """Detect ArUco markers, return dict id->center or None if <4 found."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = _aruco_detector.detectMarkers(gray)
    if ids is None or len(ids) < 4:
        return None
    result = {}
    for i, mid in enumerate(ids.flatten()):
        if mid in ARUCO_IDS:
            c = corners[i][0]
            result[int(mid)] = c.mean(axis=0)  # marker center
    if len(result) < 4:
        return None
    return result


def _compute_homography(scene_corners: dict[int, np.ndarray],
                        screen_w: int, screen_h: int) -> np.ndarray | None:
    """
    Compute homography from screen space → scene camera space.
    ArUco IDs 0,1,2,3 = TL, TR, BL, BR.
    """
    margin = 0.07  # ArUco markers drawn at 7% inset from screen edges
    mx, my = screen_w * margin, screen_h * margin
    # Screen positions of the four ArUco marker centres (same layout as webapp)
    screen_pts = np.array([
        [mx,             my],              # 0 TL
        [screen_w - mx,  my],              # 1 TR
        [mx,             screen_h - my],   # 2 BL
        [screen_w - mx,  screen_h - my],   # 3 BR
    ], dtype=np.float32)
    scene_pts = np.array([
        scene_corners[0],
        scene_corners[1],
        scene_corners[2],
        scene_corners[3],
    ], dtype=np.float32)
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
    global _homography
    snap_url = f"{RECEIVER_URL}/jpeg/{SCENE_CAM_ID}"
    while True:
        try:
            with urllib.request.urlopen(snap_url, timeout=2) as resp:
                data = resp.read()
            arr = np.frombuffer(data, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is not None:
                detected = _detect_aruco_corners(frame)
                if detected:
                    # Need screen dimensions — use last known from browser
                    sw, sh = _last_screen_size
                    if sw and sh:
                        H = _compute_homography(detected, sw, sh)
                        with _homography_lock:
                            _homography = H
        except Exception:
            pass
        time.sleep(0.5)

_last_screen_size = [0, 0]   # updated by websocket messages


# ── Eye vector poller ─────────────────────────────────────────────────────────
def _eye_poll_thread():
    """Poll receiver /stats for live pccr_vector and buffer it."""
    stats_url = f"{RECEIVER_URL}/stats"
    while True:
        if _calibrating:
            try:
                with urllib.request.urlopen(stats_url, timeout=1) as resp:
                    d = json.loads(resp.read())
                vec = d.get("pccr_vector")
                if vec and len(vec) == 2:
                    ts = time.time() * 1000  # ms
                    with _eye_lock:
                        _eye_buf.append({"ts": ts, "dx": vec[0], "dy": vec[1]})
                        # Keep last 10 seconds
                        cutoff = ts - 10000
                        while _eye_buf and _eye_buf[0]["ts"] < cutoff:
                            _eye_buf.pop(0)
            except Exception:
                pass
        time.sleep(0.033)  # ~30 Hz


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

    samples = []
    ei = 0
    for t in targets:
        # Binary-search for closest eye sample
        best = None
        best_dt = float("inf")
        # Walk forward until we pass the window
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
            continue

        scene_xy = _screen_to_scene(t["x"], t["y"])
        if scene_xy is None:
            continue
        X, Y = scene_xy
        samples.append({"dx": best["dx"], "dy": best["dy"], "X": X, "Y": Y})

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
    global _calibrating
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

            elif mtype == "start":
                global _calib_mode
                _calibrating = True
                _calib_mode  = msg.get("mode", "sweep")
                with _target_lock:
                    _target_buf.clear()
                with _eye_lock:
                    _eye_buf.clear()
                with _saccade_lock:
                    _saccade_samples.clear()
                _ws_send_frame(wfile, json.dumps({"type": "ack", "msg": f"calibration started ({_calib_mode})"}))

            elif mtype == "target":
                if _calibrating:
                    with _target_lock:
                        _target_buf.append({
                            "ts": msg["ts"],
                            "x":  msg["x"],
                            "y":  msg["y"],
                        })

            elif mtype == "fixation":
                # Saccade mode: eye has settled on this point — find closest eye vector
                if _calibrating:
                    ts = msg["ts"]
                    with _eye_lock:
                        eyes = list(_eye_buf)
                    best = min(eyes, key=lambda e: abs(e["ts"] - ts), default=None)
                    if best and abs(best["ts"] - ts) <= SYNC_WINDOW_MS * 3:
                        scene_xy = _screen_to_scene(msg["x"], msg["y"])
                        if scene_xy:
                            X, Y = scene_xy
                            with _saccade_lock:
                                _saccade_samples.append({
                                    "dx": best["dx"], "dy": best["dy"],
                                    "X": X, "Y": Y,
                                    "sx": msg["x"], "sy": msg["y"],  # screen coords for live cursor
                                })
                            n = len(_saccade_samples)
                            diag = _refit_models()
                            if diag:
                                _broadcast(json.dumps({
                                    "type": "ready",
                                    "n": n,
                                    "r2_x": round(diag["r2_x"], 3),
                                    "r2_y": round(diag["r2_y"], 3),
                                }))

            elif mtype == "stop":
                _calibrating = False
                if _calib_mode == "saccade":
                    with _saccade_lock:
                        samples = list(_saccade_samples)
                else:
                    sw, sh = _last_screen_size
                    samples = _sync_and_build_dataset(sw, sh)
                result = {"type": "result", "n_samples": len(samples)}
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
button.live-on { background: #1a3320; color: #44ff88; border-color: #44ff88; }
#status {
  color: #888; font-size: 13px; min-width: 320px; text-align: center;
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
  <span id="status">Connecting…</span>
</div>
<canvas id="c"></canvas>
<script>
const canvas = document.getElementById('c');
const ctx = canvas.getContext('2d');
const statusEl  = document.getElementById('status');
const btnStart  = document.getElementById('btnStart');
const btnStop   = document.getElementById('btnStop');
const btnLive   = document.getElementById('btnLive');
const btnSweep  = document.getElementById('btnSweep');
const btnSaccade= document.getElementById('btnSaccade');

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
};
ws.onclose = () => statusEl.textContent = 'Disconnected';
ws.onmessage = e => {
  const m = JSON.parse(e.data);
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
    if (m.ok) {
      statusEl.textContent = `Done! n=${m.n_samples}  R²x=${m.r2_x.toFixed(3)}  R²y=${m.r2_y.toFixed(3)}`;
    } else {
      statusEl.textContent = `Failed: ${m.error}`;
    }
  }
};

// ── ArUco corner markers (drawn as patterns) ─────────────────────────────────
// Render simple ArUco-style markers (4x4 bit patterns for IDs 0-3) in corners.
// These match the DICT_4X4_50 patterns so OpenCV can detect them.
// Bits read MSB-first, row by row (4x4 inner grid + 1px black border = 6x6 cells).
const ARUCO_BITS = {
  0: [[0,1,1,0],[1,1,0,1],[0,0,1,1],[0,0,0,0]],
  1: [[1,1,0,1],[1,0,0,0],[0,0,0,1],[1,0,1,1]],
  2: [[1,1,1,0],[0,1,1,1],[0,0,1,1],[0,1,1,0]],
  3: [[1,0,1,0],[0,0,0,0],[1,0,0,1],[0,1,1,1]],
};


// ── Offscreen background (ArUco corners, redrawn only on resize) ─────────────
let bgCanvas = null;

function buildBackground() {
  bgCanvas = document.createElement('canvas');
  bgCanvas.width  = W;
  bgCanvas.height = H;
  const bctx = bgCanvas.getContext('2d');
  bctx.fillStyle = '#111';
  bctx.fillRect(0, 0, W, H);
  const M = Math.min(W, H) * 0.07;
  const S = Math.min(W, H) * 0.10;
  // Reuse drawAruco but on bctx — swap ctx temporarily
  const saved = ctx;
  // Draw directly using bctx
  function bDrawAruco(id, cx, cy, size) {
    const cell = size / 6;
    bctx.fillStyle = '#fff';
    bctx.fillRect(cx - size/2, cy - size/2, size, size);
    bctx.fillStyle = '#000';
    bctx.fillRect(cx - size/2, cy - size/2, size, cell);
    bctx.fillRect(cx - size/2, cy + size/2 - cell, size, cell);
    bctx.fillRect(cx - size/2, cy - size/2, cell, size);
    bctx.fillRect(cx + size/2 - cell, cy - size/2, cell, size);
    const bits = ARUCO_BITS[id];
    for (let r = 0; r < 4; r++) {
      for (let c = 0; c < 4; c++) {
        bctx.fillStyle = bits[r][c] === 0 ? '#000' : '#fff';
        bctx.fillRect(cx - size/2 + (c+1)*cell, cy - size/2 + (r+1)*cell, cell, cell);
      }
    }
  }
  bDrawAruco(0, M,     M,     S);
  bDrawAruco(1, W-M,   M,     S);
  bDrawAruco(2, M,     H-M,   S);
  bDrawAruco(3, W-M,   H-M,   S);
}
buildBackground();
window.addEventListener('resize', () => { resize(); buildBackground(); });

// ── Sweep mode (reading lines) ────────────────────────────────────────────────
const TARGET_R = 12;
const LINES    = 7;
const LINE_MS  = 5000;
const TOTAL_MS = LINES * LINE_MS;

function readingPos(t) {
  const pad  = Math.min(W, H) * 0.15;
  const xMin = pad, xMax = W - pad;
  const yMin = pad, yMax = H - pad;
  const cycle   = t % TOTAL_MS;
  const lineIdx = Math.floor(cycle / LINE_MS);
  const lineT   = (cycle % LINE_MS) / LINE_MS;
  return {
    x: xMin + lineT * (xMax - xMin),
    y: yMin + (lineIdx / (LINES - 1)) * (yMax - yMin),
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
  const pad   = Math.min(W, H) * 0.12;
  const cellW = (W - 2*pad) / 3;
  const cellH = (H - 2*pad) / 3;
  const inset = Math.min(cellW, cellH) * 0.15;
  const col   = ZONE_POS[zoneIdx][1];
  const row   = ZONE_POS[zoneIdx][0];
  // Center zone snaps to exact screen center for the "true zero" baseline
  if (zoneIdx === CENTER_ZONE) return { x: W / 2, y: H / 2 };
  return {
    x: pad + col * cellW + inset + Math.random() * (cellW - 2*inset),
    y: pad + row * cellH + inset + Math.random() * (cellH - 2*inset),
  };
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
</script>
</body>
</html>"""


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

        else:
            self.send_response(404); self.end_headers()


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


def main():
    threading.Thread(target=_scene_cam_thread, daemon=True).start()
    threading.Thread(target=_eye_poll_thread,  daemon=True).start()

    server = ThreadedHTTPServer(("", 8090), Handler)
    print("Calibration server on http://localhost:8090")
    print(f"Scene cam: cam{SCENE_CAM_ID} (world cam from rig_config)")
    print(f"Scene cam: cam{SCENE_CAM_ID} via {RECEIVER_URL}")
    print(f"Model path: {MODEL_PATH}")
    if _model.trained:
        print("Existing gaze model loaded.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
