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
import time
import threading
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

HTTP_PORT      = 8080
DISCOVERY_PORT = 5004
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

    # ── /set?quality=X&fps=Y&cam=1 ───────────────────────────────────────────
    def _set_cmd(self, query: str):
        import urllib.parse
        params = dict(urllib.parse.parse_qsl(query))
        cmds = []
        if "quality" in params:
            cmds.append(f"q:{max(0, min(63, int(params['quality'])))}" .encode())
        if "fps" in params:
            cmds.append(f"f:{max(1, min(60, int(params['fps'])))}" .encode())

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

        try:
            body = json.dumps({"ok": bool(cmds)}).encode()
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
    <button onclick="applySettings()">Apply</button>
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
function applySettings() {
  const q   = document.getElementById('quality').value;
  const f   = document.getElementById('fps').value;
  const cam = document.getElementById('target').value;
  const fb  = document.getElementById('feedback');
  fb.textContent = 'Sending...';
  const camParam = cam ? '&cam=' + cam : '';
  Promise.all([
    fetch('/set?quality=' + q + camParam).then(r => r.json()),
    fetch('/set?fps='     + f + camParam).then(r => r.json()),
  ]).then(([rq, rf]) => {
    const who = cam ? 'cam' + cam : 'both';
    fb.textContent = (rq.ok && rf.ok)
      ? `Applied to ${who} — quality=${q}  fps=${f}`
      : 'Send failed';
  }).catch(e => { fb.textContent = 'Error: ' + e; });
}
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


def beacon_thread():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", DISCOVERY_PORT))
    sock.settimeout(1.0)

    own_ip = _own_ip()
    msg = f"LAPTOP:{own_ip}".encode()
    print(f"Beacon: {msg.decode()}  UDP:{DISCOVERY_PORT}")

    while True:
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
                except ValueError:
                    pass
        except socket.timeout:
            pass
        time.sleep(4)


if __name__ == "__main__":
    print(f"Browser → http://localhost:{HTTP_PORT}")
    threading.Thread(target=QuietHTTPServer(("0.0.0.0", HTTP_PORT), MJPEGHandler).serve_forever,
                     daemon=True).start()
    threading.Thread(target=beacon_thread, daemon=True).start()
    threading.Thread(target=sync_tracker, daemon=True).start()
    threading.Thread(target=watchdog,     daemon=True).start()
    threading.Thread(target=udp_serve, args=(CAMS[1],), daemon=True).start()
    udp_serve(CAMS[2])
