"""
Dual-camera UDP receiver with browser-based MJPEG viewer.

Packet format (little-endian):
  [4B frame_id][2B chunk_idx][2B total_chunks][8B timestamp_us][JPEG chunk]

Ports:
  cam1 stream → 5000   cam1 cmd → 5001
  cam2 stream → 5002   cam2 cmd → 5003
  discovery   → 5004

Open http://localhost:8080 in your browser.
Press Ctrl+C to quit.
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
FRAME_TIMEOUT  = 0.25   # discard incomplete frames older than this (s)

HDR      = struct.Struct("<IHHq")   # frame_id, chunk_idx, total_chunks, ts_us
HDR_SIZE = HDR.size                  # 16 bytes


# ── Per-camera state ───────────────────────────────────────────────────────────
class CamState:
    def __init__(self, cam_id: int):
        self.cam_id   = cam_id
        self.udp_port = 5000 + (cam_id - 1) * 2
        self.cmd_port = 5001 + (cam_id - 1) * 2
        self.ip       = ""          # discovered via beacon

        self._frames: dict = {}
        self._frames_lock  = threading.Lock()

        self.latest_frame: bytes = b""
        self.frame_lock  = threading.Lock()
        self.frame_event = threading.Event()

        self.stats = {"fps": 0.0, "latency_ms": 0.0, "dropped": 0, "online": False}

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
        if len(pkt) < HDR_SIZE:
            return
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

            if len(entry["chunks"]) == entry["total"]:
                jpeg = b"".join(entry["chunks"][i] for i in range(entry["total"]))
                del self._frames[frame_id]
                with self.frame_lock:
                    self.latest_frame = jpeg
                self.frame_event.set()

                now_us = int(time.time() * 1e6)
                self.stats["latency_ms"] = (now_us - entry["ts"]) / 1000.0
                return jpeg
        return None


CAMS = {1: CamState(1), 2: CamState(2)}


# ── HTTP server ────────────────────────────────────────────────────────────────
class MJPEGHandler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/":
            self._index()
        elif path == "/stream/1":
            self._mjpeg(CAMS[1])
        elif path == "/stream/2":
            self._mjpeg(CAMS[2])
        elif path == "/stats":
            self._stats()
        elif self.path.startswith("/set?"):
            self._set_cmd(self.path[5:])
        else:
            try:
                self.send_error(404)
            except BrokenPipeError:
                pass

    def _stats(self):
        body = json.dumps({
            "cam1": CAMS[1].stats,
            "cam2": CAMS[2].stats,
        }).encode()
        try:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except BrokenPipeError:
            pass

    def _set_cmd(self, query: str):
        import urllib.parse
        params = dict(urllib.parse.parse_qsl(query))

        cmds = []
        if "quality" in params:
            val = max(0, min(63, int(params["quality"])))
            cmds.append(f"q:{val}".encode())
        if "fps" in params:
            val = max(1, min(60, int(params["fps"])))
            cmds.append(f"f:{val}".encode())

        # cam=1, cam=2, or both if not specified
        target_ids = []
        if "cam" in params:
            try:
                target_ids = [int(params["cam"])]
            except ValueError:
                pass
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

    def _index(self):
        html = """<!DOCTYPE html>
<html>
<head>
  <title>ESP32-CAM Dual Stream</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { background: #111; color: #eee; font-family: monospace;
           display: flex; flex-direction: column; align-items: center;
           padding: 16px; gap: 12px; min-height: 100vh; }
    h2 { font-size: 16px; color: #8cf; letter-spacing: 1px; }

    .feeds { display: flex; gap: 12px; flex-wrap: wrap; justify-content: center; width: 100%; }
    .feed  { display: flex; flex-direction: column; align-items: center; gap: 6px; flex: 1; min-width: 280px; }
    .feed-label { font-size: 13px; color: #fa0; font-weight: bold; }
    .feed img  { width: 100%; border: 2px solid #333; border-radius: 4px; }
    .feed img.online  { border-color: #4a4; }
    .feed img.offline { border-color: #622; }
    .feed-stats { font-size: 12px; color: #8f8; }

    .ctrl { display: flex; gap: 16px; flex-wrap: wrap; justify-content: center; align-items: flex-end; }
    .ctrl-group { display: flex; flex-direction: column; gap: 4px; font-size: 12px; }
    .ctrl-group select { background: #222; color: #eee; border: 1px solid #444;
                         padding: 2px 4px; border-radius: 3px; }
    input[type=range] { width: 140px; }
    button { padding: 5px 16px; background: #2a2; color: #fff;
             border: none; border-radius: 4px; cursor: pointer; font-size: 12px; }
    button:active { background: #1a1; }
    #feedback { font-size: 11px; color: #fa0; min-height: 14px; }
  </style>
</head>
<body>
  <h2>ESP32-CAM DUAL STREAM</h2>

  <div class="feeds">
    <div class="feed">
      <div class="feed-label">CAM 1</div>
      <img id="img1" src="/stream/1" />
      <div class="feed-stats" id="stats1">connecting...</div>
    </div>
    <div class="feed">
      <div class="feed-label">CAM 2</div>
      <img id="img2" src="/stream/2" />
      <div class="feed-stats" id="stats2">connecting...</div>
    </div>
  </div>

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

    function fmtStats(s) {
      if (!s.online) return 'offline';
      return `FPS: ${s.fps.toFixed(1)}  lat: ${s.latency_ms.toFixed(0)}ms  drop: ${s.dropped}`;
    }

    setInterval(() => {
      fetch('/stats').then(r => r.json()).then(d => {
        document.getElementById('stats1').textContent = fmtStats(d.cam1);
        document.getElementById('stats2').textContent = fmtStats(d.cam2);
        document.getElementById('img1').className = d.cam1.online ? 'online' : 'offline';
        document.getElementById('img2').className = d.cam2.online ? 'online' : 'offline';
      }).catch(() => {});
    }, 1000);
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
    print(f"cam{cam.cam_id}: listening on UDP :{cam.udp_port}")

    frame_count = 0
    fps_ts = time.monotonic()

    while True:
        pkt, _ = sock.recvfrom(65535)
        jpeg = cam.ingest(pkt)
        if jpeg is None:
            continue

        cam.stats["online"] = True
        frame_count += 1
        elapsed = time.monotonic() - fps_ts
        if elapsed >= 2.0:
            cam.stats["fps"] = frame_count / elapsed
            print(
                f"cam{cam.cam_id}  FPS:{cam.stats['fps']:.1f}"
                f"  lat:{cam.stats['latency_ms']:.0f}ms"
                f"  frame:{len(jpeg)}B"
                f"  drop:{cam.stats['dropped']}"
            )
            frame_count = 0
            fps_ts = time.monotonic()


# ── Offline watchdog — marks cam offline if no frame for 3s ───────────────────
def watchdog():
    last_seen = {1: 0.0, 2: 0.0}
    while True:
        time.sleep(1)
        now = time.monotonic()
        for cid, cam in CAMS.items():
            with cam.frame_lock:
                has_frame = bool(cam.latest_frame)
            if has_frame:
                # Track last time we had a frame by checking event
                pass
            # simpler: check if fps dropped to 0 for 3+ seconds
            if cam.stats["fps"] > 0:
                last_seen[cid] = now
            if now - last_seen[cid] > 3.0 and cam.stats["online"]:
                cam.stats["online"] = False
                print(f"cam{cid}: offline")


# ── Beacon: broadcast LAPTOP:<ip>, collect cam IPs ────────────────────────────
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
    print(f"Beacon: {msg.decode()}  discovery UDP:{DISCOVERY_PORT}")

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
    threading.Thread(target=watchdog,      daemon=True).start()
    threading.Thread(target=udp_serve, args=(CAMS[1],), daemon=True).start()
    udp_serve(CAMS[2])   # main thread handles cam2
