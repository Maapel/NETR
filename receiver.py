"""
Single-camera UDP receiver with browser-based MJPEG viewer.

Packet format (little-endian):
  [4B frame_id][2B chunk_idx][2B total_chunks][8B timestamp_us][JPEG chunk]

Open http://localhost:8080 in your browser.
Press Ctrl+C to quit.
"""

import socket
import struct
import time
import threading
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from collections import defaultdict

UDP_PORT  = 5000
HTTP_PORT = 8080
HDR       = struct.Struct("<IHHq")   # frame_id, chunk_idx, total_chunks, ts_us
HDR_SIZE  = HDR.size                 # 16 bytes

# Incomplete frame buffer: frame_id -> {"ts": int, "total": int, "chunks": dict}
_frames: dict = {}
_frames_lock = threading.Lock()

# Shared latest complete JPEG
latest_frame: bytes = b""
frame_lock   = threading.Lock()
frame_event  = threading.Event()

stats = {"fps": 0.0, "latency_ms": 0.0, "dropped": 0}

FRAME_TIMEOUT_S = 0.25   # discard incomplete frames older than this


# ── MJPEG HTTP server ─────────────────────────────────────────────────────────
class MJPEGHandler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_GET(self):
        if self.path == "/":
            self._index()
        elif self.path == "/stream":
            self._mjpeg()
        elif self.path == "/stats":
            body = json.dumps(stats).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404)

    def _index(self):
        html = """<!DOCTYPE html>
<html>
<head>
  <title>ESP32-CAM UDP Stream</title>
  <style>
    body { background:#111; color:#eee; font-family:monospace;
           display:flex; flex-direction:column; align-items:center; padding:20px; }
    img  { max-width:100%; border:2px solid #444; }
    #stats { margin-top:10px; font-size:14px; color:#8f8; }
  </style>
</head>
<body>
  <h2>ESP32-CAM Live (UDP)</h2>
  <img src="/stream" />
  <div id="stats">connecting...</div>
  <script>
    setInterval(() => {
      fetch('/stats').then(r=>r.json()).then(d=>{
        document.getElementById('stats').textContent =
          `FPS: ${d.fps.toFixed(1)}  latency: ${d.latency_ms.toFixed(1)} ms  dropped: ${d.dropped}`;
      });
    }, 1000);
  </script>
</body>
</html>"""
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(html.encode())

    def _mjpeg(self):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        try:
            while True:
                frame_event.wait(timeout=2)
                frame_event.clear()
                with frame_lock:
                    data = latest_frame
                if not data:
                    continue
                hdr = (b"--frame\r\nContent-Type: image/jpeg\r\n"
                       b"Content-Length: " + str(len(data)).encode() + b"\r\n\r\n")
                self.wfile.write(hdr + data + b"\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass


def http_thread():
    HTTPServer(("0.0.0.0", HTTP_PORT), MJPEGHandler).serve_forever()


# ── UDP receiver ──────────────────────────────────────────────────────────────
def _expire_old_frames(now: float):
    """Discard frames that never completed within the timeout window."""
    expired = [fid for fid, f in _frames.items()
               if now - f["recv_time"] > FRAME_TIMEOUT_S]
    for fid in expired:
        stats["dropped"] += 1
        del _frames[fid]


def udp_serve():
    global latest_frame

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
    sock.bind(("0.0.0.0", UDP_PORT))
    print(f"Listening for UDP on :{UDP_PORT}")
    print(f"Browser stream → http://localhost:{HTTP_PORT}")

    frame_count = 0
    fps_ts = time.monotonic()

    while True:
        pkt, _ = sock.recvfrom(65535)
        if len(pkt) < HDR_SIZE:
            continue

        frame_id, chunk_idx, total_chunks, ts_us = HDR.unpack_from(pkt)
        payload = pkt[HDR_SIZE:]
        now = time.monotonic()

        with _frames_lock:
            _expire_old_frames(now)

            if frame_id not in _frames:
                _frames[frame_id] = {
                    "ts":        ts_us,
                    "total":     total_chunks,
                    "chunks":    {},
                    "recv_time": now,
                }

            entry = _frames[frame_id]
            entry["chunks"][chunk_idx] = payload

            if len(entry["chunks"]) == entry["total"]:
                # Reassemble in order
                jpeg = b"".join(entry["chunks"][i] for i in range(entry["total"]))
                del _frames[frame_id]

                with frame_lock:
                    latest_frame = jpeg
                frame_event.set()

                now_us = int(time.time() * 1e6)
                stats["latency_ms"] = (now_us - entry["ts"]) / 1000.0

                frame_count += 1
                elapsed = time.monotonic() - fps_ts
                if elapsed >= 2.0:
                    stats["fps"] = frame_count / elapsed
                    print(
                        f"FPS: {stats['fps']:.1f}  "
                        f"latency: {stats['latency_ms']:.1f} ms  "
                        f"frame: {len(jpeg)} B  "
                        f"dropped: {stats['dropped']}"
                    )
                    frame_count = 0
                    fps_ts = time.monotonic()


if __name__ == "__main__":
    threading.Thread(target=http_thread, daemon=True).start()
    udp_serve()
