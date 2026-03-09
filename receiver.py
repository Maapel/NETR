"""
Single-camera TCP receiver with browser-based MJPEG viewer.

Receives: [8B int64 timestamp µs] [4B uint32 image_len] [JPEG bytes]

Open http://localhost:8080 in your browser to view the stream.
Press Ctrl+C to quit.
"""

import socket
import struct
import time
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

HOST = "0.0.0.0"
TCP_PORT = 5000
HTTP_PORT = 8080

# Shared latest frame (bytes) + lock
latest_frame: bytes = b""
frame_lock = threading.Lock()
frame_event = threading.Event()

# Stats
stats = {"fps": 0.0, "latency_ms": 0.0}


# ── MJPEG HTTP server ─────────────────────────────────────────────────────────
class MJPEGHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # silence access logs

    def do_GET(self):
        if self.path == "/":
            self._serve_index()
        elif self.path == "/stream":
            self._serve_mjpeg()
        else:
            self.send_error(404)

    def _serve_index(self):
        html = f"""<!DOCTYPE html>
<html>
<head>
  <title>ESP32-CAM Stream</title>
  <style>
    body {{ background:#111; color:#eee; font-family:monospace;
            display:flex; flex-direction:column; align-items:center; padding:20px; }}
    img {{ max-width:100%; border:2px solid #444; }}
    #stats {{ margin-top:10px; font-size:14px; color:#8f8; }}
  </style>
</head>
<body>
  <h2>ESP32-CAM Live</h2>
  <img src="/stream" />
  <div id="stats">connecting...</div>
  <script>
    // Poll stats endpoint every second
    setInterval(() => {{
      fetch('/stats').then(r=>r.json()).then(d=>{{
        document.getElementById('stats').textContent =
          `FPS: ${{d.fps.toFixed(1)}}  latency: ${{d.latency_ms.toFixed(1)}} ms`;
      }});
    }}, 1000);
  </script>
</body>
</html>"""
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(html.encode())

    def _serve_mjpeg(self):
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
                header = (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(data)).encode() + b"\r\n\r\n"
                )
                self.wfile.write(header + data + b"\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):  # noqa: F811
        if self.path == "/":
            self._serve_index()
        elif self.path == "/stream":
            self._serve_mjpeg()
        elif self.path == "/stats":
            import json
            body = json.dumps(stats).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404)


def http_thread():
    server = HTTPServer(("0.0.0.0", HTTP_PORT), MJPEGHandler)
    print(f"Browser stream → http://localhost:{HTTP_PORT}")
    server.serve_forever()


# ── TCP receiver ──────────────────────────────────────────────────────────────
def recv_exact(conn: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise EOFError("Connection closed")
        buf.extend(chunk)
    return bytes(buf)


def tcp_serve():
    global latest_frame
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((HOST, TCP_PORT))
        srv.listen(1)
        print(f"Waiting for ESP32-CAM on TCP :{TCP_PORT}...")

        while True:
            conn, addr = srv.accept()
            print(f"ESP32 connected: {addr}")
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

            frame_count = 0
            fps_ts = time.monotonic()

            try:
                with conn:
                    while True:
                        header = recv_exact(conn, 12)
                        cam_ts_us, img_len = struct.unpack("<qI", header)
                        jpeg = recv_exact(conn, img_len)

                        with frame_lock:
                            latest_frame = jpeg
                        frame_event.set()

                        now_us = int(time.time() * 1e6)
                        stats["latency_ms"] = (now_us - cam_ts_us) / 1000.0

                        frame_count += 1
                        now = time.monotonic()
                        if now - fps_ts >= 2.0:
                            stats["fps"] = frame_count / (now - fps_ts)
                            print(
                                f"FPS: {stats['fps']:.1f}  "
                                f"latency: {stats['latency_ms']:.1f} ms  "
                                f"frame_size: {img_len} B"
                            )
                            frame_count = 0
                            fps_ts = now

            except EOFError:
                print(f"ESP32 disconnected: {addr}")


if __name__ == "__main__":
    threading.Thread(target=http_thread, daemon=True).start()
    tcp_serve()
