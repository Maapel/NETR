#!/usr/bin/env python3
"""
IoT gateway — single public port (8090) routes to backend services.

Routes:
  /iot-dashboard           → dashboard HTML (process manager)
  /iot-dashboard/ws        → dashboard WebSocket (logs / status)
  /calibration/ws          → proxy WS → localhost:8092/ws
  /calibration/...         → proxy HTTP → localhost:8092/...
  /engine/...              → proxy HTTP → localhost:8081/...
  /...                     → proxy HTTP → localhost:8091 (receiver)

Backend ports (env-overridable):
  GATEWAY_PORT   8090
  RECEIVER_PORT  8091
  CALIB_PORT     8092
  ENGINE_PORT    8081
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

from aiohttp import web, ClientSession, ClientConnectorError, WSMsgType
import aiohttp

GATEWAY_PORT  = int(os.environ.get("GATEWAY_PORT",  8090))
RECEIVER_PORT = int(os.environ.get("RECEIVER_PORT", 8091))
CALIB_PORT    = int(os.environ.get("CALIB_PORT",    8092))
ENGINE_PORT   = int(os.environ.get("ENGINE_PORT",   8081))

PROJECT_DIR = Path(__file__).parent
PYTHON = str(PROJECT_DIR / ".venv/bin/python3")
if not Path(PYTHON).exists():
    PYTHON = sys.executable

# ── Process registry ──────────────────────────────────────────────────────────
_procs: dict[str, asyncio.subprocess.Process] = {}
_log_buffers: dict[str, list[str]] = {"receiver": [], "calibration": [], "engine": []}
_ws_clients: set[web.WebSocketResponse] = set()

SERVICE_CMDS = {
    "receiver":    [PYTHON, "-u", str(PROJECT_DIR / "receiver.py")],
    "calibration": [PYTHON, "-u", str(PROJECT_DIR / "calibration_server.py")],
    "engine":      [PYTHON, "-u", str(PROJECT_DIR / "engine.py")],
}
SERVICE_ENV = {
    "receiver":    {"HTTP_PORT": str(RECEIVER_PORT)},
    "calibration": {"CALIB_PORT": str(CALIB_PORT),
                    "HTTP_PORT": str(RECEIVER_PORT)},
    "engine":      {"ENGINE_PORT": str(ENGINE_PORT),
                    "HTTP_PORT": str(RECEIVER_PORT)},
}


def _svc_status(name: str) -> str:
    p = _procs.get(name)
    if p is None:
        return "stopped"
    if p.returncode is None:
        return "running"
    return f"exited({p.returncode})"


async def _broadcast(msg: dict):
    global _ws_clients
    dead = set()
    for ws in _ws_clients:
        try:
            await ws.send_json(msg)
        except Exception:
            dead.add(ws)
    _ws_clients -= dead


async def _pipe_output(name: str, stream):
    while True:
        line = await stream.readline()
        if not line:
            break
        text = line.decode(errors="replace").rstrip()
        buf = _log_buffers[name]
        buf.append(text)
        if len(buf) > 500:
            buf.pop(0)
        await _broadcast({"type": "log", "service": name, "line": text})
    await _broadcast({"type": "status", "service": name, "state": _svc_status(name)})


async def _start_service(name: str):
    if name not in SERVICE_CMDS:
        return {"ok": False, "error": "unknown service"}
    p = _procs.get(name)
    if p and p.returncode is None:
        return {"ok": False, "error": "already running"}

    env = {**os.environ, **SERVICE_ENV.get(name, {})}
    proc = await asyncio.create_subprocess_exec(
        *SERVICE_CMDS[name],
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=str(PROJECT_DIR),
        env=env,
    )
    _procs[name] = proc  # type: ignore[assignment]
    _log_buffers[name].clear()
    asyncio.create_task(_pipe_output(name, proc.stdout))
    await _broadcast({"type": "status", "service": name, "state": "running"})
    return {"ok": True}


async def _stop_service(name: str):
    p = _procs.get(name)
    if p is None or p.returncode is not None:
        return {"ok": False, "error": "not running"}
    p.terminate()
    try:
        await asyncio.wait_for(p.wait(), 5)
    except asyncio.TimeoutError:
        p.kill()
        await p.wait()
    await _broadcast({"type": "status", "service": name, "state": _svc_status(name)})
    return {"ok": True}


# ── Dashboard HTML ────────────────────────────────────────────────────────────
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>IoT Dashboard</title>
<style>
  body{font-family:monospace;background:#0d1117;color:#c9d1d9;margin:0;padding:20px}
  h1{color:#58a6ff;margin-bottom:20px}
  .services{display:flex;gap:16px;flex-wrap:wrap;margin-bottom:24px}
  .card{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px;min-width:200px}
  .card h2{margin:0 0 8px;font-size:14px;color:#8b949e;text-transform:uppercase;letter-spacing:.05em}
  .status{font-size:13px;margin-bottom:12px}
  .status.running{color:#3fb950}
  .status.stopped{color:#8b949e}
  .status.exited{color:#f85149}
  button{padding:6px 14px;border:none;border-radius:4px;cursor:pointer;font-size:12px;margin-right:6px}
  .btn-start{background:#238636;color:#fff}
  .btn-start:hover{background:#2ea043}
  .btn-stop{background:#b62324;color:#fff}
  .btn-stop:hover{background:#da3633}
  .logs{background:#0d1117;border:1px solid #30363d;border-radius:6px;padding:12px;
        height:400px;overflow-y:auto;font-size:12px;white-space:pre-wrap}
  .log-header{display:flex;gap:8px;margin-bottom:8px;align-items:center}
  .log-header label{font-size:12px;color:#8b949e}
  select{background:#161b22;color:#c9d1d9;border:1px solid #30363d;padding:4px 8px;border-radius:4px}
  .links{margin-top:16px}
  .links a{color:#58a6ff;margin-right:16px;font-size:13px}
</style>
</head>
<body>
<h1>IoT Dashboard</h1>
<div class="services" id="services">
  <div class="card" id="card-receiver">
    <h2>Receiver</h2>
    <div class="status stopped" id="st-receiver">stopped</div>
    <button class="btn-start" onclick="start('receiver')">Start</button>
    <button class="btn-stop" onclick="stop('receiver')">Stop</button>
  </div>
  <div class="card" id="card-calibration">
    <h2>Calibration</h2>
    <div class="status stopped" id="st-calibration">stopped</div>
    <button class="btn-start" onclick="start('calibration')">Start</button>
    <button class="btn-stop" onclick="stop('calibration')">Stop</button>
  </div>
  <div class="card" id="card-engine">
    <h2>Engine</h2>
    <div class="status stopped" id="st-engine">stopped</div>
    <button class="btn-start" onclick="start('engine')">Start</button>
    <button class="btn-stop" onclick="stop('engine')">Stop</button>
  </div>
</div>
<div class="links">
  <a href="/" target="_blank">Receiver UI</a>
  <a href="/calibration/" target="_blank">Calibration UI</a>
</div>
<div class="log-header">
  <label>Logs:</label>
  <select id="log-select" onchange="switchLog(this.value)">
    <option value="receiver">receiver</option>
    <option value="calibration">calibration</option>
    <option value="engine">engine</option>
  </select>
</div>
<div class="logs" id="log-box"></div>
<script>
const ws = new WebSocket(`ws://${location.host}/iot-dashboard/ws`);
const logBufs = {receiver:[], calibration:[], engine:[]};
let currentLog = 'receiver';

ws.onmessage = e => {
  const msg = JSON.parse(e.data);
  if (msg.type === 'status') {
    const el = document.getElementById('st-' + msg.service);
    if (el) { el.textContent = msg.state; el.className = 'status ' + msg.state.split('(')[0]; }
  } else if (msg.type === 'log') {
    logBufs[msg.service].push(msg.line);
    if (logBufs[msg.service].length > 500) logBufs[msg.service].shift();
    if (msg.service === currentLog) appendLog(msg.line);
  } else if (msg.type === 'init') {
    for (const [svc, state] of Object.entries(msg.statuses)) {
      const el = document.getElementById('st-' + svc);
      if (el) { el.textContent = state; el.className = 'status ' + state.split('(')[0]; }
    }
    for (const [svc, lines] of Object.entries(msg.logs)) {
      logBufs[svc] = lines.slice();
    }
    renderLog();
  }
};

function appendLog(line) {
  const box = document.getElementById('log-box');
  box.textContent += line + '\\n';
  box.scrollTop = box.scrollHeight;
}
function renderLog() {
  const box = document.getElementById('log-box');
  box.textContent = logBufs[currentLog].join('\\n');
  box.scrollTop = box.scrollHeight;
}
function switchLog(svc) { currentLog = svc; renderLog(); }
function start(svc) { fetch('/iot-dashboard/api/start/' + svc, {method:'POST'}); }
function stop(svc)  { fetch('/iot-dashboard/api/stop/'  + svc, {method:'POST'}); }
</script>
</body>
</html>
"""


# ── Dashboard handlers ────────────────────────────────────────────────────────
async def handle_dashboard(request: web.Request):
    return web.Response(text=DASHBOARD_HTML, content_type="text/html")


async def handle_dashboard_ws(request: web.Request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    _ws_clients.add(ws)

    await ws.send_json({
        "type": "init",
        "statuses": {n: _svc_status(n) for n in SERVICE_CMDS},
        "logs": {n: list(_log_buffers[n]) for n in _log_buffers},
    })

    async for msg in ws:
        if msg.type == WSMsgType.ERROR:
            break

    _ws_clients.discard(ws)
    return ws


async def handle_api_start(request: web.Request):
    name = request.match_info["name"]
    result = await _start_service(name)
    return web.json_response(result)


async def handle_api_stop(request: web.Request):
    name = request.match_info["name"]
    result = await _stop_service(name)
    return web.json_response(result)


# ── Generic HTTP proxy ────────────────────────────────────────────────────────
async def _proxy_http(request: web.Request, target_port: int, strip_prefix: str = ""):
    path = request.path
    if strip_prefix and path.startswith(strip_prefix):
        path = path[len(strip_prefix):] or "/"

    url = f"http://127.0.0.1:{target_port}{path}"
    if request.query_string:
        url += "?" + request.query_string

    try:
        async with ClientSession() as session:
            async with session.request(
                request.method, url,
                headers={k: v for k, v in request.headers.items()
                         if k.lower() not in ("host", "content-length")},
                data=await request.read() if request.method not in ("GET", "HEAD") else None,
                allow_redirects=False,
            ) as resp:
                body = await resp.read()
                return web.Response(
                    status=resp.status,
                    headers={k: v for k, v in resp.headers.items()
                             if k.lower() not in ("transfer-encoding", "content-encoding")},
                    body=body,
                )
    except ClientConnectorError:
        return web.Response(status=502, text=f"Backend on port {target_port} not reachable")


# ── WebSocket proxy ───────────────────────────────────────────────────────────
async def _proxy_ws(request: web.Request, target_port: int, target_path: str):
    ws_client = web.WebSocketResponse()
    await ws_client.prepare(request)

    url = f"ws://127.0.0.1:{target_port}{target_path}"
    try:
        async with ClientSession() as session:
            async with session.ws_connect(url) as ws_backend:
                async def fwd_to_backend():
                    async for msg in ws_client:
                        if msg.type == WSMsgType.TEXT:
                            await ws_backend.send_str(msg.data)
                        elif msg.type == WSMsgType.BINARY:
                            await ws_backend.send_bytes(msg.data)
                        elif msg.type in (WSMsgType.CLOSE, WSMsgType.ERROR):
                            break

                async def fwd_to_client():
                    async for msg in ws_backend:
                        if msg.type == WSMsgType.TEXT:
                            await ws_client.send_str(msg.data)
                        elif msg.type == WSMsgType.BINARY:
                            await ws_client.send_bytes(msg.data)
                        elif msg.type in (WSMsgType.CLOSE, WSMsgType.ERROR):
                            break

                await asyncio.gather(fwd_to_backend(), fwd_to_client())
    except ClientConnectorError:
        pass

    return ws_client


# ── Route handlers ────────────────────────────────────────────────────────────
async def handle_calibration(request: web.Request):
    if request.headers.get("Upgrade", "").lower() == "websocket":
        return await _proxy_ws(request, CALIB_PORT, "/ws")
    return await _proxy_http(request, CALIB_PORT, strip_prefix="/calibration")


async def handle_engine(request: web.Request):
    return await _proxy_http(request, ENGINE_PORT, strip_prefix="/engine")


async def handle_receiver(request: web.Request):
    if request.headers.get("Upgrade", "").lower() == "websocket":
        path = request.path
        return await _proxy_ws(request, RECEIVER_PORT, path)
    return await _proxy_http(request, RECEIVER_PORT)


# ── App setup ─────────────────────────────────────────────────────────────────
app = web.Application()
app.router.add_get("/iot-dashboard",      handle_dashboard)
app.router.add_get("/iot-dashboard/",     handle_dashboard)
app.router.add_get("/iot-dashboard/ws",   handle_dashboard_ws)
app.router.add_post("/iot-dashboard/api/start/{name}", handle_api_start)
app.router.add_post("/iot-dashboard/api/stop/{name}",  handle_api_stop)
app.router.add_route("*", "/calibration",      handle_calibration)
app.router.add_route("*", "/calibration/{path:.*}", handle_calibration)
# Calibration JS builds ws://<host>/ws (not /calibration/ws) — catch it here
app.router.add_get("/ws",                      handle_calibration)
app.router.add_route("*", "/engine",           handle_engine)
app.router.add_route("*", "/engine/{path:.*}", handle_engine)
app.router.add_route("*", "/{path:.*}",        handle_receiver)


if __name__ == "__main__":
    print(f"[gateway] port {GATEWAY_PORT}")
    print(f"[gateway] receiver  → localhost:{RECEIVER_PORT}")
    print(f"[gateway] calibration → localhost:{CALIB_PORT}")
    print(f"[gateway] engine    → localhost:{ENGINE_PORT}")
    web.run_app(app, host="127.0.0.1", port=GATEWAY_PORT, access_log=None)
