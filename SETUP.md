# NETR Setup & Development Guide

Dual ESP32-CAM XR rig: two AI-Thinker cameras stream JPEG over UDP to a Python receiver on the laptop. One cam is the **eye** (pupil tracking), the other is the **world** (scene). Calibration maps PCCR vectors to gaze coordinates via a 2nd-order polynomial + ArUco homography.

Read `FLOW.md` for architecture. Read `CLAUDE.md` for hardware rules & task conventions.

---

## 1. Hardware

- **2× ESP32-CAM (AI-Thinker)** modules
  - cam1 MAC `c4:dd:57:ea:28:5c` → CAM_ID=1, stream UDP 5000, cmd 5001
  - cam2 MAC `c4:dd:57:ea:3d:84` → CAM_ID=2, stream UDP 5002, cmd 5003
- **USB-to-serial adapter** (CP2102 / CH340) for first-time flashing
- **Jumper wire** (IO0 → GND) to enter ESP32 flash mode
- Host laptop on same Wi-Fi subnet as the cams

See `CLAUDE.md` for LED patterns, OTA rules, and FreeRTOS task layout.

---

## 2. Laptop Prerequisites

### OS
Tested Linux (Arch, `6.18.22-1-lts`). Should work on macOS. Windows untested.

### System packages
- Python 3.11+
- git

### Python venv
```bash
python3 -m venv ~/pio-venv
source ~/pio-venv/bin/activate
pip install -r requirements.txt
```

`platformio` installed via pip. No system `pio` package needed.

### Serial permissions (Linux)
```bash
sudo usermod -aG dialout $USER   # log out/in after this
```

---

## 3. Clone & Configure

```bash
git clone <repo-url> netr
cd netr
```

Edit `rig_config.json` to pick which cam ID is eye vs world:
```json
{"eye_cam": 2, "world_cam": 1}
```

---

## 4. Flash Firmware

### First flash (wired)
Put ESP32-CAM in flash mode: jumper IO0→GND, press RST. Then:
```bash
~/pio-venv/bin/pio run -e cam1 --target upload --upload-port /dev/ttyUSB0
~/pio-venv/bin/pio run -e cam2 --target upload --upload-port /dev/ttyUSB0
```
Remove jumper, press RST. Watch serial at 115200 baud — LED blinks fast if no Wi-Fi, slow if connected but no laptop beacon, solid when streaming.

### Configure Wi-Fi
Edit `esp32cam-stream/src/main.cpp` for SSID / PSK, re-flash.

### Subsequent flashes (OTA)
After first wired flash, update `upload_port` IPs in `esp32cam-stream/platformio.ini`, then:
```bash
~/pio-venv/bin/pio run -e cam1_ota --target upload
~/pio-venv/bin/pio run -e cam2_ota --target upload
```
OTA password: `esp32ota`. OTA port: 3232.

### Find cam IPs
```bash
~/pio-venv/bin/python discover.py
```
Works if AP allows UDP broadcast. On locked networks (iitm_wifi_) scan by MAC instead.

---

## 5. Run the Stack

Three processes. Open three terminals.

### Terminal 1 — NTP server (optional but recommended for timestamp sync)
```bash
sudo ~/pio-venv/bin/python ntp_server.py
```
Port 123 needs sudo. Cams use Cristian's algorithm every 10s to sync capture timestamps.

### Terminal 2 — Compute engine (pupil/glint/gaze)
```bash
~/pio-venv/bin/python compute/engine.py
```
Port 8081. Loads `gaze_model.json` if present.

### Terminal 3 — Receiver (UDP reassembly + HTTP/MJPEG)
```bash
~/pio-venv/bin/python receiver.py
```
Port 8080. Open `http://localhost:8080` in browser.

### Optional — Calibration server
```bash
~/pio-venv/bin/python calibration_server.py
```
Port 8090. Open `http://localhost:8090` fullscreen. Saccade mode only (sweep deprecated).

### Optional — TUI
```bash
~/pio-venv/bin/python tui_revamped.py
```
Textual UI for managing cams, OTA, discovery, logs.

---

## 6. Calibration Workflow

1. Start receiver + engine. Turn on analysis in browser (toggle "Analysis" checkbox).
2. Open `http://localhost:8090` fullscreen on the target screen.
3. Press **SACCADE** mode (default).
4. Press **START** — eye must fixate on each target. 9 zones × multiple rounds.
5. After 6+ samples, **LIVE** button enables — shows red crosshair of predicted gaze.
6. Press **STOP** — models saved to `gaze_model.json` + `screen_model.json`.
7. Recording folder under `recordings/<timestamp>/` holds raw AVIs + `screen_events.jsonl`.

---

## 7. Recording & Playback

- Receiver keeps rolling **40s buffer** of every frame (both cams).
- **Save 40s Buffer** button dumps `recordings/<ts>/camN.avi`.
- Checkbox **Annotate Eye** saves pupil-overlay AVI alongside raw.
- Checkbox **Annotate World** saves gaze-crosshair AVI alongside raw.
- Playback: `http://localhost:8080/player`.

---

## 8. Ports Reference

| Port | Proto | Purpose |
|------|-------|---------|
| 5000 | UDP   | cam1 stream → laptop |
| 5001 | UDP   | cam1 cmd ← laptop |
| 5002 | UDP   | cam2 stream → laptop |
| 5003 | UDP   | cam2 cmd ← laptop |
| 5004 | UDP   | discovery beacon |
| 5005 | UDP   | time sync (Cristian) |
| 5010 | UDP   | cam log channel |
| 123  | UDP   | NTP (sudo required) |
| 3232 | TCP   | OTA |
| 8080 | HTTP  | receiver web UI |
| 8081 | HTTP  | compute engine API |
| 8090 | HTTP  | calibration server + WS |

---

## 9. Development Notes

### Code layout
- `esp32cam-stream/src/main.cpp` — firmware (single source for both cams, CAM_ID build flag)
- `receiver.py` — UDP reassembly, HTTP/MJPEG, rolling rec buffer, browser UI
- `compute/engine.py` — pupil/glint pipeline, gaze prediction (`/process`, `/result`)
- `compute/eye_pipeline.py` — detection algorithms (threshold/seed-flood)
- `calibration_server.py` — calibration web UI, ArUco homography, model fitting
- `gaze_model.py` — 2nd-order polynomial `predict(dx, dy) → (X, Y)`
- `rig_config.py` — eye vs world cam mapping (persisted in `rig_config.json`)
- `tui_revamped.py` — Textual TUI for management
- `discover.py` — find cam IPs, patch `platformio.ini`

### Persisted state
- `cam_settings.json` — sensor settings (brightness, quality, fps, etc.); auto-saved on Apply
- `eye_settings.json` — pupil/glint detector params
- `rig_config.json` — eye/world cam assignment
- `gaze_model.json` — scene-space gaze polynomial (A, B coeffs)
- `screen_model.json` — screen-space polynomial (legacy, not currently used by live)
- `calib_dataset.json` — latest calibration samples

### Rules (from `CLAUDE.md`)
- Add LED feedback for new firmware states (GPIO 33, active LOW)
- Every FreeRTOS task must pause on `g_ota_active`
- Always init camera at `FRAMESIZE_UXGA`, then downscale — never init small
- After `set_framesize` runtime call, flush 3 frames
- Never hardcode laptop IP in firmware — use beacon discovery

---

## 10. Remote Setup (Oracle Cloud)

Use this when the ESP32 cams are **not on the same LAN** as the laptop.  
A single public UDP port funnels all camera traffic through an Oracle VM.  
A gateway process serves the full web UI through one public TCP port.

### Architecture

```
ESP32-CAM  ──UDP 8877──▶  relay.py (Oracle)  ──UDP──▶  receiver.py (localhost:8091)
Browser    ──TCP 8877──▶  nginx → gateway.py (localhost:8090)
                             /             → receiver   (localhost:8091)
                             /calibration/ → calibration_server.py (localhost:8092)
                             /iot-dashboard → process manager UI
```

TCP and UDP can share port 8877 — nginx handles TCP, relay.py handles UDP.

### Oracle prerequisites

- OCI VM with Ubuntu 22.04 (or 24.04)
- Port 8877 open in both OCI Security List **and** `ufw` / `iptables`
- Python 3.12+ with a venv at `~/iot-project/.venv`
- `pip install aiohttp` in that venv
- nginx installed: `sudo apt install nginx`

### 1 — Clone the repo on Oracle

```bash
ssh ubuntu@<oracle-ip>
git clone <repo-url> ~/iot-project
cd ~/iot-project
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

### 2 — Create the relay secret

Pick a random hex string (e.g. `openssl rand -hex 8`) and keep it in two places:

**On Oracle** — export before starting:
```bash
export RELAY_SECRET=<your_secret>
```
Or add it to `~/.profile` / a secrets file that is never committed.

**On the ESP32** — create `esp32cam-stream/src/secrets.h` (gitignored):
```cpp
#pragma once
#define RELAY_SECRET "<your_secret>"
```
See `esp32cam-stream/src/secrets.h.example` as a template.

### 3 — Enable relay mode in firmware

In `esp32cam-stream/src/main.cpp`, ensure the block near the top reads:
```cpp
#define USE_RELAY
#ifdef USE_RELAY
  #define RELAY_IP   "<oracle-public-ip>"
  #define RELAY_PORT 8877
  #include "secrets.h"
#endif
```
Comment out `#define USE_RELAY` to fall back to LAN beacon mode unchanged.

Re-flash via OTA or wired (see §4 above).

### 4 — Configure nginx on Oracle

```bash
sudo tee /etc/nginx/sites-available/iot-cam > /dev/null << 'EOF'
server {
    listen 8877;
    server_name _;
    location / {
        proxy_pass http://127.0.0.1:8090;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_buffering off;
        proxy_read_timeout 86400s;
    }
}
EOF
sudo ln -sf /etc/nginx/sites-available/iot-cam /etc/nginx/sites-enabled/iot-cam
sudo nginx -t && sudo systemctl reload nginx
```

### 5 — Start services on Oracle

```bash
cd ~/iot-project
export RELAY_SECRET=<your_secret>
./start-iot.sh
```

`start-iot.sh` starts:
1. `relay.py` — UDP 0.0.0.0:8877, forwards frames to localhost:5000/5002
2. `gateway.py` — HTTP localhost:8090, routes paths to backend processes

The receiver, calibration server, and engine are **not started automatically** — use the dashboard.

### 6 — Dashboard

Open `http://<oracle-ip>:8877/iot-dashboard` in your browser.

- **Start / Stop** each service (Receiver, Calibration, Engine)
- **Live logs** stream via WebSocket
- Links to **Receiver UI** (`/`) and **Calibration UI** (`/calibration/`)

### 7 — Remote ports reference

| Port | Proto | Where | Purpose |
|------|-------|-------|---------|
| 8877 | UDP | Oracle public | ESP32 → relay (auth + frames + keepalives) |
| 8877 | TCP | Oracle public | Browser → nginx → gateway |
| 5000 | UDP | Oracle localhost | relay → receiver cam1 |
| 5002 | UDP | Oracle localhost | relay → receiver cam2 |
| 5001 | UDP | Oracle localhost | gateway cmd → relay → ESP32 cam1 |
| 5003 | UDP | Oracle localhost | gateway cmd → relay → ESP32 cam2 |
| 8090 | TCP | Oracle localhost | gateway (nginx upstream) |
| 8091 | TCP | Oracle localhost | receiver HTTP/MJPEG |
| 8092 | TCP | Oracle localhost | calibration server + WS |
| 8081 | TCP | Oracle localhost | compute engine |

### Auth flow

1. On boot, ESP32 sends `AUTH:<cam_id>:<secret>` to Oracle UDP 8877.
2. Relay verifies secret → sends `AUTH_OK` → whitelists source IP.
3. All subsequent packets from that IP (frames, keepalives) are forwarded.
4. Packets from un-authed IPs are silently dropped.
5. For commands (OTA, settings), ESP32 sends `CMD_KA:<cam_id>` keepalives every 5s to punch a NAT return path; relay sends commands back through the same mapping.

### Troubleshooting

**Cams not connecting:**
```bash
# On Oracle — watch relay log
cat /proc/$(pgrep -f relay.py)/fd/1
# Should show: [relay] cam1 authed @ (<esp32-ip>, <port>)
```

**Receiver shows fps=0:**
Check the relay log for `cam1 authed`. Then check receiver is running (`ss -tlnp | grep 8091`).

**Calibration WS fails:**
Gateway routes `ws://<host>/ws` → calibration's `/ws` endpoint.
Verify calibration is running: `ss -tlnp | grep 8092`.

**Services die on Oracle restart:**
Add a systemd unit or cron `@reboot` that exports `RELAY_SECRET` and runs `start-iot.sh`.

### Logs

```bash
# relay
cat /proc/$(pgrep -f relay.py)/fd/1

# gateway + subprocess output visible in dashboard /iot-dashboard
# or directly:
cat /proc/$(pgrep -f gateway.py)/fd/1
```
