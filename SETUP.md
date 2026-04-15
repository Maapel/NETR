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
