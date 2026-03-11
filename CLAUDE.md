# ESP32-CAM XR Rig — Claude Instructions

## Project overview
Dual ESP32-CAM (AI-Thinker) streaming rig for XR use.
Cameras stream JPEG frames over UDP to a Python receiver on the laptop.

## Hardware
- **cam1** MAC `c4:dd:57:ea:28:5c` — CAM_ID=1, stream UDP 5000, cmd UDP 5001
- **cam2** MAC `c4:dd:57:ea:3d:84` — CAM_ID=2, stream UDP 5002, cmd UDP 5003
- Discovery/beacon channel: UDP 5004
- OTA port: 3232, password: `esp32ota`
- LED: GPIO 33, active LOW

## Key files
- `esp32cam-stream/src/main.cpp` — firmware (single source for both cams)
- `esp32cam-stream/platformio.ini` — build environments (cam1/cam2/cam1_ota/cam2_ota)
- `receiver.py` — laptop-side UDP receiver + MJPEG HTTP server (port 8080)
- `ntp_server.py` — local NTP server (needs sudo, port 123)
- `discover.py` — finds camera IPs on network, patches platformio.ini

## Rules — always follow these

### LED debugging
- **Always add LED feedback** for any new state or error condition
- GPIO 33 is the debug LED (active LOW: `digitalWrite(33, LOW)` = ON)
- GPIO 4 is the flash LED — avoid using it (too bright, shares camera power)
- Current patterns:
  - Fast blink 100ms — no WiFi
  - Slow blink 500ms — WiFi OK, no laptop beacon yet
  - Solid ON — streaming
  - 3 rapid flashes loop — camera init failed

### OTA workflow
- **Never hardcode IPs** in firmware — laptop IP is discovered via beacon
- First flash on new hardware: wired via USB-serial (IO0→GND, RST to enter flash mode)
- All subsequent flashes: OTA via `pio run -e cam1_ota --target upload`
- If OTA fails: check ping latency — university AP throttles client-to-client traffic, retry
- If cameras not found: run `discover.py` first (only works if AP allows UDP broadcast)
- On restricted networks (e.g. iitm_wifi_): scan by MAC instead

### Flashing commands
```bash
# Wired (first time only)
/home/maadhav/pio-venv/bin/pio run -e cam1 --target upload --upload-port /dev/ttyUSB0
/home/maadhav/pio-venv/bin/pio run -e cam2 --target upload --upload-port /dev/ttyUSB0

# OTA (all subsequent)
/home/maadhav/pio-venv/bin/pio run -e cam1_ota --target upload
/home/maadhav/pio-venv/bin/pio run -e cam2_ota --target upload
```

### FreeRTOS task layout
| Task | Core | Purpose |
|------|------|---------|
| captureTask | 0 | Camera capture at target FPS |
| sendTask | 1 | UDP fragmentation + send |
| cmdTask | 1 | Receive quality/fps commands |
| otaTask | 1 | ArduinoOTA handler |
| discoveryTask | 1 | Beacon broadcast + laptop IP discovery |
| ledTask | 1 | LED state machine |
| wifiTask | 1 | WiFi watchdog + auto-reconnect |

### Code conventions
- Use `volatile` for globals shared between tasks
- Only call sensor functions (e.g. `set_quality`) when value actually changes
- Use `vTaskDelayUntil` for precise FPS pacing; reset `lastWake` when FPS changes
- Always reset `g_laptop_ip` to `""` when WiFi drops

### Running the receiver
```bash
# Terminal 1 (optional — for timestamp sync on local network)
sudo python3 /home/maadhav/iot-project/ntp_server.py

# Terminal 2
/home/maadhav/pio-venv/bin/python /home/maadhav/iot-project/receiver.py

# Browser
http://localhost:8080
```

## Network history
| Network | Subnet | cam1 IP | cam2 IP |
|---------|--------|---------|---------|
| 215 (hotspot) | 192.168.137.x | .149 | .34 |
| iitm_wifi_ | 10.150.37.x | .102 | .163 |
