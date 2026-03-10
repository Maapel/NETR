"""
Discover ESP32-CAM IPs on the current network and update platformio.ini.

Run before OTA flashing on a new network:
    python discover.py

The ESP32s broadcast "CAM:<id>" every 10s on UDP port 5004.
This script listens for 15 seconds, collects all announcements,
then patches the upload_port lines in platformio.ini.
"""

import socket
import time
import re
import os

DISCOVERY_PORT = 5004
LISTEN_SECS    = 15
INI_PATH       = os.path.join(os.path.dirname(__file__),
                              "esp32cam-stream", "platformio.ini")


def discover() -> dict[str, str]:
    """Listen for CAM:<id> beacons. Returns {cam_id: ip}."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    sock.bind(("0.0.0.0", DISCOVERY_PORT))
    sock.settimeout(1.0)

    # Broadcast a laptop beacon so cameras respond immediately
    own_ip = _own_ip()
    beacon = f"LAPTOP:{own_ip}".encode()

    found: dict[str, str] = {}
    deadline = time.monotonic() + LISTEN_SECS

    print(f"Listening for cameras for {LISTEN_SECS}s "
          f"(broadcasting beacon as {own_ip})...")

    while time.monotonic() < deadline:
        sock.sendto(beacon, ("255.255.255.255", DISCOVERY_PORT))
        try:
            data, addr = sock.recvfrom(64)
            txt = data.decode(errors="ignore").strip()
            if txt.startswith("CAM:"):
                cam_id = txt[4:]
                if cam_id not in found:
                    found[cam_id] = addr[0]
                    print(f"  Found cam{cam_id} at {addr[0]}")
        except socket.timeout:
            pass

    sock.close()
    return found


def _own_ip() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]


def patch_ini(cameras: dict[str, str]):
    """Update upload_port for each cam_ota env in platformio.ini."""
    with open(INI_PATH) as f:
        content = f.read()

    for cam_id, ip in cameras.items():
        # Match the cam<id>_ota section and replace its upload_port line
        pattern = r'(\[env:cam' + cam_id + r'_ota\].*?upload_port\s*=\s*)[^\n]+'
        replacement = r'\g<1>' + ip
        new_content, n = re.subn(pattern, replacement, content, flags=re.DOTALL)
        if n:
            content = new_content
            print(f"  platformio.ini: cam{cam_id}_ota upload_port → {ip}")
        else:
            print(f"  WARNING: could not find cam{cam_id}_ota section in {INI_PATH}")

    with open(INI_PATH, "w") as f:
        f.write(content)


if __name__ == "__main__":
    cameras = discover()
    if not cameras:
        print("No cameras found. Make sure they are powered and connected to WiFi.")
    else:
        print(f"\nFound {len(cameras)} camera(s): {cameras}")
        patch_ini(cameras)
        print("\nplatformio.ini updated. You can now run:")
        for cam_id in cameras:
            print(f"  pio run -e cam{cam_id}_ota --target upload")
