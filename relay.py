#!/usr/bin/env python3
"""
ESP32-CAM relay — funnels all camera traffic through one public UDP port.

Auth flow:
  ESP32 sends  AUTH:<cam_id>:<secret>  once on boot.
  Relay verifies secret, whitelists source IP.
  All subsequent packets from that IP are accepted (frames, keepalives).
  Packets from unknown IPs are silently dropped.

Packet types (after auth):
  CAM:<id>     — stream heartbeat, refreshes NAT mapping for frame routing
  CMD_KA:<id>  — cmd keepalive, registers return address for commands
  <16+ bytes>  — frame data, forwarded to localhost:5000/5002

Localhost cmd inputs (127.0.0.1:5001 / 5003):
  any bytes    — command from receiver.py, forwarded to ESP32's cmd channel
"""

import socket
import threading
import time

RELAY_PORT   = 8877
SECRET       = "e0201424befd0e31"
CAM_PORTS    = {1: 5000, 2: 5002}
CMD_IN_PORTS = {1: 5001, 2: 5003}

authed_ips:      dict[str, int]   = {}  # src_ip → cam_id  (whitelisted after AUTH)
cam_stream_addr: dict[int, tuple] = {}  # cam_id → (ip, port)
cam_cmd_addr:    dict[int, tuple] = {}  # cam_id → (ip, port)
lock = threading.Lock()

public_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
public_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
public_sock.bind(("0.0.0.0", RELAY_PORT))

_local_socks = {cid: socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                for cid in CAM_PORTS}


def _public_listener():
    while True:
        data, addr = public_sock.recvfrom(65536)
        src_ip = addr[0]

        try:
            text = data[:64].decode("ascii", errors="replace").strip()
        except Exception:
            text = ""

        # ── Auth handshake (open to all IPs, secret verified here) ────────────
        if text.startswith("AUTH:"):
            parts = text.split(":")
            if len(parts) == 3 and parts[2] == SECRET:
                try:
                    cid = int(parts[1])
                    with lock:
                        authed_ips[src_ip] = cid
                        cam_stream_addr[cid] = addr
                    public_sock.sendto(b"AUTH_OK", addr)
                    print(f"[relay] cam{cid} authed @ {addr}")
                except ValueError:
                    pass
            else:
                public_sock.sendto(b"AUTH_FAIL", addr)
                print(f"[relay] AUTH_FAIL from {addr}")
            continue

        # ── Drop everything from non-authed IPs ───────────────────────────────
        with lock:
            cid = authed_ips.get(src_ip)
        if cid is None:
            continue

        # ── Routed packets from whitelisted IPs ───────────────────────────────
        if text.startswith("CAM:"):
            with lock:
                cam_stream_addr[cid] = addr

        elif text.startswith("CMD_KA:"):
            with lock:
                cam_cmd_addr[cid] = addr

        else:
            # Frame packet
            with lock:
                cam_stream_addr[cid] = addr
            if cid in CAM_PORTS:
                _local_socks[cid].sendto(data, ("127.0.0.1", CAM_PORTS[cid]))


def _cmd_listener(cam_id: int, in_port: int):
    """Relay commands from receiver.py (localhost) → ESP32 cmd channel."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("127.0.0.1", in_port))
    print(f"[relay] cmd input for cam{cam_id} on localhost:{in_port}")
    while True:
        data, _ = s.recvfrom(256)
        with lock:
            dest = cam_cmd_addr.get(cam_id)
        if dest:
            public_sock.sendto(data, dest)
        else:
            print(f"[relay] cmd cam{cam_id} dropped — not authed yet")


if __name__ == "__main__":
    threading.Thread(target=_public_listener, daemon=True, name="public").start()
    for cid, port in CMD_IN_PORTS.items():
        threading.Thread(target=_cmd_listener, args=(cid, port),
                         daemon=True, name=f"cmd{cid}").start()

    print(f"[relay] UDP 0.0.0.0:{RELAY_PORT}  secret=****{SECRET[-4:]}")
    print(f"[relay] Stream → localhost {CAM_PORTS}")
    print(f"[relay] Cmd   ← localhost {CMD_IN_PORTS}")
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("[relay] stopped")
