"""
Minimal SNTP server (RFC 4330 stratum-1).
Binds on UDP 123 — run with sudo.
"""
import socket
import struct
import time

NTP_EPOCH_DELTA = 2208988800  # seconds between 1900-01-01 and 1970-01-01
PORT = 123


def to_ntp_ts(t: float) -> tuple[int, int]:
    t += NTP_EPOCH_DELTA
    sec = int(t)
    frac = int((t - sec) * (2**32))
    return sec, frac


def serve():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", PORT))
    print(f"NTP server listening on UDP :{PORT}")
    while True:
        data, addr = sock.recvfrom(1024)
        recv_time = time.time()
        if len(data) < 48:
            continue

        # Parse client transmit timestamp (bytes 40-47) to echo back as orig
        orig_sec, orig_frac = struct.unpack("!II", data[40:48])

        recv_sec, recv_frac = to_ntp_ts(recv_time)
        send_sec, send_frac = to_ntp_ts(time.time())

        # Build NTP response
        # LI=0, VN=4, Mode=4 (server), Stratum=1, Poll=6, Precision=-20
        flags = (0 << 6) | (4 << 3) | 4
        resp = struct.pack(
            "!BBbbIIIIIIIIIII",
            flags,        # LI + VN + Mode
            1,            # Stratum
            6,            # Poll interval
            -20,          # Precision
            0,            # Root delay
            0,            # Root dispersion
            0x4C4F434C,   # Reference ID ("LOCL")
            recv_sec, recv_frac,   # Reference timestamp
            orig_sec, orig_frac,   # Origin timestamp
            recv_sec, recv_frac,   # Receive timestamp
            send_sec, send_frac,   # Transmit timestamp
        )

        sock.sendto(resp, addr)
        print(f"NTP response → {addr[0]}  time={send_sec - NTP_EPOCH_DELTA:.3f}")


if __name__ == "__main__":
    serve()
