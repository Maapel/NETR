"""
ESP32-CAM XR Rig Manager — TUI
Run: /home/maadhav/pio-venv/bin/python tui.py
"""

import asyncio
import ipaddress
import json
import os
import re
import socket
import struct
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, ScrollableContainer
from textual.reactive import reactive
from textual.widgets import (
    Button, DataTable, Footer, Header, Label, RichLog, Static
)

# ── Project paths ─────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent
PIO        = Path("/home/maadhav/pio-venv/bin/pio")
PYTHON     = Path("/home/maadhav/pio-venv/bin/python")
PIO_INI    = ROOT / "esp32cam-stream" / "platformio.ini"
RECEIVER   = ROOT / "receiver.py"
NTP_SRV    = ROOT / "ntp_server.py"
SERIAL_DEV = "/dev/ttyUSB0"

DISCOVERY_PORT = 5004
LOG_PORT       = 5010   # UDP debug log from ESP32s
ESP32_MACS = {
    "c4:dd:57:ea:28:5c": "cam1",
    "c4:dd:57:ea:3d:84": "cam2",
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def now() -> str:
    return datetime.now().strftime("%H:%M:%S")

def own_ip() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]

def subnet_base() -> str:
    ip = own_ip()
    return ".".join(ip.split(".")[:3])

def ping_host(ip: str) -> float | None:
    """Return RTT in ms or None."""
    try:
        r = subprocess.run(
            ["ping", "-c1", "-W1", ip],
            capture_output=True, timeout=2
        )
        if r.returncode == 0:
            m = re.search(r"time=([\d.]+)", r.stdout.decode())
            return float(m.group(1)) if m else 0.0
    except Exception:
        pass
    return None

def scan_for_esps() -> dict[str, str]:
    """Ping sweep + ARP lookup. Returns {cam_name: ip}."""
    base = subnet_base()
    results: dict[str, str] = {}

    def check(i: int):
        ip = f"{base}.{i}"
        if ping_host(ip) is not None:
            r = subprocess.run(["ip", "neigh", "show", ip],
                               capture_output=True, text=True)
            for mac, cam in ESP32_MACS.items():
                if mac in r.stdout.lower():
                    results[cam] = ip

    threads = [threading.Thread(target=check, args=(i,)) for i in range(1, 255)]
    for t in threads: t.start()
    for t in threads: t.join()
    return results

def update_pio_ini(cam: str, ip: str):
    """Patch upload_port for cam_ota env in platformio.ini."""
    content = PIO_INI.read_text()
    pattern = r'(\[env:' + cam + r'_ota\].*?upload_port\s*=\s*)[^\n]+'
    new, n = re.subn(pattern, r'\g<1>' + ip, content, flags=re.DOTALL)
    if n:
        PIO_INI.write_text(new)
        return True
    return False


# ── Camera state ──────────────────────────────────────────────────────────────
class CamState:
    def __init__(self, name: str):
        self.name   = name
        self.ip     = ""
        self.online = False
        self.ping   = None
        self.note   = "not found"


# ── CSS ───────────────────────────────────────────────────────────────────────
CSS = """
Screen {
    background: $surface;
}

#title-bar {
    height: 1;
    background: $accent;
    color: $text;
    content-align: center middle;
    text-style: bold;
}

#main {
    height: 1fr;
}

#left-panel {
    width: 30;
    border: solid $accent;
    padding: 0 1;
}

#right-panel {
    width: 1fr;
    border: solid $accent;
}

.panel-title {
    background: $accent;
    color: $text;
    text-style: bold;
    width: 100%;
    content-align: center middle;
    height: 1;
}

.cam-card {
    height: 6;
    border: solid $primary-darken-2;
    margin: 1 0;
    padding: 0 1;
}

.cam-card.online {
    border: solid $success;
}

.cam-card.offline {
    border: solid $error-darken-1;
}

.cam-name {
    text-style: bold;
}

.status-online  { color: $success; }
.status-offline { color: $error; }

#services {
    height: 5;
    border: solid $accent;
    padding: 0 1;
}

.service-row {
    height: 3;
    align: left middle;
}

Button {
    margin: 0 1;
    min-width: 12;
}

Button.running {
    background: $success-darken-2;
}

#log {
    height: 1fr;
    border: solid $accent;
}

#bottom-bar {
    height: 1;
    background: $primary-darken-3;
    padding: 0 1;
    color: $text-muted;
}
"""


# ── Main App ──────────────────────────────────────────────────────────────────
class RigManager(App):
    CSS = CSS
    TITLE = "ESP32-CAM XR Rig Manager"
    BINDINGS = [
        Binding("r", "toggle_receiver", "Receiver"),
        Binding("n", "toggle_ntp",      "NTP"),
        Binding("d", "discover",        "Discover"),
        Binding("u", "upload_menu",     "Upload"),
        Binding("s", "scan",            "Scan"),
        Binding("m", "monitor",         "Monitor"),
        Binding("q", "quit",            "Quit"),
    ]

    receiver_running = reactive(False)
    ntp_running      = reactive(False)

    def __init__(self):
        super().__init__()
        self.cams: dict[str, CamState] = {
            "cam1": CamState("cam1"),
            "cam2": CamState("cam2"),
        }
        self._receiver_proc: subprocess.Popen | None = None
        self._ntp_proc: subprocess.Popen | None = None
        self._upload_proc: subprocess.Popen | None = None
        self._monitor_proc: subprocess.Popen | None = None

    # ── Layout ────────────────────────────────────────────────────────────────
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        with Horizontal(id="main"):
            with Vertical(id="left-panel"):
                yield Static("── CAMERAS ──", classes="panel-title")
                yield Static("", id="cam1-card", classes="cam-card offline")
                yield Static("", id="cam2-card", classes="cam-card offline")

                yield Static("── SERVICES ──", classes="panel-title")
                with Horizontal(classes="service-row"):
                    yield Button("▶ Receiver", id="btn-receiver", variant="default")
                    yield Button("▶ NTP",      id="btn-ntp",      variant="default")

                yield Static("── UPLOAD ──", classes="panel-title")
                with Horizontal(classes="service-row"):
                    yield Button("⬆ cam1", id="btn-cam1-ota", variant="primary")
                    yield Button("⬆ cam2", id="btn-cam2-ota", variant="primary")

                yield Static("── TOOLS ──", classes="panel-title")
                with Horizontal(classes="service-row"):
                    yield Button("🔍 Discover", id="btn-discover", variant="warning")
                    yield Button("📡 Scan",     id="btn-scan",     variant="default")

                with Horizontal(classes="service-row"):
                    yield Button("🔌 Monitor",  id="btn-monitor",  variant="default")

            with Vertical(id="right-panel"):
                yield Static("── LOG ──", classes="panel-title")
                yield RichLog(id="log", highlight=True, markup=True)

        yield Footer()

    def on_mount(self) -> None:
        self.log_msg(f"[bold cyan]ESP32-CAM Rig Manager[/] started")
        self.log_msg(f"Laptop IP: [bold]{own_ip()}[/]")
        self._update_cam_cards()
        self.set_interval(30, self._auto_ping)
        self.set_interval(2,  self._poll_procs)
        threading.Thread(target=self._udp_log_listener, daemon=True).start()

    # ── UDP log listener ───────────────────────────────────────────────────────
    def _udp_log_listener(self):
        """Receive CAMx|<message> packets from ESP32s on LOG_PORT and show in log."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("0.0.0.0", LOG_PORT))
            sock.settimeout(1.0)
        except Exception as e:
            self.call_from_thread(self.log_msg, f"[red]UDP log listener failed: {e}[/]")
            return

        cam_colors = {"1": "cyan", "2": "magenta"}
        while True:
            try:
                data, addr = sock.recvfrom(512)
                msg = data.decode(errors="ignore").strip()
                # Expected format: CAM1|<text>  or  CAM2|<text>
                if msg.startswith("CAM") and "|" in msg:
                    cam_id, text = msg[3:].split("|", 1)
                    color = cam_colors.get(cam_id, "white")
                    self.call_from_thread(
                        self.log_msg,
                        f"[[bold {color}]CAM{cam_id}[/]] {text}"
                    )
                else:
                    self.call_from_thread(self.log_msg, f"[dim]{addr[0]}[/] {msg}")
            except socket.timeout:
                continue
            except Exception:
                continue

    # ── Logging ───────────────────────────────────────────────────────────────
    def log_msg(self, msg: str):
        self.query_one("#log", RichLog).write(f"[dim]{now()}[/]  {msg}")

    # ── Camera cards ──────────────────────────────────────────────────────────
    def _update_cam_cards(self):
        for name, cam in self.cams.items():
            card = self.query_one(f"#{name}-card", Static)
            if cam.online:
                ping_str = f"{cam.ping:.0f}ms" if cam.ping else "?"
                card.update(
                    f"[bold green]● {name.upper()}[/]\n"
                    f"  IP:   {cam.ip}\n"
                    f"  Ping: {ping_str}\n"
                    f"  [dim]{cam.note}[/]"
                )
                card.set_class(True, "online")
                card.set_class(False, "offline")
            else:
                card.update(
                    f"[bold red]○ {name.upper()}[/]\n"
                    f"  IP:   {cam.ip or '—'}\n"
                    f"  [dim]{cam.note}[/]"
                )
                card.set_class(False, "online")
                card.set_class(True,  "offline")

    # ── Auto-ping ─────────────────────────────────────────────────────────────
    def _auto_ping(self):
        for cam in self.cams.values():
            if cam.ip:
                rtt = ping_host(cam.ip)
                cam.online = rtt is not None
                cam.ping   = rtt
                cam.note   = "streaming" if cam.online else "unreachable"
        self._update_cam_cards()

    # ── Poll subprocesses ─────────────────────────────────────────────────────
    def _poll_procs(self):
        if self._receiver_proc and self._receiver_proc.poll() is not None:
            self._receiver_proc = None
            self.receiver_running = False
            self.query_one("#btn-receiver", Button).label = "▶ Receiver"
            self.query_one("#btn-receiver", Button).remove_class("running")
            self.log_msg("[yellow]Receiver stopped[/]")

        if self._ntp_proc and self._ntp_proc.poll() is not None:
            self._ntp_proc = None
            self.ntp_running = False
            self.query_one("#btn-ntp", Button).label = "▶ NTP"
            self.query_one("#btn-ntp", Button).remove_class("running")
            self.log_msg("[yellow]NTP server stopped[/]")

        if self._upload_proc and self._upload_proc.poll() is not None:
            self._upload_proc = None   # result already logged by _stream_upload thread

    # ── Actions ───────────────────────────────────────────────────────────────
    def action_toggle_receiver(self):
        self.on_button_pressed(Button.Pressed(self.query_one("#btn-receiver")))

    def action_toggle_ntp(self):
        self.on_button_pressed(Button.Pressed(self.query_one("#btn-ntp")))

    def action_discover(self):
        self.on_button_pressed(Button.Pressed(self.query_one("#btn-discover")))

    def action_upload_menu(self):
        self.log_msg("Use [bold]⬆ cam1[/] or [bold]⬆ cam2[/] buttons to upload")

    def action_scan(self):
        self.on_button_pressed(Button.Pressed(self.query_one("#btn-scan")))

    def action_monitor(self):
        self.on_button_pressed(Button.Pressed(self.query_one("#btn-monitor")))

    # ── Button handler ────────────────────────────────────────────────────────
    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id

        if bid == "btn-receiver":
            self._toggle_receiver()
        elif bid == "btn-ntp":
            self._toggle_ntp()
        elif bid == "btn-cam1-ota":
            self._ota_upload("cam1")
        elif bid == "btn-cam2-ota":
            self._ota_upload("cam2")
        elif bid == "btn-discover":
            self.run_worker(self._discover(), exclusive=True)
        elif bid == "btn-scan":
            self.run_worker(self._scan(), exclusive=True)
        elif bid == "btn-monitor":
            self._open_monitor()

    # ── Receiver ──────────────────────────────────────────────────────────────
    def _toggle_receiver(self):
        btn = self.query_one("#btn-receiver", Button)
        if self._receiver_proc and self._receiver_proc.poll() is None:
            self._receiver_proc.terminate()
            self._receiver_proc = None
            self.receiver_running = False
            btn.label = "▶ Receiver"
            btn.remove_class("running")
            self.log_msg("[yellow]Receiver stopped[/]")
        else:
            self._receiver_proc = subprocess.Popen(
                [str(PYTHON), str(RECEIVER)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            self.receiver_running = True
            btn.label = "■ Receiver"
            btn.add_class("running")
            self.log_msg(f"[green]Receiver started[/] → [link]http://localhost:8080[/link]")

    # ── NTP ───────────────────────────────────────────────────────────────────
    def _toggle_ntp(self):
        btn = self.query_one("#btn-ntp", Button)
        if self._ntp_proc and self._ntp_proc.poll() is None:
            self._ntp_proc.terminate()
            self._ntp_proc = None
            self.ntp_running = False
            btn.label = "▶ NTP"
            btn.remove_class("running")
            self.log_msg("[yellow]NTP stopped[/]")
        else:
            self._ntp_proc = subprocess.Popen(
                ["sudo", str(PYTHON), str(NTP_SRV)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            self.ntp_running = True
            btn.label = "■ NTP"
            btn.add_class("running")
            self.log_msg("[green]NTP server started[/] on UDP:123")

    # ── OTA upload ────────────────────────────────────────────────────────────
    def _ota_upload(self, cam: str):
        if self._upload_proc and self._upload_proc.poll() is None:
            self.log_msg("[yellow]Upload already in progress — wait for it to finish[/]")
            return

        cam_state = self.cams[cam]
        if not cam_state.ip:
            self.log_msg(f"[red]{cam} IP unknown — run Discover first[/]")
            return

        self.log_msg(f"[cyan]OTA uploading to {cam} ({cam_state.ip})...[/]")
        self._upload_proc = subprocess.Popen(
            [str(PIO), "run", "-e", f"{cam}_ota", "--target", "upload"],
            cwd=ROOT / "esp32cam-stream",
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        threading.Thread(
            target=self._stream_upload,
            args=(self._upload_proc, cam),
            daemon=True
        ).start()

    def _stream_upload(self, proc: subprocess.Popen, cam: str):
        """Read espota output byte-by-byte, show progress bar every 10%."""
        last_pct = -1
        buf = ""

        while True:
            ch = proc.stdout.read(1)
            if not ch:
                break
            c = ch.decode(errors="ignore")
            if c in ("\n", "\r"):
                line = buf.strip()
                buf = ""
                if not line:
                    continue

                # Progress bar: "Uploading: [====   ] 45%"
                if "%" in line:
                    m = re.search(r"(\d+)%", line)
                    if m:
                        pct = int(m.group(1))
                        # Report every 10% step (and 100%)
                        if pct >= last_pct + 10 or pct == 100:
                            last_pct = pct
                            filled = int(20 * pct / 100)
                            bar = "█" * filled + "░" * (20 - filled)
                            self.call_from_thread(
                                self.log_msg,
                                f"[cyan]  [{bar}] {pct}%[/]"
                            )
                elif "Sending invitation" in line:
                    self.call_from_thread(
                        self.log_msg, f"[dim]  Connecting to {cam}...[/]"
                    )
                elif "Upload size:" in line:
                    self.call_from_thread(self.log_msg, f"[dim]  {line}[/]")
                elif "[ERROR]" in line or "Error" in line:
                    self.call_from_thread(self.log_msg, f"[red]  {line}[/]")
                # Skip verbose DEBUG/INFO lines from espota
            else:
                buf += c

        code = proc.wait()
        if code == 0:
            self.call_from_thread(self.log_msg, f"[green]✓ OTA {cam} upload SUCCESS[/]")
        else:
            self.call_from_thread(
                self.log_msg, f"[red]✗ OTA {cam} upload FAILED (exit {code})[/]"
            )

    # ── Discover ──────────────────────────────────────────────────────────────
    async def _discover(self):
        self.log_msg("[cyan]Discovering cameras (beacon + subnet scan)...[/]")
        found: dict[str, str] = {}

        # Phase 1: broadcast beacon + listen for CAM announcements (12s)
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("0.0.0.0", DISCOVERY_PORT))
            sock.settimeout(1.0)
            beacon = f"LAPTOP:{own_ip()}".encode()
            deadline = time.monotonic() + 12

            while time.monotonic() < deadline:
                sock.sendto(beacon, ("255.255.255.255", DISCOVERY_PORT))
                try:
                    data, addr = sock.recvfrom(64)
                    txt = data.decode(errors="ignore").strip()
                    if txt.startswith("CAM:"):
                        cam_id = "cam" + txt[4:]
                        if cam_id in self.cams and cam_id not in found:
                            found[cam_id] = addr[0]
                            self.log_msg(f"[green]Beacon: {cam_id} → {addr[0]}[/]")
                except socket.timeout:
                    pass
                await asyncio.sleep(0.1)
            sock.close()
        except Exception as e:
            self.log_msg(f"[yellow]Beacon phase error: {e}[/]")

        # Phase 2: subnet scan for any ESPs not caught by beacon
        if len(found) < 2:
            self.log_msg("[cyan]Scanning subnet for remaining cameras...[/]")
            loop = asyncio.get_event_loop()
            scanned = await loop.run_in_executor(None, scan_for_esps)
            for cam, ip in scanned.items():
                if cam not in found:
                    found[cam] = ip
                    self.log_msg(f"[green]Scan: {cam} → {ip}[/]")

        if not found:
            self.log_msg("[red]No cameras found[/]")
            return

        # Update state + platformio.ini
        for cam, ip in found.items():
            self.cams[cam].ip     = ip
            self.cams[cam].online = True
            self.cams[cam].note   = "discovered"
            if update_pio_ini(cam, ip):
                self.log_msg(f"[green]platformio.ini updated: {cam}_ota → {ip}[/]")

        self._update_cam_cards()

    # ── Scan ──────────────────────────────────────────────────────────────────
    async def _scan(self):
        self.log_msg("[cyan]Scanning subnet...[/]")
        loop  = asyncio.get_event_loop()
        found = await loop.run_in_executor(None, scan_for_esps)

        if not found:
            self.log_msg("[red]No ESP32-CAMs found on subnet[/]")
            return

        for cam, ip in found.items():
            rtt = ping_host(ip)
            self.cams[cam].ip     = ip
            self.cams[cam].online = rtt is not None
            self.cams[cam].ping   = rtt
            self.cams[cam].note   = "found via scan"
            self.log_msg(f"[green]{cam}[/] → {ip}  ping={rtt:.0f}ms")

        self._update_cam_cards()

    # ── Serial monitor ────────────────────────────────────────────────────────
    def _open_monitor(self):
        if not Path(SERIAL_DEV).exists():
            self.log_msg(f"[red]{SERIAL_DEV} not found — is USB-serial connected?[/]")
            return
        # Open in a new terminal window
        subprocess.Popen([
            "bash", "-c",
            f"{PIO} device monitor --port {SERIAL_DEV} --baud 115200"
            f" || read -p 'Press enter to close'"
        ])
        self.log_msg(f"[cyan]Serial monitor opened on {SERIAL_DEV}[/]")


if __name__ == "__main__":
    RigManager().run()
