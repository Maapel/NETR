"""
NETR — ESP32-CAM XR Rig Manager TUI
Run: /home/maadhav/pio-venv/bin/python tui.py
"""

import asyncio
import re
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import datetime
from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import Button, Footer, RichLog, Static

# ── Pupil detection availability check ────────────────────────────────────────
# Actual detection runs in receiver.py; we just check cv2 is importable here.
try:
    import cv2 as _cv2_check  # noqa: F401
    _PUPIL_OK = True
except Exception:
    _PUPIL_OK = False

# ── Project paths ──────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent
PIO        = Path("/home/maadhav/pio-venv/bin/pio")
PYTHON     = Path("/home/maadhav/pio-venv/bin/python")
PIO_INI    = ROOT / "esp32cam-stream" / "platformio.ini"
RECEIVER   = ROOT / "receiver.py"
ENGINE     = ROOT / "compute" / "engine.py"
CALIBRATION = ROOT / "calibration_server.py"
NTP_SRV    = ROOT / "ntp_server.py"
SERIAL_DEV = "/dev/ttyUSB0"

DISCOVERY_PORT = 5004
LOG_PORT       = 5010
ESP32_MACS = {
    "c4:dd:57:ea:28:5c": "cam1",
    "c4:dd:57:ea:3d:84": "cam2",
}

# ── ASCII banner ───────────────────────────────────────────────────────────────
BANNER = """\
  ╭──────────────────────────────────────────────────────────────────────────╮
 ╱  ╭───╮                                                         ╭───╮     ╲
│  ╱  ◉  ╲        [bold white]N  ·  E  ·  T  ·  R[/]   [dim]·   ESP32-CAM  XR  RIG[/]       ╱  ◉  ╲   │
 ╲  ╰───╯                                                         ╰───╯     ╱
  ╰──────────────────────────────────────────────────────────────────────────╯\
"""


# ── Helpers ───────────────────────────────────────────────────────────────────
def now() -> str:
    return datetime.now().strftime("%H:%M:%S")

def own_ip() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]

def subnet_base() -> str:
    return ".".join(own_ip().split(".")[:3])

def ping_host(ip: str) -> float | None:
    try:
        r = subprocess.run(["ping", "-c1", "-W1", ip], capture_output=True, timeout=2)
        if r.returncode == 0:
            m = re.search(r"time=([\d.]+)", r.stdout.decode())
            return float(m.group(1)) if m else 0.0
    except Exception:
        pass
    return None

def scan_for_esps() -> dict[str, str]:
    base = subnet_base()
    results: dict[str, str] = {}
    def check(i: int):
        ip = f"{base}.{i}"
        if ping_host(ip) is not None:
            r = subprocess.run(["ip", "neigh", "show", ip], capture_output=True, text=True)
            for mac, cam in ESP32_MACS.items():
                if mac in r.stdout.lower():
                    results[cam] = ip
    threads = [threading.Thread(target=check, args=(i,)) for i in range(1, 255)]
    for t in threads: t.start()
    for t in threads: t.join()
    return results

def update_pio_ini(cam: str, ip: str) -> bool:
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
Screen { background: $surface; }

#banner {
    height: 5;
    background: $panel;
    color: $accent;
    padding: 0 1;
    text-align: center;
}

#main { height: 1fr; }

#left-panel {
    width: 32;
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
    height: 5;
    border: solid $primary-darken-2;
    margin: 0 0 1 0;
    padding: 0 1;
}
.cam-card.online  { border: solid $success; }
.cam-card.offline { border: solid $error-darken-1; }

.service-row {
    height: 3;
    align: left middle;
}

Button {
    margin: 0 1;
    min-width: 10;
}
Button.running { background: $success-darken-2; }
Button.usb     { background: $warning-darken-2; }

#log {
    height: 1fr;
    border: solid $accent;
}

.log-filters {
    height: 3;
    align: left middle;
}
.log-filters Button {
    min-width: 6;
    margin: 0;
}
.log-filters Button.filter-active {
    background: $accent;
    color: $text;
    text-style: bold;
}

.pupil-row {
    height: 1;
    padding: 0 1;
    color: $text-muted;
}

#analysis-section {
    height: 5;
    border: solid $accent;
    padding: 0 1;
}
"""


# ── Main App ───────────────────────────────────────────────────────────────────
class RigManager(App):
    CSS = CSS
    TITLE = "NETR"
    BINDINGS = [
        Binding("r",     "toggle_receiver", "Receiver"),
        Binding("d",     "discover",        "Discover"),
        Binding("s",     "scan",            "Scan"),
        Binding("m",     "monitor",         "Monitor"),
        Binding("q",     "quit",            "Quit"),
    ]

    receiver_running    = reactive(False)
    engine_running      = reactive(False)
    calibration_running = reactive(False)
    analysis_enabled    = reactive(False)

    LOG_GROUPS = ("all", "cam1", "cam2", "ota", "sys")

    def __init__(self):
        super().__init__()
        self.cams: dict[str, CamState] = {
            "cam1": CamState("cam1"),
            "cam2": CamState("cam2"),
        }
        self._receiver_proc:    subprocess.Popen | None = None
        self._engine_proc:      subprocess.Popen | None = None
        self._calibration_proc: subprocess.Popen | None = None
        self._upload_procs:  dict[str, subprocess.Popen] = {}  # keyed by cam id
        self._log_entries: list[tuple[str, str]] = []  # (group, rendered_msg)
        self._log_filter: str = "all"

    # ── Layout ────────────────────────────────────────────────────────────────
    def compose(self) -> ComposeResult:
        yield Static(BANNER, id="banner", markup=True)

        with Horizontal(id="main"):
            with Vertical(id="left-panel"):

                yield Static("── CAMERAS ──", classes="panel-title")
                yield Static("", id="cam1-card", classes="cam-card offline")
                yield Static("", id="cam2-card", classes="cam-card offline")

                yield Static("── SERVICES ──", classes="panel-title")
                with Horizontal(classes="service-row"):
                    yield Button("▶ Receiver",    id="btn-receiver")
                with Horizontal(classes="service-row"):
                    yield Button("▶ Engine",      id="btn-engine")
                with Horizontal(classes="service-row"):
                    yield Button("▶ Calibration", id="btn-calibration")

                yield Static("── OTA UPLOAD ──", classes="panel-title")
                with Horizontal(classes="service-row"):
                    yield Button("⬆ cam1", id="btn-cam1-ota", variant="primary")
                    yield Button("⬆ cam2", id="btn-cam2-ota", variant="primary")

                yield Static("── RESET ──", classes="panel-title")
                with Horizontal(classes="service-row"):
                    yield Button("↺ cam1", id="btn-cam1-rst", variant="error")
                    yield Button("↺ cam2", id="btn-cam2-rst", variant="error")

                yield Static("── USB FLASH ──", classes="panel-title")
                with Horizontal(classes="service-row"):
                    yield Button("⚡ cam1", id="btn-cam1-usb", classes="usb")
                    yield Button("⚡ cam2", id="btn-cam2-usb", classes="usb")

                yield Static("── TOOLS ──", classes="panel-title")
                with Horizontal(classes="service-row"):
                    yield Button("⟳ Discover", id="btn-discover", variant="warning")
                    yield Button("⊙ Scan",     id="btn-scan")
                with Horizontal(classes="service-row"):
                    yield Button("⌥ Monitor",  id="btn-monitor")

            with Vertical(id="right-panel"):
                yield Static("── LOG ──", classes="panel-title")
                with Horizontal(classes="log-filters"):
                    yield Button("All", id="flt-all", classes="filter-active")
                    yield Button("CAM1", id="flt-cam1")
                    yield Button("CAM2", id="flt-cam2")
                    yield Button("OTA", id="flt-ota")
                    yield Button("Sys", id="flt-sys")
                yield RichLog(id="log", highlight=True, markup=True)
                yield Static("── PUPIL ANALYSIS ──", classes="panel-title")
                with Vertical(id="analysis-section"):
                    with Horizontal(classes="service-row"):
                        yield Button("⊙ Overlay OFF", id="btn-analysis", variant="default")
                    yield Static(
                        "[dim]cv2/numpy missing[/]" if not _PUPIL_OK else "[dim]press to enable pupil overlay in browser[/]",
                        id="analysis-status", classes="pupil-row", markup=True
                    )

        yield Footer()

    def on_mount(self) -> None:
        self._update_cam_cards()
        self.log_msg("[bold cyan]NETR[/] started", "sys")
        self.log_msg(f"Laptop IP: [bold]{own_ip()}[/]", "sys")
        self.set_interval(30, self._auto_ping)
        self.set_interval(2,  self._poll_procs)
        threading.Thread(target=self._udp_log_listener, daemon=True).start()
        threading.Thread(target=self._timesync_server,  daemon=True).start()

    def on_unmount(self) -> None:
        for proc in (self._receiver_proc, self._engine_proc, self._calibration_proc):
            if proc and proc.poll() is None:
                proc.terminate()
        for p in self._upload_procs.values():
            if p.poll() is None:
                p.terminate()

    # ── Timesync responder ────────────────────────────────────────────────────
    def _timesync_server(self):
        """Respond to SYNC_REQ from ESP32s so round-trip time sync works
        even when receiver.py is not running."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("0.0.0.0", 5005))
        except Exception as e:
            self.call_from_thread(self.log_msg, f"[red]Timesync server failed: {e}[/]")
            return
        while True:
            try:
                data, addr = sock.recvfrom(128)
                t2 = int(time.time() * 1e6)
                msg = data.decode(errors="ignore").strip()
                if msg.startswith("SYNC_REQ:"):
                    parts = msg[9:].split(":", 1)
                    if len(parts) == 2:
                        t3 = int(time.time() * 1e6)
                        sock.sendto(f"SYNC_RESP:{parts[1]}:{t2}:{t3}".encode(), addr)
            except Exception:
                continue

    # ── Pupil overlay toggle ──────────────────────────────────────────────────
    def _toggle_analysis(self):
        if not _PUPIL_OK:
            self.log_msg("[red]Pupil analysis unavailable — cv2/numpy missing[/]"); return
        if not self.receiver_running:
            self.log_msg("[yellow]Start receiver first[/]"); return
        self.analysis_enabled = not self.analysis_enabled
        val = "1" if self.analysis_enabled else "0"
        try:
            urllib.request.urlopen(
                f"http://localhost:8080/set?analysis={val}", timeout=1
            ).close()
        except Exception:
            pass
        btn = self.query_one("#btn-analysis", Button)
        if self.analysis_enabled:
            btn.label = "⊙ Overlay ON"
            btn.variant = "success"
            self.log_msg("[green]Pupil overlay enabled[/]")
        else:
            btn.label = "⊙ Overlay OFF"
            btn.variant = "default"
            self.log_msg("[dim]Pupil overlay disabled[/]")

    # ── UDP log listener ──────────────────────────────────────────────────────
    def _udp_log_listener(self):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("0.0.0.0", LOG_PORT))
            sock.settimeout(1.0)
        except Exception as e:
            self.call_from_thread(self.log_msg, f"[red]UDP log listener failed: {e}[/]")
            return

        while True:
            try:
                data, addr = sock.recvfrom(512)
                msg = data.decode(errors="ignore").strip()
                if msg.startswith("CAM") and "|" in msg:
                    cam_id, text = msg[3:].split("|", 1)
                    group = f"cam{cam_id}"
                    self.call_from_thread(self.log_msg, text, group)
                else:
                    self.call_from_thread(self.log_msg, f"[dim]{addr[0]}[/] {msg}", "sys")
            except socket.timeout:
                continue
            except Exception:
                continue

    # ── Logging ───────────────────────────────────────────────────────────────
    def log_msg(self, msg: str, group: str = "sys"):
        tag_colors = {"cam1": "cyan", "cam2": "magenta", "ota": "yellow", "sys": "dim white"}
        color = tag_colors.get(group, "white")
        tag = f"[{color}]{group.upper():>4}[/]"
        rendered = f"[dim]{now()}[/] {tag}  {msg}"
        self._log_entries.append((group, rendered))
        if self._log_filter == "all" or self._log_filter == group:
            self.query_one("#log", RichLog).write(rendered)

    def _refilter_log(self):
        log = self.query_one("#log", RichLog)
        log.clear()
        for group, rendered in self._log_entries:
            if self._log_filter == "all" or self._log_filter == group:
                log.write(rendered)

    # ── Camera cards ──────────────────────────────────────────────────────────
    def _update_cam_cards(self):
        for name, cam in self.cams.items():
            card = self.query_one(f"#{name}-card", Static)
            if cam.online:
                ping_str = f"{cam.ping:.0f}ms" if cam.ping else "?"
                card.update(
                    f"[bold green]● {name.upper()}[/]\n"
                    f"  IP:   {cam.ip}\n"
                    f"  Ping: {ping_str}  [dim]{cam.note}[/]"
                )
                card.set_class(True, "online"); card.set_class(False, "offline")
            else:
                card.update(
                    f"[bold red]○ {name.upper()}[/]\n"
                    f"  IP:   {cam.ip or '—'}\n"
                    f"  [dim]{cam.note}[/]"
                )
                card.set_class(False, "online"); card.set_class(True, "offline")

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
            btn = self.query_one("#btn-receiver", Button)
            btn.label = "▶ Receiver"; btn.remove_class("running")
            self.log_msg("[yellow]Receiver stopped[/]", "sys")

        if self._engine_proc and self._engine_proc.poll() is not None:
            self._engine_proc = None
            self.engine_running = False
            btn = self.query_one("#btn-engine", Button)
            btn.label = "▶ Engine"; btn.remove_class("running")
            self.log_msg("[yellow]Engine stopped[/]", "sys")

        if self._calibration_proc and self._calibration_proc.poll() is not None:
            self._calibration_proc = None
            self.calibration_running = False
            btn = self.query_one("#btn-calibration", Button)
            btn.label = "▶ Calibration"; btn.remove_class("running")
            self.log_msg("[yellow]Calibration server stopped[/]", "sys")

        done = [cam for cam, p in self._upload_procs.items() if p.poll() is not None]
        for cam in done:
            del self._upload_procs[cam]

    # ── Actions ───────────────────────────────────────────────────────────────
    def action_toggle_receiver(self):
        self.on_button_pressed(Button.Pressed(self.query_one("#btn-receiver")))

    def action_discover(self):
        self.on_button_pressed(Button.Pressed(self.query_one("#btn-discover")))

    def action_scan(self):
        self.on_button_pressed(Button.Pressed(self.query_one("#btn-scan")))

    def action_monitor(self):
        self.on_button_pressed(Button.Pressed(self.query_one("#btn-monitor")))

    # ── Button handler ────────────────────────────────────────────────────────
    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id
        if   bid == "btn-receiver":    self._toggle_receiver()
        elif bid == "btn-engine":      self._toggle_engine()
        elif bid == "btn-calibration": self._toggle_calibration()
        elif bid == "btn-cam1-ota":    self._ota_upload("cam1")
        elif bid == "btn-cam2-ota":  self._ota_upload("cam2")
        elif bid == "btn-cam1-rst":  self._reset_cam("cam1")
        elif bid == "btn-cam2-rst":  self._reset_cam("cam2")
        elif bid == "btn-cam1-usb":  self._usb_flash("cam1")
        elif bid == "btn-cam2-usb":  self._usb_flash("cam2")
        elif bid == "btn-analysis":  self._toggle_analysis()
        elif bid == "btn-discover":  self.run_worker(self._discover(), exclusive=True)
        elif bid == "btn-scan":      self.run_worker(self._scan(),    exclusive=True)
        elif bid == "btn-monitor":   self._open_monitor()
        elif bid and bid.startswith("flt-"):
            group = bid[4:]  # "all", "cam1", "cam2", "ota", "sys"
            self._log_filter = group
            for g in self.LOG_GROUPS:
                btn = self.query_one(f"#flt-{g}", Button)
                if g == group:
                    btn.add_class("filter-active")
                else:
                    btn.remove_class("filter-active")
            self._refilter_log()

    # ── Receiver ──────────────────────────────────────────────────────────────
    def _toggle_receiver(self):
        btn = self.query_one("#btn-receiver", Button)
        if self._receiver_proc and self._receiver_proc.poll() is None:
            self._receiver_proc.terminate()
            self._receiver_proc = None
            self.receiver_running = False
            btn.label = "▶ Receiver"; btn.remove_class("running")
            self.log_msg("[yellow]Receiver stopped[/]", "sys")
        else:
            self._receiver_proc = subprocess.Popen(
                [str(PYTHON), str(RECEIVER)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            self.receiver_running = True
            btn.label = "■ Receiver"; btn.add_class("running")
            self.log_msg("[green]Receiver started[/] → http://localhost:8080", "sys")

    # ── Engine ────────────────────────────────────────────────────────────────
    def _toggle_engine(self):
        btn = self.query_one("#btn-engine", Button)
        if self._engine_proc and self._engine_proc.poll() is None:
            self._engine_proc.terminate()
            self._engine_proc = None
            self.engine_running = False
            btn.label = "▶ Engine"; btn.remove_class("running")
            self.log_msg("[yellow]Engine stopped[/]", "sys")
        else:
            self._engine_proc = subprocess.Popen(
                [str(PYTHON), str(ENGINE)],
                cwd=ROOT / "compute",
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            self.engine_running = True
            btn.label = "■ Engine"; btn.add_class("running")
            self.log_msg("[green]Engine started[/] → http://localhost:8081", "sys")

    # ── Calibration ───────────────────────────────────────────────────────────
    def _toggle_calibration(self):
        btn = self.query_one("#btn-calibration", Button)
        if self._calibration_proc and self._calibration_proc.poll() is None:
            self._calibration_proc.terminate()
            self._calibration_proc = None
            self.calibration_running = False
            btn.label = "▶ Calibration"; btn.remove_class("running")
            self.log_msg("[yellow]Calibration server stopped[/]", "sys")
        else:
            self._calibration_proc = subprocess.Popen(
                [str(PYTHON), str(CALIBRATION)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            self.calibration_running = True
            btn.label = "■ Calibration"; btn.add_class("running")
            self.log_msg("[green]Calibration started[/] → http://localhost:8090", "sys")

    # ── Software reset ─────────────────────────────────────────────────────────
    _CMD_PORTS = {"cam1": 5001, "cam2": 5003}

    def _reset_cam(self, cam: str):
        cam_state = self.cams[cam]
        if not cam_state.ip:
            self.log_msg(f"[red]{cam} IP unknown — run Discover first[/]", "sys"); return
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.sendto(b"rst", (cam_state.ip, self._CMD_PORTS[cam]))
        sock.close()
        self.log_msg(f"[yellow]Reset sent to {cam} ({cam_state.ip})[/]", "sys")

    # ── OTA upload ────────────────────────────────────────────────────────────
    def _ota_upload(self, cam: str):
        if cam in self._upload_procs and self._upload_procs[cam].poll() is None:
            self.log_msg(f"[yellow]{cam} OTA already in progress[/]", "ota"); return

        cam_state = self.cams[cam]
        if not cam_state.ip:
            self.log_msg(f"[red]{cam} IP unknown — run Discover first[/]", "ota"); return

        self.log_msg(f"[cyan]OTA → {cam} ({cam_state.ip})[/]", "ota")
        proc = subprocess.Popen(
            [str(PIO), "run", "-e", f"{cam}_ota", "--target", "upload"],
            cwd=ROOT / "esp32cam-stream",
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        self._upload_procs[cam] = proc
        threading.Thread(
            target=self._stream_upload, args=(proc, cam, "OTA"),
            daemon=True
        ).start()

    # ── USB flash ─────────────────────────────────────────────────────────────
    def _usb_flash(self, cam: str):
        if cam in self._upload_procs and self._upload_procs[cam].poll() is None:
            self.log_msg(f"[yellow]{cam} upload already in progress[/]", "ota"); return

        if not Path(SERIAL_DEV).exists():
            self.log_msg(
                f"[red]{SERIAL_DEV} not found[/] — connect USB-serial, "
                f"hold IO0→GND then press RST to enter flash mode"
            )
            return

        self.log_msg(
            f"[yellow]USB flash → {cam} on {SERIAL_DEV}[/]  "
            f"[dim](IO0→GND + RST to enter flash mode)[/]"
        )
        proc = subprocess.Popen(
            [str(PIO), "run", "-e", cam, "--target", "upload",
             "--upload-port", SERIAL_DEV],
            cwd=ROOT / "esp32cam-stream",
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        self._upload_procs[cam] = proc
        threading.Thread(
            target=self._stream_upload, args=(proc, cam, "USB"),
            daemon=True
        ).start()

    def _stream_upload(self, proc: subprocess.Popen, cam: str, method: str):
        last_pct = -1
        buf = ""
        while True:
            ch = proc.stdout.read(1)
            if not ch:
                break
            c = ch.decode(errors="ignore")
            if c in ("\n", "\r"):
                line = buf.strip(); buf = ""
                if not line:
                    continue
                if "%" in line:
                    m = re.search(r"(\d+)%", line)
                    if m:
                        pct = int(m.group(1))
                        if pct >= last_pct + 10 or pct == 100:
                            last_pct = pct
                            filled = int(20 * pct / 100)
                            bar = "█" * filled + "░" * (20 - filled)
                            self.call_from_thread(
                                self.log_msg, f"[cyan]  [{bar}] {pct}%[/]", "ota"
                            )
                elif "Sending invitation" in line:
                    self.call_from_thread(self.log_msg, f"[dim]  Connecting to {cam}...[/]", "ota")
                elif "Upload size:" in line or "Writing at" in line[:20]:
                    self.call_from_thread(self.log_msg, f"[dim]  {line}[/]", "ota")
                elif "[ERROR]" in line or "Error" in line or "Failed" in line:
                    self.call_from_thread(self.log_msg, f"[red]  {line}[/]", "ota")
                elif "Chip is" in line or "Flash params" in line:
                    self.call_from_thread(self.log_msg, f"[dim]  {line}[/]", "ota")
            else:
                buf += c

        code = proc.wait()
        if code == 0:
            self.call_from_thread(self.log_msg, f"[green]✓ {method} {cam} SUCCESS[/]", "ota")
        else:
            self.call_from_thread(self.log_msg, f"[red]✗ {method} {cam} FAILED (exit {code})[/]", "ota")

    # ── Discover ──────────────────────────────────────────────────────────────
    async def _discover(self):
        self.log_msg("[cyan]Discovering cameras...[/]", "sys")
        found: dict[str, str] = {}

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("0.0.0.0", DISCOVERY_PORT))
            sock.settimeout(1.0)
            deadline = time.monotonic() + 12
            while time.monotonic() < deadline:
                beacon = f"LAPTOP:{own_ip()}:{int(time.time()*1e6):.0f}".encode()
                sock.sendto(beacon, ("255.255.255.255", DISCOVERY_PORT))
                try:
                    data, addr = sock.recvfrom(64)
                    txt = data.decode(errors="ignore").strip()
                    if txt.startswith("CAM:"):
                        cam_id = "cam" + txt[4:]
                        if cam_id in self.cams and cam_id not in found:
                            found[cam_id] = addr[0]
                            self.log_msg(f"[green]Beacon: {cam_id} → {addr[0]}[/]", "sys")
                except socket.timeout:
                    pass
                await asyncio.sleep(0.1)
            sock.close()
        except Exception as e:
            self.log_msg(f"[yellow]Beacon error: {e}[/]", "sys")

        if len(found) < 2:
            self.log_msg("[cyan]Subnet scan for remaining cameras...[/]", "sys")
            loop    = asyncio.get_event_loop()
            scanned = await loop.run_in_executor(None, scan_for_esps)
            for cam, ip in scanned.items():
                if cam not in found:
                    found[cam] = ip
                    self.log_msg(f"[green]Scan: {cam} → {ip}[/]", "sys")

        if not found:
            self.log_msg("[red]No cameras found[/]", "sys"); return

        for cam, ip in found.items():
            self.cams[cam].ip     = ip
            self.cams[cam].online = True
            self.cams[cam].note   = "discovered"
            if update_pio_ini(cam, ip):
                self.log_msg(f"[dim]platformio.ini: {cam}_ota → {ip}[/]", "sys")

        self._update_cam_cards()

    # ── Scan ──────────────────────────────────────────────────────────────────
    async def _scan(self):
        self.log_msg("[cyan]Scanning subnet...[/]", "sys")
        loop  = asyncio.get_event_loop()
        found = await loop.run_in_executor(None, scan_for_esps)
        if not found:
            self.log_msg("[red]No ESP32-CAMs found[/]", "sys"); return
        for cam, ip in found.items():
            rtt = ping_host(ip)
            self.cams[cam].ip     = ip
            self.cams[cam].online = rtt is not None
            self.cams[cam].ping   = rtt
            self.cams[cam].note   = "found via scan"
            self.log_msg(f"[green]{cam}[/] → {ip}  {rtt:.0f}ms", "sys")
        self._update_cam_cards()

    # ── Serial monitor ────────────────────────────────────────────────────────
    def _open_monitor(self):
        if not Path(SERIAL_DEV).exists():
            self.log_msg(f"[red]{SERIAL_DEV} not found[/]"); return
        subprocess.Popen([
            "bash", "-c",
            f"{PIO} device monitor --port {SERIAL_DEV} --baud 115200"
            " || read -p 'Press enter to close'"
        ])
        self.log_msg(f"[cyan]Serial monitor on {SERIAL_DEV}[/]")


if __name__ == "__main__":
    RigManager().run()
