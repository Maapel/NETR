"""
Compute Engine — port 8081

Single HTTP server responsible for all eye analysis:
  - Pupil + glint detection (PCCR vector)
  - Gaze mapping (polynomial model)
  - Debug settings (pipeline parameters)

API
───
POST /process          JPEG bytes in body → runs pipeline, stores result
GET  /result           Latest {pccr_vector, pupil, glint, gaze, ts}
GET  /frame            Latest annotated JPEG (for debug display)
GET  /settings         All pipeline params as JSON
POST /settings         Update params (JSON body or query string)
GET  /gaze_model       Current gaze model coefficients
POST /gaze_model/load  Reload model from disk (optional JSON body {"path": "relative/or/abs under project"})

Receiver calls POST /process for every frame when analysis is ON.
Calibration server polls the receiver GET /stats for pccr_vector and pccr_ts_ms.
Debug UI calls GET/POST /settings.
"""

import json
import sys
import time
import threading
import pathlib
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

import cv2
import numpy as np

from eye_pipeline import EyePipeline, EyeResult
from gaze_model import GazeModel

# ── Paths ─────────────────────────────────────────────────────────────────────
_DIR = pathlib.Path(__file__).parent
GAZE_MODEL_PATH   = _DIR.parent / "gaze_model.json"
EYE_SETTINGS_PATH = _DIR.parent / "eye_settings.json"
_PROJECT_ROOT     = GAZE_MODEL_PATH.parent.resolve()


def _resolve_gaze_model_load_path(path_s: str | None) -> pathlib.Path | None:
    """Return path to load, or None if path escapes project root."""
    if path_s is None or not str(path_s).strip():
        return GAZE_MODEL_PATH.resolve()
    raw = pathlib.Path(str(path_s).strip())
    cand = raw.resolve() if raw.is_absolute() else (_PROJECT_ROOT / raw).resolve()
    try:
        cand.relative_to(_PROJECT_ROOT)
    except ValueError:
        return None
    return cand

# ── Rig config ────────────────────────────────────────────────────────────────
sys.path.insert(0, str(_DIR.parent))
try:
    import rig_config as _rig_cfg
    EYE_CAM_ID = _rig_cfg.eye_cam()
except Exception:
    EYE_CAM_ID = 2

# ── Pipeline ──────────────────────────────────────────────────────────────────
_pipe = EyePipeline()
_pipe_lock = threading.Lock()

def _load_settings() -> dict:
    try:
        with open(EYE_SETTINGS_PATH) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def _save_settings():
    data = _pipe.get_params()
    data["glint_switch_dx"] = _pipe.switch_dx
    with open(EYE_SETTINGS_PATH, "w") as f:
        json.dump(data, f, indent=2)

_boot_settings = _load_settings()
with _pipe_lock:
    _pipe.update_params(_boot_settings)
    _pipe.switch_dx = float(_boot_settings.get("glint_switch_dx", 0.0))

# ── Glint sweep state ─────────────────────────────────────────────────────────
_sweep_lock    = threading.Lock()
_sweep_samples: list[dict] = []   # {"dx": float, "glint_offset": float}
_sweep_active  = False

# ── Gaze model ────────────────────────────────────────────────────────────────
_gaze_model = GazeModel()
_gaze_model.load(GAZE_MODEL_PATH)

# ── Latest result ─────────────────────────────────────────────────────────────
_latest_lock   = threading.Lock()
_latest_result: EyeResult | None = None
_latest_frame:  bytes | None = None   # annotated JPEG
_latest_ts:     float = 0.0
_latest_gaze:   tuple[float, float] | None = None

# ── Debug view ────────────────────────────────────────────────────────────────
_debug_view = "original"   # "original" | "p_suppressed" | "p_blurred" | "p_thresh" | "p_morph" | ...


def _process(jpeg: bytes) -> bytes:
    """Run full pipeline on a JPEG frame. Returns annotated JPEG."""
    global _latest_result, _latest_frame, _latest_ts, _latest_gaze

    arr = np.frombuffer(jpeg, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        return jpeg

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    with _pipe_lock:
        result = _pipe.process(gray)

    # Gaze mapping
    gaze = None
    if result.pccr_vector and _gaze_model.trained:
        try:
            gaze = _gaze_model.predict(*result.pccr_vector, result.pccr_side)
        except Exception:
            pass

    # Choose output frame
    if _debug_view != "original" and _debug_view in result.intermediate_frames:
        dbg = result.intermediate_frames[_debug_view]
        if len(dbg.shape) == 2:
            dbg = cv2.cvtColor(dbg, cv2.COLOR_GRAY2BGR)
        out = dbg
    else:
        out = EyePipeline.draw(bgr, result)

    _, enc = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 85])
    annotated = enc.tobytes()

    # Collect sweep samples while active
    with _sweep_lock:
        collecting = _sweep_active
    if collecting and result.pccr_vector is not None:
        dx = result.pccr_vector[0]
        # glint_offset = glint_x - pupil_x (positive = glint right of pupil)
        if result.glint_pos and result.pupil_center:
            glint_offset = float(result.glint_pos[0]) - float(result.pupil_center[0])
            with _sweep_lock:
                _sweep_samples.append({"dx": dx, "glint_offset": glint_offset})

    with _latest_lock:
        _latest_result = result
        _latest_frame  = annotated
        _latest_ts     = time.time()
        _latest_gaze   = gaze

    return annotated


# ── Settings helpers ──────────────────────────────────────────────────────────
# Eye pipeline parameter ranges for validation (mirrors receiver.py)
_EYE_PARAM_RANGES = {
    "p_glint_thresh":         (10,  255),
    "p_blur_ksize":           (3,   21),
    "p_dark_percentile":      (0.0, 100.0),
    "p_thresh_offset":        (0,   100),
    "p_morph_ksize":          (3,   15),
    "p_min_radius":           (5,   200),
    "p_max_radius":           (20,  400),
    "p_circularity_min":      (0.1, 1.0),
    "p_canny_low":            (5,   200),
    "p_canny_high":           (20,  300),
    "p_hough_param1":         (10,  300),
    "p_hough_param2":         (5,   100),
    "p_gradient_downscale":   (1,   4),
    "p_seed_flood_tolerance": (5,   100),
    "g_brightness_thresh":    (150, 255),
    "g_min_area":             (1,   500),
    "g_max_area":             (50,  5000),
    "g_search_radius_factor": (1.0, 5.0),
    "g_circularity_min":      (0.1, 1.0),
    "g_iris_radius_factor": (1.2, 3.0),
    "g_ellipse_slack":        (0.0, 0.5),
}
_VALID_ALGORITHMS = ("threshold", "edge", "gradient", "seed")
_VALID_DEBUG_VIEWS = ("original", "p_suppressed", "p_blurred", "p_thresh",
                      "p_morph", "p_edges", "g_thresh", "g_morph")


def _apply_params(params: dict, save: bool = False) -> dict:
    """Validate and apply a flat params dict. Returns applied keys."""
    global _debug_view
    updates = {}

    if "debug_view" in params:
        v = params["debug_view"]
        if v in _VALID_DEBUG_VIEWS:
            _debug_view = v

    if "p_algorithm" in params and params["p_algorithm"] in _VALID_ALGORITHMS:
        updates["p_algorithm"] = params["p_algorithm"]

    for key, (lo, hi) in _EYE_PARAM_RANGES.items():
        if key in params:
            try:
                val = float(params[key])
                val = max(lo, min(hi, val))
                if isinstance(lo, int) and isinstance(hi, int):
                    val = int(val)
                    if key in ("p_blur_ksize", "p_morph_ksize") and val % 2 == 0:
                        val += 1
                updates[key] = val
            except (ValueError, TypeError):
                pass

    if updates:
        with _pipe_lock:
            _pipe.update_params(updates)

    if save:
        _save_settings()

    return updates


# ── HTTP handler ──────────────────────────────────────────────────────────────
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_): pass

    def _send(self, code: int, body: bytes, ctype: str = "application/json"):
        try:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
        except BrokenPipeError:
            pass

    # ── POST /process ─────────────────────────────────────────────────────────
    def _handle_process(self):
        length = int(self.headers.get("Content-Length", 0))
        jpeg   = self.rfile.read(length)
        if not jpeg:
            self._send(400, b'{"error":"empty body"}')
            return
        annotated = _process(jpeg)
        # Include PCCR + timestamp in response headers so receiver gets them
        # inline — no separate /result round-trip needed.
        with _latest_lock:
            res = _latest_result
            ts  = _latest_ts
        try:
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(annotated)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("X-Pccr-Ts", f"{ts:.6f}")
            if res and res.pccr_vector:
                self.send_header("X-Pccr-Dx", f"{res.pccr_vector[0]:.4f}")
                self.send_header("X-Pccr-Dy", f"{res.pccr_vector[1]:.4f}")
                self.send_header("X-Pccr-Side", f"{res.pccr_side:.4f}")
            if res and res.pupil_radius is not None:
                self.send_header("X-Pupil-Radius", f"{res.pupil_radius:.2f}")
            self.end_headers()
            self.wfile.write(annotated)
        except BrokenPipeError:
            pass

    # ── GET /result ───────────────────────────────────────────────────────────
    def _handle_result(self):
        with _latest_lock:
            res   = _latest_result
            ts    = _latest_ts
            gaze  = _latest_gaze

        if res is None:
            self._send(
                200,
                json.dumps(
                    {
                        "ready": False,
                        "gaze_model_trained": _gaze_model.trained,
                        "gaze_scene_width": getattr(_gaze_model, "scene_width", None),
                        "gaze_scene_height": getattr(_gaze_model, "scene_height", None),
                    }
                ).encode(),
            )
            return

        gr = res.glint
        out = {
            "ready":       True,
            "ts":          ts,
            "gaze_model_trained": _gaze_model.trained,
            "gaze_scene_width": getattr(_gaze_model, "scene_width", None),
            "gaze_scene_height": getattr(_gaze_model, "scene_height", None),
            "pccr_vector": list(res.pccr_vector) if res.pccr_vector else None,
            "pccr_side":   res.pccr_side,
            "pupil": {
                "center":     res.pupil_center,
                "radius":     res.pupil_radius,
                "confidence": res.pupil.confidence,
            },
            "glint": {
                "primary": res.glint_pos,
                "all":     gr.glints,
            },
            "gaze": list(gaze) if gaze else None,
        }
        self._send(200, json.dumps(out).encode())

    # ── GET /frame ────────────────────────────────────────────────────────────
    def _handle_frame(self):
        with _latest_lock:
            frame = _latest_frame
        if frame is None:
            self._send(404, b'{"error":"no frame yet"}')
            return
        self._send(200, frame, "image/jpeg")

    # ── GET /settings ─────────────────────────────────────────────────────────
    def _handle_get_settings(self):
        with _pipe_lock:
            params = _pipe.get_params()
        params["debug_view"] = _debug_view
        self._send(200, json.dumps(params).encode())

    # ── POST /settings ────────────────────────────────────────────────────────
    def _handle_post_settings(self):
        import urllib.parse
        length = int(self.headers.get("Content-Length", 0))
        raw    = self.rfile.read(length).decode()
        ctype  = self.headers.get("Content-Type", "")

        if "application/json" in ctype:
            try:
                params = json.loads(raw)
            except json.JSONDecodeError:
                self._send(400, b'{"error":"bad JSON"}')
                return
        else:
            params = dict(urllib.parse.parse_qsl(raw))

        save    = bool(params.pop("save", False))
        applied = _apply_params(params, save=save)
        self._send(200, json.dumps({"ok": True, "applied": applied}).encode())

    # ── GET /gaze_model ───────────────────────────────────────────────────────
    def _handle_gaze_model(self):
        if _gaze_model.trained:
            body = json.dumps({
                "trained": True,
                "n_terms": _gaze_model.n_terms,
                "A": _gaze_model.A.tolist(),
                "B": _gaze_model.B.tolist(),
            }).encode()
        else:
            body = json.dumps({"trained": False}).encode()
        self._send(200, body)

    # ── POST /gaze_model/load ─────────────────────────────────────────────────
    def _handle_gaze_model_load(self, body: bytes):
        path_s = None
        if body:
            try:
                path_s = json.loads(body.decode()).get("path")
            except (json.JSONDecodeError, UnicodeDecodeError):
                self._send(400, b'{"error":"invalid JSON body"}')
                return
        target = _resolve_gaze_model_load_path(path_s)
        if target is None:
            self._send(400, json.dumps({"ok": False, "error": "path outside project root"}).encode())
            return
        ok = _gaze_model.load(target)
        self._send(200, json.dumps({"ok": ok, "path": str(target)}).encode())

    # ── GET /glint_sweep/status ───────────────────────────────────────────────
    def _handle_sweep_status(self):
        with _sweep_lock:
            active  = _sweep_active
            n       = len(_sweep_samples)
        self._send(200, json.dumps({
            "active": active,
            "n_samples": n,
            "switch_dx": _pipe.switch_dx,
        }).encode())

    # ── POST /glint_sweep/start ───────────────────────────────────────────────
    def _handle_sweep_start(self):
        global _sweep_active, _sweep_samples
        with _sweep_lock:
            _sweep_samples = []
            _sweep_active  = True
        self._send(200, json.dumps({"ok": True, "msg": "sweep started — look left→right slowly"}).encode())

    # ── POST /set_switch_dx ───────────────────────────────────────────────────
    def _handle_set_switch_dx(self):
        length = int(self.headers.get("Content-Length", 0))
        body   = self.rfile.read(length) if length > 0 else b""
        try:
            data = json.loads(body)
            sw   = float(data["switch_dx"])
        except (KeyError, ValueError, json.JSONDecodeError) as e:
            self._send(400, json.dumps({"error": str(e)}).encode())
            return
        with _pipe_lock:
            _pipe.switch_dx = sw
        _save_settings()
        print(f"[engine] switch_dx set to {sw:.3f}", flush=True)
        self._send(200, json.dumps({"ok": True, "switch_dx": round(sw, 3)}).encode())

    # ── POST /glint_sweep/stop ────────────────────────────────────────────────
    def _handle_sweep_stop(self):
        """Stop collection, fit switch_dx from observations, save to settings."""
        global _sweep_active
        with _sweep_lock:
            _sweep_active = False
            samples = list(_sweep_samples)

        if len(samples) < 10:
            self._send(200, json.dumps({
                "ok": False, "reason": f"too few samples ({len(samples)}, need ≥10)",
            }).encode())
            return

        import numpy as _np
        dx_arr     = _np.array([s["dx"]           for s in samples], dtype=float)
        offset_arr = _np.array([s["glint_offset"] for s in samples], dtype=float)

        # Fit linear: glint_offset = m * dx + b  → switch at dx = -b/m
        coeffs = _np.polyfit(dx_arr, offset_arr, 1)
        m, b   = float(coeffs[0]), float(coeffs[1])

        if abs(m) < 1e-6:
            self._send(200, json.dumps({
                "ok": False, "reason": "no slope detected — glint_offset constant across dx range",
            }).encode())
            return

        switch_dx = float(-b / m)

        with _pipe_lock:
            _pipe.switch_dx = switch_dx
        _save_settings()

        self._send(200, json.dumps({
            "ok": True,
            "switch_dx":  round(switch_dx, 3),
            "slope":      round(m, 4),
            "intercept":  round(b, 4),
            "n_samples":  len(samples),
            "dx_range":   [round(float(dx_arr.min()), 2), round(float(dx_arr.max()), 2)],
        }).encode())

    # ── Router ────────────────────────────────────────────────────────────────
    def do_POST(self):
        if self.path == "/process":
            self._handle_process()
        elif self.path == "/settings":
            self._handle_post_settings()
        elif self.path == "/gaze_model/load":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length) if length > 0 else b""
            self._handle_gaze_model_load(body)
        elif self.path == "/glint_sweep/start":
            self._handle_sweep_start()
        elif self.path == "/glint_sweep/stop":
            self._handle_sweep_stop()
        elif self.path == "/set_switch_dx":
            self._handle_set_switch_dx()
        else:
            self._send(404, b'{"error":"not found"}')

    def do_GET(self):
        import urllib.parse
        path = urllib.parse.urlparse(self.path).path
        qs   = urllib.parse.urlparse(self.path).query

        # Allow GET /settings?p_blur_ksize=9&save=1 for quick tuning from browser
        if path == "/settings" and qs:
            params = dict(urllib.parse.parse_qsl(qs))
            save   = bool(params.pop("save", False))
            _apply_params(params, save=save)
            self._handle_get_settings()
        elif path == "/settings":
            self._handle_get_settings()
        elif path == "/result":
            self._handle_result()
        elif path == "/frame":
            self._handle_frame()
        elif path == "/gaze_model":
            self._handle_gaze_model()
        elif path == "/glint_sweep/status":
            self._handle_sweep_status()
        else:
            self._send(404, b'{"error":"not found"}')

    def do_OPTIONS(self):
        # CORS preflight
        try:
            self.send_response(200)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()
        except BrokenPipeError:
            pass


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


ENGINE_PORT = 8081


def main():
    server = ThreadedHTTPServer(("", ENGINE_PORT), Handler)
    print(f"Compute engine on http://localhost:{ENGINE_PORT}")
    print(f"Eye cam: cam{EYE_CAM_ID} (from rig_config)")
    print(f"Gaze model: {'loaded' if _gaze_model.trained else 'not trained'}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
