"""Interactive parameter tuner for TextROIDetector.

    python tools/text_tuner.py <image_path> [--port 8100]

Open http://localhost:8100 — sliders for every detector param, live preview
of region / line / debug / split views. Method dropdown picks the
line-detection algorithm; per-method sliders + description show/hide.
"""
from __future__ import annotations

import argparse
import sys
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from compute.text_detector import TextROIDetector

_img_lock = threading.Lock()
_src_bgr: np.ndarray | None = None


def _jpeg(img: np.ndarray, q: int = 85) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    return buf.tobytes() if ok else b""


def _find_gutter(bgr: np.ndarray) -> int:
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    col_mean = gray.mean(axis=0)
    lo, hi = int(w * 0.35), int(w * 0.65)
    return lo + int(col_mean[lo:hi].argmin())


def _build_detector(params: dict) -> TextROIDetector:
    return TextROIDetector(
        min_area=int(params.get("min_area", 60)),
        max_area=int(params.get("max_area", 14000)),
        min_aspect=float(params.get("min_aspect", 0.2)),
        max_aspect=float(params.get("max_aspect", 10.0)),
        blur_ksize=int(params.get("blur_ksize", 3)),
        merge_y_overlap=float(params.get("merge_y_overlap", 0.5)),
        merge_x_gap=int(params.get("merge_x_gap", 40)),
        x_gap_scale=float(params.get("x_gap_scale", 2.0)),
        min_line_len=int(params.get("min_line_len", 2)),
        skew_method=str(params.get("skew_method", "projection")),
        line_method=str(params.get("line_method", "mser_global")),
        max_line_angle=float(params.get("max_line_angle", 45.0)),
        nms_iou=float(params.get("nms_iou", 0.5)),
        docstrum_k=int(params.get("docstrum_k", 6)),
        docstrum_angle_tol=float(params.get("docstrum_angle_tol", 15.0)),
        docstrum_ht_ratio=float(params.get("docstrum_ht_ratio", 1.8)),
        docstrum_min_len=int(params.get("docstrum_min_len", 3)),
        docstrum_max_dist=float(params.get("docstrum_max_dist", 4.0)),
        patch_grid=int(params.get("patch_grid", 3)),
    )


def _render(bgr: np.ndarray, det: TextROIDetector, view: str) -> np.ndarray:
    if view == "regions":
        return det.annotate(bgr, level="region", color=(255, 0, 0), thickness=1)
    if view == "lines":
        return det.annotate(bgr, level="line", color=(0, 255, 0), thickness=2)
    if view == "debug":
        return det.annotate_debug(bgr)
    if view == "knn_graph":
        return det.annotate_knn_graph(bgr)
    if view == "gray":
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if det.blur_ksize >= 3:
            g = cv2.GaussianBlur(g, (det.blur_ksize, det.blur_ksize), 0)
        return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    if view == "mser_raw":
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if det.blur_ksize >= 3:
            g = cv2.GaussianBlur(g, (det.blur_ksize, det.blur_ksize), 0)
        out = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        regs, _ = det._mser.detectRegions(g)
        for pts in regs:
            x, y, w, h = cv2.boundingRect(pts.reshape(-1, 1, 2))
            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 200, 255), 1)
        inv = cv2.bitwise_not(g)
        regs_inv, _ = det._mser.detectRegions(inv)
        for pts in regs_inv:
            x, y, w, h = cv2.boundingRect(pts.reshape(-1, 1, 2))
            cv2.rectangle(out, (x, y), (x + w, y + h), (255, 100, 200), 1)
        return out
    return bgr


def _render_split(bgr: np.ndarray, det: TextROIDetector, view: str) -> tuple[np.ndarray, dict]:
    h, w = bgr.shape[:2]
    gutter = _find_gutter(bgr)
    margin = max(2, int(w * 0.02))
    left = bgr[:, :max(1, gutter - margin)]
    right = bgr[:, min(w - 1, gutter + margin):]

    l_out = _render(left, det, view)
    r_out = _render(right, det, view)

    pad = 4
    H = max(l_out.shape[0], r_out.shape[0])
    canvas = np.full((H, l_out.shape[1] + pad + r_out.shape[1], 3), 30, dtype=np.uint8)
    canvas[:l_out.shape[0], :l_out.shape[1]] = l_out
    canvas[:r_out.shape[0], l_out.shape[1] + pad:] = r_out
    cv2.line(canvas, (l_out.shape[1] + pad // 2, 0),
             (l_out.shape[1] + pad // 2, H), (0, 0, 255), 2)

    l_regs = det.detect_regions(left)
    l_quads, l_th = det.detect_lines(left)
    r_regs = det.detect_regions(right)
    r_quads, r_th = det.detect_lines(right)
    stats = {
        "gutter_x": gutter,
        "left": {"regions": len(l_regs), "lines": len(l_quads),
                 "skew": (None if not np.isfinite(l_th) else round(l_th, 1))},
        "right": {"regions": len(r_regs), "lines": len(r_quads),
                  "skew": (None if not np.isfinite(r_th) else round(r_th, 1))},
    }
    return canvas, stats


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a, **k):
        pass

    def do_GET(self):
        try:
            u = urllib.parse.urlparse(self.path)
            if u.path == "/":
                return self._html()
            if u.path == "/img":
                return self._img(dict(urllib.parse.parse_qsl(u.query)))
            if u.path == "/stats":
                return self._stats(dict(urllib.parse.parse_qsl(u.query)))
            self.send_error(404)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _img(self, qs: dict):
        view = qs.get("view", "debug")
        split = qs.get("split", "0") == "1"
        det = _build_detector(qs)
        with _img_lock:
            bgr = _src_bgr.copy()
        img, _ = _render_split(bgr, det, view) if split else (_render(bgr, det, view), None)
        body = _jpeg(img, q=80)
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _stats(self, qs: dict):
        import json
        import time
        split = qs.get("split", "0") == "1"
        det = _build_detector(qs)
        with _img_lock:
            bgr = _src_bgr.copy()
        t0 = time.perf_counter()
        if split:
            _, stats = _render_split(bgr, det, "debug")
        else:
            regs = det.detect_regions(bgr)
            quads, theta = det.detect_lines(bgr)
            stats = {
                "regions": len(regs),
                "lines": len(quads),
                "skew": (None if not np.isfinite(theta) else round(theta, 1)),
            }
        stats["elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        stats["method"] = det.line_method
        body = json.dumps(stats).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _html(self):
        body = HTML.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


HTML = """<!doctype html>
<meta charset=utf-8>
<title>Text ROI Tuner</title>
<style>
  body { font: 13px/1.4 system-ui, sans-serif; background:#111; color:#ddd; margin:0; }
  #wrap { display:grid; grid-template-columns: 340px 1fr; height:100vh; }
  #ctrl { padding:12px 16px; overflow:auto; background:#181818; border-right:1px solid #2a2a2a; }
  #ctrl h1 { font-size:14px; margin:0 0 10px; color:#8cf; }
  h3 { font-size:11px; text-transform:uppercase; letter-spacing:.08em; color:#789; margin:14px 0 6px; border-bottom:1px solid #2a2a2a; padding-bottom:3px; }
  .row { margin:8px 0; }
  .row label { display:flex; justify-content:space-between; font-size:12px; color:#aaa; }
  .row input[type=range] { width:100%; }
  .row .val { color:#fff; font-family: ui-monospace, monospace; }
  select { width:100%; background:#000; color:#fff; border:1px solid #355; padding:3px; }
  #desc { background:#0c1418; border:1px solid #234; padding:8px 10px; font-size:11px; color:#9bc; margin:8px 0 4px; border-radius:3px; }
  #desc b { color:#cde; }
  #views { padding:8px; overflow:auto; }
  #views h2 { font-size:12px; margin:6px 2px; color:#9a9; font-weight:600; }
  .panel { background:#000; border:1px solid #2a2a2a; margin-bottom:10px; padding:4px; }
  .panel img { max-width:100%; display:block; }
  #stats { font-family: ui-monospace, monospace; font-size:11px; background:#000; padding:6px 8px; border:1px solid #2a2a2a; margin-bottom:10px; white-space:pre; }
  .toggle { display:flex; align-items:center; gap:6px; margin:6px 0; }
  button { background:#233; color:#fff; border:1px solid #355; padding:4px 10px; cursor:pointer; }
  button:hover { background:#345; }
  .hide { display:none; }
</style>
<body>
<div id=wrap>
  <div id=ctrl>
    <h1>Text ROI Tuner</h1>

    <h3>Method</h3>
    <div class=row><label>line_method</label>
      <select id=line_method>
        <option value=mser_global>mser_global — fast, single skew</option>
        <option value=docstrum>docstrum — per-region orientation</option>
        <option value=local_patch>local_patch — grid of mser_global</option>
      </select></div>
    <div id=desc></div>

    <div class=toggle><input type=checkbox id=split><label for=split>Split by gutter (two-page spread)</label></div>

    <h3>MSER region filter</h3>
    <div class=row><label>min_area <span class=val id=v_min_area></span></label>
      <input type=range id=min_area min=5 max=500 step=1 value=60></div>
    <div class=row><label>max_area <span class=val id=v_max_area></span></label>
      <input type=range id=max_area min=500 max=50000 step=100 value=14000></div>
    <div class=row><label>min_aspect <span class=val id=v_min_aspect></span></label>
      <input type=range id=min_aspect min=0.02 max=1.0 step=0.01 value=0.2></div>
    <div class=row><label>max_aspect <span class=val id=v_max_aspect></span></label>
      <input type=range id=max_aspect min=1.0 max=30 step=0.1 value=10></div>
    <div class=row><label>blur_ksize <span class=val id=v_blur_ksize></span></label>
      <input type=range id=blur_ksize min=0 max=9 step=1 value=3></div>
    <div class=row><label>max_line_angle&deg; <span class=val id=v_max_line_angle></span></label>
      <input type=range id=max_line_angle min=5 max=45 step=1 value=45></div>
    <div class=row><label>nms_iou <span class=val id=v_nms_iou></span></label>
      <input type=range id=nms_iou min=0.2 max=0.95 step=0.05 value=0.5></div>

    <div id=g_mser_global>
      <h3>mser_global — line merge</h3>
      <div class=row><label>skew_method</label>
        <select id=skew_method>
          <option value=projection selected>projection (periodic variance)</option>
          <option value=nn>nn (nearest-neighbor median)</option>
        </select></div>
      <div class=row><label>merge_y_overlap <span class=val id=v_merge_y_overlap></span></label>
        <input type=range id=merge_y_overlap min=0 max=1 step=0.01 value=0.5></div>
      <div class=row><label>merge_x_gap (px) <span class=val id=v_merge_x_gap></span></label>
        <input type=range id=merge_x_gap min=0 max=200 step=1 value=40></div>
      <div class=row><label>x_gap_scale (× char h) <span class=val id=v_x_gap_scale></span></label>
        <input type=range id=x_gap_scale min=0.5 max=6 step=0.1 value=2.0></div>
      <div class=row><label>min_line_len <span class=val id=v_min_line_len></span></label>
        <input type=range id=min_line_len min=1 max=8 step=1 value=2></div>
    </div>

    <div id=g_docstrum class=hide>
      <h3>docstrum — kNN graph</h3>
      <div class=row><label>docstrum_k <span class=val id=v_docstrum_k></span></label>
        <input type=range id=docstrum_k min=2 max=12 step=1 value=6></div>
      <div class=row><label>docstrum_angle_tol&deg; <span class=val id=v_docstrum_angle_tol></span></label>
        <input type=range id=docstrum_angle_tol min=3 max=45 step=1 value=15></div>
      <div class=row><label>docstrum_ht_ratio <span class=val id=v_docstrum_ht_ratio></span></label>
        <input type=range id=docstrum_ht_ratio min=1.1 max=5 step=0.1 value=1.8></div>
      <div class=row><label>docstrum_max_dist (× median h) <span class=val id=v_docstrum_max_dist></span></label>
        <input type=range id=docstrum_max_dist min=1 max=10 step=0.1 value=4></div>
      <div class=row><label>docstrum_min_len <span class=val id=v_docstrum_min_len></span></label>
        <input type=range id=docstrum_min_len min=1 max=10 step=1 value=3></div>
    </div>

    <div id=g_local_patch class=hide>
      <h3>local_patch — grid</h3>
      <div class=row><label>patch_grid <span class=val id=v_patch_grid></span></label>
        <input type=range id=patch_grid min=1 max=8 step=1 value=3></div>
      <div class=row><label>skew_method (per patch)</label>
        <select id=skew_method_lp>
          <option value=projection selected>projection</option>
          <option value=nn>nn</option>
        </select></div>
    </div>

    <button onclick="resetAll()">Reset</button>
    <button onclick="dumpJSON()">Copy params JSON</button>
    <pre id=jsonOut style="background:#000;padding:6px;font-size:11px;overflow:auto;"></pre>
  </div>

  <div id=views>
    <div id=stats>...</div>
    <h2>debug (regions + lines + hud)</h2>
    <div class=panel><img id=v_debug></div>
    <h2 id=h_knn>knn graph (docstrum only)</h2>
    <div class=panel id=p_knn><img id=v_knn_graph></div>
    <h2>lines only</h2>
    <div class=panel><img id=v_lines></div>
    <h2>regions only</h2>
    <div class=panel><img id=v_regions></div>
    <h2>mser raw (yellow=direct, pink=inverted)</h2>
    <div class=panel><img id=v_mser_raw></div>
    <h2>preprocessed gray</h2>
    <div class=panel><img id=v_gray></div>
  </div>
</div>

<script>
const MSER_IDS = ["min_area","max_area","min_aspect","max_aspect","blur_ksize","max_line_angle","nms_iou"];
const MG_IDS   = ["merge_y_overlap","merge_x_gap","x_gap_scale","min_line_len"];
const DS_IDS   = ["docstrum_k","docstrum_angle_tol","docstrum_ht_ratio","docstrum_max_dist","docstrum_min_len"];
const LP_IDS   = ["patch_grid"];
const ALL_IDS  = [...MSER_IDS, ...MG_IDS, ...DS_IDS, ...LP_IDS];

const DEFAULTS = {
  min_area:60, max_area:14000, min_aspect:0.2, max_aspect:10, blur_ksize:3,
  max_line_angle:45, nms_iou:0.5,
  merge_y_overlap:0.5, merge_x_gap:40, x_gap_scale:2.0, min_line_len:2,
  docstrum_k:6, docstrum_angle_tol:15, docstrum_ht_ratio:1.8, docstrum_max_dist:4, docstrum_min_len:3,
  patch_grid:3,
};

const DESCRIPTIONS = {
  mser_global: "<b>mser_global</b> — estimates ONE dominant skew for the whole frame (projection-profile variance or NN median), rotates box corners, merges by y-overlap. Fast (~5–15ms). Best when the page is roughly planar and all text shares a tilt. Fails on curved pages or two-page spreads where two pages tilt oppositely.",
  docstrum: "<b>docstrum</b> — every region gets its OWN local orientation from its k-nearest neighbors, smoothed by complex-mean. Edges connecting neighbors within angle_tol form a graph; connected components = lines. Tolerates curved pages, perspective, any plane. O(N·k) — ~15–40ms. Tune <i>angle_tol</i> loose for curl, tight for noise.",
  local_patch: "<b>local_patch</b> — splits the image into a patch_grid × patch_grid grid (with 10% overlap), runs mser_global per patch so each patch gets its own skew. Handles non-planar pages crudely but DOES NOT stitch lines across patch boundaries. Cheapest mental model; use 2–4 grid.",
};

function currentMethod() { return document.getElementById("line_method").value; }

function qs() {
  const p = new URLSearchParams();
  for (const k of ALL_IDS) p.set(k, document.getElementById(k).value);
  if (document.getElementById("split").checked) p.set("split","1");
  p.set("line_method", currentMethod());
  const m = currentMethod();
  if (m === "mser_global") p.set("skew_method", document.getElementById("skew_method").value);
  else if (m === "local_patch") p.set("skew_method", document.getElementById("skew_method_lp").value);
  return p.toString();
}

function updateMethodUI() {
  const m = currentMethod();
  document.getElementById("g_mser_global").classList.toggle("hide", m !== "mser_global");
  document.getElementById("g_docstrum").classList.toggle("hide", m !== "docstrum");
  document.getElementById("g_local_patch").classList.toggle("hide", m !== "local_patch");
  document.getElementById("desc").innerHTML = DESCRIPTIONS[m];
  const showKnn = (m === "docstrum");
  document.getElementById("h_knn").classList.toggle("hide", !showKnn);
  document.getElementById("p_knn").classList.toggle("hide", !showKnn);
}

function refresh() {
  for (const k of ALL_IDS) {
    const el = document.getElementById("v_"+k);
    if (el) el.textContent = document.getElementById(k).value;
  }
  const q = qs();
  const views = ["debug","lines","regions","mser_raw","gray"];
  if (currentMethod() === "docstrum") views.splice(1, 0, "knn_graph");
  views.forEach(v => {
    const img = document.getElementById("v_"+v);
    if (img) img.src = "/img?view=" + v + "&" + q + "&_=" + Date.now();
  });
  fetch("/stats?" + q).then(r => r.json()).then(s => {
    document.getElementById("stats").textContent = JSON.stringify(s, null, 2);
  }).catch(()=>{});
}
let t=null;
function schedule(){ clearTimeout(t); t=setTimeout(refresh, 150); }

ALL_IDS.forEach(id => document.getElementById(id).addEventListener("input", schedule));
document.getElementById("split").addEventListener("change", refresh);
document.getElementById("skew_method").addEventListener("change", refresh);
document.getElementById("skew_method_lp").addEventListener("change", refresh);
document.getElementById("line_method").addEventListener("change", () => { updateMethodUI(); refresh(); });

function resetAll(){
  for (const k in DEFAULTS) document.getElementById(k).value = DEFAULTS[k];
  refresh();
}
function dumpJSON(){
  const o = { line_method: currentMethod() };
  for (const k of ALL_IDS) {
    const v = document.getElementById(k).value;
    o[k] = (v.indexOf('.')>=0 ? parseFloat(v) : parseInt(v));
  }
  document.getElementById("jsonOut").textContent = JSON.stringify(o, null, 2);
}

updateMethodUI();
refresh();
</script>
"""


def main():
    global _src_bgr
    ap = argparse.ArgumentParser()
    ap.add_argument("image_path")
    ap.add_argument("--port", type=int, default=8100)
    args = ap.parse_args()

    bgr = cv2.imread(args.image_path)
    if bgr is None:
        sys.exit(f"could not read {args.image_path}")
    h, w = bgr.shape[:2]
    if max(h, w) < 800:
        scale = 800 / max(h, w)
        bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        print(f"upscaled {w}x{h} -> {bgr.shape[1]}x{bgr.shape[0]}")
    _src_bgr = bgr

    srv = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"serving http://localhost:{args.port}  (image: {args.image_path})")
    srv.serve_forever()


if __name__ == "__main__":
    main()
