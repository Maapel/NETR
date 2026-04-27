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
        clahe_clip=float(params.get("clahe_clip", 0.0)),
        clahe_tile=int(params.get("clahe_tile", 8)),
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
        chain_merge_gap=float(params.get("chain_merge_gap", 4.0)),
        binarize=params.get("binarize", "0") == "1",
        split_pages=params.get("split_pages", "0") == "1",
        adaptive_thresh=params.get("adaptive_thresh", "0") == "1",
        adaptive_block=int(params.get("adaptive_block", 51)),
        adaptive_c=float(params.get("adaptive_c", 10.0)),
        chain_break_dist=float(params.get("chain_break_dist", 0.0)),
        chain_merge_dist=float(params.get("chain_merge_dist", 0.0)),
        gap_threshold=float(params.get("gap_threshold", 0.75)),
        gap_min_h=int(params.get("gap_min_h", 2)),
        patch_grid=int(params.get("patch_grid", 3)),
        strip_count=int(params.get("strip_count", 24)),
        peak_min_height=float(params.get("peak_min_height", 0.28)),
        peak_min_dist=int(params.get("peak_min_dist", 10)),
        y_tol=int(params.get("y_tol", 10)),
        track_max_gap=int(params.get("track_max_gap", 2)),
        smooth_win=int(params.get("smooth_win", 7)),
        min_track_len=int(params.get("min_track_len", 4)),
        book_mask=params.get("book_mask", "0") == "1",
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
        return cv2.cvtColor(det._prep(bgr), cv2.COLOR_GRAY2BGR)
    if view == "binary":
        g = det._prep(bgr)
        _, bw = cv2.threshold(g if g.max() > 1 else (g * 255).astype(np.uint8),
                              0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
    if view == "word_chain":
        return det.annotate_word_chains(bgr)
    if view == "gap_mask":
        return det.annotate_gap_mask(bgr)
    if view == "curve_track":
        return det.annotate_curve_track(bgr)
    if view == "mser_raw":
        g = det._prep(bgr)
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
  .row label { display:flex; justify-content:space-between; font-size:12px; color:#aaa; align-items:center; }
  .row input[type=range] { width:100%; }
  .row .val { color:#fff; font-family: ui-monospace, monospace; }
  .tip { cursor:help; color:#567; font-size:10px; margin-left:3px; flex-shrink:0; }
  .tip:hover { color:#9bc; }
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
        <option value=vector_flow>vector_flow — chains + convex hull (curved)</option>
        <option value=local_patch>local_patch — grid of mser_global</option>
        <option value=word_chain>word_chain — rotated boxes + ray-cast chaining</option>
        <option value=gap_scan>gap_scan — white inter-line gaps → strip borders</option>
        <option value=curve_track>curve_track — strip projection peaks, follows page curvature</option>
      </select></div>
    <div id=desc></div>

    <div class=toggle><input type=checkbox id=split><label for=split>Split by gutter (two-page spread)</label></div>
    <div class=toggle><input type=checkbox id=split_pages><label for=split_pages>split_pages — process left/right pages separately (word_chain)</label></div>
    <div class=toggle><input type=checkbox id=adaptive_thresh><label for=adaptive_thresh>adaptive_thresh — local binarisation before MSER (non-uniform light)</label></div>
    <div class=toggle><input type=checkbox id=binarize><label for=binarize>binarize — Otsu threshold after CLAHE+blur (cleaner binary input for MSER)</label></div>

    <h3>MSER region filter</h3>
    <div class=row><label>min_area <span class=val id=v_min_area></span><span class=tip title="Minimum bounding-box area (px²) to keep an MSER region. Raise to discard tiny noise dots.">ⓘ</span></label>
      <input type=range id=min_area min=5 max=500 step=1 value=60></div>
    <div class=row><label>max_area <span class=val id=v_max_area></span><span class=tip title="Maximum bounding-box area (px²). Lower to reject large non-text blobs (hands, page edges).">ⓘ</span></label>
      <input type=range id=max_area min=500 max=50000 step=100 value=14000></div>
    <div class=row><label>min_aspect <span class=val id=v_min_aspect></span><span class=tip title="Min w/h ratio. 0.2 means height can be at most 5× width. Filters near-vertical strokes and serif noise.">ⓘ</span></label>
      <input type=range id=min_aspect min=0.02 max=1.0 step=0.01 value=0.2></div>
    <div class=row><label>max_aspect <span class=val id=v_max_aspect></span><span class=tip title="Max w/h ratio. Filters very wide flat blobs that are unlikely to be single characters.">ⓘ</span></label>
      <input type=range id=max_aspect min=1.0 max=30 step=0.1 value=10></div>
    <div class=row><label>blur_ksize <span class=val id=v_blur_ksize></span><span class=tip title="Gaussian blur kernel before MSER (0 = off, must be odd). Smooths noise at the cost of fine detail.">ⓘ</span></label>
      <input type=range id=blur_ksize min=0 max=9 step=1 value=3></div>
    <div class=row><label>clahe_clip (0=off) <span class=val id=v_clahe_clip></span><span class=tip title="CLAHE contrast limit. 0 = off. Use 2–4 for dark or low-contrast images (book photos, dim lighting). Dramatically improves MSER region count on poorly-lit frames.">ⓘ</span></label>
      <input type=range id=clahe_clip min=0 max=6 step=0.5 value=0></div>
    <div class=row><label>clahe_tile <span class=val id=v_clahe_tile></span><span class=tip title="CLAHE tile grid size (px). Smaller = more local contrast boost. 8 is a good default; reduce to 4 for very uneven lighting.">ⓘ</span></label>
      <input type=range id=clahe_tile min=2 max=32 step=2 value=8></div>
    <div class=row><label>adaptive_block (px) <span class=val id=v_adaptive_block></span><span class=tip title="Block size for adaptive threshold (must be odd). Larger = smoother local regions; smaller = more local contrast. Relevant only when adaptive_thresh is on.">ⓘ</span></label>
      <input type=range id=adaptive_block min=11 max=201 step=2 value=51></div>
    <div class=row><label>adaptive_c <span class=val id=v_adaptive_c></span><span class=tip title="Constant subtracted from local mean in adaptive threshold. Higher = only keep very dark text; lower = picks up faint text too.">ⓘ</span></label>
      <input type=range id=adaptive_c min=1 max=40 step=1 value=10></div>
    <div class=row><label>max_line_angle° <span class=val id=v_max_line_angle></span><span class=tip title="Reject text lines whose orientation is more than this many degrees from horizontal. Lines are always near-horizontal; 45° allows tilted pages but rejects vertical noise.">ⓘ</span></label>
      <input type=range id=max_line_angle min=5 max=45 step=1 value=45></div>
    <div class=row><label>nms_iou <span class=val id=v_nms_iou></span><span class=tip title="IoU threshold for quad deduplication (NMS). Lower = more aggressive — overlapping quads are merged. Mainly affects local_patch seam duplicates.">ⓘ</span></label>
      <input type=range id=nms_iou min=0.2 max=0.95 step=0.05 value=0.5></div>

    <div id=g_mser_global>
      <h3>mser_global — line merge</h3>
      <div class=row><label>skew_method<span class=tip title="How to estimate the dominant page skew. 'projection' sweeps angles and picks the one with highest histogram variance (accurate, ~10ms). 'nn' takes the median nearest-neighbor angle (fast, ~2ms, less robust).">ⓘ</span></label>
        <select id=skew_method>
          <option value=projection selected>projection (periodic variance)</option>
          <option value=nn>nn (nearest-neighbor median)</option>
        </select></div>
      <div class=row><label>merge_y_overlap <span class=val id=v_merge_y_overlap></span><span class=tip title="Y-band tolerance for grouping regions into the same line, as a fraction of median char height. 0.5 = regions whose centres are within ½ char height are on the same line.">ⓘ</span></label>
        <input type=range id=merge_y_overlap min=0 max=1 step=0.01 value=0.5></div>
      <div class=row><label>merge_x_gap (px) <span class=val id=v_merge_x_gap></span><span class=tip title="Fixed pixel gap: adjacent regions this far apart horizontally are still merged into the same line. Used as a lower bound; x_gap_scale may widen it further.">ⓘ</span></label>
        <input type=range id=merge_x_gap min=0 max=200 step=1 value=40></div>
      <div class=row><label>x_gap_scale (× char h) <span class=val id=v_x_gap_scale></span><span class=tip title="Scale-invariant word gap = max(merge_x_gap, x_gap_scale × median char height). Adapts automatically to zoom/DPI so word spaces are bridged correctly at any resolution.">ⓘ</span></label>
        <input type=range id=x_gap_scale min=0.5 max=6 step=0.1 value=2.0></div>
      <div class=row><label>min_line_len <span class=val id=v_min_line_len></span><span class=tip title="Minimum number of regions required to output a line. Raise to suppress single-character or two-character noise lines.">ⓘ</span></label>
        <input type=range id=min_line_len min=1 max=8 step=1 value=2></div>
    </div>

    <div id=g_docstrum class=hide>
      <h3>docstrum / vector_flow — kNN graph</h3>
      <div class=row><label>docstrum_k <span class=val id=v_docstrum_k></span><span class=tip title="Number of nearest neighbours used to estimate each region's local text direction. Higher = more robust on sparse text; lower = faster and more local.">ⓘ</span></label>
        <input type=range id=docstrum_k min=2 max=12 step=1 value=6></div>
      <div class=row><label>docstrum_angle_tol° <span class=val id=v_docstrum_angle_tol></span><span class=tip title="Max angle deviation (°) between a region's smoothed orientation and a candidate edge direction. Smaller = stricter grouping (fewer false merges); larger = more tolerant of curved/tilted text.">ⓘ</span></label>
        <input type=range id=docstrum_angle_tol min=3 max=45 step=1 value=15></div>
      <div class=row><label>docstrum_ht_ratio <span class=val id=v_docstrum_ht_ratio></span><span class=tip title="Height compatibility ratio: two regions are only connected if their heights differ by less than this factor. Prevents linking capital letters to sub/superscripts across lines.">ⓘ</span></label>
        <input type=range id=docstrum_ht_ratio min=1.1 max=5 step=0.1 value=1.8></div>
      <div class=row><label>docstrum_max_dist (× h) <span class=val id=v_docstrum_max_dist></span><span class=tip title="Maximum kNN edge length in units of median character height. Edges longer than this are discarded. Raise for widely spaced text; lower to prevent inter-line connections.">ⓘ</span></label>
        <input type=range id=docstrum_max_dist min=1 max=10 step=0.1 value=4></div>
      <div class=row><label>docstrum_min_len <span class=val id=v_docstrum_min_len></span><span class=tip title="Minimum number of regions in a connected component before it becomes a line. Filters isolated noise blobs.">ⓘ</span></label>
        <input type=range id=docstrum_min_len min=1 max=10 step=1 value=3></div>
      <div id=g_wc_extra class=hide>
        <h3>word_chain — break / merge</h3>
        <div class=row><label>chain_break_dist (× h) <span class=val id=v_chain_break_dist></span><span class=tip title="Break chain if gap between consecutive boxes exceeds N × median char height. 0 = off. Use 3–6 to split chains that jump across a word space or gutter.">ⓘ</span></label>
          <input type=range id=chain_break_dist min=0 max=20 step=0.5 value=0></div>
        <div class=row><label>chain_merge_dist (× h) <span class=val id=v_chain_merge_dist></span><span class=tip title="Merge chain endpoints within N × median char height if angle is compatible. 0 = off. Use 3–8 to bridge word spaces and stitch same-line fragments.">ⓘ</span></label>
          <input type=range id=chain_merge_dist min=0 max=20 step=0.5 value=0></div>
      </div>
      <div id=g_vf_extra class=hide>
        <h3>vector_flow — chain merge</h3>
        <div class=row><label>chain_merge_gap (× h) <span class=val id=v_chain_merge_gap></span><span class=tip title="How far (in units of median char height) the direction ray from a chain's tail can reach to merge with an adjacent chain's hull. Raise to bridge larger word spaces.">ⓘ</span></label>
          <input type=range id=chain_merge_gap min=1 max=10 step=0.5 value=4></div>
      </div>
    </div>

    <div id=g_gap_scan class=hide>
      <h3>gap_scan — inter-line gaps</h3>
      <div class=row><label>gap_threshold <span class=val id=v_gap_threshold></span><span class=tip title="Normalised row brightness (0–1) above which a row is considered an inter-line gap. 0.75 works well for clean scans; lower if gaps are not bright enough to register.">ⓘ</span></label>
        <input type=range id=gap_threshold min=0.3 max=0.99 step=0.01 value=0.75></div>
      <div class=row><label>gap_min_h (px) <span class=val id=v_gap_min_h></span><span class=tip title="Minimum consecutive bright-row height (px) to be counted as a real inter-line gap. Raise to ignore narrow bright stripes from descenders or noise.">ⓘ</span></label>
        <input type=range id=gap_min_h min=1 max=20 step=1 value=2></div>
    </div>

    <div id=g_local_patch class=hide>
      <h3>local_patch — grid</h3>
      <div class=row><label>patch_grid <span class=val id=v_patch_grid></span><span class=tip title="Split image into N×N patches (with 10% overlap). Each patch runs mser_global independently with its own skew estimate. Use 2–4 for books with page curl or two-page spreads.">ⓘ</span></label>
        <input type=range id=patch_grid min=1 max=8 step=1 value=3></div>
      <div class=row><label>skew_method (per patch)<span class=tip title="Skew method applied independently to each patch. 'projection' is more accurate but slower per patch; 'nn' is faster for many small patches.">ⓘ</span></label>
        <select id=skew_method_lp>
          <option value=projection selected>projection</option>
          <option value=nn>nn</option>
        </select></div>
    </div>

    <div id=g_curve_track class=hide>
      <h3>curve_track — strip projection peaks</h3>
      <div class=row><label>strip_count <span class=val id=v_strip_count></span><span class=tip title="Number of vertical strips to slice the image into. More strips = finer curvature tracking; fewer = faster and more robust on noisy images.">ⓘ</span></label>
        <input type=range id=strip_count min=4 max=80 step=1 value=24></div>
      <div class=row><label>peak_min_height <span class=val id=v_peak_min_height></span><span class=tip title="Min peak height as a fraction of the strip's max projection value. Raise to suppress weak/partial lines; lower to pick up faint text.">ⓘ</span></label>
        <input type=range id=peak_min_height min=0.05 max=0.9 step=0.01 value=0.28></div>
      <div class=row><label>peak_min_dist (px) <span class=val id=v_peak_min_dist></span><span class=tip title="Minimum y-distance between two peaks in the same strip. Should be roughly the inter-line gap in pixels. Too small = double-detects one line; too large = merges adjacent lines.">ⓘ</span></label>
        <input type=range id=peak_min_dist min=2 max=60 step=1 value=10></div>
      <div class=row><label>y_tol (px) <span class=val id=v_y_tol></span><span class=tip title="Max y-drift allowed between a track's last peak and a candidate peak in the next strip. Controls how much the tracked line centre can shift per strip — set to roughly the peak shift expected from page curvature.">ⓘ</span></label>
        <input type=range id=y_tol min=1 max=60 step=1 value=10></div>
      <div class=row><label>track_max_gap <span class=val id=v_track_max_gap></span><span class=tip title="Max number of consecutive strips a track can be absent (no matching peak) before it is considered dead. Raise for images where some strips have no text (margins, illustrations).">ⓘ</span></label>
        <input type=range id=track_max_gap min=0 max=10 step=1 value=2></div>
      <div class=row><label>smooth_win <span class=val id=v_smooth_win></span><span class=tip title="Box-filter kernel size for smoothing the per-strip projection profile before peak finding. Larger = smoother curve, less noise sensitivity; smaller = sharper peaks, more responsive to local density.">ⓘ</span></label>
        <input type=range id=smooth_win min=1 max=31 step=1 value=7></div>
      <div class=row><label>min_track_len <span class=val id=v_min_track_len></span><span class=tip title="Minimum number of strips a track must span to be kept as a detected line. Raise to suppress short spurious tracks (noise, margin annotations).">ⓘ</span></label>
        <input type=range id=min_track_len min=1 max=30 step=1 value=4></div>
      <div class=toggle><input type=checkbox id=book_mask><label for=book_mask>book_mask — discard lines outside detected page quad (Canny)</label></div>
    </div>

    <button onclick="resetAll()">Reset</button>
    <button onclick="dumpJSON()">Copy params JSON</button>
    <pre id=jsonOut style="background:#000;padding:6px;font-size:11px;overflow:auto;"></pre>
  </div>

  <div id=views>
    <div id=stats>...</div>
    <h2>debug — regions (orange) + lines (green) + HUD</h2>
    <div class=panel><img id=v_debug></div>
    <h2>knn graph — green=kept edges, red=rejected, yellow=orientation arrows, magenta=chain order</h2>
    <div class=panel><img id=v_knn_graph></div>
    <h2>lines only</h2>
    <div class=panel><img id=v_lines></div>
    <h2>regions only</h2>
    <div class=panel><img id=v_regions></div>
    <h2>mser raw — yellow=direct MSER, pink=inverted MSER</h2>
    <div class=panel><img id=v_mser_raw></div>
    <h2>preprocessed gray (input to MSER, after CLAHE + blur)</h2>
    <div class=panel><img id=v_gray></div>
    <h2>binary — Otsu on preprocessed gray (what binarize=on feeds to MSER)</h2>
    <div class=panel><img id=v_binary></div>
    <h2>word chain — rotated bboxes (white) + chain arrows (coloured) (word_chain only)</h2>
    <div class=panel><img id=v_word_chain></div>
    <h2>gap mask — thresholded inter-line gaps (gap_scan only)</h2>
    <div class=panel><img id=v_gap_mask></div>
    <h2>curve track — coloured polyline per text line (curve_track only)</h2>
    <div class=panel><img id=v_curve_track></div>
  </div>
</div>

<script>
const MSER_IDS = ["min_area","max_area","min_aspect","max_aspect","blur_ksize","clahe_clip","clahe_tile","adaptive_block","adaptive_c","max_line_angle","nms_iou"];
const MG_IDS   = ["merge_y_overlap","merge_x_gap","x_gap_scale","min_line_len"];
const DS_IDS   = ["docstrum_k","docstrum_angle_tol","docstrum_ht_ratio","docstrum_max_dist","docstrum_min_len","chain_merge_gap","chain_break_dist","chain_merge_dist"];
const GS_IDS   = ["gap_threshold","gap_min_h"];
const LP_IDS   = ["patch_grid"];
const CT_IDS   = ["strip_count","peak_min_height","peak_min_dist","y_tol","track_max_gap","smooth_win","min_track_len"];
const ALL_IDS  = [...MSER_IDS, ...MG_IDS, ...DS_IDS, ...GS_IDS, ...LP_IDS, ...CT_IDS];

const DEFAULTS = {
  min_area:60, max_area:14000, min_aspect:0.2, max_aspect:10, blur_ksize:3,
  clahe_clip:0.0, clahe_tile:8, adaptive_block:51, adaptive_c:10,
  max_line_angle:45, nms_iou:0.5,
  merge_y_overlap:0.5, merge_x_gap:40, x_gap_scale:2.0, min_line_len:2,
  docstrum_k:6, docstrum_angle_tol:15, docstrum_ht_ratio:1.8, docstrum_max_dist:4, docstrum_min_len:3,
  chain_merge_gap:4.0, chain_break_dist:0.0, chain_merge_dist:0.0,
  gap_threshold:0.75, gap_min_h:2,
  patch_grid:3,
  strip_count:24, peak_min_height:0.28, peak_min_dist:10,
  y_tol:10, track_max_gap:2, smooth_win:7, min_track_len:4,
};

const DESCRIPTIONS = {
  mser_global: "<b>mser_global</b> — estimates ONE dominant skew for the whole frame (projection-profile variance or NN median), rotates box corners, merges by y-overlap. Fast (~5–15ms). Best when the page is roughly planar and all text shares a tilt. Fails on curved pages or two-page spreads where two pages tilt oppositely.",
  docstrum: "<b>docstrum</b> — every region gets its OWN local orientation from its k-nearest neighbors, smoothed by complex-mean. Edges connecting neighbors within angle_tol form a graph; connected components = lines. Tolerates curved pages, perspective, any plane. O(N·k) — ~15–40ms. Tune <i>angle_tol</i> loose for curl, tight for noise.",
  vector_flow: "<b>vector_flow</b> — same kNN angle graph as docstrum, but groups members into a reading-order <i>chain</i> (sorted by projection onto mean text direction) and outputs a <b>convex hull</b> per chain instead of minAreaRect. The hull naturally wraps curved/arced lines without over-boxing. knn panel shows magenta arrows for chain order. Use same docstrum sliders.",
  word_chain: "<b>word_chain</b> — each MSER region gets a <i>rotated</i> bounding box aligned to its local smoothed angle (not axis-aligned). Then left-to-right greedy ray-cast: the leading-edge ray from box A hits the first unassigned box B on the same baseline → A chains to B. Produces text-direction-aware chains without global skew. Tune same docstrum sliders; <i>docstrum_max_dist</i> controls how far the ray reaches.",
  gap_scan: "<b>gap_scan</b> — projects the preprocessed image onto the y-axis (mean intensity per row) and finds bright horizontal bands = inter-line whitespace. Each consecutive pair of gaps becomes the top/bottom border of a text-line strip. No MSER needed. Very robust to word spaces, ligatures, and kerning gaps. Tune <i>gap_threshold</i> and enable <i>clahe_clip</i> for dark images.",
  local_patch: "<b>local_patch</b> — splits the image into a patch_grid × patch_grid grid (with 10% overlap), runs mser_global per patch so each patch gets its own skew. Handles non-planar pages crudely but DOES NOT stitch lines across patch boundaries. Cheapest mental model; use 2–4 grid.",
  curve_track: "<b>curve_track</b> — no MSER; binarises the image (adaptive threshold, controlled by <i>adaptive_block</i>/<i>adaptive_c</i>/<i>clahe_clip</i>/<i>blur_ksize</i>) then slices it into vertical strips. Per strip: horizontal projection profile → peak finding → greedy tracking across strips. Naturally follows page curvature. Use <i>split_pages</i> to handle book spreads. Output: polynomial-fit curved strip polygons. Tune <i>peak_min_dist</i> to roughly the inter-line gap in pixels.",
};

function currentMethod() { return document.getElementById("line_method").value; }

function qs() {
  const p = new URLSearchParams();
  for (const k of ALL_IDS) p.set(k, document.getElementById(k).value);
  if (document.getElementById("split").checked) p.set("split","1");
  if (document.getElementById("split_pages").checked) p.set("split_pages","1");
  if (document.getElementById("adaptive_thresh").checked) p.set("adaptive_thresh","1");
  if (document.getElementById("binarize").checked) p.set("binarize","1");
  if (document.getElementById("book_mask").checked) p.set("book_mask","1");
  p.set("line_method", currentMethod());
  const m = currentMethod();
  if (m === "mser_global") p.set("skew_method", document.getElementById("skew_method").value);
  else if (m === "local_patch") p.set("skew_method", document.getElementById("skew_method_lp").value);
  return p.toString();
}

function updateMethodUI() {
  const m = currentMethod();
  document.getElementById("g_mser_global").classList.toggle("hide", m !== "mser_global");
  document.getElementById("g_docstrum").classList.toggle("hide", m !== "docstrum" && m !== "vector_flow" && m !== "word_chain");
  document.getElementById("g_vf_extra").classList.toggle("hide", m !== "vector_flow");
  document.getElementById("g_wc_extra").classList.toggle("hide", m !== "word_chain");
  document.getElementById("g_gap_scan").classList.toggle("hide", m !== "gap_scan");
  document.getElementById("g_local_patch").classList.toggle("hide", m !== "local_patch");
  document.getElementById("g_curve_track").classList.toggle("hide", m !== "curve_track");
  document.getElementById("desc").innerHTML = DESCRIPTIONS[m] || "";
}

function refresh() {
  for (const k of ALL_IDS) {
    const el = document.getElementById("v_"+k);
    if (el) el.textContent = document.getElementById(k).value;
  }
  const q = qs();
  const views = ["debug","knn_graph","lines","regions","mser_raw","gray","binary","word_chain","gap_mask","curve_track"];
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
document.getElementById("split_pages").addEventListener("change", refresh);
document.getElementById("adaptive_thresh").addEventListener("change", refresh);
document.getElementById("binarize").addEventListener("change", refresh);
document.getElementById("book_mask").addEventListener("change", refresh);
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
