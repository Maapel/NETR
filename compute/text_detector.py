"""Text ROI detection for world-cam frames.

MSER-based. Returns word/glyph bboxes and line-level quads. No OCR.

Skew-aware: estimates dominant text-line angle from nearest-neighbor
geometry of MSER region centroids, clusters in the de-skewed frame,
and maps line polygons back to original image coords.
"""
from __future__ import annotations

import cv2
import numpy as np


Box = tuple[int, int, int, int]  # x, y, w, h — axis-aligned in image space
Quad = np.ndarray  # shape (4, 2), int32 — four corners in image space


class TextROIDetector:
    def __init__(
        self,
        min_area: int = 60,
        max_area: int = 14000,
        min_aspect: float = 0.1,
        max_aspect: float = 10.0,
        merge_y_overlap: float = 0.5,
        merge_x_gap: int = 40,
        blur_ksize: int = 3,
        skew_method: str = "projection",  # "projection" | "nn"
        skew_range: float = 45.0,  # search ±deg
        skew_coarse: float = 1.0,  # coarse step deg
        skew_fine: float = 0.2,   # refine step deg
        line_method: str = "mser_global",  # "mser_global" | "docstrum" | "local_patch"
        docstrum_k: int = 6,
        docstrum_angle_tol: float = 15.0,
        docstrum_ht_ratio: float = 1.8,
        docstrum_min_len: int = 3,
        docstrum_max_dist: float = 4.0,  # units of median region height
        patch_grid: int = 3,
    ):
        self.min_area = min_area
        self.max_area = max_area
        self.min_aspect = min_aspect
        self.max_aspect = max_aspect
        self.merge_y_overlap = merge_y_overlap
        self.merge_x_gap = merge_x_gap
        self.blur_ksize = blur_ksize
        self.skew_method = skew_method
        self.skew_range = skew_range
        self.skew_coarse = skew_coarse
        self.skew_fine = skew_fine
        self.line_method = line_method
        self.docstrum_k = docstrum_k
        self.docstrum_angle_tol = docstrum_angle_tol
        self.docstrum_ht_ratio = docstrum_ht_ratio
        self.docstrum_min_len = docstrum_min_len
        self.docstrum_max_dist = docstrum_max_dist
        self.patch_grid = patch_grid
        self._last_debug: dict = {}  # per-method aux data for debug overlay
        self._mser = cv2.MSER_create()
        self._mser.setMinArea(min_area)
        self._mser.setMaxArea(max_area)

    def _prep(self, bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
        if self.blur_ksize >= 3:
            gray = cv2.GaussianBlur(gray, (self.blur_ksize, self.blur_ksize), 0)
        return gray

    def detect_regions(self, bgr: np.ndarray) -> list[Box]:
        gray = self._prep(bgr)
        regions, _ = self._mser.detectRegions(gray)
        inv = cv2.bitwise_not(gray)
        regions_inv, _ = self._mser.detectRegions(inv)
        all_regions = list(regions) + list(regions_inv)

        boxes: list[Box] = []
        for pts in all_regions:
            x, y, w, h = cv2.boundingRect(pts.reshape(-1, 1, 2))
            if w == 0 or h == 0:
                continue
            aspect = w / h
            if aspect < self.min_aspect or aspect > self.max_aspect:
                continue
            area = w * h
            if area < self.min_area or area > self.max_area:
                continue
            boxes.append((x, y, w, h))

        return _dedupe_boxes(boxes)

    def estimate_skew(self, boxes: list[Box]) -> float:
        """Dominant text-line angle in degrees (wrapped to [-45, 45])."""
        if len(boxes) < 5:
            return 0.0
        if self.skew_method == "nn":
            return self._skew_nn(boxes)
        return self._skew_projection(boxes)

    def _skew_nn(self, boxes: list[Box]) -> float:
        """Nearest-neighbor angle voting — cheap, local."""
        c = np.array([[x + w / 2, y + h / 2] for x, y, w, h in boxes], dtype=np.float32)
        d = np.linalg.norm(c[:, None] - c[None, :], axis=2)
        np.fill_diagonal(d, np.inf)
        nn = d.argmin(axis=1)
        vecs = c[nn] - c
        ang = np.degrees(np.arctan2(vecs[:, 1], vecs[:, 0]))
        ang = (ang + 90.0) % 180.0 - 90.0
        ang = np.where(ang > 45, ang - 90, ang)
        ang = np.where(ang < -45, ang + 90, ang)
        return float(np.median(ang))

    def _skew_projection(self, boxes: list[Box]) -> float:
        """Projection-profile variance — sweeps θ, measures periodicity.

        At the true θ, centroids project onto a perpendicular axis with
        sharp per-line peaks and near-zero inter-line gaps → high variance.
        Wrong θ smears lines together → low variance.
        """
        c = np.array([[x + w / 2, y + h / 2] for x, y, w, h in boxes], dtype=np.float32)
        heights = np.array([h for _, _, _, h in boxes], dtype=np.float32)
        median_h = float(np.median(heights))
        bin_size = max(1.0, median_h * 0.5)

        def score(theta_deg: float) -> float:
            rad = np.radians(theta_deg)
            # Perpendicular-to-line axis: rotate by -theta then take y
            y_proj = -np.sin(rad) * c[:, 0] + np.cos(rad) * c[:, 1]
            span = y_proj.max() - y_proj.min()
            if span < bin_size:
                return 0.0
            n_bins = max(4, int(span / bin_size) + 1)
            hist, _ = np.histogram(y_proj, bins=n_bins)
            # Normalize by count so variance isn't dominated by total glyph count
            h = hist.astype(np.float32) / max(1.0, hist.sum())
            return float(h.var())

        coarse = np.arange(-self.skew_range, self.skew_range + 1e-9, self.skew_coarse)
        scores = [score(t) for t in coarse]
        best = float(coarse[int(np.argmax(scores))])
        # Fine refine ±coarse_step around best
        lo, hi = best - self.skew_coarse, best + self.skew_coarse
        fine = np.arange(lo, hi + 1e-9, self.skew_fine)
        scores_f = [score(t) for t in fine]
        return float(fine[int(np.argmax(scores_f))])

    def detect_lines(self, bgr: np.ndarray) -> tuple[list[Quad], float]:
        """Dispatch to selected line-detection method.

        Returns (line quads, skew estimate). Skew is NaN when method is local
        (docstrum, local_patch) — lines have per-line orientation, not one global.
        """
        regions = self.detect_regions(bgr)
        if not regions:
            self._last_debug = {"regions": []}
            return [], 0.0

        if self.line_method == "docstrum":
            return self._detect_lines_docstrum(regions)
        if self.line_method == "local_patch":
            return self._detect_lines_local_patch(bgr)
        return self._detect_lines_mser_global(regions)

    def _detect_lines_mser_global(self, regions: list[Box]) -> tuple[list[Quad], float]:
        theta = self.estimate_skew(regions)
        rad = -np.radians(theta)
        R = np.array([[np.cos(rad), -np.sin(rad)],
                      [np.sin(rad),  np.cos(rad)]], dtype=np.float32)
        rot_boxes: list[Box] = []
        orig_corners: list[np.ndarray] = []
        for (x, y, w, h) in regions:
            corners = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
                               dtype=np.float32)
            rc = corners @ R.T
            rx, ry = float(rc[:, 0].min()), float(rc[:, 1].min())
            rw = float(rc[:, 0].max() - rx)
            rh = float(rc[:, 1].max() - ry)
            rot_boxes.append((int(rx), int(ry), int(rw), int(rh)))
            orig_corners.append(corners)
        rot_lines = _merge_into_lines_indexed(
            rot_boxes, self.merge_y_overlap, self.merge_x_gap,
        )
        out: list[Quad] = []
        for member_idxs in rot_lines:
            pts = np.concatenate([orig_corners[i] for i in member_idxs], axis=0)
            rect = cv2.minAreaRect(pts)
            quad = cv2.boxPoints(rect).astype(np.int32)
            out.append(quad)
        self._last_debug = {"regions": regions, "skew": theta, "line_members": rot_lines}
        return out, theta

    def _detect_lines_docstrum(self, regions: list[Box]) -> tuple[list[Quad], float]:
        """Docstrum-lite (O'Gorman 1993 style).

        1. Per region: kNN among similar-height regions → nearest = local baseline dir.
        2. Smooth each region's orientation with median of its kNN's orientations.
        3. Edge (i,j) kept iff direction (i→j) matches region i's smoothed orientation.
        4. Connected components of surviving graph = text lines.
        Handles arbitrary page orientation, mild perspective, curl.
        """
        N = len(regions)
        if N < 2:
            self._last_debug = {"regions": regions, "method": "docstrum"}
            return [], float("nan")

        c = np.array([[x + w / 2, y + h / 2] for x, y, w, h in regions], dtype=np.float32)
        h_arr = np.array([h for _, _, _, h in regions], dtype=np.float32)
        median_h = float(np.median(h_arr))

        d = np.linalg.norm(c[:, None] - c[None, :], axis=2)
        np.fill_diagonal(d, np.inf)
        hr = h_arr[:, None] / np.maximum(h_arr[None, :], 1e-6)
        height_ok = (hr >= 1.0 / self.docstrum_ht_ratio) & (hr <= self.docstrum_ht_ratio)
        d = np.where(height_ok, d, np.inf)

        k = min(self.docstrum_k, N - 1)
        knn = np.argpartition(d, k, axis=1)[:, :k]

        # Raw local angle: nearest valid neighbor
        local_ang = np.zeros(N, dtype=np.float32)
        for i in range(N):
            best_j = knn[i][np.argmin(d[i, knn[i]])]
            if not np.isfinite(d[i, best_j]):
                local_ang[i] = 0.0
                continue
            dx, dy = c[best_j] - c[i]
            a = np.degrees(np.arctan2(dy, dx))
            local_ang[i] = (a + 90.0) % 180.0 - 90.0

        # Smooth via median of self + knn's angles (unwrap-safe by complex avg)
        smoothed = np.zeros(N, dtype=np.float32)
        for i in range(N):
            vals = np.concatenate(([local_ang[i]], local_ang[knn[i]]))
            # Convert to unit vectors on doubled angle so 180° flip = same
            z = np.exp(2j * np.radians(vals))
            smoothed[i] = 0.5 * np.degrees(np.angle(z.mean()))

        max_dist = self.docstrum_max_dist * median_h
        tol = self.docstrum_angle_tol

        parent = list(range(N))
        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        edges_kept: list[tuple[int, int]] = []
        edges_rejected: list[tuple[int, int]] = []
        for i in range(N):
            for j in knn[i]:
                if j <= i:  # avoid duplicate edge work
                    pass  # still keep directional check below
                dij = d[i, j]
                if not np.isfinite(dij) or dij > max_dist:
                    continue
                dx, dy = c[j] - c[i]
                a = np.degrees(np.arctan2(dy, dx))
                a = (a + 90.0) % 180.0 - 90.0
                # Angular delta with wrap
                delta = abs(((a - smoothed[i] + 90.0) % 180.0) - 90.0)
                if delta <= tol:
                    union(int(i), int(j))
                    edges_kept.append((int(i), int(j)))
                else:
                    edges_rejected.append((int(i), int(j)))

        groups: dict[int, list[int]] = {}
        for i in range(N):
            groups.setdefault(find(i), []).append(i)

        out: list[Quad] = []
        line_members: list[list[int]] = []
        for members in groups.values():
            if len(members) < self.docstrum_min_len:
                continue
            pts_list = []
            for idx in members:
                x, y, w, h = regions[idx]
                pts_list.extend([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
            pts = np.array(pts_list, dtype=np.float32)
            rect = cv2.minAreaRect(pts)
            quad = cv2.boxPoints(rect).astype(np.int32)
            out.append(quad)
            line_members.append(members)

        self._last_debug = {
            "regions": regions,
            "method": "docstrum",
            "centroids": c,
            "smoothed_ang": smoothed,
            "edges_kept": edges_kept,
            "edges_rejected": edges_rejected,
            "line_members": line_members,
        }
        return out, float("nan")

    def _detect_lines_local_patch(self, bgr: np.ndarray) -> tuple[list[Quad], float]:
        """Split image into a grid, run mser_global per patch, union quads.

        Tolerates per-patch orientation (different tilt across page) but does
        NOT stitch lines across patch boundaries — lines are cut at edges.
        """
        h, w = bgr.shape[:2]
        g = max(1, self.patch_grid)
        overlap = 0.1
        all_quads: list[Quad] = []
        patch_regions: list[Box] = []
        for iy in range(g):
            for ix in range(g):
                x0 = int(max(0, (ix - overlap) * w / g))
                x1 = int(min(w, (ix + 1 + overlap) * w / g))
                y0 = int(max(0, (iy - overlap) * h / g))
                y1 = int(min(h, (iy + 1 + overlap) * h / g))
                patch = bgr[y0:y1, x0:x1]
                regs = self.detect_regions(patch)
                if not regs:
                    continue
                for (rx, ry, rw, rh) in regs:
                    patch_regions.append((rx + x0, ry + y0, rw, rh))
                quads, _ = self._detect_lines_mser_global(regs)
                for q in quads:
                    q_shift = q.copy()
                    q_shift[:, 0] += x0
                    q_shift[:, 1] += y0
                    all_quads.append(q_shift)
        self._last_debug = {
            "regions": patch_regions,
            "method": "local_patch",
            "patch_grid": g,
        }
        return all_quads, float("nan")

    def annotate(
        self,
        bgr: np.ndarray,
        level: str = "line",
        color: tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
    ) -> np.ndarray:
        out = bgr.copy() if bgr.ndim == 3 else cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
        if level == "region":
            for x, y, w, h in self.detect_regions(bgr):
                cv2.rectangle(out, (x, y), (x + w, y + h), color, thickness)
        else:
            quads, _ = self.detect_lines(bgr)
            for q in quads:
                cv2.polylines(out, [q], True, color, thickness, cv2.LINE_AA)
        return out

    def annotate_debug(self, bgr: np.ndarray) -> np.ndarray:
        """Dimmed bg; regions (orange); line quads (green, numbered); HUD."""
        quads, theta = self.detect_lines(bgr)
        regions = self._last_debug.get("regions", [])
        base = bgr.copy() if bgr.ndim == 3 else cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
        out = cv2.addWeighted(base, 0.55, np.zeros_like(base), 0.45, 0)

        for x, y, w, h in regions:
            cv2.rectangle(out, (x, y), (x + w, y + h), (255, 128, 0), 1)

        if np.isfinite(theta):
            rad = -np.radians(theta)
            R = np.array([[np.cos(rad), -np.sin(rad)],
                          [np.sin(rad),  np.cos(rad)]], dtype=np.float32)
            def sort_key(q):
                rc = q.astype(np.float32) @ R.T
                return (rc[:, 1].mean(), rc[:, 0].mean())
        else:
            def sort_key(q):
                return (q[:, 1].mean(), q[:, 0].mean())
        quads_sorted = sorted(quads, key=sort_key)

        for idx, q in enumerate(quads_sorted, start=1):
            cv2.polylines(out, [q], True, (0, 255, 0), 2, cv2.LINE_AA)
            tx, ty = int(q[:, 0].min()), int(q[:, 1].min())
            label = str(idx)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            ty = max(th + 2, ty - 2)
            cv2.rectangle(out, (tx, ty - th - 2), (tx + tw + 4, ty + 2), (0, 0, 0), -1)
            cv2.putText(out, label, (tx + 2, ty), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (0, 255, 255), 1, cv2.LINE_AA)

        H, W = out.shape[:2]
        skew_str = "n/a (local)" if not np.isfinite(theta) else f"{theta:+.1f}deg"
        hud = f"[{self.line_method}]  regions={len(regions)}  lines={len(quads)}  skew={skew_str}"
        cv2.rectangle(out, (0, 0), (W, 22), (0, 0, 0), -1)
        cv2.putText(out, hud, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        return out

    def annotate_knn_graph(self, bgr: np.ndarray) -> np.ndarray:
        """Docstrum debug: green=kept edges, red=rejected; yellow arrows = smoothed local angle."""
        self.detect_lines(bgr)  # populate _last_debug
        base = bgr.copy() if bgr.ndim == 3 else cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
        out = cv2.addWeighted(base, 0.40, np.zeros_like(base), 0.60, 0)
        dbg = self._last_debug
        if dbg.get("method") != "docstrum":
            cv2.putText(out, "knn graph only for docstrum method",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1)
            return out
        c = dbg.get("centroids")
        for (i, j) in dbg.get("edges_rejected", []):
            p1 = tuple(c[i].astype(int)); p2 = tuple(c[j].astype(int))
            cv2.line(out, p1, p2, (0, 0, 200), 1, cv2.LINE_AA)
        for (i, j) in dbg.get("edges_kept", []):
            p1 = tuple(c[i].astype(int)); p2 = tuple(c[j].astype(int))
            cv2.line(out, p1, p2, (0, 220, 0), 1, cv2.LINE_AA)
        ang = dbg.get("smoothed_ang")
        if ang is not None and c is not None:
            h_arr = np.array([h for _, _, _, h in dbg["regions"]], dtype=np.float32)
            L = float(np.median(h_arr)) * 1.2
            for i in range(len(c)):
                rad = np.radians(ang[i])
                p1 = c[i]
                p2 = c[i] + np.array([np.cos(rad), np.sin(rad)], dtype=np.float32) * L
                cv2.arrowedLine(out, tuple(p1.astype(int)), tuple(p2.astype(int)),
                                (0, 255, 255), 1, cv2.LINE_AA, tipLength=0.3)
        H, W = out.shape[:2]
        hud = (f"edges: kept={len(dbg.get('edges_kept', []))} "
               f"rejected={len(dbg.get('edges_rejected', []))}  "
               f"regions={len(dbg.get('regions', []))}")
        cv2.rectangle(out, (0, 0), (W, 22), (0, 0, 0), -1)
        cv2.putText(out, hud, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        return out


def _dedupe_boxes(boxes: list[Box], iou_thresh: float = 0.7) -> list[Box]:
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
    kept: list[Box] = []
    for b in boxes:
        if not any(_iou(b, k) > iou_thresh for k in kept):
            kept.append(b)
    return kept


def _iou(a: Box, b: Box) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _merge_into_lines_indexed(
    boxes: list[Box], y_overlap: float, x_gap: int,
) -> list[list[int]]:
    """Like _merge_into_lines but returns member-index lists, not merged boxes."""
    if not boxes:
        return []
    heights = sorted(b[3] for b in boxes)
    median_h = heights[len(heights) // 2]
    tol = max(4, int((1.0 - y_overlap) * median_h))

    idx_sorted = sorted(range(len(boxes)), key=lambda i: boxes[i][1] + boxes[i][3] / 2)
    lines: list[list[int]] = []
    line_cy: list[float] = []
    for i in idx_sorted:
        cy = boxes[i][1] + boxes[i][3] / 2
        placed = False
        for j, lc in enumerate(line_cy):
            if abs(cy - lc) <= tol:
                lines[j].append(i)
                line_cy[j] = (lc * (len(lines[j]) - 1) + cy) / len(lines[j])
                placed = True
                break
        if not placed:
            lines.append([i])
            line_cy.append(cy)

    # Within each y-line, split by x-gap
    out: list[list[int]] = []
    for members in lines:
        members.sort(key=lambda i: boxes[i][0])
        group = [members[0]]
        for i in members[1:]:
            prev = boxes[group[-1]]
            prev_right = prev[0] + prev[2]
            if boxes[i][0] - prev_right <= x_gap:
                group.append(i)
            else:
                out.append(group)
                group = [i]
        out.append(group)
    return out
