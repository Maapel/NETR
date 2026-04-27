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
        # MSER region filter
        min_area: int = 60,
        max_area: int = 14000,
        min_aspect: float = 0.2,      # tightened: rejects near-vertical strokes
        max_aspect: float = 10.0,
        blur_ksize: int = 3,
        # preprocessing
        clahe_clip: float = 0.0,       # >0 enables CLAHE (e.g. 2.0–4.0); 0 = off
        clahe_tile: int = 8,           # CLAHE tile grid size
        binarize: bool = False,        # Otsu threshold after CLAHE+blur → clean binary input
        # line merge (mser_global)
        merge_y_overlap: float = 0.5,
        merge_x_gap: int = 40,
        x_gap_scale: float = 2.0,     # gap = max(merge_x_gap, x_gap_scale * median_h)
        min_line_len: int = 2,         # min regions per line to keep
        # skew estimation
        skew_method: str = "projection",  # "projection" | "nn"
        skew_range: float = 45.0,
        skew_coarse: float = 1.0,
        skew_fine: float = 0.2,
        # method dispatch
        line_method: str = "mser_global",  # "mser_global" | "docstrum" | "local_patch"
        max_line_angle: float = 45.0,  # reject lines more vertical than this (°)
        nms_iou: float = 0.5,          # quad NMS overlap threshold
        # docstrum
        docstrum_k: int = 6,
        docstrum_angle_tol: float = 15.0,
        docstrum_ht_ratio: float = 1.8,
        docstrum_min_len: int = 3,
        docstrum_max_dist: float = 4.0,  # units of median region height
        # vector_flow chain post-processing
        chain_merge_gap: float = 4.0,  # stitch chain endpoints if gap < N × median_h
        # gap_scan
        gap_threshold: float = 0.75,   # normalised row brightness to classify as inter-line gap
        gap_min_h: int = 2,            # min gap band height in pixels before it counts
        # word_chain
        split_pages: bool = False,       # detect spine and process left/right pages separately
        adaptive_thresh: bool = False,   # adaptive binarisation before MSER (non-uniform light)
        adaptive_block: int = 51,        # adaptive threshold block size (odd px)
        adaptive_c: float = 10.0,        # constant subtracted in adaptive threshold
        chain_break_dist: float = 0.0,   # break chain if gap > N × median_h  (0 = off)
        chain_merge_dist: float = 0.0,   # merge chain endpoints if gap < N × median_h (0 = off)
        # local_patch
        patch_grid: int = 3,
        # curve_track — strip-wise projection peak tracking
        strip_count: int = 24,
        peak_min_height: float = 0.28,  # min peak height relative to strip max
        peak_min_dist: int = 10,        # min y-distance between peaks (px)
        y_tol: int = 10,                # max y drift between strips for track continuity
        track_max_gap: int = 2,         # max strips a track can be absent
        smooth_win: int = 7,            # projection profile smoothing window
        min_track_len: int = 4,         # min strip count to keep a track
    ):
        self.min_area = min_area
        self.max_area = max_area
        self.min_aspect = min_aspect
        self.max_aspect = max_aspect
        self.blur_ksize = blur_ksize
        self.clahe_clip = clahe_clip
        self.clahe_tile = clahe_tile
        self.binarize = binarize
        self.merge_y_overlap = merge_y_overlap
        self.merge_x_gap = merge_x_gap
        self.x_gap_scale = x_gap_scale
        self.min_line_len = min_line_len
        self.skew_method = skew_method
        self.skew_range = skew_range
        self.skew_coarse = skew_coarse
        self.skew_fine = skew_fine
        self.line_method = line_method
        self.max_line_angle = max_line_angle
        self.nms_iou = nms_iou
        self.docstrum_k = docstrum_k
        self.docstrum_angle_tol = docstrum_angle_tol
        self.docstrum_ht_ratio = docstrum_ht_ratio
        self.docstrum_min_len = docstrum_min_len
        self.docstrum_max_dist = docstrum_max_dist
        self.chain_merge_gap = chain_merge_gap
        self.gap_threshold = gap_threshold
        self.gap_min_h = gap_min_h
        self.split_pages = split_pages
        self.adaptive_thresh = adaptive_thresh
        self.adaptive_block = adaptive_block
        self.adaptive_c = adaptive_c
        self.chain_break_dist = chain_break_dist
        self.chain_merge_dist = chain_merge_dist
        self.patch_grid = patch_grid
        self.strip_count = strip_count
        self.peak_min_height = peak_min_height
        self.peak_min_dist = peak_min_dist
        self.y_tol = y_tol
        self.track_max_gap = track_max_gap
        self.smooth_win = smooth_win
        self.min_track_len = min_track_len
        self._last_debug: dict = {}
        self._mser = cv2.MSER_create()
        self._mser.setMinArea(min_area)
        self._mser.setMaxArea(max_area)

    def _prep(self, bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
        if self.clahe_clip > 0:
            tile = max(1, self.clahe_tile)
            gray = cv2.createCLAHE(
                clipLimit=self.clahe_clip, tileGridSize=(tile, tile)
            ).apply(gray)
        if self.blur_ksize >= 3 and not self.adaptive_thresh:
            gray = cv2.GaussianBlur(gray, (self.blur_ksize, self.blur_ksize), 0)
        if self.adaptive_thresh:
            block = max(3, self.adaptive_block | 1)  # must be odd
            gray = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, block, int(self.adaptive_c),
            )
        elif self.binarize:
            _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
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
            y_proj = -np.sin(rad) * c[:, 0] + np.cos(rad) * c[:, 1]
            span = y_proj.max() - y_proj.min()
            if span < bin_size:
                return 0.0
            n_bins = max(4, int(span / bin_size) + 1)
            hist, _ = np.histogram(y_proj, bins=n_bins)
            h = hist.astype(np.float32) / max(1.0, hist.sum())
            return float(h.var())

        coarse = np.arange(-self.skew_range, self.skew_range + 1e-9, self.skew_coarse)
        scores = [score(t) for t in coarse]
        # Confidence guard: if no clear periodicity, don't refine noise
        if max(scores) < 1e-5:
            return 0.0
        best = float(coarse[int(np.argmax(scores))])
        lo, hi = best - self.skew_coarse, best + self.skew_coarse
        fine = np.arange(lo, hi + 1e-9, self.skew_fine)
        scores_f = [score(t) for t in fine]
        return float(fine[int(np.argmax(scores_f))])

    def detect_lines(self, bgr: np.ndarray) -> tuple[list[Quad], float]:
        """Dispatch to selected line-detection method.

        Returns (line quads, skew estimate). Skew is NaN when method is local
        (docstrum, local_patch) — lines have per-line orientation, not one global.
        """
        # these methods run their own internal MSER — bypass outer detect_regions
        if self.line_method in ("gap_scan", "word_chain", "curve_track"):
            if self.line_method == "gap_scan":
                quads, theta = self._detect_lines_gap_scan(bgr)
            elif self.line_method == "word_chain":
                quads, theta = self._detect_lines_word_chain(bgr)
            else:
                quads, theta = self._detect_lines_curve_track(bgr)
            return self._nms_quads(quads), theta

        regions = self.detect_regions(bgr)
        if not regions:
            self._last_debug = {"regions": []}
            return [], 0.0

        if self.line_method == "docstrum":
            quads, theta = self._detect_lines_docstrum(regions)
        elif self.line_method == "local_patch":
            quads, theta = self._detect_lines_local_patch(bgr)
        elif self.line_method == "vector_flow":
            quads, theta = self._detect_lines_vector_flow(regions)
        else:
            quads, theta = self._detect_lines_mser_global(regions)

        return self._nms_quads(quads), theta

    def _nms_quads(self, quads: list[Quad]) -> list[Quad]:
        """IoU-based deduplication of output quads (esp. for local_patch seams)."""
        if len(quads) < 2:
            return quads
        bboxes = [cv2.boundingRect(q) for q in quads]
        order = sorted(range(len(bboxes)),
                       key=lambda i: bboxes[i][2] * bboxes[i][3], reverse=True)
        kept: list[int] = []
        for i in order:
            if not any(_iou(bboxes[i], bboxes[j]) > self.nms_iou for j in kept):
                kept.append(i)
        return [quads[i] for i in kept]

    def _detect_lines_mser_global(self, regions: list[Box]) -> tuple[list[Quad], float]:
        theta = self.estimate_skew(regions)
        # Clamp skew to horizontal-bias constraint
        theta = float(np.clip(theta, -self.max_line_angle, self.max_line_angle))
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
            rot_boxes, self.merge_y_overlap, self.merge_x_gap, self.x_gap_scale,
        )
        out: list[Quad] = []
        for member_idxs in rot_lines:
            if len(member_idxs) < self.min_line_len:
                continue
            pts = np.concatenate([orig_corners[i] for i in member_idxs], axis=0)
            rect = cv2.minAreaRect(pts)
            quad = cv2.boxPoints(rect).astype(np.int32)
            out.append(quad)
        self._last_debug = {"regions": regions, "skew": theta, "line_members": rot_lines}
        return out, theta

    def _detect_lines_docstrum(self, regions: list[Box]) -> tuple[list[Quad], float]:
        """Docstrum-lite (O'Gorman 1993 style).

        1. Per region: kNN among similar-height regions → nearest = local baseline dir.
        2. Smooth each region's orientation with complex-mean of kNN orientations.
        3. Edge (i,j) kept iff direction (i→j) matches region i's smoothed orientation,
           using adaptive tolerance based on local angle spread.
        4. Connected components of surviving graph = text lines.
        5. Post-filter: discard near-vertical line clusters (max_line_angle).
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

        # Smooth via complex-mean on doubled angle (unwrap-safe: 0°=180°)
        smoothed = np.zeros(N, dtype=np.float32)
        for i in range(N):
            vals = np.concatenate(([local_ang[i]], local_ang[knn[i]]))
            z = np.exp(2j * np.radians(vals))
            smoothed[i] = 0.5 * np.degrees(np.angle(z.mean()))

        max_dist = self.docstrum_max_dist * median_h

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
            # Adaptive tolerance: loosen when neighbours point in diverse directions
            spread = float(np.std(smoothed[knn[i]]))
            local_tol = min(self.docstrum_angle_tol, max(5.0, spread * 1.5))
            for j in knn[i]:
                dij = d[i, j]
                if not np.isfinite(dij) or dij > max_dist:
                    continue
                dx, dy = c[j] - c[i]
                a = np.degrees(np.arctan2(dy, dx))
                a = (a + 90.0) % 180.0 - 90.0
                delta = abs(((a - smoothed[i] + 90.0) % 180.0) - 90.0)
                if delta <= local_tol:
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
            # Horizontal-bias filter: skip clusters whose mean angle is too vertical
            mean_ang = float(np.mean(np.abs(smoothed[members])))
            if mean_ang > self.max_line_angle:
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

        Tolerates per-patch orientation (different tilt across page). NMS at the
        end (inside detect_lines) removes duplicates from overlapping seam zones.
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

    def _detect_lines_vector_flow(self, regions: list[Box]) -> tuple[list[Quad], float]:
        """Vector-flow chain detection.

        Same kNN + angle-smoothing as docstrum, but:
        1. Within each connected component, members are sorted by projection
           onto the group's mean orientation vector → explicit reading order.
        2. Output is cv2.convexHull of the chain's bboxes (not minAreaRect),
           so mildly curved lines are hugged correctly instead of over-boxed.
        """
        N = len(regions)
        if N < 2:
            self._last_debug = {"regions": regions, "method": "vector_flow"}
            return [], float("nan")

        c = np.array([[x + w / 2, y + h / 2] for x, y, w, h in regions], dtype=np.float32)
        h_arr = np.array([h for _, _, _, h in regions], dtype=np.float32)
        median_h = float(np.median(h_arr))

        # --- kNN distance matrix with height-ratio gating (same as docstrum) ---
        d_mat = np.linalg.norm(c[:, None] - c[None, :], axis=2)
        np.fill_diagonal(d_mat, np.inf)
        hr = h_arr[:, None] / np.maximum(h_arr[None, :], 1e-6)
        height_ok = (hr >= 1.0 / self.docstrum_ht_ratio) & (hr <= self.docstrum_ht_ratio)
        d_mat = np.where(height_ok, d_mat, np.inf)

        k = min(self.docstrum_k, N - 1)
        knn = np.argpartition(d_mat, k, axis=1)[:, :k]

        # --- Raw angle from nearest valid neighbour ---
        local_ang = np.zeros(N, dtype=np.float32)
        for i in range(N):
            best_j = knn[i][np.argmin(d_mat[i, knn[i]])]
            if not np.isfinite(d_mat[i, best_j]):
                continue
            dx, dy = c[best_j] - c[i]
            a = np.degrees(np.arctan2(dy, dx))
            local_ang[i] = (a + 90.0) % 180.0 - 90.0

        # --- Complex-mean angle smoothing over kNN (unwrap-safe) ---
        smoothed = np.zeros(N, dtype=np.float32)
        for i in range(N):
            vals = np.concatenate(([local_ang[i]], local_ang[knn[i]]))
            z = np.exp(2j * np.radians(vals))
            smoothed[i] = 0.5 * np.degrees(np.angle(z.mean()))

        max_dist = self.docstrum_max_dist * median_h

        # --- Union-find: connect angle-consistent close neighbours ---
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
        for i in range(N):
            if abs(smoothed[i]) > self.max_line_angle:
                continue
            spread = float(np.std(smoothed[knn[i]]))
            local_tol = min(self.docstrum_angle_tol, max(5.0, spread * 1.5))
            for j in knn[i]:
                if not np.isfinite(d_mat[i, j]) or d_mat[i, j] > max_dist:
                    continue
                dx, dy = c[j] - c[i]
                a = np.degrees(np.arctan2(dy, dx))
                a = (a + 90.0) % 180.0 - 90.0
                delta = abs(((a - smoothed[i] + 90.0) % 180.0) - 90.0)
                if delta <= local_tol:
                    union(int(i), int(j))
                    edges_kept.append((int(i), int(j)))

        groups: dict[int, list[int]] = {}
        for i in range(N):
            groups.setdefault(find(i), []).append(i)

        # --- Phase 1: collect projection-sorted chains from union-find groups ---
        raw_chains: list[list[int]] = []
        for members in groups.values():
            if len(members) < self.docstrum_min_len:
                continue
            mean_ang = float(np.mean(np.abs(smoothed[members])))
            if mean_ang > self.max_line_angle:
                continue
            ang_rad = np.radians(float(np.mean(smoothed[members])))
            line_dir = np.array([np.cos(ang_rad), np.sin(ang_rad)], dtype=np.float32)
            projs = [float(np.dot(c[i], line_dir)) for i in members]
            raw_chains.append([m for _, m in sorted(zip(projs, members))])

        # --- Phase 2: break inter-line jumps, then merge collinear fragments ---
        raw_chains = _break_chains(raw_chains, c, smoothed)
        raw_chains = _merge_chains(
            raw_chains, regions, c, smoothed, median_h,
            self.chain_merge_gap, self.docstrum_angle_tol,
        )

        # --- Phase 3: min_line_len filter + convex hull per chain ---
        out: list[Quad] = []
        chains: list[list[int]] = []
        for chain in raw_chains:
            if len(chain) < self.min_line_len:
                continue
            pts: list[list[int]] = []
            for idx in chain:
                x, y, w, h = regions[idx]
                pts.extend([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
            hull = cv2.convexHull(np.array(pts, dtype=np.int32))
            out.append(hull.reshape(-1, 2).astype(np.int32))
            chains.append(chain)

        self._last_debug = {
            "regions": regions,
            "method": "vector_flow",
            "centroids": c,
            "smoothed_ang": smoothed,
            "chains": chains,
            "edges_kept": edges_kept,
            "edges_rejected": [],
        }
        return out, float("nan")

    def _detect_lines_gap_scan(self, bgr: np.ndarray) -> tuple[list[Quad], float]:
        """Detect text lines from inter-line white gaps.

        Algorithm:
        1. Binarise the preprocessed image (Otsu). White pixels = page/gap; dark = text.
        2. Morphologically close the binary to fill text holes → solid page silhouette.
        3. Largest white connected component = the page mask.
        4. Per row: compute fraction of white pixels within the page mask.
           Rows where that fraction ≥ gap_threshold are "gap" rows (pure inter-line space).
        5. Contiguous runs of gap rows → gap bands.
        6. Text-line strips lie between consecutive gap bands.
           Quads are clipped to the page's column extent.
        """
        gray = self._prep(bgr)
        H, W = gray.shape[:2]

        # --- Step 1: binarise (Otsu) ---
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # --- Step 2: close text holes to get page silhouette ---
        # Kernel wide enough to bridge inter-word spaces, tall enough to merge lines
        kw = max(5, W // 15)
        kh = max(3, H // 40)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh))
        filled = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # --- Step 3: largest white CC = page ---
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            filled, connectivity=8
        )
        if n_labels < 2:
            self._last_debug = {"regions": [], "method": "gap_scan",
                                "gaps": [], "white_fraction": np.zeros(H)}
            return [], float("nan")
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        page_mask = labels == largest          # bool (H, W)

        # Column extent of page (for quad width)
        col_counts = page_mask.sum(axis=0)
        page_cols = np.where(col_counts > 0)[0]
        x_lo = int(page_cols.min()) if len(page_cols) else 0
        x_hi = int(page_cols.max()) if len(page_cols) else W - 1

        # --- Step 4: per-row white fraction within page mask ---
        white_in_page = (binary.astype(bool) & page_mask).sum(axis=1).astype(np.float32)
        page_row_count = page_mask.sum(axis=1).astype(np.float32)
        with np.errstate(invalid="ignore", divide="ignore"):
            white_fraction = np.where(
                page_row_count > 0, white_in_page / page_row_count, 0.0
            ).astype(np.float32)

        # --- Step 5: find gap bands ---
        is_gap = white_fraction >= self.gap_threshold
        gaps: list[tuple[int, int]] = []
        in_gap = False
        g_start = 0
        for y in range(H):
            if is_gap[y] and not in_gap:
                g_start = y
                in_gap = True
            elif not is_gap[y] and in_gap:
                if y - g_start >= self.gap_min_h:
                    gaps.append((g_start, y))
                in_gap = False
        if in_gap and H - g_start >= self.gap_min_h:
            gaps.append((g_start, H))

        # --- Step 6: text strips between consecutive gap bands ---
        all_gaps: list[tuple[int, int]] = [(0, 0)] + gaps + [(H, H)]
        out: list[Quad] = []
        for i in range(len(all_gaps) - 1):
            y0 = all_gaps[i][1]
            y1 = all_gaps[i + 1][0]
            if y1 - y0 < self.gap_min_h:
                continue
            quad = np.array(
                [[x_lo, y0], [x_hi, y0], [x_hi, y1], [x_lo, y1]], dtype=np.int32
            )
            out.append(quad)

        self._last_debug = {
            "regions": [],
            "method": "gap_scan",
            "gaps": gaps,
            "white_fraction": white_fraction,
            "page_mask": page_mask,
            "page_x": (x_lo, x_hi),
        }
        return out, float("nan")

    def _word_chain_mser(
        self, gray: np.ndarray, x_off: int = 0, y_off: int = 0
    ) -> tuple[list[Box], list[tuple]]:
        """MSER + filter + minAreaRect for one image tile.  Returns (regions, rot_boxes)."""
        raw_fwd, _ = self._mser.detectRegions(gray)
        raw_inv, _ = self._mser.detectRegions(cv2.bitwise_not(gray))
        regions: list[Box] = []
        rot_boxes: list[tuple] = []
        for pts in list(raw_fwd) + list(raw_inv):
            pts2d = pts.reshape(-1, 2)
            x, y, w, h = cv2.boundingRect(pts2d.reshape(-1, 1, 2))
            if w == 0 or h == 0:
                continue
            aspect = w / h
            if not (self.min_aspect <= aspect <= self.max_aspect):
                continue
            if not (self.min_area <= w * h <= self.max_area):
                continue
            rect = cv2.minAreaRect(pts2d.astype(np.float32))
            (cx, cy), (rw, rh), angle = rect
            if rw < rh:
                rw, rh = rh, rw
                angle += 90.0
            angle = ((angle + 90.0) % 180.0) - 90.0
            θ = np.radians(float(angle))
            dir_v  = np.array([ np.cos(θ), np.sin(θ)], dtype=np.float32)
            perp_v = np.array([-np.sin(θ), np.cos(θ)], dtype=np.float32)
            regions.append((x + x_off, y + y_off, w, h))
            rot_boxes.append((
                np.array([cx + x_off, cy + y_off], dtype=np.float32),
                rw / 2.0, rh / 2.0, dir_v, perp_v,
            ))
        # Dedup
        if not regions:
            return [], []
        order = sorted(range(len(regions)),
                       key=lambda i: regions[i][2] * regions[i][3], reverse=True)
        kept: list[int] = []
        for i in order:
            if not any(_iou(regions[i], regions[j]) > 0.7 for j in kept):
                kept.append(i)
        return [regions[i] for i in kept], [rot_boxes[i] for i in kept]

    def _detect_lines_word_chain(self, bgr: np.ndarray) -> tuple[list[Quad], float]:
        """Per-region minAreaRect bbox → greedy left-to-right ray-cast chaining.

        Each MSER pixel blob gets cv2.minAreaRect — tightest rotated rectangle that
        fits the blob (min width, max length along natural axis).  Optionally splits
        the image at the page spine and applies break/merge post-processing.
        """
        from scipy.spatial import cKDTree

        gray = self._prep(bgr)
        H, W = gray.shape[:2]

        if self.split_pages:
            # Find spine: darkest column in the centre third
            col_mean = gray.mean(axis=0)
            lo, hi = int(W * 0.3), int(W * 0.7)
            gutter = lo + int(col_mean[lo:hi].argmin())
            margin = max(5, int(W * 0.015))
            l_reg, l_rot = self._word_chain_mser(gray[:, :gutter - margin])
            r_reg, r_rot = self._word_chain_mser(
                gray[:, gutter + margin:], x_off=gutter + margin
            )
            regions  = l_reg  + r_reg
            rot_boxes = l_rot + r_rot
        else:
            regions, rot_boxes = self._word_chain_mser(gray)

        if not regions:
            self._last_debug = {"regions": [], "method": "word_chain"}
            return [], float("nan")

        N = len(regions)
        c     = np.array([[x + w / 2, y + h / 2] for x, y, w, h in regions], dtype=np.float32)
        h_arr = np.array([h for _, _, _, h in regions], dtype=np.float32)
        median_h = float(np.median(h_arr))
        smoothed = np.array([
            float(np.degrees(np.arctan2(rb[3][1], rb[3][0]))) for rb in rot_boxes
        ], dtype=np.float32)

        if N < 2:
            self._last_debug = {"regions": regions, "method": "word_chain"}
            return [], float("nan")

        # Greedy left→right ray-cast chaining (vectorised candidate filtering)
        max_reach = self.docstrum_max_dist * median_h * 2.0
        tree = cKDTree(c)
        # Boolean consumed array → O(k) numpy mask instead of O(k) Python set iteration
        consumed_arr = np.zeros(N, dtype=bool)
        raw_chains: list[list[int]] = []

        # Pre-extract per-region arrays for fast numpy ops
        dir_arr  = np.array([rb[3] for rb in rot_boxes], dtype=np.float32)   # (N, 2)
        hp_arr   = np.array([rb[2] for rb in rot_boxes], dtype=np.float32)   # (N,) half_perp
        ctr_arr  = np.array([rb[0] for rb in rot_boxes], dtype=np.float32)   # (N, 2)
        ha_arr   = np.array([rb[1] for rb in rot_boxes], dtype=np.float32)   # (N,) half_along
        tol_ang  = float(self.docstrum_angle_tol)

        for start_i in np.argsort(c[:, 0]):
            start_i = int(start_i)
            if consumed_arr[start_i]:
                continue
            consumed_arr[start_i] = True
            chain = [start_i]
            cur = start_i
            curv_sign = 0  # 0 = undecided; +1 / -1 once established
            for _ in range(N):  # safety cap
                dir_v = dir_arr[cur]
                tail = ctr_arr[cur] + ha_arr[cur] * dir_v  # leading edge centre

                cands = np.asarray(tree.query_ball_point(tail, max_reach), dtype=np.int32)
                if len(cands) == 0:
                    break

                # Remove consumed (numpy boolean index — no Python loop)
                cands = cands[~consumed_arr[cands]]
                if len(cands) == 0:
                    break

                # Angle compatibility (vectorised)
                dang = np.abs(((smoothed[cands] - float(smoothed[cur]) + 90.0) % 180.0) - 90.0)
                cands = cands[dang <= tol_ang]
                if len(cands) == 0:
                    break

                # Curvature monotonicity (vectorised)
                if curv_sign != 0:
                    dth = ((smoothed[cands] - float(smoothed[cur]) + 90.0) % 180.0) - 90.0
                    bad = (np.abs(dth) > 2.0) & (np.sign(dth).astype(np.int32) != curv_sign)
                    cands = cands[~bad]
                if len(cands) == 0:
                    break

                # Forward slab test (vectorised in current box's frame)
                # Project centroid-to-centroid vector onto ray direction
                v = c[cands] - tail               # (M, 2)
                along = v @ dir_v                 # (M,) signed dist along ray
                fwd = (along > 0) & (along <= max_reach)
                cands = cands[fwd]
                along = along[fwd]
                if len(cands) == 0:
                    break

                # Perpendicular distance ≤ box j's half-height + slack
                v2   = c[cands] - tail
                perp = np.linalg.norm(v2 - np.outer(along, dir_v), axis=1)
                hits = perp <= (hp_arr[cands] + median_h * 0.3)
                cands = cands[hits]
                along = along[hits]
                if len(cands) == 0:
                    break

                best_j = int(cands[np.argmin(along)])

                # Update curvature sign from this transition
                dθ = float(smoothed[best_j]) - float(smoothed[cur])
                dθ = ((dθ + 90.0) % 180.0) - 90.0
                if curv_sign == 0 and abs(dθ) > 2.0:
                    curv_sign = int(np.sign(dθ))
                chain.append(best_j)
                consumed_arr[best_j] = True
                cur = best_j
            raw_chains.append(chain)

        # Post-process: break then merge
        raw_chains = _postprocess_word_chains(
            raw_chains, c, rot_boxes, smoothed, median_h,
            self.chain_break_dist, self.chain_merge_dist, self.docstrum_angle_tol,
        )

        out: list[Quad] = []
        kept: list[list[int]] = []
        for chain in raw_chains:
            if len(chain) < self.min_line_len:
                continue
            pts: list[list[int]] = []
            for idx in chain:
                x, y, w, h = regions[idx]
                pts.extend([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
            hull = cv2.convexHull(np.array(pts, dtype=np.int32))
            out.append(hull.reshape(-1, 2).astype(np.int32))
            kept.append(chain)

        self._last_debug = {
            "regions": regions,
            "method": "word_chain",
            "rot_boxes": rot_boxes,
            "chains": kept,
            "centroids": c,
            "smoothed_ang": smoothed,
        }
        return out, float("nan")

    def annotate_word_chains(self, bgr: np.ndarray) -> np.ndarray:
        """Show rotated per-region bboxes (white) and chain arrows (per-chain colour)."""
        self.detect_lines(bgr)
        base = bgr.copy() if bgr.ndim == 3 else cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
        out = cv2.addWeighted(base, 0.40, np.zeros_like(base), 0.60, 0)
        dbg = self._last_debug
        if dbg.get("method") != "word_chain":
            cv2.putText(out, "word_chain view: select word_chain method",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1)
            return out

        c = dbg["centroids"]
        rot_boxes = dbg["rot_boxes"]
        chains = dbg["chains"]

        # Draw every rotated bbox in dim white
        for ctr, ha, hp, dir_v, perp_v in rot_boxes:
            corners = np.array([
                ctr + ha * dir_v + hp * perp_v,
                ctr - ha * dir_v + hp * perp_v,
                ctr - ha * dir_v - hp * perp_v,
                ctr + ha * dir_v - hp * perp_v,
            ], dtype=np.int32)
            cv2.polylines(out, [corners], True, (140, 140, 140), 1, cv2.LINE_AA)

        # Draw chains — different colour per chain, arrows between members
        palette = [
            (0, 255, 255), (255, 0, 255), (0, 255, 128),
            (255, 128, 0), (128, 0, 255), (0, 180, 255),
            (255, 255, 0), (0, 128, 255),
        ]
        for ci, chain in enumerate(chains):
            col = palette[ci % len(palette)]
            # Box outlines for members of this chain in chain colour
            for idx in chain:
                ctr, ha, hp, dir_v, perp_v = rot_boxes[idx]
                corners = np.array([
                    ctr + ha * dir_v + hp * perp_v,
                    ctr - ha * dir_v + hp * perp_v,
                    ctr - ha * dir_v - hp * perp_v,
                    ctr + ha * dir_v - hp * perp_v,
                ], dtype=np.int32)
                cv2.polylines(out, [corners], True, col, 1, cv2.LINE_AA)
            # Arrows centroid-to-centroid
            for a, b in zip(chain[:-1], chain[1:]):
                cv2.arrowedLine(out,
                                tuple(c[a].astype(int)), tuple(c[b].astype(int)),
                                col, 1, cv2.LINE_AA, tipLength=0.25)
            # Mark chain start
            cv2.circle(out, tuple(c[chain[0]].astype(int)), 4, col, -1)

        H, W = out.shape[:2]
        hud = (f"[word_chain]  chains={len(chains)}  "
               f"regions={len(dbg['regions'])}")
        cv2.rectangle(out, (0, 0), (W, 22), (0, 0, 0), -1)
        cv2.putText(out, hud, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        return out

    def _detect_lines_curve_track(self, bgr: np.ndarray) -> tuple[list[Quad], float]:
        """Strip-wise projection-peak tracking for curved/book-page text lines.

        Each vertical strip of the binarised image gets a horizontal projection
        profile; peaks = text-line centres. Peaks are tracked across strips with
        greedy assignment, producing per-line polylines that follow page curvature.
        split_pages=True auto-detects the spine gutter and processes each page half.
        """
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr.copy()
        H, W = gray.shape[:2]

        if self.clahe_clip > 0:
            tile = max(1, self.clahe_tile)
            gray = cv2.createCLAHE(clipLimit=self.clahe_clip,
                                   tileGridSize=(tile, tile)).apply(gray)
        bk = self.blur_ksize | 1
        if bk >= 3:
            gray = cv2.GaussianBlur(gray, (bk, bk), 0)
        block = max(3, self.adaptive_block | 1)
        bin_inv = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
            block, int(self.adaptive_c),
        )

        split_x = _ct_find_page_split(bin_inv) if self.split_pages else None

        sc = self.strip_count
        kw = dict(
            peak_h=self.peak_min_height,
            peak_dist=self.peak_min_dist,
            y_tol=self.y_tol,
            max_gap=self.track_max_gap,
            smooth_win=self.smooth_win,
            min_track_len=self.min_track_len,
        )
        tracks: list[list[tuple[int, int]]] = []
        if split_x is None:
            tracks = _ct_detect_tracks(bin_inv, sc, **kw)
        else:
            pad = max(8, W // 120)
            left  = bin_inv[:, :max(1, split_x - pad)]
            right = bin_inv[:, min(W - 1, split_x + pad):]
            r_off = min(W - 1, split_x + pad)
            sc2 = max(2, sc // 2)
            if left.size:
                tracks += _ct_detect_tracks(left, sc2, x_offset=0, **kw)
            if right.size:
                tracks += _ct_detect_tracks(right, sc2, x_offset=r_off, **kw)

        half_h = max(4, self.peak_min_dist // 2)
        out: list[Quad] = []
        for track in tracks:
            pts = np.array(track, dtype=np.float32)
            xs, ys = pts[:, 0], pts[:, 1]
            deg = min(2 if len(pts) >= 5 else 1, len(pts) - 1)
            try:
                coeff = np.polyfit(xs, ys, deg)
                xs_d = np.linspace(xs.min(), xs.max(),
                                   max(20, int(xs.max() - xs.min()) + 1))
                ys_d = np.clip(np.polyval(coeff, xs_d), 0, H - 1)
            except Exception:
                xs_d, ys_d = xs, ys
            top = np.stack([xs_d, np.clip(ys_d - half_h, 0, H - 1)], axis=1).astype(np.int32)
            bot = np.stack([xs_d, np.clip(ys_d + half_h, 0, H - 1)], axis=1).astype(np.int32)
            out.append(np.vstack([top, bot[::-1]]))

        self._last_debug = {
            "regions": [],
            "method": "curve_track",
            "tracks": tracks,
            "bin_inv": bin_inv,
            "split_x": split_x,
        }
        return out, float("nan")

    def annotate_curve_track(self, bgr: np.ndarray) -> np.ndarray:
        """Coloured polyline per detected text line (curve_track method)."""
        self.detect_lines(bgr)
        base = bgr.copy() if bgr.ndim == 3 else cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
        out = cv2.addWeighted(base, 0.50, np.zeros_like(base), 0.50, 0)
        dbg = self._last_debug
        if dbg.get("method") != "curve_track":
            cv2.putText(out, "curve_track view: select curve_track method",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1)
            return out
        H, W = out.shape[:2]
        tracks = dbg.get("tracks", [])
        for i, track in enumerate(tracks):
            hue = int((i * 37) % 180)
            col = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]),
                               cv2.COLOR_HSV2BGR)[0, 0].tolist()
            col = tuple(int(c) for c in col)
            pts = np.array(track, dtype=np.float32)
            xs, ys = pts[:, 0], pts[:, 1]
            deg = min(2 if len(pts) >= 5 else 1, len(pts) - 1)
            try:
                coeff = np.polyfit(xs, ys, deg)
                xs_d = np.linspace(xs.min(), xs.max(),
                                   max(20, int(xs.max() - xs.min()) + 1))
                ys_d = np.clip(np.polyval(coeff, xs_d), 0, H - 1)
                poly = np.stack([xs_d, ys_d], axis=1).astype(np.int32)
            except Exception:
                poly = pts.astype(np.int32)
            cv2.polylines(out, [poly.reshape(-1, 1, 2)], False, col, 2, cv2.LINE_AA)
            step = max(1, len(poly) // 12)
            for x, y in poly[::step]:
                cv2.circle(out, (int(x), int(y)), 2, col, -1, cv2.LINE_AA)
        split_x = dbg.get("split_x")
        if split_x is not None:
            cv2.line(out, (split_x, 0), (split_x, H - 1), (0, 180, 255), 2)
        hud = f"[curve_track]  tracks={len(tracks)}"
        cv2.rectangle(out, (0, 0), (W, 22), (0, 0, 0), -1)
        cv2.putText(out, hud, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        return out

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
        """Dimmed bg; regions (orange); line quads (green, numbered); HUD.

        For gap_scan: gap bands tinted cyan; line strips tinted green; profile
        inset on the right edge showing the curve and threshold line.
        """
        quads, theta = self.detect_lines(bgr)
        regions = self._last_debug.get("regions", [])
        base = bgr.copy() if bgr.ndim == 3 else cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
        out = cv2.addWeighted(base, 0.55, np.zeros_like(base), 0.45, 0)

        if self._last_debug.get("method") == "gap_scan":
            H, W = out.shape[:2]
            overlay = out.copy()
            # Tint the page region blue so you can see what was segmented
            page_mask = self._last_debug.get("page_mask")
            if page_mask is not None:
                overlay[page_mask] = (overlay[page_mask].astype(np.int32)
                                      + np.array([60, 20, 0], dtype=np.int32)
                                      ).clip(0, 255).astype(np.uint8)
            # Highlight gap bands cyan/yellow
            x_lo, x_hi = self._last_debug.get("page_x", (0, W - 1))
            for y0, y1 in self._last_debug.get("gaps", []):
                cv2.rectangle(overlay, (x_lo, y0), (x_hi, y1), (0, 230, 255), -1)
            cv2.addWeighted(overlay, 0.40, out, 0.60, 0, out)

            # White-fraction profile inset on the right (80 px)
            wf = self._last_debug.get("white_fraction")
            if wf is not None and len(wf) == H:
                pw = 80
                cv2.rectangle(out, (W - pw - 1, 0), (W - 1, H - 1), (15, 15, 15), -1)
                for y in range(1, H):
                    x1 = W - pw + int(wf[y - 1] * (pw - 2))
                    x2 = W - pw + int(wf[y]     * (pw - 2))
                    cv2.line(out, (x1, y - 1), (x2, y), (180, 180, 180), 1)
                # Threshold line in yellow
                tx = W - pw + int(self.gap_threshold * (pw - 2))
                cv2.line(out, (tx, 0), (tx, H - 1), (0, 220, 255), 1)
        else:
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

    def annotate_gap_mask(self, bgr: np.ndarray) -> np.ndarray:
        """gap_scan only: shows detected gap bands in yellow on a dimmed image."""
        self.detect_lines(bgr)
        base = bgr.copy() if bgr.ndim == 3 else cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
        out = cv2.addWeighted(base, 0.35, np.zeros_like(base), 0.65, 0)
        H, W = out.shape[:2]
        if self._last_debug.get("method") != "gap_scan":
            cv2.putText(out, "gap_mask view: gap_scan method only",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1)
            return out
        x_lo, x_hi = self._last_debug.get("page_x", (0, W - 1))
        for y0, y1 in self._last_debug.get("gaps", []):
            cv2.rectangle(out, (x_lo, y0), (x_hi, y1), (0, 230, 255), -1)
        return out

    def annotate_knn_graph(self, bgr: np.ndarray) -> np.ndarray:
        """Graph debug for docstrum and vector_flow.

        docstrum: green=kept edges, red=rejected, yellow=smoothed angle arrows.
        vector_flow: same edges + cyan arrows connecting sequential chain members.
        """
        self.detect_lines(bgr)  # populate _last_debug
        base = bgr.copy() if bgr.ndim == 3 else cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
        out = cv2.addWeighted(base, 0.40, np.zeros_like(base), 0.60, 0)
        dbg = self._last_debug
        method = dbg.get("method", "")
        if method not in ("docstrum", "vector_flow"):
            cv2.putText(out, "knn graph: docstrum / vector_flow only",
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
        # vector_flow extra: draw chain reading-order arrows (magenta)
        if method == "vector_flow":
            for chain in dbg.get("chains", []):
                for a, b in zip(chain[:-1], chain[1:]):
                    pa = tuple(c[a].astype(int))
                    pb = tuple(c[b].astype(int))
                    cv2.arrowedLine(out, pa, pb, (255, 0, 255), 1,
                                    cv2.LINE_AA, tipLength=0.25)
        H, W = out.shape[:2]
        n_chains = len(dbg.get("chains", []))
        hud = (f"[{method}]  edges={len(dbg.get('edges_kept', []))}  "
               f"chains={n_chains}  regions={len(dbg.get('regions', []))}")
        cv2.rectangle(out, (0, 0), (W, 22), (0, 0, 0), -1)
        cv2.putText(out, hud, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        return out


def _postprocess_word_chains(
    chains: list[list[int]],
    c: np.ndarray,
    rot_boxes: list[tuple],
    smoothed: np.ndarray,
    median_h: float,
    break_dist: float,
    merge_dist: float,
    angle_tol: float,
) -> list[list[int]]:
    """Break chains with large inter-box gaps; merge chains whose endpoints are close."""
    # --- Break ---
    if break_dist > 0:
        thresh = break_dist * median_h
        broken: list[list[int]] = []
        for chain in chains:
            seg = [chain[0]]
            for k in range(1, len(chain)):
                if float(np.linalg.norm(c[chain[k]] - c[chain[k - 1]])) > thresh:
                    broken.append(seg)
                    seg = [chain[k]]
                else:
                    seg.append(chain[k])
            broken.append(seg)
        chains = broken

    # --- Merge ---
    if merge_dist > 0:
        thresh = merge_dist * median_h
        consumed: set[int] = set()
        result: list[list[int]] = []
        order = sorted(range(len(chains)), key=lambda i: float(c[chains[i][0]][0]))
        for a in order:
            if a in consumed:
                continue
            chain_a = list(chains[a])
            # Repeatedly try to extend chain_a forward
            while True:
                tail_pt = c[chain_a[-1]]
                dir_a   = rot_boxes[chain_a[-1]][3]
                ang_a   = float(smoothed[chain_a[-1]])
                best_b: int | None = None
                best_d = float("inf")
                for b in range(len(chains)):
                    if b == a or b in consumed:
                        continue
                    head_pt = c[chains[b][0]]
                    # Angle check
                    delta = abs(((float(smoothed[chains[b][0]]) - ang_a + 90.0) % 180.0) - 90.0)
                    if delta > angle_tol:
                        continue
                    # Distance check
                    d = float(np.linalg.norm(head_pt - tail_pt))
                    if d > thresh or d >= best_d:
                        continue
                    # Must be forward along text direction
                    if float(np.dot(head_pt - tail_pt, dir_a)) <= 0:
                        continue
                    best_d = d
                    best_b = b
                if best_b is None:
                    break
                chain_a.extend(chains[best_b])
                consumed.add(best_b)
            result.append(chain_a)
        chains = result

    return chains


def _blob_axis_angles(
    regions: list[Box],
    gray_shape: tuple[int, int],
    median_h: float,
) -> np.ndarray:
    """Per-region text-direction angle derived purely from blob geometry.

    1. Draw MSER bboxes on a blank mask, dilate horizontally → word-level blobs.
    2. For each blob with ≥2 member regions: PCA on member centroids → axis direction.
    3. Single-member blobs inherit the angle from the nearest multi-member blob.

    This avoids kNN-direction bias toward horizontal and handles perspective/curve
    because PCA on the actual character layout gives the true reading direction.
    """
    from collections import defaultdict

    H, W = gray_shape[:2]
    N = len(regions)
    c = np.array([[x + w / 2, y + h / 2] for x, y, w, h in regions], dtype=np.float32)

    # Draw filled bboxes → binary mask
    mask = np.zeros((H, W), dtype=np.uint8)
    for x, y, w, h in regions:
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

    # Dilate: bridge inter-character gaps to form word blobs
    kw = max(3, int(median_h * 1.0))   # ~1 char-height wide gap closes
    kh = max(3, int(median_h * 0.35))  # limited height so lines don't merge
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh))
    dilated = cv2.dilate(mask, kernel)

    _, labels = cv2.connectedComponents(dilated, connectivity=8)

    # Assign each region to the label under its centroid
    region_labels = np.zeros(N, dtype=np.int32)
    for i, (x, y, w, h) in enumerate(regions):
        cx_i = min(W - 1, max(0, int(x + w / 2)))
        cy_i = min(H - 1, max(0, int(y + h / 2)))
        region_labels[i] = int(labels[cy_i, cx_i])

    # Group regions by blob label
    blob_members: dict[int, list[int]] = defaultdict(list)
    for i, lbl in enumerate(region_labels):
        blob_members[lbl].append(i)

    # PCA on member centroids → text axis per blob
    blob_angle: dict[int, float | None] = {}
    multi_blobs: list[tuple[float, np.ndarray]] = []  # (angle, mean_pt)

    for lbl, members in blob_members.items():
        if len(members) >= 2:
            pts = c[members]
            mean_pt = pts.mean(axis=0)
            centered = pts - mean_pt
            if len(members) == 2:
                axis = centered[1] - centered[0]
            else:
                _, _, vt = np.linalg.svd(centered, full_matrices=False)
                axis = vt[0]
            angle = float(np.degrees(np.arctan2(axis[1], axis[0])))
            angle = (angle + 90.0) % 180.0 - 90.0  # normalise to [-90, 90]
            blob_angle[lbl] = angle
            multi_blobs.append((angle, mean_pt))
        else:
            blob_angle[lbl] = None  # single region — unknown

    # Build per-region angle array; inherit from nearest multi-member blob if unknown
    angles = np.zeros(N, dtype=np.float32)
    for i in range(N):
        a = blob_angle.get(region_labels[i])
        if a is not None:
            angles[i] = float(a)
        elif multi_blobs:
            dists = [float(np.linalg.norm(c[i] - mp)) for _, mp in multi_blobs]
            angles[i] = multi_blobs[int(np.argmin(dists))][0]

    return angles


def _make_rot_boxes(
    regions: list[Box],
    smoothed: np.ndarray,
) -> list[tuple]:
    """Per-region: rotated bbox using MSER blob dimensions, oriented to text direction.

    Keeps the same width/height as the MSER axis-aligned bbox — no inflation.
    Returns list of (center, half_along, half_perp, dir_v, perp_v).
    Corners: center ± half_along*dir_v ± half_perp*perp_v.
    """
    boxes = []
    for i, (x, y, w, h) in enumerate(regions):
        cx, cy = x + w / 2.0, y + h / 2.0
        θ = float(np.radians(smoothed[i]))
        dir_v = np.array([np.cos(θ), np.sin(θ)], dtype=np.float32)
        perp_v = np.array([-np.sin(θ), np.cos(θ)], dtype=np.float32)
        boxes.append((
            np.array([cx, cy], dtype=np.float32),
            w / 2.0,   # half-width along text direction — same as MSER bbox
            h / 2.0,   # half-height perpendicular — same as MSER bbox
            dir_v, perp_v,
        ))
    return boxes


def _ray_hits_rot_box(
    origin: np.ndarray,
    ray_dir: np.ndarray,
    box: tuple,
    max_t: float,
) -> tuple[bool, float]:
    """Slab-method ray–rotated-box intersection.

    box = (center, half_along, half_perp, dir_v, perp_v) from _make_rot_boxes.
    Returns (hit, t_enter) where t_enter is the distance along the ray to the
    nearest intersection point. hit=False if no intersection in (0, max_t].
    """
    ctr, ha, hp, dj, pj = box
    v = origin - ctr         # vector from box centre to ray origin

    # Project ray origin and direction onto box axes
    ra = float(v @ dj);  da = float(ray_dir @ dj)
    rp = float(v @ pj);  dp = float(ray_dir @ pj)

    # Slab along text-direction axis
    if abs(da) > 1e-9:
        t1, t2 = (-ha - ra) / da, (ha - ra) / da
        ta_en, ta_ex = min(t1, t2), max(t1, t2)
    else:
        if abs(ra) > ha:
            return False, float("inf")
        ta_en, ta_ex = -1e18, 1e18

    # Slab perpendicular axis
    if abs(dp) > 1e-9:
        t1, t2 = (-hp - rp) / dp, (hp - rp) / dp
        tp_en, tp_ex = min(t1, t2), max(t1, t2)
    else:
        if abs(rp) > hp:
            return False, float("inf")
        tp_en, tp_ex = -1e18, 1e18

    t_en = max(ta_en, tp_en)
    t_ex = min(ta_ex, tp_ex)
    if t_en > t_ex or t_ex <= 0 or t_en > max_t:
        return False, float("inf")
    return True, max(0.0, t_en)


def _break_chains(
    chains: list[list[int]],
    c: np.ndarray,
    smoothed: np.ndarray,
) -> list[list[int]]:
    """Split a chain at bad connections using two conditions:

    1. Inter-line jump: gap vector > 45° from text direction (perp > along).
    2. Abrupt turn: consecutive gap vectors point in opposite hemispheres
       (dot < 0), i.e. direction reverses by more than 90°.
    """
    result: list[list[int]] = []
    for chain in chains:
        if len(chain) <= 1:
            result.append(chain)
            continue
        # Mean text direction for this chain
        z = np.exp(2j * np.radians(smoothed[chain]))
        mean_ang = float(0.5 * np.degrees(np.angle(z.mean())))
        dir_v = np.array([np.cos(np.radians(mean_ang)),
                          np.sin(np.radians(mean_ang))], dtype=np.float32)

        seg: list[int] = [chain[0]]
        prev_v_norm: np.ndarray | None = None
        for k in range(1, len(chain)):
            v = c[chain[k]] - c[chain[k - 1]]
            dist = float(np.linalg.norm(v))
            if dist < 1e-6:
                seg.append(chain[k])
                continue
            v_norm = v / dist
            along = abs(float(np.dot(v_norm, dir_v)))
            perp = float(np.sqrt(max(0.0, 1.0 - along * along)))
            inter_line = perp > along  # gap goes more up/down than left/right
            abrupt = prev_v_norm is not None and float(np.dot(v_norm, prev_v_norm)) < 0
            if inter_line or abrupt:
                result.append(seg)
                seg = [chain[k]]
                prev_v_norm = None
            else:
                seg.append(chain[k])
                prev_v_norm = v_norm
        result.append(seg)
    return result


def _merge_chains(
    chains: list[list[int]],
    regions: list[Box],
    c: np.ndarray,
    smoothed: np.ndarray,
    median_h: float,
    merge_gap: float,
    angle_tol: float,
) -> list[list[int]]:
    """Stitch chain fragments on the same text line by ray–hull collision.

    For chain A, cast a ray from its tail centroid along the text direction.
    If that ray passes through (or within median_h/2 of) any bbox corner of
    chain B within merge_gap × median_h → A and B are collinear → merge.
    Two passes handle A→B→C multi-hop cases.
    """
    from scipy.spatial import cKDTree

    if len(chains) <= 1:
        return chains

    def _mean_angle(idxs: list[int]) -> float:
        z = np.exp(2j * np.radians(smoothed[idxs]))
        return float(0.5 * np.degrees(np.angle(z.mean())))

    def _hull_corners(chain: list[int]) -> np.ndarray:
        """All four bbox corners for every member in the chain — shape (4N, 2)."""
        pts: list[list[float]] = []
        for idx in chain:
            x, y, w, h = regions[idx]
            pts.extend([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
        return np.array(pts, dtype=np.float32)

    def _ray_hits_hull(
        tail_pt: np.ndarray,
        dir_a: np.ndarray,
        b_corners: np.ndarray,
        max_along: float,
        half_width: float,
    ) -> tuple[bool, float]:
        """Return (hit, along_distance) for the closest hit in [0, max_along]."""
        v = b_corners - tail_pt  # (M, 2)
        along = v @ dir_a        # (M,) — projection onto ray
        perp = np.linalg.norm(v - np.outer(along, dir_a), axis=1)  # (M,)
        mask = (along > 0) & (along <= max_along) & (perp <= half_width)
        if not mask.any():
            return False, float("inf")
        return True, float(along[mask].min())

    def _run_pass(
        result: list[list[int]],
        mean_angles: list[float],
        consumed: set[int],
    ) -> None:
        active = [i for i in range(len(result)) if i not in consumed]
        if len(active) < 2:
            return
        # Index hull corners per active chain for fast lookup
        b_corners_map = {i: _hull_corners(result[i]) for i in active}
        # cKDTree on hull corners of all active chains for spatial pre-filtering
        all_pts = np.vstack([b_corners_map[i] for i in active])
        chain_ids = np.concatenate([
            np.full(len(b_corners_map[i]), i, dtype=int) for i in active
        ])
        tree = cKDTree(all_pts)

        max_search = merge_gap * median_h * 1.1
        half_w = median_h * 0.5

        order = sorted(active, key=lambda i: float(c[result[i][-1]][0]))
        for a_idx in order:
            if a_idx in consumed:
                continue
            tail_pt = c[result[a_idx][-1]]
            ang_rad = np.radians(mean_angles[a_idx])
            dir_a = np.array([np.cos(ang_rad), np.sin(ang_rad)], dtype=np.float32)

            # Query all hull corners within the ray's reach
            cand_pt_idxs = tree.query_ball_point(tail_pt, max_search)
            cand_chains: set[int] = {
                int(chain_ids[j]) for j in cand_pt_idxs
                if chain_ids[j] != a_idx and chain_ids[j] not in consumed
            }

            best_b: int | None = None
            best_along = float("inf")
            for b_idx in cand_chains:
                # Angle compatibility
                delta = abs(((mean_angles[b_idx] - mean_angles[a_idx] + 90.0) % 180.0) - 90.0)
                if delta >= angle_tol:
                    continue
                hit, along = _ray_hits_hull(
                    tail_pt, dir_a, b_corners_map[b_idx],
                    merge_gap * median_h, half_w,
                )
                if hit and along < best_along:
                    best_along = along
                    best_b = b_idx

            if best_b is not None:
                result[a_idx].extend(result[best_b])
                mean_angles[a_idx] = _mean_angle(result[a_idx])
                consumed.add(best_b)

    result = [list(ch) for ch in chains]
    mean_angles = [_mean_angle(ch) for ch in result]
    consumed: set[int] = set()
    _run_pass(result, mean_angles, consumed)
    _run_pass(result, mean_angles, consumed)  # second pass for A→B→C
    return [result[i] for i in range(len(result)) if i not in consumed]


def _ct_find_peaks(
    profile: np.ndarray,
    min_rel_height: float,
    min_dist: int,
) -> list[int]:
    """Non-maximum-suppression peak finder for 1-D projection profiles."""
    if len(profile) < 3:
        return []
    mx = float(np.max(profile))
    if mx <= 0:
        return []
    threshold = mx * min_rel_height
    candidates: list[tuple[int, float]] = []
    for i in range(1, len(profile) - 1):
        if profile[i] >= profile[i - 1] and profile[i] > profile[i + 1] and profile[i] >= threshold:
            candidates.append((i, float(profile[i])))
    candidates.sort(key=lambda x: x[1], reverse=True)
    selected: list[tuple[int, float]] = []
    for idx, val in candidates:
        if all(abs(idx - s) >= min_dist for s, _ in selected):
            selected.append((idx, val))
    selected.sort(key=lambda x: x[0])
    return [idx for idx, _ in selected]


def _ct_find_page_split(bin_inv: np.ndarray, search_frac: float = 0.18) -> int | None:
    """Return the x coordinate of the book spine gutter, or None if not detected."""
    h, w = bin_inv.shape[:2]
    vprof = bin_inv.sum(axis=0).astype(np.float32)
    if vprof.max() <= 0:
        return None
    k = max(9, (w // 50) | 1)
    vsm = np.convolve(vprof, np.ones(k, np.float32) / k, mode="same")
    mid, span = w // 2, max(10, int(w * search_frac))
    lo, hi = max(1, mid - span), min(w - 2, mid + span)
    if hi <= lo:
        return None
    valley_idx = int(np.argmin(vsm[lo:hi])) + lo
    left  = vsm[max(0, valley_idx - span):valley_idx]
    right = vsm[valley_idx + 1:min(w, valley_idx + span)]
    if not len(left) or not len(right):
        return None
    neighbor = float((np.median(left) + np.median(right)) / 2.0)
    if neighbor <= 1e-6:
        return None
    return valley_idx if float(vsm[valley_idx]) <= neighbor * 0.72 else None



def _ct_fwhm(profile: np.ndarray, peak: int, max_hw: int) -> int:
    """Full-width at half-maximum of a peak in a 1-D profile.

    Walks outward from the peak until the profile drops below half the peak
    value (or hits the max_hw boundary).  Returns the total width in rows —
    this is the estimated text-line height in pixels for that strip.
    """
    half = profile[peak] * 0.5
    lo = peak
    while lo > max(0, peak - max_hw) and profile[lo - 1] >= half:
        lo -= 1
    hi = peak
    while hi < min(len(profile) - 1, peak + max_hw) and profile[hi + 1] >= half:
        hi += 1
    return hi - lo


def _ct_detect_tracks(
    bin_inv: np.ndarray,
    strip_count: int,
    peak_h: float,
    peak_dist: int,
    y_tol: int,
    max_gap: int,
    smooth_win: int,
    min_track_len: int,
    x_offset: int = 0,
) -> list[list[tuple[int, int]]]:
    """Greedy strip-to-strip peak tracking. Returns tracks as (x, y) lists."""
    h, w = bin_inv.shape[:2]
    strip_w = max(1, w // strip_count)

    # ── Pass 1: compute profiles and peaks for every strip ───────────────────
    strip_profiles: list[np.ndarray] = []
    strip_peaks: list[list[int]] = []
    strip_x_centers: list[int] = []
    for s in range(strip_count):
        x0 = s * strip_w
        x1 = w if s == strip_count - 1 else min(w, (s + 1) * strip_w)
        strip = bin_inv[:, x0:x1]
        if strip.size == 0:
            strip_profiles.append(np.zeros(1, np.float32))
            strip_peaks.append([])
            strip_x_centers.append(x_offset + (x0 + x1) // 2)
            continue
        profile = np.sum(strip > 0, axis=1).astype(np.float32)
        if smooth_win > 1:
            kernel = np.ones(smooth_win, np.float32) / smooth_win
            profile = np.convolve(profile, kernel, mode="same")
        strip_profiles.append(profile)
        strip_peaks.append(_ct_find_peaks(profile, min_rel_height=peak_h, min_dist=peak_dist))
        strip_x_centers.append(x_offset + (x0 + x1) // 2)

    # ── Pass 2: FWHM outlier filter ───────────────────────────────────────────
    # For every peak compute its FWHM (line-height in pixels).  The median
    # FWHM across all peaks is the global reference for "what a real text line
    # looks like".  Peaks whose FWHM exceeds the median by more than 2.5×
    # are wide diffuse bumps (hand shadow, background object) and are dropped.
    #
    # Using the global median (not just the immediate neighbour) means this
    # works even when the hand covers the full frame width — neighbouring
    # strips both have wide bumps, but they're still outliers relative to
    # the majority of peaks on the actual text.
    #
    # After FWHM filtering, also require y-neighbour support (at least one
    # adjacent strip must have a surviving peak within y_tol) to catch any
    # remaining single-strip noise spikes.
    max_hw = peak_dist  # FWHM walk bounded to ±peak_dist px each side

    # Collect all FWHMs across all strips
    peak_fwhms: list[list[int]] = []
    all_fwhms: list[int] = []
    for s, (peaks, prof) in enumerate(zip(strip_peaks, strip_profiles)):
        fw = [_ct_fwhm(prof, p, max_hw) for p in peaks]
        peak_fwhms.append(fw)
        all_fwhms.extend(fw)

    if all_fwhms:
        median_fwhm = float(np.median(all_fwhms))
        fwhm_limit  = max(median_fwhm * 2.5, 4.0)  # never reject very short profiles
    else:
        fwhm_limit = float("inf")

    # First pass: drop FWHM outliers
    fwhm_filtered: list[list[int]] = [
        [p for p, fw in zip(peaks, fws) if fw <= fwhm_limit]
        for peaks, fws in zip(strip_peaks, peak_fwhms)
    ]

    # Second pass: require at least one y-neighbour among FWHM-clean peaks
    def _has_neighbour(s: int, y: int) -> bool:
        for nb in (s - 1, s + 1):
            if 0 <= nb < len(fwhm_filtered):
                if any(abs(y - p) <= y_tol for p in fwhm_filtered[nb]):
                    return True
        return False

    filtered_peaks = [
        [y for y in peaks if _has_neighbour(s, y)]
        for s, peaks in enumerate(fwhm_filtered)
    ]

    # ── Pass 3: greedy track association on filtered peaks ────────────────────
    tracks: list[dict] = []
    for s, peaks in enumerate(filtered_peaks):
        x_center = strip_x_centers[s]
        used_peaks: set[int] = set()
        for tr in tracks:
            if tr["missed"] > max_gap:
                continue
            best_p, best_d = None, None
            for p in peaks:
                if p in used_peaks:
                    continue
                d = abs(p - tr["last_y"])
                if d <= y_tol and (best_d is None or d < best_d):
                    best_p, best_d = p, d
            if best_p is not None:
                tr["pts"].append((x_center, best_p))
                tr["last_y"] = best_p
                tr["missed"] = 0
                used_peaks.add(best_p)
            else:
                tr["missed"] += 1
        for p in peaks:
            if p not in used_peaks:
                tracks.append({"pts": [(x_center, p)], "last_y": p, "missed": 0})

    final = [tr["pts"] for tr in tracks if len(tr["pts"]) >= min_track_len]
    final.sort(key=lambda pts: float(np.mean([p[1] for p in pts])))

    # ── Post-process: drop shadow-edge pairs ──────────────────────────────────
    # A shadow/hand boundary creates two closely-spaced tracks (top and bottom
    # edge of the shadow band).  Real text lines have roughly uniform spacing;
    # a pair closer than 55% of the median inter-track gap is very likely an
    # edge artefact, not two separate lines.  Of the pair, drop whichever has
    # fewer strip points (less supported); if equal, drop the lower one.
    if len(final) >= 3:
        ys = [float(np.mean([p[1] for p in t])) for t in final]
        gaps = [ys[i + 1] - ys[i] for i in range(len(ys) - 1)]
        med_gap = float(np.median(gaps))
        threshold = med_gap * 0.55
        drop: set[int] = set()
        for i, g in enumerate(gaps):
            if g < threshold and i not in drop and (i + 1) not in drop:
                # Keep whichever track has more strip support
                if len(final[i]) >= len(final[i + 1]):
                    drop.add(i + 1)
                else:
                    drop.add(i)
        if drop:
            final = [t for i, t in enumerate(final) if i not in drop]

    return final


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
    boxes: list[Box], y_overlap: float, x_gap: int, x_gap_scale: float = 2.0,
) -> list[list[int]]:
    """Returns member-index lists grouped into text lines.

    x_gap is the minimum pixel gap; x_gap_scale * median_h gives a
    scale-invariant gap that adapts to character height automatically.
    """
    if not boxes:
        return []
    heights = sorted(b[3] for b in boxes)
    median_h = heights[len(heights) // 2]
    tol = max(4, int((1.0 - y_overlap) * median_h))
    # Scale-invariant x gap: whichever is larger of fixed pixels or height-relative
    effective_x_gap = max(x_gap, int(median_h * x_gap_scale))

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
            if boxes[i][0] - prev_right <= effective_x_gap:
                group.append(i)
            else:
                out.append(group)
                group = [i]
        out.append(group)
    return out
