"""
Pupil centre detection for IR eye images.

Designed for head-mounted IR camera setup (ESP32-CAM + NIR LEDs).
Works on grayscale eye-ROI frames where the pupil is the darkest region.

Algorithms:
  1. "threshold" — Adaptive dark-blob thresholding + contour scoring (original)
  2. "edge"      — Canny edges + HoughCircles to find iris boundary,
                   then locate darkest region inside it
  3. "gradient"  — Timm & Barth gradient voting: each gradient vector votes
                   for the center it points toward; no thresholding needed
  4. "seed"      — Find darkest point, flood-fill outward to grow the pupil
                   region, then fit a tight ellipse to the contour

Usage:
    from pupil_detector import PupilDetector
    det = PupilDetector(algorithm="threshold")
    result = det.detect(gray_frame)
"""

import cv2
import numpy as np
from dataclasses import dataclass, field

ALGORITHMS = ("threshold", "edge", "gradient", "seed")


@dataclass
class PupilResult:
    center: tuple[int, int] | None = None       # (cx, cy)
    radius: int | None = None                    # approximate radius
    ellipse: tuple | None = None                 # cv2 RotatedRect (center, axes, angle)
    confidence: float = 0.0                      # 0..1
    debug_mask: np.ndarray | None = None         # binary threshold (for visualisation)
    intermediate_frames: dict = field(default_factory=dict)


class PupilDetector:
    """Detect the pupil as the darkest circular blob in an IR eye image."""

    def __init__(
        self,
        algorithm: str = "threshold",
        glint_thresh: int = 200,
        blur_ksize: int = 7,
        thresh_offset: int = 30,
        dark_percentile: float = 10.0,
        morph_ksize: int = 5,
        min_radius: int = 15,
        max_radius: int = 150,
        circularity_min: float = 0.4,
        # Edge algorithm params
        canny_low: int = 30,
        canny_high: int = 100,
        hough_dp: float = 1.5,
        hough_param1: int = 100,
        hough_param2: int = 30,
        # Gradient algorithm params
        gradient_downscale: int = 2,
        # Seed algorithm params
        seed_flood_tolerance: int = 40,
    ):
        self.algorithm = algorithm if algorithm in ALGORITHMS else "threshold"
        self.glint_thresh = glint_thresh
        self.blur_ksize = blur_ksize
        self.thresh_offset = thresh_offset
        self.dark_percentile = dark_percentile
        self.morph_ksize = morph_ksize
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.circularity_min = circularity_min
        # Edge params
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.hough_dp = hough_dp
        self.hough_param1 = hough_param1
        self.hough_param2 = hough_param2
        # Gradient params
        self.gradient_downscale = gradient_downscale
        # Seed params
        self.seed_flood_tolerance = seed_flood_tolerance

    def detect(self, gray: np.ndarray) -> PupilResult:
        if self.algorithm == "edge":
            return self._detect_edge(gray)
        elif self.algorithm == "gradient":
            return self._detect_gradient(gray)
        elif self.algorithm == "seed":
            return self._detect_seed(gray)
        else:
            return self._detect_threshold(gray)

    # ── Algorithm 1: Threshold (original) ─────────────────────────────────────

    def _detect_threshold(self, gray: np.ndarray) -> PupilResult:
        h, w = gray.shape
        intermediate = {}

        clean = self._suppress_glints(gray)
        intermediate["p_suppressed"] = clean.copy()

        ksize = (self.blur_ksize, self.blur_ksize)
        blurred = cv2.GaussianBlur(clean, ksize, 0)
        intermediate["p_blurred"] = blurred.copy()

        dark_val = int(np.percentile(blurred, self.dark_percentile))
        tval = max(dark_val + self.thresh_offset, 5)
        _, mask = cv2.threshold(blurred, tval, 255, cv2.THRESH_BINARY_INV)
        intermediate["p_thresh"] = mask.copy()

        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (self.morph_ksize, self.morph_ksize))
        mask = cv2.dilate(mask, k, iterations=2)
        mask = cv2.erode(mask, k, iterations=1)
        intermediate["p_morph"] = mask.copy()

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return PupilResult(debug_mask=mask, intermediate_frames=intermediate)

        min_area = np.pi * self.min_radius ** 2
        max_area = np.pi * self.max_radius ** 2

        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (min_area <= area <= max_area):
                continue
            perim = cv2.arcLength(cnt, True)
            if perim < 1:
                continue
            circ = 4 * np.pi * area / (perim ** 2)
            if circ < self.circularity_min:
                continue
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            mx = int(M["m10"] / M["m00"])
            my = int(M["m01"] / M["m00"])
            dist_from_center = np.sqrt((mx - w/2)**2 + (my - h/2)**2)
            centrality = 1.0 - min(dist_from_center / (max(w, h) / 2), 1.0)
            score = area * circ * (0.5 + 0.5 * centrality)
            candidates.append((cnt, score, circ, mx, my, area))

        if not candidates:
            return PupilResult(debug_mask=mask, intermediate_frames=intermediate)

        best = max(candidates, key=lambda x: x[1])
        cnt, score, circ, mx, my, area = best

        ellipse = None
        radius = int(np.sqrt(area / np.pi))
        if len(cnt) >= 5:
            ellipse = cv2.fitEllipse(cnt)
            mx = int(ellipse[0][0])
            my = int(ellipse[0][1])
            radius = int((ellipse[1][0] + ellipse[1][1]) / 4)

        confidence = min(circ, 1.0)

        return PupilResult(
            center=(mx, my), radius=radius, ellipse=ellipse,
            confidence=confidence, debug_mask=mask,
            intermediate_frames=intermediate,
        )

    # ── Algorithm 2: Edge (Canny + HoughCircles) ─────────────────────────────

    def _detect_edge(self, gray: np.ndarray) -> PupilResult:
        """
        Use edge detection to find circular boundaries, then pick the circle
        whose interior is darkest (pupil = dark disk inside lighter iris).
        """
        h, w = gray.shape
        intermediate = {}

        clean = self._suppress_glints(gray)
        intermediate["p_suppressed"] = clean.copy()

        ksize = (self.blur_ksize, self.blur_ksize)
        blurred = cv2.GaussianBlur(clean, ksize, 0)
        intermediate["p_blurred"] = blurred.copy()

        # Canny edge detection
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)
        intermediate["p_edges"] = edges.copy()

        # Dilate edges slightly to close small gaps
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges_d = cv2.dilate(edges, k, iterations=1)
        intermediate["p_thresh"] = edges_d.copy()

        # HoughCircles to find candidate circles
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT,
            dp=self.hough_dp,
            minDist=max(self.min_radius * 2, 30),
            param1=self.hough_param1,
            param2=self.hough_param2,
            minRadius=self.min_radius,
            maxRadius=self.max_radius,
        )

        if circles is None:
            return PupilResult(debug_mask=edges, intermediate_frames=intermediate)

        circles = np.round(circles[0]).astype(int)

        # Score each circle: prefer dark interior (low mean) + central position
        best_score = -1
        best_circle = None
        for cx, cy, r in circles:
            # Create a mask for this circle's interior
            cmask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(cmask, (cx, cy), r, 255, -1)
            interior = blurred[cmask > 0]
            if len(interior) == 0:
                continue
            mean_brightness = float(interior.mean())
            # Lower brightness = better (pupil is darkest)
            # Also score centrality
            dist = np.sqrt((cx - w/2)**2 + (cy - h/2)**2)
            centrality = 1.0 - min(dist / (max(w, h) / 2), 1.0)
            # Invert brightness: max brightness is 255
            darkness_score = (255 - mean_brightness) / 255.0
            score = darkness_score * (0.6 + 0.4 * centrality)
            if score > best_score:
                best_score = score
                best_circle = (cx, cy, r)

        if best_circle is None:
            return PupilResult(debug_mask=edges, intermediate_frames=intermediate)

        cx, cy, r = best_circle

        # Build a debug visualization: edges with the winning circle
        morph_vis = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        for ccx, ccy, cr in circles:
            cv2.circle(morph_vis, (ccx, ccy), cr, (80, 80, 80), 1)
        cv2.circle(morph_vis, (cx, cy), r, (255, 255, 255), 2)
        morph_gray = cv2.cvtColor(morph_vis, cv2.COLOR_BGR2GRAY)
        intermediate["p_morph"] = morph_gray

        confidence = min(best_score * 1.2, 1.0)

        return PupilResult(
            center=(cx, cy), radius=r, ellipse=None,
            confidence=confidence, debug_mask=edges,
            intermediate_frames=intermediate,
        )

    # ── Algorithm 3: Gradient voting (Timm & Barth) ──────────────────────────

    def _detect_gradient(self, gray: np.ndarray) -> PupilResult:
        """
        Each pixel's gradient direction votes for the center it points toward.
        The pixel with the most weighted votes is the pupil center.
        Robust — no thresholding or shape assumptions needed.
        """
        h, w = gray.shape
        intermediate = {}

        clean = self._suppress_glints(gray)
        intermediate["p_suppressed"] = clean.copy()

        ksize = (self.blur_ksize, self.blur_ksize)
        blurred = cv2.GaussianBlur(clean, ksize, 0)
        intermediate["p_blurred"] = blurred.copy()

        # Downscale for speed
        ds = max(1, self.gradient_downscale)
        if ds > 1:
            small = cv2.resize(blurred, (w // ds, h // ds), interpolation=cv2.INTER_AREA)
        else:
            small = blurred
        sh, sw = small.shape

        # Compute gradients
        gx = cv2.Sobel(small, cv2.CV_64F, 1, 0, ksize=5)
        gy = cv2.Sobel(small, cv2.CV_64F, 0, 1, ksize=5)
        mag = np.sqrt(gx**2 + gy**2)

        # Only use strong gradients (edges)
        mag_thresh = np.percentile(mag, 90)
        strong = mag > mag_thresh
        intermediate["p_thresh"] = (strong.astype(np.uint8) * 255)

        # Normalize gradient directions — negate so they point inward
        # (pupil is dark: gradients naturally point outward from dark→light,
        #  we want them pointing toward the center)
        mag_safe = np.where(mag > 1e-6, mag, 1.0)
        dx = -gx / mag_safe
        dy = -gy / mag_safe

        # Darkness weight: darker pixels get more votes as center candidates
        inv = 255.0 - small.astype(np.float64)
        inv = cv2.GaussianBlur(inv, (5, 5), 0)

        # Accumulator
        accum = np.zeros((sh, sw), dtype=np.float64)

        # Get strong gradient pixel locations
        ys, xs = np.where(strong)
        if len(xs) == 0:
            return PupilResult(intermediate_frames=intermediate)

        dxs = dx[ys, xs]
        dys = dy[ys, xs]

        # For each candidate center, sum dot products with gradient directions
        # Use vectorized sampling: pick a grid of candidate centers
        step = max(1, min(sh, sw) // 40)
        cy_range = np.arange(self.min_radius // ds, sh - self.min_radius // ds, step)
        cx_range = np.arange(self.min_radius // ds, sw - self.min_radius // ds, step)

        if len(cy_range) == 0 or len(cx_range) == 0:
            return PupilResult(intermediate_frames=intermediate)

        for cy in cy_range:
            for cx in cx_range:
                # Vector from gradient pixel to candidate center
                vx = cx - xs
                vy = cy - ys
                vmag = np.sqrt(vx**2 + vy**2)
                vmag = np.where(vmag > 1e-6, vmag, 1.0)
                vx_n = vx / vmag
                vy_n = vy / vmag
                # Dot product: how well does gradient point toward this center?
                dot = dxs * vx_n + dys * vy_n
                dot = np.maximum(dot, 0)  # only positive votes
                # Weight by darkness at the candidate center
                score = np.sum(dot) * inv[cy, cx]
                accum[cy, cx] = score

        # Refine: find peak in accumulator
        accum_blur = cv2.GaussianBlur(accum, (5, 5), 0)
        intermediate["p_morph"] = (accum_blur / (accum_blur.max() + 1e-6) * 255).astype(np.uint8)

        _, _, _, max_loc = cv2.minMaxLoc(accum_blur)
        best_cx, best_cy = max_loc  # in downscaled coords

        # Subpixel refinement: fit parabola around peak
        best_cx, best_cy = self._refine_peak(accum_blur, best_cx, best_cy)

        # Scale back up
        cx_full = int(best_cx * ds)
        cy_full = int(best_cy * ds)

        # Estimate radius: find the dark region around the center
        radius = self._estimate_radius_from_center(blurred, cx_full, cy_full)

        confidence = float(accum_blur[int(best_cy), int(best_cx)]) / (accum_blur.max() + 1e-6)
        confidence = min(confidence * 1.2, 1.0)

        return PupilResult(
            center=(cx_full, cy_full), radius=radius, ellipse=None,
            confidence=confidence, debug_mask=intermediate.get("p_morph"),
            intermediate_frames=intermediate,
        )

    def _refine_peak(self, acc: np.ndarray, cx: int, cy: int) -> tuple[float, float]:
        """Parabolic subpixel refinement around peak."""
        h, w = acc.shape
        rx, ry = float(cx), float(cy)
        if 1 <= cx < w - 1:
            l, c, r = acc[cy, cx-1], acc[cy, cx], acc[cy, cx+1]
            denom = 2 * (2*c - l - r)
            if abs(denom) > 1e-6:
                rx = cx + (l - r) / denom
        if 1 <= cy < h - 1:
            t, c, b = acc[cy-1, cx], acc[cy, cx], acc[cy+1, cx]
            denom = 2 * (2*c - t - b)
            if abs(denom) > 1e-6:
                ry = cy + (t - b) / denom
        return rx, ry

    def _estimate_radius_from_center(self, blurred: np.ndarray, cx: int, cy: int) -> int:
        """Estimate pupil radius by scanning outward from center until brightness jumps."""
        h, w = blurred.shape
        cx = max(0, min(w-1, cx))
        cy = max(0, min(h-1, cy))
        center_val = float(blurred[cy, cx])

        best_r = self.min_radius
        for r in range(self.min_radius, self.max_radius):
            # Sample points on circle at radius r
            angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
            xs = (cx + r * np.cos(angles)).astype(int)
            ys = (cy + r * np.sin(angles)).astype(int)
            valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
            if valid.sum() < 4:
                break
            ring_mean = float(blurred[ys[valid], xs[valid]].mean())
            # When ring brightness exceeds center by enough, we've left the pupil
            if ring_mean > center_val + 30:
                best_r = r
                break
        return best_r

    # ── Algorithm 4: Seed (flood-fill from darkest point + ellipse fit) ─────

    def _detect_seed(self, gray: np.ndarray) -> PupilResult:
        """
        1. Suppress glints, blur
        2. Find the darkest small patch as seed point
        3. Flood-fill outward from seed — pixels within tolerance of the
           seed's brightness are considered part of the pupil
        4. Find the contour of the filled region
        5. Fit an ellipse tightly to that contour
        """
        h, w = gray.shape
        intermediate = {}

        clean = self._suppress_glints(gray)
        intermediate["p_suppressed"] = clean.copy()

        ksize = (self.blur_ksize, self.blur_ksize)
        blurred = cv2.GaussianBlur(clean, ksize, 0)
        intermediate["p_blurred"] = blurred.copy()

        # Find seed: darkest NxN patch (not single pixel — more robust to noise)
        patch = 7
        # Use integral image for fast patch-mean computation
        integral = cv2.integral(blurred)
        # Compute mean brightness over patch_size x patch_size windows
        p = patch // 2
        # Crop ranges to stay within bounds
        y1s = np.arange(p, h - p)
        x1s = np.arange(p, w - p)
        if len(y1s) == 0 or len(x1s) == 0:
            return PupilResult(intermediate_frames=intermediate)

        # Vectorized patch mean using integral image
        # integral[r2+1,c2+1] - integral[r1,c2+1] - integral[r2+1,c1] + integral[r1,c1]
        r1 = y1s - p
        r2 = y1s + p
        c1 = x1s - p
        c2 = x1s + p
        # Create 2D grids
        R1, C1 = np.meshgrid(r1, c1, indexing='ij')
        R2, C2 = np.meshgrid(r2, c2, indexing='ij')
        area = patch * patch
        patch_sums = (integral[R2+1, C2+1] - integral[R1, C2+1]
                      - integral[R2+1, C1] + integral[R1, C1])
        patch_means = patch_sums / area

        # Find the darkest patch
        min_idx = np.unravel_index(patch_means.argmin(), patch_means.shape)
        seed_y = int(y1s[min_idx[0]])
        seed_x = int(x1s[min_idx[1]])
        seed_val = float(blurred[seed_y, seed_x])

        # Show seed on debug
        seed_vis = blurred.copy()
        cv2.circle(seed_vis, (seed_x, seed_y), 5, 255, 2)
        intermediate["p_thresh"] = seed_vis

        # Flood fill from seed
        # cv2.floodFill uses a mask that is 2 pixels larger than the image
        flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
        fill_img = blurred.copy()
        lo_diff = int(self.seed_flood_tolerance)
        hi_diff = int(self.seed_flood_tolerance)
        # Fill with a known value (255)
        # FLOODFILL_FIXED_RANGE: compare each pixel against the seed value,
        # not against its neighbor — prevents leaking through uniform background
        cv2.floodFill(fill_img, flood_mask, (seed_x, seed_y),
                       newVal=255, loDiff=(lo_diff,), upDiff=(hi_diff,),
                       flags=cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE
                             | (255 << 8) | 4)

        # Extract the filled region from the mask (inner part, skip 1px border)
        region = flood_mask[1:-1, 1:-1]
        intermediate["p_morph"] = region.copy()

        # Find contours
        contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return PupilResult(debug_mask=region, intermediate_frames=intermediate)

        # Pick the largest contour that fits radius constraints
        min_area = np.pi * self.min_radius ** 2
        max_area = np.pi * self.max_radius ** 2

        # Sort by area descending
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        best_cnt = None
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_area <= area <= max_area:
                best_cnt = cnt
                break

        if best_cnt is None:
            # Fallback: take the largest contour if it exists
            if cv2.contourArea(contours[0]) > 20:
                best_cnt = contours[0]
            else:
                return PupilResult(debug_mask=region, intermediate_frames=intermediate)

        area = cv2.contourArea(best_cnt)
        perim = cv2.arcLength(best_cnt, True)
        circ = (4 * np.pi * area / (perim ** 2)) if perim > 0 else 0

        # Fit ellipse (needs >= 5 points)
        ellipse = None
        M = cv2.moments(best_cnt)
        if M["m00"] == 0:
            return PupilResult(debug_mask=region, intermediate_frames=intermediate)
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        radius = int(np.sqrt(area / np.pi))

        if len(best_cnt) >= 5:
            ellipse = cv2.fitEllipse(best_cnt)
            cx = int(ellipse[0][0])
            cy = int(ellipse[0][1])
            radius = int((ellipse[1][0] + ellipse[1][1]) / 4)

        confidence = min(circ * 1.2, 1.0)

        return PupilResult(
            center=(cx, cy), radius=radius, ellipse=ellipse,
            confidence=confidence, debug_mask=region,
            intermediate_frames=intermediate,
        )

    # ── Shared helpers ────────────────────────────────────────────────────────

    def _suppress_glints(self, gray: np.ndarray) -> np.ndarray:
        """Inpaint bright corneal reflections so they don't split the pupil blob."""
        _, bright = cv2.threshold(gray, self.glint_thresh, 255, cv2.THRESH_BINARY)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        bright = cv2.dilate(bright, k, iterations=2)
        if np.count_nonzero(bright) == 0:
            return gray
        return cv2.inpaint(gray, bright, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
