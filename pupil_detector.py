"""
Pupil centre detection for IR eye images.

Designed for head-mounted IR camera setup (ESP32-CAM + NIR LEDs).
Works on grayscale eye-ROI frames where the pupil is the darkest region.

Usage:
    from pupil_detector import PupilDetector
    det = PupilDetector()
    result = det.detect(gray_frame)
    # result.center, result.radius, result.ellipse, result.confidence
"""

import cv2
import numpy as np
from dataclasses import dataclass


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
        glint_thresh: int = 200,        # suppress pixels brighter than this
        blur_ksize: int = 7,
        thresh_offset: int = 30,        # threshold = mean_of_dark_region + offset
        morph_ksize: int = 5,
        min_radius: int = 15,
        max_radius: int = 150,
        circularity_min: float = 0.4,
    ):
        self.glint_thresh = glint_thresh
        self.blur_ksize = blur_ksize
        self.thresh_offset = thresh_offset
        self.morph_ksize = morph_ksize
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.circularity_min = circularity_min

    def detect(self, gray: np.ndarray) -> PupilResult:
        """
        Detect pupil in a grayscale eye image.

        Steps:
          1. Suppress corneal reflections (inpaint bright spots)
          2. Blur to reduce noise
          3. Adaptive threshold to isolate the darkest blob (pupil)
          4. Morphological cleanup
          5. Find contours, score by area + circularity
          6. Fit ellipse to best contour
        """
        h, w = gray.shape
        intermediate = {}

        # 1. Suppress glints so they don't fragment the pupil blob
        clean = self._suppress_glints(gray)
        intermediate["p_suppressed"] = clean.copy()

        # 2. Blur
        ksize = (self.blur_ksize, self.blur_ksize)
        blurred = cv2.GaussianBlur(clean, ksize, 0)
        intermediate["p_blurred"] = blurred.copy()

        # 3. Adaptive threshold: find pixels darker than (darkest_percentile + offset)
        #    The pupil is the darkest region — use a threshold relative to its own intensity.
        dark_val = int(np.percentile(blurred, 10))
        tval = dark_val + self.thresh_offset
        tval = max(tval, 30)
        _, mask = cv2.threshold(blurred, tval, 255, cv2.THRESH_BINARY_INV)
        intermediate["p_thresh"] = mask.copy()

        # 4. Morphological close to fill internal holes, then open to remove noise
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (self.morph_ksize, self.morph_ksize))
        mask = cv2.dilate(mask, k, iterations=2)
        mask = cv2.erode(mask, k, iterations=1)
        intermediate["p_morph"] = mask.copy()

        # 5. Find contours and score
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
            # Score: prefer large area + high circularity + central location
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

        # Pick best scoring contour
        best = max(candidates, key=lambda x: x[1])
        cnt, score, circ, mx, my, area = best

        # 6. Fit ellipse
        ellipse = None
        radius = int(np.sqrt(area / np.pi))
        if len(cnt) >= 5:
            ellipse = cv2.fitEllipse(cnt)
            # Use ellipse center for better subpixel accuracy
            mx = int(ellipse[0][0])
            my = int(ellipse[0][1])
            radius = int((ellipse[1][0] + ellipse[1][1]) / 4)

        confidence = min(circ, 1.0)

        return PupilResult(
            center=(mx, my),
            radius=radius,
            ellipse=ellipse,
            confidence=confidence,
            debug_mask=mask,
            intermediate_frames=intermediate,
        )

    def _suppress_glints(self, gray: np.ndarray) -> np.ndarray:
        """Inpaint bright corneal reflections so they don't split the pupil blob."""
        _, bright = cv2.threshold(gray, self.glint_thresh, 255, cv2.THRESH_BINARY)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        bright = cv2.dilate(bright, k, iterations=2)
        if np.count_nonzero(bright) == 0:
            return gray
        return cv2.inpaint(gray, bright, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
