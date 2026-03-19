"""
Corneal reflection (glint) detection for IR eye images.

Detects the bright spots created by NIR LEDs on the cornea surface.
Used together with pupil_detector to compute the PCCR pupil-glint vector.

Usage:
    from glint_detector import GlintDetector
    det = GlintDetector()
    result = det.detect(gray_frame, pupil_center=(cx, cy), pupil_radius=r)
    # result.glints  ->  list of (x, y) positions
    # result.primary ->  the single brightest/best glint for PCCR
"""

import cv2
import numpy as np
from dataclasses import dataclass, field


@dataclass
class GlintResult:
    glints: list[tuple[int, int]] = field(default_factory=list)   # all detected glints
    primary: tuple[int, int] | None = None                        # best glint for PCCR
    debug_mask: np.ndarray | None = None
    intermediate_frames: dict = field(default_factory=dict)


class GlintDetector:
    """Detect IR corneal reflections (glints) in an eye image."""

    def __init__(
        self,
        brightness_thresh: int = 230,   # minimum pixel brightness for a glint
        min_area: int = 5,              # minimum blob area (pixels)
        max_area: int = 800,            # maximum blob area (too large = not a glint)
        search_radius_factor: float = 2.5,  # search within N * pupil_radius of pupil center
        circularity_min: float = 0.3,
    ):
        self.brightness_thresh = brightness_thresh
        self.min_area = min_area
        self.max_area = max_area
        self.search_radius_factor = search_radius_factor
        self.circularity_min = circularity_min

    def detect(
        self,
        gray: np.ndarray,
        pupil_center: tuple[int, int] | None = None,
        pupil_radius: int | None = None,
    ) -> GlintResult:
        """
        Detect glints in a grayscale eye image.

        If pupil_center/radius are provided, the search is constrained to the
        region around the pupil (corneal reflections appear near or on the pupil).

        Steps:
          1. Threshold for very bright pixels
          2. Morphological cleanup
          3. Find small, bright, roughly-round blobs
          4. Filter by proximity to pupil
          5. Rank: closest to pupil center = primary glint
        """
        h, w = gray.shape
        intermediate = {}

        # 1. Threshold bright spots
        _, mask = cv2.threshold(gray, self.brightness_thresh, 255, cv2.THRESH_BINARY)
        intermediate["g_thresh"] = mask.copy()

        # 2. Light morphological cleanup
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.erode(mask, k, iterations=1)
        mask = cv2.dilate(mask, k, iterations=1)
        intermediate["g_morph"] = mask.copy()

        # 3. Find blobs
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (self.min_area <= area <= self.max_area):
                continue

            # Circularity filter (glints are roughly round)
            perim = cv2.arcLength(cnt, True)
            if perim < 1:
                continue
            circ = 4 * np.pi * area / (perim ** 2)
            if circ < self.circularity_min:
                continue

            # Centroid
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            gx = int(M["m10"] / M["m00"])
            gy = int(M["m01"] / M["m00"])

            # Peak brightness in the blob region
            blob_mask = np.zeros_like(gray)
            cv2.drawContours(blob_mask, [cnt], -1, 255, -1)
            peak = int(gray[blob_mask > 0].max())

            candidates.append((gx, gy, area, circ, peak))

        # 4. Filter by proximity to pupil if known
        if pupil_center is not None and pupil_radius is not None:
            px, py = pupil_center
            max_dist = pupil_radius * self.search_radius_factor
            candidates = [
                c for c in candidates
                if np.sqrt((c[0] - px)**2 + (c[1] - py)**2) <= max_dist
            ]

        if not candidates:
            return GlintResult(debug_mask=mask, intermediate_frames=intermediate)

        # 5. Sort: prefer glints close to pupil, with high brightness
        if pupil_center is not None:
            px, py = pupil_center
            candidates.sort(
                key=lambda c: np.sqrt((c[0]-px)**2 + (c[1]-py)**2)
            )
        else:
            # Without pupil info, sort by brightness (brightest first)
            candidates.sort(key=lambda c: -c[4])

        glints = [(c[0], c[1]) for c in candidates]
        primary = glints[0]

        return GlintResult(
            glints=glints,
            primary=primary,
            debug_mask=mask,
            intermediate_frames=intermediate,
        )
