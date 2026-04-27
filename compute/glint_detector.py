"""
Corneal reflection (glint) detection for IR eye images.

Detects the bright spots created by NIR LEDs on the cornea surface.
With two LEDs, two glints may appear when gaze is forward; at lateral extremes
often only one is visible. A validated pair yields a virtual reference (midpoint)
and inter-glint separation for extended gaze models.
"""

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass, field


@dataclass
class GlintResult:
    glints: list[tuple[float, float]] = field(default_factory=list)
    primary: tuple[float, float] | None = None
    reference_point: tuple[float, float] | None = None
    pair_valid: bool = False
    glint_a: tuple[float, float] | None = None
    glint_b: tuple[float, float] | None = None
    inter_glint_sep: float | None = None
    debug_mask: np.ndarray | None = None
    intermediate_frames: dict = field(default_factory=dict)


class GlintDetector:
    """Detect IR corneal reflections (glints) in an eye image."""

    def __init__(
        self,
        brightness_thresh: int = 230,
        min_area: int = 5,
        max_area: int = 800,
        search_radius_factor: float = 2.5,
        circularity_min: float = 0.3,
        pair_min_sep_factor: float = 0.35,
        pair_max_sep_factor: float = 3.5,
        pair_search_top: int = 4,
    ):
        self.brightness_thresh = brightness_thresh
        self.min_area = min_area
        self.max_area = max_area
        self.search_radius_factor = search_radius_factor
        self.circularity_min = circularity_min
        self.pair_min_sep_factor = pair_min_sep_factor
        self.pair_max_sep_factor = pair_max_sep_factor
        self.pair_search_top = max(2, int(pair_search_top))

    def detect(
        self,
        gray: np.ndarray,
        pupil_center: tuple[int, int] | None = None,
        pupil_radius: int | None = None,
    ) -> GlintResult:
        intermediate = {}

        _, mask = cv2.threshold(gray, self.brightness_thresh, 255, cv2.THRESH_BINARY)
        intermediate["g_thresh"] = mask.copy()

        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.erode(mask, k, iterations=1)
        mask = cv2.dilate(mask, k, iterations=1)
        intermediate["g_morph"] = mask.copy()

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates: list[tuple[float, float, float, float, int]] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (self.min_area <= area <= self.max_area):
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
            gx = M["m10"] / M["m00"]
            gy = M["m01"] / M["m00"]
            blob_mask = np.zeros_like(gray)
            cv2.drawContours(blob_mask, [cnt], -1, 255, -1)
            peak = int(gray[blob_mask > 0].max())
            candidates.append((gx, gy, area, circ, peak))

        if pupil_center is not None and pupil_radius is not None:
            px, py = pupil_center
            max_dist = pupil_radius * self.search_radius_factor
            candidates = [
                c for c in candidates
                if np.hypot(c[0] - px, c[1] - py) <= max_dist
            ]

        if not candidates:
            return GlintResult(debug_mask=mask, intermediate_frames=intermediate)

        if pupil_center is not None:
            px, py = pupil_center
            candidates.sort(key=lambda c: np.hypot(c[0] - px, c[1] - py))
        else:
            candidates.sort(key=lambda c: -c[4])

        glints = [(c[0], c[1]) for c in candidates]
        primary = glints[0]

        pair_valid = False
        glint_a = glint_b = None
        inter_sep = None
        ref = primary

        r = float(pupil_radius) if pupil_radius and pupil_radius > 0 else 20.0
        dmin = self.pair_min_sep_factor * r
        dmax = self.pair_max_sep_factor * r

        if len(candidates) >= 2 and pupil_center is not None:
            top = min(self.pair_search_top, len(candidates))
            best: tuple[float, int, int] | None = None
            for i in range(top):
                for j in range(i + 1, top):
                    xi, yi, _, _, pi = candidates[i][:5]
                    xj, yj, _, _, pj = candidates[j][:5]
                    sep = float(np.hypot(xi - xj, yi - yj))
                    if not (dmin <= sep <= dmax):
                        continue
                    score = pi + pj
                    if best is None or score > best[0]:
                        best = (score, i, j)
            if best is not None:
                _, i, j = best
                ga = (candidates[i][0], candidates[i][1])
                gb = (candidates[j][0], candidates[j][1])
                if ga[0] <= gb[0]:
                    glint_a, glint_b = ga, gb
                else:
                    glint_a, glint_b = gb, ga
                pair_valid = True
                inter_sep = float(np.hypot(glint_a[0] - glint_b[0], glint_a[1] - glint_b[1]))
                ref = (
                    0.5 * (glint_a[0] + glint_b[0]),
                    0.5 * (glint_a[1] + glint_b[1]),
                )

        return GlintResult(
            glints=glints,
            primary=primary,
            reference_point=ref,
            pair_valid=pair_valid,
            glint_a=glint_a,
            glint_b=glint_b,
            inter_glint_sep=inter_sep,
            debug_mask=mask,
            intermediate_frames=intermediate,
        )
