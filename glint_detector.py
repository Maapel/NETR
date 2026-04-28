"""
Corneal reflection (glint) detection for IR eye images.

Bright blobs are filtered to those lying on the pupil/iris (not sclera), then
the best candidate (closest to pupil center, tie-break peak brightness) is primary.
"""

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass, field


def _point_in_ellipse(px: float, py: float, ellipse: tuple, slack: float) -> bool:
    """True if (px,py) lies inside the pupil ellipse expanded by slack (fraction of semi-axes)."""
    (cx, cy), (w, h), ang_deg = ellipse
    if w < 1e-6 or h < 1e-6:
        return False
    a = (w / 2.0) * (1.0 + slack)
    b = (h / 2.0) * (1.0 + slack)
    rad = np.radians(float(ang_deg))
    ca, sa = np.cos(rad), np.sin(rad)
    dx, dy = float(px) - float(cx), float(py) - float(cy)
    xr = ca * dx + sa * dy
    yr = -sa * dx + ca * dy
    return (xr / a) ** 2 + (yr / b) ** 2 <= 1.0 + 1e-9


def _point_in_iris_disk(
    px: float, py: float,
    pcx: float, pcy: float,
    pupil_radius: float,
    iris_radius_factor: float,
) -> bool:
    """True if glint lies inside limbus-sized disk around pupil center."""
    r = max(pupil_radius, 1.0) * iris_radius_factor
    return np.hypot(px - pcx, py - pcy) <= r + 1e-9


def _glint_on_pupil_iris(
    gx: float, gy: float,
    pupil_center: tuple[float, float] | None,
    pupil_radius: int | None,
    pupil_ellipse: tuple | None,
    iris_radius_factor: float,
    ellipse_slack: float,
) -> bool:
    if pupil_center is None or pupil_radius is None:
        return True
    pcx, pcy = float(pupil_center[0]), float(pupil_center[1])
    r = float(pupil_radius)
    if pupil_ellipse is not None:
        return _point_in_ellipse(gx, gy, pupil_ellipse, ellipse_slack)
    return _point_in_iris_disk(gx, gy, pcx, pcy, r, iris_radius_factor)


@dataclass
class GlintResult:
    glints: list[tuple[float, float]] = field(default_factory=list)
    primary: tuple[float, float] | None = None
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
        iris_radius_factor: float = 2.15,
        ellipse_slack: float = 0.12,
    ):
        self.brightness_thresh = brightness_thresh
        self.min_area = min_area
        self.max_area = max_area
        self.search_radius_factor = search_radius_factor
        self.circularity_min = circularity_min
        self.iris_radius_factor = iris_radius_factor
        self.ellipse_slack = ellipse_slack

    def detect(
        self,
        gray: np.ndarray,
        pupil_center: tuple[float, float] | None = None,
        pupil_radius: int | None = None,
        pupil_ellipse: tuple | None = None,
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
            px, py = float(pupil_center[0]), float(pupil_center[1])
            max_dist = float(pupil_radius) * self.search_radius_factor
            candidates = [
                c for c in candidates
                if np.hypot(c[0] - px, c[1] - py) <= max_dist
            ]

        valid = [
            c for c in candidates
            if _glint_on_pupil_iris(
                c[0], c[1],
                pupil_center, pupil_radius, pupil_ellipse,
                self.iris_radius_factor, self.ellipse_slack,
            )
        ]

        if not valid:
            return GlintResult(debug_mask=mask, intermediate_frames=intermediate)

        if pupil_center is not None:
            px, py = float(pupil_center[0]), float(pupil_center[1])
            valid.sort(
                key=lambda c: (
                    np.hypot(c[0] - px, c[1] - py),
                    -c[4],
                )
            )
        else:
            valid.sort(key=lambda c: -c[4])

        glints = [(c[0], c[1]) for c in valid]
        primary = glints[0]

        return GlintResult(
            glints=glints,
            primary=primary,
            debug_mask=mask,
            intermediate_frames=intermediate,
        )
