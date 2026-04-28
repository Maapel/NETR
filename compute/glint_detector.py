"""
Corneal reflection (glint) detection for IR eye images.

Iris is segmented by casting radial rays from the pupil edge and finding the
iris→sclera intensity jump (limbus boundary). Glints are then found by
adaptive thresholding within the iris annulus mask — threshold adapts to the
99th-percentile brightness in that region, so it works across IR illumination
levels without manual tuning.

Falls back to disk-approximation + fixed threshold when pupil is not detected
or limbus detection fails.
"""

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass, field


# ── Iris segmentation ─────────────────────────────────────────────────────────

def _estimate_limbus_radius(
    gray: np.ndarray,
    cx: float,
    cy: float,
    pupil_radius: float,
    n_rays: int = 16,
    max_factor: float = 4.0,
    min_gradient: float = 4.0,
    smooth_k: int = 5,
) -> float | None:
    """Cast radial rays from pupil edge, find iris→sclera intensity jump.

    Returns median limbus radius (px from pupil center), or None if < half
    the rays found a clear edge.
    """
    h, w = gray.shape
    r_min = int(pupil_radius * 1.05)
    r_max = int(pupil_radius * max_factor)
    if r_max <= r_min:
        return None

    radii = np.arange(r_min, r_max, dtype=np.float32)
    angles = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
    limbus_rs: list[float] = []

    kernel = np.ones(smooth_k, np.float32) / smooth_k

    for angle in angles:
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        xs = np.clip(cx + cos_a * radii, 0, w - 1).astype(np.int32)
        ys = np.clip(cy + sin_a * radii, 0, h - 1).astype(np.int32)
        profile = gray[ys, xs].astype(np.float32)

        smoothed = np.convolve(profile, kernel, mode="same")
        grad = np.diff(smoothed)
        # Skip first smooth_k//2 samples — zero-padding edge artefact
        skip = smooth_k // 2
        grad = grad[skip:]
        if len(grad) < 2:
            continue
        peak_i = int(np.argmax(grad)) + skip
        if grad[peak_i - skip] < min_gradient:
            continue
        limbus_rs.append(float(radii[peak_i]))

    if len(limbus_rs) < n_rays // 2:
        return None
    return float(np.median(limbus_rs))


def _build_iris_mask(
    gray: np.ndarray,
    pupil_center: tuple[float, float],
    pupil_radius: float,
    pupil_ellipse: tuple | None,
    iris_radius_factor: float,
    limbus_radius: float | None,
) -> tuple[np.ndarray, float]:
    """Build annular iris mask and return (mask, used_limbus_radius).

    Outer boundary: detected limbus circle (or disk fallback).
    Inner boundary: pupil ellipse or circle, eroded slightly so glints right
    on the pupil edge are still kept.
    """
    h, w = gray.shape
    cx, cy = int(round(pupil_center[0])), int(round(pupil_center[1]))
    pr = float(pupil_radius)

    outer_r = limbus_radius if limbus_radius is not None else pr * iris_radius_factor

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), int(outer_r), 255, -1)

    # Subtract pupil — shrink slightly so glints on cornea edge aren't clipped
    inner_r = max(1, int(pr * 0.85))
    if pupil_ellipse is not None:
        (ecx, ecy), (ew, eh), eang = pupil_ellipse
        shrunk = (ecx, ecy), (ew * 0.85, eh * 0.85), eang
        cv2.ellipse(mask, shrunk, 0, -1)
    else:
        cv2.circle(mask, (cx, cy), inner_r, 0, -1)

    return mask, outer_r


# ── Glint threshold ───────────────────────────────────────────────────────────

def _adaptive_glint_thresh(
    gray: np.ndarray,
    iris_mask: np.ndarray,
    glint_margin: int,
    fallback_thresh: int,
) -> int:
    """99th-percentile brightness in iris region minus margin.

    Adapts to per-frame IR illumination level. Falls back to fixed threshold
    when the iris mask is empty or the computed value would be unreasonably low.
    """
    pixels = gray[iris_mask > 0]
    if pixels.size == 0:
        return fallback_thresh
    peak = int(np.percentile(pixels, 99))
    thresh = peak - glint_margin
    # Never go below fallback — avoids detecting iris texture as glints
    return max(thresh, fallback_thresh)


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class GlintResult:
    glints: list[tuple[float, float]] = field(default_factory=list)
    primary: tuple[float, float] | None = None
    debug_mask: np.ndarray | None = None
    iris_mask: np.ndarray | None = None
    limbus_radius: float | None = None
    intermediate_frames: dict = field(default_factory=dict)


# ── Detector ──────────────────────────────────────────────────────────────────

class GlintDetector:
    """Detect IR corneal reflections (glints) in an eye image.

    When pupil_center + pupil_radius are provided the detector:
      1. Estimates the limbus radius via radial intensity profiling.
      2. Builds an iris annulus mask (limbus circle minus pupil area).
      3. Computes an adaptive threshold from the iris region's peak brightness.
      4. Finds bright blobs only within the iris mask.

    Without pupil info falls back to global threshold + disk filter.
    """

    def __init__(
        self,
        # Adaptive threshold: glint must be within `glint_margin` px of
        # the iris peak brightness.  Lower = stricter (fewer false positives).
        glint_margin: int = 30,
        # Hard floor: adaptive threshold never goes below this value.
        brightness_thresh: int = 180,
        min_area: int = 5,
        max_area: int = 800,
        circularity_min: float = 0.3,
        # Limbus detection
        limbus_n_rays: int = 16,
        limbus_max_factor: float = 4.0,
        limbus_min_gradient: float = 4.0,
        # Fallback disk (used when limbus detection fails)
        iris_radius_factor: float = 2.15,
        # Legacy — still used for search_radius pre-filter
        search_radius_factor: float = 2.5,
        ellipse_slack: float = 0.12,
    ):
        self.glint_margin = glint_margin
        self.brightness_thresh = brightness_thresh
        self.min_area = min_area
        self.max_area = max_area
        self.circularity_min = circularity_min
        self.limbus_n_rays = limbus_n_rays
        self.limbus_max_factor = limbus_max_factor
        self.limbus_min_gradient = limbus_min_gradient
        self.iris_radius_factor = iris_radius_factor
        self.search_radius_factor = search_radius_factor
        self.ellipse_slack = ellipse_slack

    def detect(
        self,
        gray: np.ndarray,
        pupil_center: tuple[float, float] | None = None,
        pupil_radius: int | None = None,
        pupil_ellipse: tuple | None = None,
    ) -> GlintResult:
        intermediate: dict = {}
        iris_mask: np.ndarray | None = None
        limbus_radius: float | None = None

        if pupil_center is not None and pupil_radius is not None:
            pr = float(pupil_radius)
            cx, cy = float(pupil_center[0]), float(pupil_center[1])

            limbus_radius = _estimate_limbus_radius(
                gray, cx, cy, pr,
                n_rays=self.limbus_n_rays,
                max_factor=self.limbus_max_factor,
                min_gradient=self.limbus_min_gradient,
            )

            iris_mask, limbus_radius = _build_iris_mask(
                gray, pupil_center, pr, pupil_ellipse,
                self.iris_radius_factor, limbus_radius,
            )
            intermediate["g_iris_mask"] = iris_mask

            thresh = _adaptive_glint_thresh(
                gray, iris_mask, self.glint_margin, self.brightness_thresh,
            )
        else:
            thresh = self.brightness_thresh

        _, bin_mask = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
        intermediate["g_thresh"] = bin_mask.copy()

        # Restrict to iris region when available
        if iris_mask is not None:
            bin_mask = cv2.bitwise_and(bin_mask, iris_mask)

        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        bin_mask = cv2.erode(bin_mask, k, iterations=1)
        bin_mask = cv2.dilate(bin_mask, k, iterations=1)
        intermediate["g_morph"] = bin_mask.copy()

        contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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

        # Legacy distance pre-filter when no iris mask (no pupil info)
        if iris_mask is None and pupil_center is not None and pupil_radius is not None:
            px, py = float(pupil_center[0]), float(pupil_center[1])
            max_dist = float(pupil_radius) * self.search_radius_factor
            candidates = [
                c for c in candidates
                if np.hypot(c[0] - px, c[1] - py) <= max_dist
            ]

        if not candidates:
            return GlintResult(
                debug_mask=bin_mask,
                iris_mask=iris_mask,
                limbus_radius=limbus_radius,
                intermediate_frames=intermediate,
            )

        if pupil_center is not None:
            px, py = float(pupil_center[0]), float(pupil_center[1])
            candidates.sort(key=lambda c: (np.hypot(c[0] - px, c[1] - py), -c[4]))
        else:
            candidates.sort(key=lambda c: -c[4])

        glints = [(c[0], c[1]) for c in candidates]
        primary = glints[0]

        return GlintResult(
            glints=glints,
            primary=primary,
            debug_mask=bin_mask,
            iris_mask=iris_mask,
            limbus_radius=limbus_radius,
            intermediate_frames=intermediate,
        )
