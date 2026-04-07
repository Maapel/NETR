"""
Eye analysis pipeline — combines pupil + glint detection for PCCR.

This is the single entry point to drop into receiver.py. It takes a
grayscale frame and returns pupil center, glint position, and the
pupil-glint vector needed for gaze mapping.

Usage:
    from eye_pipeline import EyePipeline
    pipe = EyePipeline()
    result = pipe.process(gray_frame)
    # result.pupil_center, result.glint, result.pccr_vector, ...

    # Or draw the overlay on a BGR frame:
    annotated = pipe.draw(bgr_frame, result)
"""

import cv2
import numpy as np
from dataclasses import dataclass, field

from pupil_detector import PupilDetector, PupilResult, ALGORITHMS
from glint_detector import GlintDetector, GlintResult


@dataclass
class EyeResult:
    pupil: PupilResult
    glint: GlintResult
    pupil_center: tuple[int, int] | None = None
    pupil_radius: int | None = None
    glint_pos: tuple[int, int] | None = None
    pccr_vector: tuple[float, float] | None = None   # (dx, dy) glint→pupil
    intermediate_frames: dict = field(default_factory=dict)


class EyePipeline:
    """Full eye analysis: pupil detection → glint detection → PCCR vector."""

    def __init__(self, pupil_kwargs=None, glint_kwargs=None):
        self._pupil_det = PupilDetector(**(pupil_kwargs or {}))
        self._glint_det = GlintDetector(**(glint_kwargs or {}))

    def update_params(self, params: dict):
        """Update detector parameters at runtime. Keys prefixed with 'p_' go to
        PupilDetector, 'g_' go to GlintDetector."""
        for k, v in params.items():
            if k == "p_algorithm":
                if v in ALGORITHMS:
                    self._pupil_det.algorithm = v
            elif k.startswith("p_"):
                attr = k[2:]
                if hasattr(self._pupil_det, attr):
                    setattr(self._pupil_det, attr, type(getattr(self._pupil_det, attr))(v))
            elif k.startswith("g_"):
                attr = k[2:]
                if hasattr(self._glint_det, attr):
                    setattr(self._glint_det, attr, type(getattr(self._glint_det, attr))(v))

    def get_params(self) -> dict:
        """Return all tuneable parameters as a flat dict."""
        d = {}
        d["p_algorithm"] = self._pupil_det.algorithm
        for attr in ("glint_thresh", "blur_ksize", "thresh_offset", "dark_percentile",
                      "morph_ksize", "min_radius", "max_radius", "circularity_min",
                      "canny_low", "canny_high", "hough_dp", "hough_param1",
                      "hough_param2", "gradient_downscale",
                      "seed_flood_tolerance"):
            d["p_" + attr] = getattr(self._pupil_det, attr)
        for attr in ("brightness_thresh", "min_area", "max_area",
                      "search_radius_factor", "circularity_min"):
            d["g_" + attr] = getattr(self._glint_det, attr)
        return d

    def process(self, gray: np.ndarray) -> EyeResult:
        """Run full pipeline on a grayscale eye frame."""
        pr = self._pupil_det.detect(gray)
        gr = self._glint_det.detect(
            gray,
            pupil_center=pr.center,
            pupil_radius=pr.radius,
        )

        pccr = None
        if pr.center and gr.primary:
            dx = pr.center[0] - gr.primary[0]
            dy = pr.center[1] - gr.primary[1]
            pccr = (float(dx), float(dy))

        # Merge intermediate frames
        intermediate = {**pr.intermediate_frames, **gr.intermediate_frames}

        return EyeResult(
            pupil=pr,
            glint=gr,
            pupil_center=pr.center,
            pupil_radius=pr.radius,
            glint_pos=gr.primary,
            pccr_vector=pccr,
            intermediate_frames=intermediate,
        )

    @staticmethod
    def draw(bgr: np.ndarray, result: EyeResult) -> np.ndarray:
        """Draw detection overlay on a BGR frame. Returns the annotated frame."""
        out = bgr.copy()
        pr = result.pupil
        gr = result.glint

        # Pupil ellipse + crosshair
        if pr.center:
            cx, cy = pr.center
            r = pr.radius or 20
            if pr.ellipse:
                cv2.ellipse(out, pr.ellipse, (0, 200, 255), 2)
            else:
                cv2.circle(out, (cx, cy), r, (0, 200, 255), 2)
            cv2.circle(out, (cx, cy), 3, (0, 255, 0), -1)
            cv2.line(out, (cx - r, cy), (cx + r, cy), (0, 255, 0), 1)
            cv2.line(out, (cx, cy - r), (cx, cy + r), (0, 255, 0), 1)

        # All glints
        for i, (gx, gy) in enumerate(gr.glints):
            color = (0, 255, 255) if i == 0 else (200, 200, 0)
            cv2.circle(out, (gx, gy), 6, color, 2)
            cv2.circle(out, (gx, gy), 2, color, -1)

        # PCCR vector arrow
        if result.pccr_vector and result.glint_pos and result.pupil_center:
            cv2.arrowedLine(out, result.glint_pos, result.pupil_center,
                            (255, 0, 255), 2, tipLength=0.15)

        # HUD text
        h = bgr.shape[0]
        if pr.center:
            cv2.putText(out, f"Pupil ({pr.center[0]},{pr.center[1]}) r={pr.radius}",
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        if gr.primary:
            cv2.putText(out, f"Glint ({gr.primary[0]},{gr.primary[1]})",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        if result.pccr_vector:
            dx, dy = result.pccr_vector
            cv2.putText(out, f"PCCR vec ({dx:.0f},{dy:.0f})",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1)

        return out
