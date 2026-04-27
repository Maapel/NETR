"""
Eye analysis pipeline — combines pupil + glint detection for PCCR.

PCCR uses the virtual corneal reference: midpoint of a validated dual-glint pair
when both LEDs are visible, otherwise the closest single glint.
Optional pccr_sep (inter-glint distance) feeds the extended gaze polynomial.
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
    pccr_vector: tuple[float, float] | None = None
    pccr_sep: float = 0.0
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
                      "search_radius_factor", "circularity_min",
                      "pair_min_sep_factor", "pair_max_sep_factor", "pair_search_top"):
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
        pccr_sep = 0.0
        glint_pos = None
        ref = gr.reference_point
        if pr.center and ref is not None:
            dx = pr.center[0] - ref[0]
            dy = pr.center[1] - ref[1]
            pccr = (float(dx), float(dy))
            if gr.pair_valid and gr.inter_glint_sep is not None:
                pccr_sep = float(gr.inter_glint_sep)
            glint_pos = (int(round(ref[0])), int(round(ref[1])))

        intermediate = {**pr.intermediate_frames, **gr.intermediate_frames}

        return EyeResult(
            pupil=pr,
            glint=gr,
            pupil_center=pr.center,
            pupil_radius=pr.radius,
            glint_pos=glint_pos,
            pccr_vector=pccr,
            pccr_sep=pccr_sep,
            intermediate_frames=intermediate,
        )

    @staticmethod
    def draw(bgr: np.ndarray, result: EyeResult) -> np.ndarray:
        """Draw detection overlay on a BGR frame. Returns the annotated frame."""
        out = bgr.copy()
        pr = result.pupil
        gr = result.glint

        if pr.center:
            cx, cy = int(pr.center[0]), int(pr.center[1])
            r = int(pr.radius or 20)
            if pr.ellipse:
                cv2.ellipse(out, pr.ellipse, (0, 200, 255), 2)
            else:
                cv2.circle(out, (cx, cy), r, (0, 200, 255), 2)
            cv2.circle(out, (cx, cy), 3, (0, 255, 0), -1)
            cv2.line(out, (cx - r, cy), (cx + r, cy), (0, 255, 0), 1)
            cv2.line(out, (cx, cy - r), (cx, cy + r), (0, 255, 0), 1)

        if gr.pair_valid and gr.glint_a and gr.glint_b:
            ax, ay = int(round(gr.glint_a[0])), int(round(gr.glint_a[1]))
            bx, by = int(round(gr.glint_b[0])), int(round(gr.glint_b[1]))
            cv2.line(out, (ax, ay), (bx, by), (180, 180, 255), 1)

        for i, (gx, gy) in enumerate(gr.glints):
            color = (0, 255, 255) if i == 0 else (200, 200, 0)
            igx, igy = int(round(gx)), int(round(gy))
            cv2.circle(out, (igx, igy), 6, color, 2)
            cv2.circle(out, (igx, igy), 2, color, -1)

        if result.pccr_vector and result.glint_pos and result.pupil_center:
            gp = (int(result.glint_pos[0]), int(result.glint_pos[1]))
            pp = (int(result.pupil_center[0]), int(result.pupil_center[1]))
            cv2.arrowedLine(out, gp, pp, (255, 0, 255), 2, tipLength=0.15)

        if pr.center:
            cv2.putText(out, f"Pupil ({pr.center[0]:.1f},{pr.center[1]:.1f}) r={pr.radius:.1f}",
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        if gr.primary:
            cv2.putText(out, f"Glint closest ({gr.primary[0]:.1f},{gr.primary[1]:.1f})",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        if gr.pair_valid:
            cv2.putText(out, f"Pair sep={result.pccr_sep:.1f}px",
                        (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 220, 255), 1)
        if result.pccr_vector:
            dx, dy = result.pccr_vector
            cv2.putText(out, f"PCCR ({dx:.0f},{dy:.0f})",
                        (10, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1)

        return out
