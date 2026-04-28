"""
Eye analysis pipeline — combines pupil + glint detection for PCCR.

Glints outside the pupil/iris region are rejected. One primary glint drives PCCR.
Augmented gaze uses side = sign(glint_x - pupil_x) in {-1, +1}.
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
    pccr_side: float = 0.0
    # labeled_glints: list of (gx, gy, side) for all detected glints this frame
    # side ∈ {-1, +1} — derived from switch_dx calibration or sign(gx - pupil_cx)
    labeled_glints: list[tuple[float, float, float]] = field(default_factory=list)
    intermediate_frames: dict = field(default_factory=dict)


class EyePipeline:
    """Full eye analysis: pupil detection → glint detection → PCCR vector."""

    def __init__(self, pupil_kwargs=None, glint_kwargs=None, switch_dx: float = 0.0,
                 preferred_side: float = 1.0):
        self._pupil_det = PupilDetector(**(pupil_kwargs or {}))
        self._glint_det = GlintDetector(**(glint_kwargs or {}))
        # dx threshold for glint labeling: side=+1 if dx>switch_dx else -1
        # dx = pupil_cx - glint_x (PCCR convention); 0.0 = use sign of (glint_x - pupil_x)
        self.switch_dx: float = float(switch_dx)
        # When both LED glints are visible, always use this side as primary.
        # Convention: +1 (right LED). Makes prediction deterministic near crossover.
        self.preferred_side: float = 1.0 if float(preferred_side) >= 0 else -1.0

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
        for attr in ("glint_margin", "brightness_thresh", "min_area", "max_area",
                      "circularity_min", "iris_radius_factor",
                      "limbus_n_rays", "limbus_max_factor", "limbus_min_gradient",
                      "search_radius_factor", "ellipse_slack"):
            d["g_" + attr] = getattr(self._glint_det, attr)
        return d

    def _label_side(self, dx: float) -> float:
        """Label a glint as LED_left(-1) or LED_right(+1) using switch_dx.

        dx = pupil_cx - glint_x  (positive when glint is left of pupil).
        switch_dx is the crossing point found during glint sweep calibration.
        Above switch_dx → glint is to the left → LED_right is visible → side=+1.
        Below switch_dx → glint is to the right → LED_left is visible → side=-1.
        With switch_dx=0 this reduces to sign(glint_x - pupil_x).
        """
        return 1.0 if dx > self.switch_dx else -1.0

    def process(self, gray: np.ndarray) -> EyeResult:
        """Run full pipeline on a grayscale eye frame."""
        pr = self._pupil_det.detect(gray)
        gr = self._glint_det.detect(
            gray,
            pupil_center=pr.center,
            pupil_radius=pr.radius,
            pupil_ellipse=pr.ellipse,
        )

        pccr = None
        pccr_side = 0.0
        glint_pos = None
        labeled: list[tuple[float, float, float]] = []

        if pr.center and gr.glints:
            pcx, pcy = float(pr.center[0]), float(pr.center[1])
            for gx, gy in gr.glints:
                dx = pcx - gx
                side = self._label_side(dx)
                labeled.append((gx, gy, side))

            # Primary glint selection:
            # When both LED sides are visible, always pick the preferred_side glint
            # (not just the closest) so the gaze model gets a consistent side.
            sides_seen = {s for _, _, s in labeled}
            if len(sides_seen) == 2:
                preferred = [(gx, gy) for gx, gy, s in labeled if s == self.preferred_side]
                gx, gy = preferred[0] if preferred else (gr.primary[0], gr.primary[1])
            else:
                gx, gy = gr.primary

            dx = pcx - gx
            dy = pcy - gy
            pccr = (float(dx), float(dy))
            pccr_side = self._label_side(dx)
            glint_pos = (int(round(gx)), int(round(gy)))

        intermediate = {**pr.intermediate_frames, **gr.intermediate_frames}

        return EyeResult(
            pupil=pr,
            glint=gr,
            pupil_center=pr.center,
            pupil_radius=pr.radius,
            glint_pos=glint_pos,
            pccr_vector=pccr,
            pccr_side=pccr_side,
            labeled_glints=labeled,
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
            # Limbus boundary (green dashed-look: thin circle)
            if gr.limbus_radius is not None:
                cv2.circle(out, (cx, cy), int(gr.limbus_radius), (0, 180, 0), 1)

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
            cv2.putText(out, f"Glint ({gr.primary[0]:.1f},{gr.primary[1]:.1f})",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        if result.pccr_vector:
            dx, dy = result.pccr_vector
            sd = result.pccr_side
            cv2.putText(out, f"PCCR ({dx:.0f},{dy:.0f}) side={sd:+.0f}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1)

        return out
