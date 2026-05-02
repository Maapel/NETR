"""
Gaze mapping models.

GazeModel — single 6-term polynomial:
  1, dx, dy, dx·dy, dx², dy²

DualGazeModel — two independent GazeModel instances, one per LED side:
  model_pos  → side = +1 (right LED)
  model_neg  → side = -1 (left LED)
  fallback   → all samples combined (used when one side has <MIN_SAMPLES)

At prediction: use the side's dedicated model when trained, else fallback.
When both glints are visible the caller should pick a preferred side
consistently (convention: +1) rather than let geometry decide.

TwoGlintGazeModel — uses PCCR vectors from BOTH glints simultaneously:
  Features: [1, dx1, dy1, dx2, dy2, dx1*dy1, dx2*dy2, dx1*dx2, dy1*dy2, dx1+dx2, dy1+dy2]
  11-term polynomial. Trained only on frames where both LED glints were detected.
  Falls back to DualGazeModel when only one glint is available at prediction time.
  Requires samples to have dx1/dy1 (right LED) and dx2/dy2 (left LED) fields.

File format: {"type":"dual", "pos":{A,B}, "neg":{A,B}, "fallback":{A,B}, ...}
             {"type":"two_glint", "two":{A,B}, "fallback_pos":{A,B}, "fallback_neg":{A,B}, ...}
Legacy single-model files (no "type" key) are loaded as fallback only.
"""

import json
import numpy as np
import pathlib


class GazeModel:
    def __init__(self):
        self.A = None
        self.B = None
        self.trained = False
        self.scene_width: int | None = None
        self.scene_height: int | None = None
        self._n_terms = 6

    @property
    def n_terms(self) -> int:
        return self._n_terms if self.trained and self.A is not None else 6

    @staticmethod
    def _design(dx, dy, n_terms: int = 6) -> np.ndarray:
        dx   = np.asarray(dx, dtype=float)
        dy   = np.asarray(dy, dtype=float)
        ones = np.ones_like(dx)
        return np.column_stack([ones, dx, dy, dx * dy, dx**2, dy**2])

    def fit(self, samples: list[dict]) -> dict:
        if len(samples) < 6:
            raise ValueError(f"Need at least 6 samples, got {len(samples)}")
        dx = np.array([s["dx"] for s in samples], dtype=float)
        dy = np.array([s["dy"] for s in samples], dtype=float)
        Ux = np.array([s["X"]  for s in samples], dtype=float)
        Uy = np.array([s["Y"]  for s in samples], dtype=float)
        M  = self._design(dx, dy)
        self.A, _, _, _ = np.linalg.lstsq(M, Ux, rcond=None)
        self.B, _, _, _ = np.linalg.lstsq(M, Uy, rcond=None)
        self._n_terms = 6
        self.trained  = True
        Ux_pred = M @ self.A
        Uy_pred = M @ self.B
        r2_x = 1 - np.var(Ux - Ux_pred) / (np.var(Ux) + 1e-9)
        r2_y = 1 - np.var(Uy - Uy_pred) / (np.var(Uy) + 1e-9)
        return {"r2_x": float(r2_x), "r2_y": float(r2_y), "n_samples": len(samples)}

    def predict(self, dx: float, dy: float, side: float = 1.0) -> tuple[float, float]:
        if not self.trained or self.A is None:
            raise RuntimeError("Model not trained")
        row = self._design([dx], [dy])[0]
        return float(row @ self.A), float(row @ self.B)

    def save(self, path: str | pathlib.Path):
        if not self.trained:
            raise RuntimeError("Nothing to save — model not trained")
        data: dict = {"A": self.A.tolist(), "B": self.B.tolist(), "n_terms": 6}
        if self.scene_width  is not None: data["scene_width"]  = self.scene_width
        if self.scene_height is not None: data["scene_height"] = self.scene_height
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str | pathlib.Path) -> bool:
        try:
            with open(path) as f:
                data = json.load(f)
            if data.get("type") == "dual":
                return False   # not our format
            self.A = np.array(data["A"])
            self.B = np.array(data["B"])
            self._n_terms = 6
            self.trained  = True
            sw = data.get("scene_width");  self.scene_width  = int(sw) if sw is not None else None
            sh = data.get("scene_height"); self.scene_height = int(sh) if sh is not None else None
            return True
        except (FileNotFoundError, KeyError, json.JSONDecodeError):
            return False


class DualGazeModel:
    """Two independent 6-term models (one per LED side) + a shared fallback.

    Fitting: samples are split by sign(side). Each sub-model is trained if
    its side has ≥ MIN_SAMPLES; otherwise only the fallback is used for that side.

    Prediction: use the side-specific model when trained, else fallback.
    The caller is responsible for passing a consistent side convention
    (+1 = right LED preferred when both glints are visible).
    """

    MIN_SAMPLES = 6

    def __init__(self):
        self.model_pos = GazeModel()   # side = +1
        self.model_neg = GazeModel()   # side = -1
        self.fallback  = GazeModel()   # all samples, no side split
        self.trained        = False
        self.scene_width:  int | None = None
        self.scene_height: int | None = None

    def _propagate_scene_size(self):
        for m in (self.model_pos, self.model_neg, self.fallback):
            m.scene_width  = self.scene_width
            m.scene_height = self.scene_height

    def fit(self, samples: list[dict]) -> dict:
        if len(samples) < self.MIN_SAMPLES:
            raise ValueError(f"Need at least {self.MIN_SAMPLES} samples, got {len(samples)}")

        pos = [s for s in samples if float(s.get("side", 1.0)) >= 0]
        neg = [s for s in samples if float(s.get("side", 1.0)) <  0]

        diag_fb  = self.fallback.fit(samples)
        diag_pos = self.model_pos.fit(pos) if len(pos) >= self.MIN_SAMPLES else None
        diag_neg = self.model_neg.fit(neg) if len(neg) >= self.MIN_SAMPLES else None

        self.trained = True
        if self.scene_width is not None:
            self._propagate_scene_size()

        return {
            "r2_x":     diag_fb["r2_x"],
            "r2_y":     diag_fb["r2_y"],
            "n_samples": len(samples),
            "n_pos":    len(pos),
            "n_neg":    len(neg),
            "pos_ok":   diag_pos is not None,
            "neg_ok":   diag_neg is not None,
        }

    def predict(self, dx: float, dy: float, side: float = 1.0) -> tuple[float, float]:
        if not self.trained:
            raise RuntimeError("Model not trained")
        sd = 1.0 if float(side) >= 0.0 else -1.0
        if sd >= 0 and self.model_pos.trained:
            return self.model_pos.predict(dx, dy)
        if sd < 0 and self.model_neg.trained:
            return self.model_neg.predict(dx, dy)
        return self.fallback.predict(dx, dy)

    def save(self, path: str | pathlib.Path):
        if not self.trained:
            raise RuntimeError("Nothing to save — model not trained")
        data: dict = {"type": "dual"}
        if self.model_pos.trained:
            data["pos"] = {"A": self.model_pos.A.tolist(), "B": self.model_pos.B.tolist()}
        if self.model_neg.trained:
            data["neg"] = {"A": self.model_neg.A.tolist(), "B": self.model_neg.B.tolist()}
        data["fallback"] = {"A": self.fallback.A.tolist(), "B": self.fallback.B.tolist()}
        if self.scene_width  is not None: data["scene_width"]  = self.scene_width
        if self.scene_height is not None: data["scene_height"] = self.scene_height
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str | pathlib.Path) -> bool:
        try:
            with open(path) as f:
                data = json.load(f)
            if data.get("type") != "dual":
                # Legacy single-model file — load into fallback only
                ok = self.fallback.load(path)
                if ok:
                    self.trained = True
                return ok
            if "pos" in data:
                self.model_pos.A = np.array(data["pos"]["A"])
                self.model_pos.B = np.array(data["pos"]["B"])
                self.model_pos._n_terms = 6
                self.model_pos.trained  = True
            if "neg" in data:
                self.model_neg.A = np.array(data["neg"]["A"])
                self.model_neg.B = np.array(data["neg"]["B"])
                self.model_neg._n_terms = 6
                self.model_neg.trained  = True
            fb = data.get("fallback")
            if fb:
                self.fallback.A = np.array(fb["A"])
                self.fallback.B = np.array(fb["B"])
                self.fallback._n_terms = 6
                self.fallback.trained  = True
            sw = data.get("scene_width");  self.scene_width  = int(sw) if sw is not None else None
            sh = data.get("scene_height"); self.scene_height = int(sh) if sh is not None else None
            self.trained = self.fallback.trained
            return self.trained
        except (FileNotFoundError, KeyError, json.JSONDecodeError):
            return False


class TwoGlintGazeModel:
    """11-term polynomial using both LED glint PCCRs simultaneously.

    Sample dict must have: dx1, dy1 (right LED PCCR), dx2, dy2 (left LED PCCR), X, Y.
    At prediction: if only one glint visible, falls back to dedicated single-glint models.
    """

    MIN_SAMPLES = 8

    @staticmethod
    def _design(dx1, dy1, dx2, dy2) -> np.ndarray:
        dx1 = np.asarray(dx1, dtype=float); dy1 = np.asarray(dy1, dtype=float)
        dx2 = np.asarray(dx2, dtype=float); dy2 = np.asarray(dy2, dtype=float)
        ones = np.ones_like(dx1)
        return np.column_stack([
            ones, dx1, dy1, dx2, dy2,
            dx1 * dy1, dx2 * dy2,
            dx1 * dx2, dy1 * dy2,
            dx1 + dx2, dy1 + dy2,
        ])

    def __init__(self):
        self.A = None; self.B = None
        self.fallback_pos = GazeModel()  # right LED only
        self.fallback_neg = GazeModel()  # left LED only
        self.trained = False
        self.scene_width:  int | None = None
        self.scene_height: int | None = None

    def fit(self, samples: list[dict]) -> dict:
        two_glint = [s for s in samples if "dx1" in s and "dx2" in s]
        if len(two_glint) < self.MIN_SAMPLES:
            raise ValueError(f"Need ≥{self.MIN_SAMPLES} two-glint samples, got {len(two_glint)}")

        dx1 = np.array([s["dx1"] for s in two_glint])
        dy1 = np.array([s["dy1"] for s in two_glint])
        dx2 = np.array([s["dx2"] for s in two_glint])
        dy2 = np.array([s["dy2"] for s in two_glint])
        Ux  = np.array([s["X"]   for s in two_glint])
        Uy  = np.array([s["Y"]   for s in two_glint])
        M   = self._design(dx1, dy1, dx2, dy2)
        self.A, _, _, _ = np.linalg.lstsq(M, Ux, rcond=None)
        self.B, _, _, _ = np.linalg.lstsq(M, Uy, rcond=None)
        self.trained = True

        Ux_pred = M @ self.A; Uy_pred = M @ self.B
        r2_x = 1 - np.var(Ux - Ux_pred) / (np.var(Ux) + 1e-9)
        r2_y = 1 - np.var(Uy - Uy_pred) / (np.var(Uy) + 1e-9)

        # Fit per-side fallbacks from single-glint-equiv samples
        pos = [{"dx": s["dx1"], "dy": s["dy1"], "X": s["X"], "Y": s["Y"]} for s in two_glint]
        neg = [{"dx": s["dx2"], "dy": s["dy2"], "X": s["X"], "Y": s["Y"]} for s in two_glint]
        if len(pos) >= 6: self.fallback_pos.fit(pos)
        if len(neg) >= 6: self.fallback_neg.fit(neg)

        if self.scene_width is not None:
            for m in (self.fallback_pos, self.fallback_neg):
                m.scene_width = self.scene_width; m.scene_height = self.scene_height

        return {"r2_x": float(r2_x), "r2_y": float(r2_y),
                "n_two_glint": len(two_glint), "n_total": len(samples)}

    def predict(self, dx: float, dy: float, side: float = 1.0,
                dx2: float | None = None, dy2: float | None = None) -> tuple[float, float]:
        """Predict gaze. Use dual-glint model when dx2/dy2 provided, else fallback."""
        if not self.trained:
            raise RuntimeError("Model not trained")
        if dx2 is not None and dy2 is not None and self.A is not None:
            # Both glints: right LED = dx1/dy1 (side +1), left = dx2/dy2 (side -1)
            if float(side) >= 0:
                d1, d1y, d2, d2y = dx, dy, dx2, dy2
            else:
                d1, d1y, d2, d2y = dx2, dy2, dx, dy
            row = self._design([d1], [d1y], [d2], [d2y])[0]
            return float(row @ self.A), float(row @ self.B)
        # Fallback: single glint
        if float(side) >= 0 and self.fallback_pos.trained:
            return self.fallback_pos.predict(dx, dy)
        if float(side) < 0 and self.fallback_neg.trained:
            return self.fallback_neg.predict(dx, dy)
        # Last resort: mirror PCCR to approximate missing second glint
        if self.A is not None:
            row = self._design([dx], [dy], [-dx], [dy])[0]
            return float(row @ self.A), float(row @ self.B)
        raise RuntimeError("No trained sub-model available")

    def save(self, path: str | pathlib.Path):
        if not self.trained:
            raise RuntimeError("Nothing to save — model not trained")
        data: dict = {
            "type": "two_glint",
            "two": {"A": self.A.tolist(), "B": self.B.tolist()},
        }
        if self.fallback_pos.trained:
            data["fallback_pos"] = {"A": self.fallback_pos.A.tolist(), "B": self.fallback_pos.B.tolist()}
        if self.fallback_neg.trained:
            data["fallback_neg"] = {"A": self.fallback_neg.A.tolist(), "B": self.fallback_neg.B.tolist()}
        if self.scene_width  is not None: data["scene_width"]  = self.scene_width
        if self.scene_height is not None: data["scene_height"] = self.scene_height
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str | pathlib.Path) -> bool:
        try:
            with open(path) as f:
                data = json.load(f)
            if data.get("type") != "two_glint":
                return False
            two = data["two"]
            self.A = np.array(two["A"]); self.B = np.array(two["B"])
            fp = data.get("fallback_pos")
            if fp:
                self.fallback_pos.A = np.array(fp["A"])
                self.fallback_pos.B = np.array(fp["B"])
                self.fallback_pos._n_terms = 6
                self.fallback_pos.trained  = True
            fn = data.get("fallback_neg")
            if fn:
                self.fallback_neg.A = np.array(fn["A"])
                self.fallback_neg.B = np.array(fn["B"])
                self.fallback_neg._n_terms = 6
                self.fallback_neg.trained  = True
            sw = data.get("scene_width");  self.scene_width  = int(sw) if sw is not None else None
            sh = data.get("scene_height"); self.scene_height = int(sh) if sh is not None else None
            self.trained = True
            return True
        except (FileNotFoundError, KeyError, json.JSONDecodeError):
            return False
