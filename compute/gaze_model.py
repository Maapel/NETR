"""
Gaze mapping model — Phase 5.

2nd-order polynomial in (dx, dy) with LED-side identity interactions:

  6-term:   1, dx, dy, dx·dy, dx², dy²
  10-term:  same + side, side·dx, side·dy, side·dx·dy

  side ∈ {-1, +1} — which LED produced the glint (sign of glint_x - pupil_x).

The interaction terms let the model learn a different dx/dy sensitivity per LED,
which corrects for each LED being at a different physical position on the rig.
Works correctly even when only one LED is visible at a time — side is always
known for whichever glint is detected.

Term selection:
  ≥ 10 samples, side varies across samples  →  10-term (full side-interaction model)
  ≥  6 samples  (side constant or too few)  →   6-term (no side)

Usage:
    model = GazeModel()
    model.fit(samples)              # list of {"dx","dy","X","Y"} + optional "side"
    model.predict(dx, dy, side)     # side ∈ {-1,+1} or any float → snapped to ±1
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
    def _design(dx, dy, side, n_terms: int) -> np.ndarray:
        dx = np.asarray(dx, dtype=float)
        dy = np.asarray(dy, dtype=float)
        sd = np.asarray(side, dtype=float)
        ones = np.ones_like(dx)
        base = [ones, dx, dy, dx * dy, dx**2, dy**2]
        if n_terms == 10:
            base += [sd, sd * dx, sd * dy, sd * dx * dy]
        return np.column_stack(base)

    def fit(self, samples: list[dict]) -> dict:
        if len(samples) < 6:
            raise ValueError(f"Need at least 6 samples, got {len(samples)}")

        dx   = np.array([s["dx"] for s in samples], dtype=float)
        dy   = np.array([s["dy"] for s in samples], dtype=float)
        Ux   = np.array([s["X"]  for s in samples], dtype=float)
        Uy   = np.array([s["Y"]  for s in samples], dtype=float)
        raw  = [float(s.get("side", 1.0)) for s in samples]
        side = np.array([1.0 if v >= 0.0 else -1.0 for v in raw], dtype=float)

        # Use 10-term when enough samples AND both LEDs represented
        side_varies = bool(np.any(side > 0) and np.any(side < 0))
        if len(samples) >= 10 and side_varies:
            n_terms = 10
        else:
            n_terms = 6

        M = self._design(dx, dy, side, n_terms)
        self.A, _, _, _ = np.linalg.lstsq(M, Ux, rcond=None)
        self.B, _, _, _ = np.linalg.lstsq(M, Uy, rcond=None)
        self._n_terms = n_terms
        self.trained = True

        Ux_pred = M @ self.A
        Uy_pred = M @ self.B
        r2_x = 1 - np.var(Ux - Ux_pred) / (np.var(Ux) + 1e-9)
        r2_y = 1 - np.var(Uy - Uy_pred) / (np.var(Uy) + 1e-9)
        return {
            "r2_x": float(r2_x), "r2_y": float(r2_y),
            "n_samples": len(samples), "n_terms": n_terms,
            "side_varies": side_varies,
        }

    def predict(self, dx: float, dy: float, side: float = 1.0) -> tuple[float, float]:
        if not self.trained or self.A is None:
            raise RuntimeError("Model not trained")
        n  = self.n_terms
        sd = (1.0 if float(side) >= 0.0 else -1.0) if n == 10 else 0.0
        row = self._design([dx], [dy], [sd], n)[0]
        return float(row @ self.A), float(row @ self.B)

    def save(self, path: str | pathlib.Path):
        if not self.trained:
            raise RuntimeError("Nothing to save — model not trained")
        data: dict = {
            "A": self.A.tolist(),
            "B": self.B.tolist(),
            "n_terms": int(self._n_terms),
            "aug_feature": "side",
        }
        if self.scene_width is not None:
            data["scene_width"] = self.scene_width
        if self.scene_height is not None:
            data["scene_height"] = self.scene_height
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str | pathlib.Path) -> bool:
        try:
            with open(path) as f:
                data = json.load(f)
            self.A = np.array(data["A"])
            self.B = np.array(data["B"])
            self._n_terms = int(data.get("n_terms", len(self.A)))
            self.trained = True
            sw = data.get("scene_width")
            sh = data.get("scene_height")
            self.scene_width = int(sw) if sw is not None else None
            self.scene_height = int(sh) if sh is not None else None
            return True
        except (FileNotFoundError, KeyError, json.JSONDecodeError):
            return False
