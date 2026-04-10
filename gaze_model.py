"""
Gaze mapping model — Phase 4.

2nd-order polynomial regression mapping (dx, dy) PCCR vectors
to scene camera coordinates (X, Y).

Model:
  X = a0 + a1*dx + a2*dy + a3*dx*dy + a4*dx² + a5*dy²
  Y = b0 + b1*dx + b2*dy + b3*dx*dy + b4*dx² + b5*dy²

Usage:
    model = GazeModel()
    model.fit(samples)        # list of {"dx","dy","X","Y"}
    model.save("gaze_model.json")
    model.load("gaze_model.json")
    X, Y = model.predict(dx, dy)
"""

import json
import numpy as np
import pathlib


class GazeModel:
    def __init__(self):
        self.A = None  # shape (6,) — X coefficients
        self.B = None  # shape (6,) — Y coefficients
        self.trained = False
        self.scene_width: int | None = None
        self.scene_height: int | None = None

    # ── Design matrix ────────────────────────────────────────────────────────
    @staticmethod
    def _design(dx, dy):
        """Build (N,6) design matrix from dx/dy arrays."""
        dx = np.asarray(dx, dtype=float)
        dy = np.asarray(dy, dtype=float)
        ones = np.ones_like(dx)
        return np.column_stack([ones, dx, dy, dx * dy, dx**2, dy**2])

    # ── Training ─────────────────────────────────────────────────────────────
    def fit(self, samples: list[dict]) -> dict:
        """
        Fit model from a list of dicts with keys: dx, dy, X, Y.
        Returns fit diagnostics (r2_x, r2_y, n_samples).
        """
        if len(samples) < 6:
            raise ValueError(f"Need at least 6 samples, got {len(samples)}")

        dx = np.array([s["dx"] for s in samples])
        dy = np.array([s["dy"] for s in samples])
        Ux = np.array([s["X"] for s in samples])
        Uy = np.array([s["Y"] for s in samples])

        M = self._design(dx, dy)
        # OLS: A = (MᵀM)⁻¹ Mᵀ Ux
        MtM = M.T @ M
        self.A = np.linalg.solve(MtM, M.T @ Ux)
        self.B = np.linalg.solve(MtM, M.T @ Uy)
        self.trained = True

        # R² diagnostics
        Ux_pred = M @ self.A
        Uy_pred = M @ self.B
        r2_x = 1 - np.var(Ux - Ux_pred) / (np.var(Ux) + 1e-9)
        r2_y = 1 - np.var(Uy - Uy_pred) / (np.var(Uy) + 1e-9)
        return {"r2_x": float(r2_x), "r2_y": float(r2_y), "n_samples": len(samples)}

    # ── Inference ────────────────────────────────────────────────────────────
    def predict(self, dx: float, dy: float) -> tuple[float, float]:
        """Map a single (dx, dy) PCCR vector to scene (X, Y)."""
        if not self.trained:
            raise RuntimeError("Model not trained")
        row = self._design([dx], [dy])[0]
        return float(row @ self.A), float(row @ self.B)

    # ── Persistence ──────────────────────────────────────────────────────────
    def save(
        self,
        path: str | pathlib.Path,
        *,
        scene_width: int | None = None,
        scene_height: int | None = None,
    ):
        if not self.trained:
            raise RuntimeError("Nothing to save — model not trained")
        data: dict = {"A": self.A.tolist(), "B": self.B.tolist()}
        if scene_width is not None:
            data["scene_width"] = int(scene_width)
        if scene_height is not None:
            data["scene_height"] = int(scene_height)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str | pathlib.Path) -> bool:
        try:
            with open(path) as f:
                data = json.load(f)
            self.A = np.array(data["A"])
            self.B = np.array(data["B"])
            self.trained = True
            sw = data.get("scene_width")
            sh = data.get("scene_height")
            self.scene_width = int(sw) if sw is not None else None
            self.scene_height = int(sh) if sh is not None else None
            return True
        except (FileNotFoundError, KeyError, json.JSONDecodeError):
            return False
