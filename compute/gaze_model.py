"""
Gaze mapping model — Phase 4.

2nd-order polynomial in (dx, dy), optionally extended with inter-glint separation `sep`:

  6-term:  X = a0 + a1*dx + a2*dy + a3*dx*dy + a4*dx² + a5*dy²
  7-term:  same + a6*sep

Samples may include optional "sep" (default 0). If all sep ≈ 0, fit uses 6 terms
(rank-safe). If any |sep| is meaningful, fit uses 7 terms (need ≥7 samples).

Usage:
    model = GazeModel()
    model.fit(samples)   # {"dx","dy","X","Y"} or + "sep"
    model.predict(dx, dy, sep=0.0)
"""

import json
import numpy as np
import pathlib


class GazeModel:
    def __init__(self):
        self.A = None
        self.B = None
        self.trained = False
        self._n_terms = 6

    @property
    def n_terms(self) -> int:
        return self._n_terms if self.trained and self.A is not None else 6

    @staticmethod
    def _design(dx, dy, sep, n_terms: int):
        dx = np.asarray(dx, dtype=float)
        dy = np.asarray(dy, dtype=float)
        sp = np.asarray(sep, dtype=float)
        ones = np.ones_like(dx)
        base = [ones, dx, dy, dx * dy, dx**2, dy**2]
        if n_terms == 7:
            base.append(sp)
        return np.column_stack(base)

    def fit(self, samples: list[dict]) -> dict:
        if len(samples) < 6:
            raise ValueError(f"Need at least 6 samples, got {len(samples)}")

        dx = np.array([s["dx"] for s in samples], dtype=float)
        dy = np.array([s["dy"] for s in samples], dtype=float)
        sep = np.array([float(s.get("sep", 0.0)) for s in samples], dtype=float)
        Ux = np.array([s["X"] for s in samples])
        Uy = np.array([s["Y"] for s in samples])

        use_sep = bool(np.max(np.abs(sep)) >= 1e-6)
        n_terms = 7 if use_sep else 6
        if use_sep and len(samples) < 7:
            raise ValueError(f"Extended model needs at least 7 samples when sep is used, got {len(samples)}")

        M = self._design(dx, dy, sep, n_terms)
        MtM = M.T @ M
        self.A = np.linalg.solve(MtM, M.T @ Ux)
        self.B = np.linalg.solve(MtM, M.T @ Uy)
        self._n_terms = n_terms
        self.trained = True

        Ux_pred = M @ self.A
        Uy_pred = M @ self.B
        r2_x = 1 - np.var(Ux - Ux_pred) / (np.var(Ux) + 1e-9)
        r2_y = 1 - np.var(Uy - Uy_pred) / (np.var(Uy) + 1e-9)
        return {"r2_x": float(r2_x), "r2_y": float(r2_y), "n_samples": len(samples), "n_terms": n_terms}

    def predict(self, dx: float, dy: float, sep: float = 0.0) -> tuple[float, float]:
        if not self.trained or self.A is None:
            raise RuntimeError("Model not trained")
        n = len(self.A)
        sp = float(sep) if n == 7 else 0.0
        row = self._design([dx], [dy], [sp], n)[0]
        return float(row @ self.A), float(row @ self.B)

    def save(self, path: str | pathlib.Path):
        if not self.trained:
            raise RuntimeError("Nothing to save — model not trained")
        data = {"A": self.A.tolist(), "B": self.B.tolist(), "n_terms": int(len(self.A))}
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
            return True
        except (FileNotFoundError, KeyError, json.JSONDecodeError):
            return False
