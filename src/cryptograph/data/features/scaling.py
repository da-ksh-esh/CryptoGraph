"""Robust (median / IQR) scaling fitted on training rows only.

Per-coin scalers are used for coins with enough training rows; others fall
back to the global (pooled) training scaler. The clipped copy is for model
INPUTS only — evaluation targets are never clipped.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RobustScalerSet:
    feature_names: list[str]
    coins: list[str]
    median: np.ndarray = field(default=None)        # (N, F)
    iqr: np.ndarray = field(default=None)           # (N, F)
    global_median: np.ndarray = field(default=None) # (F,)
    global_iqr: np.ndarray = field(default=None)    # (F,)
    uses_global: np.ndarray = field(default=None)   # (N, F) bool — per-coin fallback to global
    n_train_rows: np.ndarray = field(default=None)  # (N, F)
    fit_dates: tuple[str, str] | None = None

    @staticmethod
    def _stats(x: np.ndarray) -> tuple[float, float]:
        x = x[np.isfinite(x)]
        if x.size == 0:
            return np.nan, np.nan
        q25, med, q75 = np.percentile(x, [25, 50, 75])
        iqr = q75 - q25
        return float(med), float(iqr if iqr > 1e-12 else (np.std(x) if np.std(x) > 1e-12 else 1.0))

    @classmethod
    def fit(cls, X: np.ndarray, train_rows: np.ndarray, feature_names: list[str], coins: list[str],
            min_rows: int = 180, fit_dates: tuple[str, str] | None = None) -> "RobustScalerSet":
        """X: (T, N, F) with NaN for missing; train_rows: (T,) bool selecting training dates ONLY."""
        Xt = X[train_rows]
        T, N, F = Xt.shape
        s = cls(feature_names=list(feature_names), coins=list(coins), fit_dates=fit_dates)
        s.global_median = np.zeros(F)
        s.global_iqr = np.ones(F)
        for f in range(F):
            s.global_median[f], s.global_iqr[f] = cls._stats(Xt[:, :, f].ravel())
        s.median = np.tile(s.global_median, (N, 1))
        s.iqr = np.tile(s.global_iqr, (N, 1))
        s.uses_global = np.ones((N, F), dtype=bool)
        s.n_train_rows = np.isfinite(Xt).sum(axis=0)
        for n in range(N):
            for f in range(F):
                if s.n_train_rows[n, f] >= min_rows:
                    s.median[n, f], s.iqr[n, f] = cls._stats(Xt[:, n, f])
                    s.uses_global[n, f] = False
        return s

    def transform(self, X: np.ndarray, scope: str = "per_coin", clip: float | None = None) -> np.ndarray:
        if scope == "per_coin":
            out = (X - self.median[None]) / self.iqr[None]
        elif scope == "global":
            out = (X - self.global_median[None, None]) / self.global_iqr[None, None]
        else:
            raise ValueError(scope)
        if clip is not None:
            out = np.clip(out, -clip, clip)
        return out.astype(np.float32)

    def state_dict(self) -> dict:
        return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in self.__dict__.items()}
