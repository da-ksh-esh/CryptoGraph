"""Naive forecasting baselines — the bar every DL model must beat."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def naive_regression(y: np.ndarray, mask: np.ndarray, last_value: np.ndarray, train_mean: np.ndarray,
                     horizons: list[int], target: str, split: str) -> list[dict]:
    """y, last_value: (S, N, H); train_mean: (N, H). Returns rows of MSE/MAE for zero / persistence / mean."""
    rows = []
    preds = {"zero": np.zeros_like(y), "persistence": last_value,
             "historical_mean": np.broadcast_to(train_mean[None], y.shape)}
    for name, p in preds.items():
        for k, h in enumerate(horizons):
            m = mask[..., k] & np.isfinite(p[..., k]) & np.isfinite(y[..., k])
            err = (p[..., k] - y[..., k])[m]
            rows.append({"split": split, "target": target, "horizon": h, "model": name, "n": int(m.sum()),
                         "MSE": float(np.mean(err ** 2)), "MAE": float(np.mean(np.abs(err)))})
    return rows


def shock_persistence_rule(shock_today: np.ndarray, y_shock: np.ndarray, mask: np.ndarray, split: str) -> dict:
    """'Shock tomorrow if shock today' (and its |z| score version) — AUROC / AUPRC."""
    s = shock_today[mask].astype(float)
    y = y_shock[mask].astype(int)
    return {"split": split, "target": "shock_next", "horizon": 1, "model": "shock_today_rule", "n": int(mask.sum()),
            "AUROC": float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else np.nan,
            "AUPRC": float(average_precision_score(y, s)) if y.sum() else np.nan, "base_rate": float(y.mean())}


def shock_score_rule(score: np.ndarray, y_shock: np.ndarray, mask: np.ndarray, split: str, name: str) -> dict:
    s, y = score[mask], y_shock[mask].astype(int)
    ok = np.isfinite(s)
    return {"split": split, "target": "shock_next", "horizon": 1, "model": name, "n": int(ok.sum()),
            "AUROC": float(roc_auc_score(y[ok], s[ok])), "AUPRC": float(average_precision_score(y[ok], s[ok])),
            "base_rate": float(y[ok].mean())}


def to_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def trailing_rv(r: np.ndarray, h: int) -> np.ndarray:
    """(T, N) returns -> sqrt(mean of squared returns over t-h+1..t); NaN if any bar is missing. This is the
    persistence forecast of the forward realized volatility over the next h bars."""
    sq = r.astype(np.float64) ** 2
    csum = np.vstack([np.zeros((1, r.shape[1])), np.cumsum(np.where(np.isnan(sq), 0.0, sq), axis=0)])
    cnan = np.vstack([np.zeros((1, r.shape[1])), np.cumsum(np.isnan(sq), axis=0)])
    out = np.full(r.shape, np.nan)
    t = np.arange(h - 1, r.shape[0])
    v = np.sqrt((csum[t + 1] - csum[t + 1 - h]) / h)
    v[(cnan[t + 1] - cnan[t + 1 - h]) > 0] = np.nan
    out[t] = v
    return out


def har_design(r: np.ndarray, windows: tuple[int, ...] = (1, 24, 168)) -> np.ndarray:
    """HAR-RV regressors (Corsi 2009) in bars: trailing realized volatility over each window -> (T, N, len(windows))."""
    return np.stack([trailing_rv(r, w) for w in windows], axis=-1)


def har_fit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """OLS of y on [1, X] over finite rows. X (S, P), y (S,) -> coefficients (P + 1,)."""
    ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
    Z = np.hstack([np.ones((int(ok.sum()), 1)), X[ok]])
    return np.linalg.lstsq(Z, y[ok], rcond=None)[0]


def har_predict(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return np.clip(beta[0] + X @ beta[1:], 0.0, None)
