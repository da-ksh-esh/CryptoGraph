"""Diebold-Yilmaz spillover networks from the generalized FEVD (directed: theta[i, j] = share of i's
forecast-error variance due to shocks in j), on fixed or rolling trailing windows.

The GFEVD math lives in `cryptograph.analysis.connectedness`; this module adds a fast OLS VAR for
the many rolling windows and the coin selection per window.
"""

from __future__ import annotations

import numpy as np

from ..analysis.connectedness import generalized_fevd


def var_ols(X: np.ndarray, p: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """VAR(p) with intercept by OLS on a complete (T, N) array. Returns coefs (p, N, N) with
    coefs[l][i, j] = effect of j at lag l+1 on i, and the residual covariance (N, N)."""
    T, N = X.shape
    Y = X[p:]
    Z = np.hstack([np.ones((T - p, 1))] + [X[p - lag - 1: T - lag - 1] for lag in range(p)])
    B, *_ = np.linalg.lstsq(Z, Y, rcond=None)                 # (1 + N p, N)
    resid = Y - Z @ B
    dof = max(T - p - N * p - 1, 1)
    sigma = resid.T @ resid / dof
    coefs = np.stack([B[1 + lag * N: 1 + (lag + 1) * N].T for lag in range(p)])
    return coefs, sigma


def window_gfevd(R: np.ndarray, H: int, p: int = 1, min_rows: int = 100) -> np.ndarray | None:
    """GFEVD (x100, rows sum to 100) of the rows of R (T, n) without missing values; None if too few rows."""
    X = R[np.isfinite(R).all(axis=1)]
    if len(X) < max(min_rows, 3 * (R.shape[1] * p + 1)):
        return None
    coefs, sigma = var_ols(X, p)
    return generalized_fevd(coefs, sigma, H)


def spillover_summary(theta: np.ndarray) -> dict:
    """TO / FROM / NET per node and the total spillover index from a GFEVD matrix (x100)."""
    off = theta.copy()
    np.fill_diagonal(off, 0.0)
    to_, from_ = off.sum(axis=0), off.sum(axis=1)
    return {"to": to_, "from": from_, "net": to_ - from_, "total": float(off.sum() / theta.shape[0])}


def window_coins(r: np.ndarray, active: np.ndarray, t: int, window: int, min_coverage: float) -> np.ndarray:
    """Nodes active at t with at least `min_coverage` of bars observed in (t-window, t]."""
    w = r[t - window + 1: t + 1]
    return np.flatnonzero(active[t] & (np.isfinite(w).mean(axis=0) >= min_coverage))


def rolling_spillover(r: np.ndarray, active: np.ndarray, snap_t: np.ndarray, window: int, H: int, p: int = 1,
                      min_coverage: float = 0.9) -> list[tuple[np.ndarray, np.ndarray | None]]:
    """For each snapshot bar t: (coin indices, GFEVD over (t-window, t]) — trailing data only."""
    out = []
    for t in snap_t:
        if t < window - 1:
            out.append((np.array([], int), None))
            continue
        idx = window_coins(r, active, t, window, min_coverage)
        theta = window_gfevd(r[t - window + 1: t + 1][:, idx], H, p) if len(idx) >= 3 else None
        out.append((idx, theta))
    return out
