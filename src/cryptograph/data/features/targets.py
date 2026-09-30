"""Targets shared by every model (defined once, from config).

Conventions for a sample dated t (last input day):
  fwd_ret[h]   = log(C[t+h] / C[t])
  fwd_rv[h]    = sqrt(mean(r[t+1..t+h]^2))          (h=1 -> |r[t+1]|)
  shock_next   = |r[t+1]| / sigma30[t] > z*          (sigma30[t] uses returns up to t)
  contagion[k] = shock_i(t) AND any shock_j(t+1..t+k), where shock_i(t) = |r[t]| / sigma30[t-1] > z*
All targets are strictly after t; NaN / mask=False where any needed value is missing.
"""

from __future__ import annotations

import numpy as np


def forward_returns(log_close: np.ndarray, horizons: list[int]) -> np.ndarray:
    """(T, N) log close -> (T, N, H) forward log returns."""
    T = log_close.shape[0]
    out = np.full((*log_close.shape, len(horizons)), np.nan, dtype=np.float32)
    for k, h in enumerate(horizons):
        out[: T - h, :, k] = log_close[h:] - log_close[: T - h]
    return out


def forward_realized_vol(r: np.ndarray, horizons: list[int]) -> np.ndarray:
    """(T, N) daily log returns -> (T, N, H) sqrt(mean of squared returns over t+1..t+h)."""
    T = r.shape[0]
    sq = r.astype(np.float64) ** 2
    # cumulative sums that propagate NaN: any missing day in the window -> NaN
    csum = np.vstack([np.zeros((1, r.shape[1])), np.cumsum(np.where(np.isnan(sq), 0.0, sq), axis=0)])
    cnan = np.vstack([np.zeros((1, r.shape[1])), np.cumsum(np.isnan(sq), axis=0)])
    out = np.full((*r.shape, len(horizons)), np.nan, dtype=np.float32)
    for k, h in enumerate(horizons):
        t = np.arange(T - h)
        s = csum[t + h + 1] - csum[t + 1]
        n_missing = cnan[t + h + 1] - cnan[t + 1]
        v = np.sqrt(s / h)
        v[n_missing > 0] = np.nan
        out[: T - h, :, k] = v
    return out


def shock_z(r: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """z[t] = r[t] / sigma[t-1] (sigma estimated strictly before the return it scales)."""
    z = np.full_like(r, np.nan, dtype=np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        z[1:] = r[1:] / sigma[:-1]
    z[~np.isfinite(z)] = np.nan
    return z


def shock_indicator(z: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Binary shock at t (|z|>thr) and its validity mask."""
    valid = np.isfinite(z)
    return (np.abs(np.nan_to_num(z)) > threshold) & valid, valid


def next_day_shock(z: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Label at t = shock at t+1."""
    s, v = shock_indicator(z, threshold)
    lab = np.zeros_like(s)
    msk = np.zeros_like(v)
    lab[:-1], msk[:-1] = s[1:], v[1:]
    return lab, msk


def contagion_labels(shock: np.ndarray, valid: np.ndarray, ks: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """(T, N) shocks -> labels/mask of shape (T, N_src, N_dst, K).

    label[t,i,j,k] = shock[t,i] & any(shock[t+1..t+k, j]); diagonal is masked out.
    mask requires the source shock indicator at t and all target indicators in the window to be valid.
    """
    T, N = shock.shape
    K = len(ks)
    lab = np.zeros((T, N, N, K), dtype=bool)
    msk = np.zeros((T, N, N, K), dtype=bool)
    off_diag = ~np.eye(N, dtype=bool)
    for kk, k in enumerate(ks):
        fut_any = np.zeros((T, N), dtype=bool)
        fut_valid = np.zeros((T, N), dtype=bool)
        for t in range(T - k):
            fut_any[t] = shock[t + 1: t + k + 1].any(axis=0)
            fut_valid[t] = valid[t + 1: t + k + 1].all(axis=0)
        lab[:, :, :, kk] = shock[:, :, None] & fut_any[:, None, :] & off_diag
        msk[:, :, :, kk] = valid[:, :, None] & fut_valid[:, None, :] & off_diag
    lab &= msk
    return lab, msk
