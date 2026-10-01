"""Hourly lead-lag: cross-correlation profiles, a circular-shift null, Granger tests and the Epps effect.

All functions take (time x coin) data that the caller has already restricted to one split
(the notebook uses train for discovery and validation for replication).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def _standardise(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = np.isfinite(x)
    v = np.where(m, x, 0.0)
    if m.sum() > 1:
        v = np.where(m, (v - v[m].mean()) / (v[m].std() + 1e-12), 0.0)
    return v, m.astype(float)


def _xcorr(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    """c[k] = sum_t a[t] * b[(t + k) mod n] for k = 0..n-1 (FFT)."""
    return np.fft.irfft(np.conj(np.fft.rfft(a, n)) * np.fft.rfft(b, n), n)


def ccf_profile(x: np.ndarray, y: np.ndarray, max_lag: int) -> np.ndarray:
    """corr(x[t], y[t+k]) for k = -max_lag..max_lag (positive k = x leads y), NaN-aware.

    Each series is standardised over its own valid bars; the sum over pairwise-valid bars is
    divided by their count, which approximates the pairwise correlation for long series.
    """
    xs, mx = _standardise(np.asarray(x, float))
    ys, my = _standardise(np.asarray(y, float))
    n = len(xs) + max_lag + 1                           # zero padding: no wrap-around for |k| <= max_lag
    s, c = _xcorr(xs, ys, n), _xcorr(mx, my, n)
    ks = np.arange(-max_lag, max_lag + 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = s[ks % n] / c[ks % n]
    out[c[ks % n] < 30] = np.nan
    return out


def circular_null_max(x: np.ndarray, y: np.ndarray, max_lag: int, n_null: int = 500, min_shift: int = 720,
                      seed: int = 0) -> np.ndarray:
    """Null distribution of max_{0<|k|<=max_lag} |ccf_k| when y is circularly shifted by a random offset.

    A shift keeps each series' own autocorrelation and volatility clustering but breaks their
    alignment. Because the circular cross-correlation of a shifted y is a shifted window of the
    circular cross-correlation, every null draw is read off one FFT.
    """
    xs, mx = _standardise(np.asarray(x, float))
    ys, my = _standardise(np.asarray(y, float))
    T = len(xs)
    s, c = _xcorr(xs, ys, T), _xcorr(mx, my, T)
    with np.errstate(invalid="ignore", divide="ignore"):
        circ = np.abs(s / c)
    rng = np.random.default_rng(seed)
    centers = rng.integers(min_shift, T - min_shift, n_null)
    offs = np.r_[np.arange(-max_lag, 0), np.arange(1, max_lag + 1)]
    return np.nanmax(circ[(centers[:, None] + offs[None, :]) % T], axis=1)


def lead_lag_table(z: pd.DataFrame, leaders: list[str], max_lag: int = 24, n_null: int = 500, seed: int = 0,
                   min_obs: int = 2000) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per (leader, coin): the ccf profile (long) and a summary with the circular-shift null.

    summary columns: contemporaneous ccf, the largest |ccf| at a non-zero lag and its lag,
    the null's 95th/99th percentile for that maximum, and whether it is exceeded.
    """
    prof, rows = [], []
    ks = np.arange(-max_lag, max_lag + 1)
    for li, lead in enumerate(leaders):
        for c in z.columns:
            if c == lead or c in leaders[:li] or z[c].notna().sum() < min_obs:
                continue
            cc = ccf_profile(z[lead].to_numpy(), z[c].to_numpy(), max_lag)
            prof.append(pd.DataFrame({"leader": lead, "coin": c, "lag": ks, "ccf": cc}))
            nz = np.where(ks != 0, np.abs(cc), np.nan)
            k_best = int(ks[np.nanargmax(nz)])
            null = circular_null_max(z[lead].to_numpy(), z[c].to_numpy(), max_lag, n_null, seed=seed + len(rows))
            q95, q99 = np.nanquantile(null, [0.95, 0.99])
            rows.append({"leader": lead, "coin": c, "contemporaneous": float(cc[max_lag]),
                         "max_abs_nonzero": float(np.nanmax(nz)), "best_lag": k_best, "null_q95": float(q95),
                         "null_q99": float(q99), "exceeds_q99": bool(np.nanmax(nz) > q99)})
    return pd.concat(prof, ignore_index=True), pd.DataFrame(rows)


def granger_leader_table(z: pd.DataFrame, leaders: list[str], lags: int = 3, min_obs: int = 2000) -> pd.DataFrame:
    """Bivariate Granger F-tests (lags 1..`lags`) leader -> coin and coin -> leader, BH-adjusted together.

    `dR2` is the in-sample R^2 the cause's lags add to the effect's own lags (effect size; with
    ~20k hourly bars even tiny effects are significant).
    """
    from statsmodels.stats.multitest import multipletests
    from statsmodels.tsa.stattools import grangercausalitytests

    rows = []
    for li, lead in enumerate(leaders):
        for c in z.columns:
            if c == lead or c in leaders[:li] or z[c].notna().sum() < min_obs:
                continue
            for cause, effect, direction in ((lead, c, "leader -> coin"), (c, lead, "coin -> leader")):
                d = z[[effect, cause]].dropna()
                with warnings.catch_warnings():          # statsmodels deprecates `verbose` but prints without it
                    warnings.simplefilter("ignore", FutureWarning)
                    res = grangercausalitytests(d, maxlag=[lags], verbose=False)[lags]
                tss = float(((d[effect] - d[effect].mean()) ** 2).sum())
                rows.append({"leader": lead, "coin": c, "direction": direction, "cause": cause, "effect": effect,
                             "n": len(d), "F": float(res[0]["ssr_ftest"][0]), "p": float(res[0]["ssr_ftest"][1]),
                             "dR2": (res[1][0].ssr - res[1][1].ssr) / tss})
    out = pd.DataFrame(rows)
    out["p_bh"] = multipletests(out["p"], method="fdr_bh")[1]
    return out


def epps_curve(log_close: pd.DataFrame, intervals_hours: list[int], bar_hours: int = 1, min_periods: int = 200
               ) -> pd.DataFrame:
    """Mean / median off-diagonal correlation of returns sampled every k hours (Epps effect)."""
    rows = []
    for k in intervals_hours:
        step = max(1, k // bar_hours)
        rk = log_close.iloc[::step].diff()
        C = rk.corr(min_periods=max(10, min_periods // step)).to_numpy()
        np.fill_diagonal(C, np.nan)
        rows.append({"interval_hours": k, "mean_corr": float(np.nanmean(C)), "median_corr": float(np.nanmedian(C)),
                     "n_pairs": int(np.isfinite(C).sum() // 2)})
    return pd.DataFrame(rows)
