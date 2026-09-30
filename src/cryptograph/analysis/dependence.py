"""Cross-asset dependence: correlation structure, lead-lag and Granger causality."""

from __future__ import annotations

import numpy as np
import pandas as pd


def cross_correlation(lead: pd.Series, others: pd.DataFrame, lags=range(-5, 6)) -> pd.DataFrame:
    """corr(lead[t], other[t+lag]) — positive lag = `lead` leads the other coin."""
    out = {}
    for c in others.columns:
        out[c] = {lag: lead.corr(others[c].shift(-lag)) for lag in lags}
    return pd.DataFrame(out)


def granger_matrix(returns: pd.DataFrame, lag: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """p-values of 'column j Granger-causes row i' (bivariate, `lag` lags) and BH-adjusted p-values."""
    from statsmodels.stats.multitest import multipletests
    from statsmodels.tsa.stattools import grangercausalitytests

    cols = list(returns.columns)
    P = pd.DataFrame(np.nan, index=cols, columns=cols)
    for i in cols:
        for j in cols:
            if i == j:
                continue
            d = returns[[i, j]].dropna()
            if len(d) < 100:
                continue
            res = grangercausalitytests(d, maxlag=[lag], verbose=False)
            P.loc[i, j] = res[lag][0]["ssr_ftest"][1]
    flat = P.to_numpy().ravel()
    ok = np.isfinite(flat)
    adj = np.full_like(flat, np.nan)
    adj[ok] = multipletests(flat[ok], method="fdr_bh")[1]
    return P, pd.DataFrame(adj.reshape(P.shape), index=cols, columns=cols)


def average_pairwise_corr(r: pd.DataFrame, window: int = 60, min_obs: int = 40) -> pd.Series:
    """Mean off-diagonal rolling correlation over coins active in the window."""
    from ..data.graphs.correlation import pairwise_corr

    X = r.to_numpy()
    vals = np.full(len(X), np.nan)
    for t in range(window - 1, len(X)):
        C = pairwise_corr(X[t - window + 1: t + 1], X[t - window + 1: t + 1], min_obs)
        np.fill_diagonal(C, np.nan)
        vals[t] = np.nanmean(C) if np.isfinite(C).any() else np.nan
    return pd.Series(vals, index=r.index, name=f"avg_corr_{window}d")
