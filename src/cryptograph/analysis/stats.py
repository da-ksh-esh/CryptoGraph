"""Statistical properties that drive model design (distribution, stationarity, dependence in time, shift)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def return_distribution_table(r: pd.DataFrame) -> pd.DataFrame:
    rows = {}
    for c in r.columns:
        x = r[c].dropna().to_numpy()
        if len(x) < 30:
            continue
        rows[c] = {"n": len(x), "mean": x.mean(), "std": x.std(ddof=1), "skew": stats.skew(x),
                   "excess_kurtosis": stats.kurtosis(x), "q0.1%": np.quantile(x, 0.001), "q1%": np.quantile(x, 0.01),
                   "q99%": np.quantile(x, 0.99), "q99.9%": np.quantile(x, 0.999), "min": x.min(), "max": x.max(),
                   "jarque_bera_p": stats.jarque_bera(x).pvalue}
    return pd.DataFrame(rows).T


def adf_table(prices: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    from arch.unitroot import ADF

    rows = {}
    for c in prices.columns:
        lp, rr = np.log(prices[c]).dropna(), returns[c].dropna()
        if len(rr) < 200:
            continue
        a_p, a_r = ADF(lp, trend="c"), ADF(rr, trend="c")
        rows[c] = {"adf_stat_logprice": a_p.stat, "p_logprice": a_p.pvalue, "adf_stat_return": a_r.stat,
                   "p_return": a_r.pvalue}
    return pd.DataFrame(rows).T


def acf_summary(r: pd.DataFrame, nlags: int = 20) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """ACF of returns, |returns|^2, and Ljung-Box p-values at lag 10 for both."""
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.tsa.stattools import acf

    ra, sa, lb = {}, {}, {}
    for c in r.columns:
        x = r[c].dropna()
        if len(x) < 200:
            continue
        ra[c] = acf(x, nlags=nlags, fft=True)[1:]
        sa[c] = acf(x ** 2, nlags=nlags, fft=True)[1:]
        lb[c] = {"lb10_p_returns": float(acorr_ljungbox(x, lags=[10])["lb_pvalue"].iloc[0]),
                 "lb10_p_squared": float(acorr_ljungbox(x ** 2, lags=[10])["lb_pvalue"].iloc[0])}
    lags = range(1, nlags + 1)
    return pd.DataFrame(ra, index=lags), pd.DataFrame(sa, index=lags), pd.DataFrame(lb).T


def ks_shift(X: np.ndarray, split: np.ndarray, names: list[str]) -> pd.DataFrame:
    """Two-sample KS (train vs val, train vs test) per feature, pooled over coins."""
    rows = []
    for f, n in enumerate(names):
        tr = X[split == "train"][..., f].ravel()
        tr = tr[np.isfinite(tr)]
        row = {"feature": n}
        for other in ("val", "test"):
            o = X[split == other][..., f].ravel()
            o = o[np.isfinite(o)]
            if len(tr) and len(o):
                ks = stats.ks_2samp(tr, o)
                row[f"KS_train_vs_{other}"], row[f"p_train_vs_{other}"] = ks.statistic, ks.pvalue
        for s in ("train", "val", "test"):
            v = X[split == s][..., f].ravel()
            v = v[np.isfinite(v)]
            row[f"mean_{s}"], row[f"std_{s}"] = (v.mean(), v.std()) if len(v) else (np.nan, np.nan)
        rows.append(row)
    return pd.DataFrame(rows).set_index("feature")


def mutual_information(X: np.ndarray, y: np.ndarray, names: list[str], seed: int = 0) -> pd.Series:
    """MI between each (finite) feature and a binary label, rows with all features finite."""
    from sklearn.feature_selection import mutual_info_classif

    ok = np.isfinite(X).all(axis=1)
    mi = mutual_info_classif(X[ok], y[ok].astype(int), random_state=seed, n_neighbors=5)
    return pd.Series(mi, index=names).sort_values(ascending=False)
