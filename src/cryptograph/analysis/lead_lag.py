"""Lead-lag relationship analysis, Granger causality tests with BH correction, and shuffled null validation."""

from typing import List, Tuple, Dict, Optional, Union
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import grangercausalitytests
from ..common.logger import get_logger

logger = get_logger(__name__)


def compute_cross_correlation(
    series_x: pd.Series,
    series_y: pd.Series,
    max_lag: int = 24
) -> pd.Series:
    """
    Computes cross-correlation between series_x and series_y for lags tau in [-max_lag, +max_lag].
    tau > 0: series_x leads series_y (x_{t} correlated with y_{t + tau}).
    tau < 0: series_y leads series_x.
    """
    clean = pd.concat([series_x, series_y], axis=1).dropna()
    x = clean.iloc[:, 0].values
    y = clean.iloc[:, 1].values

    lags = range(-max_lag, max_lag + 1)
    corrs = []

    for lag in lags:
        if lag < 0:
            # y leads x: x shifted forward
            c = np.corrcoef(x[-lag:], y[:lag])[0, 1]
        elif lag > 0:
            # x leads y: y shifted forward
            c = np.corrcoef(x[:-lag], y[lag:])[0, 1]
        else:
            c = np.corrcoef(x, y)[0, 1]
        corrs.append(float(c) if np.isfinite(c) else 0.0)

    return pd.Series(corrs, index=lags, name=f"{series_x.name}->{series_y.name}")


def run_granger_causality(
    leader_series: pd.Series,
    follower_series: pd.Series,
    max_lag: int = 4
) -> Tuple[float, float]:
    """
    Runs Granger causality test of H0: leader does NOT Granger-cause follower.
    Returns (min_f_stat, min_p_value) across lags 1..max_lag.
    """
    df = pd.concat([follower_series, leader_series], axis=1).dropna()
    if len(df) < 50:
        return 0.0, 1.0

    try:
        # grangercausalitytests expects [y, x] in first two columns
        res = grangercausalitytests(df.iloc[:, :2], maxlag=max_lag, verbose=False)
        f_stats = [res[lag][0]["ssr_ftest"][0] for lag in range(1, max_lag + 1)]
        p_vals = [res[lag][0]["ssr_ftest"][1] for lag in range(1, max_lag + 1)]
        best_idx = int(np.argmin(p_vals))
        return float(f_stats[best_idx]), float(p_vals[best_idx])
    except Exception as e:
        logger.warning(f"Granger test failed: {e}")
        return 0.0, 1.0


def granger_tests_with_bh_correction(
    returns_df: pd.DataFrame,
    leaders: List[str] = ["BTC", "ETH"],
    max_lag: int = 4,
    alpha: float = 0.05
) -> pd.DataFrame:
    """
    Performs pairwise Granger causality tests from leaders to all other assets.
    Applies Benjamini-Hochberg (BH) False Discovery Rate (FDR) correction for multiple testing.
    """
    records = []
    followers = [c for c in returns_df.columns if c not in leaders]

    for leader in leaders:
        if leader not in returns_df.columns:
            continue
        l_s = returns_df[leader]

        for follower in returns_df.columns:
            if follower == leader:
                continue
            f_s = returns_df[follower]
            f_stat, p_val = run_granger_causality(l_s, f_s, max_lag=max_lag)
            records.append({
                "leader": leader,
                "follower": follower,
                "f_stat": f_stat,
                "p_value": p_val
            })

    res_df = pd.DataFrame(records)
    if res_df.empty:
        return res_df

    # Benjamini-Hochberg FDR correction
    m = len(res_df)
    res_df = res_df.sort_values("p_value").reset_index(drop=True)
    res_df["rank"] = np.arange(1, m + 1)
    res_df["bh_critical_value"] = (res_df["rank"] / m) * alpha
    res_df["significant_bh"] = res_df["p_value"] <= res_df["bh_critical_value"]

    # Calculate adjusted p-values (q-values)
    q_values = np.zeros(m)
    running_min = 1.0
    for i in reversed(range(m)):
        q = min(1.0, (res_df.loc[i, "p_value"] * m) / res_df.loc[i, "rank"])
        running_min = min(running_min, q)
        q_values[i] = running_min
    res_df["q_value"] = q_values

    return res_df


def shuffled_null_lead_lag(
    series_x: pd.Series,
    series_y: pd.Series,
    max_lag: int = 24,
    n_permutations: int = 100,
    confidence_level: float = 0.95
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Computes cross-correlation alongside empirical upper/lower quantile bounds
    from time-shuffled permutations (null hypothesis of no temporal dependence).

    Returns:
        (empirical_corr, null_upper, null_lower)
    """
    emp_corr = compute_cross_correlation(series_x, series_y, max_lag=max_lag)

    clean = pd.concat([series_x, series_y], axis=1).dropna()
    x = clean.iloc[:, 0].values
    y = clean.iloc[:, 1].values
    T = len(y)

    perm_corrs = np.zeros((n_permutations, len(emp_corr)))

    for p in range(n_permutations):
        y_shuffled = np.random.permutation(y)
        shuff_s_y = pd.Series(y_shuffled, index=clean.index, name=series_y.name)
        c = compute_cross_correlation(clean.iloc[:, 0], shuff_s_y, max_lag=max_lag)
        perm_corrs[p, :] = c.values

    alpha_tail = (1.0 - confidence_level) / 2.0
    null_upper = pd.Series(np.percentile(perm_corrs, (1.0 - alpha_tail) * 100.0, axis=0), index=emp_corr.index)
    null_lower = pd.Series(np.percentile(perm_corrs, alpha_tail * 100.0, axis=0), index=emp_corr.index)

    return emp_corr, null_upper, null_lower


def compute_lead_lag_strength_matrix(
    returns_df: pd.DataFrame,
    max_lag: int = 6
) -> pd.DataFrame:
    """
    Computes directed lead-lag strength matrix L_{i -> j} across all asset pairs.
    L_{i -> j} = max_{1 <= tau <= max_lag} rho_{ij}(tau) - max_{1 <= tau <= max_lag} rho_{ij}(-tau)
    Values > 0 indicate that asset i temporally leads asset j.
    """
    cols = returns_df.columns
    N = len(cols)
    lead_matrix = np.zeros((N, N), dtype=np.float32)

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            cc = compute_cross_correlation(returns_df[cols[i]], returns_df[cols[j]], max_lag=max_lag)
            lead_corr = cc.loc[1:max_lag].max()   # i leads j
            lag_corr = cc.loc[-max_lag:-1].max()  # j leads i
            diff = float(lead_corr - lag_corr)
            lead_matrix[i, j] = max(0.0, diff)

    return pd.DataFrame(lead_matrix, index=cols, columns=cols)
