"""Unit tests for lead-lag cross-correlation, Granger causality, and Epps effect."""

import numpy as np
import pandas as pd
import pytest
from cryptograph.analysis.lead_lag import (
    compute_cross_correlation,
    run_granger_causality,
    granger_tests_with_bh_correction,
    shuffled_null_lead_lag,
    compute_lead_lag_strength_matrix
)
from cryptograph.analysis.epps_effect import EppsEffectAnalyzer


def test_synthetic_leader_follower_detection():
    """Verify that a synthetic leader series with lag 2 is identified correctly by cross-corr and Granger."""
    np.random.seed(42)
    T = 1500
    leader = np.random.normal(0, 1, size=T)
    noise = np.random.normal(0, 0.5, size=T)

    # Follower lags leader by exactly 2 steps: y_t = 0.8 * x_{t-2} + noise
    follower = np.zeros(T)
    follower[2:] = 0.8 * leader[:-2] + noise[2:]

    s_leader = pd.Series(leader, name="LEADER")
    s_follower = pd.Series(follower, name="FOLLOWER")

    # 1. Cross-correlation
    cc = compute_cross_correlation(s_leader, s_follower, max_lag=5)
    # Peak must be at lag = +2 (LEADER leads FOLLOWER)
    assert cc.idxmax() == 2
    assert cc.loc[2] > 0.65

    # 2. Granger test: LEADER -> FOLLOWER must be highly significant
    f_stat, p_val = run_granger_causality(s_leader, s_follower, max_lag=3)
    assert p_val < 1e-4

    # Reverse Granger: FOLLOWER -> LEADER should NOT be significant
    f_rev, p_rev = run_granger_causality(s_follower, s_leader, max_lag=3)
    assert p_rev > 0.05

    # 3. Shuffled null envelope: true peak at lag 2 must exceed the 95% null bound
    emp, null_up, null_low = shuffled_null_lead_lag(s_leader, s_follower, max_lag=5, n_permutations=50)
    assert emp.loc[2] > null_up.loc[2]


def test_epps_effect_correlation_attenuation():
    """Verify that high-frequency aggregation causes monotonic decay in cross-asset correlation."""
    dates = pd.date_range("2024-01-01", periods=24 * 60, freq="1h", tz="UTC")
    np.random.seed(42)

    # Underlying common drift + idiosyncratic noise
    common = np.cumsum(np.random.normal(0, 0.01, size=len(dates)))
    p1 = 100 * np.exp(common + np.random.normal(0, 0.02, size=len(dates)))
    p2 = 50 * np.exp(common + np.random.normal(0, 0.02, size=len(dates)))

    df_prices = pd.DataFrame({"COIN1": p1, "COIN2": p2}, index=dates)

    analyzer = EppsEffectAnalyzer(frequencies=["1h", "4h", "1d"])
    df_epps = analyzer.compute_epps_curve(df_prices)

    assert len(df_epps) == 3
    # Daily correlation should be higher than hourly due to time-averaging out micro-noise
    corr_1h = df_epps.loc[df_epps["frequency"] == "1h", "mean_correlation"].values[0]
    corr_1d = df_epps.loc[df_epps["frequency"] == "1d", "mean_correlation"].values[0]
    assert corr_1d > corr_1h
