import numpy as np
import pandas as pd

from cryptograph.analysis.lead_lag import (ccf_profile, circular_null_max, epps_curve, granger_leader_table,
                                           lead_lag_table)
from cryptograph.data.graphs.correlation import threshold_for_density


def _leader_panel(T=6000, lag=2, beta=0.4, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=T)
    y = beta * np.roll(x, lag) + rng.normal(size=T)       # x leads y by `lag`
    y[:lag] = rng.normal(size=lag)
    w = rng.normal(size=T)                                # unrelated coin
    return pd.DataFrame({"LEAD": x, "FOL": y, "IND": w})


def test_ccf_profile_finds_the_lag_and_matches_pandas():
    z = _leader_panel()
    cc = ccf_profile(z["LEAD"].to_numpy(), z["FOL"].to_numpy(), 5)
    lags = np.arange(-5, 6)
    assert lags[np.argmax(np.abs(cc))] == 2
    ref = z["LEAD"].corr(z["FOL"].shift(-2))
    assert abs(cc[5 + 2] - ref) < 0.01


def test_ccf_profile_handles_missing_values():
    z = _leader_panel()
    x, y = z["LEAD"].to_numpy().copy(), z["FOL"].to_numpy().copy()
    x[:1000] = np.nan
    y[3000:3500] = np.nan
    cc = ccf_profile(x, y, 3)
    ref = pd.Series(x).corr(pd.Series(y).shift(-2))
    assert np.isfinite(cc).all() and abs(cc[3 + 2] - ref) < 0.02


def test_circular_null_separates_leader_from_independent():
    z = _leader_panel()
    q99 = np.quantile(circular_null_max(z["LEAD"].to_numpy(), z["IND"].to_numpy(), 5, n_null=300, min_shift=200), 0.99)
    obs_ind = np.nanmax(np.abs(np.delete(ccf_profile(z["LEAD"].to_numpy(), z["IND"].to_numpy(), 5), 5)))
    obs_fol = np.nanmax(np.abs(np.delete(ccf_profile(z["LEAD"].to_numpy(), z["FOL"].to_numpy(), 5), 5)))
    assert obs_fol > q99 > obs_ind and q99 < 0.1


def test_lead_lag_table_and_granger_direction():
    z = _leader_panel()
    prof, summ = lead_lag_table(z, ["LEAD"], max_lag=5, n_null=200, min_obs=100)
    s = summ.set_index("coin")
    assert s.loc["FOL", "best_lag"] == 2 and s.loc["FOL", "exceeds_q99"]
    assert set(prof["coin"]) == {"FOL", "IND"}
    g = granger_leader_table(z, ["LEAD"], lags=3, min_obs=100).set_index(["coin", "direction"])
    assert g.loc[("FOL", "leader -> coin"), "p_bh"] < 1e-6 and g.loc[("FOL", "leader -> coin"), "dR2"] > 0.05
    assert g.loc[("FOL", "coin -> leader"), "p_bh"] > 0.01


def test_epps_curve_rises_with_interval_for_asynchronous_prices():
    rng = np.random.default_rng(3)
    T = 24 * 400
    common = rng.normal(size=T)
    a = common + 0.5 * rng.normal(size=T)
    b = np.roll(common, 1) + 0.5 * rng.normal(size=T)     # b reacts one bar late (stale quotes)
    lc = pd.DataFrame({"A": np.cumsum(a), "B": np.cumsum(b)})
    e = epps_curve(lc, [1, 4, 24]).set_index("interval_hours")["mean_corr"]
    assert e[1] < e[4] < e[24]


def test_threshold_for_density_hits_target():
    rng = np.random.default_rng(0)
    mats, acts = [], []
    for _ in range(20):
        M = rng.uniform(0, 1, (10, 10))
        M = (M + M.T) / 2
        np.fill_diagonal(M, np.nan)
        mats.append(M)
        acts.append(np.ones(10, bool))
    thr, dens = threshold_for_density(mats, acts, 0.30)
    assert dens <= 0.30 and 0.6 < thr < 0.8
    thr2, _ = threshold_for_density(mats, acts, 0.30 + 1e-9)
    assert thr2 == thr
    # inactive nodes are ignored
    acts2 = [np.r_[np.ones(5, bool), np.zeros(5, bool)] for _ in acts]
    thr3, d3 = threshold_for_density(mats, acts2, 0.30)
    assert d3 <= 0.30
