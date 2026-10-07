import numpy as np
import pandas as pd

from cryptograph.analysis.persistence import (abs_z_profile, block_bootstrap_ci, fresh_shocks, persistence_table,
                                              shock_breadth, shock_type)


def test_breadth_counts_other_active_coins_only():
    shock = np.array([[1, 1, 0, 0], [1, 0, 0, 0]], bool)
    active = np.array([[1, 1, 1, 0], [1, 1, 1, 1]], bool)
    b = shock_breadth(shock, active)
    assert b[0, 0] == 0.5 and b[0, 2] == 1.0               # coin 0: 1 of the 2 other active coins shocked
    assert b[1, 0] == 0.0 and abs(b[1, 1] - 1 / 3) < 1e-12
    assert list(shock_type(np.array([0.3, 0.1, 0.05]))) == ["market-wide", "partial", "idiosyncratic"]


def test_fresh_shocks_skip_clusters():
    s = np.zeros((40, 1), bool)
    s[[5, 6, 20, 35], 0] = True
    f = fresh_shocks(s, fresh_hours=10)
    assert np.flatnonzero(f[:, 0]).tolist() == [5, 20, 35]    # 6 follows 5; 20 is 14 h after 6


def test_persistence_measures_on_a_synthetic_shock():
    T, N = 200, 3
    rng = np.random.default_rng(0)
    z = rng.normal(0, 0.5, (T, N))
    z[100, 0] = -5.0                                        # crash of coin 0 at t = 100
    z[101:104, 0] = [3.0, 2.5, 2.0]                         # elevated |z| for 3 hours, then back to normal
    sigma = np.full((T, N), 0.01)
    r = z * sigma
    shock = np.abs(z) > 2.29
    valid = active = np.ones((T, N), bool)
    times = pd.date_range("2024-01-01", periods=T, freq="h")
    df = persistence_table(r, z, sigma, shock, valid, active, times, ["A", "B", "C"])
    row = df[(df["coin"] == "A") & (df["t"] == 100)].iloc[0]
    assert row["direction"] == "crash" and row["type"] == "idiosyncratic"
    assert row["reshock_24h"]                               # hours 101-103 are shocks again
    assert 3 <= row["hours_to_normal"] <= 10
    assert row["continuation_1h"] < 0                       # +3 sigma after a crash = reversal
    prof = abs_z_profile(z, df, ["A", "B", "C"], pre=2, post=2)
    assert len(prof) == 5 * df["type"].nunique()
    ci = block_bootstrap_ci(df, "hours_to_normal", ["type"], n_boot=50)
    assert (ci["ci_low"] <= ci["median"]).all() and (ci["median"] <= ci["ci_high"]).all()
