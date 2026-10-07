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


def _decaying_panel(T=24 * 120, N=6, half_life=6.0, seed=0):
    """Calm noise; at planted fresh shocks |r| jumps and its excess decays with the given half-life."""
    rng = np.random.default_rng(seed)
    r = rng.normal(0, 1.0, (T, N))
    shock = np.zeros((T, N), bool)
    for c in range(N):
        for t in range(400 + 37 * c, T - 200, 300):
            shock[t, c] = True
            r[t, c] = 8.0
            h = np.arange(1, 73)
            r[t + 1:t + 73, c] *= 1 + 4 * 0.5 ** ((h - 1) / half_life)
    return r, shock, np.ones((T, N), bool), pd.date_range("2024-01-01", periods=T, freq="h")


def test_decay_half_life_interpolates():
    from cryptograph.analysis.persistence import decay_half_life

    assert decay_half_life(np.array([4.0, 3.0, 2.0, 1.0])) == 3.0           # 2.0 reached exactly at hour 3
    assert decay_half_life(np.array([4.0, 3.0, 1.0])) == 2.5
    assert decay_half_life(np.array([4.0, 3.9])) == 2.0                     # never halves: the window length
    assert np.isnan(decay_half_life(np.array([-1.0, 0.5])))


def test_excess_vs_matched_null_recovers_the_planted_half_life():
    from cryptograph.analysis.persistence import excess_vs_null, sector_half_life

    r, shock, act, times = _decaying_panel(half_life=6.0)
    ep = excess_vs_null(r, shock, act, act, times, horizon=72)
    assert len(ep["t"]) > 30 and ep["sh"].shape == ep["nu"].shape == (len(ep["t"]), 72)
    assert set(np.unique(ep["coin"])) == set(range(6))
    coins = list("ABCDEF")
    sectors = {"A": "x", "B": "x", "C": "x", "D": "y", "E": "y", "F": "y"}
    tab, pairs = sector_half_life(ep, coins, sectors, times, n_boot=100)
    assert set(tab.sector) == {"x", "y"} and (tab.gap_h1 > 1).all() and (tab.p_gap_h1_not_positive == 0).all()
    assert tab.half_life_h.between(4, 9).all()                              # planted: excess |r| halves in ~6 h
    assert (tab.ci_low <= tab.half_life_h).all() and (tab.half_life_h <= tab.ci_high).all()
    assert len(pairs) == 1 and pairs.ci_low.iloc[0] <= 0 <= pairs.ci_high.iloc[0]   # same process in both sectors
    tab2, _ = sector_half_life(ep, coins, {**sectors, "F": "z"}, times, n_boot=10)
    assert set(tab2.sector) == {"x"}                                        # y has 2 coins left, z has 1


def test_event_blast_radius_anchors_on_the_shock_hour():
    from cryptograph.analysis.persistence import event_blast_radius

    T, N = 24 * 30, 5
    times = pd.date_range("2024-01-01", periods=T, freq="h")
    shock = np.zeros((T, N), bool)
    t0 = 24 * 20 + 14                                                        # 14:00 on day 20
    shock[t0, :3] = True                                                     # 3 of 5 coins: market-wide
    shock[t0 + 5, 3] = True                                                  # a follower 5 hours later
    act = np.ones((T, N), bool)
    events = pd.DataFrame({"date": [str(times[t0].date()), "2030-01-01"], "name": ["crash", "outside"]})
    out = event_blast_radius(shock, act, act, times, list("ABCDE"), events)
    assert len(out) == 1
    row = out.iloc[0]
    assert row.anchor_hour == str(times[t0]) and row.coins_shocking_at_anchor == 3 and row.type == "market-wide"
    assert row.coins_hit_72h == 4 and row.peak_spillover_hour == 5 and row.control_shock_rate == 0
    assert row.abnormal_shock_rate > 0
