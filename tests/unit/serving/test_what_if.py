import numpy as np
import pandas as pd
import pytest

from cryptograph.serving import what_if as wi


def _returns(T=600, seed=0):
    """B follows A in the same hour (beta 0.8), C follows A one hour later (0.5), D is on its own."""
    rng = np.random.default_rng(seed)
    a = 0.01 * rng.normal(size=T)
    b = 0.8 * a + 0.002 * rng.normal(size=T)
    c = 0.5 * np.r_[0.0, a[:-1]] + 0.002 * rng.normal(size=T)
    d = 0.01 * rng.normal(size=T)
    times = [str(x) for x in pd.date_range("2026-01-01", periods=T, freq="h")]
    return np.c_[a, b, c, d], times, ["A", "B", "C", "D"]


def test_shock_spreads_in_the_same_hour_and_through_the_lags():
    r, times, coins = _returns()
    m = wi.fit_shock_model(r, times, coins, t_end=len(r) - 1, window=500, horizon=6)
    assert m.coins == coins and m.irf.shape == (7, 4, 4) and m.n_hours == 500 and m.start == times[100]
    assert np.allclose(np.diag(m.irf[0]), 1.0)                       # a coin's own shock moves it one for one
    sim = wi.simulate(m, "A", -10)
    h0 = sim[sim["hour"] == 0].set_index("coin")
    assert np.isclose(h0.loc["A", "change_pct"], -10.0)
    assert np.isclose(h0.loc["B", "change_pct"], 100 * np.expm1(0.8 * np.log(0.9)), atol=0.5)
    assert abs(h0.loc["C", "change_pct"]) < 0.5 and abs(h0.loc["D", "change_pct"]) < 0.5
    h1 = sim[sim["hour"] == 1].set_index("coin")
    assert np.isclose(h1.loc["C", "move_pct"], 100 * np.expm1(0.5 * np.log(0.9)), atol=0.5)  # one hour later
    assert h0.loc["B", "big"] and not h0.loc["C", "big"] and h1.loc["C", "big"]   # big moves count from that hour
    assert not sim.loc[sim["source"], "big"].any() and not h0.loc["D", "big"]
    up = wi.simulate(m, "A", 10).set_index(["hour", "coin"])["move_pct"]           # linear in the log return
    log_up = np.log1p(up / 100)
    log_down = np.log1p(sim.set_index(["hour", "coin"])["move_pct"] / 100)
    assert np.allclose(log_up / np.log(1.1), log_down / np.log(0.9))


def test_at_hour_options_and_sentence():
    r, times, coins = _returns()
    m = wi.fit_shock_model(r, times, coins, t_end=len(r) - 1, window=500, horizon=3)
    sim = wi.simulate(m, "A", -10)
    assert wi.at_hour(sim, 0)["coin"].tolist()[:2] == ["A", "B"]          # the shocked coin first, then the largest
    words = wi.what_if_sentence(sim, "A", -10, 0)
    assert "**A** dropped **10%**" in words and "**B** would move most" in words and "**1 of 3**" in words
    assert "no other coin would move" in wi.what_if_sentence(wi.simulate(m, "A", 0), "A", 0, 0)
    ranks = pd.DataFrame({"coin": ["A", "B", "C"], "dy_net": [-1.0, 5.0, 2.0]})
    assert wi.coin_options(coins, ranks) == ["B", "C", "A", "D"]        # biggest spreaders first, unranked last
    assert wi.coin_options(["B", "A"], pd.DataFrame()) == ["A", "B"]


def test_sparse_coins_are_left_out_and_too_little_data_is_refused():
    r, times, coins = _returns()
    r[:, 3][::3] = np.nan                                               # D observed in 2/3 of the hours
    m = wi.fit_shock_model(r, times, coins, t_end=len(r) - 1, window=300)
    assert m.coins == ["A", "B", "C"] and m.n_hours == 300
    with pytest.raises(ValueError, match="too little data"):
        wi.fit_shock_model(r, times, coins, t_end=5, window=300)
