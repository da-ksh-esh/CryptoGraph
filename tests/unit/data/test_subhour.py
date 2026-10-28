"""5-minute features: computed from bars inside the hour only (sandbox exp/accuracy-v3)."""

import numpy as np
import pandas as pd

from cryptograph.data.features.subhour import subhour_features


def _bars(hours=4, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=hours * 12 + 1, freq="5min")
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.001, len(idx))))
    k5 = pd.DataFrame({"close": close}, index=pd.MultiIndex.from_product([idx, ["A"]], names=["time", "coin"]))
    return k5, pd.date_range("2024-01-01", periods=hours, freq="h")


def test_realized_vol_matches_the_hour_bars():
    k5, times = _bars()
    f = subhour_features(k5, times, ["A"], np.ones(len(times)), np.ones((len(times), 1)))
    c = k5["close"].droplevel("coin")
    r = np.log(c).diff()
    h1 = r[(r.index >= times[1]) & (r.index < times[2])]          # the 12 returns of the bar opening at 01:00
    assert len(h1) == 12
    assert np.isclose(f["rv5_ratio"][1, 0], np.log(np.sqrt((h1 ** 2).sum())), atol=1e-5)
    assert np.isclose(f["last15"][1, 0], h1[h1.index.minute >= 45].sum(), atol=1e-6)


def test_future_bars_do_not_change_earlier_hours():
    k5, times = _bars()
    base = subhour_features(k5, times, ["A"], np.ones(len(times)), np.ones((len(times), 1)))
    k5b = k5.copy()
    k5b.loc[k5b.index.get_level_values("time") >= times[2], "close"] *= 1.5   # change everything from 02:00 on
    alt = subhour_features(k5b, times, ["A"], np.ones(len(times)), np.ones((len(times), 1)))
    for k in base:
        np.testing.assert_allclose(base[k][:2], alt[k][:2], equal_nan=True)
