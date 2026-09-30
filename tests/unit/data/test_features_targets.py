import numpy as np
import pandas as pd

from cryptograph.data.features.node_features import FeatureSpec, build_node_features
from cryptograph.data.features.scaling import RobustScalerSet
from cryptograph.data.features.targets import (contagion_labels, forward_realized_vol, forward_returns,
                                               next_day_shock, shock_z)


def _master(close: pd.DataFrame) -> pd.DataFrame:
    long = close.stack(future_stack=True).rename("close").to_frame()
    long.index.names = ["date", "coin"]
    long["open"] = long["close"]
    long["high"] = long["close"] * 1.02
    long["low"] = long["close"] * 0.98
    long["volume"] = np.where(long["close"].notna(), 1e6, np.nan)
    long["market_cap"] = long["close"] * 1e3
    return long


def test_features_are_causal(synthetic_panel):
    """Changing prices after day t must not change any feature at or before t."""
    coins = list(synthetic_panel.columns)
    X1, dates, names = build_node_features(_master(synthetic_panel), coins)
    perturbed = synthetic_panel.copy()
    perturbed.iloc[300:] *= 3.0
    X2, _, _ = build_node_features(_master(perturbed), coins)
    t = 299
    a, b = X1[: t + 1], X2[: t + 1]
    same = (a == b) | (np.isnan(a) & np.isnan(b))
    # mcap_share at t is cross-sectional on day t only, also unaffected
    assert same.all()


def test_no_features_before_listing(synthetic_panel):
    X, _, names = build_node_features(_master(synthetic_panel), list(synthetic_panel.columns))
    assert np.isnan(X[:100, 2]).all()
    assert names == FeatureSpec().names()


def test_targets_strictly_after_t():
    lc = np.log(np.array([[1.0], [2.0], [4.0], [8.0]]))
    y = forward_returns(lc, [1, 2])
    assert np.isclose(y[0, 0, 0], np.log(2)) and np.isclose(y[0, 0, 1], np.log(4))
    assert np.isnan(y[3, 0, 0]) and np.isnan(y[2, 0, 1])
    r = np.array([[np.nan], [0.1], [-0.2], [0.3]])
    v = forward_realized_vol(r, [1, 2])
    assert np.isclose(v[0, 0, 0], 0.1) and np.isclose(v[1, 0, 1], np.sqrt((0.04 + 0.09) / 2))


def test_shock_uses_past_sigma_and_contagion():
    r = np.array([[0.0, 0.0], [0.5, 0.0], [0.0, 0.5], [0.0, 0.0]])
    sigma = np.full_like(r, 0.1)
    z = shock_z(r, sigma)
    assert np.isnan(z[0]).all() and np.isclose(z[1, 0], 5.0)
    lab, msk = next_day_shock(z, 2.5)
    assert lab[0, 0] and not lab[1, 0]  # label at t = shock at t+1
    s = np.abs(np.nan_to_num(z)) > 2.5
    v = np.isfinite(z)
    cl, cm = contagion_labels(s, v, [1, 2])
    assert cl[1, 0, 1, 0]            # shock at coin0 on day1 followed by coin1 on day2
    assert not cl[1, 0, 0, 0] and not cm[1, 0, 0, 0]  # diagonal masked


def test_scaler_fit_only_on_train_rows(rng):
    X = rng.normal(size=(300, 2, 1))
    train = np.zeros(300, bool)
    train[:200] = True
    X[200:] += 1000.0  # huge shift outside train must not affect the scaler
    s = RobustScalerSet.fit(X, train, ["f"], ["a", "b"], min_rows=50)
    assert np.all(np.abs(s.median) < 1.0) and np.all(np.abs(s.global_median) < 1.0)
    s2 = RobustScalerSet.fit(np.where(train[:, None, None], X, np.nan), train, ["f"], ["a", "b"], min_rows=50)
    assert np.allclose(s.median, s2.median) and np.allclose(s.iqr, s2.iqr)
