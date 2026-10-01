import numpy as np
import pandas as pd

from cryptograph.data.datasets.indexed import sample_mask_times, window_index
from cryptograph.data.features.intraday import (SeasonalityProfile, bucket_index, build_intraday_features,
                                                prune_redundant)
from cryptograph.data.features.targets import future_any


def _master(hours=24 * 60, coins=("A", "B"), seed=0):
    rng = np.random.default_rng(seed)
    t = pd.date_range("2021-01-04", periods=hours, freq="h", name="time")
    season = 1 + 2.0 * (t.hour == 14)                      # strong 14:00 UTC volatility/volume
    frames = []
    for c in coins:
        r = rng.normal(0, 0.01, hours) * np.sqrt(season)
        close = 100 * np.exp(np.cumsum(r))
        vol = np.exp(rng.normal(5, 0.2, hours)) * season
        frames.append(pd.DataFrame({"open": close, "high": close * 1.002, "low": close * 0.998, "close": close,
                                    "volume": vol}, index=pd.MultiIndex.from_product([t, [c]], names=["time", "coin"])))
    return pd.concat(frames).sort_index()


def test_bucket_index():
    t = pd.DatetimeIndex(["2024-01-01 00:00", "2024-01-01 14:00", "2024-01-07 23:00"])   # Monday 00h, Monday 14h, Sunday 23h
    assert bucket_index(t, 1).tolist() == [0, 14, 6 * 24 + 23]
    assert bucket_index(t, 2).tolist() == [0, 7, 6 * 12 + 11]


def test_seasonality_profile_found_and_fitted_on_train_only():
    m = _master()
    close = m["close"].unstack("coin")
    vol = m["volume"].unstack("coin")
    r = np.log(close).diff()
    train = np.zeros(len(r), bool)
    train[: len(r) // 2] = True
    p = SeasonalityProfile.fit(r, vol, train, 1)
    b14 = [k for k in range(168) if k % 24 == 14]
    assert p.vol[b14].mean() > 2 * np.delete(p.vol, b14).mean()        # the 14:00 effect is found
    # changing data outside train does not change the profile
    r2, v2 = r.copy(), vol.copy()
    r2.iloc[len(r) // 2:] *= 50
    v2.iloc[len(r) // 2:] *= 50
    p2 = SeasonalityProfile.fit(r2, v2, train, 1)
    assert np.allclose(p.vol, p2.vol) and np.allclose(p.logvol, p2.logvol)
    # deseasonalised returns no longer show the 14:00 effect
    rd = p.deseason_returns(r)
    ratio = (rd[r.index.hour == 14] ** 2).mean().mean() / (rd[r.index.hour != 14] ** 2).mean().mean()
    assert 0.7 < ratio < 1.4


def test_intraday_features_are_causal():
    m = _master()
    coins = ["A", "B"]
    close = m["close"].unstack("coin")
    r = np.log(close).diff()
    p = SeasonalityProfile.fit(r, m["volume"].unstack("coin"), np.ones(len(r), bool), 1)
    X1, t, names, _ = build_intraday_features(m, coins, p, [24, 72], 24, 168, 17520)
    m2 = m.copy()
    cut = t[1000]
    m2.loc[m2.index.get_level_values("time") > cut, ["open", "high", "low", "close", "volume"]] *= 3
    X2, *_ = build_intraday_features(m2, coins, p, [24, 72], 24, 168, 17520)
    a, b = X1[:1001], X2[:1001]
    assert ((a == b) | (np.isnan(a) & np.isnan(b))).all()


def test_prune_redundant_keeps_first_and_protected():
    rng = np.random.default_rng(1)
    base = rng.normal(size=(500, 3))
    X = np.stack([base[:, 0], base[:, 1], base[:, 1] * 2 + 0.01 * base[:, 2], base[:, 2]], axis=-1)[:, None, :]
    kept, dropped = prune_redundant(X, ["log_return", "rv_24h", "rv_72h", "vol"], np.ones(500, bool), 0.9)
    assert kept == ["log_return", "rv_24h", "vol"] and dropped[0][:2] == ("rv_72h", "rv_24h")


def test_hourly_sample_mask_embargo_and_windows():
    t = pd.date_range("2022-06-25", "2022-07-20 23:00", freq="h")
    bar = pd.Timedelta(hours=1)
    train = sample_mask_times(t, "2022-01-01", "2022-06-30", bar=bar, embargo=pd.Timedelta(0), lookback_bars=24,
                              horizon=pd.Timedelta(hours=24))
    val = sample_mask_times(t, "2022-07-01", "2023-12-31", bar=bar, embargo=pd.Timedelta(days=7), lookback_bars=168,
                            horizon=pd.Timedelta(hours=24))
    assert t[train].max() == pd.Timestamp("2022-06-29 23:00")          # target window ends by 30 Jun 23:00
    assert t[val].min() == pd.Timestamp("2022-07-08 00:00")            # 7-day embargo (window also inside val)
    assert not (train & val).any()


def test_window_index_requires_fully_active_window():
    T = 50
    active = np.ones((T, 2), bool)
    active[20, 0] = False
    y_ok = np.ones((T, 2), bool)
    rows = np.ones(T, bool)
    idx = window_index(active, y_ok, rows, [0, 1], lookback=10)
    t0 = idx[idx[:, 1] == 0, 0]
    assert not np.isin(t0, np.arange(20, 30)).any() and t0.min() == 9
    assert (idx[idx[:, 1] == 1, 0] == np.arange(9, T)).all()


def test_future_any_strictly_after_t():
    s = np.zeros((6, 1), bool)
    s[3] = True
    anyf, allv = future_any(s, np.ones_like(s), 2)
    assert anyf[:, 0].tolist() == [False, True, True, False, False, False]
    assert allv[:, 0].tolist() == [True, True, True, True, False, False]


def test_hourly_snapshot_dataset_selects_split_bars_and_graph_coins(tmp_path):
    import torch

    from cryptograph.data.datasets.indexed import HourlySnapshotDataset

    T, N, F = 40, 3, 2
    X = torch.arange(T * N * F, dtype=torch.float32).reshape(T, N, F)
    sm = {"train": torch.zeros(T, dtype=torch.bool), "val": torch.zeros(T, dtype=torch.bool)}
    sm["train"][5:20] = True
    sm["val"][25:40] = True
    ones = torch.ones(T, N, dtype=torch.bool)
    panel = {"X": X, "X_mask": torch.ones(T, N, F, dtype=torch.bool), "active": ones, "y_ret": torch.zeros(T, N, 2),
             "y_rv": torch.zeros(T, N, 2), "y_mask": torch.ones(T, N, 2, dtype=torch.bool), "y_shock": ones,
             "y_shock_mask": ones, "sample_mask": sm, "feature_names": ["a", "b"], "bar_hours": 1}
    snap_t = torch.arange(4, T, 8)                        # 4, 12, 20, 28, 36
    e = torch.tensor([[0, 1], [1, 0]])
    snaps = {"snapshot_t": snap_t, "graph_index": [0, 2], "graph_coins": ["A", "C"], "undirected_threshold": 0.7,
             "undirected_edge_index": [e] * len(snap_t), "undirected_edge_weight": [torch.ones(2)] * len(snap_t)}
    torch.save(panel, tmp_path / "panel.pt")
    torch.save(snaps, tmp_path / "graph_snapshots.pt")
    tr = HourlySnapshotDataset(tmp_path / "panel.pt", tmp_path / "graph_snapshots.pt", "train")
    va = HourlySnapshotDataset(tmp_path / "panel.pt", tmp_path / "graph_snapshots.pt", "val")
    assert [int(d.t) for d in tr] == [12] and [int(d.t) for d in va] == [28, 36]
    d = va[0]
    assert d.num_nodes == 2 and torch.equal(d.x, X[28][[0, 2]])
