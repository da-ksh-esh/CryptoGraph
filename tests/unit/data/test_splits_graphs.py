from datetime import date

import numpy as np
import pandas as pd

from cryptograph.data.datasets.lstm import build_windows
from cryptograph.data.datasets.splits import SplitPlan
from cryptograph.data.graphs.correlation import directed_edges, pairwise_corr, rolling_corr_mats, undirected_edges
from cryptograph.data.graphs.temporal_stream import build_event_arrays, shock_edges, to_temporal_data


def test_shock_edges_only_from_shocked_active_sources():
    shock = np.array([True, False, True, False])
    active = np.array([True, True, False, True])
    ei, w = shock_edges(shock, active, np.full((4, 4), 0.1))
    pairs = set(map(tuple, ei.T.tolist()))
    assert pairs == {(0, 1), (0, 3)}      # node 2 shocked but inactive; no self loops
    assert np.allclose(w, 0.1)

BOUNDS = {"train": (date(2020, 1, 1), date(2020, 12, 31)), "val": (date(2021, 1, 1), date(2021, 6, 30)),
          "test": (date(2021, 7, 1), date(2021, 12, 31))}


def _plan():
    return SplitPlan.from_config(BOUNDS, embargo_days=30, lookback=30, max_horizon=7)


def test_splits_disjoint_and_embargo():
    dates = pd.date_range("2020-01-01", "2021-12-31", freq="D")
    plan = _plan()
    masks = {s: plan.sample_mask(dates, s) for s in BOUNDS}
    assert not (masks["train"] & masks["val"]).any() and not (masks["val"] & masks["test"]).any()
    for s, (lo, hi) in BOUNDS.items():
        d = dates[masks[s]]
        emb = 0 if s == "train" else 30
        assert d.min() >= pd.Timestamp(lo) + pd.Timedelta(days=max(emb, 29))   # embargo + input window inside split
        assert d.max() + pd.Timedelta(days=7) <= pd.Timestamp(hi)             # target window inside split
    # nothing within 30 days after a boundary
    assert not masks["val"][(dates >= "2021-01-01") & (dates < "2021-01-31")].any()


def test_window_never_extends_past_sample_date():
    T, N, F = 120, 2, 3
    X = np.arange(T, dtype=np.float32)[:, None, None].repeat(N, 1).repeat(F, 2)
    ones = np.ones((T, N), bool)
    y = np.zeros((T, N, 2), np.float32)
    days = np.zeros(T, bool)
    days[40:60] = True
    w = build_windows(X, np.ones_like(X, bool), ones, y, y, np.zeros((T, N), bool), ones, days, [0, 1], lookback=30)
    assert (w["X"][:, -1, 0] == w["day"]).all()          # last input row is day t itself
    assert (w["X"].max(axis=(1, 2)) <= w["day"]).all()   # nothing after t
    # inactive day inside the window drops the sample
    act = ones.copy()
    act[45, 0] = False
    w2 = build_windows(X, np.ones_like(X, bool), act, y, y, np.zeros((T, N), bool), ones, days, [0], lookback=30)
    assert not np.isin(w2["day"], np.arange(45, 75)).any()


def test_pairwise_corr_matches_pandas(rng):
    a = rng.normal(size=(50, 3))
    a[5, 1] = np.nan
    c = pairwise_corr(a, a, min_obs=10)
    ref = pd.DataFrame(a).corr().to_numpy()
    assert np.allclose(c, ref, atol=1e-10)


def test_graphs_use_only_past_and_are_valid(rng):
    r = rng.normal(size=(80, 4))
    r[:, 1] = r[:, 0] + 0.1 * rng.normal(size=80)
    r[1:, 2] = r[:-1, 0]  # coin 0 leads coin 2 by one day
    C, D = rolling_corr_mats(r, window=30, min_obs=20)
    r2 = r.copy()
    r2[60:] = rng.normal(size=(20, 4))
    C2, D2 = rolling_corr_mats(r2, window=30, min_obs=20)
    assert np.allclose(np.nan_to_num(C[:60]), np.nan_to_num(C2[:60]))
    assert np.allclose(np.nan_to_num(D[:60]), np.nan_to_num(D2[:60]))
    act = np.ones(4, bool)
    ei, w = undirected_edges(C[50], act, 0.5, 0)
    assert {(0, 1), (1, 0)} <= set(map(tuple, ei.T.tolist()))
    di, dw = directed_edges(D[50], act, 0.5)
    assert (0, 2) in set(map(tuple, di.T.tolist())) and (2, 0) not in set(map(tuple, di.T.tolist()))


def test_event_stream_sorted(rng):
    T, N = 40, 3
    edges = []
    for t in range(T):
        k = rng.integers(0, 4)
        ei = np.stack([rng.integers(0, N, k), rng.integers(0, N, k)])
        edges.append((ei, rng.normal(size=k).astype(np.float32)))
    zeros = np.zeros((T, N), np.float32)
    ev = build_event_arrays(edges, np.zeros((T, N, N), np.float32), zeros, zeros, zeros.astype(bool))
    data = to_temporal_data(ev, pd.date_range("2020-01-01", periods=T))
    assert (data.t[1:] >= data.t[:-1]).all()
    assert data.msg.shape[1] == 5 and len(data.t_unix) == len(data.t)
