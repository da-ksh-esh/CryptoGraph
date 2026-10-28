import json

import numpy as np
import pandas as pd
import pytest

from cryptograph.serving import dashboard_data as dd
from cryptograph.serving.replay import first_move_after, top_rows, window_bars, window_score, window_split

from .conftest import COINS, K_HOURS


def test_window_bars_split_and_first_move():
    times = pd.date_range("2026-01-01", periods=10, freq="h")
    assert window_bars(times, "2026-01-01 02:00", "2026-01-01 05:00") == (2, 5)
    assert window_bars(times, "2025-12-31", "2026-01-01 01:30") == (0, 1)              # clipped to the data
    with pytest.raises(ValueError, match="outside the data"):
        window_bars(times, "2026-02-01", "2026-02-02")
    sid = np.array([0, 0, 1, 1, 2, 2, -1, -1, 2, 2])
    assert window_split(sid, 4, 5) == "test" and window_split(sid, 6, 7) == "after the research period"
    assert window_split(sid, 1, 2) == "mixed" and window_split(sid, 0, 1) == "train"
    shock = np.zeros((10, 2), bool)
    shock[[3, 7], 0] = True
    fm = first_move_after(shock, np.array([0, 3, 3, 8]), np.array([0, 0, 1, 0]), horizon=4)
    assert fm[0] == 3 and fm[1] == 4 and np.isnan(fm[2]) and np.isnan(fm[3])            # t=8: data ends at 9


def test_top_rows_and_window_score_against_random_picks():
    t = pd.DataFrame({"bar": [5] * 4 + [9] * 4, "horizon_h": [1] * 8, "coin": list("ABCD") * 2,
                      "risk": [0.9, 0.8, 0.2, 0.1, 0.7, 0.6, 0.5, 0.4], "lift": [3.0, 2.0, 0.5, 0.2, 2.5, 2.0, 1.0, 0.9],
                      "score": [0.0] * 8, "actual": [1, 0, 1, 0, 1, 1, 0, np.nan],
                      "shocking_now": [True, False, False, False, False, False, False, False]})
    picks = top_rows(t, 2).sort_values(["bar", "lift"], ascending=[True, False])
    assert picks["coin"].tolist() == ["A", "B", "A", "B"]
    s = window_score(t, [1], 2)[0]
    # bar 5: picks A (hit), B (miss); share 2/4 -> chance 1.0. bar 9 (D unobserved): A, B both hit; share 2/3
    assert s["alerts"] == 4 and s["hits"] == 3 and np.isclose(s["expected_by_chance"], 1.0 + 2 * 2 / 3)
    calm = window_score(t, [1], 2, skip_moving=True)[0]
    # A is moving at bar 5: picks B (miss), C (hit) among B, C, D (share 1/3); bar 9 unchanged
    assert calm["alerts"] == 4 and calm["hits"] == 3 and np.isclose(calm["expected_by_chance"], 2 / 3 + 4 / 3)


def test_alert_status_reveals_hits_when_they_happen_and_scores_decided_alerts():
    risk = pd.DataFrame({"bar": [10, 10, 10, 12], "horizon_h": [3] * 4, "coin": ["A", "B", "C", "A"],
                         "risk": [0.5, 0.4, 0.1, 0.6], "lift": [2.0, 1.5, 0.5, 2.5], "score": [0.0] * 4,
                         "actual": [1.0, 0.0, 1.0, np.nan], "first_move_h": [2.0, np.nan, 1.0, np.nan],
                         "shocking_now": [False] * 4, "time": ["t"] * 4, "source": ["S"] * 4})
    alerts = dd.replay_alerts(risk, 3, top=2)
    assert alerts[["bar", "coin"]].values.tolist() == [[10, "A"], [10, "B"], [12, "A"]]
    s = dd.alert_status(alerts, 11, 3)                                    # A moves at 12: not known yet at 11
    assert s["status"].tolist() == ["pending", "pending"] and s["hours_left"].tolist() == [2, 2]
    s = dd.alert_status(alerts, 12, 3)
    assert s["status"].tolist() == ["hit", "pending", "pending"] and not s["decided"].any()
    s = dd.alert_status(alerts, 15, 3)
    assert s["status"].tolist() == ["hit", "miss", "unknown"] and s["decided"].tolist() == [True, True, False]
    sc = dd.replay_score(risk, s, 3)
    assert sc["decided"] == 2 and sc["hits"] == 1 and np.isclose(sc["expected"], 2 * 2 / 3)   # bar 10: 2 of 3 moved
    assert dd.score_verdict(1, 1.3, 2)[0] == "neutral"                    # too few to judge
    assert dd.score_verdict(9, 5.0, 20)[0] == "good" and dd.score_verdict(3, 5.0, 20)[0] == "weak"


@pytest.fixture
def replayed(toy_project):
    from cryptograph.training.runner import run_experiment

    root, run, d = toy_project
    cfg = {"name": "replay_toy", "model": {"name": "replay"}, "device": "cpu",
           "replay": {"target_experiment": "toy_tgn", "ig_steps": 4, "top": 2, "top_drivers": 2,
                      "windows": [{"name": "late crash", "event": "late", "start": "2026-01-10 14:00",
                                   "end": "2026-01-11 20:00"},
                                  {"name": "across splits", "start": "2026-01-09 18:00", "end": "2026-01-10 06:00",
                                   "description": "val and test"}]}}
    out = run_experiment(cfg, seed=0, runs_dir=root / "runs")
    return root, root / "runs" / out["run_id"], d, out["metrics"]


def test_replay_task_scores_every_hour_of_each_window(replayed):
    root, rd, d, m = replayed
    assert m["n_windows"] == 2 and m["rebuild_max_abs_logit_diff"] < 1e-4
    idx = json.loads((rd / "replay.json").read_text())
    w0, w1 = idx["windows"]
    assert w0["split"] == "test" and w0["description"] == "a late crash" and w0["hours"] == 31
    assert w1["split"] == "mixed" and w1["description"] == "val and test"
    R = dd.load_replay(rd, w0["id"])
    assert len(R["frames"]) == 31 and R["frames"][0]["time"] == "2026-01-10 14:00:00"
    risk = R["risk"]
    t = d.t.numpy()
    bars = set(t[(t >= 230) & (t <= 260)])
    assert set(risk["bar"]) == bars and all(f["scored"] == (f["bar"] in bars) for f in R["frames"])
    assert sorted(risk["horizon_h"].unique()) == K_HOURS and set(risk["coin"]) <= set(COINS)
    src, dst = d.src.numpy(), d.dst.numpy()
    for r in risk.sample(20, random_state=0).itertuples():             # outcome = the stream's label of that event
        p = np.flatnonzero((src == COINS.index(r.source)) & (dst == COINS.index(r.coin)) & (t == r.bar))
        k = K_HOURS.index(r.horizon_h)
        assert len(p) == 1
        assert (r.actual == float(d.y[p[0], k])) if d.y_mask[p[0], k] else pd.isna(r.actual)
    with_drivers = risk[risk["drivers"].map(len) > 0]
    assert len(with_drivers) == len(top_rows(risk, 2)) and all(len(x) == 2 for x in with_drivers["drivers"])
    import torch

    panel = torch.load(root / "data" / "processed" / "v2" / "1h" / "panel.pt", weights_only=False)
    sh = (panel["shock_today"] & panel["shock_today_mask"]).numpy()
    for r in risk.sample(20, random_state=1).itertuples():              # first big move after the hour, from shocks
        c = COINS.index(r.coin)
        nxt = np.flatnonzero(sh[r.bar + 1: r.bar + 25, c])
        assert (r.first_move_h == nxt[0] + 1) if len(nxt) else pd.isna(r.first_move_h)
        assert bool(r.shocking_now) == bool(sh[r.bar, c])
    assert dd.latest_replay_dir(root / "runs") == rd and R["score"]
