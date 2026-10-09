import json

import numpy as np
import pandas as pd

from cryptograph.serving.events import event_replay, model_vs_outcome

from .conftest import COINS, K_HOURS


def test_event_replay_frames_sequence_and_metrics():
    T, N = 40, 4
    times = pd.date_range("2024-03-01", periods=T, freq="h")
    shock = np.zeros((T, N), bool)
    active = np.ones((T, N), bool)
    active[:, 3] = False                                         # coin D not trading
    t0 = 10
    shock[t0, 0] = True                                          # A at the crisis hour
    shock[[t0 + 3, t0 + 4, t0 + 5, t0 + 8], 1] = True            # B: first 3 h later, a 3-hour run, then again
    shock[t0 - 2, 2] = True                                      # C only before the crisis hour
    ret = np.full((T, N), 0.001)
    ret[t0 + 4, 1] = -0.07
    r = event_replay(shock, active, ret, times, list("ABCD"), t0, before=5, after=20)
    assert r["window"] == [-5, 20] and len(r["frames"]) == 26
    assert r["frames"][5]["offset"] == 0 and r["frames"][5]["shocked"] == [0] and 3 not in r["frames"][5]["active"]
    seq = {s["coin"]: s for s in r["sequence"]}
    assert seq["A"]["shocking_at_anchor"] and seq["A"]["first_shock_offset_h"] == 0 and seq["A"]["first_run_h"] == 1
    assert seq["B"]["first_shock_offset_h"] == 3 and seq["B"]["first_run_h"] == 3 and seq["B"]["shock_hours_after"] == 4
    assert abs(seq["B"]["peak_abs_return_pct"] - 7.0) < 1e-9
    assert seq["C"]["first_shock_offset_h"] is None                # its shock was before the crisis hour
    m = r["metrics"]
    assert m["coins_active_at_anchor"] == 3 and m["coins_shocking_at_anchor"] == 1
    assert abs(m["reach"] - 2 / 3) < 1e-12 and m["hours_to_first_follower"] == 3 and m["median_first_run_h"] == 2.0
    clipped = event_replay(shock, active, ret, times, list("ABCD"), 2, before=48, after=500)
    assert clipped["window"] == [-2, T - 1 - 2]


def test_model_vs_outcome_counts_hits_against_chance():
    risk = pd.DataFrame({"horizon_h": [1] * 6 + [6] * 2, "risk": [0.9, 0.8, 0.7, 0.2, 0.1, 0.05, 0.5, 0.4],
                         "score": [0.0] * 8, "actual": [1, 0, 1, 0, 0, np.nan, 1, 1]})
    out = model_vs_outcome(risk, top=3).set_index("horizon_h")
    assert out.loc[1, "coins_scored"] == 5 and out.loc[1, "top_hits"] == 2 and out.loc[1, "top_n"] == 3
    assert np.isclose(out.loc[1, "share_shocked"], 0.4) and np.isclose(out.loc[1, "expected_by_chance"], 1.2)
    assert out.loc[6, "top_n"] == 2 and out.loc[6, "top_hits"] == 2                     # fewer coins than `top`


def test_events_task_end_to_end(toy_project):
    from cryptograph.training.runner import run_experiment

    root, run, d = toy_project
    cfg = {"name": "events_toy", "model": {"name": "events"}, "device": "cpu",
           "events": {"target_experiment": "toy_tgn", "ig_steps": 4, "before_hours": 6, "after_hours": 24, "top": 2}}
    out = run_experiment(cfg, seed=0, runs_dir=root / "runs")
    m = out["metrics"]
    assert m["n_events"] == 2 and m["n_skipped"] == 1 and m["rebuild_max_abs_logit_diff"] < 1e-4
    rd = root / "runs" / out["run_id"]
    idx = json.loads((rd / "events.json").read_text())
    assert [e["name"] for e in idx["events"]] == ["early", "late"] and "before data" in idx["skipped"][0]
    assert [e["split"] for e in idx["events"]] == ["train", "test"]
    late = json.loads((rd / f"event_{idx['events'][1]['id']}.json").read_text())
    t0 = late["anchor_bar"]
    assert late["anchor_time"].startswith("2026-01-12") and late["source_coins"]
    risk = pd.DataFrame(late["risk"])
    assert sorted(risk.horizon_h.unique()) == K_HOURS and set(risk.sector) <= {"x", "y"}
    src, dst, t = d.src.numpy(), d.dst.numpy(), d.t.numpy()
    for r in risk.itertuples():                                 # outcome = the stream's own label for that event
        p = np.flatnonzero((src == COINS.index(r.source)) & (dst == COINS.index(r.coin)) & (t == t0))
        k = K_HOURS.index(r.horizon_h)
        assert len(p) == 1
        if d.y_mask[p[0], k]:
            assert r.actual == float(d.y[p[0], k])
        else:
            assert r.actual is None or np.isnan(r.actual)
    assert len(late["replay"]["frames"]) == late["replay"]["window"][1] - late["replay"]["window"][0] + 1
    assert all(len(x["drivers"]) == 3 for x in late["risk"])
