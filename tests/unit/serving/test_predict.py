import json

import numpy as np
import pandas as pd
import pytest
import torch

from cryptograph.serving.predict import Calibrator, coin_risk, latest_prediction_dir, recent_positions, top_drivers

from .conftest import COINS, K_HOURS


def test_recent_positions_latest_bar_lookback_and_as_of():
    t = np.array([0, 0, 1, 3, 3, 3, 6])
    pos, as_of = recent_positions(t)
    assert as_of == 6 and pos.tolist() == [6]
    pos, as_of = recent_positions(t, lookback=4)                # bars 3..6
    assert pos.tolist() == [3, 4, 5, 6]
    pos, as_of = recent_positions(t, as_of=5)                    # no events at bar 5: use the latest bar before it
    assert as_of == 3 and pos.tolist() == [3, 4, 5]
    with pytest.raises(ValueError):
        recent_positions(t, as_of=-1)


def test_calibrator_is_monotone_in_score_and_horizon():
    rng = np.random.default_rng(0)
    n = 20000
    p = rng.random((n, 2))
    y = np.stack([rng.random(n) < 0.2 * p[:, 0], rng.random(n) < p[:, 1]], 1)   # k=0 rarer than k=1
    mask = np.ones_like(y)
    mask[:100, 0] = False
    cal = Calibrator(p, y, mask)
    grid = np.linspace(0, 1, 11)[:, None].repeat(2, 1)
    out = cal(grid)
    assert (np.diff(out, axis=0) >= 0).all()                    # higher score, higher risk
    assert (out[:, 1] >= out[:, 0]).all()                       # "within the longer horizon" is never less likely
    assert out[0, 1] < 0.1 and out[-1, 1] > 0.9 and out[-1, 0] < 0.3
    assert abs(cal.base_rate[0] - y[100:, 0].mean()) < 1e-12


def test_calibrator_avoids_0_and_1_from_few_events():
    rng = np.random.default_rng(1)
    p = rng.random((3000, 1))
    y = rng.random((3000, 1)) < 0.3                             # no signal, 30% positives
    out = Calibrator(p, y, np.ones_like(y), min_bin=500)(np.array([[0.0], [0.5], [1.0]]))
    assert ((out > 0.2) & (out < 0.4)).all()                    # plain isotonic would hit 0 or 1 at the ends


def test_coin_risk_takes_each_receivers_riskiest_event():
    coins = ["A", "B", "C"]
    src, dst, t = np.array([0, 0, 1, 1]), np.array([1, 2, 0, 2]), np.array([5, 5, 5, 5])
    prob = np.array([[0.2, 0.3], [0.6, 0.7], [0.4, 0.5], [0.1, 0.9]])
    risk = prob.copy()
    risk[3, 0] = 0.6                                             # tie with event 1 for receiver C at k=0
    df = coin_risk(prob, risk, src, dst, t, coins, [1, 6], base_rate=[0.2, 0.4])
    c1 = df[(df.coin == "C") & (df.horizon_h == 1)].iloc[0]
    assert c1.risk == 0.6 and c1.source == "A" and c1.n_events == 2      # tie broken by the raw score (0.6 > 0.1)
    c6 = df[(df.coin == "C") & (df.horizon_h == 6)].iloc[0]
    assert c6.source == "B" and c6.risk == 0.9 and np.isclose(c6.lift, 0.9 / 0.4)
    assert set(df.coin) == set(coins) and len(df) == 6
    assert df.groupby("horizon_h")["risk"].apply(lambda s: s.is_monotonic_decreasing).all()


def test_top_drivers_sorted_by_magnitude_with_sign():
    attr = [torch.tensor([[0.1, 0.0]]), torch.tensor([[0.0, 0.0]]), torch.tensor([[-0.5, 0.0]]),
            torch.tensor([[0.0, 0.2]]), torch.tensor([[0.05]])]
    out = top_drivers(attr, 0, ["x0", "x1"], ["m0"], top=3)
    assert [o["input"] for o in out] == ["source coin: x0", "receiver coin: x1", "source coin: memory + recent events"]
    assert out[0]["direction"] == "lowers" and out[1]["direction"] == "raises"


def test_shock_context_types_sources_and_uses_no_later_data():
    from cryptograph.serving.predict import shock_context

    from .conftest import toy_panel

    p = toy_panel(T=900)
    as_of, gidx = 800, list(range(len(COINS)))
    p["shock_today"][as_of] = torch.tensor([True, True, False, False, False])     # 2 of 4 others: 25% -> partial
    p["shock_z"][as_of, :2] = torch.tensor([-3.0, 3.0])
    p["X_raw"][as_of, :2, 0] = torch.tensor([-0.03, 0.03])
    src, hist = shock_context(p, gidx, as_of, [0, 1], COINS)
    assert src.coin.tolist() == ["AAA", "BBB"] and src.direction.tolist() == ["crash", "pump"]
    assert np.allclose(src.breadth, 0.25) and (src.type == "partial").all()
    assert len(hist) and hist.n_shocks.gt(0).all() and hist.median_hours_to_normal.between(1, 168).all()
    p2 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in p.items()}
    p2["shock_z"][as_of + 1:] = 50.0                                          # change everything after as_of
    p2["shock_today"][as_of + 1:] = True
    p2["X_raw"][as_of + 1:] = -1.0
    src2, hist2 = shock_context(p2, gidx, as_of, [0, 1], COINS)
    pd.testing.assert_frame_equal(src, src2)
    pd.testing.assert_frame_equal(hist, hist2)


def test_predict_task_end_to_end(toy_project):
    from cryptograph.training.runner import run_experiment

    root, run, d = toy_project
    cfg = {"name": "predict_toy", "model": {"name": "predict"}, "device": "cpu",
           "predict": {"target_experiment": "toy_tgn", "target_seed": 0, "lookback_hours": 2, "ig_steps": 16,
                       "n_alerts": 5, "top_drivers": 3}}
    out = run_experiment(cfg, seed=0, runs_dir=root / "runs")
    m = out["metrics"]
    assert m["target_run"] == run.name and m["calibrated"] and m["rebuild_max_abs_logit_diff"] < 1e-4
    assert m["ig_completeness_max_gap"] < 1e-2
    rd = root / "runs" / out["run_id"]
    risk = pd.read_csv(rd / "risk.csv")
    assert sorted(risk.horizon_h.unique()) == K_HOURS and set(risk.coin) <= set(COINS)
    assert risk.risk.between(0, 1).all() and (risk.event_bar >= int(d.t.max()) - 1).all()
    alerts = json.loads((rd / "alerts.json").read_text())
    assert len(alerts) == 5 and all(len(a["drivers"]) == 3 for a in alerts)
    assert np.isclose(alerts[0]["lift"], risk.lift.max())
    assert all(a["lift"] >= b["lift"] for a, b in zip(alerts, alerts[1:]))
    summary = json.loads((rd / "summary.json").read_text())
    assert summary["hours_since_as_of"] == 0 and summary["n_events"] == int((d.t >= d.t.max() - 1).sum())
    shocks = pd.read_csv(rd / "shock_sources.csv")
    last = d.t.max()
    assert set(shocks.coin) == {COINS[s] for s in d.src[d.t == last].unique().tolist()}
    assert latest_prediction_dir(root / "runs") == rd and latest_prediction_dir(root / "runs", "other") is None
    assert (root / "runs" / "results.csv").exists()


def test_predict_matches_the_stream_scores(toy_project):
    """The served risk comes from the same logits as an evaluation pass over the stream (no separate code path)."""
    from cryptograph.explain.tgn_capture import load_trained
    from cryptograph.training.runner import run_experiment

    from cryptograph.training.tgn_task import _batches

    root, run, d = toy_project
    runner, _ = load_trained(run)
    runner.model.eval()
    runner.reset()
    stream = []
    with torch.no_grad():                                      # the evaluation loop, without label masks
        for s, e in _batches(runner.t_np, 0, len(d.t), runner.bs):
            logits, ev = runner._step(s, e)
            runner._commit(ev)
            stream.append(torch.sigmoid(logits))
    stream = torch.cat(stream).numpy()
    cfg = {"name": "predict_toy", "model": {"name": "predict"}, "device": "cpu",
           "predict": {"target_experiment": "toy_tgn", "lookback_hours": 3, "ig_steps": 4}}
    out = run_experiment(cfg, seed=0, runs_dir=root / "runs")
    risk = pd.read_csv(root / "runs" / out["run_id"] / "risk.csv")
    assert len(risk) == len(COINS) * len(K_HOURS)
    src, dst, t = d.src.numpy(), d.dst.numpy(), d.t.numpy()
    for r in risk.itertuples():
        p = np.flatnonzero((src == COINS.index(r.source)) & (dst == COINS.index(r.coin)) & (t == r.event_bar))
        assert len(p) == 1
        assert np.isclose(r.score, stream[p[0], K_HOURS.index(r.horizon_h)], atol=1e-5)
