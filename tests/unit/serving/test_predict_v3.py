"""Serving a v3 TGN (exp/accuracy-v3 options: shock-size messages, calendar and context features, hourly memory
updates, mean aggregator, role flag, optionally no memory): the served scores match a replay of the stream built the
way training builds it, and the drivers name the new inputs."""

import json

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from .conftest import COINS, K_HOURS

V3 = {"use_memory": True, "shock_size_messages": True, "calendar_features": True, "context_features": True,
      "memory_aggregator": "mean", "role_flag": True}


def _v3_run(root, d, F, overrides, name, seed=0):
    """A trained-looking v3 run: the model built from the same tables as tgn_task (random weights)."""
    from cryptograph.data.storage import load_pt
    from cryptograph.models.tgn import ContagionTGN
    from cryptograph.training.tgn_task import CTX_GLOBAL, CTX_NODE

    torch.manual_seed(seed)
    mcfg = {"memory_dim": 8, "time_dim": 4, "embedding_dim": 8, "dropout": 0.0} | overrides
    ctx_dim = 2 * len(CTX_NODE) + len(CTX_GLOBAL)
    model = ContagionTGN(num_nodes=len(COINS), msg_dim=6, node_feat_dim=F + 3, out_horizons=len(K_HOURS),
                         memory_dim=8, time_dim=4, embedding_dim=8, dropout=0.0, use_memory=mcfg["use_memory"],
                         ctx_dim=ctx_dim, aggregator="mean", role_flag=True)
    run = root / "runs" / f"20261008-00000{seed}_{name}_{seed}_abc1234"
    run.mkdir(parents=True)
    cfg = {"name": name, "data": {"version": "v2", "resolution": "1h", "network_features": False},
           "training": {"batch_size": 20, "num_neighbors": 5, "memory_step": "hour"}}
    (run / "config.resolved.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    torch.save({"model_state": model.state_dict(), "model_config": mcfg, "k_hours": K_HOURS, "ctx_dim": ctx_dim},
               run / "checkpoint.pt")
    pv = load_pt(root / "runs" / "20261007-000000_toy_tgn_0_abc1234" / "predictions_val.pt")
    torch.save(pv, run / "predictions_val.pt")
    return run, mcfg


def _replay(root, run, mcfg) -> np.ndarray:
    """Scores of every event from an hour-by-hour pass over the stream, tables built as in tgn_task."""
    from cryptograph.data.storage import load_pt
    from cryptograph.models.tgn import ContagionTGN
    from cryptograph.training.tgn_task import (StreamRunner, calendar_table, context_tables, node_feature_table,
                                               shock_size_messages)

    base = root / "data" / "processed" / "v2" / "1h"
    panel, ev = load_pt(base / "panel.pt"), load_pt(base / "tgn_events.pt")
    d, gidx = ev["full"], ev["graph_index"]
    d.msg = shock_size_messages(d, panel, gidx)
    feats = node_feature_table(panel, None, gidx)
    cal = calendar_table(panel)
    feats = torch.cat([feats, cal[:, None, :].expand(-1, feats.shape[1], -1)], dim=-1).contiguous()
    ck = load_pt(run / "checkpoint.pt")
    model = ContagionTGN(num_nodes=len(gidx), msg_dim=6, node_feat_dim=feats.shape[-1], out_horizons=len(K_HOURS),
                         memory_dim=8, time_dim=4, embedding_dim=8, dropout=0.0, use_memory=mcfg["use_memory"],
                         ctx_dim=ck["ctx_dim"], aggregator="mean", role_flag=True)
    model.load_state_dict(ck["model_state"])
    model.eval()
    wmask = {s: panel["sample_mask"][s].numpy() for s in ("train", "val", "test")}
    runner = StreamRunner(model, d, feats, wmask, torch.device("cpu"), 20, 5, ctx=context_tables(panel, gidx),
                          memory_step="hour")
    t = d.t.numpy()
    cuts = np.r_[0, np.flatnonzero(np.diff(t)) + 1, len(t)]
    out = []
    with torch.no_grad():
        for s, e in zip(cuts[:-1], cuts[1:]):
            logits, evs = runner._step(int(s), int(e))
            runner._commit(evs)
            out.append(torch.sigmoid(logits))
    return torch.cat(out).numpy()


@pytest.mark.parametrize("memory", [True, False], ids=["memory", "no_memory"])
def test_v3_run_is_served_like_its_stream(toy_project, memory):
    from cryptograph.explain.tgn_capture import load_trained
    from cryptograph.training.runner import run_experiment

    root, _, d = toy_project
    F = 3
    name = "toy_tgn_v3" if memory else "toy_tgn_v3_nomem"
    run, mcfg = _v3_run(root, d, F, V3 | {"use_memory": memory}, name)
    runner, D = load_trained(run)
    assert D["msg_names"][-2:] == ["src_z", "dst_z"] and D["node_names"][-1] == "calendar: weekend"
    assert len(D["ctx_names"]) == 10 and D["ctx_names"][-1] == "market: breadth_mean_24h"
    assert runner.memory_step == "hour" and runner.ctx is not None

    stream = _replay(root, run, mcfg)
    cfg = {"name": "predict_toy", "model": {"name": "predict"}, "device": "cpu",
           "predict": {"target_experiment": name, "lookback_hours": 3, "ig_steps": 16, "top_drivers": 30}}
    out = run_experiment(cfg, seed=0, runs_dir=root / "runs")
    m = out["metrics"]
    assert m["target_run"] == run.name and m["rebuild_max_abs_logit_diff"] < 1e-4
    assert m["ig_completeness_max_gap"] < 1e-2
    rd = root / "runs" / out["run_id"]
    risk = pd.read_csv(rd / "risk.csv")
    src, dst, t = d.src.numpy(), d.dst.numpy(), d.t.numpy()
    for r in risk.itertuples():                     # the served score is the stream's score for that event
        p = np.flatnonzero((src == COINS.index(r.source)) & (dst == COINS.index(r.coin)) & (t == r.event_bar))
        assert len(p) == 1 and np.isclose(r.score, stream[p[0], K_HOURS.index(r.horizon_h)], atol=1e-5)
    summary = json.loads((rd / "summary.json").read_text())
    assert summary["uses_memory"] is memory
    inputs = {x["input"] for v in json.loads((rd / "drivers.json").read_text()).values() for x in v}
    assert any(i.startswith("market: ") for i in inputs) and any("calendar: " in i for i in inputs)
    assert any(i in ("event: src_z", "event: dst_z") for i in inputs)


def test_v3_inputs_read_in_plain_words():
    from cryptograph.serving import dashboard_data as dd

    say = lambda i, a=0.1: dd.plain_driver({"input": i, "attribution": a}, "JUP", "QNT")   # noqa: E731
    assert say("receiver coin: lr_rv_24h") == ("JUP: how jumpy its price was over the last day, compared with its "
                                               "usual level (raises the risk)")
    assert say("source coin: max5_ratio") == "QNT: its largest 5-minute move in the last hour (raises the risk)"
    assert say("market: breadth_now", -0.1) == ("the market: how many coins are making a big move right now "
                                                "(lowers the risk)")
    assert say("event: dst_z").startswith("how much the coin itself moved in the same hour (from QNT to JUP)")
    assert say("receiver coin: calendar: weekend") == "JUP: whether it is the weekend (raises the risk)"
    ctx = pd.DataFrame({"coin": ["JUP"], "vol_multiple": [1.4], "shocks_24h": [2]})
    assert dd.plain_driver({"input": "receiver coin: lr_rv_24h", "attribution": 0.1}, "JUP", "QNT", ctx) == \
        "JUP: price 1.4× as jumpy as usual over the last day (raises the risk)"
    assert "2 big moves in the last 24 hours" in dd.plain_driver(
        {"input": "receiver coin: shocks_last_24h", "attribution": 0.1}, "JUP", "QNT", ctx)
    assert "remembers how coins reacted" in dd.about_text(None, [6])
    no_mem = dd.about_text(None, [6], uses_memory=False)
    assert "remembers" not in no_mem and "whole market" in no_mem
