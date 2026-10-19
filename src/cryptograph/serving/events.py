"""Catalogued crises through the final TGN: what it predicted at each crisis hour, and what then happened.

For every event in data/external/events.csv inside the hourly data, the anchor is the hour of the event date with the
most shocks (`analysis.persistence.event_anchor`; the date alone can sit hours away from the crash). The whole stream
is replayed once (as in serving/predict.py, memory holding only earlier events) and every event at an anchor hour is
scored, calibrated on validation predictions and explained by integrated gradients. Each coin's risk is then set next
to its realised outcome, the event label "shocks within k hours" of the stream.

Not every crisis is a forecast: the period of the anchor hour is recorded, and only test-period events were never used
for training (train), model selection or calibration (validation).

The replay of each window (shocks per coin and hour, first shock and run length per coin, peak |return|) comes from the
panel's shock labels and needs no model. Outputs (runs/<run_id>/): events.json (index) and event_<id>.json.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from ..data.config import PROJECT_ROOT

SPLIT_NAMES = {0: "train", 1: "validation", 2: "test"}
SPLIT_NOTE = {"train": "training period: the model learned from this event, so its scores here are in-sample",
              "validation": "validation period: used to choose the model and calibrate its risk, so not a clean "
                            "forecast",
              "test": "test period: never used for training, selection or calibration; a genuine forecast",
              "between splits": "embargo gap between splits: not used for training, but next to data that was"}


def event_replay(shock: np.ndarray, active: np.ndarray, ret: np.ndarray, times: pd.DatetimeIndex, coins: list[str],
                 t0: int, before: int = 48, after: int = 168) -> dict:
    """Window [t0 - before, t0 + after] around the anchor hour t0 (clipped to the data):
    frames   per hour: offset from t0, time, indices of shocking and of active coins
    sequence per coin: shocking at t0, first shock at or after t0 (offset), length of that run of consecutive shock
             hours, shock hours after t0, peak |return| (%) from t0 on
    metrics  reach (share of the coins active at t0 that shock from t0 on), hours from t0 to the first coin that was
             not shocking at t0, median run length, coins shocking at t0."""
    T = len(times)
    lo, hi = max(0, t0 - before), min(T - 1, t0 + after)
    frames = [{"offset": int(t - t0), "time": str(times[t]), "shocked": np.flatnonzero(shock[t] & active[t]).tolist(),
               "active": np.flatnonzero(active[t]).tolist()} for t in range(lo, hi + 1)]
    post = shock[t0:hi + 1] & active[t0:hi + 1]
    seq = []
    for c, coin in enumerate(coins):
        hits = np.flatnonzero(post[:, c])
        run = 0
        if len(hits):
            run = 1
            while hits[0] + run < len(post) and post[hits[0] + run, c]:
                run += 1
        r = np.abs(ret[t0:hi + 1, c])
        seq.append({"coin": coin, "active_at_anchor": bool(active[t0, c]), "shocking_at_anchor": bool(post[0, c]),
                    "first_shock_offset_h": int(hits[0]) if len(hits) else None, "first_run_h": run,
                    "shock_hours_after": int(post[1:, c].sum()),
                    "peak_abs_return_pct": float(np.nanmax(r) * 100) if np.isfinite(r).any() else None})
    act0 = active[t0]
    reached = post.any(0) & act0
    followers = [s["first_shock_offset_h"] for s in seq if s["first_shock_offset_h"] and not s["shocking_at_anchor"]]
    runs = [s["first_run_h"] for s, a in zip(seq, reached) if a]
    return {"window": [int(lo - t0), int(hi - t0)], "frames": frames,
            "activity": [{"offset": f["offset"], "time": f["time"], "coins_shocking": len(f["shocked"])}
                         for f in frames],
            "sequence": seq,
            "metrics": {"coins_active_at_anchor": int(act0.sum()), "coins_shocking_at_anchor": int(post[0].sum()),
                        "reach": float(reached.sum() / max(act0.sum(), 1)),
                        "hours_to_first_follower": int(min(followers)) if followers else None,
                        "median_first_run_h": float(np.median(runs)) if runs else None}}


def model_vs_outcome(risk: pd.DataFrame, top: int = 5) -> pd.DataFrame:
    """Per horizon: of the `top` coins the model ranked riskiest (with an observed outcome), how many shocked within k
    hours, against the share of all scored coins that did (what `top` random picks would get on average)."""
    rows = []
    for kh, g in risk.groupby("horizon_h"):
        g = g.dropna(subset=["actual"]).sort_values(["risk", "score"], ascending=False)
        if g.empty:
            continue
        n = min(top, len(g))
        rows.append({"horizon_h": int(kh), "coins_scored": int(len(g)), "share_shocked": float(g["actual"].mean()),
                     "top_hits": int(g["actual"].head(n).sum()), "top_n": int(n),
                     "expected_by_chance": float(n * g["actual"].mean())})
    return pd.DataFrame(rows)


def events_task(cfg: dict, seed: int, device: torch.device, tracker) -> dict[str, Any]:
    from ..analysis.persistence import event_anchor
    from ..explain.attribution import completeness_gap, integrated_gradients
    from ..explain.run import _find_run
    from ..explain.tgn_capture import ExplainableTGN, capture, load_trained
    from .predict import Calibrator, coin_risk, top_drivers

    ec = cfg["events"]
    target = _find_run(ec["target_experiment"], ec.get("target_seed", 0))
    tracker.log(f"events through {target.name}")
    runner, D = load_trained(target, device="cpu")
    d, coins, K, panel, gidx = D["d"], D["coins"], list(D["k_hours"]), D["panel"], D["graph_index"]
    times = pd.DatetimeIndex(panel["times"])
    t_np, src, dst = d.t.numpy(), d.src.numpy(), d.dst.numpy()
    sh = (panel["shock_today"] & panel["shock_today_mask"]).numpy()[:, gidx]
    act = panel["active"].numpy()[:, gidx]
    li = list(panel["feature_names"]).index("log_return")
    ret = panel["X_raw"][:, gidx, li].numpy().astype(float)
    split_id = panel["split_id"].numpy()

    catalogue = pd.read_csv(PROJECT_ROOT / "data" / "external" / "events.csv")
    sector_of = {}
    sp = PROJECT_ROOT / "data" / "external" / "sectors.csv"
    if sp.exists():
        sector_of = dict(pd.read_csv(sp)[["coin", "sector"]].values)
    events, skipped = [], []
    for i, e in enumerate(catalogue.itertuples()):
        t0 = event_anchor(sh, times, e.date)
        pos = np.flatnonzero(t_np == t0) if t0 is not None else np.zeros(0, int)
        if not len(pos):
            skipped.append(f"{e.name} ({e.date}): " + ("outside the hourly data" if t0 is None
                                                       else "no shock events at its anchor hour"))
            continue
        events.append({"id": i, "row": e, "t0": int(t0), "pos": pos})
    for s in skipped:
        tracker.log(f"skipped {s}")
    sel = np.zeros(len(t_np), bool)
    for ev in events:
        sel[ev["pos"]] = True
    tracker.log(f"{len(events)} events; capturing {int(sel.sum()):,} stream events in one replay")
    cap = capture(runner, sel, end=int(np.flatnonzero(sel)[-1]) + 1)
    ex = ExplainableTGN(runner.model)
    with torch.no_grad():
        logits = ex(cap)
    err = float((logits - cap.logits).abs().max())
    assert err < 1e-4, f"captured inputs do not reproduce the model (max abs logit diff {err:.2e})"
    prob = torch.sigmoid(logits).numpy()
    cal = Calibrator.from_run(target, ec.get("calibration_min_bin", 500))
    risk_all = cal(prob) if cal is not None else np.maximum.accumulate(prob, axis=1)
    cpos = cap.pos.numpy()
    where = {int(p): j for j, p in enumerate(cpos)}

    out_dir, index, gap_max, metrics = Path(tracker.dir), [], 0.0, {"target_run": target.name}
    for ev in events:
        e, t0 = ev["row"], ev["t0"]
        rows = np.array([where[int(p)] for p in ev["pos"]])
        table = coin_risk(prob[rows], risk_all[rows], src[cpos[rows]], dst[cpos[rows]], t_np[cpos[rows]], coins, K,
                          cal.base_rate if cal is not None else None)
        pos_of = cpos[rows][table["row"].to_numpy()]                     # stream position behind each coin x k
        kk = table["horizon_h"].map({kh: k for k, kh in enumerate(K)}).to_numpy()
        obs = d.y_mask[pos_of, kk].numpy()
        table["actual"] = np.where(obs, d.y[pos_of, kk].numpy().astype(float), np.nan)
        table["sector"] = table["coin"].map(sector_of).fillna("other")
        drivers = {}
        for k, kh in enumerate(K):
            sub = table.index[table["horizon_h"] == kh]
            idx = torch.from_numpy(rows[table.loc[sub, "row"].to_numpy()])
            attr, gap = integrated_gradients(ex, cap.subset(idx), k, steps=ec.get("ig_steps", 32))
            gap_max = max(gap_max, completeness_gap(attr, gap))
            for j, r in enumerate(sub):
                drivers[r] = top_drivers(attr, j, D["node_names"], D["msg_names"], ec.get("top_drivers", 3))
        table["drivers"] = [drivers[r] for r in table.index]
        split = SPLIT_NAMES.get(int(split_id[t0]), "between splits")
        replay = event_replay(sh, act, ret, times, coins, t0, ec.get("before_hours", 48), ec.get("after_hours", 168))
        score = model_vs_outcome(table, ec.get("top", 5))
        info = {"id": ev["id"], "name": e.name, "date": str(e.date), "event_type": str(e.event_type),
                "origin_coin": None if pd.isna(e.origin_coin) else str(e.origin_coin),
                "description": "" if pd.isna(e.description) else str(e.description),
                "source_url": "" if pd.isna(e.source_url) else str(e.source_url),
                "anchor_time": str(times[t0]), "anchor_bar": t0, "split": split, "split_note": SPLIT_NOTE[split],
                "source_coins": sorted({coins[s] for s in src[ev["pos"]]}), "replay_metrics": replay["metrics"]}
        payload = info | {"risk": table.drop(columns=["row"]).replace({np.nan: None}).to_dict("records"),
                          "model_vs_outcome": score.to_dict("records"), "replay": replay,
                          "data_version": D["cfg"]["data"]["version"], "resolution": D["cfg"]["data"]["resolution"]}
        (out_dir / f"event_{ev['id']}.json").write_text(json.dumps(payload, default=float), encoding="utf-8")
        index.append(info)
        h6 = score[score["horizon_h"] == 6] if len(score) else score
        hits = (f"riskiest {int(h6.iloc[0]['top_n'])} at 6 h: {int(h6.iloc[0]['top_hits'])} shocked "
                f"(chance {h6.iloc[0]['expected_by_chance']:.1f})" if len(h6) else "no observed outcomes")
        tracker.log(f"{e.name} [{split}] anchor {times[t0]}, sources {info['source_coins'][:4]}"
                    f"{'...' if len(info['source_coins']) > 4 else ''}; {hits}")
        if len(h6):
            metrics[f"event{ev['id']}_top_hits_k6h"] = int(h6.iloc[0]["top_hits"])
    (out_dir / "events.json").write_text(json.dumps({"target_run": target.name, "events": index,
                                                     "skipped": skipped, "k_hours": K}, default=float),
                                         encoding="utf-8")
    return metrics | {"n_events": len(index), "n_skipped": len(skipped), "rebuild_max_abs_logit_diff": err,
                      "ig_completeness_max_gap": gap_max}
