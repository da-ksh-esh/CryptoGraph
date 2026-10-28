"""Hour-by-hour replay of chosen windows through the final TGN, for the dashboard's "Replay" tab: what the model said
at every hour with shock events, as it would have said it live, and what then happened.

The whole stream is replayed once (memory holds only earlier events, exactly as in serving/predict.py) and every event
inside a window is scored and calibrated on the validation predictions. Per hour with shock events and per horizon k,
each receiver coin gets its risk (the highest over the events pointing at it), its lift over the base rate, the source
behind it, the realised outcome (the stream's own label: a big move within k hours), the hours until its first big
move and whether it was already moving at that hour; the `top` coins of each hour and horizon also get their
integrated-gradient drivers. Every hour of the window, calm or not, records which coins moved, which way and how much.

Only test-period windows are genuine forecasts (train = learned from, validation = used for selection and
calibration); each window records its period. Outputs (runs/<run_id>/): replay.json (index) and replay_<id>.json.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from ..data.config import PROJECT_ROOT
from .events import SPLIT_NAMES, SPLIT_NOTE

FIRST_MOVE_HORIZON = 24                  # hours searched for a coin's first big move after an alert
PERIOD_NOTE = SPLIT_NOTE | {
    "after the research period": "live data after the research period: never used for training, selection or "
                                 "calibration; a genuine forecast",
    "mixed": "spans several periods: only its test-period and later hours are genuine forecasts"}


def window_bars(times: pd.DatetimeIndex, start, end) -> tuple[int, int]:
    """[first, last] bar of a window given in UTC hours, clipped to the data. Raises if it lies outside the data."""
    a = int(np.searchsorted(times, pd.Timestamp(start), side="left"))
    b = int(np.searchsorted(times, pd.Timestamp(end), side="right")) - 1
    if a > b or a >= len(times) or b < 0:
        raise ValueError(f"window {start} .. {end} is outside the data ({times[0]} .. {times[-1]})")
    return a, b


def window_split(split_id: np.ndarray, a: int, b: int) -> str:
    """The period of a window: train / validation / test, "after the research period" for live bars, or mixed."""
    names = {SPLIT_NAMES.get(int(s), "after the research period") for s in np.unique(split_id[a:b + 1])}
    return names.pop() if len(names) == 1 else "mixed"


def first_move_after(shock: np.ndarray, t: np.ndarray, coin: np.ndarray, horizon: int = FIRST_MOVE_HORIZON
                     ) -> np.ndarray:
    """Hours from bar t to the coin's first big move in (t, t + horizon]; NaN if none there (or the data ends)."""
    T = shock.shape[0]
    out = np.full(len(t), np.nan)
    for n, (tt, c) in enumerate(zip(t, coin)):
        hits = np.flatnonzero(shock[tt + 1: min(tt + horizon, T - 1) + 1, c])
        if len(hits):
            out[n] = hits[0] + 1
    return out


def hour_frames(shock: np.ndarray, active: np.ndarray, z: np.ndarray, ret: np.ndarray, times: pd.DatetimeIndex,
                coins: list[str], a: int, b: int, scored: set[int]) -> list[dict]:
    """Per hour in [a, b]: the coins making a big move (direction, |z|, return in %), active coins, and whether the
    model scored that hour (it scores only hours with shock events)."""
    out = []
    for t in range(a, b + 1):
        moves = [{"coin": coins[c], "direction": "crash" if ret[t, c] < 0 else "pump",
                  "z": round(float(z[t, c]), 2) if np.isfinite(z[t, c]) else None,
                  "ret_pct": round(float(ret[t, c]) * 100, 2) if np.isfinite(ret[t, c]) else None}
                 for c in np.flatnonzero(shock[t] & active[t])]
        out.append({"bar": t, "time": str(times[t]), "shocks": moves, "n_active": int(active[t].sum()),
                    "scored": t in scored})
    return out


def top_rows(table: pd.DataFrame, top: int) -> pd.DataFrame:
    """The `top` coins per hour and horizon by lift (risk breaks ties; risk alone without base rates)."""
    key = ["lift", "risk"] if "lift" in table and table["lift"].notna().any() else ["risk", "score"]
    return (table.sort_values(key, ascending=False).groupby(["bar", "horizon_h"], sort=False).head(top))


def replay_task(cfg: dict, seed: int, device: torch.device, tracker) -> dict[str, Any]:
    from ..explain.attribution import completeness_gap, integrated_gradients
    from ..explain.run import _find_run
    from ..explain.tgn_capture import ExplainableTGN, capture, load_trained
    from .predict import Calibrator, coin_risk, top_drivers

    rc = cfg["replay"]
    target = _find_run(rc["target_experiment"], rc.get("target_seed", 0))
    runner, D = load_trained(target, device="cpu", data_version=rc.get("data_version"))
    tracker.log(f"replay through {target.name} on data {D['cfg']['data']['version']}")
    d, coins, K, panel, gidx = D["d"], D["coins"], list(D["k_hours"]), D["panel"], D["graph_index"]
    times = pd.DatetimeIndex(panel["times"])
    t_np, src, dst = d.t.numpy(), d.src.numpy(), d.dst.numpy()
    sh = (panel["shock_today"] & panel["shock_today_mask"]).numpy()[:, gidx]
    act = panel["active"].numpy()[:, gidx]
    z = panel["shock_z"].numpy()[:, gidx]
    li = list(panel["feature_names"]).index("log_return")
    ret = panel["X_raw"][:, gidx, li].numpy().astype(float)
    split_id = panel["split_id"].numpy()
    sector_of = {}
    sp = PROJECT_ROOT / "data" / "external" / "sectors.csv"
    if sp.exists():
        sector_of = dict(pd.read_csv(sp)[["coin", "sector"]].values)
    catalogue = pd.DataFrame()
    cp = PROJECT_ROOT / "data" / "external" / "events.csv"
    if cp.exists():
        catalogue = pd.read_csv(cp).set_index("name")

    windows = []
    for i, w in enumerate(rc["windows"]):
        a, b = window_bars(times, w["start"], w["end"])
        pos = np.flatnonzero((t_np >= a) & (t_np <= b))
        ev = catalogue.loc[w["event"]] if w.get("event") and w["event"] in catalogue.index else None
        windows.append({"id": i, "w": w, "a": a, "b": b, "pos": pos, "event": ev})
        tracker.log(f"window {w['name']}: {times[a]} .. {times[b]} ({b - a + 1} h, {len(pos):,} events)")
    sel = np.zeros(len(t_np), bool)
    for w in windows:
        sel[w["pos"]] = True
    if not sel.any():
        raise ValueError("no shock events inside any replay window")
    cap = capture(runner, sel, end=int(np.flatnonzero(sel)[-1]) + 1)
    ex = ExplainableTGN(runner.model)
    with torch.no_grad():
        logits = ex(cap)
    err = float((logits - cap.logits).abs().max())
    assert err < 1e-4, f"captured inputs do not reproduce the model (max abs logit diff {err:.2e})"
    prob = torch.sigmoid(logits).numpy()
    cal = Calibrator.from_run(target)
    risk_all = cal(prob) if cal is not None else np.maximum.accumulate(prob, axis=1)
    cpos = cap.pos.numpy()
    where = {int(p): j for j, p in enumerate(cpos)}

    out_dir, index, metrics, gap_max = Path(tracker.dir), [], {"target_run": target.name}, 0.0
    top, n_drv = int(rc.get("top", 5)), int(rc.get("top_drivers", 3))
    for w in windows:
        tables = []
        for tt in np.unique(t_np[w["pos"]]):
            rows = np.array([where[int(p)] for p in w["pos"][t_np[w["pos"]] == tt]])
            tab = coin_risk(prob[rows], risk_all[rows], src[cpos[rows]], dst[cpos[rows]], t_np[cpos[rows]], coins, K,
                            cal.base_rate if cal is not None else None)
            tab["cap_row"] = rows[tab["row"].to_numpy()]
            tables.append(tab)
        table = pd.concat(tables, ignore_index=True).drop(columns=["row", "n_events"])
        table = table.rename(columns={"event_bar": "bar"})
        p_of = cpos[table["cap_row"].to_numpy()]
        kk = table["horizon_h"].map({kh: k for k, kh in enumerate(K)}).to_numpy()
        obs = d.y_mask[p_of, kk].numpy()
        table["actual"] = np.where(obs, d.y[p_of, kk].numpy().astype(float), np.nan)
        ci = table["coin"].map({c: j for j, c in enumerate(coins)}).to_numpy()
        table["first_move_h"] = first_move_after(sh, table["bar"].to_numpy(), ci)
        table["shocking_now"] = sh[table["bar"].to_numpy(), ci]
        table["time"] = [str(times[t]) for t in table["bar"]]
        drivers, best = {}, top_rows(table, top)
        for k, kh in enumerate(K):
            sub = best[best["horizon_h"] == kh]
            if not len(sub):
                continue
            attr, gap = integrated_gradients(ex, cap.subset(torch.from_numpy(sub["cap_row"].to_numpy())), k,
                                             steps=rc.get("ig_steps", 32))
            gap_max = max(gap_max, completeness_gap(attr, gap))
            for j, r in enumerate(sub.index):
                drivers[r] = top_drivers(attr, j, D["node_names"], D["msg_names"], n_drv)
        table["drivers"] = [drivers.get(r, []) for r in table.index]
        table = table.drop(columns=["cap_row"])
        frames = hour_frames(sh, act, z, ret, times, coins, w["a"], w["b"], set(int(x) for x in table["bar"]))
        split = window_split(split_id, w["a"], w["b"])
        ev, spec = w["event"], w["w"]
        info = {"id": w["id"], "name": spec["name"], "start": str(times[w["a"]]), "end": str(times[w["b"]]),
                "hours": w["b"] - w["a"] + 1, "hours_scored": int(table["bar"].nunique()), "split": split,
                "split_note": PERIOD_NOTE[split],
                "description": spec.get("description") or ("" if ev is None or pd.isna(ev["description"])
                                                           else str(ev["description"])),
                "source_url": spec.get("source_url") or ("" if ev is None or pd.isna(ev["source_url"])
                                                         else str(ev["source_url"])),
                "events": int(len(w["pos"]))}
        score = window_score(table, K, min(3, top))
        payload = info | {"k_hours": K, "coins": coins, "sectors": {c: sector_of.get(c, "other") for c in coins},
                          "base_rate": cal.base_rate if cal is not None else None, "frames": frames,
                          "risk": table.replace({np.nan: None}).to_dict("records"), "score": score,
                          "data_version": D["cfg"]["data"]["version"], "resolution": D["cfg"]["data"]["resolution"]}
        (out_dir / f"replay_{w['id']}.json").write_text(json.dumps(payload, default=float), encoding="utf-8")
        index.append(info | {"score": score})
        s6 = next((s for s in score if s["horizon_h"] == 6), None)
        if s6:
            metrics[f"window{w['id']}_top3_hits_k6h"] = s6["hits"]
            metrics[f"window{w['id']}_top3_chance_k6h"] = s6["expected_by_chance"]
            tracker.log(f"{spec['name']} [{split}]: {info['hours_scored']} of {info['hours']} hours scored; top 3 per "
                        f"hour at 6 h: {s6['hits']} of {s6['alerts']} came true (chance {s6['expected_by_chance']:.1f})")
    (out_dir / "replay.json").write_text(json.dumps({"target_run": target.name, "windows": index, "k_hours": K},
                                                    default=float), encoding="utf-8")
    return metrics | {"n_windows": len(index), "rebuild_max_abs_logit_diff": err, "ig_completeness_max_gap": gap_max}


def window_score(table: pd.DataFrame, k_hours: list[int], top: int, skip_moving: bool = False) -> list[dict]:
    """Per horizon over a window: of the `top` coins flagged at each scored hour (with an observed outcome), how many
    made a big move within k hours, against `top` x the share of all such coins that did (random picks)."""
    t = table[table["actual"].notna()]
    if skip_moving:
        t = t[~t["shocking_now"].astype(bool)]
    out = []
    for kh in k_hours:
        g = t[t["horizon_h"] == kh]
        if not len(g):
            continue
        picks = top_rows(g, top)
        share = g.groupby("bar")["actual"].mean()
        n = picks.groupby("bar").size()
        out.append({"horizon_h": int(kh), "hours": int(g["bar"].nunique()), "alerts": int(len(picks)),
                    "hits": int(picks["actual"].sum()), "expected_by_chance": float((n * share.reindex(n.index)).sum())})
    return out
