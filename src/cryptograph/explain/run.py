"""Explain a trained TGN run: TempME + baselines (event level) and integrated gradients (feature level).

Training events for the explainer come from the validation period; everything reported is on the test period
(the riskiest alerts and a random sample) and on three case-study windows. Outputs go to the run directory
(runs/<run_id>/) and, as small CSV/JSON tables, to reports/explain/ for notebook 04.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from ..data.config import PROJECT_ROOT
from ..data.storage import load_pt
from . import fidelity as fd
from .attribution import completeness_gap, feature_table, integrated_gradients
from .tempme import N_CAT, build_walks, null_distribution, tempme_importance, train_tempme
from .tgn_capture import ExplainableTGN, capture, load_trained

PERIOD_OF_SPLIT = {0: "train", 1: "validation", 2: "test"}


def case_windows(cases: list[dict], times: pd.DatetimeIndex, split_id: np.ndarray) -> dict[str, tuple[str, str]]:
    """Case-study windows from the experiment config (`explain.cases`: name, start, end; end exclusive), in config
    order. Each window must hold data, and a name that states its period ("(…, train period)") must match the
    period its hours actually fall in, so a label cannot go stale when the splits change."""
    import re

    out = {}
    for c in cases:
        lo, hi = pd.Timestamp(c["start"]), pd.Timestamp(c["end"])
        bars = np.flatnonzero((times >= lo) & (times < hi))
        if not len(bars):
            raise ValueError(f"case '{c['name']}' ({c['start']} to {c['end']}) has no hours in the data")
        periods = {PERIOD_OF_SPLIT.get(int(s), "between splits") for s in split_id[bars]}
        said = re.search(r"\b(train|validation|test) period\b", c["name"])
        if said and periods != {said.group(1)}:
            raise ValueError(f"case '{c['name']}' is labelled '{said.group(1)} period' but its hours fall in "
                             f"{sorted(periods)}")
        out[c["name"]] = (str(c["start"]), str(c["end"]))
    return out


def _find_run(experiment: str, seed: int, required: str = "checkpoint.pt") -> Path:
    """Latest finished run of `experiment` with this seed (exact experiment name; must contain `required`)."""
    runs = sorted(p for p in (PROJECT_ROOT / "runs").glob(f"*_{experiment}_{seed}_*")
                  if (p / required).exists() and p.name[16:].rsplit("_", 2)[0] == experiment)
    if not runs:
        raise FileNotFoundError(f"no finished run of {experiment} seed {seed}")
    return runs[-1]


def tempme_task(cfg: dict, seed: int, device: torch.device, tracker) -> dict[str, Any]:
    ec = cfg["explain"]
    rng = np.random.default_rng(seed)
    target = _find_run(ec["target_experiment"], ec.get("target_seed", 0))
    tracker.log(f"explaining {target.name}")
    runner, D = load_trained(target)
    d, panel, bounds, coins = D["d"], D["panel"], D["bounds"], D["coins"]
    K, kidx = D["k_hours"], D["k_hours"].index(ec.get("horizon_hours", 6))
    t_np, src, dst = d.t.numpy(), d.src.numpy(), d.dst.numpy()
    times = pd.DatetimeIndex(panel["times"])
    n_ev = len(t_np)

    # --- which events to capture -------------------------------------------------------------------------------
    pt = load_pt(target / "predictions_test.pt")
    pos_t, prob_t, m_t = pt["position"].numpy(), pt["prob"].float().numpy()[:, kidx], pt["mask"].numpy()[:, kidx]
    cand = np.flatnonzero(m_t)
    n_top = max(1, int(round(ec.get("top_frac", 0.01) * len(cand))))
    top = pos_t[cand[np.argsort(-prob_t[cand], kind="stable")[:n_top]]]
    top = np.sort(rng.choice(top, size=min(len(top), ec.get("max_alerts", 4000)), replace=False))
    rnd = np.sort(rng.choice(pos_t[cand], size=min(len(cand), ec.get("n_random_test", 4000)), replace=False))
    a, b = bounds["val"]
    vm = panel["sample_mask"]["val"].numpy()[t_np[a:b]] & d.y_mask[a:b, kidx].numpy()
    train_pos = np.sort(rng.choice(np.flatnonzero(vm) + a, size=min(int(vm.sum()), ec.get("n_train", 10000)),
                                   replace=False))
    case_pos = {}
    for name, (lo, hi) in case_windows(ec.get("cases", []), times, panel["split_id"].numpy()).items():
        bars = np.flatnonzero((times >= pd.Timestamp(lo)) & (times < pd.Timestamp(hi)))
        ev = np.flatnonzero(np.isin(t_np, bars))
        case_pos[name] = np.sort(rng.choice(ev, size=min(len(ev), ec.get("case_events", 1500)), replace=False))
    sel = np.zeros(n_ev, bool)
    for p in (top, rnd, train_pos, *case_pos.values()):
        sel[p] = True
    tracker.log(f"capturing {sel.sum():,} events: {len(train_pos):,} val (explainer training), {len(top):,} top-"
                f"{ec.get('top_frac', 0.01):.0%} test alerts at k={K[kidx]}h, {len(rnd):,} random test, cases "
                f"{[len(v) for v in case_pos.values()]}")
    cap_all = capture(runner, sel, end=bounds["test"][1])
    where = {int(p): i for i, p in enumerate(cap_all.pos.tolist())}
    pick = lambda ps: cap_all.subset(torch.tensor([where[int(p)] for p in ps]))
    ex = ExplainableTGN(runner.model)
    with torch.no_grad():
        err = float((ex(cap_all) - cap_all.logits).abs().max())
    tracker.log(f"rebuilt vs stream logits, max abs difference {err:.2e}")
    assert err < 1e-4, "captured inputs do not reproduce the model"

    def walks_for(cap, t_arr=t_np):
        cen = np.stack([src[cap.pos.numpy()], dst[cap.pos.numpy()]], 1)
        return build_walks(cap.pos_n.numpy(), cap.nbr.numpy(), cen, src, dst, t_arr, len(coins),
                           ec.get("R2", 3), ec.get("R3", 2))

    cap_tr = pick(train_pos)
    w_tr = walks_for(cap_tr)
    t_shuf = t_np.copy()
    rng.shuffle(t_shuf)
    null_idx = torch.from_numpy(np.sort(rng.choice(len(cap_tr), size=min(3000, len(cap_tr)), replace=False)))
    w_null = walks_for(cap_tr.subset(null_idx), t_shuf)
    null = null_distribution(w_null)
    real = null_distribution(w_tr)
    msg_all, t_all, feats = d.msg.float(), d.t, D["feats"]
    tc = ec.get("tempme", {})
    explainer, hist = train_tempme(ex, cap_tr, w_tr, null, t_all, msg_all, feats, seed=seed, log=tracker.log, **tc)
    torch.save({"state": explainer.state_dict(), "null": null, "target_run": target.name, "config": tc},
               tracker.dir / "tempme_explainer.pt")

    # --- event-level faithfulness on test ------------------------------------------------------------------------
    out_dir = PROJECT_ROOT / "reports" / "explain"
    out_dir.mkdir(parents=True, exist_ok=True)
    fid_rows, credit_rows, metrics = [], [], {"target_run": target.name, "rebuild_max_abs_logit_diff": err}
    sets = {"top alerts": pick(top), "random test events": pick(rnd)}
    imps = {}
    for set_name, cap in sets.items():
        w = walks_for(cap)
        im = {"TempME": tempme_importance(explainer, w, cap, t_all, msg_all, feats),
              "random": fd.random_importance(cap, seed),
              "recency": fd.recency_importance(cap, t_all),
              "TGN attention": fd.attention_importance(ex, cap),
              "optimised mask (per alert)": fd.mask_importance(ex, cap, steps=ec.get("mask_steps", 150), seed=seed)}
        imps[set_name] = im
        for name, imp in im.items():
            for r in fd.fidelity_curve(ex, cap, imp):
                fid_rows.append({"events": set_name, "explainer": name, **r})
        tracker.log(f"{set_name}: fidelity done")
    fid = pd.DataFrame(fid_rows)
    fid.to_csv(out_dir / "fidelity.csv", index=False)
    for set_name in sets:
        for name in ("TempME", "optimised mask (per alert)", "random"):
            r = fid[(fid["events"] == set_name) & (fid["explainer"] == name) & (fid["kept_events"] == 3)].iloc[0]
            metrics[f"{set_name.split()[0]}_{name.split()[0].lower()}_char_k3"] = float(r["characterization"])

    # attention spread, motif categories
    att = imps["top alerts"]["TGN attention"]
    valid = sets["top alerts"].mask
    n_valid = valid.sum(-1).clamp_min(1).float()
    ent = (-(att * torch.log(att + 1e-12)).sum(-1) / torch.log(n_valid.clamp_min(2)))
    metrics["attention_norm_entropy"] = float(ent[valid.any(-1)].mean())
    metrics["attention_mean_max"] = float(att.amax(-1)[valid.any(-1)].mean())
    pd.DataFrame({"category": range(N_CAT), "real": real.numpy(), "null (time-shuffled)": null.numpy()}) \
        .to_csv(out_dir / "motif_categories.csv", index=False)

    # plausibility: which coins the explanations point to (top alerts) vs network rankings
    ranking_file = PROJECT_ROOT / "data" / "processed" / D["cfg"]["data"]["version"] / "rankings.csv"
    rank = pd.read_csv(ranking_file).set_index("coin")
    from scipy.stats import spearmanr
    for name in ("TempME", "optimised mask (per alert)", "TGN attention"):
        credit = fd.coin_credit(imps["top alerts"][name], sets["top alerts"], len(coins))
        exposure = fd.coin_credit(torch.ones_like(imps["top alerts"][name]), sets["top alerts"], len(coins), top=20)
        share = pd.Series(credit / credit.sum(), index=coins)
        lift = pd.Series((credit / credit.sum()) / np.maximum(exposure / exposure.sum(), 1e-12), index=coins)
        for col in ("shock_pagerank", "dy_net", "dy_pagerank", "eigenvector", "shock_out_lift"):
            c = rank.loc[coins, col]
            credit_rows.append({"explainer": name, "ranking": col,
                                "spearman_share": spearmanr(share, c).statistic,
                                "spearman_lift_over_exposure": spearmanr(lift, c).statistic})
        if name == "TempME":
            pd.DataFrame({"coin": coins, "credit_share": share.values, "lift_over_exposure": lift.values}) \
                .to_csv(out_dir / "coin_credit_tempme.csv", index=False)
    pd.DataFrame(credit_rows).to_csv(out_dir / "plausibility.csv", index=False)

    # --- feature level: integrated gradients ----------------------------------------------------------------------
    node_names, msg_names = D["node_names"], D["msg_names"]        # from the target run's own data (load_trained)
    assert len(node_names) == feats.shape[-1]
    ig_tabs = []
    for set_name, cap in {**sets, **{f"case: {k}": pick(v) for k, v in case_pos.items()}}.items():
        attr, gap = integrated_gradients(ex, cap, kidx, steps=64)
        metrics.setdefault("ig_completeness_max_gap", 0.0)
        metrics["ig_completeness_max_gap"] = max(metrics["ig_completeness_max_gap"], completeness_gap(attr, gap))
        tab = feature_table(attr, node_names, msg_names)
        tab.insert(0, "events", set_name)
        ig_tabs.append(tab)
    pd.concat(ig_tabs).to_csv(out_dir / "feature_attributions.csv", index=False)

    # --- case studies: the riskiest events of each window, their top recent events and top features ----------------
    cases = []
    for name, ps in case_pos.items():
        cap = pick(ps)
        with torch.no_grad():
            p = torch.sigmoid(ex(cap))[:, kidx]
        order = torch.argsort(p, descending=True)[: ec.get("case_top", 5)]
        c5 = cap.subset(order)
        imp = tempme_importance(explainer, walks_for(c5), c5, t_all, msg_all, feats)
        attr, _ = integrated_gradients(ex, c5, kidx, steps=64)
        for r in range(len(c5)):
            pp = int(c5.pos[r])
            flat = imp[r].reshape(-1)
            best = torch.argsort(flat, descending=True)[:3]
            evs = []
            for f in best.tolist():
                side, l = divmod(f, imp.shape[-1])
                q = int(c5.pos_n[r, side, l])
                if q >= 0:
                    evs.append({"side": "source" if side == 0 else "receiver", "time": str(times[t_np[q]]),
                                "event": f"{coins[src[q]]} -> {coins[dst[q]]}", "importance": float(flat[f])})
            one = feature_table([x[r:r + 1] for x in attr], node_names, msg_names)
            cases.append({"case": name, "time": str(times[t_np[pp]]), "event": f"{coins[src[pp]]} -> {coins[dst[pp]]}",
                          "p": float(p[order[r]]), "label": int(d.y[pp, kidx]) if bool(d.y_mask[pp, kidx]) else None,
                          "top_recent_events": evs,
                          "top_features": one.head(5)[["input", "mean_attribution"]].to_dict("records")})
    (out_dir / "case_studies.json").write_text(json.dumps(cases, indent=1), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(metrics | {"history": hist}, indent=1), encoding="utf-8")
    metrics["history"] = hist
    return metrics
