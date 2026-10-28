"""Fair re-reporting of the existing T1 results from their saved predictions (no retraining, no new model choices).

For every final model and split it reports, per horizon:
  * event level (what notebook 03 reports): AUPRC, random level (= positive rate), lift = AUPRC / random, AUROC;
  * receiver-hour level: the ~3 events per (hour, receiver) share one label, so they are collapsed to one row
    (mean probability) before scoring;
  * the "calm receiver" slice: receivers with no own shock in the 24 hours up to t (the real contagion case);
  * precision of the top 1% receiver-hours.
Writes reports/fair_eval_v2.csv (seed means) next to the run predictions it read.

Usage: python scripts/fair_report.py --runs ../CryptoGraph/runs
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import torch  # noqa: F401  (torch before pandas on Windows)
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from cryptograph.analysis.evaluation import precision_at_top
from cryptograph.data.config import PROJECT_ROOT
from cryptograph.data.storage import load_pt
from cryptograph.training.tgn_task import context_tables

MODELS = ["tgn_1h_final_v2", "tgn_1h_nomem_final_v3", "dygformer_1h_final", "gbm_1h_final"]


def score_split(pred: dict, since: np.ndarray) -> list[dict]:
    t, dst = pred["t"].numpy(), pred["dst"].numpy()
    prob, y, m = pred["prob"].float().numpy(), pred["y"].numpy(), pred["mask"].numpy()
    rows = []
    for k, kh in enumerate(pred["k_hours"]):
        mk = m[:, k]
        yk, pk = y[mk, k], prob[mk, k]
        df = pd.DataFrame({"t": t[mk], "j": dst[mk], "y": yk, "p": pk})
        g = df.groupby(["t", "j"]).agg(y=("y", "first"), p=("p", "mean")).reset_index()
        calm = since[g.t.to_numpy(), g.j.to_numpy()] >= 24
        top = precision_at_top(g.y.to_numpy(), g.p.to_numpy(), fracs=(0.01,))
        r = {"k_hours": kh,
             "event_auprc": average_precision_score(yk, pk), "event_random": yk.mean(),
             "event_auroc": roc_auc_score(yk, pk),
             "rh_auprc": average_precision_score(g.y, g.p), "rh_random": g.y.mean(),
             "rh_auroc": roc_auc_score(g.y, g.p),
             "calm_auprc": average_precision_score(g.y[calm], g.p[calm]), "calm_random": g.y[calm].mean(),
             "calm_share": calm.mean(), "top1_precision": top["precision_top1%"], "n_receiver_hours": len(g)}
        for lvl in ("event", "rh", "calm"):
            r[f"{lvl}_lift"] = r[f"{lvl}_auprc"] / r[f"{lvl}_random"]
        rows.append(r)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(PROJECT_ROOT / "runs"))
    ap.add_argument("--out", default=str(PROJECT_ROOT / "reports" / "fair_eval_v2.csv"))
    ap.add_argument("--models", nargs="+", default=MODELS)
    args = ap.parse_args()
    base = PROJECT_ROOT / "data" / "processed" / "v2" / "1h"
    panel = load_pt(base / "panel.pt")
    gidx = load_pt(base / "tgn_events.pt")["graph_index"]
    node, _ = context_tables(panel, gidx)
    since = np.rint(np.expm1(node[:, :, 3].numpy() * np.log(169)))     # hours since the receiver's own last shock
    rows = []
    for model in args.models:
        for f in sorted(glob.glob(str(Path(args.runs) / f"*_{model}_*" / "predictions_*.pt"))):
            split = Path(f).stem.split("_")[-1]
            for r in score_split(load_pt(f), since):
                rows.append({"model": model, "split": split, "run": Path(f).parent.name, **r})
    df = pd.DataFrame(rows)
    out = df.drop(columns="run").groupby(["model", "split", "k_hours"]).mean().reset_index()
    out["n_seeds"] = df.groupby(["model", "split", "k_hours"]).size().to_numpy()
    out.to_csv(args.out, index=False)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    cols = ["model", "split", "k_hours", "event_auprc", "event_random", "event_lift", "event_auroc", "rh_auprc",
            "rh_random", "rh_lift", "calm_auprc", "calm_random", "calm_lift", "top1_precision", "n_seeds"]
    print(out[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
