"""Go / no-go for the 5-minute data: does it raise the 1h contagion score on VALIDATION? (Test is not touched.)

Rows are the TGN's events collapsed to one per (hour t, receiver j), weighted by the number of shocking sources, so
the weighted AUPRC equals the event-level AUPRC that notebook 03 reports. Logistic regression and gradient boosting
are fitted on train with fixed settings (no tuning on validation), on three input sets:
  v2      the v2 node features of the receiver + event context (what the TGN head sees, minus the graph)
  v3-1h   v3 hourly features (volatility ratios, current-hour range, trades) + context + calendar + market state
  v3-5m   v3-1h + the 5-minute features (and their market-wide means)
Writes reports/go_nogo_5m.csv.

--final scores the TEST period once with the frozen recipes (same fits on train; no choice is made on test) and writes
reports/final_1h_test.csv.
"""

from __future__ import annotations

import argparse

import torch  # noqa: F401  (torch before pandas on Windows)
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score as ap
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cryptograph.data.config import PROJECT_ROOT
from cryptograph.data.features.subhour import SUBHOUR_NAMES
from cryptograph.data.features.targets import future_any
from cryptograph.data.storage import load_pt
from cryptograph.training.tgn_task import calendar_table, context_tables

KS = [1, 3, 6, 12, 24]


def main() -> None:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--final", action="store_true")
    final = ap_.parse_args().final
    ev = "test" if final else "val"
    p2 = load_pt(PROJECT_ROOT / "data/processed/v2/1h/panel.pt")
    p3 = load_pt(PROJECT_ROOT / "data/processed/v3/1h/panel.pt")
    gidx = load_pt(PROJECT_ROOT / "data/processed/v3/1h/tgn_events.pt")["graph_index"]
    assert np.array_equal(p2["shock_today"].numpy(), p3["shock_today"].numpy()), "v3 must keep the v2 labels"
    act = p3["active"].numpy()[:, gidx]
    sh = (p3["shock_today"].numpy() & p3["shock_today_mask"].numpy())[:, gidx]
    sv = p3["shock_today_mask"].numpy()[:, gidx]
    smask = {s: p3["sample_mask"][s].numpy() for s in ("train", ev)}
    node, glob_ = (a.numpy() for a in context_tables(p3, gidx))
    cal = calendar_table(p3).numpy()
    X2, X3 = p2["X"].numpy()[:, gidx], p3["X"].numpy()[:, gidx]
    names3 = p3["feature_names"]
    i5 = [i for i, n in enumerate(names3) if n in SUBHOUR_NAMES]
    i1 = [i for i in range(len(names3)) if i not in i5]
    mkt = lambda cols: np.stack([np.where(act, X3[:, :, c], np.nan) for c in cols], -1)  # noqa: E731
    mkt1 = np.nanmean(mkt([names3.index(n) for n in ("lr_rvs_6h", "lr_parkinson_1h")]), axis=1)
    mkt5 = np.nanmean(mkt([names3.index(n) for n in ("rv5_ratio", "last15", "last15_abs")]), axis=1)
    z = np.nan_to_num(p3["shock_z"].numpy()[:, gidx])
    mz = np.stack([np.abs(z).mean(1), z.mean(1), z[:, 0]], 1)              # market |z|, market z, BTC z

    n_src = sh.sum(1)
    tt, jj = np.nonzero(act & (n_src[:, None] - sh > 0))
    w = (n_src[tt] - sh[tt, jj]).astype(float)
    ctx = np.column_stack([node[tt, jj], glob_[tt]])
    sets = {"v2": np.column_stack([X2[tt, jj], ctx]),
            "v3-1h": np.column_stack([X3[tt, jj][:, i1], ctx, cal[tt], mkt1[tt], mz[tt]]),
            "v3-5m": np.column_stack([X3[tt, jj], ctx, cal[tt], mkt1[tt], mz[tt], mkt5[tt]])}
    fut = {k: future_any(sh, sv, k) for k in KS}
    calm = node[tt, jj, 3] * np.log(169) >= np.log(25) - 1e-6           # no own shock in the last 24 h
    rows = []
    if final:                                                          # frozen: v3-5m boosting + the v2 baseline
        sets = {k: sets[k] for k in ("v2", "v3-5m")}
    for name, X in sets.items():
        X = np.where(np.isfinite(X), X, np.nan).astype(np.float32)
        for model in ("logreg", "gbm"):
            for k in KS:
                y, ym = fut[k][0][tt, jj], fut[k][1][tt, jj]
                tr, va = smask["train"][tt] & ym, smask[ev][tt] & ym
                if model == "logreg":
                    clf = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=3000))
                    clf.fit(X[tr], y[tr], logisticregression__sample_weight=w[tr])
                else:
                    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                                         min_samples_leaf=2000, l2_regularization=1.0, random_state=0)
                    clf.fit(X[tr], y[tr], sample_weight=w[tr])
                pv, yv, c = clf.predict_proba(X[va])[:, 1], y[va], calm[va]
                rows.append({"inputs": name, "model": model, "k_hours": k,
                             "event_auprc": ap(yv, pv, sample_weight=w[va]), "random": np.average(yv, weights=w[va]),
                             "rh_auprc": ap(yv, pv), "rh_random": yv.mean(),
                             "calm_auprc": ap(yv[c], pv[c]), "calm_random": yv[c].mean()})
                print(rows[-1], flush=True)
    out = pd.DataFrame(rows)
    out["lift"] = out.event_auprc / out.random
    out.to_csv(PROJECT_ROOT / "reports" / ("final_1h_test.csv" if final else "go_nogo_5m.csv"), index=False)
    print(out.pivot_table(index=["inputs", "model"], columns="k_hours", values="event_auprc").round(3).to_string())


if __name__ == "__main__":
    main()
