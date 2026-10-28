"""How well the served model did on the test period, in the terms the dashboard shows (serving/predict.py):

* top alerts: of the riskiest 1% of coin-hours (a coin's risk = the highest calibrated probability over the shock
  events pointing at it in that hour, as served), how many made a big move within k hours, against the share of all
  scored coin-hours that did (a random pick);
* calibration: the same served risk in equal-count bins against how often those coins then made a big move.

Computed from the served run's own predictions_test.pt, with the calibrator fitted on its predictions_val.pt: nothing
is fitted on the test period. Coin-hours, not events, because the label only depends on the receiving coin and the
hour, so an hour with many shocking coins would otherwise count each receiver once per source.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def coin_hours(prob: np.ndarray, risk: np.ndarray, y: np.ndarray, mask: np.ndarray, t: np.ndarray, dst: np.ndarray,
               k: int) -> pd.DataFrame:
    """One row per (hour, receiver) at horizon index k: the highest served risk and raw score, and the outcome
    (the same for every event of that receiver and hour)."""
    m = mask[:, k].astype(bool)
    n = int(dst.max()) + 1 if len(dst) else 1
    uniq, inv = np.unique(t[m].astype(np.int64) * n + dst[m], return_inverse=True)
    out = {}
    for name, v in (("risk", risk[m, k]), ("score", prob[m, k]), ("y", y[m, k])):
        agg = np.full(len(uniq), -np.inf)
        np.maximum.at(agg, inv, v.astype(np.float64))
        out[name] = agg
    return pd.DataFrame(out, index=pd.MultiIndex.from_arrays([uniq // n, uniq % n], names=["t", "coin"]))


def evidence_on_test(run_dir: Path | str, top_frac: float = 0.01, bins: int = 10) -> dict:
    """{"precision": per horizon, "calibration": per horizon and bin, "run": name}; {} without test predictions."""
    from ..data.storage import load_pt
    from .predict import Calibrator

    run_dir = Path(run_dir)
    p = run_dir / "predictions_test.pt"
    if not p.exists():
        return {}
    te = load_pt(p)
    prob, y, mask = te["prob"].float().numpy(), te["y"].numpy(), te["mask"].numpy()
    t, dst, K = te["t"].numpy(), te["dst"].numpy(), [int(k) for k in te["k_hours"]]
    cal = Calibrator.from_run(run_dir)
    risk = cal(prob) if cal is not None else np.maximum.accumulate(prob, axis=1)
    prec, rel = [], []
    for k, kh in enumerate(K):
        ch = coin_hours(prob, risk, y, mask, t, dst, k).sort_values(["risk", "score"], ascending=False)
        n = max(1, int(round(top_frac * len(ch))))
        prec.append({"horizon_h": kh, "coin_hours": int(len(ch)), "positive_rate": float(ch["y"].mean()),
                     "top_n": n, "top_precision": float(ch["y"].iloc[:n].mean()),
                     "mean_risk_top": float(ch["risk"].iloc[:n].mean())})
        q = pd.qcut(ch["risk"].rank(method="first"), bins, labels=False)
        g = ch.groupby(q).agg(predicted=("risk", "mean"), observed=("y", "mean"), n=("y", "size")).reset_index(drop=True)
        rel += [{"horizon_h": kh, "bin": b, **r} for b, r in enumerate(g.to_dict("records"))]
    return {"precision": pd.DataFrame(prec), "calibration": pd.DataFrame(rel), "run": run_dir.name,
            "calibrated": cal is not None}
