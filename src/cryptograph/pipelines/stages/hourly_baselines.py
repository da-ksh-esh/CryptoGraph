"""Hourly naive baselines for T2 (next-bar shock) and T3 (forward realized volatility).

Evaluated on exactly the LSTM sample set (lstm_index_<split>.pt: model coins, full input window, targets present),
so every model in notebook 03 is compared on the same (t, coin) pairs. Anything fitted (train means, HAR, the
logistic rule) uses train samples only. Writes reports/hourly_baselines_<res>.csv. T1 (contagion) baselines come
from the `network` stage (reports/cascade_baseline.csv).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from ...analysis.baselines import har_design, har_fit, har_predict, trailing_rv
from ...common.logger import get_logger
from ...data.config import HourlyConfig
from ...data.features.intraday import bars
from ...data.storage import write_csv

logger = get_logger(__name__)
SPLITS = ("train", "val", "test")


def _cls(y: np.ndarray, s: np.ndarray) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score

    ok = np.isfinite(s)
    y, s = y[ok].astype(int), s[ok]
    return {"n": int(ok.sum()), "base_rate": float(y.mean()),
            "AUROC": float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else np.nan,
            "AUPRC": float(average_precision_score(y, s)) if y.sum() else np.nan}


def _reg(y: np.ndarray, p: np.ndarray) -> dict:
    ok = np.isfinite(y) & np.isfinite(p)
    e = p[ok] - y[ok]
    return {"n": int(ok.sum()), "MSE": float(np.mean(e ** 2)), "MAE": float(np.mean(np.abs(e)))}


def build_resolution(cfg: HourlyConfig, res: str) -> pd.DataFrame:
    from sklearn.linear_model import LogisticRegression

    d = cfg.processed_dir(res)
    p = torch.load(d / "panel.pt", weights_only=False)
    bh = p["bar_hours"]
    idx = {s: torch.load(d / f"lstm_index_{s}.pt", weights_only=False)["index"].numpy() for s in SPLITS}
    r = p["X_raw"][:, :, p["feature_names"].index("log_return")].numpy().astype(np.float64)
    y_rv, z = p["y_rv"].numpy(), p["shock_z"].numpy()
    y_sh, y_sh_m = p["y_shock"].numpy(), p["y_shock_mask"].numpy()
    s_today = p["shock_today"].numpy() & p["shock_today_mask"].numpy()
    n_coins = len(p["coins"])
    mi = [p["coins"].index(c) for c in p["model_coins"]]
    hz = p["horizons_hours"]

    # T2 scores at (t, coin), all known at t
    abs_z = np.abs(np.nan_to_num(z))
    own_24 = pd.DataFrame(s_today.astype(float)).rolling(bars(24, bh), min_periods=1).sum().to_numpy()
    breadth = np.repeat(s_today[:, mi].sum(axis=1, keepdims=True), n_coins, axis=1).astype(float)
    feats = np.stack([abs_z, s_today.astype(float), own_24, breadth], axis=-1)
    t_tr, c_tr = idx["train"][:, 0], idx["train"][:, 1]
    m_tr = y_sh_m[t_tr, c_tr]
    logit = LogisticRegression(max_iter=1000).fit(feats[t_tr, c_tr][m_tr], y_sh[t_tr, c_tr][m_tr])
    pooled = float(y_sh[t_tr, c_tr][m_tr].mean())
    coin_rate = np.full(n_coins, pooled)
    for c in np.unique(c_tr):
        sel = (c_tr == c) & m_tr
        coin_rate[c] = y_sh[t_tr[sel], c].mean()

    # T3: HAR and train means per horizon (train samples only)
    har_X = har_design(r, (1, bars(24, bh), bars(168, bh)))
    betas, means = {}, {}
    for k, h in enumerate(hz):
        ytr = y_rv[t_tr, c_tr, k]
        betas[h] = har_fit(har_X[t_tr, c_tr], ytr)
        means[h] = np.full(n_coins, np.nanmean(ytr))
        for c in np.unique(c_tr):
            means[h][c] = np.nanmean(ytr[c_tr == c])
    trail = {h: trailing_rv(r, bars(h, bh)) for h in hz}

    rows = []
    for s in SPLITS:
        t, c = idx[s][:, 0], idx[s][:, 1]
        m = y_sh_m[t, c]
        y = y_sh[t, c][m]
        scores = {"train base rate per coin": coin_rate[c][m], "|z| now": abs_z[t, c][m],
                  "shock now": s_today[t, c][m].astype(float), "own shocks last 24h": own_24[t, c][m],
                  "market breadth now": breadth[t, c][m],
                  "logistic on the 4 rules (train-fitted)": logit.predict_proba(feats[t, c][m])[:, 1]}
        for name, sc in scores.items():
            rows.append({"resolution": res, "split": s, "task": "T2 shock next bar", "horizon_hours": bh,
                         "model": name, **_cls(y, sc)})
        for k, h in enumerate(hz):
            yv = y_rv[t, c, k]
            preds = {"zero": np.zeros_like(yv), "train mean per coin": means[h][c],
                     f"persistence (trailing {h}h RV)": trail[h][t, c],
                     "HAR-RV (train-fitted)": har_predict(har_X[t, c], betas[h])}
            for name, pr in preds.items():
                rows.append({"resolution": res, "split": s, "task": "T3 realized volatility", "horizon_hours": h,
                             "model": name, **_reg(yv, pr)})
    out = pd.DataFrame(rows)
    out["note"] = np.where(out["split"] == "train", "in-sample", "")
    return out


def run(cfg: HourlyConfig, refresh: bool = False) -> None:
    pd.set_option("display.width", 220, "display.max_columns", 20)
    for res in cfg.datasets.build_resolutions:
        df = build_resolution(cfg, res)
        write_csv(df, cfg.reports_dir / f"hourly_baselines_{res}.csv")
        v = df[df["split"] != "train"]
        t2 = v[v["task"].str.startswith("T2")]
        t3 = v[v["task"].str.startswith("T3")]
        print(f"\n===== {res} =====")
        print(t2.pivot_table(index="model", columns="split", values=["AUPRC", "AUROC"]).round(4).to_string())
        print(t3.pivot_table(index=["horizon_hours", "model"], columns="split", values="MSE").map("{:.3e}".format).to_string())
