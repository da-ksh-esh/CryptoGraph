"""E2 prototype: the contagion question at 15-minute resolution (VALIDATION only; test not touched).

Same recipe as the hourly data, one level finer:
  * 15-minute closes from the 5-minute klines; log returns deseasonalised with a (weekday x 15-minute slot) profile
    fitted on train; z[t] = r_ds[t] / std(r_ds over the previous 7 days, up to t-1); shock = |z| > z*, z* set on train
    for a 4% shock rate (as the hourly data).
  * Rows: (bar t, receiver j) where another coin shocked at t, weighted by the number of shocking coins (= events).
  * Label: j shocks in (t, t+k] for k = 1, 2, 4, 8 bars (15 min, 30 min, 1 h, 2 h).
  * Inputs known at the close of bar t: receiver |z|, z, recent shock counts, bars since its last shock, volatility
    ratios, market breadth now / last hour, market and BTC z, time of day.
  * Splits: the hourly dates; 7-day embargo before validation; label windows inside the split.
Writes reports/e2_15min.csv.

--final scores the TEST period once (same fits on train, 7-day embargo) and writes reports/final_15min_test.csv.
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
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cryptograph.data.config import PROJECT_ROOT, load_hourly_config
from cryptograph.data.features.targets import future_any, shock_z
from cryptograph.data.storage import load_pt
from cryptograph.data.universe_hourly import _ts

KS = [1, 2, 4, 8]
BAR = pd.Timedelta(minutes=15)


def build(ev: str = "val") -> dict:
    """15-minute shocks, rows, inputs and split masks; `ev` is the evaluation split ("val" or "test")."""
    cfg = load_hourly_config("configs/data/hourly_v3.yaml")
    panel = load_pt(PROJECT_ROOT / "data/processed/v3/1h/panel.pt")
    coins = panel["graph_coins"]
    k5 = pd.read_parquet(cfg.interim_dir / "klines_5m.parquet", columns=["close"])
    c5 = k5["close"].unstack("coin").reindex(columns=coins).astype(float)
    cal = pd.date_range(_ts(cfg.window.start), _ts(cfg.window.end) + pd.Timedelta(minutes=45), freq="15min")
    # close of each 15-minute bar = close of its last 5-minute bar (bars labelled by their open time); a bar needs
    # all three 5-minute bars
    close = c5.groupby(c5.index.floor("15min")).last().reindex(cal)
    have = c5.notna().groupby(c5.index.floor("15min")).sum().reindex(cal).fillna(0) >= 3
    close = close.where(have)
    r = np.log(close).diff()
    T, N = r.shape
    split = np.full(T, -1)
    for i, s in enumerate(("train", "val", "test")):
        a, b = getattr(cfg.splits, s)
        split[(cal >= pd.Timestamp(a)) & (cal < pd.Timestamp(b) + pd.Timedelta(days=1))] = i
    tr_rows = split == 0
    slot = (cal.dayofweek * 96 + cal.hour * 4 + cal.minute // 15).to_numpy()
    sq = r[tr_rows] ** 2
    prof = pd.DataFrame(sq / sq.mean()).groupby(slot[tr_rows]).mean().mean(axis=1).reindex(range(7 * 96)).to_numpy()
    prof = prof / np.nanmean(prof)
    r_ds = r.div(np.sqrt(prof[slot]), axis=0)
    W = 7 * 96
    sig = r_ds.rolling(W, min_periods=int(0.8 * W)).std()
    z = shock_z(r_ds.to_numpy(), sig.to_numpy())
    zthr = float(np.quantile(np.abs(z[tr_rows][np.isfinite(z[tr_rows])]), 0.96))
    valid = np.isfinite(z)
    sh = (np.abs(np.nan_to_num(z)) > zthr) & valid
    act = close.notna().to_numpy()
    print(f"15-min bars {T:,}; shock threshold |z| > {zthr:.2f}; train shock rate {sh[tr_rows].sum() / valid[tr_rows].sum():.3f}")

    # features known at the close of bar t
    zz = np.nan_to_num(z)
    cs = np.cumsum(np.vstack([np.zeros((1, N)), sh]), axis=0)
    cnt = lambda w: cs[1:] - cs[np.maximum(np.arange(T) + 1 - w, 0)]   # noqa: E731
    since = np.full((T, N), 96.0)
    last = np.full(N, -10**9)
    for t in range(T):
        last = np.where(sh[t], t, last)
        since[t] = np.minimum(t - last, 96)
    s = sig.to_numpy()
    rms = lambda w: np.sqrt((r_ds ** 2).rolling(w, min_periods=max(1, int(0.75 * w))).mean()).to_numpy()  # noqa: E731
    with np.errstate(divide="ignore", invalid="ignore"):
        node = {"absz": np.minimum(np.abs(zz), 10), "z": np.clip(zz, -10, 10), "shock_now": sh.astype(float),
                "shocks_1h": cnt(4), "shocks_4h": cnt(16), "shocks_24h": cnt(96), "log_since": np.log1p(since),
                "lrv_1h": np.log(rms(4) / s), "lrv_4h": np.log(rms(16) / s), "lrv_24h": np.log(rms(96) / s)}
    breadth = sh.sum(1) / np.maximum(act.sum(1), 1)
    cb = np.cumsum(np.r_[0.0, breadth])
    glob = {"breadth": breadth, "breadth_1h": (cb[1:] - cb[np.maximum(np.arange(T) + 1 - 4, 0)]) / 4,
            "mkt_absz": np.abs(zz).mean(1), "mkt_z": zz.mean(1), "btc_z": np.clip(zz[:, coins.index("BTC")], -10, 10),
            "eth_z": np.clip(zz[:, coins.index("ETH")], -10, 10),
            "tod_sin": np.sin(2 * np.pi * (slot % 96) / 96), "tod_cos": np.cos(2 * np.pi * (slot % 96) / 96)}

    n_src = sh.sum(1)
    tt, jj = np.nonzero(act & (n_src[:, None] - sh > 0))
    w = (n_src[tt] - sh[tt, jj]).astype(float)
    X = np.column_stack([v[tt, jj] for v in node.values()] + [v[tt] for v in glob.values()])
    X = np.where(np.isfinite(X), X, np.nan).astype(np.float32)
    lookback, horizon = pd.Timedelta(days=7), max(KS) * BAR
    def smask(name, embargo):                                   # noqa: E306
        a, b = getattr(cfg.splits, name)
        lo = pd.Timestamp(a) + max(embargo, lookback)
        hi = pd.Timestamp(b) + pd.Timedelta(days=1) - BAR - horizon
        return np.asarray((cal >= lo) & (cal <= hi))
    m_tr, m_va = smask("train", pd.Timedelta(0))[tt], smask(ev, pd.Timedelta(days=7))[tt]
    calm = since[tt, jj] >= 96
    return {"cfg": cfg, "cal": cal, "coins": coins, "z": z, "sh": sh, "valid": valid, "act": act, "tt": tt, "jj": jj,
            "w": w, "X": X, "names": list(node) + list(glob), "m_tr": m_tr, "m_va": m_va, "calm": calm}


def fit_eval(X, y, w, tr, va, model):
    """Fit on the train rows with fixed settings; return validation (or test) probabilities."""
    if model == "logreg":
        clf = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=3000))
        clf.fit(X[tr], y[tr], logisticregression__sample_weight=w[tr])
    else:
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                             min_samples_leaf=2000, l2_regularization=1.0, random_state=0)
        clf.fit(X[tr], y[tr], sample_weight=w[tr])
    return clf.predict_proba(X[va])[:, 1]


def main() -> None:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--final", action="store_true")
    final = ap_.parse_args().final
    ev = "test" if final else "val"
    b = build(ev)
    sh, valid, tt, jj, w, X, m_tr, m_va, calm = (b[k] for k in ("sh", "valid", "tt", "jj", "w", "X", "m_tr", "m_va", "calm"))
    rows = []
    for k in KS:
        y, ym = (a[tt, jj] for a in future_any(sh, valid, k))
        tr, va = m_tr & ym, m_va & ym
        for model in ("logreg", "gbm"):
            p, yv, c = fit_eval(X, y, w, tr, va, model), y[va], calm[va]
            base = np.average(yv, weights=w[va])
            rows.append({"k_bars": k, "minutes": 15 * k, "model": model, "event_auprc": ap(yv, p, sample_weight=w[va]),
                         "random": base, "auroc": roc_auc_score(yv, p, sample_weight=w[va]),
                         "calm_auprc": ap(yv[c], p[c], sample_weight=w[va][c]),
                         "calm_random": np.average(yv[c], weights=w[va][c]), "split": ev, "n_eval_rows": int(va.sum())})
            rows[-1]["lift"] = rows[-1]["event_auprc"] / base
            rows[-1]["calm_lift"] = rows[-1]["calm_auprc"] / rows[-1]["calm_random"]
            print({k_: (round(v, 3) if isinstance(v, float) else v) for k_, v in rows[-1].items()}, flush=True)
    pd.DataFrame(rows).to_csv(PROJECT_ROOT / "reports" / ("final_15min_test.csv" if final else "e2_15min.csv"), index=False)


if __name__ == "__main__":
    main()
