"""Sender check at 15-minute resolution (VALIDATION only; test not touched): does knowing WHICH coins shocked help?

The 15-minute model of e2_15min.py sees the receiver's own state and market-wide numbers only. This adds information
about the shocking senders, from co-movement measured over the previous 30 days (refreshed daily, using whole days
strictly before the bar's day, so nothing from the bar's own day or later):
  control  corr_all, lag_all: the receiver's mean correlation / one-bar lead-lag correlation with ALL active coins
           (how market-linked the receiver is in general, whoever shocked)
  sender   corr_max, corr_mean: correlation of the receiver with the coins that shocked now
           lag_max: strongest "sender leads receiver by one bar" correlation among them
           zcorr: shock sizes of the senders weighted by their correlation with the receiver
           nbr_share: share of the receiver's 5 most correlated coins that shocked now
Same rows, labels, weights and fixed model settings as e2_15min.py; three input sets: base, base + control,
base + control + sender. The sender set must beat base + control for "who shocked" to matter.
Writes reports/e2_sender_check.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch  # noqa: F401  (torch before pandas on Windows)
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as ap
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from e2_15min import KS, build, fit_eval  # noqa: E402

from cryptograph.data.config import PROJECT_ROOT  # noqa: E402
from cryptograph.data.features.targets import future_any  # noqa: E402

WINDOW_DAYS, TOP = 30, 5


def daily_comovement(z: np.ndarray, valid: np.ndarray):
    """Correlation and one-bar lead-lag correlation (row coin leads column coin) over the WINDOW_DAYS whole days
    before each day; z is used as a zero-mean standardised return. Returns (C, L) of shape (days, N, N)."""
    T, N = z.shape
    assert T % 96 == 0, "the 15-minute calendar must hold whole days"
    m = valid.astype(np.float64)
    zz = np.where(valid, np.clip(np.nan_to_num(z), -10, 10), 0.0)
    zl, ml = np.vstack([np.zeros((1, N)), zz[:-1]]), np.vstack([np.zeros((1, N)), m[:-1]])
    D = T // 96
    day = lambda a: a.reshape(D, 96, N)                                         # noqa: E731
    sums = {"P": np.einsum("dsi,dsj->dij", day(zz), day(zz)), "A": np.einsum("dsi,dsj->dij", day(zz ** 2), day(m)),
            "Q": np.einsum("dsi,dsj->dij", day(zl), day(zz)), "Al": np.einsum("dsi,dsj->dij", day(zl ** 2), day(m)),
            "Bl": np.einsum("dsi,dsj->dij", day(ml), day(zz ** 2)), "n": np.einsum("dsi,dsj->dij", day(m), day(m))}
    roll = {}
    for k, v in sums.items():                                                   # sum over days [d - W, d)
        cs = np.concatenate([np.zeros((1, N, N)), np.cumsum(v, axis=0)])
        lo = np.maximum(np.arange(D) - WINDOW_DAYS, 0)
        roll[k] = cs[np.arange(D)] - cs[lo]
    with np.errstate(divide="ignore", invalid="ignore"):
        C = roll["P"] / np.sqrt(roll["A"] * roll["A"].transpose(0, 2, 1))
        L = roll["Q"] / np.sqrt(roll["Al"] * roll["Bl"])
    enough = roll["n"] >= 0.5 * WINDOW_DAYS * 96
    C, L = np.where(enough, C, np.nan), np.where(enough, L, np.nan)
    eye = np.eye(N, dtype=bool)
    C[:, eye], L[:, eye] = np.nan, np.nan
    return C.astype(np.float32), L.astype(np.float32)


def sender_features(b: dict, rows: np.ndarray) -> tuple[np.ndarray, list[str]]:
    sh, act, z, tt, jj = b["sh"], b["act"], b["z"], b["tt"][rows], b["jj"][rows]
    C, L = daily_comovement(b["z"], b["valid"])
    d = tt // 96
    Cj, Lj = C[d, :, jj], L[d, :, jj]                                           # (rows, N): coin i -> receiver j
    S = sh[tt].copy()
    S[np.arange(len(tt)), jj] = False                                           # the senders of this row
    A = act[tt].copy()
    A[np.arange(len(tt)), jj] = False
    absz = np.minimum(np.abs(np.nan_to_num(z[tt])), 10)
    Cfill = np.nan_to_num(Cj, nan=-np.inf)
    top = np.argsort(-Cfill, axis=1)[:, :TOP]
    nbr_ok = np.isfinite(np.take_along_axis(Cfill, top, 1)).all(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        f = {"corr_all": np.nanmean(np.where(A, Cj, np.nan), 1), "lag_all": np.nanmean(np.where(A, Lj, np.nan), 1),
             "corr_max": np.nanmax(np.where(S, Cj, np.nan), 1), "corr_mean": np.nanmean(np.where(S, Cj, np.nan), 1),
             "lag_max": np.nanmax(np.where(S, Lj, np.nan), 1),
             "zcorr": np.nansum(np.where(S, Cj * absz, np.nan), 1) / S.sum(1),
             "nbr_share": np.where(nbr_ok, np.take_along_axis(S, top, 1).mean(1), np.nan)}
    return np.column_stack(list(f.values())).astype(np.float32), list(f)


def main() -> None:
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)                  # all-NaN rows before 30 days of history
    b = build("val")
    sh, valid, tt, jj, w, X, m_tr, m_va, calm = (b[k] for k in ("sh", "valid", "tt", "jj", "w", "X", "m_tr", "m_va", "calm"))
    keep = m_tr | m_va                                                          # test rows are never featurised
    F = np.full((len(tt), 7), np.nan, np.float32)
    F[keep], fnames = sender_features(b, np.nonzero(keep)[0])
    F = np.where(np.isfinite(F), F, np.nan)
    print({n: round(float(np.nanmean(F[m_va, i])), 3) for i, n in enumerate(fnames)}, flush=True)
    sets = {"base": X, "base+control": np.column_stack([X, F[:, :2]]), "base+control+sender": np.column_stack([X, F])}
    rows = []
    for k in KS:
        y, ym = (a[tt, jj] for a in future_any(sh, valid, k))
        tr, va = m_tr & ym, m_va & ym
        for name, Xs in sets.items():
            for model in ("logreg", "gbm"):
                p, yv, c = fit_eval(Xs, y, w, tr, va, model), y[va], calm[va]
                base = np.average(yv, weights=w[va])
                rows.append({"k_bars": k, "minutes": 15 * k, "inputs": name, "model": model,
                             "event_auprc": ap(yv, p, sample_weight=w[va]), "random": base,
                             "auroc": roc_auc_score(yv, p, sample_weight=w[va]),
                             "calm_auprc": ap(yv[c], p[c], sample_weight=w[va][c]),
                             "calm_random": np.average(yv[c], weights=w[va][c])})
                rows[-1]["lift"] = rows[-1]["event_auprc"] / base
                print({k_: (round(v, 3) if isinstance(v, float) else v) for k_, v in rows[-1].items()}, flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(PROJECT_ROOT / "reports" / "e2_sender_check.csv", index=False)
    print(out.pivot_table(index=["inputs", "model"], columns="minutes", values="event_auprc").round(3).to_string())
    print(out.pivot_table(index=["inputs", "model"], columns="minutes", values="calm_auprc").round(3).to_string())


if __name__ == "__main__":
    main()
