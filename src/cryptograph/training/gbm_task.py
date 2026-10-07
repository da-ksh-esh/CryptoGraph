"""Gradient-boosting baseline for T1 contagion: no graph, no memory, no history.

Each event (coin i shocks at hour t, receiver j) becomes one row with exactly what the TGN head sees besides its
graph embeddings: node features of i and j at t and the event message, plus the coin ids (categorical). Optional:
the leakage-free event context of `context_tables` (shock state of both coins, market breadth). One
HistGradientBoostingClassifier (LightGBM-style histogram boosting, scikit-learn) per horizon. The number of trees is
chosen on validation AUPRC per horizon; then the model is scored on val (and test once). With
`refit_train_val: true` the chosen number of trees is refitted on train + val before scoring the test.

Predictions are saved for exactly the events the stream models score (same batches of 200, never splitting an hour,
kept when the batch has a counted label), so the day-block bootstrap can compare them event by event.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch

from ..data.config import PROJECT_ROOT
from .metrics import auprc
from .tgn_task import _batches, _metrics, context_tables, node_feature_table, with_futures


def _scored_positions(t: np.ndarray, ym: np.ndarray, a: int, b: int, bs: int) -> np.ndarray:
    """ym: (b - a, K) counted-label mask of events a..b-1."""
    keep = [np.arange(s, e) for s, e in _batches(t, a, b, bs) if ym[s - a:e - a].any()]
    return np.concatenate(keep) if keep else np.zeros(0, dtype=int)


def gbm_design(cfg: dict) -> dict:
    """The boosting design matrix for every event: [x_src (F), x_dst (F), msg (M), context?, src id, dst id], with
    labels, masks and split bounds. `n_perturbable` = 2F + M (the inputs shared with the TGN head)."""
    data_cfg = cfg["data"]
    base = PROJECT_ROOT / "data" / "processed" / data_cfg["version"] / data_cfg["resolution"]
    panel = torch.load(base / "panel.pt", weights_only=False)
    ev = torch.load(base / "tgn_events.pt", weights_only=False)
    net = torch.load(base / "network_features.pt", weights_only=False) if data_cfg.get("network_features", True) else None
    d, gidx, k_hours = ev["full"], ev["graph_index"], ev["label_k_hours"]
    feats = node_feature_table(panel, net, gidx)
    feats = with_futures(feats, base, data_cfg, panel["graph_coins"])
    t_np, src, dst = d.t.numpy(), d.src.numpy(), d.dst.numpy()
    sid = panel["split_id"].numpy()[t_np]
    bounds = {s: (int(np.searchsorted(sid, i, "left")), int(np.searchsorted(sid, i, "right")))
              for i, s in enumerate(("train", "val", "test"))}
    wmask = {s: panel["sample_mask"][s].numpy() for s in ("train", "val", "test")}

    cols = [feats[d.t, d.src], feats[d.t, d.dst], d.msg.float()]
    if cfg["model"].get("context_features", False):
        node, glob = context_tables(panel, gidx)
        cols += [node[d.t, d.src], node[d.t, d.dst], glob[d.t]]
    X = torch.cat(cols, dim=-1).numpy().astype(np.float32)
    X = np.column_stack([X, src, dst]).astype(np.float32)
    cat = np.zeros(X.shape[1], dtype=bool)
    cat[-2:] = True                                   # coin ids as categorical features
    Y, YM = d.y.numpy().astype(np.int8), d.y_mask.numpy().astype(bool)
    return {"X": X, "cat": cat, "Y": Y, "YM": YM, "bounds": bounds, "wmask": wmask, "t_np": t_np, "d": d,
            "k_hours": k_hours, "n_perturbable": 2 * feats.shape[-1] + d.msg.shape[1]}


def gbm_task(cfg: dict, seed: int, device: torch.device, tracker) -> dict[str, Any]:
    from sklearn.ensemble import HistGradientBoostingClassifier

    from .runner import eval_splits

    tr = cfg["training"]
    g = gbm_design(cfg)
    X, cat, Y, YM, bounds, wmask, t_np, d, k_hours = (g[k] for k in ("X", "cat", "Y", "YM", "bounds", "wmask", "t_np",
                                                                     "d", "k_hours"))

    def split_rows(s, k):
        a, b = bounds[s]
        m = YM[a:b, k] & wmask[s][t_np[a:b]]
        return np.arange(a, b)[m]

    step, max_iter = tr.get("tree_step", 50), tr.get("max_trees", 1000)
    params = dict(learning_rate=tr.get("learning_rate", 0.05), max_leaf_nodes=tr.get("max_leaf_nodes", 31),
                  min_samples_leaf=tr.get("min_samples_leaf", 200), l2_regularization=tr.get("l2", 1.0),
                  categorical_features=cat, early_stopping=False, random_state=seed)
    tracker.log(f"GBM: {X.shape[1]} features (context {bool(cfg['model'].get('context_features', False))}); events train/val/test {bounds['train'][1] - bounds['train'][0]:,}/"
                f"{bounds['val'][1] - bounds['val'][0]:,}/{bounds['test'][1] - bounds['test'][0]:,}")

    models, chosen, history = [], [], []
    for k, kh in enumerate(k_hours):
        tr_rows, va_rows = split_rows("train", k), split_rows("val", k)
        clf = HistGradientBoostingClassifier(max_iter=step, warm_start=True, **params)
        best, best_n, bad = -np.inf, step, 0
        for n in range(step, max_iter + 1, step):
            clf.set_params(max_iter=n)
            clf.fit(X[tr_rows], Y[tr_rows, k])
            score = auprc(Y[va_rows, k], clf.predict_proba(X[va_rows])[:, 1])
            history.append({"k_hours": kh, "trees": n, "val_auprc": score})
            if score > best:
                best, best_n, best_clf, bad = score, n, copy.deepcopy(clf), 0
            else:
                bad += 1
                if bad >= tr.get("patience", 4):
                    break
        tracker.log(f"k={kh}h: {best_n} trees, val AUPRC {best:.4f}")
        models.append(best_clf)
        chosen.append(best_n)

    def predict(rows):
        return np.stack([m.predict_proba(X[rows])[:, 1] for m in models], axis=1)

    def score(s):
        a, b = bounds[s]
        ym = YM[a:b] & wmask[s][t_np[a:b]][:, None]
        pos = _scored_positions(t_np, ym, a, b, tr.get("batch_size", 200))
        p = predict(pos)
        lg = torch.logit(torch.from_numpy(p).clamp(1e-6, 1 - 1e-6))
        y, m = torch.from_numpy(Y[pos]), torch.from_numpy(YM[pos] & wmask[s][t_np[pos]][:, None])
        torch.save({"position": torch.from_numpy(pos), "t": d.t[pos], "src": d.src[pos], "dst": d.dst[pos],
                    "prob": torch.from_numpy(p).half(), "y": y, "mask": m, "k_hours": k_hours},
                   tracker.dir / f"predictions_{s}.pt")
        return _metrics(s, lg, y, m, k_hours)

    metrics: dict[str, Any] = {"trees": chosen, "evaluated_test": "test" in eval_splits(cfg)}
    tr_m = {}
    for k, kh in enumerate(k_hours):                  # train metrics for the fit diagnostic
        rows = split_rows("train", k)
        p = models[k].predict_proba(X[rows])[:, 1]
        tr_m |= {f"train_auprc_k{kh}h": auprc(Y[rows, k], p), f"train_posrate_k{kh}h": float(Y[rows, k].mean())}
    metrics |= tr_m | score("val")
    if tr.get("refit_train_val", False):              # same number of trees, refitted on train + val
        for k in range(len(k_hours)):
            rows = np.concatenate([split_rows("train", k), split_rows("val", k)])
            clf = HistGradientBoostingClassifier(**(params | {"max_iter": chosen[k], "warm_start": False}))
            models[k] = clf.fit(X[rows], Y[rows, k])
        metrics["refit_train_val"] = True
        tracker.log("refitted on train + val")
    if "test" in eval_splits(cfg):
        metrics |= score("test")
    metrics["history"] = history
    return metrics
