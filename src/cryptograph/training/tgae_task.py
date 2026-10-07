"""Training task for the snapshot T-GAE (models/tgae.SnapshotTGAE): anomaly scores of daily market graphs.

Snapshots: graph_snapshots.pt (one per day, the trailing-168h correlation graph at 23:00 UTC; leakage-free). Node
features (`model.node_features`): "daily" = each coin's activity during that day (mean/max |z|, shocks, day return,
volume surge, end-of-day volatility) + network features; "hour" = panel features at 23:00 only + network features
(first design: a single hour cannot show that a day was unusual). Trained to reconstruct train-period
snapshots (truncated back-propagation through time over chunks of days), early stopping on validation
reconstruction loss. The anomaly score of a day is its reconstruction loss (adjacency BCE + feature MSE).

Evaluation against an independent label: a **stress day** has at least one hour in which >= `stress_breadth` of the
active coins shocked. Scores are compared on the same day and as an early warning for the next day, against two
naive scores: the day's turbulence (mean |z| over its 24 hours) and the density of its correlation graph.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pandas as pd
import torch

from ..data.config import PROJECT_ROOT
from ..data.storage import load_pt
from .early_stopping import EarlyStopping

DAILY_FEATURES = ["mean |z|", "max |z|", "shocks / 24 h", "day return (z units)", "volume surge (mean)",
                  "24h volatility at day end"]


def daily_node_features(panel, gi, st: np.ndarray, train: np.ndarray) -> torch.Tensor:
    """(S, N, 6) activity of each coin during the 24 hours of each snapshot day (hours t-23..t), standardised with
    the training snapshots: mean and max |z|, shock count / 24, sum of z / sqrt(24), mean volume_rel_168h, rv_24h at t."""
    z = np.nan_to_num(panel["shock_z"].numpy()[:, gi].astype(float))
    sh = (panel["shock_today"] & panel["shock_today_mask"]).numpy()[:, gi].astype(float)
    fn = panel["feature_names"]
    X = panel["X"].numpy()[:, gi]
    vol, rv = np.nan_to_num(X[..., fn.index("volume_rel_168h")]), np.nan_to_num(X[..., fn.index("rv_24h")])
    out = np.zeros((len(st), len(gi), len(DAILY_FEATURES)))
    for s, t in enumerate(st):
        lo = max(0, t - 23)
        az = np.abs(z[lo:t + 1])
        out[s] = np.stack([az.mean(0), az.max(0), sh[lo:t + 1].mean(0), z[lo:t + 1].sum(0) / np.sqrt(t + 1 - lo),
                           vol[lo:t + 1].mean(0), rv[t]], axis=-1)
    mu, sd = out[train].mean((0, 1)), out[train].std((0, 1))
    return torch.from_numpy((out - mu) / np.where(sd > 0, sd, 1.0)).float()


def _snapshot_data(base, panel, stress_breadth: float, node_features: str = "daily"):
    snaps = load_pt(base / "graph_snapshots.pt")
    net = load_pt(base / "network_features.pt")
    gi = snaps["graph_index"]
    st = snaps["snapshot_t"].numpy()
    assert np.array_equal(net["snapshot_t"].numpy(), st), "network features must follow the graph snapshots"
    netX = torch.nan_to_num(net["X"].float())
    if node_features == "daily":                      # the day's own activity of each coin + network features
        X = torch.cat([daily_node_features(panel, gi, st, panel["split_id"].numpy()[st] == 0), netX], dim=-1)
    else:                                             # "hour": panel features at the snapshot hour only
        X = torch.cat([panel["X"][st][:, gi].float(), netX], dim=-1)
    X = torch.nan_to_num(X)
    active = panel["active"][st][:, gi]
    N = len(gi)
    A = torch.zeros(len(st), N, N)
    for s, (ei, w) in enumerate(zip(snaps["undirected_edge_index"], snaps["undirected_edge_weight"])):
        A[s, ei[0], ei[1]] = 1.0
        A[s, ei[1], ei[0]] = 1.0
    # independent stress label and naive scores from the 24 hours of each snapshot day
    sh = (panel["shock_today"] & panel["shock_today_mask"]).numpy()[:, gi]
    act = (panel["active"] & panel["shock_today_mask"]).numpy()[:, gi]
    breadth = sh.sum(1) / np.maximum(act.sum(1), 1)
    az = np.abs(np.nan_to_num(panel["shock_z"].numpy()[:, gi]))
    stress, turb = np.zeros(len(st), bool), np.zeros(len(st))
    for s, t in enumerate(st):
        lo = max(0, t - 23)
        stress[s] = (breadth[lo:t + 1] >= stress_breadth).any()
        turb[s] = az[lo:t + 1][act[lo:t + 1]].mean() if act[lo:t + 1].any() else np.nan
    density = A.sum((1, 2)).numpy() / np.maximum(active.sum(1).numpy() * (active.sum(1).numpy() - 1), 1)
    split = panel["split_id"].numpy()[st]
    return snaps, st, X, A, active, stress, turb, density, split


def tgae_task(cfg: dict, seed: int, device: torch.device, tracker) -> dict[str, Any]:
    from sklearn.metrics import roc_auc_score

    from ..models.tgae import SnapshotTGAE
    from .runner import eval_splits

    data_cfg, tr = cfg["data"], cfg["training"]
    base = PROJECT_ROOT / "data" / "processed" / data_cfg["version"] / data_cfg["resolution"]
    panel = load_pt(base / "panel.pt")
    snaps, st, X, A, active, stress, turb, density, split = _snapshot_data(base, panel, tr.get("stress_breadth", 0.7),
                                                                           cfg["model"].get("node_features", "daily"))
    mc = cfg["model"]
    model = SnapshotTGAE(X.shape[-1], mc.get("hidden_dim", 32), mc.get("latent_dim", 16), mc.get("dropout", 0.1))
    opt = torch.optim.Adam(model.parameters(), lr=tr.get("learning_rate", 1e-3), weight_decay=tr.get("weight_decay", 0.0))
    lam, chunk = tr.get("feature_weight", 1.0), tr.get("chunk_days", 30)
    predict = mc.get("mode", "reconstruct") == "predict"     # score = reconstruction or one-day-ahead surprise
    ei_list = [torch.stack([torch.cat([e[0], e[1]]), torch.cat([e[1], e[0]])]) for e in snaps["undirected_edge_index"]]
    w_list = [torch.cat([w, w]).float() for w in snaps["undirected_edge_weight"]]
    S, N = len(st), X.shape[1]
    train_end = int(np.flatnonzero(split == 0).max()) + 1
    tracker.log(f"T-GAE: {sum(p.numel() for p in model.parameters()):,} parameters; {S} snapshots "
                f"({(split == 0).sum()} train / {(split == 1).sum()} val / {(split == 2).sum()} test); "
                f"stress days: {stress.mean():.1%}")

    def run(lo, hi, h, train):
        model.train(train)
        scores = []
        loss_acc = 0.0
        for s in range(lo, hi):
            h, adj, xr = model.step(X[s], ei_list[s], w_list[s], h, predict=predict)
            la, lf = SnapshotTGAE.losses(adj, A[s], xr, X[s], active[s])
            scores.append((float(la), float(lf)))
            if train and split[s] == 0:
                loss_acc = loss_acc + la + lam * lf
            if train and ((s - lo + 1) % chunk == 0 or s == hi - 1):
                if torch.is_tensor(loss_acc):
                    opt.zero_grad()
                    loss_acc.backward()
                    opt.step()
                loss_acc = 0.0
                h = h.detach()
        return h, scores

    stopper = EarlyStopping(patience=tr.get("patience", 10), mode="min")
    best, history = None, []
    for epoch in range(tr.get("epochs", 200)):
        h0 = torch.zeros(N, model.latent_dim)
        h, sc_tr = run(0, train_end, h0, True)
        with torch.no_grad():
            _, sc = run(train_end, S, h.detach(), False)
        val_idx = [i for i in range(train_end, S) if split[i] == 1]
        vloss = float(np.mean([sc[i - train_end][0] + lam * sc[i - train_end][1] for i in val_idx]))
        tloss = float(np.mean([a + lam * b for (a, b), s_ in zip(sc_tr, split[:train_end]) if s_ == 0]))
        history.append({"epoch": epoch, "train_loss": tloss, "val_loss": vloss})
        tracker.log(f"epoch {epoch}: train recon {tloss:.4f}, val recon {vloss:.4f}")
        if best is None or stopper.improved(vloss):
            best = copy.deepcopy(model.state_dict())
        if stopper.step(vloss) and epoch + 1 >= tr.get("min_epochs", 0):
            tracker.log(f"early stop at epoch {epoch} (best {stopper.best_epoch})")
            break
    model.load_state_dict(best)
    tracker.save_checkpoint({"model_state": best, "model_config": mc, "in_features": X.shape[-1]})
    with torch.no_grad():
        _, sc = run(0, S, torch.zeros(N, model.latent_dim), False)
    adj_l, feat_l = np.array(sc).T
    score = adj_l + lam * feat_l
    times = pd.DatetimeIndex(panel["times"])[st]
    df = pd.DataFrame({"time": times, "split": np.array(["train", "val", "test", "gap"])[np.clip(split, 0, 3)],
                       "score": score, "adj_loss": adj_l, "feat_loss": feat_l, "stress": stress,
                       "turbulence": turb, "density": density})
    df["stress_next_day"] = np.r_[stress[1:], False]
    df.loc[df.index[-1], "stress_next_day"] = np.nan
    df.to_csv(tracker.dir / "tgae_scores.csv", index=False)
    metrics: dict[str, Any] = {"best_epoch": stopper.best_epoch, "epochs_run": len(history),
                               "evaluated_test": "test" in eval_splits(cfg), "stress_rate": float(stress.mean())}
    for s in ("train",) + eval_splits(cfg):
        d = df[(df["split"] == s)].dropna(subset=["stress_next_day", "turbulence"])
        for target in ("stress", "stress_next_day"):
            y = d[target].astype(bool)
            if 0 < y.sum() < len(y):
                for name in ("score", "turbulence", "density"):
                    metrics[f"{s}_auroc_{target}_{name}"] = float(roc_auc_score(y, d[name]))
    if "test" in eval_splits(cfg):                    # does the score add to turbulence? fit on val, score test once
        from sklearn.linear_model import LogisticRegression

        fit, ev = (df[(df["split"] == s)].dropna(subset=["stress_next_day", "turbulence"]) for s in ("val", "test"))
        for target in ("stress", "stress_next_day"):
            for cols in (["turbulence"], ["turbulence", "score"]):
                lr = LogisticRegression().fit(fit[cols], fit[target].astype(bool))
                key = "turbulence" if len(cols) == 1 else "turbulence_plus_score"
                metrics[f"test_auroc_{target}_logit_{key}"] = float(
                    roc_auc_score(ev[target].astype(bool), lr.predict_proba(ev[cols])[:, 1]))
    metrics["history"] = history
    return metrics
