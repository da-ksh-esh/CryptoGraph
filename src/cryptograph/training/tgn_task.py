"""Training task for ContagionTGN on the shock-triggered event stream (T1 contagion).

Stream handling (no look-ahead):
  * Events are processed in time order over the full stream (train, val, test), in batches.
  * For each batch: embeddings come from memory and neighbours that hold only earlier batches; the decoder uses
    node features at each event's own bar t (panel features + network features of the latest snapshot <= t).
  * After the prediction, memory and the neighbour loader are updated with the batch's events (known at t).
  * Loss (train) and metrics (val/test) only count events whose bar t is a sample bar of that split
    (embargo respected, label window inside the split) and whose label is observed (y_mask).
Memory is reset at the start of each epoch; validation continues from the end of the train pass, and the final
evaluation replays train (no gradient), then val, then test.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch

from ..data.config import PROJECT_ROOT
from .early_stopping import EarlyStopping
from .losses import focal_loss, weighted_bce
from .metrics import auprc, auroc


def node_feature_table(panel: dict, net: dict | None, gidx: list[int]) -> torch.Tensor:
    """(T, N_graph, F) float32: scaled panel features at t, plus network features of the latest snapshot <= t
    (zeros before the first snapshot and where undefined)."""
    X = panel["X"][:, gidx].float()
    if net is None:
        return X.contiguous()
    snap_t = net["snapshot_t"].numpy()
    T = X.shape[0]
    pos = np.searchsorted(snap_t, np.arange(T), side="right") - 1          # latest snapshot at or before t
    NF = torch.nan_to_num(net["X"].float())
    Xn = torch.zeros(T, NF.shape[1], NF.shape[2])
    ok = pos >= 0
    Xn[torch.from_numpy(ok)] = NF[torch.from_numpy(pos[ok])]
    return torch.cat([X, Xn], dim=-1).contiguous()


def _batches(t: np.ndarray, start: int, end: int, size: int) -> list[tuple[int, int]]:
    """[start, end) cut into consecutive slices of about `size` events, never splitting one bar."""
    out, a = [], start
    while a < end:
        b = min(a + size, end)
        while b < end and t[b] == t[b - 1]:
            b += 1
        out.append((a, b))
        a = b
    return out


class StreamRunner:
    def __init__(self, model, data, feats: torch.Tensor, weight_mask: dict[str, np.ndarray], device, batch_size: int,
                 num_neighbors: int = 10):
        from torch_geometric.nn.models.tgn import LastNeighborLoader

        self.model, self.d, self.feats, self.device = model, data, feats, device
        self.wmask = weight_mask
        self.bs = batch_size
        self.N = model.memory.num_nodes
        self.loader = LastNeighborLoader(self.N, size=num_neighbors, device=device)
        self.assoc = torch.empty(self.N, dtype=torch.long, device=device)
        self.t_np = data.t.numpy()
        # the loader numbers inserted events 0, 1, ... after each reset; map them back to stream positions
        self.pos_of_eid = torch.zeros(len(self.t_np), dtype=torch.long)
        self.n_inserted = 0

    def reset(self):
        self.model.memory.reset_state()
        self.loader.reset_state()
        self.n_inserted = 0

    def _step(self, a: int, b: int):
        d, dev = self.d, self.device
        src, dst = d.src[a:b].to(dev), d.dst[a:b].to(dev)
        t, msg = d.t[a:b].to(dev), d.msg[a:b].to(dev)
        n_id = torch.cat([src, dst]).unique()
        n_id, edge_index, e_id = self.loader(n_id)
        self.assoc[n_id] = torch.arange(n_id.size(0), device=dev)
        pos = self.pos_of_eid[e_id.cpu()]
        z = self.model.embed(n_id, edge_index, d.t[pos].to(dev), d.msg[pos].to(dev))
        tt = d.t[a:b]
        x_src = self.feats[tt, d.src[a:b]].to(dev)
        x_dst = self.feats[tt, d.dst[a:b]].to(dev)
        logits = self.model.decode(z[self.assoc[src]], z[self.assoc[dst]], x_src, x_dst, msg)
        return logits, (a, b, src, dst, t, msg)

    def _commit(self, ev):
        a, b, src, dst, t, msg = ev
        self.model.memory.update_state(src, dst, t, msg)
        self.loader.insert(src, dst)
        self.pos_of_eid[self.n_inserted: self.n_inserted + (b - a)] = torch.arange(a, b)
        self.n_inserted += b - a

    def run(self, a: int, b: int, split: str, optimizer=None, loss_fn=None, max_batches: int | None = None):
        """Process events [a, b). Returns (mean loss, logits, labels, label masks) for the counted events."""
        train = optimizer is not None
        self.model.train(train)
        losses, out_l, out_y, out_m, out_p = [], [], [], [], []
        for nb, (s, e) in enumerate(_batches(self.t_np, a, b, self.bs)):
            if max_batches is not None and nb >= max_batches:
                break
            w = torch.from_numpy(self.wmask[split][self.t_np[s:e]])
            y, ym = self.d.y[s:e], self.d.y_mask[s:e] & w[:, None]
            with torch.set_grad_enabled(train):
                logits, ev = self._step(s, e)
                if loss_fn is not None and ym.any():          # loss_fn None = replay only (memory warm-up)
                    loss = loss_fn(logits, y.to(self.device), ym.to(self.device))
                    if train:
                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        optimizer.step()
                    losses.append(float(loss.detach()))
            self._commit(ev)
            self.model.memory.detach()
            if not train and ym.any():
                out_l.append(logits.detach().cpu())
                out_y.append(y)
                out_m.append(ym)
                out_p.append(torch.arange(s, e))
        cat = (lambda xs: torch.cat(xs)) if out_l else (lambda xs: None)
        self.last_positions = cat(out_p)                  # stream positions of the returned events
        return float(np.mean(losses)) if losses else float("nan"), cat(out_l), cat(out_y), cat(out_m)


def _metrics(prefix: str, logits, y, m, k_hours) -> dict:
    out = {}
    if logits is None:
        return out
    p = torch.sigmoid(logits).numpy()
    for k, kh in enumerate(k_hours):
        mk = m[:, k].numpy()
        out[f"{prefix}_auprc_k{kh}h"] = auprc(y[:, k].numpy(), p[:, k], mk)
        out[f"{prefix}_auroc_k{kh}h"] = auroc(y[:, k].numpy(), p[:, k], mk)
        out[f"{prefix}_posrate_k{kh}h"] = float(y[:, k].numpy()[mk].mean()) if mk.any() else float("nan")
    return out


def tgn_task(cfg: dict, seed: int, device: torch.device, tracker) -> dict[str, Any]:
    from ..models.tgn import ContagionTGN
    from .runner import _read_yaml, eval_splits

    data_cfg, tr = cfg["data"], cfg["training"]
    base = PROJECT_ROOT / "data" / "processed" / data_cfg["version"] / data_cfg["resolution"]
    panel = torch.load(base / "panel.pt", weights_only=False)
    ev = torch.load(base / "tgn_events.pt", weights_only=False)
    net = torch.load(base / "network_features.pt", weights_only=False) if data_cfg.get("network_features", True) else None
    d, gidx, k_hours = ev["full"], ev["graph_index"], ev["label_k_hours"]
    feats = node_feature_table(panel, net, gidx)
    sid = panel["split_id"].numpy()[d.t.numpy()]
    bounds = {s: (int(np.searchsorted(sid, i, "left")), int(np.searchsorted(sid, i, "right")))
              for i, s in enumerate(("train", "val", "test"))}
    assert (np.diff(sid) >= 0).all(), "events must be sorted by time and split"
    wmask = {s: panel["sample_mask"][s].numpy() for s in ("train", "val", "test")}

    mcfg = _read_yaml(cfg["model"]["config"]) | cfg["model"].get("overrides", {})
    model = ContagionTGN(num_nodes=len(gidx), msg_dim=d.msg.shape[1], node_feat_dim=feats.shape[-1],
                         memory_dim=mcfg["memory_dim"], time_dim=mcfg["time_dim"],
                         embedding_dim=mcfg["embedding_dim"], out_horizons=len(k_hours),
                         dropout=mcfg["dropout"], use_memory=mcfg.get("use_memory", True)).to(device)
    runner = StreamRunner(model, d, feats, wmask, device, tr["batch_size"], tr.get("num_neighbors", 10))

    a, b = bounds["train"]
    tw = wmask["train"][d.t.numpy()[a:b]]
    ym_tr = (d.y_mask[a:b] & torch.from_numpy(tw)[:, None]).numpy()
    pos_rate = (d.y[a:b].numpy() & ym_tr).sum(0) / np.maximum(ym_tr.sum(0), 1)
    if tr.get("loss", "focal") == "focal":
        def loss_fn(lg, y, m):
            return focal_loss(lg, y, m, gamma=tr.get("focal_gamma", 2.0), alpha=tr.get("focal_alpha", 0.25))
    else:
        pw = torch.tensor((1 - pos_rate) / np.maximum(pos_rate, 1e-6), dtype=torch.float32, device=device)

        def loss_fn(lg, y, m):
            return weighted_bce(lg, y, m, pos_weight=pw)
    tracker.log(f"TGN: {sum(p.numel() for p in model.parameters()):,} parameters; events train/val/test "
                f"{b - a:,}/{bounds['val'][1] - bounds['val'][0]:,}/{bounds['test'][1] - bounds['test'][0]:,}; "
                f"node features {feats.shape[-1]}; train positive rate per k {np.round(pos_rate, 3).tolist()}")

    opt = torch.optim.Adam(model.parameters(), lr=tr["learning_rate"], weight_decay=tr.get("weight_decay", 0.0))
    stopper = EarlyStopping(patience=tr.get("patience", 5), mode="max")
    sel_key = f"val_auprc_k{k_hours[0]}h"
    best, history = None, []
    mb, mbe = tr.get("max_train_batches"), tr.get("max_eval_batches")
    for epoch in range(tr["epochs"]):
        runner.reset()
        tl, *_ = runner.run(*bounds["train"], "train", opt, loss_fn, max_batches=mb)
        vl, lg, y, m = runner.run(*bounds["val"], "val", loss_fn=loss_fn, max_batches=mbe)   # continues the stream
        vm = _metrics("val", lg, y, m, k_hours)
        history.append({"epoch": epoch, "train_loss": tl, "val_loss": vl, **vm})
        tracker.log(f"epoch {epoch}: train loss {tl:.5f}, val loss {vl:.5f}, {sel_key} {vm.get(sel_key, float('nan')):.4f}")
        score = vm.get(sel_key, float("nan"))
        if best is None or stopper.improved(score):
            best = copy.deepcopy(model.state_dict())
        if stopper.step(score if np.isfinite(score) else -np.inf):
            tracker.log(f"early stop at epoch {epoch} (best {stopper.best_epoch})")
            break

    model.load_state_dict(best)
    tracker.save_checkpoint({"model_state": best, "model_config": mcfg, "k_hours": k_hours,
                             "node_feat_dim": feats.shape[-1], "msg_names": ev["msg_names"]})
    runner.reset()                                 # final pass: replay train (no gradient), then val and test
    runner.run(*bounds["train"], "train", max_batches=mb)
    metrics: dict[str, Any] = {"best_epoch": stopper.best_epoch, "epochs_run": len(history),
                               "evaluated_test": "test" in eval_splits(cfg)}
    for s in eval_splits(cfg):
        loss, lg, y, m = runner.run(*bounds[s], s, loss_fn=loss_fn, max_batches=mbe)
        metrics |= {f"{s}_loss": loss, **_metrics(s, lg, y, m, k_hours)}
        if lg is not None:                            # per-event predictions for bootstrap tests (notebook 03)
            pos = runner.last_positions
            torch.save({"position": pos, "t": d.t[pos], "src": d.src[pos], "dst": d.dst[pos],
                        "prob": torch.sigmoid(lg).half(), "y": y, "mask": m, "k_hours": k_hours},
                       tracker.dir / f"predictions_{s}.pt")
    metrics["history"] = history
    return metrics
