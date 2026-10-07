"""Training task for DyGFormer on the shock-triggered event stream (T1 contagion).

Same protocol as the TGN task (training/tgn_task.py): events in time order, batches never split an hour, the history
only holds events of earlier batches when a batch is predicted, loss/metrics only on sample-bar events with an
observed label, early stopping on validation AUPRC (k = 1h), final pass = train replay then val (and test once).
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch

from ..data.config import PROJECT_ROOT
from ..data.storage import load_pt
from .early_stopping import EarlyStopping
from .losses import focal_loss
from .tgn_task import _batches, _metrics, node_feature_table, with_futures


class HistoryRunner:
    def __init__(self, model, data, feats, weight_mask, device, batch_size):
        from ..models.dygformer import InteractionHistory

        self.model, self.d, self.feats, self.device, self.bs = model, data, feats, device, batch_size
        self.wmask = weight_mask
        self.hist = InteractionHistory(model.num_nodes, model.S, device)
        self.t_np = data.t.numpy()
        self.msg_dev = data.msg.to(device)
        self.t_dev = data.t.to(device)
        self.feats_dev = feats.to(device)

    def reset(self):
        self.hist.reset()

    def _side(self, nodes):
        nbr, tt, pos = self.hist.get(nodes)
        safe_pos, safe_nbr = pos.clamp_min(0), nbr.clamp_min(0)
        feats = self.feats_dev[tt, safe_nbr] * (nbr >= 0).unsqueeze(-1)          # neighbour features at that hour
        msg = self.msg_dev[safe_pos] * (pos >= 0).unsqueeze(-1)
        return (nbr, tt), feats, msg

    def run(self, a: int, b: int, split: str, optimizer=None, loss_fn=None, max_batches: int | None = None):
        train = optimizer is not None
        self.model.train(train)
        losses, out_l, out_y, out_m, out_p = [], [], [], [], []
        for nb, (s, e) in enumerate(_batches(self.t_np, a, b, self.bs)):
            if max_batches is not None and nb >= max_batches:
                break
            dev = self.device
            src, dst = self.d.src[s:e].to(dev), self.d.dst[s:e].to(dev)
            t, msg = self.t_dev[s:e], self.msg_dev[s:e]
            w = torch.from_numpy(self.wmask[split][self.t_np[s:e]])
            y, ym = self.d.y[s:e], self.d.y_mask[s:e] & w[:, None]
            with torch.set_grad_enabled(train):
                hs, fs, ms = self._side(src)
                hd, fd, md = self._side(dst)
                logits = self.model(hs, hd, fs, fd, ms, md, t, self.feats_dev[t, src], self.feats_dev[t, dst], msg)
                if loss_fn is not None and ym.any():
                    loss = loss_fn(logits, y.to(dev), ym.to(dev))
                    if train:
                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        optimizer.step()
                    losses.append(float(loss.detach()))
            self.hist.insert(src, dst, t, torch.arange(s, e, device=dev))       # only after the prediction
            if not train and ym.any():
                out_l.append(logits.detach().cpu())
                out_y.append(y)
                out_m.append(ym)
                out_p.append(torch.arange(s, e))
        cat = (lambda xs: torch.cat(xs)) if out_l else (lambda xs: None)
        self.last_positions = cat(out_p)
        return float(np.mean(losses)) if losses else float("nan"), cat(out_l), cat(out_y), cat(out_m)


def dygformer_task(cfg: dict, seed: int, device: torch.device, tracker) -> dict[str, Any]:
    from ..models.dygformer import DyGFormer
    from .refit import is_refit, refit_and_score
    from .runner import _read_yaml, eval_splits

    data_cfg, tr = cfg["data"], cfg["training"]
    base = PROJECT_ROOT / "data" / "processed" / data_cfg["version"] / data_cfg["resolution"]
    panel = load_pt(base / "panel.pt")
    ev = load_pt(base / "tgn_events.pt")
    net = load_pt(base / "network_features.pt") if data_cfg.get("network_features", True) else None
    d, gidx, k_hours = ev["full"], ev["graph_index"], ev["label_k_hours"]
    feats = node_feature_table(panel, net, gidx)
    feats = with_futures(feats, base, data_cfg, panel["graph_coins"])
    sid = panel["split_id"].numpy()[d.t.numpy()]
    bounds = {s: (int(np.searchsorted(sid, i, "left")), int(np.searchsorted(sid, i, "right")))
              for i, s in enumerate(("train", "val", "test"))}
    wmask = {s: panel["sample_mask"][s].numpy() for s in ("train", "val", "test")}

    mcfg = _read_yaml(cfg["model"]["config"]) | cfg["model"].get("overrides", {})
    model = DyGFormer(num_nodes=len(gidx), msg_dim=d.msg.shape[1], node_feat_dim=feats.shape[-1],
                      seq_len=mcfg["max_input_sequence_length"], patch=mcfg["patch_size"], d_model=mcfg["d_model"],
                      n_layers=mcfg["num_layers"], n_heads=mcfg["num_heads"], out_horizons=len(k_hours),
                      dropout=mcfg["dropout"], encoder_dropout=mcfg.get("encoder_dropout", 0.0)).to(device)
    runner = HistoryRunner(model, d, feats, wmask, device, tr["batch_size"])

    def loss_fn(lg, y, m):
        return focal_loss(lg, y, m, gamma=tr.get("focal_gamma", 2.0), alpha=tr.get("focal_alpha", 0.25))

    tracker.log(f"DyGFormer: {sum(p.numel() for p in model.parameters()):,} parameters; history {model.S}, "
                f"patch {model.P}; events train/val/test {bounds['train'][1] - bounds['train'][0]:,}/"
                f"{bounds['val'][1] - bounds['val'][0]:,}/{bounds['test'][1] - bounds['test'][0]:,}")
    opt = torch.optim.Adam(model.parameters(), lr=tr["learning_rate"], weight_decay=tr.get("weight_decay", 0.0))
    if is_refit(tr):                                  # final refit on train + val, fixed epochs, test once
        return refit_and_score(runner, d, bounds, wmask, tr["epochs"], opt, loss_fn, k_hours, tracker,
                               {"model_config": mcfg, "k_hours": k_hours, "node_feat_dim": feats.shape[-1],
                                "msg_names": ev["msg_names"]})
    stopper = EarlyStopping(patience=tr.get("patience", 10), mode="max")
    sel_key = f"val_auprc_k{k_hours[0]}h"
    best, history = None, []
    mb, mbe = tr.get("max_train_batches"), tr.get("max_eval_batches")
    for epoch in range(tr["epochs"]):
        runner.reset()
        tl, *_ = runner.run(*bounds["train"], "train", opt, loss_fn, max_batches=mb)
        vl, lg, y, m = runner.run(*bounds["val"], "val", loss_fn=loss_fn, max_batches=mbe)
        vm = _metrics("val", lg, y, m, k_hours)
        history.append({"epoch": epoch, "train_loss": tl, "val_loss": vl, **vm})
        tracker.log(f"epoch {epoch}: train loss {tl:.5f}, val loss {vl:.5f}, {sel_key} {vm.get(sel_key, float('nan')):.4f}")
        score = vm.get(sel_key, float("nan"))
        if best is None or stopper.improved(score):
            best = copy.deepcopy(model.state_dict())
        if stopper.step(score if np.isfinite(score) else -np.inf) and epoch + 1 >= tr.get("min_epochs", 0):
            tracker.log(f"early stop at epoch {epoch} (best {stopper.best_epoch})")
            break
    model.load_state_dict(best)
    tracker.save_checkpoint({"model_state": best, "model_config": mcfg, "k_hours": k_hours,
                             "node_feat_dim": feats.shape[-1], "msg_names": ev["msg_names"]})
    runner.reset()
    _, lg_tr, y_tr, m_tr = runner.run(*bounds["train"], "train", max_batches=mb)
    metrics: dict[str, Any] = {"best_epoch": stopper.best_epoch, "epochs_run": len(history),
                               "evaluated_test": "test" in eval_splits(cfg), **_metrics("train", lg_tr, y_tr, m_tr, k_hours)}
    for s in eval_splits(cfg):
        loss, lg, y, m = runner.run(*bounds[s], s, loss_fn=loss_fn, max_batches=mbe)
        metrics |= {f"{s}_loss": loss, **_metrics(s, lg, y, m, k_hours)}
        if lg is not None:
            pos = runner.last_positions
            torch.save({"position": pos, "t": d.t[pos], "src": d.src[pos], "dst": d.dst[pos],
                        "prob": torch.sigmoid(lg).half(), "y": y, "mask": m, "k_hours": k_hours},
                       tracker.dir / f"predictions_{s}.pt")
    metrics["history"] = history
    return metrics
