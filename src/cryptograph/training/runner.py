"""Run one experiment (config + seed + device) with local run tracking.

Tasks wired up: the LSTM baseline, the TGN contagion model (training/tgn_task.py) and DyGFormer
(training/dygformer_task.py). Other models (snapshot GNN) raise NotReadyError until they are built.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from ..data.config import PROJECT_ROOT
from .device import describe_device, resolve_device
from .early_stopping import EarlyStopping
from .losses import focal_loss, huber_loss, masked_mse, pos_weight_from_rate, weighted_bce
from .loop import evaluate, train_epoch
from .metrics import auprc, auroc, mae, mse
from .seed import seed_everything
from .tracking import RunTracker

SPLITS = ("train", "val", "test")


def _read_yaml(rel: str) -> dict:
    p = Path(rel)
    with open(p if p.is_absolute() else PROJECT_ROOT / p, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def eval_splits(cfg: dict) -> tuple[str, ...]:
    """Splits scored at the end of a run. Test is scored only when the experiment sets `evaluate_test: true`
    (final models), so tuning runs never look at the test period."""
    return ("val", "test") if cfg.get("evaluate_test", False) else ("val",)


def _datasets(data: dict) -> dict:
    from ..data.datasets import LSTMWindowDataset, PanelWindowDataset

    base = PROJECT_ROOT / "data" / "processed" / data["version"]
    if data.get("resolution"):
        d = base / data["resolution"]
        return {s: PanelWindowDataset(d / "panel.pt", d / f"lstm_index_{s}.pt") for s in SPLITS}
    return {s: LSTMWindowDataset(base / f"lstm_{s}.pt") for s in SPLITS}


def _lstm_task(cfg: dict, seed: int, device: torch.device, tracker: RunTracker) -> dict[str, Any]:
    from ..models.lstm import CryptoLSTM

    tr = cfg["training"]
    target = cfg.get("target", "y_rv")
    ds = _datasets(cfg["data"])
    x0 = ds["train"][0]
    classify = target == "y_shock"                    # T2: next-bar shock (one logit); otherwise regression
    n_feat, n_out = int(x0["x"].shape[-1]), 1 if classify else int(x0[target].shape[-1])
    mcfg = _read_yaml(cfg["model"]["config"]) | cfg["model"].get("overrides", {})
    model = CryptoLSTM(in_features=n_feat, hidden_dim=mcfg["hidden_dim"], num_layers=mcfg["num_layers"],
                       out_horizons=n_out, dropout=mcfg["dropout"]).to(device)
    tracker.log(f"LSTM: {sum(p.numel() for p in model.parameters()):,} parameters; target {target} ({n_out} horizons); "
                f"samples train/val/test {len(ds['train']):,}/{len(ds['val']):,}/{len(ds['test']):,}")

    g = torch.Generator().manual_seed(seed)
    train_set = ds["train"]
    if tr.get("max_train_batches"):
        n = min(len(train_set), tr["max_train_batches"] * tr["batch_size"])
        train_set = Subset(train_set, torch.randperm(len(train_set), generator=g)[:n].tolist())
    loaders = {"train": DataLoader(train_set, batch_size=tr["batch_size"], shuffle=True, generator=g,
                                   num_workers=tr.get("num_workers", 0))}
    for s in ("val", "test"):
        sub = ds[s]
        if tr.get("max_eval_samples") and len(sub) > tr["max_eval_samples"]:
            sub = Subset(sub, np.linspace(0, len(sub) - 1, tr["max_eval_samples"]).astype(int).tolist())
        loaders[s] = DataLoader(sub, batch_size=tr["batch_size"], shuffle=False, num_workers=tr.get("num_workers", 0))

    if classify:
        loss_name = cfg.get("loss", "focal")
        if loss_name == "focal":
            def loss_fn(pred, y, mask):
                return focal_loss(pred, y, mask, gamma=tr.get("focal_gamma", 2.0), alpha=tr.get("focal_alpha", 0.25))
        else:
            ys = ds["train"].y_shock[ds["train"].index[:, 0], ds["train"].index[:, 1]]                 if hasattr(ds["train"], "index") else ds["train"].y_shock
            pw = pos_weight_from_rate(float(ys.float().mean()))

            def loss_fn(pred, y, mask):
                return weighted_bce(pred, y, mask, pos_weight=pw)
    else:
        loss_fn = {"huber": huber_loss, "mse": masked_mse}[cfg.get("loss", "huber")]

    def step(m, batch, dev):
        x = torch.nan_to_num(batch["x"].to(dev).float())
        y = batch[target].to(dev).float()
        if classify:
            y = y.reshape(-1, 1)
        mask = torch.isfinite(y)
        pred = m(x)
        return loss_fn(pred, torch.nan_to_num(y), mask), (pred.detach().cpu(), y.cpu())

    def score(outs) -> float:                          # early-stopping metric (higher is better)
        P = torch.cat([o[0] for o in outs]).numpy()
        Y = torch.cat([o[1] for o in outs]).numpy()
        return auprc(Y, P) if classify else -mse(Y, P)

    opt = torch.optim.AdamW(model.parameters(), lr=tr["learning_rate"], weight_decay=tr.get("weight_decay", 0.0))
    stopper = EarlyStopping(patience=tr.get("patience", 10), mode="max")
    best_state, history = None, []
    for epoch in range(tr["epochs"]):
        tl = train_epoch(model, loaders["train"], step, opt, device, tr.get("grad_clip", 1.0))
        vl, outs = evaluate(model, loaders["val"], step, device)
        vs = score(outs)
        history.append({"epoch": epoch, "train_loss": tl, "val_loss": vl, "val_score": vs})
        tracker.log(f"epoch {epoch}: train loss {tl:.6f}, val loss {vl:.6f}, val {'AUPRC' if classify else '-MSE'} {vs:.6g}")
        if stopper.improved(vs) or best_state is None:
            best_state = copy.deepcopy(model.state_dict())
        if stopper.step(vs):
            tracker.log(f"early stop at epoch {epoch} (best {stopper.best_epoch})")
            break
    model.load_state_dict(best_state)
    tracker.save_checkpoint({"model_state": best_state, "model_config": mcfg, "n_features": n_feat,
                             "n_outputs": n_out, "target": target})
    metrics: dict[str, Any] = {"best_epoch": stopper.best_epoch, "epochs_run": len(history),
                               "evaluated_test": "test" in eval_splits(cfg)}
    for s in eval_splits(cfg):
        loss, outs = evaluate(model, loaders[s], step, device)
        P = torch.cat([o[0] for o in outs]).numpy()
        Y = torch.cat([o[1] for o in outs]).numpy()
        if classify:
            metrics |= {f"{s}_loss": loss, f"{s}_auprc": auprc(Y, P), f"{s}_auroc": auroc(Y, P),
                        f"{s}_posrate": float(np.nanmean(Y))}
        else:
            metrics |= {f"{s}_loss": loss, f"{s}_mse": mse(Y, P), f"{s}_mae": mae(Y, P)}
    metrics["history"] = history
    return metrics


def _tgn_task(cfg, seed, device, tracker):
    from .tgn_task import tgn_task

    return tgn_task(cfg, seed, device, tracker)


def _dygformer_task(cfg, seed, device, tracker):
    from .dygformer_task import dygformer_task

    return dygformer_task(cfg, seed, device, tracker)


TASKS = {"lstm": _lstm_task, "tgn": _tgn_task, "dygformer": _dygformer_task}


def run_experiment(cfg: dict, seed: int | None = None, device: str | None = None,
                   runs_dir: Path | str = PROJECT_ROOT / "runs") -> dict:
    """Resolve seed/device overrides, seed everything, run the task, write runs/<run_id>/ and results.csv."""
    from ..pipelines._experiment import NotReadyError

    cfg = copy.deepcopy(cfg)
    cfg["seed"] = int(seed if seed is not None else cfg.get("seed", 0))
    cfg["device"] = device or cfg.get("device", "auto")
    m = cfg.get("model")
    name = m if isinstance(m, str) else (m or {}).get("name")
    if name not in TASKS:
        raise NotReadyError(f"Training for model '{name}'")
    seed_everything(cfg["seed"], deterministic=cfg.get("deterministic", True))
    dev = resolve_device(cfg["device"])
    tracker = RunTracker(cfg["name"], cfg["seed"], cfg, describe_device(dev), runs_dir)
    try:
        metrics = TASKS[name](cfg, cfg["seed"], dev, tracker)
    except Exception as e:
        tracker.finish({"error": repr(e)[:300]}, status="failed")
        raise
    row = tracker.finish(metrics)
    return {"run_id": tracker.run_id, "run_dir": str(tracker.dir), "metrics": metrics, "row": row}
