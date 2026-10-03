"""Training and evaluation harness for CryptoLSTM non-graph baseline."""

from typing import Dict, Any, List, Optional
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
import numpy as np

from ..models.lstm import CryptoLSTM
from ..training.losses import WeightedBCELoss
from ..training.metrics import compute_classification_metrics
from ..training.early_stopping import EarlyStopping
from ..training.tracking import ExperimentTracker
from ..common.logger import get_logger

logger = get_logger(__name__)


def evaluate_lstm(
    model: nn.Module,
    features: torch.Tensor,
    targets: torch.Tensor,
    validity_mask: torch.Tensor,
    active_mask: torch.Tensor,
    target_indices: List[int],
    horizons: List[int],
    lookback_window: int = 24,
    batch_size: int = 64,
    device: torch.device = torch.device("cpu")
) -> Dict[str, float]:
    """
    Evaluates CryptoLSTM on a specified chronological split without leakage.
    Returns per-horizon and aggregate AUPRC, AUROC, Precision, Recall, and F1.
    """
    model.eval()
    L = lookback_window
    offsets = torch.arange(-L, 0, dtype=torch.long, device=device)
    valid_targets = [t for t in target_indices if t >= L]
    target_tensor = torch.tensor(valid_targets, dtype=torch.long, device=device)
    num_samples = len(target_tensor)

    all_scores = [[] for _ in range(len(horizons))]
    all_trues = [[] for _ in range(len(horizons))]

    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            batch_t = target_tensor[i : i + batch_size]
            b_len = len(batch_t)
            indices = batch_t[:, None] + offsets[None, :]

            # Gather and reshape
            x = features[indices].permute(0, 2, 1, 3).reshape(b_len * 30, L, -1).to(device)
            y = targets[batch_t].float().reshape(b_len * 30, -1).to(device)
            mask = (validity_mask[batch_t] & active_mask[batch_t].unsqueeze(-1)).reshape(b_len * 30, -1).to(device)

            logits = model(x)
            probs = torch.sigmoid(logits)

            for k in range(len(horizons)):
                valid_k = mask[:, k]
                if valid_k.sum() > 0:
                    all_scores[k].append(probs[:, k][valid_k].cpu().numpy())
                    all_trues[k].append(y[:, k][valid_k].cpu().numpy())

    metrics = {}
    auprcs = []
    aurocs = []
    f1s = []

    for k, h in enumerate(horizons):
        if len(all_scores[k]) > 0:
            s_k = np.concatenate(all_scores[k])
            t_k = np.concatenate(all_trues[k])
            m_k = compute_classification_metrics(t_k, s_k, threshold=0.5)
            metrics[f"auprc_{h}h"] = m_k["auprc"]
            metrics[f"auroc_{h}h"] = m_k["auroc"]
            metrics[f"f1_{h}h"] = m_k["f1"]
            metrics[f"precision_{h}h"] = m_k["precision"]
            metrics[f"recall_{h}h"] = m_k["recall"]
            auprcs.append(m_k["auprc"])
            aurocs.append(m_k["auroc"])
            f1s.append(m_k["f1"])

    metrics["mean_auprc"] = float(np.mean(auprcs)) if auprcs else 0.0
    metrics["mean_auroc"] = float(np.mean(aurocs)) if aurocs else 0.5
    metrics["mean_f1"] = float(np.mean(f1s)) if f1s else 0.0

    return metrics


def train_lstm_experiment(
    config: Dict[str, Any],
    dataset_dict: Dict[str, Any],
    device: torch.device,
    tracker: ExperimentTracker
) -> Dict[str, Any]:
    """
    Executes end-to-end training and evaluation of CryptoLSTM baseline.
    """
    features = dataset_dict["features"]
    targets = dataset_dict["targets_shock"]
    val_mask = dataset_dict["target_validity_mask"]
    act_mask = dataset_dict["active_mask"]
    splits = dataset_dict["split_indices"]
    horizons = dataset_dict.get("horizons", [1, 3, 6, 12, 24])

    train_indices = splits["train_indices"]
    val_indices = splits["val_indices"]
    test_indices = splits["test_indices"]

    m_cfg = config.get("model", {})
    t_cfg = config.get("training", {})
    d_cfg = config.get("data", {})

    lookback_window = d_cfg.get("lookback_window", 24)
    in_features = m_cfg.get("in_features", features.shape[-1])
    if in_features != features.shape[-1]:
        logger.warning(
            f"Configured in_features ({in_features}) does not match dataset feature dimension ({features.shape[-1]}). "
            f"Adapting in_features to {features.shape[-1]}."
        )
        in_features = features.shape[-1]
    hidden_dim = m_cfg.get("hidden_dim", 64)
    num_layers = m_cfg.get("num_layers", 2)
    out_horizons = len(horizons)
    dropout = m_cfg.get("dropout", 0.2)

    batch_size = t_cfg.get("batch_size", 64)
    lr = float(t_cfg.get("learning_rate", 1e-3))
    weight_decay = float(t_cfg.get("weight_decay", 1e-5))
    max_epochs = t_cfg.get("max_epochs", 15)
    patience = t_cfg.get("patience", 5)
    pos_weight_val = float(t_cfg.get("pos_weight", 10.0))

    model = CryptoLSTM(
        in_features=in_features,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        out_horizons=out_horizons,
        dropout=dropout
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val], device=device))

    early_stopping = EarlyStopping(patience=patience, min_delta=1e-4, mode="max")

    L = lookback_window
    offsets = torch.arange(-L, 0, dtype=torch.long, device=device)
    train_valid_targets = [t for t in train_indices if t >= L]
    train_target_tensor = torch.tensor(train_valid_targets, dtype=torch.long, device=device)
    num_train_samples = len(train_target_tensor)

    logger.info(f"Starting CryptoLSTM training: {num_train_samples} train timesteps, {max_epochs} epochs max...")

    best_val_auprc = -1.0
    best_state = None

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        model.train()
        perm = torch.randperm(num_train_samples, device=device)
        shuffled_targets = train_target_tensor[perm]

        total_loss = 0.0
        steps = 0

        for i in range(0, num_train_samples, batch_size):
            batch_t = shuffled_targets[i : i + batch_size]
            b_len = len(batch_t)
            indices = batch_t[:, None] + offsets[None, :]

            x = features[indices].permute(0, 2, 1, 3).reshape(b_len * 30, L, -1).to(device)
            y = targets[batch_t].float().reshape(b_len * 30, -1).to(device)
            mask = (val_mask[batch_t] & act_mask[batch_t].unsqueeze(-1)).reshape(b_len * 30, -1).to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits[mask], y[mask])
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            steps += 1

        avg_train_loss = total_loss / max(1, steps)

        # Validation evaluation
        val_metrics = evaluate_lstm(
            model=model,
            features=features,
            targets=targets,
            validity_mask=val_mask,
            active_mask=act_mask,
            target_indices=val_indices,
            horizons=horizons,
            lookback_window=L,
            batch_size=batch_size * 2,
            device=device
        )
        val_auprc = val_metrics["mean_auprc"]
        val_auroc = val_metrics["mean_auroc"]
        elapsed = time.time() - t0

        logger.info(
            f"Epoch {epoch:02d}/{max_epochs} [{elapsed:.1f}s]: "
            f"Train Loss={avg_train_loss:.4f}, Val Mean AUPRC={val_auprc:.4f}, Val Mean AUROC={val_auroc:.4f}"
        )

        if val_auprc > best_val_auprc:
            best_val_auprc = val_auprc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if early_stopping(val_auprc):
            logger.info(f"Early stopping triggered at epoch {epoch}.")
            break

    # Restore best checkpoint
    if best_state is not None:
        model.load_state_dict(best_state)

    # Save checkpoint
    ckpt_path = tracker.save_checkpoint(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "best_val_auprc": best_val_auprc,
            "config": config,
            "seed": tracker.seed
        },
        filename="best_lstm_checkpoint.pt"
    )

    # Final evaluation on test split
    test_metrics = evaluate_lstm(
        model=model,
        features=features,
        targets=targets,
        validity_mask=val_mask,
        active_mask=act_mask,
        target_indices=test_indices,
        horizons=horizons,
        lookback_window=L,
        batch_size=batch_size * 2,
        device=device
    )

    logger.info(
        f"Test Results: Mean AUPRC={test_metrics['mean_auprc']:.4f}, Mean AUROC={test_metrics['mean_auroc']:.4f}"
    )

    # Compile all logged metrics
    all_metrics = {
        "train_loss": float(avg_train_loss),
        "val_mean_auprc": float(best_val_auprc),
        "val_mean_auroc": float(val_metrics["mean_auroc"]),
        "test_mean_auprc": float(test_metrics["mean_auprc"]),
        "test_mean_auroc": float(test_metrics["mean_auroc"]),
        **{f"test_{k}": v for k, v in test_metrics.items()}
    }

    tracker.log_metrics(all_metrics)
    return all_metrics
