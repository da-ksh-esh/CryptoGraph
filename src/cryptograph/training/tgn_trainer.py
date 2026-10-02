"""Continuous-time dynamic graph training and evaluation harness for CryptoTGN."""

from typing import Dict, Any, List, Optional, Tuple
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
import numpy as np

from ..models.tgn import CryptoTGN
from ..training.losses import WeightedBCELoss
from ..training.metrics import compute_classification_metrics
from ..training.early_stopping import EarlyStopping
from ..training.tracking import ExperimentTracker
from ..common.logger import get_logger

logger = get_logger(__name__)


def build_topk_dynamic_edges(
    returns: np.ndarray,
    top_k: int = 5,
    window: int = 24
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Constructs dynamic Top-K correlation edges for each timestep t >= window.
    Returns:
        edge_indices: list of torch.LongTensor (2, E)
        edge_weights: list of torch.FloatTensor (E,)
    """
    T, N = returns.shape
    edge_indices = []
    edge_weights = []

    for t in range(T):
        if t < window:
            edge_indices.append(torch.empty((2, 0), dtype=torch.long))
            edge_weights.append(torch.empty((0,), dtype=torch.float32))
            continue

        r_slice = returns[t - window : t]
        stds = np.std(r_slice, axis=0)
        valid = stds > 1e-6
        if np.sum(valid) < 2:
            edge_indices.append(torch.empty((2, 0), dtype=torch.long))
            edge_weights.append(torch.empty((0,), dtype=torch.float32))
            continue

        corr = np.corrcoef(r_slice, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0)
        np.fill_diagonal(corr, -1.0)

        # For each node, take top_k peers
        k = min(top_k, N - 1)
        top_peers = np.argsort(corr, axis=1)[:, -k:]
        src = np.repeat(np.arange(N), k)
        dst = top_peers.flatten()
        weights = corr[src, dst]

        # Filter strictly non-negative correlation edges
        pos_mask = weights > 0.0
        src = src[pos_mask]
        dst = dst[pos_mask]
        weights = weights[pos_mask]

        if len(src) == 0:
            edge_indices.append(torch.empty((2, 0), dtype=torch.long))
            edge_weights.append(torch.empty((0,), dtype=torch.float32))
        else:
            e_idx = torch.tensor(np.vstack([src, dst]), dtype=torch.long)
            e_w = torch.tensor(weights, dtype=torch.float32)
            edge_indices.append(e_idx)
            edge_weights.append(e_w)

    return edge_indices, edge_weights


def evaluate_tgn(
    model: CryptoTGN,
    features: torch.Tensor,
    shocks: torch.Tensor,
    targets: torch.Tensor,
    validity_mask: torch.Tensor,
    active_mask: torch.Tensor,
    target_indices: List[int],
    horizons: List[int],
    edge_indices: List[torch.Tensor],
    edge_weights: List[torch.Tensor],
    device: torch.device = torch.device("cpu"),
    reset_memory: bool = False
) -> Dict[str, float]:
    """
    Evaluates CryptoTGN causally across target_indices without lookahead leakage.

    NOTE ON EVENT CONSTRUCTION:
    Interaction events in this framework are SYNTHETIC SHOCK-TRIGGERED CONTAGION BROADCASTS,
    not individual microstructural exchange trade executions. Whenever an asset exceeds
    its rolling volatility shock threshold (shocks[t, s] = True), synthetic broadcast messages
    (s -> d) are created to all active peer assets d != s.

    MEMORY LEAKAGE CONTROL:
    If reset_memory=True, internal node memory s_i(t) is reinitialized to zero before the split.
    During evaluation, memory evolves strictly causally: at timestep t, predictions are generated
    using memory states s_i(t) BEFORE the current shock interaction is incorporated into memory.
    """
    model.eval()
    if reset_memory:
        model.reset_memory()

    all_scores = [[] for _ in range(len(horizons))]
    all_trues = [[] for _ in range(len(horizons))]

    with torch.no_grad():
        for t in target_indices:
            # 1. Spatial forward pass using current memory s_i(t) (BEFORE shock update at t)
            e_idx = edge_indices[t].to(device)
            e_w = edge_weights[t].to(device) if edge_weights[t].numel() > 0 else None
            x_t = features[t].to(device)

            preds, _ = model(x_t, e_idx, e_w)
            probs = torch.sigmoid(preds)

            mask_t = (validity_mask[t] & active_mask[t].unsqueeze(-1)).to(device)
            y_t = targets[t].float().to(device)

            for k in range(len(horizons)):
                valid_k = mask_t[:, k]
                if valid_k.sum() > 0:
                    all_scores[k].append(probs[:, k][valid_k].cpu().numpy())
                    all_trues[k].append(y_t[:, k][valid_k].cpu().numpy())

            # 2. Causal memory update: process synthetic shock broadcast occurring at t AFTER predicting t
            shocked = torch.where(shocks[t] & active_mask[t])[0]
            active = torch.where(active_mask[t])[0]

            if len(shocked) > 0 and len(active) > 1:
                src_list, dst_list = [], []
                for s in shocked:
                    for d in active:
                        if d != s:
                            src_list.append(s)
                            dst_list.append(d)
                if len(src_list) > 0:
                    src = torch.tensor(src_list, dtype=torch.long, device=device)
                    dst = torch.tensor(dst_list, dtype=torch.long, device=device)
                    ts = torch.full((len(src_list),), t, dtype=torch.long, device=device)
                    msg = features[t, src, :model.raw_msg_dim].to(device)
                    model.update_memory(src, dst, msg, ts)

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


def train_tgn_experiment(
    config: Dict[str, Any],
    dataset_dict: Dict[str, Any],
    device: torch.device,
    tracker: ExperimentTracker,
    network_features_dict: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Executes end-to-end continuous-time training and evaluation of CryptoTGN model.
    """
    features = dataset_dict["features"] # (T, N, F)
    shocks = dataset_dict["shocks"]     # (T, N)
    targets = dataset_dict["targets_shock"] # (T, N, K)
    val_mask = dataset_dict["target_validity_mask"]
    act_mask = dataset_dict["active_mask"]
    splits = dataset_dict["split_indices"]
    horizons = dataset_dict.get("horizons", [1, 3, 6, 12, 24])

    # Optional concatenation of causal network features
    if network_features_dict is not None and "network_features" in network_features_dict:
        net_feats = network_features_dict["network_features"] # (T, N, 7)
        features = torch.cat([features, net_feats], dim=-1)
        logger.info(f"Fused node technical features with causal network features -> shape: {features.shape}")

    train_indices = splits["train_indices"]
    val_indices = splits["val_indices"]
    test_indices = splits["test_indices"]

    m_cfg = config.get("model", {})
    t_cfg = config.get("training", {})

    num_nodes = m_cfg.get("num_nodes", features.shape[1])
    node_feat_dim = features.shape[-1]
    raw_msg_dim = m_cfg.get("raw_msg_dim", 9)
    memory_dim = m_cfg.get("memory_dim", 64)
    time_dim = m_cfg.get("time_dim", 16)
    embedding_dim = m_cfg.get("embedding_dim", 64)
    out_horizons = len(horizons)
    dropout = m_cfg.get("dropout", 0.2)

    lr = float(t_cfg.get("learning_rate", 5e-4))
    weight_decay = float(t_cfg.get("weight_decay", 1e-5))
    max_epochs = t_cfg.get("max_epochs", 10)
    patience = t_cfg.get("patience", 4)
    pos_weight_val = float(t_cfg.get("pos_weight", 10.0))
    grad_accum_steps = t_cfg.get("grad_accum_steps", 32)

    # 1. Build Top-5 Dynamic Correlation Network Edges
    logger.info("Building dynamic Top-5 correlation edges for spatial convolution...")
    returns_np = dataset_dict["features"][:, :, 0].numpy()
    edge_indices, edge_weights = build_topk_dynamic_edges(returns_np, top_k=5, window=24)
    logger.info(f"Built {len(edge_indices)} dynamic graph snapshots.")

    model = CryptoTGN(
        num_nodes=num_nodes,
        raw_msg_dim=raw_msg_dim,
        node_feat_dim=node_feat_dim,
        memory_dim=memory_dim,
        time_dim=time_dim,
        embedding_dim=embedding_dim,
        out_horizons=out_horizons,
        dropout=dropout
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val], device=device))
    early_stopping = EarlyStopping(patience=patience, min_delta=1e-4, mode="max")

    logger.info(f"Starting CryptoTGN dynamic training: {len(train_indices)} train timesteps, {max_epochs} epochs max...")

    best_val_auprc = -1.0
    best_state = None

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        model.train()
        model.reset_memory()

        total_loss = 0.0
        steps = 0
        optimizer.zero_grad()

        for step_i, t in enumerate(train_indices):
            # Dynamic memory update on shock occurrences
            shocked = torch.where(shocks[t] & act_mask[t])[0]
            active = torch.where(act_mask[t])[0]

            if len(shocked) > 0 and len(active) > 1:
                src_list, dst_list = [], []
                for s in shocked:
                    for d in active:
                        if d != s:
                            src_list.append(s)
                            dst_list.append(d)
                if len(src_list) > 0:
                    src = torch.tensor(src_list, dtype=torch.long, device=device)
                    dst = torch.tensor(dst_list, dtype=torch.long, device=device)
                    ts = torch.full((len(src_list),), t, dtype=torch.long, device=device)
                    msg = features[t, src, :raw_msg_dim].to(device)
                    model.update_memory(src, dst, msg, ts)

            # Spatial message passing
            e_idx = edge_indices[t].to(device)
            e_w = edge_weights[t].to(device) if edge_weights[t].numel() > 0 else None
            x_t = features[t].to(device)

            preds, _ = model(x_t, e_idx, e_w)
            mask_t = (val_mask[t] & act_mask[t].unsqueeze(-1)).to(device)
            y_t = targets[t].float().to(device)

            if mask_t.sum() > 0:
                loss = criterion(preds[mask_t], y_t[mask_t]) / grad_accum_steps
                loss.backward()
                total_loss += loss.item() * grad_accum_steps
                steps += 1

            if (step_i + 1) % grad_accum_steps == 0 or (step_i + 1) == len(train_indices):
                optimizer.step()
                optimizer.zero_grad()

        avg_train_loss = total_loss / max(1, steps)

        # Validation evaluation causally continuing forward
        val_metrics = evaluate_tgn(
            model=model,
            features=features,
            shocks=shocks,
            targets=targets,
            validity_mask=val_mask,
            active_mask=act_mask,
            target_indices=val_indices,
            horizons=horizons,
            edge_indices=edge_indices,
            edge_weights=edge_weights,
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
    tracker.save_checkpoint(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "best_val_auprc": best_val_auprc,
            "config": config,
            "seed": tracker.seed
        },
        filename="best_tgn_checkpoint.pt"
    )

    # Final evaluation on test split with explicit memory reset to prevent cross-split leakage
    test_metrics = evaluate_tgn(
        model=model,
        features=features,
        shocks=shocks,
        targets=targets,
        validity_mask=val_mask,
        active_mask=act_mask,
        target_indices=test_indices,
        horizons=horizons,
        edge_indices=edge_indices,
        edge_weights=edge_weights,
        device=device,
        reset_memory=True
    )

    logger.info(
        f"TGN Test Results: Mean AUPRC={test_metrics['mean_auprc']:.4f}, Mean AUROC={test_metrics['mean_auroc']:.4f}"
    )

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
