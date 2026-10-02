"""Security and Robustness evaluation harness: T-GAE anomaly scoring and perturbation degradation curves."""

from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from ..models.tgae import TemporalGAE
from ..robustness.anomaly import AnomalyScorer
from ..robustness.attacks import AdversarialAttacker
from ..robustness.perturbation import DualPerturbationTester
from ..explain.tempme import TempME
from ..common.logger import get_logger

logger = get_logger(__name__)


def evaluate_tgae_anomalies(
    dataset_path: str = "data/processed/v2/1h/processed_dataset.pt",
    events_csv_path: str = "data/external/events.csv",
    output_csv_path: str = "reports/tgae_anomalies.csv"
) -> pd.DataFrame:
    """
    Trains an unsupervised Temporal GAE on normal training hours and measures
    reconstruction error surges across historical stress events without inventing synthetic labels.
    """
    data = torch.load(dataset_path, weights_only=False)
    features = data["features"] # (T, N, F)
    timestamps = [str(ts) for ts in data["timestamps"]]
    splits = data["split_indices"]
    asset_names = data["asset_names"]
    train_idxs = splits["train_indices"]

    T, N, F = features.shape

    from ..training.tgn_trainer import build_topk_dynamic_edges
    returns_np = features[:, :, 0].numpy()
    edge_indices, _ = build_topk_dynamic_edges(returns_np, top_k=5, window=24)

    # Train Spatio-Temporal GAE on training set
    gae = TemporalGAE(in_features=F, hidden_dim=64, latent_dim=32)
    optimizer = torch.optim.Adam(gae.parameters(), lr=1e-3)

    logger.info("Training Spatio-Temporal GAE on train split with joint feature + adjacency loss...")
    gae.train()
    sample_idxs = train_idxs[:500]
    h_prev = None
    for _ in range(3):
        for t in sample_idxs:
            x_t = features[t]
            e_idx = edge_indices[t]
            adj_target = torch.zeros((N, N), dtype=torch.float32)
            if e_idx.numel() > 0:
                adj_target[e_idx[0], e_idx[1]] = 1.0
            np.fill_diagonal(adj_target.numpy(), 1.0)

            optimizer.zero_grad()
            x_recon, adj_recon, h_prev = gae(x_t, e_idx, h_prev.detach() if h_prev is not None else None)
            loss = gae.loss_function(x_t, x_recon, adj_target, adj_recon, alpha=0.7)
            loss.backward()
            optimizer.step()

    # Score historical events
    gae.eval()
    events_df = pd.read_csv(events_csv_path)
    records = []

    from scipy import stats

    with torch.no_grad():
        # Baseline normal reconstruction error and variance
        normal_errors = []
        for t in train_idxs[:500]:
            x_t = features[t]
            e_idx = edge_indices[t]
            x_rec, _, _ = gae(x_t, e_idx)
            normal_errors.append(float(torch.mean((x_t - x_rec) ** 2).item()))
        baseline_err = float(np.mean(normal_errors))
        baseline_std = float(np.std(normal_errors, ddof=1))

        for _, row in events_df.iterrows():
            ev_name = str(row.get("name", row.get("event", "Event")))
            ev_date = str(row["date"])

            matching = [idx for idx, ts in enumerate(timestamps) if ts.startswith(ev_date)]
            if not matching:
                continue

            t_ev = matching[0]
            x_ev = features[t_ev]
            e_idx_ev = edge_indices[t_ev]
            x_rec, _, _ = gae(x_ev, e_idx_ev)
            per_node_err = torch.mean((x_ev - x_rec) ** 2, dim=-1).cpu().numpy()
            event_mean_err = float(np.mean(per_node_err))

            # Asset with highest reconstruction surge
            top_asset_idx = int(np.argmax(per_node_err))
            top_asset = asset_names[top_asset_idx]

            anomaly_ratio = event_mean_err / max(1e-6, baseline_err)
            z_score = float((event_mean_err - baseline_err) / max(1e-6, baseline_std))
            p_val = float(1.0 - stats.norm.cdf(z_score))
            is_stat_anom = bool(p_val < 0.01)

            records.append({
                "event_name": ev_name,
                "event_date": ev_date,
                "timestamp": timestamps[t_ev],
                "baseline_reconstruction_error": baseline_err,
                "baseline_std": baseline_std,
                "event_reconstruction_error": event_mean_err,
                "anomaly_ratio": anomaly_ratio,
                "z_score": z_score,
                "p_value": p_val,
                "most_anomalous_asset": top_asset,
                "is_statistical_anomaly": is_stat_anom
            })

    res_df = pd.DataFrame(records)
    Path(output_csv_path).parent.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(output_csv_path, index=False)
    logger.info(f"Saved T-GAE anomaly evaluation with rigorous p-values to {output_csv_path}.")
    return res_df


def evaluate_perturbation_degradation(
    model: nn.Module,
    features: torch.Tensor,
    edge_index: torch.Tensor,
    target_node_idx: int = 0,
    epsilon_levels: List[float] = [0.01, 0.05, 0.10, 0.20, 0.35],
    output_csv_path: str = "reports/robustness_degradation.csv"
) -> pd.DataFrame:
    """
    Evaluates adversarial resilience and degradation curves under feature and topology noise.
    """
    explainer = TempME(model, num_epochs=5, lr=0.1)
    tester = DualPerturbationTester(tgn_model=model, tempme_explainer=explainer)

    results = tester.run_stress_test(
        node_features=features,
        edge_index=edge_index,
        target_idx=target_node_idx,
        epsilon_levels=epsilon_levels
    )

    df_deg = pd.DataFrame(results)
    Path(output_csv_path).parent.mkdir(parents=True, exist_ok=True)
    df_deg.to_csv(output_csv_path, index=False)
    logger.info(f"Saved perturbation degradation curves to {output_csv_path}.")
    return df_deg
