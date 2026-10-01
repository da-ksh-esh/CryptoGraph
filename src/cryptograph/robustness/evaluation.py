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

    # Train Temporal GAE on training set
    gae = TemporalGAE(in_features=F, hidden_dim=64, latent_dim=32)
    optimizer = torch.optim.Adam(gae.parameters(), lr=1e-3)

    logger.info("Training TemporalGAE on train split for anomaly calibration...")
    gae.train()
    # Train over a representative sample of 1000 train timesteps
    sample_idxs = train_idxs[:1000]
    for _ in range(5):
        for t in sample_idxs:
            x_t = features[t]
            optimizer.zero_grad()
            x_recon, adj_recon, _ = gae(x_t)
            loss = torch.mean((x_t - x_recon) ** 2)
            loss.backward()
            optimizer.step()

    # Score historical events
    gae.eval()
    events_df = pd.read_csv(events_csv_path)
    records = []

    with torch.no_grad():
        # Baseline normal reconstruction error
        normal_errors = []
        for t in train_idxs[:200]:
            x_t = features[t]
            x_rec, _, _ = gae(x_t)
            normal_errors.append(float(torch.mean((x_t - x_rec) ** 2).item()))
        baseline_err = float(np.mean(normal_errors))

        for _, row in events_df.iterrows():
            ev_name = str(row.get("name", row.get("event", "Event")))
            ev_date = str(row["date"])

            matching = [idx for idx, ts in enumerate(timestamps) if ts.startswith(ev_date)]
            if not matching:
                continue

            t_ev = matching[0]
            x_ev = features[t_ev]
            x_rec, _, _ = gae(x_ev)
            per_node_err = torch.mean((x_ev - x_rec) ** 2, dim=-1).cpu().numpy()
            event_mean_err = float(np.mean(per_node_err))

            # Asset with highest reconstruction surge
            top_asset_idx = int(np.argmax(per_node_err))
            top_asset = asset_names[top_asset_idx]

            anomaly_ratio = event_mean_err / max(1e-6, baseline_err)

            records.append({
                "event_name": ev_name,
                "event_date": ev_date,
                "timestamp": timestamps[t_ev],
                "baseline_reconstruction_error": baseline_err,
                "event_reconstruction_error": event_mean_err,
                "anomaly_ratio": anomaly_ratio,
                "most_anomalous_asset": top_asset,
                "is_statistical_anomaly": bool(anomaly_ratio >= 1.5)
            })

    res_df = pd.DataFrame(records)
    Path(output_csv_path).parent.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(output_csv_path, index=False)
    logger.info(f"Saved T-GAE anomaly evaluation to {output_csv_path}.")
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
