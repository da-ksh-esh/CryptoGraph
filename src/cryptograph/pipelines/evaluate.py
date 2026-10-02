"""Evaluation pipeline reading checkpoints and computing comprehensive test metrics."""

import argparse
from pathlib import Path
import json
import yaml
import torch
import numpy as np
import pandas as pd

from ..training.device import resolve_device
from ..training.metrics import compute_classification_metrics
from ..common.logger import get_logger

from ..models.tgn import CryptoTGN
from ..models.lstm import CryptoLSTM
from ..models.dygformer import PatchTemporalTransformer
from ..training.tgn_trainer import evaluate_tgn, build_topk_dynamic_edges
from ..training.lstm_trainer import evaluate_lstm
from ..training.dygformer_trainer import evaluate_dygformer

logger = get_logger(__name__)


def run_evaluation(
    config_path: str,
    checkpoint_path: str,
    dataset_path: str = "data/processed/v2/1h/processed_dataset.pt",
    network_features_path: str = "data/processed/v2/network_features.pt",
    output_path: str = None,
    device_name: str = "auto"
) -> dict:
    """
    Executes comprehensive model evaluation on the chronological test split.
    Loads trained model weights, runs causal evaluation with memory warmup (for TGN),
    and computes multi-horizon classification metrics.
    """
    device = resolve_device(device_name)
    logger.info(f"Loading checkpoint {checkpoint_path} onto {device}...")

    ckpt_file = Path(checkpoint_path)
    if not ckpt_file.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_file}")

    ckpt = torch.load(ckpt_file, map_location=device, weights_only=False)
    ckpt_config = ckpt.get("config", {})

    # Detect model type
    model_type = ckpt_config.get("model", {}).get("type")
    if not model_type:
        fname = ckpt_file.name.lower()
        if "tgn" in fname:
            model_type = "tgn"
        elif "dygformer" in fname or "transformer" in fname:
            model_type = "dygformer"
        elif "lstm" in fname:
            model_type = "lstm"
        else:
            model_type = "tgn"

    logger.info(f"Detected model architecture: {model_type.upper()}")

    # Load dataset
    data_file = Path(dataset_path)
    if not data_file.exists():
        raise FileNotFoundError(f"Dataset not found: {data_file}")

    data = torch.load(data_file, map_location="cpu", weights_only=False)
    features = data["features"]
    shocks = data["shocks"]
    targets = data["targets_shock"]
    val_mask = data["target_validity_mask"]
    act_mask = data["active_mask"]
    horizons = data.get("horizons", [1, 3, 6, 12, 24])
    splits = data["split_indices"]
    test_indices = splits["test_indices"]
    val_indices = splits["val_indices"]
    asset_names = data.get("asset_names", [f"Asset_{i}" for i in range(features.shape[1])])

    metrics = {}

    if model_type == "tgn":
        # Fuse network features if available
        net_file = Path(network_features_path)
        if net_file.exists():
            net_data = torch.load(net_file, map_location="cpu", weights_only=False)
            fused_features = torch.cat([features, net_data["network_features"]], dim=-1)
        else:
            fused_features = features

        model = CryptoTGN(
            num_nodes=len(asset_names),
            raw_msg_dim=9,
            node_feat_dim=fused_features.shape[-1],
            memory_dim=ckpt_config.get("model", {}).get("memory_dim", 64),
            embedding_dim=ckpt_config.get("model", {}).get("embedding_dim", 64),
            out_horizons=len(horizons)
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])

        returns_np = data["features"][:, :, 0].numpy()
        edge_indices, edge_weights = build_topk_dynamic_edges(returns_np, top_k=5, window=24)

        warmup_hist = val_indices[-168:] if len(val_indices) >= 168 else val_indices
        logger.info(f"Evaluating CryptoTGN across {len(test_indices)} test timesteps (with 168h historical warmup)...")
        metrics = evaluate_tgn(
            model=model,
            features=fused_features,
            shocks=shocks,
            targets=targets,
            validity_mask=val_mask,
            active_mask=act_mask,
            target_indices=test_indices,
            horizons=horizons,
            edge_indices=edge_indices,
            edge_weights=edge_weights,
            device=device,
            reset_memory=True,
            warmup_indices=warmup_hist
        )

    elif model_type in ("dygformer", "transformer"):
        model = PatchTemporalTransformer(
            in_features=features.shape[-1],
            patch_size=ckpt_config.get("model", {}).get("patch_size", 4),
            hidden_dim=ckpt_config.get("model", {}).get("hidden_dim", 64),
            num_heads=ckpt_config.get("model", {}).get("num_heads", 4),
            num_layers=ckpt_config.get("model", {}).get("num_layers", 2),
            out_horizons=len(horizons)
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])

        logger.info(f"Evaluating PatchTemporalTransformer across {len(test_indices)} test timesteps...")
        metrics = evaluate_dygformer(
            model=model,
            features=features,
            targets=targets,
            validity_mask=val_mask,
            active_mask=act_mask,
            target_indices=test_indices,
            horizons=horizons,
            lookback_window=24,
            device=device
        )

    else:  # LSTM
        model = CryptoLSTM(
            in_features=features.shape[-1],
            hidden_dim=ckpt_config.get("model", {}).get("hidden_dim", 64),
            num_layers=ckpt_config.get("model", {}).get("num_layers", 2),
            out_horizons=len(horizons)
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])

        logger.info(f"Evaluating CryptoLSTM across {len(test_indices)} test timesteps...")
        metrics = evaluate_lstm(
            model=model,
            features=features,
            targets=targets,
            validity_mask=val_mask,
            active_mask=act_mask,
            target_indices=test_indices,
            horizons=horizons,
            lookback_window=24,
            device=device
        )

    # Display results
    print("\n" + "=" * 70)
    print(f"  MODEL TEST EVALUATION: {model_type.upper()}")
    print("=" * 70)
    print(f"  Mean Test AUPRC: {metrics.get('mean_auprc', 0.0):.4f}")
    print(f"  Mean Test AUROC: {metrics.get('mean_auroc', 0.0):.4f}")
    print(f"  Mean Test F1:    {metrics.get('mean_f1', 0.0):.4f}")
    print("-" * 70)
    print(f"{'Horizon':<12} {'AUPRC':<12} {'AUROC':<12} {'F1':<12} {'Precision':<12} {'Recall':<12}")
    print("-" * 70)
    for h in horizons:
        auprc = metrics.get(f"auprc_{h}h", 0.0)
        auroc = metrics.get(f"auroc_{h}h", 0.0)
        f1 = metrics.get(f"f1_{h}h", 0.0)
        prec = metrics.get(f"precision_{h}h", 0.0)
        rec = metrics.get(f"recall_{h}h", 0.0)
        print(f"{h:02d}h          {auprc:<12.4f} {auroc:<12.4f} {f1:<12.4f} {prec:<12.4f} {rec:<12.4f}")
    print("=" * 70 + "\n")

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        if out_p.suffix == ".json":
            with open(out_p, "w") as f:
                json.dump(metrics, f, indent=2)
        else:
            rows = [{"horizon_hours": h, **{k.split(f"_{h}h")[0]: v for k, v in metrics.items() if f"_{h}h" in k}} for h in horizons]
            pd.DataFrame(rows).to_csv(out_p, index=False)
        logger.info(f"Saved evaluation metrics to {out_p}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="CryptoGraph Standalone Evaluation CLI")
    parser.add_argument("--config", type=str, default="configs/model/tgn.yaml", help="Path to config YAML.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint .pt file.")
    parser.add_argument("--dataset-path", type=str, default="data/processed/v2/1h/processed_dataset.pt", help="Path to dataset.")
    parser.add_argument("--network-features-path", type=str, default="data/processed/v2/network_features.pt", help="Path to network features.")
    parser.add_argument("--output-file", type=str, default=None, help="Optional output path for evaluation metrics.")
    parser.add_argument("--device", type=str, default="auto", help="Compute device: auto | cpu | cuda.")
    args = parser.parse_args()

    run_evaluation(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        dataset_path=args.dataset_path,
        network_features_path=args.network_features_path,
        output_path=args.output_file,
        device_name=args.device
    )


if __name__ == "__main__":
    main()
