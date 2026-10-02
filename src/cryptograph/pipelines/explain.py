"""Explanation generation pipeline reading models and computing spatio-temporal edge attributions."""

import argparse
from pathlib import Path
import json
import glob
import torch
import pandas as pd

from ..models.tgn import CryptoTGN
from ..explain.tempme import TempME, TemporalEdgeMaskExplainer
from ..explain.evaluate_explanations import run_explanation_evaluation
from ..training.device import resolve_device
from ..common.logger import get_logger

logger = get_logger(__name__)


def run_single_node_explanation(
    checkpoint_path: str,
    target_idx: int = 0,
    timestep: int = 5208,
    dataset_path: str = "data/processed/v2/1h/processed_dataset.pt",
    network_features_path: str = "data/processed/v2/network_features.pt",
    device_name: str = "auto"
):
    device = resolve_device(device_name)
    logger.info(f"Generating explanation for node {target_idx} at timestep {timestep} using {checkpoint_path} on {device}")

    data = torch.load(dataset_path, map_location="cpu", weights_only=False)
    net_data = torch.load(network_features_path, map_location="cpu", weights_only=False)
    features = torch.cat([data["features"], net_data["network_features"]], dim=-1)
    assets = data["asset_names"]

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = CryptoTGN(
        num_nodes=len(assets),
        raw_msg_dim=9,
        node_feat_dim=features.shape[-1],
        memory_dim=64,
        embedding_dim=64,
        out_horizons=len(data.get("horizons", [1, 3, 6, 12, 24]))
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Dynamic Top-5 edges at timestep
    from ..training.tgn_trainer import build_topk_dynamic_edges
    w_start = max(0, timestep - 24)
    r_slice = data["features"][w_start : timestep + 1, :, 0].numpy()
    e_indices, e_weights = build_topk_dynamic_edges(r_slice, top_k=5, window=24)
    e_idx = e_indices[-1].to(device)
    e_w = e_weights[-1].to(device) if e_weights[-1].numel() > 0 else None

    explainer = TempME(model, num_epochs=20, lr=0.05)
    exp = explainer.explain(target_idx, features[timestep].to(device), e_idx, e_w)

    print(f"\n=== Spatio-Temporal Explanation for Asset: {assets[target_idx]} (t={timestep}) ===")
    print(f"Sparsity: {exp['sparsity']:.2%}")
    print(f"Entropy:  {exp['motif_entropy']:.4f}")
    print("\nTop Explanatory Transmission Channels:")
    for edge in exp["top_edges"]:
        src_name = assets[edge["source"]]
        dst_name = assets[edge["target"]]
        print(f"  {src_name:>6} -> {dst_name:<6} | Importance: {edge['importance']:.4f}")
    print()
    return exp


def main():
    parser = argparse.ArgumentParser(description="Explanation runner CLI.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to TGN model checkpoint.")
    parser.add_argument("--dataset-path", type=str, default="data/processed/v2/1h/processed_dataset.pt")
    parser.add_argument("--network-features-path", type=str, default="data/processed/v2/network_features.pt")
    parser.add_argument("--target-idx", type=int, default=0, help="Node index to explain.")
    parser.add_argument("--timestep", type=int, default=5208, help="Timestep to explain.")
    parser.add_argument("--batch", action="store_true", help="Run full evaluation across crisis sample events.")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    ckpt_path = args.checkpoint
    if ckpt_path is None:
        ckpts = sorted(glob.glob("runs/*tgn*/best_tgn_checkpoint.pt"))
        if ckpts:
            ckpt_path = ckpts[-1]
        else:
            raise FileNotFoundError("No trained TGN checkpoint found in runs/.")

    if args.batch:
        logger.info("Executing batch explanation evaluation across crisis events...")
        metrics = run_explanation_evaluation(
            checkpoint_path=ckpt_path,
            dataset_path=args.dataset_path,
            network_features_path=args.network_features_path,
            output_dir="reports"
        )
        print("\nBatch Explanation Validation Summary:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
    else:
        run_single_node_explanation(
            checkpoint_path=ckpt_path,
            target_idx=args.target_idx,
            timestep=args.timestep,
            dataset_path=args.dataset_path,
            network_features_path=args.network_features_path,
            device_name=args.device
        )


if __name__ == "__main__":
    main()
