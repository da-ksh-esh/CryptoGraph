"""Script to execute the Phase 10 Network Compression research experiment with downstream model evaluation."""

from pathlib import Path
import glob
import numpy as np
import pandas as pd
import torch
import networkx as nx

from cryptograph.network.compression import NetworkCompressor
from cryptograph.models.tgn import CryptoTGN
from cryptograph.training.tgn_trainer import evaluate_tgn, build_topk_dynamic_edges


def build_mst_dynamic_edges(returns: np.ndarray, window: int = 24):
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
        np.fill_diagonal(corr, 0.0)
        dist = np.sqrt(2.0 * np.clip(1.0 - corr, 0.0, 2.0))

        G = nx.Graph()
        for i in range(N):
            G.add_node(i)
        for i in range(N):
            for j in range(i + 1, N):
                G.add_edge(i, j, weight=float(dist[i, j]))

        mst = nx.minimum_spanning_tree(G)
        src = [e[0] for e in mst.edges()]
        dst = [e[1] for e in mst.edges()]
        weights = [float(abs(corr[e[0], e[1]])) for e in mst.edges()]

        e_idx = torch.tensor(np.vstack([src + dst, dst + src]), dtype=torch.long)
        e_w = torch.tensor(weights + weights, dtype=torch.float32)
        edge_indices.append(e_idx)
        edge_weights.append(e_w)

    return edge_indices, edge_weights


def main():
    dataset_path = Path("data/processed/v2/1h/processed_dataset.pt")
    net_path = Path("data/processed/v2/network_features.pt")
    data = torch.load(dataset_path, map_location="cpu", weights_only=False)
    net_data = torch.load(net_path, map_location="cpu", weights_only=False)
    features = torch.cat([data["features"], net_data["network_features"]], dim=-1)

    assets = data["asset_names"]
    test_indices = data["split_indices"]["test_indices"]
    shocks = data["shocks"]
    targets = data["targets_shock"]
    val_mask = data["target_validity_mask"]
    act_mask = data["active_mask"]
    horizons = data["horizons"]
    returns_np = data["features"][:, :, 0].numpy()

    # Load trained TGN model
    tgn_ckpts = sorted(glob.glob("runs/*tgn*/best_tgn_checkpoint.pt"))
    if not tgn_ckpts:
        raise FileNotFoundError("No TGN checkpoint found in runs/.")
    ckpt_file = tgn_ckpts[-1]
    ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)

    model = CryptoTGN(
        num_nodes=len(assets),
        raw_msg_dim=9,
        node_feat_dim=features.shape[-1],
        memory_dim=64,
        embedding_dim=64,
        out_horizons=len(horizons)
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Correlation matrix over frozen test period
    test_ret = returns_np[test_indices]
    corr = np.corrcoef(test_ret, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)

    print("Running structural compression benchmark...")
    df_structural = NetworkCompressor.run_benchmark(corr, assets, seed=42)

    # Now evaluate downstream model performance under each representation
    print("Evaluating downstream TGN contagion prediction performance...")
    configs = [
        ("Full_Graph", 29),
        ("TopK_8", 8),
        ("TopK_5", 5),
        ("TopK_3", 3),
        ("MST_Backbone", "mst")
    ]

    model_auprcs = {}
    model_aurocs = {}

    for name, k_val in configs:
        print(f"  Testing downstream representation: {name}...")
        if k_val == "mst":
            e_idx, e_w = build_mst_dynamic_edges(returns_np, window=24)
        else:
            e_idx, e_w = build_topk_dynamic_edges(returns_np, top_k=k_val, window=24)

        eval_res = evaluate_tgn(
            model=model,
            features=features,
            shocks=shocks,
            targets=targets,
            validity_mask=val_mask,
            active_mask=act_mask,
            target_indices=test_indices,
            horizons=horizons,
            edge_indices=e_idx,
            edge_weights=e_w,
            reset_memory=True
        )
        model_auprcs[name] = round(eval_res["mean_auprc"], 4)
        model_aurocs[name] = round(eval_res["mean_auroc"], 4)
        print(f"    -> Test Mean AUPRC: {eval_res['mean_auprc']:.4f}, Mean AUROC: {eval_res['mean_auroc']:.4f}")

    # Map into dataframe
    df_structural["downstream_test_auprc"] = df_structural["representation"].map(model_auprcs)
    df_structural["downstream_test_auroc"] = df_structural["representation"].map(model_aurocs)

    rep_path = Path("reports/network_compression_experiment.csv")
    rep_path.parent.mkdir(parents=True, exist_ok=True)
    df_structural.to_csv(rep_path, index=False)

    print("\n=== NETWORK COMPRESSION & MODEL PERFORMANCE BENCHMARK ===")
    print(df_structural[["representation", "num_edges", "edge_reduction_pct", "spearman_pagerank", "downstream_test_auprc", "downstream_test_auroc"]].to_string(index=False))
    print(f"\nSaved updated compression report to {rep_path}")


if __name__ == "__main__":
    main()
