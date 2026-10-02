"""Evaluation script for the Independent Cascade (IC) propagation baseline.
Fits edge activation probabilities strictly on train cascades and evaluates AUPRC / AUROC on val and test.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

from cryptograph.network.cascade import IndependentCascadeModel
from cryptograph.training.metrics import compute_classification_metrics


def build_cascade_graph(correlation_matrix: np.ndarray, asset_names: list, threshold: float = 0.35) -> nx.Graph:
    G = nx.Graph()
    for a in asset_names:
        G.add_node(a)
    n = len(asset_names)
    for i in range(n):
        for j in range(i + 1, n):
            c = abs(correlation_matrix[i, j])
            if c >= threshold:
                G.add_edge(asset_names[i], asset_names[j], weight=float(c))
    return G


def evaluate_split(
    model: IndependentCascadeModel,
    G: nx.Graph,
    shocks: np.ndarray,
    targets: np.ndarray,
    validity_mask: np.ndarray,
    split_indices: np.ndarray,
    asset_names: list,
    horizon_idx: int = 2  # default 6h
) -> dict:
    all_y_true = []
    all_y_pred = []

    for t in split_indices:
        shocked_indices = np.where(shocks[t])[0]
        if len(shocked_indices) == 0:
            continue

        seed_nodes = {asset_names[i] for i in shocked_indices}

        # Run Monte Carlo simulations to get activation probability per node
        n_mc = 20
        activation_counts = {a: 0 for a in asset_names}
        for _ in range(n_mc):
            active_set = model.simulate(G, seed_nodes)
            for a in active_set:
                activation_counts[a] += 1

        for idx, a in enumerate(asset_names):
            if idx in shocked_indices:
                continue  # Exclude seed nodes from contagion receiver scoring
            if not validity_mask[t, idx, horizon_idx]:
                continue

            y_true = int(targets[t, idx, horizon_idx])
            y_score = activation_counts[a] / n_mc

            all_y_true.append(y_true)
            all_y_pred.append(y_score)

    if not all_y_true or sum(all_y_true) == 0:
        return {
            "auroc": 0.5,
            "auprc": 0.0,
            "f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "samples": len(all_y_true),
            "positives": sum(all_y_true),
        }

    m = compute_classification_metrics(np.array(all_y_true), np.array(all_y_pred), threshold=0.5)

    return {
        "auroc": m["auroc"],
        "auprc": m["auprc"],
        "f1": m["f1"],
        "precision": m["precision"],
        "recall": m["recall"],
        "samples": len(all_y_true),
        "positives": int(sum(all_y_true))
    }


def main():
    dataset_path = Path("data/processed/v2/1h/processed_dataset.pt")
    data = torch.load(dataset_path, map_location="cpu", weights_only=False)

    assets = data["asset_names"]
    shocks = data["shocks"].numpy()
    targets = data["targets_shock"].numpy()
    validity = data["target_validity_mask"].numpy()
    split_indices = data["split_indices"]

    train_idx = np.array(split_indices["train_indices"])
    val_idx = np.array(split_indices["val_indices"])
    test_idx = np.array(split_indices["test_indices"])

    # 1. Extract cascades on train partition
    train_cascades = []
    for t in train_idx:
        shocked = [assets[i] for i in np.where(shocks[t])[0]]
        if len(shocked) >= 2:
            train_cascades.append(shocked)

    # 2. Build graph from training correlation
    close = data["close_prices"].numpy()
    train_close = close[train_idx]
    train_ret = np.diff(np.log(train_close), axis=0)
    train_corr = np.corrcoef(train_ret, rowvar=False)

    G = build_cascade_graph(train_corr, assets, threshold=0.35)

    # 3. Fit Independent Cascade model strictly on train
    ic_model = IndependentCascadeModel(default_p=0.15, seed=42)
    ic_model.fit_probabilities(train_cascades, G)

    print(f"Fitted Independent Cascade on {len(train_cascades)} train cascades across {G.number_of_edges()} edges.")

    # 4. Evaluate on Val and Test splits across horizons
    horizons = data["horizons"]
    records = []

    for h_idx, h in enumerate(horizons):
        val_metrics = evaluate_split(ic_model, G, shocks, targets, validity, val_idx, assets, horizon_idx=h_idx)
        test_metrics = evaluate_split(ic_model, G, shocks, targets, validity, test_idx, assets, horizon_idx=h_idx)

        records.append({
            "model": "IndependentCascade",
            "horizon_hours": h,
            "val_auroc": val_metrics["auroc"],
            "val_auprc": val_metrics["auprc"],
            "val_f1": val_metrics["f1"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_samples": val_metrics["samples"],
            "val_positives": val_metrics["positives"],
            "test_auroc": test_metrics["auroc"],
            "test_auprc": test_metrics["auprc"],
            "test_f1": test_metrics["f1"],
            "test_precision": test_metrics["precision"],
            "test_recall": test_metrics["recall"],
            "test_samples": test_metrics["samples"],
            "test_positives": test_metrics["positives"],
        })

    df_out = pd.DataFrame(records)
    rep_path = Path("reports/cascade_baseline.csv")
    rep_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(rep_path, index=False)
    print("=== CASCADE BASELINE PERFORMANCE ===")
    print(df_out[["horizon_hours", "val_auroc", "val_auprc", "val_f1", "test_auroc", "test_auprc", "test_f1"]].to_string(index=False))
    print(f"Saved results to {rep_path}")


if __name__ == "__main__":
    main()
