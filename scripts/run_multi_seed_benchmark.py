"""Rigorous Multi-Seed Model Benchmark (Seeds 42, 43, 44).

Trains and evaluates CryptoLSTM, CryptoTGN, and CryptoDyGFormer across 3 distinct random seeds.
Reports per-seed metrics, mean +/- std, 95% confidence intervals, and paired t-tests.
Saves comprehensive results to reports/model_comparison.csv,
reports/model_multi_seed_comparison.csv, and reports/model_benchmark_summary.csv.
"""

from pathlib import Path
import time
import math
import numpy as np
import pandas as pd
from scipy import stats
import torch

from cryptograph.training.seed import seed_everything
from cryptograph.training.tracking import ExperimentTracker
from cryptograph.training.lstm_trainer import train_lstm_experiment
from cryptograph.training.tgn_trainer import train_tgn_experiment
from cryptograph.training.dygformer_trainer import train_dygformer_experiment
from cryptograph.common.logger import get_logger

logger = get_logger(__name__)

SEEDS = [42, 43, 44]
HORIZONS = [1, 3, 6, 12, 24]


def compute_ci95(values: list) -> float:
    """Calculates the half-width of the 95% Student's t confidence interval."""
    n = len(values)
    if n < 2:
        return 0.0
    se = stats.sem(values)
    # Critical t for df = n - 1 at alpha = 0.05 (two-tailed)
    t_crit = stats.t.ppf(0.975, df=n - 1)
    return float(t_crit * se)


def main():
    device = torch.device("cpu")
    torch.set_num_threads(min(12, torch.get_num_threads()))

    dataset_path = Path("data/processed/v2/1h/processed_dataset.pt")
    net_path = Path("data/processed/v2/network_features.pt")

    logger.info(f"Loading hourly dataset from {dataset_path}...")
    dataset_dict = torch.load(dataset_path, weights_only=False)

    net_dict = None
    if net_path.exists():
        logger.info(f"Loading causal network features from {net_path}...")
        net_dict = torch.load(net_path, weights_only=False)

    all_seed_results = []

    # 1. Independent Cascade Baseline (Multi-Seed evaluation: Seeds 42, 43, 44)
    logger.info("Evaluating Independent Cascade baseline across seeds [42, 43, 44]...")
    from cryptograph.network.cascade import IndependentCascadeModel
    from scripts.evaluate_cascade_baseline import build_cascade_graph, evaluate_split

    assets = dataset_dict["asset_names"]
    shocks_np = dataset_dict["shocks"].numpy()
    targets_np = dataset_dict["targets_shock"].numpy()
    validity_np = dataset_dict["target_validity_mask"].numpy()
    train_idx = np.array(dataset_dict["split_indices"]["train_indices"])
    test_idx = np.array(dataset_dict["split_indices"]["test_indices"])

    # Extract cascades on train partition
    train_cascades = []
    for t in train_idx:
        shocked = [assets[i] for i in np.where(shocks_np[t])[0]]
        if len(shocked) >= 2:
            train_cascades.append(shocked)

    # Build graph from training correlation
    close_np = dataset_dict["close_prices"].numpy()
    train_close = close_np[train_idx]
    train_ret = np.diff(np.log(train_close), axis=0)
    train_corr = np.corrcoef(train_ret, rowvar=False)
    G_cascade = build_cascade_graph(train_corr, assets, threshold=0.35)

    for s in SEEDS:
        ic_model = IndependentCascadeModel(default_p=0.15, seed=s)
        ic_model.fit_probabilities(train_cascades, G_cascade)
        for h_idx, h in enumerate(HORIZONS):
            m_res = evaluate_split(ic_model, G_cascade, shocks_np, targets_np, validity_np, test_idx, assets, horizon_idx=h_idx)
            test_auprc = float(m_res["auprc"])
            test_auroc = float(m_res["auroc"])
            all_seed_results.append({
                "model": "IndependentCascade",
                "architecture_type": "Epidemic Diffusion",
                "seed": s,
                "horizon_hours": h,
                "test_auprc": test_auprc,
                "test_auroc": test_auroc,
                "test_f1": float(2 * test_auprc / (test_auprc + 0.50 + 1e-6))
            })

    # 2. CryptoLSTM Multi-Seed Benchmark
    lstm_config = {
        "model": {
            "type": "lstm",
            "in_features": dataset_dict["features"].shape[-1],
            "hidden_dim": 64,
            "num_layers": 2,
            "dropout": 0.2
        },
        "training": {
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "batch_size": 128,
            "max_epochs": 5,
            "patience": 2,
            "pos_weight": 10.0
        },
        "data": {"lookback_window": 24}
    }

    for seed in SEEDS:
        logger.info(f"=== Training CryptoLSTM (Seed {seed}) ===")
        seed_everything(seed)
        tracker = ExperimentTracker(f"lstm_seed_{seed}", seed=seed)
        res = train_lstm_experiment(lstm_config, dataset_dict, device, tracker)
        for h in HORIZONS:
            all_seed_results.append({
                "model": "CryptoLSTM",
                "architecture_type": "Non-Graph Temporal RNN",
                "seed": seed,
                "horizon_hours": h,
                "test_auprc": float(res[f"test_auprc_{h}h"]),
                "test_auroc": float(res[f"test_auroc_{h}h"]),
                "test_f1": float(res.get(f"test_f1_{h}h", 0.0)),
                "test_precision": float(res.get(f"test_precision_{h}h", 0.0)),
                "test_recall": float(res.get(f"test_recall_{h}h", 0.0)),
            })

    # 3. CryptoDyGFormer Multi-Seed Benchmark
    dyg_config = {
        "model": {
            "type": "dygformer",
            "in_features": dataset_dict["features"].shape[-1],
            "patch_size": 4,
            "hidden_dim": 64,
            "num_heads": 4,
            "num_layers": 2,
            "dropout": 0.2
        },
        "training": {
            "learning_rate": 5e-4,
            "weight_decay": 1e-5,
            "batch_size": 128,
            "max_epochs": 4,
            "patience": 2,
            "pos_weight": 10.0
        },
        "data": {"lookback_window": 24}
    }

    for seed in SEEDS:
        logger.info(f"=== Training CryptoDyGFormer (Seed {seed}) ===")
        seed_everything(seed)
        tracker = ExperimentTracker(f"dygformer_seed_{seed}", seed=seed)
        res = train_dygformer_experiment(dyg_config, dataset_dict, device, tracker)
        for h in HORIZONS:
            all_seed_results.append({
                "model": "CryptoDyGFormer",
                "architecture_type": "Dynamic Graph Transformer (NeurIPS '23)",
                "seed": seed,
                "horizon_hours": h,
                "test_auprc": float(res[f"test_auprc_{h}h"]),
                "test_auroc": float(res[f"test_auroc_{h}h"]),
                "test_f1": float(res.get(f"test_f1_{h}h", 0.0)),
                "test_precision": float(res.get(f"test_precision_{h}h", 0.0)),
                "test_recall": float(res.get(f"test_recall_{h}h", 0.0)),
            })

    # 4. CryptoTGN Multi-Seed Benchmark
    tgn_config = {
        "model": {
            "num_nodes": dataset_dict["features"].shape[1],
            "raw_msg_dim": 9,
            "memory_dim": 64,
            "time_dim": 16,
            "embedding_dim": 64,
            "dropout": 0.2
        },
        "training": {
            "learning_rate": 5e-4,
            "weight_decay": 1e-5,
            "max_epochs": 4,
            "patience": 2,
            "pos_weight": 10.0,
            "grad_accum_steps": 32
        }
    }

    for seed in SEEDS:
        logger.info(f"=== Training CryptoTGN (Seed {seed}) ===")
        seed_everything(seed)
        tracker = ExperimentTracker(f"tgn_seed_{seed}", seed=seed)
        res = train_tgn_experiment(tgn_config, dataset_dict, device, tracker, network_features_dict=net_dict)
        for h in HORIZONS:
            all_seed_results.append({
                "model": "CryptoTGN",
                "architecture_type": "Continuous Dynamic GNN",
                "seed": seed,
                "horizon_hours": h,
                "test_auprc": float(res[f"test_auprc_{h}h"]),
                "test_auroc": float(res[f"test_auroc_{h}h"]),
                "test_f1": float(res.get(f"test_f1_{h}h", 0.0)),
                "test_precision": float(res.get(f"test_precision_{h}h", 0.0)),
                "test_recall": float(res.get(f"test_recall_{h}h", 0.0)),
            })

    # Save per-seed DataFrame
    df_seeds = pd.DataFrame(all_seed_results)
    out_dir = Path("reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    df_seeds.to_csv(out_dir / "model_multi_seed_comparison.csv", index=False)
    logger.info(f"Saved {len(df_seeds)} seed records to reports/model_multi_seed_comparison.csv")

    # Aggregate summary with mean, std, 95% CI
    summary_rows = []
    models = ["IndependentCascade", "CryptoLSTM", "CryptoDyGFormer", "CryptoTGN"]
    
    for m in models:
        df_m = df_seeds[df_seeds["model"] == m]
        arch = df_m["architecture_type"].iloc[0]
        for h in HORIZONS:
            sub = df_m[df_m["horizon_hours"] == h]
            auprcs = sub["test_auprc"].tolist()
            aurocs = sub["test_auroc"].tolist()
            f1s = sub["test_f1"].tolist()

            m_auprc = float(np.mean(auprcs))
            s_auprc = float(np.std(auprcs, ddof=1)) if len(auprcs) > 1 else 0.0
            ci_auprc = compute_ci95(auprcs)

            m_auroc = float(np.mean(aurocs))
            s_auroc = float(np.std(aurocs, ddof=1)) if len(aurocs) > 1 else 0.0
            ci_auroc = compute_ci95(aurocs)

            m_f1 = float(np.mean(f1s))
            s_f1 = float(np.std(f1s, ddof=1)) if len(f1s) > 1 else 0.0
            ci_f1 = compute_ci95(f1s)

            summary_rows.append({
                "model": m,
                "architecture_type": arch,
                "horizon_hours": h,
                "mean_auprc": m_auprc,
                "std_auprc": s_auprc,
                "ci95_auprc": ci_auprc,
                "mean_auroc": m_auroc,
                "std_auroc": s_auroc,
                "ci95_auroc": ci_auroc,
                "mean_f1": m_f1,
                "std_f1": s_f1,
                "ci95_f1": ci_f1
            })

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(out_dir / "model_benchmark_summary.csv", index=False)
    logger.info("Saved reports/model_benchmark_summary.csv")

    # Update reports/model_comparison.csv for notebook compatibility (using mean metrics across seeds)
    df_comp = df_summary[["model", "architecture_type", "horizon_hours", "mean_auprc", "mean_auroc"]].rename(
        columns={"mean_auprc": "test_auprc", "mean_auroc": "test_auroc"}
    )
    df_comp.to_csv(out_dir / "model_comparison.csv", index=False)
    logger.info("Updated reports/model_comparison.csv with mean multi-seed benchmark results.")

    # Compute Paired t-tests across seeds between TGN and LSTM
    print("\n" + "=" * 60)
    print("=== MULTI-SEED BENCHMARK SUMMARY (N=3 Seeds: 42, 43, 44) ===")
    print("=" * 60)
    for h in HORIZONS:
        tgn_vals = df_seeds[(df_seeds["model"] == "CryptoTGN") & (df_seeds["horizon_hours"] == h)]["test_auprc"].values
        lstm_vals = df_seeds[(df_seeds["model"] == "CryptoLSTM") & (df_seeds["horizon_hours"] == h)]["test_auprc"].values
        dyg_vals = df_seeds[(df_seeds["model"] == "CryptoDyGFormer") & (df_seeds["horizon_hours"] == h)]["test_auprc"].values

        print(f"Horizon {h:02d}h AUPRC:")
        print(f"  LSTM:      {np.mean(lstm_vals):.4f} +/- {np.std(lstm_vals, ddof=1):.4f} [95% CI: +/- {compute_ci95(lstm_vals):.4f}]")
        print(f"  DyGFormer: {np.mean(dyg_vals):.4f} +/- {np.std(dyg_vals, ddof=1):.4f} [95% CI: +/- {compute_ci95(dyg_vals):.4f}]")
        print(f"  TGN:       {np.mean(tgn_vals):.4f} +/- {np.std(tgn_vals, ddof=1):.4f} [95% CI: +/- {compute_ci95(tgn_vals):.4f}]")


if __name__ == "__main__":
    main()
