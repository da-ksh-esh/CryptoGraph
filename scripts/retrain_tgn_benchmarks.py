"""Retrains CryptoTGN across multi-seed benchmark (42, 43, 44) with memory-gradient fix active."""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import torch

from cryptograph.training.seed import seed_everything
from cryptograph.training.tracking import ExperimentTracker
from cryptograph.training.tgn_trainer import train_tgn_experiment
from cryptograph.common.logger import get_logger

logger = get_logger(__name__)

SEEDS = [42, 43, 44]
HORIZONS = [1, 3, 6, 12, 24]


def compute_ci95(values: list) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    se = stats.sem(values)
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

    tgn_seed_results = []
    for seed in SEEDS:
        logger.info(f"=== Retraining CryptoTGN with Memory Gradients (Seed {seed}) ===")
        seed_everything(seed)
        tracker = ExperimentTracker(f"tgn_seed_{seed}", seed=seed)
        res = train_tgn_experiment(tgn_config, dataset_dict, device, tracker, network_features_dict=net_dict)
        for h in HORIZONS:
            tgn_seed_results.append({
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

    # Load existing benchmark records
    multi_seed_path = Path("reports/model_multi_seed_comparison.csv")
    if multi_seed_path.exists():
        df_existing = pd.read_csv(multi_seed_path)
        # Keep non-TGN models
        df_other = df_existing[df_existing["model"] != "CryptoTGN"]
        df_tgn = pd.DataFrame(tgn_seed_results)
        df_combined = pd.concat([df_other, df_tgn], ignore_index=True)
    else:
        df_combined = pd.DataFrame(tgn_seed_results)

    out_dir = Path("reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    df_combined.to_csv(multi_seed_path, index=False)
    logger.info(f"Saved updated multi-seed comparison to {multi_seed_path}")

    # Re-aggregate summary
    summary_rows = []
    models = ["IndependentCascade", "CryptoLSTM", "CryptoDyGFormer", "CryptoTGN"]

    for m in models:
        df_m = df_combined[df_combined["model"] == m]
        if df_m.empty:
            continue
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
    logger.info("Saved updated reports/model_benchmark_summary.csv")

    df_comp = df_summary[["model", "architecture_type", "horizon_hours", "mean_auprc", "mean_auroc"]].rename(
        columns={"mean_auprc": "test_auprc", "mean_auroc": "test_auroc"}
    )
    df_comp.to_csv(out_dir / "model_comparison.csv", index=False)
    logger.info("Saved updated reports/model_comparison.csv")

    print("\n" + "=" * 60)
    print("=== TGN RETRAINING & BENCHMARK UPDATE COMPLETE ===")
    print("=" * 60)
    print(df_summary[df_summary["model"] == "CryptoTGN"][["horizon_hours", "mean_auprc", "ci95_auprc", "mean_auroc", "mean_f1"]].to_string(index=False))


if __name__ == "__main__":
    main()
