"""End-to-End Smoke Test: Data -> Preprocessing -> Training -> Checkpoint -> Evaluation -> Inference."""

from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import torch

from cryptograph.data.targets import ContagionTargetBuilder
from cryptograph.models.tgn import CryptoTGN
from cryptograph.training.tgn_trainer import train_tgn_experiment, evaluate_tgn, build_topk_dynamic_edges
from cryptograph.training.tracking import ExperimentTracker


def test_end_to_end_pipeline_smoke(tmp_path: Path):
    """
    Validates complete end-to-end workflow:
    1. Data preparation & feature generation
    2. Contagion target builder with train thresholding
    3. Model initialization & training
    4. Checkpoint serialization
    5. Evaluation pipeline
    6. Single-step / multi-horizon inference
    """
    torch.manual_seed(42)
    np.random.seed(42)

    T = 80
    N = 4
    F = 9
    horizons = [1, 3]

    assets = ["BTC", "ETH", "SOL", "BNB"]
    timestamps = [f"2025-01-01 {h:02d}:00:00" for h in range(T)]

    # 1. Synthetic features, returns, close prices
    returns = np.random.normal(0, 0.02, size=(T, N)).astype(np.float32)
    close_prices = 100.0 * np.exp(np.cumsum(returns, axis=0))
    volatility = np.abs(returns) * 1.5

    features = torch.randn(T, N, F, dtype=torch.float32)
    features[:, :, 0] = torch.from_numpy(returns)
    features[:, :, 1] = torch.from_numpy(volatility)

    active_mask = np.ones((T, N), dtype=bool)

    # 2. Target Builder
    target_builder = ContagionTargetBuilder(horizons=horizons, shock_quantile=0.85)
    train_mask = np.zeros(T, dtype=bool)
    train_mask[:50] = True
    val_mask = np.zeros(T, dtype=bool)
    val_mask[50:65] = True
    test_mask = np.zeros(T, dtype=bool)
    test_mask[65:] = True

    target_builder.fit_thresholds(volatility, train_mask)
    shocks = target_builder.identify_shocks(volatility)
    shock_targets, return_targets, validity_mask = target_builder.build_targets(
        close_prices=close_prices,
        shocks=shocks,
        active_mask=active_mask
    )

    dataset_dict = {
        "features": features,
        "shocks": torch.from_numpy(shocks),
        "targets_shock": torch.from_numpy(shock_targets),
        "target_validity_mask": torch.from_numpy(validity_mask),
        "active_mask": torch.from_numpy(active_mask),
        "close_prices": torch.from_numpy(close_prices),
        "timestamps": timestamps,
        "asset_names": assets,
        "horizons": horizons,
        "split_indices": {
            "train_indices": list(range(0, 50)),
            "val_indices": list(range(50, 65)),
            "test_indices": list(range(65, T))
        }
    }

    # 3. Training config
    config = {
        "model": {
            "num_nodes": N,
            "raw_msg_dim": F,
            "node_feat_dim": F,
            "memory_dim": 16,
            "time_dim": 8,
            "embedding_dim": 16,
            "dropout": 0.0
        },
        "training": {
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "max_epochs": 1,
            "patience": 1,
            "pos_weight": 2.0,
            "grad_accum_steps": 1
        }
    }

    tracker = ExperimentTracker("smoke_test", runs_dir=str(tmp_path / "runs"), seed=42)
    device = torch.device("cpu")

    train_res = train_tgn_experiment(config, dataset_dict, device, tracker)
    assert "train_loss" in train_res
    assert "val_mean_auprc" in train_res
    assert "test_mean_auprc" in train_res

    # 4. Checkpoint Serialization & Reloading
    ckpt_path = tracker.run_path / "best_tgn_checkpoint.pt"
    assert ckpt_path.exists()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert "model_state_dict" in ckpt

    model = CryptoTGN(
        num_nodes=N,
        raw_msg_dim=F,
        node_feat_dim=F,
        memory_dim=16,
        time_dim=8,
        embedding_dim=16,
        out_horizons=len(horizons)
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # 5. Evaluation pipeline
    edge_indices, edge_weights = build_topk_dynamic_edges(returns, top_k=2, window=10)
    eval_res = evaluate_tgn(
        model=model,
        features=features,
        shocks=dataset_dict["shocks"],
        targets=dataset_dict["targets_shock"],
        validity_mask=dataset_dict["target_validity_mask"],
        active_mask=dataset_dict["active_mask"],
        target_indices=dataset_dict["split_indices"]["test_indices"],
        horizons=horizons,
        edge_indices=edge_indices,
        edge_weights=edge_weights,
        reset_memory=True
    )
    assert "mean_auroc" in eval_res
    assert "mean_auprc" in eval_res
    assert 0.0 <= eval_res["mean_auroc"] <= 1.0
    assert 0.0 <= eval_res["mean_auprc"] <= 1.0

    # 6. Single snapshot inference (forward pass)
    with torch.no_grad():
        x_last = features[-1]
        e_last = edge_indices[-1]
        w_last = edge_weights[-1]
        preds, embs = model(x_last, e_last, edge_weight=w_last)
        probs = torch.sigmoid(preds).numpy()

    assert preds.shape == (N, len(horizons))
    assert embs.shape == (N, 16)
    assert probs.shape == (N, len(horizons))
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
    assert not np.any(np.isnan(probs))
