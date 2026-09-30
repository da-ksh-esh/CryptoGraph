import pytest
import torch
import numpy as np
import shutil
from pathlib import Path

from cryptograph.training.device import resolve_device
from cryptograph.training.seed import seed_everything
from cryptograph.training.losses import HuberLoss, WeightedBCELoss, FocalLoss
from cryptograph.training.metrics import compute_regression_metrics, compute_classification_metrics
from cryptograph.training.early_stopping import EarlyStopping
from cryptograph.training.tracking import ExperimentTracker

def test_resolve_device():
    dev = resolve_device("cpu")
    assert dev.type == "cpu"

def test_seed_everything():
    seed_everything(123)
    t1 = torch.randn(5)
    seed_everything(123)
    t2 = torch.randn(5)
    assert torch.equal(t1, t2)

def test_losses():
    huber = HuberLoss()
    pred = torch.tensor([1.0, 2.0])
    target = torch.tensor([1.1, 1.9])
    assert huber(pred, target) > 0.0

    wbce = WeightedBCELoss(pos_weight=2.0)
    logits = torch.tensor([0.5, -0.5])
    labels = torch.tensor([1.0, 0.0])
    assert wbce(logits, labels) > 0.0

    focal = FocalLoss()
    assert focal(logits, labels) > 0.0

def test_metrics():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 1.9, 3.2])
    reg = compute_regression_metrics(y_true, y_pred)
    assert "rmse" in reg
    assert "mae" in reg

    y_true_cls = np.array([0, 1, 1, 0])
    y_score = np.array([0.1, 0.9, 0.8, 0.3])
    cls_metrics = compute_classification_metrics(y_true_cls, y_score)
    assert "auroc" in cls_metrics
    assert "auprc" in cls_metrics

def test_early_stopping():
    es = EarlyStopping(patience=2, min_delta=1e-3, mode="min")
    assert not es(1.0)
    assert not es(0.9)  # improvement
    assert not es(0.9)  # no improvement (count 1)
    assert es(0.91)     # no improvement (count 2 -> stop)

def test_experiment_tracker(tmp_path):
    tracker = ExperimentTracker("dummy_exp", seed=42, runs_dir=str(tmp_path))
    tracker.log_environment({"param": 1}, device="cpu")
    tracker.log_metrics({"rmse": 0.05})
    ckpt = tracker.save_checkpoint({"weight": [1, 2]})

    assert (tracker.run_path / "config.resolved.yaml").exists()
    assert (tracker.run_path / "env.json").exists()
    assert (tracker.run_path / "metrics.json").exists()
    assert (tmp_path / "results.csv").exists()
    assert ckpt.exists()
