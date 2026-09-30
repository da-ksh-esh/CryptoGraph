"""Evaluation Metrics for Regression and Imbalanced Contagion Classification."""

from typing import Dict, Any
import numpy as np
import torch
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score
)

def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Computes RMSE and MAE."""
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    return {
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "mae": float(mae)
    }

def compute_classification_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """
    Computes AUROC, AUPRC (Average Precision), Precision, Recall, and F1.
    AUPRC is emphasized due to heavy negative class dominance in shock contagion.
    """
    y_true_bin = (y_true > 0.5).astype(int)
    y_pred_bin = (y_score >= threshold).astype(int)

    # Handle single class edge cases in small batches
    if len(np.unique(y_true_bin)) > 1:
        auroc = float(roc_auc_score(y_true_bin, y_score))
        auprc = float(average_precision_score(y_true_bin, y_score))
    else:
        auroc = 0.5
        auprc = float(np.mean(y_true_bin))

    precision = float(precision_score(y_true_bin, y_pred_bin, zero_division=0))
    recall = float(recall_score(y_true_bin, y_pred_bin, zero_division=0))
    f1 = float(f1_score(y_true_bin, y_pred_bin, zero_division=0))

    return {
        "auroc": auroc,
        "auprc": auprc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
