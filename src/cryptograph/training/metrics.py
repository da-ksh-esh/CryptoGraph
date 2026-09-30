"""Evaluation metrics (regression and rare-event classification), mask-aware."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def _flat(y_true, y_pred, mask=None):
    y_true, y_pred = np.asarray(y_true, dtype=float).ravel(), np.asarray(y_pred, dtype=float).ravel()
    keep = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask is not None:
        keep &= np.asarray(mask, dtype=bool).ravel()
    return y_true[keep], y_pred[keep]


def mse(y_true, y_pred, mask=None) -> float:
    t, p = _flat(y_true, y_pred, mask)
    return float(np.mean((t - p) ** 2)) if t.size else float("nan")


def mae(y_true, y_pred, mask=None) -> float:
    t, p = _flat(y_true, y_pred, mask)
    return float(np.mean(np.abs(t - p))) if t.size else float("nan")


def auroc(y_true, score, mask=None) -> float:
    t, s = _flat(y_true, score, mask)
    return float(roc_auc_score(t.astype(int), s)) if 0 < t.sum() < t.size else float("nan")


def auprc(y_true, score, mask=None) -> float:
    """Average precision; its chance level equals the positive rate (report it alongside)."""
    t, s = _flat(y_true, score, mask)
    return float(average_precision_score(t.astype(int), s)) if t.sum() > 0 else float("nan")
