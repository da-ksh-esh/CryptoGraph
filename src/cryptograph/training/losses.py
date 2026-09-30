"""Losses for heavy-tailed regression and rare-event classification, with optional masks.

A mask is a boolean tensor broadcastable to the prediction; masked-out entries do not
contribute. The mean is taken over the kept entries only.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _masked_mean(values: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return values.mean()
    m = mask.to(values.dtype).expand_as(values)
    return (values * m).sum() / m.sum().clamp_min(1.0)


def huber_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None,
               delta: float = 1.0) -> torch.Tensor:
    """Huber loss (quadratic within delta, linear outside): robust to fat-tailed targets."""
    return _masked_mean(F.huber_loss(pred, target, reduction="none", delta=delta), mask)


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    return _masked_mean((pred - target) ** 2, mask)


def weighted_bce(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None,
                 pos_weight: float | None = None) -> torch.Tensor:
    """Binary cross-entropy on logits; pos_weight up-weights the rare positive class."""
    pw = None if pos_weight is None else torch.as_tensor(pos_weight, dtype=logits.dtype, device=logits.device)
    loss = F.binary_cross_entropy_with_logits(logits, target.to(logits.dtype), reduction="none", pos_weight=pw)
    return _masked_mean(loss, mask)


def focal_loss(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None,
               gamma: float = 2.0, alpha: float | None = 0.25) -> torch.Tensor:
    """Binary focal loss (Lin et al. 2017). gamma=0 and alpha=None reduce it to plain BCE."""
    t = target.to(logits.dtype)
    bce = F.binary_cross_entropy_with_logits(logits, t, reduction="none")
    p = torch.sigmoid(logits)
    p_t = p * t + (1 - p) * (1 - t)
    loss = (1 - p_t) ** gamma * bce
    if alpha is not None:
        loss = (alpha * t + (1 - alpha) * (1 - t)) * loss
    return _masked_mean(loss, mask)


def pos_weight_from_rate(positive_rate: float) -> float:
    """pos_weight = negatives / positives, from a positive rate measured on TRAIN only."""
    if not 0 < positive_rate < 1:
        raise ValueError("positive_rate must be in (0, 1)")
    return (1 - positive_rate) / positive_rate
