"""Custom Loss Functions for Extreme Events and Imbalance."""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

class HuberLoss(nn.Module):
    """Robust smooth L1 loss for financial return and volatility regression."""
    def __init__(self, delta: float = 1.0):
        super().__init__()
        self.delta = delta

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.huber_loss(pred, target, delta=self.delta)

class WeightedBCELoss(nn.Module):
    """Binary cross-entropy with positive class weighting for rare shock events."""
    def __init__(self, pos_weight: float = 5.0):
        super().__init__()
        self.pos_weight = torch.tensor([pos_weight])

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        weight = self.pos_weight.to(logits.device)
        return F.binary_cross_entropy_with_logits(logits, target, pos_weight=weight)

class FocalLoss(nn.Module):
    """Focal loss (Lin et al., 2017) down-weighting easy negatives in class-imbalanced contagion."""
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = bce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        return loss.mean()
