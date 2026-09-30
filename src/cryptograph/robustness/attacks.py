"""Adversarial perturbation attacks: Feature attacks (FGSM/PGD)."""

from typing import Tuple, Optional
import torch
import torch.nn as nn

class AdversarialAttacker:
    """
    Adversarial attacks on graph neural network inputs.
    """
    def __init__(self, eps: float = 0.05, alpha: float = 0.01, steps: int = 10):
        self.eps = eps
        self.alpha = alpha
        self.steps = steps

    def fgsm_feature_attack(
        self,
        model: nn.Module,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        y: torch.Tensor
    ) -> torch.Tensor:
        """
        Fast Gradient Sign Method (FGSM) on input node features.
        TODO: Generalize across both regression and classification loss criteria.
        """
        x_adv = x.clone().detach().requires_grad_(True)
        out = model(x_adv, edge_index)
        pred = out[0] if isinstance(out, tuple) else out

        loss = nn.MSELoss()(pred, y)
        loss.backward()

        with torch.no_grad():
            if x_adv.grad is not None:
                x_adv = x_adv + self.eps * x_adv.grad.sign()
        return x_adv.detach()
