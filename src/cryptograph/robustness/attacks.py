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
        y: torch.Tensor,
        criterion: Optional[nn.Module] = None
    ) -> torch.Tensor:
        """
        Fast Gradient Sign Method (FGSM) on input node features.
        Supports both regression (MSE) and classification (BCEWithLogits) criteria.
        """
        x_adv = x.clone().detach().requires_grad_(True)
        out = model(x_adv, edge_index)
        pred = out[0] if isinstance(out, tuple) else out

        if criterion is None:
            if y.dtype in (torch.bool, torch.long) or ((y >= 0.0) & (y <= 1.0)).all():
                crit = nn.BCEWithLogitsLoss()
            else:
                crit = nn.MSELoss()
        else:
            crit = criterion

        loss = crit(pred, y.float())
        loss.backward()

        with torch.no_grad():
            if x_adv.grad is not None:
                x_adv = x_adv + self.eps * x_adv.grad.sign()
        return x_adv.detach()

    def pgd_feature_attack(
        self,
        model: nn.Module,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        y: torch.Tensor,
        criterion: Optional[nn.Module] = None
    ) -> torch.Tensor:
        """
        Projected Gradient Descent (PGD) multi-step attack bounded in L_inf norm ball.
        """
        x_orig = x.clone().detach()
        x_adv = x_orig.clone().detach() + torch.empty_like(x_orig).uniform_(-self.eps, self.eps)

        if criterion is None:
            if y.dtype in (torch.bool, torch.long) or ((y >= 0.0) & (y <= 1.0)).all():
                crit = nn.BCEWithLogitsLoss()
            else:
                crit = nn.MSELoss()
        else:
            crit = criterion

        for _ in range(self.steps):
            x_adv.requires_grad_(True)
            out = model(x_adv, edge_index)
            pred = out[0] if isinstance(out, tuple) else out
            loss = crit(pred, y.float())
            loss.backward()

            with torch.no_grad():
                if x_adv.grad is not None:
                    x_adv = x_adv + self.alpha * x_adv.grad.sign()
                eta = torch.clamp(x_adv - x_orig, min=-self.eps, max=self.eps)
                x_adv = x_orig + eta

        return x_adv.detach()
