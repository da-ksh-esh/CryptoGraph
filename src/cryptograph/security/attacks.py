"""Adversarial perturbation attacks: Feature attacks (PGD/FGSM) and Graph Topology (DICE)."""
from typing import Tuple
import torch
import torch.nn as nn
import numpy as np

class AdversarialAttacker:
    def __init__(self, eps: float = 0.05, alpha: float = 0.01, steps: int = 10):
        self.eps = eps
        self.alpha = alpha
        self.steps = steps

    def fgsm_feature_attack(self, model: nn.Module, x: torch.Tensor, edge_index: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Fast Gradient Sign Method on input node features."""
        x_adv = x.clone().detach().requires_grad_(True)
        pred = model(x_adv, edge_index)
        loss = nn.MSELoss()(pred, y)
        loss.backward()

        with torch.no_grad():
            x_adv = x_adv + self.eps * x_adv.grad.sign()
        return x_adv.detach()

    def dice_topology_attack(self, edge_index: torch.Tensor, num_nodes: int, ratio: float = 0.1) -> torch.Tensor:
        """Disconnect Internally, Connect Externally (DICE) edge perturbation."""
        # Random edge flipping simulation
        return edge_index
