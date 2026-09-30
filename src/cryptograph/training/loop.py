"""Modular Headless Training Loop."""

from typing import Dict, Any, Callable, Optional
import torch
import torch.nn as nn
from torch.optim import Optimizer
from .losses import HuberLoss, WeightedBCELoss
from .metrics import compute_regression_metrics, compute_classification_metrics
from .early_stopping import EarlyStopping
from ..common.logger import get_logger

logger = get_logger(__name__)

class ModelTrainer:
    """
    Standardized training loop supporting both node regression (volatility/returns)
    and node classification (contagion shock risk).
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: nn.Module,
        device: torch.device,
        early_stopping: Optional[EarlyStopping] = None
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.early_stopping = early_stopping

    def train_epoch(self, dataloader) -> float:
        self.model.train()
        total_loss = 0.0
        steps = 0

        for batch in dataloader:
            self.optimizer.zero_grad()
            if isinstance(batch, (list, tuple)):
                x, y = batch[0].to(self.device), batch[1].to(self.device)
                out = self.model(x)
            elif hasattr(batch, "x") and hasattr(batch, "edge_index"):
                x = batch.x.to(self.device)
                edge_index = batch.edge_index.to(self.device)
                y = batch.y.to(self.device)
                out = self.model(x, edge_index)
            else:
                continue

            preds = out[0] if isinstance(out, tuple) else out
            loss = self.criterion(preds, y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            steps += 1

        return total_loss / max(1, steps)
