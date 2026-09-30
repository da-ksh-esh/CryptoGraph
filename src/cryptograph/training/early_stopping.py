"""Early stopping monitor for loss and evaluation metrics."""

from typing import Optional
import numpy as np

class EarlyStopping:
    """
    Early stopping tracker that interrupts training when validation performance stagnates.
    """

    def __init__(self, patience: int = 10, min_delta: float = 1e-4, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False

    def __call__(self, current_score: float) -> bool:
        if self.best_score is None:
            self.best_score = current_score
            return False

        if self.mode == "min":
            improvement = self.best_score - current_score > self.min_delta
        else:
            improvement = current_score - self.best_score > self.min_delta

        if improvement:
            self.best_score = current_score
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return self.early_stop
