"""Early stopping on a validation metric."""

from __future__ import annotations

import math


class EarlyStopping:
    """Stop when `metric` has not improved by more than `min_delta` for `patience` checks.

    mode="min" for losses, "max" for scores such as AUPRC.
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = "min"):
        if mode not in ("min", "max"):
            raise ValueError("mode must be 'min' or 'max'")
        self.patience, self.min_delta, self.mode = patience, min_delta, mode
        self.best = math.inf if mode == "min" else -math.inf
        self.best_epoch = -1
        self.bad_epochs = 0
        self.epoch = -1

    def improved(self, value: float) -> bool:
        if self.mode == "min":
            return value < self.best - self.min_delta
        return value > self.best + self.min_delta

    def step(self, value: float) -> bool:
        """Record one validation result; return True when training should stop."""
        self.epoch += 1
        if self.improved(value):
            self.best, self.best_epoch, self.bad_epochs = value, self.epoch, 0
        else:
            self.bad_epochs += 1
        return self.bad_epochs >= self.patience
