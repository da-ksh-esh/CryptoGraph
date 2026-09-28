"""Tracks shock propagation across assets: magnitude, transmission delay, and persistence."""
from typing import List, Dict
import numpy as np
from ..common.types import SpilloverMetrics

class SpilloverPropagationTracker:
    def __init__(self, max_lag: int = 24):
        self.max_lag = max_lag

    def compute_spillover(
        self,
        source_asset: str,
        target_asset: str,
        source_returns: np.ndarray,
        target_returns: np.ndarray
    ) -> SpilloverMetrics:
        """Estimates cross-correlation lag (delay), peak magnitude, and half-life decay (persistence)."""
        # Cross-correlation across lags
        cross_corr = [
            float(np.corrcoef(source_returns[:-lag or None], target_returns[lag:])[0, 1])
            if lag > 0 else float(np.corrcoef(source_returns, target_returns)[0, 1])
            for lag in range(self.max_lag)
        ]

        delay = int(np.argmax(np.abs(cross_corr)))
        magnitude = float(cross_corr[delay])
        half_life = float(delay * 1.5)  # Placeholder half-life decay calculation

        return SpilloverMetrics(
            source_asset=source_asset,
            target_asset=target_asset,
            magnitude=magnitude,
            delay_steps=delay,
            half_life_persistence=half_life,
            transmission_path=[source_asset, target_asset]
        )
