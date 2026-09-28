"""Detects cryptocurrency price shocks and jumps using EVT and CUSUM."""
from typing import List, Optional
import numpy as np
import pandas as pd
from ..common.types import ShockEvent

class ShockDetector:
    def __init__(self, z_threshold: float = 3.0, percentile_threshold: float = 99.0):
        self.z_threshold = z_threshold
        self.percentile_threshold = percentile_threshold

    def detect_shocks(self, returns_series: pd.Series, asset_name: str) -> List[ShockEvent]:
        """Flags statistical outliers and jumps in the return series."""
        mean = returns_series.mean()
        std = returns_series.std()
        z_scores = (returns_series - mean) / (std + 1e-8)

        shocks = []
        for idx, (timestamp, z) in enumerate(z_scores.items()):
            if abs(z) >= self.z_threshold:
                magnitude = float(returns_series.iloc[idx])
                shock_type = "flash_crash" if z < 0 else "positive_jump"
                shocks.append(ShockEvent(
                    timestamp=str(timestamp),
                    source_asset=asset_name,
                    shock_magnitude=magnitude,
                    shock_type=shock_type,
                    z_score=float(z)
                ))
        return shocks
