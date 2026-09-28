"""Scores anomalies and detects oracle/market manipulation via reconstruction error."""
from typing import List
import numpy as np
import torch
from ..common.types import AnomalyAlert

class AnomalyScorer:
    def __init__(self, threshold_percentile: float = 98.0):
        self.threshold_percentile = threshold_percentile

    def evaluate_anomalies(
        self,
        x_orig: torch.Tensor,
        x_recon: torch.Tensor,
        assets: List[str]
    ) -> List[AnomalyAlert]:
        """Calculates per-node reconstruction errors and flags statistical anomalies."""
        errors = torch.mean((x_orig - x_recon) ** 2, dim=-1).detach().cpu().numpy()
        threshold = np.percentile(errors, self.threshold_percentile)

        alerts = []
        for i, asset in enumerate(assets):
            score = float(errors[i])
            if score > threshold:
                alerts.append(AnomalyAlert(
                    timestamp="latest",
                    asset=asset,
                    anomaly_score=score,
                    threshold=float(threshold),
                    suspected_type="oracle_manipulation" if score > threshold * 1.5 else "wash_trading",
                    reconstruction_error=score
                ))
        return alerts
