import pytest
import torch
from cryptograph.robustness.anomaly import AnomalyScorer

def test_anomaly_scorer():
    scorer = AnomalyScorer(threshold_percentile=50.0)
    orig = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    recon = torch.tensor([[1.05, 1.95], [5.0, 8.0]])  # second has high error
    alerts = scorer.evaluate_anomalies(orig, recon, ["BTC", "ETH"])
    assert len(alerts) >= 1
    assert all(a.suspected_type == "unclassified" for a in alerts)
