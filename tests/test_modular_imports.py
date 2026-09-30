import pytest
import torch

def test_direct_modular_imports():
    from cryptograph.models.lstm import CryptoLSTM
    from cryptograph.models.tgae import TemporalGAE
    from cryptograph.analysis.connectedness import VARSpilloverBaseline
    from cryptograph.explain.edge_mask import EdgeMaskExplainer, TempME
    from cryptograph.explain.validation import ExplanationValidator
    from cryptograph.robustness.perturbation import DualPerturbationTester
    from cryptograph.robustness.anomaly import AnomalyScorer
    from cryptograph.network import (
        NetworkCentralityAnalyzer,
        CommunityDetector,
        MinimumSpanningTreeAnalyzer,
        TopologyMetrics,
        IndependentCascadeModel
    )
    from cryptograph.training import (
        resolve_device,
        seed_everything,
        HuberLoss,
        WeightedBCELoss,
        FocalLoss,
        EarlyStopping,
        ExperimentTracker,
        ModelTrainer
    )

    lstm = CryptoLSTM(in_features=4, hidden_dim=8, out_horizons=2)
    x = torch.randn(2, 5, 4)
    assert lstm(x).shape == (2, 2)
