import pytest
import torch
import pandas as pd
import numpy as np

from cryptograph.models.tgn import CryptoTGN
from cryptograph.models.lstm import CryptoLSTM
from cryptograph.analysis.var_spillover import VARSpilloverBaseline
from cryptograph.models.tgae import TemporalGAE
from cryptograph.spillover.tempme import TempME
from cryptograph.spillover.validation import ExplanationValidator
from cryptograph.security.perturbation import DualPerturbationTester

def test_crypto_tgn():
    num_nodes = 5
    in_features = 8
    model = CryptoTGN(num_nodes=num_nodes, raw_msg_dim=in_features, memory_dim=16, embedding_dim=16, out_horizons=3)
    x = torch.randn(num_nodes, in_features)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)

    preds, z = model(x, edge_index)
    assert preds.shape == (num_nodes, 3)
    assert z.shape == (num_nodes, 16)

def test_crypto_lstm():
    lstm = CryptoLSTM(in_features=8, hidden_dim=16, out_horizons=3)
    dummy_x = torch.randn(5, 10, 8)  # 5 nodes/batches, 10 timesteps, 8 features
    out = lstm(dummy_x)
    assert out.shape == (5, 3)

def test_var_spillover_baseline():
    baseline = VARSpilloverBaseline(maxlags=2, forecast_horizon=5)
    np.random.seed(42)
    sample_returns = pd.DataFrame(
        np.random.randn(50, 4) * 0.02,
        columns=["BTC", "ETH", "SOL", "BNB"]
    )
    baseline.fit(sample_returns)
    table = baseline.compute_spillover_table()
    assert "total_spillover_index" in table
    assert table["total_spillover_index"] >= 0.0

def test_tempme_and_validation():
    num_nodes = 4
    in_features = 8
    model = CryptoTGN(num_nodes=num_nodes, raw_msg_dim=in_features, memory_dim=16, embedding_dim=16, out_horizons=2)
    x = torch.randn(num_nodes, in_features)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)

    explainer = TempME(model, num_epochs=5)
    exp = explainer.explain(target_node_idx=0, node_features=x, edge_index=edge_index)
    assert "edge_mask" in exp
    assert "top_edges" in exp

    validator = ExplanationValidator(model)
    fidelity = validator.compute_fidelity_plus(x, edge_index, exp["edge_mask"], target_idx=0)
    assert isinstance(fidelity, float)

def test_temporal_gae():
    gae = TemporalGAE(in_features=8, hidden_dim=16, latent_dim=8)
    x = torch.randn(5, 8)
    x_rec, adj_rec, z = gae(x)
    assert x_rec.shape == (5, 8)
    assert adj_rec.shape == (5, 5)

def test_dual_perturbation():
    num_nodes = 4
    model = CryptoTGN(num_nodes=num_nodes, raw_msg_dim=8, memory_dim=16, embedding_dim=16, out_horizons=2)
    x = torch.randn(num_nodes, 8)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    explainer = TempME(model, num_epochs=3)

    tester = DualPerturbationTester(model, explainer)
    results = tester.run_stress_test(x, edge_index, target_idx=0, epsilon_levels=[0.05])
    assert len(results) == 1
    assert "prediction_drift_mse" in results[0]
    assert "explanation_jaccard_stability" in results[0]
