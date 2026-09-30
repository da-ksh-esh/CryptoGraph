import torch

from cryptograph.explain.edge_mask import EdgeMaskExplainer
from cryptograph.models.tgn import CryptoTGN
from cryptograph.robustness.perturbation import DualPerturbationTester


def test_dual_perturbation():
    num_nodes = 4
    model = CryptoTGN(num_nodes=num_nodes, raw_msg_dim=8, memory_dim=16, embedding_dim=16, out_horizons=2)
    x = torch.randn(num_nodes, 8)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    explainer = EdgeMaskExplainer(model, num_epochs=3)

    tester = DualPerturbationTester(model, explainer)
    results = tester.run_stress_test(x, edge_index, target_idx=0, epsilon_levels=[0.05])
    assert len(results) == 1
    assert "prediction_drift_mse" in results[0]
    assert "explanation_jaccard_stability" in results[0]
