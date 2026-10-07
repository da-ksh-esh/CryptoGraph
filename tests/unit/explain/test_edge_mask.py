import torch

from cryptograph.explain.edge_mask import EdgeMaskExplainer
from cryptograph.explain.validation import ExplanationValidator
from cryptograph.models.tgn import CryptoTGN


def test_edge_mask_and_validation():
    num_nodes = 4
    in_features = 8
    model = CryptoTGN(num_nodes=num_nodes, raw_msg_dim=in_features, memory_dim=16, embedding_dim=16, out_horizons=2)
    x = torch.randn(num_nodes, in_features)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)

    explainer = EdgeMaskExplainer(model, num_epochs=5)
    exp = explainer.explain(target_node_idx=0, node_features=x, edge_index=edge_index)
    assert "edge_mask" in exp
    assert "top_edges" in exp

    validator = ExplanationValidator(model)
    fidelity = validator.compute_fidelity_plus(x, edge_index, exp["edge_mask"], target_idx=0)
    assert isinstance(fidelity, float)
