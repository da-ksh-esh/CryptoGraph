import pytest
import torch
import numpy as np
from cryptograph.models.tgn import CryptoTGN
from cryptograph.explain.tempme import TempME
from cryptograph.explain.validation import ExplanationValidator

def test_tempme_motif_discovery():
    num_nodes = 4
    in_features = 6
    model = CryptoTGN(
        num_nodes=num_nodes,
        raw_msg_dim=in_features,
        node_feat_dim=in_features,
        memory_dim=16,
        embedding_dim=16,
        out_horizons=2
    )

    x = torch.randn(num_nodes, in_features)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)

    explainer = TempME(model, num_epochs=5, lr=0.1)
    res = explainer.explain(target_node_idx=0, node_features=x, edge_index=edge_index)

    assert "edge_mask" in res
    assert len(res["edge_mask"]) == 4
    assert "top_edges" in res
    assert "sparsity" in res
    assert "motif_entropy" in res

def test_explanation_validator_fidelity_minus():
    num_nodes = 4
    in_features = 6
    model = CryptoTGN(
        num_nodes=num_nodes,
        raw_msg_dim=in_features,
        node_feat_dim=in_features,
        memory_dim=16,
        embedding_dim=16,
        out_horizons=2
    )

    x = torch.randn(num_nodes, in_features)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    edge_mask = np.array([0.9, 0.1, 0.8, 0.2])

    validator = ExplanationValidator(model)
    f_plus = validator.compute_fidelity_plus(x, edge_index, edge_mask, target_idx=0)
    f_minus = validator.compute_fidelity_minus(x, edge_index, edge_mask, target_idx=0)

    assert isinstance(f_plus, float)
    assert isinstance(f_minus, float)
