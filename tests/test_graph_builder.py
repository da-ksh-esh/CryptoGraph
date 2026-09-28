import pytest
import pandas as pd
from cryptograph.data.graph_builder import DynamicGraphBuilder

def test_dynamic_graph_builder():
    builder = DynamicGraphBuilder(correlation_threshold=0.5)
    returns = pd.DataFrame({
        "BTC": [0.01, -0.02, 0.03, -0.01],
        "ETH": [0.012, -0.018, 0.028, -0.009],
        "SOL": [-0.01, 0.02, -0.01, 0.01]
    })
    edge_index, edge_weight = builder.build_adjacency(returns)
    assert edge_index.shape[0] == 2
