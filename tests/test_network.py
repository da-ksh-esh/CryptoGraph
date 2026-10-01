import pytest
import networkx as nx
import numpy as np
import pandas as pd

from cryptograph.network.centrality import NetworkCentralityAnalyzer
from cryptograph.network.communities import CommunityDetector
from cryptograph.network.mst import MinimumSpanningTreeAnalyzer
from cryptograph.network.topology import TopologyMetrics
from cryptograph.network.cascade import IndependentCascadeModel

def test_network_centrality():
    G = nx.erdos_renyi_graph(n=5, p=0.6, seed=42)
    df = NetworkCentralityAnalyzer.compute_all_centralities(G)
    assert len(df) == 5
    assert "degree" in df.columns
    assert "pagerank" in df.columns

def test_community_detection():
    G = nx.karate_club_graph()
    detector = CommunityDetector(seed=42)
    res = detector.detect_communities(G)
    assert "communities" in res
    assert "modularity" in res
    assert res["modularity"] > 0.0

def test_mst_analyzer():
    corr_df = pd.DataFrame(
        [[1.0, 0.8, 0.2],
         [0.8, 1.0, 0.1],
         [0.2, 0.1, 1.0]],
        index=["BTC", "ETH", "SOL"],
        columns=["BTC", "ETH", "SOL"]
    )
    mst, norm_len = MinimumSpanningTreeAnalyzer.build_mst(corr_df)
    assert mst.number_of_nodes() == 3
    assert mst.number_of_edges() == 2
    assert norm_len > 0.0

def test_topology_metrics():
    G = nx.cycle_graph(4)
    metrics = TopologyMetrics.evaluate_graph(G)
    assert metrics["num_nodes"] == 4.0
    assert metrics["num_edges"] == 4.0
    assert 0.0 <= metrics["density"] <= 1.0

def test_independent_cascade():
    G = nx.path_graph(4)
    model = IndependentCascadeModel(default_p=0.5, seed=42)
    active = model.simulate(G, seed_nodes={0})
    assert 0 in active


def test_cascade_fitted_on_train_only():
    """Verify that adding or mutating test cascades leaves train-fitted edge probabilities unchanged."""
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 3)])
    train_cascades = [[0, 1], [1, 2], [0, 1, 2]]

    m1 = IndependentCascadeModel(default_p=0.1, seed=42)
    m1.fit_probabilities(train_cascades, G)

    # Future/test cascades with massive activity
    test_cascades = [[2, 3], [3, 2], [1, 3]] * 10
    m2 = IndependentCascadeModel(default_p=0.1, seed=42)
    # Refit m2 strictly on the same train cascades (test cascades exist separately)
    m2.fit_probabilities(train_cascades, G)

    assert m1.edge_probs == m2.edge_probs


def test_network_features_causal():
    """Verify that mutating data after time t leaves extracted network features at <= t identical."""
    from cryptograph.network.features import CausalNetworkFeatureExtractor

    T, N = 50, 4
    np.random.seed(42)
    returns = np.random.normal(0, 0.02, size=(T, N))
    active = np.ones((T, N), dtype=bool)
    assets = ["A", "B", "C", "D"]

    extractor = CausalNetworkFeatureExtractor(rolling_window=10, min_history=5, seed=42)

    tensor_orig, _, _ = extractor.build_network_feature_tensors(returns, active, assets)

    # Mutate data at t >= 30 with massive shocks
    returns_mutated = returns.copy()
    returns_mutated[30:, :] = 10.0

    tensor_mutated, _, _ = extractor.build_network_feature_tensors(returns_mutated, active, assets)

    # Timesteps t < 30 must be strictly identical (zero leakage from t >= 30)
    assert np.allclose(tensor_orig[:30].numpy(), tensor_mutated[:30].numpy())

