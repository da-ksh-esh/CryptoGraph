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
