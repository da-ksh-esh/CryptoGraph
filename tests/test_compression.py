"""Unit tests for NetworkCompressor: Top-K, Thresholding, and MST structural preservation."""

import numpy as np
import pandas as pd
import networkx as nx
import pytest
from cryptograph.network.compression import NetworkCompressor


def test_compression_edge_counts_and_mst():
    """Verify that compressed representations enforce expected edge counts."""
    N = 10
    asset_names = [f"COIN_{i}" for i in range(N)]
    np.random.seed(42)
    # Synthetic correlation matrix
    A = np.random.uniform(0.1, 0.9, size=(N, N))
    corr = (A + A.T) / 2.0
    np.fill_diagonal(corr, 1.0)

    # 1. Full Graph
    G_full = NetworkCompressor.build_full_graph(corr, asset_names)
    assert G_full.number_of_nodes() == N
    assert G_full.number_of_edges() == N * (N - 1) // 2

    # 2. MST Backbone
    G_mst = NetworkCompressor.build_mst_graph(corr, asset_names)
    assert G_mst.number_of_nodes() == N
    assert G_mst.number_of_edges() == N - 1
    assert nx.is_tree(G_mst)

    # 3. Top-K Graph (k=3)
    G_top3 = NetworkCompressor.build_topk_graph(corr, asset_names, k=3)
    assert G_top3.number_of_nodes() == N
    # Each node has degree >= 3
    for node in G_top3.nodes():
        assert G_top3.degree(node) >= 3


def test_compression_benchmark_metrics():
    """Verify that run_benchmark computes all metrics and Spearman rank correlations."""
    N = 8
    asset_names = [f"COIN_{i}" for i in range(N)]
    np.random.seed(42)
    A = np.random.uniform(0.2, 0.8, size=(N, N))
    corr = (A + A.T) / 2.0
    np.fill_diagonal(corr, 1.0)

    df_bench = NetworkCompressor.run_benchmark(corr, asset_names, seed=42)

    assert len(df_bench) >= 6
    assert "representation" in df_bench.columns
    assert "edge_reduction_pct" in df_bench.columns
    assert "spearman_degree" in df_bench.columns
    assert "community_nmi" in df_bench.columns

    # Full graph has 0% edge reduction
    row_full = df_bench[df_bench["representation"] == "Full_Graph"].iloc[0]
    assert row_full["edge_reduction_pct"] == 0.0
    assert row_full["spearman_degree"] == 1.0

    # MST has highest edge reduction
    row_mst = df_bench[df_bench["representation"] == "MST_Backbone"].iloc[0]
    assert row_mst["edge_reduction_pct"] > 60.0
