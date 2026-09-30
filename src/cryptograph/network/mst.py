"""Minimum Spanning Tree (MST): Onnela / Mantegna Ultrametric Space Analysis."""

from typing import Tuple
import networkx as nx
import numpy as np
import pandas as pd

class MinimumSpanningTreeAnalyzer:
    """
    Computes correlation distance metric d_{ij} = sqrt(2 * (1 - rho_{ij}))
    and constructs the Minimum Spanning Tree (MST) to evaluate market topological compression.
    """

    @staticmethod
    def correlation_to_distance(corr_matrix: np.ndarray) -> np.ndarray:
        """Mantegna (1999) metric distance transformation."""
        dist = np.sqrt(np.clip(2.0 * (1.0 - corr_matrix), 0.0, 4.0))
        np.fill_diagonal(dist, 0.0)
        return dist

    @classmethod
    def build_mst(cls, corr_df: pd.DataFrame) -> Tuple[nx.Graph, float]:
        """
        Builds MST from correlation dataframe and returns (mst_graph, normalized_tree_length).
        Onnela et al. (2003) normalized tree length: L(t) = (1 / (N - 1)) * sum_{(i,j) in MST} d_{ij}
        """
        assets = list(corr_df.columns)
        n = len(assets)
        if n <= 1:
            return nx.Graph(), 0.0

        dist_matrix = cls.correlation_to_distance(corr_df.values)

        full_graph = nx.Graph()
        for i in range(n):
            for j in range(i + 1, n):
                full_graph.add_edge(assets[i], assets[j], weight=float(dist_matrix[i, j]))

        mst = nx.minimum_spanning_tree(full_graph, weight="weight")
        total_tree_length = sum(d["weight"] for _, _, d in mst.edges(data=True))
        normalized_tree_length = total_tree_length / (n - 1)

        return mst, float(normalized_tree_length)
