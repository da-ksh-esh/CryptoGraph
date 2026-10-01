"""Network Centrality Measures: Degree, Betweenness, Closeness, Eigenvector, and PageRank."""

from typing import Dict
import networkx as nx
import numpy as np
import pandas as pd

class NetworkCentralityAnalyzer:
    """Computes node centrality measures on static and dynamic network snapshots."""

    @staticmethod
    def compute_all_centralities(G: nx.Graph) -> pd.DataFrame:
        """
        Computes degree, betweenness, closeness, eigenvector centrality, and PageRank (if directed).
        """
        degree = nx.degree_centrality(G)
        strength = dict(G.degree(weight="weight"))
        betweenness = nx.betweenness_centrality(G)
        closeness = nx.closeness_centrality(G)

        try:
            eigenvector = nx.eigenvector_centrality_numpy(G, weight="weight")
        except Exception:
            eigenvector = {n: 0.0 for n in G.nodes()}

        if G.is_directed():
            pagerank = nx.pagerank(G, alpha=0.85, weight="weight")
        else:
            pagerank = nx.pagerank(G.to_directed(), alpha=0.85, weight="weight")

        df = pd.DataFrame({
            "degree": degree,
            "strength": strength,
            "betweenness": betweenness,
            "closeness": closeness,
            "eigenvector": eigenvector,
            "pagerank": pagerank
        })
        return df
