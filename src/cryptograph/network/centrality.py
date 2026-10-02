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

        # Weighted eigenvector centrality via principal eigenvector of adjacency matrix
        nodes = list(G.nodes())
        num_nodes = len(nodes)
        if num_nodes == 0:
            eigenvector = {}
        elif G.number_of_edges() == 0:
            eigenvector = {n: 0.0 for n in nodes}
        else:
            try:
                adj = nx.to_numpy_array(G, nodelist=nodes, weight="weight")
                if G.is_directed():
                    from scipy.sparse.linalg import eigs
                    vals, vecs = eigs(adj.T, k=1, which="LM")
                    ev = np.abs(vecs[:, 0].real)
                else:
                    vals, vecs = np.linalg.eigh(adj)
                    ev = np.abs(vecs[:, -1])
                s = np.sum(ev)
                if s > 1e-12:
                    ev = ev / s
                eigenvector = {n: float(ev[idx]) for idx, n in enumerate(nodes)}
            except Exception:
                try:
                    eigenvector = nx.eigenvector_centrality(G, weight="weight", max_iter=1000)
                except Exception:
                    eigenvector = {n: 0.0 for n in nodes}

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
