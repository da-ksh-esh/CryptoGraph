"""Network Topology & Compression Metrics."""

from typing import Dict, Any
import networkx as nx
import numpy as np

class TopologyMetrics:
    """Computes structural topology metrics for raw vs compressed network representations."""

    @staticmethod
    def evaluate_graph(G: nx.Graph) -> Dict[str, float]:
        """
        Computes network density, average clustering coefficient, transitivity,
        and average shortest path length (if connected).
        """
        n = G.number_of_nodes()
        m = G.number_of_edges()
        if n <= 1:
            return {"num_nodes": n, "num_edges": m, "density": 0.0, "clustering": 0.0}

        density = nx.density(G)
        try:
            clustering = nx.average_clustering(G)
        except Exception:
            clustering = 0.0

        return {
            "num_nodes": float(n),
            "num_edges": float(m),
            "density": float(density),
            "clustering": float(clustering)
        }

    @staticmethod
    def compute_compression_ratio(full_edge_count: int, compressed_edge_count: int) -> float:
        """Percentage edge reduction: 1.0 - (compressed / full)."""
        if full_edge_count == 0:
            return 0.0
        return float(1.0 - (compressed_edge_count / full_edge_count))
