"""Community Detection: Seeded Louvain Communities & Modularity Analysis."""

from typing import List, Set, Dict, Any
import networkx as nx
from networkx.algorithms.community import louvain_communities, modularity
import numpy as np
import pandas as pd

class CommunityDetector:
    """
    Detects dynamic cryptocurrency market communities using seeded Louvain modularity optimization.
    """

    def __init__(self, seed: int = 42, resolution: float = 1.0):
        self.seed = seed
        self.resolution = resolution

    def detect_communities(self, G: nx.Graph, weight: str = "weight") -> Dict[str, Any]:
        """
        Detects communities and computes modularity score.
        """
        if G.number_of_nodes() == 0:
            return {"communities": [], "modularity": 0.0, "node_community": {}}

        # Louvain on undirected graph with positive weights
        undirected_G = G.to_undirected() if G.is_directed() else G
        comms = louvain_communities(undirected_G, weight=weight, resolution=self.resolution, seed=self.seed)

        mod_score = modularity(undirected_G, comms, weight=weight)

        node_map = {}
        for c_idx, comm in enumerate(comms):
            for node in comm:
                node_map[node] = c_idx

        return {
            "communities": comms,
            "modularity": float(mod_score),
            "node_community": node_map
        }
