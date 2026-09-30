"""Independent Cascade Propagation-Process Baseline."""

from typing import Set, List, Dict
import numpy as np
import networkx as nx

class IndependentCascadeModel:
    """
    Classic Independent Cascade (IC) diffusion baseline for shock contagion.
    Estimates empirical edge activation probabilities from training shock cascades.
    """

    def __init__(self, default_p: float = 0.1, max_steps: int = 10, seed: int = 42):
        self.default_p = default_p
        self.max_steps = max_steps
        self.rng = np.random.RandomState(seed)
        self.edge_probs: Dict[tuple, float] = {}

    def fit_probabilities(self, train_cascades: List[List[str]], G: nx.Graph):
        """
        Calibrates pairwise activation probabilities P(v | u) based on co-occurrence in training cascades.
        """
        pair_counts: Dict[tuple, int] = {}
        source_counts: Dict[str, int] = {}

        for cascade in train_cascades:
            for i, u in enumerate(cascade):
                source_counts[u] = source_counts.get(u, 0) + 1
                for v in cascade[i + 1:]:
                    pair = (u, v)
                    pair_counts[pair] = pair_counts.get(pair, 0) + 1

        for u, v in G.edges():
            pair = (u, v)
            if pair in pair_counts and u in source_counts:
                self.edge_probs[pair] = min(0.95, max(0.01, pair_counts[pair] / source_counts[u]))
            else:
                self.edge_probs[pair] = self.default_p

    def simulate(self, G: nx.Graph, seed_nodes: Set[str]) -> Set[str]:
        """
        Simulates cascade diffusion starting from seed active nodes.
        """
        active: Set[str] = set(seed_nodes)
        newly_active: Set[str] = set(seed_nodes)

        for step in range(self.max_steps):
            if not newly_active:
                break
            candidates: Set[str] = set()
            for u in newly_active:
                for v in G.neighbors(u):
                    if v not in active:
                        p = self.edge_probs.get((u, v), self.edge_probs.get((v, u), self.default_p))
                        if self.rng.rand() < p:
                            candidates.add(v)
            newly_active = candidates - active
            active.update(newly_active)

        return active
