"""Social-network analysis and graph theory on the coin networks (Stage 5, notebook 02_network_analysis).

- centrality: strength, eigenvector; PageRank on directed networks only
- communities: seeded Louvain (networkx) on market-mode-removed correlations, modularity, NMI against sectors
- mst: minimum spanning trees (Mantegna 1999), normalized tree length (Onnela et al. 2003)
- topology: density, clustering, shortest paths, k-core, sector assortativity
- spillover: Diebold-Yilmaz GFEVD networks on fixed or rolling windows
- cascade: independent cascade baseline (Kempe, Kleinberg & Tardos 2003) and rolling shock-transmission networks
- features: leakage-free per-snapshot node features for the models

All networks are built from trailing windows only (no look-ahead).
"""

from .cascade import IndependentCascade, rolling_transmission
from .features import NODE_FEATURES, NetworkParams, compute_network_features

__all__ = ["IndependentCascade", "rolling_transmission", "NODE_FEATURES", "NetworkParams", "compute_network_features"]
