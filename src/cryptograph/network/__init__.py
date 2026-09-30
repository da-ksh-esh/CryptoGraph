"""Social-network analysis and graph theory on the coin networks (filled in during Stage 5).

- centrality: strength, eigenvector, betweenness; PageRank on directed networks only
- communities: seeded Louvain (networkx), modularity over time, NMI against sectors
- mst: minimum spanning trees (Mantegna 1999), normalized tree length (Onnela et al. 2003)
- topology: density, clustering, shortest paths, k-core around events
- cascade: independent cascade baseline (Kempe, Kleinberg & Tardos 2003)

All networks are built from trailing windows only (no look-ahead).
"""
