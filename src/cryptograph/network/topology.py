"""Network topology of one thresholded snapshot: density, clustering, path length, k-core, assortativity."""

from __future__ import annotations

import numpy as np


def threshold_adjacency(C: np.ndarray, active: np.ndarray, threshold: float) -> np.ndarray:
    """Undirected adjacency: |corr| >= threshold among active nodes (no top-k additions)."""
    A = (np.abs(np.nan_to_num(C, nan=0.0)) >= threshold) & active[:, None] & active[None, :]
    np.fill_diagonal(A, False)
    return A | A.T


def topology_metrics(A: np.ndarray, active: np.ndarray, attr: np.ndarray | None = None) -> dict:
    """density, average clustering, average shortest path (largest component), largest k-core index,
    number of components, and attribute assortativity (e.g. sector) when `attr` is given."""
    import networkx as nx

    idx = np.flatnonzero(active)
    n = len(idx)
    G = nx.Graph()
    G.add_nodes_from(idx.tolist())
    rows, cols = np.nonzero(np.triu(A[np.ix_(idx, idx)], 1))
    G.add_edges_from((int(idx[a]), int(idx[b])) for a, b in zip(rows, cols))
    out = {"n_nodes": n, "n_edges": G.number_of_edges(),
           "density": G.number_of_edges() / (n * (n - 1) / 2) if n > 1 else np.nan,
           "avg_clustering": nx.average_clustering(G) if n else np.nan,
           "n_components": nx.number_connected_components(G) if n else 0,
           "max_core": max(nx.core_number(G).values()) if G.number_of_edges() else 0}
    if n > 1:
        big = G.subgraph(max(nx.connected_components(G), key=len))
        out["avg_shortest_path"] = nx.average_shortest_path_length(big) if big.number_of_nodes() > 1 else np.nan
        out["largest_component_frac"] = big.number_of_nodes() / n
    else:
        out["avg_shortest_path"], out["largest_component_frac"] = np.nan, np.nan
    if attr is not None:
        nx.set_node_attributes(G, {int(i): attr[i] for i in idx}, "attr")
        try:
            with np.errstate(invalid="ignore", divide="ignore"):
                out["attr_assortativity"] = float(nx.attribute_assortativity_coefficient(G, "attr"))
        except (ZeroDivisionError, ValueError):
            out["attr_assortativity"] = np.nan
    return out
