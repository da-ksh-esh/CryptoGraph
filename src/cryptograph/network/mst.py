"""Minimum spanning trees of correlation distance (Mantegna 1999) and normalized tree length (Onnela et al. 2003).

Distance d_ij = sqrt(2 (1 - rho_ij)) is 0 for perfectly correlated coins and 2 for perfectly
anti-correlated ones. The MST keeps the N-1 shortest links that connect all coins. Its normalized
length L = mean edge distance shrinks when the market moves as one block (crises).
"""

from __future__ import annotations

import numpy as np


def corr_distance(C: np.ndarray) -> np.ndarray:
    return np.sqrt(np.clip(2.0 * (1.0 - C), 0.0, 4.0))


def mst_edges(C: np.ndarray, active: np.ndarray) -> list[tuple[int, int, float]]:
    """MST over the active nodes with a finite correlation row; returns (i, j, distance) in global indices."""
    from scipy.sparse.csgraph import minimum_spanning_tree

    ok = active & (np.isfinite(np.where(np.eye(len(C), dtype=bool), 0.0, C)).sum(axis=1) > 1)
    idx = np.flatnonzero(ok)
    if len(idx) < 2:
        return []
    D = corr_distance(C[np.ix_(idx, idx)])
    D = np.where(np.isfinite(D), D + 1e-9, 0.0)          # 0 = "no edge" for scipy; keep true zeros as tiny
    np.fill_diagonal(D, 0.0)
    T = minimum_spanning_tree(D).tocoo()
    return [(int(idx[a]), int(idx[b]), float(w - 1e-9)) for a, b, w in zip(T.row, T.col, T.data)]


def normalized_tree_length(C: np.ndarray, active: np.ndarray) -> float:
    e = mst_edges(C, active)
    return float(np.mean([d for _, _, d in e])) if e else float("nan")


def mst_degree(C: np.ndarray, active: np.ndarray) -> np.ndarray:
    """Number of MST links per node (hubs of the tree); NaN for nodes outside the tree."""
    deg = np.full(C.shape[0], np.nan)
    e = mst_edges(C, active)
    if e:
        nodes = sorted({i for i, _, _ in e} | {j for _, j, _ in e})
        deg[nodes] = 0
        for i, j, _ in e:
            deg[i] += 1
            deg[j] += 1
    return deg
