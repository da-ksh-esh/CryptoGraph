"""Centrality on one network snapshot (strength, eigenvector; PageRank on directed networks only).

Matrices are (N, N) over all graph coins; `active` (N,) marks the nodes present in the snapshot.
Inactive nodes get NaN.
"""

from __future__ import annotations

import numpy as np


def _block(W: np.ndarray, active: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    idx = np.flatnonzero(active)
    B = np.nan_to_num(W[np.ix_(idx, idx)].astype(float), nan=0.0)
    np.fill_diagonal(B, 0.0)
    return B, idx


def strength(C: np.ndarray, active: np.ndarray) -> np.ndarray:
    """Mean |weight| of a node's links to the other active nodes."""
    B, idx = _block(np.abs(C), active)
    out = np.full(C.shape[0], np.nan)
    if len(idx) > 1:
        out[idx] = B.sum(axis=1) / (len(idx) - 1)
    return out


def eigenvector_centrality(C: np.ndarray, active: np.ndarray) -> np.ndarray:
    """Perron eigenvector of the |weight| matrix (undirected), scaled to sum to 1."""
    B, idx = _block(np.abs(C), active)
    out = np.full(C.shape[0], np.nan)
    if len(idx) < 2 or not B.any():
        return out
    w, v = np.linalg.eigh((B + B.T) / 2)
    vec = np.abs(v[:, np.argmax(w)])
    out[idx] = vec / vec.sum()
    return out


def pagerank(W: np.ndarray, active: np.ndarray, alpha: float = 0.85, tol: float = 1e-12, max_iter: int = 1000
             ) -> np.ndarray:
    """Weighted PageRank of a DIRECTED network; W[i, j] >= 0 is the weight of edge i -> j.

    Rank flows along edges, so a node scores high when important nodes point to it. Nodes
    without out-links spread their rank uniformly (as in networkx).
    """
    B, idx = _block(np.clip(W, 0, None), active)
    out = np.full(W.shape[0], np.nan)
    n = len(idx)
    if n == 0:
        return out
    rs = B.sum(axis=1, keepdims=True)
    P = np.where(rs > 0, B / np.where(rs > 0, rs, 1), 1.0 / n)
    pr = np.full(n, 1.0 / n)
    for _ in range(max_iter):
        new = (1 - alpha) / n + alpha * (pr @ P)
        if np.abs(new - pr).sum() < tol:
            pr = new
            break
        pr = new
    out[idx] = pr / pr.sum()
    return out


def transmitter_pagerank(theta: np.ndarray, active: np.ndarray, alpha: float = 0.85) -> np.ndarray:
    """PageRank of a "depends on" network: theta[i, j] = how much i is driven by j (e.g. j's share of
    i's forecast-error variance, or i's shock response to j). Edge i -> j carries theta[i, j], so rank
    collects at the coins that others depend on: the systemically important transmitters."""
    return pagerank(theta, active, alpha)
