"""Communities: seeded Louvain (networkx) on correlations with the market mode removed, NMI against sectors.

Crypto returns share one dominant factor (the "market mode", the top eigenvector of the
correlation matrix), so Louvain on raw correlations mostly finds one block. Removing that mode
(C - lambda_1 v_1 v_1') leaves the group structure (MacMahon & Garlaschelli 2015); Louvain runs on
the positive residual correlations.
"""

from __future__ import annotations

import numpy as np


def remove_market_mode(C: np.ndarray, active: np.ndarray) -> np.ndarray:
    """Residual correlations after removing the largest eigenmode, over active nodes with finite rows."""
    out = np.full_like(C, np.nan, dtype=float)
    M = np.where(np.eye(len(C), dtype=bool), 1.0, C)
    idx = np.flatnonzero(active)
    idx = idx[np.isfinite(M[np.ix_(idx, idx)]).all(axis=1)]   # drop nodes with any non-estimable pair
    if len(idx) < 3:
        return out
    B = M[np.ix_(idx, idx)]
    w, v = np.linalg.eigh(B)
    R = B - w[-1] * np.outer(v[:, -1], v[:, -1])
    np.fill_diagonal(R, np.nan)
    out[np.ix_(idx, idx)] = R
    return out


def louvain(W: np.ndarray, active: np.ndarray, seed: int = 0, resolution: float = 1.0) -> np.ndarray:
    """Community label per node (-1 = inactive), using the positive weights of W. Labels are ordered
    by community size (0 = largest), ties by smallest member, so they are deterministic for a seed."""
    import networkx as nx

    labels = np.full(W.shape[0], -1)
    idx = np.flatnonzero(active & np.isfinite(np.where(np.eye(len(W), dtype=bool), 0.0, W)).any(axis=1))
    if len(idx) == 0:
        return labels
    G = nx.Graph()
    G.add_nodes_from(idx.tolist())
    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            w = W[idx[a], idx[b]]
            if np.isfinite(w) and w > 0:
                G.add_edge(int(idx[a]), int(idx[b]), weight=float(w))
    comms = nx.community.louvain_communities(G, weight="weight", resolution=resolution, seed=seed)
    comms = sorted((sorted(c) for c in comms), key=lambda c: (-len(c), c[0]))
    for k, c in enumerate(comms):
        labels[c] = k
    return labels


def modularity(W: np.ndarray, labels: np.ndarray) -> float:
    """Newman modularity of a labelling on the positive weights of W (active = labels >= 0)."""
    import networkx as nx

    idx = np.flatnonzero(labels >= 0)
    G = nx.Graph()
    G.add_nodes_from(idx.tolist())
    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            w = W[idx[a], idx[b]]
            if np.isfinite(w) and w > 0:
                G.add_edge(int(idx[a]), int(idx[b]), weight=float(w))
    if G.number_of_edges() == 0:
        return float("nan")
    groups = [set(np.flatnonzero(labels == k).tolist()) for k in np.unique(labels[idx])]
    return float(nx.community.modularity(G, groups, weight="weight"))


def nmi(labels: np.ndarray, reference: np.ndarray) -> float:
    """Normalized mutual information between communities and a reference partition (e.g. sectors), active nodes."""
    from sklearn.metrics import normalized_mutual_info_score

    ok = labels >= 0
    return float(normalized_mutual_info_score(reference[ok], labels[ok])) if ok.sum() > 1 else float("nan")


def align_labels(prev: np.ndarray | None, cur: np.ndarray) -> np.ndarray:
    """Relabel `cur` so communities keep the id of the previous snapshot's community they overlap most
    (greedy, largest overlaps first); unmatched communities get fresh ids. Keeps ids stable over time."""
    if prev is None:
        return cur.copy()
    out = np.full_like(cur, -1)
    pairs = []
    for c in np.unique(cur[cur >= 0]):
        for p in np.unique(prev[prev >= 0]):
            ov = int(((cur == c) & (prev == p)).sum())
            if ov:
                pairs.append((ov, int(c), int(p)))
    used_c, used_p = set(), set()
    for ov, c, p in sorted(pairs, key=lambda x: (-x[0], x[1], x[2])):
        if c not in used_c and p not in used_p:
            out[cur == c] = p
            used_c.add(c)
            used_p.add(p)
    nxt = max([int(prev.max()), int(out.max())]) + 1
    for c in np.unique(cur[cur >= 0]):
        if c not in used_c:
            out[cur == c] = nxt
            nxt += 1
    return out
