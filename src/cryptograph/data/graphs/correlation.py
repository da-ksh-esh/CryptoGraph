"""NaN-aware rolling correlation graphs (undirected contemporaneous, directed lag-1).

Every graph at day t is estimated from returns on days (t-W, t] only, over
nodes active at t, using pairwise-complete observations (>= min_obs).
"""

from __future__ import annotations

import numpy as np


def pairwise_corr(a: np.ndarray, b: np.ndarray, min_obs: int) -> np.ndarray:
    """corr(a[:, i], b[:, j]) for all i, j with pairwise-complete rows. a, b: (W, N)."""
    ma, mb = np.isfinite(a), np.isfinite(b)
    A, B = np.where(ma, a, 0.0), np.where(mb, b, 0.0)
    fa, fb = ma.astype(float), mb.astype(float)
    n = fa.T @ fb                                   # overlapping counts
    sa, sb = A.T @ fb, fa.T @ B                     # sums over the overlap
    saa, sbb = (A ** 2).T @ fb, fa.T @ (B ** 2)
    sab = A.T @ B
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = sab - sa * sb / n
        va = saa - sa ** 2 / n
        vb = sbb - sb ** 2 / n
        c = cov / np.sqrt(va * vb)
    c[(n < min_obs) | (va <= 1e-14) | (vb <= 1e-14)] = np.nan
    return np.clip(c, -1.0, 1.0)


def rolling_corr_mats(r: np.ndarray, window: int, min_obs: int) -> tuple[np.ndarray, np.ndarray]:
    """(T, N) returns -> contemporaneous C[t] = corr(r_i, r_j) and lag-1 D[t] = corr(r_i[s-1], r_j[s]),
    both over s in (t-window, t]. Shapes (T, N, N); NaN where not estimable."""
    T, N = r.shape
    C = np.full((T, N, N), np.nan, dtype=np.float32)
    D = np.full((T, N, N), np.nan, dtype=np.float32)
    for t in range(T):
        lo = max(0, t - window + 1)
        win = r[lo: t + 1]
        if win.shape[0] >= min_obs:
            C[t] = pairwise_corr(win, win, min_obs)
        if t >= 1 and win.shape[0] >= 2:
            prev = r[max(0, lo - 1): t]            # r[s-1] for s in window
            cur = r[max(1, lo): t + 1]
            m = min(len(prev), len(cur))
            if m >= min_obs:
                D[t] = pairwise_corr(prev[-m:], cur[-m:], min_obs)
    idx = np.arange(N)
    C[:, idx, idx] = np.nan
    D[:, idx, idx] = np.nan
    return C, D


def undirected_edges(C: np.ndarray, active: np.ndarray, threshold: float, top_k: int
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Edges (both directions) where |corr| >= threshold, plus each node's top-k |corr| peers.

    C: (N, N) for one day; active: (N,) bool. Returns edge_index (2, E), weight (E,).
    """
    N = C.shape[0]
    M = np.where(active[:, None] & active[None, :], C, np.nan)
    absM = np.abs(np.nan_to_num(M, nan=0.0))
    keep = absM >= threshold
    if top_k > 0:
        for i in np.flatnonzero(active):
            row = absM[i]
            cand = np.flatnonzero(np.isfinite(M[i]))
            if cand.size:
                best = cand[np.argsort(-row[cand])[:top_k]]
                keep[i, best] = True
    keep = keep | keep.T                             # symmetric
    keep &= np.isfinite(M)
    np.fill_diagonal(keep, False)
    src, dst = np.nonzero(keep)
    return np.stack([src, dst]).astype(np.int64), M[src, dst].astype(np.float32)


def directed_edges(D: np.ndarray, active: np.ndarray, threshold: float, top_k: int = 0
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Edges i -> j where corr(r_i[t-1], r_j[t]) > threshold (plus optional top-k out-edges per source)."""
    M = np.where(active[:, None] & active[None, :], D, np.nan)
    keep = np.nan_to_num(M, nan=-np.inf) > threshold
    if top_k > 0:
        for i in np.flatnonzero(active):
            cand = np.flatnonzero(np.isfinite(M[i]))
            if cand.size:
                keep[i, cand[np.argsort(-M[i, cand])[:top_k]]] = True
    np.fill_diagonal(keep, False)
    src, dst = np.nonzero(keep)
    return np.stack([src, dst]).astype(np.int64), M[src, dst].astype(np.float32)


def threshold_for_density(corr_mats: list[np.ndarray], active: list[np.ndarray], target: float,
                          grid: np.ndarray | None = None) -> tuple[float, float]:
    """Smallest |corr| threshold (0.01 grid) whose mean density over the given snapshots is <= target.

    Pass TRAIN snapshots only. Density = share of active, estimable ordered pairs with |corr| >= threshold
    (top-k additions are not counted). Returns (threshold, achieved mean density).
    """
    grid = np.round(np.arange(0.0, 1.0001, 0.01), 2) if grid is None else np.asarray(grid)
    vals = []
    for C, a in zip(corr_mats, active):
        M = np.abs(C[np.ix_(a, a)])
        off = ~np.eye(M.shape[0], dtype=bool)
        v = M[off & np.isfinite(M)]
        if v.size:
            vals.append(np.sort(v))
    if not vals:
        raise ValueError("no estimable correlation matrices to calibrate on")
    dens = np.array([np.mean([1 - np.searchsorted(v, th, side="left") / v.size for v in vals]) for th in grid])
    ok = np.flatnonzero(dens <= target)
    i = int(ok[0]) if ok.size else len(grid) - 1
    return float(grid[i]), float(dens[i])
