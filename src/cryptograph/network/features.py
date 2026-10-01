"""Leakage-free network node features and network-level series, one row per graph snapshot.

The snapshot at bar t uses returns and shocks up to bar t only:
  * correlation network: trailing `corr_window` bars (same windows as graph_snapshots.pt)
  * Diebold-Yilmaz GFEVD network: trailing `spill_window` bars
  * shock-transmission network: source shocks s with s + k <= t inside the trailing `cascade_window`
A model at bar t should use the latest snapshot with snapshot_t <= t.

Eigenvector centrality and PageRank are rescaled to mean 1 over the coins in the snapshot, so their
level does not change when coins are listed (a value of 2 = twice the average coin).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..data.features.targets import future_any
from ..data.graphs.correlation import pairwise_corr
from .cascade import rolling_transmission
from .centrality import eigenvector_centrality, strength, transmitter_pagerank
from .communities import align_labels, louvain, modularity, nmi, remove_market_mode
from .mst import mst_degree, normalized_tree_length
from .spillover import spillover_summary, window_coins, window_gfevd
from .topology import threshold_adjacency, topology_metrics

NODE_FEATURES = ["strength", "eigenvector", "mst_degree", "community_size_frac", "dy_pagerank", "dy_net",
                 "shock_pagerank", "shock_out_lift"]


@dataclass
class NetworkParams:
    corr_window: int          # bars
    spill_window: int         # bars
    spill_lags: int
    spill_horizon: int        # bars
    cascade_window: int       # bars
    cascade_k: int            # bars
    cascade_prior: float
    threshold: float          # |corr| threshold for topology metrics (the fitted graph threshold)
    louvain_resolution: float = 1.0
    pagerank_alpha: float = 0.85
    min_coverage: float = 0.9
    min_obs_frac: float = 0.8
    seed: int = 42


def compute_network_features(r: np.ndarray, shock: np.ndarray, valid: np.ndarray, active: np.ndarray,
                             snap_t: np.ndarray, sector_id: np.ndarray, p: NetworkParams
                             ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """r (T,N) deseasonalised returns, shock/valid/active (T,N) bool for the graph coins.

    Returns X (S, N, F) float32 (NaN where undefined), community (S, N) int (-1 = inactive; ids aligned
    over time) and a per-snapshot DataFrame of network-level measures."""
    S, N = len(snap_t), r.shape[1]
    X = np.full((S, N, len(NODE_FEATURES)), np.nan, np.float32)
    comm = np.full((S, N), -1, np.int64)
    src = shock & valid
    anyf, allv = future_any(src, valid, p.cascade_k)
    rows, prev = [], None
    min_obs = int(p.min_obs_frac * p.corr_window)
    for s, t in enumerate(snap_t):
        g = {"t": int(t)}
        if t >= p.corr_window - 1:
            win = r[t - p.corr_window + 1: t + 1]
            C = pairwise_corr(win, win, min_obs)
            np.fill_diagonal(C, np.nan)
            act = active[t] & (np.isfinite(C).sum(axis=1) > 0)
            X[s, :, 0] = strength(C, act)
            X[s, :, 1] = eigenvector_centrality(C, act) * act.sum()           # mean 1 over active coins
            X[s, :, 2] = mst_degree(C, act)
            R = remove_market_mode(C, act)
            lab = louvain(R, act & np.isfinite(R).any(axis=1), seed=p.seed, resolution=p.louvain_resolution)
            lab = align_labels(prev, lab)
            prev = lab
            comm[s] = lab
            n_act = (lab >= 0).sum()
            for c in np.unique(lab[lab >= 0]):
                X[s, lab == c, 3] = (lab == c).sum() / n_act
            off = C[act][:, act]
            A = threshold_adjacency(C, act, p.threshold)
            g |= {"n_active": int(act.sum()), "mean_corr": float(np.nanmean(off)),
                  "tree_length": normalized_tree_length(C, act),
                  "n_communities": int(len(np.unique(lab[lab >= 0]))), "modularity": modularity(R, lab),
                  "nmi_sector": nmi(lab, sector_id)}
            g |= {k: v for k, v in topology_metrics(A, act, sector_id).items() if k not in ("n_nodes",)}
        if t >= p.spill_window - 1:
            idx = window_coins(r, active, t, p.spill_window, p.min_coverage)
            theta = window_gfevd(r[t - p.spill_window + 1: t + 1][:, idx], p.spill_horizon, p.spill_lags) \
                if len(idx) >= 3 else None
            if theta is not None:
                Th = np.zeros((N, N))
                Th[np.ix_(idx, idx)] = theta
                m = np.zeros(N, bool)
                m[idx] = True
                X[s, :, 4] = transmitter_pagerank(Th, m, p.pagerank_alpha) * m.sum()
                summ = spillover_summary(theta)
                X[s, idx, 5] = summ["net"]
                g["total_spillover"] = summ["total"]
        if t >= p.cascade_k:
            _, lift = rolling_transmission(src, anyf, allv, int(t), p.cascade_window, p.cascade_k, p.cascade_prior)
            recv = active[t] & np.isfinite(lift).any(axis=0)               # receivers with a base rate
            act_c = recv & np.isfinite(lift[:, recv]).all(axis=1)          # sources with enough shocks
            if act_c.sum() >= 3:
                X[s, :, 6] = transmitter_pagerank(np.nan_to_num(lift.T), act_c, p.pagerank_alpha) * act_c.sum()
                L = np.where(act_c[None, :] & ~np.eye(N, dtype=bool), lift, np.nan)
                X[s, act_c, 7] = np.nanmean(L[act_c], axis=1)
                g["mean_shock_lift"] = float(np.nanmean(L[act_c]))
        rows.append(g)
    return X, comm, pd.DataFrame(rows)
