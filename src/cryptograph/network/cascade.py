"""Independent cascade baseline (Kempe, Kleinberg and Tardos 2003): activation probabilities fitted on train.

Each shocked coin i gets one chance to activate (shock) each other coin j within the next k bars,
with probability p_ij(k). Fitted on train source bars only:

    n_ij = #{t: i shocks at t, j's bars (t, t+k] are all observed}
    h_ij = #{those t where j shocks somewhere in (t, t+k]}
    p_ij = (h_ij + m * b_j) / (n_ij + m)        b_j = j's base rate of a shock in any k-bar window

(shrinkage towards the base rate with m pseudo-observations; a coin with no train data gets the
pooled train base rate, so a pair with no train evidence predicts that rate). With several coins shocked at t,
the independent-cascade probability that j is activated is 1 - prod_i (1 - p_ij).

The same counts on a trailing window give the rolling shock-transmission network.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..data.features.targets import future_any


def transmission_counts(src: np.ndarray, anyf: np.ndarray, allv: np.ndarray, rows: np.ndarray
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Counts over source bars `rows` (bool (T,) or int indices). src (T,N): shock and observed at t;
    anyf/allv (T,N): future_any outputs for horizon k. Returns n (N,N), h (N,N), base_n (N,), base_h (N,)."""
    S = src[rows].astype(np.float64)
    V = allv[rows].astype(np.float64)
    A = (anyf[rows] & allv[rows]).astype(np.float64)
    n, h = S.T @ V, S.T @ A
    np.fill_diagonal(n, 0.0)
    np.fill_diagonal(h, 0.0)
    return n, h, V.sum(axis=0), A.sum(axis=0)


def shrunk_probabilities(n: np.ndarray, h: np.ndarray, base_n: np.ndarray, base_h: np.ndarray, m: float,
                         pooled_fallback: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """p_ij shrunk towards b_j, and b_j. Diagonal = 0. Receivers without data get b_j = NaN, or the pooled
    base rate over all coins if `pooled_fallback` (coins listed after the fitting period)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        b = np.where(base_n > 0, base_h / np.maximum(base_n, 1), np.nan)
        if pooled_fallback and base_n.sum() > 0:
            b = np.where(base_n > 0, b, base_h.sum() / base_n.sum())
        P = (h + m * np.nan_to_num(b)[None, :]) / (n + m)
    np.fill_diagonal(P, 0.0)
    return P, b


@dataclass
class IndependentCascade:
    k_bars: list[int]
    prior_strength: float = 10.0
    P: dict[int, np.ndarray] = field(default_factory=dict)       # k -> (N, N) p_ij
    base: dict[int, np.ndarray] = field(default_factory=dict)    # k -> (N,) b_j
    n: dict[int, np.ndarray] = field(default_factory=dict)       # k -> (N, N) source counts

    @classmethod
    def fit(cls, shock: np.ndarray, valid: np.ndarray, source_rows: np.ndarray, k_bars: list[int],
            prior_strength: float = 10.0) -> "IndependentCascade":
        """shock, valid (T, N) bool. `source_rows` (T,) bool: train bars whose label window (t, t+k] also lies
        inside train (e.g. the panel's train sample mask) — nothing after the train period is used."""
        m = cls(list(k_bars), prior_strength)
        src = shock & valid
        for k in m.k_bars:
            anyf, allv = future_any(src, valid, k)
            n, h, bn, bh = transmission_counts(src, anyf, allv, source_rows)
            m.P[k], m.base[k] = shrunk_probabilities(n, h, bn, bh, prior_strength, pooled_fallback=True)
            m.n[k] = n
        return m

    def pairwise(self, k: int, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
        return self.P[k][src, dst]

    def receiver_base(self, k: int, dst: np.ndarray) -> np.ndarray:
        return self.base[k][dst]

    def cascade(self, k: int, shocked: np.ndarray, dst: np.ndarray) -> np.ndarray:
        """1 - prod_{i shocked} (1 - p_ij) per event; shocked (E, N) bool = coins shocked at the event's bar."""
        logq = np.log1p(-np.clip(self.P[k], 0.0, 1.0 - 1e-12))          # diagonal 0 -> log 1 = 0 (j excluded)
        return 1.0 - np.exp((shocked * logq.T[dst]).sum(axis=1))


def rolling_transmission(src: np.ndarray, anyf: np.ndarray, allv: np.ndarray, t_end: int, window: int, k: int,
                         m: float, min_sources: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Shock-transmission network known at bar t_end: sources s in [t_end-window+1, t_end-k], so every
    response window (s, s+k] ends by t_end. Returns P (N,N) and the excess over the base rate
    max(P - b_j, 0) (NaN rows for coins with fewer than `min_sources` shocks)."""
    lo, hi = max(0, t_end - window + 1), t_end - k
    if hi < lo:
        N = src.shape[1]
        return np.full((N, N), np.nan), np.full((N, N), np.nan)
    rows = np.arange(lo, hi + 1)
    n, h, bn, bh = transmission_counts(src, anyf, allv, rows)
    P, b = shrunk_probabilities(n, h, bn, bh, m)
    few = src[rows].sum(axis=0) < min_sources
    P[few, :] = np.nan
    lift = np.clip(P - b[None, :], 0.0, None)
    lift[few, :] = np.nan
    np.fill_diagonal(lift, 0.0)
    return P, lift
