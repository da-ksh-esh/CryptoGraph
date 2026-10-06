"""Model comparison: paired block bootstrap of the AUPRC difference between two scores on the same events.

Events in the same day share market conditions (one crash produces hundreds of correlated events), so the
bootstrap resamples whole days (blocks) with replacement, not single events.

A resample is expressed as integer weights (how often each day was drawn), so AUPRC can be computed with one pass
over events pre-sorted by score instead of re-sorting each time: fast enough for 1,000 resamples of ~1M events.
"""

from __future__ import annotations

import numpy as np


class _SortedAP:
    """Average precision (sklearn definition: a step function over distinct score thresholds) for one fixed score
    vector, under arbitrary non-negative sample weights."""

    def __init__(self, y: np.ndarray, score: np.ndarray):
        order = np.argsort(-score, kind="stable")
        self.order = order
        self.y = y[order].astype(np.float64)
        s = score[order]
        self.group_end = np.r_[np.flatnonzero(s[1:] != s[:-1]), len(s) - 1]   # last index of each tied-score group

    def __call__(self, w: np.ndarray) -> float:
        w = w[self.order]
        tp = np.cumsum(w * self.y)[self.group_end]
        n = np.cumsum(w)[self.group_end]
        if tp[-1] <= 0:
            return float("nan")
        precision = tp / np.maximum(n, 1e-12)
        d_recall = np.diff(np.r_[0.0, tp]) / tp[-1]
        return float(np.sum(d_recall * precision))


def weighted_auprc(y: np.ndarray, score: np.ndarray, weight: np.ndarray | None = None) -> float:
    y, score = np.asarray(y), np.asarray(score, float)
    return _SortedAP(y, score)(np.ones(len(y)) if weight is None else np.asarray(weight, float))


def block_bootstrap_auprc_diff(y: np.ndarray, score_a: np.ndarray, score_b: np.ndarray, block: np.ndarray,
                               n_boot: int = 1000, seed: int = 0) -> dict:
    """AUPRC(a) - AUPRC(b) with a 95% interval from resampling blocks; p = share of resamples with diff <= 0
    (one-sided: is a better than b?)."""
    y = np.asarray(y).astype(int)
    ap_a, ap_b = _SortedAP(y, np.asarray(score_a, float)), _SortedAP(y, np.asarray(score_b, float))
    blocks, inv = np.unique(block, return_inverse=True)
    ones = np.ones(len(y))
    obs = ap_a(ones) - ap_b(ones)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        counts = np.bincount(rng.integers(0, len(blocks), len(blocks)), minlength=len(blocks)).astype(float)
        w = counts[inv]
        diffs[i] = ap_a(w) - ap_b(w)
    diffs = diffs[np.isfinite(diffs)]
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return {"diff": float(obs), "ci_low": float(lo), "ci_high": float(hi), "p_one_sided": float(np.mean(diffs <= 0)),
            "n_blocks": int(len(blocks)), "n_events": int(len(y))}


def precision_at_top(y: np.ndarray, score: np.ndarray, fracs=(0.01, 0.05, 0.10)) -> dict[str, float]:
    """Share of real positives among the highest-scored `frac` of cases ("of the riskiest 1% of alerts, how many
    came true"), and its lift over the positive rate. Ties at the cut-off are broken by order (stable sort)."""
    y = np.asarray(y, bool)
    order = np.argsort(-np.asarray(score, float), kind="stable")
    base = float(y.mean()) if y.size else float("nan")
    out = {"positive_rate": base}
    for f in fracs:
        n = max(1, int(round(f * y.size)))
        p = float(y[order[:n]].mean())
        out[f"precision_top{f:.0%}"] = p
        out[f"lift_top{f:.0%}"] = p / base if base > 0 else float("nan")
    return out
