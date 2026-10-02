"""Model comparison: paired block bootstrap of the AUPRC difference between two scores on the same events.

Events in the same day share market conditions (one crash produces hundreds of correlated events), so the
bootstrap resamples whole days (blocks) with replacement, not single events.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score


def block_bootstrap_auprc_diff(y: np.ndarray, score_a: np.ndarray, score_b: np.ndarray, block: np.ndarray,
                               n_boot: int = 1000, seed: int = 0) -> dict:
    """AUPRC(a) - AUPRC(b) with a 95% interval from resampling blocks; p = share of resamples with diff <= 0
    (one-sided: is a better than b?)."""
    y = np.asarray(y).astype(int)
    a, b = np.asarray(score_a, float), np.asarray(score_b, float)
    blocks, inv = np.unique(block, return_inverse=True)
    members = np.split(np.argsort(inv, kind="stable"), np.cumsum(np.bincount(inv))[:-1])
    rng = np.random.default_rng(seed)
    obs = average_precision_score(y, a) - average_precision_score(y, b)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.concatenate([members[j] for j in rng.integers(0, len(blocks), len(blocks))])
        if y[idx].sum() == 0:
            diffs[i] = np.nan
            continue
        diffs[i] = average_precision_score(y[idx], a[idx]) - average_precision_score(y[idx], b[idx])
    diffs = diffs[np.isfinite(diffs)]
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return {"diff": float(obs), "ci_low": float(lo), "ci_high": float(hi), "p_one_sided": float(np.mean(diffs <= 0)),
            "n_blocks": int(len(blocks)), "n_events": int(len(y))}
