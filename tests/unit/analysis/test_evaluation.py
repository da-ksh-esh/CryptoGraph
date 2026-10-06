import numpy as np

from sklearn.metrics import average_precision_score

from cryptograph.analysis.evaluation import block_bootstrap_auprc_diff, weighted_auprc


def test_weighted_auprc_matches_sklearn_with_ties_and_weights():
    rng = np.random.default_rng(1)
    y = rng.uniform(size=3000) < 0.1
    s = np.round(rng.normal(size=3000) + y, 1)                  # many tied scores
    w = rng.integers(0, 4, size=3000).astype(float)
    assert np.isclose(weighted_auprc(y, s), average_precision_score(y, s))
    assert np.isclose(weighted_auprc(y, s, w), average_precision_score(y, s, sample_weight=w))


def test_block_bootstrap_detects_a_better_score_and_not_an_equal_one():
    rng = np.random.default_rng(0)
    n = 6000
    block = np.repeat(np.arange(200), n // 200)
    y = rng.uniform(size=n) < 0.1
    good = y * 1.0 + rng.normal(0, 0.8, n)
    noise = rng.normal(0, 1, n)
    r = block_bootstrap_auprc_diff(y, good, noise, block, n_boot=300)
    assert r["diff"] > 0 and r["ci_low"] > 0 and r["p_one_sided"] < 0.01 and r["n_blocks"] == 200
    same = block_bootstrap_auprc_diff(y, good, good + 1e-12 * rng.normal(size=n), block, n_boot=300)
    assert abs(same["diff"]) < 1e-6 and same["ci_low"] <= 0 <= same["ci_high"]


def test_precision_at_top():
    from cryptograph.analysis.evaluation import precision_at_top

    y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    s = np.array([0.9, 0.8, 0.7, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    r = precision_at_top(y, s, fracs=(0.1, 0.3))
    assert r["positive_rate"] == 0.2
    assert r["precision_top10%"] == 1.0 and r["lift_top10%"] == 5.0
    assert abs(r["precision_top30%"] - 2 / 3) < 1e-12
