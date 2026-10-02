import numpy as np

from cryptograph.analysis.evaluation import block_bootstrap_auprc_diff


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
