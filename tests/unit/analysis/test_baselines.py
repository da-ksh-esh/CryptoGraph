import numpy as np

from cryptograph.analysis.baselines import har_design, har_fit, har_predict, trailing_rv
from cryptograph.data.features.targets import forward_realized_vol


def test_trailing_rv_is_the_backward_mirror_of_the_forward_target():
    rng = np.random.default_rng(0)
    r = rng.normal(0, 0.01, (200, 3))
    r[50, 1] = np.nan
    tr = trailing_rv(r, 6)
    fw = forward_realized_vol(r, [6])[..., 0]
    assert np.allclose(tr[10:], fw[4:-6], equal_nan=True)          # trailing at t == forward at t-6
    assert np.isnan(tr[:5]).all() and np.isnan(tr[50:56, 1]).all() and np.isfinite(tr[56, 1])


def test_har_recovers_a_linear_volatility_model():
    rng = np.random.default_rng(1)
    r = rng.normal(0, 0.01, (3000, 2)) * np.exp(rng.normal(0, 0.3, (3000, 2)))
    X = har_design(r, (1, 6, 24))
    y = 0.001 + 0.2 * X[..., 0] + 0.5 * X[..., 2] + rng.normal(0, 1e-4, X.shape[:2])
    beta = har_fit(X.reshape(-1, 3), y.reshape(-1))
    assert np.allclose(beta, [0.001, 0.2, 0.0, 0.5], atol=2e-3)
    assert (har_predict(X[23:].reshape(-1, 3), beta) >= 0).all()          # first 23 bars have no 24-bar window
