import numpy as np
import pandas as pd

from cryptograph.analysis.connectedness import fit_var, generalized_fevd, spillover_table


def _returns(rng, n=600, k=4):
    e = rng.normal(size=(n, k)) * 0.02
    x = np.zeros((n, k))
    A = np.array([[0.1, 0.2, 0.0, 0.0], [0.0, 0.1, 0.0, 0.0], [0.3, 0.0, 0.05, 0.0], [0.0, 0.0, 0.2, 0.0]])
    for t in range(1, n):
        x[t] = A @ x[t - 1] + e[t]
    x[:, 1] += 0.5 * e[:, 0]
    return pd.DataFrame(x, columns=["A", "B", "C", "D"])


def test_gfevd_rows_sum_to_100(rng):
    res = fit_var(_returns(rng), lags=1)
    theta = generalized_fevd(res.coefs, np.asarray(res.sigma_u), H=10)
    assert np.allclose(theta.sum(axis=1), 100.0)
    assert (theta >= 0).all()


def test_gfevd_permutation_invariant(rng):
    df = _returns(rng)
    perm = [2, 0, 3, 1]
    r1 = fit_var(df, lags=2)
    r2 = fit_var(df.iloc[:, perm], lags=2)
    t1 = generalized_fevd(r1.coefs, np.asarray(r1.sigma_u), 10)
    t2 = generalized_fevd(r2.coefs, np.asarray(r2.sigma_u), 10)
    assert np.allclose(t1[np.ix_(perm, perm)], t2, atol=1e-8)
    n1 = spillover_table(t1, list(df.columns))["summary"]["NET"]
    n2 = spillover_table(t2, list(df.columns[perm]))["summary"]["NET"]
    assert np.allclose(n1, n2.reindex(n1.index))
    assert abs(n1.sum()) < 1e-8  # NET spillovers sum to zero
