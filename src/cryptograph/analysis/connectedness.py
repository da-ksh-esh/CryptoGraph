"""Diebold-Yilmaz (2012) connectedness with the generalized FEVD (Pesaran-Shin 1998).

The generalized FEVD does not depend on variable ordering, unlike the Cholesky
FEVD of `statsmodels` `VARResults.fevd()`, so the NET ranking is well defined.

    theta_ij(H) = sigma_jj^-1 * sum_{h<H} (e_i' Phi_h Sigma e_j)^2 / sum_{h<H} (e_i' Phi_h Sigma Phi_h' e_i)
    rows normalised to 1 (reported x100).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def ma_coefficients(coefs: np.ndarray, H: int) -> np.ndarray:
    """VAR(p) coefficients (p, k, k) -> MA matrices Phi_0..Phi_{H-1} (H, k, k)."""
    p, k, _ = coefs.shape
    phi = np.zeros((H, k, k))
    phi[0] = np.eye(k)
    for h in range(1, H):
        for lag in range(1, min(h, p) + 1):
            phi[h] += coefs[lag - 1] @ phi[h - lag]
    return phi


def generalized_fevd(coefs: np.ndarray, sigma: np.ndarray, H: int = 10) -> np.ndarray:
    """Row-normalised generalized FEVD in percent; rows sum to 100. Entry [i, j] = share of i's
    H-step forecast error variance due to shocks in j."""
    phi = ma_coefficients(coefs, H)
    k = sigma.shape[0]
    num = np.zeros((k, k))
    den = np.zeros(k)
    sdiag = np.diag(sigma)
    for h in range(H):
        ps = phi[h] @ sigma                      # (k, k): e_i' Phi_h Sigma e_j
        num += ps ** 2
        den += np.einsum("ij,jk,ik->i", phi[h], sigma, phi[h])  # e_i' Phi Sigma Phi' e_i
    theta = num / sdiag[None, :] / den[:, None]
    return 100.0 * theta / theta.sum(axis=1, keepdims=True)


def spillover_table(theta: np.ndarray, names: list[str]) -> dict:
    """TO / FROM / NET and the total spillover index from a row-normalised (x100) FEVD matrix."""
    k = len(names)
    off = theta.copy()
    np.fill_diagonal(off, 0.0)
    to_ = off.sum(axis=0)
    from_ = off.sum(axis=1)
    table = pd.DataFrame(theta, index=names, columns=names)
    table["FROM"] = from_
    summary = pd.DataFrame({"TO": to_, "FROM": from_, "NET": to_ - from_}, index=names)
    return {"matrix": table, "summary": summary, "total": float(off.sum() / k),
            "net_ranking": summary["NET"].sort_values(ascending=False)}


def fit_var(returns: pd.DataFrame, maxlags: int = 5, ic: str = "bic", lags: int | None = None):
    from statsmodels.tsa.api import VAR

    model = VAR(returns.dropna())
    if lags is not None:
        return model.fit(lags)
    res = model.fit(maxlags=maxlags, ic=ic)
    return res if res.k_ar > 0 else model.fit(1)


def dy_connectedness(returns: pd.DataFrame, H: int = 10, maxlags: int = 5, ic: str = "bic", lags: int | None = None
                     ) -> dict:
    res = fit_var(returns, maxlags=maxlags, ic=ic, lags=lags)
    theta = generalized_fevd(res.coefs, np.asarray(res.sigma_u), H)
    out = spillover_table(theta, list(returns.columns))
    out["lag_order"] = int(res.k_ar)
    out["ic_table"] = None
    return out


def rolling_total_spillover(returns: pd.DataFrame, window: int = 200, lags: int = 1, H: int = 10, step: int = 5
                            ) -> pd.Series:
    """Total spillover index on rolling windows (fixed lag order for speed/stability)."""
    from statsmodels.tsa.api import VAR

    vals, idx = [], []
    X = returns.dropna()
    for end in range(window, len(X) + 1, step):
        w = X.iloc[end - window: end]
        res = VAR(w).fit(lags)
        theta = generalized_fevd(res.coefs, np.asarray(res.sigma_u), H)
        off = theta.copy()
        np.fill_diagonal(off, 0.0)
        vals.append(off.sum() / theta.shape[0])
        idx.append(w.index[-1])
    return pd.Series(vals, index=pd.DatetimeIndex(idx), name="total_spillover")
