"""Shock persistence at hourly resolution, by shock type.

A shock of coin c at hour t is |z| > z* with z = r_t / sigma30_{t-1} (the project's leakage-free definition,
data/features/targets.py). Only *fresh* shocks are studied: no shock of the same coin in the previous `fresh_hours`
hours, so one turbulent episode counts once. Each fresh shock is typed by

  * breadth: share of the other active coins that shocked in the same hour
      market-wide  >= 30%   |   partial  10-30%   |   idiosyncratic  < 10%
  * direction: crash (r_t < 0) or pump (r_t > 0)

and described by
  * hours_to_normal  first h >= 1 at which the mean |z| over hours t+h-5..t+h is back at or below its mean over the
                     24 hours before the shock (capped at `horizon`; NaN if never within it)
  * reshock_24h      another shock of the same coin within t+1..t+24
  * continuation_h   sign(r_t) * (cumulative return over t+1..t+h) / sigma30_{t-1}; > 0 = the move continues,
                     < 0 = it reverses (h = 1, 6, 24, 72)
  * followers_6h     number of other coins with a shock in t+1..t+6

Everything is descriptive (no model is fitted), so all periods can be used; results are also split by period.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TYPE_ORDER = ["market-wide", "partial", "idiosyncratic"]


def shock_breadth(shock: np.ndarray, active: np.ndarray) -> np.ndarray:
    """(T, N) -> (T, N): share of the *other* active coins that shocked in the same hour."""
    n_sh = shock.sum(1, keepdims=True)
    n_act = active.sum(1, keepdims=True)
    return (n_sh - shock) / np.maximum(n_act - active, 1)


def shock_type(breadth: np.ndarray, wide: float = 0.30, partial: float = 0.10) -> np.ndarray:
    return np.where(breadth >= wide, "market-wide", np.where(breadth >= partial, "partial", "idiosyncratic"))


def fresh_shocks(shock: np.ndarray, fresh_hours: int = 24) -> np.ndarray:
    """Shocks with no shock of the same coin in the previous `fresh_hours` hours."""
    prev = pd.DataFrame(shock.astype(np.int8)).rolling(fresh_hours, min_periods=1).sum().shift(1).fillna(0).to_numpy()
    return shock & (prev == 0)


def persistence_table(r: np.ndarray, z: np.ndarray, sigma: np.ndarray, shock: np.ndarray, valid: np.ndarray,
                      active: np.ndarray, times: pd.DatetimeIndex, coins: list[str], horizon: int = 168,
                      fresh_hours: int = 24, wide: float = 0.30, partial: float = 0.10) -> pd.DataFrame:
    """One row per fresh shock. r, z, sigma (= sigma30 at t-1, the scale of z): (T, N); shock/valid/active: (T, N)."""
    sh = shock & valid
    fresh = fresh_shocks(sh, fresh_hours)
    br = shock_breadth(sh, active & valid)
    az = np.abs(np.nan_to_num(z, nan=np.nan))
    T, N = r.shape
    rows = []
    for t, c in zip(*np.nonzero(fresh)):
        if t < 24 or t + 72 >= T:
            continue
        pre = np.nanmean(az[t - 24:t, c])
        post = az[t + 1:t + 1 + horizon, c]
        roll = pd.Series(post).rolling(6, min_periods=6).mean().to_numpy()
        ok = np.flatnonzero(roll <= pre)
        h_norm = float(ok[0] + 1) if len(ok) and np.isfinite(pre) else np.nan
        s, sc = np.sign(r[t, c]), sigma[t, c]
        cont = {f"continuation_{h}h": float(s * np.nansum(r[t + 1:t + 1 + h, c]) / sc) if sc > 0 else np.nan
                for h in (1, 6, 24, 72)}
        others = np.delete(np.arange(N), c)
        rows.append({"time": times[t], "t": int(t), "coin": coins[c], "z": float(z[t, c]),
                     "direction": "crash" if r[t, c] < 0 else "pump", "breadth": float(br[t, c]),
                     "type": str(shock_type(br[t, c], wide, partial)),
                     "pre_abs_z_24h": float(pre), "hours_to_normal": h_norm,
                     "reshock_24h": bool(sh[t + 1:t + 25, c].any()), **cont,
                     "followers_6h": int(sh[t + 1:t + 7][:, others].any(0).sum())})
    return pd.DataFrame(rows)


def block_bootstrap_ci(df: pd.DataFrame, col: str, by: list[str], stat: str = "median", n_boot: int = 500,
                       seed: int = 0) -> pd.DataFrame:
    """Statistic of `col` per group with a 95% interval from resampling whole days (shocks of one day together)."""
    rng = np.random.default_rng(seed)
    d = df.dropna(subset=[col]).copy()
    d["day"] = pd.factorize(pd.to_datetime(d["time"]).dt.floor("D"))[0]
    f = (lambda x: np.median(x)) if stat == "median" else (lambda x: np.mean(x))
    out = []
    for key, g in d.groupby(by):
        groups = {int(k): v[col].to_numpy(dtype=float) for k, v in g.groupby("day")}
        days = np.fromiter(groups.keys(), dtype=np.int64)
        bs = []
        for _ in range(n_boot):
            pick = rng.choice(days, size=len(days), replace=True)
            bs.append(f(np.concatenate([groups[p] for p in pick])))
        key = key if isinstance(key, tuple) else (key,)
        out.append({**dict(zip(by, key)), "n_shocks": len(g), "n_days": len(days), stat: f(g[col].to_numpy()),
                    "ci_low": float(np.percentile(bs, 2.5)), "ci_high": float(np.percentile(bs, 97.5))})
    return pd.DataFrame(out)


def abs_z_profile(z: np.ndarray, shocks: pd.DataFrame, coins: list[str], by: str = "type", pre: int = 24,
                  post: int = 72) -> pd.DataFrame:
    """Mean |z| of the shocked coin from `pre` hours before to `post` hours after each shock, per group of `by`."""
    az = np.abs(z)
    ci = {c: i for i, c in enumerate(coins)}
    rows = []
    for key, g in shocks.groupby(by):
        t = g["t"].to_numpy()
        c = g["coin"].map(ci).to_numpy()
        ok = (t - pre >= 0) & (t + post < len(az))
        offs = np.arange(-pre, post + 1)
        m = np.nanmean(az[t[ok, None] + offs[None, :], c[ok, None]], axis=0)
        rows.append(pd.DataFrame({"hour": offs, "mean_abs_z": m, by: key}))
    return pd.concat(rows, ignore_index=True)
