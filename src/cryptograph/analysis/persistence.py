"""Shock persistence at hourly resolution, by shock type.

A shock of coin c at hour t is |z| > z* with z = r_t / sigma_{t-1}, sigma = trailing volatility (168 hours on the
hourly data, `targets.shock_vol_window_hours`; the project's leakage-free definition, data/features/targets.py).
Only *fresh* shocks are studied: no shock of the same coin in the previous `fresh_hours` hours, so one turbulent
episode counts once. Each fresh shock is typed by

  * breadth: share of the other active coins that shocked in the same hour
      market-wide  >= 30%   |   partial  10-30%   |   idiosyncratic  < 10%
  * direction: crash (r_t < 0) or pump (r_t > 0)

and described by
  * hours_to_normal  first h >= 1 at which the mean |z| over hours t+h-5..t+h is back at or below its mean over the
                     24 hours before the shock (capped at `horizon`; NaN if never within it)
  * reshock_24h      another shock of the same coin within t+1..t+24
  * continuation_h   sign(r_t) * (cumulative return over t+1..t+h) / sigma_{t-1}; > 0 = the move continues,
                     < 0 = it reverses (h = 1, 6, 24, 72)
  * followers_6h     number of other coins with a shock in t+1..t+6

Everything is descriptive (no model is fitted), so all periods can be used; results are also split by period.

Against a matched null (ported from the earlier persistence work; `excess_vs_null`, `sector_half_life`): for each
fresh shock, |r_ds| over the next `horizon` hours relative to the coin's median |r_ds| from 168 to 24 hours before
(r_ds = deseasonalised return, no rolling smoothing), minus the same profile after a random calm hour of the same coin
in the same calendar month (no shock in the 24 hours up to it). The gap is the excess volatility that a shock adds over
an ordinary hour in the same regime; its half-life (first hour at which it is <= half its value at hour 1) is reported
per sector with >= 3 coins, with day-block bootstrap intervals. `event_blast_radius` measures the catalogued crises:
coins shocking within 72 hours, abnormal shock rate against the 168-24 hours before, and the hour of peak spillover.
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
    """One row per fresh shock. r, z, sigma (= trailing volatility at t-1, the scale of z): (T, N); shock/valid/active: (T, N)."""
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


def decay_half_life(gap: np.ndarray) -> float:
    """First hour (linearly interpolated, 1-based) at which `gap` is <= half its value at hour 1; NaN if the gap at
    hour 1 is not positive, len(gap) if it never halves."""
    gap = np.asarray(gap, float)
    if not len(gap) or not np.isfinite(gap[0]) or gap[0] <= 0:
        return float("nan")
    half = 0.5 * gap[0]
    cross = np.flatnonzero(gap <= half)
    if not len(cross):
        return float(len(gap))
    i = int(cross[0])
    return float(i + (gap[i - 1] - half) / (gap[i - 1] - gap[i]))       # i >= 1, since gap[0] > half


def excess_vs_null(r_ds: np.ndarray, shock: np.ndarray, valid: np.ndarray, active: np.ndarray,
                   times: pd.DatetimeIndex, horizon: int = 72, base_from: int = 168, base_to: int = 24,
                   fresh_hours: int = 24, seed: int = 0) -> dict:
    """Shock and matched-null |r_ds| profiles (see the module docstring). Returns t, coin index (n,), and sh, nu (n, H)
    profiles relative to each episode's own pre-window median."""
    rng = np.random.default_rng(seed)
    sh = shock & valid
    fresh = fresh_shocks(sh, fresh_hours)
    calm = ~(pd.DataFrame(sh.astype(np.int8)).rolling(fresh_hours + 1, min_periods=1).max().to_numpy() > 0)
    absr = np.abs(r_ds)
    T, N = r_ds.shape
    ym = times.year.to_numpy() * 100 + times.month.to_numpy()
    full = lambda t, c: bool(active[t - base_from:t + horizon + 1, c].all())                        # noqa: E731

    def profile(t, c):
        base = np.nanmedian(absr[t - base_from:t - base_to, c])
        return absr[t + 1:t + horizon + 1, c] / base if base > 1e-8 else None

    out = {"t": [], "coin": [], "sh": [], "nu": []}
    for c in range(N):
        ok = np.arange(T)
        ok = ok[(ok >= base_from) & (ok + horizon < T)]
        cand = ok[calm[ok, c] & active[ok, c]]                          # calm, active hours with a full window
        nulls = pd.Series(cand).groupby(ym[cand]).apply(np.asarray)
        for t in np.flatnonzero(fresh[:, c]):
            if t < base_from or t + horizon >= T or not full(t, c) or ym[t] not in nulls.index:
                continue
            ps = profile(t, c)
            tn = int(rng.choice(nulls[ym[t]]))
            pn = profile(tn, c) if full(tn, c) else None
            if ps is None or pn is None or not (np.isfinite(ps).all() and np.isfinite(pn).all()):
                continue
            out["t"].append(t), out["coin"].append(c), out["sh"].append(ps), out["nu"].append(pn)
    return {"t": np.asarray(out["t"], np.int64), "coin": np.asarray(out["coin"], np.int64),
            "sh": np.asarray(out["sh"]).reshape(-1, horizon), "nu": np.asarray(out["nu"]).reshape(-1, horizon)}


def sector_half_life(ep: dict, coins: list[str], sector_of: dict[str, str], times: pd.DatetimeIndex,
                     min_coins: int = 3, n_boot: int = 500, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per sector with >= `min_coins` coins: episodes, excess gap at 1 / 24 / 72 h, half-life of the gap with a 95%
    interval and p(gap at 1 h <= 0) from resampling whole days (the same days for every sector), plus pairwise
    half-life differences between sectors with their intervals."""
    sec = np.array([sector_of.get(coins[c], "other") for c in ep["coin"]])
    counts = pd.Series([sector_of.get(c, "other") for c in coins]).value_counts()
    sectors = sorted(s for s, n in counts.items() if n >= min_coins and s != "other")
    day = pd.factorize(times[ep["t"]].floor("D"))[0] if len(ep["t"]) else np.zeros(0, int)
    n_days = int(day.max()) + 1 if len(day) else 0
    rng = np.random.default_rng(seed)
    draws = [np.bincount(rng.integers(0, n_days, n_days), minlength=n_days) for _ in range(n_boot)] if n_days else []
    H = ep["sh"].shape[1]
    point, boot, rows = {}, {}, []
    for s in sectors:
        m = sec == s
        if not m.any():
            continue
        gap = ep["sh"][m].mean(0) - ep["nu"][m].mean(0)
        point[s] = decay_half_life(gap)
        hl_b, g1_b = [], []
        for w_day in draws:
            w = w_day[day[m]].astype(float)
            if w.sum() == 0:
                hl_b.append(np.nan), g1_b.append(np.nan)
                continue
            gb = (w[:, None] * (ep["sh"][m] - ep["nu"][m])).sum(0) / w.sum()
            hl_b.append(decay_half_life(gb)), g1_b.append(gb[0])
        boot[s] = np.asarray(hl_b)
        hb, gb1 = boot[s][np.isfinite(boot[s])], np.asarray(g1_b)[np.isfinite(g1_b)]
        rows.append({"sector": s, "coins": int(counts[s]), "episodes": int(m.sum()),
                     "days": int(len(np.unique(day[m]))), "gap_h1": float(gap[0]),
                     "gap_h24": float(gap[min(23, H - 1)]), "gap_h72": float(gap[min(71, H - 1)]),
                     "half_life_h": point[s],
                     "ci_low": float(np.quantile(hb, 0.025)) if len(hb) else np.nan,
                     "ci_high": float(np.quantile(hb, 0.975)) if len(hb) else np.nan,
                     "p_gap_h1_not_positive": float(np.mean(gb1 <= 0)) if len(gb1) else np.nan})
    pairs = []
    done = list(point)
    for i, a in enumerate(done):
        for b in done[i + 1:]:
            dd = boot[a] - boot[b]
            dd = dd[np.isfinite(dd)]
            pairs.append({"sector_a": a, "sector_b": b, "half_life_diff_h": point[a] - point[b],
                          "ci_low": float(np.quantile(dd, 0.025)) if len(dd) else np.nan,
                          "ci_high": float(np.quantile(dd, 0.975)) if len(dd) else np.nan})
    return pd.DataFrame(rows), pd.DataFrame(pairs)


def event_blast_radius(shock: np.ndarray, valid: np.ndarray, active: np.ndarray, times: pd.DatetimeIndex,
                       coins: list[str], events: pd.DataFrame, horizon: int = 72, control_from: int = 168,
                       control_to: int = 24) -> pd.DataFrame:
    """One row per catalogued event inside the data: anchored at the first hour of the event date with the most
    shocks (the calendar date alone can sit hours away from the shock), typed by breadth at that hour. (No "most hit
    sector": the crises hit nearly every coin, so it would only name the sector with the most coins.)"""
    sh, act = shock & valid, active & valid
    br = shock_breadth(sh, act)
    rows = []
    for e in events.itertuples():
        day = np.flatnonzero(times.floor("D") == pd.Timestamp(e.date).floor("D"))
        if not len(day) or day[0] < control_from or day[0] + horizon >= len(times):
            continue
        t0 = int(day[np.argmax(sh[day].sum(1))])
        first = np.flatnonzero(sh[t0])
        win, ctrl = slice(t0, t0 + horizon), slice(t0 - control_from, t0 - control_to)
        rate = lambda s: float(sh[s].sum() / max(act[s].sum(), 1))                   # noqa: E731
        hit = sh[win].any(0)
        later = sh[t0 + 1:t0 + horizon].sum(1)
        mean_br = float(br[t0, first].mean()) if len(first) else 0.0
        rows.append({"event": e.name, "date": str(e.date), "anchor_hour": str(times[t0]),
                     "coins_shocking_at_anchor": int(len(first)),
                     "type": str(shock_type(mean_br)) if len(first) else "none",
                     "coins_hit_72h": int(hit.sum()), "share_hit_72h": float(hit.sum() / max(act[t0].sum(), 1)),
                     "shock_rate_72h": rate(win), "control_shock_rate": rate(ctrl),
                     "abnormal_shock_rate": rate(win) - rate(ctrl),
                     "peak_spillover_hour": int(np.argmax(later) + 1) if later.max() > 0 else None})
    return pd.DataFrame(rows)


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
