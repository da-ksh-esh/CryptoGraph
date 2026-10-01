"""Intraday (hourly / 2-hourly) features with seasonality removed.

Crypto volume and volatility follow strong hour-of-day / day-of-week patterns, so a
normal 14:00 UTC volume spike must not look like a shock. A profile per
(day-of-week, hour-of-day) bucket is estimated on TRAIN rows only, pooled across
coins after normalising each coin by its own train level, and then used to
standardise every row:
    r_ds      = r / sqrt(f_vol[bucket])            (deseasonalised return)
    logvol_ds = log(volume) - f_logvol[bucket]
All features at time t use data up to t only. Windows are given in hours and
converted to bars (1h or 2h).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


def bars(hours: int, bar_hours: int) -> int:
    return max(1, int(round(hours / bar_hours)))


def bucket_index(times: pd.DatetimeIndex, bar_hours: int) -> np.ndarray:
    """Seasonal bucket of each bar: day_of_week * (24 / bar_hours) + hour // bar_hours."""
    per_day = 24 // bar_hours
    return (times.dayofweek * per_day + times.hour // bar_hours).to_numpy()


@dataclass
class SeasonalityProfile:
    bar_hours: int
    vol: np.ndarray            # (n_buckets,) mean normalised squared return, mean 1
    logvol: np.ndarray         # (n_buckets,) mean de-meaned log volume, mean 0
    fit_start: str = ""
    fit_end: str = ""
    n_coins: int = 0
    meta: dict = field(default_factory=dict)

    @classmethod
    def fit(cls, r: pd.DataFrame, volume: pd.DataFrame, train_mask: np.ndarray, bar_hours: int) -> "SeasonalityProfile":
        """r, volume: (time x coin) frames; train_mask: bool per row (train dates only)."""
        b = bucket_index(r.index, bar_hours)
        n_b = 7 * (24 // bar_hours)
        rt, vt, bt = r[train_mask], volume.where(volume > 0)[train_mask], b[train_mask]
        sq = rt ** 2
        sq_norm = sq / sq.mean()                                  # each coin relative to its own train level
        lv = np.log(vt)
        lv_dm = lv - lv.mean()
        vol = np.array([np.nanmean(sq_norm.to_numpy()[bt == k]) for k in range(n_b)])
        logvol = np.array([np.nanmean(lv_dm.to_numpy()[bt == k]) for k in range(n_b)])
        vol = vol / np.nanmean(vol)
        logvol = logvol - np.nanmean(logvol)
        idx = r.index[train_mask]
        return cls(bar_hours, vol, logvol, str(idx.min()), str(idx.max()), r.shape[1])

    def deseason_returns(self, r: pd.DataFrame) -> pd.DataFrame:
        f = self.vol[bucket_index(r.index, self.bar_hours)]
        return r.div(np.sqrt(f), axis=0)

    def deseason_logvol(self, logvol: pd.DataFrame) -> pd.DataFrame:
        g = self.logvol[bucket_index(logvol.index, self.bar_hours)]
        return logvol.sub(g, axis=0)

    def to_json(self) -> dict:
        return {"bar_hours": self.bar_hours, "vol": self.vol.round(6).tolist(), "logvol": self.logvol.round(6).tolist(),
                "fit_start": self.fit_start, "fit_end": self.fit_end, "n_coins": self.n_coins,
                "bucket": "day_of_week * (24 / bar_hours) + hour // bar_hours (UTC; Monday = 0)"}


def wide_frame(master: pd.DataFrame, col: str, coins: list[str]) -> pd.DataFrame:
    return master[col].unstack("coin").reindex(columns=coins).astype(float)


def build_intraday_features(master: pd.DataFrame, coins: list[str], profile: SeasonalityProfile,
                            rv_windows_hours: list[int], range_window_hours: int, volume_window_hours: int,
                            listing_cap_hours: int) -> tuple[np.ndarray, pd.DatetimeIndex, list[str], pd.DataFrame]:
    """X (T, N, F) float32 with NaN where unavailable, times, names, and the deseasonalised returns."""
    bh = profile.bar_hours
    o, h, l, c = (wide_frame(master, k, coins) for k in ("open", "high", "low", "close"))
    vol = wide_frame(master, "volume", coins)
    times = c.index
    r = np.log(c).diff()
    r_ds = profile.deseason_returns(r)

    feats: dict[str, pd.DataFrame] = {"log_return": r}
    for w in rv_windows_hours:
        n = bars(w, bh)
        feats[f"rv_{w}h"] = r_ds.rolling(n, min_periods=max(2, int(0.8 * n))).std()
    n = bars(range_window_hours, bh)
    pk_var = np.log(h / l) ** 2 / (4 * np.log(2.0))
    f_bucket = profile.vol[bucket_index(times, bh)]
    feats[f"parkinson_{range_window_hours}h"] = np.sqrt(pk_var.div(f_bucket, axis=0)
                                                         .rolling(n, min_periods=max(2, int(0.7 * n))).mean())
    lv = np.log(vol.where(vol > 0))
    lv_ds = profile.deseason_logvol(lv)
    feats["log_volume_ds"] = lv_ds
    n = bars(volume_window_hours, bh)
    feats[f"volume_rel_{volume_window_hours}h"] = lv_ds - lv_ds.shift(1).rolling(n, min_periods=n // 2).median()
    first = c.apply(lambda s: s.first_valid_index())
    hours = pd.DataFrame({k: (times - first[k]) / pd.Timedelta(hours=1) if first[k] is not None else np.nan
                          for k in coins}, index=times)
    feats["log_hours_listed"] = np.log1p(hours.clip(upper=listing_cap_hours).where(c.notna()))

    names = list(feats)
    X = np.stack([feats[n_].reindex(columns=coins).to_numpy(dtype=np.float32) for n_ in names], axis=-1)
    X[~np.isfinite(X)] = np.nan
    return X, times, names, r_ds


def prune_redundant(X: np.ndarray, names: list[str], train_rows: np.ndarray, threshold: float,
                    protected: tuple[str, ...] = ("log_return",)) -> tuple[list[str], list[tuple[str, str, float]]]:
    """Greedy pruning on train data: walk the features in order and drop any feature whose |Spearman rho| with an
    already-kept feature is >= threshold. Returns (kept names, dropped pairs)."""
    flat = X[train_rows].reshape(-1, X.shape[-1])
    ok = np.isfinite(flat).all(axis=1)
    sample = flat[ok]
    if len(sample) > 400_000:
        sample = sample[np.random.default_rng(0).choice(len(sample), 400_000, replace=False)]
    rho = pd.DataFrame(sample, columns=names).rank().corr().to_numpy()
    kept: list[int] = []
    dropped: list[tuple[str, str, float]] = []
    for i, n in enumerate(names):
        clash = [(k, abs(rho[i, k])) for k in kept if abs(rho[i, k]) >= threshold]
        if clash and n not in protected:
            k, v = max(clash, key=lambda x: x[1])
            dropped.append((n, names[k], round(float(v), 3)))
        else:
            kept.append(i)
    return [names[i] for i in kept], dropped
