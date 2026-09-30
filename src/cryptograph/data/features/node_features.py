"""Per-coin, per-day node features — every value at day t uses data up to t only.

Inputs are wide frames (date x coin) built from the master table. Missing
stays missing (NaN); the availability mask is `np.isfinite(feature)`.
No filling across launch dates, no clipping of returns.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

LN2 = np.log(2.0)


@dataclass(frozen=True)
class FeatureSpec:
    vol_windows: tuple[int, ...] = (7, 14, 30)
    range_window: int = 7
    volume_median_window: int = 30
    rsi_window: int = 14
    days_listed_cap: int | None = 730   # "new coin" signal without encoding calendar era

    def names(self) -> list[str]:
        return (["log_return"] + [f"rv_{w}" for w in self.vol_windows] +
                [f"parkinson_{self.range_window}", f"garman_klass_{self.range_window}",
                 "log_volume", f"volume_rel_{self.volume_median_window}", f"rsi_{self.rsi_window}",
                 "macd_norm", "mcap_share", "log_days_listed"])


def wide(master: pd.DataFrame, col: str, coins: list[str]) -> pd.DataFrame:
    """date x coin frame for one column of the (date, coin)-indexed master table."""
    return master[col].unstack("coin").reindex(columns=coins).astype(float)


def log_returns(close: pd.DataFrame) -> pd.DataFrame:
    return np.log(close).diff()


def realized_vol(r: pd.DataFrame, w: int) -> pd.DataFrame:
    """Trailing std of daily log returns (needs >= 80% of the window observed)."""
    return r.rolling(w, min_periods=max(2, int(0.8 * w))).std()


def parkinson(high: pd.DataFrame, low: pd.DataFrame, w: int) -> pd.DataFrame:
    var = np.log(high / low) ** 2 / (4 * LN2)
    return np.sqrt(var.rolling(w, min_periods=max(2, int(0.7 * w))).mean())


def garman_klass(o: pd.DataFrame, h: pd.DataFrame, l: pd.DataFrame, c: pd.DataFrame, w: int) -> pd.DataFrame:
    var = 0.5 * np.log(h / l) ** 2 - (2 * LN2 - 1) * np.log(c / o) ** 2
    return np.sqrt(var.clip(lower=0).rolling(w, min_periods=max(2, int(0.7 * w))).mean())


def rsi(close: pd.DataFrame, w: int) -> pd.DataFrame:
    """Wilder RSI scaled to [0, 1]; NaN until w observations exist."""
    d = close.diff()
    gain = d.clip(lower=0).ewm(alpha=1 / w, min_periods=w, adjust=False, ignore_na=True).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1 / w, min_periods=w, adjust=False, ignore_na=True).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 1 - 1 / (1 + rs)
    return out.where(~(loss.eq(0) & gain.gt(0)), 1.0).where(close.notna())


def macd_norm(close: pd.DataFrame) -> pd.DataFrame:
    ema12 = close.ewm(span=12, min_periods=26, adjust=False, ignore_na=True).mean()
    ema26 = close.ewm(span=26, min_periods=26, adjust=False, ignore_na=True).mean()
    return ((ema12 - ema26) / close).where(close.notna())


def build_node_features(master: pd.DataFrame, coins: list[str], spec: FeatureSpec = FeatureSpec()
                        ) -> tuple[np.ndarray, pd.DatetimeIndex, list[str]]:
    """Return X (T, N, F) float32 with NaN where unavailable, the dates, and feature names."""
    o, h, l, c = (wide(master, k, coins) for k in ("open", "high", "low", "close"))
    vol, mcap = wide(master, "volume", coins), wide(master, "market_cap", coins)
    dates = c.index
    r = log_returns(c)

    feats: dict[str, pd.DataFrame] = {"log_return": r}
    for w in spec.vol_windows:
        feats[f"rv_{w}"] = realized_vol(r, w)
    feats[f"parkinson_{spec.range_window}"] = parkinson(h, l, spec.range_window)
    feats[f"garman_klass_{spec.range_window}"] = garman_klass(o, h, l, c, spec.range_window)
    v = vol.where(vol > 0)
    feats["log_volume"] = np.log(v)
    med = v.shift(1).rolling(spec.volume_median_window, min_periods=spec.volume_median_window // 2).median()
    feats[f"volume_rel_{spec.volume_median_window}"] = np.log(v / med)
    feats[f"rsi_{spec.rsi_window}"] = rsi(c, spec.rsi_window)
    feats["macd_norm"] = macd_norm(c)
    m = mcap.where((mcap > 0) & c.notna())
    feats["mcap_share"] = m.div(m.sum(axis=1, min_count=1), axis=0)
    first = c.apply(lambda s: s.first_valid_index())
    days = pd.DataFrame({k: (dates - first[k]).days if first[k] is not None else np.nan for k in coins}, index=dates)
    if spec.days_listed_cap is not None:
        days = days.clip(upper=spec.days_listed_cap)
    feats["log_days_listed"] = np.log1p(days.where(c.notna()))

    names = spec.names()
    X = np.stack([feats[n].reindex(columns=coins).to_numpy(dtype=np.float32) for n in names], axis=-1)
    X[~np.isfinite(X)] = np.nan
    return X, dates, names
