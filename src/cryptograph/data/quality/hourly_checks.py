"""Quality checks for one coin's hourly Binance series (continuous hourly calendar).

Same bitmask as the daily data (`cryptograph.data.schema.QC`); in the hourly table
MISSING_DAY means "missing hour". Large hourly moves are checked against the daily
second source (Yahoo / Coin Metrics) and the event catalogue:
  * confirmed  — a same-day daily move of at least `daily_confirm_logret`, or a
                 catalogued event within `event_window_days`;
  * bad tick   — an isolated spike that reverses in the next hour and is not
                 confirmed: the hour's prices are set missing (then handled by the
                 gap rule);
  * unverified — otherwise; kept and flagged.
Nothing is clipped; returns are computed later from the cleaned closes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..schema import QC
from .checks import stale_mask


@dataclass(frozen=True)
class HourlyCheckParams:
    stale_hours: int = 6
    volume_spike_mult: float = 20.0
    volume_spike_window_hours: int = 168
    abs_logret_flag: float = 0.25
    daily_confirm_logret: float = 0.10
    event_window_days: int = 3


def _set(flags: np.ndarray, where, flag: QC) -> None:
    flags[np.asarray(pd.Series(where).fillna(False), dtype=bool)] |= int(flag)


def run_hourly_checks(df: pd.DataFrame, daily_ref_logret: pd.Series | None, event_dates: list[pd.Timestamp],
                      params: HourlyCheckParams = HourlyCheckParams()) -> pd.DataFrame:
    """df: continuous hourly index, columns open/high/low/close/volume. Returns a copy with `qc_flags`."""
    out = df.copy()
    flags = np.zeros(len(out), dtype=np.int64)

    for col in ["open", "high", "low", "close"]:
        bad = out[col] <= 0
        _set(flags, bad, QC.NONPOSITIVE_PRICE)
        out.loc[bad, col] = np.nan
    hl = out["high"] < out["low"]
    _set(flags, hl, QC.HIGH_LT_LOW)
    out.loc[hl, ["open", "high", "low"]] = np.nan
    eps = 1e-9
    outside = ((out["open"] > out["high"] * (1 + eps)) | (out["open"] < out["low"] * (1 - eps)) |
               (out["close"] > out["high"] * (1 + eps)) | (out["close"] < out["low"] * (1 - eps)))
    _set(flags, outside, QC.OC_OUTSIDE_HL)

    has = out["close"].notna()
    _set(flags, has & out["volume"].eq(0), QC.ZERO_VOLUME)
    _set(flags, has & out["volume"].isna(), QC.MISSING_VOLUME)
    vol = out["volume"].where(out["volume"] > 0)
    med = vol.shift(1).rolling(params.volume_spike_window_hours, min_periods=24).median()
    _set(flags, vol > params.volume_spike_mult * med, QC.VOLUME_SPIKE)
    _set(flags, stale_mask(out["close"], params.stale_hours), QC.STALE_CLOSE)

    r = np.log(out["close"]).diff()
    large = (r.abs() > params.abs_logret_flag).fillna(False)
    _set(flags, large, QC.LARGE_RETURN)
    ev = pd.DatetimeIndex(event_dates).normalize()
    idx = {t: i for i, t in enumerate(out.index)}
    for t in out.index[large]:
        day = t.normalize()
        near_event = len(ev) and (np.abs((ev - day).days) <= params.event_window_days).any()
        daily_move = daily_ref_logret.get(day, np.nan) if daily_ref_logret is not None else np.nan
        if near_event or (np.isfinite(daily_move) and abs(daily_move) >= params.daily_confirm_logret):
            flags[idx[t]] |= int(QC.LARGE_RETURN_CONFIRMED)
            continue
        i = idx[t]
        nxt = r.iloc[i + 1] if i + 1 < len(r) else np.nan
        if np.isfinite(nxt) and abs(nxt) > params.abs_logret_flag and np.sign(nxt) != np.sign(r.iloc[i]):
            out.iloc[i, [out.columns.get_loc(c) for c in ["open", "high", "low", "close"]]] = np.nan
            flags[i] |= int(QC.BAD_TICK_CORRECTED)
        else:
            flags[i] |= int(QC.LARGE_RETURN_UNVERIFIED)

    out["qc_flags"] = flags
    return out


def resample_2h(h: pd.DataFrame) -> pd.DataFrame:
    """1h -> 2h bars (left-labelled). A 2h bar is valid only if both hours have a close;
    open=first, high=max, low=min, close=last, volume/quote_volume/trades=sum; flags OR-ed."""
    g = h.resample("2h", label="left", closed="left")
    n_valid = g["close"].count()
    out = pd.DataFrame({
        "open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(), "close": g["close"].last(),
        "volume": g["volume"].sum(min_count=1), "quote_volume": g["quote_volume"].sum(min_count=1),
        "trades": g["trades"].sum(min_count=1),
        "is_filled": g["is_filled"].max().fillna(False).astype(bool),
        "qc_flags": g["qc_flags"].agg(lambda s: int(np.bitwise_or.reduce(s.to_numpy(dtype=np.int64))) if len(s) else 0),
    })
    incomplete = n_valid < 2
    out.loc[incomplete, ["open", "high", "low", "close", "volume", "quote_volume", "trades"]] = np.nan
    out["is_active"] = out["close"].notna()
    return out
