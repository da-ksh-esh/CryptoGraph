"""Data-quality checks on one coin's reconciled daily frame.

Each check sets bits in `qc_flags` and (where the value is provably wrong)
masks it. Nothing is interpolated here; gap filling lives in cleaning.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..schema import QC


@dataclass(frozen=True)
class CheckParams:
    volume_spike_mult: float = 20.0
    volume_spike_window: int = 30
    stale_days: int = 3
    abs_logret_flag: float = 0.40
    event_window_days: int = 3        # a large move within +-N days of a catalogued event is "confirmed"
    xsrc_confirm_ratio: float = 0.5   # second-source return must be >= this share of Yahoo's, same sign


def _set(flags: pd.Series, where: pd.Series, flag: QC) -> None:
    flags[where.fillna(False).to_numpy()] |= int(flag)


def stale_mask(close: pd.Series, n: int) -> pd.Series:
    """True on days belonging to a run of >= n identical consecutive closes."""
    same = close.diff().eq(0) & close.notna()
    run_id = (~same).cumsum()
    run_len = same.groupby(run_id).transform("sum") + 1  # +1: first day of the run
    in_run = same | same.shift(-1, fill_value=False)
    return in_run & (run_len >= n)


def run_checks(df: pd.DataFrame, cm_close: pd.Series | None, event_dates: list[pd.Timestamp],
               params: CheckParams = CheckParams(), n_duplicates: pd.Series | None = None) -> pd.DataFrame:
    """Return a copy of `df` (continuous calendar index) with `qc_flags` populated.

    Required columns: open, high, low, close, volume, close_source, xsrc_flag, xsrc_resolution.
    """
    out = df.copy()
    flags = pd.Series(np.zeros(len(out), dtype=np.int64), index=out.index)

    if n_duplicates is not None:
        _set(flags, n_duplicates.reindex(out.index).fillna(0) > 0, QC.DUPLICATE_DATE)

    # non-positive prices -> missing
    for col in ["open", "high", "low", "close"]:
        bad = out[col] <= 0
        _set(flags, bad, QC.NONPOSITIVE_PRICE)
        out.loc[bad, col] = np.nan

    # OHLC consistency
    hl = out["high"] < out["low"]
    _set(flags, hl, QC.HIGH_LT_LOW)
    out.loc[hl, ["open", "high", "low"]] = np.nan
    eps = 1e-9
    outside = ((out["open"] > out["high"] * (1 + eps)) | (out["open"] < out["low"] * (1 - eps)) |
               (out["close"] > out["high"] * (1 + eps)) | (out["close"] < out["low"] * (1 - eps)))
    outside &= out["close_source"].eq("yahoo")  # corrected/backfilled closes have no O/H/L
    _set(flags, outside, QC.OC_OUTSIDE_HL)

    # volume
    has_close = out["close"].notna()
    _set(flags, has_close & out["volume"].eq(0), QC.ZERO_VOLUME)
    _set(flags, has_close & out["volume"].isna(), QC.MISSING_VOLUME)
    vol = out["volume"].where(out["volume"] > 0)
    med = vol.shift(1).rolling(params.volume_spike_window, min_periods=10).median()
    _set(flags, vol > params.volume_spike_mult * med, QC.VOLUME_SPIKE)

    # stale closes
    _set(flags, stale_mask(out["close"], params.stale_days), QC.STALE_CLOSE)

    # cross-source outcomes from reconcile
    _set(flags, out["xsrc_resolution"].eq("R3_keep_yahoo_flagged"), QC.XSRC_DISAGREE)
    _set(flags, out["close_source"].eq("coinmetrics_corrected"), QC.BAD_TICK_CORRECTED)
    _set(flags, out["close_source"].eq("coinmetrics"), QC.CM_BACKFILL)

    # large returns: confirm by second source or event, otherwise correct (if contradicted) or keep
    r = np.log(out["close"]).diff()
    large = r.abs() > params.abs_logret_flag
    _set(flags, large, QC.LARGE_RETURN)
    ev = pd.DatetimeIndex(event_dates)
    rc = np.log(cm_close.reindex(out.index)).diff() if cm_close is not None else None
    for d in out.index[large.fillna(False)]:
        near_event = len(ev) and (np.abs((ev - d).days) <= params.event_window_days).any()
        if rc is not None and pd.notna(rc.at[d]):
            agrees = np.sign(rc.at[d]) == np.sign(r.at[d]) and abs(rc.at[d]) >= params.xsrc_confirm_ratio * abs(r.at[d])
            if agrees or near_event:
                flags.at[d] |= int(QC.LARGE_RETURN_CONFIRMED)
            else:
                # second source contradicts the move -> bad tick; use the second source's close
                out.at[d, "close"] = float(cm_close.at[d])
                out.loc[d, ["open", "high", "low"]] = np.nan
                out.at[d, "close_source"] = "coinmetrics_corrected"
                flags.at[d] |= int(QC.BAD_TICK_CORRECTED)
        elif near_event:
            flags.at[d] |= int(QC.LARGE_RETURN_CONFIRMED)
        else:
            flags.at[d] |= int(QC.LARGE_RETURN_UNVERIFIED)

    out["qc_flags"] = flags
    return out
