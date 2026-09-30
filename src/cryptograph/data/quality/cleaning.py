"""Gap handling: never invent data across launch dates or long gaps.

Rules
  * Before a coin's first valid close and after its last one: missing, inactive.
  * Inside the active span, runs of missing closes of length <= max_gap are
    forward-filled (close only; O/H/L/volume stay missing) and flagged FILLED.
  * Longer runs stay missing, inactive, flagged LONG_GAP.
  * Returns are never clipped.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..schema import QC


def gap_runs(missing: pd.Series) -> pd.Series:
    """Length of the missing-run each day belongs to (0 for non-missing days)."""
    run_id = (~missing).cumsum()
    lengths = missing.groupby(run_id).transform("sum")
    return lengths.where(missing, 0).astype(int)


def clean_coin(df: pd.DataFrame, max_gap: int) -> pd.DataFrame:
    out = df.copy()
    close = out["close"]
    valid = close.notna()
    out["is_filled"] = False
    if not valid.any():
        out["is_active"] = False
        return out
    first, last = close.first_valid_index(), close.last_valid_index()
    span = (out.index >= first) & (out.index <= last)

    # anything outside the active span is not data: blank it (defensive; sources may carry stray values)
    value_cols = [c for c in ["open", "high", "low", "close", "volume", "market_cap"] if c in out]
    out.loc[~span, value_cols] = np.nan
    out.loc[~span, "close_source"] = None

    missing = close.isna() & span
    runs = gap_runs(pd.Series(missing, index=out.index))
    flags = out["qc_flags"].to_numpy(dtype=np.int64).copy()
    flags[missing.to_numpy()] |= int(QC.MISSING_DAY)

    short = missing & (runs <= max_gap)
    long_ = missing & (runs > max_gap)
    out.loc[short, "close"] = close.ffill()[short]
    out.loc[short, "close_source"] = "ffill"
    out.loc[short, "is_filled"] = True
    flags[short.to_numpy()] |= int(QC.FILLED)
    flags[long_.to_numpy()] |= int(QC.LONG_GAP)
    out["qc_flags"] = flags
    out["is_active"] = out["close"].notna()
    return out
