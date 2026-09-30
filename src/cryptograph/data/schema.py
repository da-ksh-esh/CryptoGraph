"""Canonical schema of the tidy master table and the QC flag bitmask."""

from __future__ import annotations

from enum import IntFlag

import pandas as pd


class QC(IntFlag):
    """Per coin-day quality flags (stored as an int64 bitmask in `qc_flags`)."""

    NONE = 0
    DUPLICATE_DATE = 1 << 0          # raw source had >1 row for the date (last kept)
    MISSING_DAY = 1 << 1             # no close from any source inside the coin's active span
    NONPOSITIVE_PRICE = 1 << 2       # a price <= 0 (set missing)
    HIGH_LT_LOW = 1 << 3             # high < low (O/H/L set missing)
    OC_OUTSIDE_HL = 1 << 4           # open or close outside [low, high]
    ZERO_VOLUME = 1 << 5
    MISSING_VOLUME = 1 << 6
    VOLUME_SPIKE = 1 << 7            # volume > k x trailing median
    STALE_CLOSE = 1 << 8             # close unchanged for >= stale_days consecutive days
    LARGE_RETURN = 1 << 9            # |log return| > abs_logret_flag
    LARGE_RETURN_CONFIRMED = 1 << 10 # ... confirmed by second source or a catalogued event
    LARGE_RETURN_UNVERIFIED = 1 << 11  # ... no second source / event available; kept
    BAD_TICK_CORRECTED = 1 << 12     # close replaced by Coin Metrics (R2 or large-return check)
    XSRC_DISAGREE = 1 << 13          # Yahoo vs Coin Metrics close differ > tol, Yahoo kept (R3)
    CM_BACKFILL = 1 << 14            # close backfilled from Coin Metrics (O/H/L/volume missing)
    FILLED = 1 << 15                 # forward-filled gap (<= max_ffill_gap days)
    LONG_GAP = 1 << 16               # inside a gap longer than max_ffill_gap (left missing, inactive)
    REDENOMINATION = 1 << 17         # documented ticker migration / redenomination date
    USDT_DEPEG = 1 << 18             # hourly (USDT-quoted) data: USDT off its USD peg by more than the tolerance that day


def decode(mask: int) -> list[str]:
    return [f.name for f in QC if f and (int(mask) & int(f))]


MASTER_COLUMNS = {
    "date": "datetime64[ns]",
    "coin": "string",
    "open": "float64",
    "high": "float64",
    "low": "float64",
    "close": "float64",
    "volume": "float64",
    "market_cap": "float64",
    "close_source": "string",
    "is_active": "bool",
    "is_filled": "bool",
    "panel_A": "bool",
    "panel_B": "bool",
    "is_event_coin": "bool",
    "qc_flags": "int64",
}


def validate_master(df: pd.DataFrame) -> None:
    """Raise if the master table violates the schema or core invariants."""
    missing = set(MASTER_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"master table missing columns: {sorted(missing)}")
    if df.duplicated(["date", "coin"]).any():
        raise ValueError("duplicate (date, coin) rows")
    if (df["close"] <= 0).any():
        raise ValueError("non-positive close in master")
    if (df["is_active"] != df["close"].notna()).any():
        raise ValueError("is_active must equal close.notna()")
    for coin, g in df.groupby("coin"):
        d = g["date"].sort_values()
        if not (d.diff().dropna() == pd.Timedelta(days=1)).all():
            raise ValueError(f"{coin}: calendar not continuous")
        first = g.loc[g["close"].notna(), "date"].min()
        if pd.notna(first) and g.loc[g["date"] < first, ["open", "high", "low", "close", "volume"]].notna().any().any():
            raise ValueError(f"{coin}: values before first valid date")
