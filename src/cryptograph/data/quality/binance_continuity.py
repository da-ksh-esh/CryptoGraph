"""Stitch a coin's Binance symbol segments into one continuous hourly history.

Rules (oldest segment first):
  * a segment with `until` is cut after that UTC date (inclusive): rows after it
    belong to a different asset when a symbol is reused (LUNAUSDT after the Terra
    collapse became the new LUNA); they are dropped and counted;
  * if the next segment starts before the previous one ends (both symbols traded
    during a migration), the previous segment is cut at the next one's first hour;
  * each boundary is reported: last hour of the old symbol, first hour of the new
    one, the gap in hours and the log price change across it. For a 1:1 ticker
    migration the price change should be small; a large change within a week is
    flagged.
Pure functions: inputs are per-symbol kline frames indexed by UTC hour.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

HOUR = pd.Timedelta(hours=1)


@dataclass
class SegmentData:
    symbol: str
    frame: pd.DataFrame          # kline frame indexed by UTC hour (open time)
    until: pd.Timestamp | None   # last valid date (inclusive) or None


def stitch_segments(segments: list[SegmentData], max_price_jump: float = 0.05
                    ) -> tuple[pd.DataFrame, list[dict]]:
    """Return (stitched frame with a `symbol` column, report rows)."""
    report: list[dict] = []
    clipped: list[tuple[str, pd.DataFrame]] = []
    for seg in segments:
        f = seg.frame[~seg.frame.index.duplicated(keep="last")].sort_index()
        if seg.until is not None:
            cut = pd.Timestamp(seg.until) + pd.Timedelta(days=1)
            after = f.loc[cut:]
            if len(after):
                report.append({"type": "reused_symbol_rows_dropped", "symbol": seg.symbol, "rows_dropped": len(after),
                               "first_dropped_hour": after.index.min(),
                               "median_close_of_dropped_rows": float(after["close"].median()), "flag": False})
            f = f.loc[: cut - HOUR]
        clipped.append((seg.symbol, f))

    # hand over to the newer symbol where both traded at the same time
    for i in range(len(clipped) - 1):
        sym, f = clipped[i]
        nxt = clipped[i + 1][1]
        if len(f) and len(nxt):
            clipped[i] = (sym, f.loc[: nxt.index.min() - HOUR])

    kept = [(s, f) for s, f in clipped if len(f)]
    for (s0, f0), (s1, f1) in zip(kept, kept[1:]):
        gap_h = (f1.index.min() - f0.index.max()) / HOUR - 1
        jump = float(np.log(f1["open"].iloc[0] / f0["close"].iloc[-1]))
        report.append({"type": "symbol_boundary", "from_symbol": s0, "to_symbol": s1,
                       "old_last_hour": f0.index.max(), "new_first_hour": f1.index.min(), "gap_hours": gap_h,
                       "log_price_change": jump, "flag": bool(abs(jump) > max_price_jump and gap_h <= 24 * 7)})
    if not kept:
        return pd.DataFrame(), report
    out = pd.concat([f.assign(symbol=s) for s, f in kept]).sort_index()
    dup = out.index.duplicated(keep="last")
    if dup.any():
        report.append({"type": "overlap_rows_dropped", "rows_dropped": int(dup.sum()), "flag": False})
        out = out[~dup]
    return out, report
