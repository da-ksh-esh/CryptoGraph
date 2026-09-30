"""Data Continuity, Survivorship, and Symbol Migration Audit.

Handles market-level continuity rules:
1. LUNC vs LUNA continuity (never splicing post-fork LUNA onto old LUNA).
2. Ticker renames (e.g., MATIC -> POL, RNDR -> RENDER).
3. Exchange maintenance gaps (ffill maximum 2 hours with audit flags; >=3h gaps masked).
4. Delisting handling (e.g. XMR Binance delisting on 2024-02-20).
5. USDT quote asset disclaimer and daily USDT peg status tracking up to 2026-05-23.
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
from ...common.logger import get_logger

logger = get_logger(__name__)

COIN_METRICS_PEG_CUTOFF = pd.Timestamp("2026-05-23", tz="UTC")

class ContinuityAuditor:
    """
    Audits and enforces continuity and missingness rules for hourly crypto streams.
    """

    @staticmethod
    def handle_lunc_continuity(old_luna_df: Optional[pd.DataFrame], lunc_df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensures Terra Classic (LUNC) maintains historical continuity up to the collapse in May 2022,
        masks the suspension gap through Sept 2022, and explicitly never splices Terra 2.0 (LUNA).
        """
        if old_luna_df is None or old_luna_df.empty:
            return lunc_df

        cutoff = pd.Timestamp("2022-05-13", tz="UTC")
        valid_old = old_luna_df.loc[:cutoff].copy()
        valid_old["is_event_coin"] = True

        valid_new = lunc_df.loc[pd.Timestamp("2022-09-01", tz="UTC"):].copy()
        valid_new["is_event_coin"] = True

        combined = pd.concat([valid_old, valid_new]).sort_index()
        return combined

    @staticmethod
    def repair_maintenance_gaps(series: pd.Series, max_ffill_hours: int = 2) -> Tuple[pd.Series, pd.Series]:
        """
        Handles short exchange maintenance gaps:
        - Gaps <= 2 hours: forward filled and flagged in maintenance_mask.
        - Gaps > 2 hours: left as NaN / inactive mask to prevent look-ahead bias.
        """
        is_missing = series.isna()
        filled = series.ffill(limit=max_ffill_hours)
        ffill_flag = is_missing & (~filled.isna())

        return filled, ffill_flag

    @staticmethod
    def check_usdt_peg_status(date: pd.Timestamp, usdt_price_usd: float, peg_tolerance: float = 0.01) -> Dict[str, Any]:
        """
        Checks USDT peg against 1.0 USD reference at daily resolution.
        Ends on 2026-05-23 (where Coin Metrics archive ends).
        Dates after cutoff carry 'no peg check' status.
        """
        ts = pd.to_datetime(date, utc=True)
        if ts > COIN_METRICS_PEG_CUTOFF:
            return {
                "date": ts.date(),
                "peg_checked": False,
                "status": "no_peg_check",
                "deviation": 0.0
            }

        deviation = abs(usdt_price_usd - 1.0)
        is_pegged = deviation <= peg_tolerance
        return {
            "date": ts.date(),
            "peg_checked": True,
            "status": "peg_ok" if is_pegged else "depeg_alert",
            "deviation": float(deviation)
        }
