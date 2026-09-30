"""Hourly Stage 1 — 30-coin universe audit (checkpoint).

Writes reports/universe_audit_v2.csv, reports/sector_evidence.csv and
data/interim/hourly/universe_v2.json, then prints a summary.
"""

from __future__ import annotations

import pandas as pd

from ...data.config import HourlyConfig
from ...data.universe_hourly import HourlyUniverseAuditor


def run(cfg: HourlyConfig, refresh: bool = False) -> None:
    df, summary = HourlyUniverseAuditor(cfg, refresh=refresh).run()
    u = cfg.universe
    pd.set_option("display.width", 250, "display.max_rows", 200, "display.max_colwidth", 70)
    show = df.copy()
    show["mcap_$B"] = (show["market_cap_usd"] / 1e9).round(2)
    for c in ("window_coverage", "test_coverage", "daily_yahoo_coverage"):
        show[c] = (show[c] * 100).round(1)
    show["history_days"] = show["history_days"].round(0)
    for c in ("hourly_first", "hourly_last"):
        show[c] = pd.to_datetime(show[c]).dt.strftime("%Y-%m-%d")
    cols = ["rank_all", "coin", "mcap_$B", "sector", "binance_symbol", "hourly_first", "hourly_last",
            "history_days", "test_coverage", "daily_yahoo_coverage", "status", "reason"]
    print(show[cols].to_string(index=False))
    print("\nIdentity checks (included coins):")
    inc = df[df["status"].eq("included")]
    print(inc[["coin", "binance_symbol", "reference_price", "binance_ref_close", "identity_note", "yahoo_ticker"]]
          .to_string(index=False))
    print(f"\nSector counts: {summary['sector_counts']}; missing sector: {summary['missing_sector']}")
    print(f"\nIncluded ({summary['n_included']}/{u.target_size}), deepest non-stable rank used: "
          f"{summary['deepest_nonstable_rank_used']}: {summary['included']}")
    print(f"Passing but not needed: {summary['spare_passing']}")
    print(f"Event coins: {summary['event_coins']}   ranking snapshot: {summary['ranking_snapshot']}")
    if summary["n_included"] < u.target_size:
        print(f"\nCHECKPOINT: only {summary['n_included']} coins pass within the top {u.candidate_pool} non-stable coins.")
