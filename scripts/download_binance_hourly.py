"""Execution script for Binance hourly data ingestion and data quality auditing."""

import argparse
from pathlib import Path
import pandas as pd

from cryptograph.data.universe import TARGET_UNIVERSE_30, EVENT_COINS
from cryptograph.data.sources.binance import BinanceArchiveDownloader
from cryptograph.common.logger import get_logger

logger = get_logger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Ingest Binance hourly archives and audit quality.")
    parser.add_argument("--coins", nargs="+", default=TARGET_UNIVERSE_30, help="List of coins to ingest.")
    parser.add_argument("--start-date", type=str, default="2024-01-01")
    parser.add_argument("--end-date", type=str, default="2026-09-27")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--daily-ref", type=str, default="data/raw/crypto_top20_10y_1d.parquet")
    parser.add_argument("--reports-dir", type=str, default="reports")
    args = parser.parse_args()

    downloader = BinanceArchiveDownloader()
    logger.info(f"Starting hourly Binance ingestion for {len(args.coins)} assets from {args.start_date} to {args.end_date}...")

    hourly_dict = downloader.ingest_coin_universe(
        coins=args.coins,
        start_date=args.start_date,
        end_date=args.end_date,
        max_workers=args.workers
    )

    report_path = downloader.generate_data_quality_report(
        hourly_dict=hourly_dict,
        daily_ref_path=args.daily_ref,
        reports_dir=args.reports_dir
    )

    logger.info(f"Ingestion and quality auditing complete! Report written to {report_path}")

if __name__ == "__main__":
    main()
