"""Binance Public Data Archive Ingestion & Checksum Verification.

Fetches monthly and daily OHLCV klines from data.binance.vision, validates
cryptographic SHA-256 checksums, and standardizes kline dataframes.
"""

import os
import io
import zipfile
import hashlib
import urllib.request
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from ...common.logger import get_logger

logger = get_logger(__name__)

BINANCE_ARCHIVE_BASE = "https://data.binance.vision/data/spot"

BINANCE_KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "count",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "ignore"
]

class BinanceArchiveDownloader:
    """
    Downloads historical spot klines from data.binance.vision public archives
    with automated SHA-256 checksum verification.
    """

    def __init__(self, raw_dir: str = "data/raw/binance/1h"):
        self.raw_dir = Path(raw_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def get_monthly_url(self, symbol: str, year: int, month: int, interval: str = "1h") -> Tuple[str, str]:
        """Constructs data and checksum URLs for monthly archive."""
        file_name = f"{symbol}-{interval}-{year}-{month:02d}.zip"
        data_url = f"{BINANCE_ARCHIVE_BASE}/monthly/klines/{symbol}/{interval}/{file_name}"
        checksum_url = f"{data_url}.CHECKSUM"
        return data_url, checksum_url

    def get_daily_url(self, symbol: str, year: int, month: int, day: int, interval: str = "1h") -> Tuple[str, str]:
        """Constructs data and checksum URLs for daily archive."""
        file_name = f"{symbol}-{interval}-{year}-{month:02d}-{day:02d}.zip"
        data_url = f"{BINANCE_ARCHIVE_BASE}/daily/klines/{symbol}/{interval}/{file_name}"
        checksum_url = f"{data_url}.CHECKSUM"
        return data_url, checksum_url

    @staticmethod
    def verify_sha256(content_bytes: bytes, checksum_text: str) -> bool:
        """
        Compares computed SHA-256 of content against .CHECKSUM file content.
        Binance checksum file format: '<sha256_hash>  <filename>'
        """
        computed_hash = hashlib.sha256(content_bytes).hexdigest().lower()
        expected_hash = checksum_text.strip().split()[0].lower()
        return computed_hash == expected_hash

    def fetch_and_verify(self, data_url: str, checksum_url: str) -> Optional[bytes]:
        """Downloads archive zip bytes and verifies against remote checksum."""
        try:
            req_data = urllib.request.Request(data_url, headers={"User-Agent": "CryptoGraph/1.0"})
            with urllib.request.urlopen(req_data, timeout=30) as resp:
                data_bytes = resp.read()

            req_chk = urllib.request.Request(checksum_url, headers={"User-Agent": "CryptoGraph/1.0"})
            with urllib.request.urlopen(req_chk, timeout=15) as resp:
                checksum_text = resp.read().decode("utf-8")

            if not self.verify_sha256(data_bytes, checksum_text):
                logger.error(f"Checksum mismatch for {data_url}")
                return None

            return data_bytes
        except Exception as e:
            logger.warning(f"Could not fetch {data_url}: {e}")
            return None

    @staticmethod
    def parse_kline_zip(zip_bytes: bytes) -> pd.DataFrame:
        """Extracts and parses CSV klines inside zip buffer into DataFrame."""
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
            if not csv_names:
                raise ValueError("No CSV file found in archive zip.")
            with zf.open(csv_names[0]) as csv_file:
                df = pd.read_csv(
                    csv_file,
                    header=None,
                    names=BINANCE_KLINE_COLUMNS
                )

        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
        numeric_cols = ["open", "high", "low", "close", "volume", "quote_volume", "count"]
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
        df.set_index("open_time", inplace=True)
        return df

    @staticmethod
    def resample_1h_to_2h(df_1h: pd.DataFrame) -> pd.DataFrame:
        """
        Resamples 1h OHLCV bars into 2h computational fallback bars:
        Open: first, High: max, Low: min, Close: last, Volume: sum.
        """
        agg_rules = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
            "quote_volume": "sum",
            "count": "sum"
        }
        valid_rules = {col: agg_rules[col] for col in agg_rules if col in df_1h.columns}
        df_2h = df_1h.resample("2h").agg(valid_rules).dropna(subset=["close"])
        return df_2h
