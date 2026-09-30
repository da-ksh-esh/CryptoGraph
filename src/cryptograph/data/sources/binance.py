"""Binance public data archive (https://data.binance.vision) — spot klines, no API key.

Files are monthly zips (`<SYM>-1h-YYYY-MM.zip`) plus daily zips for the current,
partial month, each with a `.CHECKSUM` (SHA-256). Every download is verified against
its checksum before it is kept. Prices are quoted in the quote asset (USDT), not USD.

Kline CSVs have no header. Columns: open_time, open, high, low, close, volume,
close_time, quote_volume, trades, taker_buy_base, taker_buy_quote, ignore.
Timestamps are in **milliseconds** in older files and **microseconds** from 2025 on;
`read_klines` normalises both to a UTC hourly index.
"""

from __future__ import annotations

import hashlib
import io
import re
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ...common.logger import get_logger
from ..storage import FetchLog, utc_now, write_json
from .base import with_retries

logger = get_logger(__name__)

KLINE_COLUMNS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "trades",
                 "taker_buy_base", "taker_buy_quote", "ignore"]
_S3_NS = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
_MONTH_RE = re.compile(r"-(\d{4}-\d{2})\.zip$")
_DAY_RE = re.compile(r"-(\d{4}-\d{2}-\d{2})\.zip$")


class ChecksumError(RuntimeError):
    pass


@dataclass(frozen=True)
class SymbolFiles:
    """Kline files available in the archive for one symbol."""
    symbol: str
    months: tuple[str, ...]     # "YYYY-MM" with a monthly zip
    days: tuple[str, ...]       # "YYYY-MM-DD" with a daily zip (only listed for requested months)

    @property
    def exists(self) -> bool:
        return bool(self.months or self.days)


def parse_checksum(text: str) -> str:
    """`<sha256>  <filename>` -> sha256 (lower-case)."""
    token = text.strip().split()[0] if text.strip() else ""
    if not re.fullmatch(r"[0-9a-fA-F]{64}", token):
        raise ChecksumError(f"malformed checksum file: {text[:80]!r}")
    return token.lower()


def verify_bytes(blob: bytes, expected_sha256: str) -> None:
    got = hashlib.sha256(blob).hexdigest()
    if got != expected_sha256.lower():
        raise ChecksumError(f"SHA-256 mismatch: expected {expected_sha256}, got {got}")


def read_klines(source: bytes | Path) -> pd.DataFrame:
    """Parse a kline zip (bytes or path) into a frame indexed by UTC open time (tz-naive)."""
    blob = source.read_bytes() if isinstance(source, Path) else source
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        raw = zf.read(zf.namelist()[0])
    df = pd.read_csv(io.BytesIO(raw), header=None, names=KLINE_COLUMNS)
    if len(df) and not str(df.iloc[0, 0]).strip().lstrip("-").isdigit():   # some files carry a header row
        df = df.iloc[1:].reset_index(drop=True)
    ot = pd.to_numeric(df["open_time"]).astype("int64").to_numpy()
    # microseconds (>= 1e15 for any date after 2001) vs milliseconds -> nanoseconds
    ns = np.where(np.abs(ot) >= 10**15, ot * 1_000, ot * 1_000_000)
    out = df[["open", "high", "low", "close", "volume", "quote_volume", "trades"]].apply(pd.to_numeric, errors="coerce")
    out.index = pd.DatetimeIndex(pd.to_datetime(ns, unit="ns"), name="time")
    return out.sort_index()


class BinanceArchive:
    name = "binance"

    def __init__(self, archive_url: str, list_url: str, market: str = "spot", interval: str = "1h",
                 retries: int = 5, backoff: float = 2.0):
        self.archive_url = archive_url.rstrip("/")
        self.list_url = list_url.rstrip("/")
        self.market, self.interval = market, interval
        self.retries, self.backoff = retries, backoff

    # ---- listing -------------------------------------------------------------------
    def _get(self, url: str, timeout: int = 60) -> bytes:
        req = urllib.request.Request(url, headers={"User-Agent": "cryptograph"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()

    def list_keys(self, prefix: str) -> list[str]:
        """All object keys under `prefix` (S3 ListObjects v1 with marker pagination)."""
        keys: list[str] = []
        marker = ""
        while True:
            url = f"{self.list_url}?delimiter=/&prefix={prefix}" + (f"&marker={marker}" if marker else "")
            root = ET.fromstring(with_retries(lambda: self._get(url), retries=self.retries, backoff=self.backoff,
                                              what=f"binance list {prefix}"))
            page = [c.find("s:Key", _S3_NS).text for c in root.findall("s:Contents", _S3_NS)]
            keys += page
            truncated = (root.find("s:IsTruncated", _S3_NS).text or "").lower() == "true"
            if not truncated or not page:
                return keys
            marker = page[-1]

    def _prefix(self, period: str, symbol: str) -> str:
        return f"data/{self.market}/{period}/klines/{symbol}/{self.interval}/"

    def symbol_files(self, symbol: str, daily_months: tuple[str, ...] = ()) -> SymbolFiles:
        """Monthly files for `symbol`, plus daily files inside `daily_months` (e.g. the current partial month)."""
        months = sorted({m.group(1) for k in self.list_keys(self._prefix("monthly", symbol)) if (m := _MONTH_RE.search(k))})
        days: set[str] = set()
        for ym in daily_months:
            keys = self.list_keys(self._prefix("daily", symbol) + f"{symbol}-{self.interval}-{ym}")
            days |= {m.group(1) for k in keys if (m := _DAY_RE.search(k))}
        return SymbolFiles(symbol, tuple(months), tuple(sorted(days)))

    # ---- download --------------------------------------------------------------------
    def file_key(self, symbol: str, period: str, stamp: str) -> str:
        return self._prefix(period, symbol) + f"{symbol}-{self.interval}-{stamp}.zip"

    def download_many(self, keys: list[str], out_dir: Path, *, refresh: bool = False, workers: int = 8,
                      log_every: int = 200) -> tuple[list[Path], list[tuple[str, str]]]:
        """Download many zips in parallel (each checksum-verified). Returns (paths, failures).

        The fetch log is written once, from the calling thread, to avoid concurrent writes.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        paths: list[Path] = []
        failures: list[tuple[str, str]] = []
        fresh: list[tuple[Path, str, str]] = []
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(self._download_verified, k, out_dir, refresh): k for k in keys}
            for n, f in enumerate(as_completed(futs), 1):
                try:
                    path, checksum, was_new = f.result()
                    paths.append(path)
                    if was_new:
                        fresh.append((path, futs[f], checksum))
                except Exception as e:  # recorded and reported, never silently dropped
                    failures.append((futs[f], repr(e)))
                if n % log_every == 0:
                    logger.info(f"downloaded {n}/{len(keys)} files into {out_dir.name}")
        by_dir: dict[Path, list] = {}
        for path, key, checksum in fresh:
            by_dir.setdefault(path.parent, []).append((path, key, checksum))
        for d, items in by_dir.items():
            log = FetchLog(d / "fetch_log.json")
            for path, key, checksum in items:
                log.entries[path.name] = {"source": "binance", "url": f"{self.archive_url}/{key}",
                                          "fetched_utc": utc_now(), "rows": -1, "sha256": checksum,
                                          "checksum_sha256": checksum, "verified": True}
            write_json(log.entries, log.path)
        return sorted(paths), failures

    def _download_verified(self, key: str, out_dir: Path, refresh: bool) -> tuple[Path, str, bool]:
        """(path, sha256, newly_downloaded) — no fetch-log write (thread-safe)."""
        out = out_dir / Path(key).name
        if out.exists() and not refresh:
            return out, "", False
        url = f"{self.archive_url}/{key}"
        blob = with_retries(lambda: self._get(url, timeout=120), retries=self.retries, backoff=self.backoff,
                            what=f"binance {Path(key).name}")
        expected = parse_checksum(with_retries(lambda: self._get(url + ".CHECKSUM").decode(), retries=self.retries,
                                               backoff=self.backoff, what=f"binance checksum {Path(key).name}"))
        verify_bytes(blob, expected)
        out_dir.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".part")
        tmp.write_bytes(blob)
        tmp.replace(out)          # atomic: a crash never leaves a truncated zip behind
        return out, expected, True

    def download(self, key: str, out_dir: Path, *, refresh: bool = False) -> Path:
        """Download one zip, verify it against its .CHECKSUM, and keep it (write-once unless refresh)."""
        out = out_dir / Path(key).name
        if out.exists() and not refresh:
            return out
        url = f"{self.archive_url}/{key}"
        blob = with_retries(lambda: self._get(url, timeout=120), retries=self.retries, backoff=self.backoff,
                            what=f"binance {Path(key).name}")
        expected = parse_checksum(with_retries(lambda: self._get(url + ".CHECKSUM").decode(), retries=self.retries,
                                               backoff=self.backoff, what=f"binance checksum {Path(key).name}"))
        verify_bytes(blob, expected)
        out_dir.mkdir(parents=True, exist_ok=True)
        out.write_bytes(blob)
        FetchLog(out_dir / "fetch_log.json").record(out, source="binance", url=url, rows=-1,
                                                    extra={"checksum_sha256": expected, "verified": True})
        return out
