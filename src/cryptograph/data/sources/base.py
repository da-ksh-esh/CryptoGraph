"""Common interface for market-data sources.

Every source returns a daily frame indexed by a tz-naive UTC ``DatetimeIndex``
named ``date`` with (a subset of) the canonical columns in ``OHLCV_COLUMNS``.
New providers (Binance, Kaiko, ...) plug in by implementing ``DataSource``.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Protocol, TypeVar, runtime_checkable

import pandas as pd

from ...common.logger import get_logger

logger = get_logger(__name__)

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume", "market_cap"]

T = TypeVar("T")


@runtime_checkable
class DataSource(Protocol):
    """A provider of daily USD price history for one asset at a time."""

    name: str

    def snapshot(self, asset: str, out_dir: Path, *, refresh: bool = False) -> Path | None:
        """Download the raw file for `asset` once into `out_dir` (reuse unless refresh)."""

    def load(self, path: Path) -> pd.DataFrame:
        """Parse a raw snapshot into the canonical daily frame."""


def with_retries(fn: Callable[[], T], *, retries: int, backoff: float, what: str) -> T:
    """Call `fn` with exponential backoff (backoff * 2**attempt seconds)."""
    last: Exception | None = None
    for attempt in range(retries):
        try:
            return fn()
        except Exception as exc:  # network errors differ per client library
            last = exc
            wait = backoff * (2 ** attempt)
            logger.warning(f"{what}: attempt {attempt + 1}/{retries} failed ({exc!r}); retrying in {wait:.1f}s")
            time.sleep(wait)
    raise RuntimeError(f"{what}: all {retries} attempts failed") from last


def to_utc_daily_index(idx: pd.Index) -> pd.DatetimeIndex:
    """Normalise any datetime-like index to tz-naive UTC midnight dates."""
    dt = pd.DatetimeIndex(pd.to_datetime(idx, utc=True))
    return dt.tz_convert("UTC").tz_localize(None).normalize().rename("date")
