"""Binance public REST klines (no API key): the newest hourly bars, which the archive publishes only a day later.

    GET <base>/api/v3/klines?symbol=BTCUSDT&interval=1h&startTime=<ms>&endTime=<ms>&limit=1000

Rows come in the archive's kline columns (`binance.KLINE_COLUMNS`), with the open time in milliseconds. The last row
can be the bar that is still trading, so only bars whose close time has passed are kept. Base URLs are tried in
order: data-api.binance.vision serves public market data only, api.binance.com is the main API. Server errors and
timeouts are retried with exponential backoff; a 4xx answer (e.g. an unknown symbol) is not, so a delisted coin fails
fast. REST bars carry no checksum, unlike the archive (`binance.BinanceArchive`): the live pipeline uses them only
after the research period (`serving/live.py`).
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Callable

import pandas as pd

from ...common.logger import get_logger
from .binance import KLINE_COLUMNS

logger = get_logger(__name__)

DEFAULT_BASE_URLS = ("https://data-api.binance.vision", "https://api.binance.com")
VALUE_COLUMNS = ["open", "high", "low", "close", "volume", "quote_volume", "trades"]


class SymbolError(ValueError):
    """The API refused the request (4xx), e.g. an unknown or delisted symbol: retrying will not help."""

    def __init__(self, message: str, code: int | None = None):
        super().__init__(message)
        self.code = code


BLOCKED_CODES = (403, 451)                  # this endpoint is not available from here (e.g. by region): try the next


def to_ms(ts) -> int:
    """UTC timestamp (tz-aware or tz-naive UTC) -> epoch milliseconds."""
    t = pd.Timestamp(ts)
    if t.tzinfo is not None:
        t = t.tz_convert("UTC").tz_localize(None)
    return int(t.value // 1_000_000)


def utc_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC").tz_localize(None)


def klines_frame(rows: list[list], now_ms: int | None = None) -> pd.DataFrame:
    """REST kline rows -> frame indexed by UTC open time (tz-naive) with the columns of `binance.read_klines`.
    Bars that have not closed by `now_ms` are dropped."""
    if not rows:
        return pd.DataFrame({c: pd.Series(dtype=float) for c in VALUE_COLUMNS},
                            index=pd.DatetimeIndex([], name="time"))
    df = pd.DataFrame(rows, columns=KLINE_COLUMNS)
    if now_ms is not None:
        df = df[pd.to_numeric(df["close_time"]) < now_ms]
    out = df[VALUE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    out.index = pd.DatetimeIndex(pd.to_datetime(pd.to_numeric(df["open_time"]).astype("int64"), unit="ms"), name="time")
    return out[~out.index.duplicated(keep="last")].sort_index()


def _urlopen(url: str, timeout: float) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "cryptograph-live/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


class BinanceRestKlines:
    name = "binance-rest"

    def __init__(self, base_urls: tuple[str, ...] | list[str] = DEFAULT_BASE_URLS, interval: str = "1h",
                 limit: int = 1000, retries: int = 3, backoff: float = 1.0, timeout: float = 20.0,
                 opener: Callable[[str, float], bytes] | None = None):
        self.base_urls = [u.rstrip("/") for u in base_urls]
        self.interval, self.limit = interval, int(limit)
        self.retries, self.backoff, self.timeout = retries, backoff, timeout
        self.step_ms = int(pd.Timedelta(interval).value // 1_000_000)
        self._open = opener or _urlopen

    def _get(self, url: str) -> bytes:
        """One URL with retries on server errors, rate limits and network failures (not on other 4xx)."""
        last: Exception | None = None
        for attempt in range(self.retries):
            try:
                return self._open(url, self.timeout)
            except urllib.error.HTTPError as e:
                if 400 <= e.code < 500 and e.code not in (418, 429):
                    body = e.read().decode(errors="replace")[:200] if e.fp is not None else ""
                    raise SymbolError(f"HTTP {e.code} for {url}: {body}", e.code) from e
                last = e
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
                last = e
            wait = self.backoff * 2 ** attempt
            logger.warning(f"binance rest: attempt {attempt + 1}/{self.retries} failed ({last!r}); retry in {wait:.0f}s")
            time.sleep(wait)
        raise RuntimeError(f"binance rest: all {self.retries} attempts failed for {url}") from last

    def get_json(self, path: str, params: dict) -> object:
        """GET `path` on the first base URL that answers. A 4xx answer is final (SymbolError), except 403 / 451 (the
        endpoint is not available from here), which moves on to the next base URL like a network failure."""
        query = urllib.parse.urlencode(params)
        errors = []
        for base in self.base_urls:
            try:
                return json.loads(self._get(f"{base}{path}?{query}"))
            except SymbolError as e:
                if e.code not in BLOCKED_CODES:
                    raise
                errors.append(f"{base}: {e}")
            except Exception as e:                       # this mirror is unreachable: try the next one
                errors.append(f"{base}: {e!r}")
        raise RuntimeError("no Binance REST endpoint answered: " + "; ".join(errors))

    def klines(self, symbol: str, start, end, now=None) -> pd.DataFrame:
        """Closed bars of `symbol` with open time in [start, end] (UTC), paginated by `limit` rows per request."""
        now_ms = to_ms(utc_now() if now is None else now)
        cur, end_ms = to_ms(start), to_ms(end)
        frames = []
        while cur <= end_ms:
            rows = self.get_json("/api/v3/klines", {"symbol": symbol, "interval": self.interval, "startTime": cur,
                                                    "endTime": end_ms, "limit": self.limit})
            if not rows:
                break
            frames.append(klines_frame(rows, now_ms))
            if len(rows) < self.limit:
                break
            cur = int(rows[-1][0]) + self.step_ms
        if not frames:
            return klines_frame([])
        out = pd.concat(frames)
        return out[~out.index.duplicated(keep="last")].sort_index()
