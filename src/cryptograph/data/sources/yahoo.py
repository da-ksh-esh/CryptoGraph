"""Yahoo Finance daily OHLCV via yfinance — one ticker per request.

Batch ``yf.download`` silently drops single failing tickers, so each asset is
fetched on its own with retries. ``auto_adjust=False`` keeps raw prices.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from ...common.logger import get_logger
from ..storage import FetchLog
from .base import to_utc_daily_index, with_retries

logger = get_logger(__name__)


class YahooSource:
    name = "yahoo"

    def __init__(self, start: date, end: date, retries: int = 5, backoff: float = 2.0):
        self.start = start
        self.end = end  # inclusive
        self.retries = retries
        self.backoff = backoff

    # ---- network ---------------------------------------------------------------
    def fetch(self, ticker: str) -> pd.DataFrame:
        import yfinance as yf  # lazy: keeps module importable without network deps

        def _call() -> pd.DataFrame:
            return yf.Ticker(ticker).history(
                start=self.start.isoformat(),
                end=(self.end + timedelta(days=1)).isoformat(),  # yfinance end is exclusive
                interval="1d",
                auto_adjust=False,
                actions=False,
                raise_errors=True,
            )

        raw = with_retries(_call, retries=self.retries, backoff=self.backoff, what=f"yahoo {ticker}")
        return self._normalise(raw)

    @staticmethod
    def _normalise(raw: pd.DataFrame) -> pd.DataFrame:
        if raw is None or raw.empty:
            return pd.DataFrame(columns=["open", "high", "low", "close", "adj_close", "volume"],
                                index=pd.DatetimeIndex([], name="date"))
        df = raw.rename(columns=lambda c: c.lower().replace(" ", "_"))
        df.index = to_utc_daily_index(df.index)
        keep = [c for c in ["open", "high", "low", "close", "adj_close", "volume"] if c in df.columns]
        return df[keep].sort_index()

    def info(self, ticker: str) -> dict:
        import yfinance as yf

        try:
            return with_retries(lambda: yf.Ticker(ticker).info or {}, retries=2, backoff=self.backoff,
                                what=f"yahoo info {ticker}")
        except RuntimeError:
            return {}

    def search(self, symbol: str) -> list[dict]:
        """Yahoo search results for `symbol` restricted to crypto quotes against USD."""
        import yfinance as yf

        try:
            res = with_retries(lambda: yf.Search(symbol, max_results=15, news_count=0, lists_count=0).quotes,
                               retries=2, backoff=self.backoff, what=f"yahoo search {symbol}")
        except RuntimeError:
            return []
        return [q for q in res if q.get("quoteType") == "CRYPTOCURRENCY" and str(q.get("symbol", "")).endswith("-USD")]

    # ---- snapshot ----------------------------------------------------------------
    def snapshot(self, coin: str, ticker: str, out_dir: Path, *, refresh: bool = False) -> Path | None:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{coin}.csv"
        if path.exists() and not refresh:
            return path
        df = self.fetch(ticker)
        if df.empty:
            logger.warning(f"yahoo {ticker}: no rows in window")
            return None
        df.to_csv(path, index_label="date")
        FetchLog(out_dir / "fetch_log.json").record(
            path, source="yahoo", url=f"yfinance.Ticker('{ticker}').history(auto_adjust=False)", rows=len(df),
            extra={"ticker": ticker, "start": self.start.isoformat(), "end": self.end.isoformat()},
        )
        return path

    @staticmethod
    def load(path: Path) -> pd.DataFrame:
        df = pd.read_csv(path, index_col="date", parse_dates=["date"])
        df.index = to_utc_daily_index(df.index)
        return df
