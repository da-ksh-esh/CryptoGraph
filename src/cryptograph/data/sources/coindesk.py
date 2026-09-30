"""CoinDesk Data (formerly CryptoCompare) daily OHLCV — optional tiebreaker.

Only active when an API key is present in the configured environment variable;
otherwise every call is a no-op and disputed days are resolved without it.
"""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from ..storage import FetchLog
from .base import to_utc_daily_index, with_retries

HISTODAY_URL = "https://min-api.cryptocompare.com/data/v2/histoday"


class CoinDeskSource:
    name = "coindesk"

    def __init__(self, api_key_env: str, end: date, retries: int = 3, backoff: float = 2.0):
        self.api_key = os.environ.get(api_key_env)
        self.end = end
        self.retries = retries
        self.backoff = backoff

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def _download(self, symbol: str) -> bytes:
        ts = int(datetime(self.end.year, self.end.month, self.end.day, tzinfo=timezone.utc).timestamp())
        q = urllib.parse.urlencode({"fsym": symbol, "tsym": "USD", "limit": 2000, "toTs": ts})
        req = urllib.request.Request(f"{HISTODAY_URL}?{q}", headers={"authorization": f"Apikey {self.api_key}"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.read()

    def snapshot(self, symbol: str, out_dir: Path, *, refresh: bool = False) -> Path | None:
        if not self.enabled:
            return None
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{symbol}.json"
        if path.exists() and not refresh:
            return path
        blob = with_retries(lambda: self._download(symbol), retries=self.retries, backoff=self.backoff,
                            what=f"coindesk {symbol}")
        path.write_bytes(blob)
        FetchLog(out_dir / "fetch_log.json").record(path, source="coindesk", url=HISTODAY_URL, rows=-1,
                                                    extra={"symbol": symbol})
        return path

    @staticmethod
    def load(path: Path) -> pd.DataFrame:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        rows = payload.get("Data", {}).get("Data", [])
        df = pd.DataFrame(rows)
        if df.empty:
            return pd.DataFrame(columns=["cd_close"], index=pd.DatetimeIndex([], name="date"))
        df.index = to_utc_daily_index(pd.to_datetime(df["time"], unit="s", utc=True))
        out = df[["close"]].rename(columns={"close": "cd_close"})
        return out[out["cd_close"] > 0]
