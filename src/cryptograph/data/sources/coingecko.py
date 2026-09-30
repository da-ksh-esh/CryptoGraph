"""CoinGecko — used ONLY for the current market-cap ranking snapshot.

The free Demo plan serves just the last 365 days, so it is never used for
price history. The ranking snapshot is saved with its fetch timestamp so the
universe selection is reproducible from disk.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import pandas as pd

from ..storage import FetchLog
from .base import with_retries


class CoinGeckoRanking:
    name = "coingecko"

    def __init__(self, url: str, retries: int = 5, backoff: float = 2.0):
        self.url = url
        self.retries = retries
        self.backoff = backoff

    def _download(self, url: str) -> bytes:
        req = urllib.request.Request(url, headers={"User-Agent": "cryptograph", "accept": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.read()

    def _snap(self, url: str, path: Path, refresh: bool) -> Path:
        if path.exists() and not refresh:
            return path
        blob = with_retries(lambda: self._download(url), retries=self.retries, backoff=self.backoff,
                            what=f"coingecko {path.stem}")
        path.write_bytes(blob)
        FetchLog(path.parent / "fetch_log.json").record(path, source="coingecko", url=url, rows=len(json.loads(blob)))
        return path

    def snapshot(self, out_dir: Path, *, ids: list[str] | None = None, refresh: bool = False) -> list[Path]:
        """Top-250 ranking snapshot, plus a by-id snapshot so candidates outside the top 250 get a market cap."""
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = [self._snap(self.url, out_dir / "markets_snapshot.json", refresh)]
        if ids:
            base = self.url.split("?")[0]
            url = f"{base}?vs_currency=usd&ids={','.join(sorted(ids))}&per_page=250"
            paths.append(self._snap(url, out_dir / "markets_candidates.json", refresh))
        return paths

    @staticmethod
    def load(paths: Path | list[Path]) -> pd.DataFrame:
        frames = []
        for p in [paths] if isinstance(paths, Path) else paths:
            frames.append(pd.DataFrame(json.loads(Path(p).read_text(encoding="utf-8"))))
        df = pd.concat(frames, ignore_index=True).drop_duplicates("id")
        df = df[["id", "symbol", "name", "market_cap_rank", "market_cap", "current_price",
                 "total_volume", "last_updated"]]
        return df.rename(columns={"id": "coingecko_id"})
