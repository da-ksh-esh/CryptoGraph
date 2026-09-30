"""Coin Metrics Community Data archive (https://github.com/coinmetrics/data).

Licensed CC BY-NC 4.0 — cite "Coin Metrics Community Network Data".
Used for close validation, close backfill and market capitalisation.
The raw CSV is snapshotted verbatim; only the columns below are parsed.
"""

from __future__ import annotations

import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

from ...common.logger import get_logger
from ..storage import FetchLog
from .base import to_utc_daily_index, with_retries

logger = get_logger(__name__)

CM_COLUMNS = {
    "PriceUSD": "cm_price",
    "ReferenceRateUSD": "cm_refrate",
    "CapMrktCurUSD": "cm_mcap",
    "volume_reported_spot_usd_1d": "cm_volume",
}


class CoinMetricsSource:
    name = "coinmetrics"

    def __init__(self, url_template: str, retries: int = 5, backoff: float = 2.0):
        self.url_template = url_template
        self.retries = retries
        self.backoff = backoff

    def url(self, asset: str) -> str:
        return self.url_template.format(asset=asset)

    def exists(self, asset: str) -> bool:
        req = urllib.request.Request(self.url(asset), method="HEAD", headers={"User-Agent": "cryptograph"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status == 200
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return False
            raise

    def _download(self, asset: str) -> bytes:
        req = urllib.request.Request(self.url(asset), headers={"User-Agent": "cryptograph"})
        with urllib.request.urlopen(req, timeout=180) as r:
            return r.read()

    def snapshot(self, asset: str, out_dir: Path, *, refresh: bool = False) -> Path | None:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{asset}.csv"
        if path.exists() and not refresh:
            return path
        try:
            if not with_retries(lambda: self.exists(asset), retries=self.retries, backoff=self.backoff,
                                what=f"coinmetrics HEAD {asset}"):
                return None
        except RuntimeError:
            return None
        blob = with_retries(lambda: self._download(asset), retries=self.retries, backoff=self.backoff,
                            what=f"coinmetrics {asset}")
        path.write_bytes(blob)
        rows = blob.count(b"\n") - 1
        FetchLog(out_dir / "fetch_log.json").record(path, source="coinmetrics", url=self.url(asset), rows=rows,
                                                    extra={"license": "CC BY-NC 4.0"})
        return path

    @staticmethod
    def load(path: Path) -> pd.DataFrame:
        header = pd.read_csv(path, nrows=0).columns
        use = ["time"] + [c for c in CM_COLUMNS if c in header]
        df = pd.read_csv(path, usecols=use, low_memory=False)
        df.index = to_utc_daily_index(df.pop("time"))
        df = df.rename(columns=CM_COLUMNS).apply(pd.to_numeric, errors="coerce")
        for col in CM_COLUMNS.values():
            if col not in df.columns:
                df[col] = float("nan")
        df = df[~df.index.duplicated(keep="last")].sort_index()
        return df[list(CM_COLUMNS.values())]
