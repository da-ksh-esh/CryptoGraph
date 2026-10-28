"""Download checksum-verified Binance 5-minute spot klines for the hourly universe and stack them into
data/interim/hourly/klines_5m.parquet (time, coin) -> open, high, low, close, volume, trades.

Same archive, symbol history and window as the 1h data (configs/data/hourly.yaml); 5m bars are only aggregated
into hourly features (data/features/subhour.py), never used as a separate stream.
"""

from __future__ import annotations

import torch  # noqa: F401  (torch before pandas on Windows)
import pandas as pd

from cryptograph.common.logger import get_logger
from cryptograph.data.config import load_hourly_config
from cryptograph.data.sources.binance import BinanceArchive, read_klines
from cryptograph.data.universe_hourly import _ts
from cryptograph.pipelines.stages.hourly_fetch import hourly_universe

logger = get_logger(__name__)
INTERVAL = "5m"


def main() -> None:
    cfg = load_hourly_config("configs/data/hourly.yaml")
    b = cfg.binance
    arch = BinanceArchive(b.archive_url, b.list_url, b.market, INTERVAL, b.retries, b.backoff_base_seconds)
    raw = cfg.raw_dir / "binance" / INTERVAL
    start, end = _ts(cfg.window.start), _ts(cfg.window.end)
    coins, hist, _ = hourly_universe(cfg)
    last_month = end.strftime("%Y-%m")
    frames, failures = [], []
    for coin in coins:
        for ov in hist[coin]:
            limit = end if ov.until is None else min(end, pd.Timestamp(ov.until) + pd.Timedelta(days=1)
                                                     - pd.Timedelta(minutes=5))
            files = arch.symbol_files(ov.symbol, (last_month,))
            keys = [arch.file_key(ov.symbol, "monthly", m) for m in files.months
                    if start.strftime("%Y-%m") <= m <= limit.strftime("%Y-%m")]
            covered = set(files.months)
            keys += [arch.file_key(ov.symbol, "daily", d) for d in files.days
                     if d[:7] not in covered and start <= pd.Timestamp(d) <= limit]
            paths, fails = arch.download_many(keys, raw / ov.symbol, workers=b.download_workers)
            failures += fails
            if not paths:
                continue
            kl = pd.concat([read_klines(p) for p in paths]).sort_index()
            kl = kl[~kl.index.duplicated()].loc[start: limit + pd.Timedelta(minutes=55)]
            frames.append(kl.assign(coin=coin)[["coin", "open", "high", "low", "close", "volume", "trades"]])
            logger.info(f"{coin}/{ov.symbol}: {len(paths)} files, {len(kl):,} rows")
    out = pd.concat(frames).reset_index().set_index(["time", "coin"]).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    out.to_parquet(cfg.interim_dir / "klines_5m.parquet")
    logger.info(f"{len(out):,} 5m rows; {len(failures)} failed downloads {failures[:5]}")


if __name__ == "__main__":
    main()
