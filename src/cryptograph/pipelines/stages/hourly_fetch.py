"""Hourly Stage 2a — download (checksum-verified) and stitch Binance 1h klines.

For every model coin and event coin: list the archive files inside the hourly
window (monthly zips, plus daily zips for the partial last month), download them
in parallel (write-once snapshots in data/raw/binance/1h/<SYMBOL>/), parse, and
stitch symbol segments (ticker migrations, reused LUNAUSDT) into one history.

Writes data/interim/hourly/raw_1h.parquet, reports/binance_continuity.csv and
the Coin Metrics USDT snapshot used for the peg check.
"""

from __future__ import annotations

import json

import pandas as pd

from ...common.logger import get_logger
from ...data.config import HourlyConfig, SymbolOverride
from ...data.quality.binance_continuity import SegmentData, stitch_segments
from ...data.sources import CoinMetricsSource, read_klines
from ...data.sources.binance import BinanceArchive
from ...data.storage import write_csv, write_parquet
from ...data.universe_hourly import HOUR, _ts

logger = get_logger(__name__)


def hourly_universe(cfg: HourlyConfig) -> tuple[list[str], dict[str, list[SymbolOverride]], dict[str, str]]:
    """(ordered coins incl. event coins, coin -> symbol history, coin -> coingecko id)."""
    summary = json.loads((cfg.interim_dir / "universe_v2.json").read_text(encoding="utf-8"))
    audit = pd.read_csv(cfg.reports_dir / "universe_audit_v2.csv")
    ids = dict(zip(audit["coin"], audit["coingecko_id"]))
    hist: dict[str, list[SymbolOverride]] = {}
    for coin in summary["included"]:
        cg = ids[coin]
        hist[coin] = cfg.universe.symbol_overrides.get(cg) or [SymbolOverride(symbol=f"{coin}{cfg.universe.quote_asset}")]
    for coin, h in cfg.universe.event_coins.items():
        hist[coin] = h
    return summary["included"] + list(cfg.universe.event_coins), hist, ids


def run(cfg: HourlyConfig, refresh: bool = False) -> None:
    b = cfg.binance
    arch = BinanceArchive(b.archive_url, b.list_url, b.market, b.interval, b.retries, b.backoff_base_seconds)
    raw = cfg.raw_dir / "binance" / b.interval
    start, end = _ts(cfg.window.start), _ts(cfg.window.end)
    coins, hist, _ = hourly_universe(cfg)
    last_month = end.strftime("%Y-%m")

    frames, report, failures, n_files = [], [], [], 0
    for coin in coins:
        segs = []
        for ov in hist[coin]:
            limit = end if ov.until is None else min(end, pd.Timestamp(ov.until) + pd.Timedelta(days=1) - HOUR)
            files = arch.symbol_files(ov.symbol, (last_month,))
            keys = [arch.file_key(ov.symbol, "monthly", m) for m in files.months
                    if start.strftime("%Y-%m") <= m <= limit.strftime("%Y-%m")]
            covered = set(files.months)
            keys += [arch.file_key(ov.symbol, "daily", d) for d in files.days
                     if d[:7] not in covered and start <= pd.Timestamp(d) <= limit]
            paths, fails = arch.download_many(keys, raw / ov.symbol, refresh=refresh, workers=b.download_workers)
            failures += [(ov.symbol, k, e) for k, e in fails]
            n_files += len(paths)
            if not paths:
                logger.warning(f"{coin}: no files for {ov.symbol}")
                continue
            kl = pd.concat([read_klines(p) for p in paths])
            kl = kl.loc[start: end]
            segs.append(SegmentData(ov.symbol, kl, pd.Timestamp(ov.until) if ov.until else None))
            logger.info(f"{coin}/{ov.symbol}: {len(paths)} files, {len(kl)} rows")
        stitched, rep = stitch_segments(segs, cfg.quality.rename_max_price_jump)
        report += [{"coin": coin, **r} for r in rep]
        if len(stitched):
            frames.append(stitched.assign(coin=coin))

    out = pd.concat(frames).reset_index().rename(columns={"index": "time"})
    out = out.set_index(["time", "coin"]).sort_index()
    write_parquet(out, cfg.interim_dir / "raw_1h.parquet")
    rep = pd.DataFrame(report)
    write_csv(rep, cfg.reports_dir / "binance_continuity.csv")

    # USDT/USD reference for the peg check (Coin Metrics, daily; archive end date recorded)
    from ...data.config import load_config

    CoinMetricsSource(load_config().sources.coinmetrics_url).snapshot("usdt", cfg.raw_dir / "coinmetrics", refresh=refresh)

    print(f"files: {n_files} downloaded/verified, failures: {len(failures)}")
    for f in failures[:20]:
        print("  FAILED", f)
    print(f"rows: {len(out):,} coin-hours for {out.index.get_level_values('coin').nunique()} coins")
    if len(rep):
        print(rep.to_string(index=False))
