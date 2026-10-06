"""Binance USD-M futures data from the same public archive (https://data.binance.vision): leverage pressure.

Two series per coin, both monthly zips, each verified against its `.CHECKSUM` like the spot klines:
  * premiumIndexKlines 1h: (perpetual price - spot index) / index, as an hourly kline. Its close is the premium at
    the end of the bar; indexed by bar open time exactly like the spot klines, so bar t of both refers to [t, t+1h).
  * fundingRate: the rate settled every 8h (some coins 4h). A rate is known from its settlement time `calc_time`.

Futures symbols differ from spot for a few coins: SHIB and PEPE trade as 1000SHIB / 1000PEPE (the premium and the
funding rate are ratios, so the 1000x lot does not matter), and POL's history before the 2024-09 rename is MATIC.
"""

from __future__ import annotations

import io
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from ...common.logger import get_logger
from .binance import _MONTH_RE, BinanceArchive, read_klines

logger = get_logger(__name__)

# coin -> futures symbols, oldest first (later symbols win where they overlap)
FUTURES_SYMBOLS = {"SHIB": ["1000SHIBUSDT"], "PEPE": ["1000PEPEUSDT"], "POL": ["MATICUSDT", "POLUSDT"]}


def futures_symbols(coin: str) -> list[str]:
    return FUTURES_SYMBOLS.get(coin, [f"{coin}USDT"])


def _prefix(kind: str, symbol: str) -> str:
    if kind == "premium":
        return f"data/futures/um/monthly/premiumIndexKlines/{symbol}/1h/"
    if kind == "funding":
        return f"data/futures/um/monthly/fundingRate/{symbol}/"
    raise ValueError(kind)


def download_futures(archive: BinanceArchive, coins: list[str], raw_dir: Path, workers: int = 8) -> dict:
    """Download all monthly premium-index (1h) and funding-rate zips for `coins`. Returns a per-file report."""
    report = {"files": 0, "failures": [], "symbols": {}}
    for coin in coins:
        for sym in futures_symbols(coin):
            for kind in ("premium", "funding"):
                keys = [k for k in archive.list_keys(_prefix(kind, sym)) if _MONTH_RE.search(k)]
                paths, fails = archive.download_many(keys, raw_dir / kind / sym, workers=workers)
                report["files"] += len(paths)
                report["failures"] += fails
                report["symbols"][f"{sym}/{kind}"] = len(paths)
        logger.info(f"futures {coin}: {futures_symbols(coin)} done")
    return report


def read_funding(path: Path) -> pd.Series:
    """Funding-rate zip -> Series of rates indexed by settlement time (UTC, tz-naive)."""
    with zipfile.ZipFile(path) as zf:
        df = pd.read_csv(io.BytesIO(zf.read(zf.namelist()[0])))
    if "calc_time" not in df.columns:                   # header-less older files
        df.columns = ["calc_time", "funding_interval_hours", "last_funding_rate"][: df.shape[1]]
    t = pd.to_numeric(df["calc_time"]).astype("int64").to_numpy()
    ns = np.where(np.abs(t) >= 10**15, t * 1_000, t * 1_000_000)
    return pd.Series(pd.to_numeric(df["last_funding_rate"], errors="coerce").to_numpy(),
                     index=pd.DatetimeIndex(pd.to_datetime(ns, unit="ns"), name="time"), name="funding").sort_index()


def load_coin(raw_dir: Path, coin: str) -> tuple[pd.Series, pd.Series]:
    """(hourly premium close indexed by bar open time, funding rates indexed by settlement time) for one coin."""
    prem, fund = [], []
    for sym in futures_symbols(coin):
        prem += [read_klines(p)["close"] for p in sorted((raw_dir / "premium" / sym).glob("*.zip"))]
        fund += [read_funding(p) for p in sorted((raw_dir / "funding" / sym).glob("*.zip"))]
    p = pd.concat(prem) if prem else pd.Series(dtype=float)
    f = pd.concat(fund) if fund else pd.Series(dtype=float)
    p = p[~p.index.duplicated(keep="last")].sort_index()
    f = f[~f.index.duplicated(keep="last")].sort_index()
    return p, f


FUTURES_FEATURES = ["premium", "premium_mean_24h", "funding_last", "market_premium", "futures_available"]


def futures_feature_table(times: pd.DatetimeIndex, coins: list[str], raw_dir: Path, train_mask: np.ndarray,
                          clip: float = 5.0) -> tuple[np.ndarray, list[str]]:
    """(T, N, F) float32 features known at the end of bar t (no look-ahead):

    premium            futures premium at the close of bar t (same timestamp convention as the spot bar t)
    premium_mean_24h   mean premium over bars t-23..t
    funding_last       last funding rate settled at or before the end of bar t
    market_premium     cross-coin mean premium at t (crowded leverage across the market)
    futures_available  1 where the coin had futures data at t, else 0 (features are 0 there)

    The first four are standardised with the pooled mean/std of the training rows (train_mask over T) and clipped.
    """
    T, N = len(times), len(coins)
    prem = np.full((T, N), np.nan, np.float64)
    fund = np.full((T, N), np.nan, np.float64)
    bar_end = times + pd.Timedelta(hours=1)
    for j, c in enumerate(coins):
        p, f = load_coin(raw_dir, c)
        if len(p):
            prem[:, j] = p.reindex(times).to_numpy()
        if len(f):
            # last settlement with calc_time <= end of bar t; nothing before the first settlement
            pos = np.searchsorted(f.index.values, bar_end.values, side="right") - 1
            ok = pos >= 0
            fund[ok, j] = f.to_numpy()[pos[ok]]
            last_seen = p.index.max() if len(p) else f.index.max()
            fund[times > last_seen, j] = np.nan          # coin no longer listed on futures
            if len(p):
                fund[times < p.index.min(), j] = np.nan
    avail = np.isfinite(prem)
    prem_24 = pd.DataFrame(prem).rolling(24, min_periods=12).mean().to_numpy()
    with warnings.catch_warnings():                    # hours with no futures data at all -> NaN -> 0 below
        warnings.simplefilter("ignore", RuntimeWarning)
        mkt = np.nanmean(np.where(avail, prem, np.nan), axis=1)
    mkt = np.repeat(mkt[:, None], N, axis=1)
    feats = []
    for x in (prem, prem_24, fund, mkt):
        tr = x[train_mask]
        mu, sd = np.nanmean(tr), np.nanstd(tr)
        z = np.clip((x - mu) / (sd if sd > 0 else 1.0), -clip, clip)
        feats.append(np.where(np.isfinite(z), z, 0.0))
    feats.append(avail.astype(np.float64))
    out = np.stack(feats, axis=-1).astype(np.float32)
    out[~avail] = 0.0
    out[..., -1] = avail
    return out, FUTURES_FEATURES
