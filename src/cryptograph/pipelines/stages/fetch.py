"""Stage 2 — raw snapshots (reused unless --refresh) + cross-source reconciliation."""

from __future__ import annotations

import pandas as pd

from ...common.logger import get_logger
from ...data.config import DataConfig
from ...data.quality.reconcile import ReconcileParams, reconcile_coin
from ...data.sources import CoinDeskSource, CoinMetricsSource, YahooSource
from ...data.storage import write_csv, write_parquet
from . import load_universe

logger = get_logger(__name__)


def run(cfg: DataConfig, refresh: bool = False) -> None:
    audit, _, coins = load_universe(cfg)
    s, w = cfg.sources, cfg.window
    yahoo = YahooSource(w.start_date, w.end_date, retries=s.retries, backoff=s.backoff_base_seconds)
    cm = CoinMetricsSource(s.coinmetrics_url, retries=s.retries, backoff=s.backoff_base_seconds)
    cd = CoinDeskSource(s.coindesk_api_key_env, w.end_date)
    logger.info(f"CoinDesk tiebreaker {'enabled' if cd.enabled else 'disabled (no API key)'}")
    params = ReconcileParams(tol=cfg.quality.xsource_close_tol)

    frames, reports = [], []
    for coin in coins:
        row = audit.loc[coin]
        ydf = None
        if isinstance(row["yahoo_ticker"], str) and bool(row["ticker_verified"]):
            p = yahoo.snapshot(coin, row["yahoo_ticker"], cfg.raw_dir / "yahoo", refresh=refresh)
            ydf = yahoo.load(p) if p else None
        n_dup = pd.Series(dtype=float)
        if ydf is not None:
            dup = ydf.index.duplicated(keep="last")
            n_dup = pd.Series(1.0, index=ydf.index[ydf.index.duplicated(keep=False)]).groupby(level=0).sum()
            ydf = ydf[~dup]
            ydf = ydf.loc[pd.Timestamp(w.start_date): pd.Timestamp(w.end_date)]
        cdf = None
        if isinstance(row.get("coinmetrics_asset"), str):
            p = cm.snapshot(row["coinmetrics_asset"], cfg.raw_dir / "coinmetrics", refresh=refresh)
            cdf = cm.load(p).loc[pd.Timestamp(w.start_date): pd.Timestamp(w.end_date)] if p else None
        ddf = None
        if cd.enabled:
            p = cd.snapshot(coin, cfg.raw_dir / "coindesk", refresh=refresh)
            ddf = cd.load(p) if p else None

        merged, rep = reconcile_coin(ydf, cdf, ddf, params)
        merged["raw_duplicates"] = n_dup.reindex(merged.index).fillna(0)
        merged["cm_close_aligned"] = (cdf["cm_price"].shift(-(rep["lag"] or 0)).reindex(merged.index)
                                      if cdf is not None and rep["lag"] is not None else float("nan"))
        merged["coin"] = coin
        frames.append(merged.reset_index())
        reports.append({"coin": coin, "yahoo_ticker": row["yahoo_ticker"],
                        "coinmetrics_asset": row.get("coinmetrics_asset"),
                        "raw_duplicate_dates": int((n_dup > 0).sum()), **rep})
        logger.info(f"{coin}: overlap={rep['overlap_days']} lag={rep['lag']} flagged={rep['n_flagged']} "
                    f"R2={rep['n_corrected_R2']} backfilled={rep['n_backfilled']}")

    reconciled = pd.concat(frames, ignore_index=True)
    write_parquet(reconciled.set_index(["coin", "date"]), cfg.interim_dir / "reconciled.parquet")
    rep_df = pd.DataFrame(reports)
    write_csv(rep_df, cfg.reports_dir / "cross_source_report.csv")
    print(rep_df[["coin", "overlap_days", "lag", "n_flagged", "discrepancy_rate", "median_rel_diff",
                  "n_corrected_R2", "n_kept_flagged_R3", "backfill_ok", "n_backfilled"]].to_string(index=False))
