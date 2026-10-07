"""Hourly Stage 2b — quality checks, cleaning, 2h resampling, USDT peg flag and daily cross-check.

Inputs: data/interim/hourly/raw_1h.parquet (from the fetch stage).
Writes:
  data/interim/hourly/master_1h.parquet, master_2h.parquet
  reports/data_quality_hourly.csv       (per-coin check counts, coverage, gaps)
  reports/hourly_crosscheck.csv         (hourly-derived daily close vs the daily dataset)
  reports/usdt_peg.csv                  (days USDT was off its peg; check ends where Coin Metrics ends)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...common.logger import get_logger
from ...data.config import HourlyConfig, load_config
from ...data.quality.cleaning import clean_coin, gap_runs
from ...data.quality.hourly_checks import HourlyCheckParams, resample_2h, run_hourly_checks
from ...data.schema import QC
from ...data.sources import CoinMetricsSource, YahooSource
from ...data.storage import write_csv, write_parquet
from ...data.universe_hourly import HOUR, _ts
from .hourly_fetch import hourly_universe

logger = get_logger(__name__)


def daily_reference(coins: list[str], ids: dict[str, str], cfg: HourlyConfig) -> tuple[pd.DataFrame, dict[str, str]]:
    """Daily USD close per coin from the daily dataset, else a verified Yahoo snapshot, else Coin Metrics."""
    dcfg = load_config()
    master_path = dcfg.interim_dir / "master.parquet"
    if not master_path.exists():
        raise FileNotFoundError(f"{master_path} is missing: the hourly clean stage cross-checks against the daily data; "
                                "run `cryptograph-data --stage all` (daily config) first")
    master = pd.read_parquet(master_path)["close"].unstack("coin")
    audit_v2 = pd.read_csv(cfg.reports_dir / "universe_audit_v2.csv").set_index("coin")
    audit_v1 = pd.read_csv(cfg.path(cfg.paths.daily_audit)).set_index("coin")
    ref, src = {}, {}
    for c in coins:
        if c in master.columns and master[c].notna().any():
            ref[c], src[c] = master[c], "daily dataset (Yahoo + Coin Metrics)"
            continue
        yt = audit_v2["yahoo_ticker"].get(c) if c in audit_v2.index else None
        verified_v1 = c in audit_v1.index and bool(audit_v1.loc[c, "ticker_verified"])
        p = dcfg.raw_dir / "yahoo" / f"{c}.csv"
        if p.exists() and (isinstance(yt, str) or verified_v1):
            ref[c], src[c] = YahooSource.load(p)["close"], f"Yahoo {yt if isinstance(yt, str) else audit_v1.loc[c, 'yahoo_ticker']}"
            continue
        cm_asset = audit_v1["coinmetrics_asset"].get(c) if c in audit_v1.index else None
        cp = dcfg.raw_dir / "coinmetrics" / f"{cm_asset}.csv"
        if isinstance(cm_asset, str) and cp.exists():
            ref[c], src[c] = CoinMetricsSource.load(cp)["cm_price"], f"Coin Metrics {cm_asset}"
        else:
            src[c] = "none"
    return pd.DataFrame(ref), src


def run(cfg: HourlyConfig, refresh: bool = False) -> None:
    q = cfg.quality
    raw = pd.read_parquet(cfg.interim_dir / "raw_1h.parquet")
    coins, hist, ids = hourly_universe(cfg)
    events = pd.read_csv(cfg.external_dir / "events.csv", parse_dates=["date"])
    ref, ref_src = daily_reference(coins, ids, cfg)
    ref = ref.where(ref > 0)                       # a zero daily price must never "confirm" an hourly move
    ref_ret = np.log(ref).diff()
    w0, w1 = _ts(cfg.window.start), _ts(cfg.window.end)

    # USDT peg (daily, Coin Metrics); the check ends where the archive ends
    usdt = CoinMetricsSource.load(cfg.raw_dir / "coinmetrics" / "usdt.csv")["cm_price"].loc[w0.normalize(): w1]
    peg_end = usdt.dropna().index.max()
    depeg_days = usdt[(usdt - 1).abs() > q.usdt_peg_tol]
    write_csv(pd.DataFrame({"date": depeg_days.index.date, "usdt_usd": depeg_days.values}),
              cfg.reports_dir / "usdt_peg.csv")

    params = HourlyCheckParams(q.stale_hours, q.volume_spike_mult, q.volume_spike_window_hours, q.abs_logret_flag,
                               q.daily_confirm_logret, q.event_window_days)
    boundaries = pd.read_csv(cfg.reports_dir / "binance_continuity.csv") if (cfg.reports_dir / "binance_continuity.csv").exists() else pd.DataFrame()

    out1, out2, rep_rows, xc_rows = [], [], [], []
    for coin in coins:
        g = raw.xs(coin, level="coin").sort_index()
        first, last = g.index.min(), g.index.max()
        cal = pd.date_range(max(first, w0), min(last, w1), freq="h", name="time")
        h = g.reindex(cal)
        checked = run_hourly_checks(h, ref_ret[coin] if coin in ref_ret else None, list(events["date"]), params)
        # ticker-migration boundaries
        if len(boundaries):
            for _, b in boundaries[(boundaries["coin"] == coin) & (boundaries["type"] == "symbol_boundary")].iterrows():
                t = pd.Timestamp(b["new_first_hour"])
                if t in checked.index:
                    checked.loc[t, "qc_flags"] |= int(QC.REDENOMINATION)
        # USDT peg flag
        day = checked.index.normalize()
        depeg = day.isin(depeg_days.index)
        checked.loc[depeg, "qc_flags"] = checked.loc[depeg, "qc_flags"] | int(QC.USDT_DEPEG)
        checked["usdt_peg_checked"] = day <= peg_end
        checked["close_source"] = np.where(checked["close"].notna(), "binance", None)
        cleaned = clean_coin(checked, q.max_ffill_hours)
        cleaned["coin"] = coin
        cleaned["symbol"] = cleaned["symbol"].ffill()
        out1.append(cleaned.reset_index())
        two = resample_2h(cleaned)
        two["coin"] = coin
        out2.append(two.reset_index())

        # --- per-coin report ---
        f = cleaned["qc_flags"]
        miss = cleaned["close"].isna()
        runs = gap_runs(pd.Series(checked["close"].isna().to_numpy(), index=cleaned.index))
        t0, t1 = pd.Timestamp(cfg.splits.test[0]), pd.Timestamp(cfg.splits.test[1]) + pd.Timedelta(days=1) - HOUR
        win_hours = len(pd.date_range(w0, w1, freq="h"))
        test_hours = len(pd.date_range(t0, t1, freq="h"))
        row = {"coin": coin, "symbols": " -> ".join(o.symbol for o in hist[coin]), "first_hour": first, "last_hour": last,
               "active_hours": int((~miss).sum()), "window_coverage": float((~miss).sum() / win_hours),
               "test_coverage": float((~miss).loc[t0:t1].sum() / test_hours),
               "missing_hours_raw": int(checked["close"].isna().sum()),
               "n_gaps": int((runs.diff() > 0).sum() if len(runs) else 0),
               "longest_gap_hours": int(runs.max() if len(runs) else 0)}
        for flag in QC:
            if flag is not QC.NONE:
                row[flag.name.lower()] = int(((f & int(flag)) > 0).sum())
        rep_rows.append(row)

        # --- daily cross-check (hourly close of the 23:00 bar vs daily USD close) ---
        if coin in ref:
            hc = cleaned["close"].where(~cleaned["is_filled"])
            d = hc[hc.index.hour == 23]
            d.index = d.index.normalize()
            both = pd.concat([d.rename("hourly"), ref[coin].rename("daily")], axis=1).dropna()
            both = both[(both["daily"] > 0)]
            rel = (both["hourly"] / both["daily"] - 1).abs()
            xc_rows.append({"coin": coin, "reference": ref_src[coin], "days_compared": len(rel),
                            "median_rel_diff": float(rel.median()) if len(rel) else np.nan,
                            "share_gt_1pct": float((rel > q.crosscheck_tol).mean()) if len(rel) else np.nan,
                            "share_gt_5pct": float((rel > 0.05).mean()) if len(rel) else np.nan,
                            "worst_day": rel.idxmax().date() if len(rel) else None,
                            "worst_rel_diff": float(rel.max()) if len(rel) else np.nan})
        logger.info(f"{coin}: {row['active_hours']} active hours, test coverage {row['test_coverage']:.1%}")

    m1 = pd.concat(out1, ignore_index=True).set_index(["time", "coin"]).sort_index()
    m2 = pd.concat(out2, ignore_index=True).set_index(["time", "coin"]).sort_index()
    m1["is_event_coin"] = m1.index.get_level_values("coin").isin(list(cfg.universe.event_coins))
    m2["is_event_coin"] = m2.index.get_level_values("coin").isin(list(cfg.universe.event_coins))
    write_parquet(m1, cfg.interim_dir / "master_1h.parquet")
    write_parquet(m2, cfg.interim_dir / "master_2h.parquet")

    rep = pd.DataFrame(rep_rows)
    total = rep.drop(columns=["coin", "symbols", "first_hour", "last_hour"]).sum(numeric_only=True)
    total["window_coverage"] = rep["window_coverage"].mean()
    total["test_coverage"] = rep["test_coverage"].mean()
    total["longest_gap_hours"] = rep["longest_gap_hours"].max()
    total["coin"] = "ALL"
    rep = pd.concat([rep, total.to_frame().T], ignore_index=True)
    write_csv(rep, cfg.reports_dir / "data_quality_hourly.csv")
    xc = pd.DataFrame(xc_rows)
    write_csv(xc, cfg.reports_dir / "hourly_crosscheck.csv")

    pd.set_option("display.width", 250, "display.max_columns", 40)
    print(f"USDT peg: {len(depeg_days)} days off peg by > {q.usdt_peg_tol:.1%} "
          f"(check covers {usdt.index.min().date()} .. {peg_end.date()}, where the Coin Metrics archive ends)")
    cov = rep[rep["coin"] != "ALL"][["coin", "symbols", "first_hour", "last_hour", "active_hours", "window_coverage",
                                     "test_coverage", "n_gaps", "longest_gap_hours", "filled", "long_gap"]]
    print(cov.to_string(index=False))
    qcols = ["large_return", "large_return_confirmed", "large_return_unverified", "bad_tick_corrected", "stale_close",
             "volume_spike", "zero_volume", "high_lt_low", "oc_outside_hl", "nonpositive_price", "missing_day",
             "filled", "long_gap", "usdt_depeg", "redenomination"]
    print("\nQuality totals:", rep[rep["coin"] == "ALL"][qcols].iloc[0].astype(int).to_dict())
    print("\nDaily cross-check (hourly 23:00 close vs daily USD close):")
    print(xc.to_string(index=False))
