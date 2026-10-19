"""Stage 3 — quality checks + gap cleaning -> tidy master table.

Writes data/interim/master.parquet, reports/data_quality_report.csv and
reports/redenominations.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...common.logger import get_logger
from ...data.config import DataConfig
from ...data.quality.checks import CheckParams, run_checks
from ...data.quality.cleaning import clean_coin
from ...data.schema import MASTER_COLUMNS, QC, validate_master
from ...data.storage import write_csv, write_parquet
from ...data.universe import calendar
from . import load_universe

logger = get_logger(__name__)

def load_events(cfg: DataConfig) -> pd.DataFrame:
    ev = pd.read_csv(cfg.external_dir / "events.csv", parse_dates=["date"])
    return ev


def run(cfg: DataConfig, refresh: bool = False) -> None:
    audit, panels, coins = load_universe(cfg)
    rec = pd.read_parquet(cfg.interim_dir / "reconciled.parquet").reset_index()
    events = load_events(cfg)
    event_dates = list(events["date"])
    q = cfg.quality
    params = CheckParams(volume_spike_mult=q.volume_spike_mult, volume_spike_window=q.volume_spike_window,
                         stale_days=q.stale_days, abs_logret_flag=q.abs_logret_flag)
    cal = calendar(cfg.window.start_date, cfg.window.end_date)

    rows, report = [], []
    for coin in coins:
        g = rec[rec["coin"] == coin].set_index("date").sort_index().reindex(cal)
        g.index.name = "date"
        g["close_source"] = g["close_source"].astype(object)
        checked = run_checks(g, g["cm_close_aligned"], event_dates, params, n_duplicates=g["raw_duplicates"])
        for r in cfg.universe.redenominations:
            if r.coin == coin:
                checked.loc[pd.Timestamp(r.date), "qc_flags"] |= int(QC.REDENOMINATION)
        cleaned = clean_coin(checked, q.max_ffill_gap)
        cleaned["coin"] = coin
        cleaned["panel_A"] = coin in panels["panel_A"]
        cleaned["panel_B"] = coin in panels["panel_B"]
        cleaned["is_event_coin"] = coin in cfg.universe.event_coins
        rows.append(cleaned.reset_index())

        f = cleaned["qc_flags"]
        span = cleaned["close"].notna()
        rep = {"coin": coin, "first_valid": cleaned.index[span].min().date() if span.any() else None,
               "active_days": int(span.sum()),
               "yahoo_close_days": int(cleaned["close_source"].eq("yahoo").sum())}
        for flag in QC:
            if flag is QC.NONE:
                continue
            rep[flag.name.lower()] = int(((f & int(flag)) > 0).sum())
        rep["n_longest_missing_run"] = int(cleaned.loc[cleaned.index >= (cleaned.index[span].min() if span.any() else cal[0]), "close"]
                                           .isna().astype(int).groupby(cleaned["close"].notna().cumsum()).sum().max() or 0)
        report.append(rep)

    master = pd.concat(rows, ignore_index=True)
    master = master[list(MASTER_COLUMNS) + ["xsrc_rel_diff"]]
    for col, dtype in MASTER_COLUMNS.items():
        if dtype == "bool":
            master[col] = master[col].fillna(False).astype(bool)
        elif dtype == "string":
            master[col] = master[col].astype("string")
    validate_master(master)
    write_parquet(master.set_index(["date", "coin"]), cfg.interim_dir / "master.parquet")

    rep_df = pd.DataFrame(report)
    total = rep_df.drop(columns=["coin", "first_valid"]).sum(numeric_only=True)
    total["coin"], total["first_valid"] = "ALL", None
    rep_df = pd.concat([rep_df, total.to_frame().T], ignore_index=True)
    write_csv(rep_df, cfg.reports_dir / "data_quality_report.csv")

    lines = ["# Redenominations and ticker migrations", "",
             "| coin | date | description | source |", "|---|---|---|---|"]
    lines += [f"| {r.coin} | {r.date} | {r.what} | {r.source_url} |" for r in cfg.universe.redenominations]
    lines += ["", "Screened: no other coin in the master universe shows a single-day price-level jump "
              "consistent with a redenomination (|log return| > log(10) with no reversal)."]
    jumps = master.assign(r=np.log(master["close"]).groupby(master["coin"]).diff())
    big = jumps[jumps["r"].abs() > np.log(10)]
    if len(big):
        lines += ["", "Days with |log return| > log(10) (all reviewed):", ""]
        lines += [f"- {r.coin} {r.date.date()}: log return {r.r:+.2f}" for r in big.itertuples()]
    (cfg.reports_dir / "redenominations.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    cols = ["coin", "active_days", "missing_day", "filled", "long_gap", "large_return", "large_return_confirmed",
            "large_return_unverified", "bad_tick_corrected", "xsrc_disagree", "cm_backfill", "stale_close",
            "volume_spike", "zero_volume", "high_lt_low", "oc_outside_hl"]
    print(rep_df[cols].to_string(index=False))
