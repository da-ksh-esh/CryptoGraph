"""Universe selection and availability audit (Stage 1).

Pipeline:
  1. CoinGecko ranking snapshot (current market caps) -> non-stable ranking.
  2. For each candidate: resolve / verify the Yahoo ticker, snapshot Yahoo and
     Coin Metrics raw files, measure availability per source.
  3. Define Panel A (balanced, full history) and Panel B (current top-N,
     unbalanced), and quantify alternatives when Panel A cannot be filled.

All measurement helpers are pure functions on pandas objects (unit-tested);
`run_audit` is the only function doing I/O.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from ..common.logger import get_logger
from .config import Candidate, DataConfig
from .sources import CoinGeckoRanking, CoinMetricsSource, YahooSource
from .storage import write_csv, write_json

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# pure helpers
# ---------------------------------------------------------------------------
def calendar(start: date, end: date) -> pd.DatetimeIndex:
    return pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="D", name="date")


def availability_stats(series: pd.Series, start: date, end: date) -> dict:
    """Coverage and gap statistics of a daily series over the inclusive window.

    A day counts as valid when the value is finite and > 0. Gaps are runs of
    invalid days strictly between the first and last valid day.
    """
    cal = calendar(start, end)
    s = series.reindex(cal)
    valid = np.isfinite(s.to_numpy(dtype=float)) & (s.to_numpy(dtype=float) > 0)
    n = len(cal)
    if not valid.any():
        return {"first_valid": None, "last_valid": None, "coverage": 0.0, "n_valid": 0,
                "n_gaps": 0, "longest_gap": 0}
    idx = np.flatnonzero(valid)
    first, last = idx[0], idx[-1]
    inner = ~valid[first:last + 1]
    # run-length encode the missing runs inside [first, last]
    n_gaps, longest, run = 0, 0, 0
    for miss in inner:
        if miss:
            run += 1
        else:
            if run:
                n_gaps += 1
                longest = max(longest, run)
            run = 0
    return {
        "first_valid": cal[first].date().isoformat(),
        "last_valid": cal[last].date().isoformat(),
        "coverage": float(valid.sum() / n),
        "n_valid": int(valid.sum()),
        "n_gaps": int(n_gaps),
        "longest_gap": int(longest),
    }


def combined_close(yahoo_close: pd.Series | None, cm_price: pd.Series | None) -> pd.Series:
    """Yahoo close where present, else Coin Metrics price (candidate backfill, pre-validation)."""
    parts = [s for s in (yahoo_close, cm_price) if s is not None and len(s)]
    if not parts:
        return pd.Series(dtype=float)
    out = parts[0].copy()
    for p in parts[1:]:
        out = out.combine_first(p)
    return out.where(out > 0)


def earliest_start_for_n(valid: pd.DataFrame, n: int, min_coverage: float, end: date) -> tuple[date | None, list[str]]:
    """Earliest month-start date s such that >= n coins cover [s, end] with >= min_coverage.

    `valid` is a boolean frame (calendar x coin), already restricted to eligible coins.
    """
    for s in pd.date_range(valid.index[0], pd.Timestamp(end), freq="MS"):
        cov = valid.loc[s:].mean()
        ok = cov[cov >= min_coverage].index.tolist()
        if len(ok) >= n:
            return s.date(), ok
    return None, []


@dataclass
class ResolvedTicker:
    ticker: str | None
    yahoo_name: str | None
    verified: bool
    note: str


def verify_identity(yahoo_last_close: float | None, reference_price: float | None, tol: float) -> bool:
    if yahoo_last_close is None or reference_price is None or not reference_price > 0:
        return False
    return abs(yahoo_last_close / reference_price - 1.0) <= tol


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------
class UniverseAuditor:
    def __init__(self, cfg: DataConfig, refresh: bool = False):
        self.cfg = cfg
        self.refresh = refresh
        w, s = cfg.window, cfg.sources
        self.yahoo = YahooSource(w.start_date, w.end_date, retries=s.retries, backoff=s.backoff_base_seconds)
        self.cm = CoinMetricsSource(s.coinmetrics_url, retries=s.retries, backoff=s.backoff_base_seconds)
        self.cg = CoinGeckoRanking(s.coingecko_markets_url, retries=s.retries, backoff=s.backoff_base_seconds)
        self.dirs = {k: cfg.raw_dir / k for k in ("yahoo", "coinmetrics", "coingecko")}

    # -- ranking ------------------------------------------------------------------
    def ranking(self) -> pd.DataFrame:
        ids = [c.coingecko for c in self.cfg.universe.candidates]
        rk = self.cg.load(self.cg.snapshot(self.dirs["coingecko"], ids=ids, refresh=self.refresh))
        rk["market_cap"] = pd.to_numeric(rk["market_cap"], errors="coerce").fillna(0.0)
        excluded = self.cfg.universe.excluded_symbols
        rk["excluded"] = rk["symbol"].str.lower().isin(excluded) | rk["coingecko_id"].str.lower().isin(excluded)
        rk = rk.sort_values("market_cap", ascending=False).reset_index(drop=True)
        rk["nonstable_rank"] = np.where(rk["excluded"], np.nan, (~rk["excluded"]).cumsum())
        return rk

    # -- ticker resolution --------------------------------------------------------
    def resolve(self, cand: Candidate, ref_price: float | None, yahoo_df: pd.DataFrame | None,
                names: tuple[str, ...] = ()) -> ResolvedTicker:
        """Configured ticker, else Yahoo search by symbol and by name.

        Only exact symbol forms `COIN-USD` / `COIN<digits>-USD` are considered
        (e.g. WBT must not resolve to WBTC-USD), then verified by price level.
        """
        tol = self.cfg.sources.identity_price_tolerance
        tried: list[str] = []
        options = [cand.yahoo] if cand.yahoo else []
        if not cand.yahoo:
            symbols = {cand.coin.upper(), *(n.upper() for n in names if n and n.isalnum() and len(n) <= 8)}
            pattern = re.compile(rf"^({'|'.join(map(re.escape, symbols))})\d*-USD$")
            for query in (cand.coin, *names):
                options += [q["symbol"] for q in self.yahoo.search(query) if pattern.match(str(q["symbol"]).upper())]
        for tk in dict.fromkeys(options):  # de-duplicate, keep order
            tried.append(tk)
            try:
                df = yahoo_df if (yahoo_df is not None and tk == cand.yahoo) else self.yahoo.fetch(tk)
            except RuntimeError:
                continue
            last = float(df["close"].dropna().iloc[-1]) if len(df.get("close", pd.Series(dtype=float)).dropna()) else None
            if ref_price is None:
                return ResolvedTicker(tk, None, last is not None, "no reference price; accepted on data presence")
            if verify_identity(last, ref_price, tol):
                return ResolvedTicker(tk, None, True, f"last close {last:.6g} vs CoinGecko {ref_price:.6g}")
        return ResolvedTicker(options[0] if options else None, None, False,
                              f"no Yahoo ticker matched CoinGecko price {ref_price}; tried {tried}")

    # -- per-candidate audit ---------------------------------------------------------
    def audit_candidate(self, cand: Candidate, rank_row: pd.Series | None) -> dict:
        w = self.cfg.window
        ref_price = float(rank_row["current_price"]) if rank_row is not None else None
        row: dict = {
            "coin": cand.coin, "category": cand.category, "coingecko_id": cand.coingecko,
            "mcap_rank_all": int(rank_row["market_cap_rank"]) if rank_row is not None and pd.notna(rank_row["market_cap_rank"]) else None,
            "nonstable_rank": int(rank_row["nonstable_rank"]) if rank_row is not None and pd.notna(rank_row["nonstable_rank"]) else None,
            "market_cap_usd": float(rank_row["market_cap"]) if rank_row is not None else np.nan,
            "mcap_source": "coingecko_snapshot" if rank_row is not None else None,
        }

        # Yahoo: fetch configured ticker once (reused by identity check)
        yahoo_df = None
        if cand.yahoo:
            path = self.yahoo.snapshot(cand.coin, cand.yahoo, self.dirs["yahoo"], refresh=self.refresh)
            yahoo_df = self.yahoo.load(path) if path else pd.DataFrame(columns=["close", "volume"])
        names = (str(rank_row["symbol"]), str(rank_row["name"])) if rank_row is not None else ()
        res = self.resolve(cand, ref_price, yahoo_df, names)
        if res.ticker and res.ticker != cand.yahoo:
            # never keep a snapshot for an unverified (possibly wrong-coin) ticker
            path = self.dirs["yahoo"] / f"{cand.coin}.csv"
            if res.verified:
                path = self.yahoo.snapshot(cand.coin, res.ticker, self.dirs["yahoo"], refresh=True)
                yahoo_df = self.yahoo.load(path) if path else None
            else:
                path.unlink(missing_ok=True)
                yahoo_df = None
        row.update({"yahoo_ticker": res.ticker, "ticker_verified": res.verified, "ticker_note": res.note})

        y_close = yahoo_df["close"] if yahoo_df is not None and "close" in yahoo_df else None
        ystats = availability_stats(y_close if y_close is not None else pd.Series(dtype=float), w.start_date, w.end_date)
        row.update({f"yahoo_{k}": v for k, v in ystats.items()})
        if yahoo_df is not None and "volume" in yahoo_df and len(yahoo_df):
            vol = yahoo_df["volume"].where(yahoo_df["volume"] > 0)
            row["median_volume_usd"] = float(vol.median()) if vol.notna().any() else np.nan
            row["median_volume_usd_last365"] = float(vol.iloc[-365:].median()) if vol.notna().any() else np.nan
        else:
            row["median_volume_usd"] = row["median_volume_usd_last365"] = np.nan

        # Coin Metrics: first asset id that exists
        cm_df, cm_asset = None, None
        for asset in cand.coinmetrics:
            path = self.cm.snapshot(asset, self.dirs["coinmetrics"], refresh=self.refresh)
            if path is not None:
                cm_df, cm_asset = self.cm.load(path), asset
                break
        cm_price = cm_df["cm_price"] if cm_df is not None else None
        cstats = availability_stats(cm_price if cm_price is not None else pd.Series(dtype=float), w.start_date, w.end_date)
        row["coinmetrics_asset"] = cm_asset
        row.update({f"cm_{k}": v for k, v in cstats.items()})
        row["cm_archive_last_date"] = cm_df.index.max().date().isoformat() if cm_df is not None and len(cm_df) else None
        if cm_df is not None and cm_df["cm_mcap"].notna().any():
            ref = cm_df["cm_mcap"].loc[: pd.Timestamp(w.reference_date)].dropna()
            row["cm_market_cap_at_ref"] = float(ref.iloc[-1]) if len(ref) else np.nan
            row["cm_market_cap_date"] = ref.index[-1].date().isoformat() if len(ref) else None
        else:
            row["cm_market_cap_at_ref"], row["cm_market_cap_date"] = np.nan, None

        # combined close availability (Yahoo, backfilled by CM where Yahoo missing)
        comb = combined_close(y_close, cm_price)
        kstats = availability_stats(comb, w.start_date, w.end_date)
        row.update({f"combined_{k}": v for k, v in kstats.items()})
        a_start = self.cfg.universe.panel_a_start_date or w.start_date
        row["panelA_window_coverage"] = availability_stats(comb, a_start, w.end_date)["coverage"]
        test_start = pd.Timestamp(self.cfg.splits.test[0])
        row["active_days_before_test"] = int((comb.loc[: test_start - pd.Timedelta(days=1)] > 0).sum()) if len(comb) else 0

        # cross-source price-level agreement on overlap (identity sanity check, not the full reconciliation)
        if y_close is not None and cm_price is not None:
            both = pd.concat([y_close.rename("y"), cm_price.rename("c")], axis=1).dropna()
            both = both[(both["y"] > 0) & (both["c"] > 0)]
            if len(both):
                rel = (both["y"] / both["c"] - 1).abs()
                row["xsrc_overlap_days"] = int(len(both))
                row["xsrc_median_abs_diff"] = float(rel.median())
            else:
                row["xsrc_overlap_days"], row["xsrc_median_abs_diff"] = 0, np.nan
        else:
            row["xsrc_overlap_days"], row["xsrc_median_abs_diff"] = 0, np.nan
        return row

    # -- panels -----------------------------------------------------------------------
    def build_panels(self, audit: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        q, u = self.cfg.quality, self.cfg.universe
        n = u.panel_size
        a = audit.copy()
        a["has_data"] = a["ticker_verified"].fillna(False) & a["yahoo_n_valid"].fillna(0).gt(0)
        a["liquid"] = a["median_volume_usd"].fillna(0) >= q.min_median_volume_usd
        a["full_history"] = a["panelA_window_coverage"].fillna(0) >= q.min_coverage_panel_a
        a["train_history"] = a["active_days_before_test"].fillna(0) >= u.panel_b_min_train_days
        is_event = a["coin"].isin(u.event_coins)
        ranked = a[~is_event & a["nonstable_rank"].notna()].sort_values("market_cap_usd", ascending=False)

        eligible_a = ranked[ranked["has_data"] & ranked["liquid"] & ranked["full_history"]]
        panel_a = eligible_a.head(n)["coin"].tolist()
        pool_b = ranked[ranked["has_data"] & ranked["liquid"] & ranked["train_history"]]
        panel_b = pool_b.head(n)["coin"].tolist()
        # why each top-ranked coin was skipped for Panel B (reported at the checkpoint)
        skipped = ranked[~ranked["coin"].isin(panel_b)]
        skipped = skipped[skipped["market_cap_usd"] >= pool_b.head(n)["market_cap_usd"].min()]
        a["panel_b_skip_reason"] = ""
        for idx, r in skipped.iterrows():
            reasons = [k for k, ok in (("no verified data", r["has_data"]), ("illiquid", r["liquid"]),
                                       (f"<{u.panel_b_min_train_days} train days", r["train_history"])) if not ok]
            a.loc[idx, "panel_b_skip_reason"] = "; ".join(reasons)
        if u.include_event_coins_in_panel_b:
            panel_b += [c for c in u.event_coins if c in a.loc[a["has_data"], "coin"].tolist()]

        a["panel_A"] = a["coin"].isin(panel_a)
        a["panel_B"] = a["coin"].isin(panel_b)
        a["is_event_coin"] = is_event

        # alternatives when Panel A is short
        liquid = ranked[ranked["has_data"] & ranked["liquid"]]
        options: dict = {"panel_a_size": len(panel_a), "panel_a_target": n}
        if len(panel_a) < n:
            w = self.cfg.window
            cal = calendar(w.start_date, w.end_date)
            valid = pd.DataFrame(index=cal)
            for _, r in liquid.iterrows():
                fv = r["combined_first_valid"]
                valid[r["coin"]] = cal >= pd.Timestamp(fv) if fv else False
            # (a) later start date (approximation: coverage from first valid date; gaps are rare, see audit)
            s_date, coins_a = earliest_start_for_n(valid, n, q.min_coverage_panel_a, w.end_date)
            options["a_later_start"] = {"start_date": s_date.isoformat() if s_date else None,
                                        "n_days": int((pd.Timestamp(w.end_date) - pd.Timestamp(s_date)).days + 1) if s_date else None,
                                        "coins": liquid[liquid["coin"].isin(coins_a)].head(n)["coin"].tolist()}
            # (b) relaxed coverage: coverage of the n-th best liquid coin
            cov_sorted = liquid.sort_values("combined_coverage", ascending=False)
            if len(cov_sorted) >= n:
                nth = cov_sorted.iloc[n - 1]
                top = cov_sorted.head(n)
                options["b_relaxed_coverage"] = {"threshold_needed": round(float(nth["combined_coverage"]), 4),
                                                 "coins": top["coin"].tolist(),
                                                 "limiting_coin": nth["coin"]}
            # (c) smaller balanced panel
            options["c_smaller_panel"] = {"size": len(panel_a), "coins": panel_a}
            # number of liquid coins with >= min coverage for a range of start dates
            grid = ["2016-09-28", "2017-11-09", "2018-01-01", "2018-07-01", "2019-01-01", "2019-04-01",
                    "2019-10-01", "2020-01-01"]
            options["eligible_by_start"] = {
                s: int((valid.loc[pd.Timestamp(s):].mean() >= q.min_coverage_panel_a).sum()) for s in grid}
        return a, {"panel_A": panel_a, "panel_B": panel_b, "options": options}

    # -- entry point --------------------------------------------------------------------
    def run(self) -> tuple[pd.DataFrame, dict]:
        rk = self.ranking()
        by_id = rk.set_index("coingecko_id")
        rows = []
        for cand in self.cfg.universe.candidates:
            logger.info(f"auditing {cand.coin}")
            rank_row = by_id.loc[cand.coingecko] if cand.coingecko in by_id.index else None
            rows.append(self.audit_candidate(cand, rank_row))
        audit = pd.DataFrame(rows)
        audit, panels = self.build_panels(audit)

        # top non-stable coins that are NOT in the candidate list (coverage gap of the config)
        cand_ids = {c.coingecko for c in self.cfg.universe.candidates}
        top_ns = rk[rk["nonstable_rank"].notna() & (rk["nonstable_rank"] <= self.cfg.universe.panel_size + 10)]
        panels["uncovered_top_nonstable"] = top_ns[~top_ns["coingecko_id"].isin(cand_ids)][
            ["coingecko_id", "symbol", "nonstable_rank", "market_cap"]].to_dict("records")
        panels["ranking_fetched"] = rk["last_updated"].max()
        panels["excluded_in_top50"] = rk.head(50).loc[rk.head(50)["excluded"], "symbol"].tolist()

        audit = audit.sort_values(["panel_B", "market_cap_usd"], ascending=[False, False])
        write_csv(audit, self.cfg.reports_dir / "universe_audit.csv")
        write_json(panels, self.cfg.interim_dir / "universe_panels.json")
        return audit, panels


def format_summary(audit: pd.DataFrame, panels: dict, cfg: DataConfig) -> str:
    q = cfg.quality
    lines = [
        f"Window {cfg.window.start_date} .. {cfg.window.end_date} "
        f"({len(calendar(cfg.window.start_date, cfg.window.end_date))} days); ranking snapshot {panels.get('ranking_fetched')}",
        "",
        f"{'coin':6s} {'rk':>3s} {'mcap$B':>8s} {'yahoo_tkr':14s} {'ok':>2s} {'Y_first':10s} {'Y_cov':>6s} "
        f"{'CM_first':10s} {'CM_cov':>6s} {'comb_cov':>8s} {'gaps':>4s} {'maxgap':>6s} {'medVol$M':>9s} A B",
    ]
    for _, r in audit.iterrows():
        mc = r["market_cap_usd"] / 1e9 if pd.notna(r["market_cap_usd"]) else float("nan")
        mv = r["median_volume_usd"] / 1e6 if pd.notna(r["median_volume_usd"]) else float("nan")
        rk = "" if pd.isna(r["nonstable_rank"]) else str(int(r["nonstable_rank"]))
        lines.append(
            f"{r['coin']:6s} {rk:>3s} {mc:8.2f} {str(r['yahoo_ticker']):14s} {'Y' if r['ticker_verified'] else 'N':>2s} "
            f"{str(r['yahoo_first_valid']):10s} {r['yahoo_coverage']:6.1%} {str(r['cm_first_valid']):10s} "
            f"{r['cm_coverage']:6.1%} {r['combined_coverage']:8.1%} {r['combined_n_gaps']:4d} "
            f"{r['combined_longest_gap']:6d} {mv:9.1f} {'A' if r['panel_A'] else '.'} {'B' if r['panel_B'] else '.'}"
        )
    skipped = audit[audit["panel_b_skip_reason"].fillna("") != ""]
    if len(skipped):
        lines += ["", "Skipped for Panel B despite rank: "
                  + ", ".join(f"{r.coin} ({r.panel_b_skip_reason})" for r in skipped.itertuples())]
    a_start = cfg.universe.panel_a_start_date or cfg.window.start_date
    lines += ["", f"Panel A window {a_start} .. {cfg.window.end_date}",
              f"Panel A ({len(panels['panel_A'])}/{cfg.universe.panel_size}, coverage >= {q.min_coverage_panel_a:.0%}, "
              f"median vol >= ${q.min_median_volume_usd/1e6:.0f}M): {panels['panel_A']}",
              f"Panel B ({len(panels['panel_B'])}): {panels['panel_B']}"]
    if panels.get("uncovered_top_nonstable"):
        lines.append(f"Top non-stable coins not in candidate list: "
                     f"{[(d['symbol'], int(d['nonstable_rank'])) for d in panels['uncovered_top_nonstable']]}")
    lines.append(f"Excluded (stable/wrapped/RWA) in CoinGecko top 50: {panels.get('excluded_in_top50')}")
    return "\n".join(lines)
