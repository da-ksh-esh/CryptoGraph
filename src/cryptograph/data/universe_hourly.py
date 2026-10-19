"""Stage 1 (hourly): 30-coin model universe from Binance USDT spot markets.

For every candidate (the top-N non-stable coins by market cap at the reference
date, plus the event coins) it:
  1. maps the coin to its Binance symbol history (config overrides for renames),
  2. lists the archive's monthly/daily kline files for each symbol,
  3. downloads (checksum-verified) the first and last file of each symbol segment
     to pin the exact first/last hour and the last close,
  4. estimates hourly coverage from the listed months/days, clipped to those exact
     endpoints and to each segment's validity (maintenance gaps inside a month are
     measured in Stage 2),
  5. verifies identity (Binance last close vs the CoinGecko reference price),
  6. applies the rules: >= min_history_days of data in the window and
     >= min_test_coverage of test-period hours.
The first `target_size` coins that pass, in market-cap order, become model coins.

Coverage helpers are pure functions (unit-tested); `HourlyUniverseAuditor.run`
does the I/O.
"""

from __future__ import annotations

import calendar as _cal
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ..common.logger import get_logger
from .config import HourlyConfig, SymbolOverride
from .sources.binance import BinanceArchive, SymbolFiles, read_klines
from .storage import write_csv, write_json

logger = get_logger(__name__)
HOUR = pd.Timedelta(hours=1)


# ---------------------------------------------------------------------------
# pure helpers
# ---------------------------------------------------------------------------
def _ts(x) -> pd.Timestamp:
    t = pd.Timestamp(x)
    return t.tz_convert("UTC").tz_localize(None) if t.tzinfo is not None else t


def file_intervals(files: SymbolFiles) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """[start, end) intervals covered by the listed monthly and daily files."""
    out = []
    for ym in files.months:
        y, m = map(int, ym.split("-"))
        start = pd.Timestamp(y, m, 1)
        out.append((start, start + pd.Timedelta(days=_cal.monthrange(y, m)[1])))
    for d in files.days:
        start = pd.Timestamp(d)
        out.append((start, start + pd.Timedelta(days=1)))
    return out


def merge_intervals(intervals: list[tuple[pd.Timestamp, pd.Timestamp]]) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    merged: list[list[pd.Timestamp]] = []
    for s, e in sorted(i for i in intervals if i[1] > i[0]):
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def clip_intervals(intervals, lo: pd.Timestamp, hi: pd.Timestamp):
    """Intersect [start, end) intervals with [lo, hi)."""
    return [(max(s, lo), min(e, hi)) for s, e in intervals if min(e, hi) > max(s, lo)]


def hours_in(intervals, lo: pd.Timestamp, hi: pd.Timestamp) -> float:
    return sum((e - s) / HOUR for s, e in clip_intervals(intervals, lo, hi))


@dataclass
class Segment:
    symbol: str
    files: SymbolFiles
    first_open: pd.Timestamp | None = None     # exact first hour in the segment's first file
    last_open: pd.Timestamp | None = None      # exact last hour in the segment's last file
    last_close: float | None = None
    valid_until: date | None = None            # config: symbol means this coin only up to this date

    def intervals(self) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
        iv = merge_intervals(file_intervals(self.files))
        lo = self.first_open if self.first_open is not None else pd.Timestamp.min
        hi = self.last_open + HOUR if self.last_open is not None else pd.Timestamp.max
        if self.valid_until is not None:
            hi = min(hi, pd.Timestamp(self.valid_until) + pd.Timedelta(days=1))
        return clip_intervals(iv, lo, hi)


def coverage_summary(segments: list[Segment], window: tuple[pd.Timestamp, pd.Timestamp],
                     test: tuple[pd.Timestamp, pd.Timestamp]) -> dict:
    iv = merge_intervals([i for s in segments for i in s.intervals()])
    w_lo, w_hi = window
    t_lo, t_hi = test
    in_window = clip_intervals(iv, w_lo, w_hi)
    return {
        "hourly_first": in_window[0][0] if in_window else None,
        "hourly_last": in_window[-1][1] - HOUR if in_window else None,
        "history_days": hours_in(iv, w_lo, w_hi) / 24.0,
        "window_coverage": hours_in(iv, w_lo, w_hi) / ((w_hi - w_lo) / HOUR),
        "test_coverage": hours_in(iv, t_lo, t_hi) / ((t_hi - t_lo) / HOUR),
        "n_file_gaps": max(len(in_window) - 1, 0),
    }


def decide(row: dict, rules) -> tuple[bool, str]:
    """(passes, reason) for one audited candidate, in rule order."""
    # Order matters: coverage rules first (a delisted market's last price cannot be compared with
    # today's reference price), identity last.
    if not row.get("binance_symbol"):
        return False, f"no Binance {rules.quote_asset} spot market"
    first, last = row.get("hourly_first"), row.get("hourly_last")
    span = f"hourly data {first:%Y-%m-%d} → {last:%Y-%m-%d}" if first is not None and last is not None else "no data in window"
    if row["history_days"] < rules.min_history_days:
        return False, f"only {row['history_days']:.0f} days of hourly data (< {rules.min_history_days}; {span})"
    if row["test_coverage"] < rules.min_test_coverage:
        return False, f"test-period coverage {row['test_coverage']:.0%} < {rules.min_test_coverage:.0%} ({span})"
    if not row.get("identity_ok"):
        return False, (f"Binance symbol is a different asset (last close {row.get('binance_last_close')} "
                       f"vs reference {row.get('reference_price')})")
    return True, "passes"


# ---------------------------------------------------------------------------
# sectors (CoinGecko categories -> project taxonomy)
# ---------------------------------------------------------------------------
SECTOR_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("meme", ("meme",)),
    ("exchange token", ("exchange-based tokens", "centralized exchange (cex) token", "exchange token")),
    ("privacy", ("privacy coins", "privacy")),
    ("AI", ("artificial intelligence (ai)",)),
    ("DeFi / oracle", ("oracle", "decentralized exchange (dex)", "lending/borrowing", "decentralized finance (defi)")),
    ("smart-contract platform", ("smart contract platform",)),
    ("payments", ("payment solutions",)),
]


def map_sector(categories: list[str]) -> tuple[str | None, str]:
    """First matching project sector for CoinGecko categories, with the category that matched."""
    lower = [c.lower() for c in categories if c]
    for sector, keys in SECTOR_RULES:
        for c in lower:
            if any(k == c or (k == "meme" and "meme" in c) for k in keys):
                return sector, c
    return None, ""


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------
class HourlyUniverseAuditor:
    def __init__(self, cfg: HourlyConfig, refresh: bool = False):
        self.cfg, self.refresh = cfg, refresh
        b = cfg.binance
        self.archive = BinanceArchive(b.archive_url, b.list_url, b.market, b.interval, b.retries, b.backoff_base_seconds)
        self.raw_bin = cfg.raw_dir / "binance" / b.interval
        w = cfg.window
        self.window = (_ts(w.start), _ts(w.end) + HOUR)
        self.test = (pd.Timestamp(cfg.splits.test[0]), pd.Timestamp(cfg.splits.test[1]) + pd.Timedelta(days=1))
        self.current_month = self.window[1].strftime("%Y-%m") if self.window[1].day > 1 else None

    # -- ranking ----------------------------------------------------------------------
    def ranking(self) -> pd.DataFrame:
        rk = pd.DataFrame(json.loads(self.cfg.path(self.cfg.paths.coingecko_snapshot).read_text(encoding="utf-8")))
        rk = rk.sort_values("market_cap", ascending=False).reset_index(drop=True)
        rk["rank_all"] = np.arange(1, len(rk) + 1)
        ex = self.cfg.universe.excluded_symbols
        u = self.cfg.universe
        def why(r):
            s, i = str(r["symbol"]).lower(), str(r["id"]).lower()
            if s in {x.lower() for x in u.stablecoins} or i in {x.lower() for x in u.stablecoins}:
                return "stablecoin"
            if s in {x.lower() for x in u.wrapped_or_staked} or i in {x.lower() for x in u.wrapped_or_staked}:
                return "wrapped or staked token"
            if s in {x.lower() for x in u.rwa_or_pegged} or i in {x.lower() for x in u.rwa_or_pegged}:
                return "tokenised real-world asset / pegged"
            return ""
        rk["exclusion"] = rk.apply(why, axis=1)
        rk["excluded"] = rk["exclusion"] != ""
        del ex
        rk["nonstable_rank"] = np.where(rk["excluded"], np.nan, (~rk["excluded"]).cumsum())
        return rk

    # -- one coin -----------------------------------------------------------------------
    def symbol_history(self, coingecko_id: str, symbol: str) -> list[SymbolOverride]:
        ov = self.cfg.universe.symbol_overrides.get(coingecko_id)
        return ov if ov else [SymbolOverride(symbol=f"{symbol.upper()}{self.cfg.universe.quote_asset}")]

    def _segment(self, ov: SymbolOverride) -> Segment:
        daily_months = (self.current_month,) if self.current_month else ()
        files = self.archive.symbol_files(ov.symbol, daily_months)
        seg = Segment(ov.symbol, files, valid_until=ov.until)
        if not files.exists:
            return seg
        out_dir = self.raw_bin / ov.symbol
        first_key = (self.archive.file_key(ov.symbol, "monthly", files.months[0]) if files.months
                     else self.archive.file_key(ov.symbol, "daily", files.days[0]))
        first = read_klines(self.archive.download(first_key, out_dir, refresh=self.refresh))
        # last file inside the window (and inside the segment's validity for reused symbols):
        # newest daily file if present, else newest monthly file
        limit = self.window[1] - HOUR
        if ov.until is not None:
            limit = min(limit, pd.Timestamp(ov.until) + pd.Timedelta(days=1) - HOUR)
        days_ok = [d for d in files.days if pd.Timestamp(d) <= limit]
        months_ok = [m for m in files.months if pd.Timestamp(m + "-01") <= limit]
        if days_ok and (not months_ok or pd.Timestamp(days_ok[-1]) >= pd.Timestamp(months_ok[-1] + "-01")):
            last_key = self.archive.file_key(ov.symbol, "daily", days_ok[-1])
        elif months_ok:
            last_key = self.archive.file_key(ov.symbol, "monthly", months_ok[-1])
        else:
            last_key = first_key
        last = read_klines(self.archive.download(last_key, out_dir, refresh=self.refresh))
        if ov.until is not None:
            last = last.loc[: pd.Timestamp(ov.until) + pd.Timedelta(days=1) - HOUR]
        last = last.loc[: self.window[1] - HOUR]
        seg.first_open = first.index.min()
        seg.last_open = last.index.max() if len(last) else None
        seg.last_close = float(last["close"].iloc[-1]) if len(last) else None
        return seg

    def price_at(self, seg: Segment, when: pd.Timestamp) -> float | None:
        """Close of the last full hour before `when`, from the symbol's daily file for that date."""
        day = when.strftime("%Y-%m-%d")
        if day not in seg.files.days:
            files = self.archive.symbol_files(seg.symbol, (when.strftime("%Y-%m"),))
            if day not in files.days:
                return None
        k = read_klines(self.archive.download(self.archive.file_key(seg.symbol, "daily", day),
                                              self.raw_bin / seg.symbol, refresh=self.refresh))
        k = k.loc[: when.floor("h") - HOUR]
        return float(k["close"].iloc[-1]) if len(k) else None

    def audit_coin(self, coin: str, coingecko_id: str, history: list[SymbolOverride], ref_price: float | None,
                   is_event: bool, ref_time: pd.Timestamp | None = None) -> tuple[dict, list[Segment]]:
        segments = [self._segment(ov) for ov in history]
        found = [s for s in segments if s.files.exists]
        row = {"coin": coin, "coingecko_id": coingecko_id, "is_event_coin": is_event,
               "binance_symbols": " -> ".join(s.symbol for s in segments),
               "binance_symbol": found[-1].symbol if found else None}
        if not found:
            row.update({"history_days": 0.0, "window_coverage": 0.0, "test_coverage": 0.0, "identity_ok": False})
            return row, segments
        row.update(coverage_summary(found, self.window, self.test))
        last = found[-1]
        row["binance_last_close"] = last.last_close
        row["reference_price"] = ref_price
        if is_event or ref_price is None:
            row["identity_ok"] = last.last_close is not None
            row["identity_note"] = "event coin: identity checked in Stage 2" if is_event else "no reference price"
            return row, segments
        # compare at the same moment as the CoinGecko reference price (its last_updated hour)
        p = self.price_at(last, ref_time) if ref_time is not None else None
        basis = f"Binance close before {ref_time:%Y-%m-%d %H:%M} UTC" if p is not None else "last close in window"
        p = p if p is not None else last.last_close
        if p is None:
            row["identity_ok"], row["identity_note"] = False, "no Binance price to compare"
            return row, segments
        rel = abs(p / ref_price - 1)
        row["binance_ref_close"] = p
        row["identity_ok"] = rel <= self.cfg.universe.identity_price_tolerance
        row["identity_note"] = f"{basis} vs CoinGecko reference differs by {rel:.1%}"
        return row, segments

    def yahoo_daily_coverage(self, coin: str, coingecko_id: str, ref_price: float, names: tuple[str, ...]) -> dict:
        """Resolve and verify the Yahoo ticker for a coin not in the daily dataset; coverage of the daily window."""
        from .config import Candidate, load_config
        from .sources import YahooSource
        from .universe import UniverseAuditor, availability_stats

        dcfg = load_config()
        aud = UniverseAuditor(dcfg)
        cand = Candidate(coin=coin, coingecko=coingecko_id, yahoo=self.cfg.universe.yahoo_tickers.get(coingecko_id),
                         coinmetrics=[], category="large_cap")
        res = aud.resolve(cand, ref_price, None, names)
        if not (res.ticker and res.verified):
            return {"yahoo_ticker": res.ticker, "daily_yahoo_coverage": np.nan, "yahoo_note": res.note}
        ys = YahooSource(dcfg.window.start_date, dcfg.window.end_date, dcfg.sources.retries, dcfg.sources.backoff_base_seconds)
        path = ys.snapshot(coin, res.ticker, dcfg.raw_dir / "yahoo", refresh=self.refresh)
        if path is None:
            return {"yahoo_ticker": res.ticker, "daily_yahoo_coverage": np.nan, "yahoo_note": "no rows in window"}
        st = availability_stats(ys.load(path)["close"], dcfg.window.start_date, dcfg.window.end_date)
        return {"yahoo_ticker": res.ticker, "daily_yahoo_coverage": st["coverage"],
                "yahoo_note": f"{res.note}; Yahoo daily from {st['first_valid']}"}

    # -- sectors --------------------------------------------------------------------------
    def coingecko_categories(self, coingecko_id: str) -> list[str]:
        out_dir = self.cfg.raw_dir / "coingecko" / "coins"
        path = out_dir / f"{coingecko_id}.json"
        if not path.exists() or self.refresh:
            url = (f"https://api.coingecko.com/api/v3/coins/{coingecko_id}?localization=false&tickers=false"
                   "&market_data=false&community_data=false&developer_data=false&sparkline=false")
            for attempt in range(6):
                try:
                    req = urllib.request.Request(url, headers={"User-Agent": "cryptograph", "accept": "application/json"})
                    with urllib.request.urlopen(req, timeout=60) as r:
                        blob = r.read()
                    break
                except urllib.error.HTTPError as e:
                    if e.code == 429 and attempt < 5:
                        time.sleep(20 * (attempt + 1))
                        continue
                    raise
            out_dir.mkdir(parents=True, exist_ok=True)
            path.write_bytes(blob)
            time.sleep(4)  # public API rate limit
        return json.loads(path.read_text(encoding="utf-8")).get("categories") or []

    # -- entry point -------------------------------------------------------------------------
    def run(self) -> tuple[pd.DataFrame, dict]:
        u = self.cfg.universe
        rk = self.ranking()
        pool_end = int(rk.loc[rk["nonstable_rank"] == u.candidate_pool, "rank_all"].iloc[0])
        region = rk[rk["rank_all"] <= pool_end]
        daily = pd.read_csv(self.cfg.path(self.cfg.paths.daily_audit))
        daily_cov = dict(zip(daily["coingecko_id"], daily["combined_coverage"]))
        rows = []
        for _, r in region.iterrows():
            base = {"coin": str(r["symbol"]).upper(), "coingecko_id": r["id"], "rank_all": int(r["rank_all"]),
                    "nonstable_rank": r["nonstable_rank"], "market_cap_usd": float(r["market_cap"]),
                    "reference_price": float(r["current_price"]), "daily_yahoo_coverage": daily_cov.get(r["id"], np.nan)}
            if r["excluded"]:
                rows.append({**base, "status": "excluded", "reason": r["exclusion"]})
                continue
            logger.info(f"auditing {base['coin']} ({r['id']})")
            audit, _ = self.audit_coin(base["coin"], r["id"], self.symbol_history(r["id"], r["symbol"]),
                                       base["reference_price"], is_event=False, ref_time=_ts(r["last_updated"]))
            ok, reason = decide(audit, u)
            rows.append({**base, **audit, "passes_rules": ok, "reason": reason})
        # event coins
        for coin, hist in u.event_coins.items():
            logger.info(f"auditing event coin {coin}")
            cg = u.event_coin_coingecko.get(coin, coin.lower())
            audit, _ = self.audit_coin(coin, cg, hist, None, is_event=True)
            m = rk[rk["id"] == cg]
            rows.append({**audit, "rank_all": int(m["rank_all"].iloc[0]) if len(m) else np.nan,
                         "market_cap_usd": float(m["market_cap"].iloc[0]) if len(m) else np.nan,
                         "daily_yahoo_coverage": daily_cov.get(cg, np.nan), "status": "event coin",
                         "reason": "event coin (Terra / FTX crisis origin); kept for event studies"})

        df = pd.DataFrame(rows)
        is_event = df["status"].eq("event coin")
        passes = df["passes_rules"].eq(True) & ~is_event
        cand = df[passes].sort_values("market_cap_usd", ascending=False)
        chosen = cand.head(u.target_size)["coin"].tolist()
        spare = cand.iloc[u.target_size:]["coin"].tolist()
        df["status"] = np.where(is_event, "event coin", np.where(df["coin"].isin(chosen), "included", "excluded"))
        df.loc[df["coin"].isin(spare), "reason"] = f"passes, but outside the top {u.target_size} that pass"

        # daily Yahoo coverage for included coins that are not in the daily dataset
        names = dict(zip(rk["id"], zip(rk["symbol"].astype(str), rk["name"].astype(str))))
        for i in df.index[df["status"].eq("included") & df["daily_yahoo_coverage"].isna()]:
            r = df.loc[i]
            logger.info(f"resolving Yahoo daily ticker for {r['coin']}")
            y = self.yahoo_daily_coverage(r["coin"], r["coingecko_id"], float(r["reference_price"]),
                                          names.get(r["coingecko_id"], ()))
            for k, v in y.items():
                df.loc[i, k] = v

        # sectors: hand-curated file (data/external/sectors.csv); CoinGecko categories kept as evidence
        curated = pd.read_csv(self.cfg.external_dir / "sectors.csv")
        df = df.merge(curated[["coin", "sector"]], on="coin", how="left")
        ev_rows = []
        for _, r in df[df["status"].isin(["included", "event coin"])].iterrows():
            cats = self.coingecko_categories(r["coingecko_id"])
            ev_rows.append({"coin": r["coin"], "coingecko_id": r["coingecko_id"], "curated_sector": r["sector"],
                            "auto_rule_sector": map_sector(cats)[0], "coingecko_categories": "; ".join(cats),
                            "coingecko_url": f"https://www.coingecko.com/en/coins/{r['coingecko_id']}"})
        evidence = pd.DataFrame(ev_rows)
        missing_sector = df.loc[df["status"].isin(["included", "event coin"]) & df["sector"].isna(), "coin"].tolist()

        cols = ["coin", "coingecko_id", "rank_all", "nonstable_rank", "market_cap_usd", "sector", "binance_symbols",
                "binance_symbol", "hourly_first", "hourly_last", "history_days", "window_coverage", "test_coverage",
                "n_file_gaps", "yahoo_ticker", "daily_yahoo_coverage", "yahoo_note", "reference_price", "binance_ref_close",
                "binance_last_close", "identity_ok", "identity_note", "status", "reason"]
        df = df.reindex(columns=cols)
        write_csv(df, self.cfg.reports_dir / "universe_audit_v2.csv")
        write_csv(evidence, self.cfg.path("reports/sector_evidence.csv"))
        summary = {"included": chosen, "n_included": len(chosen), "event_coins": list(u.event_coins),
                   "deepest_nonstable_rank_used": int(df.loc[df["coin"].isin(chosen), "nonstable_rank"].max()) if chosen else None,
                   "spare_passing": spare, "missing_sector": missing_sector,
                   "sector_counts": df.loc[df["status"].eq("included"), "sector"].value_counts().to_dict(),
                   "ranking_snapshot": str(rk["last_updated"].max())}
        write_json(summary, self.cfg.interim_dir / "universe_v2.json")
        return df, summary
