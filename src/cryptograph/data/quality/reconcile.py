"""Cross-source reconciliation: Yahoo (primary) vs Coin Metrics (validation / close backfill).

Resolution rule for a coin-day whose closes differ by more than `tol` (documented
in the data dictionary and README):

  R1  CoinDesk tiebreak (only if an API key is configured): keep the source whose
      close is closer to CoinDesk's close.
  R2  Isolated Yahoo spike: the Yahoo return into the day and out of it are both
      larger than `spike` in magnitude with opposite signs, while Coin Metrics'
      returns over the same two days are not -> Yahoo close replaced by Coin
      Metrics close (close_source = "coinmetrics_corrected"); O/H/L of that day set
      missing because they share the bad tick.
  R3  Otherwise keep Yahoo and flag the day (qc flag XSRC_DISAGREE).

Backfill: where Yahoo has no close but Coin Metrics has one, the close is taken
from Coin Metrics (close_source = "coinmetrics") with O/H/L/volume missing —
only if the coin's overlap validation passes (`backfill_ok`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

LAGS = (-1, 0, 1)


@dataclass(frozen=True)
class ReconcileParams:
    tol: float = 0.02               # relative close difference that flags a day
    spike: float = 0.10             # |log return| that qualifies as a spike leg for R2
    max_discrepancy_rate: float = 0.05   # backfill allowed only if overlap flag rate below this
    max_median_diff: float = 0.01        # ... and median relative difference below this
    min_overlap: int = 180               # ... and at least this many overlapping days


def best_lag(primary: pd.Series, secondary: pd.Series, lags=LAGS) -> tuple[int, dict[int, float]]:
    """Lag L maximising corr(r_primary[t], r_secondary[t+L]) on log returns."""
    rp = np.log(primary.where(primary > 0)).diff()
    rs = np.log(secondary.where(secondary > 0)).diff()
    corrs = {lag: float(rp.corr(rs.shift(-lag))) for lag in lags}
    corrs = {k: (v if np.isfinite(v) else -np.inf) for k, v in corrs.items()}
    return max(corrs, key=corrs.get), corrs


def align(series: pd.Series, lag: int) -> pd.Series:
    """Re-date `series` so that value at t is the secondary's value at t+lag."""
    return series.shift(-lag)


def reconcile_coin(yahoo: pd.DataFrame | None, cm: pd.DataFrame | None, coindesk: pd.DataFrame | None,
                   params: ReconcileParams = ReconcileParams()) -> tuple[pd.DataFrame, dict]:
    """Merge one coin's sources into a daily frame + a per-coin report row.

    Output columns: open, high, low, close, volume, market_cap, close_source,
    xsrc_rel_diff, xsrc_flag, xsrc_resolution.
    """
    cols = ["open", "high", "low", "close", "volume"]
    y = (yahoo[cols].copy() if yahoo is not None and len(yahoo) else
         pd.DataFrame(columns=cols, index=pd.DatetimeIndex([], name="date"), dtype=float))
    y = y.astype(float)  # non-positive prices are flagged later by quality.checks, not silently dropped
    report: dict = {"overlap_days": 0, "lag": None, "corr_lag_-1": np.nan, "corr_lag_0": np.nan, "corr_lag_1": np.nan,
                    "n_flagged": 0, "discrepancy_rate": np.nan, "median_rel_diff": np.nan,
                    "n_corrected_R1": 0, "n_corrected_R2": 0, "n_kept_flagged_R3": 0,
                    "backfill_ok": False, "n_backfilled": 0}

    cm_close = cm_mcap = None
    if cm is not None and len(cm) and cm["cm_price"].notna().any():
        lag, corrs = best_lag(y["close"], cm["cm_price"]) if y["close"].notna().sum() > 30 else (0, {})
        cm_close = align(cm["cm_price"], lag)
        cm_mcap = align(cm["cm_mcap"], lag)
        report.update({"lag": lag, **{f"corr_lag_{k}": v for k, v in corrs.items()}})
    elif cm is not None and len(cm):
        cm_mcap = cm["cm_mcap"]

    idx = y.index
    for s in (cm_close, cm_mcap, coindesk["cd_close"] if coindesk is not None and len(coindesk) else None):
        if s is not None:
            idx = idx.union(s.dropna().index)
    out = y.reindex(idx)
    out["market_cap"] = cm_mcap.reindex(idx) if cm_mcap is not None else np.nan
    out["close_source"] = np.where(out["close"].notna(), "yahoo", None)
    out["xsrc_rel_diff"] = np.nan
    out["xsrc_flag"] = False
    out["xsrc_resolution"] = None

    if cm_close is not None:
        c = cm_close.reindex(idx)
        both = out["close"].notna() & c.notna() & (c > 0)
        rel = (out["close"] / c - 1).abs().where(both)
        out["xsrc_rel_diff"] = rel
        flagged = rel > params.tol
        out["xsrc_flag"] = flagged.fillna(False)
        n_overlap = int(both.sum())
        report.update({"overlap_days": n_overlap, "n_flagged": int(flagged.sum()),
                       "discrepancy_rate": float(flagged.sum() / n_overlap) if n_overlap else np.nan,
                       "median_rel_diff": float(rel.median()) if n_overlap else np.nan})

        cd = coindesk["cd_close"].reindex(idx) if coindesk is not None and len(coindesk) else None
        ry = np.log(out["close"]).diff()
        rc = np.log(c).diff()
        for d in out.index[out["xsrc_flag"]]:
            yv, cv = out.at[d, "close"], c.at[d]
            if cd is not None and pd.notna(cd.at[d]):
                if abs(cv - cd.at[d]) < abs(yv - cd.at[d]):
                    _replace(out, d, cv, "coinmetrics_corrected", "R1_coindesk")
                    report["n_corrected_R1"] += 1
                else:
                    out.at[d, "xsrc_resolution"] = "R1_keep_yahoo"
                continue
            pos = out.index.get_loc(d)
            if 0 < pos < len(out) - 1:
                nxt = out.index[pos + 1]
                y_in, y_out = ry.at[d], ry.at[nxt]
                c_in, c_out = rc.at[d], rc.at[nxt]
                yahoo_spike = (abs(y_in) > params.spike and abs(y_out) > params.spike and np.sign(y_in) != np.sign(y_out))
                cm_smooth = not (abs(c_in) > params.spike and abs(c_out) > params.spike)
                if yahoo_spike and cm_smooth and np.isfinite(cv):
                    _replace(out, d, cv, "coinmetrics_corrected", "R2_isolated_spike")
                    report["n_corrected_R2"] += 1
                    continue
            out.at[d, "xsrc_resolution"] = "R3_keep_yahoo_flagged"
            report["n_kept_flagged_R3"] += 1

        ok = (n_overlap >= params.min_overlap
              and report["discrepancy_rate"] <= params.max_discrepancy_rate
              and report["median_rel_diff"] <= params.max_median_diff)
        report["backfill_ok"] = bool(ok)
        if ok:
            fill = out["close"].isna() & c.notna() & (c > 0)
            out.loc[fill, "close"] = c[fill]
            out.loc[fill, "close_source"] = "coinmetrics"
            report["n_backfilled"] = int(fill.sum())

    out.index.name = "date"
    return out.sort_index(), report


def _replace(out: pd.DataFrame, d: pd.Timestamp, value: float, source: str, rule: str) -> None:
    out.at[d, "close"] = value
    out.loc[d, ["open", "high", "low"]] = np.nan
    out.at[d, "close_source"] = source
    out.at[d, "xsrc_resolution"] = rule
