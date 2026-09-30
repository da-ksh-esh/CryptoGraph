"""Event studies and a first look at shock persistence."""

from __future__ import annotations

import numpy as np
import pandas as pd


def event_window(r: pd.DataFrame, date: pd.Timestamp, pre: int = 10, post: int = 30) -> pd.DataFrame:
    """Cumulative log return per coin from day -pre to +post (0 = event day, cum. return 0 at day -1)."""
    idx = pd.date_range(date - pd.Timedelta(days=pre), date + pd.Timedelta(days=post), freq="D")
    w = r.reindex(idx)
    base = w.loc[: date - pd.Timedelta(days=1)]
    cum = w.fillna(0).cumsum() - base.fillna(0).sum()
    cum[w.isna()] = np.nan
    cum.index = range(-pre, post + 1)
    return cum


def persistence(r: pd.DataFrame, rv7: pd.DataFrame, date: pd.Timestamp, pre: int = 60, post: int = 120,
                band_q: float = 0.95) -> pd.DataFrame:
    """Per coin: vol ratio (30d after / 30d before) and days until 7-day vol returns inside the pre-event band.

    Band = [q_{2.5%}, q_{97.5%}] of rv7 over the `pre` days before the event; 'days_to_normal' is the first day
    k >= 1 after the event with rv7 inside the band for 3 consecutive days (NaN if not within `post`).
    """
    rows = {}
    lo_q, hi_q = (1 - band_q) / 2, 1 - (1 - band_q) / 2
    for c in r.columns:
        before = r[c].loc[date - pd.Timedelta(days=30): date - pd.Timedelta(days=1)]
        after = r[c].loc[date: date + pd.Timedelta(days=29)]
        if before.notna().sum() < 20 or after.notna().sum() < 20:
            continue
        pre_rv = rv7[c].loc[date - pd.Timedelta(days=pre): date - pd.Timedelta(days=1)].dropna()
        lo, hi = pre_rv.quantile(lo_q), pre_rv.quantile(hi_q)
        post_rv = rv7[c].loc[date + pd.Timedelta(days=1): date + pd.Timedelta(days=post)]
        inside = (post_rv >= lo) & (post_rv <= hi)
        run = inside.rolling(3).sum() == 3
        days = (run.idxmax() - date).days - 2 if run.any() else np.nan
        rows[c] = {"cum_ret_0_30": float(after.sum()), "vol_ratio": float(after.std() / before.std()),
                   "days_to_normal": days}
    return pd.DataFrame(rows).T
