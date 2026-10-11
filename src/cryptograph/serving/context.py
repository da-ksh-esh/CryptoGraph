"""Real (unscaled) values per coin at one hour, so the dashboard can say "trading at 14.6× its usual volume" instead of
naming an input. Uses data up to that hour only; features the panel lacks come out as NaN."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def coin_context(panel: dict, gidx: list[int], bar: int, coins: list[str], vol_window: int = 720) -> pd.DataFrame:
    """Per coin at `bar`: last-hour move, shock size (z), volume vs its usual week (volume_rel_168h is the log ratio to
    the trailing 7-day median), volatility vs its median over the previous `vol_window` hours, shocks in the last 24 h,
    and whether it shocked this hour."""
    fn = list(panel["feature_names"])
    X = panel["X_raw"]
    lo = max(0, bar - vol_window)
    nan = np.full(len(gidx), np.nan)

    def now(name):                                     # value at `bar`, NaN if the panel lacks the feature
        return X[bar, gidx, fn.index(name)].numpy().astype(float) if name in fn else nan

    def vs_usual(name):                                # value at `bar` / its median over the previous window
        if name not in fn or bar <= lo:
            return nan
        hist = X[lo:bar + 1, gidx, fn.index(name)].numpy().astype(float)
        with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)          # coins with no history yet -> NaN
            return hist[-1] / np.nanmedian(hist[:-1], axis=0)

    sh = (panel["shock_today"] & panel["shock_today_mask"])[:bar + 1, gidx].numpy()
    return pd.DataFrame({
        "coin": coins, "active": panel["active"][bar, gidx].numpy(), "ret_1h": np.expm1(now("log_return")),
        "z": panel["shock_z"][bar, gidx].numpy().astype(float), "volume_multiple": np.exp(now("volume_rel_168h")),
        "vol_multiple": vs_usual("rv_24h"), "vol72_multiple": vs_usual("rv_72h"),
        "shocks_24h": sh[max(0, bar - 23):bar + 1].sum(0), "shock_now": sh[-1]})
