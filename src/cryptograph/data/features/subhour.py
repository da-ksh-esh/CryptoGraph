"""Hourly features built from the 5-minute bars inside each hour (Binance 5m klines, same archive as the 1h data).

An hourly bar labelled t (its open time) covers [t, t + 1h); its 5-minute bars open at t, t+5m, ..., t+55m, so every
feature of bar t is known when bar t closes. 5-minute returns are deseasonalised with the same hour-of-week profile
as the hourly returns and divided by the trailing 168h volatility of the deseasonalised hourly returns (the shock
label's own scale), so the features are comparable across coins and market regimes:
  rv5_ratio      log(realized volatility of the hour from its 12 five-minute returns / sigma168)
  rv5_3h_ratio   the same over the last 3 hours
  max5_ratio     largest |5-minute return| in the hour / sigma168
  jump_share     largest squared 5-minute return / sum of squares (1 = one jump, 1/12 = an even hour)
  last15         signed return of the last 15 minutes / sigma168
  last15_abs     its absolute value
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SUBHOUR_NAMES = ["rv5_ratio", "rv5_3h_ratio", "max5_ratio", "jump_share", "last15", "last15_abs"]


def subhour_features(k5: pd.DataFrame, times: pd.DatetimeIndex, coins: list[str], season_vol_per_bar: np.ndarray,
                     sigma168: np.ndarray) -> dict[str, np.ndarray]:
    """k5: (time, coin) -> close, 5-minute bars; season_vol_per_bar: (T,) hourly seasonal variance factor of each bar;
    sigma168: (T, N) trailing 168h std of the deseasonalised hourly returns, up to and including bar t."""
    close = k5["close"].unstack("coin").reindex(columns=coins).astype(float)
    r5 = np.log(close).diff()
    # a 5-minute return across a gap longer than one bar is not a 5-minute return
    gap = close.index.to_series().diff() != pd.Timedelta(minutes=5)
    r5[gap.to_numpy()] = np.nan
    hour = r5.index.floor("h")
    ss = (r5 ** 2).groupby(hour).sum(min_count=10)                # >= 10 of the 12 returns present
    mx = r5.abs().groupby(hour).max()
    tail = r5.index.minute >= 45
    last = r5[tail].groupby(hour[tail]).sum(min_count=3)
    ss, mx, last = (f.reindex(index=times, columns=coins).to_numpy() for f in (ss, mx, last))
    scale = np.sqrt(season_vol_per_bar)[:, None] * sigma168
    with np.errstate(divide="ignore", invalid="ignore"):
        rv = np.sqrt(ss)
        ss3 = pd.DataFrame(ss).rolling(3, min_periods=3).sum().to_numpy()
        out = {"rv5_ratio": np.log(rv / scale), "rv5_3h_ratio": np.log(np.sqrt(ss3 / 3) / scale),
               "max5_ratio": mx / scale, "jump_share": mx ** 2 / ss, "last15": last / scale,
               "last15_abs": np.abs(last) / scale}
    return {k: np.where(np.isfinite(v), v, np.nan).astype(np.float32) for k, v in out.items()}
