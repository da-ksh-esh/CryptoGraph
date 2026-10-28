"""What-if shocks for the dashboard (dashboard/app.py, "What if?" tab): move one coin's price by x% in one hour and see
how the move spreads to the other coins, hour by hour.

A VAR on the hourly log returns of the trailing window before the prediction run's moment (data up to that hour
only), read through generalized impulse responses (Pesaran and Shin 1998): a shock d to coin j's log return moves
coin i by

    psi_h[i] = d * (Phi_h Sigma e_j)[i] / sigma_jj        in hour h (h = 0: the same hour)

so the same-hour move is the regression of i's return on j's (no ordering of the coins is assumed) and later hours
follow the VAR's lags. Responses are linear in d (in log returns): a 20% drop spreads exactly twice as far as a 10% one,
which real crashes do not quite follow (coins move more alike in a crash).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..analysis.connectedness import ma_coefficients
from ..data.config import PROJECT_ROOT
from ..network.spillover import var_ols

# trailing windows the dashboard offers (hours)
WINDOWS = {"past week": 168, "past month": 720, "past 3 months": 2160}


@dataclass
class ShockModel:
    coins: list[str]
    irf: np.ndarray            # (H+1, N, N): [h, i, j] = log-return move of i in hour h after a unit shock to j
    usual_swing: np.ndarray    # (N,) std of each coin's hourly log return over the trailing week
    threshold: float           # a big move: |hourly return| > threshold x usual swing
    start: str
    end: str
    n_hours: int               # hours the VAR was fitted on (complete rows)
    lags: int


def fit_shock_model(r: np.ndarray, times: list, coins: list[str], t_end: int, window: int, horizon: int = 24,
                    lags: int = 1, threshold: float = 2.29, min_coverage: float = 0.9, swing_hours: int = 168
                    ) -> ShockModel:
    """VAR(lags) on r (T, N) hourly log returns over (t_end - window, t_end]; coins observed in less than
    `min_coverage` of those hours are left out, then hours with any missing return."""
    lo = max(0, t_end - window + 1)
    w = r[lo: t_end + 1]
    keep = np.flatnonzero(np.isfinite(w).mean(axis=0) >= min_coverage)
    X = w[:, keep]
    X = X[np.isfinite(X).all(axis=1)]
    if len(keep) < 2 or len(X) < 3 * (len(keep) * lags + 1):
        raise ValueError(f"too little data for a VAR: {len(X)} complete hours, {len(keep)} coins")
    coefs, sigma = var_ols(X, lags)
    irf = ma_coefficients(coefs, horizon + 1) @ sigma / np.diag(sigma)[None, None, :]
    swing = np.nanstd(r[max(0, t_end - swing_hours + 1): t_end + 1][:, keep], axis=0)
    return ShockModel([coins[k] for k in keep], irf, swing, float(threshold), str(times[lo]), str(times[t_end]),
                      len(X), lags)


def load_returns(summary: dict, root: Path = PROJECT_ROOT) -> tuple[np.ndarray, list, list[str], float] | None:
    """Hourly log returns (T, N) of the graph coins from the run's panel, its times, the coins and the big-move
    threshold; None when the panel is missing."""
    from ..data.storage import load_pt

    p = Path(root) / "data" / "processed" / summary.get("data_version", "v2") / summary.get("resolution", "1h") \
        / "panel.pt"
    if not p.exists():
        return None
    panel = load_pt(p)
    coins = list(panel["graph_coins"])
    cols = [list(panel.get("coins", coins)).index(c) for c in coins]
    li = list(panel["feature_names"]).index("log_return")
    r = panel["X_raw"][:, cols, li].numpy().astype(float)
    return r, list(panel["times"]), coins, float(panel.get("shock_threshold", 2.29))


def simulate(m: ShockModel, coin: str, shock_pct: float) -> pd.DataFrame:
    """One row per hour and coin after `coin` moves by `shock_pct` percent (-10 = a 10% drop) in hour 0: the hour's
    move and the price change so far (%), the hour's move in multiples of the coin's usual hourly swing, and `big`
    = the coin has made a big move of its own by that hour (the shocked coin itself never counts)."""
    j = m.coins.index(coin)
    step = np.log1p(shock_pct / 100.0) * m.irf[:, :, j]                      # (H+1, N) log returns
    with np.errstate(divide="ignore", invalid="ignore"):
        times_usual = np.abs(step) / m.usual_swing[None, :]
    big = np.logical_or.accumulate(np.nan_to_num(times_usual) > m.threshold, axis=0)
    big[:, j] = False
    H, N = step.shape
    return pd.DataFrame({"hour": np.repeat(np.arange(H), N), "coin": m.coins * H,
                         "move_pct": np.expm1(step).ravel() * 100, "change_pct": np.expm1(np.cumsum(step, 0)).ravel()
                         * 100, "times_usual": times_usual.ravel(), "big": big.ravel(),
                         "source": np.tile(np.arange(N) == j, H)})


def at_hour(sim: pd.DataFrame, hour: int) -> pd.DataFrame:
    """Every coin at one hour, the shocked coin first and the others by how far their price changed."""
    t = sim[sim["hour"] == hour]
    return t.assign(_k=-t["change_pct"].abs()).sort_values(["source", "_k"], ascending=[False, True]) \
        .drop(columns="_k").reset_index(drop=True)


def coin_options(coins: list[str], ranks: pd.DataFrame) -> list[str]:
    """Coins to shock, the biggest spreaders of price swings first (net spillover from the network rankings)."""
    if ranks.empty or "dy_net" not in ranks:
        return sorted(coins)
    net = ranks.set_index("coin")["dy_net"]
    return sorted(coins, key=lambda c: (-net.get(c, -np.inf), c))


def what_if_sentence(sim: pd.DataFrame, coin: str, shock_pct: float, hour: int) -> str:
    """The simulation in words at one hour: who moves most, how many coins make a big move of their own."""
    t = at_hour(sim, hour)
    others = t[~t["source"]]
    move = "dropped" if shock_pct < 0 else "jumped"
    when = "in the same hour" if hour == 0 else f"after {hour} hour{'s' if hour != 1 else ''}"
    out = f"If **{coin}** {move} **{abs(shock_pct):.0f}%** in one hour, "
    if others.empty or shock_pct == 0:
        return out + "no other coin would move."
    top = others.iloc[0]
    n_big = int(others["big"].sum())
    out += (f"**{top['coin']}** would move most ({top['change_pct']:+.1f}% {when}), and the typical coin "
            f"{others['change_pct'].median():+.1f}%. ")
    return out + (f"**{n_big} of {len(others)}** other coins would make a big move of their own." if n_big
                  else "No other coin would move by enough to count as a big move of its own.")
