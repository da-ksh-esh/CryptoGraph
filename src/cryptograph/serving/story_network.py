"""The network analysis behind the dashboard's "The network" page, recomputed from the processed panel with the same
functions and windows as notebooks 01 and 02 (network/, analysis/). Every function takes the panel view built by
`panel_view` and returns small frames; nothing here is used by any model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.config import PROJECT_ROOT

TEST_WINDOW = ("2024-04-02", None)          # test period from the day all 30 coins trade (ENA), as in notebook 02
FTX_WINDOW = ("2022-11-06", "2022-11-11")


@dataclass
class PanelView:
    times: pd.DatetimeIndex
    coins: list[str]                 # the model (graph) coins
    r: np.ndarray                    # (T, N) rush-hour-adjusted hourly returns
    z: np.ndarray                    # (T, N) shock score
    shock: np.ndarray                # (T, N) shock and observed
    valid: np.ndarray                # (T, N) shock label observable
    active: np.ndarray               # (T, N) trading
    split: np.ndarray                # (T,) 0 train, 1 validation, 2 test, -1 outside
    sample: dict                     # split name -> (T,) usable sample bars
    threshold: float
    all_coins: list[str]
    z_all: np.ndarray                # (T, all coins): includes the event coins LUNC and FTT


def panel_view(root: Path = PROJECT_ROOT, version: str = "v3", resolution: str = "1h") -> PanelView | None:
    from ..data.storage import load_pt

    p = Path(root) / "data" / "processed" / version / resolution / "panel.pt"
    if not p.exists():
        return None
    P = load_pt(p)
    need = {"r_ds", "shock_z", "shock_today", "shock_today_mask", "active", "times", "split_id"}
    if not need <= set(P):
        return None
    all_coins = list(P.get("coins", P.get("graph_coins", [])))
    coins = list(P.get("graph_coins", all_coins))
    cols = [all_coins.index(c) for c in coins if c in all_coins]

    def g(key):
        a = P[key].numpy()
        return a[:, cols] if a.shape[1] == len(all_coins) else a

    sample = {k: v.numpy().astype(bool) for k, v in P.get("sample_mask", {}).items()}
    return PanelView(pd.to_datetime(P["times"]), coins, g("r_ds").astype(float), np.nan_to_num(g("shock_z")),
                     g("shock_today") & g("shock_today_mask"), g("shock_today_mask"), g("active"), P["split_id"].numpy(),
                     sample, float(P.get("shock_threshold", 2.29)), all_coins, np.nan_to_num(P["shock_z"].numpy()))


def _rows(v: PanelView, start: str | None, end: str | None) -> np.ndarray:
    m = np.ones(len(v.times), bool)
    if start:
        m &= v.times >= pd.Timestamp(start)
    if end:
        m &= v.times < pd.Timestamp(end)
    return m


def window_corr(v: PanelView, start: str | None = TEST_WINDOW[0], end: str | None = TEST_WINDOW[1]) -> np.ndarray:
    x = pd.DataFrame(v.r[_rows(v, start, end)], columns=v.coins).dropna()
    return x.corr().to_numpy()


# --- chapter: no leader --------------------------------------------------------------------------------------------
def lead_lag(v: PanelView, leaders=("BTC", "ETH"), max_lag: int = 24, split: int = 0) -> pd.DataFrame:
    """Median over coins of corr(leader at t, coin at t+k), on shock scores of the training years (notebook 01 §15)."""
    from ..analysis.lead_lag import ccf_profile

    rows = v.split == split
    out = []
    for lead in leaders:
        if lead not in v.coins:
            continue
        x = v.z[rows, v.coins.index(lead)]
        prof = [ccf_profile(x, v.z[rows, j], max_lag) for j, c in enumerate(v.coins)
                if c != lead and v.active[rows, j].sum() >= 2000]
        med = np.nanmedian(np.vstack(prof), axis=0)
        out += [{"leader": lead, "lag": k, "corr": float(c)} for k, c in zip(range(-max_lag, max_lag + 1), med)]
    return pd.DataFrame(out)


# --- chapter: spillover ----------------------------------------------------------------------------------------------
def spillover(v: PanelView, horizon: int = 24, start: str | None = TEST_WINDOW[0]) -> dict | None:
    """Diebold-Yilmaz generalized FEVD (VAR(1), 24-hour horizon) on the test window: who sends to whom."""
    from ..network.spillover import spillover_summary, window_gfevd

    theta = window_gfevd(v.r[_rows(v, start, None)], H=horizon, p=1)
    if theta is None:
        return None
    s = spillover_summary(theta)
    table = pd.DataFrame({"coin": v.coins, "to": s["to"], "from": s["from"], "net": s["net"]})
    return {"theta": theta, "table": table.sort_values("net", ascending=False).reset_index(drop=True),
            "total": s["total"]}


# --- chapter: communities ---------------------------------------------------------------------------------------------
def communities(rankings: pd.DataFrame, n_null: int = 2000, seed: int = 0) -> dict | None:
    """Louvain communities (market mode removed, from notebook 02's rankings) against sectors: NMI and its shuffle test."""
    from sklearn.metrics import normalized_mutual_info_score

    if rankings.empty or not {"community", "sector"} <= set(rankings):
        return None
    obs = normalized_mutual_info_score(rankings["sector"], rankings["community"])
    rng = np.random.default_rng(seed)
    null = np.array([normalized_mutual_info_score(rng.permutation(rankings["sector"].to_numpy()), rankings["community"])
                     for _ in range(n_null)])
    groups = (rankings.groupby("community")
              .apply(lambda d: ", ".join(f"{c} ({s})" for c, s in zip(d["coin"], d["sector"])), include_groups=False))
    return {"nmi": float(obs), "null_mean": float(null.mean()), "null_q95": float(np.quantile(null, 0.95)),
            "p": float((null >= obs).mean()), "groups": groups}


# --- chapter: the minimum spanning tree -------------------------------------------------------------------------------
def mst(v: PanelView) -> dict | None:
    """The 29 strongest links joining all coins (correlation distance sqrt(2(1-rho)), test window), with a layout."""
    import networkx as nx

    from ..network.mst import mst_edges

    C = window_corr(v)
    edges = mst_edges(C, np.ones(len(v.coins), bool))
    if not edges:
        return None
    G = nx.Graph()
    G.add_weighted_edges_from((v.coins[i], v.coins[j], d) for i, j, d in edges)
    pos = nx.kamada_kawai_layout(G, weight="weight")
    deg = dict(G.degree)
    e = pd.DataFrame([(v.coins[i], v.coins[j], d, C[i, j]) for i, j, d in edges], columns=["a", "b", "distance", "corr"])
    return {"edges": e.sort_values("distance").reset_index(drop=True), "pos": pos, "degree": deg,
            "mean_length": float(e["distance"].mean())}


# --- chapter: crises tighten the network -------------------------------------------------------------------------------
CRISIS_MEASURES = {"mean_corr": "average correlation", "tree_length": "tree length", "density": "share of pairs linked",
                   "modularity": "group structure", "total_spillover": "spillover index", "mean_shock_lift": "shock lift"}


def crisis_changes(ts: pd.DataFrame, crises: pd.DataFrame, days: int = 14) -> pd.DataFrame:
    """Mean of each measure in the `days` after each crisis minus the `days` before (notebook 02 §6)."""
    rows = []
    for e in crises.itertuples():
        before = ts[(ts["time"] >= e.date - pd.Timedelta(days=days)) & (ts["time"] < e.date)]
        after = ts[(ts["time"] >= e.date) & (ts["time"] < e.date + pd.Timedelta(days=days))]
        rows.append({"crisis": f"{e.date:%b %Y} · {e.name.split('/')[0].strip()}",
                     **{lab: after[m].mean() - before[m].mean() for m, lab in CRISIS_MEASURES.items() if m in ts}})
    return pd.DataFrame(rows)


# --- chapter: sectors and contagion -------------------------------------------------------------------------------------
def sector_contagion(v: PanelView, sector_of: dict, k: int = 1) -> pd.DataFrame:
    """After coin i shocks, the chance coin j shocks within k hours: same sector vs other sectors vs any hour."""
    from ..data.features.targets import future_any
    from ..network.cascade import transmission_counts

    sec = np.array([sector_of.get(c, "other") for c in v.coins])
    same = (sec[:, None] == sec[None, :]) & ~np.eye(len(sec), dtype=bool)
    cross = sec[:, None] != sec[None, :]
    anyf, allv = future_any(v.shock, v.valid, k)
    out = []
    for split, rows in v.sample.items():
        n, h, bn, bh = transmission_counts(v.shock, anyf, allv, rows)
        out += [{"period": split, "pairs": "same sector", "rate": h[same].sum() / max(n[same].sum(), 1)},
                {"period": split, "pairs": "different sectors", "rate": h[cross].sum() / max(n[cross].sum(), 1)},
                {"period": split, "pairs": "any hour (base rate)", "rate": bh.sum() / max(bn.sum(), 1)}]
        for s in sorted(set(sec)):
            m = sec == s
            if m.sum() >= 2 and split == "test":
                w = (m[:, None] & m[None, :]) & ~np.eye(len(sec), dtype=bool)
                out.append({"period": "test", "pairs": f"inside {s}", "rate": h[w].sum() / max(n[w].sum(), 1)})
    return pd.DataFrame(out)


# --- chapter: FTX hour by hour ----------------------------------------------------------------------------------------
def ftx_hours(v: PanelView, window=FTX_WINDOW, coin: str = "FTT") -> pd.DataFrame:
    """FTT's shock score and how many model coins shocked down / up in each hour of the FTX collapse."""
    if coin not in v.all_coins:
        return pd.DataFrame()
    rows = _rows(v, window[0], window[1])
    z = v.z[rows]
    thr = v.threshold
    return pd.DataFrame({"time": v.times[rows], "ftt_z": v.z_all[rows, v.all_coins.index(coin)],
                         "down": ((z < -thr) & v.active[rows]).sum(1), "up": ((z > thr) & v.active[rows]).sum(1),
                         "trading": v.active[rows].sum(1)})


def chance_shocks_per_hour(v: PanelView) -> float:
    """How many model coins shock in an average training hour."""
    rows = v.split == 0
    return float(v.shock[rows].sum() / max(rows.sum(), 1))


# --- chapter: what the model gets from the network ---------------------------------------------------------------------
NETWORK_FEATURES = {
    "strength": "how strongly the coin is tied to all others (average correlation, past week)",
    "eigenvector": "how central it is: tied to coins that are themselves central",
    "mst_degree": "how many links it has in the market's skeleton (minimum spanning tree)",
    "community_size_frac": "how big the group of coins it moves with is",
    "dy_pagerank": "how much others depend on it in the spillover network (PageRank)",
    "dy_net": "whether it sends or receives more spillover (Diebold-Yilmaz NET)",
    "shock_pagerank": "how much others' shocks follow its shocks (PageRank, time-ordered)",
    "shock_out_lift": "the extra chance other coins shock right after it does",
}
