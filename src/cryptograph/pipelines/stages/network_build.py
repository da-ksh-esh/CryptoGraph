"""Hourly Stage 5 — network features, network time series, rankings and the independent cascade baseline.

Reads data/processed/v2/<res>/{panel.pt, graph_snapshots.pt, tgn_events.pt} and writes:
  data/processed/v2/<res>/network_features.pt   per-snapshot node features (trailing windows only)
  data/processed/v2/rankings.csv (+ rankings.json)  coin rankings on the test window (1h): DY TO/FROM/NET,
                                                PageRank on the DY and shock-transmission networks, eigenvector
  reports/network_timeseries_<res>.csv          network-level measures per snapshot
  reports/cascade_baseline.csv                  independent cascade vs simpler baselines on the TGN events
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from ...common.logger import get_logger
from ...data.config import HourlyConfig
from ...data.features.intraday import bars
from ...data.features.targets import future_any
from ...data.graphs.correlation import pairwise_corr
from ...data.storage import load_pt, utc_now, write_csv, write_json
from ...network.cascade import IndependentCascade, rolling_transmission
from ...network.centrality import eigenvector_centrality, transmitter_pagerank
from ...network.communities import louvain, remove_market_mode
from ...network.features import NODE_FEATURES, NetworkParams, compute_network_features
from ...network.mst import mst_degree
from ...network.spillover import spillover_summary

logger = get_logger(__name__)
SPLITS = ("train", "val", "test")


def load_sectors(cfg: HourlyConfig, coins: list[str]) -> tuple[np.ndarray, list[str], list[str]]:
    s = pd.read_csv(cfg.external_dir / "sectors.csv").set_index("coin")["sector"]
    names = sorted(set(s.reindex(coins).fillna("unknown")))
    labels = s.reindex(coins).fillna("unknown").tolist()
    return np.array([names.index(x) for x in labels]), names, labels


def _scores(y: np.ndarray, s: np.ndarray) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score

    if len(np.unique(y)) < 2:
        return {"auroc": np.nan, "auprc": np.nan}
    return {"auroc": float(roc_auc_score(y, s)), "auprc": float(average_precision_score(y, s))}


def cascade_baseline(cfg: HourlyConfig, res: str, p: dict, ev: dict, shock: np.ndarray, valid: np.ndarray,
                     active: np.ndarray) -> list[dict]:
    bh = p["bar_hours"]
    k_hours = cfg.targets.contagion_k_hours
    kb = [bars(k, bh) for k in k_hours]
    train_rows = p["sample_mask"]["train"].numpy()
    model = IndependentCascade.fit(shock, valid, train_rows, sorted(set(kb)), cfg.network.cascade_prior_strength)
    src_all = shock & valid & active
    rows = []
    for s in SPLITS:
        d = ev[s]
        t, src, dst = d.t.numpy(), d.src.numpy(), d.dst.numpy()
        shocked = src_all[t]
        on_sample = p["sample_mask"][s].numpy()[t]          # same rule as the models: embargo, label window in split
        for ki, (kh, k) in enumerate(zip(k_hours, kb)):
            m = d.y_mask[:, ki].numpy() & on_sample
            y = d.y[:, ki].numpy()[m]
            scores = {"receiver base rate": model.receiver_base(k, dst[m]),
                      "pairwise p_ij": model.pairwise(k, src[m], dst[m]),
                      "independent cascade": model.cascade(k, shocked[m], dst[m])}
            for name, sc in scores.items():
                rows.append({"resolution": res, "split": s, "k_hours": kh, "model": name, "events": int(m.sum()),
                             "positive_rate": float(y.mean()), **_scores(y, sc),
                             "note": "in-sample (fitted on train)" if s == "train" else ""})
    return rows


def rankings(cfg: HourlyConfig, p: dict, gidx: list[int], gcoins: list[str], sectors: list[str], r: np.ndarray,
             shock: np.ndarray, valid: np.ndarray, active: np.ndarray) -> tuple[pd.DataFrame, dict]:
    """Rankings on the test window from the first bar at which every graph coin is listed."""
    from ...analysis.connectedness import fit_var, generalized_fevd

    times = pd.to_datetime(p["times"])
    split = p["split_id"].numpy()
    test_rows = np.flatnonzero(split == 2)
    first = max(int(test_rows[0]), *[int(np.flatnonzero(active[:, j])[0]) for j in range(len(gcoins))])
    last = int(test_rows[-1])
    R = pd.DataFrame(r[first: last + 1], index=times[first: last + 1], columns=gcoins)
    nc = cfg.network
    H = bars(nc.spillover_horizon_hours, p["bar_hours"])
    res = fit_var(R.dropna(), maxlags=nc.ranking_var_maxlags, ic="bic")
    theta = generalized_fevd(res.coefs, np.asarray(res.sigma_u), H)
    summ = spillover_summary(theta)
    allon = np.ones(len(gcoins), bool)
    C = pairwise_corr(R.to_numpy(), R.to_numpy(), 100)
    np.fill_diagonal(C, np.nan)
    src = shock & valid
    k = bars(nc.cascade_k_hours, p["bar_hours"])
    anyf, allv = future_any(src, valid, k)
    _, lift = rolling_transmission(src, anyf, allv, last, last - first + 1, k, nc.cascade_prior_strength)
    comm = louvain(remove_market_mode(C, allon), allon, seed=cfg.seed, resolution=nc.louvain_resolution)
    df = pd.DataFrame({
        "coin": gcoins, "sector": sectors,
        "dy_to": summ["to"], "dy_from": summ["from"], "dy_net": summ["net"],
        "dy_pagerank": transmitter_pagerank(theta, allon, nc.pagerank_alpha),
        "shock_pagerank": transmitter_pagerank(np.nan_to_num(lift.T), allon, nc.pagerank_alpha),
        "shock_out_lift": np.nanmean(np.where(np.eye(len(gcoins), dtype=bool), np.nan, lift), axis=1),
        "eigenvector": eigenvector_centrality(C, allon), "mst_degree": mst_degree(C, allon),
        "community": comm,
    })
    for c in ("dy_net", "dy_pagerank", "shock_pagerank", "eigenvector"):
        df[f"rank_{c}"] = df[c].rank(ascending=False, method="min").astype(int)
    meta = {"window": [str(times[first]), str(times[last])], "rows_used": int(res.nobs),
            "var_lag_order_bic": int(res.k_ar), "gfevd_horizon_bars": H, "total_spillover_index": summ["total"],
            "cascade_k_bars": k, "note": "evaluation benchmark for explanations (TempME); computed on the test "
                                         "window, never used as a model input"}
    return df.sort_values("dy_net", ascending=False).reset_index(drop=True), meta


def network_params(cfg: HourlyConfig, bh: int, threshold: float) -> NetworkParams:
    """Window lengths in bars and the other network settings; `threshold` = the graph's fitted |corr| threshold."""
    nc = cfg.network
    return NetworkParams(
        corr_window=bars(nc.corr_window_hours, bh), spill_window=bars(nc.spillover_window_hours, bh),
        spill_lags=nc.spillover_var_lags, spill_horizon=bars(nc.spillover_horizon_hours, bh),
        cascade_window=bars(nc.cascade_window_hours, bh), cascade_k=bars(nc.cascade_k_hours, bh),
        cascade_prior=nc.cascade_prior_strength, threshold=float(threshold),
        louvain_resolution=nc.louvain_resolution, pagerank_alpha=nc.pagerank_alpha, min_coverage=nc.min_coverage,
        min_obs_frac=cfg.graph.min_obs_frac, seed=cfg.seed)


def build_resolution(cfg: HourlyConfig, res: str) -> dict:
    out_dir = cfg.processed_dir(res)
    p = load_pt(out_dir / "panel.pt")
    g = load_pt(out_dir / "graph_snapshots.pt")
    bh = p["bar_hours"]
    gidx, gcoins = g["graph_index"], g["graph_coins"]
    r = p["r_ds"].numpy()[:, gidx].astype(np.float64)
    shock = p["shock_today"].numpy()[:, gidx]
    valid = p["shock_today_mask"].numpy()[:, gidx]
    active = p["active"].numpy()[:, gidx]
    sector_id, sector_names, sectors = load_sectors(cfg, gcoins)
    params = network_params(cfg, bh, g["undirected_threshold"])
    snap_t = g["snapshot_t"].numpy()
    logger.info(f"[{res}] network features for {len(snap_t)} snapshots × {len(gcoins)} coins")
    X, comm, glob = compute_network_features(r, shock, valid, active, snap_t, sector_id, params)
    times = pd.to_datetime(p["times"])
    glob.insert(0, "time", times[snap_t])
    glob.insert(1, "split", np.array(["outside", *SPLITS])[p["split_id"].numpy()[snap_t] + 1])
    torch.save({"snapshot_t": torch.from_numpy(snap_t), "times": [str(x) for x in times[snap_t]],
                "graph_coins": gcoins, "graph_index": gidx, "feature_names": NODE_FEATURES,
                "X": torch.from_numpy(X), "X_mask": torch.from_numpy(np.isfinite(X)),
                "community": torch.from_numpy(comm), "sector_id": torch.from_numpy(sector_id),
                "sector_names": sector_names, "params": params.__dict__, "created_utc": utc_now(),
                "usage": "at bar t use the latest snapshot with snapshot_t <= t; NaN where undefined (X_mask)"},
               out_dir / "network_features.pt")
    write_csv(glob, cfg.reports_dir / f"network_timeseries_{res}.csv")

    ev = load_pt(out_dir / "tgn_events.pt")
    cas = cascade_baseline(cfg, res, p, ev, shock, valid, active)
    del ev
    out = {"res": res, "cascade": cas, "glob": glob}
    if res == "1h":
        rk, meta = rankings(cfg, p, gidx, gcoins, sectors, r, shock, valid, active)
        out["rankings"], out["rankings_meta"] = rk, meta
    return out


def run(cfg: HourlyConfig, refresh: bool = False) -> None:
    results = [build_resolution(cfg, res) for res in cfg.datasets.build_resolutions]
    cas = pd.DataFrame([row for r in results for row in r["cascade"]])
    write_csv(cas, cfg.reports_dir / "cascade_baseline.csv")
    rk = next((r for r in results if "rankings" in r), None)
    if rk is not None:
        base = cfg.path(cfg.paths.processed_dir) / cfg.dataset_version
        write_csv(rk["rankings"], base / "rankings.csv")
        write_json(rk["rankings_meta"], base / "rankings.json")
    pd.set_option("display.width", 220, "display.max_columns", 30)
    piv = cas[cas["split"] != "train"].pivot_table(index=["resolution", "split", "k_hours"], columns="model",
                                                    values="auprc")
    print("Cascade baseline AUPRC (val/test):")
    print(piv.round(3).to_string())
    for r in results:
        gl = r["glob"]
        print(f"\n[{r['res']}] network series (mean by split):")
        print(gl.groupby("split")[["mean_corr", "tree_length", "modularity", "nmi_sector", "density",
                                   "total_spillover", "mean_shock_lift"]].mean().round(3).to_string())
    if rk is not None:
        print("\nRankings (test window):", rk["rankings_meta"])
        print(rk["rankings"].round(3).to_string(index=False))
