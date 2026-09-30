"""Stage 4 — features, targets, splits, train-only scalers, graphs and model datasets.

Writes into data/processed/<dataset_version>/ (plus the legacy
data/processed/crypto_graph_dataset.pt):
  panel.pt, lstm_{train,val,test}.pt, graph_snapshots.pt, tgn_events.pt,
  tgn_node_features.pt, splits.json, scalers.pkl, metadata.json, data_dictionary.md
and reports/label_balance.csv, reports/sample_counts.csv.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ...common.logger import get_logger
from ...data.config import PROJECT_ROOT, DataConfig
from ...data.datasets.lstm import build_windows
from ...data.datasets.splits import SPLITS, SplitPlan
from ...data.features.node_features import FeatureSpec, build_node_features, wide
from ...data.features.scaling import RobustScalerSet
from ...data.features.targets import (contagion_labels, forward_realized_vol, forward_returns, next_day_shock,
                                      shock_indicator, shock_z)
from ...data.graphs.correlation import directed_edges, rolling_corr_mats, undirected_edges
from ...data.graphs.temporal_stream import MSG_NAMES, build_event_arrays, shock_edges, split_events, to_temporal_data
from ...data.legacy import write_legacy_graph_dataset
from ...data.storage import hash_tree, library_versions, sha256_file, utc_now, write_csv, write_json
from . import load_universe

logger = get_logger(__name__)


def _t(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(a))


def run(cfg: DataConfig, refresh: bool = False) -> None:
    out_dir = cfg.processed_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    audit, panels, coins = load_universe(cfg)
    master = pd.read_parquet(cfg.interim_dir / "master.parquet")

    # ---------------- features (all master coins) ----------------
    spec = FeatureSpec(vol_windows=tuple(cfg.features.vol_windows), volume_median_window=cfg.features.volume_median_window,
                       rsi_window=cfg.features.rsi_window, days_listed_cap=cfg.features.days_listed_cap)
    X_raw, dates, fnames = build_node_features(master, coins, spec)
    T, N, F = X_raw.shape
    active = wide(master, "close", coins).notna().to_numpy()
    X_mask = np.isfinite(X_raw)
    logger.info(f"features: T={T} N={N} F={F}")

    # ---------------- targets ----------------
    horizons = cfg.targets.horizons
    log_close = np.log(wide(master, "close", coins).to_numpy())
    r = X_raw[:, :, fnames.index("log_return")].astype(np.float64)
    sigma = X_raw[:, :, fnames.index(f"rv_{cfg.targets.shock_vol_window}")].astype(np.float64)
    y_ret = forward_returns(log_close, horizons)
    y_rv = forward_realized_vol(r, horizons)
    y_mask = np.isfinite(y_ret) & np.isfinite(y_rv)
    z = shock_z(r, sigma)

    # ---------------- splits ----------------
    plan = SplitPlan.from_config(cfg.splits.as_dict(), cfg.splits.embargo_days, cfg.lookback_window, max(horizons))
    day_split = plan.split_of_date(dates)
    sample_mask = {s: plan.sample_mask(dates, s) for s in SPLITS}
    train_days = day_split == "train"

    # shock threshold: set on TRAIN only (config value, rate reported on train)
    zthr = cfg.targets.shock_z
    s_today, s_valid = shock_indicator(z, zthr)
    train_rate = float(s_today[train_days][s_valid[train_days]].mean())
    y_shock, y_shock_mask = next_day_shock(z, zthr)
    logger.info(f"shock |z|>{zthr}: train positive rate {train_rate:.4f}")

    # ---------------- scaling (fit on train dates only) ----------------
    scalers = RobustScalerSet.fit(X_raw, train_days, fnames, coins, fit_dates=(str(dates[train_days][0].date()),
                                                                               str(dates[train_days][-1].date())))
    X_scaled = scalers.transform(X_raw, scope=cfg.scaling.scope)
    X_in = np.nan_to_num(np.clip(X_scaled, -cfg.scaling.clip, cfg.scaling.clip), nan=0.0).astype(np.float32)
    X_in[~X_mask] = 0.0
    with open(out_dir / "scalers.pkl", "wb") as f:
        pickle.dump({"per_coin_and_global": scalers, "scope_used": cfg.scaling.scope, "clip": cfg.scaling.clip}, f)

    # ---------------- graph node set (Panel B [+ event coins]) ----------------
    gcoins = list(panels["panel_B"])
    if cfg.universe.include_event_coins_in_panel_b:
        gcoins += [c for c in cfg.universe.event_coins if c not in gcoins]
    gidx = [coins.index(c) for c in gcoins]
    rb = r[:, gidx]
    g = cfg.graph
    C, D = rolling_corr_mats(rb, g.rolling_window, g.min_obs)
    act_b = active[:, gidx]
    und = [undirected_edges(C[t], act_b[t], g.threshold, g.top_k) for t in range(T)]
    dirg = [directed_edges(D[t], act_b[t], g.directed_lag_threshold, g.directed_top_k) for t in range(T)]
    logger.info(f"graphs: mean undirected edges/day {np.mean([e[0].shape[1] for e in und]):.1f}, "
                f"directed {np.mean([e[0].shape[1] for e in dirg]):.1f}")

    # contagion labels on graph nodes
    sb, vb = s_today[:, gidx], s_valid[:, gidx]
    cont, cont_mask = contagion_labels(sb, vb, cfg.targets.contagion_k)

    # ---------------- panel.pt (all master coins; EDA + reuse) ----------------
    date_str = [d.date().isoformat() for d in dates]
    panel = {
        "dates": date_str, "coins": coins, "feature_names": fnames, "horizons": horizons,
        "panel_A": [c in panels["panel_A"] for c in coins], "panel_B": [c in panels["panel_B"] for c in coins],
        "is_event_coin": [c in cfg.universe.event_coins for c in coins],
        "X_raw": _t(X_raw), "X": _t(X_in), "X_mask": _t(X_mask), "active": _t(active),
        "y_ret": _t(y_ret), "y_rv": _t(y_rv), "y_mask": _t(y_mask),
        "shock_z": _t(z.astype(np.float32)), "shock_today": _t(s_today), "shock_today_mask": _t(s_valid),
        "y_shock": _t(y_shock), "y_shock_mask": _t(y_shock_mask), "shock_threshold": zthr,
        "day_split": list(day_split), "sample_mask": {s: _t(m) for s, m in sample_mask.items()},
    }
    torch.save(panel, out_dir / "panel.pt")

    # ---------------- LSTM windows (Panel B nodes) ----------------
    counts = []
    for s in SPLITS:
        w = build_windows(X_in, X_mask, active, y_ret, y_rv, y_shock, y_shock_mask, sample_mask[s], gidx,
                          cfg.lookback_window)
        payload = {k: _t(v) for k, v in w.items()}
        payload.update({"coins": coins, "feature_names": fnames, "horizons": horizons, "dates": date_str,
                        "split": s, "lookback": cfg.lookback_window})
        torch.save(payload, out_dir / f"lstm_{s}.pt")
        counts.append({"dataset": "lstm_windows", "split": s, "samples": int(len(w["X"])),
                       "positives_shock": int(w["y_shock"].sum()),
                       "first_date": date_str[w["day"].min()] if len(w["day"]) else None,
                       "last_date": date_str[w["day"].max()] if len(w["day"]) else None})

    # ---------------- graph snapshots (Panel B nodes) ----------------
    Xb, Mb = X_in[:, gidx], X_mask[:, gidx]
    snap = {
        "dates": date_str, "coins": gcoins, "feature_names": fnames, "horizons": horizons,
        "X": _t(Xb), "X_mask": _t(Mb), "active": _t(act_b),
        "y_ret": _t(np.nan_to_num(y_ret[:, gidx])), "y_rv": _t(np.nan_to_num(y_rv[:, gidx])),
        "y_mask": _t(y_mask[:, gidx]), "y_shock": _t(y_shock[:, gidx]), "y_shock_mask": _t(y_shock_mask[:, gidx]),
        "contagion": _t(cont), "contagion_mask": _t(cont_mask), "contagion_k": cfg.targets.contagion_k,
        "undirected_edge_index": [_t(e[0]) for e in und], "undirected_edge_weight": [_t(e[1]) for e in und],
        "directed_edge_index": [_t(e[0]) for e in dirg], "directed_edge_weight": [_t(e[1]) for e in dirg],
        "day_split": list(day_split), "sample_mask": {s: _t(m) for s, m in sample_mask.items()},
        "graph_params": g.model_dump(),
    }
    torch.save(snap, out_dir / "graph_snapshots.pt")
    for s in SPLITS:
        counts.append({"dataset": "graph_snapshots", "split": s, "samples": int(sample_mask[s].sum()),
                       "positives_shock": int((y_shock[:, gidx] & y_shock_mask[:, gidx])[sample_mask[s]].sum()),
                       "first_date": date_str[np.flatnonzero(sample_mask[s])[0]],
                       "last_date": date_str[np.flatnonzero(sample_mask[s])[-1]]})

    # ---------------- legacy crypto_graph_dataset.pt (existing format) ----------------
    legacy_path = write_legacy_graph_dataset(
        cfg.path(cfg.paths.legacy_graph_dataset),
        features=_t(Xb), targets=_t(np.nan_to_num(y_ret[:, gidx]).astype(np.float32)), active_mask=_t(act_b),
        edge_indices=snap["undirected_edge_index"], edge_weights=snap["undirected_edge_weight"],
        timestamps=date_str, asset_names=gcoins, feature_names=fnames, horizons=horizons,
        metadata={"num_timesteps": T, "num_assets": len(gcoins), "num_features": F,
                  "rolling_window": g.rolling_window, "correlation_threshold": g.threshold,
                  "dataset_version": cfg.dataset_version,
                  "note": "features are train-fitted robust-scaled + clipped; targets NaN->0 (use active_mask)"},
    )

    # ---------------- TGN event stream ----------------
    li, lv = fnames.index("log_return"), fnames.index("rv_7")
    if cfg.tgn.event_source == "shock":
        ev_edges = [shock_edges(sb[t], act_b[t], D[t]) for t in range(T)]
    else:
        ev_edges = dirg
    ev = build_event_arrays(ev_edges, C, Xb[:, :, li], Xb[:, :, lv], sb)
    logger.info(f"TGN events ({cfg.tgn.event_source}): {len(ev['t'])} total")
    full = to_temporal_data(ev, dates)
    parts = split_events(full, day_split)
    torch.save({"full": full, **parts, "msg_names": MSG_NAMES, "coins": gcoins, "dates": date_str,
                "t_unit": "day index since window.start_date (t_unix: UTC seconds)"}, out_dir / "tgn_events.pt")
    torch.save({"t": _t(np.arange(T)), "x": _t(Xb), "x_mask": _t(Mb), "active": _t(act_b), "dates": date_str,
                "coins": gcoins, "feature_names": fnames}, out_dir / "tgn_node_features.pt")
    for s in SPLITS:
        counts.append({"dataset": "tgn_events", "split": s, "samples": int(parts[s].num_events if parts[s] is not None else 0),
                       "positives_shock": int(parts[s].msg[:, -1].sum().item()) if parts[s] is not None else 0,
                       "first_date": date_str[int(parts[s].t.min())] if parts[s] is not None else None,
                       "last_date": date_str[int(parts[s].t.max())] if parts[s] is not None else None})

    # ---------------- reports: sample counts and label balance ----------------
    write_csv(pd.DataFrame(counts), cfg.reports_dir / "sample_counts.csv")
    bal = []
    for s in SPLITS:
        m = sample_mask[s]
        ys, ym = y_shock[m][:, gidx], y_shock_mask[m][:, gidx]
        bal.append({"label": f"shock_next(|z|>{zthr})", "split": s, "n": int(ym.sum()),
                    "positives": int((ys & ym).sum()), "rate": float((ys & ym).sum() / max(ym.sum(), 1))})
        for kk, k in enumerate(cfg.targets.contagion_k):
            cm, cl = cont_mask[m][..., kk], cont[m][..., kk]
            src = (sb[m][:, :, None] & cm)
            bal.append({"label": f"contagion_k{k}", "split": s, "n": int(cm.sum()), "positives": int(cl.sum()),
                        "rate": float(cl.sum() / max(cm.sum(), 1)),
                        "rate_given_source_shock": float(cl.sum() / max(src.sum(), 1))})
    bal_df = pd.DataFrame(bal)
    bal_df.loc[len(bal_df)] = {"label": "shock_today_train_rate", "split": "train", "n": int(s_valid[train_days].sum()),
                               "positives": int(s_today[train_days].sum()), "rate": train_rate}
    write_csv(bal_df, cfg.reports_dir / "label_balance.csv")

    # ---------------- splits.json, metadata.json, data_dictionary.md ----------------
    write_json({**plan.to_json(), "sample_days": {s: int(m.sum()) for s, m in sample_mask.items()}},
               out_dir / "splits.json")
    processed_files = [p for p in out_dir.iterdir() if p.suffix in {".pt", ".pkl", ".json"} and p.name != "metadata.json"]
    raw_files = [p for p in cfg.raw_dir.rglob("*") if p.is_file() and p.name != ".gitkeep"]
    meta = {
        "created_utc": utc_now(), "dataset_version": cfg.dataset_version, "seed": cfg.seed,
        "config_fingerprint": cfg.fingerprint(), "config": cfg.model_dump(mode="json"),
        "libraries": library_versions(),
        "universe": {"panel_A": panels["panel_A"], "panel_B": panels["panel_B"], "graph_nodes": gcoins,
                     "master_coins": coins, "event_coins": cfg.universe.event_coins},
        "shape": {"T": T, "N_master": N, "N_graph": len(gcoins), "F": F},
        "shock_threshold": zthr, "shock_train_rate": train_rate,
        "raw_sha256": hash_tree(raw_files, PROJECT_ROOT),
        "interim_sha256": hash_tree(list(cfg.interim_dir.glob("*.parquet")), PROJECT_ROOT),
        "processed_sha256": {**hash_tree(processed_files, PROJECT_ROOT),
                             legacy_path.relative_to(PROJECT_ROOT).as_posix(): sha256_file(legacy_path)},
        "sources": {"yahoo": "yfinance, auto_adjust=False, UTC daily",
                    "coinmetrics": "Coin Metrics Community Data (CC BY-NC 4.0), PriceUSD aligned at empirical lag",
                    "coingecko": "current market-cap ranking snapshot only"},
    }
    write_json(meta, out_dir / "metadata.json")
    write_data_dictionary(out_dir, panel, snap, fnames, horizons, gcoins, cfg)
    print(pd.DataFrame(counts).to_string(index=False))
    print(bal_df.to_string(index=False))


def _shape(x) -> str:
    return str(tuple(x.shape)) if hasattr(x, "shape") else f"list[{len(x)}]"


def write_data_dictionary(out_dir: Path, panel: dict, snap: dict, fnames: list[str], horizons: list[int],
                          gcoins: list[str], cfg: DataConfig) -> None:
    T = len(panel["dates"])
    lines = [
        f"# Data dictionary — dataset `{cfg.dataset_version}`", "",
        "Generated by `cryptograph-data --stage build`. Dates are UTC calendar days; day index `t` = position in "
        f"`dates` (0 = {panel['dates'][0]}, {T - 1} = {panel['dates'][-1]}).", "",
        "Sample convention: a sample dated t uses inputs from days ≤ t and targets strictly after t. "
        "See `splits.json` for split bounds, the 30-day embargo and first/last sample dates.", "",
        "## Features (F = %d, order = last axis)" % len(fnames), "",
        "| # | name | definition (all trailing, data ≤ t) |", "|---|---|---|",
    ]
    defs = {
        "log_return": "log(C_t / C_{t-1}); never clipped in X_raw",
        "rv_7": "std of log returns over t-6..t (≥80% obs)", "rv_14": "std of log returns over t-13..t",
        "rv_30": "std of log returns over t-29..t", "parkinson_7": "sqrt(mean_7[ln(H/L)^2 / (4 ln 2)]) — needs OHLC",
        "garman_klass_7": "sqrt(mean_7[0.5 ln(H/L)^2 − (2ln2−1) ln(C/O)^2]) — needs OHLC",
        "log_volume": "ln(USD volume)", "volume_rel_30": "ln(volume_t / median(volume_{t-30..t-1}))",
        "rsi_14": "Wilder RSI / 100", "macd_norm": "(EMA12 − EMA26) / C_t",
        "mcap_share": "market cap / sum of market caps of master coins with a cap on day t (Coin Metrics; ends at archive date)",
        "log_days_listed": "ln(1 + days since first valid close)",
    }
    lines += [f"| {i} | `{n}` | {defs.get(n, '')} |" for i, n in enumerate(fnames)]
    lines += ["", f"Horizons H = {horizons}.", "",
              "## panel.pt — all master coins", "", "| key | shape | dtype | description |", "|---|---|---|---|"]
    desc = {
        "X_raw": "unscaled features, NaN where unavailable", "X": "robust-scaled (train-fitted), clipped ±%g, NaN→0" % cfg.scaling.clip,
        "X_mask": "feature availability (isfinite(X_raw))", "active": "coin has a close on day t",
        "y_ret": "forward log return log(C_{t+h}/C_t)", "y_rv": "forward realized vol sqrt(mean r_{t+1..t+h}^2)",
        "y_mask": "both regression targets finite", "shock_z": "r_t / rv_30_{t-1}",
        "shock_today": "|shock_z_t| > threshold", "shock_today_mask": "shock_z_t finite",
        "y_shock": "label: shock on day t+1", "y_shock_mask": "label defined",
    }
    for k, v in panel.items():
        if isinstance(v, torch.Tensor):
            lines.append(f"| `{k}` | {_shape(v)} | {str(v.dtype).replace('torch.', '')} | {desc.get(k, '')} |")
    lines += ["", "Also: `dates`, `coins`, `feature_names`, `horizons`, `panel_A`/`panel_B`/`is_event_coin` (per coin), "
              "`day_split` (per day: train/val/test), `sample_mask[split]` (T,) bool, `shock_threshold`.", "",
              f"## graph_snapshots.pt — graph nodes N = {len(gcoins)}: {', '.join(gcoins)}", "",
              "| key | shape | dtype | description |", "|---|---|---|---|"]
    sdesc = {**desc, "y_ret": "forward log return (NaN→0; use y_mask)", "y_rv": "forward realized vol (NaN→0; use y_mask)",
             "contagion": "label[t,i,j,k]: shock at i on t AND shock at j within t+1..t+k", "contagion_mask": "label defined (i≠j, indicators valid)"}
    for k, v in snap.items():
        if isinstance(v, torch.Tensor):
            lines.append(f"| `{k}` | {_shape(v)} | {str(v.dtype).replace('torch.', '')} | {sdesc.get(k, '')} |")
    lines += [f"| `undirected_edge_index` / `_weight` | list[T] of (2,E_t) / (E_t,) | int64 / float32 | "
              f"rolling {cfg.graph.rolling_window}d corr, |ρ| ≥ {cfg.graph.threshold} or top-{cfg.graph.top_k}, symmetric |",
              f"| `directed_edge_index` / `_weight` | list[T] of (2,E_t) / (E_t,) | int64 / float32 | "
              f"i→j if corr(r_i,t−1 , r_j,t) over {cfg.graph.rolling_window}d > {cfg.graph.directed_lag_threshold} |", "",
              "## lstm_{train,val,test}.pt — Panel B coins, one sample per (coin, day)", "",
              "| key | shape | dtype | description |", "|---|---|---|---|",
              f"| `X` | (S, {cfg.lookback_window}, {len(fnames)}) | float32 | scaled+clipped inputs, days t−L+1..t, all active |",
              f"| `X_mask` | (S, {cfg.lookback_window}, {len(fnames)}) | bool | feature availability |",
              f"| `y_ret` / `y_rv` | (S, {len(horizons)}) | float32 | targets (unclipped) |",
              "| `y_shock` | (S,) | bool | shock on t+1 |",
              "| `coin_id` | (S,) | int64 | index into `coins` (panel.pt order) |",
              "| `day` | (S,) | int64 | day index t (date = dates[day]) |", "",
              "## tgn_events.pt / tgn_node_features.pt", "",
              "`full`, `train`, `val`, `test`: `torch_geometric.data.TemporalData` with `src`, `dst` (graph-node index), "
              "`t` (day index, non-decreasing), `t_unix` (UTC seconds), `msg` (E, 5) = "
              f"{MSG_NAMES}. Split by event date using the split bounds (not fractions).", "",
              f"Event source: `{cfg.tgn.event_source}`. " + (
                  "On each day t where node i has a shock (|r_t|/σ30_{t−1} > "
                  f"{cfg.targets.shock_z}), one event i→j is emitted for every other active node j; "
                  "`lag1_corr` is the trailing 30-day corr(r_i,t−1, r_j,t) (0 if not estimable). "
                  "Pair with `contagion` labels in graph_snapshots.pt (shock at j within t+1..t+k)."
                  if cfg.tgn.event_source == "shock" else
                  "Events are the directed lag-1 snapshot edges."), "",
              "`tgn_node_features.pt`: `x` (T, N, F) and `x_mask`, `active` keyed by day index `t` (row t).", "",
              "## Other files", "",
              "- `splits.json` — split bounds, embargo, first/last sample dates, sample-day counts.",
              "- `scalers.pkl` — `RobustScalerSet` (per-coin median/IQR, global fallback, fit dates).",
              "- `metadata.json` — config + fingerprint, library versions, SHA-256 of raw/interim/processed files.",
              "- `../crypto_graph_dataset.pt` — legacy format for `load_processed_dataset` (graph nodes, undirected edges)."]
    (out_dir / "data_dictionary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
