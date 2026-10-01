"""Hourly Stage 3 — features, labels, splits, train-only scaling, graphs and datasets (1h and 2h).

For each resolution in `datasets.build_resolutions`, reads data/interim/hourly/master_<res>.parquet
and writes data/processed/v2/<res>/:
  panel.pt              features (raw, scaled+clipped, masks), targets, shock labels, split labels
  lstm_index_<split>.pt (t, coin) samples for PanelWindowDataset (windows sliced on the fly)
  graph_snapshots.pt    undirected and directed edges every `snapshot_step_hours`
  tgn_events.pt         shock-triggered TemporalData with per-event contagion labels — only if the event
                        count is <= datasets.tgn_max_events (otherwise tgn_event_count.json and a checkpoint)
  seasonality.json, scalers.pkl, splits.json, metadata.json, data_dictionary.md
and reports/hourly_<res>_sample_counts.csv, reports/hourly_<res>_label_balance.csv.
"""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import torch

from ...common.logger import get_logger
from ...data.config import PROJECT_ROOT, HourlyConfig
from ...data.datasets.indexed import sample_mask_times, window_index
from ...data.features.intraday import SeasonalityProfile, bars, build_intraday_features, prune_redundant
from ...data.features.scaling import RobustScalerSet
from ...data.features.targets import forward_realized_vol, forward_returns, next_day_shock, shock_indicator, shock_z
from ...data.graphs.correlation import directed_edges, pairwise_corr, undirected_edges
from ...data.storage import hash_tree, library_versions, utc_now, write_csv, write_json
from ...data.universe_hourly import _ts
from .hourly_fetch import hourly_universe

logger = get_logger(__name__)
SPLITS = ("train", "val", "test")


def _t(a) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(a))


def future_any(flag: np.ndarray, valid: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """any(flag[t+1..t+k]) and all(valid[t+1..t+k]) for every t (T, N); False/invalid where the window runs out."""
    T = flag.shape[0]
    cs = np.vstack([np.zeros((1, flag.shape[1]), np.int64), np.cumsum(flag, axis=0)])
    cv = np.vstack([np.zeros((1, valid.shape[1]), np.int64), np.cumsum(valid, axis=0)])
    anyf = np.zeros_like(flag, dtype=bool)
    allv = np.zeros_like(valid, dtype=bool)
    t = np.arange(T - k)
    anyf[t] = (cs[t + k + 1] - cs[t + 1]) > 0
    allv[t] = (cv[t + k + 1] - cv[t + 1]) == k
    return anyf & allv, allv


def build_resolution(cfg: HourlyConfig, res: str) -> dict:
    bh = 1 if res == "1h" else 2
    bar = pd.Timedelta(hours=bh)
    out_dir = cfg.processed_dir(res)
    out_dir.mkdir(parents=True, exist_ok=True)
    coins, _, _ = hourly_universe(cfg)
    model_coins = [c for c in coins if c not in cfg.universe.event_coins]
    gcoins = coins if cfg.datasets.include_event_coins_in_graph else model_coins
    gidx = [coins.index(c) for c in gcoins]

    master = pd.read_parquet(cfg.interim_dir / f"master_{res}.parquet")
    cal = pd.date_range(_ts(cfg.window.start), _ts(cfg.window.end), freq=f"{bh}h", name="time")
    full = pd.MultiIndex.from_product([cal, coins], names=["time", "coin"])
    master = master.reindex(full)
    T, N = len(cal), len(coins)

    # split label per bar (0/1/2, -1 outside)
    split_id = np.full(T, -1, np.int8)
    for i, s in enumerate(SPLITS):
        a, b = getattr(cfg.splits, s)
        split_id[(cal >= pd.Timestamp(a)) & (cal < pd.Timestamp(b) + pd.Timedelta(days=1))] = i
    train_rows = split_id == 0

    # seasonality (train only, model coins)
    close = master["close"].unstack("coin").reindex(columns=coins)
    vol = master["volume"].unstack("coin").reindex(columns=coins)
    r = np.log(close).diff()
    profile = SeasonalityProfile.fit(r[model_coins], vol[model_coins], train_rows, bh)
    write_json(profile.to_json(), out_dir / "seasonality.json")

    # features
    f = cfg.features
    X_all, times, names_all, r_ds = build_intraday_features(master, coins, profile, f.rv_windows_hours,
                                                            f.range_window_hours, f.volume_window_hours,
                                                            f.listing_cap_hours)
    kept, dropped = prune_redundant(X_all[:, [coins.index(c) for c in model_coins]], names_all, train_rows,
                                    f.redundancy_threshold)
    fi_all = {n: i for i, n in enumerate(names_all)}
    X_raw = X_all[:, :, [fi_all[n] for n in kept]]
    fnames = kept
    X_mask = np.isfinite(X_raw)
    active = close.notna().to_numpy()
    logger.info(f"[{res}] features kept {kept}; dropped {dropped}")

    # targets
    t_cfg = cfg.targets
    hz_bars = [bars(h, bh) for h in t_cfg.horizons_hours]
    log_close = np.log(close.to_numpy())
    rr = r.to_numpy()
    y_ret = forward_returns(log_close, hz_bars)
    y_rv = forward_realized_vol(rr, hz_bars)
    y_mask = np.isfinite(y_ret) & np.isfinite(y_rv)
    n_sig = bars(t_cfg.shock_vol_window_hours, bh)
    sigma = r_ds.rolling(n_sig, min_periods=int(0.8 * n_sig)).std().to_numpy()
    z = shock_z(r_ds.to_numpy(), sigma)
    mi = [coins.index(c) for c in model_coins]
    zt = np.abs(z[train_rows][:, mi])
    zt = zt[np.isfinite(zt)]
    zthr = float(np.round(np.quantile(zt, 1 - t_cfg.shock_target_rate), 2))
    s_today, s_valid = shock_indicator(z, zthr)
    y_shock, y_shock_mask = next_day_shock(z, zthr)
    logger.info(f"[{res}] shock threshold |z| > {zthr} (train rate {t_cfg.shock_target_rate:.1%} target)")

    # sample masks
    max_h = pd.Timedelta(hours=max(max(t_cfg.horizons_hours), max(t_cfg.contagion_k_hours)))
    L = bars(cfg.datasets.lookback_hours, bh)
    embargo = pd.Timedelta(days=cfg.splits.embargo_days)
    # embargo only at the start of validation and test (train has nothing before it to leak from)
    sample_mask = {s: sample_mask_times(cal, *getattr(cfg.splits, s), bar=bar,
                                        embargo=pd.Timedelta(0) if s == "train" else embargo, lookback_bars=L,
                                        horizon=max_h) for s in SPLITS}

    # train-only scaling
    scalers = RobustScalerSet.fit(X_raw, train_rows, fnames, coins, min_rows=cfg.datasets.scaler_min_train_hours // bh,
                                  fit_dates=(str(cal[train_rows][0]), str(cal[train_rows][-1])))
    X_in = np.nan_to_num(scalers.transform(X_raw, scope=cfg.scaling.scope, clip=cfg.scaling.clip), nan=0.0)
    X_in[~X_mask] = 0.0
    with open(out_dir / "scalers.pkl", "wb") as fh:
        pickle.dump({"per_coin_and_global": scalers, "scope_used": cfg.scaling.scope, "clip": cfg.scaling.clip}, fh)

    # contagion: base rates and rates after a source shock, per split (graph coins)
    kb = [bars(k, bh) for k in t_cfg.contagion_k_hours]
    sb, vb = s_today[:, gidx], s_valid[:, gidx]
    fut = {k: future_any(sb, vb, k) for k in sorted(set(kb))}
    bal = []
    for si, s in enumerate(SPLITS):
        m = sample_mask[s]
        ys, ym = y_shock[m][:, gidx], y_shock_mask[m][:, gidx]
        bal.append({"label": f"shock_next(|z|>{zthr})", "split": s, "n": int(ym.sum()), "positives": int((ys & ym).sum()),
                    "rate": float((ys & ym).sum() / max(ym.sum(), 1))})
        for kh, k in zip(t_cfg.contagion_k_hours, kb):
            anyf, allv = fut[k]
            base = float(anyf[m].sum() / max(allv[m].sum(), 1))
            src = sb[m] & vb[m]
            # pairs (i shocked at t, j != i): fraction where j shocks within k bars
            hits = (src.astype(np.int64)[:, :, None] & anyf[m][:, None, :]).sum() - (src & anyf[m]).sum()
            pairs = (src.astype(np.int64)[:, :, None] & allv[m][:, None, :]).sum() - (src & allv[m]).sum()
            bal.append({"label": f"contagion_k{kh}h", "split": s, "n": int(pairs), "positives": int(hits),
                        "rate": float(hits / max(pairs, 1)), "base_rate_any_coin": base})
    write_csv(pd.DataFrame(bal), cfg.reports_dir / f"hourly_{res}_label_balance.csv")

    # panel
    panel = {
        "times": [str(t) for t in cal], "times_unix": _t(cal.asi8 // 10**9), "bar_hours": bh, "coins": coins,
        "model_coins": model_coins, "graph_coins": gcoins, "feature_names": fnames, "features_dropped": dropped,
        "horizons_hours": t_cfg.horizons_hours, "contagion_k_hours": t_cfg.contagion_k_hours,
        "X_raw": _t(X_raw), "X": _t(X_in.astype(np.float32)), "X_mask": _t(X_mask), "active": _t(active),
        "y_ret": _t(y_ret), "y_rv": _t(y_rv), "y_mask": _t(y_mask), "shock_z": _t(z.astype(np.float32)),
        "shock_today": _t(s_today), "shock_today_mask": _t(s_valid), "y_shock": _t(y_shock),
        "y_shock_mask": _t(y_shock_mask), "shock_threshold": zthr, "split_id": _t(split_id),
        "sample_mask": {s: _t(m) for s, m in sample_mask.items()},
    }
    torch.save(panel, out_dir / "panel.pt")

    # LSTM index (model coins; full window active, targets present)
    counts = []
    y_ok = y_mask.all(-1) & y_shock_mask
    for s in SPLITS:
        idx = window_index(active, y_ok, sample_mask[s], mi, L, cfg.datasets.sample_stride_hours // bh or 1)
        torch.save({"index": _t(idx), "lookback": L, "split": s}, out_dir / f"lstm_index_{s}.pt")
        counts.append({"dataset": "lstm_windows", "split": s, "samples": int(len(idx)),
                       "first": str(cal[idx[:, 0].min()]) if len(idx) else None,
                       "last": str(cal[idx[:, 0].max()]) if len(idx) else None})

    # graph snapshots every step (undirected + directed lag-1), trailing window only
    g = cfg.graph
    W = bars(g.corr_window_hours, bh)
    step = bars(g.snapshot_step_hours, bh)
    snap_t = np.arange(W - 1, T, step)
    rds = r_ds.to_numpy()[:, gidx]
    act_g = active[:, gidx]
    und, dirg, dens = [], [], []
    min_obs = int(g.min_obs_frac * W)
    for t in snap_t:
        win = rds[t - W + 1: t + 1]
        C = pairwise_corr(win, win, min_obs)
        D = pairwise_corr(rds[t - W: t], win, min_obs) if t >= W else np.full_like(C, np.nan)
        np.fill_diagonal(C, np.nan)
        np.fill_diagonal(D, np.nan)
        und.append(undirected_edges(C, act_g[t], g.threshold, g.top_k))
        dirg.append(directed_edges(D, act_g[t], g.directed_lag_threshold))
        na = act_g[t].sum()
        dens.append((und[-1][0].shape[1] / max(na * (na - 1), 1), dirg[-1][0].shape[1] / max(na * (na - 1), 1)))
    torch.save({"snapshot_t": _t(snap_t), "graph_coins": gcoins, "graph_index": gidx, "params": g.model_dump(),
                "undirected_edge_index": [_t(e[0]) for e in und], "undirected_edge_weight": [_t(e[1]) for e in und],
                "directed_edge_index": [_t(e[0]) for e in dirg], "directed_edge_weight": [_t(e[1]) for e in dirg]},
               out_dir / "graph_snapshots.pt")
    for si, s in enumerate(SPLITS):
        m = sample_mask[s][snap_t]
        counts.append({"dataset": "graph_snapshots", "split": s, "samples": int(m.sum()),
                       "first": str(cal[snap_t[m]].min()) if m.any() else None,
                       "last": str(cal[snap_t[m]].max()) if m.any() else None})
    dens = np.array(dens)

    # TGN event count (shock-triggered: each shocked graph coin -> every other active graph coin)
    src_all = sb & vb & act_g
    n_dst = act_g.sum(axis=1) - 1
    per_bar = (src_all.sum(axis=1) * np.maximum(n_dst, 0)).astype(np.int64)
    ev_counts = {s: int(per_bar[split_id == i].sum()) for i, s in enumerate(SPLITS)}
    ev_total = int(sum(ev_counts.values()))
    shocks = {s: int(src_all[split_id == i].sum()) for i, s in enumerate(SPLITS)}
    top_k_opts = {k: int((src_all.sum(axis=1) * np.minimum(np.maximum(n_dst, 0), k)).sum()) for k in (3, 5, 10)}
    tgn_info = {"resolution": res, "events_per_split": ev_counts, "events_total": ev_total, "source_shocks": shocks,
                "top_k_receivers_total": top_k_opts, "max_events": cfg.datasets.tgn_max_events,
                "built": False}
    k_rx = cfg.datasets.tgn_top_k_receivers
    allowed = ev_total <= cfg.datasets.tgn_max_events or (k_rx is not None and top_k_opts.get(k_rx, 0) <= cfg.datasets.tgn_max_events)
    if allowed:
        tgn_info["built"] = True
        build_tgn_stream(out_dir, cal, coins, gidx, src_all, act_g, rds, X_in, fnames, kb, fut, split_id, W, min_obs,
                         k_rx, t_cfg.contagion_k_hours)
        for s in SPLITS:
            counts.append({"dataset": "tgn_events", "split": s, "samples": ev_counts[s], "first": None, "last": None})
    write_json(tgn_info, out_dir / "tgn_event_count.json")
    write_csv(pd.DataFrame(counts), cfg.reports_dir / f"hourly_{res}_sample_counts.csv")

    # splits.json, metadata, dictionary
    write_json({"bar_hours": bh, "embargo_days": cfg.splits.embargo_days, "lookback_bars": L,
                "max_target_horizon_hours": max_h / pd.Timedelta(hours=1),
                "rule": "S.start+embargo <= t, t-(L-1)*bar >= S.start, t+max_horizon <= S.end",
                "splits": {s: {"start": str(getattr(cfg.splits, s)[0]), "end": str(getattr(cfg.splits, s)[1]),
                               "sample_bars": int(sample_mask[s].sum())} for s in SPLITS}}, out_dir / "splits.json")
    files = [p for p in out_dir.iterdir() if p.suffix in {".pt", ".pkl", ".json"} and p.name != "metadata.json"]
    write_json({"created_utc": utc_now(), "dataset_version": cfg.dataset_version, "resolution": res, "seed": cfg.seed,
                "config_fingerprint": cfg.fingerprint(), "libraries": library_versions(),
                "coins": coins, "model_coins": model_coins, "graph_coins": gcoins,
                "features": fnames, "features_dropped_as_redundant": dropped, "shock_threshold": zthr,
                "shape": {"T": T, "N": N, "F": len(fnames)},
                "graph_density_mean": {"undirected": float(np.nanmean(dens[:, 0])), "directed": float(np.nanmean(dens[:, 1]))},
                "tgn": tgn_info, "quote_currency": "USDT",
                "interim_sha256": hash_tree([cfg.interim_dir / f"master_{res}.parquet"], PROJECT_ROOT),
                "processed_sha256": hash_tree(files, PROJECT_ROOT)}, out_dir / "metadata.json")
    write_dictionary(out_dir, res, bh, fnames, dropped, zthr, L, cfg, T, N)
    return {"res": res, "features": fnames, "dropped": dropped, "zthr": zthr, "counts": counts, "balance": bal,
            "tgn": tgn_info, "density": dens.mean(axis=0).tolist()}


def build_tgn_stream(out_dir, cal, coins, gidx, src_all, act_g, rds, X_in, fnames, kb, fut, split_id, W, min_obs,
                     top_k, k_hours) -> None:
    from torch_geometric.data import TemporalData

    li, vi = fnames.index("log_return"), fnames.index(next(n for n in fnames if n.startswith("rv_")))
    ts, ss, ds, msgs, ys, yms = [], [], [], [], [], []
    for t in np.flatnonzero(src_all.any(axis=1)):
        if t < W:
            continue
        win = rds[t - W + 1: t + 1]
        C = pairwise_corr(win, win, min_obs)
        D = pairwise_corr(rds[t - W: t], win, min_obs)
        for i in np.flatnonzero(src_all[t]):
            dst = np.flatnonzero(act_g[t])
            dst = dst[dst != i]
            if top_k is not None and len(dst) > top_k:
                dst = dst[np.argsort(-np.nan_to_num(D[i, dst], nan=-1))[:top_k]]
            for j in dst:
                gi = gidx[i]
                msgs.append([np.nan_to_num(D[i, j]), np.nan_to_num(C[i, j]), X_in[t, gi, li], X_in[t, gi, vi], 1.0])
                ys.append([fut[k][0][t, j] for k in kb])
                yms.append([fut[k][1][t, j] for k in kb])
                ts.append(t); ss.append(i); ds.append(j)
    data = TemporalData(src=torch.tensor(ss), dst=torch.tensor(ds), t=torch.tensor(ts),
                        msg=torch.tensor(np.asarray(msgs, np.float32)))
    data.y = torch.tensor(np.asarray(ys, bool))
    data.y_mask = torch.tensor(np.asarray(yms, bool))
    sid = split_id[np.asarray(ts)]
    parts = {s: data[torch.from_numpy(np.flatnonzero(sid == i))] for i, s in enumerate(SPLITS)}
    torch.save({"full": data, **parts, "msg_names": ["lag1_corr", "contemp_corr", "src_return_scaled", "src_vol_z", "src_shock"],
                "label_k_hours": k_hours, "graph_index": gidx, "t_unit": "bar index into panel.times"},
               out_dir / "tgn_events.pt")


def write_dictionary(out_dir, res, bh, fnames, dropped, zthr, L, cfg, T, N) -> None:
    lines = [f"# Data dictionary — hourly dataset `{cfg.dataset_version}/{res}`", "",
             f"Bars of {bh} hour(s), UTC, {T} bars × {N} coins (30 model coins + event coins LUNC, FTT). Prices in USDT.",
             "Bar index `t` = position in `panel.pt['times']`. Inputs at t use data up to t; targets are strictly after t.",
             "", f"## Features (F = {len(fnames)})", ""]
    defs = {"log_return": "log(C_t / C_{t-1}) (raw, never clipped in X_raw)",
            "log_volume_ds": "log volume minus the train-fitted (day-of-week × hour) seasonal profile",
            "log_hours_listed": "log(1 + hours since first bar), capped at 2 years"}
    for n in fnames:
        d = defs.get(n) or (f"std of deseasonalised returns over the trailing {n.split('_')[1]}" if n.startswith("rv_") else
                            f"Parkinson range volatility over the trailing {n.split('_')[1]} (seasonality removed)" if n.startswith("parkinson") else
                            f"deseasonalised log volume minus its trailing {n.split('_')[2]} median" if n.startswith("volume_rel") else "")
        lines.append(f"- `{n}`: {d}")
    if dropped:
        lines += ["", "Dropped as redundant on train (|Spearman ρ| ≥ %.2f): " % cfg.features.redundancy_threshold
                  + ", ".join(f"`{a}` (ρ={v} with `{b}`)" for a, b, v in dropped)]
    lines += ["", "## panel.pt", "",
              "`X_raw` (T,N,F) unscaled with NaN; `X` robust-scaled per coin (train-fitted), clipped ±%g, NaN→0; `X_mask`; `active` (T,N);" % cfg.scaling.clip,
              f"`y_ret`, `y_rv` (T,N,H) forward log return / realized volatility at {cfg.targets.horizons_hours} hours; `y_mask`;",
              f"`shock_z` = deseasonalised return / trailing {cfg.targets.shock_vol_window_hours}h volatility (to t-1); "
              f"`shock_today` = |z| > {zthr} (threshold set on train for a ~{cfg.targets.shock_target_rate:.0%} rate); `y_shock` = shock at t+1;",
              "`split_id` (T,) 0 train / 1 val / 2 test / -1 outside; `sample_mask[split]` (T,) bars usable as samples (embargo, windows inside the split).",
              "", "## lstm_index_<split>.pt", "",
              f"`index` (S, 2) = (t, coin) with all {L} input bars active; use `cryptograph.data.datasets.PanelWindowDataset(panel.pt, lstm_index_<split>.pt)`.",
              "", "## graph_snapshots.pt", "",
              f"`snapshot_t` (bar indices every {cfg.graph.snapshot_step_hours}h); per snapshot undirected edges (|ρ| ≥ {cfg.graph.threshold} or top-{cfg.graph.top_k}) "
              f"and directed lag-1 edges (ρ(r_i,t−1, r_j,t) > {cfg.graph.directed_lag_threshold}) over the trailing {cfg.graph.corr_window_hours}h of deseasonalised returns; node order = `graph_coins`.",
              "", "## tgn_events.pt (if built — see tgn_event_count.json)", "",
              "TemporalData: `src`, `dst` (graph-node index), `t` (bar index), `msg` (E,5) = lag-1 corr, contemporaneous corr, "
              "source return (scaled), source volatility (scaled), source shock flag; `y` (E,K) = destination shocks within "
              f"{cfg.targets.contagion_k_hours} hours; `y_mask` (E,K).",
              "", "Other files: `seasonality.json`, `scalers.pkl`, `splits.json`, `metadata.json`, `tgn_event_count.json`."]
    (out_dir / "data_dictionary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(cfg: HourlyConfig, refresh: bool = False) -> None:
    results = [build_resolution(cfg, res) for res in cfg.datasets.build_resolutions]
    pd.set_option("display.width", 220, "display.max_columns", 20)
    for r in results:
        print(f"\n===== {r['res']} =====")
        print("features kept:", r["features"])
        print("dropped as redundant:", r["dropped"])
        print(f"shock threshold |z| > {r['zthr']}; mean graph density undirected/directed: "
              f"{r['density'][0]:.2f} / {r['density'][1]:.3f}")
        print(pd.DataFrame(r["counts"]).to_string(index=False))
        print(pd.DataFrame(r["balance"]).to_string(index=False))
        t = r["tgn"]
        print(f"TGN shock-triggered events: {t['events_per_split']} (total {t['events_total']:,}); "
              f"source shocks {t['source_shocks']}; top-k receivers totals {t['top_k_receivers_total']}; built={t['built']}")
    big = [r for r in results if not r["tgn"]["built"]]
    if big:
        print(f"\nCHECKPOINT: the shock-triggered TGN stream exceeds {cfg.datasets.tgn_max_events:,} events for "
              f"{[r['res'] for r in big]}; it was not built. Choose an option (2h resolution or top-k receivers per shock).")
