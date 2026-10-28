"""Live scanning: extend the hourly data past the research period with Binance's newest bars and score the latest
shocks with the final TGN, once per closed hour (`cryptograph-live`, configs/serving/live_1h.yaml).

One cycle (`run_cycle`):
  1. fetch    closed 1h bars after the research data end, from the public REST API (data/sources/binance_rest.py),
              for each coin's current Binance symbol. Bars from earlier cycles are cached (rest_1h.parquet), so only
              the missing hours are requested. The research raw table plus these bars become the live raw table.
  2. rebuild  the research pipeline's own clean and build stages (1h only) on the extended window, into dataset version
              `v2_live` (data/processed/v2_live/1h/). The research files (data/interim/hourly, data/processed/v2,
              reports/) are only read.
  3. network  node features: snapshots up to the research data end are copied from v2 (each snapshot uses trailing
              windows only), every later one is computed with the research stage's parameters.
  4. verify   everything fitted (seasonality, scalers, shock threshold, graph threshold) is fitted on the train rows,
              which do not change, so v2_live must reproduce v2 exactly on the research period: the same inputs,
              shocks, splits and shock events on every bar before its last `tail_grace_bars` (where a short gap or a
              one-bar spike is only resolved once the next bar is known). A mismatch stops the cycle.
  5. predict  the serving task (serving/predict.py) on v2_live, as of the newest bar with shock events; runs go to
              runs/live/, apart from the research runs.
  6. alerts   the `alerts_top` coins with the highest lift per horizon join runs/live/alert_log.csv; every alert is
              scored as its window closes (did the coin make a big move within k hours?) against the share of the other
              scored coins that did in the same hours, i.e. what a random pick would have got (`track_record`).
Bars after the research period have split -1: they are never used for training, model selection or calibration.
runs/live/status.json describes the last cycle for the dashboard.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from pydantic import BaseModel, ConfigDict

from ..common.logger import get_logger
from ..data.config import PROJECT_ROOT, HourlyConfig
from ..data.sources.binance_rest import DEFAULT_BASE_URLS, BinanceRestKlines, SymbolError, utc_now
from ..data.storage import load_pt, write_json, write_parquet

logger = get_logger(__name__)

HOUR = pd.Timedelta(hours=1)
RES = "1h"
LIVE_CONFIG_PATH = "configs/serving/live_1h.yaml"
ALERT_COLUMNS = ["issued_at", "issued_bar", "run_id", "coin", "horizon_h", "risk", "lift", "base_rate", "source",
                 "due_at", "receivers", "status", "first_shock_at", "chance", "resolved"]


class LiveConfig(BaseModel):
    """configs/serving/live_1h.yaml (paths relative to the project root)."""
    model_config = ConfigDict(extra="forbid")

    data_config: str = "configs/data/hourly.yaml"         # the research data config: universe, features, splits
    reference_version: str = "v2"                         # research dataset that the live data must reproduce
    dataset_version: str = "v2_live"                      # where the extended data is written
    interim_dir: str = "data/interim/hourly_live"         # live raw/master tables and stage reports
    predict_config: str = "configs/experiment/predict_tgn_1h.yaml"
    runs_dir: str = "runs/live"                           # live prediction runs, alert log, status
    rest_base_urls: list[str] = list(DEFAULT_BASE_URLS)
    max_gap_days: int = 90                                # a longer gap is bridged by the archive fetch stage instead
    tail_grace_bars: int = 3                              # last research bars that may change once the next is known
    alerts_top: int = 3                                   # coins per horizon logged as alerts each cycle
    cycle_minute: int = 2                                 # --loop: start each cycle this many minutes past the hour


class LiveDataMismatch(RuntimeError):
    """The live dataset does not reproduce the research dataset on the research period."""


class CycleBusy(RuntimeError):
    """Another live cycle is running (runs/live/.lock)."""


def project_path(p: str | Path) -> Path:
    """A path relative to the project root (absolute paths stay as they are)."""
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_live_config(path: str | Path | None = None) -> LiveConfig:
    p = project_path(path or LIVE_CONFIG_PATH)
    return LiveConfig.model_validate(yaml.safe_load(p.read_text(encoding="utf-8")) or {}) if p.exists() \
        else LiveConfig()


def newest_closed_hour(now: pd.Timestamp | None = None) -> pd.Timestamp:
    """Open time of the newest hourly bar that has closed by `now` (UTC): the bar opened at 13:00 closes at 14:00."""
    now = utc_now() if now is None else pd.Timestamp(now)
    if now.tzinfo is not None:
        now = now.tz_convert("UTC").tz_localize(None)
    return now.floor("h") - HOUR


def data_configs(lc: LiveConfig, end: pd.Timestamp) -> tuple[HourlyConfig, HourlyConfig]:
    """(research config, live config). The live config is the research config with the window extended to `end`, the
    dataset version `lc.dataset_version`, its own interim and reports folders and the 1h resolution only."""
    raw = yaml.safe_load(project_path(lc.data_config).read_text(encoding="utf-8"))
    ref = HourlyConfig.model_validate(raw)
    if ref.dataset_version != lc.reference_version:
        raise ValueError(f"{lc.data_config} builds {ref.dataset_version}, not the reference {lc.reference_version}")
    live = dict(raw)
    live["window"] = {**raw["window"], "end": pd.Timestamp(end).strftime("%Y-%m-%dT%H:%M:%SZ")}
    live["dataset_version"] = lc.dataset_version
    live["paths"] = {**raw.get("paths", {}), "interim_dir": lc.interim_dir, "reports_dir": f"{lc.interim_dir}/reports"}
    live["datasets"] = {**raw.get("datasets", {}), "build_resolutions": [RES]}
    return ref, HourlyConfig.model_validate(live)


def prepare_live_dirs(ref: HourlyConfig, live: HourlyConfig) -> None:
    """Copy the research universe files the live stages read (the universe audit is reused, never re-run)."""
    for a, b in ((ref.interim_dir / "universe_v2.json", live.interim_dir / "universe_v2.json"),
                 (ref.reports_dir / "universe_audit_v2.csv", live.reports_dir / "universe_audit_v2.csv"),
                 (ref.reports_dir / "binance_continuity.csv", live.reports_dir / "binance_continuity.csv")):
        b.parent.mkdir(parents=True, exist_ok=True)
        if not b.exists() or b.read_bytes() != a.read_bytes():
            shutil.copyfile(a, b)


def fetch_new_bars(ref: HourlyConfig, live: HourlyConfig, client, end: pd.Timestamp, max_gap_days: int) -> dict:
    """Bring the live raw table up to `end`: the research raw table plus every closed bar after it (REST, cached).

    Each coin is fetched under the last symbol of its Binance history (configs/data/hourly.yaml); a coin whose history
    is closed, or whose symbol the API refuses, gets no new bars (it is inactive from then on)."""
    from ..pipelines.stages.hourly_fetch import hourly_universe

    ref_raw = pd.read_parquet(ref.interim_dir / "raw_1h.parquet")
    ref_end = ref_raw.index.get_level_values("time").max()
    if end - ref_end > pd.Timedelta(days=max_gap_days):
        raise RuntimeError(f"the research data ends {ref_end}, {(end - ref_end).days} days before {end}: extend it with "
                           "the archive fetch stage first (more than max_gap_days of hourly bars by REST)")
    cache_path = live.interim_dir / "rest_1h.parquet"
    cache = pd.read_parquet(cache_path) if cache_path.exists() else ref_raw.iloc[:0]
    cache = cache[cache.index.get_level_values("time") > ref_end]
    coins, hist, _ = hourly_universe(live)
    have = cache.reset_index().groupby("coin")["time"].max().to_dict() if len(cache) else {}
    new, failures = [], {}
    for coin in coins:
        seg = hist[coin][-1]
        if seg.until is not None:                                    # symbol history closed: no live trading
            continue
        start = max(have.get(coin, ref_end), ref_end) + HOUR
        if start > end:
            continue
        try:
            kl = client.klines(seg.symbol, start, end)
        except SymbolError as e:
            failures[coin] = str(e)[:200]
            logger.warning(f"{coin}: {seg.symbol} refused by the API ({e}); no new bars")
            continue
        if len(kl):
            new.append(kl.assign(symbol=seg.symbol, coin=coin).set_index("coin", append=True))
    if new:
        cache = pd.concat([cache, *[x[cache.columns] for x in new]])
        cache = cache[~cache.index.duplicated(keep="last")].sort_index()
        write_parquet(cache, cache_path)
    tail = cache[cache.index.get_level_values("time") <= end]
    raw_path = live.interim_dir / "raw_1h.parquet"
    if new or not raw_path.exists():
        write_parquet(pd.concat([ref_raw, tail]).sort_index(), raw_path)
    t = tail.index.get_level_values("time")
    return {"reference_end": str(ref_end), "data_end": str(t.max() if len(t) else ref_end),
            "new_bars": int(sum(len(x) for x in new)), "live_bars_total": int(len(tail)),
            "coins_with_live_bars": int(tail.index.get_level_values("coin").nunique()) if len(tail) else 0,
            "failures": failures}


def rebuild(live: HourlyConfig, log_path: Path) -> dict[str, float]:
    """The research clean and build stages on the live config; their printed tables go to `log_path`."""
    from ..pipelines.stages import hourly_build, hourly_clean

    timings = {}
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as fh, contextlib.redirect_stdout(fh):
        for name, stage in (("clean", hourly_clean), ("build", hourly_build)):
            t0 = time.perf_counter()
            stage.run(live)
            timings[f"{name}_s"] = round(time.perf_counter() - t0, 1)
    return timings


def extend_network_features(ref: HourlyConfig, live: HourlyConfig, recompute_from: int) -> dict:
    """network_features.pt (and rankings) of the live version: research snapshots taken before bar `recompute_from`
    are copied, later ones are computed with the research stage's parameters. One copied snapshot is recomputed as a
    check that the computation reproduces the research values."""
    from ..network.features import NODE_FEATURES, compute_network_features
    from ..pipelines.stages.network_build import load_sectors, network_params

    ref_dir, out_dir = ref.processed_dir(RES), live.processed_dir(RES)
    old = load_pt(ref_dir / "network_features.pt")
    p, g = load_pt(out_dir / "panel.pt"), load_pt(out_dir / "graph_snapshots.pt")
    gidx, gcoins = g["graph_index"], list(g["graph_coins"])
    if list(old["graph_coins"]) != gcoins:
        raise LiveDataMismatch("graph coins differ from the research network features")
    snap_t, old_t = g["snapshot_t"].numpy(), old["snapshot_t"].numpy()
    n_keep = int((old_t < recompute_from).sum())
    if not np.array_equal(snap_t[:n_keep], old_t[:n_keep]):
        raise LiveDataMismatch("the graph snapshot grid differs from the research one")
    r = p["r_ds"].numpy()[:, gidx].astype(np.float64)
    shock, valid, active = (p[k].numpy()[:, gidx] for k in ("shock_today", "shock_today_mask", "active"))
    sector_id, sector_names, _ = load_sectors(live, gcoins)
    params = network_params(live, p["bar_hours"], g["undirected_threshold"])
    todo = snap_t[n_keep:]
    X, comm, _ = compute_network_features(r, shock, valid, active, todo, sector_id, params)
    chk, _, _ = compute_network_features(r, shock, valid, active, old_t[n_keep - 1: n_keep], sector_id, params)
    ref_X = old["X"].numpy()[n_keep - 1]
    same = bool(np.allclose(chk[0], ref_X, rtol=1e-6, atol=1e-8, equal_nan=True))
    if not same:
        raise LiveDataMismatch(f"recomputing research snapshot {int(old_t[n_keep - 1])} does not reproduce its network "
                               f"features (max diff {np.nanmax(np.abs(chk[0] - ref_X)):.3g})")
    X_all = np.concatenate([old["X"].numpy()[:n_keep], X])
    comm_all = np.concatenate([old["community"].numpy()[:n_keep], comm])
    times = pd.to_datetime(p["times"])
    torch.save({"snapshot_t": torch.from_numpy(snap_t), "times": [str(x) for x in times[snap_t]],
                "graph_coins": gcoins, "graph_index": gidx, "feature_names": NODE_FEATURES,
                "X": torch.from_numpy(X_all), "X_mask": torch.from_numpy(np.isfinite(X_all)),
                "community": torch.from_numpy(comm_all), "sector_id": torch.from_numpy(sector_id),
                "sector_names": sector_names, "params": params.__dict__, "created_utc": str(utc_now()),
                "copied_from": f"{ref.dataset_version} snapshots before bar {recompute_from}",
                "usage": "at bar t use the latest snapshot with snapshot_t <= t; NaN where undefined (X_mask)"},
               out_dir / "network_features.pt")
    for f in ("rankings.csv", "rankings.json"):                  # test-window rankings: explanation reference only
        src = ref_dir.parent / f
        if src.exists():
            shutil.copyfile(src, out_dir.parent / f)
    return {"snapshots_copied": n_keep, "snapshots_computed": int(len(todo)), "network_check_snapshot": int(old_t[n_keep - 1])}


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Exact equality, NaN equal to NaN."""
    if a.shape != b.shape:
        return False
    if a.is_floating_point():
        return bool(torch.equal(torch.isnan(a), torch.isnan(b)) and torch.equal(torch.nan_to_num(a), torch.nan_to_num(b)))
    return bool(torch.equal(a, b))


def verify_against_reference(ref_dir: Path, live_dir: Path, grace: int) -> dict[str, Any]:
    """The live dataset must reproduce the research dataset on the research period (every bar before its last
    `grace` bars): fitted thresholds, seasonality, splits, raw and scaled inputs, shocks and the shock-event stream.
    Raises LiveDataMismatch naming the failed checks; returns the checks and how much was compared."""
    a, b = load_pt(ref_dir / "panel.pt"), load_pt(live_dir / "panel.pt")
    T = len(a["times"])
    n = T - grace
    checks = {
        "times": list(b["times"][:T]) == list(a["times"]),
        "coins_and_features": list(a["coins"]) == list(b["coins"]) and list(a["feature_names"]) == list(b["feature_names"]),
        "shock_threshold": a["shock_threshold"] == b["shock_threshold"],
        "split_id": _same(a["split_id"], b["split_id"][:T]),
        "seasonality": json.loads((ref_dir / "seasonality.json").read_text(encoding="utf-8"))
        == json.loads((live_dir / "seasonality.json").read_text(encoding="utf-8")),
    }
    for k in ("X_raw", "X", "shock_today", "shock_today_mask", "active", "shock_z"):
        checks[k] = _same(a[k][:n], b[k][:n])
    ga, gb = load_pt(ref_dir / "graph_snapshots.pt"), load_pt(live_dir / "graph_snapshots.pt")
    checks["graph_threshold"] = ga["undirected_threshold"] == gb["undirected_threshold"]
    ea, eb = load_pt(ref_dir / "tgn_events.pt")["full"], load_pt(live_dir / "tgn_events.pt")["full"]
    m = int(np.searchsorted(ea.t.numpy(), n))
    checks["shock_events"] = all(_same(getattr(ea, k)[:m], getattr(eb, k)[:m]) for k in ("src", "dst", "t", "msg"))
    tail_changed = int((~torch.isclose(a["X"][n:], b["X"][n:T], equal_nan=True)).flatten(1).any(1).sum())
    failed = [k for k, ok in checks.items() if not ok]
    if failed:
        raise LiveDataMismatch(f"the live data differs from {ref_dir.parent.name} on the research period: {failed}")
    return {"checks": checks, "bars_compared": n, "events_compared": m, "research_bars": T,
            "live_bars": len(b["times"]) - T, "events_total": int(len(eb.t)), "tail_bars_changed": tail_changed}


def predict_latest(lc: LiveConfig, as_of: str | None = None) -> Path:
    """The serving task (serving/predict.py) on the live dataset, tracked under runs/live/."""
    from ..pipelines._experiment import load_experiment
    from ..training.runner import run_experiment

    cfg = load_experiment(lc.predict_config)
    cfg["name"] = "live_" + cfg["name"].removeprefix("predict_")
    cfg.setdefault("predict", {}).update({"data_version": lc.dataset_version, "live": True})
    if as_of:
        cfg["predict"]["as_of"] = as_of
    out = run_experiment(cfg, seed=cfg.get("seed", 0), device="cpu", runs_dir=project_path(lc.runs_dir))
    return Path(out["run_dir"])


# --- alert log and track record ---------------------------------------------------------------------------------------

def read_alert_log(path: Path) -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame(columns=ALERT_COLUMNS)
    log = pd.read_csv(path)
    log["resolved"] = log["resolved"].fillna(False).astype(bool)
    return log


def new_alerts(run_dir: Path, top: int) -> pd.DataFrame:
    """Each horizon's `top` coins by lift over its base rate (by risk without base rates) in one prediction run."""
    risk = pd.read_csv(Path(run_dir) / "risk.csv")
    s = json.loads((Path(run_dir) / "summary.json").read_text(encoding="utf-8"))
    issued = pd.Timestamp(s["as_of"])
    rows = []
    for kh, g in risk.groupby("horizon_h"):
        key = "lift" if "lift" in g and g["lift"].notna().any() else "risk"
        receivers = ";".join(sorted(g["coin"]))
        for r in g.sort_values([key, "risk"], ascending=False).head(top).to_dict("records"):
            rows.append({"issued_at": str(issued), "issued_bar": int(s["as_of_bar"]), "run_id": Path(run_dir).name,
                         "coin": r["coin"], "horizon_h": int(kh), "risk": float(r["risk"]), "lift": r.get("lift"),
                         "base_rate": r.get("base_rate"), "source": r["source"],
                         "due_at": str(issued + int(kh) * HOUR), "receivers": receivers, "status": "pending",
                         "first_shock_at": None, "chance": None, "resolved": False})
    return pd.DataFrame(rows, columns=ALERT_COLUMNS)


def score_alerts(log: pd.DataFrame, shock: np.ndarray, active: np.ndarray, times, coins: list[str]) -> pd.DataFrame:
    """Status of every unresolved alert from the shocks known so far (shock, active: (T, N) over `coins`).

    hit     the coin made a big move within (issued, issued + k] (known as soon as it happens)
    miss    the window has passed without one
    pending neither yet
    Once the whole window is known the alert is `resolved` and gets `chance`: the share of the other coins scored at the
    same hour (`receivers`, active then) that made a big move in the same window, i.e. a random pick's hit rate."""
    log = log.copy()
    pos = {t: i for i, t in enumerate(pd.DatetimeIndex(times))}
    col = {c: j for j, c in enumerate(coins)}
    last = len(pos) - 1
    for idx in log.index[~log["resolved"].astype(bool)]:
        a = log.loc[idx]
        i, j = pos.get(pd.Timestamp(a["issued_at"])), col.get(a["coin"])
        if i is None or j is None:
            continue
        k = int(a["horizon_h"])
        hi = min(i + k, last)
        hits = np.flatnonzero(shock[i + 1: hi + 1, j])
        done = i + k <= last
        log.loc[idx, "status"] = "hit" if len(hits) else ("miss" if done else "pending")
        log.loc[idx, "first_shock_at"] = str(pd.DatetimeIndex(times)[i + 1 + hits[0]]) if len(hits) else None
        if done:
            rx = [col[c] for c in str(a["receivers"]).split(";") if c in col]
            ok = [c for c in rx if active[i, c]]
            moved = shock[i + 1: i + k + 1][:, ok].any(axis=0) if ok else np.zeros(0, bool)
            log.loc[idx, "chance"] = float(moved.mean()) if len(moved) else np.nan
            log.loc[idx, "resolved"] = True
    return log


TRACK_COLUMNS = ["horizon_h", "alerts", "hits", "hit_rate", "expected_by_chance", "chance_rate", "times_chance"]


def track_record(log: pd.DataFrame) -> pd.DataFrame:
    """Per horizon (and overall): resolved alerts, how many came true, and how many random picks would have.
    No rows (but the columns) while nothing is resolved."""
    done = log[log["resolved"].astype(bool)]
    rows = []
    for kh, g in [*done.groupby("horizon_h"), ("all", done)]:
        if not len(g):
            continue
        hits = int((g["status"] == "hit").sum())
        expected = float(pd.to_numeric(g["chance"], errors="coerce").sum())
        rows.append({"horizon_h": kh, "alerts": int(len(g)), "hits": hits, "hit_rate": hits / len(g),
                     "expected_by_chance": expected, "chance_rate": expected / len(g),
                     "times_chance": hits / expected if expected > 0 else np.nan})
    return pd.DataFrame(rows, columns=TRACK_COLUMNS)


def update_alert_log(log_path: Path, run_dir: Path | None, top: int, panel: dict, gidx: list[int]) -> pd.DataFrame:
    """Add the alerts of `run_dir` (once per prediction hour) and score every unresolved alert with `panel`."""
    log = read_alert_log(log_path)
    if run_dir is not None:
        add = new_alerts(run_dir, top)
        if len(add) and not (log["issued_at"].astype(str) == add["issued_at"].iloc[0]).any():
            log = pd.concat([log, add], ignore_index=True) if len(log) else add
    sh = (panel["shock_today"] & panel["shock_today_mask"]).numpy()[:, gidx]
    log = score_alerts(log, sh, panel["active"].numpy()[:, gidx], panel["times"], list(panel["graph_coins"]))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log.to_csv(log_path, index=False)
    return log


# --- one cycle ----------------------------------------------------------------------------------------------------------

@contextlib.contextmanager
def _cycle_lock(path: Path, stale_after_s: float = 3 * 3600):
    """One cycle at a time: a second `cryptograph-live` stops at once (CycleBusy). A lock older than
    `stale_after_s` is taken to be left over from a crash."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and time.time() - path.stat().st_mtime < stale_after_s:
        raise CycleBusy(f"another live cycle is running ({path.read_text(encoding='utf-8')}); "
                        f"delete {path} if it crashed")
    path.write_text(f"pid {os.getpid()} since {utc_now()} UTC", encoding="utf-8")
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


def _built_end(live_dir: Path) -> str | None:
    meta = live_dir / "metadata.json"
    if not meta.exists() or not (live_dir / "network_features.pt").exists():
        return None
    return json.loads(meta.read_text(encoding="utf-8")).get("window_end")


def run_cycle(lc: LiveConfig | None = None, now=None, client=None, predict: bool = True) -> dict[str, Any]:
    """Fetch, rebuild, verify, predict and score alerts once; returns (and writes) runs/live/status.json.
    Raises CycleBusy, without touching status.json, while another cycle runs."""
    lc = lc or load_live_config()
    runs = project_path(lc.runs_dir)
    with _cycle_lock(runs / ".lock"):
        return _cycle(lc, now, client, predict, runs)


def _cycle(lc: LiveConfig, now, client, predict: bool, runs: Path) -> dict[str, Any]:
    t0 = time.perf_counter()
    status: dict[str, Any] = {"started_utc": str(utc_now()), "status": "running", "dataset_version": lc.dataset_version}
    try:
        end = newest_closed_hour(now)
        status["target_end"] = str(end)
        ref, live = data_configs(lc, end)
        live_dir = live.processed_dir(RES)
        prepare_live_dirs(ref, live)
        status["fetch"] = fetch_new_bars(ref, live, client or BinanceRestKlines(lc.rest_base_urls), end,
                                         lc.max_gap_days)
        data_end = pd.Timestamp(status["fetch"]["data_end"])
        live = data_configs(lc, data_end)[1]                      # the window ends at the last bar that exists
        status["rebuilt"] = _built_end(live_dir) != str(data_end)
        if status["rebuilt"]:
            status["timings"] = rebuild(live, runs / "stages.log")
            T_ref = len(load_pt(ref.processed_dir(RES) / "panel.pt")["times"])
            t1 = time.perf_counter()
            status["network"] = extend_network_features(ref, live, T_ref - lc.tail_grace_bars)
            status["timings"]["network_s"] = round(time.perf_counter() - t1, 1)
            status["verify"] = verify_against_reference(ref.processed_dir(RES), live_dir, lc.tail_grace_bars)
            meta_p = live_dir / "metadata.json"
            write_json(json.loads(meta_p.read_text(encoding="utf-8")) | {
                "window_end": str(data_end), "live_verified_utc": str(utc_now()), "live_verify": status["verify"]},
                meta_p)
        run_dir, last = None, _latest_live_run(runs)
        if predict and (status["rebuilt"] or last is None or _run_data_end(last) != str(data_end)):
            t1 = time.perf_counter()
            run_dir = predict_latest(lc)
            status.setdefault("timings", {})["predict_s"] = round(time.perf_counter() - t1, 1)
        panel = load_pt(live_dir / "panel.pt")
        gidx = load_pt(live_dir / "graph_snapshots.pt")["graph_index"]
        log = update_alert_log(runs / "alert_log.csv", run_dir, lc.alerts_top, panel, gidx)
        rd = run_dir or _latest_live_run(runs)
        summary = json.loads((rd / "summary.json").read_text(encoding="utf-8")) if rd else {}
        status |= {"status": "ok", "data_end": str(data_end), "run_id": rd.name if rd else None,
                   "as_of": summary.get("as_of"), "source_coins": summary.get("source_coins"),
                   "alerts_logged": int(len(log)), "alerts_resolved": int(log["resolved"].astype(bool).sum())}
    except Exception as e:
        status |= {"status": "failed", "error": f"{type(e).__name__}: {e}"[:500]}
        raise
    finally:
        status["finished_utc"] = str(utc_now())
        status["duration_s"] = round(time.perf_counter() - t0, 1)
        write_json(status, runs / "status.json")
    return status


def _latest_live_run(runs: Path) -> Path | None:
    from .predict import PREDICT_FILES

    if not runs.exists():
        return None
    done = sorted(p for p in runs.iterdir() if p.is_dir() and all((p / f).exists() for f in PREDICT_FILES))
    return done[-1] if done else None


def _run_data_end(run_dir: Path) -> str | None:
    s = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    return s.get("data_end")
