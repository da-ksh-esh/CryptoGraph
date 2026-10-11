"""Per-coin contagion risk after the most recent shocks, with the drivers of each alert (`cryptograph-predict`).

The final TGN scores events "coin i shocks at bar t -> will receiver j shock within k hours". For the latest bars that
have shock events (the newest data in data/processed/v2/<res>/), the whole stream is replayed so memory holds
exactly what evaluation sees, and each receiver coin gets, per horizon k in {1, 3, 6, 12, 24} h:
    risk  = the highest calibrated probability over the recent events that point at it,
    source = the shocking coin of that event.
Scores come from a focal-loss model, so they rank well but are not probabilities. They are calibrated by binned isotonic
regression on the validation-period predictions of the same run (`predictions_val.pt`; the test period is never used)
and made non-decreasing in k, since "shocks within 6 h" includes "shocks within 3 h".
Drivers are integrated gradients on the prediction head (explain/attribution.py) for each coin's top event; TempME is
not used per alert because it is weakly faithful (notebook 04).

Outputs (runs/<run_id>/): risk.csv (coin x horizon), context.csv (real values per coin, for the dashboard's
reasons), drivers.json (drivers of every coin x horizon), alerts.json (the
coin-horizons with the largest lift over their horizon's base rate, with drivers), summary.json, shock_sources.csv and
persistence.csv (type of the source shocks and how long such shocks lasted before). The dashboard reads the latest of
these runs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from ..data.config import PROJECT_ROOT
from ..data.storage import load_pt
from .context import coin_context

PREDICT_FILES = ("risk.csv", "alerts.json", "summary.json")


def recent_positions(t: np.ndarray, as_of: int | None = None, lookback: int = 1) -> tuple[np.ndarray, int]:
    """Stream positions of the events in the `lookback` bars up to `as_of` (default: the latest bar with events).
    Returns (positions, as_of bar)."""
    t = np.asarray(t)
    if as_of is None:
        as_of = int(t[-1])
    else:
        upto = t[t <= as_of]
        if not len(upto):
            raise ValueError(f"no shock events at or before bar {as_of}")
        as_of = int(upto[-1])                                  # latest bar with events at or before the requested one
    return np.flatnonzero((t > as_of - lookback) & (t <= as_of)), as_of


class Calibrator:
    """Monotone map from the model's score to the observed contagion rate, one per horizon (fitted on validation).

    Validation events are sorted by score and cut into equal-count bins of at least `min_bin` events; isotonic
    regression (weighted by bin size) runs on the bin means, and scores are interpolated linearly between bins. Plain
    isotonic regression on single events gives 0% / 100% steps wherever a few events happen to share a label, which is
    exactly the sparse top tail where alerts live; with bins a risk is 0 or 1 only if a whole bin was."""

    def __init__(self, prob: np.ndarray, y: np.ndarray, mask: np.ndarray, min_bin: int = 500):
        from sklearn.isotonic import IsotonicRegression

        self.x, self.fx, self.base_rate = [], [], []
        for k in range(prob.shape[1]):
            m = mask[:, k].astype(bool)
            s, yk = prob[m, k].astype(np.float64), y[m, k].astype(np.float64)
            order = np.argsort(s, kind="stable")
            bins = np.array_split(order, max(1, len(s) // min_bin))
            xs = np.array([s[b].mean() for b in bins])
            ys = np.array([yk[b].mean() for b in bins])
            w = np.array([len(b) for b in bins], float)
            fx = IsotonicRegression(y_min=0.0, y_max=1.0).fit(xs, ys, sample_weight=w).predict(xs)
            self.x.append(xs)
            self.fx.append(fx)
            self.base_rate.append(float(yk.mean()))

    @classmethod
    def from_run(cls, run_dir: Path) -> "Calibrator | None":
        p = Path(run_dir) / "predictions_val.pt"
        if not p.exists():
            return None
        pv = load_pt(p)
        return cls(pv["prob"].float().numpy(), pv["y"].numpy(), pv["mask"].numpy())

    def __call__(self, prob: np.ndarray) -> np.ndarray:
        """(N, K) scores -> (N, K) calibrated risk, non-decreasing in k ("within 6 h" includes "within 3 h")."""
        out = np.stack([np.interp(prob[:, k].astype(np.float64), x, fx)
                        for k, (x, fx) in enumerate(zip(self.x, self.fx))], 1)
        return np.maximum.accumulate(out, axis=1)


def coin_risk(prob: np.ndarray, risk: np.ndarray, src: np.ndarray, dst: np.ndarray, t: np.ndarray,
              coins: list[str], k_hours: list[int], base_rate: list[float] | None = None) -> pd.DataFrame:
    """Long table (coin, horizon): the receiver's highest risk over the given events and the event behind it.
    `row` indexes the given events (used to pick the event to explain)."""
    rows = []
    for j in np.unique(dst):
        ev = np.flatnonzero(dst == j)
        for k, kh in enumerate(k_hours):
            b = ev[np.argmax(risk[ev, k] + 1e-9 * prob[ev, k])]   # ties (isotonic steps) broken by the raw score
            rows.append({"coin": coins[j], "horizon_h": int(kh), "risk": float(risk[b, k]), "score": float(prob[b, k]),
                         "source": coins[src[b]], "event_bar": int(t[b]), "n_events": int(len(ev)), "row": int(b)})
    df = pd.DataFrame(rows)
    if base_rate is not None:
        br = dict(zip(k_hours, base_rate))
        df["base_rate"] = df["horizon_h"].map(br)
        df["lift"] = df["risk"] / df["base_rate"].where(df["base_rate"] > 0)
    return df.sort_values(["horizon_h", "risk"], ascending=[True, False]).reset_index(drop=True)


def top_drivers(attr: list[torch.Tensor], r: int, node_names: list[str], msg_names: list[str],
                top: int = 5) -> list[dict]:
    """The `top` inputs with the largest |attribution| for row r (sign: + raises the risk, - lowers it)."""
    from ..explain.attribution import feature_table

    tab = feature_table([a[r:r + 1] for a in attr], node_names, msg_names).head(top)
    return [{"input": x["input"], "attribution": round(float(x["mean_attribution"]), 6),
             "direction": "raises" if x["mean_attribution"] > 0 else "lowers"} for x in tab.to_dict("records")]


def shock_context(panel: dict, gidx: list[int], as_of: int, sources: list[int], coins: list[str],
                  window: int = 168) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Type of each source shock at `as_of` (notebook 05 definitions) and how long such shocks lasted before.

    Returns (sources: coin, z, direction, breadth, type) and (history per type x direction: n_shocks, median and
    quartiles of hours_to_normal, share normal within 24 h, re-shock and follower rates). History uses only data up
    to `as_of`, and only shocks whose whole `window` (168 h) after the shock is observed by then."""
    from ..analysis.persistence import persistence_table, shock_breadth, shock_type

    T = as_of + 1
    li = list(panel["feature_names"]).index("log_return")
    r = panel["X_raw"][:T, gidx, li].numpy().astype(float)
    z = panel["shock_z"][:T, gidx].numpy().astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        sig = np.where(np.abs(z) > 1e-9, r / z, np.nan)
    sh, va, act = (panel[k][:T, gidx].numpy() for k in ("shock_today", "shock_today_mask", "active"))
    br = shock_breadth(sh & va, act & va)[as_of]
    src = pd.DataFrame([{"coin": coins[c], "z": float(z[as_of, c]), "direction": "crash" if r[as_of, c] < 0 else "pump",
                         "breadth": float(br[c]), "type": str(shock_type(br[c]))} for c in sources])
    ps = persistence_table(r, z, sig, sh, va, act, pd.DatetimeIndex(panel["times"][:T]), coins, horizon=window)
    hist = pd.DataFrame()
    if len(ps):
        ps = ps[ps["t"] + window <= as_of]
        hist = ps.groupby(["type", "direction"]).agg(
            n_shocks=("hours_to_normal", "size"), median_hours_to_normal=("hours_to_normal", "median"),
            q25_hours=("hours_to_normal", lambda x: x.quantile(0.25)),
            q75_hours=("hours_to_normal", lambda x: x.quantile(0.75)),
            share_normal_24h=("hours_to_normal", lambda x: float((x <= 24).mean())),
            reshock_24h=("reshock_24h", "mean"), followers_6h=("followers_6h", "mean")).reset_index()
    return src, hist


def latest_prediction_dir(runs_dir: Path | str = PROJECT_ROOT / "runs", experiment: str | None = None) -> Path | None:
    """The newest finished prediction run (contains all of PREDICT_FILES), optionally of one experiment name."""
    runs = sorted(p for p in Path(runs_dir).glob("*") if p.is_dir() and all((p / f).exists() for f in PREDICT_FILES)
                  and (experiment is None or p.name[16:].rsplit("_", 2)[0] == experiment))
    return runs[-1] if runs else None


def predict_task(cfg: dict, seed: int, device: torch.device, tracker) -> dict[str, Any]:
    from ..explain.attribution import completeness_gap, integrated_gradients
    from ..explain.run import _find_run
    from ..explain.tgn_capture import ExplainableTGN, capture, load_trained

    pc = cfg["predict"]
    target = (Path(pc["target_run"]) if pc.get("target_run")
              else _find_run(pc["target_experiment"], pc.get("target_seed", 0)))
    if not target.is_absolute():
        target = PROJECT_ROOT / target
    tracker.log(f"predicting with {target.name}")
    runner, D = load_trained(target, device="cpu")
    d, coins, K = D["d"], D["coins"], list(D["k_hours"])
    times = pd.DatetimeIndex(D["panel"]["times"])
    t_np, src, dst = d.t.numpy(), d.src.numpy(), d.dst.numpy()

    as_of = None
    if pc.get("as_of"):
        as_of = int(np.searchsorted(times, pd.Timestamp(pc["as_of"]), side="right") - 1)
    pos, as_of = recent_positions(t_np, as_of, pc.get("lookback_hours", 1))
    sel = np.zeros(len(t_np), bool)
    sel[pos] = True
    tracker.log(f"as of {times[as_of]} (data end {times[-1]}): {len(pos):,} events from "
                f"{sorted({coins[s] for s in src[pos]})}; replaying {pos[-1] + 1:,} events")
    cap = capture(runner, sel, end=int(pos[-1]) + 1)
    ex = ExplainableTGN(runner.model)
    with torch.no_grad():
        logits = ex(cap)
    err = float((logits - cap.logits).abs().max())
    assert err < 1e-4, f"captured inputs do not reproduce the model (max abs logit diff {err:.2e})"
    prob = torch.sigmoid(logits).numpy()

    cal = Calibrator.from_run(target)
    if cal is None:
        tracker.log("no predictions_val.pt in the target run: risk = uncalibrated score")
    risk = cal(prob) if cal is not None else np.maximum.accumulate(prob, axis=1)
    p_src, p_dst, p_t = src[cap.pos.numpy()], dst[cap.pos.numpy()], t_np[cap.pos.numpy()]
    table = coin_risk(prob, risk, p_src, p_dst, p_t, coins, K, cal.base_rate if cal is not None else None)

    # drivers: integrated gradients for each coin's top event, per horizon (batched over coins)
    drivers, gap_max = {}, 0.0
    for k, kh in enumerate(K):
        rows = table.loc[table["horizon_h"] == kh, "row"].to_numpy()
        attr, gap = integrated_gradients(ex, cap.subset(torch.from_numpy(rows)), k, steps=pc.get("ig_steps", 64))
        gap_max = max(gap_max, completeness_gap(attr, gap))
        for i, r in enumerate(rows):
            drivers[(int(r), kh)] = top_drivers(attr, i, D["node_names"], D["msg_names"], pc.get("top_drivers", 5))
    table["drivers"] = [drivers[(r, h)] for r, h in zip(table["row"], table["horizon_h"])]
    table["event_time"] = [str(times[b]) for b in table["event_bar"]]

    out = table.drop(columns=["row", "drivers"])
    out.to_csv(tracker.dir / "risk.csv", index=False)
    coin_context(D["panel"], D["graph_index"], as_of, coins).to_csv(tracker.dir / "context.csv", index=False)
    (tracker.dir / "drivers.json").write_text(json.dumps(       # the drivers of every coin x horizon (dashboard)
        {f"{c}|{kh}": dv for c, kh, dv in zip(table["coin"], table["horizon_h"], table["drivers"])},
        default=float), encoding="utf-8")
    # riskiest = largest lift over the horizon's base rate (absolute risk always favours the 24 h horizon)
    rank_by = ["lift", "risk"] if "lift" in table else ["risk"]
    alerts = table.sort_values(rank_by, ascending=False).head(pc.get("n_alerts", 20))
    alert_cols = ["coin", "horizon_h", "risk", "score", "base_rate", "lift", "source", "event_time", "drivers"]
    alert_list = [{c: r[c] for c in alert_cols if c in r} for r in alerts.to_dict("records")]
    summary = {"target_run": target.name, "as_of": str(times[as_of]), "as_of_bar": as_of, "data_end": str(times[-1]),
               "data_version": D["cfg"]["data"]["version"], "resolution": D["cfg"]["data"]["resolution"],
               "hours_since_as_of": int(len(times) - 1 - as_of), "n_events": int(len(pos)),
               "source_coins": sorted({coins[s] for s in p_src}), "k_hours": K, "calibrated": cal is not None,
               "base_rate": cal.base_rate if cal is not None else None, "rebuild_max_abs_logit_diff": err,
               "ig_completeness_max_gap": gap_max}
    if pc.get("shock_context", True):                         # source shock type + typical duration (notebook 05)
        gidx = load_pt(PROJECT_ROOT / "data" / "processed" / D["cfg"]["data"]["version"]
                       / D["cfg"]["data"]["resolution"] / "tgn_events.pt")["graph_index"]
        last = sorted({int(s) for s, tt in zip(p_src, p_t) if tt == as_of})
        src_tab, hist = shock_context(D["panel"], gidx, as_of, last, coins)
        src_tab.to_csv(tracker.dir / "shock_sources.csv", index=False)
        hist.to_csv(tracker.dir / "persistence.csv", index=False)
        for x in src_tab.itertuples():
            h = hist[(hist["type"] == x.type) & (hist["direction"] == x.direction)] if len(hist) else hist
            typical = f"typically back to normal in {h['median_hours_to_normal'].iloc[0]:.0f} h" if len(h) else ""
            tracker.log(f"source {x.coin}: {x.type} {x.direction} (z {x.z:+.1f}, breadth {x.breadth:.0%}) {typical}")
    (tracker.dir / "alerts.json").write_text(json.dumps(alert_list, indent=1, default=float), encoding="utf-8")
    (tracker.dir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")

    for kh in K:
        top = out[out["horizon_h"] == kh].head(3)
        tracker.log(f"k={kh}h: " + ", ".join(f"{c} {r:.1%} (from {s})" for c, r, s in
                                              zip(top["coin"], top["risk"], top["source"])))
    best = alert_list[0] if alert_list else {}
    return {"target_run": target.name, "as_of": summary["as_of"], "n_events": summary["n_events"],
            "calibrated": summary["calibrated"], "top_alert": f"{best.get('coin')} k={best.get('horizon_h')}h",
            "top_alert_risk": float(best.get("risk", float("nan"))), "rebuild_max_abs_logit_diff": err,
            "ig_completeness_max_gap": gap_max}
