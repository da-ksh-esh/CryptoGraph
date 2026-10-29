"""Data for the dashboard's story pages (dashboard/app.py): the project's results, read from the report files that the
pipeline and the notebooks write (reports/, data/processed/, runs/). Nothing here retrains or rescores a model; every
reader returns an empty frame (or None) when its file is missing, so the pages degrade instead of failing."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.config import PROJECT_ROOT

K_HOURS = [1, 3, 6, 12, 24]

# model id in the report files -> (label on the page, family) ; the order is the leaderboard's
MODELS = {
    "independent cascade": ("Independent cascade", "network baseline"),
    "gbm_1h_final": ("Gradient boosting", "machine learning"),
    "v2|logreg": ("Logistic regression (4-number rules)", "machine learning"),
    "v3-5m|gbm": ("Boosting, new inputs + 5-minute data", "machine learning"),
    "v3-5m|logreg": ("Logistic regression, new inputs + 5-minute data", "machine learning"),
    "dygformer_1h_final": ("DyGFormer (Transformer)", "temporal graph"),
    "tgn_1h_nomem_final_v3": ("TGN without memory (report)", "temporal graph"),
    "tgn_1h_final_v2": ("TGN, the report's model", "temporal graph"),
    "tgn_1h_v3_final": ("TGN v3, after the report", "temporal graph"),
}


def _csv(root: Path, rel: str) -> pd.DataFrame:
    p = Path(root) / rel
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def _json(path: Path) -> dict | None:
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


# --- headline numbers -------------------------------------------------------------------------------------------------
def dataset_facts(root: Path = PROJECT_ROOT, version: str = "v3", resolution: str = "1h") -> dict:
    """Coins, hours, events and the date range of the processed dataset."""
    base = Path(root) / "data" / "processed" / version / resolution
    meta, ev = _json(base / "metadata.json"), _json(base / "tgn_event_count.json")
    if meta is None:
        return {}
    out = {"coins": len(meta.get("model_coins", [])), "hours": int(meta["shape"]["T"]),
           "features": int(meta["shape"]["F"]), "shock_threshold": meta.get("shock_threshold")}
    if ev:
        out["events"] = int(ev["events_total"])
        out["events_per_split"] = ev["events_per_split"]
    splits = _json(base / "splits.json")
    if splits:
        sp = splits.get("splits", splits)
        out["start"] = sp.get("train", {}).get("start")
        out["end"] = sp.get("test", {}).get("end")
    return out


def fair_scores(root: Path = PROJECT_ROOT) -> pd.DataFrame:
    """Test-period scores of the TGNs and baselines, per event, per coin-hour and for calm coins (fair_report.py)."""
    f = pd.concat([_csv(root, "reports/fair_eval_v2.csv"), _csv(root, "reports/fair_eval_v3.csv")], ignore_index=True)
    return f[f["split"] == "test"].reset_index(drop=True) if not f.empty else f


def headline(root: Path = PROJECT_ROOT) -> dict:
    """The few numbers the home page leads with."""
    out = {}
    f = fair_scores(root)
    for model, key in (("tgn_1h_v3_final", "v3"), ("tgn_1h_final_v2", "v2")):
        r = f[(f["model"] == model) & (f["k_hours"] == 1)] if not f.empty else f
        if len(r):
            r = r.iloc[0]
            out[key] = {"auprc": r["event_auprc"], "random": r["event_random"], "lift": r["event_lift"],
                        "auroc": r["event_auroc"], "rh_lift": r["rh_lift"], "top1": r["top1_precision"]}
    q = _csv(root, "reports/final_15min_test.csv")
    if not q.empty:
        r = q[(q["model"] == "logreg") & (q["k_bars"] == 1)]
        if len(r):
            r = r.iloc[0]
            out["m15"] = {"auprc": r["event_auprc"], "random": r["random"], "lift": r["lift"], "auroc": r["auroc"]}
    return out


# --- results ----------------------------------------------------------------------------------------------------------
def leaderboard(root: Path = PROJECT_ROOT) -> pd.DataFrame:
    """Test AUPRC of every contagion model at every window: model, family, k_hours, auprc, random."""
    rows = []
    c = _csv(root, "reports/cascade_baseline.csv")
    if not c.empty:
        for r in c[(c["resolution"] == "1h") & (c["split"] == "test") & (c["model"] == "independent cascade")].itertuples():
            rows.append(("independent cascade", r.k_hours, r.auprc, r.positive_rate))
    for r in fair_scores(root).itertuples():
        rows.append((r.model, r.k_hours, r.event_auprc, r.event_random))
    t = _csv(root, "reports/final_1h_test.csv")
    for r in t.itertuples():
        rows.append((f"{r.inputs}|{r.model}", r.k_hours, r.event_auprc, r.random))
    df = pd.DataFrame(rows, columns=["id", "k_hours", "auprc", "random"])
    df = df[df["id"].isin(MODELS)].drop_duplicates(["id", "k_hours"], keep="last")
    df["model"] = df["id"].map(lambda m: MODELS[m][0])
    df["family"] = df["id"].map(lambda m: MODELS[m][1])
    df["lift"] = df["auprc"] / df["random"]
    order = {m: i for i, m in enumerate(MODELS)}
    return df.sort_values(["k_hours", "id"], key=lambda s: s.map(order) if s.name == "id" else s).reset_index(drop=True)


def bootstrap_v3_vs_v2(root: Path = PROJECT_ROOT) -> pd.DataFrame:
    """Day-block bootstrap of the v3 TGN against the report's TGN on the test events."""
    return _csv(root, "reports/final_tgn_v3_vs_v2_test.csv")


def fifteen_minutes(root: Path = PROJECT_ROOT) -> pd.DataFrame:
    return _csv(root, "reports/final_15min_test.csv")


def contagion_jump(root: Path = PROJECT_ROOT, resolution: str = "1h") -> pd.DataFrame:
    """After a shock: how often another coin shocks within k hours, against any k-hour window (split, k, rate, base)."""
    b = _csv(root, f"reports/hourly_{resolution}_label_balance.csv")
    if b.empty:
        return b
    c = b[b["label"].str.startswith("contagion")].copy()
    c["k_hours"] = c["label"].str.extract(r"k(\d+)h")[0].astype(int)
    c["lift"] = c["rate"] / c["base_rate_any_coin"]
    return c[["split", "k_hours", "rate", "base_rate_any_coin", "lift"]].reset_index(drop=True)


# --- the network ------------------------------------------------------------------------------------------------------
def network_timeseries(root: Path = PROJECT_ROOT, resolution: str = "1h") -> pd.DataFrame:
    t = _csv(root, f"reports/network_timeseries_{resolution}.csv")
    if not t.empty:
        t["time"] = pd.to_datetime(t["time"])
    return t


def crisis_dates(root: Path = PROJECT_ROOT, start: str = "2020-01-01") -> pd.DataFrame:
    e = _csv(root, "data/external/events.csv")
    if e.empty:
        return e
    e["date"] = pd.to_datetime(e["date"])
    return e[e["date"] >= start].reset_index(drop=True)


def deseasonalised_returns(root: Path = PROJECT_ROOT, version: str = "v3", resolution: str = "1h"):
    """Rush-hour-adjusted hourly returns (T, N) of the graph coins and their times, as notebook 02 uses them."""
    from ..data.storage import load_pt

    p = Path(root) / "data" / "processed" / version / resolution / "panel.pt"
    if not p.exists():
        return None
    panel = load_pt(p)
    if "r_ds" not in panel or "times" not in panel:
        return None
    graph = list(panel.get("graph_coins", panel.get("coins", [])))
    coins = list(panel.get("coins", graph))
    cols = [coins.index(c) for c in graph if c in coins]
    r = panel["r_ds"].numpy()
    r = r[:, cols] if r.shape[1] == len(coins) else r
    return r.astype(float), pd.to_datetime(panel["times"])


def market_mode(r_ds: np.ndarray, rows: np.ndarray) -> dict | None:
    """Eigenvalues of the coins' correlation matrix over the chosen hours, the market factor's share, and the largest
    eigenvalue pure noise could give (Marchenko-Pastur)."""
    x = pd.DataFrame(np.asarray(r_ds)[rows]).dropna(axis=1, thresh=int(0.9 * rows.sum())).dropna()
    if x.shape[0] < 50 or x.shape[1] < 3:
        return None
    eig = np.sort(np.linalg.eigvalsh(np.corrcoef(x.to_numpy().T)))[::-1]
    n, t = x.shape[1], x.shape[0]
    return {"eigenvalues": eig, "coins": n, "hours": t, "share": float(eig[0] / n),
            "noise_bound": float((1 + np.sqrt(n / t)) ** 2)}


# --- the model --------------------------------------------------------------------------------------------------------
PARAM_PARTS = {"memory.gru": "Memory updater (GRU)", "gnn.conv": "Attention over recent events",
               "head": "Prediction head", "time_enc": "Time encoders"}


def parameter_breakdown(run_dir: Path | str) -> pd.DataFrame:
    """Trainable parameters by part of a saved TGN, counting only the parts its settings switch on."""
    import torch

    p = Path(run_dir) / "checkpoint.pt"
    if not p.exists():
        return pd.DataFrame(columns=["part", "parameters"])
    ck = torch.load(p, weights_only=False, map_location="cpu")
    uses_memory = ck.get("model_config", {}).get("use_memory", True)
    counts = Counter()
    for name, v in ck["model_state"].items():
        if not hasattr(v, "numel") or name.endswith(("memory.memory", "last_update", "_assoc")):
            continue                                                     # buffers, not learned
        part = next((label for key, label in PARAM_PARTS.items() if key in name), None)
        if part is None or (not uses_memory and part != "Prediction head"):
            continue
        counts[part] += v.numel()
    return pd.DataFrame(counts.items(), columns=["part", "parameters"])


def head_inputs(run_dir: Path | str) -> int | None:
    import torch

    p = Path(run_dir) / "checkpoint.pt"
    if not p.exists():
        return None
    w = torch.load(p, weights_only=False, map_location="cpu")["model_state"].get("head.0.weight")
    return None if w is None else int(w.shape[1])


def find_run(runs_dir: Path | str, experiment: str, seed: int = 0) -> Path | None:
    runs = sorted(Path(runs_dir).glob(f"*_{experiment}_{seed}_*")) if Path(runs_dir).exists() else []
    runs = [r for r in runs if (r / "checkpoint.pt").exists()]
    return runs[-1] if runs else None


def run_settings(run_dir: Path | str | None) -> dict:
    if run_dir is None or not (Path(run_dir) / "config.resolved.yaml").exists():
        return {}
    import yaml

    return yaml.safe_load((Path(run_dir) / "config.resolved.yaml").read_text()) or {}


# --- explanations and stress tests -------------------------------------------------------------------------------------
def attribution_groups(root: Path = PROJECT_ROOT, events: str = "top alerts") -> pd.DataFrame:
    a = _csv(root, "reports/explain/feature_attributions.csv")
    if a.empty:
        return a
    a = a[a["events"] == events]
    return a.groupby("group", as_index=False)["share_of_abs"].sum().sort_values("share_of_abs", ascending=False)


def top_inputs(root: Path = PROJECT_ROOT, events: str = "top alerts", n: int = 8) -> pd.DataFrame:
    a = _csv(root, "reports/explain/feature_attributions.csv")
    return a[a["events"] == events].nlargest(n, "share_of_abs") if not a.empty else a


def explainer_fidelity(root: Path = PROJECT_ROOT, events: str = "top alerts", kept: int = 3) -> pd.DataFrame:
    f = _csv(root, "reports/explain/fidelity.csv")
    if f.empty:
        return f
    return f[(f["events"] == events) & (f["kept_events"] == kept)].sort_values("characterization")


def stress_tests(root: Path = PROJECT_ROOT) -> pd.DataFrame:
    s = pd.concat([_csv(root, "reports/robustness/stress_tests.csv"), _csv(root, "reports/robustness/stress_tests_pgd.csv")],
                  ignore_index=True)
    if s.empty:
        return s
    base = s[s["perturbation"] == "none"].groupby("model")["auprc_k1h"].first()
    s["change_k1h"] = s["auprc_k1h"] / s["model"].map(base) - 1
    return s.drop_duplicates(["model", "perturbation", "level"], keep="first").reset_index(drop=True)
