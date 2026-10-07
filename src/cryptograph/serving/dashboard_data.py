"""Data for the Streamlit dashboard (dashboard/app.py): reads prediction runs (serving/predict.py) and the network
artifacts, and shapes them for display. No model is run here; the dashboard only shows `cryptograph-predict` output.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.config import PROJECT_ROOT
from .predict import PREDICT_FILES


@dataclass
class Prediction:
    run_dir: Path
    summary: dict
    risk: pd.DataFrame
    alerts: list[dict]
    sources: pd.DataFrame = field(default_factory=pd.DataFrame)
    persistence: pd.DataFrame = field(default_factory=pd.DataFrame)
    model_metrics: dict = field(default_factory=dict)


def prediction_runs(runs_dir: Path | str = PROJECT_ROOT / "runs") -> list[Path]:
    """Finished prediction runs, newest first."""
    runs_dir = Path(runs_dir)
    if not runs_dir.exists():
        return []
    return sorted((p for p in runs_dir.iterdir() if p.is_dir() and all((p / f).exists() for f in PREDICT_FILES)),
                  reverse=True)


def load_prediction(run_dir: Path | str) -> Prediction:
    run_dir = Path(run_dir)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    opt = lambda f: pd.read_csv(run_dir / f) if (run_dir / f).exists() else pd.DataFrame()   # noqa: E731
    target_metrics = run_dir.parent / summary["target_run"] / "metrics.json"
    metrics = json.loads(target_metrics.read_text(encoding="utf-8")) if target_metrics.exists() else {}
    return Prediction(run_dir, summary, pd.read_csv(run_dir / "risk.csv"),
                      json.loads((run_dir / "alerts.json").read_text(encoding="utf-8")),
                      opt("shock_sources.csv"), opt("persistence.csv"),
                      {k: v for k, v in metrics.items() if not isinstance(v, (list, dict))})


def risk_matrix(risk: pd.DataFrame) -> pd.DataFrame:
    """coin x horizon (hours) risk, coins ordered by their mean lift over the base rates (by mean risk without
    base rates; the longest horizon would otherwise decide the order)."""
    m = risk.pivot(index="coin", columns="horizon_h", values="risk")
    key = risk.pivot(index="coin", columns="horizon_h", values="lift") if "lift" in risk else m
    return m.loc[key.mean(axis=1).sort_values(ascending=False, kind="stable").index]


def drivers_frame(alert: dict) -> pd.DataFrame:
    d = pd.DataFrame(alert.get("drivers", []))
    if d.empty:
        return pd.DataFrame(columns=["input", "attribution", "direction"])
    return d.iloc[d["attribution"].abs().argsort()[::-1]].reset_index(drop=True)


def typical_duration(persistence: pd.DataFrame, shock_type: str, direction: str) -> dict | None:
    """Median and quartiles of hours until volatility is normal again, for one shock type x direction."""
    if persistence.empty:
        return None
    h = persistence[(persistence["type"] == shock_type) & (persistence["direction"] == direction)]
    return None if h.empty else h.iloc[0].to_dict()


def model_test_metrics(model_metrics: dict, k_hours: list[int]) -> pd.DataFrame:
    """The served model's own test AUPRC and positive rate per horizon (from its training run)."""
    rows = [{"horizon_h": k, "test_auprc": model_metrics.get(f"test_auprc_k{k}h"),
             "positive_rate": model_metrics.get(f"test_posrate_k{k}h")} for k in k_hours]
    df = pd.DataFrame(rows).dropna()
    if len(df):
        df["lift"] = df["test_auprc"] / df["positive_rate"]
    return df


def load_rankings(version: str = "v2", root: Path = PROJECT_ROOT) -> pd.DataFrame:
    p = Path(root) / "data" / "processed" / version / "rankings.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def graph_at(snapshots: dict, as_of_bar: int) -> tuple[pd.DataFrame, int | None]:
    """Edges (a, b, weight) of the latest correlation snapshot at or before `as_of_bar`, and its bar."""
    st = np.asarray(snapshots["snapshot_t"])
    i = int(np.searchsorted(st, as_of_bar, side="right")) - 1
    if i < 0:
        return pd.DataFrame(columns=["a", "b", "weight"]), None
    coins = list(snapshots["graph_coins"])
    ei = np.asarray(snapshots["undirected_edge_index"][i])
    w = np.asarray(snapshots["undirected_edge_weight"][i], dtype=float)
    keep = ei[0] < ei[1] if ei.size else np.zeros(0, bool)               # one row per undirected edge
    return pd.DataFrame({"a": [coins[x] for x in ei[0][keep]], "b": [coins[x] for x in ei[1][keep]],
                         "weight": w[keep]}), int(st[i])


def load_graph_at(summary: dict, root: Path = PROJECT_ROOT) -> tuple[pd.DataFrame, int | None]:
    from ..data.storage import load_pt

    p = Path(root) / "data" / "processed" / summary.get("data_version", "v2") / summary.get("resolution", "1h") \
        / "graph_snapshots.pt"
    if not p.exists():
        return pd.DataFrame(columns=["a", "b", "weight"]), None
    return graph_at(load_pt(p), int(summary["as_of_bar"]))


def layout(coins: list[str], edges: pd.DataFrame, seed: int = 0) -> dict[str, tuple[float, float]]:
    """Force-directed positions (fixed seed, so the picture is stable between reloads)."""
    import networkx as nx

    g = nx.Graph()
    g.add_nodes_from(coins)
    g.add_weighted_edges_from(edges[["a", "b", "weight"]].itertuples(index=False))
    return {k: (float(v[0]), float(v[1])) for k, v in nx.spring_layout(g, seed=seed, weight="weight").items()}
