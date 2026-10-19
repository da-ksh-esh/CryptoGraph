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
    drivers: dict = field(default_factory=dict)            # "COIN|h" -> drivers (older runs: only the alerts')
    context: pd.DataFrame = field(default_factory=pd.DataFrame)   # real values per coin (context.csv)


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
    alerts = json.loads((run_dir / "alerts.json").read_text(encoding="utf-8"))
    dp = run_dir / "drivers.json"
    drivers = (json.loads(dp.read_text(encoding="utf-8")) if dp.exists()
               else {f"{a['coin']}|{a['horizon_h']}": a.get("drivers", []) for a in alerts})
    return Prediction(run_dir, summary, pd.read_csv(run_dir / "risk.csv"), alerts,
                      opt("shock_sources.csv"), opt("persistence.csv"),
                      {k: v for k, v in metrics.items() if not isinstance(v, (list, dict))}, drivers,
                      opt("context.csv"))


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


# --- event explorer (serving/events.py runs) ---------------------------------------------------------------------

def latest_events_dir(runs_dir: Path | str = PROJECT_ROOT / "runs") -> Path | None:
    """The newest run of the events task (contains events.json)."""
    runs_dir = Path(runs_dir)
    if not runs_dir.exists():
        return None
    runs = sorted(p for p in runs_dir.iterdir() if p.is_dir() and (p / "events.json").exists())
    return runs[-1] if runs else None


def load_events_index(run_dir: Path | str) -> dict:
    return json.loads((Path(run_dir) / "events.json").read_text(encoding="utf-8"))


def load_event(run_dir: Path | str, event_id: int) -> dict:
    ev = json.loads((Path(run_dir) / f"event_{event_id}.json").read_text(encoding="utf-8"))
    ev["risk"] = pd.DataFrame(ev["risk"])
    ev["model_vs_outcome"] = pd.DataFrame(ev["model_vs_outcome"])
    return ev


def load_sectors(root: Path = PROJECT_ROOT) -> dict[str, str]:
    p = Path(root) / "data" / "external" / "sectors.csv"
    return dict(pd.read_csv(p)[["coin", "sector"]].values) if p.exists() else {}


def circle_layout(coins: list[str], sector_of: dict[str, str]) -> dict[str, tuple[float, float]]:
    """Coins on a circle, grouped by sector (then by name), starting at the top and going clockwise: every coin keeps
    its place in every view and every frame."""
    order = sorted(coins, key=lambda c: (sector_of.get(c, "other"), c))
    n = max(len(order), 1)
    return {c: (float(np.cos(np.pi / 2 - 2 * np.pi * i / n)), float(np.sin(np.pi / 2 - 2 * np.pi * i / n)))
            for i, c in enumerate(order)}


NODE_STATES = ("shocking now", "shocked earlier in the window", "not yet", "inactive")


def node_states(frames: list[dict], n_coins: int) -> list[list[str]]:
    """Per frame and coin: shocking now, shocked earlier in the window, not yet, or inactive (not trading)."""
    seen = np.zeros(n_coins, bool)
    out = []
    for f in frames:
        now = np.zeros(n_coins, bool)
        now[f["shocked"]] = True
        act = np.zeros(n_coins, bool)
        act[f["active"]] = True
        out.append([NODE_STATES[3] if not act[c] else NODE_STATES[0] if now[c] else NODE_STATES[1] if seen[c]
                    else NODE_STATES[2] for c in range(n_coins)])
        seen |= now
    return out


def outcome_table(risk: pd.DataFrame, horizon: int, sectors: list[str] | None = None) -> pd.DataFrame:
    """The model's risk per coin at one horizon next to what happened, riskiest first."""
    t = risk[risk["horizon_h"] == horizon].sort_values(["risk", "score"], ascending=False)
    if sectors:
        t = t[t["sector"].isin(sectors)]
    t = t.assign(outcome=t["actual"].map({1.0: "shocked", 0.0: "no shock"}).fillna("not observed"),
                 drivers=[", ".join(f"{d['input']} ({'+' if d['attribution'] > 0 else '−'})" for d in ds)
                          for ds in t["drivers"]])
    cols = ["coin", "sector", "risk", "score", "lift", "outcome", "source", "drivers"]   # score orders equal risks
    return t[[c for c in cols if c in t]].reset_index(drop=True)


# --- plain language (the dashboard is for readers without a data-science background) -----------------------------

# (label, colour, lower bound of the lift over a typical coin)
RISK_LEVELS = (("Much higher than usual", "#d62728", 1.5), ("Higher than usual", "#ff7f0e", 1.15),
               ("About normal", "#9aa5b1", 0.85), ("Lower than usual", "#1f77b4", 0.0))

FEATURE_PLAIN = {
    "log_return": "price change in the last hour",
    "rv_24h": "how jumpy its price was over the last day",
    "rv_72h": "how jumpy its price was over the last 3 days",
    "log_volume_ds": "trading volume",
    "volume_rel_168h": "trading volume compared with its past week",
    "log_hours_listed": "how long it has been trading",
    "network: strength": "how strongly it moves with other coins",
    "network: eigenvector": "how central it is in the market",
    "network: mst_degree": "how many coins it is most closely linked to",
    "network: community_size_frac": "the size of its group of closely linked coins",
    "network: dy_pagerank": "its role in spreading price swings",
    "network: dy_net": "whether it tends to spread or absorb price swings",
    "network: shock_pagerank": "how often its big moves are followed by others",
    "network: shock_out_lift": "how often other coins follow its big moves",
    "memory + recent events": "its recent history of big moves",
}
EVENT_PLAIN = {
    "lag1_corr": "how closely the two coins moved together an hour apart",
    "contemp_corr": "how closely the two coins move together",
    "src_return_scaled": "the size of the move that started it",
    "src_vol_z": "how unusual the triggering coin's swings were",
    "src_shock": "that the triggering coin is in a big move",
}


def risk_level(lift: float | None) -> tuple[str, str]:
    """(label, colour) for a coin's risk compared with a typical coin (lift = risk / typical risk)."""
    if lift is None or not np.isfinite(lift):
        return "Unknown", "#9aa5b1"
    for label, colour, lo in RISK_LEVELS:
        if lift >= lo:
            return label, colour
    return RISK_LEVELS[-1][:2]


def horizon_label(h: int) -> str:
    return f"{h} hour" + ("" if h == 1 else "s")


def plain_driver(d: dict, coin: str, source: str, context: pd.DataFrame | None = None) -> str:
    """One driver (from integrated gradients) in words, e.g. "JUP: trading volume compared with its past week (raises
    the risk)". With the prediction's context.csv, the coin's real value is added where it reads naturally, e.g.
    "JUP: trading at 14.6× its usual volume (raises the risk)"."""
    side, _, name = d["input"].partition(": ")
    if side == "event":
        what = f"{EVENT_PLAIN.get(name, name)} (from {source} to {coin})"
    else:
        who = coin if side.startswith("receiver") else source
        what = f"{who}: {real_value(name, who, context) or FEATURE_PLAIN.get(name, name)}"
    return f"{what} ({'raises' if d['attribution'] > 0 else 'lowers'} the risk)"


def real_value(feature: str, coin: str, context: pd.DataFrame | None) -> str | None:
    """The coin's actual value of an input in words, or None when it is unknown or has no natural reading."""
    if context is None or context.empty or coin not in set(context["coin"]):
        return None
    c = context.set_index("coin").loc[coin]
    ok = lambda k: k in c and pd.notna(c[k]) and np.isfinite(c[k])   # noqa: E731
    if feature == "volume_rel_168h" and ok("volume_multiple"):
        return f"trading at {c['volume_multiple']:.1f}× its usual volume"
    if feature == "rv_24h" and ok("vol_multiple"):
        return f"price {c['vol_multiple']:.1f}× as jumpy as usual over the last day"
    if feature == "rv_72h" and ok("vol72_multiple"):
        return f"price {c['vol72_multiple']:.1f}× as jumpy as usual over the last 3 days"
    if feature == "log_return" and ok("ret_1h"):
        return f"moved {c['ret_1h']:+.1%} in the last hour"
    if feature == "memory + recent events" and ok("shocks_24h"):
        n = int(c["shocks_24h"])
        return f"{n} big move{'s' if n != 1 else ''} in the last 24 hours (its recent history)"
    return None


def shock_sentence(src: dict, typical: dict | None, recent_shocks: int | None = None) -> str:
    """What just happened, in words: one source shock, how many coins moved with it (its type comes from
    analysis.persistence, so the breadth cut-offs live in one place), how many big moves the coin made in the last
    24 hours, and how long such shocks usually last."""
    move = "jumped" if src["direction"] == "pump" else "dropped"
    others = f"{src['breadth']:.0%} of the other coins"
    if src["breadth"] <= 0:
        together = "on its own; no other coin moved sharply in the same hour"
    elif src["type"] == "idiosyncratic":
        together = f"mostly on its own (only {others} moved sharply in the same hour)"
    else:
        together = f"together with {others}" + (", a market-wide move" if src["type"] == "market-wide" else "")
    out = f"**{src['coin']}** {move} sharply, about **{abs(src['z']):.1f} times** its usual hourly move, {together}."
    if recent_shocks is not None and recent_shocks > 1:
        out += f" It has made {recent_shocks} big moves in the last 24 hours, counting this one."
    if typical:
        out += (f" Shocks like this have usually calmed down within about "
                f"**{typical['median_hours_to_normal']:.0f} hours**.")
    return out


def about_text(facts: dict | None, k_hours: list[int]) -> str:
    """The "How it works" explanation, with every number read from the prediction run's data facts (older runs
    without them get the same text without the numbers)."""
    f = facts or {}
    thr = f.get("shock_threshold")
    size = f" (more than {thr:.2f} times its usual hourly swing)" if thr else ""
    scope = (f" across {f['n_coins']} coins since {pd.Timestamp(f['start']):%B %Y}"
             if f.get("n_coins") and f.get("start") else "")
    horizon = (f"within {min(k_hours)} to {max(k_hours)} hours" if len(k_hours) > 1
               else f"within {horizon_label(k_hours[0])}" if k_hours else "soon after")
    return ("**What is a big move?** An hourly price change much larger than the coin's usual swings over the past "
            f"week{size}, up or down.\n\n"
            f"**How does the model work?** It watches every big move{scope} and remembers how coins reacted to each "
            "other. When a coin makes a big move, it estimates for every other coin the chance of a big move of its "
            f"own {horizon}. Its scores are turned into chances using past data the model was not trained on.\n\n"
            "**What do the reasons mean?** For each coin the model lists the inputs that changed its score the most, "
            "compared with an average situation. They explain the model's reasoning, not what will happen in the "
            "market.")


def test_period_phrase(facts: dict | None) -> str:
    """ "2024–2026 data" from the run's recorded test period, or plain "data" when it is unknown."""
    test = ((facts or {}).get("periods") or {}).get("test")
    if not test:
        return "data"
    a, b = pd.Timestamp(test[0]).year, pd.Timestamp(test[1]).year
    return f"{a} data" if a == b else f"{a}–{b} data"


def crisis_verdict(hits: int, n: int, expected: float, share_shocked: float) -> tuple[str, str]:
    """(kind, sentence) on how the model's top picks did against random picks: kind is good / neutral / weak."""
    if share_shocked >= 0.95:
        return "neutral", ("Almost every coin made a big move, so any pick would have been right: this crisis does not "
                           "show whether the model is useful.")
    if hits >= expected + 1:
        return "good", "The model did clearly better than picking coins at random."
    if hits > expected - 1:
        return "neutral", "The model did about as well as picking coins at random."
    return "weak", "The model did worse than picking coins at random here."
