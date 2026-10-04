"""Local JSON/data server for the standalone CryptoGraph historical replay UI."""

from __future__ import annotations

import csv
import glob
import json
import sys
from functools import lru_cache
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))
from cryptograph.data.universe import SECTOR_TAXONOMY  # noqa: E402

EVENTS_FILE = REPO_ROOT / "data" / "external" / "events.csv"
DATASET_PATHS = {
    "hourly": REPO_ROOT / "data" / "processed" / "v2" / "1h" / "processed_dataset.pt",
    "daily": REPO_ROOT / "data" / "processed" / "crypto_graph_dataset.pt",
}
BAR_HOURS = {"hourly": 1, "daily": 24}


@lru_cache(maxsize=1)
def get_events() -> list[dict]:
    if not EVENTS_FILE.is_file():
        return []
    with EVENTS_FILE.open("r", encoding="utf-8-sig", newline="") as stream:
        events = list(csv.DictReader(stream))
    for index, event in enumerate(events):
        event["id"] = index
        event["is_crisis"] = str(event.get("is_crisis", "")).strip().lower() == "true"
    return events


@lru_cache(maxsize=2)
def load_dataset(resolution: str) -> dict:
    if resolution not in DATASET_PATHS:
        raise ValueError("Resolution must be 'hourly' or 'daily'.")
    path = DATASET_PATHS[resolution]
    if not path.is_file():
        raise FileNotFoundError(f"The {resolution} processed dataset is missing.")
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    features = bundle["features"].detach().cpu().numpy()
    active = bundle["active_mask"].detach().cpu().numpy().astype(bool)
    timestamps = pd.to_datetime(bundle["timestamps"], utc=True, errors="coerce")
    if len(timestamps) != len(features) or not len(timestamps):
        raise ValueError(f"Invalid timestamps in {resolution} dataset.")
    # Binance hourly rows are keyed by candle open time; features use the completed
    # OHLCV candle, so the interface presents them at candle close.
    if resolution == "hourly":
        timestamps = timestamps + pd.Timedelta(hours=1)
    shocks = None
    if resolution == "hourly" and "shocks" in bundle:
        shocks = bundle["shocks"].detach().cpu().numpy().astype(bool)
    return {
        "features": features,
        "returns": np.nan_to_num(features[:, :, 0].astype(float), nan=0.0, posinf=0.0, neginf=0.0),
        "active": active,
        "shocks": shocks,
        "timestamps": timestamps,
        "assets": [str(asset).replace("-USD", "") for asset in bundle["asset_names"]],
    }


def get_event_catalogue() -> list[dict]:
    available = {}
    for resolution in DATASET_PATHS:
        if DATASET_PATHS[resolution].is_file():
            dataset = load_dataset(resolution)
            available[resolution] = (dataset["timestamps"].min(), dataset["timestamps"].max())
    result = []
    for event in get_events():
        date = pd.Timestamp(event["date"], tz="UTC")
        event_item = {**event, "available": {}}
        for resolution, (start, end) in available.items():
            event_item["available"][resolution] = bool(start.normalize() <= date.normalize() <= end.normalize())
        result.append(event_item)
    return result


def _read_csv_report(name: str) -> list[dict]:
    allowed = {
        "rankings": "rankings.csv",
        "persistence": "persistence_analysis.csv",
        "explanations": "tempme_explanations.csv",
        "anomalies": "tgae_anomalies.csv",
        "robustness": "robustness_degradation.csv",
    }
    if name not in allowed:
        raise ValueError("Unknown research report.")
    path = REPO_ROOT / "reports" / allowed[name]
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    # CSV is the persisted format; convert scalar values so the browser gets
    # usable numbers and booleans rather than formatted strings.
    converted = []
    for row in rows:
        item = {}
        for key, value in row.items():
            if value is None or value == "":
                item[key] = None
            elif value.lower() in {"true", "false"}:
                item[key] = value.lower() == "true"
            else:
                try:
                    number = float(value)
                    item[key] = int(number) if number.is_integer() else number
                except ValueError:
                    item[key] = value
        converted.append(item)
    return converted


@lru_cache(maxsize=1)
def get_forecast_options() -> dict:
    path = DATASET_PATHS["hourly"]
    if not path.is_file():
        return {"available": False, "reason": "Hourly processed dataset is missing."}
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    split_indices = bundle.get("split_indices", {})
    test_indices = [int(value) for value in split_indices.get("test_indices", [])]
    if not test_indices:
        return {"available": False, "reason": "Dataset has no held-out test indices."}
    checkpoints = {}
    for model in ("tgn", "lstm"):
        matches = sorted(glob.glob(str(REPO_ROOT / "runs" / f"*{model}*" / f"best_{model}_checkpoint.pt")))
        if matches:
            checkpoints[model] = matches[-1]
    if not checkpoints:
        return {"available": False, "reason": "No trained TGN or LSTM checkpoint is available."}
    timestamps = bundle["timestamps"]
    horizons = bundle.get("horizons", [1, 3, 6, 12, 24])
    return {
        "available": True,
        "models": list(checkpoints),
        "horizons": [int(value) for value in horizons],
        "test_indices": test_indices,
        "timestamps": [str(timestamps[index]) for index in test_indices],
    }


def run_forecast(model: str, timestep: int, horizon: int, threshold: float, top_k: int) -> dict:
    options = get_forecast_options()
    if not options["available"]:
        raise ValueError(options["reason"])
    if model not in options["models"]:
        raise ValueError("No checkpoint is available for the selected model.")
    if timestep not in options["test_indices"]:
        raise ValueError("Forecast timestep must belong to the held-out test split.")
    if horizon not in options["horizons"]:
        raise ValueError("Unsupported forecast horizon.")
    if not (0.1 <= threshold <= 0.9) or not (1 <= top_k <= 30):
        raise ValueError("Threshold or result count is outside the supported range.")
    from cryptograph.pipelines.predict import predict_contagion

    checkpoint = next(
        (Path(path) for path in sorted(glob.glob(str(REPO_ROOT / "runs" / f"*{model}*" / f"best_{model}_checkpoint.pt")))[-1:]),
        None,
    )
    if checkpoint is None:
        raise ValueError("The selected model checkpoint disappeared.")
    return predict_contagion(
        model_type=model,
        checkpoint_path=str(checkpoint),
        dataset_path=str(DATASET_PATHS["hourly"]),
        network_features_path=str(REPO_ROOT / "data" / "processed" / "v2" / "network_features.pt"),
        timestep=timestep,
        horizon_hours=horizon,
        threshold=threshold,
        top_k=top_k,
        explain=(model == "tgn"),
        device_name="auto",
    )


def _daily_shocks(dataset: dict, anchor: int) -> np.ndarray:
    returns, active = dataset["returns"], dataset["active"]
    abs_returns = np.abs(returns)
    start = max(0, anchor - 365)
    baseline = abs_returns[start:anchor]
    baseline_active = active[start:anchor]
    thresholds = np.full(returns.shape[1], np.nan)
    for node in range(returns.shape[1]):
        values = baseline[:, node][baseline_active[:, node] & np.isfinite(baseline[:, node])]
        if len(values) >= 30:
            thresholds[node] = np.quantile(values, 0.95)
    shocks = abs_returns >= thresholds[None, :]
    shocks[:, ~np.isfinite(thresholds)] = False
    return shocks & active


def _network_edges(dataset: dict, anchor: int, resolution: str) -> list[dict]:
    bar_hours = BAR_HOURS[resolution]
    start = max(0, anchor - int(90 * 24 / bar_hours))
    returns = dataset["returns"][start:anchor].copy()
    active = dataset["active"][start:anchor]
    returns[~active] = np.nan
    if returns.shape[0] < 3:
        return []
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.ma.corrcoef(np.ma.masked_invalid(returns), rowvar=False).filled(0.0)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 0.0)
    pairs: dict[tuple[int, int], float] = {}
    for source in range(corr.shape[0]):
        kept = 0
        for target in np.argsort(np.abs(corr[source]))[::-1]:
            weight = float(abs(corr[source, target]))
            if source == target or weight < 0.35:
                continue
            pair = (min(source, int(target)), max(source, int(target)))
            pairs[pair] = max(pairs.get(pair, 0.0), weight)
            kept += 1
            if kept == 3:
                break
    return [{"source": a, "target": b, "strength": round(w, 4)} for (a, b), w in pairs.items()]


def _stable_positions(assets: list[str]) -> list[dict]:
    sectors = {asset: SECTOR_TAXONOMY.get(asset, "Other") for asset in assets}
    ordered = sorted(assets, key=lambda asset: (sectors[asset], asset))
    positions = {}
    for index, asset in enumerate(ordered):
        angle = 2 * np.pi * index / max(1, len(ordered)) - np.pi / 2
        positions[asset] = (round(float(np.cos(angle)), 5), round(float(np.sin(angle)), 5))
    return [
        {"id": index, "asset": asset, "sector": sectors[asset], "x": positions[asset][0], "y": positions[asset][1]}
        for index, asset in enumerate(assets)
    ]


def build_replay(event_id: int, resolution: str, before_hours: int, after_hours: int) -> dict:
    events = get_events()
    if event_id < 0 or event_id >= len(events):
        raise ValueError("Unknown event selection.")
    if resolution not in DATASET_PATHS:
        raise ValueError("Choose a supported dataset resolution.")
    if not (0 <= before_hours <= 24 * 14 and 1 <= after_hours <= 24 * 14):
        raise ValueError("Replay window is outside the supported range.")

    event = events[event_id]
    dataset = load_dataset(resolution)
    timestamps = dataset["timestamps"]
    event_date = pd.Timestamp(event["date"], tz="UTC")
    matching = np.flatnonzero(timestamps.normalize() == event_date.normalize())
    if not len(matching):
        raise ValueError("This event date is outside the selected dataset coverage.")
    anchor = int(matching[0])
    bar_hours = BAR_HOURS[resolution]
    before_bars = int(before_hours // bar_hours)
    after_bars = max(1, int(np.ceil(after_hours / bar_hours)))
    start = max(0, anchor - before_bars)
    end = min(len(timestamps) - 1, anchor + after_bars)
    steps = np.arange(start, end + 1, dtype=int)

    shocks = dataset["shocks"] if resolution == "hourly" and dataset["shocks"] is not None else _daily_shocks(dataset, anchor)
    active, returns = dataset["active"], dataset["returns"]
    assets = dataset["assets"]
    frames = []
    for step in steps:
        frames.append({
            "index": int(step),
            "timestamp": timestamps[step].isoformat(),
            "offset_hours": int((step - anchor) * bar_hours),
            "shocked": np.flatnonzero(shocks[step] & active[step]).tolist(),
            "active": np.flatnonzero(active[step]).tolist(),
            "returns_pct": np.round(returns[step] * 100, 5).tolist(),
        })

    first_shock: dict[int, int] = {}
    duration: dict[int, int] = {}
    for node in range(len(assets)):
        hits = [int(step) for step in steps if step > anchor and shocks[step, node] and active[step, node]]
        if not hits:
            continue
        first_shock[node] = hits[0]
        run = 0
        next_step = hits[0]
        for step in hits:
            if step != next_step:
                break
            run += 1
            next_step += 1
        duration[node] = run * bar_hours

    event_active = active[anchor]
    affected = [node for node in first_shock if event_active[node]]
    distinct_times = sorted({first_shock[node] for node in affected})
    first_to_next = (distinct_times[1] - distinct_times[0]) * bar_hours if len(distinct_times) > 1 else None
    durations = [duration[node] for node in affected if node in duration]
    edges = _network_edges(dataset, anchor, resolution)
    nodes = _stable_positions(assets)
    nodes_by_id = []
    for node in nodes:
        index = node["id"]
        first = first_shock.get(index)
        nodes_by_id.append({
            **node,
            "active_at_event": bool(event_active[index]),
            "first_shock_offset_hours": int((first - anchor) * bar_hours) if first is not None else None,
            "contiguous_duration_hours": duration.get(index, 0),
        })

    return {
        "event": {
            "id": event_id, "name": event["name"], "date": event["date"],
            "description": event.get("description", ""), "source_url": event.get("source_url", ""),
            "is_crisis": event["is_crisis"],
        },
        "resolution": resolution,
        "coverage": {"start": timestamps.min().isoformat(), "end": timestamps.max().isoformat()},
        "bar_hours": bar_hours,
        "anchor_index": anchor,
        "anchor_timestamp": timestamps[anchor].isoformat(),
        "nodes": nodes_by_id,
        "edges": edges,
        "frames": frames,
        "metrics": {
            "event_active_assets": int(event_active.sum()),
            "affected_assets": len(affected),
            "reach_fraction": len(affected) / max(1, int(event_active.sum())),
            "first_to_next_shock_hours": first_to_next,
            "median_contiguous_duration_hours": float(np.median(durations)) if durations else None,
        },
        "method": (
            "Saved train-calibrated hourly shock labels; top pre-event absolute-return correlations are undirected associations."
            if resolution == "hourly" and dataset["shocks"] is not None else
            "Daily event-specific 95th-percentile absolute-return thresholds from pre-event history; visualization only."
        ),
    }


class RequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_ROOT), **kwargs)

    def _json(self, status: int, payload: dict | list) -> None:
        body = json.dumps(payload, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlsplit(self.path)
        try:
            if parsed.path == "/api/events":
                self._json(200, get_event_catalogue())
                return
            if parsed.path == "/api/research/network":
                rankings = _read_csv_report("rankings")
                network = None
                if DATASET_PATHS["hourly"].is_file():
                    dataset = load_dataset("hourly")
                    anchor = len(dataset["timestamps"]) - 1
                    nodes = _stable_positions(dataset["assets"])
                    metrics = {str(row["coin"]).replace("-USD", ""): row for row in rankings}
                    for node in nodes:
                        node.update(metrics.get(node["asset"], {}))
                    network = {"nodes": nodes, "edges": _network_edges(dataset, anchor, "hourly")}
                self._json(200, {"rankings": rankings, "network": network})
                return
            if parsed.path == "/api/research/spillover":
                self._json(200, {
                    "persistence": _read_csv_report("persistence"),
                    "explanations": _read_csv_report("explanations"),
                })
                return
            if parsed.path == "/api/research/security":
                manifest_path = REPO_ROOT / "checkpoints" / "manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
                self._json(200, {
                    "anomalies": _read_csv_report("anomalies"),
                    "robustness": _read_csv_report("robustness"),
                    "manifest": manifest,
                })
                return
            if parsed.path == "/api/forecast/options":
                self._json(200, get_forecast_options())
                return
            if parsed.path == "/api/forecast":
                query = parse_qs(parsed.query)
                result = run_forecast(
                    model=query.get("model", ["tgn"])[0],
                    timestep=int(query.get("timestep", ["-1"])[0]),
                    horizon=int(query.get("horizon", ["1"])[0]),
                    threshold=float(query.get("threshold", ["0.5"])[0]),
                    top_k=int(query.get("top_k", ["10"])[0]),
                )
                self._json(200, result)
                return
            if parsed.path == "/api/replay":
                query = parse_qs(parsed.query)
                result = build_replay(
                    event_id=int(query.get("event", ["0"])[0]),
                    resolution=query.get("resolution", ["daily"])[0],
                    before_hours=int(query.get("before", ["48"])[0]),
                    after_hours=int(query.get("after", ["168"])[0]),
                )
                self._json(200, result)
                return
            if parsed.path.startswith("/api/"):
                self._json(404, {"error": "Unknown API endpoint."})
                return
            self.path = parsed.path or "/"
            super().do_GET()
        except (ValueError, FileNotFoundError, KeyError) as exc:
            self._json(400, {"error": str(exc)})
        except Exception as exc:  # Return a useful local error without exposing a traceback.
            self._json(500, {"error": f"Local service failed: {exc}"})

    def log_message(self, fmt: str, *args) -> None:
        print(f"[web] {self.address_string()} - {fmt % args}")


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 8765), RequestHandler)
    print("CryptoGraph Historical Replay: http://127.0.0.1:8765")
    print("Local only. Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping CryptoGraph Historical Replay…")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
