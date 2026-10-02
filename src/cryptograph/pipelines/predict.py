"""Predict Pipeline & CLI: Standalone contagion risk prediction and spatio-temporal explainability engine.

Ingests processed multi-asset data feeds, loads trained CryptoTGN / CryptoLSTM models,
and outputs real-time shock probabilities, expected persistence half-lives, and TempME
explanatory transmission channels.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import sys
import os
import argparse
import json
import glob
from pathlib import Path
import torch
import numpy as np
import pandas as pd

from ..models.tgn import CryptoTGN
from ..models.lstm import CryptoLSTM
from ..explain.tempme import TempME
from ..data.universe import TARGET_UNIVERSE_30, SECTOR_TAXONOMY
from ..training.tgn_trainer import build_topk_dynamic_edges
from ..training.device import resolve_device
from ..common.logger import get_logger

logger = get_logger(__name__)

# Sector-specific persistence half-life lookup table derived from empirical study (Phase 17)
SECTOR_PERSISTENCE_HALFLIFE: Dict[str, float] = {
    "Layer 1": 23.5,
    "Layer 2": 16.0,
    "Payments": 18.2,
    "Interoperability": 15.4,
    "Infrastructure": 14.8,
    "DePIN": 14.0,
    "AI": 11.5,
    "DeFi": 13.2,
    "Meme": 6.8,
}
DEFAULT_PERSISTENCE_HALFLIFE: float = 12.0


def validate_checkpoint_schema(ckpt: Dict[str, Any], expected_model_type: str) -> None:
    """Validates that the loaded checkpoint dictionary matches expected schema and architecture."""
    if not isinstance(ckpt, dict):
        raise TypeError(f"Checkpoint must be a dictionary, got {type(ckpt).__name__}")
    if "model_state_dict" not in ckpt:
        raise KeyError("Checkpoint missing required key 'model_state_dict'")
    state_dict = ckpt["model_state_dict"]
    if not isinstance(state_dict, dict) or len(state_dict) == 0:
        raise ValueError("Checkpoint 'model_state_dict' must be a non-empty dictionary.")

    expected = expected_model_type.lower()
    keys = list(state_dict.keys())
    if expected == "tgn":
        is_tgn = any("memory" in k or "spatial_conv" in k for k in keys)
        if not is_tgn:
            raise ValueError(
                f"Checkpoint architecture mismatch: expected '{expected}', "
                f"but state_dict keys do not match TGN (found: {keys[:4]})."
            )
    elif expected == "lstm":
        is_lstm = any("lstm" in k for k in keys)
        if not is_lstm:
            raise ValueError(
                f"Checkpoint architecture mismatch: expected '{expected}', "
                f"but state_dict keys do not match LSTM (found: {keys[:4]})."
            )


def validate_dataset_schema(data: Dict[str, Any]) -> None:
    """Validates that the processed dataset dictionary matches required structure and shapes."""
    if not isinstance(data, dict):
        raise TypeError(f"Dataset must be a dictionary, got {type(data).__name__}")
    required_keys = ["features", "timestamps", "asset_names"]
    for k in required_keys:
        if k not in data:
            raise KeyError(f"Dataset missing required key '{k}'")

    features = data["features"]
    if not isinstance(features, torch.Tensor) or features.ndim != 3:
        raise ValueError(
            f"Dataset 'features' must be a 3D Tensor (T, N, F), "
            f"got {type(features)} with shape {getattr(features, 'shape', None)}"
        )

    T, N, _ = features.shape
    if len(data["timestamps"]) != T:
        raise ValueError(
            f"Dataset 'timestamps' length ({len(data['timestamps'])}) does not match features T dimension ({T})"
        )
    if len(data["asset_names"]) != N:
        raise ValueError(
            f"Dataset 'asset_names' length ({len(data['asset_names'])}) does not match features N dimension ({N})"
        )


def validate_compatibility(
    model_type: str,
    ckpt: Dict[str, Any],
    data: Dict[str, Any],
    features_to_use: torch.Tensor,
    horizons: List[int],
    asset_names: List[str]
) -> None:
    """
    Validates dimensional and structural compatibility between loaded checkpoint and dataset.
    Ensures feature dimensions, number of assets, and prediction horizons match exactly.
    """
    state_dict = ckpt["model_state_dict"]
    expected = model_type.lower()

    if expected == "tgn":
        # Check node count from memory buffer
        if "memory.memory" in state_dict:
            ckpt_nodes = state_dict["memory.memory"].shape[0]
            if ckpt_nodes != len(asset_names):
                raise ValueError(
                    f"Asset count mismatch: checkpoint expects {ckpt_nodes} assets, "
                    f"but dataset contains {len(asset_names)} assets."
                )
        # Check input feature dimension to spatial convolution
        if "spatial_conv.lin.weight" in state_dict:
            expected_fused_dim = state_dict["spatial_conv.lin.weight"].shape[1]
            memory_dim = ckpt.get("config", {}).get("model", {}).get("memory_dim", 64)
            actual_fused_dim = features_to_use.shape[-1] + memory_dim
            if expected_fused_dim != actual_fused_dim:
                raise ValueError(
                    f"Feature dimension mismatch: checkpoint spatial_conv expects fused dim {expected_fused_dim} "
                    f"(memory_dim={memory_dim} + node_feat_dim={expected_fused_dim - memory_dim}), "
                    f"but provided dataset fused features have dim {features_to_use.shape[-1]} "
                    f"(total fused {actual_fused_dim})."
                )
        # Check forecast head horizons
        forecast_weights = [v for k, v in state_dict.items() if "forecast_head" in k and "weight" in k]
        if forecast_weights:
            ckpt_horizons = forecast_weights[-1].shape[0]
            if ckpt_horizons != len(horizons):
                raise ValueError(
                    f"Horizon count mismatch: checkpoint forecast head predicts {ckpt_horizons} horizons, "
                    f"but dataset specifies {len(horizons)} horizons ({horizons})."
                )

    elif expected == "lstm":
        if "lstm.weight_ih_l0" in state_dict:
            expected_feat_dim = state_dict["lstm.weight_ih_l0"].shape[1]
            actual_feat_dim = features_to_use.shape[-1]
            if expected_feat_dim != actual_feat_dim:
                raise ValueError(
                    f"Feature dimension mismatch: checkpoint LSTM expects {expected_feat_dim} features, "
                    f"but dataset has {actual_feat_dim} features."
                )
        fc_weights = [v for k, v in state_dict.items() if ("fc" in k or "forecast" in k) and "weight" in k]
        if fc_weights:
            ckpt_horizons = fc_weights[-1].shape[0]
            if ckpt_horizons != len(horizons):
                raise ValueError(
                    f"Horizon count mismatch: checkpoint LSTM head predicts {ckpt_horizons} horizons, "
                    f"but dataset specifies {len(horizons)} horizons ({horizons})."
                )


def find_latest_checkpoint(model_type: str = "tgn") -> Optional[Path]:
    """Finds the most recent best checkpoint for the given model architecture without cross-model fallback."""
    pattern = f"runs/*{model_type}*/best_{model_type}_checkpoint.pt"
    candidates = sorted(glob.glob(pattern))
    if candidates:
        return Path(candidates[-1])
    return None


def resolve_timestep(
    timestamps: List[str],
    timestep_arg: Union[int, str]
) -> int:
    """
    Resolves an index or date string to a valid timestep index.
    Raises IndexError if integer index is out of bounds.
    Raises ValueError if date string cannot be matched to dataset timestamps.
    """
    if isinstance(timestep_arg, int):
        idx = len(timestamps) + timestep_arg if timestep_arg < 0 else timestep_arg
        if idx < 0 or idx >= len(timestamps):
            raise IndexError(
                f"Timestep index {timestep_arg} is out of bounds for dataset of length {len(timestamps)} "
                f"(valid range: [{-len(timestamps)}, {len(timestamps) - 1}])."
            )
        return idx

    # String argument: try parsing integer first
    try:
        t_int = int(timestep_arg)
        idx = len(timestamps) + t_int if t_int < 0 else t_int
        if idx < 0 or idx >= len(timestamps):
            raise IndexError(
                f"Timestep index {timestep_arg} is out of bounds for dataset of length {len(timestamps)} "
                f"(valid range: [{-len(timestamps)}, {len(timestamps) - 1}])."
            )
        return idx
    except ValueError:
        pass

    # String date lookup
    query = str(timestep_arg).strip()
    matches = [i for i, ts in enumerate(timestamps) if query in ts]
    if matches:
        return matches[-1]

    # Timestamp comparison if query looks like a datetime
    try:
        query_dt = pd.to_datetime(query, utc=True)
        ts_series = pd.to_datetime(timestamps, utc=True)
        deltas = np.abs((ts_series - query_dt).total_seconds())
        min_idx = int(np.argmin(deltas))
        # Disallow match if closest timestamp is more than 7 days away
        if deltas[min_idx] > 7 * 86400:
            raise ValueError(
                f"Timestep query '{query}' is too far from nearest dataset timestamp "
                f"({timestamps[min_idx]}, delta: {deltas[min_idx]/86400:.1f} days)."
            )
        return min_idx
    except Exception as e:
        if isinstance(e, ValueError):
            raise
        raise ValueError(f"Could not resolve timestep query '{timestep_arg}' to any valid dataset timestamp: {e}")


def predict_contagion(
    checkpoint_path: Optional[str] = None,
    model_type: str = "tgn",
    dataset_path: str = "data/processed/v2/1h/processed_dataset.pt",
    network_features_path: str = "data/processed/v2/network_features.pt",
    timestep: Union[int, str] = -1,
    horizon_hours: int = 1,
    threshold: float = 0.50,
    top_k: int = 10,
    explain: bool = True,
    device_name: str = "auto"
) -> Dict[str, Any]:
    """
    Executes inference for all universe assets at the specified timestep.

    Returns:
        Structured dictionary containing run metadata, asset risk predictions,
        market alert aggregates, and TempME explanatory channels.
    """
    device = resolve_device(device_name)
    logger.info(f"Executing predict pipeline with device: {device}")

    # 1. Load Checkpoint
    if checkpoint_path is None or not Path(checkpoint_path).exists():
        found = find_latest_checkpoint(model_type)
        if not found:
            raise FileNotFoundError(f"No checkpoint found for model type '{model_type}'.")
        checkpoint_path = str(found)

    logger.info(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ckpt_config = ckpt.get("config", {})

    # Detect model type if not explicitly set
    if "tgn" in Path(checkpoint_path).name.lower():
        model_type = "tgn"
    elif "lstm" in Path(checkpoint_path).name.lower():
        model_type = "lstm"

    validate_checkpoint_schema(ckpt, model_type)

    # 2. Load Processed Dataset
    data_file = Path(dataset_path)
    if not data_file.exists():
        raise FileNotFoundError(f"Processed dataset not found at: {data_file}")

    data = torch.load(data_file, map_location="cpu", weights_only=False)
    validate_dataset_schema(data)
    features = data["features"]  # (T, N, F_tech)
    timestamps = data["timestamps"]
    horizons = data.get("horizons", [1, 3, 6, 12, 24])
    asset_names = data.get("asset_names", TARGET_UNIVERSE_30)
    close_prices = data.get("close_prices")

    t_idx = resolve_timestep(timestamps, timestep)
    current_timestamp = timestamps[t_idx]
    logger.info(f"Target timestep: {t_idx} ({current_timestamp})")

    if horizon_hours not in horizons:
        # Default to index 0 (1h)
        h_idx = 0
        horizon_hours = horizons[0]
    else:
        h_idx = horizons.index(horizon_hours)

    # 3. Model Instantiation & Inference
    asset_risks = []
    explanations_dict = {}

    if model_type == "tgn":
        # Load network features if available
        net_file = Path(network_features_path)
        if net_file.exists():
            net_data = torch.load(net_file, map_location="cpu", weights_only=False)
            net_feats = net_data["network_features"]
            fused_features = torch.cat([features, net_feats], dim=-1)
        else:
            fused_features = features

        validate_compatibility("tgn", ckpt, data, fused_features, horizons, asset_names)

        node_feat_dim = fused_features.shape[-1]
        raw_msg_dim = ckpt_config.get("model", {}).get("raw_msg_dim", 9)
        memory_dim = ckpt_config.get("model", {}).get("memory_dim", 64)
        embedding_dim = ckpt_config.get("model", {}).get("embedding_dim", 64)
        out_horizons = len(horizons)

        model = CryptoTGN(
            num_nodes=len(asset_names),
            raw_msg_dim=raw_msg_dim,
            node_feat_dim=node_feat_dim,
            memory_dim=memory_dim,
            embedding_dim=embedding_dim,
            out_horizons=out_horizons
        ).to(device)

        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        model.reset_memory()

        # Reconstruct historical memory states over preceding 168 hours
        shocks = data.get("shocks")
        act_mask = data.get("active_mask")
        warmup_start = max(0, t_idx - 168)
        if shocks is not None and act_mask is not None:
            logger.info(f"Reconstructing historical TGN memory over warmup timesteps [{warmup_start}:{t_idx}]...")
            with torch.no_grad():
                for tau in range(warmup_start, t_idx + 1):
                    shocked = torch.where(shocks[tau] & act_mask[tau])[0]
                    active = torch.where(act_mask[tau])[0]
                    if len(shocked) > 0 and len(active) > 1:
                        src_list, dst_list = [], []
                        for s in shocked:
                            for d in active:
                                if d != s:
                                    src_list.append(s)
                                    dst_list.append(d)
                        if len(src_list) > 0:
                            src = torch.tensor(src_list, dtype=torch.long, device=device)
                            dst = torch.tensor(dst_list, dtype=torch.long, device=device)
                            ts = torch.full((len(src_list),), tau, dtype=torch.long, device=device)
                            msg = fused_features[tau, src, :raw_msg_dim].to(device)
                            model.update_memory(src, dst, msg, ts)

        # Build dynamic correlation graph at t_idx
        lookback_window = 24
        start_t = max(0, t_idx - lookback_window)
        returns_slice = features[start_t : t_idx + 1, :, 0].numpy()
        edge_indices, edge_weights = build_topk_dynamic_edges(returns_slice, top_k=5, window=lookback_window)
        e_idx = edge_indices[-1].to(device)
        e_w = edge_weights[-1].to(device) if edge_weights[-1].numel() > 0 else None

        x_t = fused_features[t_idx].to(device)
        with torch.no_grad():
            preds, _ = model(x_t, e_idx, e_w)
            probs = torch.sigmoid(preds)[:, h_idx].cpu().numpy()

        # Run TempME for highest risk assets if requested
        if explain and e_idx.size(1) > 0:
            explainer = TempME(model, num_epochs=15, lr=0.05)
            # Explain top 3 assets
            top_candidate_indices = np.argsort(probs)[-3:][::-1]
            for c_idx in top_candidate_indices:
                try:
                    exp_res = explainer.explain(int(c_idx), x_t, e_idx, e_w)
                    top_edges = exp_res.get("top_edges", [])
                    edge_strs = []
                    for e in top_edges[:3]:
                        src_name = asset_names[e["source"]]
                        dst_name = asset_names[e["target"]]
                        imp = e.get("importance", 0.0)
                        edge_strs.append(f"{src_name}->{dst_name}({imp:.2f})")
                    explanations_dict[c_idx] = "; ".join(edge_strs) if edge_strs else "Structural mode"
                except Exception as ex:
                    logger.debug(f"TempME explanation failed for node {c_idx}: {ex}")
                    explanations_dict[c_idx] = "Correlation cluster"

    else:  # LSTM
        validate_compatibility("lstm", ckpt, data, features, horizons, asset_names)

        in_features = features.shape[-1]
        hidden_dim = ckpt_config.get("model", {}).get("hidden_dim", 64)
        num_layers = ckpt_config.get("model", {}).get("num_layers", 2)
        out_horizons = len(horizons)

        model = CryptoLSTM(
            in_features=in_features,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            out_horizons=out_horizons
        ).to(device)

        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        lookback_window = 24
        start_t = max(0, t_idx - lookback_window + 1)
        seq_slice = features[start_t : t_idx + 1]  # (seq_len, N, F)
        if seq_slice.size(0) < lookback_window:
            pad = torch.zeros(lookback_window - seq_slice.size(0), len(asset_names), in_features)
            seq_slice = torch.cat([pad, seq_slice], dim=0)

        # Transpose to (N, seq_len, in_features)
        x_seq = seq_slice.permute(1, 0, 2).to(device)
        with torch.no_grad():
            preds = model(x_seq)
            probs = torch.sigmoid(preds)[:, h_idx].cpu().numpy()

    # 4. Assemble Per-Asset Records
    for i, coin in enumerate(asset_names):
        prob = float(probs[i])
        price = float(close_prices[t_idx, i]) if close_prices is not None else 0.0
        sector = SECTOR_TAXONOMY.get(coin, "General")
        halflife = SECTOR_PERSISTENCE_HALFLIFE.get(sector, DEFAULT_PERSISTENCE_HALFLIFE)
        alert_status = "ALERT" if prob >= threshold else "NORMAL"
        explanation = explanations_dict.get(i, "None (Normal baseline)") if alert_status == "ALERT" else "Normal baseline"

        asset_risks.append({
            "asset": coin,
            "sector": sector,
            "price_usdt": price,
            "shock_probability": prob,
            "status": alert_status,
            "persistence_halflife_hours": halflife,
            "explanation": explanation
        })

    # Sort descending by shock probability
    asset_risks = sorted(asset_risks, key=lambda x: x["shock_probability"], reverse=True)
    for rank_idx, record in enumerate(asset_risks, start=1):
        record["rank"] = rank_idx

    alert_count = sum(1 for r in asset_risks if r["status"] == "ALERT")
    mean_market_prob = float(np.mean(probs))

    result = {
        "metadata": {
            "timestamp": current_timestamp,
            "timestep_index": t_idx,
            "horizon_hours": horizon_hours,
            "threshold": threshold,
            "model_type": model_type.upper(),
            "checkpoint": str(checkpoint_path),
            "device": str(device),
            "total_assets_evaluated": len(asset_names),
            "assets_in_alert": alert_count,
            "market_mean_shock_probability": round(mean_market_prob, 4),
            "market_risk_regime": "HIGH RISK / SYSTEMIC ALERT" if alert_count >= 5 else (
                "ELEVATED RISK" if alert_count >= 2 else "CALM / STABLE"
            )
        },
        "predictions": asset_risks[:top_k]
    }

    return result


def render_table_output(res: Dict[str, Any]):
    """Renders formatted ASCII table with high-visibility banners."""
    meta = res["metadata"]
    preds = res["predictions"]

    regime = meta["market_risk_regime"]
    regime_color = "\033[91m" if "HIGH" in regime else ("\033[93m" if "ELEVATED" in regime else "\033[92m")
    reset_color = "\033[0m"

    print("\n" + "=" * 92)
    print(f"  CRYPTOGRAPH-SECURE CONTAGION PREDICTION ENGINE (Phase 21)")
    print("=" * 92)
    print(f"  Timestamp:         {meta['timestamp']} (t={meta['timestep_index']})")
    print(f"  Model Engine:      {meta['model_type']} | Device: {meta['device']}")
    print(f"  Forecast Horizon:  +{meta['horizon_hours']} Hours | Alert Threshold: P >= {meta['threshold']:.2f}")
    print(f"  Market Regime:     {regime_color}{regime}{reset_color} ({meta['assets_in_alert']}/{meta['total_assets_evaluated']} assets in ALERT)")
    print(f"  Mean Risk P(t):    {meta['market_mean_shock_probability']:.4f}")
    print("-" * 92)
    print(f"{'Rank':<5} {'Asset':<7} {'Sector':<16} {'Price ($)':<12} {'Risk P(t)':<11} {'Status':<10} {'t_1/2':<8} {'Explanatory Transmission Channel'}")
    print("-" * 92)

    for r in preds:
        status_str = f"\033[91m{r['status']:<10}\033[0m" if r["status"] == "ALERT" else f"\033[92m{r['status']:<10}\033[0m"
        price_str = f"${r['price_usdt']:,.2f}" if r["price_usdt"] >= 1.0 else f"${r['price_usdt']:.4f}"
        print(f"{r['rank']:<5} {r['asset']:<7} {r['sector']:<16} {price_str:<12} {r['shock_probability']:>9.4f}  {status_str} {r['persistence_halflife_hours']:>4.1f}h   {r['explanation'][:32]}")

    print("=" * 92 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="CryptoGraph Standalone Contagion Prediction CLI (Phase 21)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to .pt checkpoint file.")
    parser.add_argument("--model-type", type=str, default="tgn", choices=["tgn", "lstm"], help="Model architecture.")
    parser.add_argument("--dataset-path", type=str, default="data/processed/v2/1h/processed_dataset.pt", help="Path to processed dataset.")
    parser.add_argument("--network-features-path", type=str, default="data/processed/v2/network_features.pt", help="Path to network features.")
    parser.add_argument("--timestep", type=str, default="-1", help="Timestep index (-1 for latest) or ISO date (e.g. 2024-08-05).")
    parser.add_argument("--horizon", type=int, default=1, choices=[1, 3, 6, 12, 24], help="Prediction horizon in hours.")
    parser.add_argument("--threshold", type=float, default=0.50, help="Probability threshold for contagion alert.")
    parser.add_argument("--top-k", type=int, default=10, help="Number of highest-risk assets to return.")
    parser.add_argument("--no-explain", action="store_true", help="Disable TempME explanation generation.")
    parser.add_argument("--format", type=str, default="table", choices=["table", "json", "csv"], help="Output display format.")
    parser.add_argument("--output-file", type=str, default=None, help="Optional file path to write results.")
    parser.add_argument("--device", type=str, default="auto", help="Compute device: auto | cpu | cuda.")

    args = parser.parse_args()

    results = predict_contagion(
        checkpoint_path=args.checkpoint,
        model_type=args.model_type,
        dataset_path=args.dataset_path,
        network_features_path=args.network_features_path,
        timestep=args.timestep,
        horizon_hours=args.horizon,
        threshold=args.threshold,
        top_k=args.top_k,
        explain=not args.no_explain,
        device_name=args.device
    )

    if args.format == "table":
        render_table_output(results)
    elif args.format == "json":
        json_str = json.dumps(results, indent=2)
        print(json_str)
        if args.output_file:
            with open(args.output_file, "w", encoding="utf-8") as f:
                f.write(json_str)
            logger.info(f"Wrote JSON output to {args.output_file}")
    elif args.format == "csv":
        df = pd.DataFrame(results["predictions"])
        csv_str = df.to_csv(index=False)
        print(csv_str)
        if args.output_file:
            df.to_csv(args.output_file, index=False)
            logger.info(f"Wrote CSV output to {args.output_file}")


if __name__ == "__main__":
    main()
