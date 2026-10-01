"""Dataset loaders and preprocessing serialization pipeline for dynamic spatio-temporal crypto graphs."""

from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import argparse
import yaml
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from .collector import DEFAULT_TICKERS
from .feature_builder import FeatureBuilder
from .graph_builder import DynamicGraphBuilder
from ..common.logger import get_logger

logger = get_logger(__name__)

class CryptoSnapshotDataset(Dataset):
    """
    Dataset returning dynamic graph snapshots at each individual timestep t.
    """

    def __init__(
        self,
        node_features: torch.Tensor,
        edge_indices: List[torch.Tensor],
        edge_weights: List[torch.Tensor],
        targets: torch.Tensor,
        active_mask: torch.Tensor
    ):
        self.node_features = node_features   # shape: (T, num_nodes, num_features)
        self.edge_indices = edge_indices     # list of length T
        self.edge_weights = edge_weights     # list of length T
        self.targets = targets               # shape: (T, num_nodes, num_horizons)
        self.active_mask = active_mask       # shape: (T, num_nodes)

    def __len__(self) -> int:
        return self.node_features.size(0)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "x": self.node_features[idx],
            "edge_index": self.edge_indices[idx],
            "edge_weight": self.edge_weights[idx],
            "y": self.targets[idx],
            "mask": self.active_mask[idx],
            "timestep": idx
        }

# Alias for backward compatibility
CryptoGraphDataset = CryptoSnapshotDataset


class CryptoSequenceDataset(Dataset):
    """
    Sliding window dataset for sequence models (LSTM / TGN history windows).
    Takes a window of past L timesteps: [t - L, t) to forecast at t.
    """

    def __init__(
        self,
        node_features: torch.Tensor,
        edge_indices: List[torch.Tensor],
        edge_weights: List[torch.Tensor],
        targets: torch.Tensor,
        active_mask: torch.Tensor,
        lookback_window: int = 30
    ):
        self.node_features = node_features
        self.edge_indices = edge_indices
        self.edge_weights = edge_weights
        self.targets = targets
        self.active_mask = active_mask
        self.lookback_window = lookback_window

    def __len__(self) -> int:
        return max(0, self.node_features.size(0) - self.lookback_window)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        start = idx
        end = idx + self.lookback_window
        target_step = end - 1

        return {
            "x_seq": self.node_features[start:end],         # (L, num_nodes, F)
            "x_latest": self.node_features[target_step],    # (num_nodes, F)
            "edge_index": self.edge_indices[target_step],   # (2, E)
            "edge_weight": self.edge_weights[target_step], # (E,)
            "y": self.targets[target_step],                 # (num_nodes, horizons)
            "mask": self.active_mask[target_step],          # (num_nodes,)
            "target_idx": target_step
        }


class HourlyWindowDataset(Dataset):
    """
    Sliding window dataset for sequence forecasting models (e.g. CryptoLSTM).
    Extracts rolling history [t - lookback_window : t] to forecast targets at timestep t.
    Uses target_indices to guarantee strict chronological split boundaries without leakage.
    """
    def __init__(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
        validity_mask: torch.Tensor,
        active_mask: torch.Tensor,
        target_indices: List[int],
        lookback_window: int = 24
    ):
        self.features = features
        self.targets = targets
        self.validity_mask = validity_mask
        self.active_mask = active_mask
        self.lookback_window = lookback_window
        # Only retain indices where t >= lookback_window so history is valid
        self.valid_targets = [t for t in target_indices if t >= lookback_window]

    def __len__(self) -> int:
        return len(self.valid_targets)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        t = self.valid_targets[idx]
        # x_window: past L timesteps across all assets: (L, N, F) -> permute to (N, L, F)
        x_window = self.features[t - self.lookback_window : t].permute(1, 0, 2) # (N, L, F)
        y = self.targets[t].float()                                            # (N, K)
        mask = (self.validity_mask[t] & self.active_mask[t].unsqueeze(-1)).bool() # (N, K)
        return {
            "x": x_window,
            "y": y,
            "mask": mask,
            "timestep": torch.tensor(t, dtype=torch.long)
        }


def collate_hourly_windows(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collates a list of timestep samples into a batched dictionary.
    Flattens (B, N, L, F) to (B * N, L, F) for efficient sequence batching.
    """
    x = torch.stack([b["x"] for b in batch], dim=0)       # (B, N, L, F)
    y = torch.stack([b["y"] for b in batch], dim=0)       # (B, N, K)
    mask = torch.stack([b["mask"] for b in batch], dim=0) # (B, N, K)
    timesteps = torch.stack([b["timestep"] for b in batch], dim=0)

    B, N, L, F = x.shape
    K = y.shape[-1]

    return {
        "x": x.reshape(B * N, L, F),
        "y": y.reshape(B * N, K),
        "mask": mask.reshape(B * N, K),
        "timesteps": timesteps,
        "batch_size": B,
        "num_nodes": N
    }


def prepare_processed_dataset(
    raw_parquet_path: str | Path,
    output_path: str | Path,
    config_path: Optional[str | Path] = None,
    horizons: List[int] = [1, 3, 7, 14, 30],
    rolling_window: int = 30,
    correlation_threshold: float = 0.35
) -> Path:
    """
    Transforms raw OHLCV Parquet dataset into engineered node feature and dynamic graph tensors.
    """
    raw_path = Path(raw_parquet_path)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data file not found: {raw_path}")

    # Load configuration if provided
    tickers = DEFAULT_TICKERS
    if config_path and Path(config_path).exists():
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            tickers = cfg.get("tickers", DEFAULT_TICKERS)
            horizons = cfg.get("forecast_horizons", horizons)
            rolling_window = cfg.get("graph", {}).get("rolling_window", rolling_window)
            correlation_threshold = cfg.get("graph", {}).get("threshold", correlation_threshold)

    logger.info(f"Loading raw dataset from {raw_path}...")
    df_raw = pd.read_parquet(raw_path)

    # 1. Feature Engineering
    feature_builder = FeatureBuilder(horizons=horizons)
    features_np, targets_np, active_mask_np, timestamps = feature_builder.build_multi_asset_tensors(
        raw_df=df_raw,
        tickers=tickers
    )

    # 2. Dynamic Graph Topology Construction
    # Use log returns (feature index 0) to compute dynamic rolling correlations
    returns_matrix = features_np[:, :, 0]
    graph_builder = DynamicGraphBuilder(
        correlation_threshold=correlation_threshold,
        rolling_window=rolling_window,
        min_top_k=3
    )
    edge_indices, edge_weights = graph_builder.build_dynamic_sequence(
        returns_matrix=returns_matrix,
        active_mask=active_mask_np
    )

    # 3. Serialization to PyTorch format
    dataset_payload = {
        "features": torch.from_numpy(features_np),       # (T, N, F)
        "targets": torch.from_numpy(targets_np),         # (T, N, Horizons)
        "active_mask": torch.from_numpy(active_mask_np), # (T, N)
        "edge_indices": edge_indices,
        "edge_weights": edge_weights,
        "timestamps": [str(ts) for ts in timestamps],
        "asset_names": [t.replace("-USD", "") for t in tickers],
        "feature_names": feature_builder.feature_names,
        "horizons": horizons,
        "metadata": {
            "num_timesteps": len(timestamps),
            "num_assets": len(tickers),
            "num_features": features_np.shape[-1],
            "rolling_window": rolling_window,
            "correlation_threshold": correlation_threshold
        }
    }

    torch.save(dataset_payload, out_path)
    file_size_mb = out_path.stat().st_size / 1e6
    logger.info(f"Processed dataset successfully saved to: {out_path} ({file_size_mb:.2f} MB)")
    return out_path


def load_processed_dataset(file_path: str | Path) -> Dict[str, Any]:
    """Loads processed PyTorch dataset dictionary from disk."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Processed dataset not found: {path}")
    return torch.load(path)


def main():
    parser = argparse.ArgumentParser(description="Process raw crypto OHLCV into dynamic graph tensors.")
    parser.add_argument("--raw-file", type=str, default="data/raw/crypto_top20_10y_1d.parquet")
    parser.add_argument("--out-file", type=str, default="data/processed/crypto_graph_dataset.pt")
    parser.add_argument("--config", type=str, default="configs/data/crypto_default.yaml")
    args = parser.parse_args()

    prepare_processed_dataset(
        raw_parquet_path=args.raw_file,
        output_path=args.out_file,
        config_path=args.config
    )


if __name__ == "__main__":
    main()
