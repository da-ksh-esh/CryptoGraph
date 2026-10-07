"""Compatibility with the original `crypto_graph_dataset.pt` format.

The original pipeline (collector / feature_builder / graph_builder / dataloader
modules, now removed) wrote one dict per dataset. Notebook 01 and older callers
still read that format, so it is written alongside the v1 outputs and loaded
here. New code should use `cryptograph.data.datasets` instead.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .storage import load_pt


def write_legacy_graph_dataset(
    path: str | Path,
    *,
    features: torch.Tensor,
    targets: torch.Tensor,
    active_mask: torch.Tensor,
    edge_indices: list[torch.Tensor],
    edge_weights: list[torch.Tensor],
    timestamps: list[str],
    asset_names: list[str],
    feature_names: list[str],
    horizons: list[int],
    metadata: dict[str, Any],
) -> Path:
    """Write the dict layout of the original `crypto_graph_dataset.pt`.

    features (T, N, F), targets (T, N, H), active_mask (T, N); edge lists of length T.
    """
    payload = {
        "features": features, "targets": targets, "active_mask": active_mask,
        "edge_indices": edge_indices, "edge_weights": edge_weights,
        "timestamps": timestamps, "asset_names": asset_names, "feature_names": feature_names, "horizons": horizons,
        "metadata": metadata,
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, p)
    return p


def load_processed_dataset(file_path: str | Path) -> dict[str, Any]:
    """Load a dataset written by `write_legacy_graph_dataset` (the original format)."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Processed dataset not found: {path}")
    return load_pt(path)
