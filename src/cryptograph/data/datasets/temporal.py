"""TGN event stream loaders."""

from __future__ import annotations

from pathlib import Path

from ..storage import load_pt


def load_event_stream(path: str | Path) -> dict:
    """Returns {"full", "train", "val", "test": TemporalData | None, "msg_names", "coins", "dates"}."""
    return load_pt(path)


def load_tgn_node_features(path: str | Path) -> dict:
    """Returns {"t": (T,), "x": (T,N,F), "x_mask": (T,N,F), "active": (T,N), "dates", "coins", "feature_names"}.
    Row t of x holds node features on day index t (the same index as TemporalData.t)."""
    return load_pt(path)
