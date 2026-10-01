"""Model-ready datasets: chronological splits, LSTM windows, graph snapshots, TGN event stream."""

from .indexed import PanelWindowDataset, sample_mask_times, window_index
from .lstm import LSTMWindowDataset, build_windows
from .snapshot import GraphSnapshotDataset
from .splits import SPLITS, SplitPlan
from .temporal import load_event_stream, load_tgn_node_features

__all__ = [
    "SPLITS", "SplitPlan", "build_windows", "LSTMWindowDataset", "GraphSnapshotDataset",
    "load_event_stream", "load_tgn_node_features", "PanelWindowDataset", "sample_mask_times", "window_index",
]
