"""Data layer: config -> sources -> quality -> features/targets -> graphs -> datasets.

Built by the `cryptograph-data` command (cryptograph.pipelines.build_data).
`load_processed_dataset` reads the original single-file format
(`crypto_graph_dataset.pt`) that is still written for older notebooks.
"""

from .config import DataConfig, load_config
from .datasets import GraphSnapshotDataset, LSTMWindowDataset, SplitPlan, load_event_stream, load_tgn_node_features
from .legacy import load_processed_dataset, write_legacy_graph_dataset

__all__ = [
    "DataConfig",
    "load_config",
    "SplitPlan",
    "LSTMWindowDataset",
    "GraphSnapshotDataset",
    "load_event_stream",
    "load_tgn_node_features",
    "load_processed_dataset",
    "write_legacy_graph_dataset",
]
