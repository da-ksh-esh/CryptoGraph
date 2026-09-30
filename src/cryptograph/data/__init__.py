"""Data layer: config -> sources -> quality -> features/targets -> graphs -> datasets.

Legacy names (CryptoDataCollector, FeatureBuilder, DynamicGraphBuilder,
CryptoSnapshotDataset, CryptoSequenceDataset, prepare_processed_dataset,
load_processed_dataset) are kept for existing callers. New pipelines use
`cryptograph-data` (cryptograph.pipelines.build_data) and the subpackages.
"""

from .collector import CryptoDataCollector
from .config import DataConfig, load_config
from .dataloader import (
    CryptoGraphDataset,
    CryptoSequenceDataset,
    CryptoSnapshotDataset,
    load_processed_dataset,
    prepare_processed_dataset,
)
from .datasets import GraphSnapshotDataset, LSTMWindowDataset, SplitPlan, load_event_stream, load_tgn_node_features
from .feature_builder import FeatureBuilder
from .graph_builder import DynamicGraphBuilder

__all__ = [
    # legacy
    "CryptoDataCollector",
    "FeatureBuilder",
    "DynamicGraphBuilder",
    "CryptoSnapshotDataset",
    "CryptoSequenceDataset",
    "CryptoGraphDataset",
    "prepare_processed_dataset",
    "load_processed_dataset",
    # new
    "DataConfig",
    "load_config",
    "SplitPlan",
    "LSTMWindowDataset",
    "GraphSnapshotDataset",
    "load_event_stream",
    "load_tgn_node_features",
]
