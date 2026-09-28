from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np

@dataclass
class ShockEvent:
    timestamp: str
    source_asset: str
    shock_magnitude: float
    shock_type: str  # 'positive_jump', 'flash_crash', 'volatility_spike'
    z_score: float

@dataclass
class SpilloverMetrics:
    source_asset: str
    target_asset: str
    magnitude: float
    delay_steps: int
    half_life_persistence: float
    transmission_path: List[str] = field(default_factory=list)

@dataclass
class AnomalyAlert:
    timestamp: str
    asset: str
    anomaly_score: float
    threshold: float
    suspected_type: str  # 'oracle_manipulation', 'wash_trading', 'flash_loan'
    reconstruction_error: float

@dataclass
class GraphSnapshot:
    timestamp: str
    nodes: List[str]
    edge_index: np.ndarray  # shape (2, num_edges)
    edge_weight: np.ndarray  # shape (num_edges,)
    node_features: np.ndarray  # shape (num_nodes, num_features)
