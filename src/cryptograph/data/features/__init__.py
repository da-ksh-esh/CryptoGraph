"""Node features, targets and train-only scaling."""

from .node_features import FeatureSpec, build_node_features
from .scaling import RobustScalerSet
from .targets import (contagion_labels, forward_realized_vol, forward_returns, next_day_shock, shock_indicator,
                      shock_z)

__all__ = [
    "FeatureSpec", "build_node_features", "RobustScalerSet",
    "forward_returns", "forward_realized_vol", "shock_z", "shock_indicator", "next_day_shock", "contagion_labels",
]
