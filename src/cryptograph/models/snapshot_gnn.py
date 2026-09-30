"""Snapshot GNN Baseline Stub.

Discrete dynamic graph snapshot neural network processing sequential graph slices
using spatial GCN/GAT aggregators followed by recurrent GRU/LSTM layers.
"""
from typing import Optional, Tuple
import torch
import torch.nn as nn

class SnapshotGNN(nn.Module):
    """Discrete-time snapshot GNN baseline."""
    def __init__(self, in_features: int = 8, hidden_dim: int = 64, out_horizons: int = 5):
        super().__init__()
        self.linear = nn.Linear(in_features, hidden_dim)
        self.head = nn.Linear(hidden_dim, out_horizons)

    def forward(self, x: torch.Tensor, edge_index: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = torch.relu(self.linear(x))
        return self.head(h)
