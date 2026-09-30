"""Non-graph baseline: Multi-asset LSTM time-series predictor."""

import torch
import torch.nn as nn
from typing import Optional

class CryptoLSTM(nn.Module):
    """
    Standard LSTM baseline operating on individual crypto asset sequences
    without inter-asset graph relational information.
    """
    def __init__(
        self,
        in_features: int = 8,
        hidden_dim: int = 64,
        num_layers: int = 2,
        out_horizons: int = 5,
        dropout: float = 0.2
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, out_horizons)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, in_features) or (num_nodes, seq_len, in_features)
        Returns:
            Tensor of shape (batch_size, out_horizons)
        """
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        return self.fc(last_hidden)
