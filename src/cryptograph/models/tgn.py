"""Contagion Predictor: Temporal Graph Network (TGN) with dynamic memory and spatial aggregation."""

from typing import Tuple, Optional
import torch
import torch.nn as nn
from torch_geometric.nn.models.tgn import (
    TGNMemory,
    IdentityMessage,
    LastAggregator
)
from torch_geometric.nn import GCNConv

class CryptoTGN(nn.Module):
    """
    Temporal Graph Network (TGN) for continuous-time cryptocurrency shock contagion prediction.
    Maintains persistent memory states s_i(t) for each crypto node and aggregates
    dynamic neighborhood signals across inter-asset edges.
    """

    def __init__(
        self,
        num_nodes: int = 30,
        raw_msg_dim: int = 9,
        node_feat_dim: Optional[int] = None,
        memory_dim: int = 64,
        time_dim: int = 16,
        embedding_dim: int = 64,
        out_horizons: int = 5,
        dropout: float = 0.2
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.memory_dim = memory_dim
        self.raw_msg_dim = raw_msg_dim
        self.node_feat_dim = node_feat_dim if node_feat_dim is not None else raw_msg_dim

        # 1. TGN Dynamic Memory Module
        self.memory = TGNMemory(
            num_nodes=num_nodes,
            raw_msg_dim=raw_msg_dim,
            memory_dim=memory_dim,
            time_dim=time_dim,
            message_module=IdentityMessage(raw_msg_dim, memory_dim, time_dim),
            aggregator_module=LastAggregator()
        )

        # 2. Time Embedding Projection
        self.time_proj = nn.Linear(1, time_dim)

        # 3. Spatial Neighbor Aggregation across dynamic edges
        self.spatial_conv = GCNConv(memory_dim + self.node_feat_dim, embedding_dim)

        # 4. Multi-Horizon Contagion Forecast Head
        self.forecast_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 2, out_horizons)
        )

    def reset_memory(self):
        """Resets the internal node memory states for a new sequence or epoch."""
        self.memory.reset_state()

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        timestamps: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the TGN:
        1. Retrieves node memory states.
        2. Fuses with latest features.
        3. Aggregates spatial neighborhood messages via GCNConv.
        4. Generates multi-horizon shock & return predictions.

        Returns:
            predictions: Tensor of shape (num_nodes, out_horizons)
            embeddings: Tensor of shape (num_nodes, embedding_dim)
        """
        # Read current memory states (detached for truncated backprop across snapshots)
        node_ids = torch.arange(self.num_nodes, device=node_features.device)
        with torch.no_grad():
            mem, _ = self.memory(node_ids)
            mem = mem.detach()

        # Fuse node memory with instantaneous features
        fused = torch.cat([mem, node_features], dim=-1)

        # Spatial aggregation over dynamic edges
        if edge_index is not None and edge_index.size(1) > 0:
            norm_w = torch.abs(edge_weight).clamp(min=1e-4, max=1.0) if edge_weight is not None else None
            z = torch.relu(self.spatial_conv(fused, edge_index, norm_w))
        else:
            z = torch.relu(self.spatial_conv.lin(fused))

        # Generate multi-horizon forecasts
        preds = self.forecast_head(z)
        return preds, z

    def update_memory(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        edge_features: torch.Tensor,
        timestamps: torch.Tensor
    ):
        """Updates internal memory module given a batch of observed interactions/edges."""
        self.memory.update_state(
            src.to(dtype=torch.long),
            dst.to(dtype=torch.long),
            timestamps.to(dtype=torch.long),
            edge_features.to(dtype=torch.float32)
        )
