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
        num_nodes: int = 20,
        raw_msg_dim: int = 8,
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
        self.spatial_conv = GCNConv(memory_dim + raw_msg_dim, embedding_dim)

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
        # Read current memory states
        node_ids = torch.arange(self.num_nodes, device=node_features.device)
        if self.training:
            mem, _ = self.memory(node_ids)
        else:
            with torch.no_grad():
                mem, _ = self.memory(node_ids)
                mem = mem.detach()

        # Fuse node memory with instantaneous features
        fused = torch.cat([mem, node_features], dim=-1)

        # Spatial aggregation over dynamic edges
        if edge_index is not None and edge_index.size(1) > 0:
            z = torch.relu(self.spatial_conv(fused, edge_index, edge_weight))
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
        self.memory.update_state(src, dst, timestamps, edge_features)


class GraphAttentionEmbedding(nn.Module):
    """Attention over each node's most recent events, with the time since its last update encoded
    (as in the reference PyG TGN; Rossi et al. 2020)."""

    def __init__(self, in_channels: int, out_channels: int, msg_dim: int, time_enc: nn.Module, dropout: float = 0.1):
        super().__init__()
        from torch_geometric.nn import TransformerConv

        self.time_enc = time_enc
        self.conv = TransformerConv(in_channels, out_channels // 2, heads=2, dropout=dropout,
                                    edge_dim=msg_dim + time_enc.out_channels)

    def forward(self, x, last_update, edge_index, t, msg):
        rel_t = (last_update[edge_index[0]] - t).to(x.dtype)
        edge_attr = torch.cat([self.time_enc(rel_t), msg], dim=-1)
        return self.conv(x, edge_index, edge_attr)


class ContagionTGN(nn.Module):
    """Edge-level contagion model on the shock-triggered event stream.

    For an event (i shocks at bar t, receiver j) it outputs one logit per horizon k: will j shock within k bars?
    Inputs: node memory (updated only with past events), attention over recent events, the node features of i and
    j at bar t, and the event message. The caller must update memory / neighbours only AFTER predicting a batch.
    """

    def __init__(self, num_nodes: int, msg_dim: int, node_feat_dim: int, memory_dim: int = 64, time_dim: int = 16,
                 embedding_dim: int = 64, out_horizons: int = 5, dropout: float = 0.2, use_memory: bool = True,
                 ctx_dim: int = 0):
        super().__init__()
        from torch_geometric.nn.models.tgn import TimeEncoder

        self.use_memory = use_memory          # False = ablation: the head sees only node features and the message
        self.embedding_dim = embedding_dim
        self.ctx_dim = ctx_dim                # event context known at t (market breadth, shock state of i and j)

        self.memory = TGNMemory(num_nodes, msg_dim, memory_dim, time_dim,
                                message_module=IdentityMessage(msg_dim, memory_dim, time_dim),
                                aggregator_module=LastAggregator())
        self.gnn = GraphAttentionEmbedding(memory_dim, embedding_dim, msg_dim, TimeEncoder(time_dim), dropout=0.1)
        d_in = 2 * embedding_dim + 2 * node_feat_dim + msg_dim + ctx_dim
        self.head = nn.Sequential(nn.Linear(d_in, embedding_dim), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(embedding_dim, out_horizons))

    def embed(self, n_id: torch.Tensor, edge_index: torch.Tensor, t_nb: torch.Tensor, msg_nb: torch.Tensor):
        if not self.use_memory:
            return torch.zeros(n_id.size(0), self.embedding_dim, device=n_id.device)
        z, last_update = self.memory(n_id)
        return self.gnn(z, last_update, edge_index, t_nb, msg_nb)

    def decode(self, z_src, z_dst, x_src, x_dst, msg, ctx=None) -> torch.Tensor:
        parts = [z_src, z_dst, x_src, x_dst, msg] + ([ctx] if self.ctx_dim else [])
        return self.head(torch.cat(parts, dim=-1))
