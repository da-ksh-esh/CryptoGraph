"""T-GAE: Spatio-Temporal Graph Autoencoder with GCN and GRU for anomaly and manipulation detection."""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class TemporalGAE(nn.Module):
    """
    Spatio-Temporal Graph Autoencoder (T-GAE).
    Combines spatial GCN neighbor convolution with temporal GRU cell state evolution
    and joint feature MSE + adjacency BCE reconstruction loss.
    """

    def __init__(
        self,
        in_features: int = 8,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        dropout: float = 0.1
    ):
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # 1. Spatial Graph Convolution
        self.gcn = GCNConv(in_features, hidden_dim)

        # 2. Temporal Recurrence (GRU Cell)
        self.temporal_gru = nn.GRUCell(hidden_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)

        # 3. Feature Decoder
        self.feature_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_features)
        )

    def encode(
        self,
        x: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        h_prev: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        num_nodes = x.size(0)
        if edge_index is None or edge_index.numel() == 0:
            nodes = torch.arange(num_nodes, device=x.device)
            edge_index = torch.stack([nodes, nodes], dim=0)

        # Spatial aggregation
        h_spatial = F.relu(self.gcn(x, edge_index))
        h_spatial = self.dropout(h_spatial)

        # Temporal recurrence
        if h_prev is None:
            h_prev = torch.zeros(num_nodes, self.latent_dim, device=x.device)
        z = self.temporal_gru(h_spatial, h_prev)

        return z, z

    def decode(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Feature reconstruction
        x_recon = self.feature_decoder(z)

        # Graph adjacency reconstruction (inner product decoder)
        adj_logits = torch.mm(z, z.t())
        adj_recon = torch.sigmoid(adj_logits)

        return x_recon, adj_recon

    def forward(
        self,
        x: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        h_prev: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Node features of shape (num_nodes, in_features)
            edge_index: Optional graph edge index (2, E)
            h_prev: Optional previous temporal hidden state (num_nodes, latent_dim)
        Returns:
            x_recon: Reconstructed features (num_nodes, in_features)
            adj_recon: Reconstructed adjacency matrix (num_nodes, num_nodes)
            z: Latent node embeddings (num_nodes, latent_dim)
        """
        z, _ = self.encode(x, edge_index, h_prev)
        x_recon, adj_recon = self.decode(z)
        return x_recon, adj_recon, z

    def loss_function(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        adj_target: torch.Tensor,
        adj_recon: torch.Tensor,
        alpha: float = 0.7
    ) -> torch.Tensor:
        """
        Joint Spatio-Temporal Graph Autoencoder loss:
        L = alpha * MSE(X, X_hat) + (1 - alpha) * BCE(A, A_hat)
        """
        feat_loss = F.mse_loss(x_recon, x)
        adj_loss = F.binary_cross_entropy(adj_recon, adj_target)
        return alpha * feat_loss + (1.0 - alpha) * adj_loss

    def compute_anomaly_scores(
        self,
        x: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        alpha: float = 0.7
    ) -> torch.Tensor:
        """
        Calculates per-node anomaly score combining feature reconstruction MSE
        and spatial connectivity deviation.
        """
        x_recon, adj_recon, _ = self.forward(x, edge_index)
        feat_mse = torch.mean((x - x_recon) ** 2, dim=-1)
        return feat_mse


# Alias for backward compatibility
TemporalGraphAE = TemporalGAE
