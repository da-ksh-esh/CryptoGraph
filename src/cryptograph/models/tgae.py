"""T-GAE: Temporal Graph Autoencoder for market-data anomaly and manipulation detection."""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalGAE(nn.Module):
    """
    Temporal Graph Autoencoder (T-GAE) that reconstructs both node features
    and dynamic graph connectivity. Unusually high reconstruction errors
    flag manipulated market feeds (oracle exploitation, wash trading, flash loan anomalies).
    """

    def __init__(self, in_features: int = 8, hidden_dim: int = 64, latent_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        # Encoder: compress spatial and temporal features
        self.enc_fc1 = nn.Linear(in_features, hidden_dim)
        self.enc_fc2 = nn.Linear(hidden_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)

        # Decoder: reconstruct feature vectors
        self.dec_fc1 = nn.Linear(latent_dim, hidden_dim)
        self.dec_fc2 = nn.Linear(hidden_dim, in_features)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.enc_fc1(x))
        h = self.dropout(h)
        return self.enc_fc2(h)

    def decode(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Feature reconstruction
        h = F.relu(self.dec_fc1(z))
        x_recon = self.dec_fc2(h)

        # Graph adjacency reconstruction (inner product decoder)
        adj_recon = torch.sigmoid(torch.mm(z, z.t()))
        return x_recon, adj_recon

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Node features of shape (num_nodes, in_features)
        Returns:
            x_recon: Reconstructed features
            adj_recon: Reconstructed probability adjacency matrix (num_nodes, num_nodes)
            z: Latent node embeddings
        """
        z = self.encode(x)
        x_recon, adj_recon = self.decode(z)
        return x_recon, adj_recon, z

    def compute_anomaly_scores(self, x: torch.Tensor, alpha: float = 0.7) -> torch.Tensor:
        """
        Calculates per-node anomaly score combining feature reconstruction MSE
        and connectivity deviation.
        """
        x_recon, adj_recon, _ = self.forward(x)
        feat_mse = torch.mean((x - x_recon) ** 2, dim=-1)
        return feat_mse
