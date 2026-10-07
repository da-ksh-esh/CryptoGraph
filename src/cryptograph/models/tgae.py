"""T-GAE: Temporal Graph Autoencoder for market-data anomaly and manipulation detection."""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalGAE(nn.Module):
    """LEGACY feature autoencoder (no graph convolution, no time); not used in any result -- see SnapshotTGAE.

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


class SnapshotTGAE(nn.Module):
    """Temporal graph autoencoder over a sequence of graph snapshots (one per day of hourly data).

    For snapshot s: node embeddings u_s = GCN(x_s, A_s) (two layers), temporal state h_s = GRU(u_s, h_{s-1}) per node,
    then decode the adjacency (inner product h h^T -> edge logits) and the node features (MLP), either from h_s
    (reconstruction) or from h_{s-1} (prediction: how surprising snapshot s is given the past). The anomaly score of a
    snapshot is that loss.
    """

    def __init__(self, in_features: int, hidden_dim: int = 32, latent_dim: int = 16, dropout: float = 0.1):
        super().__init__()
        from torch_geometric.nn import GCNConv

        self.conv1 = GCNConv(in_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, latent_dim)
        self.gru = nn.GRUCell(latent_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)
        self.feat_dec = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, in_features))
        self.latent_dim = latent_dim

    def step(self, x, edge_index, edge_weight, h, predict: bool = False):
        """predict=False: reconstruct snapshot s from the state after seeing it.
        predict=True: forecast snapshot s from the state before seeing it (anomaly score = surprise), then update."""
        if predict:
            adj, xr = h @ h.t(), self.feat_dec(h)
        u = F.relu(self.conv1(x, edge_index, edge_weight))
        u = self.conv2(self.dropout(u), edge_index, edge_weight)
        h = self.gru(u, h)
        if not predict:
            adj, xr = h @ h.t(), self.feat_dec(h)
        return h, adj, xr

    @staticmethod
    def losses(adj_logits, adj_true, x_rec, x, node_mask):
        """Per-snapshot adjacency BCE (off-diagonal, active pairs) and feature MSE (active nodes)."""
        pair = node_mask[:, None] & node_mask[None, :]
        pair = pair & ~torch.eye(len(node_mask), dtype=torch.bool, device=pair.device)
        adj = F.binary_cross_entropy_with_logits(adj_logits[pair], adj_true[pair]) if pair.any() else adj_logits.sum() * 0
        feat = ((x_rec - x) ** 2)[node_mask].mean() if node_mask.any() else x.sum() * 0
        return adj, feat
