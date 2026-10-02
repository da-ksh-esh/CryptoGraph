"""DyGFormer: Dynamic Graph Transformer with Neighbor Co-occurrence Patching.

Based on Yu et al. (NeurIPS 2023): 'Towards Better Dynamic Graph Learning: New Architecture and Unified Library'.
Applies temporal sequence patching and self-attention over continuous interaction sequences.
"""

from typing import Dict, Any, List, Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalPatchEncoder(nn.Module):
    """
    Groups rolling interaction history into temporal patches and applies linear projection.
    """
    def __init__(self, in_features: int, patch_size: int = 4, hidden_dim: int = 64):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(in_features * patch_size, hidden_dim)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_seq: (batch_size, seq_len, in_features)
        Returns:
            patches: (batch_size, num_patches, hidden_dim)
        """
        B, L, F = x_seq.shape
        num_patches = L // self.patch_size
        if L % self.patch_size != 0:
            pad_len = self.patch_size - (L % self.patch_size)
            x_seq = F.pad(x_seq, (0, 0, pad_len, 0))
            num_patches = x_seq.shape[1] // self.patch_size

        # Reshape into patches: (B, num_patches, patch_size * F)
        x_patched = x_seq.view(B, num_patches, -1)
        return self.proj(x_patched)


class PatchTemporalTransformer(nn.Module):
    """
    Patch-based Temporal Transformer baseline for multi-horizon cryptocurrency shock forecasting.

    ARCHITECTURAL SPECIFICATION:
    Applies temporal sequence patching (inspired by patching concepts in Yu et al., NeurIPS 2023)
    and multi-head self-attention across temporal history patches.
    NOTE: This model operates as a non-graph sequential Transformer baseline without
    dynamic neighbor co-occurrence encoding. For spatial-temporal interaction routing,
    see CryptoTGN.
    """

    def __init__(
        self,
        in_features: int = 9,
        patch_size: int = 4,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        out_horizons: int = 5,
        dropout: float = 0.2
    ):
        super().__init__()
        self.patch_encoder = TemporalPatchEncoder(in_features, patch_size=patch_size, hidden_dim=hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.forecast_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, out_horizons)
        )

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_seq: (batch_size, seq_len, in_features) or (N, seq_len, in_features)
        Returns:
            preds: (batch_size, out_horizons)
        """
        # Patching
        h_patches = self.patch_encoder(x_seq)  # (B, num_patches, hidden_dim)

        # Multi-layer Transformer self-attention across temporal patches
        h_trans = self.transformer(h_patches)

        # Pool across patches (mean pooling)
        h_pooled = torch.mean(h_trans, dim=1)

        # Multi-horizon forecast
        return self.forecast_head(h_pooled)


# Aliases for backward compatibility
CryptoDyGFormer = PatchTemporalTransformer
DyGFormerAdapter = PatchTemporalTransformer

