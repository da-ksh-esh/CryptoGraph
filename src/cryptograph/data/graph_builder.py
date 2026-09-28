"""Dynamic graph builder module: converts time-series returns into rolling dynamic graph topologies."""

from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
from ..common.logger import get_logger

logger = get_logger(__name__)

class DynamicGraphBuilder:
    """
    Constructs time-evolving cryptocurrency graph topologies.
    Computes rolling correlation matrices and extracts dynamic edge indices and weights,
    ensuring active node filtering and top-k connectivity guarantees.
    """

    def __init__(
        self,
        correlation_threshold: float = 0.35,
        rolling_window: int = 30,
        min_top_k: int = 3
    ):
        self.threshold = correlation_threshold
        self.rolling_window = rolling_window
        self.min_top_k = min_top_k

    def build_adjacency_for_window(
        self,
        returns_slice: np.ndarray,
        active_nodes_mask: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes dynamic adjacency for a single time window.

        Args:
            returns_slice: shape (window_len, num_nodes)
            active_nodes_mask: shape (num_nodes,) boolean

        Returns:
            edge_index: torch.LongTensor of shape (2, num_edges)
            edge_weight: torch.FloatTensor of shape (num_edges,)
        """
        num_nodes = returns_slice.shape[1]
        active_indices = np.where(active_nodes_mask)[0]

        if len(active_indices) < 2:
            return torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=torch.float)

        # Slice returns for active coins only
        active_returns = returns_slice[:, active_indices]

        # Handle constant/zero volatility columns to prevent NaN correlation
        stds = np.std(active_returns, axis=0)
        valid_std = stds > 1e-6
        if np.sum(valid_std) < 2:
            return torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=torch.float)

        corr = np.corrcoef(active_returns, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0)
        np.fill_diagonal(corr, 0.0)

        edges = []
        weights = []

        # Build edges among active nodes
        for local_i, global_i in enumerate(active_indices):
            row_corrs = corr[local_i]
            abs_corrs = np.abs(row_corrs)

            # Top-k strongest peer connections for connectivity guarantee
            k = min(self.min_top_k, len(active_indices) - 1)
            top_k_locals = np.argpartition(-abs_corrs, k)[:k]

            for local_j in range(len(active_indices)):
                if local_i == local_j:
                    continue
                c_val = float(row_corrs[local_j])
                global_j = active_indices[local_j]

                # Condition: Above threshold OR in top-k
                if abs(c_val) >= self.threshold or (local_j in top_k_locals and abs(c_val) > 0.05):
                    edges.append((global_i, global_j))
                    weights.append(c_val)

        if not edges:
            return torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=torch.float)

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_weight = torch.tensor(weights, dtype=torch.float)
        return edge_index, edge_weight

    def build_dynamic_sequence(
        self,
        returns_matrix: np.ndarray,
        active_mask: np.ndarray
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Iterates over all timesteps and generates dynamic graph sequence.

        Args:
            returns_matrix: shape (T, num_nodes)
            active_mask: shape (T, num_nodes) boolean

        Returns:
            list_edge_indices: List of length T
            list_edge_weights: List of length T
        """
        num_timesteps = returns_matrix.shape[0]
        list_edge_indices = []
        list_edge_weights = []

        logger.info(f"Building dynamic graph sequence over {num_timesteps} timesteps (window={self.rolling_window})...")

        for t in range(num_timesteps):
            # Window slicing: start from max(0, t - rolling_window + 1) to t + 1
            start_idx = max(0, t - self.rolling_window + 1)
            ret_slice = returns_matrix[start_idx : t + 1]
            act_mask = active_mask[t]

            # Require at least 5 timesteps of history to compute meaningful correlation
            if ret_slice.shape[0] < 5:
                edge_idx = torch.empty((2, 0), dtype=torch.long)
                edge_w = torch.empty((0,), dtype=torch.float)
            else:
                edge_idx, edge_w = self.build_adjacency_for_window(ret_slice, act_mask)

            list_edge_indices.append(edge_idx)
            list_edge_weights.append(edge_w)

        avg_edges = np.mean([e.size(1) for e in list_edge_indices])
        logger.info(f"Dynamic graph generated: {num_timesteps} snapshots. Average edges per snapshot: {avg_edges:.1f}")
        return list_edge_indices, list_edge_weights

    def build_adjacency(self, returns_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Backward compatibility for static or single-window DataFrame."""
        returns = returns_df.fillna(0.0).values
        mask = np.ones(returns.shape[1], dtype=bool)
        edge_idx, edge_w = self.build_adjacency_for_window(returns, mask)
        return edge_idx.numpy(), edge_w.numpy()
