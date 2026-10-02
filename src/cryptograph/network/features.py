"""Dynamic, trailing correlation-network feature extraction across cryptocurrency graphs.

Extracts dynamic topological features (centralities, modularity, MST length) strictly
from trailing rolling windows with zero forward lookahead.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import networkx as nx
import torch
import torch.nn as nn
from sklearn.metrics import normalized_mutual_info_score

from .centrality import NetworkCentralityAnalyzer
from .communities import CommunityDetector
from .mst import MinimumSpanningTreeAnalyzer
from ..common.logger import get_logger

logger = get_logger(__name__)


class DynamicNetworkFeatureExtractor:
    """
    Computes per-timestep dynamic network features strictly using trailing windows.
    Guarantees zero future lookahead (strictly trailing correlation graphs).
    """

    def __init__(
        self,
        rolling_window: int = 24,
        min_history: int = 12,
        correlation_threshold: float = 0.35,
        seed: int = 42
    ):
        self.rolling_window = rolling_window
        self.min_history = min_history
        self.threshold = correlation_threshold
        self.seed = seed
        self.community_detector = CommunityDetector(seed=seed)

    def extract_features_at_t(
        self,
        trailing_returns: np.ndarray,
        active_nodes: np.ndarray,
        asset_names: List[str]
    ) -> Dict[str, np.ndarray]:
        """
        Computes network metrics for a single trailing slice of returns [t - W + 1 : t + 1].

        Args:
            trailing_returns: shape (W, N) array of past returns up to time t.
            active_nodes: shape (N,) boolean active mask at time t.
            asset_names: list of N asset names.

        Returns:
            Dictionary mapping feature names to shape (N,) arrays.
        """
        N = len(asset_names)
        active_indices = np.where(active_nodes)[0]

        degree = np.zeros(N, dtype=np.float32)
        betweenness = np.zeros(N, dtype=np.float32)
        closeness = np.zeros(N, dtype=np.float32)
        eigenvector = np.zeros(N, dtype=np.float32)
        pagerank = np.zeros(N, dtype=np.float32)
        community_id = np.zeros(N, dtype=np.int64)

        if len(active_indices) < 3 or trailing_returns.shape[0] < self.min_history:
            return {
                "degree": degree,
                "betweenness": betweenness,
                "closeness": closeness,
                "eigenvector": eigenvector,
                "pagerank": pagerank,
                "community_id": community_id,
                "modularity": 0.0,
                "mst_norm_length": 0.0
            }

        act_returns = trailing_returns[:, active_indices]
        corr = np.corrcoef(act_returns, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0)
        np.fill_diagonal(corr, 0.0)

        # Build NetworkX graph
        G = nx.Graph()
        for local_idx, global_idx in enumerate(active_indices):
            G.add_node(asset_names[global_idx])

        for i in range(len(active_indices)):
            for j in range(i + 1, len(active_indices)):
                c = corr[i, j]
                if abs(c) >= self.threshold:
                    u = asset_names[active_indices[i]]
                    v = asset_names[active_indices[j]]
                    G.add_edge(u, v, weight=float(abs(c)))

        # Centralities
        try:
            cent_df = NetworkCentralityAnalyzer.compute_all_centralities(G)
            for global_idx in active_indices:
                name = asset_names[global_idx]
                if name in cent_df.index:
                    degree[global_idx] = cent_df.loc[name, "degree"]
                    betweenness[global_idx] = cent_df.loc[name, "betweenness"]
                    closeness[global_idx] = cent_df.loc[name, "closeness"]
                    eigenvector[global_idx] = cent_df.loc[name, "eigenvector"]
                    pagerank[global_idx] = cent_df.loc[name, "pagerank"]
        except Exception:
            pass

        # Communities
        comm_res = self.community_detector.detect_communities(G)
        mod_score = comm_res.get("modularity", 0.0)
        node_map = comm_res.get("node_community", {})
        for global_idx in active_indices:
            name = asset_names[global_idx]
            community_id[global_idx] = node_map.get(name, 0)

        # MST Normalized Length
        corr_df = pd.DataFrame(corr, index=[asset_names[i] for i in active_indices], columns=[asset_names[i] for i in active_indices])
        _, mst_len = MinimumSpanningTreeAnalyzer.build_mst(corr_df)

        return {
            "degree": degree,
            "betweenness": betweenness,
            "closeness": closeness,
            "eigenvector": eigenvector,
            "pagerank": pagerank,
            "community_id": community_id,
            "modularity": float(mod_score),
            "mst_norm_length": float(mst_len)
        }

    def build_network_feature_tensors(
        self,
        returns_matrix: np.ndarray,
        active_mask: np.ndarray,
        asset_names: List[str],
        sectors_dict: Optional[Dict[str, str]] = None,
        categorical_encoding: str = "continuous"
    ) -> Tuple[torch.Tensor, List[str], pd.DataFrame]:
        """
        Iterates across all timesteps T to extract strictly trailing network feature tensor.

        Args:
            returns_matrix: (T, N) log return matrix.
            active_mask: (T, N) boolean active mask.
            asset_names: list of asset names.
            sectors_dict: mapping from coin name to sector string.
            categorical_encoding: 'continuous' (legacy 7-dim tensor with warning) or 'one_hot'
                                  (continuous centralities + one-hot binary indicator columns).

        Returns:
            features_tensor: shape (T, N, num_network_features) FloatTensor
            feature_names: list of feature names
            summary_df: DataFrame with modularity and MST length over time
        """
        T, N = returns_matrix.shape
        base_feature_names = ["degree", "betweenness", "closeness", "eigenvector", "pagerank"]
        
        sector_ids = np.zeros(N, dtype=np.int64)
        num_sectors = 1
        if sectors_dict:
            unique_sectors = sorted(list(set(sectors_dict.values())))
            num_sectors = max(1, len(unique_sectors))
            sector_map = {s: idx for idx, s in enumerate(unique_sectors)}
            for i, name in enumerate(asset_names):
                sec = sectors_dict.get(name, "Other")
                sector_ids[i] = sector_map.get(sec, 0)

        # Temporary collection of per-timestep extractions
        modularities = []
        mst_lengths = []
        raw_res = []

        logger.info(f"Extracting dynamic network features across {T} timesteps (window={self.rolling_window})...")

        for t in range(T):
            start_idx = max(0, t - self.rolling_window + 1)
            # Strictly trailing slice: start_idx to t + 1 (includes time t, zero lookahead to t+1)
            ret_slice = returns_matrix[start_idx : t + 1]
            act = active_mask[t]

            res = self.extract_features_at_t(ret_slice, act, asset_names)
            raw_res.append(res)
            modularities.append(res["modularity"])
            mst_lengths.append(res["mst_norm_length"])

        if categorical_encoding == "one_hot":
            # Collect max community ID across all timesteps
            max_comm = max(int(np.max(r["community_id"])) for r in raw_res)
            num_communities = max_comm + 1

            feature_names = list(base_feature_names)
            for c_idx in range(num_communities):
                feature_names.append(f"community_onehot_{c_idx}")
            if sectors_dict:
                for s_idx in range(num_sectors):
                    feature_names.append(f"sector_onehot_{s_idx}")

            num_feats = len(feature_names)
            tensor_out = np.zeros((T, N, num_feats), dtype=np.float32)

            for t in range(T):
                res = raw_res[t]
                tensor_out[t, :, 0] = res["degree"]
                tensor_out[t, :, 1] = res["betweenness"]
                tensor_out[t, :, 2] = res["closeness"]
                tensor_out[t, :, 3] = res["eigenvector"]
                tensor_out[t, :, 4] = res["pagerank"]

                # One-hot communities
                for node_idx in range(N):
                    c_val = int(res["community_id"][node_idx])
                    if 0 <= c_val < num_communities:
                        tensor_out[t, node_idx, 5 + c_val] = 1.0

                # One-hot sectors
                if sectors_dict:
                    offset = 5 + num_communities
                    for node_idx in range(N):
                        s_val = int(sector_ids[node_idx])
                        if 0 <= s_val < num_sectors:
                            tensor_out[t, node_idx, offset + s_val] = 1.0
        else:
            # Continuous mode (for backwards compatibility with existing 7-feature checkpoints)
            feature_names = list(base_feature_names) + ["community_id"]
            if sectors_dict:
                feature_names.append("sector_id")
            num_feats = len(feature_names)
            tensor_out = np.zeros((T, N, num_feats), dtype=np.float32)

            for t in range(T):
                res = raw_res[t]
                tensor_out[t, :, 0] = res["degree"]
                tensor_out[t, :, 1] = res["betweenness"]
                tensor_out[t, :, 2] = res["closeness"]
                tensor_out[t, :, 3] = res["eigenvector"]
                tensor_out[t, :, 4] = res["pagerank"]
                tensor_out[t, :, 5] = res["community_id"].astype(np.float32)
                if sectors_dict:
                    tensor_out[t, :, 6] = sector_ids.astype(np.float32)

        summary_df = pd.DataFrame({
            "modularity": modularities,
            "mst_norm_length": mst_lengths
        })

        return torch.from_numpy(tensor_out), feature_names, summary_df


class CategoricalFeatureEmbedding(nn.Module):
    """
    Learned embedding module for nominal graph attributes (community ID, sector ID).
    Prevents artificial continuous linear ordering on discrete categories.
    """
    def __init__(
        self,
        num_communities: int = 16,
        num_sectors: int = 20,
        comm_dim: int = 4,
        sector_dim: int = 4
    ):
        super().__init__()
        self.comm_embed = nn.Embedding(num_communities, comm_dim)
        self.sector_embed = nn.Embedding(num_sectors, sector_dim)
        self.output_dim = comm_dim + sector_dim

    def forward(self, comm_ids: torch.Tensor, sector_ids: torch.Tensor) -> torch.Tensor:
        """
        Embeds discrete categorical identifiers.
        comm_ids: shape (...,) LongTensor
        sector_ids: shape (...,) LongTensor
        Returns: shape (..., output_dim) FloatTensor
        """
        c_emb = self.comm_embed(comm_ids.long())
        s_emb = self.sector_embed(sector_ids.long())
        return torch.cat([c_emb, s_emb], dim=-1)


# Backwards-compatible alias
CausalNetworkFeatureExtractor = DynamicNetworkFeatureExtractor

