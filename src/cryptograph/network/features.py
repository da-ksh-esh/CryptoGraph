"""Causal, trailing network feature extraction across dynamic cryptocurrency graphs."""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import networkx as nx
import torch
from sklearn.metrics import normalized_mutual_info_score

from .centrality import NetworkCentralityAnalyzer
from .communities import CommunityDetector
from .mst import MinimumSpanningTreeAnalyzer
from ..common.logger import get_logger

logger = get_logger(__name__)


class CausalNetworkFeatureExtractor:
    """
    Computes per-timestep dynamic network features strictly using trailing windows.
    Guarantees zero future lookahead (strictly causal).
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
        sectors_dict: Optional[Dict[str, str]] = None
    ) -> Tuple[torch.Tensor, List[str], pd.DataFrame]:
        """
        Iterates across all timesteps T to extract strictly trailing network feature tensor.

        Returns:
            features_tensor: shape (T, N, num_network_features) FloatTensor
            feature_names: list of feature names
            summary_df: DataFrame with modularity and MST length over time
        """
        T, N = returns_matrix.shape
        feature_names = ["degree", "betweenness", "closeness", "eigenvector", "pagerank", "community_id"]
        
        sector_ids = np.zeros(N, dtype=np.float32)
        if sectors_dict:
            unique_sectors = sorted(list(set(sectors_dict.values())))
            sector_map = {s: idx for idx, s in enumerate(unique_sectors)}
            for i, name in enumerate(asset_names):
                sec = sectors_dict.get(name, "Other")
                sector_ids[i] = sector_map.get(sec, 0)
            feature_names.append("sector_id")

        num_feats = len(feature_names)
        tensor_out = np.zeros((T, N, num_feats), dtype=np.float32)

        modularities = []
        mst_lengths = []

        logger.info(f"Extracting causal network features across {T} timesteps (window={self.rolling_window})...")

        for t in range(T):
            start_idx = max(0, t - self.rolling_window + 1)
            # Strictly trailing slice: start_idx to t + 1 (includes time t, zero lookahead to t+1)
            ret_slice = returns_matrix[start_idx : t + 1]
            act = active_mask[t]

            res = self.extract_features_at_t(ret_slice, act, asset_names)

            tensor_out[t, :, 0] = res["degree"]
            tensor_out[t, :, 1] = res["betweenness"]
            tensor_out[t, :, 2] = res["closeness"]
            tensor_out[t, :, 3] = res["eigenvector"]
            tensor_out[t, :, 4] = res["pagerank"]
            tensor_out[t, :, 5] = res["community_id"].astype(np.float32)
            if sectors_dict:
                tensor_out[t, :, 6] = sector_ids

            modularities.append(res["modularity"])
            mst_lengths.append(res["mst_norm_length"])

        summary_df = pd.DataFrame({
            "modularity": modularities,
            "mst_norm_length": mst_lengths
        })

        return torch.from_numpy(tensor_out), feature_names, summary_df
