"""Network Compression Research Experiment: Top-K, Thresholding, and MST Backbone Evaluation."""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import spearmanr
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

from .centrality import NetworkCentralityAnalyzer
from .communities import CommunityDetector
from .mst import MinimumSpanningTreeAnalyzer
from ..common.logger import get_logger

logger = get_logger(__name__)


class NetworkCompressor:
    """
    Constructs compressed graph representations and quantifies preservation
    of centrality rankings and community structures against the uncompressed graph.
    """

    @staticmethod
    def build_full_graph(corr_matrix: np.ndarray, asset_names: List[str]) -> nx.Graph:
        """Constructs uncompressed full graph with all non-zero correlation edges."""
        G = nx.Graph()
        for a in asset_names:
            G.add_node(a)
        N = len(asset_names)
        for i in range(N):
            for j in range(i + 1, N):
                c = float(abs(corr_matrix[i, j]))
                if c > 1e-4:
                    G.add_edge(asset_names[i], asset_names[j], weight=c)
        return G

    @staticmethod
    def build_thresholded_graph(corr_matrix: np.ndarray, asset_names: List[str], threshold: float = 0.35) -> nx.Graph:
        """Constructs graph retaining only edges with |correlation| >= threshold."""
        G = nx.Graph()
        for a in asset_names:
            G.add_node(a)
        N = len(asset_names)
        for i in range(N):
            for j in range(i + 1, N):
                c = float(abs(corr_matrix[i, j]))
                if c >= threshold:
                    G.add_edge(asset_names[i], asset_names[j], weight=c)
        return G

    @staticmethod
    def build_topk_graph(corr_matrix: np.ndarray, asset_names: List[str], k: int = 5) -> nx.Graph:
        """Constructs graph where each node connects to its top-k strongest correlation peers."""
        G = nx.Graph()
        for a in asset_names:
            G.add_node(a)
        N = len(asset_names)
        for i in range(N):
            row_corrs = np.abs(corr_matrix[i].copy())
            row_corrs[i] = -1.0  # exclude self
            top_k_indices = np.argsort(row_corrs)[::-1][:k]
            for j in top_k_indices:
                c = float(row_corrs[j])
                if c > 1e-4:
                    G.add_edge(asset_names[i], asset_names[j], weight=c)
        return G

    @staticmethod
    def build_mst_graph(corr_matrix: np.ndarray, asset_names: List[str]) -> nx.Graph:
        """Constructs Minimum Spanning Tree (MST) backbone using Mantegna distance."""
        corr_df = pd.DataFrame(corr_matrix, index=asset_names, columns=asset_names)
        mst, _ = MinimumSpanningTreeAnalyzer.build_mst(corr_df)
        return mst

    @classmethod
    def evaluate_representation(
        cls,
        full_graph: nx.Graph,
        compressed_graph: nx.Graph,
        asset_names: List[str],
        seed: int = 42
    ) -> Dict[str, Any]:
        """
        Quantifies structural preservation: edge reduction, density,
        centrality rank preservation (Spearman rho), and community preservation (NMI, ARI).
        """
        n_nodes = len(asset_names)
        max_edges = n_nodes * (n_nodes - 1) / 2.0
        full_edges = full_graph.number_of_edges()
        comp_edges = compressed_graph.number_of_edges()

        edge_reduction_pct = (1.0 - (comp_edges / max(1, full_edges))) * 100.0
        density = comp_edges / max_edges if max_edges > 0 else 0.0

        # Centralities on both graphs
        cent_full = NetworkCentralityAnalyzer.compute_all_centralities(full_graph)
        cent_comp = NetworkCentralityAnalyzer.compute_all_centralities(compressed_graph)

        def rank_corr(col: str) -> float:
            v_full = np.array([cent_full.loc[a, col] if a in cent_full.index else 0.0 for a in asset_names])
            v_comp = np.array([cent_comp.loc[a, col] if a in cent_comp.index else 0.0 for a in asset_names])
            if np.allclose(v_full, v_comp):
                return 1.0
            if np.std(v_comp) < 1e-6 or np.std(v_full) < 1e-6:
                return 0.0
            r, _ = spearmanr(v_full, v_comp)
            return float(r) if np.isfinite(r) else 0.0

        spearman_degree = rank_corr("degree")
        spearman_strength = rank_corr("strength")
        spearman_betweenness = rank_corr("betweenness")
        spearman_eigenvector = rank_corr("eigenvector")
        spearman_pagerank = rank_corr("pagerank")

        # Community preservation
        comm_detector = CommunityDetector(seed=seed)
        res_full = comm_detector.detect_communities(full_graph)
        res_comp = comm_detector.detect_communities(compressed_graph)

        map_full = res_full["node_community"]
        map_comp = res_comp["node_community"]

        labels_full = [map_full.get(a, 0) for a in asset_names]
        labels_comp = [map_comp.get(a, 0) for a in asset_names]

        nmi = float(normalized_mutual_info_score(labels_full, labels_comp))
        ari = float(adjusted_rand_score(labels_full, labels_comp))

        return {
            "num_edges": comp_edges,
            "edge_reduction_pct": round(edge_reduction_pct, 2),
            "density": round(density, 4),
            "spearman_degree": round(spearman_degree, 4),
            "spearman_strength": round(spearman_strength, 4),
            "spearman_betweenness": round(spearman_betweenness, 4),
            "spearman_eigenvector": round(spearman_eigenvector, 4),
            "spearman_pagerank": round(spearman_pagerank, 4),
            "community_nmi": round(nmi, 4),
            "community_ari": round(ari, 4),
            "num_communities": len(res_comp["communities"]),
            "modularity": round(res_comp["modularity"], 4)
        }

    @classmethod
    def run_benchmark(
        cls,
        corr_matrix: np.ndarray,
        asset_names: List[str],
        seed: int = 42
    ) -> pd.DataFrame:
        """
        Executes full comparative benchmark across compression regimes:
        - Full Uncompressed
        - Thresholded (0.20, 0.35, 0.50, 0.65)
        - Top-K (K=3, 5, 8)
        - Minimum Spanning Tree (MST)
        """
        full_graph = cls.build_full_graph(corr_matrix, asset_names)

        regimes: Dict[str, nx.Graph] = {
            "Full_Graph": full_graph,
            "Threshold_0.20": cls.build_thresholded_graph(corr_matrix, asset_names, threshold=0.20),
            "Threshold_0.35": cls.build_thresholded_graph(corr_matrix, asset_names, threshold=0.35),
            "Threshold_0.50": cls.build_thresholded_graph(corr_matrix, asset_names, threshold=0.50),
            "Threshold_0.65": cls.build_thresholded_graph(corr_matrix, asset_names, threshold=0.65),
            "TopK_3": cls.build_topk_graph(corr_matrix, asset_names, k=3),
            "TopK_5": cls.build_topk_graph(corr_matrix, asset_names, k=5),
            "TopK_8": cls.build_topk_graph(corr_matrix, asset_names, k=8),
            "MST_Backbone": cls.build_mst_graph(corr_matrix, asset_names)
        }

        records = []
        for name, comp_graph in regimes.items():
            metrics = cls.evaluate_representation(full_graph, comp_graph, asset_names, seed=seed)
            metrics["representation"] = name
            records.append(metrics)

        df = pd.DataFrame(records)
        cols_order = [
            "representation", "num_edges", "edge_reduction_pct", "density",
            "spearman_degree", "spearman_eigenvector", "spearman_pagerank", "spearman_betweenness",
            "community_nmi", "community_ari", "modularity"
        ]
        return df[[c for c in cols_order if c in df.columns]]
