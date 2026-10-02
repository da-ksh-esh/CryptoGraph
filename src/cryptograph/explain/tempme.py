"""Temporal Edge-Mask Explainer for Dynamic Graph Neural Networks (TempME Adapter).

Learns continuous edge importance masks over dynamic computation subgraphs to causally
attribute shock contagion predictions to specific cross-asset transmission channels.
Optimization objective minimizes prediction retention MSE while penalizing mask density (L1)
and promoting concentrated attention (Shannon entropy).
"""

from typing import Dict, Any, List, Optional, Tuple
import torch
import torch.nn as nn
import numpy as np

from ..common.logger import get_logger

logger = get_logger(__name__)


class TemporalEdgeMaskExplainer:
    """
    Temporal Edge-Mask Explainer for dynamic GNNs and TGN models.
    Identifies influential interaction edges in the local computation subgraph
    responsible for predicting shock contagion at target nodes.

    Objective:
        L = MSE(f(X, E, M * W), f(X, E, W)) + lambda_sparse * mean(M) + lambda_entropy * H(M)
        (Parameterized continuous edge-mask optimization with sparsity and entropy penalties).
    """

    def __init__(
        self,
        model: nn.Module,
        num_epochs: int = 50,
        lr: float = 0.05,
        sparsity_weight: float = 0.01,
        entropy_weight: float = 0.005,
        threshold: float = 0.5
    ):
        self.model = model
        self.num_epochs = num_epochs
        self.lr = lr
        self.sparsity_weight = sparsity_weight
        self.entropy_weight = entropy_weight
        self.threshold = threshold

    def extract_computation_subgraph(
        self,
        target_idx: int,
        edge_index: torch.Tensor,
        num_hops: int = 2
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts the k-hop computation subgraph incident to target_idx.
        Returns:
            sub_edge_index: (2, E_sub)
            sub_edge_mask: boolean mask of shape (E,) indexing original edges
        """
        num_edges = edge_index.size(1) if edge_index is not None else 0
        if num_edges == 0:
            return torch.empty((2, 0), dtype=torch.long, device=edge_index.device), torch.zeros(0, dtype=torch.bool, device=edge_index.device)

        current_nodes = {target_idx}
        all_edges = set()

        for _ in range(num_hops):
            new_nodes = set()
            for e_idx in range(num_edges):
                src = int(edge_index[0, e_idx])
                dst = int(edge_index[1, e_idx])
                if dst in current_nodes or src in current_nodes:
                    all_edges.add(e_idx)
                    new_nodes.add(src)
                    new_nodes.add(dst)
            current_nodes.update(new_nodes)

        edge_mask = torch.zeros(num_edges, dtype=torch.bool, device=edge_index.device)
        if len(all_edges) > 0:
            edge_mask[list(all_edges)] = True

        return edge_index[:, edge_mask], edge_mask

    def explain(
        self,
        target_node_idx: int,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        timestamps: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Explains target node prediction by optimizing continuous motif edge weights.

        Args:
            target_node_idx: Node to explain.
            node_features: (N, F) instantaneous node features.
            edge_index: (2, E) dynamic graph edges.
            edge_weight: Optional (E,) edge weights.
            timestamps: Optional timestamp tensor.

        Returns:
            Dictionary containing edge_mask, top_edges, node_attributions, and characterization scores.
        """
        num_edges = edge_index.size(1) if edge_index is not None else 0
        if num_edges == 0:
            return {
                "edge_mask": np.array([]),
                "top_edges": [],
                "node_attributions": {},
                "sparsity": 1.0,
                "motif_entropy": 0.0
            }

        self.model.eval()
        with torch.no_grad():
            out_orig = self.model(node_features, edge_index, edge_weight)
            target_pred = out_orig[0][target_node_idx] if isinstance(out_orig, tuple) else out_orig[target_node_idx]

        # Parameterized motif mask (logits)
        mask_param = nn.Parameter(torch.randn(num_edges, device=edge_index.device) * 0.1)
        optimizer = torch.optim.Adam([mask_param], lr=self.lr)

        base_weight = edge_weight if edge_weight is not None else torch.ones(num_edges, device=edge_index.device)

        for _ in range(self.num_epochs):
            optimizer.zero_grad()
            weights = torch.sigmoid(mask_param) * base_weight

            out_masked = self.model(node_features, edge_index, weights)
            masked_pred = out_masked[0][target_node_idx] if isinstance(out_masked, tuple) else out_masked[target_node_idx]

            # 1. Fidelity loss (MSE to original prediction)
            pred_loss = torch.mean((masked_pred - target_pred) ** 2)

            # 2. Sparsity penalty
            norm_weights = torch.sigmoid(mask_param)
            sparse_loss = self.sparsity_weight * torch.mean(norm_weights)

            # 3. Entropy penalty
            p = norm_weights / (torch.sum(norm_weights) + 1e-8)
            entropy_loss = -self.entropy_weight * torch.sum(p * torch.log(p + 1e-8))

            loss = pred_loss + sparse_loss + entropy_loss
            loss.backward()
            optimizer.step()

        final_mask = torch.sigmoid(mask_param).detach().cpu().numpy()

        # Identify top explanatory motif edges
        top_indices = np.argsort(final_mask)[::-1]
        top_edges = []
        node_attribs: Dict[int, float] = {}

        for idx in top_indices:
            score = float(final_mask[idx])
            src = int(edge_index[0, idx])
            dst = int(edge_index[1, idx])
            top_edges.append({
                "source": src,
                "target": dst,
                "importance": score
            })
            node_attribs[src] = node_attribs.get(src, 0.0) + score
            node_attribs[dst] = node_attribs.get(dst, 0.0) + score

        sparsity = float(np.mean(final_mask < self.threshold))
        norm_p = final_mask / (np.sum(final_mask) + 1e-8)
        entropy = float(-np.sum(norm_p * np.log(norm_p + 1e-8)))

        return {
            "edge_mask": final_mask,
            "top_edges": top_edges[:10],
            "node_attributions": node_attribs,
            "sparsity": sparsity,
            "motif_entropy": entropy
        }


# Alias for backward compatibility
TempME = TemporalEdgeMaskExplainer

