"""TempME: Temporal Motif Explainer for Dynamic Graph Contagion Networks."""

from typing import Dict, List, Tuple, Any
import torch
import torch.nn as nn
import numpy as np

class TempME:
    """
    Temporal Motif Explainer (TempME) for Temporal Graph Networks.
    Identifies the key sequence of timestamped edges (motifs) that explain
    why a target cryptocurrency experienced a shock spillover.
    """

    def __init__(self, model: nn.Module, num_epochs: int = 50, lr: float = 0.05, sparsity_weight: float = 0.01):
        self.model = model
        self.num_epochs = num_epochs
        self.lr = lr
        self.sparsity_weight = sparsity_weight

    def explain(
        self,
        target_node_idx: int,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor = None,
        timestamps: torch.Tensor = None
    ) -> Dict[str, Any]:
        """
        Learns an edge importance mask over the temporal neighborhood of target_node_idx.

        Returns:
            Dictionary containing:
                - edge_mask: Normalized importance scores [0, 1] for each edge.
                - top_edges: Ranked list of most influential directed connections.
                - sparsity: Fraction of non-essential edges pruned.
        """
        num_edges = edge_index.size(1)
        if num_edges == 0:
            return {"edge_mask": np.array([]), "top_edges": [], "sparsity": 1.0}

        # Initialize learnable mask over edges
        mask_param = nn.Parameter(torch.zeros(num_edges, requires_grad=True))
        optimizer = torch.optim.Adam([mask_param], lr=self.lr)

        # Freeze model parameters to prevent backward conflict
        param_states = [p.requires_grad for p in self.model.parameters()]
        for p in self.model.parameters():
            p.requires_grad = False

        self.model.eval()
        with torch.no_grad():
            orig_preds, _ = self.model(node_features, edge_index, edge_weight, timestamps)
            target_orig = orig_preds[target_node_idx].detach()

        # Optimize mask to identify critical edges
        for epoch in range(self.num_epochs):
            optimizer.zero_grad()
            mask = torch.sigmoid(mask_param)

            # Pass mask as dynamic edge weights
            masked_preds, _ = self.model(node_features, edge_index, edge_weight=mask, timestamps=timestamps)

            pred_loss = nn.MSELoss()(masked_preds[target_node_idx], target_orig)
            sparsity_loss = self.sparsity_weight * torch.sum(mask)

            loss = pred_loss + sparsity_loss
            loss.backward()
            optimizer.step()

        # Restore model parameter grad states
        for p, state in zip(self.model.parameters(), param_states):
            p.requires_grad = state

        final_mask = torch.sigmoid(mask_param).detach().cpu().numpy()
        ranked_indices = np.argsort(-final_mask)

        top_edges = []
        for idx in ranked_indices[:min(10, num_edges)]:
            src = int(edge_index[0, idx])
            dst = int(edge_index[1, idx])
            score = float(final_mask[idx])
            top_edges.append({"source": src, "target": dst, "importance": score})

        return {
            "edge_mask": final_mask,
            "top_edges": top_edges,
            "sparsity": float(np.mean(final_mask < 0.5))
        }
