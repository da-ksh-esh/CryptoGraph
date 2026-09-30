"""Temporal Edge Mask Explainer (Dual-Space Sensitivity Attribution).

Identifies critical inter-asset shock propagation pathways by evaluating prediction and
representation sensitivity under counterfactual edge ablation.
"""

from typing import Dict, Any, List, Optional
import torch
import torch.nn as nn
import numpy as np

class EdgeMaskExplainer:
    """
    Evaluates temporal and spatial edge importance for GNN predictions
    using counterfactual edge ablation in forecast prediction space with
    automatic fallback to topological embedding space.
    """

    def __init__(
        self,
        model: nn.Module,
        num_epochs: int = 50,
        lr: float = 0.05,
        sparsity_weight: float = 0.001
    ):
        self.model = model
        self.num_epochs = num_epochs
        self.lr = lr
        self.sparsity_weight = sparsity_weight

    def explain(
        self,
        target_node_idx: int,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        timestamps: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Explains target node shock forecast by ranking dynamic incoming edges.
        """
        num_edges = edge_index.size(1) if edge_index is not None else 0
        if num_edges == 0:
            return {"edge_mask": np.array([]), "top_edges": [], "sparsity": 1.0}

        incident_mask = (edge_index[1] == target_node_idx)
        incident_indices = torch.where(incident_mask)[0]

        if len(incident_indices) == 0:
            incident_mask = (edge_index[0] == target_node_idx) | (edge_index[1] == target_node_idx)
            incident_indices = torch.where(incident_mask)[0]

        if edge_weight is not None:
            base_w = torch.abs(edge_weight).clamp(min=1e-4, max=1.0)
        else:
            base_w = torch.ones(num_edges, device=edge_index.device)

        self.model.eval()
        with torch.no_grad():
            out = self.model(node_features, edge_index, base_w, timestamps)
            if isinstance(out, tuple):
                orig_preds, orig_z = out[0], out[1]
            else:
                orig_preds, orig_z = out, None
            target_orig_p = orig_preds[target_node_idx].detach()
            target_orig_z = orig_z[target_node_idx].detach() if orig_z is not None else None

        sensitivities = []
        with torch.no_grad():
            for g_idx in incident_indices:
                w_ablated = base_w.clone()
                w_ablated[g_idx] = 1e-4
                ab_out = self.model(node_features, edge_index, w_ablated, timestamps)
                if isinstance(ab_out, tuple):
                    ablated_preds, ablated_z = ab_out[0], ab_out[1]
                else:
                    ablated_preds, ablated_z = ab_out, None

                p_diff = torch.norm(target_orig_p - ablated_preds[target_node_idx], p=2).item()
                if target_orig_z is not None and ablated_z is not None:
                    z_diff = torch.norm(target_orig_z - ablated_z[target_node_idx], p=2).item()
                else:
                    z_diff = 0.0

                diff = p_diff if p_diff > 1e-6 else z_diff
                sensitivities.append(diff)

        sens_arr = np.array(sensitivities)
        max_sens = np.max(sens_arr) if len(sens_arr) > 0 and np.max(sens_arr) > 1e-8 else 1.0

        global_mask = np.zeros(num_edges, dtype=np.float32)
        if len(incident_indices) > 0:
            normalized_sens = sens_arr / max_sens
            global_mask[incident_indices.cpu().numpy()] = normalized_sens

            ranked_local = np.argsort(-normalized_sens)
            top_edges = []
            for loc_idx in ranked_local:
                g_idx = incident_indices[loc_idx].item()
                src = int(edge_index[0, g_idx])
                dst = int(edge_index[1, g_idx])
                score = float(normalized_sens[loc_idx])
                top_edges.append({
                    "source": src,
                    "target": dst,
                    "importance": score,
                    "raw_impact": float(sens_arr[loc_idx])
                })
        else:
            top_edges = []

        return {
            "edge_mask": global_mask,
            "top_edges": top_edges,
            "sparsity": float(np.mean(global_mask < 0.5))
        }

# Deprecated alias for backwards compatibility
TempME = EdgeMaskExplainer
