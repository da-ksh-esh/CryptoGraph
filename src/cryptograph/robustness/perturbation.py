"""Robustness Testing: Dual perturbation testing on model predictions AND explanations."""

from typing import Dict, Any, List
import torch
import torch.nn as nn
import numpy as np

class DualPerturbationTester:
    """
    Evaluates adversarial resilience of:
    1. TGN's contagion forecasts (Prediction Degradation)
    2. Explainer's temporal motif explanations (Explanation Stability)
    under feature and graph topology perturbations.
    """

    def __init__(self, tgn_model: nn.Module, tempme_explainer):
        self.tgn_model = tgn_model
        self.tempme_explainer = tempme_explainer

    def perturb_features(self, x: torch.Tensor, epsilon: float = 0.05) -> torch.Tensor:
        """Injects bounded Gaussian noise into node features."""
        noise = torch.randn_like(x) * epsilon
        return x + noise

    def perturb_topology(self, edge_index: torch.Tensor, num_nodes: int, drop_ratio: float = 0.1) -> torch.Tensor:
        """Randomly removes edge connections to simulate corrupted market networks."""
        num_edges = edge_index.size(1)
        if num_edges == 0:
            return edge_index

        keep_count = max(1, int(num_edges * (1.0 - drop_ratio)))
        perm = torch.randperm(num_edges)[:keep_count]
        return edge_index[:, perm]

    def run_stress_test(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        target_idx: int,
        epsilon_levels: List[float] = [0.01, 0.05, 0.10, 0.20]
    ) -> List[Dict[str, Any]]:
        """
        Runs dual evaluation across varying perturbation budgets (epsilon).
        Measures:
        - prediction_mse_drop: Drift in model predictions.
        - explanation_jaccard_stability: Overlap of top temporal motif edges before vs after perturbation.
        """
        results = []

        # 1. Clean Baseline
        self.tgn_model.eval()
        with torch.no_grad():
            out_clean = self.tgn_model(node_features, edge_index)
            y_clean = out_clean[0][target_idx] if isinstance(out_clean, tuple) else out_clean[target_idx]

        clean_exp = self.tempme_explainer.explain(target_idx, node_features, edge_index)
        clean_top_edges = {(e["source"], e["target"]) for e in clean_exp.get("top_edges", [])[:5]}

        # 2. Perturbation Suite
        for eps in epsilon_levels:
            x_pert = self.perturb_features(node_features, epsilon=eps)
            edge_pert = self.perturb_topology(edge_index, num_nodes=node_features.size(0), drop_ratio=eps)

            # Measure prediction drift
            with torch.no_grad():
                out_pert = self.tgn_model(x_pert, edge_pert)
                y_pert = out_pert[0][target_idx] if isinstance(out_pert, tuple) else out_pert[target_idx]
                pred_drift = float(torch.mean((y_clean - y_pert) ** 2).item())

            # Measure explanation stability
            pert_exp = self.tempme_explainer.explain(target_idx, x_pert, edge_pert)
            pert_top_edges = {(e["source"], e["target"]) for e in pert_exp.get("top_edges", [])[:5]}

            # Jaccard similarity of top motif edges
            if clean_top_edges or pert_top_edges:
                intersection = len(clean_top_edges.intersection(pert_top_edges))
                union = len(clean_top_edges.union(pert_top_edges))
                jaccard_stability = float(intersection / union) if union > 0 else 1.0
            else:
                jaccard_stability = 1.0

            results.append({
                "epsilon": eps,
                "prediction_drift_mse": pred_drift,
                "explanation_jaccard_stability": jaccard_stability
            })

        return results
