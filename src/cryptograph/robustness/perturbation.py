"""Robustness Testing: Dual perturbation testing on model predictions AND explanations."""

from typing import Dict, Any, List
import torch
import torch.nn as nn
import numpy as np

class DualPerturbationTester:
    """
    Evaluates model resilience under:
    1. Random Perturbations: Gaussian feature jitter and random edge drops (sensor/marketfeed noise).
    2. Adversarial Perturbations: Fast Gradient Sign Method (FGSM) directed feature perturbations (targeted manipulation).
    Measures prediction drift (MSE) and explanation stability (Jaccard similarity).
    """

    def __init__(self, tgn_model: nn.Module, tempme_explainer):
        self.tgn_model = tgn_model
        self.tempme_explainer = tempme_explainer

    def perturb_features_gaussian(self, x: torch.Tensor, epsilon: float = 0.05) -> torch.Tensor:
        """Injects random Gaussian noise into node features: X + N(0, eps^2)."""
        noise = torch.randn_like(x) * epsilon
        return x + noise

    def perturb_features_fgsm(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        target_idx: int,
        epsilon: float = 0.05
    ) -> torch.Tensor:
        """
        Fast Gradient Sign Method (FGSM) task-specific attack directed to maximally
        flip target node shock classification decisions.
        """
        x_adv = x.clone().detach().requires_grad_(True)
        out = self.tgn_model(x_adv, edge_index)
        pred = out[0][target_idx] if isinstance(out, tuple) else out[target_idx]

        # Task-specific adversarial target: invert the current prediction
        clean_probs = torch.sigmoid(pred.detach())
        inverted_target = 1.0 - (clean_probs > 0.5).float()
        loss = nn.BCEWithLogitsLoss()(pred, inverted_target)
        loss.backward()

        with torch.no_grad():
            if x_adv.grad is not None:
                x_adv = x_adv + epsilon * x_adv.grad.sign()
        return x_adv.detach()

    def perturb_features_pgd(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        target_idx: int,
        epsilon: float = 0.05,
        steps: int = 5
    ) -> torch.Tensor:
        """
        Projected Gradient Descent (PGD) multi-step task-specific attack
        bounded in L_inf norm ball ||x_adv - x||_inf <= epsilon.
        """
        alpha = max(1e-4, epsilon / max(1, steps // 2))
        x_orig = x.clone().detach()
        x_adv = x_orig.clone().detach() + torch.empty_like(x_orig).uniform_(-epsilon / 2.0, epsilon / 2.0)

        for _ in range(steps):
            x_adv.requires_grad_(True)
            out = self.tgn_model(x_adv, edge_index)
            pred = out[0][target_idx] if isinstance(out, tuple) else out[target_idx]

            clean_probs = torch.sigmoid(pred.detach())
            inverted_target = 1.0 - (clean_probs > 0.5).float()
            loss = nn.BCEWithLogitsLoss()(pred, inverted_target)
            loss.backward()

            with torch.no_grad():
                if x_adv.grad is not None:
                    x_adv = x_adv + alpha * x_adv.grad.sign()
                eta = torch.clamp(x_adv - x_orig, min=-epsilon, max=epsilon)
                x_adv = (x_orig + eta).detach()

        return x_adv

    def perturb_topology(self, edge_index: torch.Tensor, num_nodes: int, drop_ratio: float = 0.1) -> torch.Tensor:
        """Randomly removes edge connections to simulate missing market network feeds."""
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
        epsilon_levels: List[float] = [0.01, 0.05, 0.10, 0.20, 0.35]
    ) -> List[Dict[str, Any]]:
        """
        Runs dual stress evaluation across varying perturbation budgets (epsilon).
        Tests Gaussian random perturbation, FGSM adversarial attack, and PGD adversarial attack.
        Reports prediction drift, explanation stability, and empirical attack success rate.
        """
        results = []

        # 1. Clean Baseline
        self.tgn_model.eval()
        with torch.no_grad():
            out_clean = self.tgn_model(node_features, edge_index)
            y_clean = out_clean[0][target_idx] if isinstance(out_clean, tuple) else out_clean[target_idx]
            clean_pred_bool = (torch.sigmoid(y_clean) > 0.5).cpu().numpy()

        clean_exp = self.tempme_explainer.explain(target_idx, node_features, edge_index)
        clean_top_edges = {(e["source"], e["target"]) for e in clean_exp.get("top_edges", [])[:5]}

        suites = [
            ("random_gaussian_and_edge_drop", self.perturb_features_gaussian),
            ("adversarial_fgsm", lambda x, eps: self.perturb_features_fgsm(x, edge_index, target_idx, eps)),
            ("adversarial_pgd", lambda x, eps: self.perturb_features_pgd(x, edge_index, target_idx, eps, steps=5)),
        ]

        for p_name, p_fn in suites:
            for eps in epsilon_levels:
                if p_name == "random_gaussian_and_edge_drop":
                    x_pert = p_fn(node_features, epsilon=eps)
                    edge_pert = self.perturb_topology(edge_index, num_nodes=node_features.size(0), drop_ratio=eps)
                else:
                    x_pert = p_fn(node_features, eps)
                    edge_pert = edge_index

                with torch.no_grad():
                    out_pert = self.tgn_model(x_pert, edge_pert)
                    y_pert = out_pert[0][target_idx] if isinstance(out_pert, tuple) else out_pert[target_idx]
                    drift_mse = float(torch.mean((y_clean - y_pert) ** 2).item())
                    pert_pred_bool = (torch.sigmoid(y_pert) > 0.5).cpu().numpy()
                    attack_success_rate = float(np.mean(clean_pred_bool != pert_pred_bool))

                pert_exp = self.tempme_explainer.explain(target_idx, x_pert, edge_pert)
                pert_top_edges = {(e["source"], e["target"]) for e in pert_exp.get("top_edges", [])[:5]}

                inter = len(clean_top_edges.intersection(pert_top_edges))
                union = max(1, len(clean_top_edges.union(pert_top_edges)))
                jaccard = float(inter / union)

                results.append({
                    "perturbation_type": p_name,
                    "epsilon": eps,
                    "prediction_drift_mse": drift_mse,
                    "explanation_jaccard_stability": jaccard,
                    "attack_success_rate": attack_success_rate
                })

        return results

