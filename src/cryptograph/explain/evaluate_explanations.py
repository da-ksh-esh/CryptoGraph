"""Explanation Evaluation Pipeline: TempME motif discovery, Fidelity+, Fidelity-, and Spearman Rank Correlation."""

from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import torch

from .tempme import TempME
from .validation import ExplanationValidator
from ..common.logger import get_logger

logger = get_logger(__name__)


def run_explanation_benchmark(
    model: Optional[torch.nn.Module] = None,
    checkpoint_path: Optional[str] = None,
    dataset_path: str = "data/processed/v2/1h/processed_dataset.pt",
    network_features_path: Optional[str] = "data/processed/v2/network_features.pt",
    rankings_csv_path: str = "reports/rankings.csv",
    num_sample_events: int = 10,
    output_dir: str = "reports"
) -> Dict[str, Any]:
    """
    Evaluates TempME explanations on test shock events:
    1. Extracts explanatory motifs for target coin shock forecasts.
    2. Computes Fidelity+ (drop when motif ablated) and Fidelity- (retention when only motif preserved).
    3. Computes Characterization Scores (sparsity and motif entropy).
    4. Computes Spearman rank correlation against independent Diebold-Yilmaz NET spillovers and PageRank.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = torch.load(dataset_path, weights_only=False)
    features = data["features"]
    shocks = data["shocks"]
    act_mask = data["active_mask"]
    splits = data["split_indices"]
    asset_names = data["asset_names"]
    test_indices = splits["test_indices"]

    # Fuse network features if available
    if network_features_path and Path(network_features_path).exists():
        net_data = torch.load(network_features_path, weights_only=False)
        if "network_features" in net_data:
            features = torch.cat([features, net_data["network_features"]], dim=-1)

    # Load econometric and network reference rankings
    rankings_raw = pd.read_csv(rankings_csv_path)
    idx_col = "coin" if "coin" in rankings_raw.columns else ("asset" if "asset" in rankings_raw.columns else rankings_raw.columns[0])
    rankings_df = rankings_raw.set_index(idx_col)
    dy_net_series = rankings_df["dy_net_spillover"] if "dy_net_spillover" in rankings_df else pd.Series(dtype=float)
    pagerank_series = rankings_df["pagerank"] if "pagerank" in rankings_df else pd.Series(dtype=float)

    if model is None:
        if checkpoint_path is None:
            raise ValueError("Either model or checkpoint_path must be provided.")
        from ..models.tgn import CryptoTGN
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model = CryptoTGN(
            num_nodes=len(asset_names),
            raw_msg_dim=9,
            node_feat_dim=features.shape[-1],
            memory_dim=64,
            embedding_dim=64,
            out_horizons=len(data.get("horizons", [1, 3, 6, 12, 24]))
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

    explainer = TempME(model, num_epochs=20, lr=0.05, sparsity_weight=0.01, entropy_weight=0.005)
    validator = ExplanationValidator(model)

    explanation_records = []
    f_plus_list = []
    f_minus_list = []
    sparsity_list = []
    entropy_list = []
    spearman_dy_list = []
    spearman_pr_list = []

    returns_np = data["features"][:, :, 0].numpy()
    N = len(asset_names)

    # Find test timesteps with shock occurrences and de-cluster into independent episodes (>= 24h refractory gap)
    test_shock_steps = [t for t in test_indices if (shocks[t] & act_mask[t]).any()]
    episodes = []
    current_ep = []
    for t in test_shock_steps:
        if not current_ep or (t - current_ep[-1] <= 24):
            current_ep.append(t)
        else:
            episodes.append(current_ep)
            current_ep = [t]
    if current_ep:
        episodes.append(current_ep)

    # Select representative independent events across episodes, diversifying target assets
    selected_events = []
    targeted_assets_set = set()

    for ep in episodes:
        # Find peak shock timestep in episode (maximum simultaneous shocked assets)
        peak_t = max(ep, key=lambda t: (shocks[t] & act_mask[t]).sum().item())
        shocked_nodes = torch.where(shocks[peak_t] & act_mask[peak_t])[0].tolist()
        if not shocked_nodes:
            continue

        # Prefer an asset that hasn't been targeted yet to maximize cross-sectional asset diversity
        chosen_node = None
        for node in shocked_nodes:
            if node not in targeted_assets_set:
                chosen_node = node
                break
        if chosen_node is not None:
            targeted_assets_set.add(chosen_node)
            selected_events.append((peak_t, chosen_node))

        if len(selected_events) >= num_sample_events:
            break

    # If fewer than num_sample_events found with strictly unique assets, fill with remaining episodes
    if len(selected_events) < num_sample_events:
        for ep in episodes:
            peak_t = max(ep, key=lambda t: (shocks[t] & act_mask[t]).sum().item())
            if any(ev[0] == peak_t for ev in selected_events):
                continue
            shocked_nodes = torch.where(shocks[peak_t] & act_mask[peak_t])[0].tolist()
            if shocked_nodes:
                best_node = min(shocked_nodes, key=lambda n: returns_np[peak_t, n])
                selected_events.append((peak_t, best_node))
                if len(selected_events) >= num_sample_events:
                    break

    logger.info(
        f"Running TempME explanation benchmark across {len(selected_events)} independent shock episodes "
        f"(>=24h refractory gap, {len(targeted_assets_set)} distinct target assets)..."
    )

    for t, target_node in selected_events:
        target_asset = asset_names[target_node]

        # Construct trailing correlation edges
        w_start = max(0, t - 24)
        r_slice = returns_np[w_start:t]
        corr = np.corrcoef(r_slice, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0)
        np.fill_diagonal(corr, -1.0)

        # Top-5 dynamic edges
        top_k = 5
        top_peers = np.argsort(corr, axis=1)[:, -top_k:]
        src = np.repeat(np.arange(N), top_k)
        dst = top_peers.flatten()
        weights = corr[src, dst]
        pos_mask = weights > 0.0
        src, dst, weights = src[pos_mask], dst[pos_mask], weights[pos_mask]

        if len(src) == 0:
            continue

        e_idx = torch.tensor(np.vstack([src, dst]), dtype=torch.long)
        e_w = torch.tensor(weights, dtype=torch.float32)
        x_t = features[t]

        # Reconstruct continuous memory states up to timestep t
        model.reset_memory()
        w_hist = max(0, t - 168)
        with torch.no_grad():
            for tau in range(w_hist, t + 1):
                shocked_tau = torch.where(shocks[tau] & act_mask[tau])[0]
                active_tau = torch.where(act_mask[tau])[0]
                if len(shocked_tau) > 0 and len(active_tau) > 1:
                    src_l, dst_l = [], []
                    for s in shocked_tau:
                        for d in active_tau:
                            if d != s:
                                src_l.append(s)
                                dst_l.append(d)
                    if len(src_l) > 0:
                        s_t = torch.tensor(src_l, dtype=torch.long)
                        d_t = torch.tensor(dst_l, dtype=torch.long)
                        ts_t = torch.full((len(src_l),), tau, dtype=torch.long)
                        msg_t = features[tau, s_t, :model.raw_msg_dim]
                        model.update_memory(s_t, d_t, msg_t, ts_t)

        # Generate explanation
        exp_res = explainer.explain(
            target_node_idx=target_node,
            node_features=x_t,
            edge_index=e_idx,
            edge_weight=e_w
        )

        edge_mask = exp_res["edge_mask"]
        top_edges = exp_res["top_edges"]
        sparsity = exp_res["sparsity"]
        entropy = exp_res["motif_entropy"]

        # Quantitative validation metrics with exact top-k edge selection and preserved edge weights
        f_plus = validator.compute_fidelity_plus(x_t, e_idx, edge_mask, target_idx=target_node, edge_weight=e_w, top_k=5)
        f_minus = validator.compute_fidelity_minus(x_t, e_idx, edge_mask, target_idx=target_node, edge_weight=e_w, top_k=5)

        # Convert node attributions to asset names
        node_attrib_named = {asset_names[node_id]: score for node_id, score in exp_res["node_attributions"].items()}

        corr_dy = validator.compute_rank_correlation(node_attrib_named, dy_net_series)
        corr_pr = validator.compute_rank_correlation(node_attrib_named, pagerank_series)

        f_plus_list.append(f_plus)
        f_minus_list.append(f_minus)
        sparsity_list.append(sparsity)
        entropy_list.append(entropy)
        spearman_dy_list.append(corr_dy["spearman_rho"])
        spearman_pr_list.append(corr_pr["spearman_rho"])

        # Top edge summary string
        top_edges_str = ";".join([f"{asset_names[e['source']]}->{asset_names[e['target']]}({e['importance']:.2f})" for e in top_edges[:3]])

        explanation_records.append({
            "timestep": t,
            "target_asset": target_asset,
            "fidelity_plus": f_plus,
            "fidelity_minus": f_minus,
            "sparsity": sparsity,
            "motif_entropy": entropy,
            "spearman_dy_net": corr_dy["spearman_rho"],
            "spearman_pagerank": corr_pr["spearman_rho"],
            "top_explanatory_edges": top_edges_str
        })

    # Save detailed explanations table
    df_exp = pd.DataFrame(explanation_records)
    exp_csv = out_dir / "tempme_explanations.csv"
    df_exp.to_csv(exp_csv, index=False)

    # Statistical significance testing on Spearman rank agreement with DY NET
    spearman_dy_arr = np.array(spearman_dy_list) if spearman_dy_list else np.array([])
    if len(spearman_dy_arr) >= 5:
        from scipy import stats
        # Fisher z-transformation
        z_scores = np.arctanh(np.clip(spearman_dy_arr, -0.9999, 0.9999))
        t_stat, p_val_fisher = stats.ttest_1samp(z_scores, 0.0)
        try:
            w_stat, p_val_wilcoxon = stats.wilcoxon(spearman_dy_arr)
        except Exception:
            w_stat, p_val_wilcoxon = 0.0, 1.0
    else:
        t_stat, p_val_fisher = 0.0, 1.0
        w_stat, p_val_wilcoxon = 0.0, 1.0

    # Save aggregate validation table
    summary_metrics = {
        "mean_fidelity_plus": float(np.mean(f_plus_list)) if f_plus_list else 0.0,
        "mean_fidelity_minus": float(np.mean(f_minus_list)) if f_minus_list else 0.0,
        "mean_sparsity": float(np.mean(sparsity_list)) if sparsity_list else 1.0,
        "mean_motif_entropy": float(np.mean(entropy_list)) if entropy_list else 0.0,
        "mean_spearman_dy_net": float(np.mean(spearman_dy_list)) if spearman_dy_list else 0.0,
        "fisher_z_t_statistic": float(t_stat),
        "fisher_z_p_value": float(p_val_fisher),
        "wilcoxon_w_statistic": float(w_stat),
        "wilcoxon_p_value": float(p_val_wilcoxon),
        "mean_spearman_pagerank": float(np.mean(spearman_pr_list)) if spearman_pr_list else 0.0,
        "sample_events_evaluated": len(explanation_records)
    }

    df_summary = pd.DataFrame([summary_metrics])
    val_csv = out_dir / "explanation_validation.csv"
    df_summary.to_csv(val_csv, index=False)

    logger.info(f"Saved TempME explanation benchmarks to {exp_csv} and {val_csv}.")
    return summary_metrics


# Alias for backward compatibility
run_explanation_evaluation = run_explanation_benchmark
