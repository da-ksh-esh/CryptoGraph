"""DyGLib Data Adapter: Exports continuous-time interaction stream to official DyGLib format."""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import json
import numpy as np
import pandas as pd
import torch

from ...common.logger import get_logger

logger = get_logger(__name__)


class DyGLibAdapter:
    """
    Exports CryptoGraph dynamic interaction stream and node features into
    the standardized data format required by official DyGLib (Yu et al., 2023).

    Generates:
        - data/{dataset_name}/ml_{dataset_name}.csv (u, i, ts, label, idx)
        - data/{dataset_name}/ml_{dataset_name}.npy (edge features, 1-indexed)
        - data/{dataset_name}/ml_{dataset_name}_node.npy (node features, 1-indexed)
        - data/{dataset_name}/metadata.json (provenance and split details)
    """

    def __init__(self, dataset_path: str = "data/processed/v2/1h/processed_dataset.pt"):
        self.dataset_path = Path(dataset_path)

    def export(
        self,
        output_dir: str = "data/external/dyglib",
        dataset_name: str = "cryptograph",
        top_k_receivers: Optional[int] = 5
    ) -> Dict[str, Any]:
        """
        Executes the export to DyGLib CSV and NumPy formats.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Processed dataset not found: {self.dataset_path}")

        logger.info(f"Loading {self.dataset_path} for DyGLib conversion...")
        data = torch.load(self.dataset_path, weights_only=False)

        features = data["features"] # (T, N, F)
        shocks = data["shocks"]     # (T, N)
        act_mask = data["active_mask"] # (T, N)
        splits = data["split_indices"]
        timestamps = data["timestamps"]
        asset_names = data["asset_names"]

        T, num_nodes, num_features = features.shape

        # Compute static/overall node features (mean over training split)
        train_idxs = splits["train_indices"]
        train_features = features[train_idxs] # (T_train, N, F)
        node_feat_mean = torch.mean(train_features, dim=0).numpy() # (N, F)

        # 1-indexed node features (index 0 is zero dummy)
        node_features_1idx = np.zeros((num_nodes + 1, num_features), dtype=np.float32)
        node_features_1idx[1:] = node_feat_mean

        # Extract continuous-time interaction events
        u_list = []
        i_list = []
        ts_list = []
        label_list = []
        edge_feat_list = []

        returns = features[:, :, 0].numpy() # column 0 is log return

        logger.info("Extracting dynamic interaction events...")
        for t in range(T):
            shocked = np.where(shocks[t] & act_mask[t])[0]
            active = np.where(act_mask[t])[0]

            if len(shocked) == 0 or len(active) <= 1:
                continue

            for src in shocked:
                # Find receivers (all active peers or top-k by correlation)
                candidates = [d for d in active if d != src]
                if top_k_receivers is not None and len(candidates) > top_k_receivers:
                    # Select top-k by correlation over trailing window
                    w_start = max(0, t - 24)
                    r_sub = returns[w_start : t + 1]
                    if len(r_sub) >= 2:
                        corr = np.corrcoef(r_sub, rowvar=False)
                        corr = np.nan_to_num(corr, nan=0.0)
                        peer_corrs = [corr[src, d] for d in candidates]
                        top_order = np.argsort(peer_corrs)[-top_k_receivers:]
                        candidates = [candidates[idx] for idx in top_order]
                    else:
                        candidates = candidates[:top_k_receivers]

                for dst in candidates:
                    # DyGLib uses 1-based node indices
                    u_list.append(int(src) + 1)
                    i_list.append(int(dst) + 1)
                    ts_list.append(float(t))
                    label_list.append(1) # shock interaction event

                    # Edge feature: source technical feature vector
                    edge_feat_list.append(features[t, src].numpy())

        num_events = len(u_list)
        logger.info(f"Generated {num_events} dynamic interaction events for DyGLib.")

        # Event indices (0 to num_events - 1)
        idx_list = list(range(num_events))

        df_events = pd.DataFrame({
            "u": u_list,
            "i": i_list,
            "ts": ts_list,
            "label": label_list,
            "idx": idx_list
        })

        # Save ml_cryptograph.csv
        csv_file = out_path / f"ml_{dataset_name}.csv"
        df_events.to_csv(csv_file, index=True) # First unnamed index column standard in DyGLib

        # Save ml_cryptograph.npy (1-indexed edge features, index 0 is zero dummy)
        edge_feats_arr = np.array(edge_feat_list, dtype=np.float32)
        edge_feats_1idx = np.zeros((num_events + 1, num_features), dtype=np.float32)
        if num_events > 0:
            edge_feats_1idx[1:] = edge_feats_arr
        npy_edge_file = out_path / f"ml_{dataset_name}.npy"
        np.save(npy_edge_file, edge_feats_1idx)

        # Save ml_cryptograph_node.npy
        npy_node_file = out_path / f"ml_{dataset_name}_node.npy"
        np.save(npy_node_file, node_features_1idx)

        meta = {
            "dataset_name": dataset_name,
            "num_nodes": num_nodes,
            "num_events": num_events,
            "node_feature_dim": num_features,
            "edge_feature_dim": num_features,
            "asset_names": asset_names,
            "train_val_test_splits": {
                "train_timesteps": len(splits["train_indices"]),
                "val_timesteps": len(splits["val_indices"]),
                "test_timesteps": len(splits["test_indices"])
            },
            "output_files": {
                "csv": str(csv_file),
                "edge_npy": str(npy_edge_file),
                "node_npy": str(npy_node_file)
            }
        }

        meta_file = out_path / f"{dataset_name}_meta.json"
        with open(meta_file, "w") as f:
            json.dump(meta, f, indent=2)

        logger.info(f"Successfully exported DyGLib artifacts to {out_path}.")
        return meta
