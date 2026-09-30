"""Daily graph snapshots (T-GAE, snapshot GNNs) as PyG `Data` objects."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class GraphSnapshotDataset(Dataset):
    """Loads data/processed/<v>/graph_snapshots.pt and yields one PyG Data per sample day of `split`.

    edge_type: "undirected" (rolling |corr| >= thr + top-k) or "directed" (lag-1 corr i->j > thr).
    Data fields: x (N,F) scaled+clipped (0 where missing), x_mask (N,F), active (N,), edge_index, edge_weight,
    y_ret (N,H), y_rv (N,H), y_mask (N,H), y_shock (N,), y_shock_mask (N,), day (1,).
    """

    def __init__(self, path: str | Path, split: str, edge_type: str = "undirected"):
        d = torch.load(path, weights_only=False)
        self.meta = {k: d[k] for k in ("coins", "feature_names", "horizons", "dates")}
        self.d = d
        self.edge_key = {"undirected": "undirected", "directed": "directed"}[edge_type]
        self.days = np.flatnonzero(d["sample_mask"][split].numpy())

    def __len__(self) -> int:
        return len(self.days)

    def __getitem__(self, i: int):
        from torch_geometric.data import Data

        t = int(self.days[i])
        d = self.d
        ei, ew = d[f"{self.edge_key}_edge_index"][t], d[f"{self.edge_key}_edge_weight"][t]
        return Data(x=d["X"][t], x_mask=d["X_mask"][t], active=d["active"][t], edge_index=ei, edge_weight=ew,
                    y_ret=d["y_ret"][t], y_rv=d["y_rv"][t], y_mask=d["y_mask"][t], y_shock=d["y_shock"][t],
                    y_shock_mask=d["y_shock_mask"][t], day=torch.tensor([t]), num_nodes=d["X"].shape[1])
