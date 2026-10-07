"""Memory-light datasets for hourly data: windows are sliced on the fly from one panel.

Materialising 168-bar windows for 30 coins x ~59k hours would need > 10 GB, so the
panel (T, N, F) is stored once together with an index of valid (t, coin) samples.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..storage import load_pt


def sample_mask_times(times: pd.DatetimeIndex, start, end, *, bar: pd.Timedelta, embargo: pd.Timedelta,
                      lookback_bars: int, horizon: pd.Timedelta) -> np.ndarray:
    """Bars t whose input window [t-(L-1)bar, t] and target window (t, t+horizon] lie inside [start, end]
    and that are at least `embargo` after the split start. `end` is a date (inclusive)."""
    s = pd.Timestamp(start)
    last_bar = pd.Timestamp(end) + pd.Timedelta(days=1) - bar
    first_ok = max(s + embargo, s + (lookback_bars - 1) * bar)
    last_ok = last_bar - horizon
    return np.asarray((times >= first_ok) & (times <= last_ok))


def window_index(active: np.ndarray, y_ok: np.ndarray, sample_rows: np.ndarray, node_idx: list[int],
                 lookback: int, stride: int = 1) -> np.ndarray:
    """(S, 2) int64 array of (t, node) where every bar in the input window is active and targets exist."""
    T = active.shape[0]
    out = []
    for n in node_idx:
        a = active[:, n].astype(np.int64)
        csum = np.concatenate([[0], np.cumsum(a)])
        t = np.arange(lookback - 1, T)
        full = (csum[t + 1] - csum[t + 1 - lookback]) == lookback
        ok = t[full & sample_rows[t] & y_ok[t, n]]
        if stride > 1:
            ok = ok[ok % stride == 0]
        out.append(np.stack([ok, np.full(len(ok), n)], axis=1))
    return np.concatenate(out).astype(np.int64) if out else np.zeros((0, 2), np.int64)


class PanelWindowDataset(Dataset):
    """Items: x (L, F), x_mask (L, F), y_ret (H,), y_rv (H,), y_shock, coin_id, t — sliced from the shared panel."""

    def __init__(self, panel_path: str | Path, index_path: str | Path):
        p = load_pt(panel_path)
        idx = load_pt(index_path)
        self.X, self.X_mask = p["X"], p["X_mask"]
        self.y_ret, self.y_rv, self.y_shock = p["y_ret"], p["y_rv"], p["y_shock"]
        self.index = idx["index"]
        self.lookback = int(idx["lookback"])
        self.meta = {k: p[k] for k in ("coins", "feature_names", "horizons_hours", "bar_hours")}
        self.meta["split"] = idx["split"]

    def __len__(self) -> int:
        return int(self.index.shape[0])

    def __getitem__(self, i: int) -> dict:
        t, n = int(self.index[i, 0]), int(self.index[i, 1])
        sl = slice(t - self.lookback + 1, t + 1)
        return {"x": self.X[sl, n], "x_mask": self.X_mask[sl, n], "y_ret": self.y_ret[t, n], "y_rv": self.y_rv[t, n],
                "y_shock": self.y_shock[t, n], "coin_id": n, "t": t}


class HourlySnapshotDataset(Dataset):
    """One PyG `Data` per hourly graph snapshot whose bar is a sample bar of `split`.

    Nodes = `graph_coins` (event coins excluded); x = scaled features at the snapshot bar; edges from
    graph_snapshots.pt (built from the trailing window up to that bar). Fields as GraphSnapshotDataset,
    with `t` (bar index) instead of `day`.
    """

    def __init__(self, panel_path: str | Path, snapshots_path: str | Path, split: str):
        p = load_pt(panel_path)
        g = load_pt(snapshots_path)
        gi = torch.as_tensor(g["graph_index"])
        self.X, self.X_mask, self.active = p["X"][:, gi], p["X_mask"][:, gi], p["active"][:, gi]
        self.y_ret, self.y_rv, self.y_mask = p["y_ret"][:, gi], p["y_rv"][:, gi], p["y_mask"][:, gi]
        self.y_shock, self.y_shock_mask = p["y_shock"][:, gi], p["y_shock_mask"][:, gi]
        self.edge_index, self.edge_weight = g["undirected_edge_index"], g["undirected_edge_weight"]
        snap_t = g["snapshot_t"].numpy()
        keep = p["sample_mask"][split].numpy()[snap_t]
        self.snap_k, self.snap_t = np.flatnonzero(keep), snap_t[keep]
        self.meta = {"graph_coins": g["graph_coins"], "feature_names": p["feature_names"], "split": split,
                     "threshold": g.get("undirected_threshold"), "bar_hours": p["bar_hours"]}

    def __len__(self) -> int:
        return len(self.snap_k)

    def __getitem__(self, i: int):
        from torch_geometric.data import Data

        k, t = int(self.snap_k[i]), int(self.snap_t[i])
        return Data(x=self.X[t], x_mask=self.X_mask[t], active=self.active[t], edge_index=self.edge_index[k],
                    edge_weight=self.edge_weight[k], y_ret=self.y_ret[t], y_rv=self.y_rv[t], y_mask=self.y_mask[t],
                    y_shock=self.y_shock[t], y_shock_mask=self.y_shock_mask[t], t=torch.tensor([t]),
                    num_nodes=self.X.shape[1])
