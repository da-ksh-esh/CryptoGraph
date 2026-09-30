"""Sliding-window samples per coin for non-graph sequence models (LSTM baseline)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def build_windows(X: np.ndarray, X_mask: np.ndarray, active: np.ndarray, y_ret: np.ndarray, y_rv: np.ndarray,
                  y_shock: np.ndarray, y_shock_mask: np.ndarray, sample_days: np.ndarray, node_idx: list[int],
                  lookback: int) -> dict[str, np.ndarray]:
    """Windows X[t-L+1..t] for each (t, coin) with t in sample_days.

    Dropped when any day in the input window is inactive or any regression / shock target is missing.
    """
    xs, ms, yr, yv, ys, cid, day = [], [], [], [], [], [], []
    y_ok = np.isfinite(y_ret).all(-1) & np.isfinite(y_rv).all(-1) & y_shock_mask
    for n in node_idx:
        for t in np.flatnonzero(sample_days):
            lo = t - lookback + 1
            if lo < 0 or not active[lo: t + 1, n].all() or not y_ok[t, n]:
                continue
            xs.append(X[lo: t + 1, n]), ms.append(X_mask[lo: t + 1, n])
            yr.append(y_ret[t, n]), yv.append(y_rv[t, n]), ys.append(y_shock[t, n])
            cid.append(n), day.append(t)
    F, H = X.shape[-1], y_ret.shape[-1]
    if not xs:
        return {"X": np.zeros((0, lookback, F), np.float32), "X_mask": np.zeros((0, lookback, F), bool),
                "y_ret": np.zeros((0, H), np.float32), "y_rv": np.zeros((0, H), np.float32),
                "y_shock": np.zeros(0, bool), "coin_id": np.zeros(0, np.int64), "day": np.zeros(0, np.int64)}
    return {"X": np.stack(xs).astype(np.float32), "X_mask": np.stack(ms), "y_ret": np.stack(yr).astype(np.float32),
            "y_rv": np.stack(yv).astype(np.float32), "y_shock": np.asarray(ys, bool),
            "coin_id": np.asarray(cid, np.int64), "day": np.asarray(day, np.int64)}


class LSTMWindowDataset(Dataset):
    """Loads data/processed/<v>/lstm_<split>.pt. Items: x (L,F), x_mask, y_ret (H,), y_rv (H,), y_shock, coin_id, day."""

    def __init__(self, path: str | Path):
        d = torch.load(path, weights_only=False)
        self.meta = {k: d[k] for k in ("coins", "feature_names", "horizons", "dates", "split", "lookback")}
        self.X, self.X_mask = d["X"], d["X_mask"]
        self.y_ret, self.y_rv, self.y_shock = d["y_ret"], d["y_rv"], d["y_shock"]
        self.coin_id, self.day = d["coin_id"], d["day"]

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        return {"x": self.X[i], "x_mask": self.X_mask[i], "y_ret": self.y_ret[i], "y_rv": self.y_rv[i],
                "y_shock": self.y_shock[i], "coin_id": self.coin_id[i], "day": self.day[i]}
