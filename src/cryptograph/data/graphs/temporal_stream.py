"""Continuous-time event stream for TGN (torch_geometric.data.TemporalData).

One event per directed edge i -> j on day t, with t = day index since window
start (int64, non-decreasing) and message features
    [lag1_corr, contemp_corr, src_return_scaled, src_vol_z, src_shock]
all computed from data up to t (scaled with the train-fitted scaler).

Edge sources:
  * `shock_edges` (default): i -> every other active j on days where i has a
    shock at t. Direction comes from time ordering; the weight is the trailing
    lag-1 correlation (0 where not estimable).
  * `directed_edges` (graphs.correlation): lag-1 correlation above a threshold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

MSG_NAMES = ["lag1_corr", "contemp_corr", "src_return_scaled", "src_vol_z", "src_shock"]


def shock_edges(shock_t: np.ndarray, active_t: np.ndarray, D_t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Edges i -> j for every shocked, active source i and every other active node j on one day."""
    src = np.flatnonzero(shock_t & active_t)
    dst = np.flatnonzero(active_t)
    if src.size == 0 or dst.size < 2:
        return np.zeros((2, 0), np.int64), np.zeros(0, np.float32)
    i, j = np.meshgrid(src, dst, indexing="ij")
    keep = i != j
    i, j = i[keep], j[keep]
    return np.stack([i, j]).astype(np.int64), np.nan_to_num(D_t[i, j]).astype(np.float32)


def build_event_arrays(edges_per_day: list[tuple[np.ndarray, np.ndarray]], C: np.ndarray,
                       src_return: np.ndarray, src_vol_z: np.ndarray, src_shock: np.ndarray
                       ) -> dict[str, np.ndarray]:
    """Flatten per-day directed edges into sorted event arrays.

    edges_per_day[t] = (edge_index (2, E_t), lag1 weight (E_t,)); C: (T, N, N) contemporaneous corr;
    src_*: (T, N) per-node values at t.
    """
    src, dst, t, msg = [], [], [], []
    for day, (ei, w) in enumerate(edges_per_day):
        if ei.shape[1] == 0:
            continue
        i, j = ei
        order = np.lexsort((j, i))
        i, j, w = i[order], j[order], w[order]
        m = np.stack([w, np.nan_to_num(C[day, i, j]), np.nan_to_num(src_return[day, i]),
                      np.nan_to_num(src_vol_z[day, i]), src_shock[day, i].astype(np.float32)], axis=1)
        src.append(i), dst.append(j), t.append(np.full(len(i), day)), msg.append(m)
    if not src:
        return {"src": np.zeros(0, np.int64), "dst": np.zeros(0, np.int64), "t": np.zeros(0, np.int64),
                "msg": np.zeros((0, len(MSG_NAMES)), np.float32)}
    return {"src": np.concatenate(src).astype(np.int64), "dst": np.concatenate(dst).astype(np.int64),
            "t": np.concatenate(t).astype(np.int64), "msg": np.concatenate(msg).astype(np.float32)}


def to_temporal_data(ev: dict[str, np.ndarray], dates: pd.DatetimeIndex):
    from torch_geometric.data import TemporalData

    data = TemporalData(src=torch.from_numpy(ev["src"]), dst=torch.from_numpy(ev["dst"]),
                        t=torch.from_numpy(ev["t"]), msg=torch.from_numpy(ev["msg"]))
    unix_days = np.asarray(dates.asi8 // 10**9, dtype=np.int64)  # UTC midnight, seconds
    data.t_unix = torch.from_numpy(unix_days[ev["t"]])
    return data


def split_events(data, day_split: np.ndarray) -> dict:
    """Split a TemporalData by the split label of each event's day (date boundaries, not fractions)."""
    labels = day_split[data.t.numpy()]
    out = {}
    for name in ("train", "val", "test"):
        idx = torch.from_numpy(np.flatnonzero(labels == name))
        out[name] = data[idx] if len(idx) else None
    return out
