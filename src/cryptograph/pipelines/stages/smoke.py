"""Stage 4.6 — smoke test: every dataset loads, iterates one epoch, and is trainable.

Asserts: no NaN/inf in model inputs, expected shapes, event-stream time
non-decreasing, every sample date inside its split's allowed sample range,
and one optimisation step of a throwaway linear model on a real batch.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from ...common.logger import get_logger
from ...data.config import DataConfig
from ...data.datasets import (SPLITS, GraphSnapshotDataset, LSTMWindowDataset, load_event_stream,
                              load_tgn_node_features)
from ...data.storage import read_json

logger = get_logger(__name__)


def _finite(t: torch.Tensor, what: str) -> None:
    assert torch.isfinite(t).all(), f"{what}: NaN/inf in inputs"


def _one_step(x: torch.Tensor, y: torch.Tensor) -> float:
    torch.manual_seed(0)
    model = nn.Linear(x.shape[-1], y.shape[-1])
    opt = torch.optim.SGD(model.parameters(), lr=1e-2)
    loss = nn.functional.huber_loss(model(x), y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    return float(loss.detach())


def run(cfg: DataConfig, refresh: bool = False) -> None:
    d = cfg.processed_dir
    splits = read_json(d / "splits.json")["splits"]
    L, H = cfg.lookback_window, len(cfg.targets.horizons)

    for s in SPLITS:
        lo, hi = pd.Timestamp(splits[s]["first_sample_date"]), pd.Timestamp(splits[s]["last_sample_date"])

        # LSTM windows
        ds = LSTMWindowDataset(d / f"lstm_{s}.pt")
        dates = pd.DatetimeIndex(ds.meta["dates"])
        n = 0
        for b in DataLoader(ds, batch_size=512, shuffle=(s == "train")):
            _finite(b["x"], f"lstm {s}")
            assert b["x"].shape[1:] == (L, len(ds.meta["feature_names"])) and b["y_ret"].shape[1] == H
            bd = dates[b["day"].numpy()]
            assert (bd >= lo).all() and (bd <= hi).all(), f"lstm {s}: sample date outside split"
            n += len(b["x"])
        assert n == len(ds)
        loss = _one_step(b["x"][:, -1, :], b["y_ret"])
        logger.info(f"lstm_{s}: {n} samples ok, linear step loss {loss:.4f}")

        # graph snapshots (PyG)
        from torch_geometric.loader import DataLoader as PygLoader

        for et in ("undirected", "directed"):
            gs = GraphSnapshotDataset(d / "graph_snapshots.pt", s, edge_type=et)
            gdates = pd.DatetimeIndex(gs.meta["dates"])
            m = 0
            for batch in PygLoader(gs, batch_size=32):
                _finite(batch.x, f"snapshot {s}")
                assert batch.edge_index.max() < batch.num_nodes if batch.edge_index.numel() else True
                bd = gdates[batch.day.numpy()]
                assert (bd >= lo).all() and (bd <= hi).all()
                m += batch.num_graphs
            assert m == len(gs)
        loss = _one_step(batch.x, batch.y_ret)
        logger.info(f"graph_snapshots {s}: {m} days ok (both edge types), linear step loss {loss:.4f}")

    # TGN event stream
    from torch_geometric.loader import TemporalDataLoader

    ev = load_event_stream(d / "tgn_events.pt")
    nf = load_tgn_node_features(d / "tgn_node_features.pt")
    full = ev["full"]
    assert (full.t[1:] >= full.t[:-1]).all(), "event stream time not sorted"
    edates = pd.DatetimeIndex(ev["dates"])
    for s in SPLITS:
        part = ev[s]
        if part is None:
            continue
        bd = edates[part.t.numpy()]
        assert (bd >= pd.Timestamp(splits[s]["start"])).all() and (bd <= pd.Timestamp(splits[s]["end"])).all()
        last_t = -1
        for batch in TemporalDataLoader(part, batch_size=200):
            _finite(batch.msg, f"tgn {s}")
            assert batch.t.min() >= last_t, "time went backwards across batches"
            last_t = int(batch.t.max())
            x_src = nf["x"][batch.t, batch.src]
            _finite(x_src, f"tgn node features {s}")
        loss = _one_step(torch.cat([batch.msg, x_src], dim=1), x_src[:, :1])
        logger.info(f"tgn_events {s}: {part.num_events} events ok, linear step loss {loss:.4f}")

    print("SMOKE TEST PASSED")
