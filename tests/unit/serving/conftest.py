"""A tiny trained-TGN run on disk (data/processed/v2/1h/ + runs/<run_id>/) under a temporary project root."""

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

K_HOURS = [1, 3, 6]
COINS = ["AAA", "BBB", "CCC", "DDD", "EEE"]
RUN_NAME = "20261007-000000_toy_tgn_0_abc1234"


def toy_panel(T=300, F=3, seed=0) -> dict:
    """Panel fields used by the TGN loader and the shock context (feature 0 = log_return)."""
    rng = np.random.default_rng(seed + 100)
    N = len(COINS)
    z = rng.standard_t(4, size=(T, N)).astype(np.float32)
    X = rng.normal(size=(T, N, F)).astype(np.float32)
    X_raw = X.copy()
    X_raw[..., 0] = 0.01 * z
    return {"X": torch.from_numpy(X), "X_raw": torch.from_numpy(X_raw), "graph_coins": COINS,
            "feature_names": ["log_return"] + [f"f{k}" for k in range(1, F)],
            "times": [str(x) for x in pd.date_range("2026-01-01", periods=T, freq="h")],
            "shock_z": torch.from_numpy(z), "shock_today": torch.from_numpy(np.abs(z) > 2.29),
            "shock_today_mask": torch.ones(T, N, dtype=torch.bool), "active": torch.ones(T, N, dtype=torch.bool)}


def _write_project(root, T=300, F=3, seed=0):
    from torch_geometric.data import TemporalData

    rng = np.random.default_rng(seed)
    N = len(COINS)
    split_id = np.r_[np.zeros(150), np.ones(70), np.full(T - 220, 2)].astype(np.int64)
    sample_mask = {s: torch.from_numpy(split_id == i) for i, s in enumerate(("train", "val", "test"))}
    src, dst, t = [], [], []
    for b in range(T):                                     # 1-2 coins shock per bar, an event to every other coin
        for i in rng.choice(N, size=rng.integers(1, 3), replace=False):
            for j in range(N):
                if j != i:
                    src.append(i), dst.append(j), t.append(b)
    E = len(t)
    d = TemporalData(src=torch.tensor(src), dst=torch.tensor(dst), t=torch.tensor(t),
                     msg=torch.from_numpy(rng.normal(size=(E, 5)).astype(np.float32)))
    d.y = torch.from_numpy(rng.random((E, len(K_HOURS))) < 0.3)
    d.y_mask = torch.from_numpy(np.asarray(t)[:, None] < T - np.asarray(K_HOURS)[None])   # no label at the end
    base = root / "data" / "processed" / "v2" / "1h"
    base.mkdir(parents=True)
    torch.save(toy_panel(T, F, seed) | {"split_id": torch.from_numpy(split_id), "sample_mask": sample_mask},
               base / "panel.pt")
    torch.save({"full": d, "graph_index": list(range(N)), "label_k_hours": K_HOURS,
                "msg_names": [f"m{k}" for k in range(5)]}, base / "tgn_events.pt")
    return d, F


def _write_run(root, d, F, seed=0):
    from cryptograph.models.tgn import ContagionTGN

    torch.manual_seed(seed)
    mcfg = {"memory_dim": 8, "time_dim": 4, "embedding_dim": 8, "dropout": 0.0, "use_memory": True}
    model = ContagionTGN(num_nodes=len(COINS), msg_dim=5, node_feat_dim=F, out_horizons=len(K_HOURS), **mcfg)
    run = root / "runs" / RUN_NAME
    run.mkdir(parents=True)
    cfg = {"name": "toy_tgn", "data": {"version": "v2", "resolution": "1h", "network_features": False},
           "training": {"batch_size": 20, "num_neighbors": 5}}
    (run / "config.resolved.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    torch.save({"model_state": model.state_dict(), "model_config": mcfg, "k_hours": K_HOURS, "ctx_dim": 0},
               run / "checkpoint.pt")
    rng = np.random.default_rng(seed)
    n = 400
    prob = rng.random((n, len(K_HOURS)))
    torch.save({"prob": torch.from_numpy(prob).half(), "y": torch.from_numpy(rng.random((n, len(K_HOURS))) < prob),
                "mask": torch.ones(n, len(K_HOURS), dtype=torch.bool), "k_hours": K_HOURS}, run / "predictions_val.pt")
    return run


@pytest.fixture
def toy_project(tmp_path, monkeypatch):
    """Temporary project root with processed data and one trained toy TGN run; returns (root, run_dir, stream)."""
    import cryptograph.data.config as config
    import cryptograph.explain.run as explain_run
    import cryptograph.serving.compress as compress
    import cryptograph.serving.events as events
    import cryptograph.serving.live as live
    import cryptograph.serving.predict as predict
    import cryptograph.serving.replay as replay

    for mod in (config, explain_run, predict, compress, events, replay, live):
        monkeypatch.setattr(mod, "PROJECT_ROOT", tmp_path)
    d, F = _write_project(tmp_path)
    ext = tmp_path / "data" / "external"
    ext.mkdir(parents=True)
    pd.DataFrame({"date": ["2025-12-01", "2026-01-03", "2026-01-12"], "name": ["before data", "early", "late"],
                  "origin_coin": [None, "AAA", None], "event_type": ["macro"] * 3,
                  "description": ["", "an early crash", "a late crash"], "source_url": ["", "https://example.org", ""]}
                 ).to_csv(ext / "events.csv", index=False)
    pd.DataFrame({"coin": COINS, "sector": ["x", "x", "x", "y", "y"]}).to_csv(ext / "sectors.csv", index=False)
    return tmp_path, _write_run(tmp_path, d, F), d
