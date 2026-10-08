import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from cryptograph.models.tgn import ContagionTGN
from cryptograph.serving.compress import (build_variant, day_blocks, model_size_bytes, nonzero_weight_share, prune_l1,
                                          quantize_int8, top_overlap, torch_linears)

from .conftest import K_HOURS


def _model(seed=0):
    torch.manual_seed(seed)
    return ContagionTGN(num_nodes=6, msg_dim=5, node_feat_dim=4, memory_dim=16, time_dim=8, embedding_dim=16,
                        out_horizons=3, dropout=0.0).eval()


def _head_inputs(seed=1, B=64):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(B, 16, generator=g), torch.randn(B, 16, generator=g), torch.randn(B, 4, generator=g),
            torch.randn(B, 4, generator=g), torch.randn(B, 5, generator=g)]


def test_torch_linears_swaps_pyg_layers_without_changing_outputs():
    from torch_geometric.nn.dense.linear import Linear as PygLinear

    m = _model()
    x = torch.randn(7, 16)
    ref = m.gnn.conv.lin_query(x)
    torch_linears(m)
    assert not any(isinstance(mod, PygLinear) for mod in m.modules())
    torch.testing.assert_close(m.gnn.conv.lin_query(x), ref)


def test_int8_quantises_every_linear_and_the_memory_gru():
    from torch.ao.nn.quantized.dynamic import GRUCell as QGRUCell, Linear as QLinear   # print as DynamicQuantized*

    q = quantize_int8(_model())
    assert isinstance(q.head[0], QLinear) and isinstance(q.gnn.conv.lin_query, QLinear)
    assert isinstance(q.memory.gru, QGRUCell)
    fp32 = [n for n, mod in q.named_modules() if type(mod) is nn.Linear]
    assert fp32 == ["memory.time_enc.lin"]                     # the only fp32 layer (TGNMemory reads its weight)
    assert q.memory.device == torch.device("cpu")
    keep = quantize_int8(_model(), keep_time=True)
    assert type(keep.gnn.time_enc.lin) is nn.Linear and type(keep.memory.time_enc.lin) is nn.Linear
    assert isinstance(keep.head[0], QLinear)


def test_int8_head_stays_close_to_fp32_and_is_smaller():
    m = _model()
    q = build_variant(m, "int8")
    x = _head_inputs()
    with torch.no_grad():
        ref, out = torch.sigmoid(m.decode(*x)), torch.sigmoid(q.decode(*x))
    assert (out - ref).abs().max() < 0.05
    assert model_size_bytes(q) < model_size_bytes(m)


def test_prune_zeroes_the_requested_share_of_each_linear_weight():
    p = prune_l1(_model(), 0.5)
    for mod in p.modules():
        if isinstance(mod, nn.Linear) and mod.weight.numel() >= 16:
            assert abs(float((mod.weight == 0).float().mean()) - 0.5) < 0.05
    assert abs(nonzero_weight_share(p) - 0.5) < 0.05 and nonzero_weight_share(_model()) == 1.0
    with pytest.raises(ValueError):
        build_variant(_model(), "fp8")


def test_day_blocks_and_top_overlap():
    times = [str(x) for x in pd.date_range("2026-01-01 22:00", periods=5, freq="h")]
    assert day_blocks(times, np.array([0, 1, 2, 4, 1])).tolist() == [0, 0, 1, 1, 0]
    a = np.arange(200, dtype=float)
    assert top_overlap(a, a, 0.05) == 1.0 and top_overlap(a, -a, 0.05) == 0.0


def test_compress_task_end_to_end(toy_project):
    from cryptograph.training.runner import run_experiment

    root, run, d = toy_project
    cfg = {"name": "compress_toy", "model": {"name": "compress"}, "device": "cpu",
           "compress": {"target_experiment": "toy_tgn", "variants": ["fp32", "int8", "prune50"], "n_boot": 50,
                        "latency_batches": 3, "threads": [1]}}
    out = run_experiment(cfg, seed=0, runs_dir=root / "runs")
    rd = root / "runs" / out["run_id"]
    acc = pd.read_csv(rd / "compression_accuracy.csv")
    lat = pd.read_csv(rd / "compression_latency.csv")
    assert set(acc.variant) == {"fp32", "int8", "prune50"} and sorted(acc.k_hours.unique()) == K_HOURS
    ref = acc[acc.variant == "fp32"]
    assert (ref.max_abs_prob_diff == 0).all() and (ref.top1pct_overlap == 1).all()
    q = acc[acc.variant == "int8"].set_index("k_hours")
    assert np.allclose(q.auprc_cost, ref.set_index("k_hours").auprc - q.auprc)
    assert ((q.cost_ci_low <= q.auprc_cost) & (q.auprc_cost <= q.cost_ci_high)).all()
    assert q.p_fp32_better.between(0, 1).all() and q.events.eq(ref.set_index("k_hours").events).all()
    assert (lat.ms_per_batch > 0).all() and lat.loc[lat.variant == "fp32", "speedup_vs_fp32"].eq(1).all()
    assert (root / "reports" / "compression" / "accuracy.csv").exists()
    assert "int8_auprc_cost_k1h" in out["metrics"]
