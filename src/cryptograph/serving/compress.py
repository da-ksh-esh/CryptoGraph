"""Compressing the final TGN for CPU serving, and what it costs (notebook-free; tables in reports/compression/).

Variants of the trained model (all on CPU):
    fp32            the trained model (reference)
    int8            dynamic int8 quantisation of every linear layer and the memory GRU. PyG's own Linear (inside the
                    attention layer) is first swapped for torch.nn.Linear so that it is quantised too. The memory's
                    1 -> time_dim time encoder stays fp32 (PyG's TGNMemory needs its float weight).
    int8_keep_time  as int8, but the attention's time encoder also stays fp32 (time gaps of up to thousands of hours)
    prune50         50% magnitude pruning of every weight matrix (linear layers, memory GRU). Dense CPU kernels: neither
                    faster nor smaller (sparse storage needs > ~70% zeros); included as the pruning comparison.

Each variant replays the whole stream exactly as evaluation does (train, val, test; memory updated after each batch),
so every variant is scored on the same test events. Accuracy cost per horizon k = AUPRC(fp32) - AUPRC(variant), with a
95% interval and a one-sided p (is fp32 better?) from the paired day-block bootstrap (analysis/evaluation.py).
Agreement = overlap of the riskiest 1% of test events. Speed = median CPU latency of one 200-event batch at the
configured thread counts, and the serialised model size.
"""

from __future__ import annotations

import copy
import io
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from ..data.config import PROJECT_ROOT
from ..data.storage import load_pt

VARIANTS = ("fp32", "int8", "int8_keep_time", "prune50")


def _copy(model: nn.Module) -> nn.Module:
    """Deep copy of a ContagionTGN in eval mode. TGN memory can hold a non-leaf tensor (deepcopy refuses those), so
    memory is reset and detached first; every replay resets it anyway."""
    model.memory.reset_state()
    model.memory.detach()
    return copy.deepcopy(model).eval()


def torch_linears(model: nn.Module) -> nn.Module:
    """Replace every torch_geometric Linear with an equal torch.nn.Linear (in place), so quantisation sees it."""
    from torch_geometric.nn.dense.linear import Linear as PygLinear

    for name, child in list(model.named_children()):
        if isinstance(child, PygLinear):
            lin = nn.Linear(child.in_channels, child.out_channels, bias=child.bias is not None)
            with torch.no_grad():
                lin.weight.copy_(child.weight)
                if child.bias is not None:
                    lin.bias.copy_(child.bias)
            setattr(model, name, lin.train(child.training))
        else:
            torch_linears(child)
    return model


def quantize_int8(model: nn.Module, keep_time: bool = False) -> nn.Module:
    """Dynamic int8 copy of a ContagionTGN (weights int8, activations quantised on the fly).

    The memory's own time encoder (1 -> time_dim) always stays fp32: PyG's TGNMemory reads its device from that
    layer's weight, which a quantised layer no longer exposes. keep_time=True also keeps the attention layer's time
    encoder in fp32."""
    from torch.ao.quantization import quantize_dynamic

    ref = torch_linears(_copy(model))
    keep = {"memory.time_enc"} | ({"gnn.time_enc"} if keep_time else set())
    m = quantize_dynamic(copy.deepcopy(ref), {nn.Linear, nn.GRUCell}, dtype=torch.qint8)
    for name in keep:                                         # put the fp32 layers back
        parent, attr = name.rsplit(".", 1)
        setattr(m.get_submodule(parent), attr, copy.deepcopy(ref.get_submodule(name)))
    return m


def prune_l1(model: nn.Module, amount: float = 0.5) -> nn.Module:
    """Copy with the smallest `amount` of every weight matrix (linear layers and the memory GRU) set to zero, per
    matrix, unstructured."""
    from torch.nn.utils import prune

    m = torch_linears(_copy(model))
    for mod in m.modules():
        names = ("weight",) if isinstance(mod, nn.Linear) else ("weight_ih", "weight_hh") \
            if isinstance(mod, nn.GRUCell) else ()
        for name in names:
            prune.l1_unstructured(mod, name, amount)
            prune.remove(mod, name)
    return m


def build_variant(model: nn.Module, name: str) -> nn.Module:
    if name == "fp32":
        return _copy(model)
    if name == "int8":
        return quantize_int8(model)
    if name == "int8_keep_time":
        return quantize_int8(model, keep_time=True)
    if name.startswith("prune"):
        return prune_l1(model, int(name[5:]) / 100)
    raise ValueError(f"unknown compression variant {name!r} (known: {VARIANTS})")


def model_size_bytes(model: nn.Module) -> int:
    """Size of the serialised state dict (dense: sparse formats only pay off above ~70% zeros)."""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes


def nonzero_weight_share(model: nn.Module) -> float:
    """Share of non-zero entries over the float weight matrices (1.0 unless pruned; int8 layers are not counted)."""
    ws = [p for n, p in model.state_dict().items() if torch.is_tensor(p) and p.is_floating_point() and p.dim() == 2]
    return float(sum(int((w != 0).sum()) for w in ws) / max(sum(w.numel() for w in ws), 1))


def day_blocks(times, t: np.ndarray) -> np.ndarray:
    """Calendar day (UTC) of each event's bar, as integer codes (the bootstrap block)."""
    days = pd.DatetimeIndex(times)[np.asarray(t)].normalize()
    return pd.factorize(days)[0]


def top_overlap(a: np.ndarray, b: np.ndarray, frac: float = 0.01) -> float:
    """Share of the top `frac` events by score a that are also in the top `frac` by score b."""
    n = max(1, int(round(frac * len(a))))
    ta = set(np.argsort(-a, kind="stable")[:n])
    tb = set(np.argsort(-b, kind="stable")[:n])
    return len(ta & tb) / n


def replay_test(runner) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluation pass over the whole stream; returns test (positions, prob, y, mask)."""
    bounds = runner.bounds
    runner.reset()
    runner.run(*bounds["train"], "train")
    runner.run(*bounds["val"], "val")
    _, lg, y, m = runner.run(*bounds["test"], "test")
    return runner.last_positions.numpy(), torch.sigmoid(lg).numpy(), y.numpy(), m.numpy()


def batch_latency_ms(runner, n_batches: int, threads: int) -> float:
    """Median wall time of one forward pass (embedding + head) over test batches, memory frozen."""
    from ..training.tgn_task import _batches

    old = torch.get_num_threads()
    if threads > 0:
        torch.set_num_threads(threads)
    try:
        a, b = runner.bounds["test"]
        out = []
        with torch.no_grad():
            for s, e in _batches(runner.t_np, a, b, runner.bs)[: n_batches + 3]:
                t0 = time.perf_counter()
                runner._step(s, e)
                out.append((time.perf_counter() - t0) * 1e3)
        return float(np.median(out[3:] if len(out) > 3 else out))      # the first batches warm up the kernels
    finally:
        torch.set_num_threads(old)


def compress_task(cfg: dict, seed: int, device: torch.device, tracker) -> dict[str, Any]:
    from ..analysis.evaluation import block_bootstrap_auprc_diff, weighted_auprc
    from ..explain.run import _find_run
    from ..explain.tgn_capture import load_trained
    from ..training.metrics import auroc
    from ..training.tgn_task import StreamRunner

    cc = cfg["compress"]
    target = _find_run(cc["target_experiment"], cc.get("target_seed", 0))
    tracker.log(f"compressing {target.name} on CPU")
    runner, D = load_trained(target, device="cpu")
    d, K = D["d"], list(D["k_hours"])
    tr = D["cfg"]["training"]
    variants = cc.get("variants", list(VARIANTS))
    assert variants[0] == "fp32", "the first variant must be the fp32 reference"
    times = D["panel"]["times"]

    preds, rows, lat_rows = {}, [], []
    for name in variants:
        model = build_variant(runner.model, name)
        r = StreamRunner(model, d, D["feats"], D["wmask"], torch.device("cpu"), tr["batch_size"],
                         tr.get("num_neighbors", 10))
        r.bounds = D["bounds"]
        t0 = time.perf_counter()
        preds[name] = replay_test(r)
        replay_s = time.perf_counter() - t0
        size, nonzero = model_size_bytes(model), nonzero_weight_share(model)
        lat = {th: batch_latency_ms(r, cc.get("latency_batches", 200), th) for th in cc.get("threads", [1, 0])}
        for th, ms in lat.items():
            lat_rows.append({"variant": name, "threads": th if th > 0 else torch.get_num_threads(),
                             "batch_events": tr["batch_size"], "ms_per_batch": ms,
                             "us_per_event": 1e3 * ms / tr["batch_size"], "full_replay_s": replay_s,
                             "size_bytes": size, "nonzero_float_weights": nonzero})
        tracker.log(f"{name}: full replay {replay_s:.0f}s, size {size / 1e3:.0f} kB, "
                    + ", ".join(f"{ms:.2f} ms/batch at {th or 'default'} threads" for th, ms in lat.items()))

    pos0, p0, y, m = preds["fp32"]
    block = day_blocks(times, d.t.numpy()[pos0])
    saved = target / "predictions_test.pt"
    metrics: dict[str, Any] = {"target_run": target.name, "n_test_events": int(len(pos0))}
    if saved.exists():                                        # the run's own scores (possibly from a GPU)
        ps = load_pt(saved)
        same = np.array_equal(ps["position"].numpy(), pos0)
        metrics["fp32_cpu_matches_saved_positions"] = bool(same)
        if same:
            metrics["fp32_cpu_vs_saved_max_abs_prob_diff"] = float(np.abs(ps["prob"].float().numpy() - p0).max())
    for name in variants:
        pos, p, _, _ = preds[name]
        assert np.array_equal(pos, pos0), "every variant must be scored on the same events"
        for k, kh in enumerate(K):
            mk = m[:, k].astype(bool)
            row = {"variant": name, "k_hours": kh, "events": int(mk.sum()), "positive_rate": float(y[mk, k].mean()),
                   "auprc": weighted_auprc(y[mk, k], p[mk, k]), "auroc": auroc(y[mk, k], p[mk, k]),
                   "max_abs_prob_diff": float(np.abs(p[mk, k] - p0[mk, k]).max()),
                   "top1pct_overlap": top_overlap(p0[mk, k], p[mk, k])}
            if name != "fp32":
                bs = block_bootstrap_auprc_diff(y[mk, k], p0[mk, k], p[mk, k], block[mk],
                                                n_boot=cc.get("n_boot", 1000), seed=seed)
                row |= {"auprc_cost": bs["diff"], "cost_ci_low": bs["ci_low"], "cost_ci_high": bs["ci_high"],
                        "p_fp32_better": bs["p_one_sided"], "n_days": bs["n_blocks"]}
                metrics[f"{name}_auprc_cost_k{kh}h"] = bs["diff"]
                metrics[f"{name}_p_k{kh}h"] = bs["p_one_sided"]
            rows.append(row)
    acc, lat = pd.DataFrame(rows), pd.DataFrame(lat_rows)
    ref_ms = lat[lat.variant == "fp32"].set_index("threads")["ms_per_batch"]
    lat["speedup_vs_fp32"] = [ref_ms[t] / ms for t, ms in zip(lat["threads"], lat["ms_per_batch"])]
    acc.to_csv(tracker.dir / "compression_accuracy.csv", index=False)
    lat.to_csv(tracker.dir / "compression_latency.csv", index=False)
    if cc.get("write_reports", True):
        out = PROJECT_ROOT / "reports" / "compression"
        out.mkdir(parents=True, exist_ok=True)
        acc.to_csv(out / "accuracy.csv", index=False)
        lat.to_csv(out / "latency.csv", index=False)
    for name in variants[1:]:
        sub = lat[(lat.variant == name)]
        metrics[f"{name}_speedup_1thread"] = float(sub.loc[sub.threads == 1, "speedup_vs_fp32"].iloc[0]) \
            if (sub.threads == 1).any() else float("nan")
    return metrics
