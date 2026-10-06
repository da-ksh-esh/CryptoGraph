"""Feature-level explanations of the TGN's contagion predictions: integrated gradients on the prediction head.

The head sees [z_i, z_j, x_i(t), x_j(t), msg]: the embeddings of source i and receiver j (memory + attention over their
recent events), the node features of both coins at t and the event message. Integrated gradients (Sundararajan et
al., 2017) attribute logit(input) - logit(baseline) to the individual inputs; the baseline is all zeros, which for
the standardised node features and messages is the training mean. Attributions satisfy completeness (they sum to
the logit difference), checked by `completeness_gap`.
"""

from __future__ import annotations

import numpy as np
import torch

from .tgn_capture import Captured, ExplainableTGN


def head_inputs(ex: ExplainableTGN, cap: Captured) -> list[torch.Tensor]:
    with torch.no_grad():
        zi, _ = ex.embed(cap.x_c[:, 0], cap.x_n[:, 0], cap.rel_t[:, 0], cap.msg_n[:, 0], cap.mask[:, 0])
        zj, _ = ex.embed(cap.x_c[:, 1], cap.x_n[:, 1], cap.rel_t[:, 1], cap.msg_n[:, 1], cap.mask[:, 1])
    if not ex.model.use_memory:
        zi, zj = torch.zeros_like(zi), torch.zeros_like(zj)
    return [zi, zj, cap.x_src, cap.x_dst, cap.msg]


def integrated_gradients(ex: ExplainableTGN, cap: Captured, horizon: int, steps: int = 32,
                         batch: int = 4096) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Attributions per input block (same shapes as `head_inputs`) for the logit of `horizon`, and the logit gap."""
    blocks = head_inputs(ex, cap)
    out = [torch.zeros_like(b) for b in blocks]
    gap = torch.zeros(len(cap))
    alphas = (torch.arange(steps, dtype=torch.float32) + 0.5) / steps                  # midpoint rule
    for a in range(0, len(cap), batch):
        xs = [b[a:a + batch] for b in blocks]
        grads = [torch.zeros_like(x) for x in xs]
        for al in alphas:
            inp = [(al * x).requires_grad_(True) for x in xs]
            logit = ex.model.decode(*inp)[:, horizon]
            g = torch.autograd.grad(logit.sum(), inp)
            grads = [gs + gi for gs, gi in zip(grads, g)]
        for o, x, gs in zip(out, xs, grads):
            o[a:a + batch] = x * gs / steps
        with torch.no_grad():
            gap[a:a + batch] = ex.model.decode(*xs)[:, horizon] - ex.model.decode(*[torch.zeros_like(x) for x in xs])[:, horizon]
    return out, gap


def completeness_gap(attr: list[torch.Tensor], gap: torch.Tensor) -> float:
    total = sum(a.sum(1) for a in attr)
    return float((total - gap).abs().max())


def feature_table(attr: list[torch.Tensor], node_names: list[str], msg_names: list[str]) -> "pd.DataFrame":
    """One row per input: mean signed and mean absolute attribution; embeddings summed into one row per side."""
    import pandas as pd

    zi, zj, xi, xj, msg = attr
    cols = {"source coin: memory + recent events": zi.sum(1), "receiver coin: memory + recent events": zj.sum(1)}
    for k, n in enumerate(node_names):
        cols[f"source coin: {n}"] = xi[:, k]
    for k, n in enumerate(node_names):
        cols[f"receiver coin: {n}"] = xj[:, k]
    for k, n in enumerate(msg_names):
        cols[f"event: {n}"] = msg[:, k]
    rows = []
    for name, v in cols.items():
        v = v.numpy()
        rows.append({"input": name, "group": name.split(":")[0], "mean_attribution": float(v.mean()),
                     "mean_abs_attribution": float(np.abs(v).mean())})
    df = pd.DataFrame(rows)
    df["share_of_abs"] = df["mean_abs_attribution"] / df["mean_abs_attribution"].sum()
    return df.sort_values("mean_abs_attribution", ascending=False).reset_index(drop=True)
