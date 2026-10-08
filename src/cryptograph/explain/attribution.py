"""Feature-level explanations of the TGN's contagion predictions: integrated gradients on the prediction head.

The head sees [z_i, z_j, x_i(t), x_j(t), msg]: the embeddings of source i and receiver j (memory + attention over their
recent events), the node features of both coins at t and the event message. Integrated gradients (Sundararajan et
al., 2017) attribute logit(input) - logit(baseline) to the individual inputs; the baseline is all zeros, which for
the standardised node features and messages is the training mean. Attributions satisfy completeness (they sum to
the logit difference), checked by `completeness_gap`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from .tgn_capture import Captured, ExplainableTGN

if TYPE_CHECKING:
    import pandas as pd


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


def flat_attributions(attr: list[torch.Tensor]) -> torch.Tensor:
    """(N, n_inputs) in the rows of `feature_table`: each embedding summed to one input, then node features of i, of j,
    and the message."""
    zi, zj, xi, xj, msg = attr
    return torch.cat([zi.sum(1, keepdim=True), zj.sum(1, keepdim=True), xi, xj, msg], dim=1)


def driver_stability(ex: ExplainableTGN, cap: Captured, horizon: int, noise: float, top: int = 5, steps: int = 32,
                     seed: int = 0) -> dict[str, float]:
    """How stable the per-alert drivers are when the observed inputs (node features of both coins and the message,
    standardised) get Gaussian noise of std `noise`: mean Jaccard overlap of the `top` inputs by |attribution|, the
    share of alerts whose top driver is unchanged, and the mean change in probability."""
    g = torch.Generator().manual_seed(seed)
    noisy = cap.subset(torch.arange(len(cap)))
    for k in ("x_src", "x_dst", "msg"):
        x = getattr(cap, k)
        setattr(noisy, k, x + noise * torch.randn(x.shape, generator=g))
    a0 = flat_attributions(integrated_gradients(ex, cap, horizon, steps)[0]).abs()
    a1 = flat_attributions(integrated_gradients(ex, noisy, horizon, steps)[0]).abs()
    t0, t1 = a0.topk(top, dim=1).indices, a1.topk(top, dim=1).indices
    sets = [(set(r0.tolist()), set(r1.tolist())) for r0, r1 in zip(t0, t1)]
    jac = [len(a & b) / len(a | b) for a, b in sets]
    with torch.no_grad():
        dp = (torch.sigmoid(ex(noisy)[:, horizon]) - torch.sigmoid(ex(cap)[:, horizon])).abs()
    return {"noise": noise, "events": len(cap), "top_k": top, "mean_jaccard_topk": float(np.mean(jac)),
            "top1_unchanged": float((t0[:, 0] == t1[:, 0]).float().mean()), "mean_abs_dp": float(dp.mean())}


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
