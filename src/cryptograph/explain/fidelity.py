"""Faithfulness of event-level explanations for the TGN, and the baseline explainers TempME is compared with.

An explanation scores each of the (up to 2 x L) recent events the TGN attends to for an explained event. At sparsity
s (the number of events kept), with p_full the model's probability, p_keep the probability with only the top-s
events kept (others weight 0) and p_drop the probability with the top-s events removed:

    Fidelity-  = mean |p_full - p_keep|   (sufficiency: lower is better)
    Fidelity+  = mean |p_full - p_drop|   (necessity:   higher is better)

Probabilities are averaged over the K horizons. Because removing *all* recent events typically moves a probability by
only ~0.02, both are also reported normalised by mean |p_full - p_empty| (all events removed), giving values on a
0-1-ish scale; the characterization score (Amara et al., 2022) is the harmonic mean of normalised Fidelity+ and
1 - normalised Fidelity-.
"""

from __future__ import annotations

import numpy as np
import torch

from .tgn_capture import Captured, ExplainableTGN


def _topk_mask(imp: torch.Tensor, valid: torch.Tensor, k: int) -> torch.Tensor:
    """(N, 2, L) importance -> (N, 2, L) bool with the k most important valid events of each instance."""
    N = imp.shape[0]
    flat = imp.reshape(N, -1).masked_fill(~valid.reshape(N, -1), float("-inf"))
    kk = min(k, flat.shape[1])
    idx = torch.topk(flat, kk, dim=1).indices
    m = torch.zeros_like(flat, dtype=torch.bool).scatter_(1, idx, True) & valid.reshape(N, -1)
    return m.reshape(imp.shape)


@torch.no_grad()
def fidelity_curve(ex: ExplainableTGN, cap: Captured, imp: torch.Tensor, ks=(1, 2, 3, 5, 10),
                   batch: int = 4096) -> list[dict]:
    rows = {k: {"keep": [], "drop": []} for k in ks}
    base, empty = [], []
    for a in range(0, len(cap), batch):
        c = cap.subset(slice(a, a + batch))
        valid = c.mask
        p_full = torch.sigmoid(ex(c))
        p_empty = torch.sigmoid(ex(c, torch.zeros_like(c.rel_t)))
        base.append(p_full)
        empty.append((p_full - p_empty).abs().mean(1))
        for k in ks:
            top = _topk_mask(imp[a:a + batch], valid, k)
            p_keep = torch.sigmoid(ex(c, top.float()))
            p_drop = torch.sigmoid(ex(c, (valid & ~top).float()))
            rows[k]["keep"].append((p_full - p_keep).abs().mean(1))
            rows[k]["drop"].append((p_full - p_drop).abs().mean(1))
    scale = float(torch.cat(empty).mean())
    out = []
    for k in ks:
        fm, fp = float(torch.cat(rows[k]["keep"]).mean()), float(torch.cat(rows[k]["drop"]).mean())
        fpn, fmn = fp / scale, fm / scale
        char = 2 * fpn * (1 - fmn) / (fpn + (1 - fmn)) if fpn > 0 and fmn < 1 else 0.0
        out.append({"kept_events": k, "fid_minus": fm, "fid_plus": fp, "fid_minus_norm": fmn, "fid_plus_norm": fpn,
                    "characterization": char, "empty_effect": scale})
    return out


# ---- baseline explainers: importance per (N, 2, L) event -------------------------------------------------------

def random_importance(cap: Captured, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.rand(cap.rel_t.shape, generator=g)


def recency_importance(cap: Captured, t_all: torch.Tensor) -> torch.Tensor:
    """Most recent events first (a strong, model-free temporal baseline)."""
    t = t_all[cap.pos_n.clamp_min(0)].float()
    return torch.where(cap.mask, t, torch.full_like(t, -1e9))


def attention_importance(ex: ExplainableTGN, cap: Captured, batch: int = 8192) -> torch.Tensor:
    """The TGN's own attention weights (mean over heads)."""
    return torch.cat([ex.attention(cap.subset(slice(a, a + batch))) for a in range(0, len(cap), batch)])


def mask_importance(ex: ExplainableTGN, cap: Captured, steps: int = 100, lr: float = 0.1, l1: float = 0.005,
                    batch: int = 4096, seed: int = 0) -> torch.Tensor:
    """Per-event soft mask optimised separately for every instance (GNNExplainer-style; explain/edge_mask.py idea):
    keep the prediction (BCE to the original probability) with as few events as possible (L1 + entropy)."""
    torch.manual_seed(seed)
    out = []
    for a in range(0, len(cap), batch):
        c = cap.subset(slice(a, a + batch))
        with torch.no_grad():
            target = torch.sigmoid(ex(c))
        theta = torch.zeros(c.rel_t.shape, requires_grad=True)
        opt = torch.optim.Adam([theta], lr=lr)
        for _ in range(steps):
            w = torch.sigmoid(theta) * c.mask
            loss = torch.nn.functional.binary_cross_entropy_with_logits(ex(c, w), target)
            ent = -(w * torch.log(w + 1e-8) + (1 - w) * torch.log(1 - w + 1e-8))
            loss = loss + l1 * w.sum((1, 2)).mean() + l1 * ent[c.mask].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        out.append(torch.sigmoid(theta).detach() * c.mask)
    return torch.cat(out)


def coin_credit(imp: torch.Tensor, cap: Captured, n_coins: int, top: int = 3) -> np.ndarray:
    """How often each coin is the counterpart of one of the `top` most important events (summed over instances,
    weighted by importance). The plausibility check compares this with network rankings."""
    m = _topk_mask(imp, cap.mask, top)
    w = (imp.clamp_min(0) * m).reshape(-1).numpy()
    nb = cap.nbr.reshape(-1).numpy()
    ok = nb >= 0
    return np.bincount(nb[ok], weights=w[ok], minlength=n_coins)
