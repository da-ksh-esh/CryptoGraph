"""Stress tests of the contagion models on the test period: noisy inputs, missing events, gradient attacks.

The TGN replays the clean history (train + validation) and is then scored on the test period with one perturbation:
  * input noise   Gaussian noise of std sigma (inputs are standardised, so sigma is in units of a feature's std)
                  added to the explained event's inputs: node features of both coins and the event message
  * missing data  a random share of test events is never written to memory / the neighbour lists (data feed gaps);
                  every event is still scored
  * FGSM attack   white-box fast-gradient-sign perturbation of the same inputs, eps * sign(grad of the loss against the
                  true labels): the worst case for a small, bounded manipulation of the observed data
Gradient boosting gets the same noise, and the TGN's FGSM perturbations are transferred to it (a black-box transfer
attack); it has no event memory, so missing events cannot affect it.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from ..training.tgn_task import _batches


def tgn_test_pass(runner, bounds: dict, *, noise: float = 0.0, drop: float = 0.0, fgsm: float = 0.0,
                  seed: int = 0, keep_inputs: bool = False) -> dict:
    """Score the test period once under one perturbation. Returns probabilities, labels, masks, positions and (with
    keep_inputs) the perturbed head inputs of the scored events."""
    model, d, dev = runner.model, runner.d, runner.device
    g = torch.Generator().manual_seed(seed)
    model.eval()
    runner.reset()
    runner.run(*bounds["train"], "train")
    runner.run(*bounds["val"], "val")
    a, b = bounds["test"]
    out = {k: [] for k in ("prob", "y", "mask", "pos", "x_src", "x_dst", "msg")}
    for s, e in _batches(runner.t_np, a, b, runner.bs):
        w = torch.from_numpy(runner.wmask["test"][runner.t_np[s:e]])
        y, ym = d.y[s:e], d.y_mask[s:e] & w[:, None]
        src, dst = d.src[s:e].to(dev), d.dst[s:e].to(dev)
        t, msg = d.t[s:e].to(dev), d.msg[s:e].to(dev).float()
        with torch.no_grad():
            n_id = torch.cat([src, dst]).unique()
            n_id, edge_index, e_id = runner.loader(n_id)
            runner.assoc[n_id] = torch.arange(n_id.size(0), device=dev)
            pos = runner.pos_of_eid[e_id.cpu()]
            z = model.embed(n_id, edge_index, d.t[pos].to(dev), d.msg[pos].to(dev))
            zs, zd = z[runner.assoc[src]], z[runner.assoc[dst]]
        tt = d.t[s:e]
        xs, xd, mg = runner.feats[tt, d.src[s:e]].to(dev), runner.feats[tt, d.dst[s:e]].to(dev), msg.clone()
        if noise > 0:
            xs = xs + noise * torch.randn(xs.shape, generator=g)
            xd = xd + noise * torch.randn(xd.shape, generator=g)
            mg = mg + noise * torch.randn(mg.shape, generator=g)
        if fgsm > 0 and ym.any():
            inp = [x.clone().requires_grad_(True) for x in (xs, xd, mg)]
            lg = model.decode(zs, zd, *inp)
            loss = F.binary_cross_entropy_with_logits(lg[ym], y[ym].float())
            grads = torch.autograd.grad(loss, inp)
            xs, xd, mg = (x.detach() + fgsm * gr.sign() for x, gr in zip(inp, grads))
        with torch.no_grad():
            logits = model.decode(zs, zd, xs, xd, mg)
        if drop > 0:                                   # data-feed gaps: some events never reach memory / neighbours
            keep = torch.rand(e - s, generator=g) >= drop
        else:
            keep = torch.ones(e - s, dtype=torch.bool)
        if keep.any():
            ks = keep.nonzero().flatten()
            model.memory.update_state(src[ks], dst[ks], t[ks], d.msg[s:e][ks].to(dev))
            runner.loader.insert(src[ks], dst[ks])
            runner.pos_of_eid[runner.n_inserted: runner.n_inserted + len(ks)] = ks + s
            runner.n_inserted += len(ks)
        model.memory.detach()
        if ym.any():
            out["prob"].append(torch.sigmoid(logits).cpu())
            out["y"].append(y)
            out["mask"].append(ym)
            out["pos"].append(torch.arange(s, e))
            if keep_inputs:
                out["x_src"].append(xs.cpu())
                out["x_dst"].append(xd.cpu())
                out["msg"].append(mg.cpu())
    return {k: torch.cat(v).numpy() for k, v in out.items() if v}


def auprc_by_k(prob: np.ndarray, y: np.ndarray, m: np.ndarray) -> list[float]:
    from ..training.metrics import auprc

    return [auprc(y[:, i], prob[:, i], m[:, i]) for i in range(y.shape[1])]
