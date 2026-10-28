"""Capture what the trained TGN sees for chosen events, so explanations can re-run its prediction exactly.

For an event p (coin i shocks at bar t, receiver j) the ContagionTGN computes
    z_c = TransformerConv(memory of c, memory of c's last L neighbours, edge attrs)   for c in (i, j)
    logits = head([z_i, z_j, x_i(t), x_j(t), msg_p])
where each neighbour edge is one past event of c (its time and message). The memory states are fixed at prediction
time, so everything the attention layer needs fits in a few tensors per event. `capture` replays the stream exactly as
evaluation does (reset, train, val, test; memory updated only after each batch is predicted) and stores them for the
selected positions; `ExplainableTGN` recomputes the logits from those tensors with an optional weight per neighbour
event that multiplies its attention (TempME's way of applying an explanation to a TGN; weights of 1 reproduce the
model exactly, 0 removes the event). Memory itself is not re-run: like TempME, explanations act on the attention over
recent events, not on the memory built from the whole past.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from ..data.storage import load_pt
from ..training.tgn_task import _batches


@dataclass
class Captured:
    pos: torch.Tensor            # (N,) stream position of the explained event
    x_c: torch.Tensor            # (N, 2, D) memory of the two centres (0 = source i, 1 = receiver j)
    x_n: torch.Tensor            # (N, 2, L, D) memory of each neighbour
    rel_t: torch.Tensor          # (N, 2, L) last update of the neighbour minus the edge time (as in the model)
    msg_n: torch.Tensor          # (N, 2, L, M) message of each neighbour event
    pos_n: torch.Tensor          # (N, 2, L) stream position of each neighbour event (-1 = padding)
    nbr: torch.Tensor            # (N, 2, L) neighbour coin (graph index, -1 = padding)
    x_src: torch.Tensor          # (N, F) node features of i at t
    x_dst: torch.Tensor          # (N, F)
    msg: torch.Tensor            # (N, M) message of the explained event
    logits: torch.Tensor         # (N, K) the model's own logits from the stream run
    ctx: torch.Tensor | None = None   # (N, C) event context: node ctx of i, of j, market (context-feature models)

    @property
    def mask(self) -> torch.Tensor:
        return self.pos_n >= 0

    def __len__(self) -> int:
        return len(self.pos)

    def subset(self, idx) -> "Captured":
        return Captured(**{k: None if getattr(self, k) is None else getattr(self, k)[idx]
                           for k in self.__dataclass_fields__})

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def _steps(runner, end: int) -> list[tuple[int, int]]:
    """The [s, e) slices the runner predicts at once before updating memory: its batches, or every hour inside them
    when the model was trained with `memory_step: hour` (as StreamRunner.run does)."""
    out = []
    for s, e in _batches(runner.t_np, 0, end, runner.bs):
        if getattr(runner, "memory_step", "batch") == "hour":
            cuts = np.flatnonzero(np.diff(runner.t_np[s:e])) + 1 + s
            out += list(zip(np.r_[s, cuts].tolist(), np.r_[cuts, e].tolist()))
        else:
            out.append((s, e))
    return out


@torch.no_grad()
def capture(runner, selected: np.ndarray, end: int, L: int = 10) -> Captured:
    """Replay events [0, end) through `runner` (a StreamRunner with the trained model, eval mode) and capture the
    attention inputs of every position p with selected[p]. Returns them in stream order."""
    model, d, dev = runner.model, runner.d, runner.device
    model.eval()
    runner.reset()
    rows = {k: [] for k in Captured.__dataclass_fields__}
    for s, e in _steps(runner, end):
        sel = np.flatnonzero(selected[s:e]) + s
        if len(sel):
            src, dst = d.src[s:e].to(dev), d.dst[s:e].to(dev)
            n_id = torch.cat([src, dst]).unique()
            n_id, edge_index, e_id = runner.loader(n_id)
            runner.assoc[n_id] = torch.arange(n_id.size(0), device=dev)
            pos_e = runner.pos_of_eid[e_id.cpu()]
            mem, last_update = model.memory(n_id)
            logits, _ = runner._step(s, e)                 # the model's own prediction for the batch
            nb_local, ctr_local = edge_index[0].cpu(), edge_index[1].cpu()
            for p in sel:
                xc, xn, rt, mn, pn, nb = [], [], [], [], [], []
                for c in (int(d.src[p]), int(d.dst[p])):
                    cl = int(runner.assoc[c])
                    m = (ctr_local == cl).nonzero().flatten()
                    m = m[torch.argsort(d.t[pos_e[m]], stable=True)][-L:]      # oldest first, at most L
                    k = len(m)
                    pad = lambda x, fill: torch.cat([x, torch.full((L - k, *x.shape[1:]), fill, dtype=x.dtype)])
                    nbl = nb_local[m]
                    xc.append(mem[cl].cpu())
                    xn.append(pad(mem[nbl].cpu(), 0.0))
                    rt.append(pad((last_update[nbl].cpu() - d.t[pos_e[m]]).float(), 0.0))
                    mn.append(pad(d.msg[pos_e[m]].float(), 0.0))
                    pn.append(pad(pos_e[m], -1))
                    nb.append(pad(n_id[nbl].cpu(), -1))
                tt = int(d.t[p])
                rows["pos"].append(torch.tensor(p))
                rows["x_c"].append(torch.stack(xc))
                rows["x_n"].append(torch.stack(xn))
                rows["rel_t"].append(torch.stack(rt))
                rows["msg_n"].append(torch.stack(mn))
                rows["pos_n"].append(torch.stack(pn))
                rows["nbr"].append(torch.stack(nb))
                rows["x_src"].append(runner.feats[tt, d.src[p]])
                rows["x_dst"].append(runner.feats[tt, d.dst[p]])
                rows["msg"].append(d.msg[p].float())
                rows["logits"].append(logits[p - s].cpu())
                if runner.ctx is not None:                 # as StreamRunner._step builds it
                    node, glob = runner.ctx
                    rows["ctx"].append(torch.cat([node[tt, d.src[p]], node[tt, d.dst[p]], glob[tt]]))
            runner._commit((s, e, src, dst, d.t[s:e].to(dev), d.msg[s:e].to(dev)))
        else:
            _, ev = runner._step(s, e)
            runner._commit(ev)
        model.memory.detach()
    return Captured(**{k: torch.stack(v) if v else None for k, v in rows.items()})


class ExplainableTGN(nn.Module):
    """The trained ContagionTGN's attention layer and head, run on captured inputs with per-event weights."""

    def __init__(self, model):
        super().__init__()
        self.model = model.eval()
        self.conv = model.gnn.conv
        self.time_enc = model.gnn.time_enc
        for p in self.model.parameters():
            p.requires_grad_(False)

    def embed(self, x_c, x_n, rel_t, msg_n, mask, w=None):
        """x_c (B, D), x_n (B, L, D), rel_t (B, L), msg_n (B, L, M), mask (B, L), w (B, L) or None -> (B, E)."""
        conv, H, C = self.conv, self.conv.heads, self.conv.out_channels
        B, L, _ = x_n.shape
        edge_attr = torch.cat([self.time_enc(rel_t.reshape(-1)).reshape(B, L, -1), msg_n], dim=-1)
        e = conv.lin_edge(edge_attr).view(B, L, H, C)
        q = conv.lin_query(x_c).view(B, 1, H, C)
        k = conv.lin_key(x_n).view(B, L, H, C) + e
        v = conv.lin_value(x_n).view(B, L, H, C) + e
        score = (q * k).sum(-1) / math.sqrt(C)                                      # (B, L, H)
        score = score.masked_fill(~mask[..., None], float("-inf"))
        alpha = torch.softmax(score, dim=1)
        alpha = torch.nan_to_num(alpha, nan=0.0)                                     # no neighbours at all
        if w is not None:
            alpha = alpha * w[..., None]                                             # TempME: attn * weight
        out = (alpha[..., None] * v).sum(1).reshape(B, H * C)
        return out + conv.lin_skip(x_c), alpha

    def forward(self, cap: Captured, w=None):
        """w: (N, 2, L) weights for the neighbour events of (i, j), or None. Returns logits (N, K)."""
        mask = cap.mask
        wi = None if w is None else w[:, 0]
        wj = None if w is None else w[:, 1]
        zi, _ = self.embed(cap.x_c[:, 0], cap.x_n[:, 0], cap.rel_t[:, 0], cap.msg_n[:, 0], mask[:, 0], wi)
        zj, _ = self.embed(cap.x_c[:, 1], cap.x_n[:, 1], cap.rel_t[:, 1], cap.msg_n[:, 1], mask[:, 1], wj)
        if not self.model.use_memory:
            zi, zj = torch.zeros_like(zi), torch.zeros_like(zj)
        return self.model.decode(zi, zj, cap.x_src, cap.x_dst, cap.msg, cap.ctx)

    @torch.no_grad()
    def attention(self, cap: Captured) -> torch.Tensor:
        """(N, 2, L) attention of each neighbour event, averaged over heads (an explanation baseline)."""
        mask = cap.mask
        out = []
        for c in (0, 1):
            _, a = self.embed(cap.x_c[:, c], cap.x_n[:, c], cap.rel_t[:, c], cap.msg_n[:, c], mask[:, c])
            out.append(a.mean(-1))
        return torch.stack(out, dim=1)


def load_trained(run_dir, device: str = "cpu"):
    """Rebuild a trained `tgn` run (runs/<run_id>/) and its StreamRunner. Returns (runner, data dict)."""
    from pathlib import Path

    import yaml

    from ..data.config import PROJECT_ROOT
    from ..models.tgn import ContagionTGN
    from ..training.tgn_task import (CTX_GLOBAL, CTX_NODE, StreamRunner, calendar_table, context_tables,
                                     node_feature_table, shock_size_messages, with_futures)

    run_dir = Path(run_dir)
    cfg = yaml.safe_load((run_dir / "config.resolved.yaml").read_text(encoding="utf-8"))
    ck = load_pt(run_dir / "checkpoint.pt", map_location=device)
    data_cfg, tr, mcfg = cfg["data"], cfg["training"], ck["model_config"]
    base = PROJECT_ROOT / "data" / "processed" / data_cfg["version"] / data_cfg["resolution"]
    panel = load_pt(base / "panel.pt")
    ev = load_pt(base / "tgn_events.pt")
    net = load_pt(base / "network_features.pt") if data_cfg.get("network_features", True) else None
    d, gidx = ev["full"], ev["graph_index"]
    feats = with_futures(node_feature_table(panel, net, gidx), base, data_cfg, panel["graph_coins"])
    sid = panel["split_id"].numpy()[d.t.numpy()]
    bounds = {s: (int(np.searchsorted(sid, i, "left")), int(np.searchsorted(sid, i, "right")))
              for i, s in enumerate(("train", "val", "test"))}
    wmask = {s: panel["sample_mask"][s].numpy() for s in ("train", "val", "test")}
    node_names = list(panel["feature_names"]) + ([f"network: {n}" for n in net["feature_names"]] if net else [])
    node_names += [f"futures: {k}" for k in range(feats.shape[-1] - len(node_names))]   # futures columns, if any
    # the v3 options, applied exactly as tgn_task does (defaults reproduce v2)
    msg_names = list(ev["msg_names"])
    if mcfg.get("shock_size_messages", False):
        d.msg = shock_size_messages(d, panel, gidx)
        msg_names = msg_names[:4] + ["src_z", "dst_z"]
    if mcfg.get("calendar_features", False):
        cal = calendar_table(panel)
        feats = torch.cat([feats, cal[:, None, :].expand(-1, feats.shape[1], -1)], dim=-1).contiguous()
        node_names += ["calendar: hour_sin", "calendar: hour_cos", "calendar: weekend"]
    ctx = context_tables(panel, gidx) if mcfg.get("context_features", False) else None
    ctx_names = ([f"source coin: {n}" for n in CTX_NODE] + [f"receiver coin: {n}" for n in CTX_NODE]
                 + [f"market: {n}" for n in CTX_GLOBAL]) if ctx is not None else []
    model = ContagionTGN(num_nodes=len(gidx), msg_dim=d.msg.shape[1], node_feat_dim=feats.shape[-1],
                         memory_dim=mcfg["memory_dim"], time_dim=mcfg["time_dim"],
                         embedding_dim=mcfg["embedding_dim"], out_horizons=len(ev["label_k_hours"]),
                         dropout=mcfg["dropout"], use_memory=mcfg.get("use_memory", True),
                         ctx_dim=ck.get("ctx_dim", 0), aggregator=mcfg.get("memory_aggregator", "last"),
                         role_flag=mcfg.get("role_flag", False)).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()
    runner = StreamRunner(model, d, feats, wmask, torch.device(device), tr["batch_size"], tr.get("num_neighbors", 10),
                          ctx=ctx, memory_step=tr.get("memory_step", "batch"))
    return runner, {"d": d, "panel": panel, "bounds": bounds, "wmask": wmask, "k_hours": ev["label_k_hours"],
                    "coins": list(panel["graph_coins"]), "feats": feats, "cfg": cfg,
                    "node_names": node_names, "msg_names": msg_names, "ctx_names": ctx_names,
                    "graph_index": list(gidx)}
