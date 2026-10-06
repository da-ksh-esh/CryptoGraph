"""TempME (Chen & Ying, NeurIPS 2023) ported to our ContagionTGN.

Follows the official implementation (github.com/Graph-and-Geometric-Learning/TempME, models/explainer.py,
temp_exp_main.py): sample temporal motifs (time-ordered walks) around both ends of the explained event, encode each
motif (event-GIN over its steps, attention over the steps, motif-category one-hot), output an importance in (0, 1)
per motif, turn motif importance into an importance per event (max over the motifs that contain it), relax it with a
concrete Bernoulli, multiply it into the TGN's attention over recent events, and train with
    loss = prediction loss(explained prediction, original prediction) + beta * KL(motif importance || null prior)
where the prior is the motif-category distribution of a temporally shuffled stream scaled by the prior sparsity.

What differs from the official code, and why:
  * Walks: from centre c, step 1 is one of c's L recent events (exactly the events our 1-layer TGN attends to), then
    up to R2 earlier events of the neighbour, then up to R3 earlier events of the next node (most recent first,
    deterministic, like TempME's "recent" sampler). Only step-1 events enter our TGN (1 layer); steps 2-3 give context.
  * Motif categories: anonymised node pattern of the walk (1, 2 or 3 steps; 8 patterns) x whether step 1 links the
    two ends of the explained event = 16 categories (official: 12, defined for its 3-step walks).
  * Prediction loss: binary cross-entropy against the model's original *probability* per horizon. The official code
    uses the hard label (p > 0.5) of a balanced link-prediction task; with contagion base rates of 10-60% almost every
    hard label would be 0 and the explainer would learn nothing about the actual risk.
  * Node features of a step: our node features at the explained bar t (the official code uses static node features).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N_CAT = 16


def _node_index(src: np.ndarray, dst: np.ndarray, t: np.ndarray, n_nodes: int):
    """For each node: positions of its events sorted by time, and their times."""
    pos = np.concatenate([np.arange(len(src)), np.arange(len(src))])
    node = np.concatenate([src, dst])
    order = np.lexsort((pos, t[pos], node))
    pos, node = pos[order], node[order]
    cuts = np.searchsorted(node, np.arange(n_nodes + 1))
    lists = [pos[cuts[u]:cuts[u + 1]] for u in range(n_nodes)]
    return lists, [t[p] for p in lists]


def _category(nodes: list[int], other_end: int) -> int:
    """Anonymised pattern of a walk c=u0 -> u1 -> u2 -> u3 (1-3 steps) x whether u1 is the other end."""
    n = len(nodes) - 1
    if n == 1:
        pat = 0
    elif n == 2:
        pat = 1 if nodes[2] == nodes[0] else 2
    else:
        u0, u1, u2, u3 = nodes
        if u2 == u0:
            pat = 3 if u3 == u1 else 4
        else:
            pat = 5 if u3 == u0 else (6 if u3 == u1 else 7)
    return pat + 8 * int(nodes[1] == other_end)


def build_walks(pos_n: np.ndarray, nbr: np.ndarray, centres: np.ndarray, src: np.ndarray, dst: np.ndarray,
                t: np.ndarray, n_nodes: int, R2: int = 3, R3: int = 2) -> dict:
    """Temporal motifs for captured events.

    pos_n, nbr: (N, 2, L) step-1 events and their other ends (-1 = none); centres: (N, 2) = (i, j).
    src, dst, t: the event stream used for steps 2-3 (the real one, or a time-shuffled one for the null model).
    Returns step positions (N, 2, L, R, 3), step nodes (N, 2, L, R, 4), categories (N, 2, L, R), walk mask, R = R2*R3.
    """
    N, _, L = pos_n.shape
    R = R2 * R3
    lists, times = _node_index(src, dst, t, n_nodes)
    steps = np.full((N, 2, L, R, 3), -1, np.int64)
    nodes = np.full((N, 2, L, R, 4), -1, np.int64)
    cat = np.zeros((N, 2, L, R), np.int64)
    wmask = np.zeros((N, 2, L, R), bool)

    def earlier(u, before, k):
        lo = np.searchsorted(times[u], before, side="left")
        return lists[u][max(0, lo - k):lo][::-1]                   # most recent first

    for n in range(N):
        for c in range(2):
            u0, other = int(centres[n, c]), int(centres[n, 1 - c])
            for l in range(L):
                p1 = int(pos_n[n, c, l])
                if p1 < 0:
                    continue
                u1 = int(nbr[n, c, l])
                r = 0
                e2s = earlier(u1, t[p1], R2)
                if len(e2s) == 0:
                    steps[n, c, l, 0, 0], nodes[n, c, l, 0, :2] = p1, (u0, u1)
                    cat[n, c, l, 0], wmask[n, c, l, 0] = _category([u0, u1], other), True
                    continue
                for p2 in e2s:
                    u2 = int(dst[p2] if src[p2] == u1 else src[p2])
                    e3s = earlier(u2, t[p2], R3)
                    if len(e3s) == 0:
                        steps[n, c, l, r, :2], nodes[n, c, l, r, :3] = (p1, p2), (u0, u1, u2)
                        cat[n, c, l, r], wmask[n, c, l, r] = _category([u0, u1, u2], other), True
                        r += 1
                        continue
                    for p3 in e3s:
                        u3 = int(dst[p3] if src[p3] == u2 else src[p3])
                        steps[n, c, l, r], nodes[n, c, l, r] = (p1, p2, p3), (u0, u1, u2, u3)
                        cat[n, c, l, r], wmask[n, c, l, r] = _category([u0, u1, u2, u3], other), True
                        r += 1
    # TempME's "edge identity" feature: how often each step's event occurs in the walks of this centre
    counts = np.zeros(steps.shape, np.float32)
    for n in range(N):
        for c in range(2):
            s = steps[n, c].reshape(-1, 3)
            vals, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
            cc = cnt[inv].reshape(s.shape).astype(np.float32)
            cc[s < 0] = 0
            counts[n, c] = cc.reshape(L, R, 3) / max(1, (s[:, 0] >= 0).sum())
    return {"steps": torch.from_numpy(steps), "nodes": torch.from_numpy(nodes), "cat": torch.from_numpy(cat),
            "wmask": torch.from_numpy(wmask), "counts": torch.from_numpy(counts)}


def null_distribution(walks_null: dict) -> torch.Tensor:
    """Motif-category frequencies of walks built on a time-shuffled stream (TempME's null model)."""
    c = walks_null["cat"][walks_null["wmask"]]
    freq = torch.bincount(c, minlength=N_CAT).float()
    return (freq + 1.0) / (freq.sum() + N_CAT)                     # add-one so no category has zero prior


class TimeEncode(nn.Module):
    """cos(w dt + b), as in TGAT / TempME."""

    def __init__(self, dim: int):
        super().__init__()
        self.lin = nn.Linear(1, dim)
        with torch.no_grad():
            self.lin.weight.copy_((1 / 10 ** torch.linspace(0, 9, dim)).reshape(dim, 1))
            self.lin.bias.zero_()

    def forward(self, dt):
        return torch.cos(self.lin(dt.unsqueeze(-1).float()))


class EventGCN(nn.Module):
    """TempME's event_gcn: MLP(src + ReLU(tgt + lin(event)))."""

    def __init__(self, event_dim: int, node_dim: int, hid: int):
        super().__init__()
        self.lin_event = nn.Linear(event_dim, node_dim)
        self.mlp = nn.Sequential(nn.Linear(node_dim, hid), nn.ReLU(), nn.Linear(hid, hid))

    def forward(self, event, src, tgt):
        return self.mlp(src + F.relu(tgt + self.lin_event(event)))


class StepAttention(nn.Module):
    """TempME's Attention: the step next to the explained event attends to the other steps of its walk."""

    def __init__(self, input_dim: int, hid: int):
        super().__init__()
        self.W1, self.W2 = nn.Linear(input_dim, input_dim), nn.Linear(input_dim, input_dim)
        self.mlp = nn.Sequential(nn.Linear(input_dim, hid), nn.ReLU(), nn.Linear(hid, hid))
        nn.init.xavier_uniform_(self.W2.weight)
        self.W2.bias.data.fill_(0.1)

    def forward(self, x, step_mask):
        """x (..., 3, D), step_mask (..., 3) -> (..., hid)."""
        q, k = self.W1(x[..., :1, :]), self.W2(x[..., 1:, :])
        score = (q * k).sum(-1).masked_fill(~step_mask[..., 1:], float("-inf"))
        a = torch.nan_to_num(torch.softmax(score, dim=-1), nan=0.0)
        return self.mlp(x[..., 0, :] + (a.unsqueeze(-1) * k).sum(-2))


class TempMEExplainer(nn.Module):
    def __init__(self, node_dim: int, msg_dim: int, time_dim: int = 16, hid: int = 64, temp: float = 0.07,
                 dropout: float = 0.1):
        super().__init__()
        self.temp = temp
        self.time_enc = TimeEncode(time_dim)
        event_dim = msg_dim + time_dim + 1
        self.event_gcn = EventGCN(event_dim, node_dim, hid)
        self.attn = StepAttention(2 * hid, hid)
        d = hid + N_CAT
        self.mlp = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d, hid), nn.ReLU(),
                                 nn.Linear(hid, 1))

    def walk_importance(self, steps, nodes, cat, wmask, counts, t_now, msg_all, t_all, feats_now):
        """Importance in (0, 1) per motif: (B, 2, L, R). feats_now: (B, n_nodes, F) node features at t."""
        smask = steps >= 0
        sp = steps.clamp_min(0)
        msg = msg_all[sp] * smask.unsqueeze(-1)                                     # (B,2,L,R,3,M)
        dt = (t_now.view(-1, 1, 1, 1, 1) - t_all[sp]).float().clamp_min(0) * smask
        event = torch.cat([msg, self.time_enc(dt), counts.unsqueeze(-1)], dim=-1)
        B = steps.shape[0]
        bidx = torch.arange(B).view(B, 1, 1, 1, 1)
        nd = nodes.clamp_min(0)
        a = feats_now[bidx, nd[..., :3]] * smask.unsqueeze(-1)                      # step k: nodes k -> k+1
        b = feats_now[bidx, nd[..., 1:]] * smask.unsqueeze(-1)
        h = torch.cat([self.event_gcn(event, a, b), self.event_gcn(event, b, a)], dim=-1)
        z = self.attn(h, smask)
        z = torch.cat([z, F.one_hot(cat, N_CAT).float()], dim=-1)
        return torch.sigmoid(self.mlp(z).squeeze(-1)) * wmask

    @staticmethod
    def event_importance(walk_imp, wmask):
        """Importance of each step-1 event (the TGN's attended events): max over the motifs that start with it."""
        return walk_imp.masked_fill(~wmask, 0.0).amax(-1)                           # (B, 2, L)

    def concrete_bern(self, p):
        u = torch.empty_like(p).uniform_(1e-10, 1 - 1e-10)
        logit = torch.log(p + 1e-10) - torch.log(1 - p + 1e-10)
        return torch.sigmoid((logit + torch.log(u) - torch.log(1 - u)) / self.temp)

    @staticmethod
    def kl_loss(walk_imp, cat, wmask, null: torch.Tensor, target: float = 0.3):
        """TempME's empirical-prior KL (official kl_loss, prior='empirical'), over valid motifs only."""
        B = walk_imp.shape[0]
        p = walk_imp.reshape(B, -1)
        m = wmask.reshape(B, -1).float()
        c = cat.reshape(B, -1)
        n = m.sum(1).clamp_min(1)
        s = (p * m).sum(1) / n                                                      # overall importance
        num = torch.zeros(B, N_CAT).scatter_add_(1, c, p * m)
        den = torch.zeros(B, N_CAT).scatter_add_(1, c, m).clamp_min(1)
        emp = s.unsqueeze(1) * (num / den)
        nul = target * null.view(1, -1)
        kl = (1 - s) * torch.log((1 - s) / (1 - target + 1e-6) + 1e-6) + \
             (emp * torch.log(emp / (nul + 1e-6) + 1e-6)).mean(1)   # official: .mean() over (B, n_cat)
        return kl.mean()


def _batch_inputs(walks: dict, idx, cap, t_all, msg_all, feats):
    t_now = t_all[cap.pos[idx]]
    return dict(steps=walks["steps"][idx], nodes=walks["nodes"][idx], cat=walks["cat"][idx],
                wmask=walks["wmask"][idx], counts=walks["counts"][idx], t_now=t_now, msg_all=msg_all, t_all=t_all,
                feats_now=feats[t_now])


@torch.no_grad()
def tempme_importance(explainer: TempMEExplainer, walks: dict, cap, t_all, msg_all, feats,
                      batch: int = 1024) -> torch.Tensor:
    """(N, 2, L) importance of each attended event (eval mode, no Bernoulli noise)."""
    explainer.eval()
    out = []
    for a in range(0, len(cap), batch):
        idx = torch.arange(a, min(a + batch, len(cap)))
        inp = _batch_inputs(walks, idx, cap, t_all, msg_all, feats)
        wi = explainer.walk_importance(**inp)
        out.append(explainer.event_importance(wi, inp["wmask"]) * cap.mask[idx])
    return torch.cat(out)


def train_tempme(ex, cap, walks: dict, null: torch.Tensor, t_all, msg_all, feats, *, epochs: int = 30,
                 batch: int = 256, lr: float = 1e-3, beta: float = 0.5, prior_p: float = 0.3, temp: float = 0.07,
                 hid: int = 64, holdout: float = 0.1, patience: int = 5, seed: int = 0, log=print):
    """Train the explainer on captured events (official defaults: lr 1e-3, beta 0.5, prior 0.3, temp 0.07, hid 64).
    Model selection on a held-out share of the training events: characterization score with 3 events kept."""
    from .fidelity import fidelity_curve

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(cap))
    n_ho = max(1, int(holdout * len(cap)))
    ho, tr = torch.from_numpy(np.sort(perm[:n_ho])), perm[n_ho:]
    cap_ho = cap.subset(ho)
    walks_ho = {k: v[ho] for k, v in walks.items()}
    explainer = TempMEExplainer(node_dim=feats.shape[-1], msg_dim=msg_all.shape[-1], hid=hid, temp=temp)
    opt = torch.optim.Adam(explainer.parameters(), lr=lr)
    with torch.no_grad():
        target = torch.sigmoid(cap.logits)
    best, best_state, bad, history = -np.inf, None, 0, []
    for epoch in range(epochs):
        explainer.train()
        rng.shuffle(tr)
        losses = []
        for a in range(0, len(tr), batch):
            idx = torch.from_numpy(np.sort(tr[a:a + batch]))
            inp = _batch_inputs(walks, idx, cap, t_all, msg_all, feats)
            wi = explainer.walk_importance(**inp)
            w = explainer.concrete_bern(explainer.event_importance(wi, inp["wmask"])) * cap.mask[idx]
            pred = ex(cap.subset(idx), w)
            loss = F.binary_cross_entropy_with_logits(pred, target[idx]) + \
                beta * explainer.kl_loss(wi, inp["cat"], inp["wmask"], null, target=prior_p)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss))
        imp = tempme_importance(explainer, walks_ho, cap_ho, t_all, msg_all, feats)
        score = fidelity_curve(ex, cap_ho, imp, ks=(3,))[0]["characterization"]
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "holdout_char_k3": score})
        log(f"TempME epoch {epoch}: loss {np.mean(losses):.5f}, hold-out characterization (3 events) {score:.4f}")
        if score > best:
            best, bad = score, 0
            best_state = {k: v.detach().clone() for k, v in explainer.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    explainer.load_state_dict(best_state)
    return explainer, history
