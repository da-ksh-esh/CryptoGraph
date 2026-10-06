"""DyGFormer (Yu et al., NeurIPS 2023), a compact re-implementation for the shock-event stream.

For an event (coin i shocks at hour t, receiver j):
  1. take each coin's last S interactions before t (neighbour, time, event message), from `InteractionHistory`;
  2. encode every interaction: neighbour's node features at that time, the event message, the time since it
     happened, and the neighbour co-occurrence counts (how often that neighbour appears in i's and in j's history);
  3. cut each history into patches of P interactions and project each patch to one token;
  4. run a Transformer over the tokens of i and j together, mean-pool each side;
  5. predict the K horizons from [h_i, h_j, node features of i and j at t, event message].
There is no memory: the model only reads the stored history, which holds events from earlier batches (hours) only.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class InteractionHistory:
    """Last S interactions of every node (neighbour id, time, event position in the stream), oldest first.
    Updated only after a batch has been predicted."""

    def __init__(self, num_nodes: int, S: int, device):
        self.N, self.S, self.device = num_nodes, S, device
        self.reset()

    def reset(self):
        self.nbr = torch.full((self.N, self.S), -1, dtype=torch.long, device=self.device)
        self.t = torch.zeros((self.N, self.S), dtype=torch.long, device=self.device)
        self.pos = torch.full((self.N, self.S), -1, dtype=torch.long, device=self.device)

    def insert(self, src: torch.Tensor, dst: torch.Tensor, t: torch.Tensor, pos: torch.Tensor):
        """Each event is an interaction for both ends (src sees dst, dst sees src), appended in stream order."""
        node = torch.cat([src, dst])
        other = torch.cat([dst, src])
        tt, pp = torch.cat([t, t]), torch.cat([pos, pos])
        order = torch.argsort(pp * 2 + torch.cat([torch.zeros_like(pos), torch.ones_like(pos)]), stable=True)
        node, other, tt, pp = node[order], other[order], tt[order], pp[order]
        for n in torch.unique(node).tolist():
            sel = node == n
            self.nbr[n] = torch.cat([self.nbr[n], other[sel]])[-self.S:]
            self.t[n] = torch.cat([self.t[n], tt[sel]])[-self.S:]
            self.pos[n] = torch.cat([self.pos[n], pp[sel]])[-self.S:]

    def get(self, nodes: torch.Tensor):
        return self.nbr[nodes], self.t[nodes], self.pos[nodes]


class TimeEncoding(nn.Module):
    """cos(w * dt + b) with learnable frequencies (as in TGAT / DyGFormer)."""

    def __init__(self, dim: int):
        super().__init__()
        self.lin = nn.Linear(1, dim)
        with torch.no_grad():
            self.lin.weight.copy_((1 / 10 ** torch.linspace(0, 9, dim)).reshape(dim, 1))
            self.lin.bias.zero_()

    def forward(self, dt: torch.Tensor) -> torch.Tensor:
        return torch.cos(self.lin(dt.unsqueeze(-1).float()))


class DyGFormer(nn.Module):
    def __init__(self, num_nodes: int, msg_dim: int, node_feat_dim: int, seq_len: int = 32, patch: int = 4,
                 time_dim: int = 16, cooc_dim: int = 16, d_model: int = 64, n_layers: int = 2, n_heads: int = 2,
                 out_horizons: int = 5, dropout: float = 0.2, encoder_dropout: float = 0.0):
        super().__init__()
        assert seq_len % patch == 0, "seq_len must be a multiple of patch"
        self.num_nodes, self.S, self.P = num_nodes, seq_len, patch
        self.time_enc = TimeEncoding(time_dim)
        self.cooc = nn.Sequential(nn.Linear(1, cooc_dim), nn.ReLU(), nn.Linear(cooc_dim, cooc_dim))
        tok_in = patch * (node_feat_dim + msg_dim + time_dim + cooc_dim)
        self.patch_proj = nn.Linear(tok_in, d_model)
        # encoder dropout is off by default: its random masks were ~40% of CPU time; regularisation comes from the
        # head dropout, weight decay and early stopping on validation
        layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=2 * d_model, dropout=encoder_dropout,
                                           batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Sequential(nn.Linear(2 * d_model + 2 * node_feat_dim + msg_dim, d_model), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(d_model, out_horizons))

    @staticmethod
    def cooccurrence(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """For each neighbour in a (B, S): how often it appears in a and in b. Padding (-1) counts as 0."""
        ca = (a.unsqueeze(-1) == a.unsqueeze(-2)).sum(-1)
        cb = (a.unsqueeze(-1) == b.unsqueeze(-2)).sum(-1)
        valid = a >= 0
        return torch.stack([ca * valid, cb * valid], dim=-1).float()

    def encode_side(self, nbr, tt, feats_nbr, msg_nbr, t_now, cooc):
        """(B, S) history of one side -> (B, S/P, d_model) patch tokens."""
        valid = (nbr >= 0).unsqueeze(-1).float()
        dt = (t_now.unsqueeze(-1) - tt).clamp_min(0)
        c = self.cooc(cooc.unsqueeze(-1)).sum(-2)                         # f(count in a) + f(count in b)
        x = torch.cat([feats_nbr, msg_nbr, self.time_enc(dt), c], dim=-1) * valid
        B = x.shape[0]
        return self.patch_proj(x.reshape(B, self.S // self.P, -1))

    def forward(self, hist_src, hist_dst, feats_src_nbr, feats_dst_nbr, msg_src_nbr, msg_dst_nbr, t_now,
                x_src, x_dst, msg):
        (ns, ts), (nd, td) = hist_src, hist_dst
        tok_s = self.encode_side(ns, ts, feats_src_nbr, msg_src_nbr, t_now, self.cooccurrence(ns, nd))
        tok_d = self.encode_side(nd, td, feats_dst_nbr, msg_dst_nbr, t_now, self.cooccurrence(nd, ns))
        z = self.encoder(torch.cat([tok_s, tok_d], dim=1))
        k = tok_s.shape[1]
        h_s, h_d = z[:, :k].mean(1), z[:, k:].mean(1)
        return self.head(torch.cat([h_s, h_d, x_src, x_dst, msg], dim=-1))
