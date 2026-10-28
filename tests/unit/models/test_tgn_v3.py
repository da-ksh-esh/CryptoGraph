"""v3 TGN options: role flag, mean aggregator, hourly memory updates (sandbox exp/accuracy-v3)."""

import numpy as np
import pytest
import torch

pytest.importorskip("torch_geometric")

from torch_geometric.data import TemporalData  # noqa: E402

from cryptograph.models.tgn import ContagionTGN, RoleMessage  # noqa: E402
from cryptograph.training.losses import focal_loss  # noqa: E402
from cryptograph.training.tgn_task import StreamRunner  # noqa: E402


def test_role_message_flags_sender_and_receiver():
    m = ContagionTGN(num_nodes=4, msg_dim=3, node_feat_dim=2, memory_dim=8, time_dim=4, embedding_dim=8,
                     aggregator="mean", role_flag=True)
    assert isinstance(m.memory.msg_s_module, RoleMessage) and m.memory.msg_s_module.role == 0.0
    assert m.memory.msg_d_module.role == 1.0
    out = m.memory.msg_d_module(torch.zeros(2, 8), torch.zeros(2, 8), torch.zeros(2, 3), torch.zeros(2, 4))
    assert out.shape == (2, m.memory.msg_s_module.out_channels) and (out[:, 8 + 8 + 3] == 1).all()


def _stream(n_hours=6, per_hour=5, n_nodes=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    t = torch.arange(n_hours).repeat_interleave(per_hour)
    src = torch.randint(0, n_nodes, (len(t),), generator=g)
    dst = (src + 1 + torch.randint(0, n_nodes - 1, (len(t),), generator=g)) % n_nodes
    d = TemporalData(src=src, dst=dst, t=t, msg=torch.randn(len(t), 3, generator=g))
    d.y = torch.rand(len(t), 2, generator=g) > 0.5
    d.y_mask = torch.ones(len(t), 2, dtype=torch.bool)
    return d


@pytest.mark.parametrize("step", ["batch", "hour"])
def test_runner_scores_every_event_once_in_order(step):
    torch.manual_seed(0)
    d = _stream()
    m = ContagionTGN(num_nodes=4, msg_dim=3, node_feat_dim=2, memory_dim=8, time_dim=4, embedding_dim=8,
                     out_horizons=2, aggregator="mean", role_flag=True)
    feats = torch.randn(6, 4, 2)
    r = StreamRunner(m, d, feats, {"val": np.ones(6, bool)}, torch.device("cpu"), batch_size=12, memory_step=step)
    r.reset()
    _, lg, y, msk = r.run(0, len(d.t), "val", loss_fn=lambda a, b, c: focal_loss(a, b, c))
    assert lg.shape == (len(d.t), 2) and torch.equal(r.last_positions, torch.arange(len(d.t)))


def test_hour_step_sees_the_previous_hour_inside_a_batch():
    """With one batch spanning two hours, hour-1 predictions differ between the modes only because the hourly
    mode has already written hour 0 into memory."""
    d = _stream(n_hours=2, per_hour=6)
    outs = {}
    for step in ("batch", "hour"):
        torch.manual_seed(0)
        m = ContagionTGN(num_nodes=4, msg_dim=3, node_feat_dim=2, memory_dim=8, time_dim=4, embedding_dim=8,
                         out_horizons=2, aggregator="mean")
        r = StreamRunner(m, d, torch.zeros(2, 4, 2), {"val": np.ones(2, bool)}, torch.device("cpu"), batch_size=100,
                         memory_step=step)
        r.reset()
        outs[step] = r.run(0, len(d.t), "val", loss_fn=lambda a, b, c: focal_loss(a, b, c))[1]
    assert torch.allclose(outs["batch"][:6], outs["hour"][:6])          # hour 0: same (empty) memory
    assert not torch.allclose(outs["batch"][6:], outs["hour"][6:])      # hour 1: hourly mode knows hour 0
