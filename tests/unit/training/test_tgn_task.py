import numpy as np
import torch

from cryptograph.models.tgn import ContagionTGN
from cryptograph.training.tgn_task import StreamRunner, _batches, node_feature_table


def _stream(E=600, N=6, T=200, seed=0):
    from torch_geometric.data import TemporalData

    g = torch.Generator().manual_seed(seed)
    t = torch.sort(torch.randint(0, T, (E,), generator=g)).values
    src = torch.randint(0, N, (E,), generator=g)
    dst = (src + torch.randint(1, N, (E,), generator=g)) % N
    d = TemporalData(src=src, dst=dst, t=t, msg=torch.randn(E, 5, generator=g))
    d.y = torch.rand(E, 2, generator=g) < 0.2
    d.y_mask = torch.ones(E, 2, dtype=torch.bool)
    return d


def test_batches_cover_the_range_and_never_split_a_bar():
    t = np.array([0, 0, 1, 1, 1, 2, 3, 3, 3, 3, 4])
    b = _batches(t, 0, len(t), 3)
    assert b[0][0] == 0 and b[-1][1] == len(t) and all(x[1] == y[0] for x, y in zip(b, b[1:]))
    assert all(t[e - 1] != t[e] for _, e in b[:-1])


def test_node_feature_table_uses_latest_snapshot_at_or_before_t():
    T, N = 10, 3
    panel = {"X": torch.arange(T * N * 2, dtype=torch.float32).reshape(T, N, 2)}
    net = {"snapshot_t": torch.tensor([3, 7]), "X": torch.tensor([[[1.0]] * N, [[2.0]] * N])}
    f = node_feature_table(panel, net, [0, 1, 2])
    assert f.shape == (T, N, 3)
    assert f[:3, :, 2].eq(0).all() and f[3:7, :, 2].eq(1).all() and f[7:, :, 2].eq(2).all()


def _predict_first_batches(d, n_batches):
    torch.manual_seed(0)
    N = 6
    model = ContagionTGN(num_nodes=N, msg_dim=5, node_feat_dim=2, memory_dim=8, time_dim=4, embedding_dim=8,
                         out_horizons=2, dropout=0.0)
    model.eval()
    feats = torch.randn(200, N, 2, generator=torch.Generator().manual_seed(1))
    runner = StreamRunner(model, d, feats, {"val": np.ones(200, bool)}, torch.device("cpu"), batch_size=50)
    runner.reset()
    _, logits, _, _ = runner.run(0, len(d.t), "val", max_batches=n_batches)
    return logits


def test_predictions_do_not_depend_on_later_events():
    d = _stream()
    base = _predict_first_batches(d, 4)
    cut = _batches(d.t.numpy(), 0, len(d.t), 50)[3][1]          # end of the 4th batch
    d2 = _stream()
    d2.msg[cut:] = 100.0                                       # change every later event and label
    d2.y[cut:] = ~d2.y[cut:]
    d2.src[cut:] = 0
    assert torch.allclose(base, _predict_first_batches(d2, 4))
    d3 = _stream()
    d3.msg[cut - 1] += 5.0                                     # an event inside the window does change them
    assert not torch.allclose(base, _predict_first_batches(d3, 4))
