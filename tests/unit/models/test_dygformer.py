import numpy as np
import torch

from cryptograph.models.dygformer import DyGFormer, InteractionHistory
from cryptograph.training.dygformer_task import HistoryRunner


def test_history_keeps_last_s_in_stream_order():
    h = InteractionHistory(4, 3, "cpu")
    h.insert(torch.tensor([0, 0]), torch.tensor([1, 2]), torch.tensor([5, 5]), torch.tensor([0, 1]))
    h.insert(torch.tensor([3, 0]), torch.tensor([0, 3]), torch.tensor([6, 7]), torch.tensor([2, 3]))
    nbr, t, pos = h.get(torch.tensor([0, 1]))
    assert nbr[0].tolist() == [2, 3, 3] and t[0].tolist() == [5, 6, 7] and pos[0].tolist() == [1, 2, 3]
    assert nbr[1].tolist() == [-1, -1, 0] and pos[1].tolist() == [-1, -1, 0]


def test_cooccurrence_counts():
    a = torch.tensor([[1, 1, 2, -1]])
    b = torch.tensor([[2, 2, 2, 3]])
    c = DyGFormer.cooccurrence(a, b)[0]
    assert c[:, 0].tolist() == [2, 2, 1, 0] and c[:, 1].tolist() == [0, 0, 3, 0]


def _stream(E=500, N=6, T=150, seed=0):
    from torch_geometric.data import TemporalData

    g = torch.Generator().manual_seed(seed)
    t = torch.sort(torch.randint(0, T, (E,), generator=g)).values
    src = torch.randint(0, N, (E,), generator=g)
    dst = (src + torch.randint(1, N, (E,), generator=g)) % N
    d = TemporalData(src=src, dst=dst, t=t, msg=torch.randn(E, 5, generator=g))
    d.y = torch.rand(E, 2, generator=g) < 0.2
    d.y_mask = torch.ones(E, 2, dtype=torch.bool)
    return d


def _first_batches(d, n):
    torch.manual_seed(0)
    model = DyGFormer(6, 5, 3, seq_len=8, patch=2, d_model=16, out_horizons=2, dropout=0.0)
    model.eval()
    feats = torch.randn(150, 6, 3, generator=torch.Generator().manual_seed(1))
    r = HistoryRunner(model, d, feats, {"val": np.ones(150, bool)}, torch.device("cpu"), batch_size=40)
    r.reset()
    return r.run(0, len(d.t), "val", max_batches=n)[1]


def test_dygformer_predictions_do_not_depend_on_later_events():
    from cryptograph.training.tgn_task import _batches

    base = _first_batches(_stream(), 4)
    cut = _batches(_stream().t.numpy(), 0, 500, 40)[3][1]
    d2 = _stream()
    d2.msg[cut:] = 50.0
    d2.y[cut:] = ~d2.y[cut:]
    d2.dst[cut:] = (d2.src[cut:] + 1) % 6
    assert torch.allclose(base, _first_batches(d2, 4))
    d3 = _stream()
    d3.msg[cut - 1] += 3.0
    assert not torch.allclose(base, _first_batches(d3, 4))
