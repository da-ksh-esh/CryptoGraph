import numpy as np
import torch

from cryptograph.explain.attribution import completeness_gap, integrated_gradients
from cryptograph.explain.fidelity import _topk_mask, fidelity_curve, random_importance
from cryptograph.explain.tempme import N_CAT, TempMEExplainer, _category, build_walks
from cryptograph.explain.tgn_capture import Captured, ExplainableTGN
from cryptograph.models.tgn import ContagionTGN


def test_motif_categories():
    assert _category([0, 1], other_end=5) == 0
    assert _category([0, 1, 0], 5) == 1 and _category([0, 1, 2], 5) == 2
    assert _category([0, 1, 0, 1], 5) == 3 and _category([0, 1, 0, 2], 5) == 4
    assert _category([0, 1, 2, 0], 5) == 5 and _category([0, 1, 2, 1], 5) == 6 and _category([0, 1, 2, 3], 5) == 7
    assert _category([0, 5, 2, 3], 5) == 15                     # step 1 links the two ends of the explained event
    assert max(_category([0, 5, 0, 5], 5), _category([0, 5], 5)) < N_CAT


def test_walk_steps_go_back_in_time():
    # events: (src, dst, t)
    src = np.array([1, 2, 3, 1, 0])
    dst = np.array([2, 3, 4, 0, 4])
    t = np.array([1, 2, 3, 5, 9])
    # explained event p=4 (0 -> 4 at t=9); step-1 event of centre 0 is p=3 (1 -> 0 at t=5), neighbour 1
    pos_n = np.full((1, 2, 2), -1)
    nbr = np.full((1, 2, 2), -1)
    pos_n[0, 0, 0], nbr[0, 0, 0] = 3, 1
    W = build_walks(pos_n, nbr, np.array([[0, 4]]), src, dst, t, n_nodes=5, R2=3, R3=2)
    steps = W["steps"][0, 0, 0]
    valid = W["wmask"][0, 0, 0]
    assert valid.sum() == 1
    s = steps[valid][0].tolist()                                # 3 (t=5) -> 0 (1->2, t=1); node 2's earlier: none
    assert s == [3, 0, -1]
    assert all(t[a] > t[b] for a, b in zip(s, s[1:]) if b >= 0)
    assert W["cat"][0, 0, 0][valid][0] == 2                     # 0 -> 1 -> 2: new node, 2 steps


def test_topk_mask_respects_validity():
    imp = torch.tensor([[[0.9, 0.1, 0.8], [0.7, 0.95, 0.0]]])
    valid = torch.tensor([[[True, True, True], [True, False, True]]])
    m = _topk_mask(imp, valid, 2)
    assert m.sum() == 2 and m[0, 0, 0] and m[0, 0, 2] and not m[0, 1, 1]


def _toy_capture(model, N=7, L=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    D, M, F = model.memory.memory_dim, 5, 6
    mask = torch.rand(N, 2, L, generator=g) > 0.3
    mask[0, 0] = False                                          # a centre with no recent events
    pos_n = torch.where(mask, torch.arange(N * 2 * L).view(N, 2, L), torch.tensor(-1))
    return Captured(pos=torch.arange(N), x_c=torch.randn(N, 2, D, generator=g), x_n=torch.randn(N, 2, L, D, generator=g),
                    rel_t=torch.randint(-50, 0, (N, 2, L), generator=g).float(), msg_n=torch.randn(N, 2, L, M, generator=g),
                    pos_n=pos_n, nbr=torch.where(mask, torch.tensor(1), torch.tensor(-1)),
                    x_src=torch.randn(N, F, generator=g), x_dst=torch.randn(N, F, generator=g),
                    msg=torch.randn(N, M, generator=g), logits=torch.zeros(N, 5))


def test_rebuilt_attention_matches_transformer_conv():
    torch.manual_seed(0)
    model = ContagionTGN(num_nodes=10, msg_dim=5, node_feat_dim=6, memory_dim=8, time_dim=4, embedding_dim=8).eval()
    ex = ExplainableTGN(model)
    cap = _toy_capture(model)
    n = 3                                                       # compare one centre against PyG's own layer
    k = int(cap.mask[n, 0].sum())
    x = torch.cat([cap.x_c[n, 0:1], cap.x_n[n, 0][cap.mask[n, 0]]])
    edge_index = torch.stack([torch.arange(1, k + 1), torch.zeros(k, dtype=torch.long)])
    last_update = torch.cat([torch.zeros(1), cap.rel_t[n, 0][cap.mask[n, 0]]]).long()
    ref = model.gnn(x, last_update, edge_index, torch.zeros(k, dtype=torch.long), cap.msg_n[n, 0][cap.mask[n, 0]])[0]
    z, _ = ex.embed(cap.x_c[n:n + 1, 0], cap.x_n[n:n + 1, 0], cap.rel_t[n:n + 1, 0], cap.msg_n[n:n + 1, 0],
                    cap.mask[n:n + 1, 0])
    torch.testing.assert_close(z[0], ref, atol=1e-5, rtol=1e-5)
    # weights of 1 = the model; weights of 0 remove every event
    with torch.no_grad():
        torch.testing.assert_close(ex(cap), ex(cap, cap.mask.float()))
        z0, _ = ex.embed(cap.x_c[:, 0], cap.x_n[:, 0], cap.rel_t[:, 0], cap.msg_n[:, 0], cap.mask[:, 0],
                         torch.zeros(len(cap), cap.rel_t.shape[-1]))
        torch.testing.assert_close(z0, ex.conv.lin_skip(cap.x_c[:, 0]))


def test_fidelity_and_integrated_gradients_run():
    torch.manual_seed(1)
    model = ContagionTGN(num_nodes=10, msg_dim=5, node_feat_dim=6, memory_dim=8, time_dim=4, embedding_dim=8).eval()
    ex = ExplainableTGN(model)
    cap = _toy_capture(model, N=20)
    rows = fidelity_curve(ex, cap, random_importance(cap), ks=(1, 8))
    assert rows[-1]["fid_minus"] <= 1e-6 or rows[-1]["kept_events"] >= cap.mask.sum((1, 2)).max()
    attr, gap = integrated_gradients(ex, cap, horizon=2, steps=64)
    assert completeness_gap(attr, gap) < 1e-2


def test_explainer_outputs_probabilities_and_finite_kl():
    torch.manual_seed(2)
    ex_net = TempMEExplainer(node_dim=6, msg_dim=5, hid=16)
    B, L, R = 3, 4, 6
    steps = torch.randint(-1, 20, (B, 2, L, R, 3))
    steps[..., 0] = torch.randint(0, 20, (B, 2, L, R))
    nodes = torch.randint(0, 10, (B, 2, L, R, 4))
    cat = torch.randint(0, N_CAT, (B, 2, L, R))
    wmask = torch.rand(B, 2, L, R) > 0.2
    wi = ex_net.walk_importance(steps, nodes, cat, wmask, torch.rand(B, 2, L, R, 3), torch.full((B,), 30),
                                torch.randn(20, 5), torch.arange(20), torch.randn(B, 10, 6))
    assert ((wi >= 0) & (wi <= 1)).all() and (wi[~wmask] == 0).all()
    kl = ex_net.kl_loss(wi, cat, wmask, torch.full((N_CAT,), 1 / N_CAT))
    assert torch.isfinite(kl)
