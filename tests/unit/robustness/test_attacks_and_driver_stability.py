import torch
import torch.nn.functional as F

from cryptograph.explain.attribution import driver_stability, flat_attributions, integrated_gradients
from cryptograph.explain.tgn_capture import ExplainableTGN
from cryptograph.models.tgn import ContagionTGN
from cryptograph.robustness.stress import gradient_attack


def _head(seed=0):
    torch.manual_seed(seed)
    m = ContagionTGN(num_nodes=6, msg_dim=5, node_feat_dim=4, memory_dim=8, time_dim=4, embedding_dim=8,
                     out_horizons=3, dropout=0.0).eval()
    g = torch.Generator().manual_seed(seed + 1)
    B = 64
    zs, zd = torch.randn(B, 8, generator=g), torch.randn(B, 8, generator=g)
    x = [torch.randn(B, 4, generator=g), torch.randn(B, 4, generator=g), torch.randn(B, 5, generator=g)]
    y = torch.rand(B, 3, generator=g) < 0.3
    ym = torch.ones(B, 3, dtype=torch.bool)
    return m, (lambda a, b, c: m.decode(zs, zd, a, b, c)), x, y, ym


def _loss(decode, x, y, ym):
    with torch.no_grad():
        return float(F.binary_cross_entropy_with_logits(decode(*x)[ym], y[ym].float()))


def test_one_step_attack_is_exactly_the_old_fgsm():
    _, decode, x, y, ym = _head()
    inp = [t.clone().requires_grad_(True) for t in x]
    grads = torch.autograd.grad(F.binary_cross_entropy_with_logits(decode(*inp)[ym], y[ym].float()), inp)
    old = [t + 0.1 * g.sign() for t, g in zip(x, grads)]              # robustness_1h's FGSM before the port
    new = gradient_attack(decode, x, y, ym, 0.1, steps=1)
    for a, b in zip(old, new):
        torch.testing.assert_close(a, b)


def test_pgd_stays_in_the_ball_and_is_at_least_as_strong_as_fgsm():
    _, decode, x, y, ym = _head()
    eps = 0.1
    fgsm = gradient_attack(decode, x, y, ym, eps, steps=1)
    pgd = gradient_attack(decode, x, y, ym, eps, steps=10, generator=torch.Generator().manual_seed(0))
    for o, a in zip(x, pgd):
        assert (a - o).abs().max() <= eps + 1e-6
    clean = _loss(decode, x, y, ym)
    assert _loss(decode, pgd, y, ym) >= _loss(decode, fgsm, y, ym) - 1e-4 > clean


def test_driver_stability_is_perfect_without_noise_and_drops_with_it():
    from tests.unit.explain.test_tempme import _toy_capture

    torch.manual_seed(3)
    model = ContagionTGN(num_nodes=10, msg_dim=5, node_feat_dim=6, memory_dim=8, time_dim=4, embedding_dim=8).eval()
    ex = ExplainableTGN(model)
    cap = _toy_capture(model, N=40)
    attr, _ = integrated_gradients(ex, cap, 1, steps=8)
    assert flat_attributions(attr).shape == (40, 2 + 6 + 6 + 5)
    s0 = driver_stability(ex, cap, 1, noise=0.0, top=5, steps=8)
    assert s0["mean_jaccard_topk"] == 1.0 and s0["top1_unchanged"] == 1.0 and s0["mean_abs_dp"] == 0.0
    s1 = driver_stability(ex, cap, 1, noise=3.0, top=5, steps=8)
    assert s1["mean_jaccard_topk"] < 1.0 and s1["mean_abs_dp"] > 0
