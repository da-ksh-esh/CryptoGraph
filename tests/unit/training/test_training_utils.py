import numpy as np
import pytest
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

from cryptograph.training import (EarlyStopping, auprc, auroc, focal_loss, huber_loss, mae, masked_mse, mse,
                                  pos_weight_from_rate, seed_everything, weighted_bce)
from cryptograph.training.loop import evaluate, train_epoch


def test_masked_losses_ignore_masked_entries():
    pred = torch.tensor([0.0, 0.0, 100.0])
    target = torch.zeros(3)
    mask = torch.tensor([True, True, False])
    assert masked_mse(pred, target, mask) == 0
    assert huber_loss(pred, target, mask) == 0
    assert huber_loss(pred, target) > 0


def test_huber_linear_in_tails():
    small = huber_loss(torch.tensor([0.5]), torch.tensor([0.0]))
    big = huber_loss(torch.tensor([10.0]), torch.tensor([0.0]))
    assert torch.isclose(small, torch.tensor(0.125)) and torch.isclose(big, torch.tensor(9.5))


def test_focal_reduces_to_bce_and_downweights_easy_examples():
    logits = torch.tensor([3.0, -3.0, 0.2, -0.1])
    y = torch.tensor([1.0, 0.0, 1.0, 0.0])
    bce = F.binary_cross_entropy_with_logits(logits, y)
    assert torch.isclose(focal_loss(logits, y, gamma=0.0, alpha=None), bce)
    assert focal_loss(logits, y, gamma=2.0, alpha=None) < bce


def test_weighted_bce_pos_weight():
    logits, y = torch.tensor([0.0]), torch.tensor([1.0])
    assert torch.isclose(weighted_bce(logits, y, pos_weight=3.0), 3 * weighted_bce(logits, y))
    assert pos_weight_from_rate(0.04) == pytest.approx(24.0)


def test_metrics_match_sklearn_and_respect_masks():
    rng = np.random.default_rng(0)
    y = rng.random(200) < 0.1
    s = rng.random(200) + y
    assert auroc(y, s) == pytest.approx(roc_auc_score(y, s))
    assert auprc(y, s) == pytest.approx(average_precision_score(y, s))
    assert mse([1, 2, np.nan], [1, 4, 0]) == pytest.approx(2.0)
    assert mae([1, 2, 3], [1, 4, 0], mask=[True, True, False]) == pytest.approx(1.0)


def test_early_stopping():
    es = EarlyStopping(patience=2, mode="min")
    assert [es.step(v) for v in (1.0, 0.8, 0.9, 0.85)] == [False, False, False, True]
    assert es.best == 0.8 and es.best_epoch == 1
    es = EarlyStopping(patience=1, mode="max")
    assert es.step(0.1) is False and es.step(0.05) is True


def test_seed_and_loop_reproducible():
    def run():
        seed_everything(7)
        model = torch.nn.Linear(3, 1)
        x, y = torch.randn(32, 3), torch.randn(32, 1)
        loader = [(x[i:i + 8], y[i:i + 8]) for i in range(0, 32, 8)]

        def step(m, batch, device):
            xb, yb = batch
            out = m(xb)
            return masked_mse(out, yb), out

        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        train_epoch(model, loader, step, opt, torch.device("cpu"))
        loss, outs = evaluate(model, loader, step, torch.device("cpu"))
        return loss, [p.detach().clone() for p in model.parameters()], len(outs)

    (l1, p1, n1), (l2, p2, n2) = run(), run()
    assert l1 == l2 and n1 == n2 == 4
    assert all(torch.equal(a, b) for a, b in zip(p1, p2))
