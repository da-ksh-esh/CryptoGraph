import numpy as np
import torch

from cryptograph.models.tgae import SnapshotTGAE
from cryptograph.training.tgae_task import DAILY_FEATURES, daily_node_features


def test_snapshot_tgae_step_shapes_and_losses():
    torch.manual_seed(0)
    m = SnapshotTGAE(in_features=5, hidden_dim=8, latent_dim=4)
    x = torch.randn(6, 5)
    ei = torch.tensor([[0, 1, 2, 1], [1, 0, 1, 2]])
    w = torch.ones(4)
    h0 = torch.zeros(6, 4)
    for predict in (False, True):
        h, adj, xr = m.step(x, ei, w, h0, predict=predict)
        assert h.shape == (6, 4) and adj.shape == (6, 6) and xr.shape == (6, 5)
    # prediction mode decodes from the state *before* the snapshot: with h0 = 0 every edge logit is 0
    _, adj_p, _ = m.step(x, ei, w, h0, predict=True)
    assert torch.allclose(adj_p, torch.zeros(6, 6))
    A = torch.zeros(6, 6)
    A[0, 1] = A[1, 0] = 1
    mask = torch.tensor([True, True, True, True, False, False])
    la, lf = SnapshotTGAE.losses(adj, A, xr, x, mask)
    assert torch.isfinite(la) and torch.isfinite(lf) and la > 0


def test_daily_features_use_only_the_day_window():
    T, N = 72, 3
    rng = np.random.default_rng(0)
    z = rng.normal(size=(T, N)).astype(np.float32)
    panel = {"shock_z": torch.from_numpy(z), "shock_today": torch.from_numpy(np.abs(z) > 2),
             "shock_today_mask": torch.ones(T, N, dtype=torch.bool), "feature_names": ["volume_rel_168h", "rv_24h"],
             "X": torch.from_numpy(rng.normal(size=(T, N, 2)).astype(np.float32))}
    st = np.array([23, 47, 71])
    f = daily_node_features(panel, [0, 1, 2], st, np.array([True, True, False]))
    assert f.shape == (3, N, len(DAILY_FEATURES))
    # changing hours after a snapshot day must not change that day's features
    z2 = z.copy()
    z2[48:] += 5
    panel2 = dict(panel, shock_z=torch.from_numpy(z2))
    f2 = daily_node_features(panel2, [0, 1, 2], st, np.array([True, True, False]))
    torch.testing.assert_close(f[:2], f2[:2])
