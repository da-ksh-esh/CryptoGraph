import pytest
import torch

from cryptograph.models.lstm import CryptoLSTM
from cryptograph.models.tgae import TemporalGAE
from cryptograph.models.tgn import CryptoTGN


def test_crypto_tgn():
    num_nodes = 5
    in_features = 8
    model = CryptoTGN(num_nodes=num_nodes, raw_msg_dim=in_features, memory_dim=16, embedding_dim=16, out_horizons=3)
    x = torch.randn(num_nodes, in_features)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)

    preds, z = model(x, edge_index)
    assert preds.shape == (num_nodes, 3)
    assert z.shape == (num_nodes, 16)


def test_crypto_lstm():
    lstm = CryptoLSTM(in_features=8, hidden_dim=16, out_horizons=3)
    dummy_x = torch.randn(5, 10, 8)  # 5 nodes/batches, 10 timesteps, 8 features
    out = lstm(dummy_x)
    assert out.shape == (5, 3)


def test_temporal_gae():
    gae = TemporalGAE(in_features=8, hidden_dim=16, latent_dim=8)
    x = torch.randn(5, 8)
    x_rec, adj_rec, z = gae(x)
    assert x_rec.shape == (5, 8)
    assert adj_rec.shape == (5, 5)
