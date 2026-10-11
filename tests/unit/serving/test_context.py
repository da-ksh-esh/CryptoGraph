import numpy as np
import pandas as pd
import torch

from cryptograph.serving import dashboard_data as dd
from cryptograph.serving.context import coin_context


def _panel(names, T=60, N=2, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 0.01, (T, N, len(names))).astype(np.float32)
    if "rv_24h" in names:
        X[:, :, names.index("rv_24h")] = 0.01
        X[-1, 0, names.index("rv_24h")] = 0.04                 # coin 0 is 4x as jumpy as usual at the last hour
    if "volume_rel_168h" in names:
        X[-1, 0, names.index("volume_rel_168h")] = np.log(3.0)  # coin 0 trades at 3x its usual volume
    shock = np.zeros((T, N), bool)
    shock[-3:, 0] = True
    return {"feature_names": names, "X_raw": torch.from_numpy(X), "shock_today": torch.from_numpy(shock),
            "shock_today_mask": torch.ones(T, N, dtype=torch.bool), "active": torch.ones(T, N, dtype=torch.bool),
            "shock_z": torch.zeros(T, N)}


def test_coin_context_real_values_and_no_future_data():
    names = ["log_return", "rv_24h", "volume_rel_168h"]
    p = _panel(names)
    c = coin_context(p, [0, 1], 59, ["A", "B"])
    a = c.set_index("coin").loc["A"]
    assert np.isclose(a["volume_multiple"], 3.0) and np.isclose(a["vol_multiple"], 4.0)
    assert a["shocks_24h"] == 3 and bool(a["shock_now"])
    assert np.isnan(a["vol72_multiple"])                        # feature not in the panel -> NaN, no error
    p2 = _panel(names)
    p2["X_raw"][59:] += 5.0                                     # changing later hours must not change an earlier hour
    pd.testing.assert_frame_equal(coin_context(p, [0, 1], 40, ["A", "B"]), coin_context(p2, [0, 1], 40, ["A", "B"]))


def test_plain_driver_adds_real_values_when_known():
    ctx = coin_context(_panel(["log_return", "rv_24h", "volume_rel_168h"]), [0, 1], 59, ["A", "B"])
    d = {"input": "receiver coin: volume_rel_168h", "attribution": 0.3}
    assert dd.plain_driver(d, "A", "B", ctx) == "A: trading at 3.0× its usual volume (raises the risk)"
    d = {"input": "receiver coin: rv_24h", "attribution": -0.1}
    s = dd.plain_driver(d, "A", "B", ctx)
    assert "4.0× as jumpy as usual" in s and "lowers" in s
    d = {"input": "receiver coin: memory + recent events", "attribution": 0.1}
    assert "3 big moves in the last 24 hours" in dd.plain_driver(d, "A", "B", ctx)
    # without context (older runs) or for inputs without a natural number: the original wording
    d = {"input": "receiver coin: volume_rel_168h", "attribution": 0.3}
    assert dd.plain_driver(d, "A", "B") == "A: trading volume compared with its past week (raises the risk)"
    d = {"input": "source coin: network: dy_net", "attribution": 0.3}
    assert dd.plain_driver(d, "A", "B", ctx) == "B: whether it tends to spread or absorb price swings (raises the risk)"
