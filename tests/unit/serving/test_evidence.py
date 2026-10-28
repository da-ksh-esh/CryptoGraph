import numpy as np
import torch

from cryptograph.serving.evidence import coin_hours, evidence_on_test


def test_coin_hours_counts_each_receiver_hour_once_at_its_highest_risk():
    prob = np.array([[0.2], [0.5], [0.1], [0.9]])
    risk = prob * 0.5
    y = np.array([[1], [1], [0], [0]], bool)
    mask = np.array([[1], [1], [1], [0]], bool)                      # the last event's outcome is unknown
    t, dst = np.array([4, 4, 4, 5]), np.array([1, 1, 2, 1])
    ch = coin_hours(prob, risk, y, mask, t, dst, 0)
    assert len(ch) == 2 and np.isclose(ch.loc[(4, 1), "risk"], 0.25) and ch.loc[(4, 1), "y"] == 1.0
    assert ch.loc[(4, 2), "y"] == 0.0


def test_evidence_on_a_run_with_test_predictions(tmp_path):
    rng = np.random.default_rng(0)
    run = tmp_path / "run"
    run.mkdir()
    n, K = 4000, [1, 6]
    s = rng.random(n)
    y1 = rng.random(n) < s * 0.6                                    # informative, nested horizons (6 h includes 1 h)
    y6 = y1 | (rng.random(n) < s * 0.4)
    common = {"prob": torch.from_numpy(np.stack([s, s], 1)).half(), "y": torch.from_numpy(np.stack([y1, y6], 1)),
              "mask": torch.ones(n, 2, dtype=torch.bool), "k_hours": K}
    torch.save(common, run / "predictions_val.pt")
    torch.save(common | {"t": torch.from_numpy(np.arange(n) // 2), "dst": torch.from_numpy(np.arange(n) % 2)},
               run / "predictions_test.pt")
    E = evidence_on_test(run, top_frac=0.05, bins=5)
    p = E["precision"].set_index("horizon_h")
    assert list(p.index) == K and (p["coin_hours"] == n).all() and E["calibrated"] and (p["top_n"] == 200).all()
    assert (p["top_precision"] > p["positive_rate"] + 0.1).all()
    c = E["calibration"]
    assert len(c) == 10 and (c.groupby("horizon_h")["n"].sum() == n).all()
    assert c.groupby("horizon_h")["predicted"].apply(lambda x: x.is_monotonic_increasing).all()
    assert np.isclose(c["predicted"], c["observed"], atol=0.08).all()   # calibrated on the same distribution
    assert evidence_on_test(tmp_path / "missing") == {}
