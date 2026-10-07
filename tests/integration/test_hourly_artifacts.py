"""Checks against the real hourly artifacts the models train on (data/processed/v2/1h); skipped when not built."""

import numpy as np
import pandas as pd
import pytest
import torch

from cryptograph.data.config import load_hourly_config
from cryptograph.data.storage import load_pt

cfg = load_hourly_config()
D = cfg.processed_dir("1h")
pytestmark = [pytest.mark.integration,
              pytest.mark.skipif(not ((D / "panel.pt").exists() and (D / "tgn_events.pt").exists()),
                                 reason="hourly artifacts not built")]


@pytest.fixture(scope="module")
def panel():
    return load_pt(D / "panel.pt")


@pytest.fixture(scope="module")
def events():
    return load_pt(D / "tgn_events.pt")


def test_sample_masks_stay_inside_their_split_with_embargo(panel):
    times = pd.DatetimeIndex(panel["times"])
    sid = panel["split_id"].numpy()
    k_max = max(panel["contagion_k_hours"])
    first, last = {}, {}
    for i, s in enumerate(("train", "val", "test")):
        hours = np.flatnonzero(panel["sample_mask"][s].numpy())
        assert len(hours) and (sid[hours] == i).all()                       # samples belong to their own split
        assert (sid[np.minimum(hours + k_max, len(sid) - 1)] == i).all()    # label windows stay inside it
        first[s], last[s] = times[hours.min()], times[hours.max()]
    assert first["val"] - last["train"] >= pd.Timedelta(days=cfg.splits.embargo_days)
    assert first["test"] - last["val"] >= pd.Timedelta(days=cfg.splits.embargo_days)


def test_event_stream_is_time_ordered_and_consistent(panel, events):
    d = events["full"]
    t = d.t.numpy()
    assert (np.diff(t) >= 0).all()                                          # time order
    assert (np.diff(panel["split_id"].numpy()[t]) >= 0).all()               # train, then val, then test
    assert (d.src != d.dst).all()
    assert torch.isfinite(d.msg).all()
    assert not (d.y & ~d.y_mask).any()                                      # no positive label where unobserved
    assert len(t) > 1_000_000 and list(events["label_k_hours"]) == list(panel["contagion_k_hours"])
    # every event is a shock of its source coin at its hour
    gi = events["graph_index"]
    sh = (panel["shock_today"] & panel["shock_today_mask"]).numpy()[:, gi]
    assert sh[t, d.src.numpy()].all()


def test_contagion_label_is_strictly_later(panel, events):
    d = events["full"]
    gi = events["graph_index"]
    sh = (panel["shock_today"] & panel["shock_today_mask"]).numpy()[:, gi]
    rng = np.random.default_rng(0)
    idx = rng.choice(len(d.t), size=2000, replace=False)
    t, j = d.t.numpy()[idx], d.dst.numpy()[idx]
    for k_i, k in enumerate(events["label_k_hours"]):
        obs = d.y_mask[idx, k_i].numpy()
        later = np.array([sh[tt + 1: tt + k + 1, jj].any() for tt, jj in zip(t, j)])
        assert (d.y[idx, k_i].numpy()[obs] == later[obs]).all()             # j shocks in (t, t+k], not at t


def test_shock_rate_on_train_matches_the_chosen_threshold(panel):
    sh, va = panel["shock_today"].numpy(), panel["shock_today_mask"].numpy()
    tr = panel["split_id"].numpy() == 0
    rate = sh[tr][va[tr]].mean()
    assert 0.03 <= rate <= 0.05                                             # z* chosen for ~4% on train
