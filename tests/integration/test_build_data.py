"""Checks against the real pipeline artifacts; skipped when they have not been built."""

import json

import numpy as np
import pandas as pd
import pytest
import torch

from cryptograph.data.config import load_config
from cryptograph.data.schema import validate_master

cfg = load_config()
MASTER = cfg.interim_dir / "master.parquet"
PROC = cfg.processed_dir
pytestmark = [pytest.mark.integration,
              pytest.mark.skipif(not (MASTER.exists() and (PROC / "panel.pt").exists()), reason="artifacts not built")]


def test_master_schema_and_no_duplicates():
    m = pd.read_parquet(MASTER).reset_index()
    validate_master(m)  # duplicates, continuity, nothing before first valid, is_active == close.notna()
    assert not m.duplicated(["date", "coin"]).any()


def test_split_dates_respected():
    sp = json.loads((PROC / "splits.json").read_text())["splits"]
    for s in ("train", "val", "test"):
        d = torch.load(PROC / f"lstm_{s}.pt", weights_only=False)
        days = pd.DatetimeIndex(d["dates"])[d["day"].numpy()]
        assert days.min() >= pd.Timestamp(sp[s]["first_sample_date"])
        assert days.max() <= pd.Timestamp(sp[s]["last_sample_date"])
        assert torch.isfinite(d["X"]).all()
    assert pd.Timestamp(sp["val"]["first_sample_date"]) >= pd.Timestamp(sp["val"]["start"]) + pd.Timedelta(days=cfg.splits.embargo_days)


def test_scaler_fit_dates_are_train_only():
    import pickle

    with open(PROC / "scalers.pkl", "rb") as f:
        sc = pickle.load(f)["per_coin_and_global"]
    lo, hi = sc.fit_dates
    assert pd.Timestamp(hi) <= pd.Timestamp(cfg.splits.train[1]) and pd.Timestamp(lo) >= pd.Timestamp(cfg.window.start_date)


def test_event_stream_sorted_and_in_split():
    ev = torch.load(PROC / "tgn_events.pt", weights_only=False)
    t = ev["full"].t
    assert (t[1:] >= t[:-1]).all()
    dates = pd.DatetimeIndex(ev["dates"])
    for s in ("train", "val", "test"):
        lo, hi = (pd.Timestamp(x) for x in getattr(cfg.splits, s))
        d = dates[ev[s].t.numpy()]
        assert d.min() >= lo and d.max() <= hi


def test_btc_largest_drops_on_known_crash_dates():
    m = pd.read_parquet(MASTER)["close"].unstack("coin")["BTC"]
    r = np.log(m).diff()
    worst = r.nsmallest(3).index.strftime("%Y-%m-%d").tolist()
    assert "2020-03-12" in worst
