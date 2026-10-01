import pytest
import numpy as np
import pandas as pd
from cryptograph.analysis.persistence import ShockPersistenceAnalyzer

def test_half_life_estimation():
    analyzer = ShockPersistenceAnalyzer()
    # Exponential decay: 100, 50, 25, 12.5...
    decay_curve = np.array([10.0, 5.0, 2.5, 1.25, 0.625])
    hl = analyzer.estimate_half_life(decay_curve)
    assert hl == 1.0  # Drops to 50% at index 1

def test_propagation_delay():
    analyzer = ShockPersistenceAnalyzer()
    T, N = 10, 4
    shocks = np.zeros((T, N), dtype=bool)
    # Trigger at t=2
    # Secondary shocks peak at t=5 (delay = 3 hours)
    shocks[5, 1] = True
    shocks[5, 2] = True
    shocks[5, 3] = True
    delay = analyzer.compute_propagation_delay(shocks, trigger_t=2, max_lookahead=8)
    assert delay == 3.0

def test_event_propagation_summary():
    analyzer = ShockPersistenceAnalyzer()
    T, N, F = 20, 3, 2
    features = np.ones((T, N, F), dtype=np.float32)
    shocks = np.zeros((T, N), dtype=bool)
    shocks[2, 0] = True # trigger
    shocks[3, 1] = True # contagion
    act_mask = np.ones((T, N), dtype=bool)
    assets = ["BTC", "ETH", "SOL"]
    sec_map = {"BTC": "Layer-1", "ETH": "Layer-1", "SOL": "Layer-1"}

    rec = analyzer.analyze_event_propagation(
        event_name="ToyEvent",
        event_timestamp="2026-01-01 02:00:00",
        trigger_t=2,
        features=features,
        shocks=shocks,
        active_mask=act_mask,
        asset_names=assets,
        sectors_map=sec_map,
        lookahead_hours=5
    )
    assert rec["blast_radius_count"] >= 1
    assert rec["event_name"] == "ToyEvent"
    assert rec["most_impacted_sector"] == "Layer-1"
