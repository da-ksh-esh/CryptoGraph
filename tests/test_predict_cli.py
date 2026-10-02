"""Tests for Phase 21: Standalone Contagion Prediction CLI & Engine."""

import pytest
import torch
from pathlib import Path

from cryptograph.pipelines.predict import predict_contagion, resolve_timestep
from cryptograph.data.universe import TARGET_UNIVERSE_30


def test_resolve_timestep():
    timestamps = [
        "2024-01-01 00:00:00+00:00",
        "2024-01-01 01:00:00+00:00",
        "2024-01-01 02:00:00+00:00",
        "2024-08-05 00:00:00+00:00",
        "2024-08-05 12:00:00+00:00",
    ]
    # Test positive integer
    assert resolve_timestep(timestamps, 2) == 2
    # Test negative integer
    assert resolve_timestep(timestamps, -1) == 4
    # Test date query match
    assert resolve_timestep(timestamps, "2024-08-05") == 4
    assert resolve_timestep(timestamps, "2024-01-01 01:00:00") == 1


def test_predict_contagion_tgn():
    dataset_path = "data/processed/v2/1h/processed_dataset.pt"
    if not Path(dataset_path).exists():
        pytest.skip("Dataset file not available.")

    res = predict_contagion(
        model_type="tgn",
        dataset_path=dataset_path,
        timestep=5208,
        horizon_hours=1,
        threshold=0.50,
        top_k=5,
        explain=False
    )

    assert "metadata" in res
    assert "predictions" in res
    assert res["metadata"]["model_type"] == "TGN"
    assert res["metadata"]["timestep_index"] == 5208
    assert res["metadata"]["total_assets_evaluated"] == len(TARGET_UNIVERSE_30)
    assert len(res["predictions"]) == 5

    # Check top prediction schema
    top_asset = res["predictions"][0]
    assert "asset" in top_asset
    assert "sector" in top_asset
    assert "shock_probability" in top_asset
    assert "status" in top_asset
    assert "persistence_halflife_hours" in top_asset
    assert 0.0 <= top_asset["shock_probability"] <= 1.0


def test_predict_contagion_lstm():
    dataset_path = "data/processed/v2/1h/processed_dataset.pt"
    if not Path(dataset_path).exists():
        pytest.skip("Dataset file not available.")

    res = predict_contagion(
        model_type="lstm",
        dataset_path=dataset_path,
        timestep=5208,
        horizon_hours=1,
        threshold=0.50,
        top_k=3,
        explain=False
    )

    assert "metadata" in res
    assert res["metadata"]["model_type"] == "LSTM"
    assert len(res["predictions"]) == 3
    assert res["predictions"][0]["rank"] == 1
