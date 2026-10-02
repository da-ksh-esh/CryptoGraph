"""Tests for Dashboard data loading and components."""

from pathlib import Path
import pytest
import pandas as pd

from dashboard.components.graph_view import load_rankings_data, load_correlation_matrix
from dashboard.components.spillover_heatmap import load_persistence_data, load_tempme_explanations
from dashboard.components.anomaly_alerts import load_tgae_anomalies
from dashboard.components.robustness_panel import load_robustness_data, load_manifest_data
from dashboard.components.live_prediction import get_dataset_metadata


def test_dashboard_data_loading():
    # 0. Live prediction dataset metadata
    timestamps, num_steps, horizons, assets = get_dataset_metadata()
    if num_steps > 0:
        assert len(timestamps) == num_steps
        assert len(assets) == 30
        assert len(horizons) >= 3
    # 1. Rankings data
    df_rankings = load_rankings_data()
    assert isinstance(df_rankings, pd.DataFrame)
    if not df_rankings.empty:
        assert "coin" in df_rankings.columns
        assert "dy_net_spillover" in df_rankings.columns
        assert "pagerank" in df_rankings.columns

    # 2. Correlation matrix
    assets, corr = load_correlation_matrix()
    if len(assets) > 0:
        assert corr.shape == (len(assets), len(assets))

    # 3. Persistence data
    df_persist = load_persistence_data()
    assert isinstance(df_persist, pd.DataFrame)
    if not df_persist.empty:
        assert "event_name" in df_persist.columns
        assert "blast_radius_pct" in df_persist.columns

    # 4. TempME explanations
    df_tempme = load_tempme_explanations()
    assert isinstance(df_tempme, pd.DataFrame)
    if not df_tempme.empty:
        assert "target_asset" in df_tempme.columns
        assert "fidelity_plus" in df_tempme.columns

    # 5. T-GAE anomalies
    df_tgae = load_tgae_anomalies()
    assert isinstance(df_tgae, pd.DataFrame)
    if not df_tgae.empty:
        assert "baseline_reconstruction_error" in df_tgae.columns
        assert "is_statistical_anomaly" in df_tgae.columns

    # 6. Robustness data and manifest
    df_rob = load_robustness_data()
    assert isinstance(df_rob, pd.DataFrame)
    if not df_rob.empty:
        assert "perturbation_type" in df_rob.columns
        assert "prediction_drift_mse" in df_rob.columns

    manifest = load_manifest_data()
    assert isinstance(manifest, dict)
    if manifest:
        assert "checkpoints" in manifest
        assert len(manifest["checkpoints"]) > 0
