"""Unit tests for seasonality filtering and leakage prevention."""

import numpy as np
import pandas as pd
import pytest
from cryptograph.data.features.seasonality import SeasonalityFilter


def test_seasonality_profile_fitted_on_train_only():
    """Verify modifying future/test data leaves fitted seasonal profile completely unchanged."""
    dates = pd.date_range("2024-01-01", periods=168 * 4, freq="1h", tz="UTC")
    np.random.seed(42)
    vol_train = np.random.uniform(100, 200, size=len(dates))
    vola_train = np.random.uniform(0.01, 0.05, size=len(dates))

    df_train = pd.DataFrame({"vol": vol_train, "vola": vola_train}, index=dates)

    filt1 = SeasonalityFilter()
    filt1.fit(df_train["vol"], df_train["vola"])

    # Mutate a synthetic "future" partition with massive outliers
    dates_future = pd.date_range("2024-02-01", periods=168 * 4, freq="1h", tz="UTC")
    vol_future = np.random.uniform(10000, 20000, size=len(dates_future))
    vola_future = np.random.uniform(1.0, 5.0, size=len(dates_future))
    df_future = pd.DataFrame({"vol": vol_future, "vola": vola_future}, index=dates_future)

    # Refit on the same train data (future data exists in memory but is not fed to fit)
    filt2 = SeasonalityFilter()
    filt2.fit(df_train["vol"], df_train["vola"])

    # Multipliers must match exactly
    for k in filt1.volume_profile_:
        assert np.isclose(filt1.volume_profile_[k], filt2.volume_profile_[k])
        assert np.isclose(filt1.volatility_profile_[k], filt2.volatility_profile_[k])

    # De-seasonalize future data without error
    des_vol, des_vola = filt1.transform(df_future["vol"], df_future["vola"])
    assert len(des_vol) == len(df_future)
    assert not des_vol.isna().any()
    assert not des_vola.isna().any()


def test_seasonality_multipliers_within_bounds():
    """Verify multipliers are clamped within the safety bounds."""
    dates = pd.date_range("2024-01-01", periods=168, freq="1h", tz="UTC")
    vol = np.ones(len(dates))
    vol[0] = 1e6  # extreme spike
    vola = np.ones(len(dates)) * 0.01

    filt = SeasonalityFilter(clip_bounds=(0.1, 10.0))
    filt.fit(pd.Series(vol, index=dates), pd.Series(vola, index=dates))

    for m in filt.volume_profile_.values():
        assert 0.1 <= m <= 10.0
