"""Unit tests for feature collinearity pruning and Parkinson protection."""

import numpy as np
import pandas as pd
import pytest
from cryptograph.data.features.pruning import CollinearityPruner


def test_collinearity_pruner_protects_parkinson():
    """Verify that when standard rolling volatility is collinear with Parkinson, Parkinson is kept."""
    np.random.seed(42)
    n_samples = 500
    base_signal = np.random.normal(0, 1, size=n_samples)

    # parkinson_volatility and a duplicate/collinear rolling volatility (rho > 0.95)
    parkinson = np.abs(base_signal)
    rolling_vol = parkinson + np.random.normal(0, 0.05, size=n_samples)
    other_feature = np.random.uniform(10, 50, size=n_samples)

    df = pd.DataFrame({
        "parkinson_volatility": parkinson,
        "volatility_24": rolling_vol,
        "other_feature": other_feature,
        "garman_klass": parkinson * 1.05  # disallowed feature
    })

    pruner = CollinearityPruner(
        correlation_threshold=0.90,
        protected_features=["parkinson_volatility"],
        disallowed_features=["garman_klass"]
    )
    pruner.fit(df)

    assert "parkinson_volatility" in pruner.kept_features_
    assert "volatility_24" not in pruner.kept_features_
    assert "garman_klass" not in pruner.kept_features_
    assert "other_feature" in pruner.kept_features_

    transformed = pruner.transform(df)
    assert list(transformed.columns) == ["parkinson_volatility", "other_feature"]
