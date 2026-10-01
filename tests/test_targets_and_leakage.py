"""Unit tests for shock threshold calibration on train and multi-horizon target construction."""

import numpy as np
import pytest
from cryptograph.data.targets import ContagionTargetBuilder


def test_shock_threshold_fitted_on_train_only():
    """Verify modifying test partition leaves fitted shock thresholds identical."""
    np.random.seed(42)
    T = 1000
    N = 5
    signals = np.random.exponential(scale=0.02, size=(T, N)).astype(np.float32)

    train_mask = np.zeros(T, dtype=bool)
    train_mask[:600] = True

    builder1 = ContagionTargetBuilder(horizons=[1, 3, 6], shock_quantile=0.95)
    builder1.fit_thresholds(signals, train_mask)
    t1 = builder1.thresholds_.copy()

    # Mutate test partition with extreme spikes
    signals_mutated = signals.copy()
    signals_mutated[700:, :] = 100.0  # Massive spikes in test

    builder2 = ContagionTargetBuilder(horizons=[1, 3, 6], shock_quantile=0.95)
    builder2.fit_thresholds(signals_mutated, train_mask)
    t2 = builder2.thresholds_.copy()

    # Must be exactly identical
    assert np.allclose(t1, t2)


def test_target_validity_masks_and_horizons():
    """Verify target validity mask handles boundary conditions and active status."""
    T = 50
    N = 3
    close = np.ones((T, N), dtype=np.float32) * 100.0
    shocks = np.zeros((T, N), dtype=bool)
    shocks[10, 0] = True  # Shock at t=10 on node 0

    active_mask = np.ones((T, N), dtype=bool)
    active_mask[30:, 1] = False  # Node 1 delisted at t=30

    builder = ContagionTargetBuilder(horizons=[1, 5, 10], shock_quantile=0.95)
    builder.thresholds_ = np.ones(N) * 0.05
    builder.is_fitted = True

    shock_targets, return_targets, validity = builder.build_targets(close, shocks, active_mask)

    # For horizon h=10, last 10 timesteps must be invalid
    assert not validity[-10:, :, 2].any()

    # For node 1, after t=30 it must be invalid
    assert not validity[30:, 1, :].any()

    # Node 0 at t=9 for horizon 1 should have positive shock target (shock is at t=10)
    # horizons index 0 is h=1
    assert shock_targets[9, 0, 0] == True
    # At t=8 for h=1, shock at t=10 is not in [9, 9], so False
    assert shock_targets[8, 0, 0] == False
    # At t=8 for h=5, shock at t=10 is in [9, 13], so True
    assert shock_targets[8, 0, 1] == True
