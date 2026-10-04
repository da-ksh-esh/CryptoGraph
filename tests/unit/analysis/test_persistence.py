from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from cryptograph.analysis.persistence import (
    ShockPersistenceAnalyzer,
    compute_excess_profile,
    estimate_decay_half_life,
)


def test_decay_half_life_synthetic_exponential():
    """Verify estimate_decay_half_life accurately recovers exponential decay half-lives."""
    h = np.arange(1, 73)  # hours 1..72
    # Known half-life of 4.0 hours: 2^(-(h-1)/4)
    # At h=1, value = 1.0; at h=5, value = 0.5. Since h=1 is index 0, half-life is 4.0 hours.
    gap_tau4 = 2.0 ** (-(h - 1) / 4.0)
    hl4 = estimate_decay_half_life(gap_tau4, interpolate=True)
    # The gap falls to 0.5 of gap[0] at h=5, i.e. 4 hours after h=1. Index 4 = hour 5.
    # The function measures hour index where value <= half_target (which occurs at index 4 = 5.0h)
    assert abs(hl4 - 5.0) < 0.05

    # Known half-life of 12.0 hours
    gap_tau12 = 2.0 ** (-(h - 1) / 12.0)
    hl12 = estimate_decay_half_life(gap_tau12, interpolate=True)
    assert abs(hl12 - 13.0) < 0.05

    # Edge cases
    assert estimate_decay_half_life(np.zeros(72)) == 0.0
    assert estimate_decay_half_life(np.ones(72) * 2.0) == 72.0  # never crosses half
    assert estimate_decay_half_life(np.array([])) == 0.0


def test_compute_excess_profile_excludes_shock_bar():
    """Verify compute_excess_profile strictly excludes shock bar t and uses baseline window."""
    T = 300
    r_ds = np.zeros(T)
    t_shock = 200

    # Fill pre-event baseline window [t-168 : t-24] with constant 2.0
    r_ds[t_shock - 168 : t_shock - 24] = 2.0
    # Shock bar t has a massive spike
    r_ds[t_shock] = 100.0
    # Post-shock hours [t+1 : t+72] have decay
    decay = np.linspace(8.0, 2.0, 72)
    r_ds[t_shock + 1 : t_shock + 73] = decay

    base, profile = compute_excess_profile(
        r_ds,
        t=t_shock,
        baseline_lookback_hours=168,
        baseline_lead_hours=24,
        horizon_hours=72,
    )

    # Baseline should be median over [t-168:t-24] = 2.0
    assert abs(base - 2.0) < 1e-6
    # Profile length should be 72
    assert len(profile) == 72
    # Shock bar t=100.0 MUST NOT be in profile
    assert 100.0 / 2.0 not in profile
    # First post-shock bar is profile[0] = decay[0] / base = 8.0 / 2.0 = 4.0
    assert abs(profile[0] - 4.0) < 1e-6
    # Last post-shock bar is profile[71] = decay[71] / base = 2.0 / 2.0 = 1.0
    assert abs(profile[71] - 1.0) < 1e-6


def test_causal_invariance_future_data():
    """
    Changing data after a shock window (t > t_shock + 72) must have ZERO effect
    on the estimates, baselines, and extracted profiles for earlier shocks.
    """
    rng = np.random.default_rng(123)
    T = 600
    r_ds = rng.normal(0, 1, size=T)
    t_shock = 200

    base_orig, prof_orig = compute_excess_profile(
        r_ds,
        t=t_shock,
        baseline_lookback_hours=168,
        baseline_lead_hours=24,
        horizon_hours=72,
    )

    # Mutate data far in the future: t in [300, 600]
    r_ds_mutated = r_ds.copy()
    r_ds_mutated[t_shock + 73 :] = rng.normal(50, 10, size=T - (t_shock + 73))

    base_mut, prof_mut = compute_excess_profile(
        r_ds_mutated,
        t=t_shock,
        baseline_lookback_hours=168,
        baseline_lead_hours=24,
        horizon_hours=72,
    )

    assert base_orig == base_mut
    np.testing.assert_allclose(prof_orig, prof_mut, rtol=1e-12, atol=1e-12)


def test_sector_filtering_min_coins():
    """Verify sectors with fewer than min_coins are excluded."""
    # Synthetic panel
    coins = ["BTC", "ETH", "SOL", "DOGE", "SHIB", "PEPE", "BNB"]
    model_coins = coins
    sectors_df = pd.DataFrame({
        "coin": coins,
        "sector": [
            "payments", "smart-contract platform", "smart-contract platform",
            "meme", "meme", "meme",
            "exchange token"  # only 1 coin
        ]
    })

    T = 300
    panel = {
        "coins": coins,
        "model_coins": model_coins,
        "r_ds": torch.randn(T, len(coins)),
        "shock_today": torch.zeros(T, len(coins), dtype=torch.bool),
        "shock_today_mask": torch.ones(T, len(coins), dtype=torch.bool),
        "active": torch.ones(T, len(coins), dtype=torch.bool),
        "times": pd.date_range("2024-01-01", periods=T, freq="h").strftime("%Y-%m-%d %H:%M:%S").tolist(),
        "split_id": torch.zeros(T, dtype=torch.int8),
    }

    analyzer = ShockPersistenceAnalyzer(horizon_hours=24, baseline_lookback_hours=48, baseline_lead_hours=12)
    eligible_sectors, _ = analyzer.extract_sector_episodes(
        panel, sectors_df, min_coins_per_sector=3
    )

    # "meme" has 3 coins -> eligible
    assert "meme" in eligible_sectors
    # "payments" has 1 coin -> excluded
    assert "payments" not in eligible_sectors
    # "exchange token" has 1 coin -> excluded
    assert "exchange token" not in eligible_sectors
    # "smart-contract platform" has 2 coins -> excluded
    assert "smart-contract platform" not in eligible_sectors


def test_shock_excess_beats_matched_null():
    """Verify that simulated elevated shock volatility significantly beats matched non-shock null."""
    rng = np.random.default_rng(42)
    T = 1000
    N = 3
    coins = ["COIN1", "COIN2", "COIN3"]
    sectors_df = pd.DataFrame({"coin": coins, "sector": ["test_sec"] * N})

    # Returns: background noise ~ N(0, 1)
    r_ds = rng.normal(0, 1, size=(T, N)).astype(np.float32)
    shocks = np.zeros((T, N), dtype=bool)

    # Inject periodic shocks at intervals of 80 hours
    shock_times = list(range(200, 800, 80))
    for t in shock_times:
        for i in range(N):
            shocks[t, i] = True
            # Elevated volatility decaying over 24h
            for h in range(1, 48):
                decay = np.exp(-h / 6.0)
                r_ds[t + h, i] += rng.normal(0, 4.0 * decay)

    # Build synthetic panel
    times = pd.date_range("2024-01-01", periods=T, freq="h").strftime("%Y-%m-%d %H:%M:%S").tolist()
    panel = {
        "coins": coins,
        "model_coins": coins,
        "r_ds": torch.tensor(r_ds),
        "shock_today": torch.tensor(shocks),
        "shock_today_mask": torch.ones(T, N, dtype=torch.bool),
        "active": torch.ones(T, N, dtype=torch.bool),
        "times": times,
        "split_id": torch.zeros(T, dtype=torch.int8),
    }

    analyzer = ShockPersistenceAnalyzer(
        horizon_hours=48,
        baseline_lookback_hours=168,
        baseline_lead_hours=24,
        min_cluster_spacing_hours=24,
    )
    eligible_sectors, sec_data = analyzer.extract_sector_episodes(
        panel, sectors_df, min_coins_per_sector=3, seed=42
    )

    assert "test_sec" in eligible_sectors
    d = sec_data["test_sec"]
    assert len(d["t"]) > 0

    mean_sh = np.mean(d["sh"], axis=0)
    mean_nu = np.mean(d["nu"], axis=0)
    gap = mean_sh - mean_nu

    # Post-shock volatility should substantially exceed matched null at h=1
    assert gap[0] > 1.0
    # Half-life should be finite and positive
    hl = estimate_decay_half_life(gap)
    assert 1.0 <= hl <= 48.0
