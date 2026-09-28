import pytest
import pandas as pd
from cryptograph.spillover.shock_detector import ShockDetector

def test_shock_detector():
    detector = ShockDetector(z_threshold=2.5)
    base_returns = [0.001, -0.002, 0.001, -0.001, 0.002] * 5  # 25 points
    jump_returns = base_returns + [0.20]  # 26th point is extreme shock
    returns = pd.Series(jump_returns, index=pd.date_range("2026-01-01", periods=len(jump_returns), freq="h"))
    shocks = detector.detect_shocks(returns, "BTC")
    assert len(shocks) >= 1
    assert shocks[0].source_asset == "BTC"
    assert shocks[0].shock_type == "positive_jump"

