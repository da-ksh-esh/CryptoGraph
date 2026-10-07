import numpy as np
import pandas as pd

from cryptograph.analysis.var_spillover import VARSpilloverBaseline


def test_var_spillover_baseline():
    baseline = VARSpilloverBaseline(maxlags=2, forecast_horizon=5)
    np.random.seed(42)
    sample_returns = pd.DataFrame(
        np.random.randn(50, 4) * 0.02,
        columns=["BTC", "ETH", "SOL", "BNB"]
    )
    baseline.fit(sample_returns)
    table = baseline.compute_spillover_table()
    assert "total_spillover_index" in table
    assert table["total_spillover_index"] >= 0.0


def test_directional_table_is_the_spillover_matrix():
    """Compatibility alias read by notebooks/archive/02_statistical_spillover_baseline.ipynb."""
    np.random.seed(0)
    baseline = VARSpilloverBaseline(maxlags=1, forecast_horizon=5)
    baseline.fit(pd.DataFrame(np.random.randn(80, 3) * 0.02, columns=["BTC", "ETH", "SOL"]))
    table = baseline.compute_spillover_table()
    pd.testing.assert_frame_equal(table["directional_table"], table["spillover_matrix"])
    assert list(table["directional_table"].columns) == ["BTC", "ETH", "SOL", "FROM"]
