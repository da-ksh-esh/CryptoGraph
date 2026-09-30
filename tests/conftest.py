"""Shared fixtures. Importing torch before pandas avoids a Windows DLL clash (pyarrow's msvcp140)."""

import torch  # noqa: F401  (must precede pandas/pyarrow on Windows)

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def synthetic_panel(rng):
    """3 coins, 400 days; coin C lists on day 100 and has a 5-day gap on days 250-254."""
    dates = pd.date_range("2020-01-01", periods=400, freq="D", name="date")
    r = rng.standard_t(4, size=(400, 3)) * 0.03
    r[:, 1] += 0.6 * r[:, 0]
    close = pd.DataFrame(100 * np.exp(np.cumsum(r, axis=0)), index=dates, columns=["A", "B", "C"])
    close.iloc[:100, 2] = np.nan
    close.iloc[250:255, 2] = np.nan
    return close
