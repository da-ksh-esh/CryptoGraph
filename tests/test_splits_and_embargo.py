"""Unit tests for chronological splits and 7-day purge embargoes."""

import pandas as pd
import numpy as np
import pytest
from cryptograph.data.splits import TemporalSplitter


def test_split_disjointness_and_embargo():
    """Verify that train, embargo, val, and test splits are mutually exclusive and have proper gaps."""
    timestamps = pd.date_range("2024-01-01", "2026-09-27 23:00:00", freq="1h", tz="UTC")
    splitter = TemporalSplitter(
        train_start="2024-01-01 00:00:00",
        train_end="2025-08-31 23:59:59",
        val_start="2025-09-08 00:00:00",
        val_end="2026-03-15 23:59:59",
        test_start="2026-03-23 00:00:00",
        test_end="2026-09-27 23:59:59",
        embargo_days=7
    )

    splits = splitter.split_timestamps(timestamps)

    # 1. Total coverage
    assert len(timestamps) == 24024
    assert splits.train_mask.sum() == 14616
    assert splits.val_mask.sum() == 4536
    assert splits.test_mask.sum() == 4536
    # Embargo 1: Sep 1 to Sep 7 (7 days = 168h), Embargo 2: Mar 16 to Mar 22 (7 days = 168h) -> 336h
    assert splits.embargo_mask.sum() == 336

    # 2. Mutually exclusive
    assert not np.any(splits.train_mask & splits.val_mask)
    assert not np.any(splits.val_mask & splits.test_mask)
    assert not np.any(splits.train_mask & splits.test_mask)
    assert not np.any(splits.train_mask & splits.embargo_mask)
    assert not np.any(splits.val_mask & splits.embargo_mask)
    assert not np.any(splits.test_mask & splits.embargo_mask)

    # 3. Exactly 7 days gap
    max_train_ts = timestamps[splits.train_indices[-1]]
    min_val_ts = timestamps[splits.val_indices[0]]
    gap_hours_1 = (min_val_ts - max_train_ts).total_seconds() / 3600.0
    assert gap_hours_1 >= 7 * 24

    max_val_ts = timestamps[splits.val_indices[-1]]
    min_test_ts = timestamps[splits.test_indices[0]]
    gap_hours_2 = (min_test_ts - max_val_ts).total_seconds() / 3600.0
    assert gap_hours_2 >= 7 * 24
