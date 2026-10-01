"""Chronological data splitting with purge embargoes to prevent temporal leakage."""

from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional
import pandas as pd
import numpy as np
from ..common.logger import get_logger

logger = get_logger(__name__)

# Standard project split schedule:
# Train: 2024-01-01 -> 2025-08-31 (609 days)
# Embargo 1: 2025-09-01 -> 2025-09-07 (7 days)
# Val: 2025-09-08 -> 2026-03-15 (189 days)
# Embargo 2: 2026-03-16 -> 2026-03-22 (7 days)
# Test: 2026-03-23 -> 2026-09-27 (189 days)
DEFAULT_TRAIN_START = "2024-01-01 00:00:00"
DEFAULT_TRAIN_END = "2025-08-31 23:59:59"
DEFAULT_VAL_START = "2025-09-08 00:00:00"
DEFAULT_VAL_END = "2026-03-15 23:59:59"
DEFAULT_TEST_START = "2026-03-23 00:00:00"
DEFAULT_TEST_END = "2026-09-27 23:59:59"
DEFAULT_EMBARGO_DAYS = 7


@dataclass
class SplitIndices:
    train_mask: np.ndarray
    val_mask: np.ndarray
    test_mask: np.ndarray
    embargo_mask: np.ndarray
    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray


class TemporalSplitter:
    """
    Chronological train/val/test splitter enforcing strict purge embargoes.
    Zero overlap or lookahead between partitions.
    """

    def __init__(
        self,
        train_start: str = DEFAULT_TRAIN_START,
        train_end: str = DEFAULT_TRAIN_END,
        val_start: str = DEFAULT_VAL_START,
        val_end: str = DEFAULT_VAL_END,
        test_start: str = DEFAULT_TEST_START,
        test_end: str = DEFAULT_TEST_END,
        embargo_days: int = DEFAULT_EMBARGO_DAYS
    ):
        self.train_start = pd.Timestamp(train_start, tz="UTC")
        self.train_end = pd.Timestamp(train_end, tz="UTC")
        self.val_start = pd.Timestamp(val_start, tz="UTC")
        self.val_end = pd.Timestamp(val_end, tz="UTC")
        self.test_start = pd.Timestamp(test_start, tz="UTC")
        self.test_end = pd.Timestamp(test_end, tz="UTC")
        self.embargo_days = embargo_days

        # Sanity check chronological ordering
        assert self.train_start <= self.train_end, "train_start must be <= train_end"
        assert self.train_end < self.val_start, "train_end must precede val_start"
        assert self.val_start <= self.val_end, "val_start must be <= val_end"
        assert self.val_end < self.test_start, "val_end must precede test_start"
        assert self.test_start <= self.test_end, "test_start must be <= test_end"

        # Check embargo duration
        embargo_1_days = (self.val_start - self.train_end).total_seconds() / 86400.0
        embargo_2_days = (self.test_start - self.val_end).total_seconds() / 86400.0
        logger.info(
            f"Initialized TemporalSplitter: "
            f"Train [{self.train_start} to {self.train_end}], "
            f"Embargo 1 ({embargo_1_days:.1f} days), "
            f"Val [{self.val_start} to {self.val_end}], "
            f"Embargo 2 ({embargo_2_days:.1f} days), "
            f"Test [{self.test_start} to {self.test_end}]"
        )

    def split_timestamps(self, timestamps: pd.DatetimeIndex) -> SplitIndices:
        """
        Creates boolean masks and integer indices for timestamps.
        Timestamps must be UTC timezone-aware or convertible.
        """
        if timestamps.tz is None:
            ts = timestamps.tz_localize("UTC")
        else:
            ts = timestamps.tz_convert("UTC")

        train_mask = (ts >= self.train_start) & (ts <= self.train_end)
        val_mask = (ts >= self.val_start) & (ts <= self.val_end)
        test_mask = (ts >= self.test_start) & (ts <= self.test_end)

        # Embargo periods fall strictly between train/val or val/test
        embargo_1 = (ts > self.train_end) & (ts < self.val_start)
        embargo_2 = (ts > self.val_end) & (ts < self.test_start)
        embargo_mask = embargo_1 | embargo_2

        # Verify disjointness
        assert not np.any(train_mask & val_mask), "Train and Val masks must not overlap!"
        assert not np.any(val_mask & test_mask), "Val and Test masks must not overlap!"
        assert not np.any(train_mask & test_mask), "Train and Test masks must not overlap!"
        assert not np.any(train_mask & embargo_mask), "Train and Embargo masks must not overlap!"
        assert not np.any(val_mask & embargo_mask), "Val and Embargo masks must not overlap!"
        assert not np.any(test_mask & embargo_mask), "Test and Embargo masks must not overlap!"

        train_indices = np.where(train_mask)[0]
        val_indices = np.where(val_mask)[0]
        test_indices = np.where(test_mask)[0]

        logger.info(
            f"Split summary: Train={len(train_indices)}, Embargo={embargo_mask.sum()}, "
            f"Val={len(val_indices)}, Test={len(test_indices)}, Total={len(ts)}"
        )

        return SplitIndices(
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            embargo_mask=embargo_mask,
            train_indices=train_indices,
            val_indices=val_indices,
            test_indices=test_indices
        )
