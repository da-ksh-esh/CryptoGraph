"""Chronological splits with embargo and boundary-safe sample selection.

A sample dated t (last input day) with lookback L and max horizon H belongs to
split S only if
    S.start + embargo <= t          (embargo; train has none)
    t - L + 1 >= S.start            (input window inside S)
    t + H     <= S.end              (target window inside S)
so no input or target window crosses a split boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class SplitPlan:
    bounds: dict[str, tuple[pd.Timestamp, pd.Timestamp]]
    embargo_days: int
    lookback: int
    max_horizon: int

    @classmethod
    def from_config(cls, bounds: dict[str, tuple[date, date]], embargo_days: int, lookback: int, max_horizon: int
                    ) -> "SplitPlan":
        b = {k: (pd.Timestamp(v[0]), pd.Timestamp(v[1])) for k, v in bounds.items()}
        return cls(b, embargo_days, lookback, max_horizon)

    def split_of_date(self, dates: pd.DatetimeIndex) -> np.ndarray:
        """Split label of each calendar date ('' if outside all splits)."""
        out = np.full(len(dates), "", dtype=object)
        for name, (s, e) in self.bounds.items():
            out[(dates >= s) & (dates <= e)] = name
        return out

    def sample_mask(self, dates: pd.DatetimeIndex, split: str) -> np.ndarray:
        s, e = self.bounds[split]
        embargo = 0 if split == "train" else self.embargo_days
        first_ok = max(s + pd.Timedelta(days=embargo), s + pd.Timedelta(days=self.lookback - 1))
        last_ok = e - pd.Timedelta(days=self.max_horizon)
        return np.asarray((dates >= first_ok) & (dates <= last_ok))

    def to_json(self) -> dict:
        out = {"embargo_days": self.embargo_days, "lookback": self.lookback, "max_horizon": self.max_horizon,
               "rule": "S.start+embargo <= t, t-L+1 >= S.start, t+H_max <= S.end", "splits": {}}
        for name, (s, e) in self.bounds.items():
            embargo = 0 if name == "train" else self.embargo_days
            first_ok = max(s + pd.Timedelta(days=embargo), s + pd.Timedelta(days=self.lookback - 1))
            out["splits"][name] = {
                "start": s.date().isoformat(), "end": e.date().isoformat(),
                "embargo_end": (s + pd.Timedelta(days=embargo - 1)).date().isoformat() if embargo else None,
                "first_sample_date": first_ok.date().isoformat(),
                "last_sample_date": (e - pd.Timedelta(days=self.max_horizon)).date().isoformat(),
            }
        return out
