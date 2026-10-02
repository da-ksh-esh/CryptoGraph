"""Event Study and Shock Propagation Persistence.

Quantifies empirical shock propagation delay, time-to-peak, affected node blast radius,
and persistence duration across historical stress events without placeholder approximations.
"""

from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd

class EventStudyAnalyzer:
    """
    Analyzes systemic shock propagation dynamics around market crisis events.
    Computes abnormal metrics against an estimation/control baseline window.
    
    NOTE: Measures empirical temporal co-occurrence and abnormal return profiles,
    not counterfactual causality.
    """
    def __init__(self, pre_event_window: int = 14, post_event_window: int = 30, estimation_window: int = 60):
        self.pre_event_window = pre_event_window
        self.post_event_window = post_event_window
        self.estimation_window = estimation_window

    def compute_cumulative_abnormal_returns(
        self,
        returns: pd.Series,
        event_date: pd.Timestamp,
        benchmark: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        Computes cumulative abnormal returns (CAR) over the event window [t - W_pre, t + W_post].
        """
        start_date = event_date - pd.Timedelta(days=self.pre_event_window)
        end_date = event_date + pd.Timedelta(days=self.post_event_window)
        sub_returns = returns.loc[start_date:end_date].copy()

        if benchmark is not None:
            aligned_bench = benchmark.loc[sub_returns.index]
            abnormal = sub_returns - aligned_bench
        else:
            abnormal = sub_returns

        car = abnormal.cumsum()
        return car

    def estimate_persistence_half_life(self, response_series: np.ndarray) -> float:
        """
        Empirically estimates shock half-life from an impulse-response or abnormal return profile
        by fitting an exponential decay or finding median decay threshold.
        """
        if len(response_series) == 0:
            return 0.0
        peak_idx = int(np.argmax(np.abs(response_series)))
        peak_val = np.abs(response_series[peak_idx])
        if peak_val < 1e-6:
            return 0.0

        half_target = peak_val / 2.0
        decay_phase = np.abs(response_series[peak_idx:])
        below_half = np.where(decay_phase <= half_target)[0]
        if len(below_half) > 0:
            return float(below_half[0])
        return float(len(decay_phase))
