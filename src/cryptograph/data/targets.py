"""Leakage-free shock detection, multi-horizon contagion targets, and target validity masks."""

from typing import List, Tuple, Dict, Optional, Union
import numpy as np
import pandas as pd
from ..common.logger import get_logger

logger = get_logger(__name__)

DEFAULT_HORIZONS = [1, 3, 6, 12, 24]


class ContagionTargetBuilder:
    """
    Constructs multi-horizon shock and return contagion targets.
    Calibrates shock thresholds strictly on training data to prevent look-ahead bias.
    """

    def __init__(
        self,
        horizons: List[int] = DEFAULT_HORIZONS,
        shock_quantile: float = 0.95,
        shock_metric: str = "volatility"  # 'volatility' or 'return_tail'
    ):
        self.horizons = sorted(horizons)
        self.shock_quantile = shock_quantile
        self.shock_metric = shock_metric
        self.thresholds_: Optional[np.ndarray] = None  # shape (num_nodes,)
        self.is_fitted = False

    def fit_thresholds(
        self,
        signal_matrix: np.ndarray,
        train_mask: np.ndarray
    ) -> "ContagionTargetBuilder":
        """
        Calibrates shock threshold per coin strictly on training timesteps.

        Args:
            signal_matrix: shape (T, num_nodes) array of volatility or absolute returns.
            train_mask: shape (T,) boolean mask indicating training timesteps.
        """
        assert signal_matrix.shape[0] == len(train_mask), "Signal matrix and train_mask length mismatch."
        num_nodes = signal_matrix.shape[1]
        thresholds = np.zeros(num_nodes, dtype=np.float32)

        train_signals = signal_matrix[train_mask]

        for node_idx in range(num_nodes):
            vals = train_signals[:, node_idx]
            valid_vals = vals[np.isfinite(vals) & (vals > 0)]
            if len(valid_vals) > 0:
                thresholds[node_idx] = float(np.percentile(valid_vals, self.shock_quantile * 100.0))
            else:
                thresholds[node_idx] = 0.05

        self.thresholds_ = thresholds
        self.is_fitted = True
        logger.info(
            f"Fitted shock thresholds on train partition ({self.shock_quantile*100:.0f}th percentile): "
            f"min={thresholds.min():.4f}, median={np.median(thresholds):.4f}, max={thresholds.max():.4f}"
        )
        return self

    def identify_shocks(self, signal_matrix: np.ndarray) -> np.ndarray:
        """
        Identifies shock events across all timesteps using the train-fitted thresholds.
        
        Returns:
            shocks: shape (T, num_nodes) boolean array.
        """
        if not self.is_fitted or self.thresholds_ is None:
            raise RuntimeError("ContagionTargetBuilder must be fitted on training data before identifying shocks.")

        shocks = signal_matrix > self.thresholds_[None, :]
        return shocks.astype(bool)

    def build_targets(
        self,
        close_prices: np.ndarray,
        shocks: np.ndarray,
        active_mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Constructs multi-horizon binary contagion shock targets and future return targets.

        Args:
            close_prices: shape (T, num_nodes) array of close prices.
            shocks: shape (T, num_nodes) boolean array of instantaneous shocks.
            active_mask: shape (T, num_nodes) boolean active coin mask.

        Returns:
            shock_targets: shape (T, num_nodes, num_horizons) boolean array.
                           True if node experiences at least one shock within [t+1, t+h].
            return_targets: shape (T, num_nodes, num_horizons) float array.
                            Cumulative log return ln(Close_{t+h} / Close_t).
            validity_mask: shape (T, num_nodes, num_horizons) boolean array.
                           False for future-truncated horizons or inactive periods.
        """
        T, num_nodes = close_prices.shape
        num_horizons = len(self.horizons)

        shock_targets = np.zeros((T, num_nodes, num_horizons), dtype=bool)
        return_targets = np.zeros((T, num_nodes, num_horizons), dtype=np.float32)
        validity_mask = np.zeros((T, num_nodes, num_horizons), dtype=bool)

        for h_idx, h in enumerate(self.horizons):
            for t in range(T):
                if t + h >= T:
                    # Beyond boundary: invalid future
                    continue

                for n in range(num_nodes):
                    if not active_mask[t, n] or not active_mask[t + h, n]:
                        continue

                    # Validity is True
                    validity_mask[t, n, h_idx] = True

                    # 1. Shock target: any shock in [t+1, t+h]
                    future_shocks = shocks[t + 1 : t + h + 1, n]
                    shock_targets[t, n, h_idx] = np.any(future_shocks)

                    # 2. Return target: ln(Close_{t+h} / Close_t)
                    c_now = close_prices[t, n]
                    c_future = close_prices[t + h, n]
                    if c_now > 1e-8 and c_future > 1e-8:
                        ret = np.log(c_future / c_now)
                        return_targets[t, n, h_idx] = np.clip(ret, -3.0, 3.0)

        logger.info(
            f"Built multi-horizon targets for horizons={self.horizons}: "
            f"shock_targets shape={shock_targets.shape}, positive rate={shock_targets[validity_mask].mean():.3%}"
        )
        return shock_targets, return_targets, validity_mask

    def compute_shock_events_count(
        self,
        shocks: np.ndarray,
        active_mask: np.ndarray,
        split_masks: Dict[str, np.ndarray],
        top_k_receivers: Optional[int] = None,
        correlation_matrix: Optional[np.ndarray] = None
    ) -> Dict[str, Dict[str, int]]:
        """
        Computes the number of shock occurrences and shock-triggered dynamic interaction events
        across each temporal split (Train, Val, Test).

        Args:
            shocks: shape (T, num_nodes) boolean array of shock indicators.
            active_mask: shape (T, num_nodes) boolean array of active nodes.
            split_masks: dictionary mapping split name ('train', 'val', 'test') to boolean mask (T,).
            top_k_receivers: if provided, source shock sends events only to top-K peers.
                             If None, broadcasts to all active peers (num_active - 1).
            correlation_matrix: optional static or trailing correlation matrix (num_nodes, num_nodes).

        Returns:
            Dictionary with statistics per split.
        """
        T, num_nodes = shocks.shape
        summary = {}

        for split_name, mask in split_masks.items():
            split_indices = np.where(mask)[0]
            split_len = len(split_indices)

            shocks_in_split = shocks[mask] & active_mask[mask]
            total_shocked_node_hours = int(np.sum(shocks_in_split))
            timesteps_with_shock = int(np.sum(np.any(shocks_in_split, axis=1)))

            # Count dynamic events
            total_broadcast_events = 0
            total_top5_events = 0
            total_top3_events = 0

            for t in split_indices:
                active_at_t = np.where(active_mask[t])[0]
                num_active = len(active_at_t)
                if num_active <= 1:
                    continue

                shocked_at_t = np.where(shocks[t] & active_mask[t])[0]
                num_shocked = len(shocked_at_t)
                if num_shocked == 0:
                    continue

                # Broadcast to all other active peers:
                total_broadcast_events += num_shocked * (num_active - 1)
                # Top-5 receivers:
                total_top5_events += num_shocked * min(5, num_active - 1)
                # Top-3 receivers:
                total_top3_events += num_shocked * min(3, num_active - 1)

            summary[split_name] = {
                "timesteps": split_len,
                "shocked_node_hours": total_shocked_node_hours,
                "timesteps_with_any_shock": timesteps_with_shock,
                "broadcast_events_all_peers": total_broadcast_events,
                "top5_receiver_events": total_top5_events,
                "top3_receiver_events": total_top3_events
            }

        return summary
