"""Seasonality filtering: fits hour-of-day x day-of-week profiles on train data only."""

from typing import Dict, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
from ...common.logger import get_logger

logger = get_logger(__name__)


class SeasonalityFilter:
    """
    Fits a 24x7 (hour-of-day x day-of-week) multiplicative profile strictly on training data
    to de-seasonalize trading volume and realized volatility.
    
    Ensures zero leakage across temporal splits.
    """

    def __init__(self, clip_bounds: Tuple[float, float] = (0.1, 10.0)):
        self.clip_bounds = clip_bounds
        self.is_fitted = False
        # Profile maps (dayofweek, hour) -> multiplier
        self.volume_profile_: Dict[Tuple[int, int], float] = {}
        self.volatility_profile_: Dict[Tuple[int, int], float] = {}
        self.global_volume_mean_: float = 1.0
        self.global_volatility_mean_: float = 1.0

    def fit(
        self,
        volume: Union[pd.Series, pd.DataFrame, np.ndarray],
        volatility: Union[pd.Series, pd.DataFrame, np.ndarray],
        timestamps: Optional[pd.DatetimeIndex] = None
    ) -> "SeasonalityFilter":
        """
        Fits 168-cell seasonal profiles strictly on the training partition.

        Args:
            volume: Training volume series, DataFrame, or 1D/2D array.
            volatility: Training volatility series, DataFrame, or 1D/2D array.
            timestamps: DatetimeIndex corresponding to the training rows if inputs are arrays.
        """
        if isinstance(volume, (pd.Series, pd.DataFrame)):
            ts = volume.index
        elif timestamps is not None:
            ts = timestamps
        else:
            raise ValueError("Timestamps must be provided if volume is not a pandas object with DatetimeIndex.")

        if ts.tz is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")

        # Convert to 1D series (pooling across assets if 2D)
        vol_arr = np.asarray(volume, dtype=np.float64)
        vola_arr = np.asarray(volatility, dtype=np.float64)

        hours = ts.hour.values
        days = ts.dayofweek.values

        # If 2D (T, N), tile timestamps across assets
        if vol_arr.ndim == 2:
            T, N = vol_arr.shape
            hours_tiled = np.repeat(hours, N)
            days_tiled = np.repeat(days, N)
            vol_flat = vol_arr.flatten()
            vola_flat = vola_arr.flatten()
        else:
            hours_tiled = hours
            days_tiled = days
            vol_flat = vol_arr
            vola_flat = vola_arr

        # Filter out NaN/infinite/negative values
        valid_vol_mask = np.isfinite(vol_flat) & (vol_flat > 0)
        valid_vola_mask = np.isfinite(vola_flat) & (vola_flat > 0)

        global_vol_mean = float(np.mean(vol_flat[valid_vol_mask])) if np.any(valid_vol_mask) else 1.0
        global_vola_mean = float(np.mean(vola_flat[valid_vola_mask])) if np.any(valid_vola_mask) else 1.0

        self.global_volume_mean_ = max(global_vol_mean, 1e-8)
        self.global_volatility_mean_ = max(global_vola_mean, 1e-8)

        df_grid = pd.DataFrame({
            "day": days_tiled,
            "hour": hours_tiled,
            "vol": vol_flat,
            "vola": vola_flat
        })

        # Compute cell means
        cell_agg = df_grid.groupby(["day", "hour"]).agg(
            vol_mean=("vol", "mean"),
            vola_mean=("vola", "mean")
        )

        min_c, max_c = self.clip_bounds
        for d in range(7):
            for h in range(24):
                if (d, h) in cell_agg.index:
                    v_cell = cell_agg.loc[(d, h), "vol_mean"]
                    va_cell = cell_agg.loc[(d, h), "vola_mean"]

                    s_vol = float(v_cell / self.global_volume_mean_) if np.isfinite(v_cell) and v_cell > 0 else 1.0
                    s_vola = float(va_cell / self.global_volatility_mean_) if np.isfinite(va_cell) and va_cell > 0 else 1.0
                else:
                    s_vol = 1.0
                    s_vola = 1.0

                self.volume_profile_[(d, h)] = float(np.clip(s_vol, min_c, max_c))
                self.volatility_profile_[(d, h)] = float(np.clip(s_vola, min_c, max_c))

        self.is_fitted = True
        logger.info(
            f"Fitted SeasonalityFilter: 168 cells. "
            f"Volume multipliers in [{min(self.volume_profile_.values()):.3f}, {max(self.volume_profile_.values()):.3f}], "
            f"Volatility multipliers in [{min(self.volatility_profile_.values()):.3f}, {max(self.volatility_profile_.values()):.3f}]"
        )
        return self

    def transform(
        self,
        volume: Union[pd.Series, pd.DataFrame, np.ndarray],
        volatility: Union[pd.Series, pd.DataFrame, np.ndarray],
        timestamps: Optional[pd.DatetimeIndex] = None
    ) -> Tuple[Any, Any]:
        """
        De-seasonalizes volume and volatility using the fitted profile.
        
        Returns:
            (volume_deseasonalized, volatility_deseasonalized) in matching types.
        """
        if not self.is_fitted:
            raise RuntimeError("SeasonalityFilter must be fitted on training data before calling transform.")

        if isinstance(volume, (pd.Series, pd.DataFrame)):
            ts = volume.index
        elif timestamps is not None:
            ts = timestamps
        else:
            raise ValueError("Timestamps must be provided if volume is not a pandas object with DatetimeIndex.")

        if ts.tz is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")

        hours = ts.hour.values
        days = ts.dayofweek.values

        # Build 1D multipliers for the timesteps
        vol_factors = np.array([self.volume_profile_.get((d, h), 1.0) for d, h in zip(days, hours)], dtype=np.float32)
        vola_factors = np.array([self.volatility_profile_.get((d, h), 1.0) for d, h in zip(days, hours)], dtype=np.float32)

        if isinstance(volume, pd.DataFrame):
            deseas_vol = volume.div(vol_factors, axis=0)
            deseas_vola = volatility.div(vola_factors, axis=0) if isinstance(volatility, pd.DataFrame) else volatility / vol_factors
            return deseas_vol, deseas_vola
        elif isinstance(volume, pd.Series):
            deseas_vol = volume / vol_factors
            deseas_vola = volatility / vola_factors
            return deseas_vol, deseas_vola
        else:
            vol_arr = np.asarray(volume, dtype=np.float32)
            vola_arr = np.asarray(volatility, dtype=np.float32)
            if vol_arr.ndim == 2:
                deseas_vol = vol_arr / vol_factors[:, None]
                deseas_vola = vola_arr / vola_factors[:, None]
            else:
                deseas_vol = vol_arr / vol_factors
                deseas_vola = vola_arr / vola_factors
            return deseas_vol, deseas_vola
