"""Resolution-aware feature engineering for multi-asset cryptocurrency dynamic graphs."""

from typing import List, Tuple, Dict, Optional, Union
import numpy as np
import pandas as pd
from ...common.logger import get_logger
from .seasonality import SeasonalityFilter
from .pruning import CollinearityPruner

logger = get_logger(__name__)


def get_resolution_windows(resolution: str) -> Tuple[int, int, int]:
    """
    Returns short, medium, and long rolling windows in bars for the given resolution.
    Targeting ~1 day, ~3 days, ~7 days.
    """
    res = resolution.lower()
    if res in ("1h", "h", "hourly"):
        return (24, 72, 168)
    elif res in ("2h", "2-hour"):
        return (12, 36, 84)
    elif res in ("1d", "d", "daily"):
        return (7, 14, 30)
    else:
        logger.warning(f"Unrecognized resolution '{resolution}', defaulting to hourly windows [24, 72, 168]")
        return (24, 72, 168)


class ResolutionAwareFeatureBuilder:
    """
    Computes stationary financial and technical features adapted to observation resolution (1h, 2h, 1d).
    Integrates Parkinson volatility, seasonality filtering, and train-calibrated scaling.
    """

    def __init__(
        self,
        resolution: str = "1h",
        use_seasonality: bool = True,
        clip_returns: float = 2.0,
        rsi_window: int = 14
    ):
        self.resolution = resolution
        self.use_seasonality = use_seasonality
        self.clip_returns = clip_returns
        self.rsi_window = rsi_window
        self.w_short, self.w_med, self.w_long = get_resolution_windows(resolution)

        self.seasonality_filter_: Optional[SeasonalityFilter] = None
        self.collinearity_pruner_: Optional[CollinearityPruner] = None
        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None
        self.feature_names_: List[str] = []

    def compute_single_asset_features(
        self,
        df: pd.DataFrame,
        seasonality_filter: Optional[SeasonalityFilter] = None
    ) -> pd.DataFrame:
        """
        Computes standardized technical features for a single coin OHLCV DataFrame.
        """
        cols = {c: c.lower() for c in df.columns}
        d = df.rename(columns=cols)

        close = d["close"].replace(0.0, np.nan).ffill().bfill()
        high = d.get("high", close)
        low = d.get("low", close)
        volume = d.get("volume", pd.Series(1.0, index=close.index))

        # 1. Log returns
        log_ret = np.log(close / close.shift(1))
        log_ret = log_ret.replace([np.inf, -np.inf], 0.0).fillna(0.0).clip(-self.clip_returns, self.clip_returns)

        # 2. Parkinson Volatility
        # Parkinson (1980): sigma_p^2 = ln(H/L)^2 / (4 * ln(2))
        hl_ratio = (high / (low + 1e-8)).clip(lower=1.0)
        parkinson_bar = (np.log(hl_ratio) ** 2) / (4.0 * np.log(2.0))
        parkinson_vol = np.sqrt(parkinson_bar.rolling(self.w_short, min_periods=2).mean()).fillna(0.0)

        # 3. Realized Volatilities (multi-scale rolling return std)
        vol_short = log_ret.rolling(self.w_short, min_periods=2).std().fillna(0.0)
        vol_med = log_ret.rolling(self.w_med, min_periods=2).std().fillna(0.0)
        vol_long = log_ret.rolling(self.w_long, min_periods=2).std().fillna(0.0)

        # 4. High-Low Spread (relative range)
        hl_spread = ((high - low) / (close + 1e-8)).fillna(0.0)

        # 5. Volume dynamics & optional seasonality
        if seasonality_filter is not None and seasonality_filter.is_fitted:
            deseas_vol, _ = seasonality_filter.transform(volume, parkinson_vol, timestamps=df.index)
        else:
            deseas_vol = volume

        log_volume = np.log1p(volume.clip(lower=0.0)).fillna(0.0)
        vol_ma = volume.rolling(self.w_short, min_periods=1).mean()
        volume_ratio = (volume / (vol_ma + 1e-8)).clip(upper=10.0).fillna(1.0)

        deseas_vol_ma = deseas_vol.rolling(self.w_short, min_periods=1).mean()
        deseas_vol_ratio = (deseas_vol / (deseas_vol_ma + 1e-8)).clip(upper=10.0).fillna(1.0)

        # 6. RSI (scaled to [0, 1])
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(self.rsi_window, min_periods=2).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(self.rsi_window, min_periods=2).mean()
        rs = gain / (loss + 1e-8)
        rsi = (100.0 - (100.0 / (1.0 + rs))) / 100.0
        rsi = rsi.fillna(0.5)

        # 7. MACD Normalized
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ((ema12 - ema26) / (close + 1e-8)).fillna(0.0)

        raw_feat_dict = {
            "log_return": log_ret,
            "parkinson_volatility": parkinson_vol,
            f"volatility_{self.w_short}": vol_short,
            f"volatility_{self.w_med}": vol_med,
            f"volatility_{self.w_long}": vol_long,
            "hl_spread": hl_spread,
            "log_volume": log_volume,
            "volume_ratio": volume_ratio,
            "volume_deseasonalized_ratio": deseas_vol_ratio,
            "rsi_14": rsi,
            "macd": macd
        }

        feat_df = pd.DataFrame(raw_feat_dict, index=df.index)
        return feat_df

    def fit_scaler(self, train_features: np.ndarray, feature_names: List[str]):
        """
        Fits robust standard scaling (median and IQR or mean and std) strictly on train data.
        
        Args:
            train_features: array of shape (N_train, num_nodes, num_features) or (N_samples, num_features)
            feature_names: names of features
        """
        self.feature_names_ = list(feature_names)
        if train_features.ndim == 3:
            flat_train = train_features.reshape(-1, train_features.shape[-1])
        else:
            flat_train = train_features

        # Filter out NaN/inf
        valid = np.isfinite(flat_train)
        means = np.zeros(flat_train.shape[1], dtype=np.float32)
        stds = np.ones(flat_train.shape[1], dtype=np.float32)

        for col_idx in range(flat_train.shape[1]):
            col_v = flat_train[valid[:, col_idx], col_idx]
            if len(col_v) > 0:
                m = float(np.mean(col_v))
                s = float(np.std(col_v))
                means[col_idx] = m
                stds[col_idx] = s if s > 1e-6 else 1.0

        self.mean_ = means
        self.scale_ = stds
        logger.info(f"Fitted FeatureScaler on {flat_train.shape[0]} samples across {len(feature_names)} features.")

    def transform_scaler(self, features: np.ndarray) -> np.ndarray:
        """
        Standardizes features using train-fitted mean and scale.
        """
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("FeatureScaler must be fitted on train features before calling transform_scaler.")

        orig_shape = features.shape
        if features.ndim == 3:
            flat = features.reshape(-1, orig_shape[-1])
            scaled = (flat - self.mean_) / self.scale_
            return scaled.reshape(orig_shape).astype(np.float32)
        else:
            scaled = (features - self.mean_) / self.scale_
            return scaled.astype(np.float32)
