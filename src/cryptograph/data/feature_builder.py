"""Legacy feature builder (kept for existing callers).

Deprecated for model data: it fills NaNs with 0, back-fills across launch dates
and clips returns. Use `cryptograph.data.features` (trailing-only features with
availability masks) via `cryptograph-data --stage build` instead.
"""

from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd
from ..common.logger import get_logger

logger = get_logger(__name__)

FEATURE_NAMES = [
    "log_return",
    "volatility_7",
    "volatility_14",
    "volatility_30",
    "hl_spread",
    "volume_ratio",
    "log_volume",
    "rsi_14",
    "macd"
]

class FeatureBuilder:
    """
    Computes stationary financial and technical features for cryptocurrency graph nodes.
    Extracts multi-horizon return targets and handles staggered coin launch dates.
    """

    def __init__(self, horizons: List[int] = [1, 3, 7, 14, 30]):
        self.horizons = horizons
        self.feature_names = FEATURE_NAMES

    def compute_single_asset_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes standardized technical features for a single cryptocurrency OHLCV DataFrame.
        """
        # Normalize column names to lowercase
        cols = {c: c.lower() for c in df.columns}
        d = df.rename(columns=cols)

        close = d["close"].replace(0.0, np.nan).ffill().bfill()
        high = d.get("high", close)
        low = d.get("low", close)
        volume = d.get("volume", pd.Series(1.0, index=close.index))

        # 1. Log Returns
        log_ret = np.log(close / close.shift(1))
        log_ret = log_ret.replace([np.inf, -np.inf], 0.0).fillna(0.0).clip(-2.0, 2.0)

        # 2. Realized Volatilities (multi-scale)
        vol_7 = log_ret.rolling(7, min_periods=2).std().fillna(0.0)
        vol_14 = log_ret.rolling(14, min_periods=2).std().fillna(0.0)
        vol_30 = log_ret.rolling(30, min_periods=2).std().fillna(0.0)

        # 3. High-Low Normalized Spread (Parkinson-type volatility proxy)
        hl_spread = ((high - low) / (close + 1e-8)).fillna(0.0)

        # 4. Volume Dynamics
        vol_ma14 = volume.rolling(14, min_periods=1).mean()
        volume_ratio = (volume / (vol_ma14 + 1e-8)).clip(upper=10.0).fillna(1.0)
        log_volume = np.log1p(volume.clip(lower=0.0)).fillna(0.0)

        # 5. RSI (14-day Relative Strength Index, scaled to [0, 1])
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14, min_periods=2).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=2).mean()
        rs = gain / (loss + 1e-8)
        rsi = (100.0 - (100.0 / (1.0 + rs))) / 100.0
        rsi = rsi.fillna(0.5)

        # 6. MACD Normalized
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ((ema12 - ema26) / (close + 1e-8)).fillna(0.0)

        feat_df = pd.DataFrame({
            "log_return": log_ret,
            "volatility_7": vol_7,
            "volatility_14": vol_14,
            "volatility_30": vol_30,
            "hl_spread": hl_spread,
            "volume_ratio": volume_ratio,
            "log_volume": log_volume,
            "rsi_14": rsi,
            "macd": macd
        }, index=df.index)

        return feat_df

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Alias for backward compatibility with single dataframe."""
        return self.compute_single_asset_features(df)

    def build_multi_asset_tensors(
        self,
        raw_df: pd.DataFrame,
        tickers: List[str]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex]:
        """
        Processes multi-asset MultiIndex DataFrame into node feature, target, and mask arrays.

        Args:
            raw_df: MultiIndex OHLCV DataFrame from yfinance.
            tickers: List of ticker strings.

        Returns:
            features: shape (T, num_nodes, num_features)
            targets: shape (T, num_nodes, len(horizons))
            active_mask: shape (T, num_nodes) boolean
            timestamps: DatetimeIndex
        """
        timestamps = raw_df.index
        num_timesteps = len(timestamps)
        num_nodes = len(tickers)
        num_features = len(self.feature_names)
        num_horizons = len(self.horizons)

        features = np.zeros((num_timesteps, num_nodes, num_features), dtype=np.float32)
        targets = np.zeros((num_timesteps, num_nodes, num_horizons), dtype=np.float32)
        active_mask = np.zeros((num_timesteps, num_nodes), dtype=bool)

        close_df = raw_df["Close"]

        for node_idx, ticker in enumerate(tickers):
            if ticker not in close_df.columns:
                logger.warning(f"Ticker {ticker} not found in Close prices.")
                continue

            # Check where this asset has valid non-null close prices
            valid_mask = close_df[ticker].notnull().values
            active_mask[:, node_idx] = valid_mask

            # Extract per-asset OHLCV sub-dataframe
            asset_dict = {}
            for metric in ["Open", "High", "Low", "Close", "Volume"]:
                if metric in raw_df.columns.levels[0] and ticker in raw_df[metric].columns:
                    asset_dict[metric.lower()] = raw_df[metric][ticker]

            asset_df = pd.DataFrame(asset_dict, index=timestamps)
            feat_df = self.compute_single_asset_features(asset_df)

            # Assign features
            features[:, node_idx, :] = feat_df[self.feature_names].values

            # Compute multi-horizon future returns as targets: ln(Close_{t+h} / Close_t)
            close_series = asset_df["close"].replace(0.0, np.nan).ffill().bfill()
            for h_idx, h in enumerate(self.horizons):
                future_ret = np.log(close_series.shift(-h) / close_series)
                future_ret = future_ret.replace([np.inf, -np.inf], 0.0).fillna(0.0).clip(-3.0, 3.0)
                targets[:, node_idx, h_idx] = future_ret.values

        logger.info(f"Engineered features shape: {features.shape}, targets shape: {targets.shape}")
        return features, targets, active_mask, timestamps
