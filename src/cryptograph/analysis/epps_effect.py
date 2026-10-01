"""Epps effect analysis: quantification of correlation attenuation across sampling frequencies."""

from typing import List, Dict, Tuple, Any
import numpy as np
import pandas as pd
from ..common.logger import get_logger

logger = get_logger(__name__)

STANDARD_FREQUENCIES = ["1h", "2h", "4h", "12h", "1d"]


class EppsEffectAnalyzer:
    """
    Quantifies the Epps effect (Epps, 1979) in high-frequency cryptocurrency markets:
    The systematic drop in measured pairwise correlation as sampling interval decreases.
    """

    def __init__(self, frequencies: List[str] = STANDARD_FREQUENCIES):
        self.frequencies = frequencies

    def resample_returns(self, hourly_prices_df: pd.DataFrame, freq: str) -> pd.DataFrame:
        """
        Resamples hourly price series to target frequency and calculates log returns.
        """
        if freq == "1h":
            resampled_prices = hourly_prices_df
        else:
            # Pandas resample rule: '2h', '4h', '12h', '1d' (or '1D'/'24h')
            f_rule = "1D" if freq.lower() in ("1d", "daily") else freq.lower()
            resampled_prices = hourly_prices_df.resample(f_rule).last().dropna(how="all")

        ret = np.log(resampled_prices / resampled_prices.shift(1)).dropna(how="all")
        return ret

    def compute_epps_curve(
        self,
        hourly_prices_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Computes average pairwise correlation, median correlation, and standard deviation
        at each sampling frequency.
        """
        records = []

        for freq in self.frequencies:
            try:
                ret = self.resample_returns(hourly_prices_df, freq)
                # Compute pairwise correlation
                corr_mat = ret.corr().values
                # Extract off-diagonal values
                tril_indices = np.tril_indices_from(corr_mat, k=-1)
                corrs = corr_mat[tril_indices]
                valid_corrs = corrs[np.isfinite(corrs)]

                records.append({
                    "frequency": freq,
                    "num_timesteps": len(ret),
                    "mean_correlation": float(np.mean(valid_corrs)),
                    "median_correlation": float(np.median(valid_corrs)),
                    "std_correlation": float(np.std(valid_corrs)),
                    "min_correlation": float(np.min(valid_corrs)),
                    "max_correlation": float(np.max(valid_corrs))
                })
            except Exception as e:
                logger.warning(f"Failed to compute Epps correlation at frequency '{freq}': {e}")

        df_epps = pd.DataFrame(records)
        logger.info(f"Computed Epps curve across {len(records)} frequencies: mean corr at 1d={df_epps.iloc[-1]['mean_correlation']:.3f} vs 1h={df_epps.iloc[0]['mean_correlation']:.3f}")
        return df_epps
