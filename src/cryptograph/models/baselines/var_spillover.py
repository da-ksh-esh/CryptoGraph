"""Statistical baseline: Vector Autoregression (VAR) & Diebold-Yilmaz Spillover Index."""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from ...common.logger import get_logger

logger = get_logger(__name__)

class VARSpilloverBaseline:
    """
    Econometric baseline computing VAR dynamics and the Diebold-Yilmaz (2012)
    Spillover Index using Generalized Forecast Error Variance Decomposition (GFEVD).
    """

    def __init__(self, maxlags: int = 5, forecast_horizon: int = 10):
        self.maxlags = maxlags
        self.forecast_horizon = forecast_horizon
        self.fitted_model = None
        self.asset_names = []

    def fit(self, returns_df: pd.DataFrame) -> "VARSpilloverBaseline":
        """
        Fits a VAR model on stationary multi-asset return series.
        """
        self.asset_names = list(returns_df.columns)
        clean_df = returns_df.dropna()
        var_model = VAR(clean_df)
        try:
            fitted = var_model.fit(maxlags=self.maxlags, ic="aic")
            if fitted.k_ar == 0:
                fitted = var_model.fit(1)
        except Exception:
            fitted = var_model.fit(1)
        self.fitted_model = fitted
        logger.info(f"Fitted VAR baseline with lag order: {self.fitted_model.k_ar}")
        return self

    def compute_spillover_table(self) -> Dict[str, Any]:
        """
        Calculates the Diebold-Yilmaz directional spillover matrix,
        TO others, FROM others, NET spillover, and Total Spillover Index (TSI).
        """
        if self.fitted_model is None:
            raise RuntimeError("Model must be fitted before computing spillover table.")

        fevd = self.fitted_model.fevd(self.forecast_horizon)
        # fevd.decomp has shape (num_variables, forecast_horizon, num_variables)
        # We take the horizon step:
        decomp_h = fevd.decomp[:, self.forecast_horizon - 1, :]
        # Normalize rows to sum to 100%
        row_sums = decomp_h.sum(axis=1, keepdims=True)
        norm_matrix = (decomp_h / row_sums) * 100.0

        n = len(self.asset_names)
        # Directional spillovers
        # Off-diagonal elements
        off_diag = norm_matrix.copy()
        np.fill_diagonal(off_diag, 0.0)

        # Spillover TO others (column sum of off-diagonal)
        to_others = off_diag.sum(axis=0)
        # Spillover FROM others (row sum of off-diagonal)
        from_others = off_diag.sum(axis=1)
        # Net spillover
        net_spillover = to_others - from_others

        # Total Spillover Index (TSI)
        total_spillover_index = float(off_diag.sum() / n)

        spillover_df = pd.DataFrame(norm_matrix, index=self.asset_names, columns=self.asset_names)
        spillover_df["FROM"] = from_others

        return {
            "spillover_matrix": spillover_df,
            "to_others": pd.Series(to_others, index=self.asset_names),
            "from_others": pd.Series(from_others, index=self.asset_names),
            "net_spillover": pd.Series(net_spillover, index=self.asset_names),
            "total_spillover_index": total_spillover_index
        }

    def get_connectedness_ranking(self) -> pd.Series:
        """
        Returns rank order of assets by NET systemic spillover (most influential transmitter to receiver).
        """
        results = self.compute_spillover_table()
        return results["net_spillover"].sort_values(ascending=False)
