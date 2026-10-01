"""Feature pruning: collinearity elimination (rho >= 0.90) and Garman-Klass drop rule."""

from typing import List, Tuple, Dict, Optional, Set
import numpy as np
import pandas as pd
from ...common.logger import get_logger

logger = get_logger(__name__)


class CollinearityPruner:
    """
    Prunes redundant collinear features based on training set correlation.
    Enforces the project design requirement:
    - Keep Parkinson volatility.
    - Drop Garman-Klass volatility (if present).
    - Drop one feature from any pair with Pearson |rho| >= correlation_threshold (default 0.90).
    """

    def __init__(
        self,
        correlation_threshold: float = 0.90,
        protected_features: Optional[List[str]] = None,
        disallowed_features: Optional[List[str]] = None
    ):
        self.correlation_threshold = correlation_threshold
        self.protected_features = set(protected_features or ["parkinson_volatility", "parkinson_vol"])
        self.disallowed_features = set(disallowed_features or ["garman_klass", "garman_klass_volatility"])
        self.kept_features_: List[str] = []
        self.dropped_features_: List[Tuple[str, str, float]] = []
        self.correlation_matrix_: Optional[pd.DataFrame] = None

    def fit(self, feature_df: pd.DataFrame) -> "CollinearityPruner":
        """
        Fits pruning rules on training feature DataFrame.

        Args:
            feature_df: DataFrame of engineered features across training timesteps.
        """
        all_cols = list(feature_df.columns)
        candidate_cols = []
        self.dropped_features_ = []

        # 1. Drop disallowed features (e.g. Garman-Klass)
        for col in all_cols:
            if col in self.disallowed_features:
                self.dropped_features_.append((col, "disallowed_rule", 1.0))
                logger.info(f"Pruning rule: dropped disallowed feature '{col}'")
            else:
                candidate_cols.append(col)

        # 2. Compute correlation matrix on valid numerical values
        clean_df = feature_df[candidate_cols].replace([np.inf, -np.inf], np.nan).dropna()
        corr_matrix = clean_df.corr().abs()
        self.correlation_matrix_ = corr_matrix

        # 3. Identify pairs with correlation >= threshold
        dropped_set: Set[str] = set()
        
        # Sort features by overall average correlation (highest redundancy first)
        mean_corr = corr_matrix.mean()

        for i in range(len(candidate_cols)):
            f1 = candidate_cols[i]
            if f1 in dropped_set:
                continue

            for j in range(i + 1, len(candidate_cols)):
                f2 = candidate_cols[j]
                if f2 in dropped_set:
                    continue

                r_val = float(corr_matrix.loc[f1, f2])
                if r_val >= self.correlation_threshold:
                    # Decide which to drop
                    # If f1 is protected and f2 is not: drop f2
                    if f1 in self.protected_features and f2 not in self.protected_features:
                        to_drop, kept = f2, f1
                    elif f2 in self.protected_features and f1 not in self.protected_features:
                        to_drop, kept = f1, f2
                    else:
                        # Drop the one with higher average correlation to others
                        if mean_corr[f1] >= mean_corr[f2]:
                            to_drop, kept = f1, f2
                        else:
                            to_drop, kept = f2, f1

                    dropped_set.add(to_drop)
                    self.dropped_features_.append((to_drop, kept, r_val))
                    logger.info(
                        f"Collinearity prune: dropped '{to_drop}' (correlated with '{kept}', |rho| = {r_val:.4f} >= {self.correlation_threshold})"
                    )

        self.kept_features_ = [col for col in candidate_cols if col not in dropped_set]
        logger.info(f"Final kept features ({len(self.kept_features_)}): {self.kept_features_}")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Slices DataFrame to retain only kept features."""
        if not self.kept_features_:
            raise RuntimeError("CollinearityPruner must be fitted before calling transform.")
        return df[[col for col in self.kept_features_ if col in df.columns]]
