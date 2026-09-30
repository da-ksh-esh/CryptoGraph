# Archive: Legacy Notebooks & Baseline Audits

This directory preserves earlier prototype and exploratory notebooks for audit trail, reproducibility, and version history.

## Archived Notebooks

### 1. `01_market_dynamics_and_spillover_baseline.ipynb`
* **Original Scope:**
  - Initial exploratory data analysis (EDA) across the daily 20-coin universe.
  - Correlation heatmaps, rolling volatility plots, and static network visualizations.
  - Initial Diebold-Yilmaz Vector Autoregression (VAR) spillover baseline and total spillover index (TSI).
* **Identified Methodological & Architectural Limitations:**
  - **Input / Feature Returns:** Returns were derived directly from intermediate model inputs rather than raw master prices, introducing subtle scaling/pre-filtering dependencies.
  - **Data Quality Inconsistencies:** The quality audit tables had conflicting launch-date masks and inactive coin filtering logic.
  - **Look-Ahead Bias in Shocks:** Standardized shock scores ($z$-scores) computed rolling statistics that included the contemporary day $t$, introducing backward leakage.
  - **Contagion vs. Co-occurrence:** Simple contemporary co-occurrence of shocks across assets was labeled as dynamic "contagion" without temporal directional lag validation.
  - **Fractional / Random Splits:** Earlier train/val/test splits used naive fractional indexing without chronological embargoes, risking temporal information leakage across split boundaries.
  - **Econometric Formulation:** Cholesky orthogonalization numbers were reported under a "GFEVD" (Generalized Forecast Error Variance Decomposition) label. Cholesky decompositions are order-dependent and sensitive to coin ordering.
  - **API Inconsistency:** References to `spillover_results["directional_table"]` were present in the notebook while the underlying repository class only provided `spillover_matrix`.

---

## Where Corrected Versions Live

The research-grade implementations resolving each of the above points have been decoupled and moved into dedicated modules:

1. **Data Preprocessing, Quality Audit, & Leakage-Free EDA:**
   - [`notebooks/01_data_preprocessing_and_eda.ipynb`](../01_data_preprocessing_and_eda.ipynb)
   - Features: raw price returns from `master.parquet`, 30-day chronological embargo, ADF stationarity, fat-tail verification, volatility clustering, and PyTorch Geometric tensor validation.
2. **Order-Invariant Generalized FEVD Econometric Baseline:**
   - [`notebooks/02_statistical_spillover_baseline.ipynb`](../02_statistical_spillover_baseline.ipynb)
   - Source: [`src/cryptograph/models/baselines/var_spillover.py`](../../src/cryptograph/models/baselines/var_spillover.py)
   - Features: exact Pesaran-Shin (1998) / Diebold-Yilmaz (2012) invariant generalized FEVD, verified order-invariant under column permutations, exporting clean TO/FROM directional tables.
3. **Dynamic Graph Neural Network & Explainability:**
   - [`notebooks/03_tgn_training_and_tempme.ipynb`](../03_tgn_training_and_tempme.ipynb)
   - Features: Temporal Graph Network (`CryptoTGN`) with snapshot-based memory state management and dual-space sensitivity explanations (`TempME`).
