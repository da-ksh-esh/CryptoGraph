# CryptoGraph-Secure 🛡️⚡

An empirical, cybersecurity-focused deep learning framework analyzing continuous-time cryptocurrency shock propagation, network spillovers, and anomaly resilience using **Temporal Graph Neural Networks (CryptoTGN)** and **Temporal Motif Explainers (TempME)**.

---

## 🌟 Key Research Capabilities

- **📊 High-Frequency 30-Coin Dynamic Graphs:** 24,024 hourly timesteps (2024–2026 Binance spot OHLCV) across a curated 30-coin universe with token migration splicing (RNDR $\to$ RENDER) and verified sector taxonomies.
- **🛡️ Strict Leakage Control & Purge Embargoes:** Chronological Train $\to$ 7-day embargo $\to$ Validation $\to$ 7-day embargo $\to$ Test splitting; 168-cell ($24 \times 7$) diurnal/weekend seasonality profiles and shock thresholds calibrated strictly on the training partition.
- **🔗 Econometric & Dynamic Network Science:**
  - **VAR-GFEVD:** Order-invariant Diebold-Yilmaz (2012) generalized forecast error variance decomposition.
  - **Directed Lead-Lag:** High-frequency cross-correlations $\tau \in [-24, +24]$h with Benjamini-Hochberg FDR-corrected Granger causality tests and shuffled permutation nulls.
  - **MST & Ultrametric Space:** Mantegna metric distance and Onnela normalized tree length $L(t)$ crisis contraction tracking.
  - **Community Detection:** Seeded Louvain modularity and Normalized Mutual Information (NMI) against external sector taxonomies.
- **🧠 Continuous-Time Dynamic Graph Learning (CryptoTGN):** Dynamic node memory modules evolving over timestamped shock and correlation event streams to forecast multi-horizon contagion ($k \in \{1, 3, 6, 12, 24\}$ hours).
- **🔍 Post-Hoc Explainability (TempME):** Temporal motif discovery extracting attributed shock transmission subgraphs, validated quantitatively via Fidelity+, Fidelity-, and Spearman rank agreement with econometric DY NET spillovers and PageRank centralities.
- **🚨 Unsupervised Anomaly Detection (T-GAE):** Spatio-temporal reconstruction error identifying abnormal market feeds and flash crash regimes without invented labels.

---

## 📂 Repository Structure

```text
CryptoGraph/
├── configs/             # YAML experiment, data, and model configurations
│   ├── data/            # daily.yaml (v1 reference), hourly.yaml (v2 1h/2h target)
│   ├── model/           # tgn.yaml, lstm.yaml, cascade.yaml, dygformer.yaml, tgae.yaml
│   └── experiment/      # Tracked experiment hyperparameter definitions
├── data/
│   ├── raw/             # Binance 1h/2h OHLCV archives (checksum-verified)
│   ├── processed/       # v1 reference (daily) & v2 target (1h, 2h) processed tensors
│   └── external/        # events.csv (crisis catalogue), sectors.csv (taxonomy)
├── notebooks/           # Structured empirical research notebooks (01–06)
│   ├── 01_data_preprocessing_and_eda.ipynb      # Part I (Daily) + Part II (Hourly EDA)
│   ├── 02_network_analysis.ipynb                # Dynamic networks, MST, SNA, & Cascade
│   ├── 03_models_vs_baselines.ipynb             # VAR, Cascade, LSTM, TGN, DyGFormer
│   ├── 04_tempme_explanations.ipynb             # TempME motif discovery & validation
│   ├── 05_persistence_by_shock_type.ipynb       # Event studies & propagation persistence
│   ├── 06_robustness_tgae.ipynb                 # T-GAE anomaly layer & perturbation curves
│   └── archive/                                 # Preserved legacy scaffold notebooks
├── src/cryptograph/     # Modular research engine
│   ├── data/            # sources, quality, features, targets, splits, dataloader
│   ├── network/         # centrality, communities, mst, topology, cascade, features
│   ├── models/          # tgn.py, lstm.py, dygformer.py, tgae.py, baselines/
│   ├── analysis/        # connectedness (VAR-GFEVD), lead_lag, epps_effect, event_study
│   ├── explain/         # TempME wrapper, edge attribution, fidelity validation
│   ├── robustness/      # anomaly scoring, adversarial perturbations (FGSM)
│   ├── training/        # loop, losses, metrics, seed, device, tracking
│   └── pipelines/       # data.py, train.py, evaluate.py, explain.py, package_data.py
├── reports/             # Verified audit CSVs, rankings, and baseline benchmarks
├── tests/               # Pytest automated verification suite (40 unit tests)
├── docs/                # model_plan.md, methodology specifications
├── colab/               # Headless cloud training runners (runner.ipynb)
└── dashboard/           # Streamlit monitoring dashboard (preserved)
```

---

## 🚀 Quickstart & Pipeline CLI

### 1. Installation
```bash
# Standard Python 3.12 environment
pip install -r requirements.txt
pip install -e .
```

### 2. Verify Test Suite
```bash
pytest
```

### 3. Build Processed Datasets
```bash
# Process 30-coin hourly and 2-hour datasets (v2)
cryptograph-data --config configs/data/hourly.yaml --stage all

# Run integrity and non-nullity smoke tests
cryptograph-data --stage smoke
```

### 4. Package Data for Remote Compute (Colab / Kaggle)
```bash
# Creates verified zip with SHA-256 manifest
cryptograph-package pack --version v2 --out data_v2.zip
```
