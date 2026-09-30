# CryptoGraph-Secure 🛡️⚡

A cybersecurity-focused deep learning framework that analyzes how price shocks propagate across interconnected cryptocurrencies using **Temporal Graph Neural Networks (TGNN)** and evaluates model resilience against adversarial market manipulation.

---

## 🌟 Key Features

- **📊 Dynamic Relationship Graph**: Rolling dynamic correlations & learned adaptive adjacency matrices between cryptocurrencies.
- **⚡ Cryptocurrency Shock Detection**: Extreme Value Theory (EVT) and jump detection for flash crashes and sudden liquidity drains.
- **🔗 Cross-Asset Spillover Analysis**: Quantifies spillover magnitude, transmission delay (lead-lag steps), and decay persistence.
- **🧠 Temporal Graph Neural Network**: GATv2 spatial attention + Bi-directional GRU for multi-horizon price/return prediction.
- **🚨 Cybersecurity & Anomaly Detection**: Unsupervised Temporal VGAE to flag wash trading, oracle manipulation, and corrupted feeds.
- **🛡️ Adversarial Robustness Testing**: Benchmarks model degradation under feature (PGD/FGSM) and graph topology (DICE) attacks.
- **📱 Interactive Dashboard**: Real-time Streamlit interface with interactive network graphs and perturbation stress-testing.

---

## 📂 Project Structure

```text
CryptoGraph/
├── configs/
│   ├── data/daily.yaml       # data window, universe, quality, splits, targets, graph, scaling
│   ├── model/                # lstm, tgn, dygformer, cascade, tgae
│   ├── experiment/           # one YAML per training experiment
│   └── robustness/           # perturbation settings (optional stage)
├── data/                     # raw / interim / processed (gitignored) + curated external/events.csv
├── dashboard/                # Streamlit UI (deferred until models are trained)
├── docs/                     # model plan, merge notes
├── notebooks/                # 01 data & EDA, 03 Diebold-Yilmaz; archive/ holds superseded notebooks
├── reports/                  # audit, quality and baseline tables, figures
├── runs/                     # local experiment runs (gitignored)
├── src/cryptograph/
│   ├── data/                 # sources, quality, features, graphs, datasets, universe; legacy.py (old file format)
│   ├── network/              # centrality, communities, MST, topology, independent cascade (Stage 5)
│   ├── models/               # LSTM, TGN, T-GAE; DyGFormer and snapshot-GNN placeholders
│   ├── explain/              # edge-mask explainer, explanation validation (TempME adapter later)
│   ├── robustness/           # perturbation tests, FGSM (TODO), anomaly scoring
│   ├── training/             # seeding, losses, metrics, early stopping, loop
│   ├── analysis/             # statistics, Diebold-Yilmaz (generalized FEVD), event studies, baselines
│   ├── pipelines/            # build_data (cryptograph-data), train, evaluate, explain
│   └── common/               # logging, shared types
└── tests/unit/{data,models,explain,robustness,training,analysis,network}, tests/integration/
```

---

## 🚀 Quickstart

### 1. Install Dependencies
```bash
pip install -r requirements.txt
pip install -e .
```

### 2. Run Tests
```bash
pytest tests/
```

### 3. Launch Monitoring Dashboard
```bash
streamlit run dashboard/app.py
```

---

## 📦 Data

### Reproduce everything

```bash
pip install -r requirements.txt
pip install -e .
cryptograph-data --stage audit     # universe audit -> reports/universe_audit.csv (checkpoint)
cryptograph-data --stage all       # fetch -> clean -> build -> smoke test
pytest
jupyter nbconvert --to notebook --execute --inplace notebooks/01_market_dynamics_and_spillover_baseline.ipynb
```

Raw downloads are write-once snapshots under `data/raw/` and are reused on later runs. Pass `--refresh` to download them again. The date window, universe, thresholds, splits and seed all live in `configs/data/daily.yaml`. On Windows, `import cryptograph` (or `torch`) before `pandas`, because pyarrow ships an older `msvcp140.dll` that breaks torch when it loads first.

### Window and universe

- **Window:** 2016-09-28 → 2026-09-27, daily, UTC (3,652 days, fixed in config, never `period="10y"`).
- **Panel A:** balanced, 20 coins, 2017-11-09 → 2026-09-27. These are the 20 largest coins at the reference date with ≥98% coverage and median volume ≥ $1M. Used for the VAR/Diebold-Yilmaz baseline and long-run EDA. Only 13 coins meet the criterion from 2016-09-28, because Yahoo history starts on 2017-11-09 for everything except BTC and LTC. The later start was the documented checkpoint choice.
- **Panel B:** the main DL panel, 20 coins, unbalanced, with an `is_active` mask. These are the top non-stable coins by market cap that have ≥365 active days before the test split and median volume ≥ $1M. HYPE, CC and SUI were excluded because they have too little training history.
- **Event coins:** LUNC (Terra Classic) and FTT are kept in the master table, flagged, and used in event studies.
- Stablecoins, wrapped/staked tokens and tokenised real-world assets are excluded.

### Sources and quality

| Source | Use |
|---|---|
| Yahoo Finance (`yfinance`, `auto_adjust=False`, one ticker per request with retries) | primary OHLCV |
| [Coin Metrics Community Data](https://github.com/coinmetrics/data) (CC BY-NC 4.0, cite "Coin Metrics Community Network Data") | close validation, close-only backfill, market cap. `PriceUSD` aligns with Yahoo at lag 0; `ReferenceRateUSD` is shifted by one day. The archive currently ends on 2026-05-24. |
| CoinGecko | current market-cap ranking snapshot only (the free plan has no long history) |
| CoinDesk Data | optional tiebreaker, only used if `COINDESK_API_KEY` is set |

**Cross-source rule.** A coin-day whose closes differ by more than 2% is resolved as follows: (R1) take CoinDesk's side if a key is set; (R2) replace an isolated, reverting Yahoo spike with the Coin Metrics close; (R3) otherwise keep Yahoo and flag the day. A coin's Coin Metrics backfill is used only if its discrepancy rate is ≤5% and its median difference is ≤1%.

**Cleaning.** Gaps of 2 days or fewer are forward-filled and flagged. Longer gaps stay missing and inactive. Nothing is filled before a listing date, and returns are never clipped. Every check is counted per coin in `reports/data_quality_report.csv`, and each row carries a bitmask of flags (`cryptograph.data.schema.QC`).

### Artifacts (`data/processed/v1/`)

`panel.pt` (all coins: features, masks, targets), `lstm_{train,val,test}.pt`, `graph_snapshots.pt` (undirected and directed daily graphs, shock and contagion labels), `tgn_events.pt` (PyG `TemporalData`, split by date), `tgn_node_features.pt`, `splits.json`, `scalers.pkl`, `metadata.json` (config fingerprint, library versions, SHA-256 of every raw, interim and processed file), and `data_dictionary.md` (every array, shape and dtype). The legacy `data/processed/crypto_graph_dataset.pt` is still written in its original format.

**No-leakage guarantees** (enforced by tests):
- Features at day t use only data up to t.
- Targets are strictly after t.
- Scalers and the shock threshold are fitted on train dates only.
- There is a 30-day embargo before val and before test.
- Any sample whose input or target window crosses a split boundary is dropped.

### Limitations

- Yahoo crypto history before 2017-11-09 exists only for BTC and LTC. Earlier closes for other coins are Coin Metrics backfill with no OHLC or volume.
- Exchange-reported volume is noisy.
- Market caps come from Coin Metrics, which covers only about 20 coins and stops at its archive date.
- Choosing coins by today's ranking introduces survivorship bias.
- The sample is small by DL standards (about 3,650 days × 20 nodes).
