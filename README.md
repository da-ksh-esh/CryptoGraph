# CryptoGraph-Secure 🛡️⚡

A deep learning framework for how price shocks spread across cryptocurrencies: hourly market data for 30 coins,
network analysis of the coin market, and temporal graph neural networks (TGN, DyGFormer) that predict which coins
shock next, with motif-based explanations (TempME) and robustness tests.

> **Confidential** — proprietary, all rights reserved (pending IP review). Do not share code, data or results outside the team.

---

## 🌟 Status

| Part | Status |
|---|---|
| Daily data (20 coins, 2016–2026; Yahoo + Coin Metrics) | done — `data/processed/v1/` |
| Hourly data (30 coins, 2020–2026; Binance, 1h + 2h) | done — `data/processed/v2/` |
| EDA (notebook 01) and network analysis (notebook 02) | done |
| Independent cascade baseline, Diebold–Yilmaz rankings, network node features | done |
| Run tracking, data packaging, Colab runner | done (smoke-tested) |
| Models: TGN (main), DyGFormer, LSTM, gradient boosting, ensemble; 5 seeds, test scored once (notebook 03) | done |
| Explanations: TempME ported to the TGN, integrated gradients, case studies (notebook 04) | done |
| Shock persistence by type (notebook 05) | done |
| Robustness: T-GAE anomaly scores, noise / missing-data / FGSM stress tests (notebook 06) | done |
| Serving: `cryptograph-predict` (calibrated risk per coin + integrated-gradient drivers), Streamlit dashboard, TGN compression for CPU | done: final runs on `tgn_1h_final_v2` seed 0; serve fp32 (int8 is 3x smaller but slower and costs ~0.005 AUPRC; `reports/compression/`, `docs/merge_notes.md`) |

---

## 📂 Project Structure

```text
CryptoGraph/
├── configs/
│   ├── data/daily.yaml, hourly.yaml   # windows, universe, quality, features, labels, splits, graphs, network
│   ├── model/                # architecture of lstm, tgn, dygformer (training settings in experiment/)
│   └── experiment/           # one YAML per run: *_dev (validation only), *_final (test), tempme, tgae, robustness,
│                             #   predict_tgn_1h, compress_tgn_1h
├── colab/                    # self-contained Colab notebooks + runner.ipynb + README (Colab / Kaggle)
├── data/                     # raw / interim / processed (gitignored) + curated external/events.csv, sectors.csv
├── dashboard/                # Streamlit UI over cryptograph-predict runs (risk, alerts + drivers, duration, network)
├── docs/                     # model_plan.md, hourly_data_notes.md, merge_notes.md
├── notebooks/                # 01 data & EDA, 02 networks, 03 models, 04 explanations, 05 persistence, 06 robustness
├── reports/                  # audit, quality, baseline, network, explain/ and robustness/ tables, figures
├── runs/                     # local experiment runs + results.csv (gitignored)
├── src/cryptograph/
│   ├── data/                 # sources, quality, features, graphs, datasets, universe (daily + hourly)
│   ├── network/              # centrality, communities, MST, topology, spillover, independent cascade, node features
│   ├── models/               # LSTM, TGN, DyGFormer, T-GAE (SnapshotTGAE); snapshot GNN not built
│   ├── explain/              # TempME port, TGN capture, fidelity baselines, integrated gradients
│   ├── robustness/           # stress tests (noise, missing events, FGSM / PGD), driver stability, anomaly scoring
│   ├── serving/              # predict (risk + drivers), compress (int8 / pruning vs fp32), dashboard data
│   ├── training/             # runner + tasks (TGN, DyGFormer, LSTM, boosting, T-GAE), refit, tracking, metrics
│   ├── analysis/             # statistics, lead-lag, Diebold-Yilmaz, event studies, persistence, evaluation, baselines
│   ├── pipelines/            # build_data (cryptograph-data), train (cryptograph-train), predict (cryptograph-predict),
│   │                         #   package_data, evaluate, explain
│   └── common/               # logging, shared types
└── tests/unit/{data,models,explain,robustness,training,analysis,network,serving}, tests/integration/
```

---

## 🚀 Quickstart

```bash
pip install -r requirements-lock.txt   # exact versions used for the results (requirements.txt = minimum versions)
pip install -e .
pytest
cryptograph-data --stage all                                        # daily data (needed by the hourly clean stage)
cryptograph-data --config configs/data/hourly.yaml --stage all      # audit -> fetch -> clean -> build -> network -> baselines
cryptograph-train --config configs/experiment/tgn_1h_final_v2.yaml --seed 0   # one final TGN run -> runs/<run_id>/
```

---

## 📦 Daily data (`data/processed/v1/`)

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

## ⏱️ Hourly data (`data/processed/v2/`) — the modelling dataset

```bash
cryptograph-data --stage all                                        # daily data first (hourly clean cross-checks it)
cryptograph-data --config configs/data/hourly.yaml --stage audit    # 30-coin universe (checkpoint)
cryptograph-data --config configs/data/hourly.yaml --stage fetch    # Binance archive, SHA-256 verified
cryptograph-data --config configs/data/hourly.yaml --stage clean    # quality, 2h bars, USDT peg, cross-check
cryptograph-data --config configs/data/hourly.yaml --stage build    # features, labels, splits, graphs, datasets
cryptograph-data --config configs/data/hourly.yaml --stage network  # network features, rankings, cascade baseline
cryptograph-data --config configs/data/hourly.yaml --stage all      # = audit, fetch, clean, build, network, baselines
```

- **Source:** Binance public archive (data.binance.vision), 1h klines, 2020-01-01 → 2026-09-27, prices in USDT;
  2h bars resampled from 1h. 30 model coins (≥ 2 years of hourly data, ≥ 90% test coverage) + LUNC, FTT as event coins.
  Pitfalls (ticker changes, LUNA/LUNC, delistings, maintenance gaps, USDT peg) are in `docs/hourly_data_notes.md`.
- **Per resolution** (`1h/`, `2h/`): `panel.pt` (features, masks, labels, deseasonalised returns), `lstm_index_<split>.pt`
  (`PanelWindowDataset`), `graph_snapshots.pt` (`HourlySnapshotDataset`; |corr| threshold fitted on train),
  `tgn_events.pt` (shock-triggered `TemporalData`, contagion labels for k = 1–24 h; 1.46M events at 1h),
  `network_features.pt` (8 leakage-free node features per daily snapshot), `seasonality.json`, `scalers.pkl`,
  `splits.json`, `metadata.json`, `data_dictionary.md`. `v2/rankings.csv`: coin rankings for explanation validation only.
- **Splits:** same dates as daily (train → 2022-06-30, val → 2023-12-31, test → 2026-09-27) with a 7-day embargo.

## 🧪 Training runs, Colab and data packages

- `cryptograph-train --config configs/experiment/<name>.yaml --seed <n> --device auto|cpu|cuda` writes
  `runs/<run_id>/` (resolved config, environment + git commit, log, metrics, checkpoint) and appends `runs/results.csv`.
- `cryptograph-package pack --version v2` → `dist/cryptograph_data_v2.zip` with a SHA-256 manifest;
  `cryptograph-package unpack <zip>` verifies every file. Data never goes into git.
- `colab/runner.ipynb` runs the same commands on a Colab GPU (`colab/README.md`, incl. Kaggle).
- **CI:** `.github/workflows/tests.yml` runs `pytest` on every push with the pinned versions of `requirements-lock.txt`
  (integration tests skip on CI because data is never in git; they run locally where the artifacts exist).

## 🔮 Serving: predictions, dashboard, compression

Needs the final TGN run (`runs/*_tgn_1h_final_v2_0_*`, with its `predictions_val.pt`) and `data/processed/v2/`.

```bash
cryptograph-predict                                   # latest shock bar -> runs/<run_id>/risk.csv, alerts.json, ...
cryptograph-predict --as-of "2025-10-10 21:00"        # the risk as it stood at that hour
streamlit run dashboard/app.py                        # shows the latest prediction run
cryptograph-train --config configs/experiment/compress_tgn_1h.yaml        # int8 / pruning vs fp32 on the test events
cryptograph-train --config configs/experiment/robustness_extra_1h.yaml    # PGD attacks + driver stability
cryptograph-train --config configs/experiment/predict_events_1h.yaml     # catalogued crises for the dashboard
```

- **Risk** per coin and horizon (1–24 h) = the highest calibrated probability over the most recent shock events that
  point at the coin. The focal-loss scores are calibrated by binned isotonic regression on the validation period (never
  the test period) and made non-decreasing in the horizon. Alerts are ranked by lift over each horizon's base rate.
- **Drivers** = integrated gradients on the prediction head for the coin's top event (completeness checked per run).
- **Shock context** = type of the source shock (notebook 05) and how long such shocks lasted before the as-of hour.
- **Event explorer** (dashboard tab, `serving/events.py`): each catalogued crisis in the hourly data, anchored at the
  hour of its date with the most shocks: an hour-by-hour replay of the shock network, coins shocking per hour, the
  per-coin shock sequence, and the model's risk at the crisis hour next to what happened (hits vs chance). Each event
  shows its period; only test-period events are genuine forecasts.
- **Compression** (`serving/compress.py`): every variant replays the whole stream and is scored on the same test
  events; AUPRC cost with the paired day-block bootstrap, top-1% alert overlap, CPU latency per 200-event batch, size
  (`reports/compression/`).
