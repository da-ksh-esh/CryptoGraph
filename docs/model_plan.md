# Model plan

Status (2026-10-07): **complete.** Every model below marked *done* is trained and evaluated (notebooks 03-06);
this file keeps the original plan and feasibility notes for reference.
Data, splits and labels are fixed by the data phase (`docs/hourly_data_notes.md`, notebooks 01–02).

## 1. Prediction tasks

| Task | Unit | Label | Data | Primary metric |
|---|---|---|---|---|
| **T1 Contagion** (main) | one shock event: coin *i* shocks at hour *t*, for every other coin *j* | does *j* shock within *k* ∈ {1, 3, 6, 12, 24} h (`y`, masked by `y_mask`) | `tgn_events.pt` (1h: 1.46M events; 2h: 0.75M) | **AUPRC** per *k* (with the positive rate), AUROC |
| **T2 Shock risk** | coin × hour | shock in the next bar (`y_shock`) | `panel.pt` + `lstm_index_<split>.pt` | AUPRC, AUROC |
| **T3 Volatility** | coin × hour | forward realized volatility at 1 / 6 / 24 h (`y_rv`) | same | MSE / MAE vs persistence |

Direction (`y_ret`) is reported only as a sanity check: notebook 01 shows it is not predictable.

**Protocol (all models):**
- Fixed-date splits (train 2020-01 → 2022-06, val 2022-07 → 2023-12, test 2024-01 → 2026-09-27) with a 7-day embargo.
- Tune on validation only, and touch test once per final model.
- Run 5 seeds and report mean ± std. Compare against the best baseline with a block bootstrap over days.
- 1h is the main resolution and 2h the robustness check.
- Every run goes through `cryptograph-train` / `cryptograph.training.runner` and is logged in `runs/` (Phase C).
- Test metrics are computed only when an experiment sets `evaluate_test: true` (final models); development and
  tuning runs report validation only.
- Compute: one TGN pass over the full 1h stream takes about 1–2 minutes on a laptop CPU (≈17 ms per 200-event
  batch), so TGN development runs locally; Colab is for seeds × configurations sweeps and DyGFormer.

**Baselines to beat (T1, 1h, from `reports/cascade_baseline.csv`):**

| k | base rate AUPRC (val / test) | independent cascade AUPRC (val / test) |
|---|---|---|
| 1h | 0.086 / 0.107 | **0.117 / 0.160** |
| 3h | 0.174 / 0.207 | 0.212 / 0.280 |
| 24h | 0.592 / 0.631 | 0.578 / 0.672 |

**Baselines to beat (T2/T3, 1h, on the LSTM sample set; `reports/hourly_baselines_1h.csv`, stage `baselines`):**

| Task | Best simple baseline | val | test | Reference |
|---|---|---|---|---|
| T2 next-hour shock (AUPRC) | logistic on 4 rules (\|z\| now, shock now, own shocks last 24h, market breadth), train-fitted | 0.079 | 0.095 | base rate 0.037 / 0.039 |
| T3 RV, 1h (MSE) | HAR-RV (1h, 24h, 168h trailing RV), train-fitted | 4.02e-5 | 5.01e-5 | persistence 6.80e-5 / 8.43e-5 |
| T3 RV, 24h (MSE) | HAR-RV | 1.47e-5 | 1.68e-5 | persistence 1.86e-5 / 2.16e-5 |

At 2h the "1h" horizon is one 2-hour bar.

## 2. Models

| Model | Kind | Tasks | Input | Status | Code |
|---|---|---|---|---|---|
| Naive rules (persistence, train mean, base rates) | baseline | T1–T3 | panel / events | ***done*** (daily + hourly) | `analysis/baselines.py` |
| Independent cascade | network baseline | T1 | panel shocks | ***done*** | `network/cascade.py` |
| Diebold–Yilmaz (VAR, GFEVD) | statistical reference | rankings only | returns | ***done*** | `analysis/connectedness.py`, `network/spillover.py` |
| LSTM (no graph) | deep baseline | T2, T3; T1 via receiver risk | 168-bar windows per coin | ***done*** (T2, T3; 5 seeds) | `models/lstm.py`, `training/runner.py` |
| Snapshot GNN (GCN/GAT + GRU) | graph ablation | T1–T3 | daily graph snapshots + panel | not built (dropped: the TGN, DyGFormer and boosting answer the same question) | `models/snapshot_gnn.py` |
| **TGN** | main model | T1 (edge head), T2 (node head) | event stream + node features | ***done*** (T1; 5 seeds, ablation, 2h, refit) | `models/tgn.py`, `training/tgn_task.py` |
| DyGFormer | comparison TGNN | T1 | event stream (compact re-implementation, not DyGLib) | ***done*** (5 seeds) | `models/dygformer.py`, `training/dygformer_task.py` |
| T-GAE | anomaly / robustness | prediction / reconstruction error | daily snapshots | ***done*** (`SnapshotTGAE`, 5 seeds; negative result) | `models/tgae.py`, `training/tgae_task.py` |
| TempME | explainer | explains T1 | trained TGNN + events | ***done*** (ported to our TGN, route B) | `explain/` |

- **LSTM:** answers "does the graph add anything?". It sees only the receiving coin's own 168-hour history, and
  scores T1 events by the receiver's next-hours shock risk. Any graph model that does not beat it on T1 has learned
  nothing about contagion.
- **Snapshot GNN:** a GCN/GAT over each daily graph snapshot (fitted threshold; `HourlySnapshotDataset`), followed by
  a GRU over snapshots. It tests whether continuous time matters, or whether daily graphs with hourly node features are enough.
- **TGN (main):** PyG `TGNMemory` with node memory updated by the shock events.
  - Messages: the 5 event features (lag-1 corr, same-hour corr, source return, source volatility, shock flag) plus time encoding.
  - Node features: the panel features at *t* and the 8 network features from the latest snapshot ≤ *t* (`network_features.pt`).
  - Head: an edge-classification MLP on [memory/embedding of *i*, of *j*, message] → 5 logits (one per *k*), trained with masked
    focal/BCE loss with `pos_weight` from train. A node head for T2 can share the encoder.
  - Batches are processed in time order and memory is reset per split. Training on 1h runs on the Colab GPU.
  - `configs/model/tgn.yaml` holds draft values.
- **DyGFormer:** a Transformer over each node's recent interactions (patching), strongest on the TGB leaderboards.
  Used as a second TGNN so the results don't hinge on one architecture.
- **T-GAE:** reconstruction-based anomaly scores for notebook 06 (robustness): which hours look unlike the learned
  structure, and how the contagion models degrade under feature or edge perturbations.
- **Independent cascade:** done (notebook 02 §9). It is the bar for T1, and it shows that most of the
  easy signal is "how many coins shocked this hour". A TGNN must add *who* is at risk.

## 3. Explanation validation (TempME or fallback)

Explanations are judged on **faithfulness** and **plausibility**:
- **Fidelity+** = drop in the predicted probability when the explanation's events are removed (higher = necessary).
- **Fidelity−** = drop when *only* the explanation is kept (lower = sufficient).
- **Characterization score** = weighted harmonic mean of Fidelity+ and 1 − Fidelity− (Amara et al., 2022).
- All three are reported over sparsity levels (share of events kept: 1–30%), as TempME's code does.
- **Plausibility:** aggregate per-coin importance and compare with the rankings in `data/processed/v2/rankings.csv`
  (Spearman).
  - The main reference is the time-ordered `shock_pagerank`, plus `dy_net`, `dy_pagerank`, `eigenvector` and the
    daily DY NET from notebook 02 §2b.
  - Notebook 02 shows hourly DY NET equals correlation centrality, so no single ranking is the ground truth.
  - Agreement with the time-ordered ranking counts most.
- **Case studies:** FTX (Nov 2022), LUNA (May 2022), the Oct-10-2025 liquidation cascade. Explanations must be
  consistent with notebook 02 §8: the market moved in the same hour, so we don't expect chains.
- **Fallback:** `explain/edge_mask.py` (a GNNExplainer-style edge mask, *not* TempME), so notebook 04 can be filled
  even if TempME integration slips.

## 4. Feasibility report (official repositories only; read 2026-10-01, no integration yet)

### DyGLib — https://github.com/yule-BUAA/DyGLib (MIT licence)

What the official code expects and does:
- **Input:** `processed_data/<name>/ml_<name>.csv` with columns `u, i, ts, label, idx`, plus `ml_<name>.npy` (edge
  features) and `ml_<name>_node.npy` (node features), made by `preprocess_data/preprocess_data.py`.
  - Node ids are shifted to start at 1 (bipartite graphs get separate id ranges).
  - Edge and node features are zero-padded to 172 dimensions.
  - Node features are static (one row per node).
- **Splits:** timestamp quantiles from `val_ratio` / `test_ratio` (70/15/15 by default). For link prediction, 10% of
  the nodes seen at test time are made "new nodes" and their edges are removed from training.
- **Tasks:**
  - Dynamic link prediction (random / historical / inductive negative sampling).
  - Dynamic node classification with one binary label per interaction (used on Wikipedia/Reddit).
  - No multi-horizon edge labels.
- **Models:** JODIE, DyRep, TGAT, TGN, CAWN, EdgeBank, TCL, GraphMixer, DyGFormer.
- **Environment:** the README lists PyTorch 1.8.1. We run torch 2.x, so expect small fixes.

Gaps for our data, and the adapter they need:

| Gap | Adapter |
|---|---|
| File format | Exporter `tgn_events.pt` → `ml_crypto1h.csv/.npy` (`u`=src+1, `i`=dst+1, `ts` = hours, 5 message features padded to 172; static node feature = sector one-hot). Mechanical. |
| Fraction splits + new-node masking | Replace their split function with our fixed dates and **no** new-node masking (all 30 coins are known). |
| Our label is per event and multi-horizon | Use DyGLib's backbone to compute src/dst temporal embeddings and attach **our** edge-classification head (5 logits, masked BCE); skip their link-prediction trainer. |
| Time-varying node features (panel, network) | Not supported by their loaders; either carry them in the edge features or extend the model's node-feature lookup with a time index. |
| Library pinning | Vendor DyGLib as a pinned git submodule (MIT: keep the licence notice), no edits inside; adapter code lives in `cryptograph/models/dygformer.py`. |

**Verdict:** feasible with an adapter; estimated **3–5 working days** (exporter 0.5 d, splits 0.5 d, edge head and
training loop 2 d, testing on a small slice 1 d). The licence permits use in proprietary work with the notice kept.

### TempME — https://github.com/Graph-and-Geometric-Learning/TempME (official, MIT licence; NeurIPS 2023)

What the official code does:
- **Explains link prediction** of its own TGAT, TGN and GraphMixer implementations, trained by `learn_base.py`
  (random negative sampling, AP/AUC). Edge labels are read but unused ("we are not predicting edge labels").
- **Pipeline:**
  1. `python learn_base.py --base_type ${type} --data ${dataset}` saves `params/tgnn/{type}_{data}.pt`.
  2. `processed/data_preprocess.py` samples temporal walks and motifs into `{data}_{mode}.h5`, `{data}_{mode}_cat.h5`
     and `{data}_{mode}_edge.npy`. It reads `ml_{data}.csv` (TGN format) and splits at timestamp quantiles 0.70 / 0.85.
     Walk settings sit in a hard-coded `degree_dict` keyed by dataset name.
  3. `python temp_exp_main.py --base_type ${type} --data ${dataset}` trains the explainer and reports fidelity
     (probability and logit), AP/AUC/accuracy over sparsity ratios 0.01–0.30.
  4. `enhance_main.py` checks whether explanations improve the base model.
- **Interface:** the base model must expose `contrast(src, dst, dst_fake, ts, e_idx, subgraph_src, ...)` (positive
  and negative logits) and `set_neighbor_sampler(...)`.

It **cannot explain our PyG TGN as it stands**. Two routes:

| Route | What | Pros / cons | Estimate |
|---|---|---|---|
| **A. Reformulate as link prediction** | Build a *contagion interaction stream*: an edge *i → j* at the hour *j* shocks within *k* after *i*. Train TempME's own TGN on it (`learn_base.py`) with our fixed-date split and a `degree_dict` entry, then run TempME. | Official code almost unchanged. But it explains *their* TGN on a link-prediction proxy with random negatives, not our evaluated model. | 3–4 days |
| **B. Port the interface** | Wrap our TGN to expose `contrast()` / `set_neighbor_sampler()`, using our edge-head logit for the event as the positive logit; run their motif sampler on our stream. | Explains the model we evaluate. Needs their neighbour-sampler format and careful time handling. | 1.5–2 weeks |

**Recommendation:** run **A** first as a feasibility prototype on the 2h stream (smaller). Move to **B** once the TGN
is trained and A has shown the motif sampler works on our data. Keep the edge-mask explainer as the fallback for
notebook 04 in the meantime.

## 5. Data → model map

| File | Used by |
|---|---|
| `v2/<res>/panel.pt`, `lstm_index_<split>.pt` (`PanelWindowDataset`) | LSTM; node features for every graph model |
| `v2/<res>/graph_snapshots.pt` (`HourlySnapshotDataset`) | snapshot GNN, T-GAE |
| `v2/<res>/tgn_events.pt` | TGN, DyGFormer (via exporter), TempME, independent cascade evaluation |
| `v2/<res>/network_features.pt` | extra node features (latest snapshot ≤ *t*) |
| `v2/rankings.csv` | explanation plausibility only; never a model input |

## 6. Notebooks

| Notebook | Content | Status |
|---|---|---|
| `01_market_dynamics_and_spillover_baseline` | data audit, EDA (daily Part I, hourly Part II) | done |
| `02_network_analysis` | networks, centrality, communities, MST, cascade baseline | done |
| `03_models_vs_baselines` | T1–T3 results table, per-k AUPRC, seeds, bootstrap tests | done |
| `04_tempme_explanations` | fidelity / sparsity curves, ranking agreement, case studies | done (TempME ported, route B) |
| `05_persistence_by_shock_type` | event study in hours, persistence by shock type (crash / squeeze / idiosyncratic) | done |
| `06_robustness_tgae` | T-GAE anomaly scores, perturbation and attack results | done |

## 7. Future work (deferred, not cancelled): a user-facing view

Once the models are trained, a small interface will show the **current contagion risk per coin**. It could be the
existing `dashboard/` (Streamlit, untouched so far) or a `cryptograph-predict` command. Per coin it would show:
- the probability of a shock within 1/3/6/12/24 h;
- the **expected time-to-hit** after a shock elsewhere;
- the expected **persistence** (from notebook 05);
- the **TempME explanation** behind each alert (which recent shocks and links drove it).

It reads a trained checkpoint from `runs/` and the latest processed data. No serving contract is designed until the
models exist, and nothing in the data or training code depends on it.

## 8. Decisions taken (formerly open)

1. Repository private (checked before every push); Colab runs use self-contained notebooks with Drive zips.
2. TempME: route B (ported to our TGN), because route A would explain a different model.
3. Snapshot GNN: not built; the TGN-vs-boosting comparison answers whether the graph matters.
4. T1 loss: focal loss (chosen on validation AUPRC).
5. Added during training: gradient-boosting baseline, validation-weighted ensemble, train+val refit, futures
   features (rejected on validation), top-alert precision (notebook 03).
