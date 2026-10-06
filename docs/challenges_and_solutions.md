# Challenges and how we solved them

A record of the real problems we ran into, why each mattered, and what we did. Grouped by project stage.
Numbers refer to the final results in `notebooks/03_models_vs_baselines.ipynb`.

---

## 1. Data

| Problem | Why it mattered | What we did |
|---|---|---|
| **Look-ahead in the first shock definition.** The original shock detector used the mean and std of the *whole* sample, and one notebook's shock z-score included day *t* itself. | The model would be "warned" by future data: results look great but are impossible in real life. | Rewrote shocks with trailing windows only (`data/features/targets.py`); threshold chosen on the training period only. Unit tests check that changing data after *t* never changes anything at or before *t*. |
| **A leaky contagion label** in the earlier pipeline on `main`, and co-occurrence (two coins moving the same hour) labelled as "contagion". | Same-hour co-movement is not spillover; the earlier high scores were not real. | New label: after coin *i* shocks at hour *t*, does coin *j* shock in (*t*, *t*+*k*]? Strictly later, for k = 1, 3, 6, 12, 24 hours. |
| **Random/fractional train-test splits** in the first notebooks. | Shuffled time series leak the future into training. | Fixed date splits (train ≤ 2022-06-30, val 2022-07 → 2023-12, test 2024-01 → 2026-09) plus a **7-day embargo** at each boundary, and sample masks so no label window crosses a split. |
| **Daily data was too coarse** (professor's review): contagion in crypto happens within hours. | Daily bars hide which coin moved first. | Moved to **hourly Binance data, 30 coins, 2020–2026**; each file verified against Binance's SHA-256 checksum. We later confirmed **1h beats 2h** for this task. |
| **Coin universe pitfalls**: LUNA collapsed and its ticker was reused for a new coin; MATIC was renamed POL; XMR was delisted in 2024. | Splicing two different coins under one name creates fake crashes; a delisted coin has no test data. | Continuity rules (`data/quality/binance_continuity.py`): never splice old and new LUNA, handle renames, mask the gaps. Inclusion rule: ≥ 2 years of data and ≥ 90% coverage of the test period; every exclusion recorded with its reason. |
| **Exchange maintenance gaps** and missing hours. | Gaps look like zero-return hours or create artificial jumps. | Forward-fill at most 2 hours (flagged); longer gaps masked, never filled. |
| **Prices are in USDT, not USD.** | If USDT de-pegs, every coin appears to move together. | Daily USDT peg check from Coin Metrics, documented as ending where that archive ends (2026-05-23); no hourly peg source exists among our data sources, stated as a limitation. |
| **Intraday seasonality**: volume and volatility follow strong hour-of-day / weekday patterns. | Models would learn "it's 14:00 UTC" instead of real risk. | Hour × weekday profile fitted on **train only**, used to standardise volume and volatility. |
| **Wrong event descriptions** in the events catalogue (e.g. the 2021-05-19 China event). | Event studies and the case studies cite these. | Re-verified every event date and source; corrected wording where sources did not support it. |
| **Cross-source consistency.** | One bad source silently corrupts everything downstream. | Hourly data aggregated to daily and compared to the independent daily dataset; differences > 1% investigated per coin. |

## 2. Turning markets into a graph problem

| Problem | Why it mattered | What we did |
|---|---|---|
| **Too many events for the temporal GNN.** Every shock creates an event towards every other coin. | Above ~1M events training on a laptop CPU becomes impractical. | Measured the counts per split first, compared options (2h bars, top-k receivers, cap), and chose the full 1h stream capped at 2M events. |
| **Is there real lead–lag at all?** | Without it a graph model has nothing to learn beyond correlation. | Lead-lag −24h…+24h, Granger tests with Benjamini–Hochberg correction and a shuffled null; Epps effect checked across 1h–1d. |
| **Network measures can leak the future too.** | Centralities computed on the full sample "know" future crashes. | All networks (correlation, MST, lead-lag, Diebold–Yilmaz spillover) built on **trailing windows only**; tested for causality. |

## 3. Models and training

| Problem | Why it mattered | What we did |
|---|---|---|
| **Rare events: "accuracy" is meaningless.** About 90% of cases are "no contagion" at 1 hour. | A model that always says "no" would get ~90% accuracy. | Report **AUPRC** (against the random baseline = positive rate) and AUROC; trained with focal loss. |
| **Leakage inside the TGN's memory.** If events of the same hour are split across batches, the model sees part of hour *t* when predicting hour *t*. | Silent, optimistic results. | Batches never split an hour; memory is updated **after** each prediction. |
| **Event IDs reset to 0** after a neighbour-loader reset, mis-aligning predictions and events. | Wrong labels attached to predictions. | Explicit position mapping (`pos_of_eid`); predictions saved with event positions and checked to match across models. |
| **Cascade baseline gave NaN** for coin pairs never seen shocking in training. | Baseline would be unusable. | Pooled fallback probability for unseen pairs. |
| **Under-trained first final run (v1).** 30 epochs, patience 4 stopped while validation was still rising (0.138). | We would have under-reported the model. | v2: 100 epochs, patience 10 → validation 0.153. v3: `min_epochs: 30` for models whose curve has a noisy early peak. All decided on validation only. |
| **Is the model overfitting?** | Train-period patterns may not hold later. | Compared train→validation drop against the cascade's (which cannot memorise): TGN 1.5 vs 1.1 at 1 hour → some short-horizon overfitting, controlled by early stopping; none at 24 hours. Regularisation reduced it only by losing accuracy, reported honestly. |
| **Many tweaks, tiny differences.** LR schedules, extra context features, regularisation changed validation by ±0.005. | Easy to pick a "winner" that won by luck. | Measured seed-to-seed noise (5 seeds) first; kept the simplest model when a tweak did not beat noise. |
| **Is the improvement statistically real?** One crash creates hundreds of correlated events, so ordinary confidence intervals are far too narrow. | Overclaiming significance. | **Day-block bootstrap** (1,000 resamples of whole days). TGN beats the cascade at every horizon; memory helps significantly only at 6–24h. |
| **Bootstrap was too slow** (sklearn AUPRC × 1,000 × 5 horizons). | Hours per comparison. | Wrote a single-pass weighted AUPRC; verified equal to sklearn, including ties and weights. |
| **Volatility: deep learning lost to a classic model.** LSTM worse than HAR-RV at every horizon. | Tempting to hide. | Reported as a finding: for single-coin volatility a linear model is the right tool; deep learning earns its place on rare events and networks. |
| **"Does the graph matter at all?"** The TGN without memory nearly matched the full TGN at 1 hour, so a reviewer would ask whether a plain machine-learning model on the same inputs does as well. | If it did, the graph model would add nothing. | Added a gradient-boosting baseline (scikit-learn histogram boosting, LightGBM-style) on exactly the TGN head's inputs, same events and masks. Validation: it **matches the TGN at 1 hour** (0.155 vs 0.153) but the **TGN is better at 6–24 hours** (0.362 vs 0.341 at 6h). Finding reported as is: hour-ahead risk is mostly in the current features; the graph memory captures slower, multi-hour contagion. Gradient boosting also memorises the training period far more (train AUPRC 0.64 vs val 0.155). |
| **Overall scores look modest** (AUROC 0.64–0.69). | Hard to judge usefulness from AUPRC alone. | Added "precision of the top alerts": of the TGN's riskiest 1% of alerts on test, 61% are followed by contagion within 3 hours (random 21%) and 76% within 6 hours (random 31%). |
| **The test period (2024–26) differs from the training period (2020–22).** | Models trained only on old data under-use the most recent behaviour. | Tried a final refit on train + validation with the number of epochs/trees fixed beforehand from validation (no early stopping on data we train on); test still scored once. **Mixed result, reported as is:** it helped gradient boosting (test AUPRC 0.178 → 0.185 at 1h) but **hurt the TGN** (0.205 → 0.188, all 5 seeds). Likely cause: without early stopping, 77 fixed epochs over 1.8× more data over-train the TGN. We did not re-tune the epoch count, because that would mean tuning on the test; the main TGN stays the train-only model fixed before the refit was tried. |
| **Gradient boosting matched the TGN on validation but not on test.** | Choosing models on validation alone could have crowned the wrong one. | On test the TGN beats boosting at 1h and 3h (+0.028 and +0.023, p ≈ 0.03 for both, with the 95% intervals just touching zero): the graph model generalised better to the new market period, while boosting had memorised the training period (train AUPRC 0.64 vs val 0.155). |
| **Ensembling: weights chosen on validation did not transfer at short horizons.** | An ensemble is not automatically better. | Weights chosen per horizon on validation only. The ensemble beats the TGN significantly at 24h (+0.008, p = 0.001) but is worse at 1–3h, where validation favoured boosting. Reported per horizon. |
| **Hitting the information ceiling.** Architecture tweaks moved validation AUPRC by only ±0.005 (seed noise), so better scores needed new information. | More tuning of the same inputs would only fit noise. | Added **Binance futures data** (same archive, 4,069 checksum-verified files): hourly perpetual-vs-spot premium and funding rates, as leakage-free node features standardised on train. **It made validation worse** (TGN 0.153 → ~0.13 at 1h, early peaks at epochs 5–9; boosting 0.155 → 0.140). Most likely the leverage regime changed: extreme funding in the 2020–22 bull market vs a calmer, more institutional market afterwards. Reported as a negative result rather than tuned until something "won" on validation. |
| **DyGFormer's official library (DyGLib)** did not fit our setup (edge-level multi-horizon labels, our split protocol). | Using it as-is would break the leakage rules. | Compact re-implementation (24k parameters) under exactly the same protocol and events as the TGN; verified both models score identical events. Encoder dropout was 40% of CPU time, so it was turned off (regularisation via head dropout, weight decay, early stopping). |

## 4. Engineering and reproducibility

| Problem | Why it mattered | What we did |
|---|---|---|
| **No GPU.** | Temporal GNNs are slow on CPU. | Kept models compact, ran 5 seeds in parallel with limited threads each; Colab notebooks prepared as an optional fallback. |
| **Results must be traceable.** | "Which code produced this number?" | Every run writes `runs/<run_id>/` with config, environment, git commit and an "uncommitted changes" flag; **final runs only from clean commits**, test scored once per final run. |
| **A silent scripting bug** (PowerShell variable names are case-insensitive: `$s` and `$S` collided) broke the first TGN seed sweep without an error. | Wrong seeds, wrong results. | Caught by checking the run registry against the expected seeds; renamed the variable and re-ran. Lesson: always verify outputs, not just exit codes. |
| **The laptop went to sleep** mid-training several times. | Runs paused for hours. | Keep-awake during runs; sleep disabled while training. |
| **Windows library-loading order** (importing pandas before torch crashed). | Random crashes. | Import torch first everywhere. |
| **Messy legacy codebase** (placeholder modules that returned their input, invented labels like "wash_trading"). | Hard to trust or extend. | Restructured into layered packages (data / network / models / training / explain / robustness); removed placeholders; verified the refactor changed no output (hash comparison); 107 automated tests. |
| **Confidentiality** (possible patent/paper) with a public GitHub repo. | Work could leak. | Repo made private, proprietary license, no pushes without review. |
| **Two people, two branches**, with the earlier results built on the leaky label. | Merging invalid results. | Clear split of work; the other branch is reviewed file by file before merging and its results re-run on the corrected pipeline. |

---

*Living document: new problems are added as they come up (next: gradient-boosting baseline, refit on train + validation, futures data, TempME).*
