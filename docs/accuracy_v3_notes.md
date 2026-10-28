# Accuracy v3 (sandbox branch `exp/accuracy-v3`, 2026-10-28)

Work done in a separate worktree (`../cg-sandbox`) with its own copy of the data; the main checkout, its data and
every reported v2 result are untouched. All choices below were made on **validation only**; the test period has not
been scored by any new model.

## Problems found in the v2 pipeline

| Problem | Evidence | Change (default keeps v2 behaviour) |
|---|---|---|
| The T1 label depends only on (hour, receiver), so each receiver-hour is scored ~3 times (once per sender) and market-wide hours dominate | 3.1 events per receiver-hour (val), 3.66 (test); scores vary little across senders (13% of the spread); 83% of 1h positives are receivers already in a shock episode | `scripts/fair_report.py`: per receiver-hour metrics, lift, AUROC, calm-receiver slice, top-1% precision → `reports/fair_eval_v2.csv` |
| Memory and neighbours were updated once per 200-event batch (~4 hours) | 31% of events were predicted without hour t-1; lift 1.6 vs 2.0 when t-1 was in memory (val) | `training.memory_step: hour` (gradients accumulated, one optimiser step per batch) |
| `LastAggregator` kept one arbitrary message per coin per update (all events of an hour share a timestamp); messages had no sender/receiver flag; `src_shock` was always 1 | memory ≈ no memory; uniform attention (notebook 04) | `memory_aggregator: mean`, `role_flag: true` (`RoleMessage`), `shock_size_messages: true` (sender and receiver z replace the constant flag) |
| Label-blind pruning dropped `rv_168h`, the shock label's own scale, and the range | `features_dropped` in v2 metadata | `features.protected_features` |
| Features in 2020-22 units drift in the calmer val/test periods | median scaled `rv_24h`: 0 / −0.58 / −0.42 (train/val/test); log(rv24/σ168): −0.10 / −0.12 / −0.10 | `ratio_features: true` (log volatility / σ168) |
| `log_hours_listed` is constant in val/test and only marks the 2020-21 era | every coin is > 2 years old from 2024 | `drop_features: [log_hours_listed]` |
| Early stopping on the 1h score only | memory mattered most at 6-24h | `select_on: mean` |
| Hourly bars miss the within-hour moves that decide the label | knowing the other coins' next hour gives val 1h AUPRC 0.62 vs 0.155 for everything known at t | 5-minute klines (`scripts/fetch_5m.py`, 19.3M bars, all checksum-verified) → `data/features/subhour.py` |

Checks: v3 keeps the v2 labels, events, masks and splits bit for bit; with the new code and the v2 settings, one
epoch reproduces the original seed-0 run exactly (train loss 0.05611, val AUPRC 0.1203); new unit tests in
`tests/unit/models/test_tgn_v3.py` and `tests/unit/data/test_subhour.py`.

## Results (validation)

**Same hourly task, better inputs** (`scripts/go_nogo_5m.py` → `reports/go_nogo_5m.csv`; fixed settings, no tuning;
event-level AUPRC, comparable with the TGN's numbers):

| Inputs | Model | 1h | 3h | 6h | 12h | 24h |
|---|---|---|---|---|---|---|
| v2 TGN (5 seeds, reference) | TGN | 0.153 | 0.257 | 0.362 | 0.481 | 0.652 |
| v2 | logistic regression | 0.155 | 0.260 | 0.358 | 0.491 | 0.662 |
| v3 hourly (ratios, range, trades, calendar, market) | boosting | 0.167 | 0.263 | 0.348 | 0.477 | 0.649 |
| v3 + 5-minute features | boosting | **0.186** | **0.279** | 0.357 | **0.497** | **0.665** |
| random | | 0.086 | 0.174 | 0.269 | 0.397 | 0.580 |

**The contagion question at 15-minute resolution** (`scripts/e2_15min.py` → `reports/e2_15min.csv`; 15-minute shocks,
z* = 2.23 on train for a 4% rate; boosting, logistic regression within 0.01):

| Horizon | AUPRC | random | lift | AUROC |
|---|---|---|---|---|
| 15 min | 0.284–0.289 | 0.117 | 2.4–2.5× | 0.72–0.73 |
| 30 min | 0.379–0.383 | 0.181 | 2.1× | 0.72 |
| 60 min | 0.473–0.478 | 0.256 | 1.9× | 0.71–0.72 |
| 120 min | 0.566–0.573 | 0.360 | 1.6× | 0.69–0.70 |

Hourly task for comparison: 1.8× (TGN) to 2.2× (boosting + 5-minute features) at 1h, AUROC about 0.64. The finer
task is easiest closest to the shock: contagion is a minutes-scale process.

**TGN variants** (seed 0, 60 epochs, lr 1e-3 halved on plateau, best epoch by mean validation AUPRC; event-level):

| Run | Inputs | Memory | 1h | 3h | 6h | 12h | 24h | mean |
|---|---|---|---|---|---|---|---|---|
| `tgn_1h_ctrl_dev` | v2 | v2 (per batch, last message) | 0.147 | 0.257 | 0.362 | 0.487 | 0.658 | 0.382 |
| `tgn_1h_fix_dev` | v2 + shock sizes | hourly, mean, role flag | 0.155 | 0.260 | 0.355 | 0.479 | 0.656 | 0.381 |
| `tgn_1h_v3_dev` | v3 + calendar + context | hourly, mean, role flag | 0.168 | 0.274 | 0.373 | 0.500 | **0.677** | 0.398 |
| `tgn_1h_v3_nomem_dev` | v3 + calendar + context | none | **0.179** | **0.280** | **0.376** | 0.499 | 0.668 | **0.400** |
| boosting, v3 + 5-minute (above) | | | 0.186 | 0.279 | 0.357 | 0.497 | 0.665 | 0.397 |

The memory fixes help a little at 1h (0.147 → 0.155) but the memory still does not beat no memory: the gain comes
from the inputs. Frozen for test: `tgn_1h_v3_final` (= `v3_nomem`, seeds 0-4), boosting v3 + 5-minute
(`go_nogo_5m.py --final`), and the 15-minute task (`e2_15min.py --final`).

## Test (scored once, frozen recipes)

Hourly task, 1h (`reports/final_1h_test.csv`; random 0.105 event-level, 0.059 per receiver-hour, 0.034 calm):

| Model | event AUPRC | per receiver-hour | calm receivers |
|---|---|---|---|
| v2 TGN (5 seeds, from the main repo) | 0.204 | 0.132 | 0.052 |
| logistic regression, v2 inputs | 0.213 | 0.139 | 0.057 |
| boosting, v3 + 5-minute (validation pick) | 0.210 | 0.131 | 0.059 |
| logistic regression, v3 + 5-minute | 0.218 | 0.148 | 0.073 |

The validation gain of the 5-minute inputs (0.155 → 0.186) mostly does not carry over to test for boosting; the
linear model keeps a small gain, largest for calm receivers (0.057 → 0.073). The hourly task looks capped near 2×
random with these inputs.

15-minute task (`reports/final_15min_test.csv`):

| Horizon | logistic regression | boosting | random | lift (LR) | AUROC (LR) |
|---|---|---|---|---|---|
| 15 min | 0.311 | 0.282 | 0.120 | 2.6× | 0.74 |
| 30 min | 0.442 | 0.421 | 0.196 | 2.3× | 0.75 |
| 60 min | 0.555 | 0.543 | 0.290 | 1.9× | 0.74 |
| 120 min | 0.652 | | 0.404 | 1.6× | 0.73 |

The 15-minute result holds on test (validation 0.289, test 0.311): the strongest result of this work.

## Sender check at 15 minutes (validation only; `scripts/e2_sender_check.py` → `reports/e2_sender_check.csv`)

Does knowing *which* coins shocked help, beyond how many? Added to the 15-minute model: the receiver's correlation
and one-bar lead-lag with the coins that shocked now, their correlation-weighted shock sizes, and the share of the
receiver's 5 closest coins that shocked (30 days of history, whole days before the bar). Control: the receiver's
correlation with all coins.

| Inputs | 15 min (boosting / LR) | 30 min | 60 min | 120 min |
|---|---|---|---|---|
| base | 0.284 / 0.289 | 0.383 / 0.379 | 0.478 / 0.473 | 0.573 / 0.566 |
| base + control | 0.283 / 0.289 | 0.378 / 0.379 | 0.477 / 0.473 | 0.572 / 0.566 |
| base + control + sender | **0.296** / 0.289 | 0.386 / 0.380 | 0.485 / 0.474 | 0.575 / 0.567 |

Boosting at 15 min: +0.013 over the control (week-block bootstrap 95% interval +0.002 to +0.021; better in 99.5% of
resamples). Real but small (about 4.5%), non-linear only, and fading with the horizon. Reason: these 30 coins move
together (mean correlation 0.59; the senders' 0.60), so who shocked says little beyond how many shocked.

## Final TGN on test (`tgn_1h_v3_final`, no memory, v3 inputs; 5 seeds, scored once)

| | 1h | 3h | 6h | 12h | 24h |
|---|---|---|---|---|---|
| v2 TGN (reported) | 0.205 | | | | 0.739 |
| v3 TGN, mean of 5 seeds | **0.231** | 0.382 | 0.500 | 0.619 | **0.762** |

Seed spread ±0.001-0.002; 1h AUROC 0.709 (v2 0.685). Unlike the tabular models, the TGN with the v3 inputs keeps its
validation gain on test: +0.026 at 1h (2.2× random vs 1.95×).

Fair metrics (`reports/fair_eval_v3.csv`, test, 1h): per receiver-hour 0.156 vs 0.059 random (v2 0.132), calm
receivers 0.070 vs 0.034 (v2 0.052), top-1% precision 0.328 (v2 0.272). v3 is ahead of v2 at every horizon on every
metric. Paired comparison on the same 796,237 test events (seed-averaged probabilities, week-block bootstrap, 300
resamples; `reports/final_tgn_v3_vs_v2_test.csv`):

| k | v2 | v3 | difference | 95% interval |
|---|---|---|---|---|
| 1h | 0.206 | 0.232 | +0.026 | +0.007 to +0.052 |
| 3h | 0.353 | 0.383 | +0.030 | +0.012 to +0.047 |
| 6h | 0.465 | 0.500 | +0.035 | +0.019 to +0.050 |
| 12h | 0.589 | 0.620 | +0.031 | +0.015 to +0.045 |
| 24h | 0.740 | 0.763 | +0.023 | +0.012 to +0.034 |
