# Additions for the report / viva (sandbox exp/accuracy-v3, 2026-10-28)

Draft text only. The submitted report's story and main model (TGN v2, test 1h AUPRC 0.205) do not change; nothing
below claims an hourly improvement.

## 1. Fair score (each coin-hour counted once)

The T1 label ("does coin j shock in the next k hours?") depends only on the hour and the receiver, so a receiver-hour
appears once per shocking sender: 3.7 times on average in the test period. Market-wide hours are therefore counted
many times. Re-scoring the saved TGN predictions with each receiver-hour counted once (`reports/fair_eval_v2.csv`):

| Test, 1h | AUPRC | random | multiple of random |
|---|---|---|---|
| per event (as reported) | 0.204 | 0.105 | 1.95× |
| per receiver-hour | 0.132 | 0.059 | 2.2× |
| receivers calm for the last 24 h | 0.052 | 0.034 | 1.5× |
| precision of the top 1% of alerts | 0.272 | 0.059 | 4.6× |

The model is about 2× random either way; the repetition does not inflate the conclusion.

## 2. Memory timing: a weakness, and what fixing it showed

The TGN's memory was written once per batch of 200 events (about 4 hours), not once per hour, so 31% of validation
events were predicted without the previous hour in memory. This loses information; it never uses future data.
In addition, the "last message" aggregator kept one arbitrary message per coin per hour, and messages carried no
sender/receiver flag. Fixing all three (hourly memory writes, averaged messages, role flag) moved validation 1h AUPRC
from 0.147 to 0.155, while the same model without memory still scored as well or better (0.179 with the v3 inputs).
Conclusion: at hourly resolution the memory adds little; the gain comes from the receiver's own recent state.

## 3. Future work: the same question at 15-minute resolution

Repeating the contagion question on 15-minute bars (built from checksum-verified Binance 5-minute klines, same split
dates, threshold set on train) gives, on test:

| Horizon | AUPRC | random | multiple of random | AUROC |
|---|---|---|---|---|
| 15 min | 0.311 | 0.120 | 2.6× | 0.74 |
| 1 hour | 0.555 | 0.290 | 1.9× | 0.74 |

(compare multiples of random, not raw AUPRC: the share of positives differs from the hourly task.) Validation and
test agree (0.289 vs 0.311), and receivers that were calm still come out at 2× random. The model is a plain logistic
regression on the receiver's own state and market-wide numbers; it uses no information about which coin shocked and
no network, so this shows that shocks cluster tightly in time, not that the graph helps. Finer timing is where the
network question should be tested next.

Sender check (validation only): adding which coins shocked (their correlation and lead-lag with the receiver, and
whether the receiver's closest coins shocked) raises the 15-minute boosting score from 0.283 to 0.296 (bootstrap 95%
interval +0.002 to +0.021), and does nothing for the linear model. So the network does carry some information at
15 minutes, but little: these 30 coins are so correlated (about 0.6 on average) that who shocked adds little beyond
how many shocked. Not scored on test.

## 4. Every test-period score

Old runs: `runs/results.csv` in the main repo (test 1h / 24h AUPRC, mean over seeds; random 0.105 at 1h).
New runs: this sandbox. Each recipe was frozen on validation before its single test scoring.

| Date | Experiment | Seeds | Test 1h | Test 24h | Note |
|---|---|---|---|---|---|
| 10-02 | tgn_1h_final | 5 | 0.201 | 0.737 | |
| 10-03 | tgn_1h_nomem_final | 1 | 0.206 | 0.730 | |
| 10-04 | tgn_1h_final_v2 | 5 | 0.205 | 0.739 | **reported main model** |
| 10-04 | tgn_2h_final_v2 | 5 | 0.165 | 0.560 | 2h bars (random 0.092) |
| 10-04 | tgn_1h_nomem_final_v2 | 5 | 0.195 | 0.726 | |
| 10-04 | tgn_1h_nomem_final_v3 | 5 | 0.208 | 0.729 | |
| 10-05 | lstm_1h_rv_final, lstm_1h_shock_final | 5 + 5 | – | – | other tasks |
| 10-06 | dygformer_1h_final | 5 | 0.208 | 0.736 | |
| 10-06 | gbm_1h_final | 1 | 0.178 | 0.736 | |
| 10-06 | gbm_1h_refit, tgn_1h_refit | 1 + 5 | 0.185 / 0.188 | 0.740 / 0.711 | refit on train + val |
| 10-07 | robustness_1h, tgae_1h_final | 1 + 5 | – | – | other analyses |
| 10-28 | fair re-scoring of tgn_1h_final_v2 | – | – | – | saved predictions, no new model |
| 10-28 | logistic regression, v2 inputs | 1 | 0.213 | 0.747 | baseline |
| 10-28 | boosting, v3 + 5-minute | 1 | 0.210 | 0.743 | the validation pick |
| 10-28 | logistic regression, v3 + 5-minute | 1 | 0.218 | 0.755 | not the pick; do not quote as the result |
| 10-28 | boosting, v2 inputs | 1 | 0.176 | 0.723 | |
| 10-28 | 15-minute task, logistic regression / boosting | 1 + 1 | 0.311 / 0.282 | – | 15-minute horizon |
| 10-28 | tgn_1h_v3_final (no memory, v3 inputs) | 5 | 0.231 | 0.762 | validation pick; seed spread ±0.001; AUROC 0.71 |

## 5. Follow-up model: TGN with the v3 inputs (frozen on validation, test scored once)

Same TGN code, memory off (it did not help on validation), v3 inputs (volatility ratios instead of raw levels, the
weekly volatility kept, coin age dropped, 5-minute features, hour of day, event context). Test, 5 seeds:

| 1h | v2 TGN (reported) | v3 TGN |
|---|---|---|
| event AUPRC (random 0.105) | 0.205 | **0.232** |
| per receiver-hour (random 0.059) | 0.132 | 0.156 |
| calm receivers (random 0.034) | 0.052 | 0.070 |
| top 1% precision | 0.272 | 0.328 |
| AUROC | 0.685 | 0.709 |

Better at every horizon (1h to 24h); the paired bootstrap interval excludes zero at all five (1h: +0.026, 95%
+0.007 to +0.052). The gain comes from the inputs, not from the graph memory. Present it as a follow-up result; the
submitted main model is unchanged.
