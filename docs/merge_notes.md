# Merge notes

## Dakshesh's local changes (not yet in the repository)

Status on 2026-09-30: the restructure (Phase B) went ahead before Dakshesh's local changes were pushed, because
the repository is being made private first.

His archived notebook `notebooks/archive/02_statistical_spillover_baseline.ipynb` reads
`VARSpilloverBaseline.compute_spillover_table()["directional_table"]`. That key exists only in his local copy.
In this repository the same information is:

- `spillover_matrix`: the directional table (rows receive, columns send, plus a `FROM` column), and
- `to_others`, `from_others`, `net_spillover`, `total_spillover_index`.

`VARSpilloverBaseline` has moved to `cryptograph.analysis.var_spillover` (exported as
`cryptograph.analysis.VARSpilloverBaseline`) and now defaults to the generalized FEVD.

**To do when his branch is pushed:** diff it against `feat/data-pipeline`, then port `directional_table` by hand
as an alias of `spillover_matrix` (or a formatted version of it) if anything still needs it. The merge is proposed
to Hardik before anything is merged.

## Dakshesh's remaining tasks (status 2026-10-07)

The research part is complete on `feat/hourly-network-models` (notebooks 01-06, all final runs). Persistence
(notebook 05) and robustness / T-GAE (notebook 06) were originally his and are now done here, so his remaining
work is the user-facing and deployment side:

1. **Start from the current branch.** Branch off `feat/hourly-network-models` (latest commit) into a new branch.
   Results from his earlier `main` work (leaky contagion label, broken cascade) are not reused; anything he keeps
   must be re-run on the v2 hourly pipeline.
2. **`cryptograph-predict` command.** Load the final TGN (`runs/*_tgn_1h_final_v2_0_*`, via
   `explain.tgn_capture.load_trained`) and the latest processed data; output, per coin, the contagion risk at
   1/3/6/12/24 h after the most recent shocks, plus the top drivers of each alert from integrated gradients
   (`explain/attribution.py`; TempME is too weakly faithful for per-alert explanations, notebook 04).
3. **Web UI** in `dashboard/` (Streamlit, untouched so far): reads the predict output; shows risk per coin, the
   riskiest alerts with their drivers, typical persistence (8-10 h, notebook 05) and the network view
   (`data/processed/v2/rankings.csv`). The current `dashboard/app.py` text ("real-time monitoring", hard-coded values) does not
   describe the project yet; replace it with real outputs.
4. **Model compression.** E.g. dynamic quantisation or pruning of the TGN for CPU serving; report the test AUPRC
   change against the uncompressed model on the same events (day-block bootstrap, `analysis/evaluation.py`) and
   the CPU latency.
5. **Reconcile his persistence / robustness code** with notebooks 05-06: port only analyses that add something
   new, without duplicate modules (`analysis/persistence.py`, `robustness/stress.py`, `models/tgae.py`).
6. **`directional_table`** alias of `spillover_matrix` if his code still needs it (section above).

**Rules for his branch:** pytest passes with tests for new code; configs in `configs/`; runs through the runner
(`runs/`, `results.csv`); no data, checkpoints or `runs/` in git; push only to the private repo; Hardik reviews and
approves the merge before anything is merged.

## Status of Dakshesh's tasks (branch `feat/predict-dashboard-compression`, 2026-10-07)

Branched from `feat/hourly-network-models` at `183d78e`. Nothing from the old `main` results is reused; everything
below runs on the v2 hourly pipeline. Full suite: 186 passed, 5 skipped (daily-artifact integration tests).

| Task | Status |
|---|---|
| 1. Start from the current branch | done |
| 2. `cryptograph-predict` | done: `serving/predict.py`, `pipelines/predict.py`, `configs/experiment/predict_tgn_1h.yaml`. Runs through the runner (`model: predict`). Risk = binned-isotonic calibration on the run's `predictions_val.pt` (validation only), non-decreasing in k; per-coin risk = max over the recent events pointing at the coin; drivers = integrated gradients (completeness gap logged); shock type + typical duration of the source shocks (notebook 05 definitions, data up to the as-of hour only). Checked end-to-end on the real 1h stream with a smoke model: 33 s on CPU, rebuilt logits identical to the stream (max diff 0.0). |
| 3. Dashboard | done: `dashboard/app.py` (data layer `serving/dashboard_data.py`), reads the latest predict run: risk per coin and horizon, riskiest alerts with driver charts, shock duration by type, correlation network at the as-of hour + `rankings.csv`, the served model's own test AUPRC. Tested with Streamlit's AppTest. |
| 4. Compression | code done: `serving/compress.py`, `configs/experiment/compress_tgn_1h.yaml` (fp32 vs dynamic int8, int8 with fp32 time encoder, 50% pruning; AUPRC cost with the day-block bootstrap on the same test events, top-1% overlap, CPU latency, size). **Final numbers pending the final TGN run.** Speed of the final architecture (64/16/64, random weights, this laptop): fp32 3.7 ms per 200-event batch on 1 thread (5.2 ms on all threads); int8 3.1x smaller (94 vs 296 kB) but 1.7x slower (6.3 ms), because the per-batch neighbour lookups and memory updates dominate, not the matrix products. |
| 5. Reconcile persistence / robustness | done, ported into the existing modules (no new module): `analysis/persistence.py` gains the matched-null excess volatility (`excess_vs_null`), sector half-lives with day-block intervals and pairwise differences (`sector_half_life`), and the crisis blast radius (`event_blast_radius`), using notebook 05's fresh-shock and type definitions; notebook 05 gets an unexecuted section 5. `robustness/stress.py` gains PGD (`attack_steps > 1`; one step is exactly the old FGSM, tested), and `explain/attribution.py` gains `driver_stability` (top-k IG drivers under input noise), both run by `configs/experiment/robustness_extra_1h.yaml` into separate report files (`robustness_1h` and notebook 06 unchanged). Not ported: the old snapshot-model attacker / T-GAE harness (superseded by `stress.py` and `SnapshotTGAE`), the old "most hit sector" event column (crises hit nearly every coin, so it only named the largest sector), the old declustering rule (notebook 05's `fresh_shocks` is used instead). The legacy `robustness/perturbation.py`, `robustness/anomaly.py` and `models/tgae.TemporalGAE` are left as they are. |
| 6. `directional_table` | done: alias of `spillover_matrix` in `VARSpilloverBaseline.compute_spillover_table()` (as in `96b271d`), for the archived notebook 02; tested. |

**Final runs (2026-10-08, commit 6f8db63, on Hardik's data package and runs).** The data package unpacked with every
SHA-256 verified (`cryptograph-package unpack`); its event counts match `reports/hourly_1h_sample_counts.csv`. The run
folders were extracted into `runs/` (the zip holds them at its top level, not under `runs/`). On CPU, the final TGN
(`20261004-065801_tgn_1h_final_v2_0_8f5d566`) replays the same 796,237 test events as its saved predictions (max
|dp| 2.4e-4, the fp16 rounding of the saved file) and its test AUPRC is identical at every horizon (0.2032 / 0.3517 /
0.4641 / 0.5880 / 0.7387 at 1 / 3 / 6 / 12 / 24 h).

- **Predict** (`runs/20261007-191158_predict_tgn_1h_0_6f8db63`): as of 2026-09-27 23:00 UTC the only shock is an
  idiosyncratic QNT crash (z -4.8; such crashes were back to normal after a median 8 h). Top alert JUP at 1.4-1.5x
  the base rate at every horizon, driven by its own relative volume. Rebuilt logits match the stream to 1e-7; IG
  completeness gap 0.004.
- **Compression** (`reports/compression/`): int8 costs 0.004-0.007 test AUPRC (about 1-1.5% relative, p <= 0.006 at
  every horizon with the day-block bootstrap), keeps 88% of the top-1% alerts, is 3.1x smaller (94 vs 296 kB) and
  1.8x *slower* on CPU (2.1 vs 1.2 ms per 200-event batch, 1 thread; more threads do not help). Keeping the time
  encoder in fp32 changes nothing. 50% pruning: no size or speed gain (0.8x), cost significant at 12 h and 24 h,
  84-86% alert overlap. **Recommendation: serve fp32** (about 6 microseconds per event on one thread); int8 only if
  the 3x smaller file matters. Per-batch neighbour lookups and memory updates, not matrix products, dominate.
- **Notebook 05** executed top to bottom on the package data (sections 1-4 unchanged); section 5: the excess
  volatility over a matched calm hour halves in 2.4-2.6 h in every sector with no sector differences, a small gap
  remains at 72 h; every catalogued crisis is market-wide and reaches all coins within 72 h, LUNA raised the shock
  rate most (+0.19), the BTC ETF approval least (+0.01).
- **Robustness additions** (`robustness_extra_1h`, `reports/robustness/stress_tests_pgd.csv`, `driver_stability.csv`):
  both gates passed (clean TGN replay 2.4e-4; refitted boosting reproduces its validation AUPRC to 1e-16).
  10-step PGD is barely stronger than notebook 06's one-step FGSM (TGN AUPRC within 0.002 at every radius and
  horizon, e.g. 0.078 vs 0.080 at eps 0.25, 1 h): the model is close to linear within these radii, so the FGSM
  results are already a near-worst case and notebook 06's conclusions stand. The per-alert IG drivers are stable: on
  the riskiest 2,000 test alerts at 6 h, the top driver is unchanged for 99 / 98 / 97% of alerts under input noise of
  0.05 / 0.1 / 0.25 std (top-5 Jaccard 0.97 / 0.94 / 0.86). `robustness_1h` and its report are unchanged.

**For Hardik's review:** the branch is ready to merge into `feat/hourly-network-models`; nothing was merged.
