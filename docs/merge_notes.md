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
   (`data/processed/v2/rankings.csv`).
4. **Model compression.** E.g. dynamic quantisation or pruning of the TGN for CPU serving; report the test AUPRC
   change against the uncompressed model on the same events (day-block bootstrap, `analysis/evaluation.py`) and
   the CPU latency.
5. **Reconcile his persistence / robustness code** with notebooks 05-06: port only analyses that add something
   new, without duplicate modules (`analysis/persistence.py`, `robustness/stress.py`, `models/tgae.py`).
6. **`directional_table`** alias of `spillover_matrix` if his code still needs it (section above).

**Rules for his branch:** pytest passes with tests for new code; configs in `configs/`; runs through the runner
(`runs/`, `results.csv`); no data, checkpoints or `runs/` in git; push only to the private repo; Hardik reviews and
approves the merge before anything is merged.
