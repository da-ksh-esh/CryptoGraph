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
