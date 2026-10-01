# Archived notebooks

These two notebooks were written and run by Dakshesh on 2026-09-29/30 against the **original** data pipeline
(9 features; coins incl. ATOM, DOT, FIL, SHIB; window 2016-09-24 → 2026-09-24 from `period="10y"`), before the
audited pipeline replaced it. They are kept here **unchanged**, with their saved outputs, for reference.
They will not run against the current repo (see "Compatibility" below).

The corrected, current versions are:

| Archived notebook | Corrected version |
|---|---|
| `01_data_preprocessing_and_eda.ipynb` | `notebooks/01_market_dynamics_and_spillover_baseline.ipynb` (layout and section order taken from the archived notebook, rebuilt on the audited data) |
| `02_statistical_spillover_baseline.ipynb` | `notebooks/02_network_analysis.ipynb`, section 2b (generalized FEVD), next to the hourly spillover network |

## `01_data_preprocessing_and_eda.ipynb`

**What it computes:** dataset dimensions, a quality table, launch dates and active masks, price growth with crises,
ADF tests, fat tails (Jarque-Bera, kurtosis), ACF of returns vs squared returns, shock and contagion counts, a scaling
plot, train/val/test split periods, rolling average correlation, graph edge counts and a tensor check.

**Differences from the audited pipeline:**

1. Return statistics are computed from `features[:, :, 0]`. In the original pipeline that feature was the model input:
   log returns clipped to ±2 with missing days set to 0. Statistics should use raw returns from the cleaned prices
   (`data/interim/master.parquet`). The kurtosis (10.03), ACF (+0.156) and correlation values differ for this reason.
2. The quality table counts days before a coin was listed as missing closes (12,973) and reports a pass rate of 81.95%.
   It also marks 217 zero-or-negative prices as "Passed".
3. The shock z-score uses a rolling mean and standard deviation that include day t itself, so each day's shock is
   measured partly against itself. The audited pipeline uses volatility up to t−1.
4. "Multi-asset contagion probability 38.71%" is the share of shock days on which two or more coins shocked on the same
   day. That measures co-occurrence. Contagion as defined in the project needs a time lag (coin j shocks within k days
   after coin i).
5. The scaling section describes train-only robust scaling. The original pipeline did not scale `log_volume`, so the plot
   shows unscaled values.
6. The splits are 70/15/15 fractions of the common period 2020-10-14 → 2026-09-24. The project config uses fixed dates
   (train to 2022-06-30, val to 2023-12-31, test from 2024-01-01) so that each split contains a major shock. The 30-day
   embargo is drawn on the chart but not applied to the datasets.
7. Edges are described as "directed, |r| ≥ 0.60". The original graph builder produced symmetric edges (each pair counted
   twice) with a threshold of 0.35–0.40 plus top-k links.
8. The claim that correlation rises "from ≈0.35 to 0.80+" is not printed from the data. 2021-05-19 is labelled "China
   Mining Ban"; the event that day was the ban on financial institutions offering crypto services.

## `02_statistical_spillover_baseline.ipynb`

**What it computes:** a VAR(2) on the common active period, a Diebold-Yilmaz spillover table, the total spillover index,
the NET ranking and an export to `data/processed/statistical_connectedness.csv`.

**Differences from the audited pipeline:**

1. The markdown describes the generalized FEVD (Pesaran-Shin), but the saved numbers (TSI 54.24%, BTC NET +731.50) come
   from the Cholesky FEVD, whose result depends on the column order. BTC's own-variance share of 97% is the typical
   first-column effect. With the generalized FEVD on the audited 20-coin balanced panel, TSI is 86.0% and ETH is the
   largest net transmitter.
2. It reads `spillover_results["directional_table"]`, a key that is not in the repository's `VARSpilloverBaseline`
   (it returns `spillover_matrix`). The code that produced it was local to the author's machine.
3. Returns come from `features[:, :, 0]` (see point 1 above).
4. The "Top 3 transmitters" list prints XRP with a negative value ("+ -3.92%"). NET values are percentage points, not
   percent. The last markdown cell is duplicated. The export has no TO/FROM columns.

## Compatibility

Both notebooks read `data/raw/crypto_top20_10y_1d.parquet` or the old 9-feature `crypto_graph_dataset.pt`, and use
`../src` as the import path. They were run from `D:\GitHubProjects\CryptoGraph\notebooks`. Against the current repo
they fail (missing raw parquet, different feature layout, missing `directional_table`).
