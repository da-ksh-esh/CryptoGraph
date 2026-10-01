# Hourly data (Binance) — source, pitfalls and quality

**Source.** Binance public data archive (`https://data.binance.vision`, spot klines, 1h), monthly zips for
2020-01 .. 2026-08 plus daily zips for 2026-09-01 .. 2026-09-27. No API key. Every file is verified against its
`.CHECKSUM` (SHA-256) before it is kept in `data/raw/binance/1h/<SYMBOL>/` (write-once; `--refresh` re-downloads).
Stage 2 run on 2026-10-01: 3,049 files, all verified, 0 failures, 1.61M coin-hours for 30 model coins + 2 event coins.
2h bars are resampled from 1h (open first, high max, low min, close last, volume sum; a 2h bar needs both hours).

## Pitfalls and how they are handled

| Issue | Handling |
|---|---|
| Prices are quoted in **USDT**, not USD | Recorded. USDT/USD from Coin Metrics (daily) flags days off peg by > 0.5%: 2020-03-12/13 (COVID) and 2023-03-11/12 (SVB/USDC). **The peg check ends on 2026-05-23**, where the Coin Metrics archive ends; later hours carry `usdt_peg_checked = False` (not "peg OK"). |
| Timestamps are milliseconds in older files and **microseconds from 2025** | `read_klines` normalises both to UTC hours. |
| `LUNAUSDT` is two assets | Old Terra LUNA's last hour is 2022-05-13 00:00. The 18 `LUNAUSDT` rows from 2022-05-31 (median close $8.84) are the new LUNA and are dropped, never spliced. LUNC resumes as `LUNCUSDT` on 2022-09-09 08:00; the gap in between stays masked. |
| MATIC → POL | `MATICUSDT` to 2024-09-10 02:00, `POLUSDT` from 2024-09-13 10:00: a 79-hour gap (masked) and a 0.0% price change across the switch (1:1 migration). |
| TON → GRAM | `TONUSDT` files end 2026-06-30, `GRAMUSDT` continues. TON is not a model coin (listed on Binance 2024-08; 78% test coverage). |
| Delistings | XMR delisted 2024-02-20 → excluded from the model universe. FTT trading on Binance was suspended from 2022-11-15 to 2023-09-22 (after the FTX collapse); that gap is masked; the collapse week itself is covered. |
| Exchange maintenance | Binance-wide gaps of a few hours (15 gaps for coins listed since 2020, longest 5 hours). Gaps of ≤ 2 hours are forward-filled and flagged (`is_filled`); longer gaps stay missing and inactive. |

## Quality summary (all coins)

- No zero or negative prices, no high < low, no open/close outside the bar's range, no duplicate hours after stitching.
- 92 hourly moves with |log return| > 0.25: 87 confirmed by the same-day daily move (≥ 10%) or a catalogued event,
  5 unverified (kept, flagged), 0 bad ticks.
- 280 hours forward-filled, 10,779 hours masked as long gaps (mostly the LUNC and FTT gaps and MATIC→POL),
  214 stale-close hours, 118 zero-volume hours, 2,894 volume spikes (> 20× trailing 7-day median).

## Cross-check against the daily data

The close of each day's 23:00 UTC bar is compared with the daily USD close (daily dataset; Yahoo or Coin Metrics for
coins that are not in it). Median difference is about 0.1% for every coin. Differences above 1% are concentrated
in 2020–2021 (10–17% of those days for DOGE, XLM, ZEC, HBAR, QNT) and fall to 0–1% from 2022, the same pattern as the
daily Yahoo vs Coin Metrics comparison: early markets were thinner and the sources aggregate different venues.

**SHIB and PEPE cannot be cross-checked with Yahoo**: Yahoo returns their prices rounded to 6 decimal places
(SHIB is stored as exactly 0.000006, with 217 days rounded to 0.000000; PEPE as 0.000004). Binance has full
precision. Yahoo's daily data for sub-$0.0001 coins is not usable.

## Stage 4 decisions (hourly EDA, notebook 01 Part II)

| Question | Finding (train unless stated) | Decision |
|---|---|---|
| TGN event count at 1h | 1,463,693 shock-triggered events (361k / 301k / 802k) | Full 1h stream built (`tgn_max_events` raised to 2M); trained on the Colab GPU. The 2h stream (752,627 events) is the robustness check. Top-k receivers rejected: it would bias the contagion labels towards the selected pairs. |
| Lead-lag (BTC, ETH → 23 coins, −24…+24 h) | Lag-0 correlation 0.65–0.68; largest non-zero-lag \|corr\| ≈ 0.035, mostly at +24 h (daily cycle). Granger (3 lags, BH): 8/45 leader → coin and 10/45 coin → leader significant, max ΔR² 0.47%; the lag-1 pattern does not replicate on validation | No lead-lag edge features; no directed lag-1 edges in hourly graphs (`directed_lag_threshold: null`). Directed networks in Stage 5 come from GFEVD spillovers. |
| Epps effect | Mean pairwise correlation 0.58 (1h), 0.60 (2h), 0.61 (4h), 0.62 (12h), 0.59 (1d) | Weak; the threshold is set by density, not by an Epps correction. |
| Undirected graph threshold | 0.5 gives 75% density at 1h | Fitted on train snapshots per resolution for 30% density (as the daily graph): 0.75 at 1h, 0.77 at 2h (`threshold: null`, `target_train_density: 0.30`). With the top-2 links per coin, mean density is 38% / 30% / 39% at 1h. |
| Contagion signal | Lift of "j shocks within k hours after i shocks" over the base rate: 2.3× (k = 1h) falling to 1.26× (24h); 2.4× val, 2.7× test | Hourly resolution is justified; contagion is mostly within the first hours. |

## Stage 5 outputs and decisions (network analysis, notebook 02)

`cryptograph-data --config configs/data/hourly.yaml --stage network` writes `data/processed/v2/<res>/network_features.pt`
(8 node features per daily snapshot, trailing windows only; use the latest snapshot at or before t),
`data/processed/v2/rankings.csv` + `rankings.json` (test window 2024-04-02 → 2026-09-27, all 30 coins listed; evaluation
benchmark only), `reports/network_timeseries_<res>.csv` and `reports/cascade_baseline.csv`.

| Question | Finding | Decision |
|---|---|---|
| Directed network for PageRank | Hourly lead-lag is absent, so hourly GFEVD is driven by same-hour correlation: DY NET = eigenvector centrality (ρ = 1.00); daily vs hourly NET ρ = 0.21 | Two directed networks: DY GFEVD (30-day windows) and a time-ordered shock-transmission network (P(j shocks ≤ 1h after i), 90-day windows). PageRank on the "depends on" graph (rank = transmitter importance). TempME is validated against several rankings, not one. |
| Communities | Louvain on raw correlations is one block (market mode = 64% of variance) | Louvain (seeded) on correlations with the market mode removed; NMI with sectors 0.47 (p = 0.003). |
| Centrality scale | Sum-to-1 centralities change level as coins list | Eigenvector and PageRank features rescaled to mean 1 over the active coins. |
| Redundancy | Max |ρ| between the 8 node features on train = 0.78 | All 8 kept. |
| Sectors | Within- vs cross-sector 1h contagion equal (9.3% vs 9.3% train), except meme coins | Sector id kept as a node attribute; graphs are not built around sectors. |
| Contagion benchmark | Independent cascade AUPRC at k = 1h: 0.160 test / 0.117 val vs base rate 0.107 / 0.086; p_ij alone ≈ base rate | The TGN must beat the cascade, not only the base rate. |
