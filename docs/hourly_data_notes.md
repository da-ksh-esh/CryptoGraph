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
