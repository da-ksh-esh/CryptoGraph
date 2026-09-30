import numpy as np
import pandas as pd

from cryptograph.data.quality.binance_continuity import SegmentData, stitch_segments
from cryptograph.data.quality.hourly_checks import HourlyCheckParams, resample_2h, run_hourly_checks
from cryptograph.data.schema import QC, decode


def _bars(start: str, closes, freq="h") -> pd.DataFrame:
    idx = pd.date_range(start, periods=len(closes), freq=freq, name="time")
    c = pd.Series(closes, index=idx, dtype=float)
    return pd.DataFrame({"open": c, "high": c * 1.01, "low": c * 0.99, "close": c, "volume": 10.0,
                         "quote_volume": 10.0 * c, "trades": 5.0})


def test_reused_symbol_rows_after_until_are_dropped_never_spliced():
    old_luna = _bars("2022-05-12 00:00", [0.0001] * 48)                 # old LUNA up to 13 May
    new_luna = _bars("2022-05-31 00:00", [7.0] * 24)                    # same symbol, new asset
    luna = pd.concat([old_luna, new_luna])
    lunc = _bars("2022-09-09 10:00", [0.0003] * 24)
    out, rep = stitch_segments([SegmentData("LUNAUSDT", luna, pd.Timestamp("2022-05-13")),
                                SegmentData("LUNCUSDT", lunc, None)])
    assert out["close"].max() < 0.001                                    # no new-LUNA price anywhere
    dropped = [r for r in rep if r["type"] == "reused_symbol_rows_dropped"]
    assert dropped and dropped[0]["rows_dropped"] == 24 and dropped[0]["median_close_of_dropped_rows"] == 7.0
    b = [r for r in rep if r["type"] == "symbol_boundary"][0]
    assert b["gap_hours"] > 24 * 100 and b["flag"] is False               # long gap: not treated as a 1:1 swap


def test_migration_handover_and_price_jump_flag():
    matic = _bars("2024-09-09 00:00", [0.40] * 48)                      # still trading on 10 Sep
    pol = _bars("2024-09-10 12:00", [0.40] * 24)
    out, rep = stitch_segments([SegmentData("MATICUSDT", matic, None), SegmentData("POLUSDT", pol, None)])
    assert not out.index.duplicated().any()
    assert out.loc["2024-09-10 12:00", "symbol"] == "POLUSDT" and out.loc["2024-09-10 11:00", "symbol"] == "MATICUSDT"
    b = [r for r in rep if r["type"] == "symbol_boundary"][0]
    assert b["gap_hours"] == 0 and b["flag"] is False
    bad = _bars("2024-09-10 12:00", [0.80] * 24)                        # price doubles across the switch
    _, rep2 = stitch_segments([SegmentData("MATICUSDT", matic, None), SegmentData("POLUSDT", bad, None)])
    assert [r for r in rep2 if r["type"] == "symbol_boundary"][0]["flag"] is True


def test_isolated_spike_is_bad_tick_but_confirmed_move_is_kept():
    closes = [100.0] * 30
    closes[10] = 150.0                                                   # +40% then back: bad tick
    closes[20:] = [60.0] * 10                                            # -51% and stays: real crash
    df = _bars("2021-01-01 00:00", closes)
    daily = pd.Series({pd.Timestamp("2021-01-01"): -0.6})                # daily source confirms a big move that day
    out = run_hourly_checks(df, daily, [], HourlyCheckParams(abs_logret_flag=0.25, daily_confirm_logret=0.10))
    t_spike, t_crash = df.index[10], df.index[20]
    # the daily move confirms both hours on 2021-01-01 -> nothing removed
    assert "LARGE_RETURN_CONFIRMED" in decode(out.loc[t_crash, "qc_flags"])
    out2 = run_hourly_checks(df, pd.Series(dtype=float), [], HourlyCheckParams(abs_logret_flag=0.25))
    assert np.isnan(out2.loc[t_spike, "close"]) and "BAD_TICK_CORRECTED" in decode(out2.loc[t_spike, "qc_flags"])
    assert out2.loc[t_crash, "close"] == 60.0 and "LARGE_RETURN_UNVERIFIED" in decode(out2.loc[t_crash, "qc_flags"])


def test_stale_and_zero_volume_flags():
    df = _bars("2021-01-01", [1.0] * 8 + [1.1, 1.2])
    df.loc[df.index[9], "volume"] = 0.0
    out = run_hourly_checks(df, None, [], HourlyCheckParams(stale_hours=6))
    assert ((out["qc_flags"].iloc[:8] & int(QC.STALE_CLOSE)) > 0).all()
    assert (out["qc_flags"].iloc[9] & int(QC.ZERO_VOLUME)) > 0


def test_resample_2h_ohlcv_and_incomplete_bars():
    df = _bars("2021-01-01 00:00", [1.0, 2.0, 3.0, 4.0])
    df["high"] = [1.5, 2.5, 3.5, 4.5]
    df["low"] = [0.5, 1.5, 2.5, 3.5]
    df["is_filled"] = [False, True, False, False]
    df["qc_flags"] = [0, int(QC.FILLED), 0, int(QC.STALE_CLOSE)]
    df.loc[df.index[3], "close"] = np.nan                                 # second bar incomplete
    two = resample_2h(df)
    b0 = two.iloc[0]
    assert (b0["open"], b0["high"], b0["low"], b0["close"], b0["volume"]) == (1.0, 2.5, 0.5, 2.0, 20.0)
    assert bool(b0["is_filled"]) and b0["qc_flags"] == int(QC.FILLED)
    assert np.isnan(two.iloc[1]["close"]) and not two.iloc[1]["is_active"]


def test_download_many_records_checksum_failures(tmp_path, monkeypatch):
    from cryptograph.data.sources.binance import BinanceArchive, ChecksumError

    arch = BinanceArchive("https://example.invalid", "https://example.invalid")

    def fake(key, out_dir, refresh):
        if "bad" in key:
            raise ChecksumError("SHA-256 mismatch")
        p = out_dir / key
        out_dir.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
        return p, "0" * 64, True

    monkeypatch.setattr(arch, "_download_verified", fake)
    paths, failures = arch.download_many(["good.zip", "bad.zip"], tmp_path, workers=2)
    assert [p.name for p in paths] == ["good.zip"]
    assert failures and failures[0][0] == "bad.zip" and "mismatch" in failures[0][1]
    assert (tmp_path / "fetch_log.json").exists()
