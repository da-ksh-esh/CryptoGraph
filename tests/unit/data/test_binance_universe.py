import hashlib
import io
import zipfile
from datetime import date
from types import SimpleNamespace

import pandas as pd
import pytest

from cryptograph.data.config import is_hourly_config, load_hourly_config
from cryptograph.data.sources.binance import ChecksumError, SymbolFiles, parse_checksum, read_klines, verify_bytes
from cryptograph.data.universe_hourly import (Segment, clip_intervals, coverage_summary, decide, file_intervals,
                                              hours_in, merge_intervals)


def _zip(rows: list[str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("X-1h.csv", "\n".join(rows) + "\n")
    return buf.getvalue()


def test_read_klines_handles_milliseconds_and_microseconds():
    ms = "1502942400000,4261.48,4313.62,4261.32,4308.83,47.18,1502945999999,202366.1,171,35.1,150952.4,0"
    us = "1785542400000000,62887.88,63024.04,62887.87,62938.01,517.26,1785545999999999,32568037.4,40193,232.5,1.0,0"
    df = read_klines(_zip([ms, us]))
    assert list(df.index) == [pd.Timestamp("2017-08-17 04:00"), pd.Timestamp("2026-08-01 00:00")]
    assert df["close"].tolist() == [4308.83, 62938.01]


def test_read_klines_skips_header_row():
    rows = ["open_time,open,high,low,close,volume,close_time,quote_volume,count,tbb,tbq,ignore",
            "1502942400000,1,2,0.5,1.5,10,1502945999999,15,3,5,7,0"]
    assert len(read_klines(_zip(rows))) == 1


def test_checksum_verification():
    blob = b"hello"
    good = hashlib.sha256(blob).hexdigest()
    assert parse_checksum(f"{good}  BTCUSDT-1h-2024-01.zip\n") == good
    verify_bytes(blob, good)
    with pytest.raises(ChecksumError):
        verify_bytes(blob + b"!", good)
    with pytest.raises(ChecksumError):
        parse_checksum("not-a-checksum")


def test_interval_arithmetic():
    files = SymbolFiles("X", months=("2024-01", "2024-02"), days=("2024-03-01",))
    iv = merge_intervals(file_intervals(files))
    assert iv == [(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-03-02"))]
    assert hours_in(iv, pd.Timestamp("2024-02-01"), pd.Timestamp("2024-02-02")) == 24
    assert clip_intervals(iv, pd.Timestamp("2025-01-01"), pd.Timestamp("2025-02-01")) == []


def test_segment_until_never_extends_past_cutoff():
    """LUNAUSDT is reused for the new LUNA after the collapse: the old-LUNA segment must stop at `until`."""
    old = Segment("LUNAUSDT", SymbolFiles("LUNAUSDT", months=("2022-04", "2022-05", "2022-06", "2022-07"), days=()),
                  first_open=pd.Timestamp("2022-04-01"), last_open=pd.Timestamp("2022-07-31 23:00"),
                  valid_until=date(2022, 5, 13))
    new = Segment("LUNCUSDT", SymbolFiles("LUNCUSDT", months=("2022-09",), days=()),
                  first_open=pd.Timestamp("2022-09-09 10:00"), last_open=pd.Timestamp("2022-09-30 23:00"))
    iv = merge_intervals([i for s in (old, new) for i in s.intervals()])
    assert iv[0][1] == pd.Timestamp("2022-05-14")                  # stops after 13 May
    assert iv[1][0] == pd.Timestamp("2022-09-09 10:00")            # LUNC resumes, gap stays a gap
    s = coverage_summary([old, new], (pd.Timestamp("2022-04-01"), pd.Timestamp("2022-10-01")),
                         (pd.Timestamp("2022-09-01"), pd.Timestamp("2022-10-01")))
    assert s["n_file_gaps"] == 1 and s["test_coverage"] < 1


def test_rules_in_order():
    rules = SimpleNamespace(quote_asset="USDT", min_history_days=730, min_test_coverage=0.9)
    base = {"binance_symbol": "XUSDT", "history_days": 1000, "test_coverage": 1.0, "identity_ok": True,
            "hourly_first": pd.Timestamp("2020-01-01"), "hourly_last": pd.Timestamp("2026-09-27")}
    assert decide(base, rules) == (True, "passes")
    assert "no Binance" in decide({**base, "binance_symbol": None}, rules)[1]
    assert "days of hourly data" in decide({**base, "history_days": 500}, rules)[1]
    # a delisted market fails on coverage, not on identity (its old price cannot be compared with today's)
    delisted = {**base, "test_coverage": 0.05, "identity_ok": False}
    assert "test-period coverage" in decide(delisted, rules)[1]
    assert "different asset" in decide({**base, "identity_ok": False}, rules)[1]


def test_hourly_config_loads_and_is_detected():
    cfg = load_hourly_config()
    assert cfg.resolution == "1h" and cfg.universe.target_size == 30
    assert cfg.splits.embargo_days == 7
    assert [s.symbol for s in cfg.universe.event_coins["LUNC"]] == ["LUNAUSDT", "LUNCUSDT"]
    assert cfg.universe.event_coins["LUNC"][0].until == date(2022, 5, 13)
    assert is_hourly_config("configs/data/hourly.yaml") and not is_hourly_config("configs/data/daily.yaml")
