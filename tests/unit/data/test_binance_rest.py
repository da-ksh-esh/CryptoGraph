import io
import json
import urllib.error

import pandas as pd
import pytest

from cryptograph.data.sources.binance_rest import BinanceRestKlines, SymbolError, klines_frame, to_ms

H = 3_600_000


def row(open_ms: int, close: float = 1.0) -> list:
    """One REST kline row (open time in ms; values as strings, like the API)."""
    return [open_ms, "1.0", "2.0", "0.5", str(close), "10.0", open_ms + H - 1, "10.0", 7, "5.0", "5.0", "0"]


def test_klines_frame_parses_rows_and_drops_the_bar_still_trading():
    t0 = to_ms("2026-10-01 00:00")
    rows = [row(t0 + i * H, close=float(i + 1)) for i in range(3)]
    df = klines_frame(rows, now_ms=t0 + 2 * H + 30 * 60_000)              # third bar is half way through its hour
    assert list(df.index) == list(pd.date_range("2026-10-01", periods=2, freq="h"))
    assert df.index.name == "time" and df["close"].tolist() == [1.0, 2.0] and df["trades"].dtype.kind == "i"
    assert list(df.columns) == ["open", "high", "low", "close", "volume", "quote_volume", "trades"]
    assert klines_frame([]).empty and list(klines_frame([]).columns) == list(df.columns)


def test_to_ms_reads_naive_and_aware_utc():
    assert to_ms("1970-01-01 01:00") == H and to_ms(pd.Timestamp("1970-01-01 02:00", tz="UTC")) == 2 * H
    assert to_ms(pd.Timestamp("1970-01-01 05:30", tz="Asia/Kolkata")) == 0


class FakeAPI:
    """Serves klines for BTCUSDT from a list of open times; records calls; can fail per base URL."""

    def __init__(self, opens, fail=None):
        self.opens, self.fail, self.calls = opens, fail or {}, []

    def __call__(self, url, timeout):
        self.calls.append(url)
        base = url.split("/api/")[0]
        if base in self.fail:
            err = self.fail[base]
            if isinstance(err, int):
                raise urllib.error.HTTPError(url, err, "err", {}, io.BytesIO(b'{"msg": "nope"}'))
            raise err
        q = dict(p.split("=") for p in url.split("?")[1].split("&"))
        if q["symbol"] != "BTCUSDT":
            raise urllib.error.HTTPError(url, 400, "bad", {}, io.BytesIO(b'{"code": -1121, "msg": "Invalid symbol."}'))
        lo, hi, lim = int(q["startTime"]), int(q["endTime"]), int(q["limit"])
        return json.dumps([row(o) for o in self.opens if lo <= o <= hi][:lim]).encode()


def test_klines_paginates_and_keeps_closed_bars_only():
    t0 = to_ms("2026-10-01 00:00")
    api = FakeAPI([t0 + i * H for i in range(25)])
    c = BinanceRestKlines(base_urls=["https://a"], limit=10, backoff=0, opener=api)
    df = c.klines("BTCUSDT", "2026-10-01 00:00", "2026-10-02 00:00", now="2026-10-01 23:10")
    assert len(api.calls) == 3                                   # 10 + 10 + 5 rows
    assert df.index[0] == pd.Timestamp("2026-10-01 00:00") and df.index[-1] == pd.Timestamp("2026-10-01 22:00")
    assert df.index.is_unique and df.index.is_monotonic_increasing


def test_unknown_symbol_fails_fast_and_dead_mirror_falls_back():
    t0 = to_ms("2026-10-01 00:00")
    api = FakeAPI([t0], fail={"https://a": urllib.error.URLError("down")})
    c = BinanceRestKlines(base_urls=["https://a", "https://b"], retries=2, backoff=0, opener=api)
    assert len(c.klines("BTCUSDT", "2026-10-01", "2026-10-01 05:00", now="2026-10-02")) == 1
    assert sum(u.startswith("https://a") for u in api.calls) == 2 and api.calls[-1].startswith("https://b")
    api.calls.clear()
    with pytest.raises(SymbolError) as e:
        c.klines("NOPEUSDT", "2026-10-01", "2026-10-01 05:00", now="2026-10-02")
    assert e.value.code == 400 and len(api.calls) == 3                 # a: 2 network failures, b: one 400, no retry


def test_region_blocked_endpoint_moves_on_and_all_blocked_is_an_error():
    t0 = to_ms("2026-10-01 00:00")
    api = FakeAPI([t0], fail={"https://a": 451})
    c = BinanceRestKlines(base_urls=["https://a", "https://b"], backoff=0, opener=api)
    assert len(c.klines("BTCUSDT", "2026-10-01", "2026-10-01 05:00", now="2026-10-02")) == 1
    api.fail["https://b"] = 403
    with pytest.raises(RuntimeError, match="no Binance REST endpoint answered"):
        c.klines("BTCUSDT", "2026-10-01", "2026-10-01 05:00", now="2026-10-02")


@pytest.mark.network
def test_live_binance_returns_closed_hourly_bars():
    c = BinanceRestKlines(retries=1, timeout=10)
    end = pd.Timestamp.now(tz="UTC").tz_localize(None).floor("h")
    try:
        df = c.klines("BTCUSDT", end - pd.Timedelta(hours=5), end)
    except Exception as e:                                            # offline, or the API is blocked here
        pytest.skip(f"Binance REST not reachable: {e}")
    assert 1 <= len(df) <= 5 and df.index.max() < end and (df["close"] > 0).all()
