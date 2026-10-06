import io
import zipfile

import numpy as np
import pandas as pd

from cryptograph.data.sources.binance_futures import futures_feature_table, futures_symbols

MS = 1_000


def _zip(path, name, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(name, text)
    path.write_bytes(buf.getvalue())


def _ms(ts):
    return int(pd.Timestamp(ts).value // 1_000_000)


def test_symbol_mapping():
    assert futures_symbols("BTC") == ["BTCUSDT"]
    assert futures_symbols("SHIB") == ["1000SHIBUSDT"]
    assert futures_symbols("POL") == ["MATICUSDT", "POLUSDT"]


def test_features_are_known_at_bar_end_and_standardised_on_train(tmp_path):
    times = pd.date_range("2024-01-01", periods=48, freq="h")
    # premium klines with a header row (as in the archive): close = hour index / 1000
    lines = ["open_time,open,high,low,close,volume,close_time,quote_volume,count,taker_buy_volume,taker_buy_quote_volume,ignore"]
    for i, t in enumerate(times[:40]):                  # coin A has futures only for the first 40 hours
        lines.append(f"{_ms(t)},0,0,0,{i / 1000},0,{_ms(t) + 3_599_999},0,1,0,0,0")
    _zip(tmp_path / "premium" / "AUSDT" / "AUSDT-1h-2024-01.zip", "a.csv", "\n".join(lines))
    # funding settles at 00:00, 08:00, 16:00 with rates 1, 2, 3 (x1e-4)
    f = ["calc_time,funding_interval_hours,last_funding_rate"]
    for j, h in enumerate((0, 8, 16)):
        f.append(f"{_ms(times[0] + pd.Timedelta(hours=h))},8,{(j + 1) * 1e-4}")
    _zip(tmp_path / "funding" / "AUSDT" / "AUSDT-fundingRate-2024-01.zip", "f.csv", "\n".join(f))

    train = np.zeros(48, bool)
    train[:24] = True
    X, names = futures_feature_table(times, ["A", "B"], tmp_path, train)
    avail = names.index("futures_available")
    assert X.shape == (48, 2, len(names))
    assert X[:40, 0, avail].all() and not X[40:, 0, avail].any()
    assert not X[:, 1].any()                            # coin B has no futures data: all zeros, flag 0
    prem = X[:, 0, names.index("premium")]
    raw = np.arange(48) / 1000
    mu, sd = raw[:24].mean(), raw[:24].std()            # train rows only
    np.testing.assert_allclose(prem[:40], np.clip((raw[:40] - mu) / sd, -5, 5), rtol=1e-5)
    # funding at bar t uses the last settlement at or before the end of bar t (t + 1h): bar 06:00 ends 07:00 -> rate 1;
    # bar 07:00 ends 08:00 -> rate 2 (settled exactly at the bar end)
    fund = X[:, 0, names.index("funding_last")]
    assert fund[6] < fund[7] == fund[14] < fund[15]
