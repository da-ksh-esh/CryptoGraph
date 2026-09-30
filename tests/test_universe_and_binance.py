import pytest
import hashlib
import pandas as pd
import numpy as np

from cryptograph.data.universe import (
    TARGET_UNIVERSE_30,
    EVENT_COINS,
    SECTOR_TAXONOMY,
    get_universe,
    get_binance_symbol,
    generate_universe_audit_report
)
from cryptograph.data.sources.binance import BinanceArchiveDownloader
from cryptograph.data.quality.binance_continuity import ContinuityAuditor

def test_universe_specifications():
    univ = get_universe()
    assert len(univ) == 30
    assert len(EVENT_COINS) == 2
    for coin in univ:
        assert coin in SECTOR_TAXONOMY
        assert get_binance_symbol(coin) == f"{coin}USDT"

def test_universe_audit_report(tmp_path):
    audit_file = generate_universe_audit_report(reports_dir=str(tmp_path))
    assert audit_file.exists()
    df = pd.read_csv(audit_file)
    assert len(df) >= 40
    assert "coin" in df.columns
    assert "reason" in df.columns
    assert "XMR" in df["coin"].values
    assert df.loc[df["coin"] == "XMR", "status"].values[0] == "Excluded"

def test_binance_archive_downloader():
    downloader = BinanceArchiveDownloader()
    data_url, chk_url = downloader.get_monthly_url("BTCUSDT", 2024, 1, "1h")
    assert "BTCUSDT-1h-2024-01.zip" in data_url
    assert chk_url == f"{data_url}.CHECKSUM"

    # Test checksum logic
    dummy_bytes = b"Binance Public Archive Kline Mock Data"
    expected_hash = hashlib.sha256(dummy_bytes).hexdigest()
    checksum_text = f"{expected_hash}  BTCUSDT-1h-2024-01.zip\n"
    assert downloader.verify_sha256(dummy_bytes, checksum_text)

def test_resample_1h_to_2h():
    dates = pd.date_range("2024-01-01 00:00", periods=4, freq="1h", tz="UTC")
    df_1h = pd.DataFrame({
        "open": [10.0, 11.0, 12.0, 13.0],
        "high": [12.0, 13.0, 14.0, 15.0],
        "low": [9.0, 10.5, 11.5, 12.5],
        "close": [11.0, 12.0, 13.0, 14.0],
        "volume": [100, 200, 300, 400],
        "quote_volume": [1100, 2300, 3700, 5300],
        "count": [10, 20, 30, 40]
    }, index=dates)

    df_2h = BinanceArchiveDownloader.resample_1h_to_2h(df_1h)
    assert len(df_2h) == 2
    # First 2h bar (00:00 - 02:00)
    assert df_2h.iloc[0]["open"] == 10.0
    assert df_2h.iloc[0]["high"] == 13.0
    assert df_2h.iloc[0]["low"] == 9.0
    assert df_2h.iloc[0]["close"] == 12.0
    assert df_2h.iloc[0]["volume"] == 300.0

def test_maintenance_gap_and_peg_check():
    # Gap fill test
    s = pd.Series([100.0, np.nan, np.nan, np.nan, 105.0])
    filled, flags = ContinuityAuditor.repair_maintenance_gaps(s, max_ffill_hours=2)
    assert filled.iloc[1] == 100.0
    assert filled.iloc[2] == 100.0
    assert np.isnan(filled.iloc[3])  # >2h gap remains NaN

    # Peg test
    pre_cutoff = pd.Timestamp("2026-01-15", tz="UTC")
    res_ok = ContinuityAuditor.check_usdt_peg_status(pre_cutoff, 0.998)
    assert res_ok["status"] == "peg_ok"

    res_depeg = ContinuityAuditor.check_usdt_peg_status(pre_cutoff, 0.95)
    assert res_depeg["status"] == "depeg_alert"

    post_cutoff = pd.Timestamp("2026-06-01", tz="UTC")
    res_post = ContinuityAuditor.check_usdt_peg_status(post_cutoff, 1.0)
    assert res_post["status"] == "no_peg_check"
