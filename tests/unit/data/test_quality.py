import numpy as np
import pandas as pd

from cryptograph.data.quality.checks import run_checks, stale_mask
from cryptograph.data.quality.cleaning import clean_coin, gap_runs
from cryptograph.data.quality.reconcile import best_lag, reconcile_coin
from cryptograph.data.schema import QC, decode
from cryptograph.data.universe import availability_stats


def _frame(close: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99, "close": close, "volume": 1e6})
    df["close_source"] = np.where(close.notna(), "yahoo", None)
    df["xsrc_flag"] = False
    df["xsrc_resolution"] = None
    return df


def test_no_values_before_first_valid_and_gap_rules(synthetic_panel):
    c = synthetic_panel["C"]
    c.iloc[300:302] = np.nan  # short gap (2 days) -> filled
    df = run_checks(_frame(c), None, [])
    out = clean_coin(df, max_gap=2)
    first = c.first_valid_index()
    assert out.loc[: first - pd.Timedelta(days=1), ["open", "high", "low", "close", "volume"]].isna().all().all()
    assert not out.loc[: first - pd.Timedelta(days=1), "is_active"].any()
    # 5-day gap stays missing and inactive
    assert out["close"].iloc[250:255].isna().all() and not out["is_active"].iloc[250:255].any()
    assert ((out["qc_flags"].iloc[250:255] & int(QC.LONG_GAP)) > 0).all()
    # 2-day gap forward-filled and flagged
    assert out["is_filled"].iloc[300:302].all()
    assert (out["close"].iloc[300:302] == out["close"].iloc[299]).all()


def test_gap_runs():
    m = pd.Series([False, True, True, False, True, False])
    assert gap_runs(m).tolist() == [0, 2, 2, 0, 1, 0]


def test_stale_mask():
    s = pd.Series([1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 4.0])
    assert stale_mask(s, 3).tolist() == [False, True, True, True, False, False, False]


def test_large_return_contradicted_by_second_source_is_corrected():
    idx = pd.date_range("2021-01-01", periods=10, freq="D")
    y = pd.Series(100.0, index=idx)
    y.iloc[5] = 200.0  # bad tick
    cm = pd.Series(100.0, index=idx)
    out = run_checks(_frame(y), cm, [])
    assert out["close"].iloc[5] == 100.0
    assert "BAD_TICK_CORRECTED" in decode(out["qc_flags"].iloc[5])


def test_reconcile_lag_and_backfill():
    idx = pd.date_range("2020-01-01", periods=600, freq="D")
    rng = np.random.default_rng(1)
    p = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.03, 600))), index=idx)
    yahoo = _frame(p)[["open", "high", "low", "close", "volume"]].iloc[100:]
    cm = pd.DataFrame({"cm_price": p, "cm_refrate": p.shift(1), "cm_mcap": p * 1e6, "cm_volume": np.nan})
    lag, corrs = best_lag(yahoo["close"], cm["cm_refrate"])
    assert lag == 1 and corrs[1] > 0.99
    out, rep = reconcile_coin(yahoo, cm, None)
    assert rep["lag"] == 0 and rep["backfill_ok"] and rep["n_backfilled"] == 100
    assert (out["close_source"].iloc[:100] == "coinmetrics").all()
    assert out[["open", "high", "low"]].iloc[:100].isna().all().all()


def test_availability_stats():
    s = pd.Series([1.0, np.nan, np.nan, 2.0, 3.0], index=pd.date_range("2020-01-02", periods=5))
    st = availability_stats(s, pd.Timestamp("2020-01-01").date(), pd.Timestamp("2020-01-06").date())
    assert st["first_valid"] == "2020-01-02" and st["n_gaps"] == 1 and st["longest_gap"] == 2
    assert abs(st["coverage"] - 3 / 6) < 1e-12
