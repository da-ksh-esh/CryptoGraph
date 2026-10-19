import numpy as np
import pandas as pd
import pytest

from cryptograph.explain.run import case_windows

TIMES = pd.date_range("2024-01-01", periods=24 * 30, freq="h")
SPLIT = np.r_[np.zeros(24 * 10), np.ones(24 * 10), np.full(24 * 10, 2)].astype(int)   # 10 days each


def test_windows_come_from_the_config_in_its_order():
    cases = [{"name": "late (test period)", "start": "2024-01-25", "end": "2024-01-26"},
             {"name": "early crash", "start": "2024-01-02", "end": "2024-01-03 12:00"}]
    w = case_windows(cases, TIMES, SPLIT)
    assert list(w) == ["late (test period)", "early crash"]
    assert w["early crash"] == ("2024-01-02", "2024-01-03 12:00")
    assert case_windows([], TIMES, SPLIT) == {}


def test_a_stated_period_must_match_the_splits():
    with pytest.raises(ValueError, match="labelled 'train period' but its hours fall in \['test'\]"):
        case_windows([{"name": "X (train period)", "start": "2024-01-25", "end": "2024-01-26"}], TIMES, SPLIT)
    with pytest.raises(ValueError, match="fall in \['train', 'validation'\]"):                # spans two periods
        case_windows([{"name": "Y (validation period)", "start": "2024-01-10", "end": "2024-01-12"}], TIMES, SPLIT)
    assert case_windows([{"name": "Z (Nov 2022, validation period)", "start": "2024-01-12", "end": "2024-01-13"}],
                        TIMES, SPLIT)


def test_a_window_outside_the_data_is_refused():
    with pytest.raises(ValueError, match="has no hours in the data"):
        case_windows([{"name": "future", "start": "2030-01-01", "end": "2030-01-02"}], TIMES, SPLIT)


def test_the_tempme_config_cases_match_the_real_splits_when_the_data_is_there():
    import yaml

    from cryptograph.data.config import PROJECT_ROOT
    from cryptograph.data.storage import load_pt

    panel_path = PROJECT_ROOT / "data" / "processed" / "v2" / "1h" / "panel.pt"
    if not panel_path.exists():
        pytest.skip("processed v2 data not built")
    cfg = yaml.safe_load((PROJECT_ROOT / "configs" / "experiment" / "tempme_tgn_1h.yaml").read_text(encoding="utf-8"))
    p = load_pt(panel_path)
    w = case_windows(cfg["explain"]["cases"], pd.DatetimeIndex(p["times"]), p["split_id"].numpy())
    assert list(w) == ["LUNA collapse (May 2022, train period)", "FTX collapse (Nov 2022, validation period)",
                       "10-Oct-2025 liquidation cascade (test period)"]         # notebook 04 reads these names
