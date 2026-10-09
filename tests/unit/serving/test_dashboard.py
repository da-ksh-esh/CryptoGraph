import numpy as np
import pandas as pd
import pytest
import torch

from cryptograph.data.config import PROJECT_ROOT
from cryptograph.serving import dashboard_data as dd

from .conftest import COINS, K_HOURS

APP = str(PROJECT_ROOT / "dashboard" / "app.py")


def test_graph_at_takes_latest_snapshot_and_dedupes_edges():
    snaps = {"snapshot_t": torch.tensor([10, 34, 58]), "graph_coins": ["A", "B", "C"],
             "undirected_edge_index": [torch.tensor([[0, 1], [1, 0]]), torch.tensor([[0, 2, 1, 2], [2, 0, 2, 1]]),
                                       torch.tensor([[0], [0]])[:, :0]],
             "undirected_edge_weight": [torch.tensor([0.9, 0.9]), torch.tensor([0.8, 0.8, 0.7, 0.7]),
                                        torch.tensor([])]}
    e, bar = dd.graph_at(snaps, 40)
    assert bar == 34 and sorted(zip(e.a, e.b)) == [("A", "C"), ("B", "C")] and np.allclose(sorted(e.weight), [0.7, 0.8])
    e, bar = dd.graph_at(snaps, 5)
    assert bar is None and e.empty
    e, bar = dd.graph_at(snaps, 100)
    assert bar == 58 and e.empty
    pos = dd.layout(["A", "B", "C"], dd.graph_at(snaps, 40)[0])
    assert set(pos) == {"A", "B", "C"} and pos == dd.layout(["A", "B", "C"], dd.graph_at(snaps, 40)[0])


def test_risk_matrix_drivers_duration_and_metrics():
    risk = pd.DataFrame({"coin": ["A", "A", "B", "B"], "horizon_h": [1, 6, 1, 6], "risk": [0.1, 0.2, 0.05, 0.6]})
    m = dd.risk_matrix(risk)
    assert m.index.tolist() == ["B", "A"] and m.columns.tolist() == [1, 6]
    risk["lift"] = risk["risk"] / risk["horizon_h"].map({1: 0.02, 6: 0.5})   # A: 5x at 1 h beats B's 1.2x at 6 h
    assert dd.risk_matrix(risk).index.tolist() == ["A", "B"]
    d = dd.drivers_frame({"drivers": [{"input": "x", "attribution": 0.1, "direction": "raises"},
                                      {"input": "y", "attribution": -0.5, "direction": "lowers"}]})
    assert d.input.tolist() == ["y", "x"] and dd.drivers_frame({}).empty
    per = pd.DataFrame({"type": ["partial"], "direction": ["crash"], "median_hours_to_normal": [9.0]})
    assert dd.typical_duration(per, "partial", "crash")["median_hours_to_normal"] == 9.0
    assert dd.typical_duration(per, "partial", "pump") is None and dd.typical_duration(pd.DataFrame(), "a", "b") is None
    tm = dd.model_test_metrics({"test_auprc_k1h": 0.2, "test_posrate_k1h": 0.1, "test_auprc_k6h": 0.5}, [1, 6])
    assert tm.horizon_h.tolist() == [1] and np.isclose(tm.lift.iloc[0], 2.0)


@pytest.fixture
def predicted(toy_project):
    from cryptograph.training.runner import run_experiment

    root, run, _ = toy_project
    cfg = {"name": "predict_toy", "model": {"name": "predict"}, "device": "cpu",
           "predict": {"target_experiment": "toy_tgn", "ig_steps": 4, "n_alerts": 6}}
    out = run_experiment(cfg, seed=0, runs_dir=root / "runs")
    return root, root / "runs" / out["run_id"]


def test_load_prediction_reads_a_predict_run(predicted):
    root, rd = predicted
    assert dd.prediction_runs(root / "runs") == [rd]
    P = dd.load_prediction(rd)
    assert P.summary["k_hours"] == K_HOURS and set(P.risk.coin) <= set(COINS) and len(P.alerts) == 6
    assert not P.sources.empty and {"type", "direction"} <= set(P.sources.columns)
    assert dd.prediction_runs(root / "missing") == []


def _app(monkeypatch, root):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("CRYPTOGRAPH_ROOT", str(root))
    monkeypatch.setenv("CRYPTOGRAPH_RUNS_DIR", str(root / "runs"))
    return AppTest.from_file(APP, default_timeout=60).run()


def test_dashboard_renders_a_prediction_run(predicted, monkeypatch):
    root, rd = predicted
    at = _app(monkeypatch, root)
    assert not at.exception, at.exception
    assert [t.label for t in at.tabs] == ["Risk per coin", "Riskiest alerts", "Shock duration", "Network",
                                          "How good is the model"]
    as_of = pd.Timestamp(dd.load_prediction(rd).summary["as_of"])
    assert at.metric[0].value == as_of.strftime("%d %b %H:%M") and len(at.metric[0].value) <= 12
    assert at.metric[0].label == f"As of (UTC, {as_of.year})"
    assert "pumpes" not in " ".join(m.value for m in at.markdown)
    assert len(at.expander) >= 3
    at.radio[0].set_value(1).run()                             # switch horizon
    assert not at.exception


def test_dashboard_without_runs_explains_what_to_do(tmp_path, monkeypatch):
    at = _app(monkeypatch, tmp_path)
    assert not at.exception and "cryptograph-predict" in at.info[0].value
