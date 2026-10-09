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
    assert [t.label for t in at.tabs] == ["📊 Who's at risk", "🔍 Why", "📜 Past crises", "🕸️ Market map",
                                          "ℹ️ How it works"]
    as_of = pd.Timestamp(dd.load_prediction(rd).summary["as_of"])
    assert any(as_of.strftime("%d %b %Y, %H:%M UTC") in m.value for m in at.markdown)
    assert at.metric[0].label == "Most likely to move next (6 hours)" and at.metric[0].value.split(":")[0] in COINS
    assert "pumpes" not in " ".join(m.value for m in at.markdown)
    assert any(e.label.startswith("Details: every time window") for e in at.expander)
    assert "dropped sharply" in " ".join(m.value for m in at.markdown) or "jumped sharply" in         " ".join(m.value for m in at.markdown)
    at.select_slider(key="horizon").set_value(1).run()          # switch the time window
    assert not at.exception, at.exception
    assert at.metric[0].label == "Most likely to move next (1 hour)"
    at.selectbox(key="why_coin").set_value(at.selectbox(key="why_coin").options[-1]).run()
    assert not at.exception, at.exception
    assert any("the risk)" in m.value for m in at.markdown)                 # plain-language reasons


def test_dashboard_without_runs_explains_what_to_do(tmp_path, monkeypatch):
    at = _app(monkeypatch, tmp_path)
    assert not at.exception and "cryptograph-predict" in at.info[0].value


def test_circle_layout_groups_sectors_and_is_stable():
    sectors = {"A": "y", "B": "x", "C": "x", "D": "y"}
    pos = dd.circle_layout(["A", "B", "C", "D"], sectors)
    assert pos == dd.circle_layout(["D", "C", "B", "A"], sectors)            # same place regardless of input order
    assert np.allclose(pos["B"], (0.0, 1.0))                                 # first of sector x at the top
    assert all(abs(np.hypot(*p) - 1) < 1e-12 for p in pos.values())
    order = sorted(pos, key=lambda c: (-np.arctan2(pos[c][1], pos[c][0]) + np.pi / 2) % (2 * np.pi))
    assert order == ["B", "C", "A", "D"]                                     # clockwise: x then y


def test_node_states_and_outcome_table():
    frames = [{"shocked": [], "active": [0, 1]}, {"shocked": [0], "active": [0, 1, 2]},
              {"shocked": [1], "active": [0, 1, 2]}]
    st = dd.node_states(frames, 3)
    assert st[0] == ["not yet", "not yet", "inactive"]
    assert st[1] == ["shocking now", "not yet", "not yet"]
    assert st[2] == ["shocked earlier in the window", "shocking now", "not yet"]
    risk = pd.DataFrame({"coin": ["A", "B", "C", "A"], "sector": ["x", "y", "x", "x"], "horizon_h": [6, 6, 6, 1],
                         "risk": [0.2, 0.5, 0.3, 0.9], "score": [0.0] * 4, "lift": [1.0, 2.5, 1.5, 3.0],
                         "actual": [1.0, 0.0, np.nan, 1.0], "source": ["B"] * 4,
                         "drivers": [[{"input": "f", "attribution": 0.2}], [{"input": "g", "attribution": -0.1}], [],
                                     []]})
    t = dd.outcome_table(risk, 6)
    assert t.coin.tolist() == ["B", "C", "A"] and t.outcome.tolist() == ["no shock", "not observed", "shocked"]
    assert t.drivers.tolist() == ["g (−)", "", "f (+)"]
    assert dd.outcome_table(risk, 6, ["x"]).coin.tolist() == ["C", "A"] and "score" in t.columns


@pytest.fixture
def with_events(predicted):
    from cryptograph.training.runner import run_experiment

    root, rd = predicted
    cfg = {"name": "events_toy", "model": {"name": "events"}, "device": "cpu",
           "events": {"target_experiment": "toy_tgn", "ig_steps": 4, "before_hours": 6, "after_hours": 24}}
    out = run_experiment(cfg, seed=0, runs_dir=root / "runs")
    return root, root / "runs" / out["run_id"]


def test_event_explorer_renders_and_switches_events(with_events, monkeypatch):
    root, ev_dir = with_events
    assert dd.latest_events_dir(root / "runs") == ev_dir and dd.prediction_runs(root / "runs")    # both kinds found
    at = _app(monkeypatch, root)
    assert not at.exception, at.exception
    box = at.selectbox(key="event")
    assert [o.split(" · ")[1] for o in box.options] == ["early", "late"]
    assert any("Not a fair test" in w.value for w in at.warning)
    late_id = dd.load_events_index(ev_dir)["events"][1]["id"]
    box.set_value(late_id).run()
    assert not at.exception, at.exception
    assert any("A real test" in s.value for s in at.success)
    assert any("flagged" in m.value and "at random" in m.value for m in at.markdown)
    at.multiselect(key="event_sectors").select("x").run()
    assert not at.exception, at.exception


def test_plain_language_helpers():
    assert dd.risk_level(1.6)[0] == "Much higher than usual" and dd.risk_level(1.2)[0] == "Higher than usual"
    assert dd.risk_level(1.0)[0] == "About normal" and dd.risk_level(0.5)[0] == "Lower than usual"
    assert dd.risk_level(None)[0] == "Unknown" and dd.risk_level(float("nan"))[0] == "Unknown"
    assert dd.horizon_label(1) == "1 hour" and dd.horizon_label(24) == "24 hours"
    up = {"input": "receiver coin: volume_rel_168h", "attribution": 0.3}
    assert dd.plain_driver(up, "JUP", "QNT") == "JUP: trading volume compared with its past week (raises the risk)"
    down = {"input": "source coin: network: dy_net", "attribution": -0.1}
    assert dd.plain_driver(down, "JUP", "QNT").startswith("QNT: whether it tends to spread") and "lowers" in \
        dd.plain_driver(down, "JUP", "QNT")
    ev = {"input": "event: contemp_corr", "attribution": 0.2}
    assert dd.plain_driver(ev, "JUP", "QNT") == ("how closely the two coins move together (from QNT to JUP) "
                                                 "(raises the risk)")
    assert dd.plain_driver({"input": "receiver coin: futures: 0", "attribution": 0.1}, "A", "B") == \
        "A: futures: 0 (raises the risk)"                                      # unknown inputs stay readable
    solo = dd.shock_sentence({"coin": "QNT", "direction": "crash", "z": -4.8, "breadth": 0.0},
                             {"median_hours_to_normal": 8.0})
    assert "QNT** dropped sharply" in solo and "4.8 times" in solo and "on its own" in solo and "8 hours" in solo
    wide = dd.shock_sentence({"coin": "BTC", "direction": "pump", "z": 3.1, "breadth": 0.6}, None)
    assert "jumped" in wide and "60% of the other coins" in wide and "market-wide" in wide and "hours" not in wide
    assert dd.crisis_verdict(5, 5, 5.0, 1.0)[0] == "neutral"                # everyone moved: says nothing
    assert dd.crisis_verdict(3, 5, 1.6, 0.32)[0] == "good"
    assert dd.crisis_verdict(2, 5, 1.9, 0.38)[0] == "neutral"
    assert dd.crisis_verdict(0, 5, 2.5, 0.5)[0] == "weak"
