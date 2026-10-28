import json
import shutil
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from cryptograph.data.config import SymbolOverride
from cryptograph.data.sources.binance_rest import SymbolError
from cryptograph.serving import live

from .conftest import COINS, K_HOURS


def test_newest_closed_hour_and_live_config():
    assert live.newest_closed_hour(pd.Timestamp("2026-10-26 19:05")) == pd.Timestamp("2026-10-26 18:00")
    assert live.newest_closed_hour(pd.Timestamp("2026-10-27 00:35", tz="Asia/Kolkata")) == pd.Timestamp("2026-10-26 18:00")
    lc = live.LiveConfig()
    ref, cfg = live.data_configs(lc, pd.Timestamp("2026-10-26 18:00"))
    assert ref.dataset_version == "v2" and cfg.dataset_version == "v2_live"
    assert cfg.window.end.replace(tzinfo=None) == pd.Timestamp("2026-10-26 18:00").to_pydatetime()
    assert cfg.window.start == ref.window.start and cfg.splits == ref.splits and cfg.features == ref.features
    assert cfg.datasets.build_resolutions == ["1h"] and cfg.interim_dir != ref.interim_dir
    assert cfg.reports_dir != ref.reports_dir and cfg.processed_dir("1h").parent.name == "v2_live"
    with pytest.raises(ValueError, match="not the reference"):
        live.data_configs(live.LiveConfig(reference_version="v9"), pd.Timestamp("2026-10-26"))
    with pytest.raises(Exception):
        live.LiveConfig(unknown_key=1)                                    # config typos are errors
    assert live.load_live_config().dataset_version == "v2_live"           # configs/serving/live_1h.yaml


class FakeClient:
    def __init__(self, refuse=()):
        self.calls, self.refuse = [], set(refuse)

    def klines(self, symbol, start, end):
        self.calls.append((symbol, pd.Timestamp(start), pd.Timestamp(end)))
        if symbol in self.refuse:
            raise SymbolError("HTTP 400: Invalid symbol", 400)
        idx = pd.date_range(start, end, freq="h", name="time")
        return pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 2.0, "volume": 5.0, "quote_volume": 10.0,
                             "trades": 3}, index=idx)


def test_fetch_new_bars_appends_closed_bars_and_requests_only_missing_hours(tmp_path, monkeypatch):
    import cryptograph.pipelines.stages.hourly_fetch as hf

    times = pd.date_range("2026-10-01", periods=3, freq="h", name="time")
    ref_raw = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0, "quote_volume": 1.0,
                            "trades": 1, "symbol": "x"},
                           index=pd.MultiIndex.from_product([times, ["AAA", "BBB", "OLD", "BAD"]], names=["time", "coin"]))
    ref = SimpleNamespace(interim_dir=tmp_path / "ref")
    lv = SimpleNamespace(interim_dir=tmp_path / "live")
    ref.interim_dir.mkdir()
    lv.interim_dir.mkdir()
    ref_raw.to_parquet(ref.interim_dir / "raw_1h.parquet")
    hist = {"AAA": [SymbolOverride(symbol="AAAUSDT")], "BBB": [SymbolOverride(symbol="BUSDT", until="2025-01-01"),
                                                              SymbolOverride(symbol="BBBUSDT")],
            "OLD": [SymbolOverride(symbol="OLDUSDT", until="2026-01-01")], "BAD": [SymbolOverride(symbol="BADUSDT")]}
    monkeypatch.setattr(hf, "hourly_universe", lambda cfg: (list(hist), hist, {}))
    c = FakeClient(refuse={"BADUSDT"})
    info = live.fetch_new_bars(ref, lv, c, pd.Timestamp("2026-10-01 05:00"), max_gap_days=90)
    assert {s for s, *_ in c.calls} == {"AAAUSDT", "BBBUSDT", "BADUSDT"}            # closed history: not fetched
    assert all(a == pd.Timestamp("2026-10-01 03:00") for _, a, _ in c.calls)
    assert info["new_bars"] == 6 and info["data_end"] == "2026-10-01 05:00:00" and "BAD" in info["failures"]
    raw = pd.read_parquet(lv.interim_dir / "raw_1h.parquet")
    assert len(raw) == len(ref_raw) + 6 and raw.loc[(pd.Timestamp("2026-10-01 04:00"), "BBB"), "symbol"] == "BBBUSDT"
    assert raw.loc[ref_raw.index].equals(ref_raw)                                    # research bars untouched
    c.calls.clear()
    info = live.fetch_new_bars(ref, lv, c, pd.Timestamp("2026-10-01 06:00"), max_gap_days=90)
    assert {(s, a) for s, a, _ in c.calls} == {("AAAUSDT", pd.Timestamp("2026-10-01 06:00")),
                                               ("BBBUSDT", pd.Timestamp("2026-10-01 06:00")),
                                               ("BADUSDT", pd.Timestamp("2026-10-01 03:00"))}
    assert info["new_bars"] == 2 and info["live_bars_total"] == 8
    with pytest.raises(RuntimeError, match="archive fetch stage"):
        live.fetch_new_bars(ref, lv, c, pd.Timestamp("2027-06-01"), max_gap_days=90)


def _dataset(path, T, extra_events=0, seed=0):
    """Minimal processed dataset: what verify_against_reference reads."""
    from torch_geometric.data import TemporalData

    rng = np.random.default_rng(seed)
    path.mkdir(parents=True)
    X = torch.from_numpy(rng.normal(size=(T, 3, 2)).astype(np.float32))
    z = rng.normal(size=(T, 3)).astype(np.float32)
    z[0] = np.nan
    torch.save({"times": [str(t) for t in pd.date_range("2026-01-01", periods=T, freq="h")], "coins": ["A", "B", "C"],
                "feature_names": ["f0", "f1"], "shock_threshold": 2.29, "split_id": torch.zeros(T, dtype=torch.int8),
                "X_raw": X.clone(), "X": X, "shock_z": torch.from_numpy(z),
                "shock_today": torch.from_numpy(np.abs(z) > 1.5), "shock_today_mask": torch.ones(T, 3, dtype=torch.bool),
                "active": torch.ones(T, 3, dtype=torch.bool)}, path / "panel.pt")
    (path / "seasonality.json").write_text(json.dumps({"hour": [1.0, 2.0]}))
    torch.save({"undirected_threshold": 0.75}, path / "graph_snapshots.pt")
    t = np.repeat(np.arange(T), 2)
    E = len(t)
    torch.save({"full": TemporalData(src=torch.zeros(E, dtype=torch.long), dst=torch.ones(E, dtype=torch.long),
                                     t=torch.from_numpy(t), msg=torch.from_numpy(rng.normal(size=(E, 5)).astype(np.float32)))},
               path / "tgn_events.pt")


def _extend(ref, out, extra):
    """`out` = `ref` plus `extra` new bars (and their events), like a live rebuild."""
    shutil.copytree(ref, out)
    p, e = torch.load(ref / "panel.pt", weights_only=False), torch.load(ref / "tgn_events.pt", weights_only=False)
    T = len(p["times"])
    p["times"] += [str(pd.Timestamp(p["times"][-1]) + pd.Timedelta(hours=k + 1)) for k in range(extra)]
    for k in ("X_raw", "X", "shock_z", "shock_today", "shock_today_mask", "active", "split_id"):
        p[k] = torch.cat([p[k], p[k][-1:].repeat(extra, *[1] * (p[k].dim() - 1))])
    p["split_id"][T:] = -1
    torch.save(p, out / "panel.pt")
    d = e["full"]
    from torch_geometric.data import TemporalData

    t = torch.arange(T, T + extra)
    e["full"] = TemporalData(src=torch.cat([d.src, torch.zeros(extra, dtype=torch.long)]),
                             dst=torch.cat([d.dst, torch.ones(extra, dtype=torch.long)]),
                             t=torch.cat([d.t, t]), msg=torch.cat([d.msg, torch.zeros(extra, 5)]))
    torch.save(e, out / "tgn_events.pt")
    return p


def test_verify_against_reference_accepts_extensions_and_catches_changes(tmp_path):
    ref = tmp_path / "v2"
    _dataset(ref, T=50)
    out = tmp_path / "v2_live"
    _extend(ref, out, extra=10)
    r = live.verify_against_reference(ref, out, grace=3)
    assert all(r["checks"].values()) and r["bars_compared"] == 47 and r["live_bars"] == 10
    assert r["events_compared"] == 94 and r["tail_bars_changed"] == 0

    p = torch.load(out / "panel.pt", weights_only=False)
    p["X"][48, 0, 0] += 1.0                                  # inside the grace tail: allowed, reported
    torch.save(p, out / "panel.pt")
    assert live.verify_against_reference(ref, out, grace=3)["tail_bars_changed"] == 1
    p["X"][10, 1, 1] += 1.0                                  # a research bar: not allowed
    torch.save(p, out / "panel.pt")
    with pytest.raises(live.LiveDataMismatch, match=r"\['X'\]"):
        live.verify_against_reference(ref, out, grace=3)

    shutil.rmtree(out)
    _extend(ref, out, extra=10)
    (out / "seasonality.json").write_text(json.dumps({"hour": [1.0, 2.5]}))
    e = torch.load(out / "tgn_events.pt", weights_only=False)
    e["full"].msg[5, 0] += 1
    torch.save(e, out / "tgn_events.pt")
    with pytest.raises(live.LiveDataMismatch, match="seasonality.*shock_events"):
        live.verify_against_reference(ref, out, grace=3)


def _predict_run(path, as_of="2026-03-01 10:00:00", bar=10):
    path.mkdir(parents=True)
    pd.DataFrame({"coin": ["A", "B", "C", "A", "B", "C"], "horizon_h": [1, 1, 1, 3, 3, 3],
                  "risk": [0.3, 0.2, 0.1, 0.5, 0.4, 0.3], "lift": [3.0, 2.0, 1.0, 2.5, 2.0, 1.5],
                  "base_rate": [0.1] * 3 + [0.2] * 3, "source": ["S"] * 6}).to_csv(path / "risk.csv", index=False)
    (path / "summary.json").write_text(json.dumps({"as_of": as_of, "as_of_bar": bar}))
    return path


def test_alert_log_scores_alerts_as_their_windows_close(tmp_path):
    rd = _predict_run(tmp_path / "run1")
    a = live.new_alerts(rd, top=2)
    assert a[["coin", "horizon_h"]].values.tolist() == [["A", 1], ["B", 1], ["A", 3], ["B", 3]]
    assert a["due_at"].tolist()[0] == "2026-03-01 11:00:00" and set(a["receivers"]) == {"A;B;C"}
    times = pd.date_range("2026-03-01", periods=14, freq="h")            # data up to 13:00
    shock = np.zeros((14, 3), bool)
    shock[12, 1] = True                                                  # B moves at 12:00
    shock[11, 2] = True                                                  # C moves at 11:00
    active = np.ones((14, 3), bool)
    s = live.score_alerts(a, shock[:12], active[:12], times[:12], ["A", "B", "C"])     # known up to 11:00
    assert s["status"].tolist() == ["miss", "miss", "pending", "pending"] and s["resolved"].tolist() == [1, 1, 0, 0]
    assert np.isclose(s.loc[0, "chance"], 1 / 3)                         # C moved within (10, 11]: 1 of 3
    s = live.score_alerts(s, shock, active, times, ["A", "B", "C"])       # up to 13:00: 3 h windows closed
    assert s["status"].tolist() == ["miss", "miss", "miss", "hit"] and s.loc[3, "first_shock_at"] == "2026-03-01 12:00:00"
    assert np.isclose(s.loc[2, "chance"], 2 / 3) and s["resolved"].all()
    early = live.track_record(a)                                        # nothing resolved yet: no rows, same columns
    assert early.empty and list(early.columns) == live.TRACK_COLUMNS
    assert early[early["horizon_h"].astype(str) == "1"].empty
    tr = live.track_record(s).set_index("horizon_h")
    assert tr.loc[1, "hits"] == 0 and tr.loc[3, "hits"] == 1 and tr.loc["all", "alerts"] == 4
    assert np.isclose(tr.loc["all", "expected_by_chance"], 2 * (1 / 3) + 2 * (2 / 3))

    panel = {"shock_today": torch.from_numpy(shock), "shock_today_mask": torch.ones(14, 3, dtype=torch.bool),
             "active": torch.from_numpy(active), "times": [str(t) for t in times], "graph_coins": ["A", "B", "C"]}
    log_path = tmp_path / "live" / "alert_log.csv"
    log = live.update_alert_log(log_path, rd, 2, panel, [0, 1, 2])
    log = live.update_alert_log(log_path, rd, 2, panel, [0, 1, 2])           # the same hour again: not duplicated
    assert len(log) == 4 and log["resolved"].all() and live.read_alert_log(log_path)["status"].tolist()[-1] == "hit"


def test_predict_reads_another_data_version(toy_project):
    from cryptograph.training.runner import run_experiment

    root, run, d = toy_project
    base = root / "data" / "processed"
    shutil.copytree(base / "v2", base / "v2_live")
    p = torch.load(base / "v2_live" / "1h" / "panel.pt", weights_only=False)
    p["split_id"][-30:] = -1                                     # bars after the research period
    torch.save(p, base / "v2_live" / "1h" / "panel.pt")
    cfg = {"name": "live_toy", "model": {"name": "predict"}, "device": "cpu",
           "predict": {"target_experiment": "toy_tgn", "ig_steps": 4, "data_version": "v2_live", "live": True}}
    out = run_experiment(cfg, seed=0, runs_dir=root / "runs" / "live")
    s = json.loads((root / "runs" / "live" / out["run_id"] / "summary.json").read_text())
    assert s["data_version"] == "v2_live" and s["live"] is True and s["rebuild_max_abs_logit_diff"] < 1e-4
    from cryptograph.explain.tgn_capture import load_trained

    _, D = load_trained(run, data_version="v2_live")
    assert D["cfg"]["data"]["version"] == "v2_live" and D["bounds"]["test"][1] <= int((d.t < 270).sum())
    assert set(pd.read_csv(root / "runs" / "live" / out["run_id"] / "risk.csv")["horizon_h"]) == set(K_HOURS)
    assert set(pd.read_csv(root / "runs" / "live" / out["run_id"] / "risk.csv")["coin"]) <= set(COINS)


def test_a_second_cycle_stops_without_touching_the_status(tmp_path):
    runs = tmp_path / "live"
    runs.mkdir()
    (runs / ".lock").write_text("pid 1 since earlier")
    (runs / "status.json").write_text('{"status": "ok"}')
    with pytest.raises(live.CycleBusy, match="another live cycle is running"):
        live.run_cycle(live.LiveConfig(runs_dir=str(runs)))
    assert json.loads((runs / "status.json").read_text()) == {"status": "ok"} and (runs / ".lock").exists()
