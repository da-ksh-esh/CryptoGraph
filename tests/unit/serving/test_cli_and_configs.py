from pathlib import Path

import pytest

from cryptograph.data.config import PROJECT_ROOT
from cryptograph.pipelines._experiment import load_experiment

EXPERIMENTS = sorted((PROJECT_ROOT / "configs" / "experiment").glob("*.yaml"))


@pytest.mark.parametrize("path", EXPERIMENTS, ids=lambda p: p.stem)
def test_every_experiment_config_names_a_registered_task(path):
    from cryptograph.training.runner import TASKS

    cfg = load_experiment(path)
    m = cfg.get("model")
    name = m if isinstance(m, str) else (m or {}).get("name")
    assert name in TASKS or name == "snapshot_gnn", f"{path.name}: unknown task {name!r}"
    if isinstance(m, dict) and m.get("config"):
        assert (PROJECT_ROOT / m["config"]).exists(), f"{path.name}: missing model config {m['config']}"


def test_serving_configs_point_at_the_final_tgn():
    for name, key in (("predict_tgn_1h", "predict"), ("compress_tgn_1h", "compress")):
        cfg = load_experiment(PROJECT_ROOT / "configs" / "experiment" / f"{name}.yaml")
        assert cfg[key]["target_experiment"] == "tgn_1h_final_v2" and cfg[key]["target_seed"] == 0
        assert (PROJECT_ROOT / "configs" / "experiment" / "tgn_1h_final_v2.yaml").exists()


def test_v3_serving_configs_point_at_the_final_v3_tgn():
    for name, key in (("predict_tgn_1h_v3", "predict"), ("predict_events_1h_v3", "events")):
        cfg = load_experiment(PROJECT_ROOT / "configs" / "experiment" / f"{name}.yaml")
        assert cfg[key]["target_experiment"] == "tgn_1h_v3_final" and cfg[key]["target_seed"] == 0
    final = load_experiment(PROJECT_ROOT / "configs" / "experiment" / "tgn_1h_v3_final.yaml")
    assert final["data"]["version"] == "v3"


def test_predict_cli_overrides_and_prints(toy_project, monkeypatch, capsys, tmp_path):
    import yaml

    import cryptograph.pipelines.predict as cli
    import cryptograph.training.runner as runner

    root, run, _ = toy_project
    cfg = tmp_path / "p.yaml"
    cfg.write_text(yaml.safe_dump({"name": "predict_toy", "model": {"name": "predict"},
                                   "predict": {"target_experiment": "nonexistent", "ig_steps": 4}}), encoding="utf-8")
    real = runner.run_experiment
    monkeypatch.setattr(cli, "run_experiment", lambda c, **kw: real(c, runs_dir=root / "runs", **kw))
    cli.main(["--config", str(cfg), "--run", str(run), "--as-of", "2026-01-12 05:30"])
    out = capsys.readouterr().out
    assert "as of 2026-01-12 05:00:00 UTC" in out and "riskiest coins" in out and "top alerts" in out
    assert Path(out.split("run dir: ")[-1].strip()).exists()
