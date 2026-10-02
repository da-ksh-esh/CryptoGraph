import pytest

from cryptograph.pipelines import evaluate, explain, train
from cryptograph.pipelines._experiment import load_experiment


@pytest.mark.parametrize("module", [train, evaluate, explain])
def test_entry_points_exit_cleanly_for_models_not_built_yet(module, tmp_path):
    cfg = tmp_path / "demo.yaml"
    cfg.write_text("model: dygformer\nseed: 1\n", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        module.main(["--config", str(cfg), "--seed", "3"])
    assert exc.value.code == 2


def test_load_experiment_defaults_name(tmp_path):
    cfg = tmp_path / "daily_lstm.yaml"
    cfg.write_text("model: lstm\n", encoding="utf-8")
    assert load_experiment(cfg) == {"model": "lstm", "name": "daily_lstm"}
