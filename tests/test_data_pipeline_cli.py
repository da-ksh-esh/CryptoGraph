from pathlib import Path

from cryptograph.pipelines.data import resolve_pipeline_targets


def test_all_defaults_to_both_hourly_resolutions():
    assert resolve_pipeline_targets("all", None, "configs/data/hourly.yaml") == [
        ("1h", Path("data/processed/v2/1h/processed_dataset.pt")),
        ("2h", Path("data/processed/v2/2h/processed_dataset.pt")),
    ]


def test_explicit_resolution_builds_only_that_resolution():
    assert resolve_pipeline_targets("all", "2h", "configs/data/hourly.yaml") == [
        ("2h", Path("data/processed/v2/2h/processed_dataset.pt"))
    ]


def test_build_uses_configured_resolution(tmp_path):
    config = tmp_path / "hourly.yaml"
    config.write_text("resolution: 2h\n", encoding="utf-8")
    assert resolve_pipeline_targets("build", None, str(config)) == [
        ("2h", Path("data/processed/v2/2h/processed_dataset.pt"))
    ]


def test_daily_config_resolves_daily_artifact():
    assert resolve_pipeline_targets("verify", None, "configs/data/daily.yaml") == [
        ("1d", Path("data/processed/crypto_graph_dataset.pt"))
    ]
