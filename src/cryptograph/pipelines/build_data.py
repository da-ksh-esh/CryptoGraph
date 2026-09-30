"""Data pipeline orchestrator.

Daily (Yahoo + Coin Metrics), configs/data/daily.yaml:
    cryptograph-data --stage audit            # universe audit (checkpoint)
    cryptograph-data --stage fetch            # raw snapshots + cross-source validation
    cryptograph-data --stage clean            # quality audit + cleaned master table
    cryptograph-data --stage build            # features, targets, scalers, graphs, datasets
    cryptograph-data --stage smoke            # DataLoader smoke test
    cryptograph-data --stage all              # fetch -> clean -> build -> smoke

Hourly (Binance), configs/data/hourly.yaml (selected because it declares `resolution`):
    cryptograph-data --config configs/data/hourly.yaml --stage audit    # 30-coin universe (checkpoint)

Add --refresh to re-download raw snapshots. Each stage reads only artifacts written by
earlier stages, so stages can be re-run independently.
"""

from __future__ import annotations

import argparse
from typing import Callable

from ..common.logger import get_logger
from ..data.config import DEFAULT_CONFIG_PATH, DataConfig, is_hourly_config, load_config, load_hourly_config
from ..training.seed import seed_everything

logger = get_logger("cryptograph.pipeline")


def stage_audit(cfg: DataConfig, refresh: bool) -> None:
    from ..data.universe import UniverseAuditor, format_summary

    audit, panels = UniverseAuditor(cfg, refresh=refresh).run()
    print(format_summary(audit, panels, cfg))
    opts = panels["options"]
    if opts["panel_a_size"] < opts["panel_a_target"]:
        print("\nCHECKPOINT: Panel A cannot be filled with the current criteria. Options:")
        for k, v in opts.items():
            if isinstance(v, dict):
                print(f"  {k}: {v}")


def _stage(name: str) -> Callable:
    def run(cfg, refresh: bool) -> None:
        import importlib

        module = importlib.import_module(f"cryptograph.pipelines.stages.{name}")
        module.run(cfg, refresh=refresh)

    return run


STAGES: dict[str, Callable] = {
    "audit": stage_audit,
    "fetch": _stage("fetch"),
    "clean": _stage("clean"),
    "build": _stage("build"),
    "smoke": _stage("smoke"),
}
ALL_ORDER = ["fetch", "clean", "build", "smoke"]

HOURLY_STAGES: dict[str, Callable] = {
    "audit": _stage("hourly_audit"),
}
HOURLY_ALL_ORDER = ["audit"]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="CryptoGraph data pipeline")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--stage", choices=sorted({*STAGES, *HOURLY_STAGES, "all"}), default="all")
    parser.add_argument("--refresh", action="store_true", help="re-download raw snapshots")
    args = parser.parse_args(argv)

    if is_hourly_config(args.config):
        cfg, stages, order = load_hourly_config(args.config), HOURLY_STAGES, HOURLY_ALL_ORDER
    else:
        cfg, stages, order = load_config(args.config), STAGES, ALL_ORDER
    seed_everything(cfg.seed)
    for name in (order if args.stage == "all" else [args.stage]):
        if name not in stages:
            parser.error(f"stage '{name}' is not available for this config yet")
        logger.info(f"=== stage: {name} ===")
        stages[name](cfg, args.refresh)


if __name__ == "__main__":
    main()
