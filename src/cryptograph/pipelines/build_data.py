"""Data pipeline orchestrator.

    cryptograph-data --stage audit            # Stage 1: universe audit (checkpoint)
    cryptograph-data --stage fetch            # Stage 2: raw snapshots + cross-source validation
    cryptograph-data --stage clean            # Stage 3: quality audit + cleaned master table
    cryptograph-data --stage build            # Stage 4: features, targets, scalers, graphs, datasets
    cryptograph-data --stage smoke            # Stage 4.6: DataLoader smoke test
    cryptograph-data --stage all              # fetch -> clean -> build -> smoke
    add --refresh to re-download raw snapshots.

Each stage reads only artifacts written by earlier stages, so stages can be
re-run independently.
"""

from __future__ import annotations

import argparse
from typing import Callable

from ..common.logger import get_logger
from ..data.config import DEFAULT_CONFIG_PATH, DataConfig, load_config
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


def _stage(name: str) -> Callable[[DataConfig, bool], None]:
    def run(cfg: DataConfig, refresh: bool) -> None:
        import importlib

        module = importlib.import_module(f"cryptograph.pipelines.stages.{name}")
        module.run(cfg, refresh=refresh)

    return run


STAGES: dict[str, Callable[[DataConfig, bool], None]] = {
    "audit": stage_audit,
    "fetch": _stage("fetch"),
    "clean": _stage("clean"),
    "build": _stage("build"),
    "smoke": _stage("smoke"),
}
ALL_ORDER = ["fetch", "clean", "build", "smoke"]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="CryptoGraph data pipeline")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--stage", choices=[*STAGES, "all"], default="all")
    parser.add_argument("--refresh", action="store_true", help="re-download raw snapshots")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    seed_everything(cfg.seed)
    for name in (ALL_ORDER if args.stage == "all" else [args.stage]):
        logger.info(f"=== stage: {name} ===")
        STAGES[name](cfg, args.refresh)


if __name__ == "__main__":
    main()
