"""Shared argument parsing and experiment-config loading for train / evaluate / explain."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from ..data.config import PROJECT_ROOT


def parse_args(description: str, argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, help="experiment config, e.g. configs/experiment/<name>.yaml")
    parser.add_argument("--seed", type=int, default=None, help="overrides the seed in the config")
    parser.add_argument("--device", default=None, help="auto | cpu | cuda (overrides the config)")
    return parser.parse_args(argv)


def load_experiment(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        p = PROJECT_ROOT / p
    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg.setdefault("name", p.stem)
    return cfg


class NotReadyError(SystemExit):
    """Raised by entry points whose stage has not been built yet (exit code 2)."""

    def __init__(self, what: str):
        super().__init__(f"{what} is not implemented yet; it is built in the training phase (see docs/model_plan.md).")
        self.code = 2
