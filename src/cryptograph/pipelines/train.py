"""Train a model from an experiment config, with local run tracking.

    python -m cryptograph.pipelines.train --config configs/experiment/<name>.yaml [--seed <n>] [--device auto|cpu|cuda]

Writes runs/<run_id>/ (config, env, log, metrics, checkpoint) and appends to runs/results.csv.
Models not wired up yet exit with code 2 (see docs/model_plan.md).
"""

from __future__ import annotations

from ..training.runner import run_experiment
from ._experiment import load_experiment, parse_args


def main(argv: list[str] | None = None) -> None:
    args = parse_args("Train a CryptoGraph model", argv)
    cfg = load_experiment(args.config)
    out = run_experiment(cfg, seed=args.seed, device=args.device)
    print(f"run_id: {out['run_id']}\nrun dir: {out['run_dir']}")


if __name__ == "__main__":
    main()
