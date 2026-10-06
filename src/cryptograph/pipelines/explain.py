"""Explain a trained TGN's contagion predictions and validate the explanations, with local run tracking.

    python -m cryptograph.pipelines.explain --config configs/experiment/<name>.yaml [--seed <n>]

The config names the trained run to explain (`explain.target_experiment`, `explain.target_seed`); see
cryptograph/explain/run.py. Writes runs/<run_id>/ and the tables in reports/explain/ used by notebook 04.
"""

from __future__ import annotations

from ..training.runner import run_experiment
from ._experiment import load_experiment, parse_args


def main(argv: list[str] | None = None) -> None:
    args = parse_args("Explain CryptoGraph predictions", argv)
    cfg = load_experiment(args.config)
    out = run_experiment(cfg, seed=args.seed, device=args.device)
    print(f"run_id: {out['run_id']}\nrun dir: {out['run_dir']}")


if __name__ == "__main__":
    main()
