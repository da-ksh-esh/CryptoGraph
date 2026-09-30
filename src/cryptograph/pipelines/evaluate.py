"""Evaluate trained runs of an experiment against the baselines.

    python -m cryptograph.pipelines.evaluate --config configs/experiment/<name>.yaml

Stub: argument parsing and config loading only.
"""

from __future__ import annotations

from ._experiment import NotReadyError, load_experiment, parse_args


def main(argv: list[str] | None = None) -> None:
    args = parse_args("Evaluate CryptoGraph runs", argv)
    cfg = load_experiment(args.config)
    raise NotReadyError(f"Evaluation for experiment '{cfg['name']}'")


if __name__ == "__main__":
    main()
