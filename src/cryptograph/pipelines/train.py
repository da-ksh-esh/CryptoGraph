"""Train a model from an experiment config.

    python -m cryptograph.pipelines.train --config configs/experiment/<name>.yaml --seed <n>

Stub: argument parsing and config loading only. Training (with run tracking under
runs/<run_id>/) is added in the training phase.
"""

from __future__ import annotations

from ._experiment import NotReadyError, load_experiment, parse_args


def main(argv: list[str] | None = None) -> None:
    args = parse_args("Train a CryptoGraph model", argv)
    cfg = load_experiment(args.config)
    raise NotReadyError(f"Training for experiment '{cfg['name']}'")


if __name__ == "__main__":
    main()
