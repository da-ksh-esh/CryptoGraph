"""Explain a trained TGN's contagion predictions and validate the explanations.

    python -m cryptograph.pipelines.explain --config configs/experiment/<name>.yaml

Stub: argument parsing and config loading only.
"""

from __future__ import annotations

from ._experiment import NotReadyError, load_experiment, parse_args


def main(argv: list[str] | None = None) -> None:
    args = parse_args("Explain CryptoGraph predictions", argv)
    cfg = load_experiment(args.config)
    raise NotReadyError(f"Explanations for experiment '{cfg['name']}'")


if __name__ == "__main__":
    main()
