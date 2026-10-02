"""Training pipeline for Temporal Graph Network (CryptoTGN) Contagion Predictor.

Headless training CLI and script for training continuous-time CryptoTGN models.
"""

import argparse
from pathlib import Path
import yaml
import torch

from .train import run_training
from ..common.logger import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="CryptoTGN Headless Training Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/model/tgn.yaml",
        help="Path to TGN experiment configuration YAML."
    )
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed.")
    parser.add_argument("--device", type=str, default="auto", help="Compute device: auto | cpu | cuda.")

    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        # Fallback to general config
        alt_cfg = Path("configs/config.yaml")
        if alt_cfg.exists():
            cfg_path = alt_cfg

    logger.info(f"Launching CryptoTGN training pipeline with config: {cfg_path}, seed: {args.seed}, device: {args.device}")
    run_id = run_training(str(cfg_path), seed=args.seed, device_name=args.device)
    logger.info(f"CryptoTGN training successfully completed. Run ID: {run_id}")


if __name__ == "__main__":
    main()
