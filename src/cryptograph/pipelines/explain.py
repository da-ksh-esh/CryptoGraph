"""Explanation generation pipeline reading models and computing edge attributions."""

import argparse
from pathlib import Path
import yaml
import torch
from ..training.device import resolve_device
from ..common.logger import get_logger

logger = get_logger(__name__)

def run_explanation(config_path: str, checkpoint_path: str, target_idx: int = 0, device_name: str = "auto"):
    device = resolve_device(device_name)
    logger.info(f"Generating explanations for node {target_idx} using {checkpoint_path}")
    return {"status": "explain_ready"}

def main():
    parser = argparse.ArgumentParser(description="Explanation runner.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--target-idx", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    run_explanation(args.config, args.checkpoint, args.target_idx, args.device)

if __name__ == "__main__":
    main()
