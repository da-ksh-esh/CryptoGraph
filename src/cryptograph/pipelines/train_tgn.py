"""Training pipeline for Temporal Graph Network (TGN) Contagion Predictor."""

import yaml
from pathlib import Path
from ..common.logger import get_logger
from ..models.tgn import CryptoTGN

logger = get_logger("train_tgn")

def main():
    logger.info("Initializing TGN Contagion Predictor training pipeline...")
    config_path = Path("configs/config.yaml")
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
            logger.info(f"Loaded project: {cfg.get('project_name')}")

if __name__ == "__main__":
    main()
