"""Deterministic random seeding configuration."""

import os
import random
import numpy as np
import torch
from ..common.logger import get_logger

logger = get_logger(__name__)

def seed_everything(seed: int = 42) -> None:
    """
    Sets deterministic seeds across python, numpy, torch CPU/CUDA,
    and configures CUBLAS workspace for reproducible experiments.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception as e:
        logger.debug(f"Deterministic algorithms note: {e}")

    logger.info(f"Initialized global deterministic seed: {seed}")
