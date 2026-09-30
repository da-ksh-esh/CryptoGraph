"""Device resolution utility."""

import torch
from ..common.logger import get_logger

logger = get_logger(__name__)

def resolve_device(device_str: str = "auto") -> torch.device:
    """
    Resolves compute device: 'auto' | 'cpu' | 'cuda'.
    """
    device_str = device_str.lower().strip()
    if device_str == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device_str == "cuda":
        if not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            dev = torch.device("cpu")
        else:
            dev = torch.device("cuda")
    else:
        dev = torch.device("cpu")

    logger.info(f"Using device: {dev}")
    return dev
