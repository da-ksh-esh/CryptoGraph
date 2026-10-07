"""Seeding for reproducible runs."""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy and (if installed) PyTorch RNGs.

    deterministic=True also asks PyTorch for deterministic kernels (warn-only, since a few
    GPU ops have no deterministic version) and sets CUBLAS_WORKSPACE_CONFIG, which cuBLAS
    needs for reproducible matrix products on CUDA >= 10.2. Set it before the first CUDA call.
    """
    random.seed(seed)
    np.random.seed(seed)
    # PYTHONHASHSEED cannot change the running interpreter (fixed at start-up); setting it here makes child
    # processes started from now on (e.g. DataLoader workers on Windows) hash reproducibly. No result depends on it.
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
