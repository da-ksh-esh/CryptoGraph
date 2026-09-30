"""Legacy data loaders and dataset serialization format compatibility."""

from pathlib import Path
from typing import Dict, Any, Optional, Union
import torch
from .dataloader import CryptoSnapshotDataset, CryptoSequenceDataset, prepare_processed_dataset

def load_processed_dataset(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Loads processed PyTorch dataset dictionary from disk."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Processed dataset not found: {path}")
    return torch.load(path, weights_only=False)

__all__ = [
    "load_processed_dataset",
    "CryptoSnapshotDataset",
    "CryptoSequenceDataset",
    "prepare_processed_dataset"
]
