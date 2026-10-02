"""Device selection: the same code runs on a laptop CPU and on a Colab / Kaggle GPU."""

from __future__ import annotations

import torch


def resolve_device(name: str | None = "auto") -> torch.device:
    """"auto" -> cuda if available, else cpu; "cpu" / "cuda" / "cuda:N" are taken literally
    ("cuda" without a GPU raises, so a GPU run never silently falls back to CPU)."""
    name = (name or "auto").lower()
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(name)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"device '{name}' requested but CUDA is not available")
    return dev


def describe_device(device: torch.device) -> dict:
    info = {"device": str(device), "torch": torch.__version__, "cuda_available": torch.cuda.is_available()}
    if device.type == "cuda":
        idx = device.index if device.index is not None else torch.cuda.current_device()
        info |= {"gpu": torch.cuda.get_device_name(idx), "cuda": torch.version.cuda,
                 "gpu_memory_gb": round(torch.cuda.get_device_properties(idx).total_memory / 1e9, 1)}
    return info
