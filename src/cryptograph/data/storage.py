"""Artifact I/O: hashing, JSON/parquet helpers, fetch log and run manifest."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import pandas as pd

TRACKED_LIBRARIES = (
    "numpy", "pandas", "pyarrow", "scipy", "statsmodels", "scikit-learn", "torch",
    "torch-geometric", "yfinance", "arch", "matplotlib", "seaborn", "networkx", "pydantic",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def library_versions() -> dict[str, str]:
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for lib in TRACKED_LIBRARIES:
        try:
            out[lib] = importlib_metadata.version(lib)
        except importlib_metadata.PackageNotFoundError:
            out[lib] = "not-installed"
    return out


def write_json(obj: Any, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
    return p


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_parquet(df: pd.DataFrame, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=True)
    return p


def write_csv(df: pd.DataFrame, path: str | Path, index: bool = False) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=index)
    return p


class FetchLog:
    """Append-only provenance log for raw downloads (one JSON file per raw dir)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.entries: dict[str, dict[str, Any]] = read_json(self.path) if self.path.exists() else {}

    def record(self, file: Path, *, source: str, url: str, rows: int, extra: dict | None = None) -> None:
        self.entries[file.name] = {
            "source": source,
            "url": url,
            "fetched_utc": utc_now(),
            "rows": int(rows),
            "sha256": sha256_file(file),
            **(extra or {}),
        }
        write_json(self.entries, self.path)

    def get(self, file: Path) -> dict[str, Any] | None:
        return self.entries.get(file.name)


def hash_tree(paths: list[Path], root: Path) -> dict[str, str]:
    """SHA-256 for each file, keyed by POSIX path relative to root."""
    return {p.relative_to(root).as_posix(): sha256_file(p) for p in sorted(paths) if p.is_file()}


def load_pt(path: str | Path, map_location=None) -> Any:
    """torch.load in safe mode (weights_only=True): tensors, containers and plain types only, so a tampered file
    cannot run code when it is opened. The PyG event stream (tgn_events.pt) also needs PyG's TemporalData and its
    storage class, which are allow-listed explicitly; any other object type is refused."""
    import torch

    allowed = []
    try:
        from torch_geometric.data.storage import GlobalStorage
        from torch_geometric.data.temporal import TemporalData

        allowed = [TemporalData, GlobalStorage]
    except ImportError:
        pass
    with torch.serialization.safe_globals(allowed):
        return torch.load(path, weights_only=True, map_location=map_location)
