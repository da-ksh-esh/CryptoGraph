"""Local run tracking (no external service).

Each run writes runs/<run_id>/ with
    config.resolved.yaml   the experiment config after overrides (seed, device)
    env.json               library versions, Python, platform, device, git commit and dirty flag
    log.txt                timestamped log lines
    metrics.json           final metrics
    checkpoint.pt          best model state (optional)
and appends one row to runs/results.csv (run_id, experiment, seed, commit, dirty, metrics...).
run_id = {YYYYmmdd-HHMMSS}_{experiment}_{seed}_{gitshort}.
"""

from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from ..data.config import PROJECT_ROOT
from ..data.storage import library_versions, utc_now, write_json

RUN_ID_RE = re.compile(r"^\d{8}-\d{6}_[A-Za-z0-9_.-]+_\d+_[0-9a-z]+$")
CODE_PATHS = ("src", "configs", "pyproject.toml", "requirements.txt", "requirements-lock.txt")
CORE_COLUMNS = ["run_id", "experiment", "seed", "commit", "dirty", "device", "started_utc", "finished_utc", "status"]


def git_info(root: Path = PROJECT_ROOT) -> dict:
    """{"commit": short hash or "nogit", "dirty": bool}.

    dirty = the code that runs may differ from the commit: any modified, staged or *untracked* file under the paths
    that affect a run (`CODE_PATHS`: src/, configs/, pyproject.toml, requirements files). Documents and notebooks do
    not affect a run and are ignored.

    Without a git checkout (e.g. code unpacked from `cryptograph-package pack-code` on Colab), the commit and dirty
    flag recorded in the code package are used when CRYPTOGRAPH_GIT_COMMIT / CRYPTOGRAPH_GIT_DIRTY are set."""
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True, text=True,
                                check=True, timeout=10).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all", "--", *CODE_PATHS],
                                cwd=root, capture_output=True, text=True, check=True, timeout=10).stdout.strip()
        return {"commit": commit or "nogit", "dirty": bool(status)}
    except (OSError, subprocess.SubprocessError):
        if os.environ.get("CRYPTOGRAPH_GIT_COMMIT"):
            return {"commit": os.environ["CRYPTOGRAPH_GIT_COMMIT"],
                    "dirty": os.environ.get("CRYPTOGRAPH_GIT_DIRTY", "true").lower() == "true", "source": "code package"}
        return {"commit": "nogit", "dirty": True}


def make_run_id(experiment: str, seed: int, commit: str, now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    exp = re.sub(r"[^A-Za-z0-9_.-]+", "-", experiment)
    return f"{now:%Y%m%d-%H%M%S}_{exp}_{int(seed)}_{commit}"


class FileLock:
    """Minimal cross-platform lock file (O_CREAT | O_EXCL), so two runs never append at the same time."""

    def __init__(self, path: Path, timeout: float = 30.0, stale_after: float = 300.0):
        self.path, self.timeout, self.stale_after = Path(path), timeout, stale_after

    def __enter__(self):
        start = time.monotonic()
        while True:
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(os.getpid()).encode())
                os.close(fd)
                return self
            except FileExistsError:
                try:
                    if time.time() - self.path.stat().st_mtime > self.stale_after:   # left by a crashed run
                        self.path.unlink(missing_ok=True)
                        continue
                except FileNotFoundError:
                    continue
                if time.monotonic() - start > self.timeout:
                    raise TimeoutError(f"could not acquire {self.path}")
                time.sleep(0.05)

    def __exit__(self, *exc):
        self.path.unlink(missing_ok=True)


def append_results(path: Path, row: dict[str, Any]) -> None:
    """Append a row to a CSV; new metric columns are added (the file is rewritten under the lock)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(path.with_suffix(path.suffix + ".lock")):
        rows, cols = [], list(CORE_COLUMNS)
        if path.exists():
            with open(path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                cols = list(reader.fieldnames or cols)
                rows = list(reader)
        new = [k for k in row if k not in cols]
        if new or not path.exists():
            cols += new
            rows.append({k: row.get(k, "") for k in cols})
            tmp = path.with_suffix(".tmp")
            with open(tmp, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=cols)
                w.writeheader()
                w.writerows(rows)
            os.replace(tmp, path)
        else:
            with open(path, "a", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=cols).writerow({k: row.get(k, "") for k in cols})


class RunTracker:
    def __init__(self, experiment: str, seed: int, config: dict, device_info: dict,
                 runs_dir: Path | str = PROJECT_ROOT / "runs"):
        self.runs_dir = Path(runs_dir)
        self.git = git_info()
        self.experiment, self.seed = experiment, int(seed)
        self.run_id = make_run_id(experiment, seed, self.git["commit"])
        self.dir = self.runs_dir / self.run_id
        self.dir.mkdir(parents=True, exist_ok=False)
        self.started = utc_now()
        self.device = device_info.get("device", "?")
        with open(self.dir / "config.resolved.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, sort_keys=False)
        write_json({"run_id": self.run_id, "started_utc": self.started, "git": self.git, "device": device_info,
                    "libraries": library_versions()}, self.dir / "env.json")
        self.log(f"run {self.run_id} on {self.device}; commit {self.git['commit']}"
                 + (" (uncommitted changes)" if self.git["dirty"] else ""))

    def log(self, msg: str) -> None:
        line = f"[{utc_now()}] {msg}"
        print(line, flush=True)
        with open(self.dir / "log.txt", "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def save_checkpoint(self, state: dict) -> Path:
        import torch

        p = self.dir / "checkpoint.pt"
        torch.save(state, p)
        return p

    def finish(self, metrics: dict[str, Any], status: str = "ok") -> dict:
        write_json(metrics, self.dir / "metrics.json")
        row = {"run_id": self.run_id, "experiment": self.experiment, "seed": self.seed, "commit": self.git["commit"],
               "dirty": self.git["dirty"], "device": self.device, "started_utc": self.started,
               "finished_utc": utc_now(), "status": status,
               **{k: (round(v, 6) if isinstance(v, float) else v) for k, v in metrics.items()
                  if not isinstance(v, (dict, list))}}
        append_results(self.runs_dir / "results.csv", row)
        self.log(f"finished ({status}): " + json.dumps({k: row[k] for k in row if k not in CORE_COLUMNS}, default=str))
        return row
