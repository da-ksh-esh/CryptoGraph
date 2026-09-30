"""Experiment run tracking and artifact provenance logging."""

import os
import sys
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import yaml
import torch

class ExperimentTracker:
    """
    Manages structured experiment runs, tracking configurations, system metadata,
    metrics, checkpoints, and appending to central results.csv.
    """

    def __init__(self, experiment_name: str, seed: int = 42, runs_dir: str = "runs"):
        self.experiment_name = experiment_name
        self.seed = seed
        self.runs_dir = Path(runs_dir)
        self.runs_dir.mkdir(parents=True, exist_ok=True)

        self.git_short, self.is_dirty = self._get_git_info()
        now_str = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        self.run_id = f"{now_str}_{experiment_name}_s{seed}_{self.git_short}"

        self.run_path = self.runs_dir / self.run_id
        self.run_path.mkdir(parents=True, exist_ok=True)

    def _get_git_info(self) -> tuple[str, bool]:
        try:
            short = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
            status = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
            return short, len(status) > 0
        except Exception:
            return "nogit", False

    def log_environment(self, config: Dict[str, Any], device: str) -> None:
        """Saves config.resolved.yaml and env.json."""
        with open(self.run_path / "config.resolved.yaml", "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        env_info = {
            "run_id": self.run_id,
            "experiment": self.experiment_name,
            "seed": self.seed,
            "git_commit": self.git_short,
            "git_dirty": self.is_dirty,
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "device": str(device),
            "timestamp_utc": datetime.utcnow().isoformat()
        }
        with open(self.run_path / "env.json", "w") as f:
            json.dump(env_info, f, indent=2)

    def log_metrics(self, metrics: Dict[str, Any]) -> None:
        """Saves metrics.json and safely appends a flat row to runs/results.csv."""
        with open(self.run_path / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        results_csv = self.runs_dir / "results.csv"
        flat_row = {
            "run_id": self.run_id,
            "experiment": self.experiment_name,
            "seed": self.seed,
            "git_commit": self.git_short,
            "git_dirty": self.is_dirty,
            "timestamp": datetime.utcnow().isoformat(),
            **metrics
        }
        row_df = pd.DataFrame([flat_row])

        if not results_csv.exists():
            row_df.to_csv(results_csv, index=False)
        else:
            row_df.to_csv(results_csv, mode="a", header=False, index=False)

    def save_checkpoint(self, state_dict: Dict[str, Any], filename: str = "checkpoint.pt") -> Path:
        ckpt_path = self.run_path / filename
        torch.save(state_dict, ckpt_path)
        return ckpt_path
