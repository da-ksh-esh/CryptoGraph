"""Generates a production checkpoint manifest with dataset, config, and checkpoint SHA256 hashes."""

import hashlib
import json
from pathlib import Path
import glob
import torch
import yaml


def compute_sha256(filepath: Path) -> str:
    """Computes SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()


def main():
    root = Path(".")
    runs_dir = root / "runs"
    dataset_path = root / "data" / "processed" / "v2" / "1h" / "processed_dataset.pt"

    dataset_hash = compute_sha256(dataset_path) if dataset_path.exists() else "not_found"

    manifest = {
        "version": "1.0.0",
        "description": "Production Checkpoint Manifest for CryptoGraph-Secure Framework",
        "dataset": {
            "path": str(dataset_path).replace("\\", "/"),
            "sha256": dataset_hash,
            "resolution": "1h",
            "assets_count": 30,
            "period": "2024-01-01 to 2026-09-27"
        },
        "checkpoints": []
    }

    # Discover checkpoints
    ckpt_patterns = [
        ("CryptoTGN", "runs/*tgn*/best_tgn_checkpoint.pt"),
        ("CryptoLSTM", "runs/*lstm*/best_lstm_checkpoint.pt"),
        ("CryptoDyGFormer", "runs/*dygformer*/best_dygformer_checkpoint.pt")
    ]

    for model_name, pattern in ckpt_patterns:
        files = sorted(glob.glob(pattern))
        for ckpt_str in files:
            ckpt_file = Path(ckpt_str)
            run_folder = ckpt_file.parent

            # Read env.json if present
            env_file = run_folder / "env.json"
            env_data = {}
            if env_file.exists():
                with open(env_file, "r") as f:
                    env_data = json.load(f)

            # Read metrics.json if present
            metrics_file = run_folder / "metrics.json"
            metrics_data = {}
            if metrics_file.exists():
                with open(metrics_file, "r") as f:
                    metrics_data = json.load(f)

            # Read resolved config
            cfg_file = run_folder / "config.resolved.yaml"
            cfg_data = {}
            cfg_hash = ""
            if cfg_file.exists():
                cfg_hash = compute_sha256(cfg_file)
                with open(cfg_file, "r") as f:
                    cfg_data = yaml.safe_load(f)

            # Load checkpoint header safely
            try:
                ckpt_data = torch.load(ckpt_file, map_location="cpu", weights_only=False)
                seed = ckpt_data.get("seed", env_data.get("seed", 42))
                epoch = ckpt_data.get("epoch", None)
            except Exception:
                seed = env_data.get("seed", 42)
                epoch = None

            ckpt_hash = compute_sha256(ckpt_file)

            entry = {
                "model_name": model_name,
                "run_id": run_folder.name,
                "seed": seed,
                "epoch": epoch,
                "checkpoint_path": str(ckpt_file).replace("\\", "/"),
                "checkpoint_sha256": ckpt_hash,
                "config_sha256": cfg_hash,
                "git_commit": env_data.get("git_commit", "unknown"),
                "git_dirty": env_data.get("git_dirty", False),
                "timestamp_utc": env_data.get("timestamp_utc", ""),
                "metrics": {
                    "test_mean_auprc": metrics_data.get("test_mean_auprc", None),
                    "test_mean_auroc": metrics_data.get("test_mean_auroc", None),
                    "val_mean_auprc": metrics_data.get("val_mean_auprc", None),
                }
            }
            manifest["checkpoints"].append(entry)

    out_file = root / "checkpoints" / "manifest.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Generated manifest at {out_file} with {len(manifest['checkpoints'])} registered checkpoints.")


if __name__ == "__main__":
    main()
