"""Data packaging and verification utility for Colab, Kaggle, and remote compute."""

import os
import sys
import json
import zipfile
import hashlib
import argparse
from pathlib import Path
from typing import Dict, Any

def compute_sha256(file_path: Path) -> str:
    """Computes SHA-256 hex digest of a file in chunks."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def pack_data(version: str = "v1", base_dir: str = "data/processed", out_zip: str = None) -> Path:
    """
    Zips data/processed/<version>/ along with metadata.json recording per-file SHA-256 hashes.
    """
    proc_dir = Path(base_dir) / version
    if not proc_dir.exists():
        # Fallback to root processed dir if sub-version does not exist yet
        proc_dir = Path(base_dir)

    if out_zip is None:
        out_zip = f"{version}.zip"

    zip_path = Path(out_zip)
    manifest = {}

    files_to_pack = [p for p in proc_dir.glob("**/*") if p.is_file() and p.name != "metadata.json"]
    for p in files_to_pack:
        rel = p.relative_to(proc_dir)
        manifest[str(rel).replace("\\", "/")] = compute_sha256(p)

    metadata_path = proc_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump({"version": version, "files": manifest}, f, indent=2)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(metadata_path, arcname="metadata.json")
        for p in files_to_pack:
            rel = p.relative_to(proc_dir)
            zf.write(p, arcname=str(rel))

    print(f"Packed {len(files_to_pack)} files into {zip_path.resolve()} with SHA-256 manifest.")
    return zip_path

def unpack_data(zip_file: str, target_dir: str = "data/processed") -> None:
    """
    Extracts zip file and verifies SHA-256 checksum of every file against manifest.
    Fails loudly with ValueError if any hash mismatch occurs.
    """
    zip_path = Path(zip_file)
    if not zip_path.exists():
        raise FileNotFoundError(f"Archive file not found: {zip_path}")

    dest = Path(target_dir)
    dest.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest)

    metadata_path = dest / "metadata.json"
    if not metadata_path.exists():
        raise ValueError(f"Extracted package missing metadata.json in {dest}")

    with open(metadata_path, "r") as f:
        meta = json.load(f)

    manifest = meta.get("files", {})
    print(f"Verifying {len(manifest)} files against SHA-256 manifest...")

    for rel_path, expected_hash in manifest.items():
        extracted_file = dest / rel_path
        if not extracted_file.exists():
            raise FileNotFoundError(f"Missing expected extracted file: {extracted_file}")
        actual_hash = compute_sha256(extracted_file)
        if actual_hash != expected_hash:
            raise ValueError(
                f"SHA-256 verification failed for {rel_path}!\n"
                f"Expected: {expected_hash}\n"
                f"Actual:   {actual_hash}"
            )

    print("All file hashes verified successfully.")

def main():
    parser = argparse.ArgumentParser(description="Package or extract verified processed datasets.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    pack_parser = subparsers.add_parser("pack", help="Pack processed dataset into verified zip.")
    pack_parser.add_argument("--version", type=str, default="v1")
    pack_parser.add_argument("--out", type=str, default=None)

    unpack_parser = subparsers.add_parser("unpack", help="Unpack and verify dataset zip.")
    unpack_parser.add_argument("zip_path", type=str)
    unpack_parser.add_argument("--target-dir", type=str, default="data/processed")

    args = parser.parse_args()
    if args.command == "pack":
        pack_data(version=args.version, out_zip=args.out)
    elif args.command == "unpack":
        unpack_data(zip_file=args.zip_path, target_dir=args.target_dir)

if __name__ == "__main__":
    main()
