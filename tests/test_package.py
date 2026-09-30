import pytest
import json
from pathlib import Path
from cryptograph.pipelines.package_data import compute_sha256, pack_data, unpack_data

def test_package_roundtrip(tmp_path):
    # Setup mock processed dataset dir
    source_dir = tmp_path / "processed" / "v1"
    source_dir.mkdir(parents=True)
    sample_file = source_dir / "sample.txt"
    sample_file.write_text("CryptoGraph deterministic test package")

    zip_file = tmp_path / "v1.zip"
    pack_data(version="v1", base_dir=str(tmp_path / "processed"), out_zip=str(zip_file))
    assert zip_file.exists()

    # Unpack into new destination
    dest_dir = tmp_path / "unpacked"
    unpack_data(str(zip_file), target_dir=str(dest_dir))
    unpacked_sample = dest_dir / "sample.txt"
    assert unpacked_sample.exists()
    assert unpacked_sample.read_text() == "CryptoGraph deterministic test package"

def test_tamper_detection(tmp_path):
    source_dir = tmp_path / "proc"
    source_dir.mkdir()
    f = source_dir / "data.txt"
    f.write_text("Original content")

    zip_file = tmp_path / "data.zip"
    pack_data(version="proc", base_dir=str(source_dir), out_zip=str(zip_file))

    # Tamper with the manifest or unpack directory
    dest_dir = tmp_path / "tampered"
    unpack_data(str(zip_file), target_dir=str(dest_dir))
    tampered_file = dest_dir / "data.txt"
    tampered_file.write_text("Tampered malicious modification")

    # Re-verifying manually
    with open(dest_dir / "metadata.json") as meta_f:
        meta = json.load(meta_f)
    expected_hash = meta["files"]["data.txt"]
    actual_hash = compute_sha256(tampered_file)
    assert actual_hash != expected_hash
