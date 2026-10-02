"""Pack processed data for Colab / Kaggle and unpack it with hash verification.

    python -m cryptograph.pipelines.package_data pack --version v2          # -> dist/cryptograph_data_v2.zip
    python -m cryptograph.pipelines.package_data unpack <zip> [--dest .]    # extract + verify every SHA-256

The zip holds data/processed/<version>/** under their project-relative paths plus
PACKAGE_MANIFEST.json (version, git commit, created time, per-file size and SHA-256). Unpacking
refuses paths outside the destination and fails loudly on any missing file or hash mismatch.
Data never goes into git; the zip is stored on Google Drive / a private Kaggle dataset.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

from ..data.config import PROJECT_ROOT
from ..data.storage import sha256_file, utc_now

MANIFEST = "PACKAGE_MANIFEST.json"


class PackageError(SystemExit):
    def __init__(self, msg: str):
        super().__init__(f"package error: {msg}")
        self.code = 3


def pack(version: str, out: Path | None = None, root: Path = PROJECT_ROOT) -> Path:
    src = root / "data" / "processed" / version
    if not src.is_dir():
        raise PackageError(f"{src} does not exist; build the data first (cryptograph-data)")
    files = sorted(p for p in src.rglob("*") if p.is_file())
    if not files:
        raise PackageError(f"{src} is empty")
    from ..training.tracking import git_info

    manifest = {"version": version, "created_utc": utc_now(), "git": git_info(root),
                "files": {p.relative_to(root).as_posix(): {"bytes": p.stat().st_size, "sha256": sha256_file(p)}
                          for p in files}}
    out = Path(out) if out else root / "dist" / f"cryptograph_data_{version}.zip"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".zip.part")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as z:
        for rel in manifest["files"]:
            z.write(root / rel, rel)
        z.writestr(MANIFEST, json.dumps(manifest, indent=2))
    tmp.replace(out)
    return out


def unpack(zip_path: Path, dest: Path = PROJECT_ROOT) -> dict:
    zip_path, dest = Path(zip_path), Path(dest).resolve()
    with zipfile.ZipFile(zip_path) as z:
        if MANIFEST not in z.namelist():
            raise PackageError(f"{zip_path} has no {MANIFEST}")
        manifest = json.loads(z.read(MANIFEST))
        for name in z.namelist():
            target = (dest / name).resolve()
            if dest not in target.parents and target != dest:
                raise PackageError(f"unsafe path in archive: {name}")
        missing = sorted(set(manifest["files"]) - set(z.namelist()))
        if missing:
            raise PackageError(f"files listed in the manifest are missing from the archive: {missing[:5]}")
        z.extractall(dest, members=[n for n in z.namelist() if n != MANIFEST])
    bad = [rel for rel, meta in manifest["files"].items() if sha256_file(dest / rel) != meta["sha256"]]
    if bad:
        raise PackageError(f"SHA-256 mismatch for {len(bad)} file(s): {bad[:5]}")
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Pack / unpack processed CryptoGraph data with hash checks")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pack")
    p.add_argument("--version", required=True, choices=["v1", "v2"])
    p.add_argument("--out", default=None)
    u = sub.add_parser("unpack")
    u.add_argument("zip")
    u.add_argument("--dest", default=str(PROJECT_ROOT))
    args = parser.parse_args(argv)
    if args.cmd == "pack":
        out = pack(args.version, Path(args.out) if args.out else None)
        print(f"packed data/processed/{args.version} -> {out} ({out.stat().st_size / 1e6:.1f} MB)")
    else:
        m = unpack(Path(args.zip), Path(args.dest))
        print(f"unpacked {len(m['files'])} files of data {m['version']} (commit {m['git']['commit']}); all SHA-256 verified")


if __name__ == "__main__":
    main()
