import csv
import json
import zipfile

import numpy as np
import pytest
import torch

from cryptograph.pipelines import package_data
from cryptograph.training.device import describe_device, resolve_device
from cryptograph.training.loop import train_epoch
from cryptograph.training.seed import seed_everything
from cryptograph.training.tracking import RUN_ID_RE, RunTracker, append_results, make_run_id


def test_resolve_device_cpu_and_auto():
    assert resolve_device("cpu").type == "cpu"
    assert resolve_device("auto").type in ("cpu", "cuda")
    assert describe_device(resolve_device("cpu"))["device"] == "cpu"
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError):
            resolve_device("cuda")


def test_run_id_format():
    rid = make_run_id("smoke lstm/daily", 3, "abc1234")
    assert RUN_ID_RE.match(rid) and rid.endswith("_smoke-lstm-daily_3_abc1234")


def test_results_csv_appends_and_adds_columns(tmp_path):
    p = tmp_path / "results.csv"
    append_results(p, {"run_id": "a", "seed": 0, "val_mse": 1.0})
    append_results(p, {"run_id": "b", "seed": 1, "val_mse": 2.0, "test_mse": 3.0})
    append_results(p, {"run_id": "c", "seed": 2})
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    assert [r["run_id"] for r in rows] == ["a", "b", "c"]
    assert rows[0]["test_mse"] == "" and rows[1]["test_mse"] == "3.0"
    assert not (tmp_path / "results.csv.lock").exists()


def test_run_tracker_writes_run_folder(tmp_path):
    t = RunTracker("unit", 7, {"name": "unit", "seed": 7}, {"device": "cpu"}, runs_dir=tmp_path)
    t.save_checkpoint({"w": torch.zeros(2)})
    row = t.finish({"val_mse": 0.5, "history": [1, 2]})
    names = {p.name for p in t.dir.iterdir()}
    assert {"config.resolved.yaml", "env.json", "log.txt", "metrics.json", "checkpoint.pt"} <= names
    env = json.loads((t.dir / "env.json").read_text())
    assert "commit" in env["git"] and "dirty" in env["git"] and "torch" in env["libraries"]
    assert row["val_mse"] == 0.5 and "history" not in row
    assert (tmp_path / "results.csv").exists()


def _tiny_run(seed):
    seed_everything(seed, deterministic=True)
    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 1))
    g = torch.Generator().manual_seed(seed)
    X, y = torch.randn(256, 4, generator=g), torch.randn(256, 1, generator=g)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, y), batch_size=32, shuffle=True,
                                         generator=torch.Generator().manual_seed(seed))
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)

    def step(m, b, d):
        out = m(b[0])
        return torch.nn.functional.mse_loss(out, b[1]), out

    for _ in range(3):
        train_epoch(model, loader, step, opt, torch.device("cpu"))
    return torch.cat([p.detach().flatten() for p in model.parameters()])


def test_seeded_runs_are_identical():
    assert torch.equal(_tiny_run(5), _tiny_run(5))
    assert not torch.equal(_tiny_run(5), _tiny_run(6))


def _fake_project(root):
    d = root / "data" / "processed" / "v9" / "1h"
    d.mkdir(parents=True)
    (d / "a.bin").write_bytes(np.arange(1000, dtype=np.int64).tobytes())
    (d.parent / "rankings.csv").write_text("coin,x\nBTC,1\n", encoding="utf-8")
    return root


def test_pack_unpack_roundtrip_verifies_hashes(tmp_path):
    src = _fake_project(tmp_path / "src")
    z = package_data.pack("v9", tmp_path / "pkg.zip", root=src)
    m = package_data.unpack(z, tmp_path / "dest")
    assert set(m["files"]) == {"data/processed/v9/1h/a.bin", "data/processed/v9/rankings.csv"}
    assert (tmp_path / "dest" / "data/processed/v9/1h/a.bin").read_bytes() == (src / "data/processed/v9/1h/a.bin").read_bytes()


def test_pack_code_has_sources_and_no_data(tmp_path):
    root = tmp_path / "proj"
    (root / "src" / "pkg" / "__pycache__").mkdir(parents=True)
    (root / "src" / "pkg" / "a.py").write_text("x = 1\n", encoding="utf-8")
    (root / "src" / "pkg" / "__pycache__" / "a.cpython-312.pyc").write_bytes(b"\0")
    (root / "configs").mkdir()
    (root / "configs" / "c.yaml").write_text("a: 1\n", encoding="utf-8")
    (root / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    _fake_project(root)
    z = package_data.pack_code(tmp_path / "code.zip", root=root)
    m = package_data.unpack(z, tmp_path / "dest")
    assert set(m["files"]) == {"src/pkg/a.py", "configs/c.yaml", "pyproject.toml"}


def test_unpack_detects_tampering_and_unsafe_paths(tmp_path):
    src = _fake_project(tmp_path / "src")
    z = package_data.pack("v9", tmp_path / "pkg.zip", root=src)
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(z) as zin, zipfile.ZipFile(bad, "w") as zout:
        for n in zin.namelist():
            data = zin.read(n)
            zout.writestr(n, data + b"x" if n.endswith("a.bin") else data)
    with pytest.raises(SystemExit) as e:
        package_data.unpack(bad, tmp_path / "d1")
    assert "mismatch" in str(e.value)
    evil = tmp_path / "evil.zip"
    with zipfile.ZipFile(evil, "w") as zout:
        zout.writestr(package_data.MANIFEST, json.dumps({"version": "v9", "files": {}}))
        zout.writestr("../outside.txt", "x")
    with pytest.raises(SystemExit) as e:
        package_data.unpack(evil, tmp_path / "d2")
    assert "unsafe" in str(e.value) and not (tmp_path / "outside.txt").exists()


def test_git_info_falls_back_to_code_package_commit(tmp_path, monkeypatch):
    from cryptograph.training.tracking import git_info

    monkeypatch.setenv("CRYPTOGRAPH_GIT_COMMIT", "abc1234")
    monkeypatch.setenv("CRYPTOGRAPH_GIT_DIRTY", "false")
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "nonexistent"))      # make git fail
    info = git_info(tmp_path)
    assert info["commit"] == "abc1234" and info["dirty"] is False
