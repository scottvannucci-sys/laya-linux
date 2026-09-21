"""Checkpoint path resolution, containment, and manifest verification (unit)."""

import json

import pytest

from laya_linux import checkpoints
from laya_linux.errors import (
    ModelChecksumError,
    ModelIncompleteError,
    ModelNotFoundError,
    NetworkDisabledError,
)


def test_resolve_rejects_remote_identifiers():
    for bad in ("https://example.com/m", "http://x/y", "hf://convaiinnovations/laya"):
        with pytest.raises(NetworkDisabledError):
            checkpoints.resolve_model_dir(bad)


def test_resolve_missing_local_path_never_downloads(tmp_path):
    with pytest.raises(ModelNotFoundError):
        checkpoints.resolve_model_dir(str(tmp_path / "does-not-exist"))


def test_resolve_accepts_existing_dir(tiny_pkg):
    resolved = checkpoints.resolve_model_dir(str(tiny_pkg))
    assert resolved.is_dir()


def test_containment_blocks_traversal(tiny_pkg):
    with pytest.raises(ModelIncompleteError):
        checkpoints.contained(tiny_pkg, "../outside.json")
    with pytest.raises(ModelIncompleteError):
        checkpoints.contained(tiny_pkg, "/etc/passwd")
    assert checkpoints.contained(tiny_pkg, "rl_agent_config.json").is_file()


def test_load_checkpoint_requires_all_files(tiny_pkg):
    import shutil

    incomplete = tiny_pkg.parent / "incomplete"
    if incomplete.exists():
        shutil.rmtree(incomplete)
    shutil.copytree(tiny_pkg, incomplete)
    (incomplete / "model.safetensors").unlink()
    with pytest.raises(ModelIncompleteError):
        checkpoints.load_checkpoint(incomplete)


def test_load_checkpoint_rejects_bad_config(tiny_pkg):
    import shutil

    bad = tiny_pkg.parent / "badcfg"
    if bad.exists():
        shutil.rmtree(bad)
    shutil.copytree(tiny_pkg, bad)
    cfg = json.loads((bad / "rl_agent_config.json").read_text())
    del cfg["head_layers"]
    (bad / "rl_agent_config.json").write_text(json.dumps(cfg))
    with pytest.raises(ModelIncompleteError):
        checkpoints.load_checkpoint(bad)


def test_manifest_checksums_verified(tiny_pkg):
    import shutil

    from laya_linux.checkpoints import sha256_file

    good = tiny_pkg.parent / "manifest-good"
    bad = tiny_pkg.parent / "manifest-bad"
    for d in (good, bad):
        if d.exists():
            shutil.rmtree(d)
        shutil.copytree(tiny_pkg, d)
    weights = good / "model.safetensors"
    digest = sha256_file(weights)
    manifest = {
        "format": "laya-linux",
        "format_version": 1,
        "files": {"model.safetensors": {"size": weights.stat().st_size, "sha256": digest}},
    }
    (good / "manifest.json").write_text(json.dumps(manifest))
    assert checkpoints.verify_manifest(good) is not None

    manifest["files"]["model.safetensors"]["sha256"] = "0" * 64
    (bad / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ModelChecksumError):
        checkpoints.verify_manifest(bad)


def test_no_manifest_is_fine(tiny_pkg):
    assert checkpoints.verify_manifest(tiny_pkg) is None
