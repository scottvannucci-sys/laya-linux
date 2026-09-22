"""Manifest schema, construction, and validation (unit)."""


import pytest

from laya_linux import manifest
from laya_linux.errors import ModelIncompleteError


def test_validate_rejects_missing_keys():
    with pytest.raises(ModelIncompleteError):
        manifest.validate_manifest({"format": "laya-linux"})
    with pytest.raises(ModelIncompleteError):
        manifest.validate_manifest("not a dict")
    with pytest.raises(ModelIncompleteError):
        manifest.validate_manifest({"format": "other", "format_version": 1})


def test_validate_rejects_bad_version():
    base = {
        "format": "laya-linux",
        "format_version": 99,
        "model_family": "laya",
        "checkpoint": "x",
        "license": "Apache-2.0",
        "dtype": "F32",
        "architecture": {"encoder": "modernbert", "hidden_size": 64, "context_limit": 512},
        "files": {"model.safetensors": {"size": 1, "sha256": "0" * 64}},
    }
    with pytest.raises(ModelIncompleteError):
        manifest.validate_manifest(base)
    base["format_version"] = 1
    assert manifest.validate_manifest(base)["format_version"] == 1


def test_validate_rejects_bad_file_records():
    base = {
        "format": "laya-linux",
        "format_version": 1,
        "model_family": "laya",
        "checkpoint": "x",
        "license": "Apache-2.0",
        "dtype": "F32",
        "architecture": {"encoder": "modernbert", "hidden_size": 64, "context_limit": 512},
        "files": {"model.safetensors": {"size": 1, "sha256": "short"}},
    }
    with pytest.raises(ModelIncompleteError):
        manifest.validate_manifest(base)
    base["files"] = {}
    with pytest.raises(ModelIncompleteError):
        manifest.validate_manifest(base)


def test_inspect_safetensors_dtype_header_only(tiny_pkg):
    dtype = manifest.inspect_safetensors_dtype(tiny_pkg / "model.safetensors")
    assert dtype == "F32"


def test_build_manifest_covers_all_files(tiny_pkg):
    built = manifest.build_manifest(tiny_pkg, license_id="Apache-2.0")
    assert built["format"] == "laya-linux"
    assert built["checkpoint"]
    assert set(built["files"]) >= {
        "model.safetensors", "rl_agent_config.json", "encoder/config.json",
        "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json",
    }
    record = built["files"]["model.safetensors"]
    assert record["size"] == (tiny_pkg / "model.safetensors").stat().st_size
    assert len(record["sha256"]) == 64
    assert built["dtype"] == "F32"
    assert built["architecture"]["encoder"] == "modernbert"


def test_write_and_load_roundtrip(tiny_pkg, tmp_path):
    import shutil

    copy = tmp_path / "copy"
    shutil.copytree(tiny_pkg, copy)
    built = manifest.build_manifest(copy, license_id="Apache-2.0")
    destination = manifest.write_manifest(copy, built)
    assert destination.name == "manifest.json"
    loaded = manifest.load_manifest(copy)
    assert loaded == built


def test_check_manifest_files_detects_tamper(tiny_pkg, tmp_path):
    import shutil

    copy = tmp_path / "tampered"
    shutil.copytree(tiny_pkg, copy)
    built = manifest.build_manifest(copy, license_id="Apache-2.0")
    assert manifest.check_manifest_files(copy, built) == []
    target = copy / "rl_agent_config.json"
    target.write_text(target.read_text() + " ")
    problems = manifest.check_manifest_files(copy, built)
    assert problems and problems[0]["file"] == "rl_agent_config.json"


def test_validate_architecture_info_rejects_unknown_encoder():
    with pytest.raises(ModelIncompleteError):
        manifest.validate_architecture_info({"model_type": "bert"})
    info = manifest.validate_architecture_info({"model_type": "modernbert", "hidden_size": 64})
    assert info["encoder"] == "modernbert" and info["hidden_size"] == 64
