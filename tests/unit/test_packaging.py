"""Offline packaging and whole-package verification (unit)."""

import json
import shutil

import pytest

from laya_linux.errors import ModelChecksumError, ModelIncompleteError
from laya_linux.packaging import package_model, verify_model


@pytest.fixture()
def raw_pkg(tmp_path):
    from conftest import build_tiny_package

    return build_tiny_package(tmp_path / "raw", seed=13)


def test_package_creates_manifest_and_verifies(raw_pkg, tmp_path):
    out = tmp_path / "packaged"
    result = package_model(raw_pkg, out, license_id="Apache-2.0", source_revision="deadbeef")
    assert result == out
    mf = json.loads((out / "manifest.json").read_text())
    assert mf["format"] == "laya-linux" and mf["license"] == "Apache-2.0"
    assert "model.safetensors" in mf["files"]
    report = verify_model(out)
    assert report["manifest"] == "present and verified"
    assert "strict match" in report["weights"]


def test_package_refuses_to_overwrite(raw_pkg, tmp_path):
    out = tmp_path / "packaged"
    package_model(raw_pkg, out, license_id="Apache-2.0")
    with pytest.raises(ModelIncompleteError):
        package_model(raw_pkg, out, license_id="Apache-2.0")


def test_verify_raw_package_without_manifest(raw_pkg):
    report = verify_model(raw_pkg)
    assert report["manifest"] == "absent (optional)"
    assert report["tokenizer"]["valid"]


def test_verify_detects_tampered_weights(raw_pkg, tmp_path):
    out = tmp_path / "packaged"
    package_model(raw_pkg, out, license_id="Apache-2.0")
    weights = out / "model.safetensors"
    data = bytearray(weights.read_bytes())
    data[-1] ^= 0xFF
    weights.write_bytes(bytes(data))
    with pytest.raises(ModelChecksumError):
        verify_model(out)


def test_verify_detects_missing_file(raw_pkg, tmp_path):
    out = tmp_path / "packaged"
    package_model(raw_pkg, out, license_id="Apache-2.0")
    (out / "tokenizer" / "tokenizer.json").unlink()
    with pytest.raises(ModelIncompleteError):
        verify_model(out)


def test_verify_detects_wrong_shapes(raw_pkg):
    import torch
    from safetensors.torch import load_file, save_file

    broken = raw_pkg.parent / "broken-shapes"
    if broken.exists():
        shutil.rmtree(broken)
    shutil.copytree(raw_pkg, broken)
    weights = load_file(str(broken / "model.safetensors"))
    first = next(iter(weights))
    weights[first] = torch.randn(3, 3)  # wrong shape on purpose
    save_file(weights, str(broken / "model.safetensors"))
    from laya_linux.errors import ModelIncompatibleError

    with pytest.raises(ModelIncompatibleError, match="architecture mismatch"):
        verify_model(broken)


def test_verify_rejects_unsupported_encoder(raw_pkg):
    cfg_path = raw_pkg / "encoder" / "config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["model_type"] = "gpt2"
    cfg_path.write_text(json.dumps(cfg))
    with pytest.raises(ModelIncompleteError, match="encoder"):
        verify_model(raw_pkg)


def test_packaged_model_predicts_identically(raw_pkg, tmp_path, agent):
    """Packaging must not alter model behavior: packaged Agent == raw Agent."""
    from laya_linux import Agent

    out = package_model(raw_pkg, tmp_path / "packaged", license_id="Apache-2.0")
    from conftest import QUESTIONS, STATE

    a_raw = Agent(str(raw_pkg), device="cpu")
    a_pkg = Agent(str(out), device="cpu")
    assert a_raw.predict(STATE, QUESTIONS) == a_pkg.predict(STATE, QUESTIONS)
