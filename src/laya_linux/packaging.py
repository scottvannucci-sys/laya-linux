"""Offline packaging: turn a raw upstream checkpoint into a laya-linux package.

Packaging is a user-invoked staging operation (requirements §7.1: never part
of normal runtime behavior). It copies a complete upstream-layout checkpoint
into a new directory, records a manifest with per-file SHA-256 digests, and
verifies the result before returning. The source directory is never modified.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from . import checkpoints
from . import manifest as manifest_mod
from .errors import ModelIncompleteError

COPIED_ITEMS = ("model.safetensors", "rl_agent_config.json", "encoder", "tokenizer")


def package_model(
    source_dir: str | Path,
    output_dir: str | Path,
    *,
    license_id: str,
    checkpoint_name: str | None = None,
    source_project: str = "",
    source_revision: str = "",
    source_subfolder: str = "",
) -> Path:
    """Package a local upstream checkpoint into ``output_dir`` with a manifest.

    ``license_id`` records the checkpoint's license as verified from its model
    card at packaging time; pass the identifier you actually verified. The
    destination must not already exist — existing packages are never
    overwritten. If packaging fails, the partial destination is removed.
    """
    source = checkpoints.resolve_model_dir(source_dir)
    info = checkpoints.load_checkpoint(source)  # completeness + config validation
    destination = Path(output_dir).expanduser()
    if destination.exists():
        raise ModelIncompleteError(
            f"Output directory already exists: {destination}; packaging never overwrites"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir()
    try:
        for item in COPIED_ITEMS:
            src = source / item
            dst = destination / item
            if src.is_dir():
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
        built = manifest_mod.build_manifest(
            destination,
            checkpoint_name=checkpoint_name or str(info["agent_config"].get("model_name") or source.name),
            license_id=license_id,
            source_project=source_project,
            source_revision=source_revision,
            source_subfolder=source_subfolder,
        )
        manifest_mod.write_manifest(destination, built)
        # The packaged copy must pass the same verification users will run.
        verify_model(destination)
    except BaseException:
        if destination.exists():
            shutil.rmtree(destination, ignore_errors=True)
        raise
    return destination


def verify_model(model_dir: str | Path) -> dict[str, Any]:
    """Perform ALL non-inference validation of a model package, offline.

    Returns a structured report. Raises on any failure. This is the engine
    behind ``laya-linux verify``: required files, configuration sanity, manifest
    checksums (when present), tokenizer special tokens, and strict
    parameter-name/shape validation against the constructed architecture.
    """
    import torch
    from safetensors.torch import load_file

    from .model import build_decision_model, verify_weights
    from .tokenizer import Tokenizer

    report: dict[str, Any] = {"offline": True}
    info = checkpoints.load_checkpoint(model_dir)  # paths, files, configs, manifest checksums
    model_dir_path: Path = info["dir"]
    report["path_resolved"] = str(model_dir_path)
    report["manifest"] = "present and verified" if info["manifest"] else "absent (optional)"
    agent_cfg = info["agent_config"]
    encoder_cfg = info["encoder_config"]

    report["architecture"] = manifest_mod.validate_architecture_info(encoder_cfg)
    report["max_len"] = agent_cfg.get("max_len", 512)
    report["head_max_len"] = agent_cfg.get("head_max_len", 192)
    temperatures = [*agent_cfg.get("temperature", []), *agent_cfg.get("temperature_by_options", {}).values()]
    report["calibration"] = "valid" if temperatures and all(float(t) > 0 for t in temperatures) else "defaults"

    weights = load_file(str(model_dir_path / "model.safetensors"))
    report["dtype"] = manifest_mod.inspect_safetensors_dtype(model_dir_path / "model.safetensors")
    report["tensor_count"] = len(weights)
    model = build_decision_model(encoder_cfg, agent_cfg)
    verify_weights(model, weights)
    report["weights"] = "strict match: names, shapes, and coverage verified"
    del weights, model

    tok = Tokenizer(model_dir_path / "tokenizer")
    report["tokenizer"] = {
        "special_tokens": {n: getattr(tok, n + "_id") for n in ("cls_token", "sep_token", "pad_token", "mask_token")},
        "valid": True,
    }
    if torch.cuda.is_available():
        report["note"] = "verification ran with this machine's local device information only"
    return report
