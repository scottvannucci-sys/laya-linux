"""Strict local checkpoint loading (architecture §5.6, requirements §9).

Checkpoints are self-contained local directories in the upstream Laya layout::

    model.safetensors
    rl_agent_config.json
    encoder/config.json
    tokenizer/tokenizer.json
    tokenizer/tokenizer_config.json

Optionally, a ``manifest.json`` in the laya-linux package format adds integrity
requirements; when present, its checksums are verified. Loading never writes to
the model directory and never resolves anything off-machine.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .errors import ModelChecksumError, ModelIncompleteError, ModelNotFoundError

REQUIRED_FILES = (
    "model.safetensors",
    "rl_agent_config.json",
    "encoder/config.json",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
)

MAX_CONFIG_BYTES = 4 * 1024 * 1024  # bounded config sizes
CHUNK = 1024 * 1024


def resolve_model_dir(path: str | os.PathLike) -> Path:
    """Resolve and normalize a local model path, rejecting anything remote-shaped.

    A string that does not resolve to a local directory fails here, before any
    network-capable library could be imported or called (architecture §5.2).
    """
    raw = str(path)
    if raw.startswith(("http://", "https://", "hf://")):
        from .errors import NetworkDisabledError

        raise NetworkDisabledError(
            f"Remote model identifiers are not supported by the core runtime: {raw!r}. "
            f"Stage the model locally and pass a filesystem path."
        )
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    candidate = candidate.resolve()
    if not candidate.is_dir():
        looks_like_hub_id = (
            not raw.startswith(("/", "./", "../", "\\\\", "C:", "D:"))
            and "/" in raw
            and not Path(raw).exists()
        )
        if looks_like_hub_id:
            from .errors import NetworkDisabledError

            raise NetworkDisabledError(
                f"{raw!r} looks like a remote model identifier. The core runtime never downloads "
                f"models and has no network capability; stage the model locally and pass its "
                f"filesystem path."
            )
        raise ModelNotFoundError(
            f"Local model directory not found: {candidate}. "
            f"The runtime never downloads models; stage a model package locally first."
        )
    return candidate


def contained(model_dir: Path, relative: str) -> Path:
    """Join a relative path and guarantee the result stays inside the model directory."""
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ModelIncompleteError(f"Illegal file reference in model package: {relative!r}")
    resolved = (model_dir / relative).resolve()
    root = model_dir.resolve()
    if resolved != root and root not in resolved.parents:
        raise ModelIncompleteError(f"Path traversal detected in model package: {relative!r}")
    return resolved


def _read_json(model_dir: Path, relative: str) -> dict[str, Any]:
    file_path = contained(model_dir, relative)
    if not file_path.is_file():
        raise ModelIncompleteError(f"Model package is missing {relative!r}")
    size = file_path.stat().st_size
    if size > MAX_CONFIG_BYTES:
        raise ModelIncompleteError(f"Model package file {relative!r} exceeds the size limit")
    try:
        return json.loads(file_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ModelIncompleteError(f"Model package file {relative!r} is not valid JSON: {exc}") from exc


def sha256_file(file_path: Path) -> str:
    digest = hashlib.sha256()
    with open(file_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest(model_dir: Path) -> dict[str, Any] | None:
    """Verify manifest.json checksums when present; return the manifest or None.

    Absence of a manifest is not an error (upstream checkpoints ship without
    one); presence with a wrong checksum is a hard failure.
    """
    manifest_path = model_dir / "manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = _read_json(model_dir, "manifest.json")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ModelChecksumError("Manifest present but contains no file records")
    for relative, record in sorted(files.items()):
        expected = record.get("sha256") if isinstance(record, dict) else None
        if not expected:
            raise ModelChecksumError(f"Manifest record for {relative!r} has no sha256")
        target = contained(model_dir, relative)
        if not target.is_file():
            raise ModelIncompleteError(f"Manifest lists {relative!r} but the file is missing")
        actual = sha256_file(target)
        if actual != expected:
            raise ModelChecksumError(f"Checksum mismatch for {relative!r}: model file may be corrupt")
    return manifest


def load_checkpoint(model_dir: Path) -> dict[str, Any]:
    """Inspect a staged checkpoint and return its parsed configuration files.

    This is the non-inference validation entry point; tensor materialization
    happens in the agent. All failures are actionable and offline.
    """
    model_dir = resolve_model_dir(model_dir)
    for relative in REQUIRED_FILES:
        file_path = contained(model_dir, relative)
        if not file_path.is_file():
            raise ModelIncompleteError(
                f"Not a complete Laya checkpoint: {relative!r} is missing from {model_dir}"
            )
    manifest = verify_manifest(model_dir)
    agent_cfg = _read_json(model_dir, "rl_agent_config.json")
    encoder_cfg = _read_json(model_dir, "encoder/config.json")
    for key in ("encoder", "head_layers"):
        if key not in agent_cfg:
            raise ModelIncompleteError(
                f"rl_agent_config.json is missing required key {key!r}; "
                f"this does not appear to be a Laya decision-model checkpoint"
            )
    return {
        "dir": model_dir,
        "agent_config": agent_cfg,
        "encoder_config": encoder_cfg,
        "manifest": manifest,
    }
