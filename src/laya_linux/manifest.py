"""The laya-linux model-package manifest: schema, construction, validation.

The manifest is the integrity and provenance record of a model package
(requirements §9, architecture §5.6). Schema rules:

- ``format`` MUST be ``laya-linux``; ``format_version`` MUST be an integer this
  runtime supports.
- Every entry in ``files`` MUST carry the file's byte size and SHA-256 digest.
- The ``license`` field records the checkpoint's license as verified at
  packaging time; it is never assumed to match the runtime license.
- Unknown keys are allowed (forward compatibility); a breaking schema change
  requires a ``format_version`` bump.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .errors import ModelIncompleteError
from .version import __version__

MANIFEST_FORMAT = "laya-linux"
SUPPORTED_FORMAT_VERSIONS = (1,)
MANIFEST_NAME = "manifest.json"
CHUNK = 1024 * 1024

REQUIRED_KEYS = ("format", "format_version", "model_family", "checkpoint", "license", "dtype", "architecture", "files")
REQUIRED_ARCHITECTURE_KEYS = ("encoder", "hidden_size", "context_limit")


def sha256_file(file_path: Path) -> str:
    digest = hashlib.sha256()
    with open(file_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_manifest(manifest: Any) -> dict[str, Any]:
    """Validate a parsed manifest, returning it normalized. Raises on any violation."""
    if not isinstance(manifest, dict):
        raise ModelIncompleteError("Manifest must be a JSON object")
    for key in REQUIRED_KEYS:
        if key not in manifest:
            raise ModelIncompleteError(f"Manifest is missing required key {key!r}")
    if manifest["format"] != MANIFEST_FORMAT:
        raise ModelIncompleteError(
            f"Manifest format is {manifest['format']!r}; expected {MANIFEST_FORMAT!r}"
        )
    version = manifest["format_version"]
    if not isinstance(version, int) or isinstance(version, bool) or version not in SUPPORTED_FORMAT_VERSIONS:
        raise ModelIncompleteError(
            f"Manifest format_version {version!r} is not supported by this runtime "
            f"(supported: {SUPPORTED_FORMAT_VERSIONS})"
        )
    for key in ("model_family", "checkpoint", "license", "dtype"):
        if not isinstance(manifest[key], str) or not manifest[key]:
            raise ModelIncompleteError(f"Manifest key {key!r} must be a nonempty string")
    architecture = manifest["architecture"]
    if not isinstance(architecture, dict):
        raise ModelIncompleteError("Manifest 'architecture' must be an object")
    for key in REQUIRED_ARCHITECTURE_KEYS:
        if key not in architecture:
            raise ModelIncompleteError(f"Manifest 'architecture' is missing {key!r}")
    if not isinstance(architecture["hidden_size"], int) or not isinstance(architecture["context_limit"], int):
        raise ModelIncompleteError("Manifest architecture sizes must be integers")
    files = manifest["files"]
    if not isinstance(files, dict) or not files:
        raise ModelIncompleteError("Manifest 'files' must be a nonempty object")
    for relative, record in files.items():
        if not isinstance(record, dict):
            raise ModelIncompleteError(f"Manifest file record for {relative!r} must be an object")
        size, digest = record.get("size"), record.get("sha256")
        if not isinstance(size, int) or size < 0:
            raise ModelIncompleteError(f"Manifest record for {relative!r} has no valid size")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ModelIncompleteError(f"Manifest record for {relative!r} has no valid sha256")
    return manifest


def inspect_safetensors_dtype(file_path: Path) -> str:
    """Read the stored dtype from a safetensors header without loading tensors.

    Only the header bytes are read (the 8-byte length prefix plus the JSON
    metadata), so this stays cheap even for multi-gigabyte weight files.
    """
    import json
    import struct

    with open(file_path, "rb") as handle:
        raw_length = handle.read(8)
        if len(raw_length) < 8:
            raise ModelIncompleteError("Safetensors file is too small to have a valid header")
        (header_length,) = struct.unpack("<Q", raw_length)
        header = json.loads(handle.read(header_length))
    header.pop("__metadata__", None)
    dtypes = {record.get("dtype") for record in header.values() if isinstance(record, dict)}
    if not dtypes:
        raise ModelIncompleteError("Safetensors file contains no tensors")
    if len(dtypes) == 1:
        return str(next(iter(dtypes)))
    return "mixed(" + ",".join(sorted(str(d) for d in dtypes)) + ")"


def build_manifest(
    source_dir: Path,
    *,
    checkpoint_name: str | None = None,
    license_id: str = "unknown",
    source_project: str = "",
    source_revision: str = "",
    source_subfolder: str = "",
) -> dict[str, Any]:
    """Build a manifest for the model package rooted at ``source_dir``.

    Checksums cover every file in the package except the manifest itself. The
    dtype and architecture fields are read from the package's own
    configuration; ``license`` must be supplied by the user after verifying the
    model card at packaging time.
    """
    source_dir = Path(source_dir)
    from .checkpoints import REQUIRED_FILES, _read_json

    agent_cfg = _read_json(source_dir, "rl_agent_config.json")
    encoder_cfg = _read_json(source_dir, "encoder/config.json")
    weights = source_dir / "model.safetensors"

    files: dict[str, Any] = {}
    for relative in sorted({*REQUIRED_FILES}):
        file_path = source_dir / relative
        files[relative] = {"size": file_path.stat().st_size, "sha256": sha256_file(file_path)}
    for extra in sorted(p for p in source_dir.rglob("*") if p.is_file()):
        relative = extra.relative_to(source_dir).as_posix()
        if relative == MANIFEST_NAME or relative in files:
            continue
        files[relative] = {"size": extra.stat().st_size, "sha256": sha256_file(extra)}

    manifest: dict[str, Any] = {
        "format": MANIFEST_FORMAT,
        "format_version": SUPPORTED_FORMAT_VERSIONS[-1],
        "model_family": "laya",
        "checkpoint": checkpoint_name or str(agent_cfg.get("model_name") or source_dir.name),
        "license": license_id,
        "dtype": inspect_safetensors_dtype(weights),
        "architecture": {
            "encoder": str(encoder_cfg.get("model_type", "unknown")),
            "hidden_size": int(encoder_cfg.get("hidden_size", 0)),
            "context_limit": int(encoder_cfg.get("max_position_embeddings", 0)),
        },
        "source": {
            "project": source_project,
            "revision": source_revision,
            "subfolder": source_subfolder or None,
        },
        "runtime_compat": {"laya_linux": f">={__version__.split('.')[0]}.{__version__.split('.')[1]}"},
        "files": files,
        "packaged_by": {"laya_linux_version": __version__, "package_id": str(uuid.uuid4())},
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    return validate_manifest(manifest)


def write_manifest(model_dir: Path, manifest: dict[str, Any]) -> Path:
    """Write manifest.json atomically (temp file + replace) inside the package."""
    validate_manifest(manifest)
    destination = Path(model_dir) / MANIFEST_NAME
    temp = destination.with_name(MANIFEST_NAME + f".tmp-{uuid.uuid4().hex[:8]}")
    temp.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temp.replace(destination)
    return destination


def load_manifest(model_dir: Path) -> dict[str, Any]:
    """Load and validate manifest.json from a model package."""
    from .checkpoints import _read_json

    manifest = _read_json(Path(model_dir), MANIFEST_NAME)
    return validate_manifest(manifest)


def check_manifest_files(model_dir: Path, manifest: dict[str, Any]) -> list[dict[str, str]]:
    """Verify every manifest-listed file exists with the recorded size and digest."""
    model_dir = Path(model_dir)
    problems: list[dict[str, str]] = []
    for relative, record in sorted(manifest["files"].items()):
        target = model_dir / relative
        if not target.is_file():
            problems.append({"file": relative, "status": "missing"})
            continue
        actual_size = target.stat().st_size
        if actual_size != record["size"]:
            problems.append({"file": relative, "status": f"size mismatch: {actual_size} != {record['size']}"})
            continue
        if sha256_file(target) != record["sha256"]:
            problems.append({"file": relative, "status": "sha256 mismatch"})
    return problems


def validate_architecture_info(encoder_cfg: dict[str, Any]) -> dict[str, Any]:
    """Summarize an encoder config into the manifest architecture shape.

    Raises when the encoder is not a supported ModernBERT configuration, so
    verification rejects incompatible checkpoints before any tensor is read.
    """
    model_type = encoder_cfg.get("model_type")
    if model_type != "modernbert":
        raise ModelIncompleteError(
            f"Unsupported encoder {model_type!r}; this runtime implements 'modernbert' only"
        )
    return {
        "encoder": "modernbert",
        "hidden_size": int(encoder_cfg.get("hidden_size", 0)),
        "context_limit": int(encoder_cfg.get("max_position_embeddings", 0)),
        "num_hidden_layers": int(encoder_cfg.get("num_hidden_layers", 0)),
        "num_attention_heads": int(encoder_cfg.get("num_attention_heads", 0)),
        "local_attention": int(encoder_cfg.get("local_attention", 128)),
        "layer_types": list(encoder_cfg.get("layer_types") or []),
    }
