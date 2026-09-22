"""Generate a CycloneDX 1.5 SBOM for the laya-linux release environment.

Inventory rule (requirements §13 "SBOM or dependency inventory"): the SBOM
lists the laya-linux package itself plus the transitive closure of its core
runtime dependencies as actually installed in the build environment — i.e.
exactly what the accompanying offline wheelhouse ships. Torch build variants
(+cpu / +cu128) are visible in the version string, which is the point: the SBOM
documents which build line a release was validated against.

Stdlib only; deterministic ordering; no build-time dependencies added.
"""

from __future__ import annotations

import argparse
import importlib.metadata as im
import json
import re
import uuid
from datetime import datetime, UTC
from email.utils import parseaddr
from pathlib import Path
from urllib.parse import quote

ROOT_DEPS = ("torch", "numpy", "safetensors", "tokenizers")


def canonical(name: str) -> str:
    """PEP 503 normalization."""
    return re.sub(r"[-_.]+", "-", name).lower()


def purl(name: str, version: str) -> str:
    return f"pkg:pypi/{canonical(name)}@{quote(version, safe='.-_')}"


def requirement_name(req_line: str) -> str | None:
    """Extract the distribution name from a Requires-Dist line."""
    line = req_line.split(";", 1)[0].strip()
    if not line:
        return None
    # handle extras and environment markers: name[extra1,extra2] (>=1.0) ; marker
    base = line.split("[", 1)[0]
    base = base.split("(", 1)[0].strip()
    # name may be followed directly by a version spec without space
    match = re.match(r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?", base)
    return match.group(0) if match else None


def walk_closure() -> dict[str, im.Distribution]:
    """Dependency closure of the core runtime deps from installed metadata."""
    by_name: dict[str, im.Distribution] = {}
    for dist in im.distributions():
        name = dist.metadata.get("Name")
        if name:
            by_name[canonical(name)] = dist

    pending = [canonical(ROOT_DEPS[0])]
    # Prefer the installed package's own declared deps as roots when available.
    self_dist = by_name.get("laya-linux")
    if self_dist is not None:
        declared = [
            requirement_name(line)
            for line in (self_dist.metadata.get_all("Requires-Dist") or [])
        ]
        roots = [canonical(n) for n in declared if n]
        if roots:
            pending = roots

    closure: dict[str, im.Distribution] = {}
    queue = list(dict.fromkeys(pending))
    while queue:
        key = queue.pop(0)
        if key in closure or key not in by_name:
            continue  # unknown/not-installed names are reported by the caller
        dist = by_name[key]
        closure[key] = dist
        for line in dist.metadata.get_all("Requires-Dist") or []:
            marker = line.split(";", 1)[1] if ";" in line else ""
            if "extra ==" in marker:
                continue  # only active when optional extras are installed
            dep = requirement_name(line)
            if dep and canonical(dep) not in closure:
                queue.append(canonical(dep))
    return closure


def license_of(dist: im.Distribution) -> list[dict] | None:
    fields = dist.metadata
    expr = fields.get("License-Expression") or None
    if expr:
        return [{"expression": expr.strip()}]
    name = fields.get("License")
    if name and len(name) < 200:
        return [{"name": name.strip()}]
    for classifier in fields.get_all("Classifier") or []:
        if classifier.startswith("License ::"):
            parts = [p.strip() for p in classifier.split("::") if p.strip()]
            return [{"name": parts[-1]}]
    return None


def build_sbom() -> dict:
    closure = walk_closure()
    self_name = "laya-linux"
    # laya-linux is the application component, not a dependency of itself;
    # resolve it from installed metadata directly.
    from importlib.metadata import distributions

    self_installed = next((d for d in distributions()
                           if (d.metadata.get("Name") or "").lower() == self_name), None)
    if self_installed is not None:
        closure[self_name] = self_installed
        missing_hint = None
    else:
        missing_hint = "laya-linux itself is not installed in this environment; install -e . first"

    components = []
    for key in sorted(closure):
        dist = closure[key]
        metadata = dist.metadata
        if key == "laya-linux":
            continue  # added below as the application component with fixed metadata
        component = {
            "type": "library",
            "bom-ref": purl(key, metadata.get("Version", "0")),
            "name": key,
            "version": metadata.get("Version", "0"),
            "purl": purl(key, metadata.get("Version", "0")),
            "description": (metadata.get("Summary") or "")[:300],
        }
        licenses = license_of(dist)
        if licenses:
            component["licenses"] = licenses
        author = parseaddr(metadata.get("Author") or metadata.get("Author-email") or "")[0]
        if author:
            component["supplier"] = {"name": author}
        components.append(component)

    self_name = "laya-linux"
    self_version = __import__("laya_linux").__version__ if _importable() else "unknown"
    components.append({
        "type": "application",
        "bom-ref": purl(self_name, self_version),
        "name": self_name,
        "version": self_version,
        "purl": purl(self_name, self_version),
        "description": "Local-first inference runtime for Laya typed-decision models on Linux",
        "supplier": {"name": "laya-linux contributors"},
    })

    components.sort(key=lambda c: (c["name"], c["version"]))
    content_blob = json.dumps(components, sort_keys=True).encode()
    serial = uuid.uuid5(uuid.NAMESPACE_URL, f"laya-linux-sbom@{self_version}+{content_blob[:64].hex()}")
    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "serialNumber": f"urn:uuid:{serial}",
        "version": 1,
        "metadata": {
            "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
            "tools": [{"name": "laya-linux generate_sbom", "version": self_version}],
            "component": {
                "type": "application",
                "bom-ref": purl(self_name, self_version),
                "name": self_name,
                "version": self_version,
            },
            "properties": [
                {"name": "laya-linux:sbom-scope",
                 "value": "core runtime dependency closure installed in the release build environment"},
            ],
        },
        "components": components,
    }, missing_hint


def _importable() -> bool:
    try:
        import laya_linux  # noqa: F401

        return True
    except Exception:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--out", default="dist/laya-linux-sbom.json")
    args = parser.parse_args()
    sbom, hint = build_sbom()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(sbom, indent=2) + "\n", encoding="utf-8")
    print(f"SBOM written: {out} ({len(sbom['components'])} components)")
    if hint:
        print(f"NOTE: {hint}")
    unknown = set(ROOT_DEPS) - {c["name"] for c in sbom["components"]}
    if unknown:
        print(f"WARNING: root dependencies not found in this environment: {sorted(unknown)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
