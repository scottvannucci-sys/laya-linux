# Release checklist (requirements §13, §17)

Every release produces the artifacts below, in this order. Steps 1–3 are
automated by committed scripts; nothing is hand-assembled.

## 1. Build artifacts

```bash
python scripts/test_clean_install.py        # includes a fresh wheel build
python scripts/generate_sbom.py             # -> dist/laya-linux-sbom.json
```

`test_clean_install.py` is the per-release gate: it builds the wheel fresh,
creates a throwaway venv, installs **strictly offline** (`--no-index` from the
wheelhouse), and proves verify+predict work there with socket creation blocked.
Exit code 0 is mandatory before tagging.

## 2. Checksums for release artifacts

```bash
cd dist && sha256sum laya_linux-*.whl laya-linux-sbom.json > SHA256SUMS
```

## 3. Validation evidence attached to the release

- CPU test results: the GitHub Actions run for the release commit (green on
  Python 3.11 and 3.12).
- GPU validation: the JSON artifacts in `benchmarks/` for each documented
  configuration (`docs/cuda-setup.md` table). A release MUST NOT claim support
  for a hardware configuration without its artifact.
- Numerical parity claims: the tolerance table in `PARITY_BASELINES.md`.

## 4. Release notes must state

- The torch build line the SBOM records (e.g. `2.14.0+cpu`) — users build their
  wheelhouse to match before installing (see `docs/offline-installation.md`).
- Model-manifest tooling compatibility: which `format_version` values this
  release reads (`manifest.SUPPORTED_FORMAT_VERSIONS`).
- Any change to the client/server protocol version, with migration notes.

## 5. Tag and publish

```bash
git tag -s v0.1.0 -m "laya-linux 0.1.0" && git push origin v0.1.0
```

Attach to the GitHub release: the wheel, the SBOM, `SHA256SUMS`, and the
validation JSON artifacts. Do not attach model weights (§13).
