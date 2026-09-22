# Offline installation

Laya-Linux is designed for air-gapped machines: nothing in the runtime performs a network
operation, and installation itself can be completed entirely from removable media.

## 1. Build a wheelhouse on a staging machine

On a machine with internet access, download the Python wheels for your target platform and
Python version:

```bash
# CPU-only PyTorch (smallest); substitute the CUDA or ROCm wheel for GPU machines
pip download torch --index-url https://download.pytorch.org/whl/cpu -d wheelhouse/

# Core runtime dependencies
pip download numpy safetensors tokenizers -d wheelhouse/

# The laya-linux wheel itself
pip wheel --no-deps -w dist/ .
```

Copy `wheelhouse/`, `dist/`, and the model package (below) to removable media.

> **GPU note:** CUDA and ROCm PyTorch builds are platform-specific. Download the exact wheel for
> the target machine's CUDA/ROCm version and Python version; see <https://pytorch.org> for the
> matrix. Never rely on a Python extra to pick the GPU build for you.

## 2. Install on the air-gapped machine

```bash
python -m venv .venv && source .venv/bin/activate
pip install --no-index --find-links wheelhouse/ torch numpy safetensors tokenizers
pip install --no-index --find-links dist/ laya-linux-<version>-py3-none-any.whl
```

`--no-index` forbids PyPI access entirely; if anything is missing the install fails loudly
instead of reaching for the network.

Verify the installation without a model:

```bash
laya-linux doctor
```

`doctor` reports verified facts (versions, devices) and, separately, suggestions. It never
requires a model or a network connection.

## 3. Transfer a model package

Models are never downloaded by the runtime. On the staging machine, either copy a checkpoint
you have already acquired, or package it with checksums:

```bash
laya-linux verify /staging/laya-typed-decisions        # confirm the source is complete
python - <<'PY'
from laya_linux.packaging import package_model
package_model(
    "/staging/laya-typed-decisions",
    "/media/usb/laya-typed-decisions",
    license_id="Apache-2.0",   # the license you verified on the model card
    source_project="convaiinnovations/laya-typed-decisions (Hugging Face)",
    source_revision="f9ab0b228f0fc0f14d873dbc99038f135c2da1b2",
)
PY
```

`package_model` writes a `manifest.json` containing the byte size and SHA-256 digest of every
file, so integrity can be re-checked on the target with no internet access.

On the air-gapped machine:

```bash
laya-linux verify /opt/laya/models/laya-typed-decisions
laya-linux predict /opt/laya/models/laya-typed-decisions \
    --state "I was charged twice." --preset triage
```

`verify` performs all non-inference validation offline: required files, configuration sanity,
manifest checksums, tokenizer special tokens, and strict parameter-name/shape checks.

## 4. Checksums without internet

Every digest lives inside the package's `manifest.json` and is verified with the standard
`hashlib` — no downloads, no key servers. To verify before trusting a package:

```bash
python - <<'PY'
from laya_linux import manifest
problems = manifest.check_manifest_files("/opt/laya/models/laya-typed-decisions",
                                         manifest.load_manifest("/opt/laya/models/laya-typed-decisions"))
print(problems or "all files match the manifest")
PY
```

A manifest is optional; without one, `verify` still validates structure, configuration,
tokenizer, and tensor names/shapes, but cannot detect content corruption — prefer manifest-carrying
packages for air-gapped transfer.

## 5. What must never happen

- The runtime must never download a model or dependency. If any command tries, that is a bug —
  the test suite enforces this by blocking sockets during import, load, and prediction.
- Do not fetch models on the air-gapped machine "just this once." Use the staging process above.
