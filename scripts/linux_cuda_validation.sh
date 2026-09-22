#!/usr/bin/env bash
# Phase 3 Linux CUDA validation for laya-linux (run on a Linux NVIDIA machine).
#
# Usage:
#   cd /path/to/laya-linux            # a clone of the repo
#   bash scripts/linux_cuda_validation.sh /path/to/model-package [out.json]
#
# Environment overrides (all optional):
#   PYTHON_BIN=python3.11   Python to build the venv from (needs >= 3.11)
#   TORCH_INDEX=https://download.pytorch.org/whl/cu126   torch wheel index
#   SKIP_VENV=1             use the ambient environment instead of a venv
#
# What it does: creates an isolated venv, installs the CUDA-enabled PyTorch
# build for this machine's driver, installs laya-linux from the local tree,
# runs the full validation battery (parity drift FP32/FP16/BF16 vs CPU FP32,
# memory stability, benchmarks), and prints a JSON block at the end.
#
# Paste the entire output (or just the JSON block) back to the maintainer for
# folding into PARITY_BASELINES.md.
set -euo pipefail

MODEL_PKG="${1:?usage: linux_cuda_validation.sh /path/to/model-package [out.json]}"
OUT="${2:-cuda-validation-linux.json}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$REPO/.venv-cuda"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"

echo "== laya-linux Linux CUDA validation =="
echo "repo:       $REPO"
echo "model pkg:  $MODEL_PKG"
echo "python:     $PYTHON_BIN"
echo "torch idx:  $TORCH_INDEX"
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv || true

# Dependency sanity checks with actionable messages.
command -v "$PYTHON_BIN" >/dev/null 2>&1 || {
  echo "ERROR: '$PYTHON_BIN' not found on PATH. Install Python >= 3.11 (e.g. 'sudo apt"
  echo "install python3.11 python3.11-venv') or rerun with PYTHON_BIN=/path/to/python."
  exit 1
}
PYVER=$("$PYTHON_BIN" -c 'import sys; print("%d.%d" % sys.version_info[:2])')
"$PYTHON_BIN" - "$PYVER" <<'PY'
import sys
major, minor = (int(x) for x in sys.argv[1].split("."))
if (major, minor) < (3, 11):
    sys.exit(f"ERROR: Python {major}.{minor} is too old; laya-linux needs >= 3.11")
PY
"$PYTHON_BIN" -m venv --help >/dev/null 2>&1 || {
  echo "ERROR: the venv module is missing. On Debian/Ubuntu: 'sudo apt install"
  echo "python3-venv' (or python3.11-venv for a specific interpreter)."
  exit 1
}

# venv + CUDA torch (cu128 line supports Blackwell and older; if the driver
# predates CUDA 12.8, rerun with TORCH_INDEX=https://download.pytorch.org/whl/cu126)
if [ "${SKIP_VENV:-0}" != "1" ]; then
  if [ ! -x "$VENV/bin/python" ]; then
    "$PYTHON_BIN" -m venv "$VENV"
    "$VENV/bin/python" -m pip install --quiet --upgrade pip
    "$VENV/bin/python" -m pip install --quiet torch --index-url "$TORCH_INDEX"
  fi
  PY="$VENV/bin/python"
else
  PY="$PYTHON_BIN"
fi
"$PY" -m pip install --quiet numpy safetensors tokenizers
"$PY" -m pip install --quiet --no-deps -e "$REPO"

"$PY" - <<PY
import torch
assert torch.cuda.is_available(), (
    "CUDA not available to the installed torch build - the most common causes are "
    "(a) a CPU-only torch wheel got installed (fix: delete the venv, rerun with "
    "TORCH_INDEX=https://download.pytorch.org/whl/cu128) or (b) the driver is older "
    "than the wheel's CUDA version (check 'nvidia-smi' and rerun with a matching "
    "TORCH_INDEX such as .../cu126 or .../cu121)."
)
print("torch:", torch.__version__, "| cuda:", torch.version.cuda, "| gpu:", torch.cuda.get_device_name(0))
PY

"$PY" "$REPO/scripts/cuda_validation.py" "$MODEL_PKG" --out "$OUT"

echo
echo "== COPY EVERYTHING BELOW INTO THE MAINTAINER REPLY =="
cat "$OUT"
