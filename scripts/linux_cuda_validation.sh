#!/usr/bin/env bash
# Phase 3 Linux CUDA validation for laya-linux (run on a Linux NVIDIA machine).
#
# Usage:
#   cd /path/to/laya-linux            # a clone of the repo
#   bash scripts/linux_cuda_validation.sh /path/to/model-package [out.json]
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

echo "== laya-linux Linux CUDA validation =="
echo "repo:       $REPO"
echo "model pkg:  $MODEL_PKG"
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv || true

# venv + CUDA torch (cu128 line supports Blackwell and older; adjust if your
# driver predates CUDA 12.8, e.g. use /whl/cu126)
if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv "$VENV"
  "$VENV/bin/python" -m pip install --quiet --upgrade pip
  "$VENV/bin/python" -m pip install --quiet torch --index-url https://download.pytorch.org/whl/cu128 \
    || "$VENV/bin/python" -m pip install --quiet torch --index-url https://download.pytorch.org/whl/cu126
fi
"$VENV/bin/python" -m pip install --quiet numpy safetensors tokenizers
"$VENV/bin/python" -m pip install --quiet --no-deps -e "$REPO"

"$VENV/bin/python" - <<PY
import torch
assert torch.cuda.is_available(), "CUDA not available to the installed torch build"
print("torch:", torch.__version__, "| cuda:", torch.version.cuda, "| gpu:", torch.cuda.get_device_name(0))
PY

"$VENV/bin/python" "$REPO/scripts/cuda_validation.py" "$MODEL_PKG" --out "$OUT"

echo
echo "== COPY EVERYTHING BELOW INTO THE MAINTAINER REPLY =="
cat "$OUT"
