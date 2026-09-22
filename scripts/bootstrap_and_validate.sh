#!/usr/bin/env bash
# One-command Phase 3 Linux CUDA validation bootstrap for laya-linux.
#
# Run from a fresh clone of the repository:
#   bash scripts/bootstrap_and_validate.sh
#
# It handles, in order:
#   0. repo sanity (right repo, up to date enough)
#   1. system dependencies (git/curl if missing, Python >= 3.11 + venv module)
#   2. driver/GPU visibility check
#   3. checkpoint staging from the pinned Hugging Face revision (with size
#      verification of the weights file)
#   4. the full validation battery via linux_cuda_validation.sh
#
# Environment overrides (optional, passed through to the validation script):
#   PYTHON_BIN=python3.12   use a specific interpreter (needs >= 3.11)
#   TORCH_INDEX=.../cu126   torch wheel line matching an older driver
#   SKIP_APT=1              never call sudo apt (deps must already exist)
#
# The final JSON block is what the maintainer folds into PARITY_BASELINES.md.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
OUT="${1:-cuda-validation-linux.json}"
CKPT_DIR="models/laya-typed-decisions"
HF_BASE="https://huggingface.co/convaiinnovations/laya-typed-decisions/resolve/f9ab0b228f0fc0f14d873dbc99038f135c2da1b2"
WEIGHTS_BYTES=842609220   # pinned revision; a different size means a truncated download

say() { printf '\n=== %s ===\n' "$*"; }

# ---- 0. repo sanity ---------------------------------------------------------
say "0. Repo sanity"
[ -f scripts/cuda_validation.py ] && [ -f scripts/linux_cuda_validation.sh ] || {
  echo "ERROR: this does not look like a laya-linux clone (missing scripts/)."
  echo "       git clone https://github.com/scottvannucci-sys/laya-linux.git"
  exit 1
}
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git fetch origin main --quiet 2>/dev/null || true
  LOCAL=$(git rev-parse HEAD 2>/dev/null || echo none)
  ORIGIN=$(git rev-parse origin/main 2>/dev/null || echo none)
  if [ "$LOCAL" != "$ORIGIN" ]; then
    echo "NOTE: local checkout differs from origin/main; pulling latest scripts."
    git pull --ff-only origin main || echo "WARNING: could not fast-forward; continuing with local scripts."
  fi
fi

# ---- 1. system dependencies -------------------------------------------------
say "1. System dependencies"
APT=""
if [ "${SKIP_APT:-0}" != "1" ] && command -v apt-get >/dev/null 2>&1; then
  if command -v sudo >/dev/null 2>&1; then APT="sudo apt-get"; else APT="apt-get"; fi
fi

install_apt() {  # install_apt <pkg...>  (no-op when apt is unavailable)
  if [ -n "$APT" ]; then
    echo "installing: $*"
    $APT update -qq && $APT install -y -qq "$@"
  else
    return 1
  fi
}

command -v curl >/dev/null 2>&1 || install_apt curl || {
  echo "ERROR: curl is required (or wget - edit this script)."; exit 1
}

# Pick an interpreter >= 3.11: $PYTHON_BIN, python3.11, python3.12, python3
PY_OK=""
for CAND in "${PYTHON_BIN:-}" python3.11 python3.12 python3; do
  [ -z "$CAND" ] && continue
  command -v "$CAND" >/dev/null 2>&1 || continue
  if "$CAND" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null; then
    PY_OK="$CAND"; break
  fi
done
if [ -z "$PY_OK" ]; then
  echo "No Python >= 3.11 found. Attempting 'apt install python3.11 python3.11-venv'..."
  install_apt python3.11 python3.11-venv || {
    echo "ERROR: could not install Python 3.11 automatically."
    echo "  Ubuntu 22.04:  sudo apt install python3.11 python3.11-venv"
    echo "  Older distros: add the deadsnakes PPA, or install via pyenv."
    echo "  Then rerun with PYTHON_BIN=python3.11"
    exit 1
  }
  PY_OK=python3.11
fi
"$PY_OK" -m venv --help >/dev/null 2>&1 || install_apt python3-venv || {
  echo "ERROR: the venv module is missing: sudo apt install python3-venv (or ${PY_OK}-venv)."
  exit 1
}
echo "using interpreter: $PY_OK ($("$PY_OK" --version))"

# ---- 2. GPU visibility ------------------------------------------------------
say "2. GPU / driver"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found. Install the NVIDIA driver for this machine first"
  echo "       (https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html)."
  exit 1
fi
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv

# ---- 3. checkpoint staging --------------------------------------------------
say "3. Checkpoint staging (pinned revision, ~840 MB download)"
mkdir -p "$CKPT_DIR/tokenizer" "$CKPT_DIR/encoder"
fetch() {  # fetch <remote-path> <dest>
  if [ -s "$2" ]; then echo "already staged: $2"; return 0; fi
  echo "downloading: $1"
  curl -L --fail --retry 3 --progress-bar "$HF_BASE/$1" -o "$2"
}
fetch "rl_agent_config.json"        "$CKPT_DIR/rl_agent_config.json"
fetch "encoder/config.json"         "$CKPT_DIR/encoder/config.json"
fetch "tokenizer/tokenizer.json"    "$CKPT_DIR/tokenizer/tokenizer.json"
fetch "tokenizer/tokenizer_config.json" "$CKPT_DIR/tokenizer/tokenizer_config.json"
if [ ! -s "$CKPT_DIR/model.safetensors" ]; then
  echo "downloading model.safetensors (~840 MB)..."
  curl -L --fail --retry 3 --progress-bar "$HF_BASE/model.safetensors" -o "$CKPT_DIR/model.safetensors"
fi
ACTUAL=$(stat -c%s "$CKPT_DIR/model.safetensors" 2>/dev/null || wc -c < "$CKPT_DIR/model.safetensors")
if [ "$ACTUAL" != "$WEIGHTS_BYTES" ]; then
  echo "ERROR: model.safetensors is $ACTUAL bytes; expected $WEIGHTS_BYTES."
  echo "       The download is truncated or corrupt - delete the file and rerun this script."
  exit 1
fi
echo "checkpoint verified: $ACTUAL bytes"

# ---- 4. validation ----------------------------------------------------------
say "4. CUDA validation battery"
export PYTHON_BIN="${PYTHON_BIN:-$PY_OK}"
bash scripts/linux_cuda_validation.sh "$CKPT_DIR" "$OUT"

say "DONE - copy the JSON block above back to the maintainer"
