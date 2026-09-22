"""Repeatable clean-environment wheel-install test (requirements §13, §17.1).

This is the Phase 2 exit gate made per-release: build the wheel, assemble an
offline wheelhouse (reusing ./wheelhouse when it already contains the needed
wheels), create a throwaway venv, install strictly with ``--no-index``, and
prove verify+predict work with socket creation blocked inside that fresh
environment.

Usage:  python scripts/test_clean_install.py [--keep]
Exit code 0 = the release is installable offline and functional.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WHEELHOUSE = REPO / "wheelhouse"
TOP_LEVEL = ("torch", "numpy", "safetensors", "tokenizers")
DEFAULT_TORCH_INDEX = "https://download.pytorch.org/whl/cpu"

TINY_ENC = {
    "model_type": "modernbert", "vocab_size": 204, "hidden_size": 64,
    "intermediate_size": 128, "num_hidden_layers": 2, "num_attention_heads": 2,
    "local_attention": 8, "global_attn_every_n_layers": 3,
    "max_position_embeddings": 512, "layer_types": ["full_attention", "sliding_attention"],
}
TINY_AGENT_CFG = {
    "encoder": "modernbert-release-test", "head_layers": 2, "max_len": 256, "head_max_len": 64,
    "temperature": [1.0, 1.0, 1.0], "temperature_by_options": {},
}
GATE_SCRIPT = r'''
import json, socket, sys

class Denied(socket.socket):
    def __init__(self, *a, **k): raise OSError("network blocked by clean-install gate")
socket.socket = Denied
socket.create_connection = lambda *a, **k: (_ for _ in ()).throw(OSError("blocked"))

import laya_linux
report = {"offline_import": True, "version": laya_linux.__version__}
agent = laya_linux.load(sys.argv[1], device="cpu")
result = agent.predict(
    {"message": "I was charged twice."},
    {"q": {"type": "noul", "instructions": "Does the customer ask for money back?"}},
)
report["predict_ok"] = result["model"] == "laya-rl-agent" and "q" in result["answers"]
report["noul"] = result["answers"]["q"]["noul"]
from laya_linux.packaging import verify_model
verify_model(sys.argv[1])
report["verify_ok"] = True
print(json.dumps(report))
'''


def sh(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    print("+", " ".join(cmd), flush=True)
    result = subprocess.run(cmd, **kwargs)
    if result.returncode != 0:
        print(f"FATAL: command failed ({result.returncode}).", file=sys.stderr)
        raise SystemExit(result.returncode)
    return result


def ensure_wheelhouse() -> Path:
    """Reuse ./wheelhouse when complete; otherwise (re)download the closure."""
    WHEELHOUSE.mkdir(exist_ok=True)
    have = {p.name.lower() for p in WHEELHOUSE.glob("*.whl")}
    missing = [name for name in TOP_LEVEL if not any(n.startswith(name) for n in have)]
    if missing:
        print(f"wheelhouse missing {missing}; downloading the full closure once...")
        sh([sys.executable, "-m", "pip", "download",
            "torch", "--index-url", DEFAULT_TORCH_INDEX,
            "numpy", "safetensors", "tokenizers",
            "-d", str(WHEELHOUSE)])
    return WHEELHOUSE


def build_tiny_package(dest: Path) -> Path:
    """Build a tiny checkpoint-compatible package in the BUILD environment."""
    import torch  # build env has torch
    from safetensors.torch import save_file
    from tokenizers import Tokenizer as TB, models, pre_tokenizers

    from laya_linux.model import build_decision_model

    torch.manual_seed(7)
    model = build_decision_model(TINY_ENC, TINY_AGENT_CFG)
    model.eval()
    (dest / "tokenizer").mkdir(parents=True, exist_ok=True)
    (dest / "encoder").mkdir(parents=True, exist_ok=True)
    words = {"<|cls|>": 0, "<|sep|>": 1, "<|pad|>": 2, "<|mask|>": 3}
    for i in range(200):
        words["tok%d" % i] = 4 + i
    backend = TB(models.WordLevel(vocab=words, unk_token="<|pad|>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.save(str(dest / "tokenizer" / "tokenizer.json"))
    (dest / "tokenizer" / "tokenizer_config.json").write_text(json.dumps({
        "cls_token": "<|cls|>", "sep_token": "<|sep|>",
        "pad_token": "<|pad|>", "mask_token": "<|mask|>",
    }))
    (dest / "encoder" / "config.json").write_text(json.dumps(TINY_ENC))
    (dest / "rl_agent_config.json").write_text(json.dumps(TINY_AGENT_CFG))
    save_file(dict(model.state_dict()), str(dest / "model.safetensors"))
    return dest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep", action="store_true", help="keep the throwaway venv for debugging")
    args = parser.parse_args()

    workdir = Path(tempfile.mkdtemp(prefix="laya-release-"))
    print(f"workspace: {workdir}")

    # 1. Build the wheel fresh (never trust a stale dist/)
    dist = REPO / "dist"
    if dist.exists():
        shutil.rmtree(dist)
    sh([sys.executable, "-m", "pip", "wheel", "--no-deps", "-w", str(dist), str(REPO)])
    wheels = list(dist.glob("laya_linux-*.whl"))
    if not wheels:
        print("FATAL: wheel was not produced", file=sys.stderr)
        return 1
    wheel = wheels[0]
    print(f"wheel: {wheel.name}")

    # 2. Wheelhouse
    wheelhouse = ensure_wheelhouse()

    # 3. Tiny model package (built with the build env's torch)
    tiny = build_tiny_package(workdir / "tiny-laya")

    # 4. Fresh venv, strictly offline install
    venv_dir = workdir / "fresh-venv"
    sh([sys.executable, "-m", "venv", str(venv_dir)])
    venv_python = str(venv_dir / "Scripts" / "python.exe" if sys.platform == "win32" else venv_dir / "bin" / "python")
    sh([venv_python, "-m", "pip", "install", "--quiet", "--no-index",
        "--find-links", str(wheelhouse), "--find-links", str(dist), str(wheel)])

    # 5. Socket-blocked functional gate inside the fresh environment
    gate = workdir / "gate.py"
    gate.write_text(GATE_SCRIPT)
    result = subprocess.run([venv_python, str(gate), str(tiny)], capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        print(result.stdout[-2000:])
        print(result.stderr[-2000:], file=sys.stderr)
        print("FATAL: the fresh environment failed the offline functional gate.", file=sys.stderr)
        if not args.keep:
            shutil.rmtree(workdir, ignore_errors=True)
        return 1
    report = json.loads(result.stdout.strip().splitlines()[-1])
    print("gate report:", json.dumps(report))

    if args.keep:
        print(f"kept workspace: {workdir}")
    else:
        shutil.rmtree(workdir, ignore_errors=True)

    ok = report.get("offline_import") and report.get("predict_ok") and report.get("verify_ok")
    print("CLEAN-INSTALL GATE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
