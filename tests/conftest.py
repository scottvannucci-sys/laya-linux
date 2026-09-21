"""Shared fixtures: deterministic tiny model packages in the upstream layout."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

ENC_DICT = {
    "model_type": "modernbert",
    "vocab_size": 204,
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "local_attention": 8,
    "global_attn_every_n_layers": 3,
    "max_position_embeddings": 512,
    "layer_types": ["full_attention", "sliding_attention"],
    "rope_parameters": {
        "full_attention": {"rope_theta": 160000.0, "rope_type": "default"},
        "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"},
    },
}

AGENT_CFG = {
    "encoder": "modernbert-test",
    "head_layers": 2,
    "max_len": 256,
    "head_max_len": 64,
    "temperature": [1.05, 1.1, 0.95],
    "temperature_by_options": {"choice:2": 1.3},
    "act_costs": {"escalate": 0.5},
}

QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "Pick the department.",
        "criteria": {"billing": "money", "technical": "bugs"},
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent?",
        "criteria": ["calm", "concerned", "angry"],
    },
    "refund": {"type": "noul", "instructions": "Does the customer ask for a refund?"},
}

STATE = {"message": "I was charged twice."}


def build_tiny_package(dest: Path, seed: int = 42) -> Path:
    """Write a tiny checkpoint package with fixed random weights."""
    import torch
    from safetensors.torch import save_file
    from tokenizers import Tokenizer as TB
    from tokenizers import models, pre_tokenizers

    from laya_linux.model import build_decision_model

    torch.manual_seed(seed)
    model = build_decision_model(ENC_DICT, AGENT_CFG)
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
    (dest / "encoder" / "config.json").write_text(json.dumps(ENC_DICT))
    (dest / "rl_agent_config.json").write_text(json.dumps(AGENT_CFG))
    save_file(dict(model.state_dict()), str(dest / "model.safetensors"))
    return dest


@pytest.fixture(scope="session")
def tiny_pkg(tmp_path_factory) -> Path:
    return build_tiny_package(tmp_path_factory.mktemp("pkg") / "tiny-laya")


@pytest.fixture(scope="session")
def agent(tiny_pkg):
    from laya_linux import Agent

    return Agent(str(tiny_pkg), device="cpu")


@pytest.fixture(scope="session")
def small_tok(tiny_pkg):
    from laya_linux.tokenizer import Tokenizer

    return Tokenizer(tiny_pkg / "tokenizer")


class SocketBlocker:
    """Deny socket creation; record any attempt (offline guarantee tests)."""

    def __init__(self):
        self.attempts = []

    def __enter__(self):
        import socket

        self._socket = socket.socket
        real = self._socket
        attempts = self.attempts

        class Denied(real):  # noqa: N801 - intentionally raises
            def __init__(self, *args, **kwargs):
                attempts.append(args)
                raise OSError("network access blocked by test")

        socket.socket = Denied
        self._create_connection = socket.create_connection

        def _deny(*args, **kwargs):
            attempts.append(args)
            raise OSError("network access blocked by test")

        socket.create_connection = _deny
        return self

    def __exit__(self, *exc):
        import socket

        socket.socket = self._socket
        socket.create_connection = self._create_connection
        return False


REPO_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_COMMON = Path(
    __import__("os").environ.get(
        "LAYA_UPSTREAM_COMMON", str(REPO_ROOT.parent / "laya-linux-refs" / "laya" / "laya" / "common.py")
    )
)
REAL_CHECKPOINT = Path(
    __import__("os").environ.get("LAYA_REAL_CHECKPOINT", str(REPO_ROOT / "models" / "laya-typed-decisions"))
)
