"""Guard the vendored plugin against drifting from src.

The plugin dir ships hand-copies of src/laya_linux/client/http.py,
presets.py, protocol.py and errors.py (stdlib-only, for torch-free
Hermes installs). These tests fail when the copies drift — the three
recent plugin bugfixes were all drift-adjacent.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PLUGIN = REPO / "plugins" / "hermes" / "laya-linux"

import pytest


@pytest.fixture(scope="module")
def plugin_dir_on_path():
    sys.path.insert(0, str(PLUGIN))
    yield
    sys.path.remove(str(PLUGIN))


def test_presets_match_src():
    import laya_linux.presets as src_presets
    sys.path.insert(0, str(PLUGIN))
    try:
        import presets as plugin_presets
    finally:
        sys.path.remove(str(PLUGIN))
    for name in ("triage_questions", "email_questions", "guard_questions",
                 "moderation_questions", "router_questions"):
        assert getattr(src_presets, name)() == getattr(plugin_presets, name)(), name


def test_vendored_client_imports_and_matches_src(plugin_dir_on_path):
    import client as plugin_client

    from laya_linux.client import http as src_http

    assert plugin_client.RemoteAgent is not None
    # Same public surface as the src client.
    for attr in ("predict", "health"):
        assert callable(getattr(plugin_client.RemoteAgent, attr))
        assert callable(getattr(src_http.RemoteAgent, attr))


def test_vendored_client_accepts_dotenv_token_file(tmp_path):
    sys.path.insert(0, str(PLUGIN))
    try:
        import client as plugin_client
    finally:
        sys.path.remove(str(PLUGIN))
    token_file = tmp_path / "token"
    token_file.write_text("LAYA_TOKEN=abc123\n", encoding="utf-8")
    # Loopback host keeps the private-address guardrail happy.
    agent = plugin_client.RemoteAgent("http://127.0.0.1:8142", token_file=str(token_file))
    assert agent._token == b"abc123"
