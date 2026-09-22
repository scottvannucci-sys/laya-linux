"""The constrained-client guarantee: the remote client works without PyTorch.

Runs in a subprocess with an import hook that makes ``import torch`` fail, so
the test reflects a machine where PyTorch is simply not installed.
"""

import json
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from conftest import REPO_ROOT

CLIENT_SCRIPT = r'''
import importlib.abc
import json
import sys

class TorchBlocker(importlib.abc.MetaPathFinder):
    """Make `import torch` raise, simulating a machine without PyTorch."""
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise ImportError("PyTorch is not installed on this client (blocked by test)")
        return None

sys.meta_path.insert(0, TorchBlocker())

for mod in list(sys.modules):
    if mod == "torch" or mod.startswith("torch."):
        del sys.modules[mod]

from laya_linux.client.http import RemoteAgent  # must not import torch

assert "torch" not in sys.modules, "client imported torch!"
client = RemoteAgent(sys.argv[1], timeout=60)
health = client.health()
result = client.predict({"message": "I was charged twice."}, {
    "q": {"type": "noul", "instructions": "Does the customer ask for money back?"}})
print(json.dumps({"torch_free": True, "health": health["status"], "noul": result["answers"]["q"]["noul"]}))
'''


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def running_server(tiny_pkg):
    from laya_linux.server.app import run_server
    from laya_linux.server.config import ServerConfig

    config = ServerConfig(models={"tiny": str(tiny_pkg)}, host="127.0.0.1", port=_free_port())
    thread = threading.Thread(target=run_server, args=(config,), daemon=True)
    thread.start()
    time.sleep(0.5)
    yield f"http://127.0.0.1:{config.port}"


def test_client_works_without_torch(running_server, tmp_path):
    script = tmp_path / "torch_free_client.py"
    script.write_text(CLIENT_SCRIPT)
    venv_python = Path(sys.executable)
    result = subprocess.run(
        [str(venv_python), str(script), running_server],
        capture_output=True, text=True, timeout=300, cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, result.stderr[-800:]
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["torch_free"] is True
    assert payload["health"] == "ok"
    assert 0.0 <= payload["noul"] <= 1.0
