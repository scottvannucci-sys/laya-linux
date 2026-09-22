"""End-to-end server tests: real loopback server, tiny model, real client."""

import json
import socket

import pytest

from conftest import QUESTIONS, STATE

torch = pytest.importorskip("torch")

from laya_linux.client.http import RemoteAgent  # noqa: E402
from laya_linux.errors import (  # noqa: E402
    MalformedRequestError,
    UnauthorizedError,
)
from laya_linux.server.app import run_server  # noqa: E402
from laya_linux.server.config import ServerConfig  # noqa: E402


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture()
def server(tiny_pkg):
    """Run a real server on a random loopback port in a background thread."""
    import threading
    import time

    config = ServerConfig(models={"tiny": str(tiny_pkg)}, host="127.0.0.1", port=_free_port())
    thread = threading.Thread(target=run_server, args=(config,), daemon=True)
    thread.start()
    time.sleep(0.5)  # give the loop time to bind
    yield f"http://127.0.0.1:{config.port}"
    # daemon thread dies with the process


def test_health_and_models(server):
    client = RemoteAgent(server)
    health = client.health()
    assert health["status"] == "ok" and health["protocol_version"] == 1
    models = client.models()
    assert models["default"] == "tiny"
    assert models["models"] == [{"alias": "tiny", "loaded": False}]


def test_system_redacts_paths(server):
    client = RemoteAgent(server)
    system = client.system()
    assert str(server).replace("http://", "") in system["config"]["bind"]
    assert "tiny_pkg" not in json.dumps(system)  # no user paths in system output


def test_predict_round_trip_matches_embedded(server, agent):
    client = RemoteAgent(server)
    remote = client.predict(STATE, QUESTIONS)
    embedded = agent.predict(STATE, QUESTIONS)
    assert remote == embedded  # same model, same inputs, identical result
    assert remote["model"] == "laya-rl-agent"


def test_protocol_version_mismatch(server):
    client = RemoteAgent(server)
    from laya_linux.protocol import PROTOCOL_VERSION

    with pytest.raises(Exception, match="protocol"):
        client._request("POST", "/v1/predict", {
            "protocol_version": PROTOCOL_VERSION + 99,
            "state": STATE,
            "questions": QUESTIONS,
        })


def test_unknown_alias_rejected(server):
    client = RemoteAgent(server)
    with pytest.raises(MalformedRequestError, match="alias"):
        client.predict(STATE, QUESTIONS, model="nonexistent")


def test_unknown_endpoint_rejected(server):
    client = RemoteAgent(server)
    with pytest.raises(MalformedRequestError, match="endpoint"):
        client._request("GET", "/v1/nope")


def test_wrong_method_rejected(server):
    client = RemoteAgent(server)
    with pytest.raises(MalformedRequestError):
        client._request("GET", "/v1/predict")


def test_malformed_json_rejected(server):
    client = RemoteAgent(server)
    with pytest.raises(MalformedRequestError):
        client._request("POST", "/v1/predict", {"state": STATE})  # no questions -> 400


def test_auth_success_and_failure(tiny_pkg):
    import tempfile
    from pathlib import Path

    token = Path(tempfile.mkdtemp()) / "token"
    token.write_text("s3cret")
    config = ServerConfig(models={"tiny": str(tiny_pkg)}, host="127.0.0.1", port=_free_port(),
                          token_file=str(token))
    import threading
    import time

    threading.Thread(target=run_server, args=(config,), daemon=True).start()
    time.sleep(0.5)
    base = f"http://127.0.0.1:{config.port}"
    with pytest.raises(UnauthorizedError):
        RemoteAgent(base).predict(STATE, QUESTIONS)  # no token
    ok = RemoteAgent(base, token_file=str(token))
    result = ok.predict(STATE, QUESTIONS)
    assert result["model"] == "laya-rl-agent"
    with pytest.raises(UnauthorizedError):
        # wrong token file
        bad = token.parent / "bad"
        bad.write_text("nope")
        RemoteAgent(base, token_file=str(bad)).predict(STATE, QUESTIONS)


def test_non_loopback_bind_requires_token(tmp_path):
    with pytest.raises(Exception, match="token"):
        ServerConfig(models={"tiny": str(tmp_path)}, host="192.168.1.9").validate()


def test_public_address_guardrail():
    with pytest.raises(MalformedRequestError, match="public"):
        RemoteAgent("http://example.com:8142")


def test_redirect_never_followed(server, monkeypatch):
    client = RemoteAgent(server)

    class FakeResponse:
        status = 302
        def getheader(self, name):
            return None
        def read(self, n=-1):
            return b""

    class FakeConnection:
        def request(self, *a, **k): ...
        def getresponse(self):
            return FakeResponse()
        def close(self): ...

    monkeypatch.setattr(client, "_connection", lambda: FakeConnection())
    with pytest.raises(MalformedRequestError, match="redirect"):
        client._request("GET", "/v1/health")


def test_server_unavailable_when_nothing_listens():
    port = _free_port()
    with pytest.raises(Exception, match="reach"):
        RemoteAgent(f"http://127.0.0.1:{port}").predict(STATE, QUESTIONS)


def test_queue_full_returns_stable_code(tiny_pkg):
    """A capacity-0-ish setup is not constructible; verify 503 mapping directly."""
    from laya_linux.server.app import _ERROR_STATUS

    assert _ERROR_STATUS["QUEUE_FULL"] == 503
    assert _ERROR_STATUS["SERVER_BUSY"] == 503
    assert _ERROR_STATUS["UNAUTHORIZED"] == 401
