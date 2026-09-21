"""Offline guarantees: no sockets during import, load, or prediction."""


import pytest

torch = pytest.importorskip("torch")

from conftest import QUESTIONS, STATE


class SocketBlocker:
    """Deny socket creation; record any attempt."""

    def __init__(self):
        self.attempts = []

    def __enter__(self):
        import socket

        self._socket = socket.socket
        real = self._socket

        class Denied(real):  # noqa: N801 - intentionally raises
            def __init__(self, *args, **kwargs):
                self.attempts.append(args)
                raise OSError("network access blocked by test")

        Denied.attempts = self.attempts
        socket.socket = Denied
        # also block higher-level helpers
        self._create_connection = socket.create_connection
        socket.create_connection = self._deny
        return self

    def _deny(self, *args, **kwargs):
        self.attempts.append(args)
        raise OSError("network access blocked by test")

    def __exit__(self, *exc):
        import socket

        socket.socket = self._socket
        socket.create_connection = self._create_connection
        return False


def test_import_load_predict_with_sockets_blocked(tiny_pkg):
    import importlib

    import laya_linux  # already imported at collection; force a fresh module check

    with SocketBlocker() as blocker:
        importlib.reload(laya_linux)
        agent = laya_linux.load(str(tiny_pkg), device="cpu")
        result = agent.predict(STATE, QUESTIONS)
    assert blocker.attempts == []
    assert set(result["answers"]) == {"department", "urgency", "refund"}


def test_missing_model_never_downloads(tmp_path):
    from laya_linux import load
    from laya_linux.errors import ModelNotFoundError

    with SocketBlocker() as blocker:
        with pytest.raises(ModelNotFoundError):
            load(str(tmp_path / "not-there"))
    assert blocker.attempts == []


def test_remote_identifier_rejected():
    from laya_linux import load
    from laya_linux.errors import NetworkDisabledError

    with pytest.raises(NetworkDisabledError):
        load("convaiinnovations/laya-typed-decisions")
