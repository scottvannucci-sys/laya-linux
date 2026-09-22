"""Server config guardrails and bearer auth (unit)."""

import pytest

from laya_linux.errors import LayaError, UnauthorizedError
from laya_linux.server.auth import BearerAuth, NoAuth
from laya_linux.server.config import ServerConfig


def _cfg(**overrides):
    base = dict(models={"typed": r"C:\models\x"})
    base.update(overrides)
    return ServerConfig(**base)


def test_loopback_default_ok(tmp_path, monkeypatch):
    monkeypatch.setattr("laya_linux.server.config.Path", __import__("pathlib").Path)
    # models path existence check: point at a real dir
    cfg = ServerConfig(models={"typed": str(tmp_path)})
    cfg.validate()  # loopback, no auth -> valid


def test_non_loopback_requires_auth(tmp_path):
    cfg = ServerConfig(models={"typed": str(tmp_path)}, host="192.168.1.50")
    with pytest.raises(LayaError, match="token"):
        cfg.validate()
    token = tmp_path / "token"
    token.write_text("s3cret")
    ServerConfig(models={"typed": str(tmp_path)}, host="192.168.1.50", token_file=str(token)).validate()


def test_invalid_model_path_rejected(tmp_path):
    with pytest.raises(LayaError, match="does not exist"):
        ServerConfig(models={"typed": str(tmp_path / "missing")}).validate()


def test_bad_alias_rejected(tmp_path):
    with pytest.raises(LayaError):
        ServerConfig(models={"a/b": str(tmp_path)}).validate()


def test_unix_socket_conflicts_with_host(tmp_path):
    with pytest.raises(LayaError):
        ServerConfig(models={"typed": str(tmp_path)}, unix_socket="/tmp/x.sock", host="0.0.0.0").validate()


def test_tls_needs_both_files(tmp_path):
    cert = tmp_path / "c.pem"
    cert.write_text("x")
    with pytest.raises(LayaError):
        ServerConfig(models={"typed": str(tmp_path)}, tls_certfile=str(cert)).validate()


def test_describe_is_redacted(tmp_path):
    cfg = ServerConfig(models={"typed": str(tmp_path)}, host="127.0.0.1")
    d = cfg.describe()
    assert str(tmp_path) not in str(d)
    assert "127.0.0.1" in d["bind"]


def test_bearer_auth_constant_time_paths(tmp_path):
    token = tmp_path / "token"
    token.write_text("s3cret\n")
    auth = BearerAuth(str(token))
    auth.check("Bearer s3cret")
    with pytest.raises(UnauthorizedError):
        auth.check("Bearer wrong")
    with pytest.raises(UnauthorizedError):
        auth.check(None)
    with pytest.raises(UnauthorizedError):
        auth.check("s3cret")  # missing scheme


def test_bearer_auth_dotenv_style(tmp_path):
    token = tmp_path / "token"
    token.write_text("LAYA_TOKEN=abc123")
    auth = BearerAuth(str(token))
    auth.check("Bearer abc123")


def test_no_auth_allows_everything():
    NoAuth().check(None)
    NoAuth().check("Bearer anything")
