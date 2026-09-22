"""Server configuration (architecture §8): secure defaults, explicit exposure."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..errors import LayaError

DEFAULT_PORT = 8142
DEFAULT_MAX_REQUEST_BYTES = 1 * 1024 * 1024
DEFAULT_MAX_QUESTIONS = 64
DEFAULT_MAX_OPTIONS = 256
DEFAULT_QUEUE_CAPACITY = 64
DEFAULT_CONCURRENCY = 1
DEFAULT_EXECUTION_TIMEOUT = 60.0


class ServerConfigError(LayaError):
    code = "SERVER_CONFIG_INVALID"


def _is_loopback(host: str) -> bool:
    return host in ("127.0.0.1", "::1", "localhost")


@dataclass
class ServerConfig:
    """All server settings; loopback-only and auth-off are the defaults."""

    models: dict[str, str] = field(default_factory=dict)
    host: str = "127.0.0.1"
    port: int = DEFAULT_PORT
    unix_socket: str | None = None
    token_file: str | None = None
    device: str | None = None
    dtype: str = "auto"
    max_request_bytes: int = DEFAULT_MAX_REQUEST_BYTES
    max_questions: int = DEFAULT_MAX_QUESTIONS
    max_options: int = DEFAULT_MAX_OPTIONS
    queue_capacity: int = DEFAULT_QUEUE_CAPACITY
    concurrency: int = DEFAULT_CONCURRENCY
    execution_timeout: float = DEFAULT_EXECUTION_TIMEOUT
    tls_certfile: str | None = None
    tls_keyfile: str | None = None
    preload: bool = False
    access_log: bool = True

    def validate(self) -> None:
        if not self.models:
            raise ServerConfigError(
                "No models configured; pass at least one alias=path pair "
                "(e.g. --model typed=/opt/laya/models/laya-typed-decisions)"
            )
        for alias, path in self.models.items():
            if not alias or "/" in alias or alias.startswith("."):
                raise ServerConfigError(f"Invalid model alias {alias!r}")
            if not Path(path).is_dir():
                raise ServerConfigError(
                    f"Model path for alias {alias!r} does not exist: {path}"
                )
        if self.unix_socket and self.host not in ("127.0.0.1",):
            raise ServerConfigError("Choose either a Unix socket or a TCP host, not both")
        if not self.unix_socket and _is_loopback(self.host) is False:
            # Non-loopback binding is an explicit act that must be paired with auth.
            if not self.token_file:
                raise ServerConfigError(
                    f"Binding to non-loopback address {self.host!r} requires authentication: "
                    f"pass --token-file. Use the default 127.0.0.1 for loopback-only service."
                )
        if self.token_file and not Path(self.token_file).is_file():
            raise ServerConfigError(f"Token file not found: {self.token_file}")
        if self.tls_certfile and not Path(self.tls_certfile).is_file():
            raise ServerConfigError(f"TLS certificate not found: {self.tls_certfile}")
        if self.tls_keyfile and not Path(self.tls_keyfile).is_file():
            raise ServerConfigError(f"TLS key not found: {self.tls_keyfile}")
        if bool(self.tls_certfile) != bool(self.tls_keyfile):
            raise ServerConfigError("TLS requires both --tls-certfile and --tls-keyfile")
        if self.max_questions < 1 or self.max_options < 1 or self.queue_capacity < 1:
            raise ServerConfigError("Limits must be positive integers")
        if self.execution_timeout <= 0:
            raise ServerConfigError("execution_timeout must be positive")

    def describe(self) -> dict:
        """Redacted description for /v1/system — no paths with usernames, no secrets."""
        return {
            "bind": self.unix_socket or f"{self.host}:{self.port}",
            "auth": "bearer-token" if self.token_file else "disabled (loopback only)",
            "tls": bool(self.tls_certfile),
            "limits": {
                "max_request_bytes": self.max_request_bytes,
                "max_questions": self.max_questions,
                "max_options": self.max_options,
                "queue_capacity": self.queue_capacity,
                "concurrency": self.concurrency,
                "execution_timeout_seconds": self.execution_timeout,
            },
        }
