"""PyTorch-free remote client implementing the DecisionAgent protocol
(architecture §5.9, requirements §8.4).

Uses only the standard library — a constrained client can call the server
without installing PyTorch or storing model weights. Guardrails:

- The configured host must be loopback or private; public endpoints are
  rejected unless explicitly allowed. This is a guardrail, not a firewall.
- Responses are never followed across hosts (no redirect handling at all).
- Non-idempotent predict requests are never retried automatically.
- Stable server error codes surface as typed exceptions.
"""

from __future__ import annotations

import http.client
import ipaddress
import json
import socket
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from ..errors import (
    LayaError,
    MalformedRequestError,
    ProtocolUnsupportedError,
    ResponseTooLargeError,
    ServerUnavailableError,
)
from ..protocol import PROTOCOL_VERSION, DecisionAgent, PredictionResult, Questions, State

DEFAULT_TIMEOUT = 30.0
DEFAULT_MAX_RESPONSE_BYTES = 8 * 1024 * 1024

_ERROR_CLASSES = {}


def _error_for(code: str, message: str) -> LayaError:
    from .. import errors as E

    mapping = {
        "UNAUTHORIZED": E.UnauthorizedError,
        "REQUEST_TOO_LARGE": E.RequestTooLargeError,
        "QUEUE_FULL": E.QueueFullError,
        "SERVER_BUSY": E.ServerBusyError,
        "PROTOCOL_UNSUPPORTED": E.ProtocolUnsupportedError,
        "MALFORMED_REQUEST": E.MalformedRequestError,
        "METHOD_NOT_ALLOWED": E.MalformedRequestError,
        "UNSUPPORTED_MEDIA_TYPE": E.UnsupportedMediaTypeError,
        "INVALID_QUESTION": E.MalformedRequestError,
        "TOKEN_BUDGET_EXCEEDED": E.RequestTooLargeError,
        "MODEL_NOT_FOUND": E.ModelNotFoundError if hasattr(E, "ModelNotFoundError") else E.LayaError,
        "MODEL_INCOMPATIBLE": E.ModelIncompatibleError if hasattr(E, "ModelIncompatibleError") else E.LayaError,
        "NON_FINITE_OUTPUT": E.NonFiniteOutputError if hasattr(E, "NonFiniteOutputError") else E.LayaError,
    }
    cls = mapping.get(code, E.LayaError)
    exc = cls(message)
    exc.code = code
    return exc


class _UnixHTTPConnection(http.client.HTTPConnection):
    def __init__(self, socket_path: str, timeout: float):
        super().__init__("localhost", timeout=timeout)
        self._socket_path = socket_path

    def connect(self) -> None:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        sock.connect(self._socket_path)
        self.sock = sock


def _is_loopback_or_private(host: str) -> bool:
    if host in ("localhost",):
        return True
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        return False  # hostnames: resolve first
    return addr.is_loopback or addr.is_private


class RemoteAgent(DecisionAgent):
    """Client-side agent speaking the laya-linux HTTP protocol.

    ``address`` is either a TCP base URL (``http://127.0.0.1:8142``) or a
    Unix-socket path (``/run/laya/laya.sock``). Unix sockets are preferred
    when both processes share a host.
    """

    def __init__(
        self,
        address: str,
        *,
        token_file: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
        allow_public: bool = False,
        model: str | None = None,
    ):
        if not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError("timeout must be a positive number")
        self.timeout = float(timeout)
        self.max_response_bytes = max_response_bytes
        self.model = model
        self._token: bytes | None
        if token_file:
            token = Path(token_file).read_text(encoding="utf-8").strip()
            if not token:
                raise ValueError("Token file is empty")
            self._token = token.encode("utf-8")
        else:
            self._token = None

        expanded = str(address)
        if expanded.startswith(("http://", "https://")):
            parsed = urlparse(expanded)
            host = parsed.hostname or ""
            port = parsed.port or 8142
            if not _is_loopback_or_private(host):
                resolved = {ai[4][0] for ai in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)}
                if not all(_is_loopback_or_private(ip) for ip in resolved) and not allow_public:
                    raise MalformedRequestError(
                        f"Address {host!r} resolves outside loopback/private networks; the client "
                        f"refuses public endpoints (allow_public=True overrides this guardrail)"
                    )
            self._socket_path: str | None = None
            self._host, self._port = host, port
        else:
            path = Path(expanded).expanduser()
            if not path.exists():
                raise ServerUnavailableError(f"Unix socket not found: {path}")
            self._socket_path = str(path)
            self._host, self._port = "localhost", None

    # --------------------------------------------------------- transport
    def _connection(self) -> http.client.HTTPConnection:
        if self._socket_path:
            return _UnixHTTPConnection(self._socket_path, self.timeout)
        assert self._port is not None
        return http.client.HTTPConnection(self._host, self._port, timeout=self.timeout)

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self._token:
            headers["Authorization"] = "Bearer " + self._token.decode("utf-8")
        return headers

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
        connection = self._connection()
        try:
            connection.request(method, path, body=body, headers=self._headers())
            response = connection.getresponse()
            if 300 <= response.status < 400:
                # Never follow redirects: the guardrail is the client's own.
                raise MalformedRequestError(
                    f"Server attempted a redirect ({response.status}); refusing to follow"
                )
            declared = response.getheader("Content-Length")
            if declared and int(declared) > self.max_response_bytes:
                raise ResponseTooLargeError(
                    f"Response of {declared} bytes exceeds the {self.max_response_bytes}-byte limit"
                )
            chunks = []
            received = 0
            while True:
                chunk = response.read(65536)
                if not chunk:
                    break
                received += len(chunk)
                if received > self.max_response_bytes:
                    raise ResponseTooLargeError(
                        f"Response exceeds the {self.max_response_bytes}-byte limit"
                    )
                chunks.append(chunk)
            raw = b"".join(chunks)
            if response.status != 200:
                try:
                    err = json.loads(raw)
                except json.JSONDecodeError:
                    raise MalformedRequestError(
                        f"Server returned HTTP {response.status} without a structured error"
                    ) from None
                raise _error_for(str(err.get("error_code", "SERVER_ERROR")), str(err.get("error", "")))
            try:
                return json.loads(raw)
            except json.JSONDecodeError as exc:
                raise MalformedRequestError("Server response is not valid JSON") from exc
        except (TimeoutError, ConnectionRefusedError, FileNotFoundError, socket.gaierror, OSError, http.client.HTTPException) as exc:
            if isinstance(exc, LayaError):
                raise
            raise ServerUnavailableError(f"Could not reach the laya-linux server: {type(exc).__name__}") from exc
        finally:
            connection.close()

    # ------------------------------------------------------------ protocol
    def predict(self, state: State, questions: Questions, model: str | None = None) -> PredictionResult:
        """Evaluate typed questions on the remote server (single request)."""
        payload = {
            "protocol_version": PROTOCOL_VERSION,
            "model": model or self.model,
            "state": state,
            "questions": questions,
        }
        result = self._request("POST", "/v1/predict", payload)
        version = result.get("protocol_version")
        if version is not None and version != PROTOCOL_VERSION:
            raise ProtocolUnsupportedError(
                f"Server protocol version {version!r} not supported by this client"
            )
        return result

    system_one = predict

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/v1/health")

    def models(self) -> dict[str, Any]:
        return self._request("GET", "/v1/models")

    def system(self) -> dict[str, Any]:
        return self._request("GET", "/v1/system")


RemoteClient = RemoteAgent  # alias for readability in server-side docs
