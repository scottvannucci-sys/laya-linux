"""The laya-linux private inference server (architecture §5.10, requirements §8).

Implemented directly on asyncio — no web framework is required, keeping the
server extra dependency-free (§15). The request pipeline is:

    connection → content-type/body-size limit → authentication → JSON/schema
    validation → semantic question limits → bounded queue → resident agent →
    prediction → response-size check → redacted access log

Each connection serves exactly one request (``Connection: close``), which
keeps the hand-rolled HTTP/1.1 parsing minimal and auditable. The default
bind is loopback TCP; Unix-domain sockets are preferred when client and
server share a host; non-loopback TCP requires an explicit token file.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from .. import __version__
from ..errors import (
    LayaError,
    MalformedRequestError,
    UnsupportedMediaTypeError,
)
from ..protocol import PROTOCOL_VERSION
from .auth import BearerAuth, NoAuth
from .config import ServerConfig
from .queue import AgentRegistry, BoundedExecutor
from .schemas import validate_predict_payload

log = logging.getLogger("laya_linux.server")

_REASONS = {
    200: "OK",
    400: "Bad Request",
    401: "Unauthorized",
    404: "Not Found",
    405: "Method Not Allowed",
    413: "Content Too Large",
    415: "Unsupported Media Type",
    500: "Internal Server Error",
    503: "Service Unavailable",
}

_ERROR_STATUS = {
    "UNAUTHORIZED": 401,
    "MALFORMED_REQUEST": 400,
    "REQUEST_TOO_LARGE": 413,
    "UNSUPPORTED_MEDIA_TYPE": 415,
    "PROTOCOL_UNSUPPORTED": 400,
    "INVALID_QUESTION": 422,
    "TOKEN_BUDGET_EXCEEDED": 422,
    "MODEL_NOT_FOUND": 404,
    "MODEL_INCOMPLETE": 422,
    "MODEL_CHECKSUM_FAILED": 422,
    "MODEL_INCOMPATIBLE": 422,
    "QUEUE_FULL": 503,
    "SERVER_BUSY": 503,
    "NON_FINITE_OUTPUT": 500,
}

MAX_RESPONSE_BYTES = 8 * 1024 * 1024


class LayaServerApp:
    """Async request handler: (method, path, headers, body) → (status, headers, body)."""

    def __init__(self, config: ServerConfig):
        config.validate()
        self.config = config
        self.auth = BearerAuth(config.token_file) if config.token_file else NoAuth()
        self.registry = AgentRegistry(config.models, device=config.device, dtype=config.dtype)
        self.executor = BoundedExecutor(config.queue_capacity, concurrency=config.concurrency)
        self.default_alias = next(iter(sorted(config.models)))

    async def start(self) -> None:
        self.executor.start()

    async def stop(self) -> None:
        await self.executor.stop()

    # ------------------------------------------------------------- routing
    async def handle(
        self, method: str, path: str, headers: dict[str, str], body: bytes
    ) -> tuple[int, dict[str, str], bytes]:
        try:
            status, extra, payload = await self._route(method, path, headers, body)
        except LayaError as exc:
            status = _ERROR_STATUS.get(exc.code, 500)
            payload: dict[str, Any] = {"error_code": exc.code, "error": str(exc)}
            extra = {}
        except Exception:  # noqa: BLE001 - never leak internals to the client
            log.exception("unhandled server error")
            status, extra = 500, {}
            payload = {"error_code": "SERVER_ERROR", "error": "internal server error"}
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        if len(encoded) > MAX_RESPONSE_BYTES:  # defense in depth; answers are small
            encoded = json.dumps({"error_code": "SERVER_ERROR", "error": "response too large"}).encode()
            status = 500
        return status, {"Content-Type": "application/json", **extra}, encoded

    async def _route(self, method: str, path: str, headers: dict[str, str], body: bytes):
        if path not in ("/v1/predict", "/v1/health", "/v1/models", "/v1/system"):
            raise MalformedRequestError(f"Unknown endpoint {path!r}")
        if path == "/v1/predict":
            if method != "POST":
                return self._method_not_allowed("POST")
        elif method != "GET":
            return self._method_not_allowed("GET")

        self.auth.check(headers.get("authorization"))
        _check_request_size(headers, body, self.config.max_request_bytes)

        if path == "/v1/health":
            return 200, {}, {"status": "ok", "protocol_version": PROTOCOL_VERSION, "version": __version__}
        if path == "/v1/models":
            return 200, {}, {
                "protocol_version": PROTOCOL_VERSION,
                "default": self.default_alias,
                "models": self.registry.describe_loaded(),
            }
        if path == "/v1/system":
            return 200, {}, {
                "protocol_version": PROTOCOL_VERSION,
                "version": __version__,
                "config": self.config.describe(),
                "queue_depth": self.executor.depth,
            }
        return await self._predict(headers, body)

    # ------------------------------------------------------------ handlers
    async def _predict(self, headers: dict[str, str], body: bytes):
        if not body:
            raise MalformedRequestError("Request body is required")
        content_type = (headers.get("content-type") or "application/json").split(";")[0].strip().lower()
        if content_type != "application/json":
            raise UnsupportedMediaTypeError(
                f"Content-Type {content_type!r} is not supported; use application/json"
            )
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise MalformedRequestError("Request body is not valid JSON") from None
        request = validate_predict_payload(
            payload, max_questions=self.config.max_questions, max_options=self.config.max_options
        )
        alias = request["model"] or self.default_alias
        if alias not in self.registry:
            raise MalformedRequestError(f"Unknown model alias {alias!r}; see /v1/models")
        # Model load (first use) and prediction both run through the bounded
        # executor in worker threads: the event loop never blocks, there is
        # always exactly one model copy per alias, and overload is bounded.
        agent = await self.executor.submit(self.registry.get, alias, timeout=self.config.execution_timeout)
        result = await self.executor.submit(
            agent.predict, request["state"], request["questions"], timeout=self.config.execution_timeout
        )
        log.info("predict ok questions=%d model=%s", len(request["questions"]), alias)
        return 200, {}, result

    @staticmethod
    def _method_not_allowed(allowed: str) -> tuple[int, dict[str, str], dict[str, Any]]:
        return 405, {"Allow": allowed}, {"error_code": "METHOD_NOT_ALLOWED", "error": f"use {allowed}"}


def _check_request_size(headers: dict[str, str], body: bytes, max_bytes: int) -> None:
    from ..errors import RequestTooLargeError

    declared = headers.get("content-length")
    if declared is not None:
        try:
            if int(declared) > max_bytes:
                raise RequestTooLargeError(f"Request body exceeds {max_bytes} bytes")
        except ValueError:
            raise MalformedRequestError("Invalid Content-Length header") from None
    if len(body) > max_bytes:
        raise RequestTooLargeError(f"Request body exceeds {max_bytes} bytes")


# ------------------------------------------------------------------ transport
class _OneRequestProtocol(asyncio.Protocol):
    """Minimal HTTP/1.1 server protocol: one request per connection."""

    def __init__(self, app: LayaServerApp, access_log: bool = True):
        self.app = app
        self.access_log = access_log
        self._buffer = bytearray()
        self._headers_done = False
        self._content_length = 0
        self._method = ""
        self._path = ""
        self._headers: dict[str, str] = {}
        self._peer = "unknown"

    def connection_made(self, transport: asyncio.BaseTransport) -> None:
        self.transport = transport  # type: ignore[assignment]
        peername = transport.get_extra_info("peername")
        # Unix sockets have no peer address; never log anything richer than needed.
        self._peer = "unix" if peername is None else (peername[0] if isinstance(peername, tuple) else "local")

    def data_received(self, data: bytes) -> None:
        self._buffer.extend(data)
        if not self._headers_done:
            head, sep, rest = bytes(self._buffer).partition(b"\r\n\r\n")
            if not sep:
                if len(self._buffer) > 65536:
                    self._abort(400, {"error_code": "MALFORMED_REQUEST", "error": "headers too large"})
                return
            self._headers_done = True
            self._parse_head(head.decode("latin-1"))
            self._buffer = bytearray(rest)
        expected = self._content_length
        if expected and len(self._buffer) < expected:
            if len(self._buffer) > 64 * 1024 * 1024:
                self._abort(413, {"error_code": "REQUEST_TOO_LARGE", "error": "body too large"})
            return
        body = bytes(self._buffer[:expected]) if expected else b""
        self._buffer.clear()
        asyncio.get_running_loop().create_task(self._respond(body))

    def _parse_head(self, head: str) -> None:
        lines = head.split("\r\n")
        try:
            self._method, self._path, _ = lines[0].split(" ", 2)
        except ValueError:
            self._abort(400, {"error_code": "MALFORMED_REQUEST", "error": "malformed request line"})
            return
        for line in lines[1:]:
            name, _, value = line.partition(":")
            self._headers[name.strip().lower()] = value.strip()
        try:
            self._content_length = max(0, int(self._headers.get("content-length", "0")))
        except ValueError:
            self._abort(400, {"error_code": "MALFORMED_REQUEST", "error": "invalid Content-Length"})

    def _abort(self, status: int, payload: dict[str, Any]) -> None:
        self._write_response(status, {}, json.dumps(payload).encode(), close=True)

    async def _respond(self, body: bytes) -> None:
        status, headers, payload = await self.app.handle(self._method, self._path, self._headers, body)
        if self.access_log:
            # Redacted access log: method, path, status, size — never the body.
            log.info("%s %s -> %d (%d bytes) from %s", self._method, self._path, status, len(payload), self._peer)
        self._write_response(status, headers, payload, close=True)

    def _write_response(self, status: int, headers: dict[str, str], payload: bytes, *, close: bool) -> None:
        reason = _REASONS.get(status, "OK")
        lines = [f"HTTP/1.1 {status} {reason}"]
        for name, value in headers.items():
            lines.append(f"{name}: {value}")
        lines.append(f"Content-Length: {len(payload)}")
        lines.append("Connection: close")
        head = ("\r\n".join(lines) + "\r\n\r\n").encode("latin-1")
        try:
            self.transport.write(head + payload)
        except Exception:  # noqa: BLE001 - client hung up first
            pass
        self.transport.close()


async def serve(config: ServerConfig, *, access_log: bool = True) -> None:
    """Run the server until cancelled. Bind target comes from the config."""
    app = LayaServerApp(config)
    await app.start()
    loop = asyncio.get_running_loop()
    if config.unix_socket:
        server = await loop.create_unix_server(lambda: _OneRequestProtocol(app, access_log), config.unix_socket)
        log.info("laya-linux server on unix:%s models=%s", config.unix_socket, sorted(config.models))
    else:
        server = await loop.create_server(
            lambda: _OneRequestProtocol(app, access_log), config.host, config.port
        )
        log.info("laya-linux server on %s:%d models=%s", config.host, config.port, sorted(config.models))
    try:
        async with server:
            await server.serve_forever()
    finally:
        await app.stop()


def run_server(config: ServerConfig, *, access_log: bool = True) -> None:
    """Blocking entry point used by ``laya-linux serve``."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    try:
        asyncio.run(serve(config, access_log=access_log))
    except KeyboardInterrupt:
        log.info("server stopped")
