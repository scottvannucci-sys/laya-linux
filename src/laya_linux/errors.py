"""Stable library exceptions with machine-readable codes (requirements §7, architecture §7)."""

from __future__ import annotations


class LayaError(Exception):
    """Base class for all laya-linux errors.

    Carries a stable machine-readable ``code`` and a safe human-readable
    explanation. Error messages MUST NOT include request bodies, tokens, or
    sensitive environment data.
    """

    code = "LAYA_ERROR"

    def __init__(self, message: str, *, code: str | None = None):
        super().__init__(message)
        if code is not None:
            self.code = code


class ModelNotFoundError(LayaError):
    code = "MODEL_NOT_FOUND"


class ModelIncompleteError(LayaError):
    code = "MODEL_INCOMPLETE"


class ModelChecksumError(LayaError):
    code = "MODEL_CHECKSUM_FAILED"


class ModelIncompatibleError(LayaError):
    code = "MODEL_INCOMPATIBLE"


class InvalidQuestionError(LayaError):
    code = "INVALID_QUESTION"


class TokenBudgetExceededError(LayaError):
    code = "TOKEN_BUDGET_EXCEEDED"


class DeviceUnavailableError(LayaError):
    code = "DEVICE_UNAVAILABLE"


class DtypeUnsupportedError(LayaError):
    code = "DTYPE_UNSUPPORTED"


class NonFiniteOutputError(LayaError):
    code = "NON_FINITE_OUTPUT"


class NetworkDisabledError(LayaError):
    """Raised when a remote-capable operation is attempted through the core runtime.

    The core runtime performs no network operations by construction; this error
    exists so a rejected remote identifier produces an actionable, stable code.
    """

    code = "NETWORK_DISABLED"


# --- server/client protocol error codes (requirements §8, architecture §7) ---


class UnauthorizedError(LayaError):
    code = "UNAUTHORIZED"


class RequestTooLargeError(LayaError):
    code = "REQUEST_TOO_LARGE"


class QueueFullError(LayaError):
    code = "QUEUE_FULL"


class ServerBusyError(LayaError):
    code = "SERVER_BUSY"


class ProtocolUnsupportedError(LayaError):
    code = "PROTOCOL_UNSUPPORTED"


class ServerUnavailableError(LayaError):
    """The configured server could not be reached (connection refused/unreachable)."""

    code = "SERVER_UNAVAILABLE"


class RequestTimeoutError(LayaError):
    """The server did not answer within the client's deadline."""

    code = "REQUEST_TIMEOUT"


class ResponseTooLargeError(LayaError):
    """The server's response exceeded the client's maximum response size."""

    code = "RESPONSE_TOO_LARGE"


class UnsupportedMediaTypeError(LayaError):
    code = "UNSUPPORTED_MEDIA_TYPE"


class MalformedRequestError(LayaError):
    code = "MALFORMED_REQUEST"
