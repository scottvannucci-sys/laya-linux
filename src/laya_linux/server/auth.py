"""Bearer-token authentication (requirements §8.3).

Tokens are read from protected files or injected files — never from command
lines that would expose them in process listings. Comparison is constant-time.
"""

from __future__ import annotations

import hmac
from pathlib import Path

from ..errors import UnauthorizedError


class BearerAuth:
    """Validates ``Authorization: Bearer <token>`` headers."""

    def __init__(self, token_file: str):
        raw = Path(token_file).read_text(encoding="utf-8")
        # Accept raw token files and trivial `NAME=value` dotenv-style lines.
        token = raw.strip()
        if "=" in token and "\n" not in token and token.split("=", 1)[0].strip().upper() in (
            "TOKEN",
            "LAYA_TOKEN",
            "BEARER",
        ):
            token = token.split("=", 1)[1].strip().strip('"').strip("'")
        if not token:
            raise ValueError("Token file is empty")
        self._token = token.encode("utf-8")

    @property
    def enabled(self) -> bool:
        return True

    def check(self, authorization_header: str | None) -> None:
        """Raise ``UnauthorizedError`` unless the header carries the right token."""
        if not authorization_header or not authorization_header.startswith("Bearer "):
            raise UnauthorizedError("Missing bearer token")
        supplied = authorization_header[len("Bearer "):].strip().encode("utf-8")
        if not hmac.compare_digest(supplied, self._token):
            raise UnauthorizedError("Invalid bearer token")


class NoAuth:
    """Auth disabled — only acceptable on loopback bindings (config validates this)."""

    @property
    def enabled(self) -> bool:
        return False

    def check(self, authorization_header: str | None) -> None:  # noqa: ARG002
        return None
