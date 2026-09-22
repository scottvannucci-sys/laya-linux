"""Laya typed-decision inference on Linux, local-first and offline by construction.

Torch-free attributes (``Router``, presets, ``RemoteAgent``, language
utilities, protocol constants) import eagerly; the heavy inference stack
(``Agent``, ``load``) loads lazily on first attribute access, so a constrained
client machine without PyTorch can import and use ``RemoteAgent``
(requirements §8.4).
"""

from importlib import import_module
from typing import Any

from .email import clean_email_body, email_state  # torch-free
from .lang import analyse as detect_language  # torch-free
from .lang import detect_script, is_english  # torch-free
from .presets import (  # torch-free
    email_questions,
    guard_questions,
    moderation_questions,
    router_questions,
    triage_questions,
)
from .protocol import PROTOCOL_VERSION, DecisionAgent  # torch-free
from .router import DEFAULT_MODELS, RouteDecision, Router  # torch-free
from .version import __version__

__version__ = __version__

_LAZY: dict[str, tuple[str, str]] = {
    "Agent": ("laya_linux.agent", "Agent"),
    "RLAgent": ("laya_linux.agent", "RLAgent"),
    "load": ("laya_linux.agent", "load"),
    "RemoteAgent": ("laya_linux.client.http", "RemoteAgent"),
}


def __getattr__(name: str) -> Any:
    spec = _LAZY.get(name)
    if spec is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = spec
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value  # cache: subsequent accesses skip the hook
    return value


def __dir__() -> list[str]:
    return sorted([*globals(), *_LAZY])


__all__ = [
    "Agent",
    "RLAgent",
    "load",
    "RemoteAgent",
    "Router",
    "RouteDecision",
    "DEFAULT_MODELS",
    "DecisionAgent",
    "PROTOCOL_VERSION",
    "detect_language",
    "detect_script",
    "is_english",
    "clean_email_body",
    "email_state",
    "email_questions",
    "guard_questions",
    "moderation_questions",
    "router_questions",
    "triage_questions",
    "__version__",
]
