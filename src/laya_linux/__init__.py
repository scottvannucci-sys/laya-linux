"""Laya typed-decision inference on Linux, local-first and offline by construction."""

from .agent import Agent, RLAgent, load
from .email import clean_email_body, email_state
from .lang import analyse as detect_language
from .lang import detect_script, is_english
from .presets import (
    email_questions,
    guard_questions,
    moderation_questions,
    router_questions,
    triage_questions,
)
from .protocol import PROTOCOL_VERSION, DecisionAgent
from .router import DEFAULT_MODELS, RouteDecision, Router
from .version import __version__

__all__ = [
    "Agent",
    "RLAgent",
    "load",
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
