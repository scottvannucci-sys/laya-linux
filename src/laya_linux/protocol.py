"""Structural interfaces shared by embedded and remote execution (architecture §5.1)."""

from __future__ import annotations

from typing import Any, Protocol

State = str | dict[str, Any] | list
Questions = dict[str, dict[str, Any]]
PredictionResult = dict[str, Any]

PROTOCOL_VERSION = 1


class DecisionAgent(Protocol):
    """Anything that can answer typed-decision questions about a state.

    Embedded ``Agent`` and the future PyTorch-free remote client both satisfy
    this protocol, so application code can accept either without knowing which
    one is in use.
    """

    def predict(self, state: State, questions: Questions) -> PredictionResult: ...

    def system_one(self, state: State, questions: Questions) -> PredictionResult: ...
