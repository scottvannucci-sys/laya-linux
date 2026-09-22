"""Predict request validation at the API boundary (architecture §5.10, §8).

The server owns semantic limits before anything reaches the model: protocol
version, question count, option count, and string sizes. Validation reuses the
embedded runtime's question rules so server and library accept exactly the
same shapes.
"""

from __future__ import annotations

from typing import Any

from ..errors import MalformedRequestError, ProtocolUnsupportedError, RequestTooLargeError
from ..protocol import PROTOCOL_VERSION

# Conservative per-string caps enforced before tokenization (requirements §8.3:
# request size and question count MUST be bounded).
MAX_STATE_CHARS = 200_000
MAX_INSTRUCTIONS_CHARS = 4_000
MAX_CRITERION_CHARS = 4_000


def validate_predict_payload(
    payload: Any,
    *,
    max_questions: int,
    max_options: int,
) -> dict[str, Any]:
    """Validate a parsed predict request; returns the normalized request dict.

    Raises stable-code errors for every violation; never echoes request
    content in error messages.
    """
    if not isinstance(payload, dict):
        raise MalformedRequestError("Request body must be a JSON object")

    version = payload.get("protocol_version", PROTOCOL_VERSION)
    if version != PROTOCOL_VERSION:
        raise ProtocolUnsupportedError(
            f"Client protocol version {version!r} is not supported; server speaks version "
            f"{PROTOCOL_VERSION}"
        )

    questions = payload.get("questions")
    if not isinstance(questions, dict):
        raise MalformedRequestError("'questions' must be an object keyed by question id")
    if len(questions) > max_questions:
        raise RequestTooLargeError(
            f"Request contains more than the maximum of {max_questions} questions"
        )
    for qid, definition in questions.items():
        if not isinstance(qid, str) or not qid:
            raise MalformedRequestError("Question ids must be nonempty strings")
        _validate_question(definition, max_options=max_options)

    state = payload.get("state")
    if isinstance(state, str):
        if len(state) > MAX_STATE_CHARS:
            raise RequestTooLargeError("State exceeds the maximum string length")
    elif isinstance(state, (dict, list)):
        # deep string bounds without importing json recursively here
        if _count_state_chars(state) > MAX_STATE_CHARS:
            raise RequestTooLargeError("State exceeds the maximum serialized size")
    else:
        raise MalformedRequestError("'state' must be a string, object, or array")

    model = payload.get("model")
    if model is not None and not isinstance(model, str):
        raise MalformedRequestError("'model' must be a string alias")

    return {"state": state, "questions": questions, "model": model}


def _validate_question(definition: Any, *, max_options: int) -> None:
    if not isinstance(definition, dict):
        raise MalformedRequestError("Each question must be an object")
    kind = definition.get("type")
    if kind not in ("choice", "score", "noul"):
        raise MalformedRequestError("Question 'type' must be choice, score, or noul")
    instructions = definition.get("instructions")
    if not isinstance(instructions, str) or not instructions:
        raise MalformedRequestError("Question 'instructions' must be a nonempty string")
    if len(instructions) > MAX_INSTRUCTIONS_CHARS:
        raise RequestTooLargeError("Question instructions exceed the maximum length")
    criteria = definition.get("criteria")
    if kind == "choice":
        if isinstance(criteria, list):
            if not all(isinstance(c, str) for c in criteria):
                raise MalformedRequestError("Choice labels must be strings")
            if len(set(criteria)) != len(criteria):
                raise MalformedRequestError("Choice labels must be unique")
            count = len(criteria)
        elif isinstance(criteria, dict):
            if not criteria:
                raise MalformedRequestError("Choice criteria must not be empty")
            count = len(criteria)
        else:
            raise MalformedRequestError("Choice criteria must be an object or list of labels")
        if count > max_options:
            raise RequestTooLargeError(
                f"Question has more than the maximum of {max_options} options"
            )
        _bound_criterion_values(criteria, kind)
    elif kind == "score":
        if not isinstance(criteria, list) or not criteria:
            raise MalformedRequestError("Score criteria must be a nonempty list")
        if len(criteria) > max_options:
            raise RequestTooLargeError(
                f"Question has more than the maximum of {max_options} options"
            )
        _bound_criterion_values(criteria, kind)
    else:  # noul
        if criteria is not None and not isinstance(criteria, dict):
            raise MalformedRequestError("Noul criteria must be an object with false/true keys")
        _bound_criterion_values(criteria or {}, kind)


def _bound_criterion_values(criteria: Any, kind: str) -> None:
    values = criteria.values() if isinstance(criteria, dict) else iter(criteria)
    for value in values:
        if value is None:
            continue
        text = value if isinstance(value, str) else None
        if text is None and isinstance(value, (dict, list, int, float, bool)):
            import json

            try:
                text = json.dumps(value, ensure_ascii=False, default=str)
            except Exception:
                text = str(value)
        if text and len(text) > MAX_CRITERION_CHARS:
            raise RequestTooLargeError("A criterion description exceeds the maximum length")


def _count_state_chars(state: Any, depth: int = 0) -> int:
    if depth > 32:
        return 0
    if isinstance(state, str):
        return len(state)
    if isinstance(state, dict):
        return sum(_count_state_chars(v, depth + 1) for v in state.values())
    if isinstance(state, (list, tuple)):
        return sum(_count_state_chars(v, depth + 1) for v in state)
    return 0
