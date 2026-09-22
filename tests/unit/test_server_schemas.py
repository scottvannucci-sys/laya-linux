"""Server schema validation: semantic limits at the API boundary (unit)."""

import pytest

from laya_linux.errors import (
    MalformedRequestError,
    ProtocolUnsupportedError,
    RequestTooLargeError,
)
from laya_linux.protocol import PROTOCOL_VERSION
from laya_linux.server.schemas import validate_predict_payload


def _valid(**overrides):
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "state": {"message": "hello"},
        "questions": {"q": {"type": "noul", "instructions": "Is this hello?"}},
    }
    payload.update(overrides)
    return payload


def test_valid_payload_passes():
    request = validate_predict_payload(_valid(), max_questions=10, max_options=32)
    assert request["model"] is None
    assert "q" in request["questions"]


def test_wrong_protocol_version_rejected():
    with pytest.raises(ProtocolUnsupportedError):
        validate_predict_payload(_valid(protocol_version=999), max_questions=10, max_options=32)


def test_missing_or_malformed_parts_rejected():
    for bad in (
        {},
        "not a dict",
        _valid(state=42),
        _valid(questions=[]),
        _valid(questions={"": {"type": "noul", "instructions": "i"}}),
    ):
        with pytest.raises(MalformedRequestError):
            validate_predict_payload(bad, max_questions=10, max_options=32)


def test_question_count_bounded():
    questions = {f"q{i}": {"type": "noul", "instructions": "i"} for i in range(5)}
    with pytest.raises(RequestTooLargeError):
        validate_predict_payload(_valid(questions=questions), max_questions=4, max_options=32)


def test_option_count_bounded():
    labels = [f"opt{i}" for i in range(6)]
    with pytest.raises(RequestTooLargeError):
        validate_predict_payload(
            _valid(questions={"c": {"type": "choice", "instructions": "i", "criteria": labels}}),
            max_questions=10,
            max_options=5,
        )


def test_bad_question_definitions_rejected():
    for bad in (
        {"type": "unknown", "instructions": "i"},
        {"type": "noul"},
        {"type": "noul", "instructions": ""},
        {"type": "choice", "instructions": "i", "criteria": ["a", "a"]},
        {"type": "choice", "instructions": "i", "criteria": "nope"},
        {"type": "score", "instructions": "i", "criteria": []},
        {"type": "noul", "instructions": "i", "criteria": ["not-a-dict"]},
        "just a string",
    ):
        with pytest.raises(MalformedRequestError):
            validate_predict_payload(_valid(questions={"q": bad}), max_questions=10, max_options=32)


def test_oversized_strings_rejected():
    with pytest.raises(RequestTooLargeError):
        validate_predict_payload(_valid(state="x" * 300_000), max_questions=10, max_options=32)
    with pytest.raises(RequestTooLargeError):
        validate_predict_payload(
            _valid(questions={"q": {"type": "noul", "instructions": "x" * 5000}}),
            max_questions=10,
            max_options=32,
        )


def test_nested_state_size_bounded():
    deep = {"a": {"b": {"c": ["x" * 1000] * 300}}}
    with pytest.raises(RequestTooLargeError):
        validate_predict_payload(_valid(state=deep), max_questions=10, max_options=32)
