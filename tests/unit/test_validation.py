"""Question validation and input handling (unit)."""

import pytest

BAD_QUESTIONS = [
    ({"x": {"type": "unknown", "instructions": "i"}}, "unknown type"),
    ({"x": {"type": "choice", "instructions": "i"}}, "choice without criteria"),
    ({"x": {"type": "choice", "instructions": "i", "criteria": []}}, "empty choice"),
    ({"x": {"type": "choice", "instructions": "i", "criteria": ["a", "a"]}}, "duplicate labels"),
    ({"x": {"type": "choice", "instructions": "i", "criteria": ["a", 2]}}, "non-string list labels"),
    ({"x": {"type": "score", "instructions": "i"}}, "score without criteria"),
    ({"x": {"type": "score", "instructions": "i", "criteria": []}}, "empty score"),
    ({"x": {"type": "noul", "instructions": "i", "criteria": ["not-a-dict"]}}, "bad noul criteria"),
    ({"x": {"instructions": "i"}}, "missing type"),
    ({"x": {"type": "noul"}}, "missing instructions"),
    ("not-a-dict", "question not a dict"),
]


@pytest.mark.parametrize("questions,label", BAD_QUESTIONS, ids=[lab for _, lab in BAD_QUESTIONS])
def test_malformed_questions_rejected(agent, questions, label):
    with pytest.raises((ValueError, TypeError)):
        agent.predict({"m": "x"}, questions)


def test_empty_questions_no_model_execution(agent):
    result = agent.predict({"m": "x"}, {})
    assert result["answers"] == {}
    assert result["usage"] == {"input_tokens": 0, "output_tokens": 0}


def test_questions_must_be_dict(agent):
    with pytest.raises(ValueError):
        agent.predict({"m": "x"}, ["not", "a", "dict"])


def test_non_string_instructions_are_json_encoded(agent):
    questions = {"q": {"type": "noul", "instructions": {"prompt": "is it true?"}}}
    result = agent.predict({"m": "x"}, questions)  # must not raise
    assert "q" in result["answers"]
