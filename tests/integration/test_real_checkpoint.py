"""Real-checkpoint integration tests (opt-in; skipped unless a checkpoint is staged)."""

import pytest

torch = pytest.importorskip("torch")

from conftest import REAL_CHECKPOINT

pytestmark = [
    pytest.mark.real_checkpoint,
    pytest.mark.skipif(not REAL_CHECKPOINT.is_dir(), reason="real checkpoint not staged locally"),
]

QUESTIONS = {
    "intent": {
        "type": "choice",
        "instructions": "What does the customer want in `message`?",
        "criteria": {
            "refund": "money returned or a duplicate charge reversed",
            "technical_help": "a bug, outage or integration problem",
            "billing_question": "a question about an invoice, plan or payment method",
        },
    },
    "is_urgent": {"type": "noul", "instructions": "Does `message` communicate time pressure or a deadline?"},
    "frustration": {
        "type": "score",
        "instructions": "How frustrated does the customer sound in `message`?",
        "criteria": ["calm and neutral", "concerned but civil", "clearly annoyed"],
    },
}
STATE = {"message": "I was charged twice on my last invoice."}


def test_real_checkpoint_loads_with_sockets_blocked():
    from conftest import SocketBlocker

    import laya_linux

    with SocketBlocker() as blocker:
        agent = laya_linux.load(str(REAL_CHECKPOINT), device="cpu")
        result = agent.predict(STATE, QUESTIONS)
    assert blocker.attempts == []
    assert set(result["answers"]) == set(QUESTIONS)
    assert result["model"] == "laya-rl-agent"


def test_real_checkpoint_deterministic():
    from laya_linux import Agent

    agent = Agent(str(REAL_CHECKPOINT), device="cpu")
    assert agent.predict(STATE, QUESTIONS) == agent.predict(STATE, QUESTIONS)


def test_real_checkpoint_probabilities_wellformed():
    from laya_linux import Agent

    result = Agent(str(REAL_CHECKPOINT), device="cpu").predict(STATE, QUESTIONS)
    for answer in result["answers"].values():
        assert 0.0 <= answer["confidence"] <= 1.0
        assert 0.0 <= answer["action"]["act_probability"] <= 1.0
