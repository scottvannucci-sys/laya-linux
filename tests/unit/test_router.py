"""Router policy: precedence, LRU residency, attach (unit, no model loads)."""

import pytest

from laya_linux.router import (
    Router,
    match_typed_decisions_workflow,
    normalise_name,
)

MODELS = {"english": "nonexistent/english", "multilingual": "nonexistent/multi"}


def test_normalise_aliases():
    assert normalise_name("en") == "english"
    assert normalise_name("typed") == "typed-decisions"
    assert normalise_name("ML") == "multilingual"
    with pytest.raises(ValueError):
        normalise_name("gpt-4")


def test_route_precedence():
    r = Router(models=MODELS)
    assert r.route("x", {}, model="multilingual").model == "multilingual"
    assert r.route("x", {}, task="typed_decisions").model == "typed-decisions"
    assert r.route("x", {}, lang="fr").model == "multilingual"
    assert r.route("x", {}, lang="en-US").model == "english"


def test_route_by_script():
    r = Router(models=MODELS)
    assert r.route("hello world this is english", {}).model == "english"
    assert r.route("こんにちは", {}).model == "multilingual"
    assert r.route("Привет мир, как дела", {}).model == "multilingual"
    assert r.route("Bonjour le monde, comment allez-vous", {}).model == "multilingual"
    assert r.route("", {}).model == "english"  # unknown script -> default


def test_typed_decisions_never_auto_selected_by_default():
    r = Router(models=MODELS)
    wf_questions = {"action": {}, "needs_review": {}, "outcome": {}, "risk": {}, "urgency": {}}
    assert match_typed_decisions_workflow(wf_questions) == "agent_trace_observability"
    assert r.route("english text", wf_questions).model == "english"
    r2 = Router(models=MODELS, auto_task_detection=True)
    assert r2.route("english text", wf_questions).model == "typed-decisions"


def test_route_decision_serializes():
    d = Router(models=MODELS).route("hello", {})
    assert isinstance(d, dict) and "model" in d and "reason" in d


class FakeAgent:
    def __init__(self):
        self.calls = 0

    def system_one(self, state, questions):
        self.calls += 1
        return {"model": "fake", "answers": {}, "usage": {}}


def test_attach_raises_residency_and_lru_evicts():
    """attach() grows residency (upstream semantics); eviction is deterministic LRU."""

    r = Router(models=MODELS, max_loaded=1)
    a1, a2 = FakeAgent(), FakeAgent()
    r.attach("english", a1)
    r.predict("hello there friend", {"q": {"type": "noul", "instructions": "i"}})
    assert a1.calls == 1
    r.attach("multilingual", a2)  # attach raises max_loaded instead of evicting
    assert r.max_loaded == 2
    assert r.loaded == ["english", "multilingual"]
    r.max_loaded = 1
    r._evict()  # deterministic bounded-residency enforcement
    assert r.loaded == ["multilingual"]  # english was least-recently-used
    # evicted, and its configured value looks like a hub id -> NETWORK_DISABLED
    from laya_linux.errors import NetworkDisabledError

    with pytest.raises(NetworkDisabledError):
        r.load("english")
    r.unload()
    assert r.loaded == []


def test_unconfigured_model_load_fails_clearly():
    r = Router(models={"english": "does/not/matter"})
    with pytest.raises(ValueError):
        r.load("multilingual")  # never configured
