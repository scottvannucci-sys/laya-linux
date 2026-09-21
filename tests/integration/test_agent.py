"""Agent behavior: schema, determinism, batching, memory stability (integration)."""


import pytest

torch = pytest.importorskip("torch")

from conftest import QUESTIONS, STATE


def test_result_schema_matches_reference(agent):
    result = agent.predict(STATE, QUESTIONS)
    assert result["model"] == "laya-rl-agent"
    assert result["usage"]["output_tokens"] == 0
    assert result["usage"]["input_tokens"] > 0
    a = result["answers"]
    assert set(a["department"]) == {"type", "choice", "probabilities", "confidence", "action"}
    assert a["department"]["choice"] in {"billing", "technical"}
    assert set(a["urgency"]) == {"type", "score", "legend", "probabilities", "confidence", "action"}
    assert set(a["refund"]) == {"type", "noul", "confidence", "action"}
    for prob in a["department"]["probabilities"].values():
        assert 0.0 <= prob <= 1.0
    assert abs(sum(a["department"]["probabilities"].values()) - 1.0) < 1e-3
    assert abs(sum(a["urgency"]["probabilities"].values()) - 1.0) < 1e-3
    for v in (*a["department"]["probabilities"].values(), a["refund"]["noul"]):
        assert round(v, 4) == v  # four-decimal public rounding


def test_confidence_bounds(agent):
    result = agent.predict(STATE, QUESTIONS)
    for answer in result["answers"].values():
        assert 0.0 <= answer["confidence"] <= 1.0
        assert 0.0 <= answer["action"]["act_probability"] <= 1.0


def test_noul_confidence_is_max_p(agent):
    r = agent.predict(STATE, {"n": {"type": "noul", "instructions": "i"}})
    n = r["answers"]["n"]
    assert n["confidence"] == round(max(n["noul"], 1.0 - n["noul"]), 4)


def test_determinism_across_repeated_calls(agent):
    r1 = agent.predict(STATE, QUESTIONS)
    r2 = agent.predict(STATE, QUESTIONS)
    r3 = agent.system_one(STATE, QUESTIONS)
    assert r1 == r2 == r3


def test_score_expected_value(agent):
    r = agent.predict(STATE, {"s": {"type": "score", "instructions": "i", "criteria": ["a", "b", "c"]}})
    s = r["answers"]["s"]
    expected = sum(i * p for i, p in enumerate(s["probabilities"].values()))
    assert abs(s["score"] - expected) < 5e-3  # rounding-aware


def test_batch_chunking_does_not_change_results(tiny_pkg):
    """Splitting questions across forward passes must not change public results."""
    from laya_linux import Agent

    many = {
        f"q{i}": {"type": "noul", "instructions": f"Question number {i}?"} for i in range(7)
    }
    a16 = Agent(str(tiny_pkg), device="cpu", batch_size=16)
    a2 = Agent(str(tiny_pkg), device="cpu", batch_size=2)
    r16, r2 = a16.predict(STATE, many), a2.predict(STATE, many)
    assert r16["answers"] == r2["answers"]


def test_usage_counts_tokens(agent):
    short = agent.predict("hi", {"n": {"type": "noul", "instructions": "i"}})
    long = agent.predict("tok1 " * 200, {"n": {"type": "noul", "instructions": "i"}})
    assert short["usage"]["input_tokens"] < long["usage"]["input_tokens"]


def test_repeated_inference_memory_stability(agent):
    """Active memory must not grow unboundedly over repeated calls."""
    if not torch.cuda.is_available():
        pytest.skip("active-memory accounting requires CUDA")
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    for _ in range(30):
        agent.predict(STATE, QUESTIONS)
    torch.cuda.synchronize()
    growth = torch.cuda.memory_allocated() - base
    assert growth < 50 * 1024 * 1024  # < 50MB residual growth over 30 calls


def test_thread_safety_documented_level(agent):
    """Concurrent predictions complete and agree with serial results (§12)."""
    import concurrent.futures

    serial = agent.predict(STATE, QUESTIONS)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(agent.predict, STATE, QUESTIONS) for _ in range(8)]
        results = [f.result() for f in futures]
    assert all(r == serial for r in results)


def test_load_rejects_remote_identifier():
    from laya_linux import load
    from laya_linux.errors import NetworkDisabledError

    with pytest.raises(NetworkDisabledError):
        load("convaiinnovations/laya")


def test_option_order_preserved(agent):
    questions = {
        "c": {"type": "choice", "instructions": "i", "criteria": {"zeta": None, "alpha": None, "mid": None}}
    }
    r = agent.predict(STATE, questions)
    assert list(r["answers"]["c"]["probabilities"].keys()) == ["zeta", "alpha", "mid"]
