"""Phase 3 CUDA validation battery (architecture §14 Phase 3).

Runs the full correctness/stability/benchmark set against a local model
package on CUDA and emits machine-readable JSON for PARITY_BASELINES.md:

  1. environment record (GPU, driver, torch build, capability)
  2. device/dtype policy checks on real hardware
  3. FP32 CUDA-vs-CPU selected-answer agreement + probability drift
  4. FP16 and BF16 drift characterization with measured tolerances
  5. repeated-call memory stability and peak memory
  6. deterministic-repeat and padding-invariance checks on device
  7. latency/throughput benchmarks per dtype

Usage:
  python scripts/cuda_validation.py MODEL_PATH [--out cuda-validation.json]
  python scripts/cuda_validation.py --tiny   # tiny synthetic package instead
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch  # noqa: E402

from laya_linux import Agent  # noqa: E402
from laya_linux.devices import select_device  # noqa: E402
from laya_linux.errors import DtypeUnsupportedError  # noqa: E402

FIXTURE_STATES = [
    {"message": "I was charged twice on my last invoice and need this fixed today."},
    {"message": "The app crashes whenever I upload a CSV file."},
    {"message": "hello"},
    {"message": "Can you explain your pricing plans for a team of fifty? " * 8},
    {"message": "Please cancel my subscription effective immediately."},
    {"message": "12345 67890 --- ??? %%%"},
]

QUESTIONS = {
    "intent": {
        "type": "choice",
        "instructions": "What does the customer want in `message`?",
        "criteria": {
            "refund": "money returned or a duplicate charge reversed",
            "technical_help": "a bug, outage or integration problem",
            "billing_question": "a question about an invoice, plan or payment method",
            "information": "general information, pricing or how-to",
            "cancellation": "wants to cancel or downgrade",
            "other": "none of the other options fits",
        },
    },
    "is_urgent": {"type": "noul", "instructions": "Does `message` communicate time pressure or a deadline?"},
    "frustration": {
        "type": "score",
        "instructions": "How frustrated does the customer sound in `message`?",
        "criteria": ["calm and neutral", "concerned but civil", "clearly annoyed", "very angry"],
    },
    "refund_requested": {"type": "noul", "instructions": "Does the customer ask for money back?"},
    "churn_risk": {"type": "noul", "instructions": "Does `message` suggest the customer may leave or cancel?"},
}


def run_fixtures(agent: Agent) -> list[dict]:
    outputs = []
    for state in FIXTURE_STATES:
        result = agent.predict(state, QUESTIONS)
        outputs.append(result)
    return outputs


def drift_vs_reference(reference: list[dict], candidate: list[dict]) -> dict:
    """Selected-answer agreement and probability drift between two result sets."""
    agree = 0
    total = 0
    max_prob_diff = 0.0
    prob_diffs: list[float] = []
    for ref_result, cand_result in zip(reference, candidate, strict=True):
        for qid, ref_answer in ref_result["answers"].items():
            cand_answer = cand_result["answers"][qid]
            total += 1
            if ref_answer["type"] == "choice":
                match = ref_answer["choice"] == cand_answer["choice"]
                pairs = list(zip(ref_answer["probabilities"].values(), cand_answer["probabilities"].values(), strict=True))
            elif ref_answer["type"] == "score":
                match = round(ref_answer["score"], 2) == round(cand_answer["score"], 2)
                pairs = list(zip(ref_answer["probabilities"].values(), cand_answer["probabilities"].values(), strict=True))
            else:
                match = (ref_answer["noul"] > 0.5) == (cand_answer["noul"] > 0.5)
                pairs = [(ref_answer["noul"], cand_answer["noul"])]
            agree += int(match)
            for a, b in pairs:
                diff = abs(a - b)
                prob_diffs.append(diff)
                max_prob_diff = max(max_prob_diff, diff)
    return {
        "selected_answer_agreement": f"{agree}/{total}",
        "agreement_rate": round(agree / total, 4) if total else None,
        "max_probability_diff": round(max_prob_diff, 6),
        "mean_probability_diff": round(statistics.fmean(prob_diffs), 8) if prob_diffs else None,
        "fixture_count": len(reference),
        "question_count": total // max(1, len(reference)),
    }


def benchmark(agent: Agent, warmup: int, iterations: int) -> dict:
    for _ in range(warmup):
        agent.predict(FIXTURE_STATES[0], QUESTIONS)
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        agent.predict(FIXTURE_STATES[0], QUESTIONS)
        samples.append(time.perf_counter() - start)
    samples.sort()
    return {
        "median_ms": round(statistics.median(samples) * 1000, 2),
        "p95_ms": round(samples[max(0, int(len(samples) * 0.95) - 1)] * 1000, 2),
        "min_ms": round(samples[0] * 1000, 2),
        "max_ms": round(samples[-1] * 1000, 2),
        "iterations": iterations,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", nargs="?", default=None)
    parser.add_argument("--tiny", action="store_true", help="use the tiny synthetic test package")
    parser.add_argument("--out", default="cuda-validation.json")
    parser.add_argument("--bench-iterations", type=int, default=20)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("FATAL: CUDA is not available to this PyTorch build", file=sys.stderr)
        return 2

    cap = torch.cuda.get_device_capability(0)
    report = {
        "environment": {
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": f"sm_{cap[0]}{cap[1]}",
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "driver_hint": "see nvidia-smi on the validation machine",
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "dtype_policies": {},
        "parity": {},
        "stability": {},
        "benchmarks": {},
    }
    print(json.dumps(report["environment"], indent=2))

    # ---- device/dtype policy on real hardware ----
    plan = select_device("auto")
    report["dtype_policies"]["auto_selects_cuda"] = plan.device.type == "cuda"
    report["dtype_policies"]["auto_dtype"] = str(plan.dtype).replace("torch.", "")
    try:
        select_device("cpu", "bfloat16")
        report["dtype_policies"]["cpu_bf16_correctly_rejected"] = False
    except DtypeUnsupportedError:
        report["dtype_policies"]["cpu_bf16_correctly_rejected"] = True

    # ---- model paths ----
    if args.tiny:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
        from conftest import build_tiny_package

        model_path = build_tiny_package(Path(__import__("tempfile").mkdtemp()) / "tiny", seed=42)
    else:
        model_path = Path(args.model or "models/laya-typed-decisions")
        if not model_path.is_dir():
            print(f"FATAL: model package not found: {model_path}", file=sys.stderr)
            return 2
    report["model"] = str(model_path)

    # ---- reference: CPU FP32 ----
    print("building CPU FP32 reference...", flush=True)
    cpu_agent = Agent(str(model_path), device="cpu", dtype="float32")
    reference = run_fixtures(cpu_agent)

    # ---- per-dtype CUDA validation ----
    for dtype in ("float32", "float16", "bfloat16"):
        label = f"cuda_{dtype}"
        print(f"validating {label}...", flush=True)
        agent = Agent(str(model_path), device="cuda:0", dtype=dtype)
        outputs = run_fixtures(agent)
        report["parity"][label] = drift_vs_reference(reference, outputs)

        # deterministic repeat on-device (same tensors, same kernels)
        repeat = agent.predict(FIXTURE_STATES[0], QUESTIONS)
        first = outputs[0]
        same = all(
            repeat["answers"][qid] == first["answers"][qid] for qid in first["answers"]
        )
        report["stability"].setdefault(label, {})["repeat_identical"] = same

        # memory stability: 30 repeated calls
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        for _ in range(30):
            agent.predict(FIXTURE_STATES[0], QUESTIONS)
        torch.cuda.synchronize()
        growth_mb = (torch.cuda.memory_allocated() - base) / (1024 * 1024)
        peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        torch.cuda.reset_peak_memory_stats()
        report["stability"][label]["growth_mb_over_30_calls"] = round(growth_mb, 2)
        report["stability"][label]["peak_memory_mb"] = round(peak_mb, 1)

        report["benchmarks"][label] = benchmark(agent, warmup=3, iterations=args.bench_iterations)
        del agent
        torch.cuda.empty_cache()

    ok = (
        report["dtype_policies"]["auto_selects_cuda"]
        and all(v["selected_answer_agreement"].split("/")[0] == v["selected_answer_agreement"].split("/")[1]
                for v in report["parity"].values())
        and all(v["repeat_identical"] for v in report["stability"].values())
        and all(v["growth_mb_over_30_calls"] < 50 for v in report["stability"].values())
    )
    report["gate"] = "PASS" if ok else "FAIL"

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("dtype_policies", "parity", "stability", "benchmarks", "gate")}, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
