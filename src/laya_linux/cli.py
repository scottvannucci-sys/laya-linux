"""The `laya-linux` command line (requirements §11).

Commands: predict, verify, doctor, benchmark. `serve` arrives with the private
server in Phase 4. Every command works without contacting the internet;
diagnostic output distinguishes verified facts from suggestions and never
includes secrets.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from .errors import LayaError
from .version import __version__

PRESETS = {
    "triage": "triage_questions",
    "email": "email_questions",
    "guard": "guard_questions",
    "moderation": "moderation_questions",
    "router": "router_questions",
}


def _load_state(args: argparse.Namespace) -> Any:
    if args.state_file:
        payload = json.loads(Path(args.state_file).read_text(encoding="utf-8"))
        return payload
    if args.state is not None:
        return args.state
    raise SystemExit("error: provide --state or --state-file")


def _load_questions(args: argparse.Namespace) -> dict[str, Any]:
    if args.questions_file:
        return json.loads(Path(args.questions_file).read_text(encoding="utf-8"))
    if args.preset:
        import laya_linux

        return getattr(laya_linux, PRESETS[args.preset])()
    raise SystemExit("error: provide --questions-file or --preset")


def cmd_predict(args: argparse.Namespace) -> int:
    from . import Agent

    agent = Agent(args.model, device=args.device, dtype=args.dtype, batch_size=args.batch_size)
    state = _load_state(args)
    questions = _load_questions(args)
    result = agent.predict(state, questions)
    json.dump(result, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    from .packaging import verify_model

    try:
        report = verify_model(args.model)
    except LayaError as exc:
        json.dump({"ok": False, "error_code": exc.code, "error": str(exc)}, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 1
    if args.json:
        json.dump({"ok": True, "report": report}, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        _print_verify_human(report)
    return 0


def _print_verify_human(report: dict[str, Any]) -> None:
    print(f"model package : {report['path_resolved']}")
    print(f"manifest      : {report['manifest']}")
    arch = report["architecture"]
    print(
        f"architecture  : {arch['encoder']} hidden={arch['hidden_size']} "
        f"layers={arch['num_hidden_layers']} heads={arch['num_attention_heads']} "
        f"context={arch['context_limit']}"
    )
    print(f"weights       : {report['weights']} ({report['tensor_count']} tensors, {report['dtype']})")
    print(f"tokenizer     : special tokens {report['tokenizer']['special_tokens']}")
    print(f"limits        : max_len={report['max_len']} head_max_len={report['head_max_len']}")
    print(f"calibration   : {report['calibration']}")
    print("verified offline: yes")


def cmd_doctor(args: argparse.Namespace) -> int:
    facts: dict[str, Any] = {}
    suggestions: list[str] = []
    facts["python"] = platform.python_version()
    facts["platform"] = platform.platform()
    facts["laya_linux_version"] = __version__
    try:
        import torch

        facts["torch"] = torch.__version__
        facts["torch_cuda_build"] = bool(torch.version.cuda)
        facts["torch_hip_build"] = bool(getattr(torch.version, "hip", None))
        facts["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            facts["cuda_device"] = torch.cuda.get_device_name(0)
            facts["cuda_capability"] = ".".join(str(n) for n in torch.cuda.get_device_capability(0))
    except Exception as exc:  # pragma: no cover - torch is a hard dependency
        facts["torch"] = f"unavailable: {type(exc).__name__}"
        suggestions.append("install a CPU, CUDA, or ROCm PyTorch build for your platform")
    for module in ("numpy", "safetensors", "tokenizers"):
        try:
            facts[module] = __import__(module).__version__
        except Exception:
            facts[module] = "unavailable"
    try:
        from .devices import select_device

        facts["device_plan"] = select_device("auto").describe()
    except Exception as exc:
        facts["device_plan"] = f"unavailable: {type(exc).__name__}"
        suggestions.append("check that your PyTorch build matches your hardware")
    report = {"facts": facts, "suggestions": suggestions}
    if args.json:
        json.dump(report, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        print(f"laya-linux {__version__} doctor")
        for key, value in facts.items():
            print(f"  {key:18s}: {value}")
        for s in suggestions:
            print(f"  suggestion: {s}")
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    from . import Agent

    questions = json.loads(Path(args.questions_file).read_text(encoding="utf-8")) if args.questions_file else None
    if questions is None:
        import laya_linux

        questions = getattr(laya_linux, PRESETS[args.preset])()
    state = _load_state(args)

    agent = Agent(args.model, device=args.device, dtype=args.dtype, batch_size=args.batch_size)
    import torch

    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device": str(agent.device),
        "backend": agent.device_plan.backend,
        "dtype": str(agent.dtype).replace("torch.", ""),
        "checkpoint": agent.cfg.get("model_name", Path(args.model).name),
        "batch_size": args.batch_size,
        "warmup": args.warmup,
        "iterations": args.iterations,
    }

    samples: list[float] = []
    token_counts: list[int] = []
    for i in range(args.warmup + args.iterations):
        start = time.perf_counter()
        result = agent.predict(state, questions)
        elapsed = time.perf_counter() - start
        if i >= args.warmup:
            samples.append(elapsed)
            token_counts.append(result["usage"]["input_tokens"])

    peak_memory = None
    if agent.device.type == "cuda":
        peak_memory = torch.cuda.max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()
    else:
        try:
            import resource

            peak_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        except Exception:
            peak_memory = None

    samples.sort()
    report = {
        "environment": environment,
        "latency_seconds": {
            "median": statistics.median(samples),
            "p95": samples[max(0, int(len(samples) * 0.95) - 1)],
            "min": samples[0],
            "max": samples[-1],
            "mean": statistics.fmean(samples),
        },
        "throughput": {
            "questions_per_second": len(questions) / statistics.median(samples),
            "tokens_per_second": statistics.fmean(token_counts) / statistics.median(samples),
        },
        "peak_memory_bytes": peak_memory,
        "raw_samples_seconds": samples,
        "note": "single process, single device; do not compare across different hardware",
    }
    json.dump(report, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from .server.app import run_server
    from .server.config import ServerConfig

    models: dict[str, str] = {}
    for pair in args.model or []:
        alias, _, path = pair.partition("=")
        if not _:
            raise SystemExit(f"error: --model expects alias=path, got {pair!r}")
        models[alias.strip()] = path.strip()
    config = ServerConfig(
        models=models,
        host=args.host,
        port=args.port,
        unix_socket=args.unix_socket,
        token_file=args.token_file,
        device=args.device,
        dtype=args.dtype,
        max_request_bytes=args.max_request_bytes,
        max_questions=args.max_questions,
        max_options=args.max_options,
        queue_capacity=args.queue_capacity,
        concurrency=args.concurrency,
        execution_timeout=args.execution_timeout,
        preload=args.preload,
        access_log=not args.no_access_log,
    )
    run_server(config, access_log=not args.no_access_log)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="laya-linux",
        description="Local-first inference for Laya typed-decision models.",
    )
    parser.add_argument("--version", action="version", version=f"laya-linux {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("predict", help="run one prediction from local files")
    p.add_argument("model", help="path to a local model package directory")
    p.add_argument("--device", default="auto", help="auto, cpu, or cuda:N (default: auto)")
    p.add_argument("--dtype", default="auto", help="auto, float32, float16, bfloat16")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--state", help="state as an inline string")
    p.add_argument("--state-file", help="JSON file containing the state")
    p.add_argument("--questions-file", help="JSON file containing the questions")
    p.add_argument("--preset", choices=sorted(PRESETS), help="use a built-in question preset")
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("verify", help="validate a model package offline")
    p.add_argument("model", help="path to a local model package directory")
    p.add_argument("--json", action="store_true", help="machine-readable report")
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("doctor", help="report local runtime and device readiness")
    p.add_argument("--json", action="store_true", help="machine-readable report")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("benchmark", help="run a reproducible local benchmark (JSON)")
    p.add_argument("model", help="path to a local model package directory")
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="auto")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--state", help="state as an inline string")
    p.add_argument("--state-file", help="JSON file containing the state")
    p.add_argument("--questions-file", help="JSON file containing the questions")
    p.add_argument("--preset", choices=sorted(PRESETS), help="use a built-in question preset")
    p.set_defaults(func=cmd_benchmark)

    p = sub.add_parser("serve", help="start the private inference server (loopback by default)")
    p.add_argument("--model", action="append", metavar="ALIAS=PATH",
                   help="model alias to local package path (repeatable)")
    p.add_argument("--host", default="127.0.0.1",
                   help="TCP host to bind (default 127.0.0.1; non-loopback requires --token-file)")
    p.add_argument("--port", type=int, default=8142)
    p.add_argument("--unix-socket", default=None, help="Unix socket path (overrides --host/--port)")
    p.add_argument("--token-file", default=None, help="bearer token file (required for non-loopback)")
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default="auto")
    p.add_argument("--max-request-bytes", type=int, default=1 * 1024 * 1024)
    p.add_argument("--max-questions", type=int, default=64)
    p.add_argument("--max-options", type=int, default=256)
    p.add_argument("--queue-capacity", type=int, default=64)
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--execution-timeout", type=float, default=60.0)
    p.add_argument("--preload", action="store_true", help="load models at startup")
    p.add_argument("--no-access-log", action="store_true", help="disable the redacted access log")
    p.set_defaults(func=cmd_serve)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except LayaError as exc:
        json.dump({"error_code": exc.code, "error": str(exc)}, sys.stderr, indent=2)
        sys.stderr.write("\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
