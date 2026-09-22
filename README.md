# laya-linux

Local-first inference runtime for [Laya](https://github.com/NandhaKishorM/laya)
typed-decision models on Linux. It evaluates constrained questions about a
supplied state in a single bidirectional model forward pass and returns
calibrated probabilities without generating text — entirely on your hardware,
with no cloud calls, telemetry, or downloads.

**Status: Phases 1–4 complete — CPU + CUDA runtimes, packaging/CLI, and the private server.** The FP32
CPU runtime matches Hugging Face transformers' ModernBERT and upstream Laya's `DecisionModel` to
**0.0 max difference**; CUDA is validated on RTX 5070 Ti (sm_120) with FP32 bit-identical to CPU
and FP16/BF16 drift tolerances recorded (see [`PARITY_BASELINES.md`](PARITY_BASELINES.md) and
`benchmarks/`). `laya-linux serve` exposes the private JSON API (loopback/Unix-socket/private-LAN
with auth; see [`docs/server.md`](docs/server.md)), and the PyTorch-free `RemoteAgent` client
implements the same prediction interface over the standard library only.

## Documents

| Document | Role |
|---|---|
| [`requirements.md`](requirements.md) | Normative product specification (MUST/SHOULD requirements, v0.1.0 acceptance criteria) |
| [`architecture.md`](architecture.md) | Technical design: repository layout, components, parity strategy, implementation phases |
| [`PARITY_BASELINES.md`](PARITY_BASELINES.md) | Machine-readable pins for upstream code revisions, checkpoint revisions, and tolerance results |

`requirements.md` wins on any conflict.

## Planned deployment modes

1. **Embedded** — import the Python package, run inference in-process.
2. **Private server** — lightweight (PyTorch-free) client → your own server on
   loopback, a Unix-domain socket, or a private LAN address. Loopback-only by
   default.

Both modes are designed to run with the network unavailable, enforced by tests
that block socket creation.

## Provenance and licensing

The implementation is informed by two Apache-2.0 projects, verified at these
revisions:

- [Laya](https://github.com/NandhaKishorM/laya) — commit `42626c348753fbb17572a813127df2278a1ec527`
- [Laya-MLX](https://github.com/mizorewww/laya-mlx) — commit `fc1df62828a3fedf4d8229fdac1cbd85f1cdf337`
  (whose parity reference is upstream Laya commit `6a5819129eb220570792e417e49723d697efd76f`)

Attribution lives in [`NOTICE`](NOTICE); the runtime is licensed under
Apache-2.0 ([`LICENSE`](LICENSE)). Model checkpoints are separate artifacts
with their own recorded licenses — currently
[convaiinnovations/laya-typed-decisions](https://huggingface.co/convaiinnovations/laya-typed-decisions)
and [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya),
both Apache-2.0 — and are never bundled in this repository or its releases.

## Contributing / agent rules

Autonomous agents MUST read both specification documents before editing, and
MUST follow the working rules in [`architecture.md`](architecture.md)
§15 — including: no implicit model downloads, no cloud fallback, no GPU
support claims without real hardware results, and tests for every externally
observable behavior change.
