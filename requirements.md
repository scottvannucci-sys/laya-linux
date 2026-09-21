# Laya-Linux Product Requirements

## 1. Document purpose

This document is the normative product specification for **Laya-Linux**. It is written so a
human contributor or an autonomous coding agent can implement the project without relying on
the conversation that created it.

The terms **MUST**, **MUST NOT**, **SHOULD**, **SHOULD NOT**, and **MAY** are used as normative
requirements.

## 2. Product definition

Laya-Linux is an open-source, local-first inference runtime for Laya typed-decision models on
Linux. It evaluates constrained questions about a supplied state in a single bidirectional model
forward pass and returns calibrated probabilities without generating text.

The project supports two deployment modes:

1. **Embedded mode:** the application imports a Python package and runs inference in-process.
2. **Private server mode:** a lightweight client sends requests to a Laya-Linux server controlled
   by the user on loopback, a Unix-domain socket, or a private local network.

Neither mode may require the public internet during normal operation.

## 3. Source projects and provenance

The implementation is informed by these Apache-2.0 projects:

- Laya: <https://github.com/NandhaKishorM/laya>
- Laya-MLX: <https://github.com/mizorewww/laya-mlx>

The initial investigation used:

- Laya commit `42626c348753fbb17572a813127df2278a1ec527`
- Laya-MLX commit `fc1df62828a3fedf4d8229fdac1cbd85f1cdf337`
- The Laya-MLX parity reference identifies upstream Laya commit
  `6a5819129eb220570792e417e49723d697efd76f`.

Agents MUST verify the applicable licenses and preserve required attribution, copyright notices,
and `NOTICE` content when adapting code. Model licenses and provenance MUST be recorded separately
from the runtime's source-code license.

## 4. Goals

Laya-Linux MUST:

- Run Laya inference locally on Linux using open model weights.
- Require no cloud inference API.
- Avoid sending prompts, state, questions, results, logs, or telemetry to the public internet.
- Support both in-process inference and a user-controlled local/private inference server.
- Preserve the public typed-decision behavior of Laya and Laya-MLX.
- Load complete model packages from local filesystem paths.
- Validate model integrity and architectural compatibility before inference.
- Provide reproducible correctness, numerical-parity, memory-stability, and performance tests.
- Provide clear CPU, NVIDIA CUDA, and AMD ROCm installation and deployment guidance.
- Fail clearly and safely when a requested model, device, or server is unavailable.

## 5. Non-goals

The initial project MUST NOT attempt to:

- Train or fine-tune models.
- Reproduce Jev's proprietary service or private implementation.
- Provide a public hosted inference service.
- Automatically fall back to a cloud service.
- Bundle large model weights in the source repository or Python wheel.
- Promise exact latency before measurements exist for the stated hardware and benchmark method.
- Support arbitrary Hugging Face architectures. Initial model support is limited to validated Laya
  checkpoints and their supported ModernBERT variants.
- Make correctness or safety claims outside the scope of published and locally reproduced tests.

ONNX Runtime, TensorRT, Intel XPU, quantization, and training support are possible future work, not
requirements for version 0.1.0.

## 6. Required typed-decision behavior

### 6.1 Inputs

The inference API MUST accept:

- `state`: a string, JSON-compatible dictionary, or JSON-compatible list.
- `questions`: a dictionary keyed by caller-defined question IDs.

Supported question types:

- `choice`: select one named option and return the probability of every option.
- `score`: return a probability distribution over ordered rubric levels and its expected
  zero-based score.
- `noul`: return `P(true)` for a proposition, with false/true criteria optionally supplied.

Question validation MUST reject malformed definitions before model execution. Validation MUST
cover unknown types, missing instructions, empty criteria, duplicate choice labels, invalid
criteria shapes, and options that cannot fit within the configured token budget.

### 6.2 Outputs

Results MUST remain compatible with the established Laya result shape:

```json
{
  "model": "laya-rl-agent",
  "answers": {
    "department": {
      "type": "choice",
      "choice": "billing",
      "probabilities": {
        "billing": 0.91,
        "technical": 0.06,
        "sales": 0.03
      },
      "confidence": 0.72,
      "action": {
        "act_probability": 0.97
      }
    }
  },
  "usage": {
    "input_tokens": 42,
    "output_tokens": 0
  }
}
```

The runtime MUST retain the checkpoint's prompt construction, option ordering, truncation rules,
temperature calibration, four-decimal public rounding, and zero output-token semantics.

### 6.3 Public Python API

Embedded inference MUST support this conceptual interface:

```python
import laya_linux as laya

agent = laya.load("/opt/laya/models/laya-typed-decisions")
result = agent.predict(state, questions)
```

The public API MUST provide:

- `load(path, ...)`
- `Agent`
- `Agent.predict(...)`
- `Agent.system_one(...)` as an alias for `predict`
- `Router`
- Built-in question presets carried over from the source projects where licensing permits
- A client object implementing the same prediction interface for private server mode

The API SHOULD remain source-compatible with Laya-MLX where backend-specific arguments do not
apply.

## 7. Offline and privacy requirements

Offline operation is a core product guarantee.

### 7.1 Runtime networking

- Embedded inference MUST perform no network operations.
- Loading a model from a local path MUST perform no network operations.
- A missing local model MUST produce an actionable error and MUST NOT trigger a download.
- Remote Hugging Face-style identifiers MUST be rejected by the core runtime.
- The core runtime MUST NOT include telemetry, analytics, crash uploading, update checks, remote
  logging, advertisements, or cloud fallbacks.
- Runtime tests MUST include a mode that blocks socket creation and confirms embedded prediction
  still works.
- Network-capable model acquisition tooling, if ever provided, MUST be a separate optional tool,
  explicitly invoked by the user, and not imported by the inference runtime.

### 7.2 Offline installation

Documentation MUST explain how to install from a local Python wheelhouse and how to transfer a
model package from a staging machine using removable media or another user-controlled process.
Checksums MUST be verifiable without internet access.

### 7.3 Data retention

- Prompts, states, questions, and results MUST NOT be persisted by default.
- Request bodies MUST NOT appear in normal server access logs.
- Debug logging that can contain user data MUST be explicitly enabled and clearly documented.
- The server MUST NOT retain request content after completion unless a user explicitly configures
  a local retention mechanism.

## 8. Local/private server requirements

### 8.1 Deployment boundary

The server MUST support:

- Loopback TCP (`127.0.0.1` and `::1`).
- Unix-domain sockets on Linux.
- Explicit binding to a private LAN address for a separate inference machine.

The default MUST be loopback only. Binding to a non-loopback address MUST require explicit user
configuration. Documentation MUST include firewall guidance for limiting access to known clients.

There MUST be no automatic public endpoint discovery and no public-internet fallback.

### 8.2 HTTP API

The first server release SHOULD use a versioned JSON HTTP API with at least:

- `POST /v1/predict`
- `GET /v1/health`
- `GET /v1/models`
- `GET /v1/system`

The predict endpoint MUST accept the same state and questions as the Python API and return the same
result shape. Protocol errors MUST use stable machine-readable error codes.

Health and system endpoints MUST NOT expose prompts, secrets, filesystem paths containing user
names, environment variables, or model weight contents.

### 8.3 Server security

- Authentication MUST be available for LAN deployments.
- Static bearer-token authentication MAY be the initial mechanism.
- TLS and mutual TLS SHOULD be documented for networks that are not physically trusted.
- API tokens MUST be accepted through environment variables, protected files, or secret managers;
  they MUST NOT be required on command lines that expose them through process listings.
- Request size, maximum question count, sequence length, queue length, and concurrency MUST be
  bounded.
- The server MUST reject unsupported content types and malformed JSON.
- Server errors MUST not include sensitive request data.

### 8.4 Resource-aware operation

- One model copy per accelerator SHOULD be the default.
- GPU deployments MUST NOT create duplicate model copies merely because an HTTP server uses
  multiple process workers.
- Concurrency SHOULD be handled with an in-process queue and, later, optional bounded dynamic
  batching.
- The server MUST provide overload responses rather than consume unbounded memory.
- A constrained client MUST be able to use the server without installing PyTorch or storing model
  weights.

## 9. Model package requirements

The runtime MUST load models from a self-contained local directory such as:

```text
laya-typed-decisions/
├── model.safetensors
├── manifest.json
├── rl_agent_config.json
├── encoder/
│   └── config.json
└── tokenizer/
    ├── tokenizer.json
    └── tokenizer_config.json
```

The manifest MUST include:

- Format name and version.
- Model family and checkpoint identifier.
- Source and conversion provenance.
- Model license identifier or local license-file reference.
- Runtime compatibility range.
- Tensor dtype.
- SHA-256 digest and byte size for every required file.
- Architecture information needed to reject incompatible checkpoints.

The loader MUST:

- Prevent path traversal when resolving files.
- Require all expected files.
- Validate parameter names and shapes strictly.
- Validate manifest checksums when a manifest is present.
- Detect unsupported encoders and unsupported RoPE scaling.
- Never modify the supplied model directory.
- Return a clear error for corrupt, incomplete, or incompatible models.

A `laya-linux verify MODEL_PATH` command MUST perform all non-inference validation offline.

## 10. Device and precision requirements

### 10.1 Version 0.1.0

Version 0.1.0 MUST support:

- Linux x86-64.
- Python 3.11 or newer within the tested compatibility matrix.
- CPU inference.
- NVIDIA CUDA inference when the user has installed a compatible PyTorch build.

AMD ROCm is a high-priority requirement for the first stable release. It MAY be marked
experimental in 0.1.0 if real ROCm hardware validation is not yet available.

### 10.2 Selection behavior

- `device="auto"` MUST select an available supported accelerator and otherwise use CPU.
- Explicit device selection MUST either use that device or fail clearly; it MUST NOT silently move
  sensitive or expensive work to another machine.
- CPU MUST default to FP32 unless a lower precision has been proven correct and beneficial.
- GPU precision defaults MUST be based on verified device capability.
- Users MUST be able to request FP32, FP16, or BF16 where supported.
- Unsupported dtype/device combinations MUST fail with an actionable error.

## 11. Command-line requirements

The project SHOULD provide one `laya-linux` executable with these initial commands:

- `predict`: run one request from command-line arguments or local JSON files.
- `serve`: start the local/private inference server.
- `verify`: validate a local model package and checksums.
- `doctor`: report local runtime, dependency, model, and device readiness.
- `benchmark`: run a reproducible local benchmark and emit JSON.

Commands MUST work without contacting the internet. Diagnostic output MUST distinguish verified
facts from suggestions and MUST not expose secrets.

## 12. Reliability and compatibility requirements

- Inference MUST run under `torch.inference_mode()` or an equivalent no-gradient mode.
- Batch chunking MUST not change public results beyond documented numeric tolerances.
- Empty question dictionaries MUST return an empty answer dictionary without model execution.
- Repeated calls MUST not exhibit unbounded active-memory growth.
- Non-finite logits or action values MUST cause an explicit failure.
- Model initialization MUST be thread-safe at the documented concurrency level.
- Router eviction and preloading behavior MUST be deterministic.
- Server and client protocol versions MUST be negotiated or rejected clearly.

## 13. Testing requirements

The test suite MUST include:

- Unit tests using tiny randomly initialized models.
- Tokenization and prompt-construction golden tests.
- Layer-level parity tests against a trusted ModernBERT reference.
- Full-model parity tests against pinned upstream Laya code.
- Real-checkpoint integration tests for every officially supported checkpoint.
- FP32 selected-answer parity tests.
- Documented FP16 and BF16 probability-error tolerances.
- CPU tests on every pull request.
- CUDA tests when suitable hardware is available.
- ROCm tests before ROCm support is declared stable.
- Tests with network/socket access blocked.
- Checkpoint corruption, checksum, path traversal, and shape mismatch tests.
- Server authentication, body-size, queue-limit, timeout, and log-redaction tests.
- Repeated inference and memory-stability tests.
- Wheel build and clean-environment installation tests.

Correctness tests MUST be completed before performance optimization. Compilation or specialized
kernels MUST not be enabled by default until their parity and stability are demonstrated.

## 14. Benchmark requirements

Benchmarks MUST:

- Identify CPU/GPU model, memory, operating system, Python, PyTorch, driver, and backend versions.
- Record checkpoint, dtype, batch size, input length, question count, warmup count, and iterations.
- Include prompt preparation, tokenization, host/device transfer, synchronized inference,
  calibration, and result formatting in end-to-end measurements.
- Run competing configurations in fresh processes when memory or compilation state could bias the
  result.
- Report raw samples or machine-readable summaries, not only a best-case number.
- Report median and tail latency plus throughput and peak memory when available.
- Never compare results across different hardware without prominently stating the difference.

## 15. Packaging and dependency requirements

Core runtime dependencies SHOULD be limited to:

- PyTorch
- NumPy
- `safetensors`
- Hugging Face `tokenizers` Rust bindings

Server dependencies MUST be optional. A likely server stack is FastAPI, Pydantic, and Uvicorn, but
the implementation may choose alternatives if the offline and API requirements remain satisfied.

Transformers SHOULD be a development/reference dependency rather than a required inference
dependency once native ModernBERT parity is established.

GPU-specific PyTorch builds MUST NOT be hidden behind misleading Python extras. Documentation MUST
instruct users to install the correct CPU, CUDA, or ROCm PyTorch wheel from their chosen offline
wheelhouse before installing Laya-Linux.

## 16. Documentation requirements

The repository MUST eventually include:

- A concise README with embedded and private-server quick starts.
- Offline installation and model-transfer instructions.
- CPU, CUDA, and ROCm setup pages.
- Threat model and privacy statement.
- Model provenance and licensing documentation.
- API reference and example payloads.
- Benchmark methodology and published raw results.
- Troubleshooting for insufficient RAM/VRAM, unsupported devices, corrupt models, and unavailable
  servers.

Examples MUST use local filesystem model paths. Any separate acquisition documentation MUST be
clearly labeled as an online staging operation, not part of normal runtime behavior.

## 17. Version 0.1.0 acceptance criteria

Version 0.1.0 is acceptable only when all of the following are true:

1. A clean Linux environment can install the project from local wheels.
2. A local Laya checkpoint can be verified and loaded with network access blocked.
3. Embedded CPU inference passes the tiny-model and real-checkpoint parity suites.
4. CUDA inference works on at least one documented NVIDIA configuration.
5. `choice`, `score`, and `noul` output schemas match the reference behavior.
6. The CLI can predict, verify, diagnose, and benchmark without internet access.
7. The server binds only to loopback by default.
8. A lightweight client can call a server on a private address without installing PyTorch.
9. The server does not log request bodies by default and passes authentication and limit tests.
10. No core-runtime path automatically downloads a model or contacts an external service.
11. License and notice requirements are satisfied.
12. Published performance statements are accompanied by reproducible methodology and hardware
    details.

## 18. Priority order

Implementation work MUST follow this priority order unless maintainers explicitly revise it:

1. Offline and privacy guarantees.
2. Correctness and upstream parity.
3. Clear failures and model validation.
4. Embedded CPU runtime.
5. CUDA runtime.
6. Local/private server and lightweight client.
7. Packaging, documentation, and reproducible benchmarks.
8. ROCm stabilization.
9. Optional optimization backends.

