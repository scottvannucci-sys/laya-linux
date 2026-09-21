# Laya-Linux Architecture

## 1. Purpose and implementation rule

This document describes the intended technical design of Laya-Linux. Read
`requirements.md` first. When this document conflicts with `requirements.md`, the requirements
document wins.

The architecture is deliberately local-first. Embedded inference has no networking capability.
Private server mode is a separate composition in which both endpoints remain controlled by the
user. No component may silently substitute a cloud service.

## 2. Architectural decisions

The following decisions are currently fixed:

1. PyTorch is the first execution backend.
2. The core runtime loads local filesystem checkpoints only.
3. The core tokenizer uses the Rust-backed `tokenizers` package directly.
4. The target runtime does not require Transformers after native ModernBERT parity is complete.
5. Original Laya safetensors checkpoints are the canonical input format.
6. Laya-MLX-compatible converted checkpoints should also be supported when practical.
7. Embedded and remote agents expose the same prediction protocol.
8. The first network API is versioned JSON over HTTP, with Unix sockets and private TCP supported.
9. Loopback is the server's default bind target.
10. Correctness and offline guarantees take precedence over optimization.

## 3. System context

### 3.1 Embedded mode

```text
┌──────────────────────── Linux process ────────────────────────┐
│                                                               │
│  User application                                             │
│       │                                                       │
│       ▼                                                       │
│  laya_linux.Agent                                             │
│       │                                                       │
│       ├── prompt/token preparation                            │
│       ├── native PyTorch model                                │
│       ├── calibration/result formatting                       │
│       └── local model files                                   │
│                                                               │
└───────────────────────────────────────────────────────────────┘

No sockets, downloads, telemetry, or remote calls.
```

### 3.2 Private server mode

```text
┌──────────── constrained Linux host ────────────┐
│ Application → laya_linux.client.RemoteAgent   │
└───────────────────────┬────────────────────────┘
                        │ Unix socket, loopback, or private LAN
                        ▼
┌──────────── inference Linux host ──────────────┐
│ API boundary                                   │
│   → validation/auth/limits                     │
│   → bounded request queue                      │
│   → one resident Agent per selected device     │
│   → CPU, CUDA, or ROCm                         │
│   → local model files                          │
└────────────────────────────────────────────────┘

No public-internet dependency or cloud fallback.
```

## 4. Proposed repository layout

```text
.
├── README.md
├── requirements.md
├── architecture.md
├── PARITY_BASELINES.md
├── LICENSE
├── NOTICE
├── pyproject.toml
├── src/
│   └── laya_linux/
│       ├── __init__.py
│       ├── agent.py
│       ├── protocol.py
│       ├── model.py
│       ├── attention.py
│       ├── tokenizer.py
│       ├── prompts.py
│       ├── prepared.py
│       ├── checkpoints.py
│       ├── manifest.py
│       ├── devices.py
│       ├── results.py
│       ├── router.py
│       ├── lang.py
│       ├── presets.py
│       ├── email.py
│       ├── errors.py
│       ├── cli.py
│       ├── client/
│       │   ├── __init__.py
│       │   └── http.py
│       └── server/
│           ├── __init__.py
│           ├── app.py
│           ├── auth.py
│           ├── config.py
│           ├── queue.py
│           └── schemas.py
├── tests/
│   ├── unit/
│   ├── parity/
│   ├── integration/
│   ├── offline/
│   ├── server/
│   └── fixtures/
├── benchmarks/
│   ├── run.py
│   ├── worker.py
│   ├── report.py
│   └── results/
├── examples/
├── docs/
├── scripts/
└── containers/
```

The layout may evolve, but core inference MUST remain importable without importing server or
network-client modules. Note that `email.py` is naming inherited from Laya-MLX: it contains
email-domain question presets and performs no network operations and no email-sending
capability.

## 5. Component responsibilities

### 5.1 `protocol.py`

Define structural interfaces shared by embedded and remote execution:

```python
class DecisionAgent(Protocol):
    def predict(self, state: State, questions: Questions) -> PredictionResult: ...
    def system_one(self, state: State, questions: Questions) -> PredictionResult: ...
```

Application code should be able to accept a `DecisionAgent` without knowing whether inference is
embedded or remote.

### 5.2 `agent.py`

`Agent` owns one loaded model and is responsible for:

- Loading and validating local configuration and weights.
- Selecting device and dtype through `devices.py`.
- Preparing questions and batching them.
- Running inference under `torch.inference_mode()`.
- Applying temperature calibration.
- Formatting the stable public result schema.
- Detecting non-finite output.

`Agent` MUST NOT know how to download a model. It accepts a `Path` or path-like value. A string that
does not resolve to a local directory must fail before any network-capable library is called.

### 5.3 `model.py` and `attention.py`

These modules implement the inference-only neural architecture:

```text
token embeddings + embedding LayerNorm
    ↓
ModernBERT encoder layers
    ├── fused QKV projection
    ├── full or sliding-window attention
    ├── per-layer RoPE base
    └── gated MLP
    ↓
final encoder LayerNorm
    ↓
question-type embedding
    ↓
zero or more pre-norm Transformer decision-head layers
    ↓
marker token scoring head
    ↓
action head using pooled state and probability features
```

Implementation details that MUST match the reference:

- The first ModernBERT encoder attention block does not add an extra attention LayerNorm where the
  reference uses identity.
- Encoder MLP behavior is gated GELU.
- Full and local attention layers use their configured RoPE bases.
- Sliding attention uses the same inclusive window boundary as the reference.
- Padded tokens cannot act as keys or contribute marker outputs.
- Padded query rows must remain finite.
- Decision-head layers are pre-normalized and use ReLU in their feed-forward blocks.
- Marker positions are gathered in option order.
- Invalid marker slots receive a sufficiently negative logit before softmax.
- Probability-derived action features match the upstream order and formula.

Prefer `torch.nn.functional.scaled_dot_product_attention` so PyTorch can select an appropriate
CPU, CUDA, or ROCm kernel. Any custom attention path must have independent parity tests.

### 5.4 `tokenizer.py` and `prompts.py`

`tokenizer.py` loads `tokenizer/tokenizer.json` with the Rust tokenizer backend and validates the
CLS, SEP, PAD, and MASK special tokens.

`prompts.py` owns all externally observable prompt behavior:

- State serialization.
- Structured criterion rendering.
- Option ordering.
- Question prefix construction.
- Per-option truncation.
- State truncation.
- Marker placement.
- Token-usage counting.

These functions should remain free of PyTorch so they can be tested cheaply and reused by tools.

### 5.5 `prepared.py`

Provide a bounded, mutation-safe prefix cache. Cache keys must include every input that affects the
result. The cache stores CPU-side token data only and does not claim to reuse encoder hidden states.

### 5.6 `checkpoints.py` and `manifest.py`

Checkpoint loading is split into inspection and materialization:

1. Resolve and normalize the local path.
2. Confirm that all resolved paths remain inside the model directory.
3. Parse configuration using bounded input sizes.
4. Validate manifest version and file checksums.
5. Construct the model architecture.
6. Load tensors with `safetensors`.
7. Map known upstream or Laya-MLX parameter names into the native layout.
8. Reject duplicate, missing, unexpected, or misshaped tensors.
9. Move the model to the selected device and dtype.
10. Mark it as evaluation-only.

Do not use pickle-based model formats. Never write into the source model directory.

Proposed manifest shape:

```json
{
  "format": "laya-linux",
  "format_version": 1,
  "model_family": "laya",
  "checkpoint": "laya-typed-decisions",
  "source": {
    "project": "convaiinnovations/laya",
    "revision": "example-revision"
  },
  "license": "Apache-2.0",
  "dtype": "float16",
  "architecture": {
    "encoder": "modernbert",
    "hidden_size": 1024,
    "context_limit": 1024
  },
  "files": {
    "model.safetensors": {
      "size": 123,
      "sha256": "example"
    }
  }
}
```

The final schema must be versioned and covered by fixtures before publication.
The manifest's `license` field MUST record the checkpoint's license as
verified from its model card at packaging time; it MUST NOT be assumed to
match the runtime's license. Upstream checkpoint revisions are pinned in
`PARITY_BASELINES.md`.

### 5.7 `devices.py`

Centralize device behavior instead of scattering backend checks throughout the runtime.

Conceptual selection order for `auto`:

1. CUDA-compatible PyTorch device, including ROCm builds that expose the CUDA API.
2. CPU.

The module should return a structured description containing device, backend, selected dtype,
capabilities, and an explanation suitable for `doctor`. Do not infer ROCm solely from the device
string; inspect PyTorch build metadata.

Explicit device requests fail if unavailable. Automatic device choice may fall back locally to CPU
only when the behavior is documented and the user did not explicitly require an accelerator.

### 5.8 `router.py`

The router chooses among locally configured model directories. It supports:

- Explicit model selection.
- Language/script-based selection.
- Explicit typed-decisions task selection.
- Preloading selected models.
- Bounded least-recently-used residency.
- Attaching an already created `DecisionAgent`.
- Deterministic unload behavior.

It MUST NOT interpret a configured model name as a remote repository identifier.

### 5.9 `client/http.py`

`RemoteAgent` is a thin dependency-light implementation of `DecisionAgent`. It should not import
PyTorch. Responsibilities:

- Serialize the versioned request.
- Authenticate when configured.
- Apply connect and request deadlines.
- Enforce a maximum response size.
- Decode stable server errors.
- Never retry non-idempotent requests automatically unless a future request ID protocol makes it
  safe.
- Never redirect to another host unless explicitly enabled.
- Never fall back to a public endpoint.

Private-network address validation is a guardrail, not a replacement for host firewall and routing
policy. Unix sockets are preferred when both processes share a host.

### 5.10 `server/`

The server composition owns networking and must not contaminate the embedded core.

Request path:

```text
connection
  → content-type and body-size limit
  → authentication
  → JSON/schema validation
  → semantic question limits
  → bounded queue
  → selected resident agent
  → prediction
  → response-size check
  → redacted access log
```

The server should run one process per accelerator by default. Multiple HTTP workers are not a safe
way to increase GPU concurrency because every process may load another model copy. Use an async
front end with bounded model execution instead.

Initial scheduling can be a FIFO queue protected by a semaphore. Dynamic batching is a later
optimization and must preserve per-request question order and numerical tolerances.

## 6. Public API sketches

### 6.1 Embedded agent

```python
from pathlib import Path
import laya_linux as laya

agent = laya.load(
    Path("/opt/laya/models/laya-typed-decisions"),
    device="auto",
    dtype="auto",
    batch_size=16,
    compile=False,
    pad_to_multiple=None,
    cache_prompts=False,
)

result = agent.predict(state, questions)
```

### 6.2 Remote agent

```python
from laya_linux.client import RemoteAgent

agent = RemoteAgent(
    "http://192.168.1.50:8142",
    token_file="/run/secrets/laya-token",
    timeout=10.0,
)

result = agent.predict(state, questions)
```

### 6.3 Predict request

```json
{
  "protocol_version": 1,
  "model": "typed-decisions",
  "state": {
    "message": "I was charged twice."
  },
  "questions": {
    "refund": {
      "type": "noul",
      "instructions": "Does the customer ask for a refund?"
    }
  }
}
```

The server owns the mapping from a public model alias to an absolute local model directory. Clients
must never submit arbitrary server filesystem paths.

## 7. Stable error model

Define library exceptions and equivalent API error codes:

- `MODEL_NOT_FOUND`
- `MODEL_INCOMPLETE`
- `MODEL_CHECKSUM_FAILED`
- `MODEL_INCOMPATIBLE`
- `INVALID_QUESTION`
- `TOKEN_BUDGET_EXCEEDED`
- `DEVICE_UNAVAILABLE`
- `DTYPE_UNSUPPORTED`
- `NON_FINITE_OUTPUT`
- `UNAUTHORIZED`
- `REQUEST_TOO_LARGE`
- `QUEUE_FULL`
- `SERVER_BUSY`
- `PROTOCOL_UNSUPPORTED`
- `NETWORK_DISABLED`

Errors should include a safe human-readable explanation and a stable code. They must not include
request bodies, tokens, or sensitive environment data.

Protocol versions are non-negative integers owned by `protocol.py` and pinned by
`PARITY_BASELINES.md`-style fixtures rather than by prose. The current protocol version is 1. A
version bump requires a compatibility note in the changelog and explicit negotiation or rejection
behavior on both client and server sides, per requirements §12.

## 8. Configuration model

Configuration precedence should be:

1. Explicit function argument or CLI flag.
2. A user-specified local configuration file.
3. Environment variable intended for non-sensitive operational settings.
4. Secure defaults.

Secrets should use protected files or dedicated secret injection. Do not echo secret values in
`doctor`, logs, or exceptions.

Server configuration should include:

- Model alias to local path mappings.
- Device and dtype.
- Bind address or Unix socket.
- Authentication mode and token-file path.
- TLS certificate paths when enabled.
- Maximum request bytes.
- Maximum questions and options.
- Queue capacity and execution timeout.
- Logging level and explicit sensitive-debug switch.

## 9. Offline enforcement strategy

Offline behavior must be enforced by construction and tests:

- Keep `huggingface_hub`, HTTP clients, and server frameworks out of core import paths.
- Resolve model arguments as local paths before importing model code.
- Do not call APIs with implicit remote resolution such as `from_pretrained` on untrusted strings.
- Load tokenizer and configuration files directly.
- Add tests that monkeypatch or deny `socket.socket` while importing, loading, and predicting.
- Run an integration job in a container with networking disabled.
- Scan core dependencies and document any known networking behavior.
- Keep optional acquisition utilities in a separately installable namespace or distribution.

Environment variables such as `HF_HUB_OFFLINE` may be defense in depth, but they are not the
primary guarantee. The core runtime should not need Hugging Face Hub code at all.

## 10. Numerical parity strategy

Use a progression of increasingly expensive gates:

1. Pure prompt/token golden fixtures.
2. Attention-mask and RoPE tests.
3. ModernBERT layer comparison against Transformers using small random configurations.
4. Decision-head comparison against pinned upstream Laya.
5. End-to-end tiny-model comparison.
6. Full-checkpoint FP32 comparison.
7. FP16/BF16 selected-answer and probability-drift characterization.
8. Repeated inference and memory-stability testing.

Selected-answer agreement and probability tolerance are separate metrics. A match in selected
labels does not establish calibration parity. Store raw validation results with hardware and
software metadata.

## 11. Test organization

### Unit tests

- Configuration parsing.
- Input validation.
- Prompt construction and truncation.
- Tokenizer special-token discovery.
- Attention masks.
- Device selection.
- Manifest parsing and path containment.
- Result formatting.
- Router policy.

### Parity tests

- Transformers ModernBERT versus native encoder.
- Upstream Laya decision head versus native decision head.
- Upstream Laya result schema versus native agent.
- Laya-MLX checkpoint name conversion when supported.

### Offline tests

- Import with sockets blocked.
- Load and predict with sockets blocked.
- Missing local path never starts a download.
- Server-disabled core installation contains no mandatory HTTP framework.

### Server tests

- Loopback default.
- Authentication success and failure.
- Private bind requires explicit configuration.
- Body and response limits.
- Queue saturation.
- Timeout behavior.
- No request-body logging.
- Client/server protocol mismatch.
- Lightweight client installation without PyTorch.

### Hardware tests

- CPU on normal CI.
- CUDA on a documented runner or manually reproducible host.
- ROCm before declaring stable support.
- Each hardware run produces a metadata-rich JSON artifact.

## 12. Dependency groups

Suggested dependency boundaries:

```text
core:
  torch
  numpy
  safetensors
  tokenizers

server extra:
  fastapi
  pydantic
  uvicorn

client extra:
  a small HTTP client only if the standard library is insufficient

dev/reference extra:
  pytest
  pytest-cov
  ruff
  build
  transformers
```

Avoid declaring CUDA or ROCm as package extras. Those builds are selected when creating the local
wheelhouse, before installing this package.

## 13. Build and release design

Each release should produce:

- Source distribution.
- Platform-appropriate pure-Python project wheel where possible; PyTorch remains external.
- Offline wheelhouse instructions.
- SBOM or dependency inventory.
- Checksums for release artifacts.
- CPU test results.
- Separately identified GPU validation results.
- Model-manifest tooling version compatibility notes.

Do not include model weights in release artifacts unless a future separately licensed model bundle
is deliberately created.

## 14. Implementation sequence for agents

An implementation agent should follow these phases and should not optimize ahead of the gates.

### Phase 0: repository foundation

1. Add Apache-2.0 license and accurate `NOTICE` material.
2. Create `pyproject.toml`, `src/` package layout, linting, test configuration, and GitHub
   Actions Ubuntu CPU CI (see requirements §11.5).
3. Add error types and public protocol definitions.
4. Add a minimal README stating that no working release exists yet if appropriate.

Exit gate: package builds and imports in an offline clean environment.

### Phase 1: CPU reference runtime

1. Port tokenizer and prompt construction.
2. Implement native PyTorch model architecture.
3. Implement strict local checkpoint loading.
4. Implement `Agent.prepare`, batching, forward execution, calibration, and formatting.
5. Port tiny-model tests and reference parity tests.

Exit gate: FP32 CPU parity passes on tiny and real checkpoints with sockets blocked.

### Phase 2: model packaging and CLI

1. Finalize the manifest schema.
2. Implement checksum verification and offline conversion/packaging from local source directories.
3. Implement `predict`, `verify`, `doctor`, and `benchmark` CLI commands.
4. Add wheelhouse and air-gapped installation documentation.

Exit gate: a fresh offline Linux environment can install, verify, and predict.

### Phase 3: CUDA

1. Add device and precision policies.
2. Validate FP32, FP16, and BF16 where supported.
3. Add memory and repeated-call tests.
4. Add reproducible end-to-end benchmarks.
5. Evaluate `torch.compile` only as an opt-in feature.

Exit gate: documented CUDA configuration passes parity and stability criteria.

### Phase 4: private server and client

1. Add optional server schemas and API.
2. Add bounded queueing and one-model-per-device lifecycle.
3. Add authentication, limits, timeouts, and redacted logging.
4. Add a PyTorch-free `RemoteAgent` client.
5. Add Unix-socket and explicit private-LAN deployment examples.

Exit gate: client/server compatibility, privacy, overload, and offline-network tests pass.

### Phase 5: ROCm and optimization

1. Validate the same PyTorch code path on real ROCm hardware.
2. Document supported hardware/software combinations.
3. Investigate dynamic batching, compilation, quantization, ONNX Runtime, or TensorRT one at a time.

Exit gate: no optimization is enabled by default without parity, stability, and benchmark evidence.

## 15. Agent working rules

Any autonomous agent working on this repository MUST:

- Read both this file and `requirements.md` before editing.
- Inspect existing changes and preserve unrelated user work.
- Use the pinned source implementations as behavioral references, not assume remembered behavior.
- Keep model weights, credentials, generated benchmark bulk data, and local caches out of Git.
- Avoid adding dependencies that are not justified by a requirement.
- Add or update tests with every externally observable behavior change.
- Run the smallest relevant tests first, then the broader suite.
- Never claim GPU support without a real hardware result.
- Never add implicit model downloads or cloud fallback for convenience.
- Preserve attribution when adapting code.
- Record unresolved compatibility questions in the repository instead of silently guessing.

## 16. Known risks

### Numerical drift

Different attention kernels and low-precision formats can change probabilities near decision
boundaries. Mitigate with FP32 references, raw error distributions, and selected-answer fixtures.

### Dynamic shape compilation

Question count, option count, and token length vary. Compilation may repeatedly specialize and
become slower than eager execution. Keep it opt-in until shape-bucket benchmarks justify it.

### GPU memory duplication

Multiple server processes can each load a 322M-421M parameter model. Use one execution owner per
device and bounded concurrency.

### Packaging GPU runtimes

CPU, CUDA, and ROCm PyTorch wheels differ. Keep backend installation explicit and provide offline
wheelhouse recipes for each validated stack.

### High-cardinality choices

Option descriptions share a fixed question-head token budget. Many options may become truncated
and indistinguishable. Preserve explicit errors where options do not fit and document hierarchical
choice strategies rather than hiding the limitation.

### Private LAN exposure

A private address is not automatically trustworthy. Default to loopback, support authentication,
recommend firewall rules, and document TLS/mTLS for sensitive networks.

## 17. Definition of architectural success

The architecture succeeds when the same application can switch between:

```python
agent = laya_linux.load("/local/model")
```

and:

```python
agent = RemoteAgent("http://private-inference-host:8142")
```

without changing the prediction inputs or outputs, while the embedded path is demonstrably
network-free and the server path remains entirely on infrastructure controlled by the user.
