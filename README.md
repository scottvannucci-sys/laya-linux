# laya-linux

Local-first inference runtime for [Laya](https://github.com/NandhaKishorM/laya)
typed-decision models on Linux. It evaluates constrained questions about a
supplied state in a single bidirectional model forward pass and returns
calibrated probabilities without generating text — entirely on your hardware,
with no cloud calls, telemetry, or downloads.

**Status: v0.1.0 feature-complete.** The FP32 CPU runtime matches Hugging Face
transformers' ModernBERT and upstream Laya's `DecisionModel` to **0.0 max
difference**; CUDA is validated on two documented configurations (RTX 5070 Ti
sm_120, NVIDIA GB10 sm_121 — see [`PARITY_BASELINES.md`](PARITY_BASELINES.md)
and `benchmarks/`). `laya-linux serve` exposes the private JSON API
(loopback/Unix-socket/private-LAN with auth; [`docs/server.md`](docs/server.md)),
and the PyTorch-free `RemoteAgent` client implements the same prediction
interface over the standard library only.

## Quick Start

### 1. Install

```bash
git clone https://github.com/scottvannucci-sys/laya-linux.git && cd laya-linux
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or a CUDA/ROCm wheel
pip install .
laya-linux doctor        # reports verified device/dtype facts, never guesses
```

(PyPI publication comes with the first tagged release; until then install from
a clone or a built wheel. Air-gapped machines: build a wheelhouse on a staging
box first — [`docs/offline-installation.md`](docs/offline-installation.md).)

### 2. Stage a model (one-time, explicit — the runtime never downloads)

```bash
mkdir -p models/laya-typed-decisions/tokenizer models/laya-typed-decisions/encoder
base="https://huggingface.co/convaiinnovations/laya-typed-decisions/resolve/f9ab0b228f0fc0f14d873dbc99038f135c2da1b2"
curl -L "$base/model.safetensors"          -o models/laya-typed-decisions/model.safetensors
curl -L "$base/rl_agent_config.json"       -o models/laya-typed-decisions/rl_agent_config.json
curl -L "$base/encoder/config.json"        -o models/laya-typed-decisions/encoder/config.json
curl -L "$base/tokenizer/tokenizer.json"   -o models/laya-typed-decisions/tokenizer/tokenizer.json
curl -L "$base/tokenizer/tokenizer_config.json" -o models/laya-typed-decisions/tokenizer/tokenizer_config.json

laya-linux verify models/laya-typed-decisions   # full offline validation
```

For air-gapped transfer with checksums, use
[`laya_linux.packaging.package_model`](docs/offline-installation.md#3-transfer-a-model-package).

### 3. Predict

```python
import laya_linux as laya

agent = laya.load("models/laya-typed-decisions")        # device="auto", CPU defaults to FP32
result = agent.predict(
    {"message": "I was charged twice and need this fixed today or I am cancelling."},
    {
        "intent": {
            "type": "choice",
            "instructions": "What does the customer want in `message`?",
            "criteria": {
                "refund": "money returned or a duplicate charge reversed",
                "technical_help": "a bug, outage or integration problem",
                "billing_question": "a question about an invoice, plan or payment method",
                "cancellation": "wants to cancel or downgrade",
                "other": "none of the other options fits",
            },
        },
        "is_urgent": {"type": "noul", "instructions": "Does `message` communicate time pressure?"},
        "frustration": {
            "type": "score",
            "instructions": "How frustrated does the customer sound in `message`?",
            "criteria": ["calm and neutral", "concerned but civil", "clearly annoyed", "very angry"],
        },
    },
)

print(result["answers"]["intent"]["choice"])        # e.g. "billing_question"
print(result["answers"]["is_urgent"]["noul"])       # P(true), 4-decimal rounding
print(result["answers"]["frustration"]["score"])    # expected score over the rubric
print(result["usage"])                              # {"input_tokens": N, "output_tokens": 0}
```

Three question types: `choice` (argmax + full distribution), `score` (expected
score over ordered rubric levels), `noul` (P(true) for a proposition). Ready-made
presets — `triage_questions()`, `email_questions()`, `guard_questions()`,
`moderation_questions()` — come from `laya_linux.presets`.

CLI equivalents:

```bash
laya-linux predict models/laya-typed-decisions --state "I was charged twice." --preset triage
laya-linux benchmark models/laya-typed-decisions --state "..." --preset triage --iterations 20
```

### 4. Serving other processes (optional)

Same host — Unix socket (preferred; no port exposed):

```bash
laya-linux serve --model typed=models/laya-typed-decisions --unix-socket /run/laya/laya.sock
```

Private LAN (auth required for non-loopback binds):

```bash
umask 077 && printf 'LAYA_TOKEN=%s\n' "$(openssl rand -hex 32)" > /etc/laya/token
laya-linux serve --model typed=models/laya-typed-decisions --host 192.168.1.50 --token-file /etc/laya/token
```

Constrained clients then talk to it **without PyTorch installed**:

```python
from laya_linux.client.http import RemoteAgent

agent = RemoteAgent("http://192.168.1.50:8142", token_file="/run/secrets/laya-token")
result = agent.predict(state, questions)     # identical interface and results
```

Full deployment guidance (firewall, TLS, limits):
[`docs/server.md`](docs/server.md). One-command local setup through model
staging, verification, and a READY server:
[`scripts/setup_and_serve.sh`](scripts/setup_and_serve.sh).

## Hooking into an agent harness (Hermes and friends)

`laya-linux` was designed for autonomous-agent integration: both deployment
modes satisfy one structural protocol (`laya_linux.protocol.DecisionAgent`), so
harness tools can accept either without knowing which is live. A one-command
local setup that ends with the server READY for harness connections:

```bash
bash scripts/setup_and_serve.sh          # deps -> model -> verify -> serving
#   or, backgrounded with a health check:
bash scripts/setup_and_serve.sh --detach
```

Detailed integration — tool wrapper, JSON schemas, Hermes/MCP specifics — lives
in [`docs/harness-integration.md`](docs/harness-integration.md). The short
version:

```python
from laya_linux.protocol import DecisionAgent, PROTOCOL_VERSION
```

### Embedded tool (in-process, one-liner MCP-style server)

The fastest harness hookup: expose `Agent.predict` as a tool in the agent's
process. Because embedded inference performs **zero network operations**, this
is the mode to use when the harness's privacy policy forbids local sockets.

```python
# hermes_tool.py — register with your harness's tool loader
from pathlib import Path
import laya_linux as laya

_agent = laya.load(Path("/opt/laya/models/laya-typed-decisions"))  # loads once

def typed_decision(state: str | dict | list, questions: dict) -> dict:
    """Evaluate calibrated typed questions (choice/score/noul) about a state.

    Args:
        state: conversation turns, a document, or a dict of fields to judge.
        questions: {question_id: {"type": "choice"|"score"|"noul",
                                  "instructions": "...", "criteria": ...}}
    Returns:
        {"answers": {qid: {choice|score|noul, probabilities, confidence,
                            action.act_probability}}, "usage": {...}}
    """
    return _agent.predict(state, questions)
```

Prompt the model in the system prompt with the question schema you registered,
and the agent fills `questions` from its own reasoning — the runtime returns
calibrated numbers instead of generated text. Question presets
(`triage_questions()` etc.) are stable tool payloads you can register verbatim.

### Harness on a different machine (client mode)

If the agent's sandbox cannot hold a 421M-parameter model (or several apps share
one inference host), run the server on the inference box and give the harness a
`RemoteAgent` — it implements the identical `DecisionAgent` protocol with the
standard library only:

```python
# same tool body as above, one import changed
from laya_linux.client.http import RemoteAgent

_agent = RemoteAgent("http://192.168.1.50:8142", token_file="/run/secrets/laya-token")
```

Harness-specific notes:

- **Hermes (Hermes Agent):** drop the embedded snippet into a skill or plugin
  tool; `typed_decision` is a pure function over JSON-able inputs/outputs, so
  it needs no shell access and no network permissions. For the client mode,
  point the token file at the harness's secret-injection path (never a
  command-line flag — process listings would expose it).
- **MCP-style harnesses:** wrap `typed_decision` as a single tool with
  `state` + `questions` JSON-schema fields; `PROTOCOL_VERSION` is exposed for
  capability negotiation.
- **Concurrency:** a single `Agent` instance is safe to call from multiple
  threads (tested in `tests/integration/test_agent.py`); server mode bounds
  concurrency via its queue instead.
- **Privacy:** embedded mode performs no sockets at all (enforced by tests);
  client mode talks only to the address you configure and refuses public
  endpoints by default.

### Question schema (what the model accepts)

| Field | `choice` | `score` | `noul` |
|---|---|---|---|
| `type` | `"choice"` | `"score"` | `"noul"` |
| `instructions` | the question text | the rubric question | the proposition |
| `criteria` | `{label: description}` or `[labels]` | `[level0, level1, ...]` | optional `{"false": ..., "true": ...}` |

Labels stay in insertion order in the response's `probabilities`; descriptions
are plain strings (structured values are JSON-rendered).

## Documents

| Document | Role |
|---|---|
| [`requirements.md`](requirements.md) | Normative product specification (MUST/SHOULD requirements, v0.1.0 acceptance criteria) |
| [`architecture.md`](architecture.md) | Technical design: repository layout, components, parity strategy, implementation phases |
| [`PARITY_BASELINES.md`](PARITY_BASELINES.md) | Machine-readable pins for upstream code revisions, checkpoint revisions, and tolerance results |
| [`benchmarks/README.md`](benchmarks/README.md) | Published-benchmark conventions, artifact index, decision-quality evals (routing baseline), and reproduction instructions |
| [`docs/server.md`](docs/server.md) | Private-server deployment: loopback, Unix sockets, private LAN, auth, TLS |
| [`docs/harness-integration.md`](docs/harness-integration.md) | Hooking laya-linux into an agent harness: tool wrapper, schemas, Hermes/MCP specifics |
| [`docs/offline-installation.md`](docs/offline-installation.md) | Air-gapped wheelhouse install and model transfer |
| [`docs/cpu-setup.md`](docs/cpu-setup.md) / [`docs/cuda-setup.md`](docs/cuda-setup.md) | Per-backend setup and documented hardware configurations |
| [`docs/release.md`](docs/release.md) | Release checklist: SBOM, clean-install gate, checksums, evidence |

`requirements.md` wins on any conflict.

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
