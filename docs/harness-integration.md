# Hooking laya-linux into an agent harness

This guide covers everything after `scripts/setup_and_serve.sh` has printed
`laya-linux is READY`. Connection facts (address, alias, token-file path) are
written to `.laya/ready.json` by the setup script — read that file from your
integration code instead of hardcoding.

## The integration contract

Both deployment modes satisfy one structural protocol:

```python
from laya_linux.protocol import DecisionAgent, PROTOCOL_VERSION

class DecisionAgent(Protocol):
    def predict(self, state, questions) -> dict: ...
    def system_one(self, state, questions) -> dict: ...   # alias of predict
```

Anything that accepts a `DecisionAgent` works unchanged whether the model is
in-process (`laya_linux.Agent`) or across a socket
(`laya_linux.client.http.RemoteAgent`). Harness tools should be written against
the protocol, not either concrete class.

## Choosing a mode

| Situation | Mode | Privacy property |
|---|---|---|
| Harness and model can share a process; policy forbids local sockets | Embedded `Agent` | performs **zero** socket operations (enforced by tests) |
| Several apps / sandboxes share one inference host | Server + `RemoteAgent` | talks only to the configured address; refuses public endpoints |
| Constrained sandbox (no PyTorch, no weights) | `RemoteAgent` | stdlib client only |

## Tool wrapper (the whole integration)

Harness tools are usually one function with a JSON schema. This is the entire
integration surface — inputs and outputs are JSON-able:

```python
# laya_tool.py
import json
from pathlib import Path

def _make_agent():
    """Build once; both modes satisfy the same protocol."""
    # Embedded (in this process, no sockets):
    # import laya_linux
    # return laya_linux.load("/opt/laya/models/laya-typed-decisions")
    # Remote (model lives on the inference host):
    from laya_linux.client.http import RemoteAgent

    ready = json.loads(Path(".laya/ready.json").read_text())
    return RemoteAgent(ready["address"], token_file=ready["token_file"])

AGENT = _make_agent()

def typed_decision(state, questions: dict) -> str:
    """Calibrated typed decisions about a state (choice / score / noul).

    Args:
        state: JSON string or object - conversation turns, a document, or
            fields to judge (the model reads the values, not the keys).
        questions: object keyed by question id; each value is
            {"type": "choice"|"score"|"noul", "instructions": str,
             "criteria": ...}. Prefer one of the ready-made presets
            (triage/email/guard/moderation/router_questions()).
    Returns:
        JSON string: {"answers": {qid: {...}}, "usage": {...}} - probabilities
        are 4-decimal rounded; output_tokens is always 0 (no text generation).
    """
    if isinstance(state, str):
        payload_state = state
    else:
        payload_state = state
    result = AGENT.predict(payload_state, questions)
    return json.dumps(result)
```

Register `typed_decision` with your harness, and put the question schema in the
tool description so the agent constructs `questions` itself. Presets are stable
objects you can also expose as separate no-argument tools.

## Tool schema (what to declare to the harness)

```json
{
  "name": "typed_decision",
  "description": "Runs calibrated typed questions (choice/score/noul) about a
                  state using a local Laya model. Returns per-question
                  probabilities, expected scores, confidence, and an action
                  probability. Use for classification, rubric scoring, and
                  boolean judgments instead of generating text.",
  "parameters": {
    "type": "object",
    "properties": {
      "state": {"type": ["string", "object", "array"],
                "description": "The content to judge: conversation turns, a document, or fields"},
      "questions": {"type": "object",
                    "description": "Map of question_id -> {type, instructions, criteria}"}
    },
    "required": ["state", "questions"]
  }
}
```

### Question types

| `type` | `criteria` | Returns |
|---|---|---|
| `choice` | `{label: description}` or `[label, ...]` | `choice` (argmax label), full `probabilities` in insertion order |
| `score` | `[level0, level1, ...]` (ordered rubric) | `score` = Σ i·p_i, `probabilities` per level, `legend` |
| `noul` | optional `{"false": desc, "true": desc}` | `noul` = P(true) |

Every answer also carries `confidence` (normalized entropy for choice/score,
max-P for noul) and `action.act_probability` (should the agent act on this).

## Prompt guidance for the calling agent

The model does not see your prompt — the *questions* carry all semantics. Tell
the calling agent:

- Write `instructions` as a self-contained question ("What does the customer
  want in `message`?") — reference state fields by backticked name.
- Use `choice` when the labels are known; use `noul` for yes/no propositions;
  use `score` for graded rubrics with ordered levels.
- Treat sub-0.55 probabilities and `action.act_probability` below ~0.5 as "not
  confident enough to act" — the model is calibrated, so these numbers are
  meaningful thresholds.

## Hermes Agent specifics

1. Save the tool wrapper as a skill or plugin tool file; `typed_decision` needs
   no shell and no network permissions in embedded mode.
2. Point `token_file` at the harness's secret-injection path (e.g.
   `/run/secrets/...`) — never pass tokens on a command line; process listings
   would expose them.
3. The server writes no request bodies to logs, and `/v1/system` redacts paths
   and secrets, so tool traffic is safe to leave in the harness's own logs.
4. If the harness runs tools in subprocesses, prefer client mode with the
   server as a system service (the setup script's `--detach` writes
   `.laya/serve.pid` for lifecycle management).

## MCP-style harnesses

Wrap `typed_decision` as a single tool with the schema above. Expose
`laya_linux.PROTOCOL_VERSION` in your server/capability handshake if your
framework negotiates tool versions. For multiple model aliases, call
`agent.models()` to enumerate what the server can serve and pass
`model=<alias>` per request.

## Operational notes

- **Thread safety:** one `Agent`/`RemoteAgent` instance is safe to call from
  multiple threads; results are deterministic for identical inputs.
- **First-call latency:** the server loads the model on the first predict
  (~1.5 s CPU); pass `--preload` to the serve command to pay it at startup.
- **Failure modes surface as typed errors** with stable codes:
  `UNAUTHORIZED`, `QUEUE_FULL`, `SERVER_BUSY`, `PROTOCOL_UNSUPPORTED`,
  `REQUEST_TOO_LARGE`. Catch `laya_linux.errors.LayaError` and branch on
  `.code` — the codes are contract, not prose.
- **Stopping the server:** `kill $(cat .laya/serve.pid)`; logs in
  `.laya/serve.log` contain method/path/status only — never request bodies.
