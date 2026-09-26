# Private server deployment

The `laya-linux serve` command exposes a versioned JSON HTTP API for constrained
clients on infrastructure you control. Defaults are safe: loopback only, no public
endpoint discovery, no cloud fallback, request bodies never logged.

## API

| Endpoint | Method | Purpose |
|---|---|---|
| `/v1/predict` | POST | Same state/questions as the embedded API; identical result shape |
| `/v1/health` | GET | Liveness + protocol version |
| `/v1/models` | GET | Configured aliases and load state |
| `/v1/system` | GET | Redacted config (bind, auth mode, limits) — never paths or secrets |

All errors are stable machine-readable codes (`UNAUTHORIZED`, `QUEUE_FULL`,
`REQUEST_TOO_LARGE`, `PROTOCOL_UNSUPPORTED`, ...).

## Example payloads

Request:

```json
{
  "protocol_version": 1,
  "model": "typed",
  "state": {"message": "I was charged twice."},
  "questions": {
    "refund": {"type": "noul", "instructions": "Does the customer ask for money back?"}
  }
}
```

Response: the same JSON the embedded `Agent.predict` returns.

## Loopback (default — one host, one user)

```bash
laya-linux serve --model typed=/opt/laya/models/laya-typed-decisions
```

Binds `127.0.0.1:8142`. Only local processes can connect. The embedded API is
preferable when the application and model share a process; use this mode to keep
model memory out of several local apps at once.

## Same host, several apps: Unix-domain socket

```bash
laya-linux serve --model typed=/opt/laya/models/laya-typed-decisions \
    --unix-socket /run/laya/laya.sock
```

Unix sockets are preferred over TCP when both processes share a host: the
filesystem enforces who may connect (set the socket directory's permissions, e.g.
a group that only your service accounts belong to). No port is exposed.

```python
from laya_linux.client.http import RemoteAgent

agent = RemoteAgent("/run/laya/laya.sock", timeout=30)
result = agent.predict(state, questions)   # same interface as the embedded Agent
```

## Separate inference machine: private LAN

```bash
# 1. Create a token file (protected location, readable by the server account only)
umask 077 && printf 'LAYA_TOKEN=%s\n' "$(openssl rand -hex 32)" > /etc/laya/token

# 2. Bind the private interface — non-loopback requires a token file
laya-linux serve --model typed=/opt/laya/models/laya-typed-decisions \
    --host 192.168.1.50 --port 8142 \
    --token-file /etc/laya/token
```

Client side:

```python
from laya_linux.client.http import RemoteAgent

agent = RemoteAgent(
    "http://192.168.1.50:8142",
    token_file="/run/secrets/laya-token",   # protected file, not a command line
    timeout=30,
)
```

Required hardening for LAN service (a private address is not automatically
trustworthy — requirements §8.1):

- **Firewall:** allow TCP 8142 only from the client subnets you name; drop
  everything else to that host.
- **TLS:** terminate traffic in a tunnel or place a TLS-terminating reverse proxy
  in front when the path between hosts leaves a physically trusted network.
  The server is a plain HTTP/1.1 + JSON protocol by design; TLS belongs to the
  transport layer (mTLS pairs naturally with the bearer token for client
  certificates).
- **Rotation:** the token is read once at startup; rotate by restarting with a
  new token file.

## Resource behavior (requirements §8.4)

- One resident model copy per alias regardless of request concurrency — the
  async front end serializes execution through a bounded queue instead of
  duplicating the 421M-parameter model per worker.
- `--queue-capacity` bounds queued requests; overflow gets `QUEUE_FULL`
  immediately instead of unbounded memory growth.
- `--execution-timeout` bounds each prediction; expired requests get
  `SERVER_BUSY` (the in-flight call finishes and its result is discarded).
  The default (60 s) is comfortably above the ~100 ms GPU path but real-world
  batch callers have hit it on ~2k-char states: long states tokenize into
  many sequences (one per question), and the deadline covers the whole
  request. Batch callers should pace client-side (`--sleep` in
  `scripts/build_routing_eval.py`, plus its `--retry-errors`) or raise
  `--execution-timeout`; single interactive calls are unaffected.
- The server is `concurrency: 1` by design (one model copy, serialized
  queue). A client firing requests back-to-back will drive queue depth up and
  its own requests past the deadline — pace or batch.
- `--preload` loads models at startup so no request pays the first-load cost.

## Constrained clients

The client (`laya_linux.client.http.RemoteAgent`) uses only the Python standard
library: **no PyTorch, no model weights** on the caller's machine — enforced by a
test that imports and uses the client with `import torch` blocked
(`tests/integration/test_client_no_torch.py`).
