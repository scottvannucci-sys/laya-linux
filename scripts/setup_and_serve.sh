#!/usr/bin/env bash
# One-command local setup for laya-linux, through the whole Quick Start:
#   dependencies -> model staging -> integrity verify -> server READY for a harness.
#
# Usage:
#   bash scripts/setup_and_serve.sh                 # setup, then serve in the foreground
#   bash scripts/setup_and_serve.sh --detach        # setup, start server in background, health-check
#   bash scripts/setup_and_serve.sh --no-serve      # setup only (deps + model + verify)
#
# Options:
#   --port N          TCP port (default 8142; auto-picks a free port if busy)
#   --unix-socket P   serve on a Unix-domain socket instead of TCP (Linux)
#   --no-token        skip bearer-token generation (loopback-only service)
#   --model-path P    model package location (default models/laya-typed-decisions)
#   --alias NAME      model alias to serve (default "typed")
#
# Environment overrides:
#   PYTHON_BIN=python3.12   interpreter to build the venv from (>= 3.11)
#   TORCH_INDEX=...         torch wheel line (default CPU; use .../cu128 for NVIDIA)
#   SKIP_APT=1              never invoke apt
#
# Idempotent: safe to rerun; existing venv, staged model, and token are reused.
# On success (with --detach or --no-serve) connection facts are written to
# .laya/ready.json for harness tooling to consume.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

MODEL_PATH="models/laya-typed-decisions"
ALIAS="typed"
PORT=""
UNIX_SOCKET=""
USE_TOKEN=1
DETACH=0
SERVE=1
HF_BASE="https://huggingface.co/convaiinnovations/laya-typed-decisions/resolve/f9ab0b228f0fc0f14d873dbc99038f135c2da1b2"
WEIGHTS_BYTES=842609220
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cpu}"

while [ $# -gt 0 ]; do
  case "$1" in
    --port) PORT="$2"; shift 2 ;;
    --unix-socket) UNIX_SOCKET="$2"; shift 2 ;;
    --no-token) USE_TOKEN=0; shift ;;
    --model-path) MODEL_PATH="$2"; shift 2 ;;
    --alias) ALIAS="$2"; shift 2 ;;
    --detach) DETACH=1; shift ;;
    --no-serve) SERVE=0; shift ;;
    *) echo "unknown option: $1"; exit 1 ;;
  esac
done

say() { printf '\n=== %s ===\n' "$*"; }

# ---- 1. Python + venv + dependencies ---------------------------------------
say "1. Dependencies"
APT=""
if [ "${SKIP_APT:-0}" != "1" ] && command -v apt-get >/dev/null 2>&1; then
  if command -v sudo >/dev/null 2>&1; then APT="sudo apt-get"; else APT="apt-get"; fi
fi
PY_OK=""
for CAND in "${PYTHON_BIN:-}" python3.12 python3.11 python3 python py; do
  [ -z "$CAND" ] && continue
  command -v "$CAND" >/dev/null 2>&1 || continue
  "$CAND" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null || continue
  PY_OK="$CAND"; break
done
if [ -z "$PY_OK" ]; then
  [ -n "$APT" ] && $APT install -y -qq python3.11 python3.11-venv && PY_OK=python3.11
fi
[ -n "$PY_OK" ] || { echo "ERROR: no Python >= 3.11 found (install python3.11 + venv, or set PYTHON_BIN)."; exit 1; }
"$PY_OK" -m venv --help >/dev/null 2>&1 || { echo "ERROR: venv module missing (sudo apt install python3-venv)."; exit 1; }

VENV="$REPO/.venv"
if [ -x "$VENV/Scripts/python.exe" ]; then PY="$VENV/Scripts/python.exe"      # Windows
elif [ -x "$VENV/bin/python" ]; then             PY="$VENV/bin/python"        # Linux/macOS
else
  "$PY_OK" -m venv "$VENV"
  if [ -x "$VENV/Scripts/python.exe" ]; then PY="$VENV/Scripts/python.exe"; else PY="$VENV/bin/python"; fi
fi
echo "interpreter: $PY ($("$PY" --version))"

for MOD in torch numpy safetensors tokenizers; do
  if "$PY" -c "import $MOD" 2>/dev/null; then
    echo "already installed: $MOD"
  else
    echo "installing: $MOD"
    if [ "$MOD" = "torch" ]; then
      "$PY" -m pip install --quiet torch --index-url "$TORCH_INDEX"
    else
      "$PY" -m pip install --quiet "$MOD"
    fi
  fi
done
if "$PY" -c "import laya_linux" 2>/dev/null; then
  echo "already installed: laya-linux (editable)"
else
  "$PY" -m pip install --quiet --no-deps -e "$REPO"
fi

# ---- 2. Model staging -------------------------------------------------------
say "2. Model staging ($MODEL_PATH)"
mkdir -p "$MODEL_PATH/tokenizer" "$MODEL_PATH/encoder"
fetch() {
  if [ -s "$2" ]; then echo "already staged: $2"; return 0; fi
  echo "downloading: $1"
  curl -L --fail --retry 3 --progress-bar "$HF_BASE/$1" -o "$2"
}
fetch "rl_agent_config.json"            "$MODEL_PATH/rl_agent_config.json"
fetch "encoder/config.json"             "$MODEL_PATH/encoder/config.json"
fetch "tokenizer/tokenizer.json"        "$MODEL_PATH/tokenizer/tokenizer.json"
fetch "tokenizer/tokenizer_config.json" "$MODEL_PATH/tokenizer/tokenizer_config.json"
if [ ! -s "$MODEL_PATH/model.safetensors" ]; then
  echo "downloading model.safetensors (~840 MB)..."
  curl -L --fail --retry 3 --progress-bar "$HF_BASE/model.safetensors" -o "$MODEL_PATH/model.safetensors"
fi
ACTUAL=$(stat -c%s "$MODEL_PATH/model.safetensors" 2>/dev/null || wc -c < "$MODEL_PATH/model.safetensors" | tr -d ' ')
[ "$ACTUAL" = "$WEIGHTS_BYTES" ] || {
  echo "ERROR: model.safetensors is $ACTUAL bytes; expected $WEIGHTS_BYTES (truncated/corrupt - delete it and rerun)."
  exit 1
}
echo "checkpoint size verified: $ACTUAL bytes"

# ---- 3. Offline integrity verification --------------------------------------
say "3. Verify model package"
"$PY" -m laya_linux.cli verify "$MODEL_PATH" --json | "$PY" -c "import json,sys; r=json.load(sys.stdin); assert r['ok'], r; print('verify:', r['report']['weights'])"

# ---- 4. Auth token -----------------------------------------------------------
say "4. Auth token"
LAYA_DIR="$REPO/.laya"
mkdir -p "$LAYA_DIR"
chmod 700 "$LAYA_DIR" 2>/dev/null || true
TOKEN_FILE="$LAYA_DIR/token"
if [ "$USE_TOKEN" = "1" ]; then
  if [ ! -s "$TOKEN_FILE" ]; then
    if command -v openssl >/dev/null 2>&1; then
      umask 077 && printf 'LAYA_TOKEN=%s\n' "$(openssl rand -hex 32)" > "$TOKEN_FILE"
    else
      umask 077 && printf 'LAYA_TOKEN=%s\n' "$("$PY" -c 'import secrets; print(secrets.token_hex(32))')" > "$TOKEN_FILE"
    fi
  fi
  echo "token file: $TOKEN_FILE (kept secret; harness reads this path)"
else
  rm -f "$TOKEN_FILE"; TOKEN_FILE=""; echo "auth disabled (loopback-only service)"
fi

# ---- 5. Endpoint -------------------------------------------------------------
say "5. Endpoint"
if [ -n "$UNIX_SOCKET" ]; then
  BIND_ARGS=(--unix-socket "$UNIX_SOCKET")
  BIND_DESC="unix:$UNIX_SOCKET"
  ADDRESS="$UNIX_SOCKET"
else
  if [ -z "$PORT" ]; then
    PORT=8142
  fi
  PORT_BUSY=$("$PY" - "$PORT" <<'PY'
import socket, sys
s = socket.socket()
try:
    s.bind(("127.0.0.1", int(sys.argv[1]))); print("free")
except OSError:
    print("busy")
finally:
    s.close()
PY
)
  if [ "$PORT_BUSY" = "busy" ]; then
    PORT=$("$PY" -c "import socket; s=socket.socket(); s.bind(('127.0.0.1',0)); print(s.getsockname()[1]); s.close()")
    echo "port 8142 busy; serving on $PORT instead"
  fi
  BIND_ARGS=(--host 127.0.0.1 --port "$PORT")
  BIND_DESC="http://127.0.0.1:$PORT"
  ADDRESS="http://127.0.0.1:$PORT"
fi
BIND_ARGS+=(--model "$ALIAS=$MODEL_PATH")
echo "bind:        $BIND_DESC"
echo "model alias: $ALIAS -> $MODEL_PATH"

[ -n "$TOKEN_FILE" ] && BIND_ARGS+=(--token-file "$TOKEN_FILE")

# Ready-report for harness tooling (token path, not token value).
"$PY" - "$ADDRESS" "$ALIAS" "$TOKEN_FILE" "$MODEL_PATH" <<'PY'
import json, sys
address, alias, token_file, model_path = sys.argv[1:5]
report = {"address": address, "alias": alias, "model_path": model_path,
          "token_file": token_file or None, "protocol_version": 1,
          "client_class": "laya_linux.client.http.RemoteAgent"}
with open(".laya/ready.json", "w") as f:
    json.dump(report, f, indent=2)
print("ready report: .laya/ready.json")
PY

# ---- Harness hookup instructions (real values) ------------------------------
ADDR_SHELL_SAFE="$ADDRESS"
if [ -n "$UNIX_SOCKET" ]; then
  CLIENT_ADDR="$UNIX_SOCKET"
else
  CLIENT_ADDR="$ADDRESS"
fi
cat <<EOF

=============================================================================
 laya-linux is READY. Hook your agent harness in with:

   # (on the same machine or from an app that can reach $( [ -n "$UNIX_SOCKET" ] && echo "this socket" || echo "$ADDRESS" ))
   from laya_linux.client.http import RemoteAgent
   agent = RemoteAgent("$CLIENT_ADDR"$( [ -n "$TOKEN_FILE" ] && printf ',\n                   token_file="%s"' "$TOKEN_FILE"))
   result = agent.predict(state, questions)   # same interface as embedded

   The client needs NO PyTorch and NO model weights.
   Full harness integration guide: docs/harness-integration.md
   Connection facts for tooling:  .laya/ready.json
=============================================================================
EOF

# ---- 6. Serve ----------------------------------------------------------------
if [ "$SERVE" = "0" ]; then
  say "Setup complete (--no-serve). Start the server with:"
  echo "  $PY -m laya_linux.cli serve ${BIND_ARGS[*]}"
  exit 0
fi

if [ "$DETACH" = "1" ]; then
  nohup "$PY" -m laya_linux.cli serve "${BIND_ARGS[@]}" > "$LAYA_DIR/serve.log" 2>&1 &
  SERVER_PID=$!
  echo "$SERVER_PID" > "$LAYA_DIR/serve.pid"
  echo "server starting in background (pid $SERVER_PID, log: $LAYA_DIR/serve.log)..."
  for i in $(seq 1 60); do
    sleep 1
    # A 401 means the server is up and auth is working; any /v1/health HTTP
    # response proves liveness.
    HEALTH=$("$PY" - "$ADDRESS" "$TOKEN_FILE" <<'PY'
import json, sys, urllib.request, urllib.error
url = sys.argv[1] + "/v1/health"
headers = {}
token_file = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else None
if token_file:
    raw = open(token_file).read().strip()
    token = raw.split("=", 1)[1].strip() if "=" in raw and "\n" not in raw else raw
    headers["Authorization"] = "Bearer " + token
request = urllib.request.Request(url, headers=headers)
try:
    with urllib.request.urlopen(request, timeout=2) as r:
        print(json.load(r)["status"])
except urllib.error.HTTPError as e:
    print("up" if e.code in (401, 403) else f"http-{e.code}")
except Exception:
    print("waiting")
PY
)
    if [ "$HEALTH" = "ok" ] || [ "$HEALTH" = "up" ]; then
      echo "HEALTH CHECK: ok"
      echo "stop it later with:  kill \$(cat $LAYA_DIR/serve.pid)"
      exit 0
    fi
  done
  echo "ERROR: server did not become healthy; see $LAYA_DIR/serve.log" >&2
  exit 1
fi

say "Serving (Ctrl-C to stop)"
exec "$PY" -m laya_linux.cli serve "${BIND_ARGS[@]}"
