# laya-linux plugin for Hermes

Typed-decision inference (choice / score / calibrated boolean) from a local,
offline-first [laya-linux](https://github.com/scottvannucci-sys/laya-linux)
server — as an importable Hermes plugin, no hand-written integration needed.

## Install

1. Run the laya-linux setup (deps, model, verify, READY server):

   ```bash
   bash scripts/setup_and_serve.sh --detach     # from a laya-linux clone
   ```

2. Install and enable this plugin:

   ```bash
   cp -r plugins/hermes/laya-linux "$HERMES_HOME/plugins/"   # e.g. ~/AppData/Local/hermes/plugins on Windows
   hermes plugins enable laya-linux
   ```

3. Point it at the server (connection facts are printed by setup_and_serve.sh
   and saved to `.laya/ready.json`):

   ```bash
   hermes config set plugins.entries.laya-linux.settings.server_url "http://127.0.0.1:8142"
   hermes config set plugins.entries.laya-linux.settings.token_file "C:/path/to/laya-linux/.laya/token"
   ```

   `token_file` is the secret-injection path — the token itself is never
   placed in config.yaml, argv, or logs.

4. Validate and restart:

   ```bash
   hermes plugins doctor "$HERMES_HOME/plugins/laya-linux" --ci
   ```

   Then start a new Hermes session (`/reset`, or relaunch `hermes`) — plugin
   tools and their settings load at session start.

## Server on a separate machine (LAN)

`setup_and_serve.sh` binds loopback, which is only reachable from the same
host. To serve other machines on your LAN, start the server bound to all
interfaces — a non-loopback bind requires a token file, which the script
already writes to `.laya/token`:

```bash
# on the server machine, from a laya-linux clone
laya-linux serve --host 0.0.0.0 --port 8142 \
    --model typed=models/laya-typed-decisions \
    --token-file .laya/token
```

On the Hermes machine, copy the token file (never the token value into
config) and point the plugin at the server:

```bash
mkdir -p ~/.laya && chmod 700 ~/.laya
scp server:/path/to/laya-linux/.laya/token ~/.laya/token && chmod 600 ~/.laya/token
hermes config set plugins.entries.laya-linux.settings.server_url "http://<server-ip>:8142"
hermes config set plugins.entries.laya-linux.settings.token_file "$HOME/.laya/token"
```

Notes:

- `.laya/token` is dotenv-format (`LAYA_TOKEN=...`). The client parses it;
  raw `curl` needs the value only (`cut -d= -f2`).
- The token is read once at server startup — restart the server after
  regenerating the file.
- The server is plain HTTP + bearer token; see `docs/server.md` for the
  hardening checklist (firewall, TLS termination) beyond a trusted LAN.

## Tools

| Tool | Purpose |
|---|---|
| `laya_predict` | Evaluate typed questions on a state (preset name, or custom questions JSON) |
| `laya_presets` | List the five presets (triage, email, guard, moderation, router) and their exact questions |
| `laya_server_status` | Server health, protocol version, loaded models |

`laya_predict` returns `summary` — `{qid: {type, value, confidence}}` — plus
the full typed payload (`probabilities`, `legend`, `action`) in `answers`.
Errors come back typed (`SERVER_UNAVAILABLE`, `UNAUTHORIZED`,
`VALIDATION`, ...) with `success: false`, never as raised exceptions.

## Self-contained client

If `laya_linux` is importable in Hermes' venv, the installed client is used;
otherwise the vendored stdlib-only copy (`client.py`, `protocol.py`,
`errors.py` — vendored from src/laya_linux at v0.1.0) is loaded. No PyTorch,
no model weights, and no dependencies beyond the Python standard library in
either path. The client enforces the same guardrails as upstream: no
redirects, loopback/private-address default, bounded responses, and
dotenv-style token files (`LAYA_TOKEN=...`).

## Updating the vendored client

`client.py` / `protocol.py` / `errors.py` are vendored copies. After changing
the upstream client, re-copy them, revert the two package-relative imports in
`client.py` (`..errors` -> `errors`, `..protocol` -> `protocol`), and re-run
`hermes plugins doctor --ci`. Keep vendoring in lockstep with releases.
