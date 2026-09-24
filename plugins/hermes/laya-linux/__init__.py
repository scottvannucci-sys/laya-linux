"""laya-linux Hermes plugin — registration and tool handlers.

Tools:
  laya_predict        evaluate typed questions (choice/score/noul) on a state
  laya_presets        list available question presets and their exact questions
  laya_server_status  health + models + protocol version of the server

The plugin is self-contained: if `laya_linux` is importable it is used;
otherwise a vendored stdlib-only client (client.py, vendored from
src/laya_linux/client/http.py at v0.1.0) talks to the server directly.

Settings (plugins.entries.laya-linux.settings in config.yaml; see plugin.yaml
config_schema): server_url, token_file, model, preset, timeout.
Secrets belong in the token file, never in config.yaml.
"""

import importlib.util
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_PLUGIN_DIR = Path(__file__).resolve().parent
_CTX = None  # PluginContext, captured at register() for settings access


def _vendored(name: str):
    """Import a vendored module (client/protocol/errors/presets) from this dir.

    Vendored modules use absolute sibling imports ("from errors import ..."),
    which only resolve when this plugin dir is importable. Importing via
    spec_from_file_location alone provides no package context, so put the dir
    on sys.path for the duration of the import.
    """
    import sys

    plugin_dir = str(_PLUGIN_DIR)
    spec = importlib.util.spec_from_file_location(
        f"laya_hermes_plugin._{name}", _PLUGIN_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, plugin_dir)
    try:
        spec.loader.exec_module(module)
    finally:
        try:
            sys.path.remove(plugin_dir)
        except ValueError:
            pass
    return module


def _settings() -> dict:
    """Plugin settings via ctx.get_config(); empty dict outside Hermes."""
    if _CTX is not None:
        try:
            out = {}
            for key in ("server_url", "token_file", "model", "preset", "timeout"):
                try:
                    val = _CTX.get_config(key, default=None)
                except ValueError:
                    # Reserved config root (e.g. 'model'); skip this key only.
                    logger.debug("settings key %r rejected by Hermes; skipped", key)
                    continue
                if val is not None:
                    out[key] = val
            return out
        except Exception:
            logger.debug("ctx.get_config failed; using defaults", exc_info=True)
    return {}


def _get_client():
    """Build a RemoteAgent from settings; installed package or vendored client."""
    settings = _settings()
    server_url = str(settings.get("server_url") or "http://127.0.0.1:8142")
    token_file = str(settings.get("token_file") or "") or None
    model = str(settings.get("model") or "") or None  # None = server's default alias
    timeout = float(settings.get("timeout") or 30.0)

    try:
        from laya_linux.client.http import RemoteAgent  # prefer installed package
    except ImportError:
        RemoteAgent = _vendored("client").RemoteAgent
    return RemoteAgent(server_url, token_file=token_file, model=model, timeout=timeout), settings


def _ok(payload: dict) -> str:
    return json.dumps({"success": True, **payload}, ensure_ascii=False)


def _err(code: str, message: str) -> str:
    return json.dumps({"success": False, "error_code": code, "error": message}, ensure_ascii=False)


def laya_predict(args: dict, **kwargs) -> str:
    """Evaluate typed questions on the state via the laya-linux server."""
    state = str(args.get("state", "")).strip()
    if not state:
        return _err("VALIDATION", "state is required")

    try:
        client, settings = _get_client()
    except Exception as exc:
        return _err("CONFIG", f"Could not configure the client: {exc}")

    # Questions: explicit JSON > preset name > configured/default preset.
    questions = None
    if args.get("questions"):
        try:
            questions = json.loads(args["questions"])
        except json.JSONDecodeError as exc:
            return _err("VALIDATION", f"'questions' is not valid JSON: {exc}")
    else:
        preset = str(args.get("preset") or settings.get("preset") or "triage")
        try:
            presets = _vendored("presets")
            fn = getattr(presets, f"{preset}_questions", None)
            if fn is None:
                return _err("VALIDATION",
                            f"Unknown preset '{preset}'. Known: triage, email, guard, moderation, router.")
            preset_kwargs = {}
            if args.get("preset_args"):
                try:
                    preset_kwargs = json.loads(args["preset_args"])
                except json.JSONDecodeError as exc:
                    return _err("VALIDATION", f"'preset_args' is not valid JSON: {exc}")
            questions = fn(**preset_kwargs)
        except Exception as exc:
            return _err("PRESET_ERROR", f"Preset '{preset}' failed: {exc}")

    try:
        result = client.predict(state, questions, model=args.get("model") or None)
        answers = result.get("answers", result)
        return _ok({
            "answers": answers,
            # Normalized quick-read: {qid: {type, value, confidence}} — the
            # full typed payload stays in "answers" for detailed consumption.
            "summary": {
                qid: {
                    "type": ans.get("type"),
                    "value": (ans.get("choice") if ans.get("type") == "choice"
                              else ans.get("noul") if ans.get("type") == "noul"
                              else ans.get("score")),
                    "confidence": ans.get("confidence"),
                }
                for qid, ans in answers.items() if isinstance(ans, dict)
            },
            "model": result.get("model"),
            "device": result.get("device"),
        })
    except Exception as exc:
        code = getattr(exc, "code", None) or type(exc).__name__
        return _err(str(code), f"{type(exc).__name__}: {exc}")


def laya_presets(args: dict, **kwargs) -> str:
    """List every preset with its full question set."""
    try:
        presets = _vendored("presets")
        out = {name: getattr(presets, f"{name}_questions")()
               for name in ("triage", "email", "guard", "moderation", "router")}
    except Exception as exc:
        return _err("PRESET_ERROR", str(exc))
    return _ok({"presets": out})


def laya_server_status(args: dict, **kwargs) -> str:
    """Health + models + protocol version of the configured server."""
    try:
        client, _ = _get_client()
    except Exception as exc:
        return _err("CONFIG", f"Could not configure the client: {exc}")
    try:
        health = client.health()
    except Exception as exc:
        return _err("SERVER_UNAVAILABLE", f"{type(exc).__name__}: {exc}")
    try:
        models = client.models()
    except Exception:
        models = {}  # health succeeded; models endpoint may be restricted
    return _ok({"health": health, "models": models})


def register(ctx):
    """Wire schemas to handlers. Called exactly once by Hermes at startup."""
    global _CTX
    _CTX = ctx

    from . import schemas  # sibling module in the plugin package

    ctx.register_tool(name="laya_predict", toolset="laya-linux",
                      schema=schemas.LAYA_PREDICT, handler=laya_predict)
    ctx.register_tool(name="laya_presets", toolset="laya-linux",
                      schema=schemas.LAYA_PRESETS, handler=laya_presets)
    ctx.register_tool(name="laya_server_status", toolset="laya-linux",
                      schema=schemas.LAYA_SERVER_STATUS, handler=laya_server_status)
    logger.info("laya-linux plugin registered: 3 tools")
