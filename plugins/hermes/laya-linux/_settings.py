"""Per-call plugin settings from Hermes' config namespace.

Imported lazily by tools.py-style handlers so the plugin can register
even when Hermes' config layer is unavailable (e.g. unit tests).
"""

import json


def _settings() -> dict:
    """Resolve plugin settings, tolerating absence of the Hermes host."""
    try:
        from hermes_constants import get_hermes_home

        base = get_hermes_home()
        cfg_path = base / "config.yaml"
        if cfg_path.exists():
            try:
                import yaml

                cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
                entry = ((cfg.get("plugins") or {}).get("entries") or {}).get("laya-linux") or {}
                return dict(entry.get("settings") or {})
            except Exception:
                return {}
    except Exception:
        pass
    return {}
