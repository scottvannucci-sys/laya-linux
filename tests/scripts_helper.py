"""Test helper: import a script module from scripts/ by name."""

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = REPO / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name.removesuffix(".py")] = module
    spec.loader.exec_module(module)
    return module
