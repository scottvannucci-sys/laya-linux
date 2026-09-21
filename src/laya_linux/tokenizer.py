# Derived from Laya-MLX (Apache-2.0); see NOTICE.
"""Load the checkpoint's Rust tokenizer without importing Transformers or torch."""

from __future__ import annotations

import json
from pathlib import Path

from tokenizers import Tokenizer as Backend

from .errors import ModelIncompatibleError


class Tokenizer:
    def __init__(self, path: Path):
        path = Path(path)
        tokenizer_file = path / "tokenizer.json"
        if not tokenizer_file.is_file():
            raise ModelIncompatibleError(f"Model package is missing {tokenizer_file.name!r} under tokenizer/")
        self.backend = Backend.from_file(str(tokenizer_file))
        self.backend.no_padding()
        self.backend.no_truncation()
        config_file = path / "tokenizer_config.json"
        if not config_file.is_file():
            raise ModelIncompatibleError("Model package is missing 'tokenizer_config.json' under tokenizer/")
        config = json.loads(config_file.read_text())
        for name in ("cls_token", "sep_token", "pad_token", "mask_token"):
            value = config.get(name)
            if isinstance(value, dict):
                value = value.get("content")
            token_id = self.backend.token_to_id(value) if isinstance(value, str) else None
            if token_id is None:
                raise ModelIncompatibleError(f"Tokenizer is missing a valid {name}")
            setattr(self, name, value)
            setattr(self, name + "_id", token_id)

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict:
        return {"input_ids": self.backend.encode(text, add_special_tokens=add_special_tokens).ids}
