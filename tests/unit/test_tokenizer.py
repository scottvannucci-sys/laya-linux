"""Tokenizer package loading (unit)."""

import json

import pytest

from laya_linux.errors import ModelIncompatibleError
from laya_linux.tokenizer import Tokenizer


def test_special_tokens_discovered(tiny_pkg):
    tok = Tokenizer(tiny_pkg / "tokenizer")
    assert tok.cls_token == "<|cls|>"
    assert tok.cls_token_id == 0
    assert tok.sep_token_id == 1
    assert tok.pad_token_id == 2
    assert tok.mask_token_id == 3


def test_missing_tokenizer_file(tmp_path):
    d = tmp_path / "tokenizer"
    d.mkdir()
    with pytest.raises(ModelIncompatibleError):
        Tokenizer(d)


def test_missing_special_token(tmp_path, tiny_pkg):
    import shutil

    d = tmp_path / "broken-tok"
    shutil.copytree(tiny_pkg / "tokenizer", d)
    cfg = json.loads((d / "tokenizer_config.json").read_text())
    del cfg["mask_token"]
    (d / "tokenizer_config.json").write_text(json.dumps(cfg))
    with pytest.raises(ModelIncompatibleError):
        Tokenizer(d)


def test_encode_no_special_tokens_by_default(tiny_pkg):
    tok = Tokenizer(tiny_pkg / "tokenizer")
    ids = tok("tok1 tok2")["input_ids"]
    assert tok.cls_token_id not in ids and tok.sep_token_id not in ids
