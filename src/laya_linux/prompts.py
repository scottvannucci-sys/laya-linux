# Derived from Laya-MLX and Laya (Apache-2.0); see NOTICE.
"""Prompt construction, truncation, and calibration math (architecture §5.4).

These functions are free of PyTorch so they can be tested cheaply and reused
by tools. The externally observable prompt behavior — serialization, option
ordering, truncation, marker placement — MUST match upstream Laya exactly.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np

QTYPES = {"choice": 0, "score": 1, "noul": 2}
QTYPE_NAMES = {v: k for k, v in QTYPES.items()}


class _TokenizerBackend(Protocol):
    """Minimal tokenizer surface used by prompt construction."""

    mask_token: str
    mask_token_id: int
    cls_token_id: int
    sep_token_id: int

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, Sequence[int]]: ...


def serialize_state(state: str | dict | list) -> str:
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False)


def render_criterion(value: Any) -> str:
    """Render one criterion value as text.

    Strings pass through; anything structured (dict, list, number) becomes compact JSON, so a
    rubric reads as JSON rather than a Python repr. Without this a dict-valued criterion
    crashed `noul` outright and leaked `{'desc': ...}` into `choice` and `score` prompts.
    """
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(", ", ": "), default=str)


def render_options(q: dict[str, Any]) -> list[str]:
    """Render option texts in label-index order. Noul is always [false, true]."""
    t, crit = q["t"], q.get("crit")
    if t == "choice":
        # only None/"" mean "no description"; 0 and False are legitimate criterion values
        return [k if v is None or v == "" else "%s: %s" % (k, render_criterion(v)) for k, v in crit.items()]
    if t == "score":
        return ["level %d: %s" % (i, render_criterion(c)) for i, c in enumerate(crit)]
    crit = crit or {}
    false_crit, true_crit = crit.get("false"), crit.get("true")
    return [
        "false: " + (render_criterion(false_crit) if false_crit not in (None, "") else "no, the statement does not hold"),
        "true: " + (render_criterion(true_crit) if true_crit not in (None, "") else "yes, the statement holds"),
    ]


def build_prefix(
    tok: _TokenizerBackend,
    q: dict[str, Any],
    head_max_len: int = 192,
    option_order: Sequence[int] | None = None,
) -> tuple[list[int], list[int]]:
    """Build the question-only prefix, before state tokens and final truncation."""
    mask_tok = tok.mask_token
    opts = render_options(q)
    order = list(option_order) if option_order is not None else list(range(len(opts)))
    ins = str(q["ins"]).replace(mask_tok, " ")
    head_ids = list(tok("%s question: %s" % (q["t"], ins), add_special_tokens=False)["input_ids"])
    opt_ids = []
    for i in order:
        opt_ids.append(
            [tok.mask_token_id]
            + list(tok(" " + opts[i].replace(mask_tok, " "), add_special_tokens=False)["input_ids"])[:48]
        )
    opt_budget = head_max_len - sum(len(o) for o in opt_ids)
    if opt_budget < 16:
        per = max(4, (head_max_len - 16) // max(1, len(opt_ids)))
        opt_ids = [o[:per] for o in opt_ids]
        opt_budget = head_max_len - sum(len(o) for o in opt_ids)
    head_ids = head_ids[: max(8, opt_budget)]
    ids = [tok.cls_token_id] + head_ids + [tok.sep_token_id]
    markers = []
    for o in opt_ids:
        markers.append(len(ids))
        ids.extend(o)
    ids.append(tok.sep_token_id)
    return ids, markers


def build_sequence(
    tok: _TokenizerBackend,
    state: str | dict | list,
    q: dict[str, Any],
    max_len: int = 512,
    head_max_len: int = 192,
    option_order: Sequence[int] | None = None,
    truncate_left: bool = False,
) -> tuple[list[int], list[int]]:
    """Format: [CLS] <type> instructions [SEP]  opt0  opt1 ... [SEP] state [SEP]."""
    ids, markers = build_prefix(tok, q, head_max_len, option_order)
    room = max(0, max_len - len(ids) - 1)
    st = list(tok(serialize_state(state).replace(tok.mask_token, " "), add_special_tokens=False)["input_ids"])
    st = st[-room:] if truncate_left else st[:room]
    ids = ids + st + [tok.sep_token_id]
    return ids[:max_len], [m for m in markers if m < max_len]


def confidence_from_probs(p: Any, k: int) -> float:
    """Normalized Shannon entropy confidence: 1 - H(p) / log(k)."""
    if k < 2:
        return 1.0
    p = np.asarray(p)[:k]
    ent = -(p * np.log(np.clip(p, 1e-12, 1.0))).sum()
    return float(np.clip(1.0 - ent / math.log(k), 0.0, 1.0))


def temp_bucket(qtype: int, k: int) -> str:
    size = "2" if k <= 2 else "3-5" if k <= 5 else "6-10" if k <= 10 else "11+"
    return "%s:%s" % (QTYPE_NAMES[int(qtype)], size)


def collate_items(items: list[dict[str, Any]], pad_id: int) -> dict[str, Any]:
    """Pad a list of prepared items into batch tensors (CPU-side, format-agnostic)."""
    import numpy as np

    if not items:
        raise ValueError("Cannot collate an empty batch")
    n, length = len(items), max(len(it["ids"]) for it in items)
    count = max(2, max(len(it["markers"]) for it in items))
    batch = {
        "input_ids": np.full((n, length), pad_id, dtype=np.int64),
        "attention_mask": np.zeros((n, length), dtype=np.int64),
        "marker_pos": np.zeros((n, count), dtype=np.int64),
        "marker_mask": np.zeros((n, count), dtype=bool),
        "qtype": np.array([it["qtype"] for it in items], dtype=np.int64),
    }
    for i, it in enumerate(items):
        ids_len, k = len(it["ids"]), len(it["markers"])
        batch["input_ids"][i, :ids_len] = it["ids"]
        batch["attention_mask"][i, :ids_len] = 1
        batch["marker_pos"][i, :k] = it["markers"]
        batch["marker_mask"][i, :k] = True
    return batch
