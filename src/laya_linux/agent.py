# Derived from Laya and Laya-MLX (Apache-2.0); see NOTICE.
"""High-level inference runtime for Laya System 1 decision models (architecture §5.2)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file

from . import checkpoints
from .devices import DevicePlan, select_device
from .errors import ModelIncompatibleError, NonFiniteOutputError, TokenBudgetExceededError
from .model import build_decision_model
from .prompts import (
    QTYPES,
    build_sequence,
    collate_items,
    confidence_from_probs,
    render_options,
    temp_bucket,
)
from .protocol import PredictionResult, Questions, State
from .tokenizer import Tokenizer

REQUIRED_WEIGHT_PREFIXES = ("encoder.", "head.", "type_emb.", "scorer.", "act_head.", "temperature")

DEFAULT_BATCH_SIZE = 16


class Agent:
    """One loaded model: local files in, typed decisions out, no network."""

    def __init__(
        self,
        model_path: str | Path,
        device: str | None = "auto",
        dtype: str = "auto",
        *,
        batch_size: int = DEFAULT_BATCH_SIZE,
        pad_to_multiple: int | None = None,
        cache_prompts: bool = False,
    ):
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if pad_to_multiple is not None and (
            not isinstance(pad_to_multiple, int) or isinstance(pad_to_multiple, bool) or pad_to_multiple < 1
        ):
            raise ValueError("pad_to_multiple must be a positive integer or None")
        self.batch_size = batch_size
        self.pad_to_multiple = pad_to_multiple
        self.model_path = model_path
        from .prepared import PrefixCache

        self._prefix_cache = PrefixCache() if cache_prompts else None

        info = checkpoints.load_checkpoint(model_path)
        self.model_dir: Path = info["dir"]
        self.cfg: dict[str, Any] = info["agent_config"]
        self.encoder_cfg_dict: dict[str, Any] = info["encoder_config"]
        self.manifest = info["manifest"]

        self.temperature = self.cfg.get("temperature", [1.0, 1.0, 1.0])
        self.temperature_by_options = self.cfg.get("temperature_by_options", {})
        if len(self.temperature) != 3 or any(
            not math.isfinite(float(t)) or float(t) <= 0
            for t in [*self.temperature, *self.temperature_by_options.values()]
        ):
            raise ModelIncompatibleError("Calibration temperatures must be finite and positive")

        self.tok = Tokenizer(self.model_dir / "tokenizer")

        self.model = build_decision_model(self.encoder_cfg_dict, self.cfg)
        weights = load_file(str(self.model_dir / "model.safetensors"))
        self._verify_compatibility(weights)
        self.model.load_state_dict(weights, strict=True)
        del weights

        plan: DevicePlan = select_device(device, dtype, checkpoint_amp_dtype=self.cfg.get("amp_dtype"))
        self.device = plan.device
        self.dtype = plan.dtype
        if self.device.type == "cpu":
            # CPU defaults to FP32 (requirements §10.2); load_state_dict already
            # cast the checkpoint's values into the model's FP32 parameters.
            self.model.to(device=self.device)
        else:
            self.model.to(device=self.device, dtype=self.dtype)
        self.model.eval()
        self.device_plan = plan

    def _verify_compatibility(self, weights: dict[str, torch.Tensor]) -> None:
        """Strict checkpoint verification before any weight is trusted (requirements §9)."""
        state_names = set(self.model.state_dict().keys())
        weight_names = set(weights.keys())
        for prefix in REQUIRED_WEIGHT_PREFIXES:
            if not any(name.startswith(prefix) for name in weight_names):
                raise ModelIncompatibleError(
                    f"Checkpoint is missing {prefix!r} parameters; "
                    f"expected a Laya decision model with encoder and decision heads"
                )
        missing = sorted(state_names - weight_names)
        if missing:
            preview = ", ".join(missing[:5])
            raise ModelIncompatibleError(
                f"Model weights incomplete: missing {len(missing)} parameter tensors (e.g. {preview})"
            )
        unexpected = sorted(weight_names - state_names)
        if unexpected:
            preview = ", ".join(unexpected[:5])
            raise ModelIncompatibleError(
                f"Checkpoint contains {len(unexpected)} parameters the architecture does not define "
                f"(e.g. {preview}); refusing to load an incompatible model"
            )
        mismatches = []
        for name, param in self.model.state_dict().items():
            supplied = weights[name]
            if tuple(supplied.shape) != tuple(param.shape):
                mismatches.append(f"{name}: expected {tuple(param.shape)}, found {tuple(supplied.shape)}")
        if mismatches:
            preview = "\n".join("  - " + m for m in mismatches[:5])
            more = f"\n  ... and {len(mismatches) - 5} more mismatched tensors." if len(mismatches) > 5 else ""
            raise ModelIncompatibleError(
                f"Model architecture mismatch:\n{preview}{more}\n"
                f"The checkpoint weights do not match the configured model architecture."
            )

    @staticmethod
    def _to_internal(qdef: Any) -> dict[str, Any]:
        """Validate a public question definition and convert it to the internal form."""
        if not isinstance(qdef, dict):
            raise ValueError("Each question must be a dictionary")
        kind = qdef.get("type")
        if kind not in QTYPES:
            raise ValueError(f"Unknown question type {kind!r}; expected 'choice', 'score', or 'noul'")
        if "instructions" not in qdef:
            raise ValueError("Question is missing instructions")
        criteria = qdef.get("criteria")
        if kind == "choice":
            if isinstance(criteria, list):
                if not all(isinstance(c, str) for c in criteria):
                    raise ValueError("Choice labels must be strings")
                if len(set(criteria)) != len(criteria):
                    raise ValueError("Choice labels must be unique")
                criteria = dict.fromkeys(criteria)
            if not isinstance(criteria, dict) or not criteria:
                raise ValueError("Choice criteria must be a nonempty dictionary or list of unique labels")
            if not all(isinstance(k, str) for k in criteria):
                raise ValueError("Choice labels must be strings")
        elif kind == "score":
            if not isinstance(criteria, list) or not criteria:
                raise ValueError("Score criteria must be a nonempty list")
        elif criteria is not None and not isinstance(criteria, dict):
            raise ValueError("Noul criteria must be a dictionary with 'false'/'true' descriptions")
        instructions = qdef["instructions"]
        if not isinstance(instructions, str):
            instructions = json.dumps(instructions)
        return {"t": kind, "ins": instructions, "crit": criteria}

    def prepare(
        self, state: State, questions: Questions
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Construct CPU-side batch items; exposed for parity tests and profiling."""
        if not isinstance(questions, dict):
            raise ValueError("questions must be a dictionary keyed by question id")
        if not questions:
            return [], []
        if self._prefix_cache is not None:
            return self._prefix_cache.prepare(self, state, questions)
        max_len = self.cfg.get("max_len", 512)
        head_max_len = self.cfg.get("head_max_len", 192)
        items, internal = [], []
        for qid, definition in questions.items():
            q = self._to_internal(definition)
            ids, markers = build_sequence(self.tok, state, q, max_len, head_max_len)
            if len(markers) != len(render_options(q)):
                raise TokenBudgetExceededError(
                    f"Question {qid!r} has too many options for the token budget"
                )
            items.append({"ids": ids, "markers": markers, "qtype": QTYPES[q["t"]]})
            internal.append(q)
        return items, internal

    def forward(self, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one prepared batch under inference mode on this agent's device."""
        tensors = {k: torch.as_tensor(v, device=self.device) for k, v in batch.items()}
        with torch.inference_mode():
            logits, act = self.model(
                tensors["input_ids"],
                tensors["attention_mask"],
                tensors["marker_pos"],
                tensors["marker_mask"],
                tensors["qtype"],
            )
        return logits, act

    def system_one(self, state: State, questions: Questions) -> PredictionResult:
        """Evaluate typed questions across the state in single forward passes."""
        items, internal = self.prepare(state, questions)
        question_ids = list(questions)
        answers: dict[str, Any] = {}
        for start in range(0, len(items), self.batch_size):
            chunk = items[start : start + self.batch_size]
            batch = collate_items(chunk, self.tok.pad_token_id)
            if self.pad_to_multiple:
                length = batch["input_ids"].shape[1]
                padded = ((length + self.pad_to_multiple - 1) // self.pad_to_multiple) * self.pad_to_multiple
                padded = min(padded, self.cfg.get("max_len", 512))
                if padded > length:
                    pad_id = self.tok.pad_token_id
                    grow_ids = np.full((batch["input_ids"].shape[0], padded - length), pad_id, dtype=np.int64)
                    grow_mask = np.zeros((batch["attention_mask"].shape[0], padded - length), dtype=np.int64)
                    batch["input_ids"] = np.concatenate([batch["input_ids"], grow_ids], axis=1)
                    batch["attention_mask"] = np.concatenate([batch["attention_mask"], grow_mask], axis=1)
            logits, act = self.forward(batch)
            logits_np, act_np = logits.float().cpu().numpy(), act.float().cpu().numpy()
            if not np.isfinite(logits_np).all() or not np.isfinite(act_np).all():
                raise NonFiniteOutputError(
                    "Model produced non-finite outputs; this indicates numerical instability"
                )
            act_np = np.exp(act_np - act_np.max(axis=-1, keepdims=True))
            act_np /= act_np.sum(axis=-1, keepdims=True)
            for row, item in enumerate(chunk):
                qid = question_ids[start + row]
                q = internal[start + row]
                k, qt = len(item["markers"]), item["qtype"]
                scale = self.temperature_by_options.get(temp_bucket(qt, k), self.temperature[qt])
                z = logits_np[row, :k] / max(1e-3, float(scale))
                p = np.exp(z - z.max())
                p /= p.sum()
                answer: dict[str, Any] = {
                    "type": q["t"],
                    "confidence": round(confidence_from_probs(p, k), 4),
                    "action": {"act_probability": round(float(act_np[row, 0]), 4)},
                }
                if q["t"] == "choice":
                    labels = list(q["crit"])
                    answer.update(
                        choice=labels[int(p.argmax())],
                        probabilities={label: round(float(v), 4) for label, v in zip(labels, p, strict=True)},
                    )
                elif q["t"] == "score":
                    answer.update(
                        score=round(float((np.arange(k) * p).sum()), 4),
                        legend={str(i): value for i, value in enumerate(q["crit"])},
                        probabilities={str(i): round(float(v), 4) for i, v in enumerate(p)},
                    )
                else:
                    answer.update(
                        noul=round(float(p[1]), 4),
                        confidence=round(max(float(p[1]), 1.0 - float(p[1])), 4),
                    )
                answers[qid] = answer
        return {
            "model": "laya-rl-agent",
            "answers": answers,
            "usage": {"input_tokens": sum(len(item["ids"]) for item in items), "output_tokens": 0},
        }

    predict = system_one


RLAgent = Agent


def load(
    model_path: str | Path,
    device: str | None = "auto",
    dtype: str = "auto",
    **kwargs: Any,
) -> Agent:
    """Load a Laya agent from a local model package directory."""
    return Agent(model_path, device=device, dtype=dtype, **kwargs)
