# Derived from Laya and Laya-MLX (Apache-2.0); see NOTICE.
"""Native PyTorch ModernBERT encoder and Laya decision heads (inference only).

Architecture follows Laya's DecisionModel and the ModernBERT implementation in
Hugging Face Transformers (architecture §5.3). Parameter names intentionally
match the upstream checkpoint layout — ``encoder.*`` uses Transformers
ModernBERT naming and the decision head uses ``nn.TransformerEncoderLayer``
naming — so original checkpoints load with ``load_state_dict(strict=True)``
and no name mapping.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .errors import ModelIncompatibleError

# Decision-head features: padded marker slots receive this logit before softmax,
# exactly as upstream Laya.
NEG_LOGIT = -1e4


@dataclass
class EncoderConfig:
    """Configuration for the ModernBERT encoder, validated like the reference."""

    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    model_type: str = "modernbert"
    norm_eps: float = 1e-5
    norm_bias: bool = False
    attention_bias: bool = False
    mlp_bias: bool = False
    hidden_activation: str = "gelu"
    local_attention: int = 128
    global_attn_every_n_layers: int = 3
    global_rope_theta: float = 160000.0
    local_rope_theta: float = 10000.0
    max_position_embeddings: int = 8192
    layer_types: list[str] | None = None
    rope_parameters: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> EncoderConfig:
        names = {f.name for f in fields(cls)}
        cfg = cls(**{k: v for k, v in value.items() if k in names})
        if cfg.model_type != "modernbert":
            raise ModelIncompatibleError(f"Unsupported encoder: {cfg.model_type!r}; expected 'modernbert'")
        if cfg.hidden_activation != "gelu":
            raise ModelIncompatibleError(f"Unsupported encoder activation: {cfg.hidden_activation!r}")
        if cfg.hidden_size % cfg.num_attention_heads:
            raise ModelIncompatibleError("hidden_size must be divisible by num_attention_heads")
        if cfg.layer_types is None:
            cfg.layer_types = [
                "full_attention" if i % cfg.global_attn_every_n_layers == 0 else "sliding_attention"
                for i in range(cfg.num_hidden_layers)
            ]
        if len(cfg.layer_types) != cfg.num_hidden_layers or set(cfg.layer_types) - {
            "full_attention",
            "sliding_attention",
        }:
            raise ModelIncompatibleError("Invalid ModernBERT layer_types")
        for kind in set(cfg.layer_types):
            params = (cfg.rope_parameters or {}).get(kind, {})
            if params.get("rope_type", "default") != "default":
                raise ModelIncompatibleError("Only default (unscaled) ModernBERT RoPE is supported")
        return cfg

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    def rope_base(self, kind: str) -> float:
        fallback = self.global_rope_theta if kind == "full_attention" else self.local_rope_theta
        return float((self.rope_parameters or {}).get(kind, {}).get("rope_theta", fallback))


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, base: float) -> torch.Tensor:
    """Apply rotary position embeddings in the Transformers/ModernBERT style.

    x: [batch, heads, seq, head_dim]. Uses float32 for the frequencies (as the
    reference does) cast to the input dtype for the elementwise product.
    """
    head_dim = x.size(-1)
    seq = x.size(-2)
    inv_freq = 1.0 / (
        base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=x.device) / head_dim)
    )
    positions = torch.arange(seq, dtype=torch.float32, device=x.device)
    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().to(dtype=x.dtype)[None, None, :, :]
    sin = emb.sin().to(dtype=x.dtype)[None, None, :, :]
    return (x * cos) + (_rotate_half(x) * sin)


def attention_masks(attention_mask: torch.Tensor, window: int) -> dict[str, torch.Tensor]:
    """Additive float attention masks for full and sliding layers.

    Padded queries keep access to valid keys so no softmax row is fully masked
    (architecture §5.3: padded query rows must remain finite). Padded tokens
    are never usable as keys, so valid-token results are unchanged.
    """
    valid = attention_mask.to(dtype=torch.bool)
    b, length = valid.shape
    positions = torch.arange(length, device=valid.device)
    dist = (positions[:, None] - positions[None, :]).abs()
    key_valid = valid[:, None, None, :]          # [B,1,1,L]: padded keys are never usable
    padded_query_access = ~valid[:, None, :, None]  # [B,1,L,1]: padded queries stay finite
    neg = torch.finfo(torch.float32).min
    full = torch.zeros((b, 1, length, length), dtype=torch.float32, device=valid.device)
    full = full.masked_fill(~key_valid, neg)
    local_window = (dist <= window // 2)[None, None]  # [1,1,L,L]: inclusive boundary
    local_allowed = (local_window | padded_query_access) & key_valid
    local = torch.zeros_like(full).masked_fill(~local_allowed, neg)
    return {"full_attention": full, "sliding_attention": local}


class Embeddings(nn.Module):
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.tok_embeddings = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.norm_eps, bias=cfg.norm_bias)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.norm(self.tok_embeddings(input_ids))


class EncoderAttention(nn.Module):
    def __init__(self, cfg: EncoderConfig, kind: str):
        super().__init__()
        self.num_heads = cfg.num_attention_heads
        self.head_dim = cfg.head_dim
        self.rope_base = cfg.rope_base(kind)
        self.Wqkv = nn.Linear(cfg.hidden_size, 3 * cfg.hidden_size, bias=cfg.attention_bias)
        self.Wo = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=cfg.attention_bias)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b, length, _ = x.shape
        qkv = self.Wqkv(x).reshape(b, length, 3, self.num_heads, self.head_dim)
        q, k, v = (qkv[:, :, i].transpose(1, 2) for i in range(3))
        q = apply_rope(q, self.rope_base)
        k = apply_rope(k, self.rope_base)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=self.head_dim**-0.5)
        return self.Wo(out.transpose(1, 2).reshape(b, length, -1))


class EncoderMLP(nn.Module):
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.Wi = nn.Linear(cfg.hidden_size, 2 * cfg.intermediate_size, bias=cfg.mlp_bias)
        self.Wo = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=cfg.mlp_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = self.Wi(x).chunk(2, dim=-1)
        return self.Wo(torch.nn.functional.gelu(value) * gate)


class EncoderLayer(nn.Module):
    def __init__(self, cfg: EncoderConfig, index: int):
        super().__init__()
        self.attention_type = cfg.layer_types[index]
        # The reference uses identity for the first layer's attention norm.
        self.attn_norm: nn.Module = (
            nn.Identity()
            if index == 0
            else nn.LayerNorm(cfg.hidden_size, eps=cfg.norm_eps, bias=cfg.norm_bias)
        )
        self.attn = EncoderAttention(cfg, self.attention_type)
        self.mlp_norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.norm_eps, bias=cfg.norm_bias)
        self.mlp = EncoderMLP(cfg)

    def forward(self, x: torch.Tensor, mask: dict[str, torch.Tensor]) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), mask[self.attention_type])
        return x + self.mlp(self.mlp_norm(x))


class ModernBert(nn.Module):
    """Inference-only ModernBERT encoder with checkpoint parameter names."""

    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.config = cfg
        self.embeddings = Embeddings(cfg)
        self.layers = nn.ModuleList(EncoderLayer(cfg, i) for i in range(cfg.num_hidden_layers))
        self.final_norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.norm_eps, bias=cfg.norm_bias)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        x = self.embeddings(input_ids)
        masks = attention_masks(attention_mask, self.config.local_attention)
        for layer in self.layers:
            x = layer(x, masks)
        return self.final_norm(x)


class DecisionHead(nn.Module):
    """Pre-norm transformer decision head matching nn.TransformerEncoderLayer naming."""

    def __init__(self, dims: int, count: int):
        super().__init__()
        if count <= 0:
            self.layers = nn.ModuleList()
            return
        nhead = max(1, dims // 64)
        # Inference runtime: dropout is disabled (upstream trains with 0.1 but
        # checkpoints carry no dropout state and evaluation disables it).
        layer = nn.TransformerEncoderLayer(
            dims, nhead, 4 * dims, dropout=0.0, batch_first=True, norm_first=True
        )
        self.layers = nn.TransformerEncoder(layer, count, enable_nested_tensor=False).layers

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=key_padding_mask)
        return x


class DecisionModel(nn.Module):
    """Bidirectional ModernBERT encoder + typed decision head (Laya architecture)."""

    def __init__(self, encoder_config: EncoderConfig, head_layers: int = 2, n_act: int = 2):
        super().__init__()
        self.encoder = ModernBert(encoder_config)
        d = encoder_config.hidden_size
        self.head = DecisionHead(d, head_layers)
        self.type_emb = nn.Embedding(3, d)
        self.scorer = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1)
        )
        self.act_head = nn.Sequential(nn.Linear(d + 4, 256), nn.GELU(), nn.Linear(256, n_act))
        # Checkpoint buffer; public calibration uses the JSON config temperatures.
        self.register_buffer("temperature", torch.ones(3))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        marker_pos: torch.Tensor,
        marker_mask: torch.Tensor,
        qtype: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(input_ids, attention_mask)
        h = h + self.type_emb(qtype)[:, None, :]
        if len(self.head.layers) > 0:
            h = self.head(h, ~attention_mask.bool())
        idx = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
        m = torch.gather(h, 1, idx)
        logits = self.scorer(m).squeeze(-1).float()
        logits = logits.masked_fill(~marker_mask, NEG_LOGIT)

        p = torch.softmax(logits.detach(), -1)
        k = marker_mask.sum(-1).clamp(min=2).float()
        ent = -(p * torch.log(p.clamp_min(1e-9))).sum(-1) / torch.log(k)
        top2 = p.topk(2, -1).values
        feats = torch.stack([top2[:, 0], top2[:, 0] - top2[:, 1], ent, k / 255.0], -1)
        pooled = h[:, 0].float()
        act_logits = self.act_head(torch.cat([pooled, feats], -1))
        return logits, act_logits


def build_decision_model(
    encoder_cfg_dict: dict[str, Any], agent_cfg: dict[str, Any]
) -> DecisionModel:
    """Construct the model from checkpoint configuration dictionaries."""
    enc = EncoderConfig.from_dict(encoder_cfg_dict)
    head_layers = agent_cfg.get("head_layers", 2)
    if not isinstance(head_layers, int) or head_layers < 0:
        raise ModelIncompatibleError("head_layers must be a non-negative integer")
    n_act = len(agent_cfg.get("act_costs", {})) + 1
    return DecisionModel(enc, head_layers=head_layers, n_act=n_act)


def expected_parameter_names(head_layers: int = 2) -> set[str]:
    """Parameter/buffer names the architecture requires from a checkpoint."""
    import itertools

    cfg = EncoderConfig(
        vocab_size=8,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        layer_types=["full_attention", "sliding_attention"],
    )
    model = DecisionModel(cfg, head_layers=head_layers, n_act=2)
    names = {n for n, _ in itertools.chain(model.named_parameters(), model.named_buffers())}
    del model
    return names
