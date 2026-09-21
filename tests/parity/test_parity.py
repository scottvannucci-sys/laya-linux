"""Native implementation vs pinned references (parity).

These tests are the Phase 1 correctness gate: with identical weights, the
native encoder must match Hugging Face Transformers' ModernBERT and the native
full model must match upstream Laya's DecisionModel. References are loaded
from the pinned revisions recorded in PARITY_BASELINES.md; they are
development-only dependencies and are never imported by the runtime.
"""

import importlib.util
import sys

import pytest

torch = pytest.importorskip("torch")

from conftest import UPSTREAM_COMMON
from laya_linux.model import build_decision_model
from laya_linux.prompts import collate_items

pytestmark = pytest.mark.parity


def _load_upstream_common():
    if not UPSTREAM_COMMON.is_file():
        pytest.skip(f"upstream reference not staged: {UPSTREAM_COMMON}")
    spec = importlib.util.spec_from_file_location("laya_upstream_common", str(UPSTREAM_COMMON))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["laya_upstream_common"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def ref_pair():
    """(native model, transformers ModernBERT module) with identical weights."""
    pytest.importorskip("transformers")
    from transformers import ModernBertConfig, ModernBertModel

    from conftest import ENC_DICT

    torch.manual_seed(7)
    native = build_decision_model(ENC_DICT, {"encoder": "x", "head_layers": 2, "act_costs": {"e": 0.5}}).eval()
    cfg = ModernBertConfig(
        vocab_size=ENC_DICT["vocab_size"],
        hidden_size=ENC_DICT["hidden_size"],
        intermediate_size=ENC_DICT["intermediate_size"],
        num_hidden_layers=ENC_DICT["num_hidden_layers"],
        num_attention_heads=ENC_DICT["num_attention_heads"],
        local_attention=ENC_DICT["local_attention"],
        global_attn_every_n_layers=ENC_DICT["global_attn_every_n_layers"],
        max_position_embeddings=ENC_DICT["max_position_embeddings"],
        layer_types=ENC_DICT["layer_types"],
        pad_token_id=2,
        reference_compile=False,
    )
    hf = ModernBertModel(cfg).eval()
    hf.load_state_dict(native.encoder.state_dict())
    return native, hf


def test_encoder_state_dict_names_match_reference(ref_pair):
    """Bidirectional strict name compatibility with Transformers ModernBERT."""
    _, hf = ref_pair
    missing, unexpected = hf.load_state_dict(ref_pair[0].encoder.state_dict(), strict=True)
    assert not missing and not unexpected


def test_encoder_parity_padded_batch(ref_pair):
    native, hf = ref_pair
    b = collate_items([
        {"ids": [0, 5, 6, 7, 1, 4, 4, 4, 4, 9, 9, 1], "markers": [5, 8], "qtype": 0},
        {"ids": [0, 8, 9, 1, 10, 11, 12, 1], "markers": [4, 6], "qtype": 2},
    ], pad_id=2)
    ids, am = torch.tensor(b["input_ids"]), torch.tensor(b["attention_mask"])
    with torch.inference_mode():
        ref = hf(input_ids=ids, attention_mask=am).last_hidden_state
        got = native.encoder(ids, am)
    for row, valid in ((0, 12), (1, 8)):
        # valid positions must match; padded rows may differ by design (never
        # used as keys, markers, or pooled outputs) but must stay finite
        assert (ref[row, :valid] - got[row, :valid]).abs().max().item() < 1e-5
    assert torch.isfinite(got).all()


def test_decision_head_names_match_upstream(ref_pair):
    """Native head parameter names are exactly upstream Laya's (minus encoder)."""
    native, hf = ref_pair
    upstream = _load_upstream_common()
    up = upstream.DecisionModel(hf, head_layers=2, n_act=2).eval()
    non_enc_native = {k for k in native.state_dict() if not k.startswith("encoder.")}
    non_enc_up = {k for k in up.state_dict() if not k.startswith("encoder.")}
    assert non_enc_native == non_enc_up


def test_full_model_parity_vs_upstream(ref_pair):
    native, hf = ref_pair
    upstream = _load_upstream_common()
    up = upstream.DecisionModel(hf, head_layers=2, n_act=2).eval()
    non_enc = {k: v for k, v in native.state_dict().items() if not k.startswith("encoder.")}
    up.load_state_dict(non_enc, strict=False)

    b = collate_items([
        {"ids": [0, 5, 6, 7, 1, 4, 4, 4, 4, 9, 9, 1], "markers": [5, 8], "qtype": 0},
        {"ids": [0, 8, 9, 1, 10, 11, 12, 1], "markers": [4, 6], "qtype": 2},
    ], pad_id=2)
    ids, am = torch.tensor(b["input_ids"]), torch.tensor(b["attention_mask"])
    args = (ids, am, torch.tensor(b["marker_pos"]), torch.tensor(b["marker_mask"]), torch.tensor(b["qtype"]))
    with torch.inference_mode():
        ref_logits, ref_act = up(*args)
        got_logits, got_act = native(*args)
    assert (ref_logits - got_logits).abs().max().item() < 1e-5
    assert (ref_act - got_act).abs().max().item() < 1e-5


@pytest.mark.parametrize("qtype", [0, 1, 2], ids=["choice", "score", "noul"])
def test_full_model_parity_unpadded_per_qtype(ref_pair, qtype):
    native, hf = ref_pair
    upstream = _load_upstream_common()
    up = upstream.DecisionModel(hf, head_layers=2, n_act=2).eval()
    non_enc = {k: v for k, v in native.state_dict().items() if not k.startswith("encoder.")}
    up.load_state_dict(non_enc, strict=False)
    b = collate_items([{"ids": [0, 5, 6, 7, 1, 4, 4, 4, 4, 9, 9, 1], "markers": [5, 8], "qtype": qtype}], pad_id=2)
    ids, am = torch.tensor(b["input_ids"]), torch.tensor(b["attention_mask"])
    args = (ids, am, torch.tensor(b["marker_pos"]), torch.tensor(b["marker_mask"]), torch.tensor(b["qtype"]))
    with torch.inference_mode():
        rl, ra = up(*args)
        gl, ga = native(*args)
    assert (rl - gl).abs().max().item() < 1e-5
    assert (ra - ga).abs().max().item() < 1e-5


def test_padding_invariance(ref_pair):
    """A sequence's results are unchanged by the presence of padded batch rows."""
    native, _ = ref_pair
    solo = collate_items([{"ids": [0, 5, 6, 7, 1, 4, 4, 4, 4, 9, 9, 1], "markers": [5, 8], "qtype": 0}], pad_id=2)
    batch = collate_items([
        {"ids": [0, 5, 6, 7, 1, 4, 4, 4, 4, 9, 9, 1], "markers": [5, 8], "qtype": 0},
        {"ids": [0, 1], "markers": [1], "qtype": 2},
    ], pad_id=2)
    with torch.inference_mode():
        l1, _ = native(torch.tensor(solo["input_ids"]), torch.tensor(solo["attention_mask"]),
                       torch.tensor(solo["marker_pos"]), torch.tensor(solo["marker_mask"]), torch.tensor(solo["qtype"]))
        l2, _ = native(torch.tensor(batch["input_ids"]), torch.tensor(batch["attention_mask"]),
                       torch.tensor(batch["marker_pos"]), torch.tensor(batch["marker_mask"]), torch.tensor(batch["qtype"]))
    assert (l1[0] - l2[0]).abs().max().item() < 1e-6
