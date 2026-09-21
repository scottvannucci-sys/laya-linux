"""Full-scale parity gate: real checkpoint vs transformers + upstream laya (manual script)."""
import importlib.util
import sys, time

sys.path.insert(0, "C:/Users/scott/Sources/laya-linux/src")
sys.path.insert(0, "C:/Users/scott/Sources/laya-linux-refs/laya")
import torch

PKG = "C:/Users/scott/Sources/laya-linux/models/laya-typed-decisions"

from laya_linux.model import build_decision_model
from laya_linux.prompts import collate_items
from laya_linux.tokenizer import Tokenizer
from laya_linux.prompts import build_sequence
import json
from pathlib import Path

pkg = Path(PKG)
cfg = json.loads((pkg / "rl_agent_config.json").read_text())
enc_dict = json.loads((pkg / "encoder" / "config.json").read_text())
native = build_decision_model(enc_dict, cfg).eval()
from safetensors.torch import load_file

weights = load_file(str(pkg / "model.safetensors"))
native.load_state_dict(weights, strict=True)
print("native model loaded from real checkpoint (strict)")

# Build a real tokenized batch
tok = Tokenizer(pkg / "tokenizer")
state = "I was charged twice on my last invoice and I need this fixed today or I am cancelling my account."
qs = [
    {"t": "choice", "ins": "What does the customer want in `message`?",
     "crit": {"refund": "money returned", "technical_help": "a bug", "billing_question": "invoice question",
              "information": "general info", "cancellation": "cancel or downgrade", "other": "none fits"}},
    {"t": "noul", "ins": "Does `message` communicate time pressure or a deadline?"},
    {"t": "score", "ins": "How frustrated?", "crit": ["calm", "concerned", "annoyed", "angry"]},
]
items = []
for q in qs:
    ids, markers = build_sequence(tok, {"message": state}, q, cfg.get("max_len", 1024), cfg.get("head_max_len", 256))
    from laya_linux.prompts import QTYPES

    items.append({"ids": ids, "markers": markers, "qtype": QTYPES[q["t"]]})
b = collate_items(items, tok.pad_token_id)
ids_t = torch.tensor(b["input_ids"])
am_t = torch.tensor(b["attention_mask"])
args = (ids_t, am_t, torch.tensor(b["marker_pos"]), torch.tensor(b["marker_mask"]), torch.tensor(b["qtype"]))
print("batch:", ids_t.shape, "| non-pad tokens:", int(am_t.sum()))

# transformers reference with the checkpoint's encoder weights
t0 = time.time()
from transformers import ModernBertConfig, ModernBertModel

hfcfg = ModernBertConfig.from_pretrained(str(pkg / "encoder"), reference_compile=False)
hf = ModernBertModel(hfcfg).eval()
hf.load_state_dict({k[len("encoder."):]: v for k, v in weights.items() if k.startswith("encoder.")})
print(f"transformers reference loaded in {time.time()-t0:.1f}s")

with torch.inference_mode():
    ref = hf(input_ids=ids_t, attention_mask=am_t).last_hidden_state
    got = native.encoder(ids_t, am_t)
valid = am_t.bool()
enc_diff = (ref[valid] - got[valid]).abs().max().item()
print(f"ENCODER PARITY (valid positions, {int(valid.sum())} tokens): {enc_diff:.3e}")

# upstream laya DecisionModel reference
spec = importlib.util.spec_from_file_location(
    "laya_upstream_common", "C:/Users/scott/Sources/laya-linux-refs/laya/laya/common.py")
upstream = importlib.util.module_from_spec(spec)
sys.modules["laya_upstream_common"] = upstream
spec.loader.exec_module(upstream)
up = upstream.DecisionModel(hf, head_layers=cfg["head_layers"], n_act=len(cfg.get("act_costs", {})) + 1).eval()
head_weights = {k: v for k, v in weights.items() if not k.startswith("encoder.")}
missing, unexpected = up.load_state_dict(head_weights, strict=False)
assert all(k.startswith("encoder.") for k in missing) and not unexpected
with torch.inference_mode():
    rl, ra = up(*args)
    gl, ga = native(*args)
dl = (rl - gl).abs().max().item()
da = (ra - ga).abs().max().item()
print(f"FULL-MODEL PARITY: logits {dl:.3e} | act {da:.3e}")

# selected answers must agree exactly
import numpy as np

gl_np, ga_np = gl.float().numpy(), torch.softmax(ga.float(), -1).numpy()
rl_np, ra_np = rl.float().numpy(), torch.softmax(ra.float(), -1).numpy()
temp = cfg.get("temperature", [1, 1, 1])
for r, item in enumerate(items):
    k = len(item["markers"])
    z_r, z_g = rl_np[r, :k] / temp[0], gl_np[r, :k] / temp[0]
    p_r, p_g = np.exp(z_r - z_r.max()), np.exp(z_g - z_g.max())
    p_r /= p_r.sum(); p_g /= p_g.sum()
    assert p_r.argmax() == p_g.argmax()
    max_p_diff = np.abs(p_r - p_g).max()
    print(f"  row {r} (qtype {item['qtype']}): argmax agree, max prob diff {max_p_diff:.2e}")
print("FULL-SCALE REAL-CHECKPOINT PARITY GATE PASSED")
