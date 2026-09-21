# Parity baselines

This file is the machine-readable record of the upstream revisions that
correctness work is measured against. `requirements.md` §3 and
`architecture.md` §10 describe the parity strategy; this file pins the exact
targets. The parity test suite MUST read its pins from this file rather than
from prose in other documents. Re-pinning is a diff to this file plus a
passing parity run — never a silent edit.

## Source-code baselines

| Role | Repository | Revision | Notes |
|---|---|---|---|
| Initial-investigation Laya | <https://github.com/NandhaKishorM/laya> | `42626c348753fbb17572a813127df2278a1ec527` | Reference used during specification |
| Parity-reference Laya | <https://github.com/NandhaKishorM/laya> | `6a5819129eb220570792e417e49723d697efd76f` | Revision identified by the Laya-MLX parity reference |
| Laya-MLX | <https://github.com/mizorewww/laya-mlx> | `fc1df62828a3fedf4d8229fdac1cbd85f1cdf337` | Apple-MLX port; source of NOTICE attribution |

`convaiinnovations/laya` identifiers refer to Hugging Face model
repositories, not GitHub repositories; Convai Innovations has no GitHub
organization.

## Checkpoint baselines

Checkpoints are Hugging Face model repositories. The runtime never downloads
them; acquisition is a user-invoked staging operation, and the staged copy is
validated by `laya-linux verify` before any parity run.

| Checkpoint | Hugging Face revision | Package layout | Role |
|---|---|---|---|
| [convaiinnovations/laya-typed-decisions](https://huggingface.co/convaiinnovations/laya-typed-decisions) | `f9ab0b228f0fc0f14d873dbc99038f135c2da1b2` | root | Primary typed-decisions checkpoint; its file layout is the reference for the §9 model-package format |
| [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya) | `1c5edc17a7acd8701df6fc341c0d179f1c62c982` | `typed-decisions/` | Same checkpoint family inside the full Laya repository |
| [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya) | `1c5edc17a7acd8701df6fc341c0d179f1c62c982` | `multilingual/` | Multilingual variant; exercises router language selection |

Checkpoint licenses are recorded at packaging time from the model card
current at that revision; they are not assumed to match the runtime license.

## Machine-readable pins

```json
{
  "code": {
    "laya_investigation": {
      "repo": "https://github.com/NandhaKishorM/laya",
      "revision": "42626c348753fbb17572a813127df2278a1ec527"
    },
    "laya_parity_reference": {
      "repo": "https://github.com/NandhaKishorM/laya",
      "revision": "6a5819129eb220570792e417e49723d697efd76f"
    },
    "laya_mlx": {
      "repo": "https://github.com/mizorewww/laya-mlx",
      "revision": "fc1df62828a3fedf4d8229fdac1cbd85f1cdf337"
    }
  },
  "checkpoints": {
    "laya-typed-decisions": {
      "hf_repo": "convaiinnovations/laya-typed-decisions",
      "revision": "f9ab0b228f0fc0f14d873dbc99038f135c2da1b2",
      "package_layout": "root"
    },
    "laya_typed-decisions_subdir": {
      "hf_repo": "convaiinnovations/laya",
      "revision": "1c5edc17a7acd8701df6fc341c0d179f1c62c982",
      "package_layout": "typed-decisions/"
    },
    "laya_multilingual": {
      "hf_repo": "convaiinnovations/laya",
      "revision": "1c5edc17a7acd8701df6fc341c0d179f1c62c982",
      "package_layout": "multilingual/"
    }
  }
}
```

## Tolerance record

Filled in by parity runs as results become available; raw results are stored
with hardware and software metadata per architecture §10.

| Gate | Status | Tolerance | Hardware | Date |
|---|---|---|---|---|
| FP32 selected-answer parity, real checkpoint | not yet run | selected answers must match exactly | — | — |
| FP32 probability drift, real checkpoint | not yet run | TBD from first run | — | — |
| FP16 selected-answer parity | not yet run | TBD | — | — |
| BF16 selected-answer parity | not yet run | TBD | — | — |
