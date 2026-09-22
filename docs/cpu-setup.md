# CPU setup

Laya-Linux CPU inference is the reference path: FP32 by default, bit-exact against the upstream
references (see `PARITY_BASELINES.md`).

## Requirements

- Linux x86-64 (Windows/macOS work for development but are not validated targets)
- Python 3.11+
- PyTorch 2.2+ (CPU build is sufficient and smallest)

## Install

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install laya-linux --no-index --find-links dist/   # or from PyPI when released
```

## Verify your device

```bash
laya-linux doctor
```

Expected `device_plan`: `device: cpu, backend: cpu, dtype: float32`.

## Behavior notes

- `device="auto"` selects CPU when no accelerator is visible to PyTorch.
- Explicit `device="cpu"` always honors CPU; FP16/BF16 requests on CPU fail with
  `DTYPE_UNSUPPORTED` rather than silently degrading (requirements §10.2).
- Checkpoint files stored in FP16 (e.g. the original Laya checkpoints) are cast to FP32 at load
  time on CPU; results match the FP32 reference parity gates.
