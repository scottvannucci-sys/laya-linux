# CUDA setup

Phase 3 (CUDA) validation status and setup guidance. This page is updated only with
real hardware results — GPU support is never claimed without one (architecture §15).

## Requirements

- An NVIDIA GPU; Blackwell (sm_120, e.g. RTX 50-series) requires a CUDA 12.8-era torch build
- A driver new enough for the chosen torch build (`nvidia-smi` to check)
- Python 3.11+

## Install

```bash
python -m venv .venv-cuda && source .venv-cuda/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128   # or cu126 for older drivers
pip install laya-linux
laya-linux doctor
```

`doctor` reports the detected device plan (backend `cuda`, capability-aware default dtype) as a
verified fact; it never guesses.

## Precision behavior

- `dtype="auto"` on CUDA selects FP16 on capability < 8.0 and BF16 on capability >= 8.0,
  reflecting verified device capability (requirements §10.2).
- FP32 is always available and is the parity reference.
- FP16/BF16 results are *not* bit-identical to CPU FP32 (different kernels); published drift
  tolerances live in `PARITY_BASELINES.md` once the validation battery has run on each
  documented configuration.

## Validation

Run the reproducible battery and keep the JSON artifact:

```bash
python scripts/cuda_validation.py /opt/laya/models/laya-typed-decisions --out cuda-validation.json
```

It records environment, dtype-policy checks, FP32/FP16/BF16 drift vs the CPU FP32 reference,
repeated-call memory stability, and latency/throughput. On Linux the same battery is wrapped by
`scripts/linux_cuda_validation.sh` (environment setup included).

## Documented configurations

| GPU | Capability | Driver | Torch build | Status |
|---|---|---|---|---|
| NVIDIA GeForce RTX 5070 Ti (16GB, Windows 11) | sm_120 | 616.92 | 2.11.0+cu128 | **validated** — see `benchmarks/cuda-validation-win-rtx5070ti.json` |
| NVIDIA GB10 (Grace Blackwell, unified memory, Linux aarch64) | sm_121 | 580.173.02 | 2.11.0+cu128 | **validated** — see `benchmarks/cuda-validation-linux-gb10.json` |

Raw artifacts live in `benchmarks/`. Auto precision on capable GPUs honors the checkpoint's
recorded `amp_dtype` (the laya-typed-decisions checkpoint records `bf16`).

### Gate criteria (recorded in every artifact)

- FP32 CUDA must match the CPU FP32 reference exactly (argmax, noul threshold, probabilities).
- Reduced-precision dtypes: argmax/noul selections may disagree only where the FP32 reference
  itself was near-tied (top-2 gap or tie margin ≤ 0.05); score questions use
  `|ΔE[score]| ≤ 0.05` (the expected score is continuous — Σ i·p_i — so per-probability drift
  accumulates); max probability drift ≤ 0.02.
- Memory: < 50 MB active growth over 30 repeated calls; per-dtype results repeat-identically.

## torch.compile (opt-in evaluation, architecture §14 Phase 3 step 5)

Not evaluated on Windows: the Triton compiler backend torch.compile relies on is not supported
there. Evaluation is deferred to the Linux validation run (`scripts/linux_cuda_validation.sh`
output should include an eager-vs-compiled comparison across shape buckets). It remains **off by
default** — §16 warns dynamic shapes (question count, option count, token length) can make
compilation repeatedly re-specialize and run slower than eager.
