# Published benchmarks

Raw, machine-readable benchmark artifacts for laya-linux. Every number claimed
anywhere in this repository must trace to a JSON file here (requirements §14:
"Report raw samples or machine-readable summaries, not only a best-case number"
and §12: "reproducible local benchmark and emit JSON").

## Conventions

1. **One file per (hardware configuration × workload).** Naming:
   `<what>-<platform>-<gpu>.json`, e.g.
   `cuda-validation-win-rtx5070ti.json`, `cuda-validation-linux-gb10.json`.
2. **Fixed workload files.** The workload (state + questions) must be
   reproducible: use a committed fixture or an inline state string that is
   recorded *inside* the artifact — the benchmark CLI now records
   `input_length_chars`, `question_count`, and `question_ids`, so a fresh
   process can be shown to have run the identical workload.
3. **Full environment recorded.** The artifact's `environment` block (device,
   backend, dtype, torch build, platform) is the identity of the run. A number
   without its environment block is not a result.
4. **Raw samples included.** Summaries (median/p95) are derived; the artifact
   must carry the raw per-iteration samples so any statistic can be recomputed.
5. **Never compare across hardware without saying so** — the artifacts state
   their platform; any document quoting numbers from two different files must
   label the difference prominently (requirements §14).
6. **Correctness accompanies speed.** GPU artifacts come from
   `scripts/cuda_validation.py`, so every latency number is published next to
   its parity-drift and memory-stability evidence. A benchmark file without
   correctness context is not accepted for a supported-configuration claim
   (architecture §15: never claim GPU support without a real hardware result).

## Reproducing

```bash
# Latency/throughput for one configuration:
laya-linux benchmark models/laya-typed-decisions \
    --state "I was charged twice." --preset triage \
    --warmup 3 --iterations 20 > my-run.json

# The full battery used for the published artifacts (CPU reference + CUDA
# FP32/FP16/BF16 parity, stability, benchmarks):
python scripts/cuda_validation.py models/laya-typed-decisions \
    --out benchmarks/cuda-validation-<your-platform>.json
```

## Index of artifacts

| File | Hardware | Workload | Headline (median) |
|---|---|---|---|
| `cuda-validation-win-rtx5070ti.json` | RTX 5070 Ti sm_120, Windows, torch 2.11.0+cu128 | 6-fixture triage battery, 5 questions/fixture | 34.8 ms FP32 · 21.1 ms FP16 · 21.2 ms BF16 |
| `cuda-validation-linux-gb10.json` | NVIDIA GB10 sm_121, Linux aarch64 (unified memory), torch 2.11.0+cu128 | same battery, same fixtures | 85.7 ms FP32 · 63.4 ms FP16 · 63.4 ms BF16 |

CPU reference numbers for the same battery are inside each artifact (the GB10
run's CPU FP32 reference and the RTX run's correspond to the embedded-runtime
path; the Phase 2 CLI benchmark on a separate tiny synthetic model measured
2.7 ms median and is not comparable to either).

Tolerance and parity status for every configuration: `PARITY_BASELINES.md`.
