# tools/profile

Two pieces of decode-performance tooling, both from the decode-performance pass (2026-09-19,
docs/perf.md "Decode performance pass"):

## Per-op decode-step profile

`r4dx::model::Model::DecodeStepProfiled` (`src/model/model.h`/`.cpp`) runs exactly one decode step
(T=1) with each op family (embed, GDN layers, attention layers, MLP, final_norm+lm_head) wrapped in
a `hipEvent` start/end pair, then does ONE `hipEventSynchronize` at the end and reads back every
pair's elapsed time -- so the GPU pipeline is never artificially serialized mid-step just to time
it. `r4dx-cli --profile` (owned by the CLI component, `src/cli/main.cpp`) calls it once on the
first generated token and prints the resulting table to stderr; see `docs/perf.md`'s "Per-op
decode-step profile" section for the table captured this way in every layout, and
`Model::StepProfile`'s own doc comment (`model.h`) for exactly what `gpu_sum_ms` /
`host_enqueue_ms` / `finish_wait_ms` / `wall_ms` do and do not add up to.

Granularity note: this breaks the step down to the same block boundaries `model.cpp` already calls
directly (one entry per GDN layer's whole `GdnLayer::Forward()`, one per attention layer's whole
`AttentionLayer::Forward()`, etc.), not down to individual GEMMs-by-shape or norm/rope/quant
kernels within a block -- doing that would mean threading a profiler handle into
`gdn_layer.cpp`/`attention_layer.hpp`/`mlp.cpp`/`final_lm_head.cpp` themselves, which this pass's
time budget did not extend to. Extending it: add another `SpanAccumulator::Add` call around the
specific sub-call you want broken out (each of those four files' `Forward()` already calls its
GEMMs/kernels as separate, named function calls, so this is mechanical, not a redesign).

## GEMM (WV,SK,MB,NPW,NT) tuning sweep

`tune_gemm.py` sweeps the legal parameter grid for all four `r4d_gemm_*_nt_m64` kernel families
(bf16/w4a16/w4a8/mxfp4) at this model's seven real (N,K) linear shapes (`gdn.in_proj_qkv`,
`gdn.out_proj`, `attn.qg`, `attn.o`, `mlp.gate_up`, `mlp.down`, `lm_head`) for M in
{1,2,4,8,16,32,64}, over the Python `r4d.pyd` binding + the read-only reference venv's torch (same
route `third_party/libr4d/bench_mxfp4_gemm.py` already used for a narrower, mxfp4-only sweep) on
HIP device 1. Each kernel's legal-parameter constraints (K/N divisibility, `WV*SK*32<=1024`,
`MB in 1..4`, the LDS-budget cap `WV*NPW*SK<=64` for the three quantized kernels) are read from that
kernel's own `.hip` source (see `tune_gemm.py`'s own file comment for the exact file:line
references) before generating candidates, not guessed; each launch is also wrapped in try/except as
a redundant second check, since the kernel itself throws `std::runtime_error` on an out-of-range
combo too.

```
C:\Users\user\dev\vLLM_for_AMD\.venv-rocm10\Scripts\python.exe tools\profile\tune_gemm.py \
    --out src\model\gemm_tuning_table.inc
```

writes `src/model/gemm_tuning_table.inc` (checked in, git-tracked -- NOT gitignored, since
`src/model/linear.cpp`'s `PickTuning` `#include`s it directly and the repo needs to build without
requiring every checkout to have `r4d.pyd`/the reference venv available). Re-run after any kernel
change that could shift which WV/SK/MB/NPW/NT wins, or after this model's shapes change (a new
`config.json`, e.g. a different `intermediate_size`) -- `PickTuning` falls back to a hand-derived,
constraint-legal-for-every-shape default (`FallbackTuning`, `linear.cpp`) for any `(layout,N,K)` the
table has no row for, so a stale or missing table degrades gracefully rather than breaking the
build or crashing at runtime.

`--quick` restricts M to {1,8,64} and halves the iteration count, for a fast sanity check that the
sweep machinery still runs after a kernel/shape change, without waiting for the full ~25-30 minute
sweep.
