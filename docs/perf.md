# r4dx end-to-end performance and correctness (assembly + CLI milestone)

> **Update (2026-09-20, Milestone 2 integration pass)**: reran the full pipeline from a clean
> `build.ps1 -Clean` rebuild (HIP device 1, 107/107 build steps) -- full `ctest --preset win-hip`
> 30/30 passing in 93.50s, `tools/server/smoke.ps1` all 20 checks passing against the 4-layer test
> container, then one `r4dx-cli` generation per quantized layout at both `--mtp 0` and `--mtp 3`
> against the real, unmodified 64-layer container (`D:\models\r4dx\qwen38-27b.r4dx`), same prompt as
> every table below. Numbers matched the FIX pass's own measurements within run-to-run noise,
> confirming Milestone 2 is reproducible end to end from a clean checkout:
>
> | Layout | mtp=0 decode | mtp=3 decode | mtp=3 acceptance | speedup |
> |---|---|---|---|---|
> | mxfp4 | 27.08 tok/s | 47.46 tok/s | 41.9% (39 rounds, 117 drafted, 49 accepted) | +75.3% |
> | w4a16 | 32.83 tok/s | 66.42 tok/s | 54.3% (35 rounds, 105 drafted, 57 accepted) | +102.3% |
> | w4a8  | 30.95 tok/s | 47.72 tok/s | 32.5% (41 rounds, 123 drafted, 40 accepted) | +54.2% |
> | bf16  | 1.41 tok/s  | 2.21 tok/s  | 48.7% (13 rounds, 39 drafted, 19 accepted), `--max-tokens 32` | +56.7% |
>
> **Milestone 1 -> Milestone 2, headline decode tok/s** (greedy, real 64-layer container, this
> file's own prompt, `--mtp 0` i.e. MTP off in both columns -- isolates the decode/prefill
> performance pass's own effect from MTP's separate, additional speedup shown in the table above):
>
> | Layout | Milestone 1 (2026-09-19) | Milestone 2, `--mtp 0` (2026-09-20) | delta |
> |---|---|---|---|
> | mxfp4 | 24.90 tok/s | 27.08 tok/s | +8.8% |
> | w4a16 | 29.08 tok/s | 32.83 tok/s | +12.9% |
> | w4a8  | 27.88 tok/s | 30.95 tok/s | +11.0% |
> | bf16  | 1.38 tok/s  | 1.41 tok/s  | +2.2% |
>
> Stacking both effects (Milestone 1 baseline -> Milestone 2 `--mtp 0` -> Milestone 2 `--mtp 3`):
> w4a16 goes from 29.08 to 66.42 tok/s (+128.4% total), the largest full-stack gain of the four
> layouts; mxfp4 24.90 -> 47.46 tok/s (+90.6%); w4a8 27.88 -> 47.72 tok/s (+71.2%); bf16 1.38 -> 2.21
> tok/s (+60.1%, on a 32-token-capped run since bf16 decode is too slow for a full 128-token sweep at
> every layout x K combination in one pass). See "Before/after, this pass" (decode/prefill
> performance pass detail) and `docs/mtp.md` (MTP detail, all five `--mtp` values per layout) for the
> full breakdowns this summary draws from. Generated text for all eight runs was coherent, on-topic,
> and (except the two `--max-tokens`-capped bf16 runs) stopped on the model's own EOS token. No
> integration issues found requiring a code fix during this pass -- Milestone 2's three prior stages
> (PERF, SERVER, MTP, plus a FIX pass closing the Opus review's blockers/majors) had already left the
> working tree in a fully green state.

> **Update (2026-09-19, decode performance pass)**: `src/model/**` (incl. `src/model/attention/**`),
> `src/kernels/**`, `tools/profile/**`. All six items measured against the real 64-layer container
> (`D:\models\r4dx\qwen38-27b.r4dx`), same prompt/settings as the table below, HIP device 1.
>
> 1. **Quantized attention projections**: `AttentionLayer` now takes `QuantLinear` for `attn.qg`/
>    `attn.o` and dispatches both through the shared `r4dx::model::ApplyLinear` (`src/model/linear.h`)
>    -- the same path GDN's `in_proj_qkv`/`out_proj` and MLP's `gate_up`/`down` already used.
>    `Container::Load`'s bf16 override is gone (`src/model/container.cpp`); every layout now loads
>    `attn.qg`/`attn.o` at its own requested layout. `tests/model/attention/test_attn_layer` gates
>    bf16 at its original tight tolerance (prefill norm rel err 1.84e-2, decode 1.54e-2, both under
>    the 2e-2 bf16 bound) and reports (does not gate) the quantized layouts' own rel-L2. **VRAM now
>    differs by layout for the first time** (previously all four layouts used the same 17.79 GiB
>    because attn.qg/o were always bf16): mxfp4/w4a16/w4a8 dropped to **15.75 GiB** (-2.04 GiB, the
>    16 full-attention layers' quantized qg/o weights); bf16 is unchanged at 31.86 GiB (it already
>    loaded qg/o as bf16). This is an intentional **precision change** for the three quantized
>    layouts, not a pure throughput fix -- see "Generated-text regression" below for what that does
>    and does not change.
> 2. **Prefill waste**: `Model::Prefill` now passes `want_logits=false` for every chunk except the
>    last (`RunChunk`'s existing parameter, `src/model/model.cpp`), so `final_norm`+full-vocab
>    `lm_head`+the D2H logits readback+the per-chunk `stream_.Synchronize()` gate that guards it only
>    run once per `Prefill()` call instead of once per <=64-row chunk. Correctness-neutral (a
>    non-final chunk's logits were never read before either); purely removes wasted GPU work +
>    host syncs on multi-chunk prefills.
> 3. **Per-op profile**: `Model::DecodeStepProfiled` (`model.h`/`.cpp`) times one decode step (T=1)
>    per op-family (embed / GDN layers / attention layers / MLP / final_norm+lm_head) with hipEvent
>    pairs recorded async, one `hipEventSynchronize` at the end. `r4dx-cli --profile` (CLI
>    component's own change, `src/cli/main.cpp`/`cli_args.h`) runs it once on the first generated
>    token and prints the table to stderr. See "Per-op decode-step profile" below for the measured
>    table in all four layouts and `tools/profile/README.md` for what this does and does not break
>    out (block-level, not sub-kernel; extending it is mechanical, see that file).
> 4. **GEMM tuning table**: `tools/profile/tune_gemm.py` sweeps the legal `(WV,SK,MB,NPW,NT)` grid
>    (constraints read from each `r4d_gemm_*_nt_m64.hip` source, see that script's own file comment)
>    for all four GEMM kernel families at this model's seven real `(N,K)` shapes, M in
>    {1,2,4,8,16,32,64}, over the `r4d.pyd` + reference-venv-torch route
>    (`third_party/libr4d/bench_mxfp4_gemm.py`'s route, generalized). Wrote 196 measured rows to
>    `src/model/gemm_tuning_table.inc` (checked in); `src/model/linear.cpp`'s `PickTuning` now looks
>    up `(layout,N,K,M)` in that table first (rounding up to the nearest measured M-band) and only
>    falls back to the old hand-derived constant when a shape is untabulated. See "GEMM tuning sweep"
>    below for the full methodology, a table excerpt, and **an important caveat**: this change is
>    NOT byte-identical-safe for the quantized layouts (see "Generated-text regression" below) --
>    different `SK` (K-split count) changes the LDS reduction's summation order, which is a real
>    rounding-level change, not a bug.
> 5. **Host overhead**: added `Model::DecodeStepGreedy` + device-side argmax (`r4dx_argmax_f32`,
>    `src/kernels/`) for the `--temperature 0` path: `Model::RunChunk` now argmaxes `logits_dev_` ON
>    DEVICE and reads back a single `int32` instead of the full `vocab_size`-length fp32 logits
>    vector every decode step. `r4dx-cli`'s greedy loop (`main.cpp`) uses `DecodeStepGreedy` instead
>    of `DecodeStep`+host-side `Argmax`. The remaining per-token D2H is that one 4-byte copy, guarded
>    by the same single `stream_.Synchronize()` `RunChunk` already did.
> 6. **Net measured impact** (same command as "Perf table" below, greedy, real container): see
>    "Before/after, this pass" for the full per-step table. Headline decode numbers, milestone-1
>    baseline -> this pass (items 1,2,3,5, pre-GEMM-tuning) -> this pass + item 4 (tuned):
>    mxfp4 24.90 -> 27.09 -> 27.04 tok/s; w4a16 29.08 -> 32.64 -> 32.78 tok/s; w4a8 27.88 -> 30.99 ->
>    30.87 tok/s; bf16 1.38 -> 1.38 -> 1.41 tok/s. Prefill (chunked, dominated by the item-4 tuning
>    once GEMMs are tabulated at M=64): mxfp4 400.21 -> 401.55 -> **556.54** tok/s; w4a16 419.44 ->
>    443.79 -> **613.18** tok/s; w4a8 410.75 -> 424.09 -> **606.06** tok/s; bf16 20.40 -> 20.41 ->
>    **37.78** tok/s. `tests/run_tests.ps1` is 29/29 passing after every step (24 from Milestone 1 +
>    5 from the in-flight `src/server` component sharing this working tree).
>
> **Generated-text regression** (same prompt/settings/greedy as "Perf table" below, verbatim text in
> "Before/after, this pass"):
> - **bf16**: byte-identical to the Milestone-1 baseline text through every step of this pass,
>   including after item 4's GEMM retuning -- expected, since bf16 has no quantization error
>   compounding logit margins tight enough for a rounding-order change to flip an argmax.
> - **mxfp4/w4a16/w4a8**: changed from the Milestone-1 baseline after item 1 (expected -- attn.qg/o
>   are now genuinely quantized for these layouts instead of silently running bf16, so the
>   forward-pass math itself changed) and changed AGAIN after item 4's GEMM retuning, even though
>   items 1-3+5 alone were byte-identical to each other. Root cause: `SK` (the GEMM's K-split count)
>   changes the order partial sums are reduced through LDS in, and these three layouts' known ~7-13%
>   per-GEMM quantization error (`docs/status.md` "Known gaps") already sits close enough to some
>   argmax decision boundaries that a different-but-equally-valid rounding order flips a token
>   choice a few dozen tokens in -- both texts remain fluent, coherent, and on-topic (see "Before/
>   after, this pass"), so this is a rounding-order effect of item 4, not a correctness regression;
>   documented here rather than silently treated as "byte-identical" per the task's own instruction
>   to report before/after per change honestly.
>
> See "Per-op decode-step profile", "GEMM tuning sweep", and "Before/after, this pass" below for the
> full data this summary is drawn from.

## Per-op decode-step profile (2026-09-19, decode performance pass)

One `Model::DecodeStepProfiled` call (`r4dx-cli --profile`, T=1, real 64-layer container, after
this pass's items 1/2/3/5 and the item-4 GEMM retune) per layout. `gpu_sum_ms` is the sum of every
hipEvent-measured span (does NOT include the final sync+readback); `finish_wait_ms` is the
host-chrono-measured cost of that final `hipEventSynchronize` + the 4-byte argmax D2H -- see
`Model::StepProfile`'s own comment (`model.h`) for why the two are not additive with each other.

**mxfp4** (wall 44.28ms, host_enqueue 4.25ms/9.6%, finish_wait 40.03ms/90.4%):

| op family | ms | calls | % gpu_sum |
|---|---|---|---|
| mlp (gate_up/down GEMMs + silu) | 22.15 | 64 | 52.5% |
| gdn_layers (GDN kernels + in/out_proj GEMMs) | 15.27 | 48 | 36.2% |
| attn_layers (attention kernels + qg/k/v/o GEMMs) | 3.49 | 16 | 8.3% |
| final_norm+lm_head (full-vocab GEMM + widen) | 1.26 | 1 | 3.0% |
| embed | 0.06 | 1 | 0.1% |

**w4a16** (wall 38.13ms, host_enqueue 2.48ms/6.5%, finish_wait 35.65ms/93.5%):

| op family | ms | calls | % gpu_sum |
|---|---|---|---|
| mlp | 18.55 | 64 | 50.9% |
| gdn_layers | 13.37 | 48 | 36.7% |
| attn_layers | 3.21 | 16 | 8.8% |
| final_norm+lm_head | 1.25 | 1 | 3.4% |
| embed | 0.07 | 1 | 0.2% |

**w4a8** (wall 38.39ms, host_enqueue 2.69ms/7.0%, finish_wait 35.70ms/93.0%):

| op family | ms | calls | % gpu_sum |
|---|---|---|---|
| mlp | 19.11 | 64 | 51.3% |
| gdn_layers | 13.74 | 48 | 36.8% |
| attn_layers | 3.16 | 16 | 8.5% |
| final_norm+lm_head | 1.24 | 1 | 3.3% |
| embed | 0.04 | 1 | 0.1% |

**bf16** (wall 737.42ms, host_enqueue 6.81ms/0.9%, finish_wait 730.61ms/99.1%):

| op family | ms | calls | % gpu_sum |
|---|---|---|---|
| mlp | 439.89 | 64 | 60.1% |
| gdn_layers | 155.97 | 48 | 21.3% |
| final_norm+lm_head | 90.97 | 1 | 12.4% |
| attn_layers | 44.79 | 16 | 6.1% |
| embed | 0.09 | 1 | 0.0% |

**Top 3 costs** (every layout): (1) **MLP gate_up/down GEMMs** (51-60% of GPU time -- the widest
GEMMs in the model, `N=34816,K=5120` and `N=5120,K=17408`, x64 layers); (2) **GDN layers** (21-37% --
48 GDN layers' `in_proj_qkv`/`out_proj` GEMMs plus the conv/kkt/recurrent-update kernels, all folded
into one entry per layer at this pass's block-level granularity); (3) **attention layers** for the
three quantized layouts (8-9%) or **final_norm+lm_head** for bf16 (12.4% -- the full 248320-row
vocab GEMM at bf16 is far more expensive than at any quantized layout, where it also runs quantized
since `lm_head`'s own layout follows `--layout` like every other linear). `finish_wait` (the actual
per-token sync-and-4-byte-readback cost item 5 targeted) is 90-99% of wall time in every layout --
expected and correct, since `finish_wait` is host time BLOCKED on the same GPU work `gpu_sum_ms`
already accounts for (see `StepProfile`'s comment), not extra work; `host_enqueue` (pure CPU launch-
issue overhead) is the number item 5 was actually trying to shrink, and at 0.9-9.6% of wall it is
not the bottleneck in any layout -- the GEMMs are.

## GEMM tuning sweep (2026-09-19, decode performance pass)

`tools/profile/tune_gemm.py` (see `tools/profile/README.md`) swept `WV in {1,2,4,8,16,32}`,
`SK in {1,2,4,8,16,32}`, `MB` fixed per M-band (`max(1,min(4,(M+15)/16))`, matching
`third_party/libr4d/bench_mxfp4_gemm.py`'s own convention), and `NPW` over each kernel's legal set
({1} for bf16, {1,4} for w4a16, {1,2,4,8} for w4a8/mxfp4) -- filtered through the legal-parameter
constraints read from each kernel's own `.hip` source (see that script's file comment for the exact
constraints and file:line references) before ever calling into `r4d.pyd`, with a try/except around
each call as a second, redundant safety net. **Caveat**: `NT` was held fixed at `1` throughout this
sweep (not itself swept) -- `r4d_gemm_w4a16_nt_m64.hip`'s own comment recommends the non-temporal
weight-load path for a weight read once per step and never reused, which is what motivated fixing
it rather than doubling the sweep's search space; a future pass could add `NT` as a swept dimension.
196 rows (7 shapes x 7 M-bands x 4 layouts) written to `src/model/gemm_tuning_table.inc`.

Excerpt (decode band M=1 and prefill band M=64, all four layouts, `mlp.gate_up` -- the single most
expensive shape per the profile above):

| layout | shape | N | K | M | WV/SK/MB/NPW/NT | measured |
|---|---|---|---|---|---|---|
| bf16 | mlp.gate_up | 34816 | 5120 | 1 | 1/8/1/1/1 | 564.09 us |
| bf16 | mlp.gate_up | 34816 | 5120 | 64 | 1/4/4/1/1 | 691.99 us |
| w4a16 | mlp.gate_up | 34816 | 5120 | 1 | 2/4/1/1/1 | 147.76 us |
| w4a16 | mlp.gate_up | 34816 | 5120 | 64 | 8/4/4/1/1 | 356.43 us |
| w4a8 | mlp.gate_up | 34816 | 5120 | 1 | (see `gemm_tuning_table.inc`) | -- |
| mxfp4 | mlp.gate_up | 34816 | 5120 | 1 | (see `gemm_tuning_table.inc`) | -- |

Effect on end-to-end throughput: decode (M=1, dominated by launch overhead + small-GEMM latency
more than by tiling choice) barely moved (mxfp4 27.09->27.04 tok/s, w4a16 32.64->32.78, w4a8
30.99->30.87, bf16 1.38->1.41 -- within run-to-run noise for all four). Prefill (M up to 64, where
`WV`/`SK`/`MB`/`NPW` genuinely change how much of the GPU's compute the launch keeps busy) improved
substantially: mxfp4 401.55->**556.54** tok/s (+38.6%), w4a16 443.79->**613.18** tok/s (+38.2%),
w4a8 424.09->**606.06** tok/s (+42.9%), bf16 20.41->**37.78** tok/s (+85.1%). See "Generated-text
regression" above for the floating-point-non-associativity caveat this retuning surfaces for the
three quantized layouts.

## Before/after, this pass (2026-09-19, decode performance pass)

Same command as the "Perf table" section below (`--prompt "Write a haiku about GPUs, then explain
what a GPU is in two sentences." --max-tokens 128 --temperature 0 --stats`, `--max-ctx 2048` except
bf16's `--max-ctx 512`), real 64-layer container, HIP device 1. Three snapshots: **baseline**
(Milestone-1 integration pass, top of this file), **items 1+2+3+5** (this pass, before the item-4
GEMM retune), **+item 4** (this pass, after the retune -- `tests/run_tests.ps1` 29/29 passing at
this point, the numbers this section ends on).

| Layout | Stage | Container load | Prefill | Decode | VRAM |
|---|---|---|---|---|---|
| mxfp4 | baseline | 10.04s | 395.68 tok/s | 24.91 tok/s | 17.79 GiB |
| mxfp4 | items 1+2+3+5 | 17.24s | 401.55 tok/s | 27.09 tok/s | **15.75 GiB** |
| mxfp4 | +item 4 | 17.02s | **556.54 tok/s** | 27.04 tok/s | 15.75 GiB |
| w4a16 | baseline | 16.71s | 416.74 tok/s | 29.04 tok/s | 17.79 GiB |
| w4a16 | items 1+2+3+5 | 15.23s | 443.79 tok/s | 32.64 tok/s | **15.75 GiB** |
| w4a16 | +item 4 | 15.17s | **613.18 tok/s** | 32.78 tok/s | 15.75 GiB |
| w4a8 | baseline | 16.88s | 407.05 tok/s | 27.90 tok/s | 17.79 GiB |
| w4a8 | items 1+2+3+5 | 14.92s | 424.09 tok/s | 30.99 tok/s | **15.75 GiB** |
| w4a8 | +item 4 | 16.12s | **606.06 tok/s** | 30.87 tok/s | 15.75 GiB |
| bf16 | baseline | 59.59s | 20.46 tok/s | 1.38 tok/s | 31.86 GiB |
| bf16 | items 1+2+3+5 | 58.34s | 20.41 tok/s | 1.38 tok/s | 31.86 GiB |
| bf16 | +item 4 | 63.55s | **37.78 tok/s** | 1.41 tok/s | 31.86 GiB |

(Container load time is dominated by disk I/O and is noisy run to run at this container size --
not a target of this pass, shown only for completeness.)

Generated text, `+item 4` stage (final state of this pass):

### `--layout mxfp4` (changed from baseline -- see "Generated-text regression" above)

```
Silicon threads weave,
Parallel light in the dark,
Pixels bloom in code.

A GPU (Graphics Processing Unit) is a specialized electronic circuit designed to rapidly manipulate and alter memory to accelerate the creation of images in a frame buffer for output to a display. It excels at handling thousands of simultaneous calculations, making it essential for rendering complex 3D graphics and increasingly vital for tasks like artificial intelligence and scientific computing.
```

### `--layout w4a16` (changed from baseline)

```
Silicon threads weave,
Parallel light in the dark,
Pixels bloom anew.

A GPU (Graphics Processing Unit) is a specialized electronic circuit designed to rapidly manipulate and alter memory to accelerate the creation of images in a frame buffer for output to a display. Unlike a CPU, which is optimized for sequential processing, a GPU is built for parallel processing, allowing it to handle thousands of calculations simultaneously.
```

### `--layout w4a8` (changed from baseline)

```
Silent silicon hums,
Thousand cores weave light and shadow,
Pixels bloom in code.

A GPU, or Graphics Processing Unit, is a specialized electronic circuit designed to rapidly build and manipulate images stored in memory for output to a display. It excels at handling large numbers of simple calculations in parallel, making it essential for rendering graphics and increasingly for general-purpose computing tasks like AI training.
```

### `--layout bf16` (byte-identical to baseline, see above)

```
Silicon threads weave,
Parallel light in the dark,
Pixels come alive.

A GPU (Graphics Processing Unit) is a specialized electronic circuit designed to rapidly manipulate and alter memory to accelerate the creation of images in a frame buffer. Unlike a CPU, it is optimized for handling thousands of simultaneous calculations, making it ideal for rendering graphics and modern machine learning tasks.
```

All four remain coherent, on-topic, grammatically correct, and hit the model's own EOS token before
the 128-token cap -- the three quantized layouts' text changes are a rounding-order effect (item 1's
intentional precision change, compounded by item 4's retune), not garbling or truncation.

> **Update (2026-09-19, Milestone 1 integration pass)**: reran the full pipeline from a clean
> `build.ps1 -Clean` rebuild (HIP device 1, 84/84 build steps) -- full `ctest --preset win-hip`
> 24/24 passing in 59.50s, then one `r4dx-cli` generation per layout against the real, unmodified
> 64-layer container (`D:\models\r4dx\qwen38-27b.r4dx`), same prompt/settings as the review-fix
> pass below. Numbers matched within run-to-run noise and generated text was byte-for-byte
> identical to the review-fix pass's own output (greedy/deterministic): mxfp4 load 9.73s / prefill
> 400.21 tok/s / decode 24.90 tok/s / 17.79 GiB; w4a16 load 16.01s / prefill 419.44 / decode 29.08
> tok/s / 17.79 GiB; w4a8 load 16.62s / prefill 410.75 / decode 27.88 tok/s / 17.79 GiB; bf16 load
> 60.77s / prefill 20.40 / decode 1.38 tok/s / 31.86 GiB (`--max-ctx 512`). This confirms the
> milestone is reproducible end to end from a clean checkout, not just an artifact of the working
> build directory the review-fix pass used. See `docs/status.md`'s "What passes (Milestone 1
> integration pass)" section for the same table alongside the test results.

> **Update (2026-09-19, review-fix pass)**: applied an Opus code review of `src/model/**`,
> `src/cli/**` on top of the assembly stage below. Fixed a blocker (GDN conv-state rolling-buffer
> depth was one element too large for a plain single-token decode, an out-of-bounds kernel read
> that was silently correct only by accident of the compiler's scratch-frame layout -- see
> `src/model/gdn_state.h`'s updated comment) and two of the perf-affecting majors: `AttentionLayer`
> no longer does ~208 `hipMalloc`/`hipFree` pairs per decode token (moved onto the shared
> `core::Arena`, `src/model/attention/include/r4dx/model/attention/attention_layer.hpp`), and
> `GdnLayer`'s per-call control-array upload no longer does 144 host-blocking `hipStreamSynchronize`
> calls per decode token (replaced `UploadArray` with `GdnControlCache`, `src/model/gdn_state.h`,
> which uploads each distinct value once ever rather than once per call). Added a value-gated
> `tests/model/test_forward_smoke.cpp` check (`Prefill(N)` vs `Prefill(N-1)+DecodeStep` must agree)
> that would have caught the blocker. The perf table, generated text, and "Bugs found" section below
> are this update's numbers; the original assembly-stage narrative follows for its still-relevant
> correctness evidence and the bugs it found and fixed.
>
> Net decode throughput impact of this pass, same prompt/settings as the table below: mxfp4 16.34 ->
> 24.91 tok/s (+52%), w4a16 18.43 -> 29.04 tok/s (+58%), w4a8 17.75 -> 27.90 tok/s (+57%), bf16 1.33
> -> 1.38 tok/s (~flat -- bf16's cost is dominated by its ~2.5GB-per-GEMM weight reads at every
> layer, not by the host-sync/malloc overhead these fixes removed). Prefill also improved (fewer
> allocations per full-attention layer even in the many-rows-per-launch prefill case): mxfp4 307.85
> -> 395.68 tok/s, w4a16 326.99 -> 416.74 tok/s, w4a8 318.99 -> 407.05 tok/s, bf16 20.08 -> 20.46
> tok/s. Generated text is byte-for-byte identical to the pre-fix run for all four layouts (greedy/
> deterministic, and the state-handoff blocker never actually manifested at this run's short
> generation length -- see the blocker's own note about "correct today only by accident") --
> confirming these are pure throughput fixes, not behavior changes.

Measured 2026-09-19 on the real, complete pipeline: `r4dx-cli` (src/cli/main.cpp) driving
`r4dx::model::Model` (src/model/model.{h,cpp}) against the real 64-layer, full-vocab container
`D:\models\r4dx\qwen38-27b.r4dx` (87.79 GiB on disk, all four quantized body layouts plus bf16 for
every linear, MTP + vision passthrough tensors present but unused by this milestone), built from
the real `C:\AI\models\Qwen3.8-27B` checkpoint. All runs on HIP device 1 (`$env:HIP_VISIBLE_DEVICES
='1'`) on the single AMD Radeon AI PRO R9700 (gfx1201, 31.86 GiB VRAM reported by `hipInfo`).

Command (identical across layouts except `--layout` and, for `bf16`, `--max-ctx` -- see "VRAM"
below):

```
$env:HIP_VISIBLE_DEVICES='1'
build\win-hip\src\cli\r4dx-cli.exe --model D:\models\r4dx\qwen38-27b.r4dx --layout <layout> ^
    --prompt "Write a haiku about GPUs, then explain what a GPU is in two sentences." ^
    --max-tokens 128 --temperature 0 --max-ctx 2048 --stats
```

`--temperature 0` selects greedy argmax decoding (`r4dx::kernels::Argmax`, `SampleParams
.temperature <= 0`). The prompt is rendered through the real `chat_template.jinja` with
`enable_thinking=false` (CLI default, i.e. "thinking off") and no system prompt, then encoded with
the real BPE tokenizer (`parse_special=true`) -- exactly what `--chat`/`--prompt` do in production
use, not a shortcut path.

## Perf table (post review-fix pass, 2026-09-19)

| Layout | Container load | Prefill (prompt=29 tok) | Decode | Generated tokens | Stopped on | VRAM used |
|---|---|---|---|---|---|---|
| mxfp4 | 10.04 s | 0.073 s (395.68 tok/s) | 3.292 s (24.91 tok/s) | 82 | eos | 17.79 GiB |
| w4a16 | 16.71 s | 0.070 s (416.74 tok/s) | 3.031 s (29.04 tok/s) | 88 | eos | 17.79 GiB |
| w4a8  | 16.88 s | 0.071 s (407.05 tok/s) | 3.119 s (27.90 tok/s) | 87 | eos | 17.79 GiB |
| bf16  | 59.59 s | 1.417 s (20.46 tok/s)  | 53.701 s (1.38 tok/s) | 74 | eos | 31.86 GiB |

(Pre-fix numbers, same prompt/settings, for comparison: mxfp4 10.51s load / 307.85 prefill tok/s /
16.34 decode tok/s; w4a16 16.89s / 326.99 / 18.43; w4a8 17.73s / 318.99 / 17.75; bf16 64.05s / 20.08
/ 1.33 -- see the "Update" note at the top of this file.)

**attn.qg/o note**: every run below prints `note: attn.qg/o load as bf16 regardless of --layout=...
(16 full-attention layers)` at load time (this pass's `Container::Load` change, see "Known
limitation" below) -- expected, not an error.

All four layouts hit the model's own EOS token before the 128-token cap (greedy decoding is
deterministic, so each layout's own quantization error is the only thing that changes the exact
generated text/length -- not a bug or a truncation).

**bf16 VRAM note**: bf16 loads every GDN/MLP/lm_head linear at 2 bytes/element (attn.qg/o are
*already* bf16-only regardless of `--layout`, see "Known limitation" below) -- this uses
31.86 GiB of the R9700's 31.86 GiB total, i.e. the entire card, with the KV cache and per-layer
scratch arena counted in. The three quantized layouts above ran with `--max-ctx 2048`; the bf16
run used `--max-ctx 4096` initially and was re-run at `--max-ctx 512` to leave enough headroom for
the fp8 KV cache (16 layers x kv_heads=4 x head_dim=256 x 2 x max_ctx bytes) not to push total
usage past the card's limit -- 512 tokens of context is more than the ~29-token prompt plus 128
generated tokens this run needs, but this is not the 131072-token default the CLI otherwise uses,
and a longer bf16 conversation would need either a bf16-specific `--max-ctx` well under the
quantized layouts' headroom or a card with more VRAM. mxfp4/w4a16/w4a8 have ample headroom (~14
GiB free at `--max-ctx 2048`) to run at the CLI's full 131072-token default in practice.

## Generated text (verbatim)

### `--layout mxfp4`

```
Silicon threads weave light,
Parallel paths solve the dark,
Graphics born in code.

A GPU (Graphics Processing Unit) is a specialized electronic circuit designed to rapidly manipulate and alter memory to accelerate the creation of images in a frame buffer for output to a display. It excels at handling thousands of simultaneous calculations, making it essential for rendering complex 3D graphics and accelerating high-performance computing tasks.
```

### `--layout w4a16`

```
Silicon threads weave,
Parallel light in the dark,
Pixels bloom anew.

A GPU (Graphics Processing Unit) is a specialized electronic circuit designed to rapidly manipulate and alter memory to accelerate the creation of images in a frame buffer for output to a display. Unlike a CPU, which is optimized for sequential processing, a GPU is built to handle thousands of simple calculations simultaneously, making it ideal for graphics rendering and modern parallel computing tasks.
```

### `--layout w4a8`

```
Silent silicon hums,
Thousand cores weave light and shadow,
Pixels bloom in code.

A GPU (Graphics Processing Unit) is a specialized electronic circuit designed to rapidly manipulate and render complex images, text, and video data for output to a display. It achieves this by using thousands of small, efficient cores to perform parallel processing tasks, which makes it significantly faster than a general-purpose CPU for graphics-intensive workloads.
```

### `--layout bf16`

```
Silicon threads weave,
Parallel light in the dark,
Pixels come alive.

A GPU (Graphics Processing Unit) is a specialized electronic circuit designed to rapidly manipulate and alter memory to accelerate the creation of images in a frame buffer. Unlike a CPU, it is optimized for handling thousands of simultaneous calculations, making it ideal for rendering graphics and modern machine learning tasks.
```

All four are coherent, on-topic, grammatical English: a haiku-styled GPU poem followed by an
accurate two-sentence GPU explanation, as the prompt asked. No layout produced garbage, so the
task's "debug by bisecting against the goldens" path was not needed for output quality -- it *was*
needed, and used, to find and fix two real bugs surfaced by this exact run (see below).

## Bugs found and fixed while producing this run

The first real (non-test-container) runs crashed or produced garbled/repeating output. Both were
missing-stream-synchronization races, not model/kernel correctness bugs -- confirmed by re-running
under `HIP_LAUNCH_BLOCKING=1` (which serializes every kernel launch): the exact same crash-then-
garbage sequence turned into a clean run with fluent, sensible text at every step, isolating the
bug to missing ordering rather than wrong math.

1. **`src/model/gdn_layer.cpp`'s `UploadArray` helper** (used for the tiny `cu`/`cache_idx`/
   `has_init`/`sidx` control arrays every GDN layer call uploads) used a plain `hipMemcpy` with no
   stream argument. `r4dx::model::Model`'s compute stream (`stream_`, `model.cpp`) is created with
   `hipStreamNonBlocking`, which by design does **not** implicitly synchronize against the legacy/
   null stream a bare `hipMemcpy` uses. Combined with `Arena::Reset()` between layers (a host-only
   offset rewind, by design -- see `arena.hpp`), a later layer's control-array upload could
   overwrite the same bump-allocator bytes an earlier layer's still-in-flight kernel was reading,
   surfacing as an intermittent `HIP error 719 (unspecified launch failure)` on the real 64-layer
   model that the 4-layer test container / `test_forward_smoke` never hit (too few layers/too
   little work in flight for the race window to matter). Fixed by routing the upload through
   `hipMemcpyAsync(..., stream)` + an explicit `hipStreamSynchronize(stream)`, i.e. ordered on the
   same stream as everything else in that layer.
2. **`src/model/model.cpp`'s `Model::RunChunk`** copied `logits_dev_` back to the host with a
   plain, synchronous `DeviceBuffer::CopyToHost` (again a bare, no-stream `hipMemcpy`) *before*
   calling `stream_.Synchronize()`, instead of after. The widen kernel that produces `logits_dev_`
   runs on `stream_`; for the same `hipStreamNonBlocking` reason as above, the plain `hipMemcpy`
   was not guaranteed to wait for it, so `CopyToHost` could read stale/partial logits. This is what
   produced coherent *first* tokens (the very first decode step happened to still be safe) followed
   by garbled, repeating continuations as the race compounded. Fixed by moving
   `stream_.Synchronize()` to immediately before `CopyToHost` (not after).

Both fixes are the same lesson: any plain (`hipMemcpy`/`DeviceBuffer::CopyToHost`/
`CopyFromHost`) host<->device copy against memory a `hipStreamNonBlocking` stream produced or
will consume needs an *explicit* wait, never an implicit one. `tests/model/attention/**`'s
`AttentionLayer` also does a few plain `CopyFromHost` calls (`pos_ids`/`slot_mapping`/
`seqused_k`), but onto freshly `hipMalloc`'d (not arena-bump-allocated) buffers freed via normal
RAII at the end of each call -- `hipFree`'s implicit device-wide synchronization (relied on
throughout this codebase's existing, already-hardware-tested components) is believed to make that
pattern safe, but it was not independently re-audited under the same rigor as the two fixes above;
flagged in "Open issues" below.

## Bugs found and fixed in the review-fix pass (2026-09-19, on top of the above)

1. **GDN conv-state rolling-buffer depth off-by-one (blocker)** -- `src/model/gdn_state.h`'s
   `GdnStateManager` sized the conv-state's per-(sequence,channel) rolling buffer as `conv_width -
   1 + max_decode_window`. `r4d_gdn_conv_update_w4_h128_bf16`'s decode-side cache rewrite
   (`third_party/libr4d/r4d_gdn_conv_w4_h128_bf16.hip:398-414`) is only self-consistent when the
   buffer depth equals `max_query_len + width - 2`, i.e. `conv_width - 2 + max_decode_window` --
   one element smaller. For a plain single-token decode (`max_decode_window=1`, `width=4`) this
   made every decode step's `r4d_gdn_conv_update_w4_h128_bf16` call read one element out of bounds
   of a stack-local `hist[CP_ST][CP_DPL]` array (`CP_ST = width-1 = 3`), landing on adjacent
   scratch memory (`xb[0]`) that happened to hold the correct channel values often enough that
   generation stayed coherent in every run so far -- confirmed on hardware (verbatim 3x sentence
   repetition test, see the assembly stage's own run) before this fix, i.e. this was silently
   correct by accident of the compiler's scratch-frame layout, not of the code. Fixed by changing
   the buffer sizing to `conv_width - 2 + max_decode_window`; verified by the new
   `test_forward_smoke` prefill/decode equivalence check (see "Correctness evidence") and by
   `test_gdn_layer` staying at its usual bf16 ~4.4e-3 / quantized ~7-8e-2 rel-L2 numbers.
2. **`AttentionLayer::Forward` per-token allocation storm (major, perf)** -- constructed 13
   `DeviceBuffer`s (13 `hipMalloc` + 13 `hipFree`, the latter device-synchronizing) per call, x16
   full-attention layers x every decode token. Moved every temporary onto the shared
   `core::Arena` (already used by GDN/MLP) and hoisted the two remaining small control buffers
   (`positions` -- which also now doubles as `slot_mapping`, since this cache's contiguous block
   table makes the two identical -- and `seqused_k`) into `Model`-owned persistent buffers,
   uploaded once per chunk instead of once per attention layer.
3. **`GdnLayer`'s `UploadArray` host-blocking syncs (major, perf)** -- every GDN layer's `cu`/
   `cache_idx`/`has_init`/`sidx` control-array upload did a `hipMemcpyAsync` + an immediate
   `hipStreamSynchronize`: 3 host-blocking pipeline drains x 48 GDN layers = 144 syncs per decode
   token. In this model's single-sequence scope these arrays are pure functions of `(T, slot)`
   with `slot` constant for the whole session, so replaced `UploadArray` with `GdnControlCache`
   (`gdn_state.h`): each distinct `(T, slot)` value is uploaded once, ever (safe without any
   stream sync, since a freshly `hipMalloc`'d buffer is never reused by anything else) and every
   later call reuses the cached device pointer.

See the "Update" note at the top of this file for the combined throughput impact.

## Correctness evidence

- **This run's own output** (above): fluent, on-topic, grammatically correct English matching the
  prompt's request, for all four layouts, greedy/deterministic.
- **Layer-level goldens** (already-passing, real-transformers-checkpoint-backed tests, run as part
  of `ctest --preset win-hip`): `test_gdn_layer` (layer 0, GDN, prefill+decode, all 4 layouts),
  `test_final_lm_head` (final_norm+lm_head, all 4 layouts), `tests/model/attention/test_attn_layer`
  (layer 3, full attention, bf16) -- see each test's own file comment for measured rel-L2 numbers
  and tolerances (bf16 tight at ~1e-4 to ~4e-3; quantized layouts ~7e-2 to ~1.3e-1, a known,
  already-flagged per-tensor-quantization accuracy gap from the model-core stage, not something
  this stage changed).
- **`tests/model/test_forward_smoke`** (assembly stage): exercises the assembled `Model` class
  (chunked prefill across a 64-token boundary + several decode steps, both GDN and full-attention
  layers, all four layouts) end-to-end for finite, correctly-shaped logits on the 4-layer test
  container.
- **`tests/model/test_forward_smoke`'s prefill/decode equivalence check** (review-fix pass, new):
  `Prefill(tokens)` vs `Prefill(tokens[:-1]) + DecodeStep(tokens[-1])` must land on the same
  next-token logits -- this exercises exactly the has_init/start_pos state handoff the GDN
  conv-state blocker lived in, which the NaN/Inf-only checks above do not. Measured on the 4-layer
  test container, HIP device 1: bf16 rel L2=1.96e-3 (tol 1e-2), mxfp4=6.99e-3, w4a16=2.09e-3,
  w4a8=3.86e-2 (quantized tol 8e-2).
- **`tools/reference/first_token.py`** (new this stage, **not executed**): a full-checkpoint,
  `transformers`-only (no r4dx code) top-5-logit dump for the exact same chat-templated prompt,
  intended for a byte-for-byte-independent cross-check of the engine's first generated token. A
  27B-parameter CPU (or CPU-competing GPU) forward pass was judged not feasible inside this
  session's remaining time budget, which the task brief explicitly allows falling back from ("only
  if feasible in <30 min; otherwise report the layer-golden results as the correctness evidence") --
  the script is provided for a future run, but its output was not gathered or compared here.

## Known limitation carried from this stage's design (not a bug)

`src/model/container.cpp`'s `Container::Load` always loads `attn.qg`/`attn.o` (the two quantized
linears inside each of the 16 full-attention layers) as **bf16**, regardless of the requested
`--layout`. `src/model/attention/`'s `AttentionLayer` (a different component, owned jointly now)
only implements a bf16 GEMM dispatch for those two linears -- extending it to dispatch through
`r4dx::model::ApplyLinear`'s quantized paths (mxfp4/w4a16/w4a8) the same way GDN's
`in_proj_qkv`/`out_proj` and MLP's `gate_up`/`down` already do is the natural next step (the
"dedupe of any duplicated Linear logic between core and attention" this stage's ownership grant
anticipated) but was not done here given the time budget -- see this Attention component's `PagedKvCache`/
`AttnConfig`/`AttnWeights` structs would need to grow to carry `QuantLinear` instead of raw
`const uint16_t*`, and `tests/model/attention/test_attn_layer`'s own link graph (it does not
currently link `r4dx_model`) would need adjusting too. Every layout's real container does carry
the bf16 tensors for these two linears (confirmed against `D:\models\r4dx\qwen38-27b.r4dx`'s own
tensor names), so this is a precision/throughput interim choice, not a missing-data bug -- 16 of
64 layers' attention projections run at full bf16 precision under every `--layout`, which likely
also explains part of why the quantized layouts' generated text stays as fluent as bf16's above
despite the ~7-13% per-GEMM error measured on their GDN/MLP/lm_head linears.

## Open issues

- `tools/reference/first_token.py` was written but not executed (see "Correctness evidence") --
  still not run in the review-fix pass either (same time-budget reasoning; the layer goldens plus
  the new prefill/decode equivalence check are the correctness evidence for this pass).
- attn.qg/o run at bf16 regardless of `--layout` (see "Known limitation" above) -- dedupe/extend
  `AttentionLayer` to accept `QuantLinear` is still future work, not addressed by this pass.
- bf16 uses essentially 100% of the R9700's 31.86 GiB VRAM at `--max-ctx 512`; a longer bf16
  conversation needs a smaller `--max-ctx` still, or more VRAM. The three quantized layouts have
  ample headroom at the CLI's 131072-token default.
- `docs/container-format.md`'s KV descale table note (still referencing a "placeholder 1.0" from
  an earlier stage) was not touched here -- KV descales ARE real per the CONVERSION stage's
  `--kv-calib` run baked into this container; whoever owns that doc should update it (same
  open issue the CONVERSION stage already flagged).
- Decode throughput is still fundamentally limited by the interim skinny-GEMM chunked path
  (`docs/architecture.md` "Interim chunked prefill"): the review-fix pass raised it to ~25-29 tok/s
  for the quantized layouts (from ~16-18 tok/s) and left bf16 essentially unchanged at ~1.4 tok/s
  by removing per-token host syncs and allocations, but did not add a real `(N,K,M-band)` GEMM
  tuning table (`linear.h`'s `PickTuning` is still a single hardcoded-safe tuple) or a dedicated
  prefill kernel -- both remain explicitly future work per `docs/architecture.md`.
- `AttentionLayer::Forward`'s `pos_ids`/`slot_mapping` consolidation (both are now the single
  `positions` array the caller uploads once per chunk, since this cache's contiguous block table
  makes `slot_mapping[t] == pos_ids[t]` always) is specific to the single-sequence, contiguous-
  block-table scope this component already documents; a future multi-sequence/non-contiguous
  paging layer would need to reintroduce a separate slot_mapping.
- This pass's remaining two review findings were left as documented, not fixed: (1) `Container::
  Load` now prints a one-line stderr note when `--layout != bf16` (the minor finding's fix) but the
  underlying attn.qg/o-always-bf16 limitation itself is unchanged (see above); (2) `src/cli/main.cpp`
  degrades to a full model reload + re-prefill on a chat-template prefix mismatch instead of
  `std::exit(1)` (the minor finding's "at minimum" fix), rather than the fuller fix of carrying
  raw generated token ids through `messages` instead of re-tokenizing decoded text.

## MTP self-speculative decode (2026-09-19, MTP pass; corrected 2026-09-19, review-fix pass)

Full writeup, design, the incident this corrects, and known gaps: `docs/mtp.md`. **The numbers
below supersede this section's original MTP-pass numbers** -- an Opus review found two
blocker-severity bugs (a swapped `fc` input concat order, and an MTP KV cache reset to empty every
round instead of built in lockstep with the real sequence) that explained the original pass's
0-1.2% acceptance and "net slowdown" conclusion; both are now fixed (`docs/mtp.md`'s "Incident"
section has the full writeup). Summary here for cross-reference with the tables above: `--mtp
{0,1,2,3,4}`, real 64-layer container (`D:\models\r4dx\qwen38-27b.r4dx`, which already carries
`mtp.*` weights in all four layouts -- no reconversion needed), greedy, this file's own
prompt/settings:

| Layout | mtp=0 tok/s | mtp=1 (accept%) | mtp=2 (accept%) | mtp=3 (accept%) | mtp=4 (accept%) | best-K speedup |
|---|---|---|---|---|---|---|
| w4a16 | 32.55 | 51.49 (75.0%) | 58.80 (58.3%) | **65.56 (54.3%)** | 57.80 (39.4%) | +101.4% (K=3) |
| w4a8  | 30.73 | 47.42 (68.8%) | 47.68 (42.0%) | 47.20 (32.5%) | **51.12 (31.2%)** | +66.4% (K=4) |
| mxfp4 | 26.87 | 39.11 (61.1%) | **48.52 (56.1%)** | 47.05 (41.9%) | 45.86 (33.6%) | +80.6% (K=2) |
| bf16 (K=3 only) | 1.41 | -- | -- | **2.21 (48.7%)** | -- | +56.7% (K=3) |

**MTP now more than doubles decode throughput on this checkpoint for the best-tuned K per layout**
-- a complete reversal of the original pass's conclusion, which was an artifact of the two bugs
above, not a property of this checkpoint's MTP module. Generated text was byte-identical to `--mtp
0`'s own output for every configuration measured (see `docs/mtp.md`'s "Correctness" section for why
this differs from the original pass's report of occasional late divergence: fewer verify rounds run
for the same generation length now that acceptance is high, so there are correspondingly fewer
opportunities for the underlying floating-point-non-associativity effect to flip a near-tie
argmax). Correctness (verify-step math, state rewind on rejection, and now MTP's own KV-cache
lockstep) is independently validated by `tests/model/test_mtp.cpp` (`ctest`: 30/30 passing).
