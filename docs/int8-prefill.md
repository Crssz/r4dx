# int8 activations in prefill: the accuracy cost, measured before any int8 GEMM is written

Status (2026-10-07, branch `int8q`): the activation switch `R4DX_FAKEQ_ACT` is written, its CPU tests pass and
its GPU runs are in the results below. The weight switch `R4DX_FAKEQ_W` (section "The weight side") is written,
built and its CPU tests pass; its kernels and their bit-test are built and NOT run (CPU-only session), and its
GPU runs are PENDING (the commands and an empty results table are in that section).

The question: a trellis linear's GEMM (docs/trellis-kernel.md, docs/trellis-m256.md) multiplies an f16
activation tile, the input after the 128-block Hadamard rotation, by on-the-fly decoded weights on f16 WMMA.
An int8 prefill GEMM would feed int8 activations instead. What do the int8 activations cost in accuracy? The
paper estimate for int8 with one scale per 128-block on rotated inputs is +0.001 to +0.003 mean KL. Our old
w4a8 (one scale per row, no rotation) doubled the KL. This branch measures it with a switch that rounds the
tile to int8 and back and runs the unchanged f16 GEMM. It measures the activation rounding only.

## The switch

`R4DX_FAKEQ_ACT`, read once per process in `Model::Load` (an unrecognized value throws there, before the weights
are read, so a typo cannot run an unquantized experiment) and logged once on stderr when active:

| value | scale group |
|---|---|
| unset, empty, `off` | nothing: the code path, the launches and the buffers are main's |
| `row` | one scale per row of the operand (K columns) |
| `blk128` | one scale per row x 128-column block (the rotation's own block) |
| `blk32` | one scale per row x 32-column block |

For each group, in fp32: `s = max|x| / 127`, `q = clamp(rint(x / s), -127, 127)` (`rint` is round half to
even), `v = q * s` rounded to fp32, then rounded to f16 and written back in place. An all-zero group gives
zeros. A group with a non-finite element is left untouched (an f16 A only gets one through an upstream bug;
the diagnostic must not hide it). The product `q * s` is kept as a separate fp32 rounding ahead of the f16
conversion, as the input transform does (`-ffp-contract=off`, an empty asm barrier).

Where: `ApplyLinear` (src/model/linear.cpp), for a trellis linear, right before the GEMM, on the exact buffer
the GEMM reads: the transform's own scratch, or the `PreQuantizedActivation` a fused producer wrote (the
shared qg / k / v and in_proj_qkv / z transforms, the silu_mul producer for mlp.down, the gate-mul producer
for attn.o). Every linear has a buffer of its own (the transform is per linear, it carries the linear's suh),
so rewriting it in place changes nothing another linear reads. Both the M = 256 launch and the 64-row slices
are covered, and the second part of mlp.gate_up (`a1`) is rounded like the first. The rotation is never fused
into the GEMM (the operand always exists in memory), so no separate rotate path was needed. The kernel is
`r4dx_fake_quant_act_f16` (src/kernels/src/trellis_transform.hip, declared in kernels.h): wave32 per
(row, 128-block) for `blk128` / `blk32`, 256 threads per row for `row`. One extra launch per part per
linear, only when the switch is on. The CPU reference and the parser are the header-only
src/model/fake_quant_act.h.

Scope, by construction: `Model::RunChunk` opens a `ScopedFakeQuantAct` around the layer loop when
`is_prefill_path` is true (every prompt-prefill chunk: 256-row super-chunks, 64-row chunks, tails of any
length, one-token `Prefill` calls, `PrefillMultimodal`'s chunks) and ends it right after the layer loop;
`PrefillProfiled` opens it per chunk. The scope is a thread-local mode that `ApplyLinear` reads, so:

| not touched | why |
|---|---|
| decode, `DecodeStep*`, `VerifyWindow` (MTP / DFlash verify) | not `is_prefill_path` |
| the MTP head's priming, the DFlash drafter's injection, the lm_head | after the scope ends (and none is a trellis linear of the backbone) |
| vision tower | outside `RunChunk` |
| in_proj_a / in_proj_b, bf16-layout attention k / v | `Layout::kBf16`, no A tile |
| the Gemma 4 models | their own `RunChunk`; not wired |

With the switch unset `RunChunk` writes one thread-local int (0) and `ApplyLinear` tests it: no kernel is
launched, no buffer changes, the bits are main's. (To be confirmed on the GPU by step a below.)

## An important property of the Rung-4 harness

`kl_rung4.ps1` runs `tool_teacher_forced_logprobs` with no `--tail-rows`: the uniform pass, `Prefill({ids[0]})`
and then **`DecodeStep` for every later token** (tests/model/teacher_forced.h). So the frozen numbers (mean KL
0.00788, top-1 95.70%) are the DECODE path's, one row at a time, and a prefill-only switch changes nothing in
that run except the first token. The statement that this tool "runs the prompt-processing path only" is true
only with `--tail-rows`, `--tail-path prefill`.

For the measurement, rows must go through the prefill path. The cheapest way that still yields a full
`[T-1, V]` dump (so `kl_report.py` scores it against the bf16 reference, `kl_rung4.ps1` unchanged otherwise) is

    --tail-rows 1023 --tail-path prefill

on the 1024-token canon segments: `Prefill(ids[0..1))`, then every later token as a one-token `Prefill` call,
all rows quantized. `kl_rung4.ps1` gained `-ToolArgs` (default none; the command line is unchanged) to pass
this. What this path does NOT exercise: the M = 256 / M = 64 GEMM kernels (a one-token prefill call runs the
decode-shaped trellis GEMM) and the chunked GDN scan. The A operand, which is what the switch rounds, is the
same f16 tile whatever M is, and the rounding is per row, so the activation cost is the same. The GEMM's
accumulation noise is not. A one-token-at-a-time prefill is also a different path from the decode one, so it
has its own baseline (step a2): read every run against an unquantized run of the same arguments.

An optional second family exercises the real chunked path (steps e): `tools/prefill/run_kl.ps1` on the same
canon tokens (`--tail-rows 256`: a 768-token prefix through three 256-row super-chunks, the last 256 rows
through `DecodeStep`), compared with `tools/prefill/kl_compare.py`. There only the last prefix row is a
quantized prefill row; the other 255 rows see the quantization through the KV cache and the GDN state. It
measures how the prefill's error propagates into generation.

## Commands (HIP device 1, one GPU job at a time)

The 10.1.0 baseline `E:\models\r4dx\rocm1010\kl` is mean KL 0.007877, top-1 95.699%, which round to the frozen
0.00788 / 95.70, so step a needs no `-ExpectKl`. Run them from PowerShell, in this order. Outputs go under
`E:\models\r4dx\int8q\`. The backtick before `$` keeps the calling shell from expanding it; the variable lives
and dies in the child `powershell`, so it cannot leak. The tool and the script are this worktree's
(`kl_rung4.ps1` resolves its tokens relative to its own repo, and this copy has `-ToolArgs`).

```
# 0. the kernel against its CPU reference (about a second)
powershell -NoProfile -Command "`$env:HIP_VISIBLE_DEVICES='1'; & C:\Users\pay20\dev\r4dx-int8q\build\win-hip\tests\kernels\test_fake_quant_act.exe; exit `$LASTEXITCODE"
# a. switch unset, the Rung-4 uniform pass, gated: byte-identical to rocm1010\kl (bits are main's)
powershell -NoProfile -Command "& C:\Users\pay20\dev\r4dx-int8q\tools\quant2\kl_rung4.ps1 -Device 1 -Tool C:\Users\pay20\dev\r4dx-int8q\build\win-hip\tests\model\tool_teacher_forced_logprobs.exe -OutDir E:\models\r4dx\int8q\kl-off -CompareDir E:\models\r4dx\rocm1010\kl; exit `$LASTEXITCODE"
# a2. switch unset, every row through a one-token Prefill call: the baseline of the measurement
powershell -NoProfile -Command "& C:\Users\pay20\dev\r4dx-int8q\tools\quant2\kl_rung4.ps1 -Device 1 -Tool C:\Users\pay20\dev\r4dx-int8q\build\win-hip\tests\model\tool_teacher_forced_logprobs.exe -OutDir E:\models\r4dx\int8q\kl-off-pfx -NoGate -CompareDir '' -ToolArgs '--tail-rows','1023','--tail-path','prefill'; exit `$LASTEXITCODE"
# b. blk128, c. row, d. blk32: the same, switch on
powershell -NoProfile -Command "`$env:R4DX_FAKEQ_ACT='blk128'; & C:\Users\pay20\dev\r4dx-int8q\tools\quant2\kl_rung4.ps1 -Device 1 -Tool C:\Users\pay20\dev\r4dx-int8q\build\win-hip\tests\model\tool_teacher_forced_logprobs.exe -OutDir E:\models\r4dx\int8q\kl-blk128 -NoGate -CompareDir '' -ToolArgs '--tail-rows','1023','--tail-path','prefill'; exit `$LASTEXITCODE"
powershell -NoProfile -Command "`$env:R4DX_FAKEQ_ACT='row'; & C:\Users\pay20\dev\r4dx-int8q\tools\quant2\kl_rung4.ps1 -Device 1 -Tool C:\Users\pay20\dev\r4dx-int8q\build\win-hip\tests\model\tool_teacher_forced_logprobs.exe -OutDir E:\models\r4dx\int8q\kl-row -NoGate -CompareDir '' -ToolArgs '--tail-rows','1023','--tail-path','prefill'; exit `$LASTEXITCODE"
powershell -NoProfile -Command "`$env:R4DX_FAKEQ_ACT='blk32'; & C:\Users\pay20\dev\r4dx-int8q\tools\quant2\kl_rung4.ps1 -Device 1 -Tool C:\Users\pay20\dev\r4dx-int8q\build\win-hip\tests\model\tool_teacher_forced_logprobs.exe -OutDir E:\models\r4dx\int8q\kl-blk32 -NoGate -CompareDir '' -ToolArgs '--tail-rows','1023','--tail-path','prefill'; exit `$LASTEXITCODE"
```

Check in each run's `tool.log` that stderr carries the `R4DX_FAKEQ_ACT=...` line (b, c, d) and not (a, a2).
Step a's gate also passes only if all four segments print `byte-identical`.

CPU analysis afterwards (`KL(off || on)` on the same rows, no bf16 reference needed; the per-segment and
overall rows are the table below):

```
C:\Users\pay20\AppData\Local\Programs\Python\Python312\python.exe C:\Users\pay20\dev\r4dx-int8q\tools\prefill\kl_compare.py --ref E:\models\r4dx\int8q\kl-off-pfx --test E:\models\r4dx\int8q\kl-blk128 --tokens C:\Users\pay20\dev\r4dx-int8q\tools\reference\kl_corpus\tokens_canon.json
```

(and `kl-row`, `kl-blk32`).

Optional, the chunked path itself (real 256-row super-chunks and the M = 256 GEMM; see above for what it
measures): `run_kl.ps1` takes the segment tail from the tokens file (256 by default) and leaves the switch to
the environment.

```
powershell -NoProfile -Command "& C:\Users\pay20\dev\r4dx-int8q\tools\prefill\run_kl.ps1 -Device 1 -Tool C:\Users\pay20\dev\r4dx-int8q\build\win-hip\tests\model\tool_teacher_forced_logprobs.exe -Tokens C:\Users\pay20\dev\r4dx-int8q\tools\reference\kl_corpus\tokens_canon.json -OutDir E:\models\r4dx\int8q\kl-chunk-off; exit `$LASTEXITCODE"
powershell -NoProfile -Command "`$env:R4DX_FAKEQ_ACT='blk128'; & C:\Users\pay20\dev\r4dx-int8q\tools\prefill\run_kl.ps1 -Device 1 -Tool C:\Users\pay20\dev\r4dx-int8q\build\win-hip\tests\model\tool_teacher_forced_logprobs.exe -Tokens C:\Users\pay20\dev\r4dx-int8q\tools\reference\kl_corpus\tokens_canon.json -OutDir E:\models\r4dx\int8q\kl-chunk-blk128; exit `$LASTEXITCODE"
```

then `kl_compare.py --ref ...\kl-chunk-off --test ...\kl-chunk-blk128 --tokens <tokens_canon.json>`.

Per step, the number that matters is `KL(on) - KL(off, same path)` from the `[kl_rung4]` line, and, more
sensitive, `kl_compare.py --ref kl-off-pfx --test kl-blk128` (KL(off || on) over the same rows, which does not
need the bf16 reference at all).

Expected wall time: a is about 2 min (decode, 29.5 ms/row on device 1 for the 10.1.0 baseline); the
prefill-path runs process 4092 one-token prefill calls each and are slower (guess: 4-6 min).

## Results (2026-10-07, ROCm 10.1.0, Huihui trellis mix4.5m)

Mean KL against the bf16 reference (kl_rung4.ps1 `[kl_rung4]` line), top-1 in %, per segment mean KL
(cpp / english / python / thai). kl-off, kl-off-pfx and kl-blk128 ran on HIP device 1, kl-row and kl-blk32
on device 0 (same card model, deterministic kernels):

| run | R4DX_FAKEQ_ACT | path | mean KL | top-1 | cpp | english | python | thai |
|---|---|---|---|---|---|---|---|---|
| kl-off | unset | decode (uniform pass) | 0.00788 | 95.70 | 0.00583 | 0.00975 | 0.00778 | 0.00814 |
| kl-off-pfx | unset | one-token prefill | 0.00751 | 95.72 | 0.00592 | 0.00844 | 0.00754 | 0.00815 |
| kl-blk128 | blk128 | one-token prefill | 0.00790 | 95.92 | 0.00593 | 0.00962 | 0.00768 | 0.00836 |
| kl-row | row | one-token prefill | 0.00834 | 95.77 | 0.00645 | 0.00939 | 0.00832 | 0.00920 |
| kl-blk32 | blk32 | one-token prefill | 0.00758 | 95.92 | 0.00596 | 0.00824 | 0.00784 | 0.00828 |

kl-off is byte-identical to the ROCm 10.1.0 main baseline (all four segments), so the unset switch is
main's path. Against kl-off-pfx: blk32 +0.00007, blk128 +0.00039, row +0.00083.

`KL(off || on)` from `kl_compare.py` (mean / p99 / max, top-1 agreement), 4092 rows:

| on | mean KL | p99 | max | top-1 agreement |
|---|---|---|---|---|
| blk128 | 0.001301 | 0.006835 | 1.1173 | 98.68 |
| row | 0.001775 | 0.011269 | 0.5361 | 98.34 |
| blk32 | 0.000969 | 0.006301 | 0.0152 | 99.00 |

The blk128 max (1.12, english_prose) is one row whose top token flipped; its mean without that row is in
line with the other segments. All three are below the paper estimate (+0.001 to +0.003 against the
reference): int8 activations with per-128-block scales cost about +0.0004 KL, per-32-block about nothing.
The weight side (below) is still unmeasured.

Reading it: a mean-KL increase of 0.001 to 0.003 for `blk128` confirms the paper estimate and makes an int8
prefill GEMM worth building; `row` is expected to be clearly worse (the w4a8 lesson). The budget the
production container has is the gap between its KL and the 0.01 gate (docs/gemma4-plan.md, docs/quant2.md).

## The weight side: R4DX_FAKEQ_W

The other half of an int8 x int8 prefill GEMM. A trellis weight is decoded in the GEMM from its code to an f16
codebook value, which is not on an int8 grid. `R4DX_FAKEQ_W` rounds each DECODED weight to a symmetric int8 grid
before it enters the WMMA, in prefill only, and runs the rest of the GEMM unchanged (f16 WMMA, fp32 accumulate,
the same FWHT / svh / bf16 epilogue). It measures the weight rounding only; with `R4DX_FAKEQ_ACT` on as well the
numerics are the full int8 x int8 ones except the accumulation (int32 in a real kernel).

### What is rounded

Q is the decoded weight in its regularized domain, Q[K][N] (the values the WMMA multiplies; W^T = diag(suh) H Q
H diag(svh)), f16. A scale group is one OUTPUT COLUMN n of Q over a block of k:

| value | scale group | scale table |
|---|---|---|
| unset, empty, `off` | nothing: the shipped kernels, bit for bit | none |
| `col128` | (column n, 128 k): the rotation's own block, matches `R4DX_FAKEQ_ACT=blk128` | fp32 [K/128][N], K * N / 32 bytes per linear |
| `col32` | (column n, 32 k): matches `blk32` | fp32 [K/32][N], 4x that |

Per group, in fp32: `s = max|w| / 127` (1.0f for an all-zero group), `rs = 1 / s` (IEEE division),
`q = clamp(rint(w * rs), -127, 127)` (`rint` is round half to even), `v = q * s` (the product is rounded to fp32
first, then once to f16). src/model/fake_quant_w.h is the CPU transcription (`FakeQuantWScaleRef`,
`FakeQuantWRoundRef`, `FakeQuantWTableRef`, `FakeQuantWApplyRef`); tests/model/test_fake_quant_act_cpu.cpp
tests the parser and the reference (hand values, symmetry, the half-s error bound, the grid, the table layout).
Read once per process in `Model::Load` (an unrecognized value throws before the weights are read), one line on
stderr when active (it prints the table's VRAM). Combine with `R4DX_FAKEQ_ACT` freely; they are independent
thread-local scopes opened around the same layer loop, `Model::RunChunk` with `is_prefill_path`, so decode,
`DecodeStep*`, `VerifyWindow`, the MTP / DFlash heads, the lm_head and the vision tower never see either
(`PrefillProfiled` opens both per chunk; the Gemma 4 models are not wired).

### Design

* A scale table per trellis linear, built once at `Model::Load` (after the weights are resident, only when the
  switch is on) by `Container::BuildTrellisWScales` -> `BuildTrellisWScale` (src/model/linear.cpp) ->
  libr4d's `r4d_trellis_wscale_f32`. The builder decodes with the GEMM's own functions
  (`r4d_trellis_k4_decode` / `k5_decode`, r4d_trellis_dq.h), one wave per (tile pair, group) with a lane xor 16
  joining the two k halves of a fragment, so the amax is taken over exactly the f16 values the WMMA sees.
  Memory at `col128`: K * N / 32 bytes per linear, about 0.8 GB for the model (weights are 16.3 GiB on a 32 GB
  card; `col32` is 4x that, about 3.4 GB; the Load line prints the measured figure, and the VRAM breakdown line
  books it under `weights=`). The table is
  `QuantLinear::trellis_wscale`, empty when the switch is off.
* The rounding happens IN the GEMM kernels, on the decoded fragment, before the WMMA. A decoded fragment (f0 or
  f1) is one lane's eight k of ONE column of Q, so one scale serves all eight; the lane loads its two scales
  per decoded (tile pair, k-tile) and pays about one division and 8 x (mul, rndne, med3, mul, cvt) per fragment
  on top of the 62-VALU decode, so the K loop is several times longer (by instruction count; not timed). This
  is a measurement tool, not a fast kernel.
  `r4d_trellis_wq_round` (r4d_trellis_dq.h) is the one function; the product barrier is an empty asm because the
  backend otherwise folds `fptrunc(q * s)` into a single-rounding `v_fma_mix_f16` (seen in the first listing),
  which is not the fp32 product followed by the f16 conversion the CPU reference does. The listing now has
  `v_mul_f32` + `v_cvt_f16_f32`.
* Why not decode to a dense f16 matrix, round it, and run a dense GEMM: it needs a dense f16 GEMM with the
  trellis epilogue (none exists), and a different summation order, which would add noise to every comparison
  against the switch-off baseline. Rounding inside the trellis kernel keeps (SK, SKG, Wc) and the epilogue
  identical, so on vs off differs by the weight rounding alone.
* What a real int8 kernel would have to do is the same: decode, multiply by a precomputed `1 / s`, round, with
  `s` from a table of the size above (or a pass over the k block); an on-the-fly amax per 128-k group would need
  the whole group decoded before the first WMMA.

### Which kernels, by measurement path

| measurement | path | kernels that run rounded |
|---|---|---|
| `kl_rung4.ps1 ... -ToolArgs '--tail-rows','1023','--tail-path','prefill'` | every row a one-token `Prefill` call: `RunChunk(is_prefill_path)` -> `ApplyLinear` with M = 1 -> the loop over <= 64-row chunks, one chunk | `r4d_gemm_trellis_nt_m64` with the shipped M = 1 tuning row: the kernel a decode step runs (decode itself is outside the scope) |
| `tools/prefill/run_kl.ps1` (a 768-token prefix through the chunked path, then the tail) | a full 256-row super-chunk when `R4DX_PREFILL_CHUNK=256` and the linear has an M = 256 plan (check the `prefill chunk:` stderr line); every other chunk and a linear without a plan: 64-row slices | `r4d_gemm_trellis_nt_m256` (the decoding wave rounds before it publishes the fragment in LDS, so all four row groups see rounded weights) and `r4d_gemm_trellis_nt_m64` for the slices, the < 256-row tail chunk (M = 1..64) and the per-linear fallback |
| not scoped | decode, verify windows, MTP / DFlash, lm_head, vision | the shipped kernels |

Every instantiation of both kernels has a rounding variant (KB 4 and 5; every (NP, U, MT) of the M <= 64 kernel,
49 of them since NT does not select a kernel in the rounding unit; every configuration of the M = 256 table, 22).
`NT` (non-temporal weight loads) is a cache hint; the rounding kernels always use temporal loads.

### OFF path: identical code, and how that was checked

Switch unset: the model never builds a table, never opens a non-zero scope, and `ApplyLinear` takes the shipped
launches (one thread-local int test). In libr4d the rounding is a template parameter that defaults to false
(`bool WQ = false` and a trailing parameter pack that is empty, so the shipped signature and mangling prefix are
as before), with every use under `if constexpr (WQ)`, and the rounding instantiations live in two units of their
own, `r4d_gemm_trellis_nt_m64_wq.hip` and `r4d_gemm_trellis_nt_m256_wq.hip`, which `#include` the shipped
sources with `R4D_TQ_WQ_UNIT` / `R4D_T256_WQ_UNIT` defined (the shipped entry points are compiled out there and
only the rounding launches in).
Two attempts that did not hold, kept because the check caught them: moving the body into a `__device__` function
called by two kernels changed 98 of 98 shipped M <= 64 kernels; and keeping the rounding variant in the SAME unit
as the shipped kernels changed every one of the 98 by a few instructions (the epilogue's lane compare), although
the rounding variant is a different function. Hence the separate units.

Checked with `hipcc -S` listings (the unit's own flags, exactly what `third_party/CMakeLists.txt` builds), from
the sources at `3685ccc` and from this branch: the shipped M <= 64 unit (98 kernels + 2 reconstruct) and the
M = 256 unit (22 kernels), instruction text with comments and symbol names normalised, per kernel, and the whole
listing (instructions and code-object metadata: `.vgpr_count`, scratch, kernarg sizes) as a sorted multiset of
lines: 0 differences, both units. `tools/reference/compare_isa_listings.ps1 <before.s> <after.s>` repeats that.
The build's own checks hold: `check_trellis_isa` on the shipped units (max 189 / 188 VGPRs, no scratch, no
spills, 0 near dependencies) and `check_qwen_attn_isa` (baseline hash unchanged). The rounding units get the
same script with `-DWQ=ON`: near dependencies gated, resources reported (49 M <= 64 rounding kernels, max 192
VGPRs, one with 88 bytes of scratch (KB 5, NP 2, U 2, MT 3); 22 M = 256 kernels, max 191 VGPRs, no scratch).
A spilling kernel is slower, not wrong.

### Commands (device 1 or 0; outputs under `E:\models\r4dx\int8q\`; one GPU job at a time)

Build: `cmake --build build\win-hip --target tool_teacher_forced_logprobs test_fake_quant_w -j 12` (done; the
exes below are this worktree's). All of these were NOT run by the session that wrote them.

```
# w0. the kernels against the CPU reference: scale table, one-hot rows of Q' through every instantiation, the
#     M = 256 kernel vs four 64-row launches on the model's linear classes, the whole linear (a minute at most)
powershell -NoProfile -Command "`$env:HIP_VISIBLE_DEVICES='1'; & C:\Users\pay20\dev\r4dx-int8q\build\win-hip\tests\kernels\test_fake_quant_w.exe; exit `$LASTEXITCODE"
# w1. switch unset: byte-identical to rocm1010\kl (all four segments must print byte-identical)
powershell -NoProfile -Command "& C:\Users\pay20\dev\r4dx-int8q\tools\quant2\kl_rung4.ps1 -Device 1 -Tool C:\Users\pay20\dev\r4dx-int8q\build\win-hip\tests\model\tool_teacher_forced_logprobs.exe -OutDir E:\models\r4dx\int8q\kl-w-off -CompareDir E:\models\r4dx\rocm1010\kl; exit `$LASTEXITCODE"
# w2. weights only, col128, on the one-token prefill path (baseline: kl-off-pfx)
powershell -NoProfile -Command "`$env:R4DX_FAKEQ_W='col128'; & C:\Users\pay20\dev\r4dx-int8q\tools\quant2\kl_rung4.ps1 -Device 1 -Tool C:\Users\pay20\dev\r4dx-int8q\build\win-hip\tests\model\tool_teacher_forced_logprobs.exe -OutDir E:\models\r4dx\int8q\kl-wcol128 -NoGate -CompareDir '' -ToolArgs '--tail-rows','1023','--tail-path','prefill'; exit `$LASTEXITCODE"
# w3. full w8a8 numerics: col128 + blk128
powershell -NoProfile -Command "`$env:R4DX_FAKEQ_W='col128'; `$env:R4DX_FAKEQ_ACT='blk128'; & C:\Users\pay20\dev\r4dx-int8q\tools\quant2\kl_rung4.ps1 -Device 1 -Tool C:\Users\pay20\dev\r4dx-int8q\build\win-hip\tests\model\tool_teacher_forced_logprobs.exe -OutDir E:\models\r4dx\int8q\kl-wcol128-ablk128 -NoGate -CompareDir '' -ToolArgs '--tail-rows','1023','--tail-path','prefill'; exit `$LASTEXITCODE"
# w4. the finer bound: col32 + blk32
powershell -NoProfile -Command "`$env:R4DX_FAKEQ_W='col32'; `$env:R4DX_FAKEQ_ACT='blk32'; & C:\Users\pay20\dev\r4dx-int8q\tools\quant2\kl_rung4.ps1 -Device 1 -Tool C:\Users\pay20\dev\r4dx-int8q\build\win-hip\tests\model\tool_teacher_forced_logprobs.exe -OutDir E:\models\r4dx\int8q\kl-wcol32-ablk32 -NoGate -CompareDir '' -ToolArgs '--tail-rows','1023','--tail-path','prefill'; exit `$LASTEXITCODE"
# w5 (optional). the real chunked path, M = 256 kernel: weights only, same form as the existing kl-chunk-off / kl-chunk-blk128
powershell -NoProfile -Command "`$env:R4DX_FAKEQ_W='col128'; & C:\Users\pay20\dev\r4dx-int8q\tools\prefill\run_kl.ps1 -Device 1 -Tool C:\Users\pay20\dev\r4dx-int8q\build\win-hip\tests\model\tool_teacher_forced_logprobs.exe -Tokens C:\Users\pay20\dev\r4dx-int8q\tools\reference\kl_corpus\tokens_canon.json -OutDir E:\models\r4dx\int8q\kl-chunk-wcol128; exit `$LASTEXITCODE"
```

Check in each run's `tool.log` (`tool.err.log` for w5: `run_kl.ps1` splits stderr into it) that stderr carries the
`R4DX_FAKEQ_W=...: scale tables built` line (w2 - w5) and
`R4DX_FAKEQ_ACT=...` where set (w3, w4), and neither in w1. The one-token path runs the rounded M = 1 kernel
4092 times per segment set and is slower than the activation runs (the K loop is several times longer; guess
10 - 15 minutes). Compare against the unquantized one-token baseline `kl-off-pfx`:

```
C:\Users\pay20\AppData\Local\Programs\Python\Python312\python.exe C:\Users\pay20\dev\r4dx-int8q\tools\prefill\kl_compare.py --ref E:\models\r4dx\int8q\kl-off-pfx --test E:\models\r4dx\int8q\kl-wcol128 --tokens C:\Users\pay20\dev\r4dx-int8q\tools\reference\kl_corpus\tokens_canon.json
```

(and `kl-wcol128-ablk128`, `kl-wcol32-ablk32`; for w5 `--ref E:\models\r4dx\int8q\kl-chunk-off --test ...\kl-chunk-wcol128`).

### Results (2026-10-07, ROCm 10.1.0)

Mean KL against the bf16 reference (`[kl_rung4]` line) and top-1 in %; the baseline is `kl-off-pfx` (0.00751,
95.72). Activations alone (from the table above): blk128 +0.00039, blk32 +0.00007. kl-w-off, kl-wcol128 and
kl-wcol128-ablk128 ran on HIP device 1, the col32 runs on device 0. test_fake_quant_w passed (361 checks) first.

| run | R4DX_FAKEQ_W | R4DX_FAKEQ_ACT | path | mean KL | top-1 | cpp | english | python | thai |
|---|---|---|---|---|---|---|---|---|---|
| kl-w-off | unset | unset | decode (uniform pass), gate vs rocm1010 | 0.00788 | 95.70 | 0.00583 | 0.00975 | 0.00778 | 0.00814 |
| kl-wcol128 | col128 | unset | one-token prefill | 0.00771 | 95.87 | 0.00605 | 0.00880 | 0.00757 | 0.00841 |
| kl-wcol32 | col32 | unset | one-token prefill | 0.00764 | 95.75 | 0.00594 | 0.00851 | 0.00781 | 0.00831 |
| kl-wcol128-ablk128 | col128 | blk128 | one-token prefill | 0.00807 | 95.75 | 0.00625 | 0.00966 | 0.00776 | 0.00859 |
| kl-wcol32-ablk32 | col32 | blk32 | one-token prefill | 0.00820 | 95.58 | 0.00600 | 0.01045 | 0.00787 | 0.00847 |

kl-w-off printed byte-identical on all four segments: the unset switch is main's path. Against kl-off-pfx:
col128 +0.00020, col32 +0.00013, col128 + blk128 +0.00056, col32 + blk32 +0.00069.

`KL(off || on)` from `kl_compare.py` against `kl-off-pfx`, 4092 rows (activation-only rows repeated for
comparison):

| on | mean KL | median | p99 | max | top-1 agreement |
|---|---|---|---|---|---|
| blk128 (A only) | 0.001301 | 0.000460 | 0.006835 | 1.1173 | 98.68 |
| col128 (W only) | 0.001272 | 0.000522 | 0.007693 | 0.4702 | 98.63 |
| col32 (W only) | 0.001058 | 0.000468 | 0.007136 | 0.0412 | 98.97 |
| col128 + blk128 | 0.001616 | 0.000636 | 0.009445 | 1.2832 | 98.63 |
| col32 + blk32 | 0.001577 | 0.000534 | 0.007402 | 1.8010 | 98.53 |

The means are pulled by single flipped rows in english_prose (max 1.28 and 1.80; one row of 1.8 adds 0.0018
to that segment's mean). The medians are the steadier read: activations and weights each move the
distribution by about the same small amount, and w8a8 at 128 is a little worse than either alone, not their
sum. col32 + blk32 is not better than col128 + blk128 against the reference (0.00820 vs 0.00807); with these
few flips the difference is noise, so finer groups buy nothing measurable. Chunked path, activations only
(`run_kl.ps1`, 768-token prefix through the M = 256 kernel, 256 decode rows scored): blk128 0.00089, blk32
0.00094 KL(off || on), top-1 agreement 99.12 / 99.22 %.

Verdict: full int8 x int8 numerics with per-128 scales cost about +0.0006 mean KL against the reference on
the prefill path (0.00751 -> 0.00807), under the paper estimate, and leave the production container well
inside the 0.01 budget. Accuracy is not the obstacle to an int8 prefill GEMM; its speed is the open question.
That question has its own bench-only prototype now, with a go / no-go rule fixed in advance: docs/int8-gemm-proto.md
(branch `int8gemm`; the GPU runs are pending).

Reading it: weights and activations roughly add if their errors are independent; a w8a8 total under the
activation-only +0.0004 plus a similar weight term keeps `col128 + blk128` well inside the production budget
(the gap to the 0.01 gate, docs/gemma4-plan.md, docs/quant2.md). A weight term far above the activation one would
say the int8 grid of a trellis weight is the obstacle and not the activations.

### Weight-side caveats

* Accumulation is f16 WMMA with fp32 accumulate, not int32; a real kernel rescales each k group's integer
  partial sum (the activation's 128-block scale times the weight's column scale).
* The scale is exact fp32 here; a stored table would be f16 or a power of two, which adds error. `col32` is the
  bound on what finer scales recover.
* Under tensor parallel a rank decodes its own shard; the table is built per rank from its own words and a group
  never crosses a shard (K and N shards are whole 128-blocks), but TP is not wired or tested for this switch:
  measure at TP = 1.
* The one-token path runs the M <= 64 kernel at M = 1 only; the M = 256 kernel's rounding is covered by the
  optional w5 run and, bit for bit, by `test_fake_quant_w` (which compares it with four 64-row launches).

## Caveats

* The weights are unrounded in everything above `R4DX_FAKEQ_W`'s section: those runs measure the activation
  rounding only. An int8 x int8 GEMM also needs int8 weights, and the trellis weights decode to f16 codebook
  values that are not on an int8 grid; the section above measures what putting them on one costs.
* The GEMM still accumulates in f16 WMMA, not int32, so the accumulation noise of a real int8 kernel is not
  modeled; with `blk128` the real kernel would also have to rescale each 128-block's int32 partial sum.
* Round-trip rounding is exact in fp32 here; a real kernel's `s` may be stored in f16 or as a power of two,
  which would add error. `blk32` is the cheap bound on how much finer scales can recover.
* Measure at TP = 1. Under tensor parallel a row-parallel linear (mlp.down, for one) quantizes
  its own rank's K shard, so `row` would take one scale per shard, not per full row (`blk128` / `blk32` are
  unchanged where the shard is whole 128-blocks).
* The one-token prefill path differs numerically from the chunked path (other GEMM and GDN kernels), so its
  baseline is not the frozen 0.00788.
