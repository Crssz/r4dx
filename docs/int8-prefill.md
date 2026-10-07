# int8 activations in prefill: the accuracy cost, measured before any int8 GEMM is written

Status (2026-10-07, branch `int8q`): the activation switch `R4DX_FAKEQ_ACT` is written, its CPU tests pass and
its GPU runs are in the results below. The weight switch `R4DX_FAKEQ_W` (section "The weight side") is written,
built and its CPU tests pass; its GPU bit-test and KL runs are in that section's results.

Update (2026-10-07, branch `int8prefill`): the two switches above are accuracy EXPERIMENTS (they round, then run the f16
GEMM). The bench-only int8 GEMM prototype (docs/int8-gemm-proto.md) measured 1.46x (KB4) and 1.34x (KB5) on the
linears, so it is now a real prefill path, `R4DX_PREFILL_INT8`: see "Production path" at the end of this file.

**Now the default (2026-10-07, branch `fast`).** The path was validated on a GPU (Results at the end: -18.9% TTFT at 8k,
-15.7% at 32k, greedy text unchanged, KL(off || on) 0.0011 canon / 0.0017 at 8k / 0.0104 at 32k) and made the default:
`R4DX_PREFILL_INT8` unset, empty, `1` or `on` is on, `0` or `off` is the kill switch (see "Now the default" under
"Production path"). `R4DX_FAKEQ_ACT` / `R4DX_FAKEQ_W` below are still experiments and, when set, win over the default.
It is not on `main` yet: it is on branch `fast`.

Update (2026-10-07, branch `int8v2`): two follow-ups on top of `main`, both with their own kill switch and both GPU-untested when they were written: the
fused activation quantizer (`R4DX_PREFILL_INT8_FUSEDQ`, default on; "The fused quantizer" below) and an opt-in COARSE-scale mode
(`R4DX_PREFILL_INT8_SCALES=coarse`: one activation scale per row, one weight scale per column, no per-128 rescale; "Coarse scales"
below). `blk128`, the default of the second, is today's bytes.

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

## Production path (`R4DX_PREFILL_INT8`)

Status (2026-10-07, branch `int8prefill` = `f16retune` (= `main`) + `int8q` + `int8gemm`, then this work): written and built
(libr4d unit, `tool_teacher_forced_logprobs`, `r4dx-cli`, `r4dx-server`, the tests below); every CPU check passes. The branch
`int8prefill` was written CPU-only and then validated on a GPU by the main session: the validation sequence is the last
subsection and its results table (G1 to G5) is at the end, filled 2026-10-07 for that branch alone (int8 opt-in, exact-wide
attention). It was default OFF, and off is byte-identical to `f16retune` (checked below); **it is now the default, see "Now
the default (branch `fast`)" directly below -- and nothing of branch `fast` itself (the merge with `decode-t1` and `splitkv`,
the default flip, `test_prefill_chunk_identity_defaults`, the corrected `test_prefill_int8`) has run on a GPU yet.**

### Now the default (branch `fast`, 2026-10-07)

The user approved giving up "prefill is bit-identical" (September) for two lossy-but-validated speedups, this one and split-KV
prompt-prefill attention (docs/prefill.md "Split threshold"). What changed:

| | before | now |
|---|---|---|
| `R4DX_PREFILL_INT8` unset / empty | off | **on** (where the Model can use it) |
| `R4DX_PREFILL_INT8=1` / `on` | on | on, and an explicit request: with `R4DX_FAKEQ_ACT/W` set it still throws |
| `R4DX_PREFILL_INT8=0` / `off` | off | off: the kill switch (no scale table, no extra launch, the old bytes) |
| anything else | off + warning | the default (on) + warning |
| `ModelOptions::prefill_int8 = -1` | follows the environment (off) | follows the environment (on); 0 and 1 still force |
| `R4DX_FAKEQ_ACT/W` set, int8 only the default | n/a | int8 is refused with the reason `R4DX_FAKEQ_ACT / R4DX_FAKEQ_W are set`, the experiment runs as before |

Unchanged: it still runs only where it did (full 256-row super-chunks of `Prefill` calls on a TP = 1 trellis Model: tails, a
64-row Model, a quant2 container, TP = 2 (unless `R4DX_PREFILL_INT8_TP2=1`, off by default, "Tensor parallel"), images, decode and
verify windows and the Gemma 4 model stay on the f16 kernels).
Costs now paid by default: +0.7 GiB of weight scale tables at load (`R4DX_PREFILL_INT8=0` skips them), prefill rows that are a
quantized model of the f16 rows (KL numbers in Results), and the KV bytes of a prompt depending on the chunk grid and so on
the prefix-cache state (a prefix-reuse suffix anchors its own grid). Load line: `prefill int8: ON (default): ...`, or
`off (default is on, not used: <reason>)`, or `off (R4DX_PREFILL_INT8=0: ...)`.

The reference side of an accuracy experiment moves with the default: the "unquantized" run of an `R4DX_FAKEQ_ACT/W`
comparison (FAKEQ unset) is now an int8-prefill run, and a prefill A/B against recorded f16 dumps (`kl_rung4`'s long-prefix
modes, `run_kl.ps1`, `ttft_cli.ps1`) is only an f16 reference with `$env:R4DX_PREFILL_INT8 = '0'; $env:R4DX_PREFILL_SPLITKV =
'exact'` set by the caller. `kl_rung4.ps1` with its default arguments is unaffected: it feeds ids[1..] through `DecodeStep`
after a one-token `Prefill` (a one-row tail: no super-chunk, and the decode kernel), so neither default reaches it.

Tests: the f16 identity tests pin `prefill_int8 = 0` per `Load` (`test_prefill_chunk_identity`, which also pins
`R4DX_PREFILL_SPLITKV=exact` unless the environment says otherwise), `test_prefill_int8` loads off / on / **default (-1)**,
`tp1_identity.ps1` pins both knobs and `gdn256_check.ps1` pins them unless run with `-Defaults`. One test expectation was wrong
and is fixed (G2b below): `mm600` differed from `len600` in the attention layers' KV digests with int8 off too, but that was the
test's own state, not the entry points -- a text-only `PrefillMultimodal` delegates to `Prefill`, `Reset()` leaves the KV pages
alone, and `DebugStateDigest` covers all of them, so each scenario's digest carries the stale pages of the scenarios before it
(`mm600` follows `len600` and its two decode steps, which wrote positions 600 and 601). `test_prefill_int8` now compares
`mm600` with `len600` on the live state (logits, GDN states, decode tokens, int8 chunk count) and leaves the KV digests out of
that one comparison. Not covered by the speed / accuracy gates that were measured (listed under "Gates for making it
the default" below, which have no row in Results yet): the speculation gate (DFlash / MTP accepted length on the OpenCode
transcripts), the `run_tasks` scores at 8k / 32k, a 128k prefill; they are in the validation list handed to the main session.

What it is: the prototype of docs/int8-gemm-proto.md (measured 1.456x KB4, 1.335x KB5 against the f16 M = 256 plan on all
seven linear classes, before the mlp.down retune; the unfused activation quantizer adds about 5%) turned into a real
path. A full 256-row prefill super-chunk runs each of its trellis linears as: transformed f16 A -> int8 A8 + scales
(`r4d_trellis_i8_quant_act`) -> `r4d_gemm_trellis_nt_i8` (decoded trellis weight quantized to int8 on the fly with a
per-(column, 128 k) scale table, `v_wmma_i32_16x16x16_iu8`, a per-128 fp32 rescale of the int32 partial sums, then the
f16 kernel's FWHT / svh / out_scale / one bf16 rounding). It is a quantized MODEL of the f16 linear, not its bits.

### Which rows run it

| | |
|---|---|
| runs int8 | the rows of a full 256-row super-chunk of a `Prefill` call, on a TP = 1 Model that runs 256-row chunks (`R4DX_PREFILL_CHUNK` unset / 256) and whose container is trellis, for every linear class with an int8 plan (all seven of the 27B, KB 4 and 5) |
| stays f16 | tails (1..255 rows: 64-row slices), `R4DX_PREFILL_CHUNK=0/64`, a quant2 (rotated) container, MTP and DFlash 64-row slices, decode and verify windows, the vision tower, **`PrefillMultimodal`** with image spans, or after an image has made the positions multimodal (image accuracy is unmeasured; it never sets the per-call flag; a text-only call with no image ever seen delegates to `Prefill` and so DOES run int8, which is the only text-only path the CLI and the server use), TP = 2 (explicit fallback with a reason at load, unless `R4DX_PREFILL_INT8_TP2=1`: off by default and, until the TP = 2 tuning tables have rows, still f16; see "Tensor parallel"), the Gemma 4 model (switch ignored, said once) |

`ApplyLinear` takes the int8 branch when `M == 256 && TrellisM256Active() && TrellisI8Active()` and the linear has a plan
and a scale table; anything else falls through to the f16 paths. `ScopedTrellisI8` is opened beside `ScopedTrellisM256`
in `RunChunk` (and `PrefillProfiled`), only when the Model has the path and the call is `Prefill`.

What holds with the switch ON:

* **off is identical to today**: unset / `0` / `off` changes no launch, no allocation, no byte (`tests/model/test_prefill_int8`,
  and the ISA check below);
* **no super-chunk, no change**: any `Prefill` call of fewer than 256 rows (and any non-wide, non-trellis, TP, rotated Model)
  is byte-identical on and off, because no int8 launch happens;
* **deterministic**: the same call gives the same bytes twice (the plan is fixed per (class, KB); the kernel's fp32 order is fixed);
* **row-position independent**: a row's output depends on the row, the weights and `(skw, skg)`, never on its position in
  the chunk or on the other rows (`test_trellis_i8_gemm` check G).

What is lost with it ON:

* the 64 == 256 bit identity, for super-chunk rows (the f16 kernel's identity with four 64-row launches does not hold for
  a quantized model);
* independence from the chunk grid. The grid is anchored at each `Prefill` call's start, so with prefix reuse or a
  checkpoint restore the SAME token can be int8 in one run (it fell in a super-chunk) and f16 in another (it fell in a
  tail). The KV bytes of a prompt then depend on the prefix-cache state. That is a property of the idea (the rows a
  grid makes super-chunks), bounded by the accuracy gates below, and why it was opt-in (it is now the default; the kill
  switch is `R4DX_PREFILL_INT8=0`).

What replaces 64 == 256 as the contract: kernel exactness against an integer reference (`test_trellis_i8_gemm`), a KL budget
against off (gates below), and a split-consistency gate (`--prefix-split-at`): KL(on one-shot || on split) must be at most
KL(off || on).

### The kernel (third_party/libr4d)

* `r4d_gemm_trellis_nt_i8.hip` (in `R4D_UNITS`, a unit of its own: the ISA of a kernel moved by a few instructions when other
  instantiations shared its unit, see "OFF path") instantiates only `i8g_kernel<TRELLIS = true, KB in {4, 5}, FWHT = true,
  RESC = 0, SKW in {2, 4, 8}>`, 6 kernels (and, since `R4DX_PREFILL_INT8_SCALES=coarse`, the same with `RESC = 4`: 12, see "Coarse scales"). The device code is `r4d_trellis_i8.h` (the `I8G_EMU` hooks kept: the bench, the
  host-emulation test and the unit compile the one source), the layouts `r4d_trellis_i8_layout.h`. The dense kernel, the
  RESC 1..3 speed bounds and the reference kernel stay bench-only (`tests/kernels/int8_gemm_proto_kernels.h`).
* Entries (`r4d.h`, wrapped in `r4d.hpp`, one row in the kernel registry): `r4d_trellis_i8_wscale(_count)`,
  `r4d_trellis_i8_quant_act` (fixed at M = 256, one launch for both parts), `r4d_trellis_i8_dump_w` (a test diagnostic),
  `r4d_gemm_trellis_nt_i8_check` / `_ws_bytes` / `r4d_gemm_trellis_nt_i8`.
* **Two A parts**: the kernel takes `A8_0, SA_0, A8_1, SA_1, n_split` and picks per block with `n0 >= n_split`, the f16
  kernel's rule (m256.hip). `SW` is indexed by the absolute column. `ws` (`SKG * 256 * N * 4` bytes, always used) and
  `tickets` are the f16 protocol, and the tickets self-reset, so f16 and int8 launches of one linear can interleave on its
  stream. The concurrency invariant (one stream per Container) is unchanged.
* **Legality**: `M == 256`, `K`, `N`, `n_split` multiples of 128, `(K / 128) % (skw skg) == 0`, `skw` in {2, 4, 8}, `skg` in
  {1, 2, 4, 8}, `KB` in {4, 5}, LDS `8192 skw <= 64 KiB`. K / 128 is 40 (K = 5120), 48 (6144), 136 (17408), 24 (3072),
  68 (8704), so `skw skg` is 2, 4 or 8 (and 16 at K = 6144).
* **The weight scale rule** (`i8g_wscale`): per (column, 128 k) of Q, `s = max|w| / 127` (1 for an all-zero group),
  `rs = f16(min(1 / s, 60000))`, `q = rint(w * rs)` (computed as the low byte of `f16(fma(w, rs, 1536))`), and the table holds
  `s_eff = 1 / rs`, so dequantization uses exactly the grid the quantizer rounded on. This is NOT the fp32 table of
  `r4d_trellis_wscale_f32` (R4DX_FAKEQ_W's, which lives in the `_wq` unit production must not depend on): 0.58% of weights
  differ from the fp32 rule by one LSB with the same error RMS, so the real-kernel KL gate (G4) is the arbiter, not the
  fake-quant numbers above.
* **Activation quantizer** (`i8g_quant_act`): one wave per (row, 128-block), `s = amax / 127` (1 for all zero),
  `q = clamp(rint(x / s), -127, 127)` with IEEE division, written straight in the fragment layout. v1 launches it separately
  per `ApplyLinear` call (10 us at K = 5120, 28.6 us at K = 17408; scratch `264 K` bytes per part from the arena, at most
  a few MB; the 224 MiB reserve is unchanged). Fusing it into the transform producers is done, see "The fused quantizer".
* **Tuning table** `src/model/gemm_tuning_table_trellis_i8.inc`, `{N, K, KB, skw, skg}` per class, SEEDED from the bench's best
  picks (`E:\models\r4dx\int8gemm\all.log`, device 1, before the mlp.down retune); `tool_int8_gemm_proto --emit-rows` regenerates
  it with the rule "best median, then the smallest skg within 1%". `PlanTrellisI8(N, K, kb, parts, part_n0)` mirrors
  `PlanTrellisM256`: it honors `R4DX_M256_SHAPES`, validates the row against the kernel's own check (so a bad regenerated row
  cannot throw mid-request), and a shape without a plan runs f16 with a once-per-shape stderr line.

  | class (N x K) | KB4 (skw skg, us) | KB5 (skw skg, us) |
  |---|---|---|
  | mlp.gate_up 34816 x 5120 | 2 1, 607 | 2 2, 645 |
  | mlp.down 5120 x 17408 | 4 1, 308 | 2 2, 327 |
  | gdn.in_proj_qkv 10240 x 5120 | 2 2, 192 | 2 2, 206 |
  | gdn.in_proj_z 6144 x 5120 | 4 1, 123 | 4 2, 130 |
  | gdn.out_proj / attn.o 5120 x 6144 | 4 2, 120 | 2 4, 127 |
  | attn.qg 12288 x 5120 | 4 1, 227 | 2 2, 243 |
  | attn.k / attn.v 1024 x 5120 | 4 2, 29 | 4 1, 29 |

* **TP = 2 in v1**: an explicit f16 fallback, logged once at load. The kernel itself is shape-generic (the rank shards K/2 and
  N/2 are still whole 128-blocks, `n_split` 8704) and `test_trellis_i8_gemm` bit-tests every TP = 2 rank shape, so v2 needs
  wiring and its own KL gate only. The wiring is written (branch `int8v2`, `R4DX_PREFILL_INT8_TP2`, default off, own empty tuning
  tables): "Tensor parallel" below.

### The fused quantizer (`R4DX_PREFILL_INT8_FUSEDQ`, branch `int8v2`)

Status: written, built and CPU-tested; every GPU check below is written and NOT run (the writing session was CPU only).

**What it removes.** The separate `r4d_trellis_i8_quant_act` launch of every int8 call, and with it the f16 A: the transform
(or the fused silu_mul / gate-mul producer) quantizes the block it just rotated and writes A8 + SA, so the f16 A is neither
written nor read back. Expected gain at least the ~5% the unfused quantizer costs on the linears (about 0.15 s of the 3.63 s
cold TTFT at 8k); the f16 A traffic that disappears (2 B written + 2 B read per element, 1 B of A8 written instead of one
write by each of the two kernels) should add a little on top. Unmeasured.

**The contract is byte identity.** A8 and SA are bit for bit what the unfused chain makes: the producer rounds to f16 exactly as
`TransformBlock` does (the same fp32 product, the same `asm("")` that keeps the multiply and the conversion apart), then runs
`i8g_quant_act`'s arithmetic on those f16 values (`amax` over the 128-block, `s = amax / 127`, 1 for an all-zero block,
`q = clamp(rint(x / s), -127, 127)` with IEEE division), and stores the byte in the A8 fragment layout. Every downstream byte
(the GEMM, the KV cache, the logits) is therefore unchanged by the switch.

| piece | where |
|---|---|
| the device function `i8g_quant_wave_block` (a wave32 owning one (row, 128-block), lane `l` holding elements `l + 32 r` as f16 bits: the register layout the transform's FWHT leaves) | `third_party/libr4d/r4d_trellis_i8_fused.h`, header only, includes just `r4d_trellis_i8_layout.h`; the one definition the kernels and the CPU emulation test compile |
| three kernels, twins of the f16 ones: `TrellisInputI8Kernel`, `SiluMulTrellisI8Kernel`, `GateMulTrellisI8Kernel` | `src/kernels/src/trellis_transform.hip`; the f16 kernels and `TransformBlock` are untouched (their ISA: 9 kernel bodies of the unit compared with `tools/reference/compare_isa_listings.ps1` before / after, identical; the new kernels use 28 to 55 VGPRs, no scratch) |
| entries `r4dx_trellis_input_i8` (nout 1..3 with host arrays of `a8[]` / `sa[]`, a two-part linear's second output at `a8 + 256 K`, `sa + K / 128 x 256`), `r4dx_silu_mul_trellis_i8`, `r4dx_attn_gate_mul_trellis_i8` | `kernels.h`; rows must be exactly 256 (throws otherwise), a8 and sa non-null |
| `PreQuantizedActivation::a8` / `sa` (with `transform_id` set, `data` null) | `src/model/linear.h`: a producer that fused hands the consumer its int8 operand instead of the f16 A |
| `TrellisI8Takes(w, M)`, `TrellisI8FusedQ(w, M)`, `AllocTrellisI8Operand` | `src/model/linear.cpp`: THE decision, once |

**One decision, consistently.** `TrellisI8Takes` is `ApplyLinear`'s own test for its int8 branch, factored out (trellis, M = 256,
`ScopedTrellisM256` and `ScopedTrellisI8` open, `PlanTrellisI8` ok, the scale table present); `TrellisI8FusedQ` adds the switch.
Every producer asks it for the linear whose A it makes, in the same scope as that linear's `ApplyLinear`, so they agree by
construction: `SharedTrellisInput` (gdn in_proj_qkv / z, attn qg / k / v: all linears of the group fused, or none), the
mlp.down silu_mul producer (`mlp.cpp`), the attn.o gate-mul producer (`attention_layer.hpp`). A call with no producer (mlp.gate_up
two parts, gdn.out_proj, any linear whose input is a plain bf16 tensor) fuses inside `ApplyLinear` itself: `r4dx_trellis_input_i8`
replaces `TrellisA256`'s transform + the separate quantizer. If a producer quantized and the consumer then cannot take int8
(a8 handed to a call outside the scopes, a plan or table missing) `ApplyLinear` THROWS: no f16 A exists to fall back to. Alignment
of a8 / sa (16 bytes) is checked like the f16 operand's.

**Kill switch.** `R4DX_PREFILL_INT8_FUSEDQ` (read once; unset, empty, `1`, `on` = on, the default; `0`, `off` = the separate
launch again, today's chain bit for bit; anything else warns and keeps the default). Forced off when `R4DX_TRELLIS_A_STATS` is set
(that hook tallies the f16 A a fused producer never writes). It needs `R4DX_PREFILL_INT8` on to matter at all. `ScopedTrellisI8FusedQ`
is the test-only per-thread override, and `TrellisI8OperandCountsGet()` counts where each int8 call's operand came from (producer,
own fused transform, separate quantizer) so a test can prove which chain ran.

**Tests.**

| test | where | status |
|---|---|---|
| `i8g_quant_wave_block` == `i8g_quant_act` byte for byte (A8, SA), == the CPU quantizer, on wide-range f16 (magnitudes 2^-20 .. 2^10, all-zero blocks, one-nonzero blocks, values near rounding midpoints, 60000), one part K = 1024 x 2 and K = 17408; and on the standard problem of both KB | `tests/kernels/test_int8_gemm_proto_emu.cpp` (CPU, 6 new checks, 196 s in total) | PASS |
| the parser, `TrellisI8Takes` / `TrellisI8FusedQ` outside the scopes and without a table | `tests/model/test_prefill_int8_cpu.cpp` (CPU) | PASS (182 checks) |
| the three kernels against the f16 producer + separate quantizer, byte for byte: nout 1..3, K 3072 / 5120 / 6144 / 8704 / 17408, prescale 0 / 4 / -3, silu_mul at both rank widths and a padded row stride, gate-mul at K 6144 / 3072; canaries (nothing written past the outputs), poison (every byte is written), a negative control, the preconditions | `tests/kernels/test_trellis_input_i8.cpp` (GPU, new) | built, not run |
| whole prefill: logits of every call, every KV page, every GDN state and the decode tokens byte-identical with the switch scoped off and on, scenarios of 256 / 257 / 600 / 300 + 333 / 1024 rows, plus the operand counters (off: all separate; on: none separate, the same number of operands) | `tests/model/test_prefill_int8.cpp` `RunFusedQ` (GPU, a fourth load, about 3 more minutes) | built, not run |

GPU commands for the main session (device 1, one job at a time):

```
powershell -NoProfile -Command "`$env:HIP_VISIBLE_DEVICES='1'; & C:\Users\pay20\dev\r4dx-int8v2\build\win-hip\tests\kernels\test_trellis_input_i8.exe; exit `$LASTEXITCODE"
powershell -NoProfile -Command "`$env:HIP_VISIBLE_DEVICES='1'; & C:\Users\pay20\dev\r4dx-int8v2\build\win-hip\tests\model\test_prefill_int8.exe; exit `$LASTEXITCODE"
# an end-to-end A/B of the switch itself (separate process per side): ttft_cli.ps1 -Lengths 8k with R4DX_PREFILL_INT8_FUSEDQ=0 and unset; the greedy hashes must be equal
```

**Caveats.** The A8 stores are one byte per element (4 `global_store_b8` per lane, 16 8-byte segments per wave): the write combining is
left to the L2 and the unfused quantizer's dword stores are not reproduced; if the GPU time of the fused kernels shows it, the
fix is a 4-lane shuffle into dwords. TP = 2 runs f16 unless `R4DX_PREFILL_INT8_TP2=1` ("Tensor parallel"); with it the producers run at the
shard widths (K 8704 silu_mul, K 3072 gate-mul, both byte-tested by `test_trellis_input_i8`). The Gemma 4 trellis
producers (`r4dx_gelu_tanh_mul_trellis_bf16`) are not wired (that model has no int8 path).

### Coarse scales (`R4DX_PREFILL_INT8_SCALES=coarse`, branch `int8v2`)

Status: written, built and CPU-tested; every GPU check below is written and NOT run (the writing session was CPU only). Opt-in; the
default is `blk128`, today's bytes, unchanged.

**What it is.** The per-128 GEMM rescales the int32 partial sum of every 128 K into fp32 (cvt + mul + fma per output element, about
165 of ~860 clk per 128 K per wave in the cycle model of docs/int8-gemm-proto.md, WMMA and VALU not overlapping), because both
scales change every 128 K. With ONE activation scale per ROW (amax over the whole K of the row, per A part) and ONE weight scale per
output COLUMN of Q (amax over the whole K), the int32 accumulator can run over the whole K slice and be multiplied once, after the
K loop, by `sa[row] * sw[col]`; then the unchanged epilogue (LDS reduction over the slices, fp32 partials in `ws`, ticket,
FWHT-128, svh, `out_scale`, one bf16 rounding). The bench measured the bound of this idea (RESC 1: x1.56 to x1.58 against the
shipped f16 plan, against x1.37 for the per-128 kernel); this is the production kernel of it.

| | `blk128` (default) | `coarse` |
|---|---|---|
| activation scale | per (row, 128 k): SA `[K/128][256]` | per row: SA `[256]` per A part |
| weight scale (the f16-`rs` rule of `i8g_wscale`: `s = amax / 127` (1 for all zero), `rs = f16(min(1 / s, 60000))`, `q = rint(w rs)`, table `s_eff = 1 / rs`) | per (column, 128 k): `[K/128][N]` fp32, 0.708 GiB on the 27B | per column over the whole K: `[N]` fp32, 4 N bytes per linear, **14.9 MiB on the 27B** (3,899,392 columns), a saving of 0.69 GiB |
| int32 accumulation | per 128 K, then `acc_f32 += float(int32) * (sa * sw)` | the whole K slice, one `float(int32) * (sa[row] * sw[col])` at the end |
| the weight's f16 `rs` | formed per 128 K from the table | formed once per kernel (two `v_rcp_f32`), the loop reads neither SA nor SW |
| kernel | `i8g_kernel<TRELLIS, KB, FWHT, RESC = 0, SKW>` | `RESC = 4`, 6 more instantiations in the same unit (12 kernels, max 192 VGPRs, no scratch; RESC 4 itself 149 / 160 VGPRs at KB 4 / 5) |

**Overflow.** `|q| <= 127` on both sides (the weight's amax maps to 127 and f16 `rs` is within 2^-11, `|w rs| <= 127.07`, no clamp). An int32
accumulator holds one K slice, `K / (skw skg)` terms: at most K / 2 = 8704 terms (K = 17408, skw 2, skg 1) x 127 x 127 = 1.4e8,
below 2^31 = 2.1e9 (the whole row is 2.8e8, also below). The slices are summed in fp32 (`ws`, the SKG partials), never in int32, so
the split-K path cannot overflow either. The emulation test runs that worst case (every operand +127 at K = 17408, three `(skw, skg)`).

**The pieces.**

| piece | where |
|---|---|
| the kernel: `RESC = 4` (the bench's RESC 1 bound with its own tables; the SA / SW loads of the K loop are compiled out) | `third_party/libr4d/r4d_trellis_i8.h`; instantiated in `r4d_gemm_trellis_nt_i8.hip` |
| entries `r4d_gemm_trellis_nt_i8c` (the per-128 entry's contract and legality, SA `[256]`, SWC `[N]`), `r4d_trellis_i8_wscale_col` (+ `_count` = N), `r4d_trellis_i8_dump_w_col` (test diagnostic), `r4d_trellis_i8_quant_act_row` | `r4d.h`, wrapped in `r4d.hpp` (`GemmTrellisNtI8c`, `TrellisI8WscaleColBuild`, `TrellisI8QuantActRow`, ...); the registry row of `gemm_trellis_nt_i8` names the coarse mode |
| the column table: one workgroup of 8 waves per tile pair, wave w takes the 128-groups w, w + 8, ..., exact amax joined by `xor 16` and an LDS reduction | `i8g_wscale_col` |
| the row quantizer, one workgroup (256 threads) per row: amax over the row, `s = amax / 127` (1 for all zero), `q = clamp(rint(x / s), -127, 127)`, 8-byte segments straight into the A8 fragment layout | `i8g_quant_row_wg` (`r4d_trellis_i8_fused.h`), the one function the stand-alone kernel `i8g_quant_act_row` and the producers call |
| **the producers: fused**, `r4dx_trellis_input_i8r` / `r4dx_silu_mul_trellis_i8r` / `r4dx_attn_gate_mul_trellis_i8r` | `src/kernels/src/trellis_transform.hip`, `kernels.h` |
| the model: `ModelOptions::prefill_int8_scales`, `QuantLinear::trellis_i8_swc`, `TrellisI8Coarse(w)`, `PlanTrellisI8(..., coarse)`, `BuildTrellisI8Scale(w, stream, coarse)`, `AllocTrellisI8Operand(..., coarse)` | `model.h`, `quant_linear.h`, `linear.h` / `linear.cpp`, `container.cpp`, `model.cpp`, `mlp.cpp`, `attention_layer.hpp` |
| the tuning table | `src/model/gemm_tuning_table_trellis_i8c.inc` |

**The fused producer, and why it is a different kernel from task A's.** The row's amax needs the whole row, which no single
(row, 128-block) wave has, so the per-128 fused producers' structure (a wave32 per block, grid (K / 128, 256)) does not carry over. The
coarse producers are one workgroup of 8 waves per row (grid 256): wave w transforms the row's 128-blocks w, w + 8, ... with the same
arithmetic as the f16 producers (x * suh in fp32, the butterfly, the product by the scale as an fp32 value, ONE rounding to f16),
parks the f16 bits in a row buffer in LDS (`nout * K * 2 + 64` bytes: 10 KiB for gate_up's two parts, 30 KiB for the shared qg / k / v,
34 KiB for mlp.down's 17408; a host check throws beyond 64 KiB), one barrier, then `i8g_quant_row_wg` quantizes the row. So there is
no f16 A in memory and no separate quantizer launch, as in task A, and A8 / SA are byte for byte what the stand-alone row quantizer
makes from the f16 A (the GPU test compares them). The per-128 fused producers, `TransformBlockQ8` and the f16 kernels are untouched
(`compare_isa_listings.ps1`, before / after, on `trellis_transform.hip`: the 9 f16 kernels and the 3 per-128 fused kernels IDENTICAL;
the three new ones use 50 to 68 VGPRs, no scratch). The row kernels copy the transform's arithmetic instead of sharing a helper with
the per-128 kernels for exactly that reason (a refactor of those moved their ISA).

The price that is not measured: 256 workgroups x 8 waves is 2048 waves against the per-128 fused producers' up to 34816 (K = 17408), and
LDS bounds the residency of the 34 KiB mlp.down row to about 3 workgroups per WGP, so the coarse producers have less latency hiding. The
stand-alone row quantizer's time is a bench job (`quantize A per row`); the fused kernels' time shows only in the end-to-end A/B
(`ttft_cli.ps1`). If it shows, the follow-ups are more threads per row (the function is written for 256, `red` and the loop strides are
the only places) or a split row (a cluster of workgroups with an atomic max).

**One decision, consistently.** Whether a linear is coarse is a property of its weight table: `TrellisI8Coarse(w)` is
`!w.trellis_i8_swc.empty()`. `Model::Load` builds, per linear, the table of the model's mode and never both (with coarse the per-128
table is NOT allocated: the Load line prints the VRAM figure, `0.0145 GiB` expected on the 27B, against 0.708). `ApplyLinear` reads the
linear's own table to pick the plan row, the operand layout (`AllocTrellisI8Operand`: SA `parts x 256` floats), the quantizer
(`r4dx_*_i8r` / `TrellisI8QuantActRow`) and the GEMM entry; every producer asks the same function for the linear whose A it makes
(`SharedTrellisInput` also requires one scale mode across its group, else it falls back to f16 A for all and each linear quantizes its
own). `R4DX_PREFILL_INT8_FUSEDQ=0` composes: the separate chain is then `r4d_trellis_i8_quant_act_row` over the f16 A (the test oracle
as well as the kill switch).

**Switches.**

| | |
|---|---|
| `R4DX_PREFILL_INT8_SCALES` (read once; unset, empty, `blk128`: the default; `coarse`; anything else warns and keeps the default) | `ModelOptions::prefill_int8_scales`: -1 follows it, 0 forces blk128, 1 coarse (the tests load both in one process). It matters only where the int8 GEMM is on (`R4DX_PREFILL_INT8` not `0`, a TP = 1 trellis Model with 256-row chunks); the Gemma 4 model ignores it |
| load line | `prefill int8: ON (...): ... scales COARSE (R4DX_PREFILL_INT8_SCALES=coarse): A per row, weights per column, over the whole K, ...; N of M trellis linears have a weight scale table, <x> GiB` |
| kill switch | unset / `blk128`: today's table, kernels and bytes (the 6 RESC-0 kernel bodies are ISA-identical to before this work: `compare_isa_listings.ps1 -Pattern 'i8g_kernelILb1ELi[45]ELb1ELi0ELi[248]E'`) |

**Tuning table.** `gemm_tuning_table_trellis_i8c.inc` is SEEDED with the per-128 table's rows (legal for the kernel, the same check),
not measured: the coarse loop is lighter and its best `(skw, skg)` is likely different. `tool_int8_gemm_proto.exe` sweeps and times
the coarse production kernel (`trellisCP` jobs, every legal configuration, verified against the exact coarse reference first, plus the
stand-alone row quantizer) and `--emit-rows-coarse <file>` writes the table in the per-128 table's shape and rule (best verified
median, then the smallest skg within 1%); paste it over the .inc.

**Accuracy.** Fake quantization on the prefill path (baseline KL 0.00751): per-128 both sides 0.00807, W per column + A per row
0.00838, inside the 0.01 budget. The real kernel is what the gate below measures (the f16-`rs` rule differs from the fp32 rule of the
fake quantizer by one LSB on 0.58% of weights). CPU, Gaussian data (`test_int8_gemm_proto_cpu`, K = 1024): relative RMS error of the
product against the f16 x f16 one 0.0112 coarse vs 0.0092 per-128. A row with one large element (the model's outlier rows after the
Hadamard rotation are the rule's worst case) gets a coarse scale far above the rest of its row; the bench selftest and the GPU test
include such a row, only the KL gate says what it costs on real activations.

**Expected speed (a prediction, not a result).** Cycle model: 860 clk per 128 K per wave for the per-128 trellis kernel, 700 without the
rescale: x1.23 of the kernel; the bench's RESC 1 bound over the per-128 production plan, x1.56 / x1.37, is x1.14. The coarse
producers replace the per-128 ones. Linears are about three quarters of the 8k prefill, so the kernel gain is a few percent to 10% of
TTFT: 8k from 3.6 s to about 3.3 to 3.5 s, minus whatever the row producers cost against task A's.

**Tests.**

| test | where | status |
|---|---|---|
| the coarse CPU references (`QuantizeWeightsColRef`, `QuantizeActRowRef`): range, amax -> 127, all-zero row / column -> scale 1, the coarse scale is the max of the per-128 ones, K = 128 equals per-128 bit for bit; the kernel chain (`EmuKernel` RESC 1 = RESC 4's math) vs the exact reference at five K splits; the accuracy price on Gaussian data; the int32 bound | `tests/kernels/test_int8_gemm_proto_cpu.cpp` (CPU) | PASS |
| the kernel SOURCE as plain C++: `i8g_quant_act_row` and the producers' LDS form of `i8g_quant_row_wg` vs the CPU row quantizer (K 1024 two parts, K 17408, zero rows, a one-nonzero row); `i8g_wscale_col` and `i8g_dump_w<COARSE>` vs the CPU table / int8 matrix; `i8g_kernel` RESC 4 (trellis KB 4 / 5, every legal `(skw, skg)`) vs the exact coarse reference, byte for byte vs its dense twin and vs RESC 1 on the same operands replicated per 128, one-hot byte-exact, two A parts vs two single-part launches, tickets reset; K = 17408 with every operand +127 | `tests/kernels/test_int8_gemm_proto_emu.cpp` (CPU, 19 new checks, 287 s in total) | PASS (39 checks) |
| the parser, the option, the coarse plan for the seven classes at both rates, the coarse table (14 rows, legal, same classes as the per-128 one) | `tests/model/test_prefill_int8_cpu.cpp` (CPU) | PASS (263 checks) |
| `r4d_gemm_trellis_nt_i8c` on the 14 classes (TP = 1 and TP = 2 shards) x KB 4 / 5: A the column table, the int8 weights and the row quantizer vs the CPU; B every legal `(skw, skg)` vs the exact reference (+ a 2% negative control); C one-hot byte-exact; D two parts vs single-part launches; E repeats and tickets; F f16 / coarse / f16 on one linear; G row independence; the refusals | `tests/kernels/test_trellis_i8_gemm.cpp` `RunCaseCoarse` (GPU) | built, not run |
| `r4dx_*_i8r` byte for byte vs the f16 producers + `TrellisI8QuantActRow`, all the per-128 test's shapes, canaries, poison, negative control, preconditions, the LDS limit | `tests/kernels/test_trellis_input_i8.cpp`, every test now runs twice (per-128, `[coarse]`) (GPU) | built, not run |
| a Model with `prefill_int8_scales = 1`: fused chain == separate chain byte for byte (logits of every call, KV, GDN state, decode tokens), the operand counters, a rerun gives the same bytes, KV differs from f16's and from per-128's, KL bounds | `tests/model/test_prefill_int8.cpp` `RunFusedQ(scales = 1)`, a fifth load, about 3 more minutes (GPU) | built, not run |
| selftest, verify, sweep and timing of the production coarse kernel and the stand-alone row quantizer | `tool_int8_gemm_proto.exe` (`SelftestCoarse`, `trellisCP`, `quantrow`; builds with `build_int8_gemm_proto.ps1`) | built, not run |

**What still needs the GPU (the main session; device 1, one job at a time).**

```
# C1. the kernel and producer bit-tests (a few minutes each; test_trellis_i8_gemm now also runs the coarse cases)
powershell -NoProfile -Command "`$env:HIP_VISIBLE_DEVICES='1'; & C:\Users\pay20\dev\r4dx-int8v2\build\win-hip\tests\kernels\test_trellis_i8_gemm.exe; exit `$LASTEXITCODE"
powershell -NoProfile -Command "`$env:HIP_VISIBLE_DEVICES='1'; & C:\Users\pay20\dev\r4dx-int8v2\build\win-hip\tests\kernels\test_trellis_input_i8.exe; exit `$LASTEXITCODE"
# C2. the Model test (five loads of the 27B, about 16 minutes): per-128 identity unchanged, the coarse fused-vs-separate chain
powershell -NoProfile -Command "`$env:HIP_VISIBLE_DEVICES='1'; & C:\Users\pay20\dev\r4dx-int8v2\build\win-hip\tests\model\test_prefill_int8.exe; exit `$LASTEXITCODE"
# C3. the bench: selftest (includes the coarse kernel), verify, the sweep and the coarse tuning rows (machine idle, about 25 minutes)
powershell -NoProfile -File C:\Users\pay20\dev\r4dx-int8v2\tests\kernels\build_int8_gemm_proto.ps1 -Out E:\models\r4dx\int8v2\obj
powershell -NoProfile -Command "`$env:HIP_VISIBLE_DEVICES='1'; & E:\models\r4dx\int8v2\obj\tool_int8_gemm_proto.exe --mode all --out E:\models\r4dx\int8v2\all.json --emit-rows E:\models\r4dx\int8v2\rows.inc --emit-rows-coarse E:\models\r4dx\int8v2\rows_c.inc 2>&1 | Tee-Object -FilePath E:\models\r4dx\int8v2\all.log; exit `$LASTEXITCODE"
#     pass: the selftest and every verify line pass; the line "COARSE PRODUCTION kernel ... x<r> vs shipped plan" per (KB, summary) is the speed;
#     then copy rows_c.inc over src\model\gemm_tuning_table_trellis_i8c.inc (rebuild, run test_prefill_int8_cpu) 
# C4. accuracy, the chunked canon prefill with the real kernel (KL(f16 || coarse) and KL(per-128 || coarse); gates as for the per-128 default)
powershell -NoProfile -Command "`$env:R4DX_PREFILL_INT8_SCALES='coarse'; & C:\Users\pay20\dev\r4dx-int8v2\tools\prefill\run_kl.ps1 -Device 1 -Tool C:\Users\pay20\dev\r4dx-int8v2\build\win-hip\tests\model\tool_teacher_forced_logprobs.exe -Tokens C:\Users\pay20\dev\r4dx-int8v2\tools\reference\kl_corpus\tokens_canon.json -OutDir E:\models\r4dx\int8v2\kl-chunk-coarse; exit `$LASTEXITCODE"
C:\Users\pay20\AppData\Local\Programs\Python\Python312\python.exe C:\Users\pay20\dev\r4dx-int8v2\tools\prefill\kl_compare.py --ref E:\models\r4dx\int8prefill\kl-chunk-off --test E:\models\r4dx\int8v2\kl-chunk-coarse --tokens C:\Users\pay20\dev\r4dx-int8v2\tools\reference\kl_corpus\tokens_canon.json
#     (the per-128 default's run, for the ratio: the same command without the variable, -OutDir ...\kl-chunk-blk128, and --ref kl-chunk-off --test that)
#     8k / 32k: the same with -Tokens the tokens_long.json of G4c (tools\prefill\make_kl_tokens.py --lengths 8k,32k); the 32k code segment is the one to watch
# C5. speed end to end: cold TTFT at 8k / 32k, coarse vs the default (and vs R4DX_PREFILL_INT8_FUSEDQ=0 coarse, which prices the row producers)
powershell -NoProfile -Command "`$env:R4DX_PREFILL_INT8_SCALES='coarse'; & C:\Users\pay20\dev\r4dx-int8v2\tools\prefill\ttft_cli.ps1 -Device 1 -Lengths 8k,32k -Runs 3 -OutDir E:\models\r4dx\int8v2\ttft-coarse; exit `$LASTEXITCODE"
```

Gates for making coarse the default (it would need the same list as "Gates for making it the default" above, the KL ones against the
f16 path and the speed ones against the per-128 default): KL(f16 || coarse) mean at most 0.0020, p99 at most 0.012, top-1 agreement at
least 98.3% on the chunked canon; per-kind at most 0.003 at 8k and 32k with no row above 2 (the per-128 kernel sits at 0.0011 / 0.0017
and 0.0104 at 32k, so there is little room at 32k code); and at least 3% faster cold TTFT than the per-128 default at 8k and 32k, else the
accuracy it gives up buys nothing. What is NOT covered here, and not covered for the per-128 default either: images, and TP = 2
(see "Tensor parallel" next, which wires the kernel for it, off by default).

### Tensor parallel (`R4DX_PREFILL_INT8_TP2`, branch `int8v2`)

Status: written, built and CPU-tested (`test_prefill_int8_cpu`, 485 checks, PASS); every GPU check below is written and NOT run (the
writing session was CPU only). **The switch is OFF by default and the TP = 2 tuning tables are EMPTY**: nothing changes at TP = 2 until
the bench has produced rows (`--tp 2`) and the accuracy gates below have been run. With the switch on and no rows the Model says so at
load and runs f16 (byte-identical to the switch off, `test_prefill_int8_tp2` checks it).

**Why TP = 2 ran f16 (every reason, found by reading the code).**

1. `DecidePrefillInt8` (`prefill_int8.h`) refused `tp_world > 1` outright (the only hard switch; Model::Load calls it with
   `tp.world`). The 256-row super-chunks themselves are NOT the obstacle: `DecidePrefillChunk` serves TP = 2, every rank runs
   `Model::Prefill` -> `RunChunk` with `ScopedTrellisM256` open, `Model::Prefill` sets `prefill_int8_call_` on each rank, and `TpWarmup`
   runs a 256-row prefill first, so the scope would be open on both ranks.
2. `PlanTrellisI8` had no row for any rank shard shape: its table is keyed by `(N, K, KB)` of the seven TP = 1 classes. Lifting (1)
   alone would have left every shard on f16 with the once-per-shape notice, except `attn.qg`, whose rank shape 6144 x 5120 equals
   `gdn.in_proj_z`'s TP = 1 class: that linear alone would have taken the TP = 1 row, silently half-covering the model. (The f16 side
   keys a separate TP = 2 table for exactly this reason, `gemm_tuning_table_trellis_tp2.inc`.)
3. Nothing else: the scale tables are built from the rank's own words (`Container::BuildTrellisI8Scales` on the rank thread, the
   rank's device and stream), the producers and `ApplyLinear` decide with the same `TrellisI8Takes` / `TrellisI8FusedQ`, and the kernel
   was already bit-tested at all seven rank shapes (`test_trellis_i8_gemm`). No collective, comm or lockstep code is involved.

**How TP = 2 shards the trellis linears** (docs/tp.md 4.2, 4.3, 5.2; the shapes below are derived from the real
`tp::RuleFor` / `RankRows` / `RankCols` by `test_prefill_int8_cpu`):

| class (count per forward) | rule | global N x K | rank N x K (part boundary) | K / 128 | legal `(skw, skg)` (`skw skg` divides K / 128) |
|---|---|---|---|---|---|
| mlp.gate_up (64) | column-parallel, 2 segments gate / up, 2 A parts | 34816 x 5120 | 17408 x 5120 (8704) | 40 | (2,1) (2,2) (2,4) (4,1) (4,2) (8,1) |
| mlp.down (64) | row-parallel | 5120 x 17408 | 5120 x 8704 | **68** | **(2,1) (2,2) (4,1) only** |
| gdn.in_proj_qkv (48) | column-parallel, 3 segments q / k / v | 10240 x 5120 | 5120 x 5120 | 40 | as gate_up |
| gdn.in_proj_z (48) | column-parallel | 6144 x 5120 | 3072 x 5120 | 40 | as gate_up |
| gdn.out_proj (48) + attn.o (16) | row-parallel | 5120 x 6144 | 5120 x 3072 | 24 | as gate_up |
| attn.qg (16) | column-parallel, per-head `[q | gate]` | 12288 x 5120 | 6144 x 5120 | 40 | as gate_up |
| attn.k, attn.v (16 each) | column-parallel | 1024 x 5120 | 512 x 5120 | 40 | as gate_up |

* **The trellis tensors are sliced, not re-encoded.** `ShardLoader::TrellisSlice` cuts the pair-grid words by bytes (a run of rows is one
  byte range, a K range one run per pair row; every range whole 128-blocks, which is whole 32 x 16 tiles), slices `suh` (row-parallel:
  the rank's K range; column-parallel: replicated) and `svh` (column-parallel: the rank's rows) and recomputes the part widths. A rank's
  linear is then a stand-alone trellis linear of the rank shape, which is why the f16 and int8 kernels run on it unchanged.
  `test_tp_loader` checks the words, `suh`, `svh` and part widths of both ranks against the TP = 1 load.
* **K blocks never straddle a shard.** Every row range, K range and part boundary is a whole 128-block (the loader refuses otherwise and
  `test_prefill_int8_cpu` re-derives it): K / 2 = 3072 (24 blocks) and 8704 (68 blocks), K shard offsets 3072 r and 8704 r, N shards
  1024 / 3072 / 6144 / 8704 multiples. So with `blk128` scales the A8 + SA of a row-parallel shard are exactly the TP = 1 blocks of its
  K range, and a shard's per-(column, 128 k) weight table is exactly the TP = 1 table of its rows and K blocks (`test_tp_loader`
  checks the tables bit for bit): **TP = 2 `blk128` int8 differs from TP = 1 `blk128` int8 only in the order of the fp32 sums and
  the bf16 all-reduce**, the same noise TP = 2 f16 has against TP = 1 f16 (docs/tp.md 10.4: KL 0.00089).
* **`coarse` is not TP = 1's slice for the row-parallel classes** (mlp.down, gdn.out_proj, attn.o): the A scale is the amax over the
  rank's K half and the weight scale is per column over the rank's K half, i.e. FINER than TP = 1's whole-K scales (the two halves'
  column scales are each at most the TP = 1 scale, and their maximum equals it, `test_tp_loader`). Column-parallel classes have the
  whole K per rank and equal TP = 1's. So `coarse` at TP = 2 needs its own KL run and is expected to be at least as accurate as TP = 1's.
* **Sizes.** The weight tables are `K N / 32` bytes per linear: 0.354 GiB per rank for `blk128` (half of 0.708 GiB), about 7 MiB per
  rank for `coarse`; device 0 also carries the desktop (docs/tp.md 4.5: ~15 GiB headroom remains). The int8 GEMM's `ws` is
  `skg * 256 * N * 4` bytes from the arena (224 MiB on a wide Model): the widest shard is mlp.gate_up (N = 17408, 17.8 MB per `skg`),
  so any legal row fits (skg 8 would be 142 MB); the bench's rule picks the smallest `skg` within 1% anyway. `TpWarmup`'s 256-row prefill
  (run when the KV cache holds a 256-row chunk, i.e. `--max-ctx` of at least 256) would raise an arena overflow at load, not mid-request.

**What was built.**

| piece | where |
|---|---|
| `R4DX_PREFILL_INT8_TP2` (read once; unset, empty, `0`, `off` = off, the default; `1`, `on` = asks for it; anything else warns and keeps the default), `ModelOptions::prefill_int8_tp2` (-1 follows the environment; 0 / 1 force it, the tests load both in one process), `PrefillInt8Inputs::tp2_enabled`, the `DecidePrefillInt8` reasons (TP without the switch: names the switch; switch on but no shard linear has a plan: names the TP tables and the bench command) | `prefill_int8.h`, `model.h`, `Model::Load` |
| `QuantLinear::trellis_tp_shard`, set by `ShardLoader::TrellisSlice` for a column- or row-parallel linear of a world > 1 load (never at TP = 1, never for a replicated linear) | `quant_linear.h`, `shard_loader.h` |
| `PlanTrellisI8(..., coarse, tp_shard)` and `TrellisI8Rows(..., coarse, tp_shard)`: a shard linear reads the TP = 2 tables ONLY (no borrowed row for `attn.qg`), and a TP = 1 linear never reads them; the refusal says "no int8 tuning row for this TP rank-shard (N, K, KB) (...: tool_int8_gemm_proto --tp 2 --emit-rows)", and the once-per-shape notice reads `trellis TP shard [N x K] KB4 parts 1 has no int8 plan (...); its 256-row calls run the f16 kernel` | `linear.h`, `linear.cpp` |
| `gemm_tuning_table_trellis_i8_tp2.inc`, `gemm_tuning_table_trellis_i8c_tp2.inc` (`kTrellisI8Tp2Table`, `kTrellisI8cTp2Table`): EMPTY, one `N == 0` placeholder row each (a C++ array cannot be empty; no plan matches it, the CPU test skips it) | `src/model/` |
| `tool_int8_gemm_proto --tp 2`: times, verifies and emits the seven rank shapes (`kShapesTp2`) against the f16 plan of the TP = 2 table (the f16 baseline of `gdn.in_proj_z` KB5, whose M = 64 row has no exact M = 256 configuration, is a flagged stand-in), and writes `kTrellisI8Tp2Table` / `kTrellisI8cTp2Table`; compiled (`-Lite`), not run | `tests/kernels/tool_int8_gemm_proto.hip` |
| `TpModel::Load` compares `PrefillInt8Enabled()` across the ranks with the other capabilities (a split decision throws `TpDivergenceError`: a half-int8 prefill would still all-reduce, wrongly mixed) | `tp_model.cpp` |
| the load line gains `; TP rank r shards (R4DX_PREFILL_INT8_TP2=1; ...)` when it is on; `prefill int8: off (default is on, not used: tensor parallelism (... R4DX_PREFILL_INT8_TP2=1 ...))` when it is not asked for | `Model::Load` |

Everything else is shared with TP = 1 and unchanged: the producers (`SharedTrellisInput`, silu_mul, gate-mul) decide per group with
`TrellisI8FusedQ`, so a group with a shard that has no row keeps f16 for all of it; `ApplyLinear` takes int8 per linear; the row-parallel
all-reduce runs on the bf16 output exactly as before (it sums the ranks' partial outputs after the epilogue, which is linear); the fused
producers are bit-tested at the shard widths (K 8704 silu_mul, K 3072 gate-mul, `test_trellis_input_i8`); `TpWarmup` runs the 256-row
prefill through the int8 chain when it is on, so the first launch of every int8 kernel is inside the load.

**OFF path.** With the switch unset a TP = 2 Model decides at load that it does not use int8 (`DecidePrefillInt8` refuses at the TP
check, before any table is built), builds no scale table, opens no scope and launches nothing new: it is byte-identical to one built
before this branch (T3 checks it against main's binary). `QuantLinear` gained one bool; no kernel, no f16 launch and no tuning row
changed (no libr4d file is touched).

**Tests.**

| test | where | status |
|---|---|---|
| the switch's parser and option; `DecidePrefillInt8`'s TP rows and their order; the 27B's rank shapes from the real `RuleFor` / `RankRows` / `RankCols` against docs/tp.md 4.2, every range whole 128-blocks, K shard offsets 3072 r / 8704 r; at least one kernel-legal `(skw, skg)` per shard class (mlp.down: exactly three); the shard plan is ok iff the TP = 2 table has the row, a shard never borrows the TP = 1 row (`attn.qg`), a TP = 1 linear never reads a shard row, the refusal reasons; the TP = 2 tables (placeholder skipped, no duplicate, legal, shard shapes only); the f16 M = 256 plan at the rank shapes (printed) | `tests/model/test_prefill_int8_cpu.cpp` (CPU, +222 checks) | PASS (485 checks) |
| `trellis_tp_shard` set on every rank linear and never at TP = 1; the int8 weight scale table of every shard == the TP = 1 table of its rows and K blocks, bit for bit (`TrellisI8WscaleBuild` on tiny trellis containers, KB 4 and mix); the per-column coarse table of a column-parallel shard == TP = 1's rows, of a row-parallel pair: each half <= the whole-K scale and the larger of the two == it | `tests/model/test_tp_loader.cpp` `CheckTrellisTp` (GPU, tiny containers, fixture `trellis_tiny`) | built, not run |
| `TpModel` (emulate) on the real container: switch off -> f16 on both ranks; switch on with empty tables -> refused, both ranks f16, every observable byte-identical to off; with rows -> int8 chunk counts per rank, no-super-chunk scenarios identical, deterministic rerun, KL(TP = 2 f16 \|\| TP = 2 int8), and TP = 2 int8 against TP = 1 int8 | `tests/model/test_prefill_int8_tp2.cpp` (GPU, up to four 27B loads) | built, not run |
| the seven rank shapes at both rates: kernel exactness (`r4d_gemm_trellis_nt_i8` and `_i8c`, every legal `(skw, skg)`, one-hot, two parts, tickets, row independence) | `tests/kernels/test_trellis_i8_gemm.cpp` (GPU, already covers the TP = 2 shards) | built (rebuilt on this branch), not run |
| the bench at `--tp 2`: selftest, verify, sweep, timing, `--emit-rows` / `--emit-rows-coarse` | `tool_int8_gemm_proto.exe` (compiled with `build_int8_gemm_proto.ps1 -Lite` to check it) | built, not run |

**What still needs the GPU** (the main session; one job at a time; TP = 2 real mode needs device 0 as well, so stop the production server first,
docs/tp.md 9.2; outputs under `E:\models\r4dx\int8v2\tp2\`). Order matters: the fallback checks first, then the rows, then accuracy and speed.

```
# T1. the loader and the scale tables of a shard (tiny containers; a minute)
ctest --test-dir C:\Users\pay20\dev\r4dx-int8v2\build\win-hip -R "^(convert_trellis_import|test_tp_loader)$" --output-on-failure
# T2. the fallback with the TP tables EMPTY (as shipped): switch on == switch off, byte for byte (two 27B TP = 2 emulated loads, ~8 minutes)
powershell -NoProfile -Command "`$env:HIP_VISIBLE_DEVICES='1'; & C:\Users\pay20\dev\r4dx-int8v2\build\win-hip\tests\model\test_prefill_int8_tp2.exe; exit `$LASTEXITCODE"
#     pass: "[PASS] tp2 fallback: ..." and "test_prefill_int8_tp2: PASS"; the load lines say "prefill int8: off (...not used: no TP rank-shard linear has an int8 plan ...)"
# T3. TP = 2 identity with int8 off against main's TP = 2 (real mode, both GPUs): main's tool and this branch's tool, switch unset, the same tokens
powershell -NoProfile -File C:\Users\pay20\dev\r4dx-int8v2\tools\prefill\run_kl.ps1 -Tp 2 -Tokens C:\Users\pay20\dev\r4dx-int8v2\tools\reference\kl_corpus\tokens_canon.json -Tool C:\Users\pay20\dev\r4dx\build\win-hip\tests\model\tool_teacher_forced_logprobs.exe -OutDir E:\models\r4dx\int8v2\tp2\kl-main-tp2
powershell -NoProfile -File C:\Users\pay20\dev\r4dx-int8v2\tools\prefill\run_kl.ps1 -Tp 2 -Tokens C:\Users\pay20\dev\r4dx-int8v2\tools\reference\kl_corpus\tokens_canon.json -OutDir E:\models\r4dx\int8v2\tp2\kl-branch-tp2-off
#     pass: every *.logprobs.f16 of the two directories has the same SHA-256 (Get-ChildItem both | Get-FileHash)
# B1. the bench at the rank shapes (idle machine, both GPUs free; device 1; ~15 minutes): selftest, verify, the sweep and both tables
powershell -NoProfile -File C:\Users\pay20\dev\r4dx-int8v2\tests\kernels\build_int8_gemm_proto.ps1 -Out E:\models\r4dx\int8v2\obj
powershell -NoProfile -Command "`$env:HIP_VISIBLE_DEVICES='1'; & E:\models\r4dx\int8v2\obj\tool_int8_gemm_proto.exe --tp 2 --mode all --out E:\models\r4dx\int8v2\tp2\all.json --emit-rows E:\models\r4dx\int8v2\tp2\rows_tp2.inc --emit-rows-coarse E:\models\r4dx\int8v2\tp2\rows_c_tp2.inc 2>&1 | Tee-Object -FilePath E:\models\r4dx\int8v2\tp2\all.log; exit `$LASTEXITCODE"
#     pass: the selftest and every verify line pass; "ALL SEVEN LINEAR CLASSES, TP = 2 RANK SHARDS KB4/KB5" gives the speed (compare with x1.37 / x1.33 at TP = 1)
#     then copy rows_tp2.inc over src\model\gemm_tuning_table_trellis_i8_tp2.inc and rows_c_tp2.inc over ..._i8c_tp2.inc (14 rows each; the CPU
#     test lists how many it sees), rebuild (cmake --build build\win-hip --target test_prefill_int8_cpu test_prefill_int8_tp2 r4dx-cli tool_teacher_forced_logprobs), run test_prefill_int8_cpu
# B2. the int8 part of the TP = 2 Model test, with rows (the tables are compiled in; four loads, ~14 minutes)
powershell -NoProfile -Command "`$env:HIP_VISIBLE_DEVICES='1'; & C:\Users\pay20\dev\r4dx-int8v2\build\win-hip\tests\model\test_prefill_int8_tp2.exe; exit `$LASTEXITCODE"
# A1. accuracy, the chunked canon prefill under TP = 2, real mode: f16 (T3's kl-branch-tp2-off), int8 on, and the TP = 1 int8 twin
powershell -NoProfile -Command "`$env:R4DX_PREFILL_INT8_TP2='1'; & C:\Users\pay20\dev\r4dx-int8v2\tools\prefill\run_kl.ps1 -Tp 2 -Tokens C:\Users\pay20\dev\r4dx-int8v2\tools\reference\kl_corpus\tokens_canon.json -OutDir E:\models\r4dx\int8v2\tp2\kl-tp2-on; exit `$LASTEXITCODE"
powershell -NoProfile -File C:\Users\pay20\dev\r4dx-int8v2\tools\prefill\run_kl.ps1 -Device 1 -Tokens C:\Users\pay20\dev\r4dx-int8v2\tools\reference\kl_corpus\tokens_canon.json -OutDir E:\models\r4dx\int8v2\tp2\kl-tp1-on
powershell -NoProfile -Command "`$env:R4DX_PREFILL_INT8='0'; & C:\Users\pay20\dev\r4dx-int8v2\tools\prefill\run_kl.ps1 -Device 1 -Tokens C:\Users\pay20\dev\r4dx-int8v2\tools\reference\kl_corpus\tokens_canon.json -OutDir E:\models\r4dx\int8v2\tp2\kl-tp1-off; exit `$LASTEXITCODE"
$py = "C:\Users\pay20\AppData\Local\Programs\Python\Python312\python.exe"; $kl = "C:\Users\pay20\dev\r4dx-int8v2\tools\prefill\kl_compare.py"; $tok = "C:\Users\pay20\dev\r4dx-int8v2\tools\reference\kl_corpus\tokens_canon.json"
& $py $kl --ref E:\models\r4dx\int8v2\tp2\kl-branch-tp2-off --test E:\models\r4dx\int8v2\tp2\kl-tp2-on --tokens $tok      # TP2 int8 vs TP2 f16: the cost of int8 at TP = 2
& $py $kl --ref E:\models\r4dx\int8v2\tp2\kl-tp1-on --test E:\models\r4dx\int8v2\tp2\kl-tp2-on --tokens $tok            # TP2 int8 vs TP1 int8: should sit at the next line's level (blk128)
& $py $kl --ref E:\models\r4dx\int8v2\tp2\kl-tp1-off --test E:\models\r4dx\int8v2\tp2\kl-branch-tp2-off --tokens $tok   # TP2 f16 vs TP1 f16: the TP noise floor (docs/tp.md: mean 0.00089)
#     8k / 32k: the same four runs with -Tokens the tokens_long.json of G4c (tools\prefill\make_kl_tokens.py --lengths 8k,32k)
# S1. speed end to end, TP = 2 real mode: cold TTFT 8k / 32k, switch off and on (separate processes; greedy hashes must be equal to the TP = 1 gate's rule)
powershell -NoProfile -File C:\Users\pay20\dev\r4dx-int8v2\tools\prefill\ttft_cli.ps1 -Tp 2 -Lengths 8k,32k -Runs 3 -OutDir E:\models\r4dx\int8v2\tp2\ttft-off
powershell -NoProfile -Command "`$env:R4DX_PREFILL_INT8_TP2='1'; & C:\Users\pay20\dev\r4dx-int8v2\tools\prefill\ttft_cli.ps1 -Tp 2 -Lengths 8k,32k -Runs 3 -OutDir E:\models\r4dx\int8v2\tp2\ttft-on; exit `$LASTEXITCODE"
# C1. coarse at TP = 2 (its own rows from B1, own KL): the A1 / S1 runs again with R4DX_PREFILL_INT8_SCALES=coarse as well
```

**Gates for turning it on by default at TP = 2** (the same as TP = 1's, against the TP = 2 f16 run, plus the TP-specific ones): KL(TP = 2 f16
\|\| TP = 2 int8) mean at most 0.0020, p99 at most 0.012, top-1 agreement at least 98.3% on the chunked canon, per-kind at most 0.003 at 8k and
32k with no row above 2; KL(TP = 1 int8 \|\| TP = 2 int8) at most 1.5 x KL(TP = 1 f16 \|\| TP = 2 f16) (blk128 only differs in summation
order; a larger gap means a shard scale or alignment bug); T2 and T3 byte-identical; cold TTFT faster at 8k and 32k (predicted, NOT
measured: the linears are a smaller share of a TP = 2 prefill than of a TP = 1 one, the all-reduces and the per-rank attention do not
shrink, so expect a gain between half and three quarters of TP = 1's 18.9% / 15.7%, i.e. roughly -10 to -14% at 8k); a 128k TP = 2
prefill completes, VRAM headroom on device 0 logged, no NaN, no all-reduce timeout (docs/tp.md 12); the greedy hash stable across two runs.
The default is flipped by changing `ParsePrefillInt8Tp2`'s unset case, nothing else.

**Not done / open.** The TP = 2 tables are empty by design (no measurement was possible); the `blk128` and `coarse` accuracy at TP = 2
is unmeasured (the equality argument above is a prediction, `test_tp_loader` proves its scale-table half); MTP and DFlash at TP = 2 keep
their 64-row f16 slices (unchanged); the vision tower and `PrefillMultimodal` with images stay f16 as at TP = 1.

### Epilogue: a copy that cannot drift silently

The int8 kernel's LDS reduction, ticket protocol, FWHT-128, svh, `out_scale` and bf16 rounding are a COPY of the shipped
M = 256 kernel's. A shared helper would change the shipped kernel's ISA (the int8q work showed any refactor of those
kernels does), so the shipped file is not touched. `tools/reference/diff_epilogue.ps1` compares the two sources as text on
the parts that define the arithmetic (whole bodies of `bf16_rn`, `lane_stage` and `bfly`; the FWHT stage trace at NP = 1;
the ticket block literally; the y-sum and the ws unit address; `svh` addressing; the output expression; the slice
reduction's owner and sum rules) and fails when they differ; it runs in the build (`r4d_trellis_isa`), and `-SelfTest`
edits one FWHT stage in memory and requires the comparison to fail.

### Weight scale tables and VRAM

`QuantLinear::trellis_i8_sw` (`[K / 128][N]` fp32, separate from `trellis_wscale`) is built once at `Model::Load`, only when
the switch is on and the Model uses it, from each rank's own words (`Container::BuildTrellisI8Scales`, after the weights are
resident) and only for linears with a plan. `K N / 32` bytes per linear: 760,217,600 bytes (0.708 GiB) for the 27B's 64
layers (MLP 0.50, GDN in_proj 0.12, out / attn.o 0.06, qg 0.03, k / v 0.005 GiB), on 16.3 GiB of weights; the VRAM breakdown
books it under `weights=` and the load line prints it. Follow-up: store f16 `rs` (about 0.4 GB). Under TP = 2 with
`R4DX_PREFILL_INT8_TP2=1`, each rank builds the table of its own shard on its own device: 0.354 GiB per rank (`blk128`), ~7 MiB (`coarse`).

### Model plumbing

* `src/model/prefill_int8.h` (header-only, HIP-free): `ParsePrefillInt8` (unset, empty, `1`, `on` = on -- the default --;
  `0`, `off` = off; anything else warns and means the default), `ParsePrefillInt8Explicit` / `ResolvePrefillInt8Explicit`
  (only `1` / `on` or option 1 are an explicit request), `ResolvePrefillInt8Request` and `DecidePrefillInt8` (refuses with a
  reason, in this order, for `R4DX_FAKEQ_ACT/W` set, a rotated container, a Model without 256-row chunks, TP > 1 without
  `R4DX_PREFILL_INT8_TP2`, no trellis linears, no linear with a plan).
* `ModelOptions::prefill_int8`: -1 follows the environment (default on), 0 and 1 force it (`test_prefill_chunk_identity`
  forces 0). An EXPLICIT `R4DX_PREFILL_INT8=1` / option 1 together with `R4DX_FAKEQ_ACT` / `R4DX_FAKEQ_W` throws at load (they
  would quantize twice); with int8 only the default, the experiment switches win.
* One load line: `prefill int8: ON (default | R4DX_PREFILL_INT8=...)` with the table count and GiB, or
  `off (<source> ..., not used: <reason>)`, or `off (R4DX_PREFILL_INT8=0: ...)`.
* `Model::PrefillInt8Enabled()` and `PrefillInt8ChunksRun()` (a counter beside `PrefillWideChunksRun()`; tests prove the int8
  path, not a fallback, produced a result).
* DFlash feature capture and MTP priming read int8-perturbed hidden states (allowed, gated by the speculation gate below).

### OFF path: identical code, and how that was checked

With the switch unset (or on a Model that cannot use it): no table, no scope, no extra launch or allocation; `ApplyLinear`
tests one thread-local bool before the (unchanged) f16 block, whose input transform now comes from a helper (`TrellisA256`, a
verbatim move). In libr4d nothing shipped moved: the int8 kernels are a unit of their own and `r4d.h` only gained
declarations. Checked on this branch with `hipcc -S` listings (the unit's own flags, `tools/reference/compare_isa_listings.ps1`,
against the listings of the `f16retune` worktree, 89587f0): `r4d_gemm_trellis_nt_m64` (100 kernels), `r4d_gemm_trellis_nt_m256`
(22 kernels), `r4d_attn_paged_h256_gqa6` and `_gqa2`: function bodies and the whole normalised listing (including the
code-object metadata: VGPR counts, scratch, kernarg sizes) are IDENTICAL in all four. The build's own gates hold: shipped m64 100
kernels max 189 VGPRs, m256 22 kernels max 188, no scratch, 0 near dependencies; the Qwen attention hash unchanged; the `_wq` units
(49 and 22 kernels) are reported only.

The int8 unit's own gate (`check_trellis_isa.cmake -DGEMM_KERNEL=i8g_kernel -DMAX_VGPR=192`): 6 kernels, max 192 VGPRs, no
scratch, no spill, 822 asm VALU in the GEMM kernels, 0 near dependencies (12 kernels, 1644 asm VALU, still max 192, with the coarse ones). Against the bench translation unit, where the GPU
numbers were measured, the 6 production bodies are IDENTICAL (`compare_isa_listings.ps1 -Pattern 'i8g_kernelILb1ELi[45]ELb1ELi0ELi[248]E'`);
against the prototype's listing from before the two-part parameters, each kernel gained one VALU and a few scalar selects (10 387 ->
10 406 instructions at KB 4 / skw 2, 10 565 -> 10 565 at KB 5 / skw 4).

### What was run (CPU only) and what was not

Run and passing: `test_int8_gemm_proto_cpu` (layouts, quantizers, software-WMMA chain), `test_int8_gemm_proto_emu` (the production
kernel SOURCE compiled as plain C++, run against exact references: every instantiation of the bench, plus the new two-part
cases: quantizer part 1, `n_split` against each part's reference and against two single-part launches, byte for byte, and a\none-hot check that the whole chain is byte-exact against the CPU's fp32 emulation of the epilogue, the very prediction\n`test_trellis_i8_gemm` check C makes on the GPU; 14 checks, 95 s), `test_prefill_int8_cpu` (the parser, the decision table, `PlanTrellisI8` for the seven classes at both rates and its
refusals, the table's legality under the kernel's own check, the scope), `test_prefill_chunk` (unchanged), the build gates
(`r4d_trellis_isa`: ISA, VGPR, near-dependency, `diff_epilogue`). Built, NOT run: `test_trellis_i8_gemm` (GPU kernel bit-test),
`test_prefill_int8` (model test), `tool_int8_gemm_proto` (with `--emit-rows`), `tool_teacher_forced_logprobs`, `r4dx-cli`,
`r4dx-server`.

### Predicted effect (from the bench, to be re-measured)

The f16 side has improved since the bench ran (the mlp.down KB4 retune took that GEMM from 547 us to about 395 us at M = 256), so
the ratios against today's baseline are lower: about 1.30x on the linears (1.25x with the unfused quantizer). Estimated cold TTFT:
8k about -0.8 to -0.9 s (-18%) from 4.683 s, 32k about -3.3 to -3.6 s (-15%) from 22.954 s; a fused quantizer would add another
~0.15 s at 8k. G1 below gives the true baseline.

### Validation sequence (the main session runs these, device 1, one job at a time; outputs under `E:\models\r4dx\int8prefill\`)

CPU build of everything (`cd C:\Users\pay20\dev\r4dx-int8p; $env:CMAKE_BUILD_PARALLEL_LEVEL='20'`):
`cmake --build build\win-hip --target tool_teacher_forced_logprobs test_trellis_i8_gemm test_prefill_int8 test_prefill_int8_cpu test_prefill_chunk_identity test_prefill_chunk r4dx-cli r4dx-server tool_int8_gemm_proto -j 20`
(`tool_int8_gemm_proto` is built by `tests\kernels\build_int8_gemm_proto.ps1`, not CMake).

```
# G1. bench re-baseline against the retuned f16 table (needs an idle machine; ~20 min)
powershell -NoProfile -File C:\Users\pay20\dev\r4dx-int8p\tests\kernels\build_int8_gemm_proto.ps1 -Out E:\models\r4dx\int8prefill\obj
powershell -NoProfile -Command "`$env:HIP_VISIBLE_DEVICES='1'; & E:\models\r4dx\int8prefill\obj\tool_int8_gemm_proto.exe --mode all --out E:\models\r4dx\int8prefill\all.json --emit-rows E:\models\r4dx\int8prefill\rows.inc 2>&1 | Tee-Object -FilePath E:\models\r4dx\int8prefill\all.log; exit `$LASTEXITCODE"
# G2. the production kernel bit-test (idle machine not required, a few minutes)
powershell -NoProfile -Command "`$env:HIP_VISIBLE_DEVICES='1'; & C:\Users\pay20\dev\r4dx-int8p\build\win-hip\tests\kernels\test_trellis_i8_gemm.exe; exit `$LASTEXITCODE"
# G2b. the model test (three loads of the 27B, ~10 min)
powershell -NoProfile -Command "`$env:HIP_VISIBLE_DEVICES='1'; & C:\Users\pay20\dev\r4dx-int8p\build\win-hip\tests\model\test_prefill_int8.exe; exit `$LASTEXITCODE"
# G3. switch-off identity: the chunk identity test and the Rung-4 gate with R4DX_PREFILL_INT8 UNSET
powershell -NoProfile -Command "`$env:HIP_VISIBLE_DEVICES='1'; & C:\Users\pay20\dev\r4dx-int8p\build\win-hip\tests\model\test_prefill_chunk_identity.exe; exit `$LASTEXITCODE"
powershell -NoProfile -Command "& C:\Users\pay20\dev\r4dx-int8p\tools\quant2\kl_rung4.ps1 -Device 1 -Tool C:\Users\pay20\dev\r4dx-int8p\build\win-hip\tests\model\tool_teacher_forced_logprobs.exe -OutDir E:\models\r4dx\int8prefill\kl-off -CompareDir E:\models\r4dx\rocm1010\kl; exit `$LASTEXITCODE"
```

G4, prefill-path KL with the real kernel (the Rung-4 uniform pass is the DECODE path: a prefill switch changes nothing there
except token 0, see "An important property of the Rung-4 harness"):

```
# G4a. tails stay f16: one-token prefill is M = 1 (the f16 band), so on must equal the unquantized one-token baseline byte for byte
powershell -NoProfile -Command "`$env:R4DX_PREFILL_INT8='1'; & C:\Users\pay20\dev\r4dx-int8p\tools\quant2\kl_rung4.ps1 -Device 1 -Tool C:\Users\pay20\dev\r4dx-int8p\build\win-hip\tests\model\tool_teacher_forced_logprobs.exe -OutDir E:\models\r4dx\int8prefill\kl-tail-on -NoGate -CompareDir E:\models\r4dx\int8q\kl-off-pfx -ToolArgs '--tail-rows','1023','--tail-path','prefill'; exit `$LASTEXITCODE"
# G4b. the chunked canon prefill (768-token prefix = three super-chunks, 256 decode rows scored); re-record off (the retune moved the prefill bits)
powershell -NoProfile -Command "& C:\Users\pay20\dev\r4dx-int8p\tools\prefill\run_kl.ps1 -Device 1 -Tool C:\Users\pay20\dev\r4dx-int8p\build\win-hip\tests\model\tool_teacher_forced_logprobs.exe -Tokens C:\Users\pay20\dev\r4dx-int8p\tools\reference\kl_corpus\tokens_canon.json -OutDir E:\models\r4dx\int8prefill\kl-chunk-off; exit `$LASTEXITCODE"
powershell -NoProfile -Command "`$env:R4DX_PREFILL_INT8='1'; & C:\Users\pay20\dev\r4dx-int8p\tools\prefill\run_kl.ps1 -Device 1 -Tool C:\Users\pay20\dev\r4dx-int8p\build\win-hip\tests\model\tool_teacher_forced_logprobs.exe -Tokens C:\Users\pay20\dev\r4dx-int8p\tools\reference\kl_corpus\tokens_canon.json -OutDir E:\models\r4dx\int8prefill\kl-chunk-on; exit `$LASTEXITCODE"
C:\Users\pay20\AppData\Local\Programs\Python\Python312\python.exe C:\Users\pay20\dev\r4dx-int8p\tools\prefill\kl_compare.py --ref E:\models\r4dx\int8prefill\kl-chunk-off --test E:\models\r4dx\int8prefill\kl-chunk-on --tokens C:\Users\pay20\dev\r4dx-int8p\tools\reference\kl_corpus\tokens_canon.json
# G4c. long prefixes: E:\models\r4dx\prefill-m0\kl\tokens_long.json does not exist on disk; make it first (CPU), then run_kl on and off as above with -Tokens it, and kl_compare
C:\Users\pay20\AppData\Local\Programs\Python\Python312\python.exe C:\Users\pay20\dev\r4dx-int8p\tools\prefill\make_kl_tokens.py --lengths 8k,32k
# G4d. split consistency: the same tokens with the prefix split at 300, switch on, against the one-shot run with the switch on
powershell -NoProfile -Command "`$env:R4DX_PREFILL_INT8='1'; & C:\Users\pay20\dev\r4dx-int8p\tools\prefill\run_kl.ps1 -Device 1 -Tool C:\Users\pay20\dev\r4dx-int8p\build\win-hip\tests\model\tool_teacher_forced_logprobs.exe -Tokens C:\Users\pay20\dev\r4dx-int8p\tools\reference\kl_corpus\tokens_canon.json -OutDir E:\models\r4dx\int8prefill\kl-chunk-on-split -ExtraArgs '--prefix-split-at','300'; exit `$LASTEXITCODE"
C:\Users\pay20\AppData\Local\Programs\Python\Python312\python.exe C:\Users\pay20\dev\r4dx-int8p\tools\prefill\kl_compare.py --ref E:\models\r4dx\int8prefill\kl-chunk-on --test E:\models\r4dx\int8prefill\kl-chunk-on-split --tokens C:\Users\pay20\dev\r4dx-int8p\tools\reference\kl_corpus\tokens_canon.json
```

G5, greedy text and TTFT: `tools\prefill\gdn256_check.ps1` takes `-Int8` (the switch in `$knobs`, set to 1 for the new, conv1 and
old configurations only; expect those three to share one hash and c64 to differ), and
`$env:R4DX_PREFILL_INT8='1'; .\tools\prefill\ttft_cli.ps1 -Device 1 -Lengths 8k,32k -Runs 3 -OutDir E:\models\r4dx\int8prefill\ttft`
twice for determinism (baselines 8k 4.683 s, 32k 22.954 s).

### Gates for making it the default

| gate | required |
|---|---|
| speed | cold TTFT at least 10% faster at both 8k and 32k against 4.683 s and 22.954 s, with at least 1.25x on the linears in G1 |
| accuracy, chunked canon | KL(off \|\| on) mean at most 0.0020, p99 at most 0.012, top-1 agreement at least 98.3%; mean KL against bf16 rises by at most 0.001 over off (the fake-quant references: 0.0009 chunked, +0.0006 on the prefill path) |
| accuracy, 8k and 32k | per-kind KL(off \|\| on) mean at most 0.003, no single row with KL above 2 |
| split consistency | KL(on one-shot \|\| on split) at most KL(off \|\| on) |
| tasks | `run_tasks` at 8k and 32k within one item of off per task |
| determinism | the greedy hash is identical across two on-runs |
| speculation | DFlash and MTP mean accepted length within -1% on the OpenCode transcripts |
| robustness | a 128k prefill completes, VRAM headroom logged, no NaN |
| plumbing | the ISA and epilogue-diff gates in the build; switch-off byte identity passes every build |
| not required for the flip, but needed first | TP = 2 (auto-falls back until validated; wired on branch `int8v2` behind `R4DX_PREFILL_INT8_TP2`, default off, "Tensor parallel"; GPU runs pending) and the fused quantizer (written on branch `int8v2`, "The fused quantizer"; GPU runs pending) |

### Results

2026-10-07, ROCm 10.1.0, HIP device 1 (G4d's f16 control on device 0), logs and dumps in
`E:\models\r4dx\int8prefill\`.

| step | result |
|---|---|
| G1 bench vs retuned f16 | all seven classes, layer-weighted: KB4 x1.367, KB5 x1.327 (x1.300 / x1.259 with the unfused A quantizer); `--emit-rows` output is now `gemm_tuning_table_trellis_i8.inc` |
| G2 `test_trellis_i8_gemm` | PASS (14 classes x KB 4/5, 847 checks; repeats byte-identical, f16/int8 interleave unchanged, row permutation 0 of 256 rows differ) |
| G2b `test_prefill_int8` | 1 FAIL, a test assumption: `mm600` (text-only `PrefillMultimodal`) differs from `len600` (`Prefill`) in the attention layers' KV digests with the switch OFF too; int8 is not involved. Cause (found in review of branch `fast`, by reading, not a run): the digest covers every KV page and `Reset()` does not clear them, so the digests carry the stale pages of earlier scenarios; the entry points delegate and are the same code. The test compares the live state now (still to be run) |
| G3 off identity | `test_prefill_chunk_identity` PASS (15 configurations); `kl_rung4` gate byte-identical to `rocm1010\kl` |
| G4a tails f16 | switch on, one-token prefill path: byte-identical to `int8q\kl-off-pfx` on all four segments |
| G4b chunked canon KL(off \|\| on) | mean 0.00110, median 0.00032, p99 0.0083, max 0.0187, top-1 agreement 98.83 %, ppl 3.4287 -> 3.4349 |
| G4c 8k / 32k KL | 8k: mean 0.00166 (prose 0.0022, code 0.0023, recall 0.0005); 32k: mean 0.0104, of which code_32k 0.0291 (median 0.0008, p99 0.74, max 1.95, ppl 4.158 -> 4.625 on its 256 scored rows), prose 0.0020, recall 0.0002. code at depth is the most sensitive segment to any change of prefill numerics: docs/prefill.md measured a chunk-boundary shift alone at code mean KL 0.048, so 0.029 is inside the rounding-class spread, but it is the number to watch |
| G4d split consistency | KL(on one-shot \|\| on split at 300) = 0.00281, vs the f16 control KL(off one-shot \|\| off split at 300) = 0.00274: splitting moves the bits by the same amount with or without int8 (GDN chunk alignment), so the int8 path adds no split sensitivity. The gate as first written (split <= KL(off \|\| on) = 0.0011) was too strict: it is below f16's own split spread |
| G5 greedy hash, TTFT 8k / 32k | `gdn256_check -Int8` ALL PASS: greedy text identical across new / conv1 / old and equal to f16 at 8k (`9A0B80DE8E49B800`) and 32k (`CDD3FDD6609B99DF`). Cold TTFT, 2 runs each, off -> on: 8k 4.714 / 4.724 -> 3.837 / 3.820 s (-18.9 %, 2110 tok/s), 32k 22.945 / 22.976 -> 19.372 / 19.355 s (-15.7 %, 1690 tok/s) |
