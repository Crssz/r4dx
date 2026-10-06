# int8 activations in prefill: the accuracy cost, measured before any int8 GEMM is written

Status (2026-10-07, branch `int8q`): the measurement switch is written and its CPU tests pass; the GPU runs are
PENDING (the commands are below, the results table is empty).

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

## Results (to fill in)

Mean KL against the bf16 reference (kl_rung4.ps1 `[kl_rung4]` line), top-1 in %, per segment mean KL
(cpp / english / python / thai):

| run | R4DX_FAKEQ_ACT | path | mean KL | top-1 | cpp | english | python | thai |
|---|---|---|---|---|---|---|---|---|
| kl-off | unset | decode (uniform pass) | | | | | | |
| kl-off-pfx | unset | one-token prefill | | | | | | |
| kl-blk128 | blk128 | one-token prefill | | | | | | |
| kl-row | row | one-token prefill | | | | | | |
| kl-blk32 | blk32 | one-token prefill | | | | | | |

`KL(off || on)` from `kl_compare.py` (mean / p99 / max, top-1 agreement):

| on | mean KL | p99 | max | top-1 agreement |
|---|---|---|---|---|
| blk128 | | | | |
| row | | | | |
| blk32 | | | | |

Reading it: a mean-KL increase of 0.001 to 0.003 for `blk128` confirms the paper estimate and makes an int8
prefill GEMM worth building; `row` is expected to be clearly worse (the w4a8 lesson). The budget the
production container has is the gap between its KL and the 0.01 gate (docs/gemma4-plan.md, docs/quant2.md).

## Caveats

* Activations only. An int8 x int8 GEMM also needs int8 weights; the trellis weights decode to f16 codebook
  values and are not int8. Whether they can be (and at what cost) is a separate question this switch cannot
  answer.
* The GEMM still accumulates in f16 WMMA, not int32, so the accumulation noise of a real int8 kernel is not
  modeled; with `blk128` the real kernel would also have to rescale each 128-block's int32 partial sum.
* Round-trip rounding is exact in fp32 here; a real kernel's `s` may be stored in f16 or as a power of two,
  which would add error. `blk32` is the cheap bound on how much finer scales can recover.
* The one-token prefill path differs numerically from the chunked path (other GEMM and GDN kernels), so its
  baseline is not the frozen 0.00788.
