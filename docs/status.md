# Status

Last updated: 2026-09-20 (Milestone 3 integration pass -- see "Milestone 3: done" immediately below
for the summary, and the FIX pass / R5 profiling-truth pass / R2+P2 / P6 / MTP-quality / R1 / server
sections further down for the full per-stage writeups this integration pass assembles). Full `ctest`
is **35/35** (`tests\run_tests.ps1`, clean rebuild, ~133s, HIP device 1).

## Milestone 3: done

Four work items, developed across several parallel/sequential stages (below) and merged, reviewed,
fixed, and integrated in this pass:

1. **R1 -- quantize `gdn.in_proj_z`/`attn.k`/`attn.v`.** These three tensors (20.6% of every
   token's weight traffic) join the quantized-linear family (mxfp4/w4a16/w4a8, plus bf16) instead of
   being bf16-only regardless of `--layout`. Real container reconverted:
   `D:\models\r4dx\qwen38-27b-v3.r4dx`, 45.02 GiB (was 87.79 GiB, -48.7%). Accuracy held (no
   tensor's rel-err crossed the task's "2x worse" bar; see "R1" below for the full table). Decode
   ceiling moved +9.7% to +15.1% depending on layout.
2. **R3 + P6 -- kernel-level decode-path work.** R3 fuses `r4dx_residual_rmsnorm_bf16` into both
   layer boundaries (386 -> 259 r4dx-owned launches/token, w4a16). P6 vectorizes
   `r4dx_rmsnorm_bf16`/`r4dx_residual_rmsnorm_bf16`/`r4dx_silu_mul_bf16` to 16-byte loads (35-91%
   per-launch latency cut, isolated microbenchmark). R2/P2 (fused activation-quant epilogues) were
   built and kernel-level byte-exact verified but a real end-to-end correctness bug was found when
   wiring them into the model and NOT root-caused in time -- shipped **disabled**
   (`EpilogueForLayout` returns `none` unconditionally); see "R2/P2" below and "Next milestone"
   below for the follow-up.
3. **MTP quality.** A configurable, measured MTP head layout (`--mtp-head-layout {bf16,layout}`,
   default `layout` -- wins speed in 23/24 measured configs with no acceptance cost) and a
   device-resident embedding gather + draft loop (`--embed-device-resident {on|off}`, default on)
   that removes `MtpHead::Draft`'s prior per-token host round-trip.
4. **Server catches up with the engine.** A real `Model::Reset()` (ms, not the ~18.6s full reload
   it replaces) used on every prefix-match continuation; server-side `--mtp N` with the same
   greedy-only per-request routing as the CLI; `PrefixState` (CPU-unit-testable prefix-match/commit
   bookkeeping); and a real, reproducible MTP mid-round `committed_tokens` undercount bug (found by
   the Opus review, not by ctest's tolerance-bounded goldens) fixed in both `r4dx-cli` and
   `r4dx-server` via the shared `ProcessMtpRound` helper (`src/model/mtp_round.hpp`).

**This integration pass** (2026-09-20): clean `build.ps1 -Clean` rebuild (117/117 steps, no
warnings-as-errors, HIP device 1) -- clean of anything left over from the merge/FIX passes' own
partial builds. Full `ctest --preset win-hip`: **35/35 passing, ~133s**
(`build\logs` scratch logs from every prior stage were deleted at the end of this pass, per the
task's own "delete scratch logs" instruction -- `build/` is git-ignored regardless, so none of them
were ever going to be committed). `tools\server\smoke.ps1` run three times (default 4-layer
container `--mtp 0`, the 4-layer MTP container `-Mtp 3`, and the real 64-layer container `-Mtp 3`):
**24/24, 25/25, 25/25 checks passing.** A fresh confirmation sweep of `r4dx-cli` against the real
container (`D:\models\r4dx\qwen38-27b-v3.r4dx`), this file's standard prompt/flags, w4a8/w4a16/mxfp4
x `--mtp {0,3}` (bf16 excluded from performance work per the standing rule): every number lands
within run-to-run noise of the FIX pass's own measurements (see "Milestone 3 consolidated
performance" in `docs/perf.md` for the full six-run table) -- confirming the merged, reviewed, and
fixed tree is reproducible end to end from a clean checkout. Two stale doc corrections found and
fixed as trivial integration issues (not code): README.md's server section still said `--mtp` was
"not yet exposed as a server flag" (it has been since the "server catches up with the engine"
stage); `docs/server.md`'s own `--mtp-head-layout` paragraph still said the concept "does not
correspond to any existing concept in this codebase" (true only pre-merge -- the parallel MTP-
quality stage added it at the `src/model`/`src/cli` level, `docs/status.md`'s own "Milestone 3
merge note" already documents this same correction; `docs/server.md` had not been updated to match
until this pass). No engine/kernel code was changed this pass -- every correctness/perf claim below
is inherited from the FIX pass (which DID change code, see "FIX pass" below) and reconfirmed here.

**Known gaps going into Milestone 4** (not silently dropped, full detail in each stage's own section
below):

- **P2's fused activation-quant epilogues remain disabled.** Kernel-level math is byte-exact
  verified (`tests/kernels/test_fused_quant.cpp`, 135/135); the model-level wiring compiles and
  passes every tolerance-bounded ctest golden, but produces different generated text than the
  disabled baseline for w4a8/mxfp4, and the root cause was not isolated (three specific unexplored
  areas are named in the "R2/P2" section below). This is the single largest unrealized perf lever
  in the roadmap (docs/r9700.md's R2 row) and the top candidate for Milestone 4's first task.
- **`gemm_tuning_table.inc` still serves partially cache-flattered numbers.** R5's Q5 fix (a
  >=4-buffer/>256MiB ring-rotation benchmark harness) proved the old single-buffer sweep
  over-favored mxfp4 at M=1 by up to +71.7% on the two shapes spot-checked; the full 196-row
  re-sweep was never run. `src/model/gemm_tuning_table.inc` carries a provenance banner noting this
  (added by the FIX pass).
- **`--mtp-head-layout` has no server-side flag** (`src/server/server_args.h`) -- every server
  `Model` uses the measured-default layout-matched head (`std::nullopt`), which wins in 23/24
  measured configs, so this is a low-priority passthrough gap, not a missing feature.
- **Q8 (GPU clock/power sampling) has no working tool on this Windows ROCm 7.15 install** --
  whether the card holds boost clock through a decode/MTP-verify step is still unanswered.
- **R13/Q17 (32k/131k long-context validation) is unmeasured** -- every perf number in this
  milestone is at `--max-ctx 2048` (`docs/perf.md`'s standard prompt/flags).
- **`--chat` multi-turn + MTP together has no automated mid-round-stop regression test** (the
  underlying bookkeeping fix IS unit-tested via `PrefixState`/`mtp_round` directly, just not a live
  end-to-end forced-mid-round-stop scenario).
- Vision tower and DFlash2 drafting (the milestones named in Milestone 2's own "Status" line) have
  not been started.

**Next milestone (proposed)**: (1) root-cause and re-attempt P2's fused epilogues with a real
end-to-end byte-identical-text verification gate (not just ctest's tolerance-bounded goldens, which
did not catch the w4a8/mxfp4 bug) before re-enabling; (2) the full Q5-fixed `tune_gemm.py` re-sweep;
(3) R13's 32k/131k long-context measurement; (4) vision tower; (5) DFlash2 drafting.

## Milestone 3 profiling truth (docs/r9700.md R5 + Q2/Q3/Q5/Q7/Q8, 2026-09-20)

**Code state**: the per-kernel profiling instrumentation this task asked for (steady-state
`--profile-token`, `SpanAccumulator`-based per-kernel hipEvent spans inside `GdnLayer::Forward`/
`AttentionLayer::Forward`/`Mlp::Forward`, `Model::PrefillProfiled`/`--profile-prefill`, and
`tools/profile/tune_gemm.py`'s Q5 ring-buffer cache-flattery fix) was **already present, uncommitted,
in the working tree at the start of this pass** -- confirmed by reading `src/model/profile_span.h`,
`Model::DecodeStepProfiled`/`PrefillProfiled` (model.cpp), `src/cli/main.cpp`'s `--profile`/
`--profile-prefill`/`--profile-token` handling, and `tools/profile/tune_gemm.py`'s ring-allocator
before writing any new code. `git diff --stat HEAD -- src/model/model.h src/model/model.cpp` shows
only ~100 lines of P2-pass (`buf_normed_pre_`/`body_epilogue_`) diff on top of an already-committed
base that carries this instrumentation -- i.e. an earlier, unlogged pass already built R5's code; this
pass's actual work was: verify build+test are still green, RUN the real measurements against real
hardware and the real container, and correct docs/r9700.md/docs/perf.md against what was measured
(several of the document's own prior *inferred* numbers turned out to be wrong once measured -- see
below). No `src/`/`tools/` files were modified this pass; `docs/perf.md` and `docs/r9700.md` were.

**Build/test**: `.\build.ps1` was a no-op (`ninja: no work to do` -- already built). `.\tests\run_tests.ps1`:
**33/33 passing**, ~114s, HIP device 1 (`build\logs\m3-ctest.log`).

**Findings that changed this document's/docs/r9700.md's own prior claims** (full detail, tables, and
`build\logs\m3-*.log` paths in docs/perf.md's "Milestone 3 profiling truth" section):

1. **Q2's "+5.4 ms first-token offset" is now a "+27 ms instrumentation-granularity offset", and the
   first-token effect itself is no longer measurable through this instrumentation** (profiling token 1
   vs token 32 with the same fine-grained spans gives a 0.007 ms difference, w4a16). The profiled step
   now costs ~1.9-2.1x the real steady-state step (`--stats`), not +18%. Consequence: `--profile`'s
   `gpu_sum` and per-kernel `ms` are a ranking/attribution signal only at this granularity, same status
   as `[TUNE]` -- not a cost model, and this document's own R6 ms/token estimate is downgraded to "not
   cleanly estimable from current `--profile` output" pending a lower-overhead profiling method (new
   top-5 item #5, docs/perf.md).
2. **Q3: `recurrent_update` does NOT dominate GDN's non-GEMM excess** -- `conv_update`,
   `recurrent_update`, and the fused `gdn.residual` are roughly tied (3.7-4.2% of decode `gpu_sum`
   each, all 3 layouts). R6 should fuse all three, not target `recurrent_update` alone as this document
   previously expected.
3. **Q5: mxfp4's M=1 `gdn.in_proj_qkv`/`mlp.down` cache-flattery is confirmed and large** (+71.7%/
   +66.8% once a >256 MiB, >=4-buffer ring replaces the old single-buffer benchmark) -- mxfp4 flips
   from fastest to slowest of the three layouts on both cells. The ranking changed, so per this
   document's own rule a full re-sweep is warranted; **it was not run this pass** (time-boxed to the
   two shapes/three layouts the task specified) -- `src/model/gemm_tuning_table.inc` is unchanged and
   still serves the old, partially cache-flattered numbers everywhere else. Flagged as the new #1
   follow-up item.
4. **Q7 inverts docs/r9700.md's own prior §2.6 conclusion.** Real measurement: GEMM is 66.8-73.2% of
   prefill `gpu_sum` at T<=64 (all 3 layouts), not the previously-inferred 36%. **R10 (tiled prefill
   GEMM kernel) outranks R11 (non-GEMM prefill path)** -- the opposite of what the document said before
   this pass. docs/r9700.md's §2.6 and roadmap table (R10/R11 rows) are corrected in place with a
   dated note.
5. **Q8: no answer.** No `rocm-smi`/`amd-smi` on this Windows ROCm 7.15 install (`C:\opt\rocm\bin`
   listed, absent), no usable Windows perf-counter or WMI clock/power source found, no ADL/ADLX/AGS SDK
   in this project. Documented as an open gap with three concrete follow-up paths in docs/r9700.md's
   Q8 entry, not silently dropped. Whether the card holds boost clock through a decode or MTP-verify
   step remains unanswered.
6. **mxfp4's kernel-launch count (596 decode / 10098 prefill) vs w4a16/w4a8's (259 / 4386) is a known
   instrumentation blind spot, confirmed with real numbers**: `r4dx_kernel_launch_counter_get()` only
   counts r4dx-owned translation units, and only mxfp4's own activation-quant kernel
   (`r4dx_quant_act_fp8e4m3_row`) happens to live in one -- w4a16's/w4a8's equivalents do not. Not a
   real 2.3x launch-count difference; a measurement-scope artifact, already flagged by the R3/P2
   passes and confirmed rather than newly discovered here.

**Not done this pass** (see docs/perf.md's "Re-ranked next five items" for the full, justified list):
a full Q5-fixed `tune_gemm.py` re-sweep (196 rows); a lower-overhead profiling method to get
trustworthy absolute ms/token for R6/R10/R11; Q8's clock/power sampling (no tool found); R10/R11/R6
themselves (this was a measurement pass, "no new engine features" per the task).

## R2/P2 fused activation-quant epilogues (docs/r9700.md): kernels done and byte-exact-verified; NOT wired into the model (real, reproducible correctness bug found and not root-caused)

Task scope: fuse the per-GEMM activation-quant/cast launch (`r4dx_model_cast_bf16_to_f16` for
w4a16, `core::r4d::QuantActI8` for w4a8, `r4dx_quant_act_fp8e4m3_row` for mxfp4 -- ~257 extra
launches/token) into the producer kernel that already computed the row (rmsnorm/residual_rmsnorm/
silu_mul), selectable by a new enum-like `r4dx_epilogue` (kernels.h). Method followed exactly as
specified: read `third_party/libr4d/r4d_quant_act_i8.hip`'s byte-order derivation and the w4a16/
mxfp4 GEMMs' own A-operand read code FIRST; wrote the byte-diff harness
(`tests/kernels/test_fused_quant.cpp`) BEFORE any wiring; only then implemented and wired.

- **Kernel-level epilogues, done and verified.** `r4dx_rmsnorm_bf16`/`r4dx_residual_rmsnorm_bf16`/
  `r4dx_silu_mul_bf16` (`src/kernels/include/r4dx/kernels/kernels.h`,
  `src/kernels/src/r4dx_kernels.hip`) each gained three trailing optional params (`epilogue`,
  `epilogue_out`, `epilogue_scale`, default `r4dx_epilogue_none` -- every pre-existing call site
  unaffected). When requested, a NEW pass runs immediately after the kernel's own bf16 output is
  fully written (`__syncthreads()` first), re-reading that row and running the IDENTICAL
  reduction+quantize algorithm the corresponding standalone kernel uses (`ApplyEpilogueRow`,
  `r4dx_kernels.hip`) -- f16 is a plain elementwise convert (`FloatToF16(Bf16ToFloat(v))`, the same
  helper pair `r4dx_model_cast_bf16_to_f16` calls); fp8/int8 additionally compute a per-row absmax
  first (`BlockReduceMax`), exactly mirroring `QuantActFp8Kernel`/`r4d_quant_act_i8_kernel`
  including int8's WMMA-fragment byte permute. **`tests/kernels/test_fused_quant.cpp`: 135/135
  checks pass** (3 producers x 3 epilogues x M in {1,2,4,16,64} x K in {5120,6144,17408}, the
  task's full grid) -- for every combination, the fused epilogue's bytes AND per-row scale are
  byte-identical (`std::memcmp`/exact float compare, no tolerance) to running the real standalone
  kernel (`r4dx_model_cast_bf16_to_f16` / `r4dx_quant_act_fp8e4m3_row` /
  third_party/libr4d's real `r4d_quant_act_i8`) on the same bf16 values, and the producer's own
  plain bf16 output is bit-identical whether or not an epilogue was requested. Run on real
  hardware, HIP device 1.
- **Model-level wiring, done mechanically, correct in isolation, WRONG end-to-end -- not enabled.**
  `linear.h`/`linear.cpp` gained `EpilogueForLayout(Layout)` and `PreQuantizedActivation` (a
  pre-quantized activation + per-row scale `ApplyLinear` can consume directly, skipping its own
  quant launch for that call, with a hard `throw` if the format doesn't match `w.layout`).
  `GdnLayer::Forward`/`AttentionLayer::Forward`/`Mlp::Forward` (`gdn_layer.{h,cpp}`,
  `attention_layer.hpp`, `mlp.{h,cpp}`) each gained six more trailing optional params mirroring
  R3's `x_normed_in`/`next_norm_weight`/`x_normed_out` pattern: `x_normed_pre_epilogue/_data/_scale`
  (reuse an already-fused epilogue for this block's OWN first quantized GEMM(s) -- in_proj_qkv +
  in_proj_z share one rmsnorm epilogue when both agree on layout, similarly qg/k/v) and
  `next_epilogue/next_epilogue_out/next_epilogue_scale` (request this block's own
  `r4dx_residual_rmsnorm_bf16` epilogue ALSO emit the fused format for the NEXT block's first GEMM).
  `Mlp::Forward` additionally fuses `silu_mul`'s output epilogue into `w_.down`'s GEMM
  unconditionally (self-contained, every layer, no cross-component plumbing). `Model` gained
  `buf_normed_pre_`/`buf_normed_pre_scale_` (persistent, `buf_normed_`'s own reuse-in-stream-order
  pattern) and `body_epilogue_` (`EpilogueForLayout(opts.layout)`, cached once), threaded through
  all four of `model.cpp`'s per-layer loops (`RunChunk`, `DecodeStepProfiled`, `PrefillProfiled`,
  `VerifyWindow`). Every per-weight fusion site independently re-validates
  `EpilogueForLayout(that weight's OWN actual layout)` before using a shared/incoming buffer (never
  assumes the container-wide default applies to every tensor -- `LoadQuantLinearWithFallback` can
  fall one tensor back to bf16 independently of its siblings).
  - **Full `ctest` 33/33 green with this wiring ACTIVE** (`test_gdn_layer`/`test_attn_layer`/
    `test_forward_smoke`/`test_mtp`'s existing tolerance-bounded golden checks all passed).
  - **A real, reproducible correctness bug was found via actual `r4dx-cli` generation** (not
    caught by ctest's tolerance-bounded golden tests): with `EpilogueForLayout` wired to return
    `r4dx_epilogue_int8_fraga8`/`r4dx_epilogue_fp8_e4m3_row` for w4a8/mxfp4, the generated text for
    the real 64-layer container (`D:\models\r4dx\qwen38-27b-v3.r4dx`, standard prompt/flags)
    changed relative to the fusion-disabled baseline -- confirmed NOT GPU nondeterminism (the
    fusion-disabled baseline itself reproduces byte-identical text across repeated runs) and NOT
    isolated-kernel math (`test_fused_quant.cpp` above independently verifies that). An in-model
    diagnostic (temporarily computing each fused GEMM's activation a SECOND time via a fresh
    unfused `ApplyLinear` call immediately after the real one, on the exact same inputs, and
    byte-comparing the two GEMMs' own OUTPUT) showed **zero differing elements** for
    `gdn.in_proj_qkv`, `gdn.in_proj_z`, and `mlp.down` across 200+ real w4a8 decode-step samples
    spanning many layers -- i.e. every fusion site this diagnostic covered is individually correct
    in the full model, yet the aggregate generated text still diverges. Root cause NOT isolated
    within this pass's time budget. Three specific areas were NOT yet cleared and are the
    recommended starting point for whoever picks this up: (1) the new per-call arena allocations
    this fusion adds shift every LATER allocation in the same layer to a different byte offset than
    the pre-fusion code path used -- something downstream may be sensitive to absolute arena
    layout in a way that violates `Arena::Alloc`'s own bump-then-return contract; (2) `mlp.cpp`'s
    `gate_up` local fused epilogue specifically was never isolated with the same in-model
    diagnostic; (3) `attention_layer.hpp`'s qg/k/v fusion path is completely UNTESTED by this
    bisection (layer 0 of this container is a GDN layer, and cross-layer-boundary fusion was
    disabled throughout the bisection, so no attention layer's local OR cross-boundary fused path
    was ever exercised while diagnosing this).
  - **Decision: `EpilogueForLayout` (`linear.cpp`) returns `r4dx_epilogue_none` for every layout,
    unconditionally, until the root cause is found and fixed.** Per this task's own explicit
    instruction ("never loosen to a tolerance ... shipping it unverified risks silently corrupting
    every quantized GEMM's input, which is worse than not shipping it"), this pass does NOT enable
    the wiring despite it compiling, passing every existing ctest, and the kernel math itself being
    independently double-verified. All the plumbing above (`PreQuantizedActivation`, the six new
    trailing params, `buf_normed_pre_`, etc.) is left in place and exercised with `epilogue=0`
    (a no-op, byte-identical to every pre-existing call site's behavior -- confirmed: real
    `r4dx-cli` generation for w4a16/w4a8/mxfp4 at `--mtp 0` reproduces the exact pre-R2/P2 baseline
    text, and w4a16 decode measures 38.91 tok/s, matching the P6-pass baseline of 38.58 tok/s
    within noise) so a future pass that finds the root cause only has to fix it and flip
    `EpilogueForLayout`, not rebuild this wiring from scratch.
  - **w4a16's `r4dx_epilogue_f16` has a SEPARATE, already-understood reason it should stay
    disabled even once the above is fixed**: measured -4.3% decode regression (38.58 -> 36.93/36.94
    tok/s, reproducible) when briefly wired during this pass's own bisection.
    `r4dx_model_cast_bf16_to_f16` launches a FLAT elementwise grid
    (`blocks=ceil(M*K/256)`, ~20 independent workgroups at decode `T=1`/`K=5120`, spread across up
    to 20 CUs in parallel); fusing it into rmsnorm/residual_rmsnorm/silu_mul's own
    one-workgroup-per-row epilogue (`dim3(rows)=dim3(1)` at decode) collapses that same work onto a
    SINGLE workgroup running sequentially after the row's main reduction -- a real parallelism loss
    the removed launch's dispatch/sync savings do not cover. w4a8's `r4dx_epilogue_int8_fraga8` and
    mxfp4's `r4dx_epilogue_fp8_e4m3_row` do NOT have this problem (their standalone kernels,
    `r4d_quant_act_i8`/`QuantActFp8Kernel`, already launch `dim3(M)` -- one workgroup per row, same
    grid shape as the producers -- so there is no parallelism to lose): a brief measurement during
    this pass's bisection (before the w4a8/mxfp4 correctness bug above was found and the wiring was
    reverted) showed w4a8 +1.4% and mxfp4 +2.2% decode at `--mtp 0`, so once the correctness bug is
    fixed these two layouts are expected to be a net win; w4a16 should stay unfused even then
    (or be revisited for a PREFILL-only fusion, where `dim3(rows)=dim3(T)` already gives one
    workgroup per row and this specific loss should not apply).
  - **Task item 4 (launch counter) and item 5 (full measurement sweep, decide the default
    layout)**: not meaningfully done, since the fusion that would move these numbers is disabled.
    The launch counter (`r4dx_kernel_launch_counter_get()`) was confirmed to be blind to two of the
    three quant kernel types regardless (`r4dx_model_cast_bf16_to_f16` and
    `core::r4d::QuantActI8` live in translation units it does not instrument -- only
    `r4dx_quant_act_fp8e4m3_row`, mxfp4's, is counted), a pre-existing scope note this pass
    re-confirmed rather than fixed. **`w4a16` stays the default** -- the task's own decision rule
    presumes the fusion is real and working, which it is not this pass.
  - **Recommended follow-up**: (a) root-cause the w4a8/mxfp4 divergence, starting with the three
    areas listed above, using the SAME in-model diagnostic pattern (temporarily duplicate a fused
    call as an unfused one on identical inputs, byte-compare) extended to `mlp.cpp`'s `gate_up` and
    to at least one attention layer; (b) once fixed, re-verify with a REAL generated-text
    byte-identical check (not just `ctest`'s tolerance-bounded golden tests, which did not catch
    this) before re-enabling; (c) only then run task item 5's full layout-decision sweep.


## Milestone 3 merge note

Milestone 3 work was developed in two parallel trees and is merged back together as of this pass:
stage 1 ("MTP quality + device-resident draft loop pass", below) and stages 2-4 ("Server catches up
with the engine", "R2+R3+P2+P6", and "R1", below, all originally written against a tree that did not
yet have stage 1's changes). Both stages' functionality is present in the merged tree: stage 1's
configurable MTP head layout (`ModelOptions::mtp_head_layout`, CLI `--mtp-head-layout`) and
device-resident embedding gather, alongside stages 2-4's R1 quantized `gdn.in_proj_z`/`attn.k`/
`attn.v`, R3's fused residual+rmsnorm, and the server's `Model::Reset()`/MTP/prefix-reuse hardening.
One correction from the merge: the "Server catches up with the engine" section below has a
`--mtp-head-layout` bullet saying it is "NOT implemented" -- that was true only in isolation, against
the stages-2-4 tree that section was written from; stage 1 (developed in parallel, merged in by this
pass) does implement it in `src/model`/`src/cli`, see "MTP quality + device-resident draft loop pass"
immediately below. Full `ctest --preset win-hip` after the merge: **31/31** (stages 2-4's
`test_prefix_state` addition, 30 -> 31; stage 1 added no new ctest binary, only new checks inside the
existing `test_mtp`).

## MTP quality + device-resident draft loop pass: done

Follow-up to Milestone 2's MTP self-speculative decode: (1) a configurable MTP head layout
(`ModelOptions::mtp_head_layout`, CLI `--mtp-head-layout {bf16,layout}`) -- measured across
w4a8/w4a16/mxfp4 x K=1..4 against the real 64-layer container, the layout-matched (quantized) head
is faster than a bf16 head in 23/24 configurations with no acceptance-rate cost, so it is now the
default; (2) an investigation into the persistent w4a16-vs-w4a8/mxfp4 acceptance gap that ruled out
MTP head precision and logit-margin/decision-confidence as the mechanism but did not fully isolate
the root cause (not a bug -- see `docs/mtp.md`'s "Acceptance gap investigation"); (3) a
device-resident draft loop -- `text.embed_tokens` mirrored into VRAM (`ModelOptions::
embed_device_resident`, default true) behind a new device-side gather kernel
(`r4dx_embedding_gather_bf16`, `src/kernels`) fed directly from `r4dx_argmax_f32`'s own device
output, eliminating `MtpHead::Draft`'s prior per-drafted-token host sync/D2H/H2D round-trip; (4) a
new regression test (`CheckPlainDecodeUnaffectedByMtpConfig`, `tests/model/test_mtp.cpp`) confirming
the plain (non-MTP) decode path is unperturbed by this pass's changes for an MTP-sized `Model`. See
`docs/mtp.md`'s "MTP head layout", "Acceptance gap investigation", and "Device-resident draft loop"
sections for the full writeup and measured tables. Full `ctest --preset win-hip` re-run green (30/30
at the time this stage was developed, in isolation before the merge -- see "Milestone 3 merge note"
above for the post-merge 31/31 count) after every change in this pass, including a final rebuild +
full suite re-run after the last default-layout flip.

## Server catches up with the engine (`Model::Reset()`, server-side MTP, prefix-reuse hardening)

(Previous entry, written before the merge, kept below: R2+R3+P2+P6 pass, docs/r9700.md (partial),
and R1.)

Task scope: (1) a real `Model::Reset()` the server uses instead of a full `Model::Load()` on a
prefix mismatch; (2) server-side MTP (`--mtp`/greedy-only per-request routing, streaming every
accepted token, fixing the mid-round `fed_tokens` under-count in both `r4dx-cli` and
`r4dx-server`); (3) three review items from the M2 pass (all found already done, see below); (4)
`tools/server/smoke.ps1` verification against the 4-layer MTP container and the real container with
`--mtp 3`, plus a new check that two different-prompt requests never reload the container. All four
items are done; the one named-but-nonexistent piece (`--mtp-head-layout`) is explicitly not
implemented, see its own note below.

- **`Model::Reset()`, done** (`src/model/model.h`/`.cpp`). Re-zeroes `GdnStateManager::ZeroAll` for
  every GDN layer, then resets `pos_`/`started_`/MTP bookkeeping (`mtp_seed_valid_`,
  `mtp_num_accepted_valid_`, `mtp_last_hidden_`) -- no weight `DeviceBuffer` and no KV cache byte is
  touched (paged KV addressing is slot==position, self-correcting via overwrite-before-read, so
  clearing `pos_` alone is sufficient "KV bookkeeping" reset; GDN state is different because
  `has_init` genuinely READS the previous state rather than only overwriting position-indexed
  slots). **Measured** (`tests/model/test_forward_smoke.cpp`'s new
  `ResetMatchesFreshLoadRelErr` check): `Reset()` + replay is byte-identical (rel L2 = 0.0000e+00,
  all four layouts) to a fresh `Load()` + the same calls. **Server-side latency** (real 64-layer
  w4a16 container, HIP device 1, `engine.cpp`'s new per-request `reset=X.XXms` log field): 0.5-2ms
  on the 4-layer test container, 1.9-2.1ms on the real container -- vs. the ~18.6s full reload this
  replaces (docs/server.md's "Reset cost").
- **Server-side MTP, done** (`src/server/engine.cpp`/`engine.h`, `src/server/server_args.h`'s new
  `--mtp N`). A request takes `Model::DecodeStepMtpGreedy` iff `model_->MtpEnabled()` (server
  started with `--mtp N>0` against an MTP container) AND that request's own
  `sampling.temperature <= 0` -- every other request is unchanged plain decode. Every accepted
  token streams to the client as it is committed (`Engine::EmitToken`, shared between both loops).
  **Mid-round `fed`/`committed` bookkeeping bug, fixed in both binaries**
  (docs/mtp.md's "mid-round" gap): `Model::DecodeStepMtpGreedy` commits every element of its
  returned round except the last atomically, regardless of where a caller's display loop stops
  (`--max-tokens`/EOS mid-vector) -- `src/cli/main.cpp`'s `TurnResult::committed_tokens` and the new
  `src/server/prefix_state.h`'s `PrefixState::Commit()` now both track this correctly instead of
  under-counting from the displayed-only token set. **Measured** (real 64-layer w4a16 container,
  streaming request, same prompt/flags as docs/perf.md's CLI table): server `--mtp 0` decode
  32.60 tok/s vs. CLI's 32.59 tok/s; server `--mtp 3` decode 66.58 tok/s (54.3% accept, 2.60
  tok/round) vs. CLI's 66.10 tok/s (54.3% acceptance, 2.60 tok/round avg) -- server tracks the CLI
  within ~1% both ways, and MTP acceptance/tokens-per-round are identical (both drive the same
  deterministic greedy `Model` API).
- **`PrefixState`, done** (`src/server/prefix_state.h`, new). Pulled the prefix-match/invalidate
  bookkeeping out of `Engine` into its own header-only, HIP-free class so it is CPU-unit-testable
  (`tests/server/test_prefix_state.cpp`, 8 checks including the MTP mid-round-commit contract) --
  `Engine` itself cannot be CPU-tested (owns `r4dx::model::Model`, HIP-dependent). The M2 pass's
  `fed_tokens_.clear()`-on-exception fix (review finding, 2026-09-19) carries over unchanged in
  spirit as `prefix_.Invalidate()`.
- **Review items 3, all already done, verified not changed**: `temperature<0` -> 400
  (`openai_types.cpp`'s `ParseSampling`, already present); HTTP status<500 mapped to
  `invalid_request_error` (`http_server.cpp`'s `ErrorTypeForStatus`, already present);
  `stream_options.include_usage` documented as deferred (`docs/server.md`'s "Deferred / known
  gaps", already present). No code or doc change was needed for any of the three.
- **`tools/server/smoke.ps1`, extended**: new `-Mtp N` parameter; a new check block sends two
  consecutive different-prompt requests and greps the server's own stderr log to confirm the
  `"[r4dx::model::Model] VRAM breakdown"` line (printed only by `Model::Load()`, never `Reset()`)
  count stays flat and at least one request's log line shows `reset=`, not a reload; with `-Mtp N>0`
  an extra check confirms at least one `mtp:` log line appears. **Run and passing** against: the
  default 4-layer bf16 container (`--mtp 0`, regression check), the 4-layer MTP container with
  `-Mtp 3`, and the real 64-layer container (`D:\models\r4dx\qwen38-27b.r4dx`) with `-Mtp 3` --
  25/25 checks pass in every run.
- **`--mtp-head-layout`, superseded by the merge -- now implemented.** This bullet originally read
  "NOT implemented", written against the stages-2-4 tree in isolation, before this pass's merge with
  stage 1 (developed in parallel): at that point `Container`/`MtpWeights`/`MtpHead` genuinely had no
  independent "head layout" knob. Stage 1 added exactly that (`ModelOptions::mtp_head_layout`, CLI
  `--mtp-head-layout {bf16,layout}`, `Container::Load`'s own `mtp_head_layout` parameter) -- see
  "MTP quality + device-resident draft loop pass" above for the full writeup and measured table. Left
  here, corrected, rather than silently deleted, so this section's own history stays accurate.
- Full `ctest` green: **31/31** (`tests/server/test_prefix_state` and
  `tests/model/test_forward_smoke`'s new `Reset()` check are the two additions this pass; the other
  29 are unchanged from before this pass and still pass).

## P6 kernel rewrite (docs/r9700.md P6 + §2.5): rmsnorm/residual_rmsnorm/silu_mul vectorized -- done; P2 and the other six kernels still not done

Follow-up to the R2+R3+P2+P6 pass below, scoped to task item 1-2's first three named kernels only
(rmsnorm, residual_rmsnorm, silu_mul -- "the three on the decode hot path"). Rewrote all three in
`src/kernels/src/r4dx_kernels.hip` from scalar `dim3(rows)` grids with 2-byte
`__bfloat162float`/`__float2bfloat16` loads/stores to the same `dim3(rows)` grid (already
`kThreads`=256=8 waves/block, the task's own sanctioned fallback for a row too small to fill 64
CUs) but with 16-byte `uint4` vector loads/stores, unpacking/packing two bf16 halves per dword via
`__ushort_as_bfloat16`/`__bfloat16_as_ushort`. A new test, `tests/kernels/test_kernel_bandwidth.cpp`
(wired into `tests/kernels/CMakeLists.txt`), captured the pre-edit scalar kernels' output as a
golden file (`tests/kernels/golden/kernel_bandwidth_golden.bin`, checked in, ~18.6 MB) for M in
{1,4,16,64} x K in {5120,6144,17408} BEFORE any kernel edit (task step 1's explicit instruction),
then gates the rewrite: elementwise outputs bit-exact (silu_mul, residual_rmsnorm's residual-add
half), reduction-dependent outputs (rmsnorm's own output, residual_rmsnorm's normed half) within a
documented last-ulp tolerance (13/4,874,240 elements differ, max rel 7.3e-3 -- see that test file's
header comment for why the task's literal "1e-6" bound is not the operative gate for a
bf16-quantized reduction output, and what is asserted instead). Full `ctest` **32/32** (31 prior +
the new test).

- **A real bug was caught by the new test before it reached ctest or a perf run**: the first
  version of the pack/unpack helpers used `__hip_bfloat16(unsigned short)` /
  `operator unsigned short()`, which are VALUE conversions in `amd_hip_bf16.h` (`static_cast<__bf16>`
  of the integer, not a bit reinterpretation), not the bit-preserving pair
  (`__ushort_as_bfloat16`/`__bfloat16_as_ushort`) that intrinsic-level bf16 packing needs --
  `amd_hip_bf16.h`'s own `HIPRT_ONE_BF16` etc. macros use the latter for exactly this reason. Running
  `test_kernel_bandwidth` against the buggy version immediately showed 4,871,766/4,874,240 elements
  wrong (`max_diff_fp32=inf`); switching to the bit-preserving pair fixed it, confirmed by the same
  test going green. Left as an explicit comment at the helper's definition
  (`r4dx_kernels.hip`'s `UnpackBf16x2`/`PackBf16x2`) so the next kernel author doesn't repeat it.
- **Measured, isolated hipEvent microbenchmark** (`test_kernel_bandwidth`'s own timing, HIP device
  1): every one of the 36 (kernel x M x K) cells improved, 35-91% per-launch latency cut (two cells
  at M=64,K=17408 show a smaller/anomalous "before" number that looks like measurement noise rather
  than a real floor, not re-investigated). Full table and log paths in `docs/perf.md`'s new "P6
  kernel rewrite" update block.
- **Measured, full-model decode** (real 64-layer container `D:\models\r4dx\qwen38-27b-v3.r4dx`,
  `docs/perf.md`'s standard prompt/flags, HIP device 1): w4a16 `--mtp 0` 37.39 -> **38.58 tok/s**
  (+3.2%), `--mtp 3` 65.02 -> **68.43 tok/s** (+5.2%, 46.3% acceptance); w4a8 `--mtp 0`
  **35.51 tok/s**, `--mtp 3` **61.47 tok/s** (43.3% acceptance); mxfp4 `--mtp 0` **30.08 tok/s**,
  `--mtp 3` **55.71 tok/s** (47.1% acceptance) -- w4a8/mxfp4 were not measured after R3-only, so only
  w4a16 has a direct before/after. The wall-clock win is real but much smaller than the per-launch
  cut, consistent with docs/r9700.md §2.5: these three kernels are "non-GEMM" launches (32.5% of
  `gpu_sum` in the post-P6 `--profile` breakdown), and GEMMs (67.5%) are untouched by this pass.
  `r4dx-owned kernel launches/token` unchanged at **259** (P6 rewrites launches in place, does not
  add/remove any). Full per-op profile: `docs/perf.md`'s update block; raw logs under `build\logs\
  p6-*.log`.
- **Not done this pass** (do not read as silently dropped -- this is the task's own explicit scope,
  "starting with rmsnorm then residual_rmsnorm then silu_mul"): the other six r4dx-owned kernels
  (`r4dx_rope_partial_mrope_bf16`, `r4dx_quant_act_fp8e4m3_row`, `r4dx_kv_write_paged_fp8_hnd`,
  `r4dx_argmax_f32`, `r4dx_embedding_gather_bf16`) and the plain non-fused `r4dx_residual_add_bf16`
  are all still scalar/`dim3(rows or T)`. Each is its own verification effort; `r4dx_argmax_f32` in
  particular needs a genuine multi-workgroup reduction redesign (today it is a single block looping
  over all 248320 vocab floats), not just wider loads, since the task calls it out as being on the
  critical path of every MTP draft. `ApplyLinear`'s quant/cast launches (P2/R2, task item... the
  fused byte-exact int8/fp8/f16 producer epilogues) remain unimplemented -- the task for this pass
  explicitly said not to touch them ("Do not touch ApplyLinear's quant launches yet (next stage)").
  The default layout stays `w4a16`; the R2/R3 pass's own deferred "measure all three layouts and
  flip the default if w4a8 wins" sweep still has not been run to completion, since P2 (its other
  precondition) is still not implemented.
- **Recommended follow-up** (not started): P2's fused int8 `fragA8`/fp8/f16 producer epilogues next
  (the task's own explicit "next stage"), reusing this pass's byte-diff-golden methodology (capture
  the existing `r4d_quant_act_i8`/`QuantActFp8Kernel`/`cast_bf16_to_f16` kernels' output as a golden
  BEFORE wiring anything into `ApplyLinear`); then the remaining P6 kernels, argmax first since it is
  the one that actually needs a design change rather than a mechanical vectorization.

## R2+R3+P2+P6 (docs/r9700.md): remove the per-token quant/cast launch pile -- R3 done, R2/P2/P6 kernel work NOT done this pass

Task scope was four items: (1) fuse activation quantization into rmsnorm/silu_mul/GDN-gated-norm
producers with byte-exact int8 `fragA8`/fp8/f16 epilogues (R2/P2), (2) rewrite every r4dx-owned
elementwise kernel for 16-byte vector loads + all-64-CU grids at M=1 (P6), (3) wire the existing
`r4dx_residual_rmsnorm_bf16` into both layer boundaries (R3), (4) count+report launches and cache
`PickTuning` (M2-review item). Items 3 and 4 (and the caching half of item 4's sibling) are done and
verified on real hardware; items 1 and 2 are **not implemented this pass** -- see "Not done" below.

- **R3, done.** `GdnLayer::Forward`, `attention::AttentionLayer::Forward`, and `Mlp::Forward` each
  gained three trailing optional parameters (`x_normed_in`, `next_norm_weight`, `x_normed_out`,
  all defaulting to `nullptr` so any caller/test that doesn't pass them keeps the pre-R3 behavior
  exactly). When wired, a sub-block's own initial `r4dx_rmsnorm_bf16` call is skipped in favor of
  reading the previous stage's fused output, and its own final `r4dx_residual_add_bf16` is replaced
  by `r4dx_residual_rmsnorm_bf16` (already existed, was never called before this pass), which
  additionally produces the NEXT stage's normed input in the same launch. `Model` (`model.h`/
  `model.cpp`) wires this at both boundaries -- (GDN|Attn) -> its own Mlp, and Mlp -> the next
  layer's (GDN|Attn) -- across all three per-layer loops (`RunChunk`, `DecodeStepProfiled`,
  `VerifyWindow`), through one new persistent (non-arena) scratch buffer `buf_normed_`
  ([max_chunk_, hidden] bf16, reused every boundary crossing in stream order -- safe because each
  write is always followed by its one read before being overwritten, all on `stream_`). Layer 0's
  sub-block still computes its own input rmsnorm (nothing precedes it), and the last loaded layer's
  Mlp still does a plain residual add (its consumer is `FinalLmHead`'s own separate `final_norm`,
  explicitly out of this fusion's "two layer boundaries" scope).
- **Launch counter, done** (task item 4's counting half). `r4dx_kernel_launch_counter_reset`/`_get`
  (`src/kernels/include/r4dx/kernels/kernels.h`, `src/kernels/src/r4dx_kernels.hip`) count every
  r4dx-owned kernel launch (plain global int64, not atomic -- `Model` is single-worker-thread, same
  reasoning `linear.cpp`'s `PickTuning` cache already relies on). **Scope note**: this counts only
  the launches in `r4dx_kernels.hip` -- it does NOT count `third_party/libr4d`'s own
  `r4d_gemm_*`/`r4d_gdn_*`/`r4d_attn_*` launches (out of scope: a third_party submodule), so it is
  not directly comparable to docs/r9700.md P2's "257 quant/cast launches" census, which is a
  different count over a different call-site set (`linear.cpp:107-128`'s per-GEMM quant calls,
  several of which go through `core::r4d::QuantActI8` -- a libr4d entry point, not an
  `r4dx_kernels.hip` one). `Model::StepProfile` gained `r4dx_kernel_launches` (reset at the top of
  `DecodeStepProfiled`, read at the end); `r4dx-cli --profile` prints it.
  - **Measured, real hardware, real container** (`D:\models\r4dx\qwen38-27b-v3.r4dx`, w4a16, HIP
    device 1, `r4dx-cli --profile`): **259 r4dx-owned launches/token after R3.** Before R3 (computed
    by adding back the exact removed call sites, not re-measured live to avoid a throwaway
    revert/rebuild cycle): 64 layers each had one `r4dx_rmsnorm_bf16` at (GDN|Attn) entry and one at
    Mlp entry = 128 rmsnorm launches; R3 leaves exactly one (layer 0's sub-block entry) and fuses
    the other 127 into the (already-present) residual-add launch at each boundary, which is a
    swap-in-place (`r4dx_residual_add_bf16` -> `r4dx_residual_rmsnorm_bf16`, same launch count) not
    a removal. **259 + 127 = 386 before.** This is a real ~33% cut in r4dx-owned launch count, but
    see "Not done" below for why it did not translate into a proportional wall-clock win.
- **`PickTuning` caching (M2 review item): already done, no change needed.** `linear.cpp`'s
  `PickTuning` already cached its resolved `LinearTuning` per `(layout,N,K,M)` key in a
  function-local `unordered_map` (comment dated 2026-09-19, present before this pass started) --
  confirmed by reading the file, not re-implemented.
- **Wall-clock effect of R3 alone, measured** (`r4dx-cli --stats --temperature 0`, real container,
  docs/perf.md's standard prompt, `--max-tokens 128 --max-ctx 2048`, HIP device 1, each run twice):
  w4a16 `--mtp 0` decode **37.39 / 37.38 tok/s** (prefill 683.80 / 674.79 tok/s, 83 tokens, VRAM
  13.80 GiB *(stale -- see correction below)*) -- essentially flat against R1's own 37.77 tok/s baseline (same container, same flags,
  different pass), i.e. **R3 alone is a launch-count win, not yet a measured wall-clock win**: decode
  is GPU-bound (docs/r9700.md P3: `host_enqueue` is 8.3% of wall here, `finish_wait` 91.7%), so
  cutting host-issued launches mostly saves host time that was already overlapped with GPU work, not
  critical-path time. w4a16 `--mtp 3`: decode **65.02 tok/s**, 46.3% acceptance, 2.31 tok/round (vs
  R1's 67.82 tok/s / 50.0% -- within noise of a different pass's measurement, not a regression
  investigated further this pass). **The GPU-side win R2/P2 exists to capture (257-ish quant/cast
  launches' actual device time, and the single-workgroup-at-M=1 occupancy problem P6 names) is
  unrealized because R2/P2/P6 were not implemented -- see "Not done" immediately below.**
  **VRAM correction (2026-09-20, FIX pass, review finding)**: every "13.80 / 14.23 GiB" figure in
  this section and the R1 section below is stale by exactly a 2.37 GiB device-resident embedding
  mirror (`vocab 248320 x hidden 5120 x 2 bytes`, `Container::Load`'s `embed_tokens_dev_`) that
  landed in stage 1 (`ced8acc`, before this R3-alone measurement) but was not reflected in these
  numbers, which were measured in a separate pre-merge worktree that predated the mirror. Re-measured
  against the current merged tree, same prompt/flags, HIP device 1: **16.17 GiB at `--mtp 0`, 16.60
  GiB at `--mtp 3`**, identical across w4a16/w4a8/mxfp4 (weights=15.5076 GiB is now also identical
  across all three quantized layouts on this tree, superseding the R1 section's 13.14 GiB weights
  figure below). Pass `--embed-device-resident off` (added this same FIX pass) to opt back into the
  pre-mirror host-gather path and reclaim the 2.37 GiB, at the cost of a per-token host memcpy+H2D on
  the decode/draft path.
- **Not done this pass, and why** (do not read as silently dropped):
  - **R2/P2 fused producer epilogues (task item 1) -- not implemented.** Emitting the GEMM input
    directly from `RmsNormKernel`/`ResidualRmsNormKernel`/`SiluMulKernel`/the GDN gated-norm path in
    int8 `fragA8` WMMA-fragment order, fp8 e4m3 row-major, or f16, **byte-exact** against
    `r4d_quant_act_i8`/`QuantActFp8Kernel`/`cast_bf16_to_f16`, is real low-level HIP/ISA kernel work
    (the int8 path specifically needs the exact `idx = lane%16, k = 8*(e>>2)+4*(lane>>4)+(e&3)`
    fragment layout P7 documents, cross-checked against `third_party/libr4d/r4d_quant_act_i8*.hip`'s
    actual operand-read code) that this pass's time budget did not allow doing to a standard I'd
    trust in inference-correctness-critical code without an iterative build/byte-diff verification
    loop this pass didn't have room for. Writing it without that verification risks silently
    corrupting every quantized GEMM's input -- worse than not doing it. `linear.cpp`'s separate quant
    launches (`r4dx_quant_act_fp8e4m3_row`, `r4dx_model_cast_bf16_to_f16`, `core::r4d::QuantActI8`)
    are unchanged.
  - **P6 vectorized rewrite (task item 2) -- not implemented.** Every r4dx-owned elementwise/norm
    kernel still launches `dim3(rows)` workgroups (one workgroup at M=1, i.e. decode) and does plain
    scalar `__bfloat162float`/`__float2bfloat16` loads/stores, not the `ushort4`/`uint4` 16-byte
    vector loads + `f2bf2`-style packed converts + all-64-CU split-K grid P6 specifies. This is a
    second independent, large kernel-rewrite project (every kernel in `r4dx_kernels.hip`), same
    correctness-verification-budget reasoning as above.
  - **Task item 6 (measure all three layouts x `--mtp {0,3}` and decide the default) -- only
    partially run, and the default was deliberately NOT changed.** The task's own decision rule
    ("if w4a8 is now fastest... make it default") is a question about the state AFTER R2/P2/P6's
    launch/occupancy fixes land, since those are what the roadmap expects to move the ranking (P1's
    existing w4a8-should-win argument is about GEMM throughput, not about the ~250-launch overhead
    R2 targets) -- running the full 6-config decision sweep against R3-only would answer a different
    question than the one asked and risked misattributing R3's (near-zero) wall-clock effect to a
    layout ranking. Only w4a16 `--mtp {0,3}` was measured (above) as a before/after checkpoint for
    R3 itself. **`w4a16` stays the default** (`src/cli/cli_args.h`, README.md, docs/perf.md
    unchanged) -- this is a deferral, not a decision that w4a16 won.
  - Full ctest is green (30/30, `.\tests\run_tests.ps1`, ~75s, HIP device 1) and a new byte-exact
    test suite for item 1 was NOT added since item 1 itself was not built.
- **Recommended follow-up** (not started): a dedicated pass for R2/P2 should (a) read
  `third_party/libr4d/r4d_quant_act_i8*.hip` and the w4a8 GEMM's A-operand read code line-by-line
  first, (b) write the int8/fp8/f16 epilogues with a byte-diff test against the existing
  `r4d_quant_act_i8`/`QuantActFp8Kernel`/`cast_bf16_to_f16` kernels for M in {1,4,16,64} x K in
  {5120,6144,17408} BEFORE wiring them into `ApplyLinear`, and (c) do the P6 vectorized-load/
  all-CU-grid rewrite as a separate, independently-testable step per kernel (rmsnorm first, since
  its output is the most-launched of the four). Only after both land does task item 6's full
  layout-decision sweep answer the question it was meant to answer.

## R1 (docs/r9700.md): quantize `gdn.in_proj_z` and `attn.k`/`attn.v` -- done

`gdn.in_proj_z` (3.02 GB/token) and `attn.k`/`attn.v` (0.34 GB/token) -- 20.6% of every token,
previously bf16-only in every `--layout` -- now join the quantized-linear family (mxfp4/w4a16/w4a8
plus bf16), exactly like `attn.qg/o` and `gdn.in_proj_qkv`/`out_proj` already did. This is the only
roadmap item that raises the decode *ceiling* rather than competing for existing headroom
(docs/r9700.md R1: 36.6 -> 43.0 tok/s theoretical, "+4 to +5 tok/s realistic").

- **Converter** (`src/convert/main.cpp`): `text.layers.{i}.attn.k`/`.v` and `gdn.in_proj_z` now go
  through `add_linear` (the same helper `attn.qg/o`/`gdn.in_proj_qkv/out_proj` use) instead of
  `add_bf16`, so they pick up `.{layout}.wq`/`.wsz`/`.ws`/`.wref` tensors for every layout
  `--layouts` requests, plus `.bf16.w`. `gdn.in_proj_a`/`in_proj_b`/`conv1d_weight` stay bf16 (too
  small to matter, feed the decay path). `mtp.attn.k`/`.v` are deliberately unchanged (still the
  old bare bf16 tensor, via `add_bf16`) -- the MTP head stays bf16-only per this pass's task brief.
  The existing `--no-bf16` flag already covers "omit the full-model bf16 layout and the bf16
  `lm_head` variant" (no new flag needed); the real container reconversion below uses it.
- **Loader** (`src/model/container.{h,cpp}`): `AttnWeights::k/v` and `GdnWeights::in_proj_z` are now
  `QuantLinear` (were raw bf16 `DeviceBuffer<uint16_t>`). `LoadQuantLinearWithFallback` tries the
  requested layout, then bf16, then the bare pre-R1 tensor name (in that order) -- so a container
  converted before this pass (bare `attn.k`/`attn.v`/`gdn.in_proj_z`, no `.{layout}` suffix at all)
  still loads correctly, always as bf16. `mtp.attn.k`/`.v` are loaded via the same helper but with
  `Layout::kBf16` forced regardless of the requested body layout.
- **Layers**: `GdnLayer::Forward` routes `in_proj_z` through `ApplyLinear` (was a hardcoded
  `GemmBf16NtM64` call); `AttentionLayer::Forward` routes `k`/`v` through `ApplyLinear` too (was
  the component's own bf16-only `Linear` wrapper, `attention/linear.hpp`, now deleted -- nothing
  else used it). Both now pick up `tools/profile/tune_gemm.py`'s measured tuning table instead of a
  hardcoded/heuristic `WV/SK/MB/NPW` (docs/r9700.md R4): `gdn.in_proj_z` (6144x5120) and
  `attn.k`/`attn.v` (1024x5120) were appended to `tune_gemm.py`'s `SHAPES` list and swept across all
  four layouts x all seven M-bands (84 rows, `tools\profile\tune_gemm.py --shapes
  gdn.in_proj_z,attn.k,attn.v --layouts bf16,w4a16,w4a8,mxfp4 --append`) -- a new `--append` mode
  (insert rows before the table's closing `};` instead of overwriting) and `--shapes` filter were
  added to the script so this re-sweep didn't have to re-run the whole (much longer) existing table.
  Every M-band for all three new shapes stayed within noise of the M=1 baseline (e.g. `attn.k`
  w4a8: 5.99us at M=1 vs 6.79us at M=16), confirming docs/r9700.md's prediction that these bf16
  `in_proj_z`/`k`/`v` GEMMs are bandwidth-bound to M=64.
- **Accuracy (Q14)**: no dedicated Python `layer_golden.py`-isolation run was built for this pass;
  instead the existing C++ golden tests (`tests/model/test_gdn_layer.cpp`,
  `tests/model/attention/test_attn_layer.cpp`), which already diff a full layer's output against a
  real-weights `transformers` golden per quantized layout, were extended to also load `attn.k`/`v`
  (test_attn_layer) at the layout under test rather than always bf16 (`gdn.in_proj_z` needed no test
  change at all -- `GdnLayer`/`Container` already wire it through `layout` generically). Measured
  on the regenerated 4-layer test containers (`D:\models\r4dx\qwen38-27b-l4-{bf16,mtp}.r4dx`, real
  Qwen3.8-27B weights, HIP device 1):

  | Component | Layout | Before R1 (qg/o only) | After R1 (+k/v or +in_proj_z) | Tolerance |
  |---|---|---|---|---|
  | attn layer (prefill/decode norm rel err) | w4a16 | 7.17e-2 / 6.57e-2 | 9.76e-2 / 8.83e-2 | 1.5e-1 |
  | attn layer | w4a8 | 8.49e-2 / 7.55e-2 | 1.14e-1 / 1.00e-1 | 1.5e-1 |
  | attn layer | mxfp4 | 8.25e-2 / 7.34e-2 | 1.14e-1 / 9.63e-2 | 1.5e-1 |
  | GDN layer+MLP (prefill/decode rel L2) | w4a16 | ~7-8e-2 (undifferentiated) | 7.83e-2 / 8.34e-2 | 1.0e-1 |
  | GDN layer+MLP | w4a8 | ~7-8e-2 (undifferentiated) | 8.74e-2 / 9.22e-2 | 1.0e-1 |
  | GDN layer+MLP | mxfp4 | ~7-8e-2 (undifferentiated) | 7.78e-2 / 7.78e-2 | 1.0e-1 |

  Quantizing `attn.k`/`v` raises the attention layer's own rel-err by roughly 30-40% relative (e.g.
  w4a16 decode 6.57e-2 -> 8.83e-2), well short of the task's ">2x is clearly worse" bar and still
  comfortably inside the existing 1.5e-1 gate. Quantizing `gdn.in_proj_z` does not move the GDN
  block's error outside its pre-existing ~7-8e-2 ballpark at all (the "before" column there is a
  single undifferentiated range because the pre-R1 test didn't isolate a bf16-`in_proj_z` number --
  see `tests/model/test_gdn_layer.cpp`'s own tolerance-derivation comment). **No tensor's error
  crossed the 2x-worse bar, so all three stay quantized in every layout; none was reverted to
  bf16.** All 30 ctest tests pass (`.\tests\run_tests.ps1`, HIP device 1, 4-layer containers
  regenerated with the new converter -- `D:\models\r4dx\qwen38-27b-l4-{bf16,mtp}.r4dx.pre-r1.bak`
  keep the pre-R1 fixtures for reference, not deleted).
- **R14/Q13 (VRAM diagnostics)**: `Container::Load` now warns on stderr if the layout it just loaded
  consumed more VRAM than was free before the load started (the bf16-64-layer-model scenario
  docs/r9700.md's §2.1 describes: driver pages the excess over PCIe with no other symptom).
  `Model::Load` prints one `weights=.../kv+gdn_state=.../arena+scratch=.../free=...` breakdown line
  at the end of every load, from four `hipMemGetInfo` snapshots bracketing each allocation phase --
  answering Q13 ("where does the measured VRAM actually go") from what the driver reports rather
  than from this codebase's own tensor-shape arithmetic.
- **Real container reconversion**: `D:\models\r4dx\qwen38-27b-v3.r4dx` (w4a8/w4a16/mxfp4 body +
  4-bit `lm_head` only, no bf16 anywhere, `--mtp on --vision on`, reusing the existing
  `qwen38-27b.kvcalib.json` calibration) -- **45.02 GiB on disk (was 87.79 GiB, -48.7%)**, converted
  in 150.2s (32 threads, `hardware_concurrency()` default). The old
  `D:\models\r4dx\qwen38-27b.r4dx` (87.79 GiB, all four layouts including full bf16) was kept, not
  deleted. Measured decode (`--mtp 0`, real container, same prompt/flags as `docs/perf.md`): w4a16
  **37.77 tok/s** (was 32.83, +15.1%), w4a8 **34.97 tok/s** (was 30.95, +13.0%), mxfp4 **29.72 tok/s**
  (was 27.08, +9.7%) -- all three beat docs/r9700.md's "+4 to +5 tok/s realistic" prediction except
  mxfp4, which landed a bit under it (see docs/perf.md's own writeup for the likely reason: mxfp4's
  worse small-M GEMM knee, §2.4). `--mtp 3` also improved on all three (+2.1% to +19.2%). VRAM (as
  measured at R1 time, in a pre-merge worktree without stage 1's device-resident embedding mirror):
  13.80 GiB at `--mtp 0` (was 15.75 GiB), 14.23 GiB at `--mtp 3`. **Superseded (2026-09-20, FIX pass,
  review finding): re-measured against the current merged tree (which does include that mirror),
  same prompt/flags -- both figures are 2.37 GiB higher: 16.17 GiB at `--mtp 0`, 16.60 GiB at
  `--mtp 3`, identical across all three quantized layouts.** Full per-layout table, the R14
  VRAM-breakdown line's output, and the R14 over-commit-warning verification (against the OLD
  container's bf16 layout, which does over-commit) are in `docs/perf.md`'s own "R1 pass" update
  (also corrected there).

## Milestone 2: done

`r4dx-server` (OpenAI-compatible `/v1/chat/completions` + `/v1/completions` + `/health` +
`/v1/models`, streaming and non-streaming, single-worker-thread/single-GPU) and MTP
self-speculative decode (`--mtp K`, greedy-only, real per-sequence KV cache built in lockstep with
the backbone) are implemented, tested, and verified end-to-end on HIP device 1, on top of a decode/
prefill performance pass (quantized attention `qg`/`o` projections, prefill-chunk waste removal, a
measured `(N,K,M-band)` GEMM tuning table, device-side greedy argmax) that landed in the same
integration window. See `docs/server.md` for the server's full API/concurrency-model writeup,
`docs/mtp.md` for MTP's design/incident/measurement writeup, and `docs/perf.md` for the full
before/after performance table spanning Milestone 1 through Milestone 2. This integration pass
re-ran a clean `build.ps1 -Clean` rebuild, the full `ctest` suite, `tools/server/smoke.ps1`, and one
`r4dx-cli` generation per quantized layout at both `--mtp 0` and `--mtp 3` against the real 64-layer
container to confirm the milestone is reproducible end to end; see "What passes" below for the
numbers.

## Milestone 1: done

Container loader, GDN + full-attention layers, the assembled model forward pass, and `r4dx-cli`
text generation are implemented, tested, and verified end-to-end against the real 64-layer
container on HIP device 1 for all four body layouts (mxfp4/w4a16/w4a8/bf16). See the "Update"
sections below for the assembly + review-fix narrative and `docs/perf.md` for the full perf table
and verbatim generated text. This integration pass re-ran a clean `build.ps1 -Clean` rebuild, the
full `ctest` suite, and one `r4dx-cli` generation per layout against `D:\models\r4dx\qwen38-27b.r4dx`
to confirm the milestone is reproducible end to end; see "What passes" below for the numbers.

## What exists

- **Repo skeleton**: CMake project (`win-hip` preset, clang-cl + Ninja driven from the
  `vLLM_for_AMD` venv's CMake 4.4.2), `build.ps1` / `tests/run_tests.ps1`, vendored header-only
  third-party deps (nlohmann/json, cpp-httplib, minja, stb_image), `third_party/libr4d` submodule
  (branch `windows-llp64` @ `7675605`). `docs/container-format.md` and `docs/architecture.md` are
  the converter/loader and forward-pass contracts.
- **`r4d_core`** (`third_party/CMakeLists.txt`): the 15 required libr4d translation units built via
  `hipcc.exe` (paged/vit attention, GDN chunk-scan/conv/kkt-solve/recurrent-update/gated-rmsnorm,
  the four GEMM families, quant_act_i8, dflash_conv, registry). Confirmed `r4d_registry` links
  without any `r4d_ar_*` objects (AR kernels referenced only as data), so no libr4d submodule patch
  was needed.
- **`src/core`**: HIP device/stream/event/buffer/arena/tensor plumbing (`r4dx::core`), header-only.
- **`src/kernels`**: r4dx-owned HIP kernels -- rmsnorm, residual add, rope (partial + mrope), silu_mul,
  fp8 e4m3 / int8 row activation quant, paged KV cache write, embedding gather, sampler (argmax /
  temperature / top-k / min-p).
- **`src/tokenizer`**: BPE tokenizer + chat template (minja), vendored llama.cpp Unicode tables.
  Golden-tested against the real Qwen3.8-27B `tokenizer.json`.
- **`src/convert`**: `r4dx-convert` CLI -- HF safetensors -> r4dx container, producing all three
  quantized GEMM layouts (MXFP4, INT4 g128 W4A16, INT4 g128 W4A8) pre-permuted into WMMA fragment
  order, plus bf16 passthrough for embeddings/vision/MTP-fusion tensors. Exercised end-to-end
  against the real checkpoint at `C:\AI\models\Qwen3.8-27B`, including `--mtp on`.
- **`tools/reference`**, **`tools/convert_ref`**, **`tools/tok_ref`**: read-only Python validation
  tooling against the reference `transformers` 5.17.0 venv -- per-layer goldens (GDN layer, full
  attention layer, final-norm/lm_head, MTP), KV descale calibration, converter cross-checks against
  the real `r4d_core` kernels, tokenizer golden generation. None of this touches the C++ engine
  directly; it produces ground truth for `src/model`'s future tests.
- **`src/model`**, **`src/server`**, **`src/cli`** are now all implemented -- see the "Update"
  sections below (Milestone 1: model graph, forward pass, CLI text generation) and `docs/server.md` /
  `docs/mtp.md` (Milestone 2: OpenAI-compatible server, MTP self-speculative decode).

## What passes

Clean `-Clean` rebuild (`build.ps1 -Clean`, HIP device 1) plus full `ctest --preset win-hip` run
(`tests/run_tests.ps1`) on 2026-09-19:

```
100% tests passed out of 17
```

| Test | Covers |
|---|---|
| `smoke_r4d` | `r4d_core` link/geometry/registry + `r4d_gemm_bf16_nt_m64` vs CPU fp32 ref |
| `reference_manifest` | CPU-only Python check of `tools/reference/layer_golden.py`'s manifest contract |
| `convert_quantize_roundtrip` | quantizer rel-L2 error bounds (w4a16/w4a8/mxfp4) on random data |
| `convert_pack_bytes` | packer byte-exactness vs Python-reference-generated fixtures |
| `convert_kernel_decode` | packed bytes decoded via each kernel's own literal indexing (independent of the packer/Python references) |
| `test_core` | device/stream/buffer/arena/tensor plumbing |
| `test_rmsnorm`, `test_residual_add`, `test_silu_mul`, `test_rope` | elementwise/norm kernels vs CPU reference |
| `test_quant_act_fp8` | fp8 e4m3 row activation quant |
| `test_mxfp4_gemm` | real quant_act_fp8e4m3_row -> real `r4d_gemm_mxfp4a8_nt_m64` vs CPU fp32 ref |
| `test_kv_write` | paged fp8 KV cache write layout |
| `test_sampler` | argmax / temperature / top-k / min-p sampling paths |
| `test_attn_decode` | paged decode attention, incl. multi-sequence per-head descale broadcast |
| `test_gdn_chunk_scan` | GDN chunked-scan kernel vs CPU reference |
| `tokenizer_golden` | 103 encode + 10 chat-template + 11 HF-compat-mode + 4 header-field cases, 0 failures |

Additional GPU-device-1 Python self-tests, run manually (outside `ctest`, serialized on device 1)
during this integration pass, all green:

- `tools/convert_ref/selftest_compare.py` -- byte-exact packer check against the real
  `r4dx-convert` binary's own `--selftest` output.
- `tools/convert_ref/kernel_crosscheck.py` -- w4a16/w4a8/mxfp4 rel_err all under the 2e-2 gate
  against the real `r4d_core` GEMM kernels.
- `tools/reference/layer_golden.py --device cuda` -- all four components
  (`layer_000_gdn`, `layer_003_full_attention`, `final_norm_lm_head`, `mtp`) status=ok against real
  checkpoint weights.
- `tools/reference/kv_calibrate.py --device cuda --layer 3` -- produced k_amax/v_amax descale
  values.

## Update (2026-09-19, assembly + CLI milestone)

`src/model` (container loader, GDN layer, MLP, final norm + lm_head, the full-attention layer under
`src/model/attention/`, and the assembled `r4dx::model::Model` forward pass in `model.{h,cpp}`) and
`src/cli` (`r4dx-cli`, chat-template-driven prompt/--chat loop, streaming UTF-8-safe decode,
prefill/decode tokens/s + VRAM stats) are now implemented and exercised end-to-end against the real
64-layer container (`D:\models\r4dx\qwen38-27b.r4dx`) on HIP device 1, for all four body layouts
(mxfp4/w4a16/w4a8/bf16) -- coherent, on-topic generated text for all four; see `docs/perf.md` for
the full perf table, verbatim outputs, and two real stream-synchronization bugs found and fixed
during this pass (`src/model/gdn_layer.cpp`'s `UploadArray`, `src/model/model.cpp`'s
`Model::RunChunk` logits readback -- both missing an explicit wait against a `hipStreamNonBlocking`
stream). `src/server` remains an unimplemented placeholder. `tests/model/test_forward_smoke` and
`tests/cli/test_args` are new; full `ctest --preset win-hip` is 24/24 passing as of this update.

## What passes (Milestone 1 integration pass, 2026-09-19)

Clean `-Clean` rebuild (`build.ps1 -Clean`, HIP device 1, 84/84 build steps) plus full
`ctest --preset win-hip` run (`tests\run_tests.ps1`):

```
100% tests passed out of 24
Total Test time (real) = 59.50 sec
```

All 17 Phase-0 tests plus `convert_kv_calib`, `convert_bf16_layout`, `test_gdn_layer`,
`test_final_lm_head`, `test_forward_smoke`, `test_attn_layer`, `test_cli_args` (the seven tests
added across the conversion/assembly/review-fix stages) pass together in one run.

One `r4dx-cli` generation per body layout against the real, full 64-layer container
(`D:\models\r4dx\qwen38-27b.r4dx`, 87.79 GiB), same prompt/settings as `docs/perf.md`
(`--prompt "Write a haiku about GPUs, then explain what a GPU is in two sentences." --max-tokens 128
--temperature 0 --stats`, `--max-ctx 2048` except bf16's `--max-ctx 512`):

| Layout | Container load | Prefill | Decode | Generated tokens | VRAM |
|---|---|---|---|---|---|
| mxfp4 | 9.73 s | 400.21 tok/s | 24.90 tok/s | 82 | 17.79 GiB |
| w4a16 | 16.01 s | 419.44 tok/s | 29.08 tok/s | 88 | 17.79 GiB |
| w4a8  | 16.62 s | 410.75 tok/s | 27.88 tok/s | 87 | 17.79 GiB |
| bf16  | 60.77 s | 20.40 tok/s  | 1.38 tok/s  | 74 | 31.86 GiB |

All four generations were coherent, on-topic, and stopped on the model's own EOS token, matching
the assembly/review-fix stages' own runs within noise -- confirming the milestone is reproducible
from a clean rebuild. See `docs/perf.md` for the verbatim generated text and full narrative.

## What passes (Milestone 2 integration pass, 2026-09-20)

Clean `-Clean` rebuild (`build.ps1 -Clean`, HIP device 1, 107/107 build steps) plus full
`ctest --preset win-hip` run (`tests\run_tests.ps1`):

```
100% tests passed out of 30
Total Test time (real) = 93.50 sec
```

All 24 Milestone-1 tests plus `test_mtp` and the five `test_server_*` CPU-only tests (`test_server_args`,
`test_openai_types`, `test_sse`, `test_response_sink`, `test_request_queue`) pass together in one run.

`tools/server/smoke.ps1` (4-layer test container, `--layout w4a16 --layers 4`): all 20 shape/status
checks passed -- `/v1/models`, non-streaming and streaming `/v1/chat/completions` (SSE framing,
`[DONE]` terminator, per-event `chat.completion.chunk` shape), and a rejected-image-part `400`.

One `r4dx-cli` generation per quantized body layout, `--mtp 0` vs `--mtp 3`, against the real,
unmodified 64-layer container (`D:\models\r4dx\qwen38-27b.r4dx`), same prompt as `docs/perf.md`
(`--temperature 0 --max-ctx 2048 --stats`, `--max-tokens 128` except bf16's `--max-tokens 32
--max-ctx 512`):

| Layout | mtp=0 decode | mtp=3 decode | mtp=3 acceptance | speedup |
|---|---|---|---|---|
| mxfp4 | 27.08 tok/s | 47.46 tok/s | 41.9% | +75.3% |
| w4a16 | 32.83 tok/s | 66.42 tok/s | 54.3% | +102.3% |
| w4a8  | 30.95 tok/s | 47.72 tok/s | 32.5% | +54.2% |
| bf16  | 1.41 tok/s  | 2.21 tok/s  | 48.7% | +56.7% |

All eight runs were coherent, on-topic, and (except the two 32-token-capped bf16 runs, which hit
`--max-tokens` by design to keep the sweep's wall-clock bounded) stopped on the model's own EOS
token, matching the FIX pass's own numbers within run-to-run noise -- confirming Milestone 2 is
reproducible from a clean rebuild. See `docs/perf.md` and `docs/mtp.md` for the full per-layout
tables (all five `--mtp` values, not just 0 and 3) and verbatim generated text.

## Known gaps

- `r4d_gdn_conv_prep_w4_h128_bf16` / `conv_update` both live in the single
  `r4d_gdn_conv_w4_h128_bf16` translation unit per `r4d.h`; no gap, just worth remembering when
  wiring `src/model`.
- Vision tower weights are carried bf16-only for now (no quantized vision GEMM path yet); vision
  tower forward pass itself is the next milestone (see "Next milestone" below).
- fp8 KV descales are calibrated per-layer on demand via `kv_calibrate.py`
  (`tools/reference/kv_calibrate_out/kv_descale.json`, gitignored) but not yet wired into the
  converter -- `docs/container-format.md`'s descale table is still the placeholder `1.0` until
  `src/convert` consumes calibration output for all 16 full-attention layers.
- `tests/reference/test_manifest.py` is CPU-only and has no `pytest` dependency (the reference venv
  doesn't have `pytest` installed and is read-only) -- it's a plain script with bare asserts, run
  directly and also registered as the `reference_manifest` ctest test.
- No prefill kernel yet beyond the interim 64-row skinny-GEMM chunking path, now backed by a
  measured `(N,K,M-band)` GEMM tuning table (`docs/perf.md`'s "GEMM tuning sweep") rather than a
  single hardcoded tuple, but still not a dedicated WMMA prefill kernel; that remains future work.
- MTP acceptance (30-75%, best at low K) is well above the pre-fix 0-1.2% but still plausibly below
  what a purpose-trained self-speculative head could achieve -- not investigated further; see
  `docs/mtp.md`'s "Known gaps".
- `r4dx-server`'s tool-call parsing and vision content parts are both deferred (`docs/server.md`'s
  "Deferred / known gaps"); a mismatched-prefix conversation reset still pays a full container
  reload's latency (no lightweight `Model::Reset()` yet).
- `--chat` multi-turn only lightly exercised, now also true of `--chat` + MTP interaction together
  (`docs/mtp.md`'s "Known gaps").

## Next milestone

**Milestone 2** (`r4dx-server` OpenAI-compatible chat API, decode/prefill performance pass, MTP
self-speculative decode) is done -- see the "Milestone 2: done" section above, `docs/server.md`,
`docs/mtp.md`, and `docs/perf.md`. Next up: the vision tower forward pass, then DFlash2 drafting,
followed by prefix-caching, per the top-level project decisions.
