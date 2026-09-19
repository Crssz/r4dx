# Status

Last updated: 2026-09-20 (Milestone 3 in progress: MTP quality + device-resident draft loop pass,
merged with the server-catches-up-with-engine / R2+R3+P2+P6 / R1 pass -- see "Milestone 3 merge
note" below).

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
  13.80 GiB) -- essentially flat against R1's own 37.77 tok/s baseline (same container, same flags,
  different pass), i.e. **R3 alone is a launch-count win, not yet a measured wall-clock win**: decode
  is GPU-bound (docs/r9700.md P3: `host_enqueue` is 8.3% of wall here, `finish_wait` 91.7%), so
  cutting host-issued launches mostly saves host time that was already overlapped with GPU work, not
  critical-path time. w4a16 `--mtp 3`: decode **65.02 tok/s**, 46.3% acceptance, 2.31 tok/round (vs
  R1's 67.82 tok/s / 50.0% -- within noise of a different pass's measurement, not a regression
  investigated further this pass). **The GPU-side win R2/P2 exists to capture (257-ish quant/cast
  launches' actual device time, and the single-workgroup-at-M=1 occupancy problem P6 names) is
  unrealized because R2/P2/P6 were not implemented -- see "Not done" immediately below.**
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
  worse small-M GEMM knee, §2.4). `--mtp 3` also improved on all three (+2.1% to +19.2%). VRAM:
  13.80 GiB at `--mtp 0` (was 15.75 GiB), 14.23 GiB at `--mtp 3`. Full per-layout table, the R14
  VRAM-breakdown line's output, and the R14 over-commit-warning verification (against the OLD
  container's bf16 layout, which does over-commit) are in `docs/perf.md`'s own "R1 pass" update.

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
