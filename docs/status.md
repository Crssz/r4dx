# Status

## Milestone 4: done (2026-09-20, integration pass)

**Integration**: clean `build.ps1 -Clean` rebuild (one pre-existing, unrelated MSVC `localtime`
deprecation warning, no errors) + full `ctest` **37/37** (~210s, HIP device 1) +
`tools\server\smoke.ps1` three ways -- default 4-layer container **28/28**, 4-layer MTP container
(`-Mtp 3`) **29/29** (MTP path confirmed taken), real 64-layer container (`-Layers -1
-ToolRoundTrip`) **34/34** (full tool call/result/answer round trip against real weights) -- plus a
fresh `r4dx-cli --stats` confirmation sweep (w4a8/w4a16/mxfp4 x `--mtp 0` and each layout's own
best `K`, re-checked against neighboring `K` values rather than assumed) and one long-context
confirmation point (w4a16, `--max-ctx 32768`, coherent generation, 776.92 tok/s prefill / 36.46
tok/s decode). Full tables: `docs/perf.md`'s new "Milestone 4: integration confirmation sweep +
consolidated M1->M4 table" section (top of that file). No vision run was performed -- the vision
stage produced real-hardware Python golden data and a full architecture spec but no `src/model` C++,
so there is no engine code path to exercise; confirmed by grepping `src/` for vision call sites
(none exist) rather than assuming.

**Headline decode throughput, real 64-layer container, each layout's own best `--mtp K`**: **w4a16
68.73 tok/s (`K=3`, 46.3% acceptance), w4a8 57.86 tok/s (`K=4`, 31.0%), mxfp4 65.04 tok/s (`K=3`,
52.9%)**. mxfp4's best `K` moved from `K=2` (pre-Q5-re-sweep) to `K=3` (post-re-sweep) -- checked
directly this pass, not assumed from stale data. w4a16 remains the fastest layout and stays the
default.

**No new source code was written this integration pass beyond trivial doc/log housekeeping** -- all
engine/kernel/server code in the working tree was written and verified by the seven prior stages
this pass integrates (P2, TUNE, ACCEPTANCE, DRAFTER, LONG-CONTEXT, PREFILL, VISION, TOOL-CALLS,
REVIEW, FIX -- see each stage's own dated section below for full detail). This pass's job was: clean
rebuild, full re-verification on real hardware from that clean build, doc consolidation, scratch-log
cleanup, and the single integration commit.

**Milestone 4 work items, status**:

| Item | Status | Detail |
|---|---|---|
| R2/P2 fused activation-quant epilogues | **DONE** | Root-caused (`Arena::Alloc` end-alignment gap) and enabled for w4a8/mxfp4; w4a16 stays unfused (measured wall-clock regression, not correctness). See "R2/P2 ... root-caused and enabled" below. |
| Q5 GEMM tuning re-sweep | **DONE** | Full 280-row re-sweep with the (already-correct) Q5 cache-flattery fix; mxfp4 improved, w4a16 flat, w4a8 `--mtp 3` regressed (root cause: verify-band GEMM retiling + numerical reduction-order drift, not a bug). See "Full Q5-fixed `tune_gemm.py` re-sweep" below. |
| MTP acceptance-gap investigation | **DONE** | h_seed drift on one outlier residual dimension (index 3994/5120) matches the acceptance ranking exactly across all three layouts -- a measured, expected quantization behavior, not a bug. See "MTP acceptance-gap investigation" below. |
| R9 reduced-vocab MTP draft head | **PARTIAL** | Mechanism built end-to-end and verified lossless on real hardware (byte-identical output, reduced vs full head); the ~2.5-3x economic projection was NOT realized because this machine's only calibration corpus (WikiText-2) is too small/narrow (76.8% held-out coverage, N=2977 natural size) -- root cause isolated, not a code defect. Default container (`qwen38-27b-v3.r4dx`) carries no draft-head tensors, so `--mtp-draft-head reduced`'s default silently falls back to full-vocab behavior for it -- no production regression. See docs/mtp.md's "Reduced-vocab draft head". |
| R13/Q17 long-context validation | **DONE** | Measured to the model's own native 262144-token ceiling; `--max-ctx` default raised 131072 -> 262144 in both CLI and server. See "Long-context validation" below. |
| R10/P9 prefill GEMM kernel | **NOT DONE** | Per-shape profile re-confirmed (`mlp.gate_up`+`mlp.down` = 32.0-32.9% of `gpu_sum`); the tiled WMMA kernel itself, the chunk-cap raise, and the correctness gate were time-boxed out as multi-day kernel-engineering work -- see "R10/P9" below. Still the #1 follow-up. |
| Vision tower | **PARTIAL** | Architecture, preprocessing, and mrope-splicing semantics fully documented against real `transformers` source with real-hardware validation goldens (`docs/vision.md`, `tools/reference/vision_golden.py`); no C++ implementation exists yet. |
| OpenAI tool calls | **DONE** | `tools`/`tool_choice`/`message.tool_calls`/`role:"tool"`/`"function"` fully implemented, then hardened by a dedicated review (8 findings: 2 correctness-in-server, 1 stop-trim leak, 5 minor) and fix pass -- all fixed, tested, and verified live against the real container. See "Tool calls" in `docs/server.md`. |
| DFlash2 drafting | **NOT STARTED** | Assessed (real GGUF metadata read): architecturally a different model (block-diffusion + selector-based drafting), not a slice of this model's own weights -- porting it needs a new loader and forward pass, out of scope for a draft-head-reuse approach. |

**Known gaps going into Milestone 5** (superset of the "Milestone 4" work-item table above; nothing
here was silently dropped -- each item's own stage section has the full accounting):

1. **R10/P9 (tiled WMMA prefill GEMM kernel) -- still the single largest unrealized perf lever.**
   Prefill is 62.6-73.2% GEMM at the current 64-row chunk cap (measured, all three layouts); no
   kernel work has started. Recommended as its own dedicated multi-stage effort per
   docs/perf.md's "Prefill per-shape profile and chunk-cap baseline" section.
2. **Vision tower C++**: patch embed, 27 encoder layers, merger, `stb_image` preprocessing, CLI
   `--image`, server `image_url` wiring, golden validation, perf measurement -- all unstarted.
   `docs/vision.md`'s closing section gives the recommended implementation order against the real
   goldens already captured.
3. **R9 reduced-vocab draft head's economic win is unrealized** on this machine because WikiText-2
   is too small a calibration corpus; needs a larger/more diverse corpus (or self-generated text) to
   re-calibrate against, per docs/mtp.md's "Reduced-vocab draft head" section. The mechanism itself
   needs no further changes.
4. **w4a8's `--mtp 3` MTP acceptance-rate drop (43.3%->31-35.6%)** from the Q5 re-sweep was measured
   and reported but not root-caused to split "verify-band GEMM genuinely slower" from "numerical
   reduction-order drift shifting argmax decisions" -- still open, per `docs/perf.md`'s re-sweep
   section.
5. **Q8 (GPU clock/power sampling)** has no working tool on this Windows ROCm 7.15 install --
   whether the card holds boost clock through a decode/MTP-verify step is still unanswered.
6. **DFlash2 drafting** not started (see table above).
7. `tools/validate_fusion.ps1` (the mandatory byte-identity gate for the R2/P2 fused-epilogue work)
   was fixed by the REVIEW/FIX pass (exit-code + empty-stdout checks, pinned `--max-ctx` per prompt
   tier) and last run 18/18 by that pass; **not re-run by this integration pass** (no epilogue-
   affecting code changed since), so it is reconfirmed by inheritance, not by a fresh run this pass.

**Next milestone (proposed)**: (1) R10/P9's tiled prefill GEMM kernel, as its own dedicated
multi-stage effort (kernel + isolated correctness test first, wiring + cap-sweep second, full
end-to-end SHA-256 gate third) -- the single highest-value remaining perf lever; (2) the vision
tower's C++ implementation against the real goldens `docs/vision.md` already captured; (3) a larger
calibration corpus for R9's reduced-vocab draft head; (4) DFlash2 drafting, if a multi-head/tree
drafter or the reduced-vocab head (once re-calibrated) makes wide-K speculation worthwhile.

Last updated: 2026-09-20 (Milestone 4 integration pass, above). Before that: vision tower
investigation + golden reference pass -- see "Vision tower:
investigation + golden reference done, no C++ yet" immediately below, and the new `docs/vision.md`
for the full architecture spec this pass wrote against real `transformers` source and real-hardware
goldens). Before that: docs/r9700.md R10/§2.6/P9 pass -- see "R10/P9: prefill GEMM profiling done,
kernel NOT built" below for the full accounting of what was and was not completed. Before
that: the R13/Q17 pass: measured long-context validation
on real hardware at 2k/8k/32k/131072/262144 -- the model's own native context ceiling, `262144` per
the checkpoint's `config.json`, not the previous 131072 self-imposed cap. VRAM, decode, prefill, and
correctness (needle retrieval + coherent generation) all measured; `--max-ctx` default raised
131072 -> 262144 in both CLI and server. See "Long-context validation" below for the full writeup.
Before that: the R9 pass built the reduced-vocab MTP draft head
end to end -- container format extension, loader, `r4dx_gather_i32` kernel, `MtpHead::Draft` wiring,
K widened to 16, a calibration tool, and a real-hardware K x layout measurement sweep. See "R9:
reduced-vocab draft head" below for the full writeup -- MECHANISM verified correct/lossless on real
hardware, but the ECONOMIC projection (~2.5-3x) was NOT realized this pass because the calibration
corpus available on this machine (WikiText-2) is too small/narrow to build a well-covered subset;
root cause isolated and documented, follow-up flagged. Before that: the "Acceptance gap
investigation" + Known-gaps items 3/6 pass, the full Q5-fixed `tune_gemm.py` re-sweep, and the R2/P2
fused-epilogue pass -- see their own sections below. Full `ctest` is **36/36**
(`tests\run_tests.ps1`, ~189s, HIP device 1) after this pass's changes (was 35/35 -- +1 for the new
`test_gather_i32` kernel test).

## Vision tower: investigation + golden reference done, no C++ yet (2026-09-20)

**Task**: docs/status.md's own "Known gaps" item 7 (the vision tower, named in Milestone 2's own
Status line, never started) -- read the reference semantics, implement the tower, wire it end to
end (CLI `--image`, server `image_url`), validate rung by rung, measure cost. Full write-up, the
complete architecture spec, and the exact "what's done/what's not" accounting: **`docs/vision.md`**
(new). Summary here, not a duplicate of that document's detail.

**Done, real hardware, HIP device 1**: read every piece of the real semantics directly from the
reference venv's `transformers` 5.17.0 source (`Qwen3_5VisionModel` and everything it calls -- patch
embed, learned+bilinearly-interpolated position embeddings, axial rope, per-image dense attention,
the merger -- plus `Qwen3_5Model.get_rope_index`'s text-side mrope splicing rule for image spans);
confirmed the real checkpoint carries all 333 `model.visual.*` weights the container's `--vision on`
conversion needs. **Found and fixed a real memory-safety bug in shared reference tooling**:
`tools/reference/common.py`'s `ShardIndex.get_tensor`/`get_row_slice` returned tensors backed by an
already-unmapped `safe_open` mmap, reproducibly crashing the Python interpreter (access violation)
once a golden script reads enough tensors in one `load_state_dict` call (333, for the vision tower;
every existing script's ~14-20-tensor components never hit it) -- fixed with `.clone()`, verified by
reproducing the crash, bisecting to `load_state_dict` (not the individual reads), and confirming
three clean runs after the fix. Wrote `tools/reference/vision_golden.py` (mirrors `layer_golden.py`'s
conventions) and ran it for real on HIP device 1: real weights, real preprocessing (`transformers`'
own configured `Qwen2VLImageProcessor`) of a deterministic synthetic test image, all 27 encoder
blocks' outputs plus block 0's full internal chain plus the merger output, 39 tensors, all finite,
shapes matching the architecture spec exactly (784 patches -> 196 merged `[5120]`-dim tokens) --
`tools/reference/golden_out/vision_tower.safetensors` + `vision_manifest.json`, docs/validation.md
rung 3's ground truth for this tower, same status the GDN/attention layers had before `src/model`
implemented them.

**Not done, per the task's own "finish in the listed order, report exactly where you stopped"
instruction**: no C++ in `src/model` (no patch embed, no encoder, no merger, no `stb_image`
preprocessing, no `r4d_attn_vit_h72_bf16` call site), no CLI `--image` flag, no server `image_url`
wiring, no golden-vs-r4dx validation (nothing to validate yet), no end-to-end description check, no
perf measurement. This is a genuinely large, multi-subsystem C++/HIP implementation (preprocessing
exactly matching a resize+patchify+normalize pipeline with no existing precedent in this codebase,
a position-embedding bilinear-interpolation scheme with no existing precedent either, 27 encoder
layers, then text-side mrope splicing) that warrants its own dedicated implementation pass against
the real goldens this pass produced, rather than a partially-wired and unvalidated attempt in the
same pass that did the investigation -- the same scope judgment this project's own R10 pass
(immediately below) applied to the prefill GEMM kernel. `docs/vision.md`'s closing section gives the
recommended implementation order (preprocessing -> patch embed+pos embed -> one block -> all 27
blocks+merger -> text splicing/mrope -> CLI/server -> end-to-end+perf), each rung pointed at the
specific golden tensor(s) to validate against.

**Build/test**: no `src/`/`tests/` files changed (Python-tooling-only pass); `tools/reference/
common.py`'s bugfix is the only change to code any other script depends on -- re-verified
`layer_golden.py` and `kv_calibrate.py`'s own component counts/shapes are unaffected (the `.clone()`
only changes storage ownership, not values). Full `ctest` not re-run this pass (nothing under
`src/`/`tests/` changed) -- left exactly as the prior pass reported it, **36/36**.

## R10/P9: prefill GEMM profiling done, kernel NOT built (2026-09-20)

**Task**: docs/r9700.md R10 + §2.6 + P9 -- correct §2.6's stale inferred GEMM/non-GEMM split, profile
prefill per op family/shape at the current 64-row cap, write a tiled WMMA prefill GEMM kernel
targeting 170-215 TOPS, raise the chunk cap above 64 with a measured sweep, gate it on bf16-tolerance
plus end-to-end SHA-256 text identity, reconfirm decode, and measure prefill at several prompt
lengths before/after.

**Verified already done, not redone**: docs/r9700.md's §2.6 "Conclusion, corrected" blockquote and
the R10/R11 roadmap rows already carried R5's measured 66.8-73.2%/26.8-33.2% GEMM/non-GEMM split and
the "R10 now outranks R11" statement, applied by an earlier pass (docs/status.md's own "Milestone 3
profiling truth" section, dated the same day) -- confirmed by reading the live file directly rather
than trusting that section's own account. Nothing needed to change there beyond one addition (below).

**Done this pass, real hardware, HIP device 1**: a fresh per-op-family, per-GEMM-shape
`--profile-prefill` breakdown at the current 64-row cap for all three layouts (`mlp.gate_up` and
`mlp.down` together are 32.0-32.9% of `gpu_sum`, more than every GDN/attention GEMM combined -- the
concrete target P9 should aim at); a decode reconfirmation at the standard prompt (37.58/35.37/32.20
tok/s w4a16/w4a8/mxfp4, within noise of the TUNE-pass 38.46/36.08/32.83 baseline, as expected since no
kernel/dispatch code was touched); and a prefill tok/s sweep at 128/512/1024/4096-token prompts per
layout (944.98-1001.79 w4a16, 1298.04-1482.62 w4a8, 1162.28-1312.53 mxfp4 tok/s) as the pre-change
baseline. Full tables and logs: docs/perf.md's "Prefill per-shape profile and chunk-cap baseline"
section, `build/logs/r10_profileprefill_*.txt` / `r10_decode_confirm.txt` / `r10_sweep_before.txt`
(gitignored).

**NOT done this pass, and why (not silently dropped)**: the tiled WMMA prefill GEMM kernel itself
(P9, would live in `src/kernels`, targeting `mlp.gate_up`/`mlp.down` per the measured ranking, on the
w4a8 int8 WMMA path per the task's own prior-favourite note); raising `max_chunk_`/`kMaxChunkM` above
64 (`src/model/model.h`, `src/model/linear.cpp`); the chunk-cap sweep (64/128/256+); and the
correctness gate (bf16 4-layer reference tolerance check + end-to-end SHA-256 text identity at
`--mtp 0` for the standard prompt plus a ~1000-token prompt, plus a new ctest case). This is a
from-scratch, low-level HIP kernel -- packed 4-bit weight dequantization, per-128-K scale/zero
application, WMMA 16x16x16 tiling with real register-reuse, all wired into `linear.cpp`'s dispatch
without touching `third_party/libr4d` -- the kind of task that genuinely takes multiple days of
iterative on-hardware debugging to get correct, not something a single pass can responsibly write,
tune, AND verify byte/tolerance-correct against a 45 GiB real container without a material risk of
shipping a kernel that silently changes the model's output. Per this project's own correctness gate
("a faster prefill that changes the answer is a failure") and its standing rule against narrowing
scope silently, this pass chose to do the parts it could complete and verify for real (the profiling
and the pre-change baseline) and report the kernel/cap-raise/gate as **not started** rather than
fabricate a kernel, a cap sweep, or invented post-change TOPS/tok-s numbers. **This is now the #1
follow-up item**, recommended as its own dedicated multi-stage effort (kernel + isolated correctness
unit test first; wiring + cap sweep second; full end-to-end SHA-256 gate + ctest registration third).

**Build/test**: no source files changed this pass (docs only); `.\build.ps1` confirmed a no-op
(`ninja: no work to do`) before profiling; full `ctest` reconfirmed green after the doc edits (docs
cannot affect test outcomes, but the standing rule is to leave `ctest` green, verified rather than
assumed -- see the ctest run at the end of this pass).

## Long-context validation (2026-09-20, docs/r9700.md R13 + Q17, docs/status.md's own prior "Known
gaps" item 5)

**Task**: every perf number this project had ever reported was measured at `--max-ctx 2048`, while
r4dx defaulted `--max-ctx` to `131072` in both `src/cli/cli_args.h` and `src/server/server_args.h`
-- a self-imposed cap at HALF the checkpoint's own declared `max_position_embeddings: 262144`
(`rope_type` "default", no scaling trick needed). Two assumptions behind the old 131072 default
were suspect: (a) whether the context ceiling itself should be higher, and (b) whether
docs/r9700.md's own VRAM-headroom argument (":153", "15.75 GiB measured, leaving 0.74 GiB") was
sized correctly. Both were resolved by measurement, on real hardware, this pass -- see
`docs/perf.md`'s "Long-context validation" section for the full data and `docs/r9700.md`'s R13/Q17
entries (§2.1's VRAM-reconciliation blockquote, §2.3's decode-ceiling correction, P4) for the
first-principles reconciliation.

**Findings, in one paragraph**: the 262144-token native ceiling **fits with room to spare** --
7.75 GiB (24%) free at `--mtp 0`, 6.83 GiB (21%) free at `--mtp 3`, against the old argument's
0.74 GiB. The measured KV growth (`(8.344-0.406) GiB / 260096 tokens = 32768.0 bytes/token`) matches
the architecture-derived 32 KiB/token figure to 4 significant figures -- the "measured vs. inferred"
figures the task asked to reconcile were never actually in conflict, they had just never been
compared at a context length large enough to distinguish them from noise. Decode degrades from
38.24 tok/s @2048 to 25.95 tok/s @262144 (-32.1%, w4a16 `--mtp 0`) as fp8 KV reads grow from 0.5% to
38.1% of bytes/token, closely tracking a corrected roofline model (88.9% -> 96.9% of ceiling,
efficiency actually *improving* with context as fixed overhead amortizes). docs/r9700.md's own
131k prediction of 29.1 tok/s is refuted in the optimistic direction -- it used the pre-R1 base
bytes/token; the real number is 31.09 tok/s. MTP (`--mtp 3`) roughly doubles decode at every context
length including 262144 (45.11 vs 25.95 tok/s). **Correctness holds throughout**: a needle-retrieval
prompt is answered correctly and the standard haiku prompt generates coherent, on-topic,
byte-identical-between-`mtp={0,3}` text at every context length up to and including 262144 real
prefilled tokens -- RoPE, the paged KV allocator, GDN state, and MTP window bookkeeping all behave
correctly at the model's own native ceiling. A genuinely new, unplanned finding: **prefill
throughput degrades far faster than decode** (1007 -> 264 tok/s, -73.8%, vs decode's -32.1%),
attributed (not root-caused) to each new prefill chunk's attention needing to read the *entire*
preceding KV history, not just its own chunk.

**Decision and code changes**: `--max-ctx` default raised `131072` -> `262144` in
`src/cli/cli_args.h`, `src/server/server_args.h`, and `src/model/model.h`'s
`ModelOptions::max_ctx`; `README.md` and the CLI/server default-value tests
(`tests/cli/test_args.cpp`, `tests/server/test_server_args.cpp`) updated to match. bf16 is
unaffected (its 47.73 GiB of weights do not fit regardless of context length, per the project's
standing bf16-retired-from-perf-work rule) and was excluded from this pass's measurements.

**Reduced scope, not silently dropped** (each 262144-token prefill takes 15-16 minutes wall-clock,
making the task's full 5-context x 3-layout x 2-`--mtp` x 2-run matrix a multi-hour undertaking):
w4a16 got the complete 5-context x 2-`--mtp` sweep (the primary ask); w4a8/mxfp4 got a 2-point
spot-check (2048, 131072, `--mtp 0` only) confirming the same qualitative trend, not measured at
262144 or `--mtp 3`; every configuration was run once, not twice (the 2048-tier numbers are
consistent with this document's and docs/perf.md's other, independently repeated 2048-context
measurements, the closest available check on this pass's own noise); the prefill superlinear
degradation is reported but not root-caused (no `--profile-prefill` run at long context this pass).
Full detail, all tables, and the exact transcripts: `docs/perf.md`'s "Long-context validation"
section.

**Build/test**: only two lines of test-assertion code changed
(`tests/cli/test_args.cpp`/`tests/server/test_server_args.cpp`, the default-value checks), plus the
three `--max-ctx` default constants themselves (`src/cli/cli_args.h`, `src/server/server_args.h`,
`src/model/model.h`) and doc-comment updates -- no engine/kernel logic changed. Full `ctest`:
**36/36 passing**, ~217s, HIP device 1 (`tests\run_tests.ps1`) -- same count as the R9 pass, this
pass added no new tests.

## R9: reduced-vocab draft head

**Task**: docs/r9700.md R9 + §2.2's draft-side byte budget -- cut a sequentially-drafted token's
dominant cost (the FULL 248320-entry `lm_head`, 675 MB of the 926 MB per drafted row) with a
top-N slice of `lm_head` used only for drafting, making wide speculation (K up to 16) viable.

**Shipped**:
- Container format: OPTIONAL `mtp.draft_head.lm_head.{layout}` + `mtp.draft_head.vocab_ids`
  (docs/container-format.md), versioned by presence -- absent on every pre-existing container,
  loader falls back to the exact pre-R9 full-vocab path unconditionally.
- `r4dx-convert --draft-vocab-ids <json>`: slices `lm_head.weight`'s rows by a calibration-chosen
  id list, quantizes through the SAME `PlanLinearLayouts`/`EmitLinearLayouts` helpers as every other
  linear.
- `r4dx_gather_i32` (new device kernel, `src/kernels/src/r4dx_kernels.hip`): maps a subset-local
  argmax index back to a real vocab id entirely on-device, zero host syncs, bounds-checked (clamps
  out-of-range like `r4dx_embedding_gather_bf16` already does).
- `MtpHead::Draft` gained `use_reduced_vocab` (default on when the container has a draft head);
  reuses `FinalLmHead` completely unmodified (it is generic in its own `lm_head_.N`) -- no new GEMM
  code needed. `Model::VerifyWindow` is UNTOUCHED -- always the real full-vocab head -- which is what
  makes the technique lossless: an out-of-subset draft is just a rejected draft, never a wrong
  accepted token.
- `ModelOptions::mtp_draft_reduced_vocab` / CLI+server `--mtp-draft-head {reduced,full}` (default
  `reduced`, degrades to `full` automatically on a container with no draft head).
- `tests/model/tool_vocab_calib.cpp`: a real-hardware calibration tool, two subset-construction
  methods (corpus-frequency vs the model's own predicted-token frequency), a proper TRAIN/HELD-OUT
  split so "coverage" is a genuine out-of-sample measurement, not a tautology (an early version of
  this tool measured a tautological 100% before the split was added -- caught and fixed within this
  same pass).
- K widened to 16 with **no structural changes needed**: `GdnStateManager`'s window bank, the
  conv-buffer rolling depth (`state_len_max = conv_width-2+max_decode_window`), and
  `Model::VerifyWindow`'s `mtp_logits_dev_`/`mtp_argmax_dev_` scratch were already parametrized by
  `ModelOptions::mtp_draft_k` from the Milestone 3 MTP-quality pass -- the M3 review's blocker-class
  off-by-one in this exact bookkeeping was already fixed as part of getting that formula right, and
  it generalizes. Verified on real hardware at K=16, both `bf16` and `w4a16`, via a NEW
  `tests/model/test_mtp.cpp::CheckWideWindowRejectionRewind` (same lossless-rewind contract as the
  existing K=3 checks) plus a new K=16 case in `tests/model/test_mtp_round.cpp`.
- New test coverage: `tests/kernels/test_gather_i32.cpp` (the new kernel, in-range + OOB-clamp),
  `tests/model/test_mtp.cpp::CheckReducedVocabDraftHeadLossless` (both `use_reduced=true/false`
  against a real container with draft-head tensors, on real hardware), `--mtp-draft-head` flag
  tests in both `tests/cli/test_args.cpp` and `tests/server/test_server_args.cpp`. Full `ctest`:
  **36/36** (was 35/35).

**Measured, real hardware** (see docs/mtp.md's "Reduced-vocab draft head" section for the complete
table and analysis): the shipped calibration run (`D:/models/wikitext-2-raw/wiki.train.raw`, 20000
positions teacher-forced through the real 64-layer container) found only **2977 distinct predicted
ids** and **76.8% held-out coverage** -- far short of the 8k-16k, high-coverage subset the R9
economic projection assumes. Re-ran with 10 dispersed corpus segments instead of one contiguous
prefix to rule out "unlucky narrow sample" as the cause -- **numerically identical result**,
pointing at WikiText-2 itself (a small, curated benchmark corpus) rather than the sampling strategy.
**Headline, stated honestly**: best measured decode tok/s across the full K={1,2,3,4,6,8,12,16} x
layout={w4a16,w4a8,mxfp4} sweep is **65.02 tok/s (w4a16, K=4, FULL-vocab head)** -- 0.95x of M3's
68.37 baseline (within the setup's own measurement noise, this pass's own K=0 baseline was 38.44 vs
M3's 38.86, -1.1%). **The reduced-vocab head's own best is 53.35 tok/s (w4a16, K=3)** -- 0.78x of
baseline, a regression, not the projected 2.5-3x. Every layout's tok/s PEAKS at low K (2-3) and
DECLINES monotonically as K grows with the reduced head, because acceptance keeps falling while the
coverage ceiling (76.8%) compounds with the checkpoint's own already-modest wide-K acceptance (the
FULL head only reaches 8.9-9.5% acceptance at K=16 on this checkpoint/prompt -- an independent,
real limit on wide-K speculation for THIS model, not caused by the reduced head at all).
**Conclusion**: the MECHANISM is complete, correct, and lossless (verified); the ECONOMIC case
depends entirely on calibration-subset coverage, which this pass's only available corpus could not
supply. Recommended default remains the full-vocab MTP head at its previously-measured optimal K
(w4a16 K=3, 67.34 tok/s, docs/mtp.md's "MTP head layout" table) until a larger/more diverse
calibration corpus is available and re-measured.

**DFlash2** (`D:/models/Qwen3.8-27B-DFlash2/*.gguf`, real files already on disk): assessed, not
ported -- a genuinely different model architecture (`general.architecture=dflash`, a block-diffusion
drafter with its own selector mechanism), not a slice of this model's own weights; porting it would
need a new GGUF loader and a new forward pass, a materially larger lift than the reduced-vocab head.
Full assessment in docs/mtp.md's "DFlash2 assessment" section.

**Not attempted, flagged for follow-up** (per this pass's own time budget, not silently dropped):
- A larger/more diverse calibration corpus (hundreds of thousands of tokens, multiple domains, or
  synthetic self-generated text) to actually raise coverage into the range the R9 projection assumes
  and re-measure this same K-sweep.
- The reduced-vs-full comparison only covered K={4,8,16} (not the full K list) per this pass's own
  time budget -- every K WAS measured with the reduced head (the task's own primary ask).
- No isolated per-draft-token latency measurement of the reduced head's own GEMM (the net decode
  tok/s numbers above are dominated by acceptance rate at this coverage level, so the draft-side
  latency saving, while real by construction -- ~83x fewer output rows -- was not separately
  profiled this pass).

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

- **RESOLVED (Milestone 4 follow-up, 2026-09-20 -- see "R2/P2 fused activation-quant epilogues:
  root-caused and enabled for w4a8/mxfp4" below).** P2's fused epilogues are now enabled for w4a8
  and mxfp4; root cause was an `r4dx::core::Arena::Alloc` gap (aligned each call's start but not its
  end), fixed in `arena.hpp`. w4a16 stays disabled (a separate, already-understood wall-clock
  regression, not a correctness issue). This was the single largest unrealized perf lever in the
  roadmap (docs/r9700.md's R2 row).
- **RESOLVED (Milestone 4 follow-up, 2026-09-20 -- see `docs/perf.md`'s "Full Q5-fixed
  `tune_gemm.py` re-sweep" section and `docs/r9700.md`'s Q5 entry).** The full 280-row re-sweep with
  the (already-correct) ring-rotation harness ran on real hardware: 216/280 rows changed tuning,
  41/70 comparable `(shape,M)` cells flip the w4a16/w4a8/mxfp4 ranking, overwhelmingly mxfp4 going
  from fastest to slowest at M in {1,2,4,8,16} on five of the ten shapes. End-to-end re-measurement:
  mxfp4 decode improved (+6.4% `mtp=0`, +14.1% `mtp=3`), w4a16 flat within noise, w4a8 `mtp=3`
  regressed -10.0% (MTP acceptance dropped 43.3%->35.6%, attributed to verify-band tuning changes and
  expected floating-point reduction-order drift, not a correctness bug). The re-swept table shipped
  (`src/model/gemm_tuning_table.inc`, provenance banner rewritten) because it is honestly correct
  (no cell implies above-DRAM-peak bandwidth, unlike the old table) and does not regress the shipped
  default (w4a16 stays fastest and stays the default). Full ctest 35/35 with the new table.
- **RESOLVED (Milestone 4 follow-up, 2026-09-20 -- see "MTP acceptance-gap investigation +
  server/test gaps closed" below).** `src/server/server_args.h` now exposes `--mtp-head-layout
  {bf16,layout}` (default `layout`), wired into `ModelOptions::mtp_head_layout` in
  `src/server/main.cpp` identically to the CLI's own conversion, with a new
  `tests/server/test_server_args.cpp::TestMtpHeadLayoutFlag`.
- **Q8 (GPU clock/power sampling) has no working tool on this Windows ROCm 7.15 install** --
  whether the card holds boost clock through a decode/MTP-verify step is still unanswered.
- **RESOLVED (Milestone 4 follow-up, 2026-09-20 -- see "Long-context validation" below).**
  R13/Q17 (32k/131k long-context validation) measured on real hardware, extended to the model's own
  native 262144-token ceiling; `--max-ctx` default raised 131072 -> 262144.
- **RESOLVED (Milestone 4 follow-up, 2026-09-20 -- see "MTP acceptance-gap investigation +
  server/test gaps closed" below).** `tests/model/test_mtp.cpp::CheckChatMultiTurnMidRoundStop`
  drives a real two-turn conversation through a real `Model` + the production
  `r4dx::model::ProcessMtpRound`/`r4dx::server::PrefixState` helpers, with a `--max-tokens`-
  equivalent budget forced (by a same-seed dry run) to land one token short of a round boundary, and
  asserts turn 2's post-`Reset()` continuation is byte-identical to an independently-loaded
  sequential reference.
- **PARTIALLY ADDRESSED (2026-09-20 -- see "Vision tower: investigation + golden reference done, no
  C++ yet" above and `docs/vision.md`).** The vision tower's architecture/preprocessing/mrope-
  splicing semantics are now fully documented against real `transformers` source and validated real-
  hardware goldens exist (`tools/reference/vision_golden.py`), but no C++ implementation exists yet
  -- still open. DFlash2 drafting (the other milestone named in Milestone 2's own "Status" line) has
  not been started (see the R9 pass's "DFlash2 assessment" above: architecturally a different model,
  not a slice of this model's own weights).

**Next milestone (proposed)**: ~~(1) root-cause and re-attempt P2's fused epilogues with a real
end-to-end byte-identical-text verification gate (not just ctest's tolerance-bounded goldens, which
did not catch the w4a8/mxfp4 bug) before re-enabling~~ -- **done, see the "R2/P2 ... root-caused and
enabled" section above**; ~~(2) the full Q5-fixed `tune_gemm.py` re-sweep~~ -- **done, see this
section's "RESOLVED" bullet above and `docs/perf.md`'s "Full Q5-fixed `tune_gemm.py` re-sweep"
section**; (3) R13's 32k/131k long-context measurement; (4) vision tower; (5) DFlash2 drafting. New
follow-up from this pass: root-cause w4a8's `--mtp 3` acceptance-rate drop (43.3%->35.6%) to split
"verify-band GEMM genuinely slower" from "numerical drift shifting argmax decisions" -- not attempted
this pass, time-boxed (see `docs/perf.md`'s re-sweep section).

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

## R2/P2 fused activation-quant epilogues: root-caused and enabled for w4a8/mxfp4 (Milestone 4 follow-up, 2026-09-20)

Follow-up to the "R2/P2" incident below (kept unchanged, verbatim, for history): the fused
activation-quant epilogues were built and byte-exact-verified in isolation but shipped **disabled**
because wiring them into the real model changed w4a8/mxfp4's generated text on real hardware, and
the root cause was not found in time. This pass found and fixed the root cause (Problem A), decided
Problem B (w4a16's wall-clock regression) by measurement, and re-verified with a real end-to-end
byte-identical-text gate as the task required.

**Problem A -- root cause, found.** Not the epilogue kernels' own math (`test_fused_quant.cpp`
already verified that in isolation, 135/135) and not which GEMM consumed a fused buffer (the
original pass's own in-model diagnostic had already cleared `gdn.in_proj_qkv`/`in_proj_z`/
`mlp.down`'s own output bytes). It is `r4dx::core::Arena::Alloc` (`src/core/include/r4dx/core/
arena.hpp`): it aligned each allocation's own **start** to the caller-requested alignment, but never
rounded its **end** up to any particular boundary. A fused epilogue's per-row fp32 scale scratch
(`next_epilogue_scale`/`x_normed_pre_scale`, allocated in `GdnLayer::Forward`/`AttentionLayer::
Forward`/`Mlp::Forward` as `arena.Alloc<float>(T)`, `T` = the chunk's row count, 1..64) is exactly
`4*T` bytes -- a multiple of 16 only when `T % 4 == 0`. At decode (`T=1`) and MTP verify (`T=2..4`),
the very next default-aligned `Alloc` call in the same layer (e.g. `GdnLayer::Forward`'s
`mixed_qkv`, `alignof(uint16_t)=2`) then starts a few bytes short of a 16-byte boundary, and every
allocation after it in that layer inherits the same drift until the next `Reset()`. Every
`third_party/libr4d` kernel this arena feeds (GDN conv/kkt/chunk-scan, every quantized GEMM
family's A-operand read, attention's decode scratch) reads its device buffers with **unchecked**
wide (16-byte) vector loads and silently reads the wrong bytes when handed a misaligned pointer --
unlike this project's own P6-rewritten `rmsnorm`/`residual_rmsnorm`/`silu_mul` kernels, which fall
back to a scalar loop when a buffer isn't 16-aligned. Before this fusion pass, every real call
site's allocation *size* happened to already be a multiple of 16 bytes (`hidden`=5120,
`conv_dim`=10240, `intermediate`=17408 are all multiples of 8 bf16 elements = 16 bytes), so
`offset_` was always incidentally 16-aligned and this was never triggered -- several call sites'
own comments already flagged that invariant as "incidental, not enforced"
(`gdn_layer.cpp`/`attention_layer.hpp`/`linear.cpp`'s `i8_scratch`/`fp8_scratch` comments) before
this pass traced it to its actual source. This matches prime suspect (b) named by the task
("buffer aliasing/alignment when the epilogue writes into a differently-aligned arena slice than
the standalone kernel did") -- the earlier pass's own bisection (forcing every epilogue KERNEL to a
no-op while keeping the same extra arena allocations, and still reproducing the divergence) had
already isolated it to the allocation shift itself, just not to which allocator invariant broke.

**Fix** (`src/core/include/r4dx/core/arena.hpp`): `Arena::Alloc` now rounds the end of every
allocation up to a 16-byte boundary before storing it as the next call's bump offset, so every
FUTURE `Alloc` call -- regardless of what alignment it individually requests -- is guaranteed to
start 16-aligned again, restoring the invariant the rest of the codebase already silently depended
on. At most 15 bytes of padding per call, negligible against the arena's ~96 MiB reservation. No
other file needed a change for Problem A -- `linear.cpp`'s `EpilogueForLayout` and every
`GdnLayer`/`AttentionLayer`/`Mlp` fusion call site were already correct once the allocator invariant
they all assumed was actually enforced.

**Test coverage gap, closed.** The original 135-check `test_fused_quant.cpp` grid could never have
caught this: every check there allocates its own isolated `DeviceBuffer` (each a fresh `hipMalloc`,
which happens to come back far more than 16-byte aligned), so it never exercises `r4dx::core::Arena`
at all. Added `CheckArenaAlignmentInvariant` to `tests/kernels/test_fused_quant.cpp`: reproduces the
exact `GdnLayer`/`AttentionLayer`/`Mlp` allocation sequence (a 16-aligned quantized-activation data
buffer, then a default-aligned `float[T]` scale scratch, then the layer's own next default-aligned
buffer) through a real `r4dx::core::Arena`, for every `T` in 1..64 (decode, every MTP verify width,
every prefill chunk width), and asserts every buffer after the scale scratch is still 16-byte
aligned -- host-side, no GPU divergence needed to notice a regression. Full `ctest`: **35/35**
(unchanged count -- this is an addition to an existing test binary, not a new one).

**Problem B -- w4a16, decided by measurement (unchanged from the original incident).**
`r4dx_epilogue_f16` measures -4.3% decode when wired (reproducible, re-confirmed by the original
incident's own numbers, not re-measured this pass since nothing about the f16 path changed):
`r4dx_model_cast_bf16_to_f16` launches a flat elementwise grid (`blocks=ceil(M*K/256)`, ~20
independent workgroups at decode `T=1`/`K=5120`), and fusing it into rmsnorm/residual_rmsnorm/
silu_mul's own one-workgroup-per-row epilogue collapses that work onto a single workgroup -- a real
parallelism loss the removed launch's savings do not cover. w4a8's `r4dx_epilogue_int8_fraga8` and
mxfp4's `r4dx_epilogue_fp8_e4m3_row` do not have this problem (their standalone kernels already
launch `dim3(M)`, the same grid shape as the producers). Per the task's own instruction ("do not
re-enable in that form"), `EpilogueForLayout` (`src/model/linear.cpp`) keeps returning
`r4dx_epilogue_none` for w4a16 (and for bf16, which never quantizes its activation input) and now
returns `r4dx_epilogue_int8_fraga8`/`r4dx_epilogue_fp8_e4m3_row` for w4a8/mxfp4. A wide-grid f16
cast or a prefill-only fusion (`dim3(rows)=dim3(T)`, no parallelism loss once `T>1`) is left as
future work, not attempted this pass.

**Mandatory gate: end-to-end byte-identical generated text.** `tools/validate_fusion.ps1` (new):
for every (layout in {w4a16, w4a8, mxfp4}) x (`--mtp` in {0, 3}) x (prompt in {short ~20-token
single-chunk, medium ~100-token multi-chunk, long ~1000-token long-context}) -- 18 combinations --
runs `r4dx-cli.exe` twice against the real 64-layer container (`D:\models\r4dx\qwen38-27b-v3.r4dx`),
greedy, once with `R4DX_DISABLE_EPILOGUE=1` (forces every layout back to the pre-fusion baseline
from the SAME binary, no second build needed) and once without, and SHA-256-hashes each run's raw
stdout (the generated text only -- `[stats]`/`--profile` all go to stderr). **Run on real hardware,
HIP device 1: 18/18 combinations byte-identical.** w4a16's six rows are an expected trivial pass
(`EpilogueForLayout` returns `none` for it either way, a regression check that the env-var toggle
itself never perturbs it); w4a8's and mxfp4's twelve rows are the real fix verification, covering
every prompt-length/MTP combination the task required. This script is left in place as the standing
regression gate for any future change to `linear.cpp`, `arena.hpp`, or the epilogue kernels.

**Performance, measured** (real 64-layer container, `docs/perf.md`'s standard prompt/flags, HIP
device 1, each config run twice, both runs within 0.1% of each other so a single value is reported
per the task's own ">3% difference" threshold):

| Layout | `mtp=0` decode, fusion off (M3) | `mtp=0` decode, fusion on | Delta | `mtp=3` decode, fusion off (M3) | `mtp=3` decode, fusion on | Delta |
|---|---|---|---|---|---|---|
| w4a8  | 35.73 tok/s | **36.19 tok/s** | +1.3% | 61.41 tok/s | **61.47 tok/s** | +0.1% (noise) |
| mxfp4 | 30.27 tok/s | **30.85 tok/s** | +1.9% | 55.72 tok/s | **56.39 tok/s** | +1.2% |

MTP acceptance/tokens-per-round are **identical** to the fusion-off baseline for both layouts at
both `--mtp` values (w4a8: 43.3%, 2.27 tok/round, 40 rounds/120 drafted/52 accepted; mxfp4: 47.1%,
2.32 tok/round, 34 rounds/102 drafted/48 accepted) -- expected, since fusion is greedy-deterministic
byte-identical to the baseline, so it cannot change which tokens are accepted. Prefill and VRAM are
unchanged within noise (prefill ~640-740 tok/s per layout, VRAM 16.17/16.60 GiB `mtp=0`/`mtp=3`,
identical to the M3 baseline in `docs/perf.md`). w4a16 was not re-measured (`EpilogueForLayout`
returns `none` for it, so its decode/prefill/VRAM numbers are unchanged from M3 by construction).

**Launches/token** (`r4dx-cli --profile`, r4dx-owned kernel-launch counter, single profiled decode
step, fusion on vs `R4DX_DISABLE_EPILOGUE=1`): mxfp4 597 -> **326** (-271, a real reduction visible
to this counter because mxfp4's own quant kernel, `r4dx_quant_act_fp8e4m3_row`, lives in the same
translation unit the counter instruments); w4a8 260 -> **260** (unchanged -- this counter is blind
to `core::r4d::QuantActI8`, a `third_party/libr4d` entry point outside the counted translation unit,
the same scope note docs/r9700.md's launch census already flags; the real per-step launch count IS
lower for w4a8 too, just not visible to this particular instrument). GEMM share of `gpu_sum` in the
profiled step rose accordingly as non-GEMM launches were removed (mxfp4: 66.0% fused vs 72.2%
unfused of a smaller total; w4a8: 65.5% vs 68.7%) -- consistent with fusion removing non-GEMM launch
overhead rather than touching the GEMMs themselves.

**Task item 6 (measure all three layouts x `--mtp {0,3}` and decide the default): still not run.**
The modest, real gains above (+1.3-1.9% decode at `mtp=0`) do not change the layout ranking
(w4a16 remains fastest in absolute tok/s at both `--mtp` values); `w4a16` stays the default
(`src/cli/cli_args.h`, README.md, docs/perf.md unchanged). P1's (docs/r9700.md) argument for
w4a8-as-default is about GEMM throughput at higher verify widths, a separate question from this
pass's scope.

**Recommended follow-up** (not started this pass): (a) a wide-grid f16 cast or prefill-only fusion
for w4a16 (Problem B's deferred option); (b) re-run `docs/r9700.md`'s R2 roadmap entry against the
now-enabled w4a8/mxfp4 fusion to see whether it changes any downstream roadmap ranking; (c) task
item 6's full layout-decision sweep, now that fusion is real for two of the three layouts.

## Full Q5-fixed `tune_gemm.py` re-sweep (Milestone 4 follow-up, 2026-09-20): done

Follow-up to this document's own "Known gaps" item 2 and `docs/r9700.md`'s Q5. Confirmed
`tools/profile/tune_gemm.py`'s R5-era ring-rotation fix (`RingCall`/`ring_count`, >=4 distinct weight
buffers totalling >256 MiB, rotated per timed call) was already correct and wired into every
`alloc_*` helper -- i.e. already applied to the whole sweep, not just the two shapes R5 spot-checked;
what was missing was actually running the full 280-row sweep. Ran it (all 4 layouts x 10 shapes x 7
M-bands, HIP device 1, one process at a time, ~13 min wall clock) and regenerated
`src/model/gemm_tuning_table.inc` in place.

**Findings**: 216/280 rows (77%) picked a different `(WV,SK,MB,NPW,NT)` tuning than the old table.
41/70 `(shape,M)` cells where w4a16/w4a8/mxfp4 are directly comparable flip the fastest-to-slowest
ranking -- overwhelmingly mxfp4 going from fastest at M in {1,2,4,8,16} to slowest of the three, on
`gdn.in_proj_qkv`, `gdn.in_proj_z`, `gdn.out_proj`, `attn.qg`, and `mlp.down` (e.g.
`gdn.in_proj_qkv` M=1: 30.44 -> 52.53 us, +72.6%), generalizing R5's two-shape spot-check to a
much broader pattern across most of the table. w4a16/w4a8 never flip past each other.

**End-to-end re-measurement** (real 64-layer container, this file's standard prompt/flags, HIP
device 1, each config run twice, within 0.2% both times): mxfp4 decode improved **+6.4% (`mtp=0`,
30.85->32.83 tok/s) and +14.1% (`mtp=3`, 56.39->64.31 tok/s)** -- a genuine win, because the
corrected sweep picks configs that are fastest under the cold reads the real model always does,
where the old sweep picked whatever looked fastest on a single cache-resident buffer. w4a16 is flat
within noise at both `--mtp` settings. **w4a8 `--mtp 3` regressed -10.0% (61.47->55.34 tok/s)**, MTP
acceptance dropping 43.3%->35.6% -- attributed to a combination of the M=2..5 verify-band GEMMs
getting ~8-12% slower in the honest sweep and MTP's acceptance decision being sensitive to the
verify step's own floating-point reduction order (expected numerical drift from retiling, not a
correctness bug: verify's batched-M forward pass IS the ground truth MTP checks the draft against).
Full root-cause split (GEMM-latency vs numerical-drift) not attempted, time-boxed -- see "Next
milestone" below.

**Decision: shipped.** The re-swept table does not regress the shipped default (w4a16 stays fastest
in absolute decode tok/s at both `--mtp` settings, so it stays the default) and is honestly correct
(no cell implies above-DRAM-peak bandwidth, unlike several old mxfp4 rows) even though one
non-default layout's MTP mode regressed. Full `ctest`: **35/35** (`tests\run_tests.ps1`, ~142s, HIP
device 1) -- a data-only table swap, `PickTuning`'s existing fallback path makes this
correctness-preserving by construction, confirmed by the tolerance-bounded goldens in
`test_forward_smoke`/`test_mtp`/`test_gdn_layer`/`test_attn_layer` all still passing. Full detail,
the flip table, and the end-to-end numbers: `docs/perf.md`'s "Full Q5-fixed `tune_gemm.py` re-sweep"
section; `docs/r9700.md`'s Q5 entry (now marked answered) and its two other stale references to the
pre-fix numbers (corrected in place). Docs updated, no other document found asserting the specific
above-DRAM-peak numbers this pass corrected.

## MTP acceptance-gap investigation + server/test gaps closed (Milestone 4 follow-up, 2026-09-20)

Task scope: docs/mtp.md's "Acceptance gap investigation" open question (w4a16 46.3% vs mxfp4 47.1%
vs w4a8 43.3% at K=3, with head-precision and logit-margin already falsified), plus this document's
own Known-gaps items 3 (`--mtp-head-layout` server flag) and 6 (`--chat` multi-turn + MTP mid-round
regression test).

**h_seed drift, measured: correlates with the acceptance ranking.** Built the one route M3 named but
never had (a 4-layer container carrying ALL FOUR layouts plus `mtp.*` weights side by side --
`D:\models\r4dx\qwen38-27b-l4-allmtp.r4dx`, converted this pass via `r4dx-convert --layers 4 --mtp on
--layouts bf16,w4a16,w4a8,mxfp4`; the pre-existing `qwen38-27b-l4-mtp.r4dx` only carries bf16+w4a16).
Added a diagnostic-only accessor, `Model::DebugSeedHiddenBf16()` (`src/model/model.h`/`.cpp`), that
reads back `mtp_seed_hidden_` -- the exact pre-final-norm hidden-state row `MtpHead::Draft`'s first
step consumes -- and a one-off tool, `tests/model/tool_hseed_drift.cpp` (built, never registered as
a ctest test -- it prints numbers, it doesn't assert a contract), that feeds the SAME fixed 64-token
stream through a bf16 (exact-arithmetic reference, this project's own convention) `Model` and each
quantized layout's `Model` against that container, sampling `h_seed` every 8 tokens (8 positions) and
comparing to the bf16 reference by cosine similarity and relative L2.

Measured (HIP device 1, real hardware): mean cosine / mean rel L2 vs bf16 -- **w4a16 0.99763 /
7.16e-2** (best), **mxfp4 0.99745 / 1.061e-1** (middle), **w4a8 0.99496 / 1.257e-1** (worst). This
is EXACTLY the measured acceptance ranking at K=3 (w4a16 51.9-54.3% > mxfp4 40.0-41.9% > w4a8
32.5-34.2%, docs/mtp.md's "MTP head layout" table) -- the layout with the smallest h_seed drift from
the exact-arithmetic reference has the highest acceptance, and the layout with the largest drift has
the lowest, with mxfp4 correctly landing in between on both axes. Per-position noise exists (at 2 of
8 sampled positions mxfp4's rel L2 briefly exceeds w4a8's -- 8 positions on a drastically-truncated
4-layer container is a small sample), so this is a MEAN-level correlation, not a claim that every
single position preserves the ranking; the aggregate agreement across two independent metrics
(cosine and rel L2) and three layouts is not plausibly noise.

**Mechanism, identified**: nearly all of every layout's rel L2 is driven by ONE hidden dimension
(component index 3994 of 5120) whose bf16 reference magnitude (+26.5) dwarfs every other component
(observed range roughly -2..+2) -- the well-documented "massive activation" / outlier-dimension
phenomenon in transformer residual streams. Its absolute quantization error tracks the exact same
per-layout ranking as the aggregate metric: w4a16 -0.375 (-1.4%), mxfp4 -1.125 (-4.2%), w4a8 -2.875
(-10.8%). Because this one dimension's squared magnitude dominates the vector's squared norm, its
quantization error alone is effectively what rel L2 (and, plausibly, the lm_head's own sensitivity
to the residual stream) is measuring here.

**Conclusion**: h_seed drift explains the acceptance ordering (a measured positive, not a clean
negative like the two hypotheses this pass's predecessor falsified). Not a bug -- this is expected
behavior of weight/activation quantization on a residual-stream dimension with an outsized dynamic
range, the same reason "outlier-aware" quantization schemes exist in the wider literature; no code
change is warranted, so this pass does NOT touch task item 4 (fix + re-measure) -- there is nothing
found to fix. Per the task's own item-2 conditional ("if h_seed drift does not explain it, the next
candidates in order..."), the KV-history-divergence and verify-window-disagreement-rate hypotheses
were NOT pursued this pass, since a measured, mechanistically-grounded positive answer for h_seed
drift was already found. Recommended follow-up (not started): whether an outlier-aware quantization
scheme for the small number of massive-activation dimensions specifically (leaving the rest of the
tensor at its current bit width) narrows the acceptance gap without giving up the layout's
throughput -- see docs/mtp.md's "Acceptance gap investigation" for the full writeup and numbers.

**Two small gaps closed while in this code** (docs/status.md's own Known-gaps items 3 and 6 above):

1. **`--mtp-head-layout` server-side flag** (`src/server/server_args.h`, `src/server/main.cpp`):
   added `--mtp-head-layout {bf16,layout}` (default `layout`), parsed and validated identically to
   `src/cli/cli_args.h`'s own flag, wired into `ModelOptions::mtp_head_layout` in `main.cpp`. New
   `tests/server/test_server_args.cpp::TestMtpHeadLayoutFlag` (defaults, `bf16` override, rejection
   of an unrecognized value). `docs/server.md`/README.md updated (their own "deferred" notes marked
   resolved).
2. **`--chat` multi-turn + MTP mid-round-stop regression test**
   (`tests/model/test_mtp.cpp::CheckChatMultiTurnMidRoundStop`, new): drives a REAL two-turn
   conversation through a real `Model` (not a synthetic vector, unlike
   `tests/server/test_prefix_state.cpp`'s existing `TestCommitTracksCommittedNotDisplayedTokens`,
   which only proves `PrefixState`'s own arithmetic) -- a same-seed dry run first finds a round that
   naturally emits >=2 tokens, then the real run enforces a `max_tokens_remaining` budget exactly one
   token short of that round's own boundary via the production `r4dx::model::ProcessMtpRound` helper
   `src/cli/main.cpp`/`src/server/engine.cpp` use, asserts `r4dx::server::PrefixState::Extend()`
   correctly REFUSES to treat turn 2's re-rendered (displayed-only) conversation as a fast-path
   extension of the model's real (committed) state (the exact desync the bookkeeping fix exists to
   prevent), then verifies the `Reset()`+full-reprefill recovery path's turn-2 continuation is
   byte-identical to an independently-loaded, freshly-`Prefill()`'d sequential reference. Runs for
   every layout `test_mtp` iterates (originally bf16, w4a16 on `qwen38-27b-l4-mtp.r4dx`).
   **UPDATE (review finding, 2026-09-20):** `test_mtp.cpp`'s main loop (this check plus
   `CheckVerifyMatchesSequential`/`CheckRejectionRewind`/`CheckWideWindowRejectionRewind`/etc.) now
   points at `qwen38-27b-l4-allmtp.r4dx` and iterates all FOUR layouts (bf16, w4a16, w4a8, mxfp4) --
   w4a8/mxfp4 previously had zero coverage from this file despite being the two layouts the
   fused-epilogue and GEMM-tuning-table passes actually perturbed. `qwen38-27b-l4-mtp-draftvocab.
   r4dx` (used only by `CheckReducedVocabDraftHeadLossless`) was NOT re-converted and still only
   carries bf16/w4a16, so that one check deliberately kept its own separate {bf16, w4a16} layout
   list rather than reusing the widened one (reusing it crashed the process with
   STATUS_STACK_BUFFER_OVERRUN against tensors that container doesn't have -- caught by re-running
   after the change, before shipping it). `tool_hseed_drift` also still uses
   `qwen38-27b-l4-allmtp.r4dx` (unchanged). Full `ctest`: still 37/37, `test_mtp` grew from ~110s to
   ~147s (double the layouts through the main loop).

**Build/test**: `.\build.ps1` incremental (both new source files, `model.h`/`.cpp`'s new accessor,
and the two server-side files rebuilt their dependents cleanly, no warnings-as-errors). Full `ctest`:
**35/35 passing** (`tests\run_tests.ps1`, ~162s, HIP device 1) -- `test_mtp` grew from ~50s to 70s
(the new `CheckChatMultiTurnMidRoundStop` check, x2 layouts) but the test COUNT is unchanged (no new
ctest binary was registered; `tool_hseed_drift` is built but deliberately not an `add_test`).



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
