# r4dx end-to-end performance and correctness (assembly + CLI milestone)

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
