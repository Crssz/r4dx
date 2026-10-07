# Pipeline-parallel (layer-split) 2-GPU prefill: Phase 0 tooling and Phase 1 (stage-capable RunChunk, PP-emulate)

Status (2026-10-08, branch `pp`, from `main` 4e710ab): **Phase 0 tooling and Phase 1 are written and build; no GPU
run has been made of any of it.** Phase 2 (the real two-GPU pipeline) is NOT started: it waits for the Phase 0
numbers. The design this implements is the Phase 0 / Phase 1 part of the pipeline-parallel design (research ideas
#26 / #33): 2 stages of 256-row super-chunks, stage A = layers [0, k) + embedding on device 0 (the desktop card),
stage B = layers [k, N) + final norm + lm_head + MTP priming + DFlash injection on device 1 (the headless card,
decode's), the boundary through pinned host memory (no peer copies: docs/tp.md 1.2), a 3-slot ring, k = 33 by default
(32..35 keep 8 attention layers per stage), PP prefill + TP=1 decode, byte-identical to the monolithic prefill.

(The full design document is not in the repo; what this branch relies on is restated here, and "the design" below means it.)
This file records what exists, how to run it, the pass rules, and what is not done. The design's numbers (TTFT
1.80x / 1.85x / 1.86x at 8k / 32k / 64k expected, 1.63x / 1.68x / 1.70x pessimistic) are DERIVED, not measured:
Phase 0 below is what measures them.

## 1. RunChunk is three phases (Phase 1, pure code motion)

`Model::RunChunk` (src/model/model.cpp) used to be one 590-line function. It is now

| phase | function | what it does |
|---|---|---|
| prologue | `ChunkPrologue(ChunkRun&)` | the id / meta upload, embedding gather, image splice, stack-entry rotation, the per-chunk device arrays (positions, seqused_k, mrope rows), the thread-local scopes the layers run under (M = 256 trellis GEMM, int8 GEMM, fake-quant switches), the carry's initial state, the TP submission-bounding setup |
| layer range | `RunLayerRange(ChunkRun&, first, last)` | the layer loop over layers [first, last): DFlash target-feature capture at the top of each layer, GDN or attention, then the Mlp with the R3 / R2 fusion; `cur` / `normed_in` / `normed_in_epilogue` go in and come out |
| epilogue | `ChunkEpilogue(ChunkRun&)` | the stack-exit rotation, MTP KV priming, final norm + lm_head, the stream synchronize and logits readback, the DFlash injection, `pos_ += T` |

`ChunkRun` (src/model/chunk_run.h) holds what used to be RunChunk's locals between the phases: the call's arguments, the
probe scope, `hidden` / `num_layers` / `has_init` / `dflash_capture_active`, the per-chunk device pointers, the four
thread-local scopes (as `std::optional`s, opened where the prologue ends, closed with the call), and the carry
(`cur`, `other`, `normed_in`, `normed_in_epilogue`). The monolithic `RunChunk` is
`ChunkPrologue; RunLayerRange(0, N); ChunkEpilogue`.

**It is a code motion, checked mechanically**: of the 570 statements of the old body, 556 appear verbatim and in the
same order in the new three functions; the other 14 are the local -> `ChunkRun` bindings (the probe scope, the four
scopes, `cur` / `other` / `normed_in` / `normed_in_epilogue`, `had`, the loop header, the two `End()` calls).
Everything the new code adds is PP-emulate and its timing, behind `ChunkRun` flags that are false on every other
path: one `if (r.poison_arena_layers)` per layer, and `if (r.timing)` after the MTP priming and around the
DFlash injection. Decode runs the same statements in the same order, so its bytes and (to within a handful of
struct stores) its host cost are unchanged. No device code changed: the libr4d ISA-hash and epilogue-diff gates of the
build pass untouched and nothing was re-baselined.

There is no env kill switch for the refactor itself (nothing behaves differently); the one behaviour it enables,
PP-emulate, is off unless asked for (section 2).

## 2. PP-emulate: two stages on one device, the carry through host memory (Phase 1)

`Model::SetPpEmulate(PpEmulateConfig)`, `ModelOptions::pp_emulate_split`, or `R4DX_PP_EMULATE=<k>` in the environment
(unset, empty, `0`, `off` = off, the default and the kill switch; a positive integer = the split layer k; anything
else is refused at `Model::Load`; `pp_emulate_split = 0` forces it off whatever the environment says).
With a split k every PROMPT-PREFILL chunk (a 256-row super-chunk or a 64-row chunk; image chunks too; never a
decode step or a verify window) runs as

```
ChunkPrologue            (stage A: ids, embedding gather, splice, scopes)
RunLayerRange(0, k)      (stage A)            -- synchronize
hop: export  D2H of the carry + DFlash columns + KV rows  ->  pinned host staging (hipHostMallocPortable)
     poison  0xFF into the device-side destinations (cur, other, buf_normed_, buf_normed_pre_, the columns, the KV rows)
     import  H2D back, cur ALWAYS into buf_a_ (as stage B on the other device will)
RunLayerRange(k, N)      (stage B)
ChunkEpilogue            (stage B: MTP priming, lm_head, DFlash injection, pos_ += T)
```

What crosses (docs/pp-prefill design 1.3, 3.1): the residual stream `cur` and the fused-norm pair (`buf_normed_`,
`buf_normed_pre_`), three T x 5120 bf16-sized buffers (7.5 MiB at 256 rows); the DFlash feature columns of the target
layers < k (a prefix of the capture's columns: the target layers are ascending), moved with `hipMemcpy2DAsync`; and
the KV blocks of stage A's attention layers that this chunk wrote (`pp::KvBlocksTouched`: whole 16-row blocks, one
contiguous byte range per layer, so 4.25 MiB for 8 layers at a block-aligned start and one block more mid-block).
The GDN state needs no hop here (one Model); its compact export / import with the conv-line pitch conversion is the
real pipeline's job, once per Prefill call, and is Phase 2.

Refused (`SetPpEmulate` throws): a tensor-parallel rank, a rotated (quant2) container (the stack-entry and -exit
rotations straddle the stages), a split outside [1, N - 1].

Test and measurement controls, all in `PpEmulateConfig`: `poison_arena` (1: the whole activation arena filled with 0xFF
at the stage boundary; 2: also after every layer's scratch release -- a kernel that reads scratch it did not write
would change a byte), `fault` (negative controls: `kShiftCurOneByte`, `kSkipNormed`, `kSkipKv`, `kSkipDflash`), and
`timing` (record `PpChunkTimes` per chunk; adds a stream synchronize after stage B's layers and after the MTP
priming, skips the poison fill so the hop is timed as a copy; the bytes are unchanged).

### The gate, G1b: `tests/model/test_pp_emulate_identity.cpp` (GPU, device 1)

One Model per configuration; each scenario runs monolithic, then two-stage at each variant's split; every observable is
compared bit for bit: the last row's logits of each Prefill call, the greedy tokens after it, the digest of ALL
per-sequence state (`Model::DebugStateDigest`; `DebugZeroKvState`, a new `R4DX_TP_TESTING` hook, zeroes the KV caches at
every scenario start so a stale page cannot depend on what an earlier scenario left -- the two runs share a Model),
the speculative rounds' tokens, and the DFlash capture drained through the per-chunk callback. It also checks that
exactly the grid's number of chunks went through the two-stage path (`PpEmulatedChunksRun`).

| part | splits | what it proves |
|---|---|---|
| 4-layer container (bf16, w4a16), plain | 1, 2, 3 of 4; arena poison at the boundary and after every layer | the carry, with attention in stage B only |
| 4-layer, target capture of layers 0..3 | 1, 2, 3 | the DFlash columns split between the stages |
| 4-layer, `--mtp 3` | 2 | priming on stage B |
| real 64-layer trellis container, plain | 33, 5, arena poison 1 and 2 | KV of 8 attention layers through the host; odd ping-pong parity (cur leaves stage A in buf_b_); the production int8 + split-KV paths; `len8145` |
| real, `--mtp 3` / `--dflash` (k = 7) | 33, 5 / 35, 21, 5 | MTP priming and DFlash injection after the hop; columns of 3 / 2 / 0 target layers in stage A |
| real, image rows (plain and `--dflash`) | 33 / 35 | the splice in stage A's prologue, the 3-axis rope read by both stages |
| NEGATIVE CONTROLS | 4-layer: cur one byte off, norm dropped, DFlash columns dropped; real: cur one byte off, norm dropped, KV dropped, DFlash dropped | each must change an observable, or the comparison proves nothing |

The shape list is `test_prefill_chunk_identity`'s: 1, 63, 64, 65, 255, 256, 257, 511, 8145 rows and the prefix-reuse
shapes 300 + 333, 64 + 511, 257 + 1, 1 + 255 + 257 (real container: 63 .. 1100 plus 8145 at k = 33). SKIPs (77) without the
containers. `R4DX_TEST_ONLY=<substring>` runs matching configurations only.

`CPU` tests (always run): `test_pp_plan_cpu` (src/model/pp_plan.h: the KV block ranges incl. the chunk-grid coverage
property, the `R4DX_PP_EMULATE` parser, the pipeline simulation against hand-computed cases and the design's closed form,
the CSV formats, the projection and the go rule).

## 3. Phase 0: measure before building (tooling)

The design's Phase 0 table, as tools. All built under `tests/model` (never `add_test()`ed: they print / write numbers):

| tool | measures |
|---|---|
| `tool_pp_stage_bench` | one card, the full model: the monolithic prefill TTFT and the PP-emulate run -- per chunk the prologue + layers [0, k) (stage A), the hop's D2H and H2D halves, layers [k, N) (stage B), the epilogue, and in it the MTP priming and the DFlash injection (`--mtp K`, `--dflash`); sizes 8k = 8145, 32k = 32623, 64k = 65529 tokens (the design's prompts) or `n<N>`; repeats keep the median; `--barrier <prefix>` starts two processes together |
| `tool_pp_hop_bench` | both cards: D2H on one card into pinned memory, H2D on the other, both orders, 2.5 / 7.5 / 12.75 MiB and 24 x 3 MiB, the two legs alone and together (the pipeline's steady state), `--duration` to cover a busy window |
| `tool_pp_project` | CPU only: the per-chunk stage times of the two cards -> the two-stage pipeline simulation (3 slots) -> projected TTFT at 8k / 32k / 64k, the go rule, the other gates; `summary.txt` |
| `pp_phase0.ps1` | the one script that runs them (below) |

**Projection** (`pp_plan.h`): stage A's per-chunk cost is device 0's prologue + layers [0, k) + the D2H half of the hop; stage B's
is device 1's H2D half + layers [k, N) + the epilogue (MTP priming, lm_head on the last chunk, DFlash injection); the real chunk
grid goes through a two-stage simulation with a bounded slot ring (A starts chunk c + 1 only when a slot is free; B starts chunk c
after A published it and after chunk c - 1) plus a 2 ms tail; the baseline is the monolithic TTFT on device 1. Two projections:
"alone" (each card measured by itself) and "both busy" (both cards measured at the same time -- carries the dual-GPU clock / power
sag); **the go rule uses the both-busy one when it exists**. The hop halves are stretched by the ratio of the emulation's own
host-copy bandwidth to the measured cross-card bandwidth when the latter is slower (never credited when faster).

**Go rule** (design 8): projected speedup >= 1.6x at BOTH 8k and 32k, else STOP and do not start Phase 2. 64k is reported, not
part of the rule. Reported next to it, informationally (the design's table): stage A cost / full prefill in [0.49, 0.54] (device 0,
8k), device skew |monolithic 8k TTFT dev 0 / dev 1 - 1| <= 6 %, hop >= 3 GB/s concurrent with compute, MTP priming + DFlash injection
< 12 ms per 256-row chunk (else plan k = 35 and the ring-window injection skip). Exit codes of `tool_pp_project` / the script:
0 GO, 3 STOP, 2 inconclusive or a failed run.

### Running it (GPU; needs the user's approval; refuses while an `r4dx-*` process runs)

```powershell
# both cards free; add -UserPresent for the both-cards-busy phases (power!) with the user at the machine
powershell -NoProfile -File <scratchpad>\pp_phase0.ps1 [-UserPresent] [-Skip64k] [-Concurrent64k] [-Resume]
# (a copy lives at tools\pp\phase0.ps1; -BuildDir defaults to build\win-hip of the checkout it is in)
```

Steps: pre-flight (refuse if `r4dx-*` runs; each card's PCIe link, read through PnP), stage bench on device 1 then device 0
(mono + emu at 8k x3, 32k, 64k), `--dflash` at k = 35 on both cards (8k, 32k) and `--mtp 3` on device 1 (8k), the hop bench idle,
(with `-UserPresent`) both stage benches at once behind a barrier with the hop bench on top and the link re-read under load,
the clock probe (`R4DX_CLOCK_PROBE=1`, one 8k emulate run per card alone and, with `-UserPresent`, together), then the
projection. Outputs in `<models root>\r4dx\pp\phase0` (default `E:\models\r4dx\pp\phase0`): the CSVs, one log per process,
`notes.txt`, `summary.txt`. About 15-25 minutes.

**Power.** Nothing on this stack reads board power in software (no amd-smi / rocm-smi in `C:\opt\rocm\bin`; AMD's
overlay / HWiNFO are the only sensors). The 2026-09-28 power-off happened with both cards ramping (docs/prefill.md), so the both-cards
phases are opt-in, bounded (~25 s of dual load; 60 s with `-Concurrent64k`) and announced: the script prints when it will release
the barrier (`-PreGoSeconds`, default 10) so the overlay can be started first. The summary carries the clocks, the link state
and a line saying power was not measured by the script -- record the overlay's reading in this file.

**Link state seen on this box (2026-10-08, idle, read through PnP, no GPU work):** both R9700s report PCIe Gen5 x16 (current = max;
bus 3 under root port GPP0, bus 7 under GPP3); the slot-width question of the design's section 1.4 is therefore answered at idle
(x16 each) and the script re-reads it under load.

## 4. What this branch does not do

* **Phase 2 is not started**: no `StageChannel`, no stage-A rank thread, no half-weight `Model` (`layer_first / layer_last /
  boundary_norm / stage_only` load options), no `PpModel`, no `--pp` flag, no GDN state export / import, no sync-back.
* PP-emulate does not exercise the GDN state handoff (one Model), the `HIP_VISIBLE_DEVICES=1,0` device-selection check, the
  per-thread scopes of two threads, or real cross-device DMA. The real-vs-emulate gate (G2a) is Phase 2's.
* Quant2 (rotated) containers, Gemma 4 (own `RunChunk`) and tensor-parallel ranks are refused by design.
* The Phase 0 numbers: nothing has run. The synthetic-CSV check of `tool_pp_project` (plausible numbers, GO / STOP exit codes) was
  done on the CPU; its numbers mean nothing.

## 5. Files

`src/model/chunk_run.h`, `src/model/pp_stage.cpp`, `src/model/pp_plan.h`, `src/model/model.{h,cpp}` (the three phases, the API),
`tests/model/test_pp_emulate_identity.cpp`, `tests/model/test_pp_plan_cpu.cpp`, `tests/model/tool_pp_{stage_bench,hop_bench,project}.cpp`,
`tools/pp/phase0.ps1`. Switches: `R4DX_PP_EMULATE=<k>` (off by default), `ModelOptions::pp_emulate_split`, `R4DX_CLOCK_PROBE=1` (existing; used by the
clock step).
