# Pipeline-parallel (layer-split) 2-GPU prefill: Phase 0 tooling, Phase 1 (stage-capable RunChunk, PP-emulate), Phase 2 (the real pipeline)

Status (2026-10-08, branch `pp`, from `main` 4e710ab):

* **Phase 0** (measurements) ran on the GPUs: both PCIe links x16 Gen5, stage cost of MTP priming / DFlash injection 8.59 / 3.51 ms
  per 256-row chunk (under the 12 ms gate), and the projection says **GO** (>= 1.6x at 8k and 32k on each card alone; the numbers
  are in section 3 and `E:\models\r4dx\pp\phase0\summary.txt`: 1.87x / 1.90x / 1.90x at 8k / 32k / 64k, stage A / stage B = 1.06).
* **Phase 1** (the three-phase RunChunk and PP-emulate) is validated on the GPUs: `test_prefill_chunk_identity` (+ defaults) PASS,
  `tp1_identity` rows equal to main, and `test_pp_emulate_identity` passed every identity case and nine negative controls; the tenth
  (real / dflash7 `neg/skip-dflash`) crashed the process -- fixed in this branch (section 2).
* **Phase 2** (the real two-GPU pipeline, section 4) is **validated on the GPUs (2026-10-08, `E:\models\r4dx\pp2`)**:
  * identity (G2a): `test_pp_emulate_identity` PASS (11 configurations, 10 negative controls), `test_pp_real_identity` PASS (9
    configurations, 6 negative controls: real pipeline == monolithic prefill, bit for bit, on the 4-layer containers and the real one,
    plain / `--mtp 3` / `--dflash` / image rows / warm turns / checkpoints), and `tp1_identity` rows 1-6, 8, 9 EQUAL to main (PP off is
    unchanged);
  * cold TTFT (G2b), median of 2 runs, HIP device 1 baseline vs `--pp 2`, greedy output identical in every arm: 8k 3.150 s -> 1.702 s
    (**1.85x**), 32k 14.034 s -> 7.392 s (**1.90x**), 64k 32.049 s -> 16.794 s (**1.91x**); with `--pp-split 32` 1.657 s (1.90x) at 8k and
    7.236 s (1.94x) at 32k. MTP and DFlash (k = 7, 8k and 32k) give the same text, ids and accept stats as the baseline; decode 37.2
    vs 37.3 tok/s (**0.997x**, gate >= 0.97), text identical;
  * failure: a hard kill of `r4dx-cli --pp 2` at 25 % and 65 % of a 32k prefill leaves no orphan and no TDR, and the next `--pp 2` run
    reproduces the baseline's bytes;
  * soak (G2c, `tools/pp/soak.ps1`): 5 minutes (17 iterations, 9 compared) and 60 minutes (199 iterations, 100
    compared, all equal; buffer drift < 0.001 MiB, VRAM drift 6 MiB; no TDR; summary and teardown `exit_code` 0). The soak's
    "mean prefill speedup 1.60x, best 1.68x" is NOT a production figure: the soak ran with `--verify`, whose `PpLiveDigest`
    copies and hashes each stage's whole live state (about 0.45 s at sync and 0.85 s at the tail of a 25k-row call), and
    `pp_prefill_s` times only each iteration's cold first prefill. Without verify the warm-turn sync-back is about 6 ms (76.5 MiB
    of GDN state, export 2.9 ms + import 2.8 ms) and the tail about 3 ms (scratchpad design_pp_warm.md, 2026-10-08).
  * defaults since 2026-10-08: split k = 32 (35 with a drafter) and `--pp-min-rows` 512 (break-even about 400 rows).

  **Placement caveat of those runs.** They ran with `HIP_VISIBLE_DEVICES=1,0`, which on this ROCm (10.1, Windows) does NOT reorder
  the devices, so stage B (decode) sat on HIP device 0 = pci 03:00 (the desktop card) and stage A on device 1 = pci 07:00 -- the
  opposite of the design (4.1). Correctness is unaffected (both cards are the same model), and the TTFT numbers above are what that
  placement gave. Placement is now explicit (`--pp-devices B,A`, `R4DX_PP_DEVICES`, default decode on the last visible ordinal =
  physical device 1 = pci 07:00 with `HIP_VISIBLE_DEVICES` unset; section 4.2) and `HIP_VISIBLE_DEVICES=1,0` is no longer needed. The
  corrected placement has to be re-confirmed on the GPUs (identity, an 8k TTFT, the log lines of 4.7).

The design this implements (research ideas #26 / #33): 2 stages of 256-row super-chunks, stage A = layers [0, k) + embedding on
device 0 (the desktop card), stage B = layers [k, N) + final norm + lm_head + MTP priming + DFlash injection on device 1 (the
headless card, decode's), the boundary through pinned host memory (no peer copies: docs/tp.md 1.2), a 3-slot ring, k = 33 by
default (32..35 keep 8 attention layers per stage; 35 with a drafter), PP prefill + TP=1 decode, byte-identical to the monolithic
prefill.

(The full design document is not in the repo; what this branch relies on is restated here, and "the design" below means it.)
This file records what exists, how to run it, the pass rules, and what is not done. The design's TTFT numbers (1.80x / 1.85x /
1.86x at 8k / 32k / 64k expected, 1.63x / 1.68x / 1.70x pessimistic) were DERIVED; Phase 0 measured the stage costs they rest on
(section 3), the real pipeline's TTFT is measured by `ttft_cli.ps1 -Pp`.
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

**GPU result (2026-10-08) and the one crash.** Every identity case passed (4-layer bf16 / w4a16 plain, capture, mtp3; real plain with
splits 33 / 5 incl. `len8145`, both poison modes, real mtp3, real vision plain, real dflash7 at 35 / 21 / 5) and nine negative controls
failed the comparison as required; then the process died with an access violation in the tenth, real / dflash7 `neg/skip-dflash`.
Cause (read from the code, not reproduced): the control dropped the DFlash columns after poisoning them with 0xFF (NaN in bf16), so the
real drafter ran on NaN features, its logits were NaN, `r4dx_topk16_f32` returned its documented `(-inf, INT32_MAX)` sentinel for
rows with fewer than 16 non-NaN values, and `DflashDraft::SelectorWalk` indexed its host codebooks with `INT32_MAX` -- a host read
far outside the vector. (The 4-layer capture case has no drafter, so nothing walked the ids.) Fixed twice: the control now
zero-fills the dropped columns (finite features -- the drafter runs, its ring differs from the monolithic run's, which the
`dflash.k` / `dflash.v` digests see), and `SelectorWalk` throws on a candidate id outside the vocabulary instead of reading past a
codebook (a latent crash for any NaN drafter output). The test needs no change; the control must still FAIL the comparison.

## 3. Phase 0: measure before building (tooling)

The design's Phase 0 table, as tools. All built under `tests/model` (never `add_test()`ed: they print / write numbers):

| tool | measures |
|---|---|
| `tool_pp_stage_bench` | one card, the full model: the monolithic prefill TTFT and the PP-emulate run -- per chunk the prologue + layers [0, k) (stage A), the hop's D2H and H2D halves, layers [k, N) (stage B), the epilogue, and in it the MTP priming and the DFlash injection (`--mtp K`, `--dflash`); sizes 8k = 8145, 32k = 32623, 64k = 65529 tokens (the design's prompts) or `n<N>`; repeats keep the median; `--layers N` loads only layers [0, N) (`--layers 32` + mono runs = the design's literal half-the-layers probe: its ratio to the full model and the skew between the cards); `--barrier <prefix>` starts two processes together |
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

**Result (2026-10-08, `E:\models\r4dx\pp\phase0\summary.txt`): GO.** Projected cold TTFT of plain prefill (int8 + split-KV defaults), k = 33:

| size | tokens | chunks | monolithic dev 1 | stage A sum | stage B sum | A / B | pipelined, each card alone |
|---|---|---|---|---|---|---|---|
| 8k | 8145 | 35 | 3195.7 ms | 1660.9 ms | 1567.5 ms | 1.060 | 1707.4 ms (**1.87x**) |
| 32k | 32623 | 129 | 14003.6 ms | 7294.2 ms | 6826.8 ms | 1.068 | 7354.7 ms (**1.90x**) |
| 64k | 65529 | 259 | 32164.4 ms | 16839.3 ms | 15768.1 ms | 1.068 | 16910.7 ms (**1.90x**) |

Both PCIe links x16 Gen5 (idle and under load); the clock probe shows ~2450 MHz on both cards alone and together (no dual-GPU
sag at 256 rows); stage B's cost of MTP priming / DFlash injection 8.59 / 3.51 ms per chunk. **Stage A is the slower stage by 6-7 %**
(its sum includes the D2H half of the hop and device 0 runs the prologue): k = 32 would move one GDN layer (~1.8 ms per chunk,
~3 %) to stage B and balance them; the default stays 33 (35 with a drafter) as the design says, `--pp-split` moves it, and the
Phase 2 TTFT run is what decides.

### Running it (GPU; needs the user's approval; refuses while an `r4dx-*` process runs)

```powershell
# both cards free; add -UserPresent for the both-cards-busy phases (power!) with the user at the machine
powershell -NoProfile -File <scratchpad>\pp_phase0.ps1 [-UserPresent] [-Skip64k] [-Concurrent64k] [-Resume]
# (a copy lives at tools\pp\phase0.ps1; -BuildDir defaults to build\win-hip of the checkout it is in)
```

Steps: pre-flight (refuse if `r4dx-*` runs; each card's PCIe link, read through PnP), stage bench on device 1 then device 0
(mono + emu at 8k x3, 32k, 64k; then `--layers 32` mono 8k / 32k on both), `--dflash` at k = 35 on both cards (8k, 32k) and `--mtp 3` on device 1 (8k), the hop bench idle,
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

## 4. Phase 2: the real two-GPU pipeline

Off by default: nothing below runs unless `--pp 2` (r4dx-cli and r4dx-server), `ModelOptions::pp = 2` or `R4DX_PP=1` asks.

### 4.1 Structure

```
facade / engine thread  (hipSetDevice(B) = physical device 1, the headless card)      stage-A thread  (hipSetDevice(A) = device 0, the desktop card)
  Model B  = the ordinary full Model: decode, MTP, DFlash, vision, checkpoints          Model A = layers [0, k] resident (k+1 layers), embedding,
  Prefill(call) ---------------- pipelined ------------------------------------------   runs layers [0, k) of every chunk; no lm_head use, no MTP,
     per chunk: prologue -> wait slot -> import -> layers [k, N) -> epilogue -> release     no drafter, bounded GPU submission (SubmitBounder)
                                          ^                                             per chunk: prologue -> layers [0, k) -> sync -> wait free slot -> export -> publish
                                          +----------- pp::StageChannel (3 pinned slots, FIFO) -----------+
     end of call: GDN state of A's layers  <---------- one pinned buffer, flag in the channel ----------- export after A's last chunk
```

* `src/model/pp_model.{h,cpp}` `PpModel : TextModel` owns B, A (built, used and destroyed on A's thread: `tp::RankWorker`), the
  ring and the three pinned buffers; every `TextModel` call forwards to B (decode, speculative rounds, checkpoints, `EncodeImages`
  are byte-for-byte today's: same Model, same device, same thread). `Prefill` / `PrefillMultimodal` of at least `--pp-min-rows`
  rows (1024) run on both stages; shorter calls run on B alone.
* Stage A is **not a new class**: the same `Model`, loaded with `layer_limit = k + 1` (`ModelOptions` already had it). Its
  `has_next_layer` rule (`i + 1 < NumLoadedLayers()`) makes layer k - 1's Mlp fuse layer k's real `input_layernorm` weight exactly as
  the monolithic run does, so the three carry buffers (`cur`, `buf_normed_`, `buf_normed_pre_`) leave A exactly as they pass from
  layer k - 1 to layer k. The design's `layer_first / layer_last / boundary_norm / stage_only` load options were not needed; the price
  is one resident layer (0.2 GiB) and an unused lm_head and, with a vision container, tower on device 0 (VRAM there is plentiful;
  `PpModel` prints both stages' VRAM at load).
* `Model` gets the stage roles (`src/model/pp_stage_real.cpp`, `Model::PpAttach`): in `RunChunk`, a prompt-prefill chunk of an
  attached Model goes to `RunChunkPpStageA` (prologue, `RunLayerRange(0, k)`, synchronize, acquire a free slot, D2H of the carry /
  DFlash columns / the KV blocks the chunk wrote, synchronize, publish; `pos_ += T`; no logits) or `RunChunkPpStageB` (prologue,
  acquire the full slot, check its header against the chunk, H2D into `buf_a_` / `buf_normed_` / `buf_normed_pre_` / the DFlash
  columns / the KV blocks, `RunLayerRange(k, N)`, `ChunkEpilogue`, release). The carry's row width is the configuration's
  `hidden_size` (`Model::PpCarryHidden`), checked against the layout on both sides (4.6 "GPU result"). **Both Models run the same `Model::Prefill` /
  `PrefillMultimodal` on the same tokens**, so the chunk grid, the int8 / M = 256 scope decisions, the mrope bookkeeping and the
  last-chunk flag agree by construction; the slot header (position, rows, DFlash columns, last) and the payload size are checked
  anyway and a disagreement throws. Everything else in `RunChunk` is untouched (one extra branch on `pp_active_`).
* `pp::StageChannel` (`src/model/pp_channel.h`, HIP-free): ring of caller-provided pinned slots, FIFO by counters, back-pressure
  (stage A blocks on a full ring), bounded waits (30 s, `PpOptions::timeout_ms`) that throw `ChannelTimeout`, a poison flag
  that wakes every waiter with `ChannelPoisoned`, per-call `BeginCall` / `Reset`, and the "GDN state complete" flag
  (`PublishBulk` / `BulkReady` / `WaitBulk`). No P2P copy, no spin kernel and no device-side wait anywhere: a slot is plain
  host memory that A fills with `hipMemcpyAsync` D2H and waits for before publishing and B drains with H2D and finishes with
  before releasing. `MakeSlotLayout` (same header) derives both sides' offsets from (position, rows, DFlash columns, attention
  layers, KV block geometry) so they cannot disagree.
* **Slot payload** for a 256-row chunk at k = 33: 3 x 2.5 MiB carry + 4 MiB KV (17 blocks of the 8 attention layers when the chunk
  starts mid-block: 4.25 MiB) + DFlash columns (2.5 MiB per column, up to 3 with k = 35); slots are sized for the worst case
  (`MaxSlotBytes`, ~24 MiB). Pinned buffers use `hipHostMallocPortable` (DMA-able from both devices, as the shared embedding table).

### 4.2 Switches

| switch | meaning | default |
|---|---|---|
| `--pp 2` (r4dx-cli, r4dx-server) / `ModelOptions::pp = 2` / `R4DX_PP=1` | run prompt prefill as the two-stage pipeline; `--pp 1` / `pp = 0` force it off | off (`pp = -1` follows `R4DX_PP`, unset = off) |
| `--pp-split N` \| `auto` (`PpOptions::split`) | k = stage A's layer count | auto: 32, 35 with a drafter |
| `--pp-min-rows N` | calls shorter than this run on the decode Model alone | 512 |
| `--pp-verify` / `R4DX_PP_VERIFY=1` | digest the live state of both stages after every sync-back and hand-off and compare (slow: copies the state to the host) | off |
| `--pp-submit-layers N`, `--pp-max-inflight K` | stage A's bounded GPU submission: a forced submission every N layers (fewer past 16k / 64k / 128k of context), at most K units queued; `0` layers = off | 32, 1 (as TP) |
| `--pp-devices B,A` (r4dx-cli, r4dx-server) / `R4DX_PP_DEVICES=B,A` / `PpOptions::devices` | the process-visible HIP **ordinals** of stage B (the decode card) and stage A, two different values inside the visible range; `auto` = the default. The flag wins over the variable | auto: B = the last visible ordinal, A = the one before it (with `HIP_VISIBLE_DEVICES` unset: B = ordinal 1 = physical device 1 = pci 07, the headless card every TP=1 run uses; A = ordinal 0 = pci 03, the desktop card) |
| `HIP_VISIBLE_DEVICES` | not needed: leave it unset. It must expose both cards (a value that leaves fewer than two is refused at load: `pipeline-parallel prefill needs two visible HIP devices ... HIP_VISIBLE_DEVICES=<value> exposes <n>`). It is never used to order the stages: on this ROCm `1,0` does not reorder | unset |
| `R4DX_PP_TRACE=1` | diagnostic: one stderr line per chunk and boundary point with a hash of each carry buffer and of the chunk's ids -- `A-dev` (stage A's device buffers after layers [0, k)), `A-slot` / `B-slot` (the slot bytes as A published / B found them), `B-dev` (B's buffers after the import), `mono` (a monolithic call on the decode Model, split at the same k). Match lines by `ids` + `pos` + `rows`. The hash of an empty region is the offset basis `14650fb0739d0383`. Synchronizes (slow) | off |

Refused at load / parse: `--tp 2` (mutually exclusive), Gemma 4, a rotated (quant2) container, a probe (`R4DX_CLOCK_PROBE`), fewer
than two visible devices, a malformed or out-of-range `--pp-devices` / `R4DX_PP_DEVICES`, both ordinals on one physical GPU (same pci
bus), GPUs of different architectures, a stage-A / stage-B disagreement on the prefill chunk size or on int8 prefill (the bits would
differ), a split outside [1, layers - 1].

**Placement.** The stages are chosen by explicit ordinal (`pp::ResolvePlacement`, `pp_sync.h`, CPU-tested), never by the order
`HIP_VISIBLE_DEVICES` lists the cards: on ROCm 10.1 / Windows `HIP_VISIBLE_DEVICES=1,0` exposes the same two cards in the natural
order (ordinal 0 = pci 03, ordinal 1 = pci 07), which put decode on the desktop card in the 2026-10-08 runs. The auto rule is
`--tp-devices auto`'s (docs/tp.md 9.2): decode on the LAST visible ordinal. A TP=1 run calls no `hipSetDevice` and uses ordinal 0 of
whatever `HIP_VISIBLE_DEVICES` exposes (the production habit `=1` makes that physical device 1), so with both cards visible decode
lands on the same physical card (device 1, pci 07) a TP=1 run uses. Because decode is then not on ordinal 0, `PpModel` binds the
calling thread with `hipSetDevice(B)` in `Load` and at the start of every entry point (`PpModel::BindStageB`; the server loads on
the main thread and runs on the engine worker thread; the destructor binds too), and stage A's thread selects A as before. The load
log names both: `[r4dx-pp] stage B (decode) -> HIP device 1 (<name>, pci 07:00); stage A -> HIP device 0 (<name>, pci 03:00) [placement:
auto; 2 visible, HIP_VISIBLE_DEVICES=<unset>]`.

### 4.3 One pipelined call

1. B collapses a speculative window (`PpPrepareCall`), reads the call's start position p0 and asks the `MirrorTracker` (4.4) what
   stage A is missing: the GDN state and/or KV rows [kv_valid_to, p0).
2. B exports those to pinned host buffers (synchronously, device idle); in verify mode it digests its own live state.
3. Stage A's command is posted: `PpSetSyncState` (position, `started`, mrope state), the drafter-injection policy, import of the
   sync-back, then `Prefill` / `PrefillMultimodal` (image rows A cannot read -- B's device rows -- are copied to the host first and
   spliced from there, the TP path), then export of its GDN state and `PublishBulk`.
4. B runs the same call with its role active. Chunk c on B overlaps chunk c + 1 on A; A runs up to three slots ahead.
5. Before B's last chunk computes, if A's GDN export is already complete (`BulkReady`) B starts importing it on a side stream
   (`Model::PpImportGdnAsync`: it writes only layers < k's state, which B's own layers never read), overlapping the chunk; otherwise
   it waits and imports after. `PpImportFence` before the call returns, so `SaveCheckpoint` and decode see a complete B.
6. B joins stage A (the watchdog: 60 s without a chunk is fatal), updates the tracker (both stages agree through the new
   position), and in verify mode compares the live digests.

The **GDN wire form** (per GDN layer in layer order, 256-byte aligned): the live recurrent slot (`SlotForSeq(0)`, 3 MiB fp32), then
the conv history as `[conv_dim][conv_width - 1]` bf16 (60 KiB). The two stages' conv lines have different row pitches
(`state_len_max = conv_width - 2 + max_decode_window`: 3 on stage A, bigger on a speculating B), so a line whose pitch equals the
history length is copied contiguously and a longer one with a 2D copy at offset 0 (the line's other entries are not live state).
~76 MiB at k = 33.

### 4.4 The mirror (stage A's copy of the conversation) and the sync-back

After a pipelined call both stages agree through the call's end position. Decode, speculative rounds, a restore, a `Reset` and any
B-only prefill then move B alone. `pp::MirrorTracker` (`src/model/pp_sync.h`) keeps two numbers, `kv_valid_to` (A's KV rows
[0, kv_valid_to) equal B's) and `gdn_pos` (the position at which A's GDN state equals B's, or -1), and its rules only ever make them
smaller or exact: any B-only call invalidates the GDN state and cuts `kv_valid_to` to the position it started at; a restore
invalidates the GDN state; `Reset` resets both Models and the tracker; a completed pipelined call sets both to its end. The next
pipelined call copies B's GDN state if `gdn_pos != p0` and KV rows [min(kv_valid_to, p0), p0) -- 16 KiB per row (8 attention
layers), ~12 ms for the GDN state, so a 1000-token reply costs 16 MiB. A sync-back above 512 MiB of KV (32k rows A is behind)
falls back to a B-only call. The rules are checked on the CPU against an abstract model of both devices over 20,000 random
sequences of 14 operations (`test_pp_sync_cpu`), and `R4DX_PP_VERIFY` checks them against the real state on the GPUs.

Prefix reuse and the server's prompt checkpoint need nothing else: `PrefixState` and `Model::SaveCheckpoint` live on B; the engine's
checkpoint-split prefill is two `Prefill` calls, each of which finishes its hand-off, and the short second one (below
`--pp-min-rows`) runs on B alone and is repaired by the next warm turn's sync-back.

### 4.5 Failure handling

`PpModel` has `TpModel`'s state machine: `kReady`, `kNeedsRecovery` (every device-work call throws until `Reset()`), `kFatal`
(restart the process). A failure on either stage poisons the channel, so the other stage's waits return; the facade joins stage A
(a stage that makes no progress for 60 s -- its heartbeat is bumped per chunk -- is `kFatal`), rethrows the root cause (stage A's
own error unless stage B's is only "the channel was poisoned"), and `Reset()` heals both Models, the channel and the tracker. The
server's existing "a failed request makes `PrefixState` demand a `Reset()`" rule covers the rest. Shutdown poisons the channel,
waits for stage A (30 s, then `quick_exit(3)` rather than free memory a kernel may still touch) and destroys A's Model on its
thread.

### 4.6 Byte identity (gate G2a) and the tests

The claim is the design's 7.1: every layer runs the same kernels, selected by shape only, with deterministic reductions, from the
same container; the carry crosses as raw bytes. New here beyond Phase 1: two threads, two devices, the GDN hand-off and the sync-back.

* `tests/model/test_pp_real_identity.cpp` (GPU, **both cards**, `HIP_VISIBLE_DEVICES` unset or `0,1` as ctest sets it, SKIP 77 with fewer than two visible devices): one `PpModel` per
  configuration (stage A loaded for the largest split, `SetSplit` moves k without reloading); every scenario runs monolithic (the
  decode Model driven directly) and then through the `PpModel` with `min_rows = 1`; compared bit for bit: each Prefill call's last-row
  logits, the greedy / speculative tokens, the digest of ALL of B's state (KV caches of both stages zeroed at the start), the DFlash
  capture, and after every pipelined call stage A's live state against stage B's (`PpLiveDigest`); `PpModel` runs with verify on, so it
  makes the same comparison itself after every sync-back and hand-off. It also checks the number of calls and chunks that went
  through the ring. Shapes: Phase 1's list (1 .. 8145 rows, the prefix-reuse shapes) plus **warm turns** (prefill, decode, prefill:
  the sync-back of GDN state and generated KV rows at a mid-block start; one-row and 63-row tails; speculative rounds between turns;
  checkpoint save / restore / re-prefill; a variant with `min_rows = 1024` where short calls run on B alone and the next long one
  repairs the mirror) and image rows (an image mid-prompt, at position 0, a conversation with decode between the image turn and a
  text continuation, so stage A must learn the mrope delta). Configurations as Phase 1: 4-layer bf16 / w4a16 plain / capture / `--mtp 3`
  at splits 1, 2, 3; the real container plain (33, 5), `--mtp 3` (33, 5), `--dflash 7` (35, 21, 5), vision. **Negative controls**
  (`PpModel::SetTestFault`): no sync-back to stage A on a warm turn (a stale mirror), no GDN import into B -- each must change an
  observable or fail a call.
* **GPU result (2026-10-08, `E:\models\r4dx\pp2\logs\test_pp_real_identity.err.txt`): every configuration FAILED; fixed,
  not yet re-run.** The pattern: every observable fed by the carry differed (B's layers >= k: GDN states, KV, logits, MTP KV, the
  capture of B's layers), from a 1-row prompt at k = 1 on; everything else that crosses -- A's GDN states and the KV blocks of
  A's attention layers (the same slot, the same pinned memory; B reads them only after its epilogue's synchronize), the GDN
  hand-off (fenced), the sync-back (`R4DX_PP_VERIFY` found nothing: it digests layers < k only) -- was exact; and the wrong bits
  depended on the configuration's history (the plain and the capture configurations, the same computation, gave different
  pipelined hashes), i.e. B's first layer read the carry buffers' previous contents. A first guess (an unsynchronized import;
  synchronizes added) changed no failure count. `R4DX_PP_TRACE=1` (4.2) then showed A's device buffers right and every slot's
  three carry regions hashing to the hash of ZERO bytes (`14650fb0739d0383`, the offset basis) at every chunk size, and B's
  buffers after the import = its own prologue's embedding / stale contents. Cause: `RunChunkPpStageA` / `B` read
  `const int64_t hidden = r.hidden;` BEFORE `ChunkPrologue(r)`, and `ChunkRun::hidden` is 0 until the prologue fills it
  (`chunk_run.h`), so `MakeSlotLayout` sized every carry copy (and the DFlash columns' width) 0 bytes on both sides -- the
  headers, the payload sizes and the KV blocks (sized from the block geometry, not `hidden`) all still agreed, and no HIP call
  failed (a 0-byte copy is legal). PP-emulate's hop reads `r.hidden` after the prologue, which is why G1b passed. Fix: the width
  is the configuration's `hidden_size` (`Model::PpCarryHidden`); both stages check `carry_bytes == T x hidden x 2` and that the
  prologue agrees; `pp::MakeSlotLayout` refuses rows or hidden < 1 (CPU-tested in `test_pp_channel_cpu`); the unneeded
  synchronizes were removed again (stream order covers the import, as everywhere else). The below-threshold / turn diffs of the threshold variant are the same bug: their first diffs are in
  a pipelined call (1100 / 1200 rows); later B-only calls differ only because the tokens decoded in between already did.
  Separately, the log showed `HIP_VISIBLE_DEVICES=1,0` did not reorder the devices on this runtime: ordinal 0 (stage B, decode) was
  pci 03:00 -- the desktop card in docs/tp.md -- and ordinal 1 (stage A) pci 07:00, the opposite of 4.1's intent. Not a correctness
  issue; fixed by explicit placement (4.2, "Placement").
* **GPU result after the fix (same day, `E:\models\r4dx\pp2\logs`):** `test_pp_real_identity` PASS (9 configurations, 6 negative
  controls), see the status block at the top for the TTFT, decode, spec, kill and soak results.
* CPU, always run: `test_pp_channel_cpu` (FIFO, back-pressure, timeouts, poison, per-call reset, bulk flag, the payload arithmetic,
  a million chunks between two threads with payload integrity), `test_pp_sync_cpu` (the tracker against the abstract model, the
  switches, the placement rule and the `--pp-devices` / `R4DX_PP_DEVICES` parser), the arg tests of both binaries.

### 4.7 Harness and how to run the gates (GPU; the user approves; nothing else may be running)

```powershell
# G2a -- needs both cards; takes a while (several model loads on each)
Remove-Item env:HIP_VISIBLE_DEVICES -ErrorAction SilentlyContinue                                  # both cards visible; the placement is PpModel's
.\build\win-hip\tests\model\test_pp_real_identity.exe                                              # R4DX_TEST_ONLY=l4/bf16/plain narrows it
# expect on every load: [r4dx-pp] stage B (decode) -> HIP device 1 (..., pci 07:00); stage A -> HIP device 0 (..., pci 03:00) [placement: auto; ...]
# G2b -- cold TTFT, baseline then pipelined (G2b: >= 1.6x at 8k and 32k, >= 1.7x at 64k) and decode tok/s unchanged
.\tools\prefill\ttft_cli.ps1 -Device 1 -Lengths 8k,32k,64k -Runs 2 -OutDir E:\models\r4dx\pp\ttft_base
.\tools\prefill\ttft_cli.ps1 -Pp -Lengths 8k,32k,64k -Runs 2 -OutDir E:\models\r4dx\pp\ttft_pp     # -ExtraArgs '--pp-split','32' to try a split
# G2c -- the soak: random 2k-32k prompts, each generated monolithically and pipelined, tokens compared, TDR watch (5 min, then 60)
.\tools\pp\soak.ps1 --layout trellis --minutes 5 --max-ctx 36864 --json build\logs\pp_soak_5min.jsonl
# or all of it, with a verdict file
.\tools\pp\phase2.ps1 [-Dflash] [-Skip64k] [-SoakMinutes 5]       # E:\models\r4dx\pp\phase2\summary.txt
# the server
Remove-Item env:HIP_VISIBLE_DEVICES -ErrorAction SilentlyContinue; .\build\win-hip\src\server\r4dx-server.exe --model <container> --layout trellis --pp 2 --prompt-checkpoint on ...   # --pp-devices B,A to override the placement
```

`tool_pp_soak` (tests/model) logs one JSON line per iteration (both paths' prefill seconds, the speedup, decode tok/s, buffer drift on
both devices) and ends with `summary` and `teardown` lines; `soak.ps1` is the G2c verdict (no TDR, exit 0, both lines). Stage A runs the
desktop card at ~100 % duty for the whole prefill; run the first 64k prefills with the user present (power: the 2026-09-28 power-off
was with both cards ramping; the soak does not read power).

### 4.8 Decisions that differ from, or add to, the design

* Stage A loads `k + 1` layers instead of `layer_first / layer_last / boundary_norm / stage_only` options: no change to `Container::Load`,
  the boundary fusion is the monolithic code path itself. Costs one layer, the lm_head and (with a vision container) the tower on
  device 0; a `stage_only` load that skips them is the follow-up if device-0 VRAM ever matters.
* No `HostMailboxComm`, no events on the channel: the export waits with `hipStreamSynchronize` (the design's v1 sync shape, ~1.3 ms
  per chunk); the double-buffered export and async metadata of Phase 4 are not done.
* The mirror's validity is two integers, not "A's `pos_`": `PpSetSyncState` hands A the position / `started` / mrope state at every
  call, so A needs no bookkeeping of its own.
* `DefaultSplit` and the rest of the policy are HIP-free (`pp_sync.h`) so they are CPU-tested.

## 5. What is not done

* **The corrected placement has not been re-run.** Phase 2 is validated (status block), but on the placement with decode on the desktop card
  (pci 03); the explicit `--pp-devices` / auto placement (decode on pci 07) and the per-entry `hipSetDevice` binding of the facade thread
  are CPU-tested and compile, and need one GPU confirmation (4.7: placement log lines, `test_pp_real_identity` l4/bf16/plain and
  real/plain, an 8k TTFT). The server smoke (`tools/server/smoke.ps1 -Pp 2`, below) has not been run either.
* The auto-split at warm-up (`--pp-split auto` is the fixed default 33 / 35); Phase 0 says stage A is 6-7 % slower, so k = 32 is the
  first thing to try (`--pp-split 32`).
* Phase 4 tuning (double-buffered export, option (b) of design 1.4: ship `cur` only), the ring-window DFlash injection skip, and Phase 5
  (TP=2 decode + PP prefill) are untouched.
* `tools/server/smoke.ps1 -Pp 2` (design G3) is not written: the server accepts `--pp 2` (arg tests pass) and `PpModel` implements the whole
  `TextModel` surface, but the smoke matrix (plain, `-Dflash -ToolRoundTrip`, `-Vision -Dflash`, `-Mtp 3`, multi-turn, checkpoint, regenerate) has not been
  adapted; `tests/model/test_pp_real_identity.cpp` covers the same state transitions at the Model level.
* Fault-injection tests of the failure path (design G2d: A throws, A stalls, B throws, then `Reset()` and the next request matches) are not written;
  the code path exists (poison, join, `kNeedsRecovery`, `Reset`).
* Quant2 (rotated) containers, Gemma 4 and tensor-parallel ranks are refused by design.

## 6. Files

Phase 1: `src/model/chunk_run.h`, `src/model/pp_stage.cpp`, `src/model/pp_plan.h`, `src/model/model.{h,cpp}` (the three phases, the API),
`tests/model/test_pp_emulate_identity.cpp`, `tests/model/test_pp_plan_cpu.cpp`, `tests/model/tool_pp_{stage_bench,hop_bench,project}.cpp`,
`tools/pp/phase0.ps1`. Switches: `R4DX_PP_EMULATE=<k>` (off by default), `ModelOptions::pp_emulate_split`, `R4DX_CLOCK_PROBE=1` (existing; used by the
clock step).

Phase 2: `src/model/pp_channel.h` (ring, poison, slot / GDN wire arithmetic), `src/model/pp_sync.h` (mirror tracker, switches, device placement),
`src/model/pp_stage_real.cpp` (the stage compositions and hand-off primitives, `Model` members), `src/model/pp_model.{h,cpp}` (`PpModel`),
`src/model/gdn_state.h` (live-state accessors), `src/model/text_model.h` (`PpOptions`, `LoadTextModel(opts, tp, pp)`), `src/model/model.{h,cpp}`
(the roles, `ModelOptions::pp`), `src/cli/{cli_args.h,main.cpp}`, `src/server/{server_args.h,main.cpp,engine.{h,cpp}}`,
`tests/model/test_pp_{channel,sync}_cpu.cpp`, `tests/model/test_pp_real_identity.cpp`, `tests/model/tool_pp_soak.cpp`, `tools/pp/{soak,phase2}.ps1`,
`tools/prefill/ttft_cli.ps1 -Pp [-PpDevices B,A]`, `tools/tp/tdr_watch.psm1 -HipVisibleDevices`. Switches: `--pp 2`, `R4DX_PP`, `R4DX_PP_VERIFY` (all off by default),
`--pp-devices` / `R4DX_PP_DEVICES` (placement; auto by default).