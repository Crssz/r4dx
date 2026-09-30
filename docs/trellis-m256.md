# M = 256 trellis GEMM and the 256-row prefill chunk

Status: the kernel (`third_party/libr4d/r4d_gemm_trellis_nt_m256.hip`, libr4d branch `linear`) is in
`R4D_UNITS`, has its `r4d.h` / `r4d.hpp` entry, its registry row and pybind binding, a host legality check and
a ctest, and `ApplyLinear` can call it. A Model runs it only when **`R4DX_PREFILL_CHUNK=256`** is set (default
OFF, nothing changes when it is unset) for a prompt `Prefill` call at TP = 1 with no MTP head, no DFlash
drafter and no image. It is a spike on branch `linear` (not merged, not pushed): the design is in "The 256-row
prefill chunk" below, the kernel in "What it is". Since stage S1 the kernel reproduces EVERY shipped M = 64
tuning row (all seven classes x KB 4/5, including mlp.down KB 4's SK 16) bit for bit; the bench
(`tests/kernels/tool_trellis_m256_bench.hip`) and real-weight check (`tool_trellis_m256_real_check.hip`,
`build_m256_bench.ps1 [-Tool real_check]`, hipcc, no CMake target) are unchanged in purpose.

## The 256-row prefill chunk (stage S2, `R4DX_PREFILL_CHUNK=256`)

Result (MEASURED, HIP device 1, TP = 1, Huihui mix4.5m, main dd38f8b + branch `linear`; logs under
`D:\models\r4dx\linear\S2\logs`, outputs under `D:\models\r4dx\linear\spike`):

* **Bit identity: all 12 KL segments (prose / code / recall x 8k / 32k / 64k / 128k) are byte-identical
  (sha256 of every `logprobs.f16`) to the dense dumps `D:\models\r4dx\pflash\baseline\kl-dense-huihui`**, mean /
  p99 / max KL exactly 0, top-1 100%. Flag OFF: prose_8k and prose_32k are byte-identical to the same dumps
  (the noise floor of the harness is exactly 0, so this is a bit test). Greedy text of 4 short prompts
  (`tests/model/mtp_prompts.txt`), 4 long prompts (495 / 1069 / 2115 / 4310 tokens: super-chunks plus a
  64-row tail chunks of every size) and, with DFlash attached, 2 more: 10 of 10 identical ON vs OFF. A prefill
  split into two calls (`--prefix-split-at 1001`, so 1001 = 3 super-chunks + 233 rows, then 6935 = 27 + 23
  rows): ON byte-identical to OFF, while the split itself differs from the unsplit dump (so the test does see
  a moved chunk grid).
* **Cold TTFT** (`ttft_cli.ps1`, one fresh process per run, interleaved OFF / ON, one discarded warm-up run,
  device 1, seconds of `[stats] prefill`):

| length (prompt tokens) | OFF | ON | speedup |
|---|---|---|---|
| 8k (8145) | 7.186, 7.246 (mean 7.216) | 5.472, 5.504 (mean 5.488) | **1.315x** (pairs 1.313, 1.316) |
| 32k (32623) | 33.333, 33.598 (mean 33.466) | 25.874, 26.026 (mean 25.950) | **1.290x** (pairs 1.288, 1.291) |
| 64k (65529) | 78.369 | 62.861 | 1.247x |
| 128k (130884) | 199.512 | 168.611 | 1.183x |

  The gate (>= 1.25x at 8k and 32k) is met. The saving is a constant 13.6 / 14.7 / 15.1 / 15.1 ms per 64 rows
  at 8k / 32k / 64k / 128k (DERIVED from the table; S1's bench estimate was 0.24-0.30 ms per token = 15-19 ms
  per 64 rows), so the speedup falls with context as the unchanged attention and GDN scan take a larger share.
  Device 1 is 3-5% slower than device 0's reference (BASELINE_KL_TTFT.md: 6.93 / 31.62 / 74.37 / 193.56 s), so
  compare ON against OFF of the same device only. The KL runs' own prefill seconds (prefix only, ON on device 1
  against the device-0 dump) are 5.20 / 25.64 / 62.40 / 168.38 s (`kl-256\vs_dense.json`).
* Where the rest goes (`--profile-prefill`, 8k, flag ON, `spike\profile-on`; hipEvent spans cost time, the
  run's wall is 6.93 s; its "128 chunks" header and per-chunk column count 64-row chunks, ON ran 31
  super-chunks + 4 chunks): mlp.gate_up 26.4% and mlp.down 15.0% of gpu_sum are still the largest lines, then
  gdn.in_proj_qkv 7.0%, gdn.conv_prep 7.3%, attn.core_prefill 6.3%, kkt_solve 4.6%, chunk_scan 4.2%, in_proj_z
  4.9%, out_proj 4.8%, the gated rmsnorm 3.1%. GEMMs are 66.6% (68.1% OFF); the four GDN sequence ops
  (19.2% of gpu_sum) and the attention core are exactly the launches four 64-row chunks would make, so they
  did not shrink. Launches: 15750 ON vs 57600 OFF.
* ctest (`-LE tp2gpu`, this build dir): 94 of 94 passed with the flag unset (820 s), 20 SKIPPED for gitignored
  golden data this worktree does not have; the list includes the two new tests. main's known 95/95 differs
  because the three python-reference tests are registered only when the build was configured with a torch
  interpreter (`R4DX_REFERENCE_PYTHON`), which this build dir was not (it is a difference in what is
  registered, not a failure). With `R4DX_PREFILL_CHUNK=256` in the environment the same 94 pass (853 s, the same
  20 skipped); most of those tests prompt with fewer than 256 tokens, so this shows the flag disturbs nothing
  else (Load with wide buffers, short prompts, the MTP / DFlash / prompt-checkpoint / server tests), not that the
  wide path was exercised there -- the KL, TTFT and text runs above are what exercise it.

### What runs where

`Model::Prefill` decides once per call (`PrefillRowsForCall`, `prefill_chunk.h`): 256 rows only for a Model
whose buffers were sized wide at load (flag on, TP = 1, no MTP head, no drafter), with no feature capture, no
image ever spliced (3-axis rope rows), no quant2 container, and no per-chunk callback. Anything else prints one
stderr line ("r4dx: R4DX_PREFILL_CHUNK=256 ignored: <reason>; using 64-row prefill chunks", once per Model)
and runs today's path; it never throws. `--tp 2` and `--mtp` follow from the same table (a CPU ctest,
`test_prefill_chunk`, covers every row of it and the parser: unset / `64` are silent, anything but `64|256` warns
and uses 64); `--dflash` was run for real (the two DFlash prompts above: the line is printed, the text is the OFF text).
TP = 2 was not run (not approved), so its fallback is verified by that unit test and by reading: TpModel calls
`Model::Prefill` on rank Models whose `comm_ != nullptr`, and `Model::Load` never sizes a TP rank wide.
The prompt-prefill call's chunk grid is anchored at the call start: `remaining >= 256` gives a 256-row
super-chunk, else the ordinary 64-row chunks (the tail is today's, identical, chunk for chunk).
`PrefillMultimodal` with images, decode, verify windows, MTP priming and `PrefillProfiled`'s callers are
unchanged (`PrefillProfiled` follows the same grid so `--profile-prefill` shows the wide path).

A super-chunk (`RunChunk` with T = 256, only reachable from `Prefill`) is layer-major over 256 rows:

| op | 256-row super-chunk | why the bytes equal four 64-row chunks |
|---|---|---|
| embedding gather, rmsnorms, fused residual+norm, residual add, silu_mul (trellis producer), split_qg, qk-norm, rope, KV write, gate-mul | one launch over 256 rows | one workgroup / block per row (or element-wise), the launch shape does not enter a row's arithmetic; the trellis input transforms (`r4dx_trellis_input_bf16`, the fused producers) are one wave per (row, 128-block) |
| trellis linears (qkv, z, out, gate_up, down, qg, k, v, o) | `ApplyLinear(M = 256)`: one `r4d_gemm_trellis_nt_m256` launch where `PlanTrellisM256` has an exact configuration, else four 64-row launches as today | the M = 256 kernel reproduces the shipped M = 64 row's (SK, SKG) summation bit for bit (below); the plan reads that row from `PickTuning` at M = 64 |
| gdn in_proj_a / in_proj_b (bf16 GEMM, M <= 64) | four 64-row launches | exactly the launches of four chunks |
| gdn conv prep, kkt solve, chunk scan (+ the state commit), gated norm | once per 64-row sub-slice, in order (`GdnLayerParams::seq_slice = 64`) | each sub-slice is the call a 64-row chunk makes: `has_init` true after the first, conv history and the fp32 state handed on through the slot |
| attention core | once per 64-row sub-slice (`attn_slice = 64`), q_len 64, `seqused_k[j] = pos + 64 (j + 1)` from a per-slice device array | each is the exact-wide (or split / dense, per `R4DX_PREFILL_SPLITKV`) launch a 64-row chunk makes; the KV write covers all 256 rows first and the causal mask hides later sub-slices' keys from earlier ones |
| lm_head / logits | last row only, from `cur + (T - 1) * hidden` | as today |

Memory: with the flag on (and a Model that can run it) the activation buffers, the position array and the
seqused array are sized for 256 rows and the arena is 224 MiB instead of 96 MiB (the widest layer, the mlp,
needs about 90 MB: gate_up 17.8, the M = 256 GEMM's split-group partials 35.6 + 10.5, h and the transformed A
8.9 each; the GDN / attention scratch is dropped with `arena_.Reset()` before the mlp, which is safe by stream
order). Measured VRAM: "arena+scratch=0.21875 GiB" in the load line; unset, the load is byte for byte as before.
The scout's "72 MB at 256 rows" missed the M = 256 GEMM's fp32 partials (`SKG x 256 x N x 4 B`).

The M = 256 kernel is reachable from `ApplyLinear` only inside `ScopedTrellisM256` (a thread-local, set by
`RunChunk` / `PrefillProfiled` around a super-chunk), so no other caller (decode, verify windows, the drafter,
tests, the vision tower) can reach it whatever M it passes. `R4DX_M256_SHAPES=NxK,...` (a debug / bisection aid,
unset normally) restricts it to the listed linear shapes; it was not needed (V2 passed first time).

Wired configurations (from `PlanTrellisM256`, printed by `test_trellis_m256`; SK / SKG are the shipped M = 64
row's; W = K slices resident per workgroup x the groups walked; all 14 rows of the model have one):

| class | KB 4 | KB 5 |
|---|---|---|
| mlp.gate_up | SK 4, W2 x 2 | SK 4, W4 x 1 |
| mlp.down | SK 16, SKG 2, W4 x 4 (34-k-tile tail) | SK 4, SKG 2, W4 x 1 |
| gdn.in_proj_qkv | SK 2, W2 | SK 2, W2 |
| gdn.in_proj_z | SK 4, W4 x 1 | SK 8, W8 x 1 |
| gdn.out_proj, attn.o | SK 4, W4 x 1 | SK 8, W8 x 1 |
| attn.qg | SK 4, W4 x 1 | SK 4, W4 x 1 |
| attn.k, attn.v | SK 8, W4 x 2 | SK 8, W8 x 1 |

The KB 5 rows with more than one group of slices (S1 used W4 x 2 for z, out and k / v at KB 5) are no longer
instantiated: they compile to 192 VGPRs (KB 5 PH > 1), and the build check below allows 190. KB 5 therefore runs
those SK 8 rows as W8 x 1 (32-wave workgroups, 184 VGPRs). S1 measured W8 x 1 at 1.54 / 1.57 (z), 1.73 / 1.71
(out) and 2.18 / 2.07 (k / v) against the shipped row, versus 1.64 / 1.67, 1.81 / 1.85 and 2.15 / 2.07 for W4 x 2:
a 0-6% loss on those classes (S1 numbers, not re-measured here).

### libr4d and build

* `r4d_gemm_trellis_nt_m256` is in `R4D_UNITS` (`r4d_core`, 17 units when this was written; the cut later left 13), declared in `r4d.h` with its contract,
  wrapped as `core::r4d::GemmTrellisNtM256`, in `r4d_registry.hip` (op `gemm_nt_m256`, never returned for a
  `gemm_nt` request), and (when this was written) in `r4d_module.hip` (pybind), `build_windows.ps1`, `build.sh` and the README table; those files are no longer vendored.
* Host legality: `r4d_gemm_trellis_nt_m256_check(M, K, N, n_split, KB, SK, NP, SKG, U, skw)` returns nullptr or a
  message naming the first broken rule (M, K / N multiples, KB, SKG, NP, n_split, NP * U == M / 64, SK, the
  k-tile divisibility, slice length, skw, waves, LDS, and "not instantiated"); the launch throws with that
  message. One predicate (`r4d_t256_inst`) is the instantiation table for both.
* `third_party/check_trellis_isa.cmake` now takes `GEMM_KERNEL` and `MAX_VGPR`; `third_party/CMakeLists.txt`
  compiles the M = 256 unit to a device listing (`--cuda-device-only -S`, the object's flags) and runs it:
  **22 instantiations, max 190 VGPRs, no scratch, no spills, 0 near dependencies behind the decode's asm** (the
  M = 64 unit: 100 kernels, max 189). To meet it three combinations that spilled or needed 192 VGPRs were
  dropped from the instantiation table: KB 4 W4 x 4 without a tail (8 B scratch; no shape of the model uses it)
  and every KB 5 configuration with more than one group of slices (above).
* `tests/kernels/test_trellis_m256` (ctest, `SKIP_RETURN_CODE 77` only without a HIP device, needs no goldens):
  for the 7 classes x KB 4 / 5, synthetic random weights / A / svh, one M = 256 launch (the plan
  `ApplyLinear` uses) against four launches of the shipped row (`PickTuning` at M = 64), one-A and (gate_up)
  two-part, 0 of up to 8.9 M bf16 differ, 3 repeats each identical, tickets reset, a negative control (a
  shipped kernel at another SK differs by 162-13485 bf16, all > 0), and the refusal messages
  (`M = 64`, KB 5 SK 16, an uncovered tail). 0.9 s.
* `tests/model/test_prefill_chunk` (CPU): the `R4DX_PREFILL_CHUNK` parser and the decision table.

### Not done / caveats of the spike

* All identity checks are at TP = 1 on real text; the M = 256 kernel's own bit tests use random or Gaussian
  activations (S1) and M is always 256 at the kernel; tails use today's 64-row kernels, never a 128-row one.
* The 64k / 128k pairs are single pairs; the speedup keeps falling with context (1.247x, 1.183x): attention and
  the GDN sequence ops are not touched.
* `PlanTrellisM256` refuses (and ApplyLinear then slices as today) a KB 5 SK 16 row, an odd k-tile tail, and any
  shape outside K, N multiples of 128; no shape of this model hits that.
* MTP, DFlash, TP = 2, vision and the server's other prefill entry points are not wired (out of scope): they fall
  back to 64 rows.
* `git`: `ctest -LE tp2gpu` skipped 20 tests for gitignored goldens; the trellis row-identity / golden tests
  (`test_trellis_gemm`, `test_trellis_input`, `test_trellis_decode`) among them, so the shipped M = 64 rows were
  checked here only through `test_trellis_m256` and the KL runs.

## What it is

One launch computes M = 256 rows (or 128) of the trellis linear with the bits the shipped M = 64 kernel gives
each 64-row slice at the same (SK, SKG). Every accumulator keeps its k-ordered chain of 16-k WMMA; the SK
slices are summed in slice order from 0.f, the SKG partials in y order from 0.f, then the stock FWHT stages,
svh, out_scale and one bf16 rounding. Row tiles are independent in WMMA, so nothing else can differ.

* Decode shared through LDS (the "M8" idea, docs/trellis-kernel.md 4.7). A workgroup is RG x SKW waves
  (RG = M / 64, SKW = K slices resident, at most 32 waves). Wave (rg, ks) owns rows 64 rg.. and K slice ks of
  the current group with the stock MT = 4 accumulator tile. Per step of U k-tiles the RG waves of one K slice
  each decode ONE of the slice's NP x U (tile pair, k-tile) blocks, store the two f16 fragments (ds_store_b128)
  to a double-buffered LDS slot, one workgroup barrier, then every wave reads all of them (ds_load_b128) and
  runs NP U 2 4 WMMA. Decode and weight loads are amortised over 256 rows. Shipped configuration: RG 4, NP 1,
  U 4 (Wc = 32 columns per block), 172 VGPRs at KB 4 and 184 at KB 5 (183-192 for the phased ones), no
  scratch in any configuration the model's shapes use, LDS 32 KB staging / 24-48 KB reduction.
* SK 16 and every other SK > SKW: PH = SK / SKW groups of SKW slices are walked one after the other by the same
  workgroup (`skw` argument; default min(SK, 4)). Slice = group * SKW + ks keeps its own k-tile range, its chain
  of 16-k WMMA restarts from zero, so each slice partial is exactly the shipped one. The shipped SK sum is a
  LEFT FOLD from 0.f in slice order (m64 kernel: `v = 0.f; for s < SK: v += red[s]`), so it can be cut into
  groups: at the end of group g the owner of tile i (slice i % SKW of the group) sums the group's SKW partials
  in slice order starting from the running sum of groups 0..g-1 instead of 0.f (group 0 starts from 0.f, as the
  shipped kernel), and parks the result in its own ws unit (the unit the final sum lands in; same lanes read it
  back, a workgroup fence and barrier between). The added terms and their order are the shipped ones, so the
  bits are. Cost: PH x (LDS reduction, barrier, one ws round trip), no extra accumulators. The phase loop is
  fully unrolled (a rolled loop spilled 28-80 B/lane: the compiler hoists the lane offsets out of it).
* A slice's k-tile count need not be a multiple of U: TAIL = ktw % U trailing k-tiles run as one short step in
  which only the waves whose block index is below TAIL decode (the last full step preloads exactly those
  waves' tail blocks; the others re-read their current block, so nothing is read outside the slice). mlp.down
  KB 4 is 34 k-tiles per slice (SK 16, SKG 2, K = 17408): 8 steps of 4 + a tail of 2. TAIL is a template
  parameter (0 everywhere; 2 instantiated for (SKW, PH) = (4, 4) and (8, 2)); another value fails with a clear
  message. k-tile order per accumulator is unchanged, hence bit identity.
* Epilogue in the accumulator layout (all lane-contiguous b128, no transposes):
  (1) group reduction: tile i is owned by slice i % SKW; the other slices store their acc[.][.][i] to LDS, the
  owner sums in slice order taking its own term from registers; (2) the owner writes the sum to ws in lane
  order and, after the last group, takes a ticket (the stock scheme: the last of the SKG x 128/Wc contributors
  finishes, tickets reset themselves, no inter-block wait); (3) the last block y-sums the ws units, runs all
  seven FWHT stages in registers (lane bits 0-2, the f pair, lane bit 3, then the block index), scales and
  stores bf16.
* ws is SKG M N x 4 B (gate_up 35.6 MB, down 10.5 MB at M = 256); tickets N/128 words, as the stock kernel.

## Bit identity against the SHIPPED rows (MEASURED, S1: logs under D:\models\r4dx\linear\S1\logs)

The comparison target is the shipped M = 64 tuning-table row of each (class, KB), read from
`src/model/gemm_tuning_table_trellis.inc` (M = 64 band): mlp.gate_up SK 4; mlp.down SK 16 (KB 4) / SK 4
(KB 5), both SKG 2; qkv SK 2; z SK 4 (KB 4) / 8 (KB 5); out and attn.o SK 4 / 8; qg SK 4; attn.k, attn.v SK 8.

* Random weights (`tool_trellis_m256_bench --mode verify --seeds 6`, verify_all_s6_final.log): C[256][N] bf16
  of one launch equals, byte for byte, four launches of the shipped kernel against the shipped row AND every
  other legal stock tuning of the same (SK, SKG) (12-25 of them per shape), for all seven classes x KB 4/5,
  EVERY (SKW, SK / SKW) the kernel has for that SK (SK 16: W4 x4, W8 x2; SK 8: W2 x4, W4 x2, W8 x1; SK 4:
  W2 x2, W4 x1; SK 2: W2), 6 seeds, one activation matrix and the n_split pass (two activation matrices,
  columns split N/2): 360 of 360 comparisons SHIPPED ROW IDENTICAL, 0 differing bytes, tickets reset every
  time. Each M = 256 launch is repeated 8 more times and must give the same bytes (a race in the running-sum
  handoff would show). M = 128 against two launches (4 seeds, `--m128`, verify_m128.log): 176 of 176 identical;
  SK 16 has no M = 128 configuration (skipped, printed).
* Real container weights (`tool_trellis_m256_real_check --dir D:\models\r4dx\linear\A2verify\real`, the
  verifier's extracted slices of the Huihui mix4.5m container, 9 tensor classes x KB 4/5 = 18 files, N(0,1)
  f16 activations, 3 seeds, one-A and two-part, each launch repeated 12 times; real_check_final.log): 246
  M = 256 launches, 0 of N x 256 bf16 differ from four shipped-row launches, 0 run-to-run differences,
  tickets reset. mlp.down KB 4 (the row that was 0.17% different at SK 4) is identical at both SK16
  configurations; a 300-repeat stress of mlp.down on real weights (real_check_down_repeat300.log): 0
  run-to-run diffs for W4 x4 and W8 x2 (KB 4) and W2 x2 / W4 x1 (KB 5), both forms.
* Negative controls: the same comparison against a stock tuning of a different SK finds 96-13626 differing
  bf16 per shape (random weights 206-13345, real weights e.g. SK 16 vs SK 4: 2236 = 0.17%), so the
  comparison does detect an accumulation-order change (every file printed a control, all > 0).
* Coverage limit: activations are random / Gaussian, not the model's; M a multiple of 64; n_split cases test
  the two-matrix path with random second matrices.

## Speed (MEASURED, device 1, time_final_run1/2.log, rounds 15, batch 6 weight copies, interleaved)

Speedup of one M = 256 launch against 4 x the SHIPPED M = 64 row on cold weights (each launch on a copy no
launch of the batch has read), median of per-round ratios; run 1 / run 2. Best (SKW x PH) per class, all of
them bit-identical to the shipped row.

| class (layers) | KB 4 | KB 5 |
|---|---|---|
| mlp.gate_up (64) | 1.555 / 1.575 (W2 x2; W4 x1 1.52 / 1.50) | 1.554 / 1.558 (W4 x1) |
| mlp.down (64) | 1.298 / 1.311 (SK 16, W4 x4; W8 x2 1.20 / 1.21) | 1.716 / 1.714 (SK 4, W4 x1) |
| gdn.in_proj_qkv (48) | 1.562 / 1.566 (SK 2) | 1.628 / 1.635 (SK 2) |
| gdn.in_proj_z (48) | 1.589 / 1.562 (SK 4, W4 x1) | 1.636 / 1.672 (SK 8, W4 x2; W8 x1 1.54 / 1.57) |
| gdn.out_proj, attn.o (64) | 1.632 / 1.684 (SK 4, W4 x1) | 1.812 / 1.845 (SK 8, W4 x2; W8 x1 1.73 / 1.71) |
| attn.qg (16) | 1.466 / 1.457 (SK 4, W4 x1) | 1.560 / 1.578 (SK 4, W4 x1) |
| attn.k, attn.v (32) | 1.91 / 1.92 (SK 8, W4 x2; W8 x1 1.79 / 1.94) | 2.18 / 2.07 (SK 8, W8 x1; W4 x2 2.15 / 2.07) |
| **MLP pair** | **1.457 / 1.477** | **1.608 / 1.600** |
| all seven classes, layer-weighted | 1.500 / 1.508 | 1.633 / 1.634 |

Model-layer sums (DERIVED from the bench, excludes in_proj_a/b and the transforms): seven classes cost 0.717 /
0.735 (KB 4) and 0.780 / 0.788 (KB 5) ms/token at the shipped M = 64 rows, and 0.476 / 0.488 (KB 4), 0.477 /
0.484 (KB 5) ms/token with the M = 256 kernel, i.e. 0.241 / 0.247 (KB 4) and 0.302 / 0.304 (KB 5) ms/token
saved. Run-to-run spread of one shape is about +-3%; per-round MLP pair ratios span 1.29-1.55 (KB 4).

What the exact SK 16 cost: the earlier prototype's non-identical SK 4 down KB 4 (485 us in the verifier's
protocol) is replaced by SK 16 in W4 x 4 (about 516-524 us here): the KB 4 down class drops from about 1.45 to
1.30 vs the shipped row, the KB 4 MLP pair from 1.55 to 1.46-1.48. Where it goes: 4 groups each pay a tail step
(34 k-tiles = 8 x 4 + 2), a fresh weight-load pipeline, a 4-pass LDS reduction with 8 barriers and a ws round
trip. W8 x 2 (32-wave workgroups) is 8% slower still.

Also measured, not on the shipped rows: the earlier M = 128 kernel (RG 2) pair 1.13, unchanged.

Regression note (MEASURED, 3 interleaved ABAB runs, gate_up KB 4): the W4 x 1 instantiation (16-wave
workgroup, the old prototype's SK 4 configuration) is about 4% slower than the previous prototype's kernel
(860-900 us vs 825-860) after the phase/tail restructuring; W2 x 2 (8-wave workgroups, two groups) equals the
old kernel (827-848 us) and is the best gate_up KB 4 pick above. Other classes' W4 x 1 timings match the old
prototype within noise (z KB 4: 148-153 vs 149 us). Cause not found (172 vs 174 VGPRs, same LDS, no scratch).
## Where the time goes (MEASURED with ablation builds, gate_up KB 4)

Loop alone 721 us, whole kernel 790 us: epilogue 69 us = SK reduction 14 + ws stores and ticket 15 + the last
block's finish 38. The loop's floor at 2.59 GHz is WMMA 552 + decode 68 = 620 us, so it runs at 86%. Removing
the barrier or the LDS reads changes nothing (the WMMA-only skeleton is 743 us); the non-temporal weight
loads change nothing. (Replacing decode or the loads by VALU stand-ins does not lower the time either, but
that ablation is not clean: the stand-ins cost about the VALU they replace.)

## What was tried and dropped

* (RG 4, NP 2, U 2): 0-15% slower than NP 1, U 4 on every class and it spills (24 B/lane at KB 4, 52-260 at
  KB 5). Removed.
* M = 128: above. 32-wave workgroups (SKW 8): slower than SKW 4 with two groups (z KB 5 SK 8: 179 vs 167 us,
  mlp.down KB 4 SK 16: 562 vs 516 us).
* Rolled phase loop: spilled 28-80 B/lane (hoisted lane offsets); an sm volatile fence on the lane value fixed
  most of it but cost ~3% on the PH = 1 instantiations, so the loop is unrolled. A runtime tail (instead of the
  TAIL template parameter) cost +7 VGPRs and ~4-5% on every PH = 1 kernel.
* Non-temporal weight loads: within noise.

## Limits of the kernel (S1 wrote this as "for whoever wires it"; S2 wired it, see the top section)

* Instantiation limits (`r4d_t256_inst`, the single table the launch and `r4d_gemm_trellis_nt_m256_check` use):
  SKW 8 needs NP 1; PH > 1 exists for (RG 4, NP 1, U 4) and KB 4 only, so M = 128 has no SK 16 and KB 5 has no
  configuration that walks more than one group of slices; TAIL 2 only for (SKW, PH) = (4, 4) and (8, 2) (other
  k-tile counts per slice are refused with a message). A model with other K would need more instantiations.
  S1's instantiations that spilled or used 192 VGPRs (KB 4 W4 x 4 TAIL 0: 8 B/lane; every KB 5 PH > 1: 192
  VGPRs, and 12 / 16 B/lane at W4 x 4) were removed in S2 so the build check (<= 190 VGPRs, no scratch) covers
  the unit; the tables and the S1 report below that mention them describe what S1 measured.
* The shipped rows in this file are the M = 64 band of the table; the M = 32 / M = 1 rows and every
  M <= 16 decode / verify path stay on the shipped kernels (docs/trellis-kernel.md 538-543).
* The per-(class, KB) (SKW, PH) choice is `PlanTrellisM256` (linear.cpp), not a tuning-table column: the
  best pick per class from the S1 table, first instantiated one wins; W4 x 1 vs W2 x 2 differ by up to 10% on
  out / z KB 4 (z and out use W4 x 1, mlp.gate_up KB 4 W2 x 2).
* Tail chunks (fewer than 256 rows left) use today's 64-row kernels; the M = 128 kernel is not used (S1: about
  1.13x, not worth a second path).
* The build's zero-scratch / 190-VGPR check (third_party/check_trellis_isa.cmake) covers this unit since S2:
  22 instantiations, max 190 VGPRs, no scratch (the S1 compile report,
  D:\models\r4dx\linear\S1\logs\resource.txt, lists the ones since removed).
* A-layout: fragment-contiguous activations (a prefill-only input-transform change, measured -5..-8% on the
  M = 64 kernel) were not tried here.
