# M = 256 trellis GEMM and the 256-row prefill chunk

Status (2026-09-30, branch `chunk-default`): **the 256-row chunk is the default** (`R4DX_PREFILL_CHUNK`
unset means 256; `0` or `64` is the kill switch that restores the 64-row engine exactly; see
[docs/prefill.md](prefill.md) "The 256-row prefill chunk" for the switch, the load line, the chunk grid and
the fallbacks). The sections below are the spike's record (stages S1 / S2, when the flag was opt-in and
TP = 1 only); where they say "ignored", "opt-in" or "not wired", the default-on work supersedes them.
The kernel (`third_party/libr4d/r4d_gemm_trellis_nt_m256.hip`) is in
`R4D_UNITS`, has its `r4d.h` / `r4d.hpp` entry, its registry row, a host legality check and
a ctest, and `ApplyLinear` calls it inside a Model's 256-row prefill super-chunk. The design is in "The
256-row prefill chunk" below, the kernel in "What it is". Since stage S1 the kernel reproduces EVERY shipped M = 64
tuning row (all seven classes x KB 4/5, including mlp.down KB 4's SK 16) bit for bit; the bench
(`tests/kernels/tool_trellis_m256_bench.hip`) and real-weight check (`tool_trellis_m256_real_check.hip`,
`build_m256_bench.ps1 [-Tool real_check]`, hipcc, no CMake target) are unchanged in purpose.

Whether the plan can take a faster (SK, SKG) than the shipped M = 64 rows give it, and why nothing was applied: see the last
section, "Retuning the plan" (2026-10-07).

## The 256-row prefill chunk (stage S2, `R4DX_PREFILL_CHUNK=256`)

Result (MEASURED, HIP device 1, TP = 1, Huihui mix4.5m, main dd38f8b + branch `linear`; logs under
`E:\models\r4dx\linear\S2\logs`, outputs under `E:\models\r4dx\linear\spike`):

* **Bit identity: all 12 KL segments (prose / code / recall x 8k / 32k / 64k / 128k) are byte-identical
  (sha256 of every `logprobs.f16`) to the dense dumps `E:\models\r4dx\pflash\baseline\kl-dense-huihui`**, mean /
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

(Spike wording. Since the default-on work (2026-09-30) the decision is made once at load,
`DecidePrefillChunk`: 256 rows unless `R4DX_PREFILL_CHUNK` is `0` / `64` or the container is quant2.
TP = 2, `--mtp`, `--dflash`, the feature capture, the per-chunk callback and image prompts all run
super-chunks now, bit-identical, as docs/prefill.md describes; only the mechanism of the table below is
unchanged.)

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
| gdn conv prep, kkt solve, chunk scan (+ the state commit), gated norm | one call each over 256 rows (`seq_slice = 0`, the default; chunk 64 inside the kernels); `R4DX_GDN_SLICE=64` restores one call per 64-row sub-slice | see "GDN sequence ops" below: the only state between chunks is the scan's fp32 state, exact whether it stays in registers or goes through the slot, and a chunk's conv halo is the same x shorts whether read from x or from the conv cache |
| attention core | once per 64-row sub-slice (`attn_slice = 64`), q_len 64, `seqused_k[j] = pos + 64 (j + 1)` from a per-slice device array | each is the exact-wide (or split / dense, per `R4DX_PREFILL_SPLITKV`) launch a 64-row chunk makes; the KV write covers all 256 rows first and the causal mask hides later sub-slices' keys from earlier ones |
| lm_head / logits | last row only, from `cur + (T - 1) * hidden` | as today |

### GDN sequence ops (branch gdn256, 2026-10-06; GPU validation and measurements PENDING)

Until this change a super-chunk ran the four GDN sequence ops as four 64-row sub-slices. At 8k they were
19.2% of `gpu_sum` under `--profile-prefill` (conv prep 7.3, kkt 4.6, scan 4.2, gated norm 3.1), and the
launch count did not shrink with M = 256. They now run once per super-chunk (`GdnLayerParams::seq_slice = 0`,
`cu = {0, 256}`), with no kernel change. The bytes are the same for these reasons (the libr4d units build
with `-ffp-contract=off` and no fast-math, so the same expression rounds the same way whatever produced its
operands):

* conv prep: each 64-token chunk is its own workgroup, and the gate cumsum is local to the chunk. A chunk
  with t0 > 0 reads its 3 halo tokens from x. A sliced call reads them from the conv cache, which holds the
  raw bf16 shorts of the same x rows. Only the t0 == 0 workgroup reads and rewrites the cache, from rows
  253..255, which are the bytes the last sub-slice wrote.
* kkt solve: one workgroup per (chunk, k head), with no state between chunks. `A_buf` is sized T x H x 64.
* chunk scan: the kernel loops over the chunks. Between chunks the fp32 state stays in registers. A sliced
  call stores it as fp32, copies it into the slot and reloads it as fp32, which is exact. The bf16 copy
  `SWRITE` derives from the state and the chunk body are the same code either way.
* gated norm: one wave per (token, head) row.

The launches per super-chunk drop from 16 + 4 memcpys to 4 + 1. kkt runs 64 live workgroups instead of 16.

conv prep was costly because it is latency-bound, not bandwidth-bound. A 64-row call is only 10 live
workgroups (80 waves on 64 CUs). Each wave walks all 64 tokens with one dependent 8-byte load per token. The
y == 0 workgroups also do all 48 heads' gating before their own conv. `r4d_gdn_conv_prep2_w4_h128_bf16`
(the same arguments, `core::r4d::GdnConvPrep2`) changes only who computes what:

* the conv is blocked by 16 tokens, and the block is narrowed until there are at least 256 conv workgroups;
* the gating runs on grid rows of its own, one wave per (64-token chunk, head);
* a block's 16 x rows are loaded before the first is used.

Every arithmetic line, operand order and lane mapping is copied from v1. That covers the conv sum, the bf16
round before the norm, the 4-dims-per-lane l2norm and its xor 16..1 butterfly, and the 2-tokens-per-lane gate
scan with `pre = p - (g0 + g1)`. It compiles to 98 VGPRs with no scratch. It is the default on a wide Model
(since the 2026-10-06 GPU validation below), where it serves every prefill call (the 64-row tail chunks too);
`R4DX_GDN_CONV=1` restores `r4d_gdn_conv_prep`. A 64-row Model always runs `r4d_gdn_conv_prep`.

Switches (read once per process, `prefill_chunk.h`, logged at load; both apply only to a wide Model):

| variable | default | other value |
|---|---|---|
| `R4DX_GDN_SLICE` | unset / `0` / `256`: one call per super-chunk | `64`: four 64-row sub-slices (anything else warns and uses 64) |
| `R4DX_GDN_CONV` | unset / `2`: `r4d_gdn_conv_prep2` | `1`: `r4d_gdn_conv_prep`, the original (anything else warns and uses 1) |

`R4DX_PREFILL_CHUNK=0` restores the whole pre-gdn256 64-row path, kernels included: a 64-row Model runs
one GDN call per 64-row chunk and `r4d_gdn_conv_prep`, whatever `R4DX_GDN_SLICE` / `R4DX_GDN_CONV` say
(`DecideGdnConv`).

Not changed, because the bits would move:
* the chunk size of 64 and the per-chunk cumsum;
* kkt's fp32 diagonal substitution (WMMA takes bf16 operands);
* the truncating bf16 packs;
* the order of any reduction;
* fusing the gated norm into the scan (a workgroup owns only half of each row);
* `in_proj_a` / `in_proj_b`, which stay 64-row launches.

kkt and the scan already run on WMMA. conv prep and the gated norm have no matrix product to put on it.
The one matmul-shaped scalar loop left is kkt's 16 x 16 fp32 forward substitution; a WMMA version would
take bf16 operands and so would not be bit-identical (it would need its own flag and the KL gate). It is not
in this branch.

Validation (`tools/prefill/gdn256_check.ps1`, HIP device 1, commit 9e333f5, 2026-10-06; at that commit
prep2 was the opt-in `R4DX_GDN_CONV=2`, so "prep2" below is today's default). ALL PASS:
* `tests/model/test_gdn_seq256_identity`: 33 / 33 byte-identical (one call against 4 x 64, conv v1 against
  v2; synthetic at the real shape, fresh and `has_init` calls of 256, 320, 200, 17, 2 and 64 rows, plus a
  random initial state). Part B (the 4-layer container) and `test_gdn_layer` skipped: data not on disk.
* `test_gdn_chunk_scan` PASS.
* `test_prefill_chunk_identity` (logits, greedy tokens and the full KV / GDN state digest of a 256-row
  Model against the true pre-gdn256 64-row Model) PASS three times: one call + `r4d_gdn_conv_prep`
  (340 s), one call + prep2 (325 s), `R4DX_GDN_SLICE=64` (319 s).
* Greedy 64-token text on the Huihui trellis mix4.5m container: the same SHA-256 across the four paths
  below, at 8k (8089 prompt tokens) and 32k (32734).

TTFT (unprofiled `[stats]`, one run each, same session):

| path | 8k | 32k |
|---|--:|--:|
| `R4DX_PREFILL_CHUNK=0` (64-row chunks) | 6.837 s (1183 tok/s) | 32.230 s |
| `R4DX_GDN_SLICE=64` (pre-gdn256 256-row path) | 5.002 s (1617 tok/s) | 24.550 s |
| one GDN call per super-chunk, `r4d_gdn_conv_prep` | 4.765 s (1698 tok/s) | 23.463 s |
| **one call + prep2 (default)** | **4.722 s (1713 tok/s), -5.6%** | **23.405 s, -4.7%** |

`--profile-prefill` at 8k, GPU ms for the four GDN sequence ops (span overhead included):

| op | sub-slices (old) | one call, prep | one call, prep2 |
|---|--:|--:|--:|
| conv prep | 384.6 | 82.8 | 3.3 |
| kkt solve | 20.2 | 4.3 | 2.3 |
| chunk scan | 86.8 | 66.4 | 45.0 |
| gated norm | 50.4 | 5.8 | 5.5 |
| total (% of gpu_sum) | 542 (11.1%) | 159 (3.5%) | 56 (1.2%) |

The GDN sequence ops are no longer a prefill target; what is left is the GEMMs and the attention core.

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
  back to 64 rows. (Superseded 2026-09-30: all of them run the 256-row chunk now, docs/prefill.md.)
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

## Bit identity against the SHIPPED rows (MEASURED, S1: logs under E:\models\r4dx\linear\S1\logs)

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
* Real container weights (`tool_trellis_m256_real_check --dir E:\models\r4dx\linear\A2verify\real`, the
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
  1.13x, not worth a second path; it saves well under 1% of a prompt, and it has no SK 16 configuration for
  mlp.down KB 4). The tail lengths and prefix-reuse suffixes are bit-tested by `test_prefill_chunk_identity`.
* The build's zero-scratch / 190-VGPR check (third_party/check_trellis_isa.cmake) covers this unit since S2:
  22 instantiations, max 190 VGPRs, no scratch (the S1 compile report,
  E:\models\r4dx\linear\S1\logs\resource.txt, lists the ones since removed).
* A-layout: fragment-contiguous activations (a prefill-only input-transform change, measured -5..-8% on the
  M = 64 kernel) were not tried here.

## Retuning the plan: what is free and what is not (2026-10-07, branch `f16retune`)

Why: the int8 prototype bench (`E:\models\r4dx\int8gemm\all.log` / `all.json`, docs/int8-gemm-proto.md) swept every
legal (SK, SKG, SKW) of this unit and found, per class, a configuration 0-15% faster than the shipped plan. The
question is whether the plan can take any of them without moving a bit of the "256-row chunk == 64-row chunk"
identity. The CPU analysis below found no candidate free from existing data; the GPU sweep then decided it.

**Result (MEASURED, ROCm 10.1.0, HIP device 1, idle machine).** `--retune --mode verify` on device 0: every M = 256
(SK, SKG, SKW) byte-identical to the stock M = 64 tunings of the same (SK, SKG), 0 failing comparisons. Two
`--retune --mode time` runs (`E:\models\r4dx\f16retune\retune1.log`, `retune2.log`); a pick was accepted only if it
is FREE in both:

| class | change (M = 64 row, and so the M = 256 plan) | M = 256 run 1 / 2 | M = 64 run 1 / 2 | taken |
|---|---|--:|--:|---|
| mlp.down KB 4 | `{1, 16, 4, 2, 0, 2, 1}` (SK 16 SKG 2) -> `{1, 4, 4, 2, 0, 1, 1}` (SK 4 SKG 1) | x1.375 / x1.412 | x1.008 / x1.062 | yes |
| in_proj_qkv KB 5 | SK 2 SKG 1 -> SK 4 SKG 1 | x1.017 / x1.028 | x1.018 / x1.008 | no (within noise) |
| z, out, down KB 5 | FREE in one run only, different families | - | - | no |

End to end with the down KB 4 row (`tools\prefill\gdn256_check.ps1 -SkipIdentity`, device 1, single runs; logs
`E:\models\r4dx\f16retune\`): cold TTFT 8k 4.836 -> **4.683 s** (1727 tok/s), 32k 23.456 -> **22.954 s** (1426 tok/s)
against the same-toolchain main (the 4.836 s baseline ran with a CPU build in parallel, so the 8k gain may be
slightly overstated). Greedy text hashes unchanged (8k `9A0B80DE8E49B800`, 32k `CDD3FDD6609B99DF`) and identical
across new / conv1 / old / c64. `test_prefill_chunk_identity` PASS (15 configurations, 64-row == 256-row bit for
bit), `test_trellis_m256`, `test_pick_tuning`, `test_prefill_chunk` PASS, `kl_rung4` byte-identical to
`E:\models\r4dx\rocm1010\kl` (decode path untouched). The prefill bits of mlp.down KB 4 layers change, so prefill
dumps recorded before this commit (dense-KL baselines) need re-recording.

### What defines the order of a sum (read from the two kernels, not assumed)

Both kernels give the wave (y, slice) the K range `kt0 = (y * SK + slice) * ktw`, `ktw = K / 16 / (SK * SKG)`,
accumulate it as one k-ordered chain of 16-k WMMA, sum the SK slices of a group in slice order from `0.f`
(`v = 0.f; for s < SK: v += red[s]`, m64 line 400; the same running sum in m256 section 1), then sum the SKG groups
in y order from `0.f` (m64 lines 436 / 449, m256 line 357), then FWHT, svh, scale, one bf16 rounding.

| parameter | what it does | changes a per-element sum? |
|---|---|---|
| SK | K is cut into SK x SKG contiguous pieces; the SK pieces of a group are folded left to right | **yes** |
| SKG | the groups' partials are folded left to right | **yes** |
| SKW (M = 256 only) | slices resident per workgroup, PH = SK / SKW groups walked in turn; the fold keeps running through `ws` | no, scheduling only (360 / 360 identical comparisons in S1, every (SKW, PH) of every shipped SK) |
| WV, NP, MT, U, NT, Wc (M = 64) | columns per block, tile pairs, row tiles, k unroll, cache hint | no (test_trellis_gemm / the bench compare 12-25 stock tunings per (SK, SKG), byte for byte) |
| RG, NP, U of the M = 256 unit | fixed by the instantiation table (RG 4, NP 1, U 4) | no |
| M, the row tile a row falls in | WMMA row tiles are independent | no |

So the order is a function of (SK, SKG) alone. Two pairs give the same bytes only if their fold trees are equal: that
holds for (P, 1) and (1, P) (a fold of P pieces either way) and for a pair with itself; (2, 2) is
`(p0 + p1) + (p2 + p3)` and (4, 1) is `((p0 + p1) + p2) + p3`, which differ. Hence the M = 256 plan can only be
identical to the 64-row path at the M = 64 row's own (SK, SKG), and `PlanTrellisM256` reads them from that row for
exactly this reason. Changing a class's M = 256 (SK, SKG) means changing the class's M = 64 row with it. (The one
exception is the alias: an M = 64 row at (1, P) gives the bytes of an M = 256 plan at (P, 1), SK 1 being legal for the
64-row kernel. It never helps: in the M5 screen the best (1, P) tuning is 6% to 4.4x slower than the best (P, 1) one
on every shape and rate except down KB 5 at P = 2 (192.2 against 194.3 us, and (2, 1) is not a candidate there), so the
sweep below does not time it.)

What a retune does and does not touch: M <= 16 (decode, verify, MTP / DFlash windows) takes the M = 1 band and M
17..32 the M = 32 band, neither is a candidate and neither moves; `kl_rung4` runs the decode path and stays
byte-identical. The M = 64 band serves the 256-row path (through `PlanTrellisM256`), every 33..64-row tail,
`R4DX_PREFILL_CHUNK=0`, quant2 and image prompts. A retune changes the bits of that class's PREFILL output for every
one of them: 64 == 256 still holds, but every prefill dump recorded before it (the dense-KL byte-identity of
"Bit identity" above, the prefill-m0 / prefill-m1 baselines, any text hash of a prefill) stops matching and has to be
re-recorded. That is a cost of the change, separate from its speed. The TP = 2 rows (`gemm_tuning_table_trellis_tp2.inc`)
and the Gemma shapes are separate rows that this sweep does not cover.

### Classification of the 14 (class, rate) results (DERIVED: all.log lines "f16 shipped plan" / "f16 alt ... best f16 config")

The M = 64 columns come from the M5 screen (`E:\models\r4dx\trellis-m5\ptune_k4.json`, `ptune_mix.json`: every legal
stock tuning per shape, flushed chains, random weights, ROCm before 10.1.0, one linear, run-to-run about +-3%) for
the best M = 64 tuning of the candidate's (SK, SKG) against the shipped row. "p*" is the share of rows that may run
through the 64-row kernel before the retune loses: `g / (g + 4 c)` for g us saved per 256 rows and c us lost per
64-row launch. "ranges" says whether the bench's own [min..max] of the two M = 256 timings overlap.

| class | KB | shipped (SK, SKG) | M256 us | best f16 config | M256 us | M64 row us | M64 us at the candidate | p* | ms saved per super-chunk (x layers) |
|---|---|---|---|---|---|---|---|---|---|
| gate_up | 4 | 4, 1 | 816.4 | 2, 1 (W2) | 811.3 (x1.006, ranges overlap) | 302.7 | 334.4 (+10.5%) | 4% | 0.33 |
| down | 4 | 16, 2 | 547.5 | 2, 2 (W2) | 406.4 (**x1.347**, separated) | 169.9 | 190.7 (+12.3%) | 63% | 9.03 |
| qkv | 4 | 2, 1 | 267.1 | 2, 2 (W2) | 255.7 (x1.045, overlap) | 92.3 | 110.2 (+19.4%) | 14% | 0.55 |
| z | 4 | 4, 1 | 162.2 | none: the shipped plan is the best f16 config | | | | | 0 |
| out, attn.o | 4 | 4, 1 | 175.0 | 2, 4 (W2) | 165.9 (x1.055, overlap) | 64.3 | 85.4 (+32.9%) | 10% | 0.58 |
| qg | 4 | 4, 1 | 309.0 | 2, 2 (W2) | 304.9 (x1.013, overlap) | 98.0 | 122.7 (+25.2%) | 4% | 0.07 |
| k, v | 4 | 8, 1 | 39.5 | 2, 2 (W2) | 38.1 (x1.037, overlap) | 15.5 | 29.4 (+90.4%) | 2% | 0.04 |
| gate_up | 5 | 4, 1 | 847.4 | 2, 1 (W2) | 834.8 (x1.015, overlap) | 315.4 | 355.1 (+12.6%) | 7% | 0.81 |
| down | 5 | 4, 2 | 427.9 | 4, 1 (W4) | 409.8 (x1.044, overlap) | 173.1 | 178.1 (+2.9%) | 47% | 1.16 |
| qkv | 5 | 2, 1 | 272.4 | 4, 1 (W4) | 259.3 (x1.051, separated) | 99.8 | 102.6 (+2.7%) | 55% | 0.63 |
| z | 5 | 8, 1 | 186.3 | 2, 2 (W2) | 162.7 (**x1.145**, separated) | 62.9 | 72.8 (+15.8%) | 37% | 1.13 |
| out, attn.o | 5 | 8, 1 | 190.6 | 2, 4 (W2) | 170.6 (x1.117, overlap) | 66.9 | 92.9 (+38.7%) | 16% | 1.28 |
| qg | 5 | 4, 1 | 312.7 | 2, 2 (W2) | 312.2 (x1.004, overlap) | 102.1 | 127.9 (+25.3%) | 0% | 0.01 |
| k, v | 5 | 8, 1 | 39.4 | 4, 1 (W4) | 37.7 (x1.045, overlap) | 17.4 | 17.4 (+0.1%) | 97% | 0.05 |

* **(A) identity-preserving with today's M = 64 rows (only SKW differs): none in the data.** `all.log` prints the best
  configuration of the whole sweep per class; every one of them has another (SK, SKG) than the shipped row (the gate_up
  KB 4 and qg KB 5 winners are inside the noise anyway), so a same-(SK, SKG) win at another SKW would only show if it
  were the global best, and none is. The shipped SKW per class is S1's measured pick, and several alternatives do not
  exist (KB 5 has no configuration that walks more than one group of slices, so SK 8 is W8 x 1 only). The sweep tool
  prints an "(A)" line for any class where the plan's SKW loses to another SKW of the same (SK, SKG), so this is
  decided on the device, not by this table.
* **(B) needs the M = 64 row changed to the same (SK, SKG): all 13 candidates.** Every one is runnable by the M = 64
  kernel (all are in the M5 screen). The M = 256 gain comes from fewer slices per workgroup and more groups (SK 2,
  SKG 2-4: more, smaller workgroups for the 256-row tile), while the M = 64 rows were chosen by the M5 sweep for the
  64-row kernel and are the best of their class there in 12 of 14 cases (the other two, out KB 4 and down KB 4, are within 1.4%).
* **(C) cannot keep identity: none.** (SK 16 at KB 5 has no M = 256 instantiation, but no candidate uses it.)

What the numbers say:

1. **The headline is mostly noise and one class.** 10 of the 13 candidates have M = 256 time ranges that overlap the
   shipped plan's, and the bench reports the MINIMUM over about 25 configurations per class (a winner's curse of a few
   per cent against a +-3% run-to-run spread, docs "Speed" above). Outside the noise: down KB 4 (x1.347), z KB 5
   (x1.145), qkv KB 5 (x1.051). Layer-weighted, all classes at one rate, one super-chunk: KB 4 saves 10.60 ms of 125.3
   (8.5%), of which down KB 4 is 9.03 ms; KB 5 saves 5.15 ms of 122.3 (4.2%; all.log "ALL SEVEN LINEAR CLASSES"). On 32
   super-chunks (8k) that is at most 160-340 ms of 4.836 s, if every winner were real and every one were taken.
2. **The M = 64 side is not free.** Taking the best-f16 family of every class makes the 64-row chunk slower by
   +14.7% (KB 4) / +12.2% (KB 5) of the linears' 43.6 / 45.6 ms per 64 rows, i.e. `R4DX_PREFILL_CHUNK=0` (the
   kill switch and the pre-change reference) would lose about 0.7-0.8 s at 8k, and every tail chunk and quant2 / image
   prompt pays too. Of the 13 candidates only k, v at KB 5 (SK 4: +0.1%) is not slower at M = 64, and its M = 256
   gain (1.7 us per layer, ranges overlapping) is worth about 0.04% of a prompt: not worth the prefill re-baseline.
   qkv KB 5 and down KB 5 (SK 4, SKG 1) cost +2.7% / +2.9% at M = 64 for x1.051 / x1.044 at M = 256.
3. **The one lead that is cheap on both sides is down KB 4.** Its shipped row is SK 16, SKG 2, which M = 256 runs as
   W4 x 4 with a 34-k-tile tail (the slowest configuration of the unit, 547 us). At M = 64 the screen puts SK 4, SKG 2
   (WV 2 NP 2 SK 4 SKG 2 U 1 MT 4 NT 0, 169.0 us) at x1.005 of the shipped row (169.9 us), and SK 8 / 16 with SKG 2 within
   1%; M5's own whole-chunk stage had chosen SK 16 over SK 4 by 0.10 ms per 64-row chunk of 45 ms (0.2%), a tie. S1's
   old non-identical SK 4 down KB 4 ran at 485 us against the SK 16 plan's 516-524 (x1.07 at M = 256). Nobody has
   measured M = 256 at SK 4 / SK 8 with SKG 2 on ROCm 10.1.0, which is the number that decides it; (2, 2) is faster
   at M = 256 (x1.347) but 12% slower at M = 64 (a TRADE, p* 63%).
4. Nothing above is a measurement on this ROCm: the M = 64 columns are the M5 screen (older ROCm), and the M = 256
   columns are one run with +-3% noise.

Decision: leave the table as it is (the retune cannot be shown not slower at M = 64, and the M = 256 gains outside
the noise are two classes), and ship the instrument that decides it.

### The sweep (`tests/kernels/tool_trellis_m256_bench.hip --retune`, built by `build_m256_bench.ps1`)

* `--retune --mode verify`: every legal M = 256 (SK, SKG, SKW) of every family (SK 2 / 4 / 8 / 16 x SKG 1 / 2 / 4)
  against stock M = 64 tunings of the SAME (SK, SKG) (WV x NP at U 1, NT 0, MT 4, plus the shipped row), byte for
  byte, one-A and `n_split`, `--seeds` seeds, 8 repeats each, tickets reset, and the negative control. S1 verified the
  shipped rows' (SK, SKG) only; this is the identity a retuned row would rest on, for the families S1 never ran
  (SKG 4, SK 2 / SKG 2 ...). Correctness, so any device.
* `--retune --mode time`: per class and rate, M = 256 for every family at the plan's SKW (the rule of
  `PlanTrellisM256`, copied as `PlanSkw`) and at the best SKW, M = 64 for every legal stock tuning (WV x NP x U x NT x
  MT in {4, 2}) screened in 3 short rounds and the best three of each family timed in the full protocol (cold weight
  copies, `--rounds 11 --batch 6`, every round times every job once, medians of per-round ratios). Per family a
  verdict: FREE (M = 256 >= `--min-gain` % faster, default 1, and M = 64 no more than `--tol64` % slower, default 1),
  or TRADE with its p*. For each class it prints the FREE pick's `.inc` row, in the table's format, ready to replace
  the class's M = 64 row, the fastest-at-M = 256 family's row, an "(A)" line when a different SKW of the shipped family
  wins, and a layer-weighted summary per rate (super-chunk and 64-row chunk, FREE picks only and fastest picks). Idle
  device only.
* A row taken from the output must keep the M = 64 band legal for `BestRow` (a split row, SKG > 1 or Wc < 128,
  needs MT 4 for a 64-row chunk; the tool never prints one that does not), and it needs no change to `PlanTrellisM256`
  (the SKW rule picks the best SKW of every candidate's family in the table above, the log's winners are W2 for SK 2 and W4 for SK 4; the tool prints the plan's SKW and the best SKW side by side, so a gap shows).
* Applying a row, in order: paste the row(s) into `src/model/gemm_tuning_table_trellis.inc`; `test_pick_tuning` (reads
  the table itself, no expectation to edit: its check count does not change) and `test_prefill_chunk` (the decision
  table, not the tuning) must still pass; `test_trellis_m256` (plan vs four launches of the new row, with its
  negative control); `test_prefill_chunk_identity` (256 vs 64, the true pre-change path, real container); the KL pair
  of `tools/prefill/run_kl.ps1` (default vs `R4DX_PREFILL_CHUNK=0`: byte-identical `logprobs.f16`, both new; the
  default `-Tokens` file `prefill-m0\kl\tokens_long.json` no longer exists on disk: write one first with
  `python tools\prefill\make_kl_tokens.py --lengths 8k --out <dir>\tokens_8k.json` (CPU, seconds, segments prose_8k /
  code_8k / recall_8k) and pass it as `-Tokens`);
  `tools/quant2/kl_rung4.ps1 -CompareDir E:\models\r4dx\rocm1010\kl` byte-identical (decode); then the cold TTFT of
  `tools/prefill/gdn256_check.ps1` against 8k 4.836 s / 32k 23.456 s. The prefill dumps recorded before the change are
  then history; re-record the KL baselines that matter.
