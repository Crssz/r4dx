# Prefill: long-prompt TTFT

**Goal:** cut cold time-to-first-token for long prompts (8k to 128k) on Qwen3.8-27B. The model is a
hybrid of 48 GDN and 16 full-attention layers, with fp8 paged KV, head 256 and gqa 6. The M0 and M1
work below ran through the base `qwen38-27b-trellis-mix45m.r4dx` on 2x R9700; since 2026-09-29 the
default of `tools/prefill/*.ps1` is the Huihui abliterated `huihui-qwen38-27b-abl-trellis-mix45m.r4dx`
(same recipe and shapes, so the M1 conclusions -- exact-wide is bit-identical to the dense kernel --
are properties of the kernels and carry over; the base container and the M0 dense KL/TTFT dumps
under `E:\models\r4dx\prefill-m0` are historical: a lossy M2 mode would need its dense baseline
re-taken on the Huihui container, `docs/huihui.md` "Frozen values").

- **M0:** measure. A long-context eval kit, a profile of where prefill time goes, and dense accuracy
  baselines.
- **M1:** lossless attention parallelism (below). The default prompt-prefill attention is the
  exact-wide launch (bit-identical to the dense kernel); split-KV is opt-in with
  `R4DX_PREFILL_SPLITKV=split`. See [M1 final](#m1-final-exact-by-default-split-kv-opt-in).
- **M2:** opt-in lossy modes, gated on the M0 KL harness.
- **256-row chunk:** lossless, default on since 2026-09-30; `R4DX_PREFILL_CHUNK=0` is the kill switch. See
  [The 256-row prefill chunk](#the-256-row-prefill-chunk-default-on-r4dx_prefill_chunk).

The kit and its commands are in [`tools/prefill/README.md`](../tools/prefill/README.md). Raw outputs
are in `E:\models\r4dx\prefill-m0\` (`profile\results.json`, `baseline\`) and are never committed.

## M0: baseline and profile (dense prefill, 64-row chunks)

### Cold TTFT

`ttft_cli.ps1` measures prefill seconds. Each run is a fresh `r4dx-cli` process.

| length | tokens | TP=1 s (2 runs) | TP=1 tok/s | TP=2 s | TP=2 tok/s | TP=2 speedup |
|---|---|---|---|---|---|---|
| 8k | 8145 | 7.23 / 7.28 | 1123 | - | - | - |
| 32k | 32623 | 38.41 / 38.56 | 848 | 33.10 / 33.58 | 979 | 1.15x |
| 64k | 65529 | 103.03 / 102.94 | 636 | - | - | - |
| 128k | 130884 | 309.21 / 309.05 | 423 | 293.36 / 316.13 | 429 | 1.01x |

- **TP=1:** HIP device 1, binary built at 60e6fae. K4m gives 37.6 s at 32k and 304.3 s at 128k.
- **TP=2:** `--tp 2 --tp-mode real` on both GPUs, with the dense pre-M1 binary. The `git` field of
  that jsonl is only the worktree HEAD label, which says 4b04ece. The 128k run 2 was lost to the
  power-off below and rerun after M1 (`profile\ttft-dense-tp2\resume-run2`, whose jsonl says run 1).
  It is 7.8% slower than run 1; the M1 final TP=2 table compares against the mean of the two.
- **Why TP=2 barely helps at depth:** a TP=2 rank has 2 KV heads, so its attention call launches 2
  workgroups. The call takes as long as at TP=1 (10.71 vs 10.52 ms at 123k), so only the linears
  get faster.
- **Warm turn at TP=1:** a ~4k-token user turn appended through server prefix reuse costs 3.5 s at
  offset 0, 6.5 s after 32k tokens and 9.8 s after 64k tokens.

### Per-chunk time vs depth

TP=1, from the probe timeline, in ms per 64-row chunk:

| depth | chunk | linears | GDN | attention core | other |
|---|---|---|---|---|---|
| 0 | 51.8 | 44.7 | 3.6 | 0.6 | 3.0 |
| 8192 | 63.7 | 44.6 | 3.4 | 12.8 | 2.9 |
| 32768 | 100.7 | 45.2 | 3.4 | 49.2 | 2.9 |
| 65536 | 152.9 | 46.5 | 3.4 | 100.0 | 2.9 |
| 122880 | 240.7 | 46.4 | 3.4 | 188.0 | 2.9 |

The fit is chunk = 51.0 ms + 1.552 ms per 1k tokens of depth, and attention accounts for 1.542 ms of
that slope.

Attention share of cold prefill, measured against the research fit (`pflash-report.md`):

| length | measured | fit |
|---|---|---|
| 8k | 11.5% | 9% |
| 32k | 32.2% | 30% |
| 64k | 48.8% | - |
| 128k | 65.6% | 63% |

### Attention kernel occupancy

The q_len 64 call has grid (1, 4, 1), 768 threads, 240 VGPRs (13 spilled) and 54016 B of LDS, so one
workgroup fits per WGP. Four workgroups on 32 WGPs use 1/8 of the device, and a TP=2 rank uses 1/16.

`tool_attn_prefill_bench` microbench, in ms per 64 query rows (split8 is the M0 emulated partial
pass):

| depth | q64 (today) | q256 | q512 | split8 |
|---|---|---|---|---|
| 8192 | 0.706 | 0.182 | 0.106 | 0.136 |
| 32768 | 2.713 | 0.696 | 0.429 | 0.463 |
| 65536 | 5.864 | 1.432 | 0.848 | 0.905 |
| 122880 | 10.867 | 2.699 | 1.586 | 1.600 |

### VRAM at 128k (max_ctx 132096)

- **TP=1, mix45m:** 20.84 GiB used. That is 16.27 GiB of weights and 4.31 GiB of KV and GDN state,
  leaving 11.03 GiB free. K4m uses 19.43 GiB.
- **TP=2:** 11.90 GiB per rank. That is 9.40 GiB of weights and 2.20 GiB of KV and state, leaving
  19.96 GiB free.

### Dense accuracy baselines

- **Task set:** 154 RULER-style items from 8k to 128k, on mix45m at TP=1 with the 60e6fae binaries
  (`baseline\tasks\mix45m\summary.json`). Score in %, items in parentheses:

  | task | 8k | 32k | 64k | 128k |
  |---|---|---|---|---|
  | niah_single | 100 (8) | 100 (8) | 100 (4) | 100 (2) |
  | niah_multikey | 100 (8) | 100 (8) | 100 (4) | 100 (2) |
  | niah_multivalue | 100 (8) | 100 (8) | 100 (4) | 100 (2) |
  | niah_multiquery | 100 (8) | 100 (8) | 100 (4) | 100 (2) |
  | vt | 97.5 (8) | 95.0 (8) | 30.0 (4) | 80.0 (2) |
  | cwe | 100 (8) | 100 (8) | 100 (4) | 100 (2) |
  | code_qa | 100 (8) | 100 (8) | 100 (4) | 50.0 (2) |
  | **mean** | 99.6 | 99.3 | 90.0 | 90.0 |
- **KL references:** prose, code and recall at 8k, 32k and 128k. Continuation perplexity:

  | length | prose | code | recall |
  |---|---|---|---|
  | 8k | 9.71 | 19.61 | 1.164 |
  | 32k | 3.915 | 1.837 | 1.310 |
  | 128k | 4.524 | 1.192 | 1.176 |

- **KL noise floor = 0:** a full dense rerun of the 32k segments (768 rows) gives mean, p99 and max
  KL of exactly 0 and 100% top-1 agreement, as did the 8k smoke rerun. Prefill is deterministic run
  to run, so any nonzero KL comes from the variant.

**Incident:** on 2026-09-28 at ~14:07 the PC powered off during TP=2 128k run 2. The likely cause
is PSU over-current with both GPUs ramping at once. Work continued on both GPUs unchanged.

**Next:** the profile points at attention occupancy. M1 below makes the attention call split-KV
(S=8 at TP=1, 16 per rank at TP=2).

## M1: split-KV prefill attention

Long-prompt prefill was bounded by one kernel. Prefill runs in 64-row chunks (`Model::max_chunk_`).
Each full-attention layer makes one `r4d_attn_prefill_h256_gqa6_fp8kv` call per chunk. That call
launches `ceil(q_len/64) x kv_heads` workgroups: 4 at TP=1 and 2 per rank at TP=2. The device holds
32 of these workgroups at once (768 threads and 54 KB of LDS each), so 1/8 of it was used. As a
result, attention grew from 0.6 ms per chunk at depth 0 to 188 ms at 128k (prefill M0,
`tools/prefill`).

M1 is lossless attention parallelism. Nothing else in the chunk changes.

### Design choice

| Option | What it changes | Verdict |
|---|---|---|
| (a) split-KV prefill attention | The KV tile range of one call is cut into S segments (grid x = q_blocks x S). Each segment writes fp32 partials, and a fixed-order merge follows. | **Chosen.** Only the attention call changes. Every invariant sized by `max_chunk_` stays as it is: activation buffers, `MtpHead::kMaxPrime`, DFlash feature buffers, GDN prewarm, the 96 MiB arena, `T <= 64` in `AttentionLayer`, `ApplyLinear`'s 64-row tiles, and the TP all-reduce channel sizes. Measured at the kernel: S=8 matches a 512-row call. |
| (b) group several chunks' attention into one call | Linears and GDN run in 64-row chunks, and attention runs over up to 512 rows at once. | Not possible without reordering layers. Layer L+1's input for chunk c needs layer L's attention output for chunk c. Batching attention across chunks means running every layer over all grouped chunks together, which is (c). |
| (c) raise `max_chunk_` to 256 or 512 | Everything. | The largest blast radius. Every buffer and invariant listed under (a) grows, and the skinny-GEMM, trellis and verify-window paths are all tuned for M <= 64. (a) gets the attention gain without any of that. |

### Mechanism (libr4d branch `prefill`)

- `r4d_attn_prefill_kernel` has a new `SPLIT` template parameter.
  - `SPLIT=0` is the old kernel: every split term folds away at compile time, and the plain entry
    points launch it exactly as before.
  - `SPLIT=1`: workgroup `(qb, sp)` runs tiles `[sp*tps, min((sp+1)*tps, ntiles))`, with
    `tps = ceil(tiles(ctx)/S)`. This is the decode kernel's partition. The workgroup writes its fp32
    accumulator, `m_ref` and row sum to the decode path's fp32 partial layout.
- `r4d_attn_splitkv_combine_kernel<256, 4, PF16=0>` (the decode merge, unchanged) combines the
  segments in fixed index order. Its per-row segment count `u` already handles the causal tail.
- New entry points:
  - `r4d_attn_prefill_splitkv_h256_gqa6_{fp8kv,bf16kv}` and `..._scratch_bytes`. `splits <= 1` is
    the plain launch: same kernel, same bits, no scratch.
  - A split launch without scratch returns -5.
  - The split count is chosen by the caller. The library has no split law.

### Split law (r4dx, `attention_layer.hpp`)

`PrefillSplitKvSplits(ctx, q_len, kv_heads)` is a pure function of the call's shape, so TP ranks,
reruns and prefix-reuse replays all agree:

- **`ctx < 8192`:** 1, i.e. the plain launch. Prompts up to 8k are bit-identical to the pre-M1
  runtime.
- **Otherwise:** the largest power of two with `q_blocks x kv_heads x S <= 32` and at least 8
  48-key tiles per segment. That gives S=8 at TP=1 (4 KV heads) and S=16 per rank at TP=2 (2 KV
  heads). This is the measured argmin at every depth, from 2k to 123k (see below).
- **Callers:** only prompt prefill turns it on (`Model::RunChunk` on its prefill path, and
  `PrefillProfiled`). Decode steps, verify windows (MTP and DFlash), MTP priming and the attention
  tests pass `prefill_split_kv = false` and keep the plain launch. Verify windows of at most 10 rows
  take the decode kernel anyway.
- **`R4DX_PREFILL_SPLITKV`:** at 4b04ece the law was the default. It is opt-in since
  [M1 final](#m1-final-exact-by-default-split-kv-opt-in) (`=split`). `0` or `1` means never split
  (checked bit-identical to the pre-M1 build at 32k). `N` forces N segments on every prompt-prefill
  call, capped at 32.

### Lossless: definition and evidence

- **Below the threshold:** outputs are bit-identical. The 8k KL segments of the M0 harness are
  byte-identical to the M0 dense dump.
- **Above the threshold:** the only differences are:
  - where each segment's lazy softmax reference max starts, which changes which f16 P roundings
    happen;
  - the fp32 order of the merge.

  No precision is dropped: the partials are fp32 accumulators, not a re-rounded copy.

`tests/kernels/test_attn_prefill_splitkv` compares against a CPU fp32 reference over the same fp8
cache. It covers depths 0 to 32700, q_len 37, 64 and 150, a permuted block table, partial pages and
tiles, 2 to 200 splits (clamped to 64), and one segment that dominates the merge.

- The split error against fp32 equals the unsplit kernel's own error (max-abs within 3% at every
  case, RMS equal to 3 digits). At depth 9000, for example: unsplit max abs 6.44e-05, rms 1.060e-05;
  split 8 gives 6.42e-05 and 1.060e-05.
- Split vs unsplit differs on 2-14% of outputs, by about 1 bf16 ulp of the output's magnitude
  (max abs 1.2e-04 at depth 9000).
- `splits <= 1` gives 0 differing bits. A split launch rerun gives 0 differing bits.

**Model level**
([`tools/prefill` KL harness](../tools/prefill/README.md), mix45m, TP=1, device 1, against M0's
dense dumps):

| length | mean KL | p99 KL | max KL | top-1 % | ppl dense -> split | prefill s dense -> split |
|---|---|---|---|---|---|---|
| 8k | 0 (bytes identical) | 0 | 0 | 100 | identical | 23.5 -> 21.1 (3 segs; device 0 vs device 1) |
| 32k | 0.00142 | 0.0191 | 0.152 | 98.44 | 2.1124 -> 2.1063 | 38.3 -> 29.3 per segment (same device) |
| 128k | 0.00077 | 0.0084 | 0.032 | 98.96 | 1.8511 -> 1.8506 | 299.6 -> 141.6 per segment |

**Calibration for the same 32k segments.** Neither of these perturbations involves split-KV at the
same count.

- **Chunk boundaries shifted by 1001 tokens:** today's kernels only (`R4DX_PREFILL_SPLITKV=0`), with
  the prefix prefilled as two calls, which is what a prefix-cache restore does. KL per segment:
  - code: mean 0.048, max 3.9
  - prose: mean 0.0044
  - recall: mean 0.0013

  That is 3 to 15 times more than split-KV.
- **Split 8 vs split 16:** KL is the same size as dense vs split 8.

The code_32k continuation has a few knife-edge rows, and any rounding-level change moves them. The
unsplit kernel is one member of that rounding class. Perplexity is not worse at any length.

### Speed

Kernel, q_len 64, ms per call (`tool_attn_prefill_bench --splitkv`, merge included):

| depth | 4 KV heads, unsplit | 4 KV heads, S=8 | 2 KV heads (TP=2 rank), unsplit | 2 KV heads, S=16 |
|---|---|---|---|---|
| 8192 | 0.685 | 0.152 (4.5x) | 0.726 | 0.089 (8.2x) |
| 32768 | 2.74 | 0.518 (5.3x) | 2.75 | 0.272 (10.1x) |
| 65536 | 5.58 | 0.974 (5.7x) | 5.72 | 0.485 (11.8x) |
| 122880 | 10.52 | 1.75 (6.0x) | 10.71 | 0.929 (11.5x) |

End to end at TP=1, prefix prefill: 32k goes from 38.3 s to 29.3 s (1.31x), and 128k from 299.6 s
to 141.6 s (2.12x). TP=2 was not run in M1: this milestone ran on HIP device 1 only.

## M1 results: the lossless gate, diagnosis and the exact-wide mode

Split-KV at 4b04ece missed the lossless gate (mean KL <= 0.0005 and top-1 >= 99.5% against the M0
dense dumps). It failed on prose_32k (0.00090, 97.7%), code_32k (0.00318, 98.0%) and prose_128k
(0.00163, 96.9%). This section diagnoses why, adds a mode that is bit-identical, and re-validates
both. Commits: libr4d `c9c0237` and `dec5a4f` (branch `prefill`), r4dx `d731bb8` and `b5e5eb9`.
Raw outputs are in `E:\models\r4dx\prefill-m1\fix\`. Everything ran at TP=1 on HIP device 1.
These runs predate the default change: "split-KV (default law)" and "branch default" below mean
`R4DX_PREFILL_SPLITKV` unset at the time, which is `=split` today.
- The exact-mode runs, ctest and the dense and split-KV TTFT reruns used the `b5e5eb9` build.
- The split-KV task set, warm turns, first TTFT pair and identity used the `d731bb8` build. Its
  split-KV and dense paths are the same code.

### Diagnosis: split-KV is differently rounded, not less accurate

Where the split and unsplit kernels can differ:

| candidate | status in 4b04ece |
|---|---|
| partial O stored at reduced precision | No: each segment writes its fp32 accumulator. |
| LSE, max or rescale order in the merge | fp32, with a fixed segment order, and deterministic. |
| P rounded to f16 against a different max | **Yes.** p = 2^(s - m_ref + SHIFT) is rounded to f16 before the PV WMMA. m_ref is the kernel's lazy running max. Each segment starts it from its own first tile, so a split P is rounded against a different reference than the unsplit P: a different rounding, not a coarser one. |
| fp32 summation order | **Yes.** Unsplit sums the tiles sequentially into one accumulator. Split sums them per segment and then merges the segments. |
| segment boundaries vs tile order | Already on 48-key tile boundaries. That does not remove either of the two differences above. |

The kernel-level test is `tool_attn_prefill_precision` (new). It runs one 64-row chunk at depth D
through the plain, exact and split-KV (S = 8/16/32) launches. The reference is fp64 on the CPU
(QK, softmax and PV in double) over the same fp8 cache and bf16 query. There are two data sets:
- **flat:** i.i.d. data (entropy 7.6-11.7 nats);
- **peaky:** keys share a mean direction and each row has 6 strongly aligned keys (top weight
  about 0.15).

Each metric compares against fp64:
- **rms err:** rms error;
- **correct:** the share of outputs equal to the correctly rounded bf16 of the fp64 value;
- **closer / farther:** the share of outputs where the variant is nearer to fp64 than plain, or
  further from it.

| data | depth | variant | rms err | correct | closer / farther than plain |
|---|---|---|---|---|---|
| flat | 2048 | plain | 2.246e-05 | 87.88% | - |
| flat | 2048 | split 8 | 2.246e-05 | 87.96% | 6.5% / 6.5% |
| flat | 32768 | plain | 5.747e-06 | 87.64% | - |
| flat | 32768 | split 8 | 5.748e-06 | 87.60% | 6.6% / 6.7% |
| flat | 122880 | plain | 2.914e-06 | 87.84% | - |
| flat | 122880 | split 8 | 2.915e-06 | 87.79% | 6.5% / 6.6% |
| peaky | 2048 | plain | 6.236e-04 | 81.33% | - |
| peaky | 2048 | split 8 | 6.235e-04 | 81.71% | 5.9% / 5.4% |
| peaky | 32768 | plain | 6.220e-04 | 83.78% | - |
| peaky | 32768 | split 8 | 6.181e-04 | 83.95% | 5.8% / 5.6% |
| peaky | 122880 | plain | 6.547e-04 | 83.50% | - |
| peaky | 122880 | split 8 | 6.163e-04 | 84.16% | 6.4% / 5.7% |
| peaky | 122880 | split 16 | 6.157e-04 | 84.28% | 6.7% / 5.9% |

Split 16 and split 32 match split 8 at every depth. The exact launch is bit-identical to plain at
every depth.

Split-KV's error against fp64 is the unsplit kernel's error:
- With flat data, rms is within 0.05%.
- With peaky data at depth, split-KV is slightly *more* accurate: rms is 6% lower at 123k, and the
  correctly-rounded share is higher. The unsplit kernel's single long sequential fp32 sum is the
  weaker of the two.
- On the outputs where the two differ, split-KV is closer to fp64 about as often as it is farther.

So the dense run is not a more correct reference. It is one member of a class of equally accurate
roundings, and the model amplifies any change within that class:
- M1's own calibration: moving the chunk boundaries with today's kernels gives 32k mean KL
  0.0013-0.048, and split 8 vs split 16 gives the same KL as dense vs split 8.
- On the task set (below), split 8 flips one knife-edge vt item at 32k, and split 4 and split 16 do
  not.

**Consequence for the gate.** "KL <= 0.0005 against the dense dump" can only be met by a kernel
that is bit-identical to the dense one. No split of the KV range can be: the unsplit accumulator
is a sequential fp32 sum under a history-dependent lazy max, and any parallel partition
reassociates it. The meaningful gate for a non-bit-identical kernel is:
1. error against an fp64 reference no worse than the unsplit kernel's (above: met);
2. KL against dense within the calibrated rounding-class spread (met: split-KV KL is below the
   chunk-shift KL at 32k);
3. task scores within the spread of equally accurate variants, and perplexity not worse.

A model-level KL against a higher-precision run (fp32 P, or HF bf16) is not available: r4dx has no
higher-precision attention path, and an HF run at 32k-128k does not fit this box (M0 caveat). The
kernel-level fp64 comparison stands in for it.

### The lossless alternative: exact-wide prefill (`R4DX_PREFILL_SPLITKV=exact`)

The plain call's parallelism can be widened along the two axes that never touch a row's
arithmetic:
- **Query rows:** fewer warps per workgroup. A wave keeps the same 16 rows, so its wave-uniform
  lazy-max decisions stay the same.
- **Output columns (DSPLIT):** DS workgroups share a q-block. Each redoes QK and the softmax, so it
  has the same m_ref sequence, f16 P and row sum. Each stages only its 1/DS slice of V and writes
  those d-tiles.

Tile size is free too, because the lazy-max check runs once per 16-key m-tile. The default
geometry is 12 warps x 4 d-slices x 96-key tiles, which gives 32 workgroups per chunk at 4 KV
heads. `R4DX_PREFILL_SPLITKV=exact` routes every prompt-prefill call there at any depth. The price
is recomputing QK and restaging K in every d-slice.

| depth | plain ms | exact ms (x) | split-KV S=8 ms (x) | 2 KV heads: plain | exact (x) | S=16 (x) |
|---|---|---|---|---|---|---|
| 8192 | 0.695 | 0.308 (2.25x) | 0.143 (4.9x) | 0.679 | 0.258 (2.6x) | 0.086 (7.9x) |
| 32768 | 2.714 | 1.175 (2.31x) | 0.496 (5.5x) | 2.790 | 1.033 (2.7x) | 0.246 (11.4x) |
| 65536 | 5.514 | 2.410 (2.29x) | 0.981 (5.6x) | 5.721 | 2.202 (2.6x) | 0.522 (11.0x) |
| 122880 | 10.859 | 4.518 (2.40x) | 1.779 (6.1x) | 11.053 | 4.365 (2.5x) | 0.960 (11.5x) |

The table is `tool_attn_prefill_bench --exact`, q_len 64. It has 0 differing bf16 outputs in every
geometry. `test_attn_prefill_splitkv` checks the same property for 11 geometries in every case:
depth 0 to 32700, q_len 37/64/150, and a permuted block table.

### Validation (TP=1, HIP device 1)

**Cold TTFT** (`ttft_cli.ps1`, 2 runs, prefill seconds). The dense rerun is the same binary with
`R4DX_PREFILL_SPLITKV=0`, run the same afternoon. It is about 5% slower at 8k than M0's morning
runs, so compare within a row against it.

| length | dense M0 | dense rerun | split-KV (default law) | exact |
|---|---|---|---|---|
| 8k | 7.23 / 7.28 | 7.62 / 7.60 | 7.58 / 7.61 (1.00x) | 7.33 / 7.37 (1.03x) |
| 32k | 38.41 / 38.56 | 38.73 / 38.59 | 31.29 / 31.18 (1.24x) | 34.04 / 33.52 (1.14x) |
| 64k | 103.03 / 102.94 | - | 66.62 / 66.70 (1.54x vs M0) | 78.35 / 78.50 (1.31x vs M0) |
| 128k | 309.21 / 309.05 | 301.30 / 301.20 | 149.86 / 149.84 (2.01x) | 199.65 / 199.38 (1.51x) |

The KL harness's prefix prefill gives the same ratios: split-KV 141.7 s vs 299.6 s at 128k, and
exact 199.1 s.

**Warm 4k turn** (`warm_delta.ps1`, prompt ms of the ~4k appended turn):

| offset | dense (M0) | split-KV | exact |
|---|---|---|---|
| 0 (cold 4k) | 3.49 s | 3.63 s | 3.61 s |
| after 32k | 6.54 s | 4.15 s | 4.96 s |
| after 64k | 9.79 s | 4.67 s | 6.28 s |

**KL against the M0 dense dumps** (`kl_compare.py`, 256 tail rows per segment):

| segment | split-KV mean KL | p99 | top-1 % | ppl dense -> split | exact |
|---|---|---|---|---|---|
| prose_8k / code_8k / recall_8k | 0 (bytes identical) | 0 | 100 | identical | 0 (bytes identical, prose_8k) |
| prose_32k | 0.00090 | 0.0057 | 97.66 | 3.915 -> 3.910 | 0 (bytes identical) |
| code_32k | 0.00318 | 0.0770 | 98.05 | 1.837 -> 1.824 | 0 (bytes identical) |
| recall_32k | 0.00018 | 0.0034 | 99.61 | 1.310 -> 1.310 | 0 (bytes identical) |
| prose_128k | 0.00163 | 0.0114 | 96.88 | 4.524 -> 4.532 | 0 (bytes identical) |
| code_128k | 0.00058 | 0.0113 | 100 | 1.192 -> 1.190 | 0 (bytes identical) |
| recall_128k | 0.00009 | 0.0020 | 100 | 1.176 -> 1.175 | 0 (bytes identical) |

- The split-KV dumps are 4b04ece's (`validate\kl\dense_vs_m1.json`). The `b5e5eb9` build
  reproduces the 32k dumps byte for byte.
- The exact dumps were made with the 12x4 geometry (48-key tiles). The 96-key default was
  rechecked on code_32k and prose_128k, and those dumps are also byte-identical to dense.
- kl_compare reports exact's top-5 on prose_128k as 99.61%. That is a tie-order artifact: the files
  are byte-identical.

**Task set against the M0 dense baseline** (score %, `score_tasks.py --compare`):

| length | dense | split-KV | identical outputs | score changes |
|---|---|---|---|---|
| 32k (56 items) | 99.3 | 97.9 | 49/56 | vt-32k-03: 1.0 -> 0.2 (vt 95.0 -> 85.0) |
| 128k (14 items) | 90.0 | 90.0 | 11/14 | none |

vt calibration at 32k, with the same binary and the same 8 items:

| mode | vt score | outputs identical to dense |
|---|---|---|
| split 8 (the default law) | 85.0 | 5/8 |
| split 4 | 95.0 | 6/8 |
| split 16 | 95.0 | 6/8 |
| exact | 95.0 | 8/8 |

The vt-32k-03 answer is knife-edge: the split-8 run stopped after the first of five names. Two
other equally accurate rounding members keep it.

**Other checks:**
- `ctest -LE tp2gpu`: 92/92 passed. That includes `test_attn_prefill_splitkv` with the exact
  geometries, and `test_forward_smoke`, which failed on both main and 4b04ece in the earlier run
  with HIP error 719.
- Short-prompt greedy identity against main, for plain, dflash7 and mtp3 with 4 prompts each:
  12/12 byte-identical text and token ids. Those prompts never reach the split threshold. The exact
  mode is bit-identical by construction.
- Decode: unchanged. Decode, verify windows and MTP priming never take either new path. 4b04ece's
  bench (`validate\decode_bench.log`: dflash7 116.37 vs 116.15 main, plain 36.54 vs 36.61) covers
  the same decode code.

**TP=2: not run in this pass** (done later: see [M1 final](#m1-final-exact-by-default-split-kv-opt-in)). TP=2 cold TTFT at 32k and 128k (split-KV and exact) and TP=2 short-prompt
identity against main TP=2 need device 0. Device 0 is held until M0's own TP=2 128k run 2 finishes
and writes `E:\models\r4dx\prefill-m0\TP2_DONE`, and that marker did not appear during this pass.
To run them afterwards (the phase script is in this session's scratchpad, and its steps are the
plain `ttft_cli.ps1 -Tp 2` calls):
- `ttft_cli.ps1 -Tp 2 -Lengths 32k,128k -Runs 2 -Cli <b5e5eb9 build>\src\cli\r4dx-cli.exe`, once with
  `R4DX_PREFILL_SPLITKV` unset and once with `=exact`;
- the identity loop with `--tp 2 --tp-mode real` against `C:\Users\pay20\dev\r4dx\build\win-hip`.

Expected from the kernel numbers: split-KV gives a TP=2 rank 16 segments, 11.5x on the attention
call at 123k, against 2.5x for exact.

### Gate decision

| variant | lossless gate (KL <= 0.0005, top-1 >= 99.5%) | fp64 error vs plain | tasks | 128k TTFT |
|---|---|---|---|---|
| split-KV (branch default) | **fails** on 3 of 9 segments | equal (slightly better at depth) | 128k equal; 32k one knife-edge vt flip, not seen at S=4/16 | 2.01x (target 2x: met) |
| exact-wide (`=exact`) | **passes** (bit-identical) | identical | identical | 1.51x (target: missed) |

No single variant meets both the lossless gate and the 2x target at TP=1. Nothing was merged to
main at this point; option 1 was chosen (M1 final, below).
Options:

1. **Exact as the default, split-KV opt-in.** Lossless by construction, 1.5x at 128k, 1.3x at 64k.
   Split-KV stays one environment variable away for users who accept rounding-class drift for
   2.0x.
2. **Split-KV as the default under the redefined gate** (fp64 error no worse than dense, KL within
   the calibrated rounding spread, task scores within the spread of equal-accuracy variants). This
   is 2.0x at 128k and 1.5x at 64k. Keep exact as the switch for bit-for-bit reproducibility
   against older runs.
3. **Bigger chunks (option c) instead.** Not a lossless path either: it moves GDN and chunk
   boundaries, which the calibration shows costs more KL than split-KV (32k mean KL 0.0013-0.048).
   It also has the largest blast radius.
4. **Push exact further.** It is capped by the redundant QK and K staging per d-slice. Measured:
   2.3-2.4x per call at depth, and about 2x on the attention share of the 128k prefill end to end
   (roughly 197 s -> 95 s, with 104 s of non-attention work). Closing the gap to
   split-KV needs the d-slices to share QK, which RDNA4 cannot do across workgroups without a trip
   through global memory.

Recommendation: option 1 if "M1 = lossless" is binding, and option 2 otherwise. The fp64 analysis
says split-KV costs no accuracy, only reproducibility against the old bits.

## M1 final: exact by default, split-KV opt-in

Decision: option 1 of the gate decision above. "M1 = lossless" is binding, so the default prompt
prefill attention is the exact-wide launch, and split-KV stays one environment variable away.

| `R4DX_PREFILL_SPLITKV` | prompt-prefill attention | bits vs dense | 128k TTFT, TP=1 |
|---|---|---|---|
| unset, empty or `exact` (**default**) | exact-wide (`r4d_attn_prefill_exact_*`, 12 warps x 4 d-slices x 96-key tiles) | identical | 1.51x |
| `split` (also `splitkv`, `auto`) | split-KV by the split law (S=8 at TP=1, 16 per rank at TP=2, from 8k context) | rounding-class drift (see above) | 2.01x |
| `0`, `1`, `off` or `dense` | the old single-workgroup dense launch | identical (it is the dense kernel) | 1.00x |
| `N` > 1 | N split-KV segments on every call, capped at 32 | rounding-class drift | - |
| anything else | a warning on stderr, then the default | identical | 1.51x |

- Only prompt prefill reads it. Decode, verify windows and MTP priming always take the plain launch.
- The parse is `ParsePrefillAttnMode` in `attention_layer.hpp`. `test_attn_layer` checks it on the
  CPU before its data-presence skip.

**Checks with the new default** (the `1af310d` build, HIP device 1, outputs in
`E:\models\r4dx\prefill-m1\final\`):
- `ctest -LE tp2gpu`: 95/95 passed, including `test_attn_prefill_splitkv` and the parse check in
  `test_attn_layer`.
- Short-prompt greedy identity against main at TP=1 (plain, dflash7 and mtp3, 4 prompts each from
  `tests/model/mtp_prompts.txt`): 12/12 byte-identical text and token ids.

### TP=2 (both GPUs, `--tp 2 --tp-mode real`)

Cold TTFT (`ttft_cli.ps1 -Tp 2`, 2 runs, prefill seconds), the `1af310d` build; the dense column is
M0's pre-M1 binary (128k run 2 measured the same evening as the M1 runs):

| length | dense (M0) | exact (default) | split-KV (`=split`) | exact vs dense | split-KV vs dense |
|---|---|---|---|---|---|
| 32k | 33.10 / 33.58 | 26.23 / 26.35 | 23.16 / 23.49 | 1.27x | 1.43x |
| 128k | 293.36 / 316.13 | 166.62 / 168.86 | 104.60 / 106.32 | 1.82x | 2.89x |

TP=2 against TP=1 (the TP=1 M1 validation table, means of 2 runs):

| length | exact TP=1 -> TP=2 | split-KV TP=1 -> TP=2 | dense TP=1 -> TP=2 (M0) |
|---|---|---|---|
| 32k | 33.78 -> 26.29 s (1.28x) | 31.24 -> 23.33 s (1.34x) | 38.49 -> 33.34 s (1.15x) |
| 128k | 199.52 -> 167.74 s (1.19x) | 149.85 -> 105.46 s (1.42x) | 309.13 -> 304.74 s (1.01x) |

- Exact gives a TP=2 rank 16 workgroups per call (2 KV heads x 8), so TP=2 now scales at depth
  instead of being bound by a 2-workgroup attention call. Split-KV gives each rank 16 segments and
  scales further.
- Short-prompt greedy identity against main's TP=2 (plain, dflash7 and mtp3, 4 prompts each,
  `--tp 2 --tp-mode real`): 12/12 byte-identical text and token ids.
- Outputs: `E:\models\r4dx\prefill-m1\final\` (`ttft\`, `identity_tp2\`, `phases.log`, and the
  runner `final.ps1`).

**Final default:** exact-wide (lossless: bit-identical to dense at TP=1 and TP=2). 1.51x at 128k
TP=1 and 1.82x at 128k TP=2 over dense. Split-KV (`R4DX_PREFILL_SPLITKV=split`) stays opt-in for
2.0x (TP=1) and 2.9x (TP=2) at 128k, with rounding-class drift against the dense bits.

## The 256-row prefill chunk (default on, `R4DX_PREFILL_CHUNK`)

Prompt prefill runs in 256-row super-chunks by default: the trellis linears see 256 rows through libr4d's
M = 256 GEMM (one weight pass per 256 rows instead of four). The attention core still runs in 64-row
sub-slices in order. The GDN sequence ops (conv prep, kkt solve, chunk scan, gated norm) run once over the
256 rows with chunk 64 inside the kernels; `R4DX_GDN_SLICE=64` restores the sub-slices. Conv prep runs
`r4d_gdn_conv_prep2` (same bytes on a grid that fills the device); `R4DX_GDN_CONV=1` restores
`r4d_gdn_conv_prep`. A 64-row Model (`R4DX_PREFILL_CHUNK=0`) ignores both knobs and runs exactly the
pre-change kernels. The result is bit-identical to the 64-row path, so no accuracy number moves. The two
GDN changes cut 8k TTFT 5.002 -> 4.722 s (1713 tok/s) and 32k 24.55 -> 23.41 s on device 1. The design,
the identity coverage, the kernels and the measurements are in [docs/trellis-m256.md](trellis-m256.md)
("GDN sequence ops").

| `R4DX_PREFILL_CHUNK` | prompt-prefill chunk |
|---|---|
| unset, empty or `256` (**default**) | 256-row super-chunks where the configuration allows it (below), else 64-row chunks |
| `0` or `64` (**kill switch**) | 64-row chunks exactly as before: the activation buffers and the arena keep their 64-row sizes (no extra VRAM) |
| anything else | a warning on stderr, then `64` |

- The variable is read once per process. `ModelOptions::prefill_chunk` (0 = follow the environment) forces
  64 or 256 for a test that needs both in one process.
- Every model load prints one line, e.g. `[r4dx::model::Model] prefill chunk: 256 rows (default;
  R4DX_PREFILL_CHUNK=0 restores 64-row chunks), GDN sequence ops once per super-chunk (R4DX_GDN_SLICE=64
  restores 64-row sub-slices)`, or `64 rows (... asked for 256, not used: <reason>)` when a 256 request
  could not be honoured. A second line names the GDN conv prep kernel (`R4DX_GDN_CONV`).
- The chunk grid is anchored at the start of each `Prefill` call: full 256-row super-chunks while at least
  256 rows remain, then the ordinary 64-row chunks for the rest (the last one holds 1..64 rows). The tail
  therefore runs the very chunks the 64-row grid makes for it, and a call split anywhere (a prefix-cache
  restore, a warm-turn suffix) shifts nothing it did not shift before. `test_prefill_chunk` walks the grid
  for the tail lengths 1, 63, 64, 65, 255, 256, 257, 511 and 8145. No 128-row super-chunk exists: the
  tail is under 256 rows, and the M = 128 kernel would save well under 1% of a prompt.
- `test_prefill_chunk_identity` (device 1, needs the 4-layer container for its fast part and the trellis
  container for the real part) loads a 64-row and a 256-row Model one after the other and compares
  every observable bit for bit: the last-row logits, the greedy tokens after, and a digest of all
  per-sequence state (each KV cache, each GDN recurrent / conv state; with a drafter also the MTP KV and
  the DFlash ring) for single calls of 1, 63, 64, 65, 255, 256, 257, 511 (and 8145 on the 4-layer
  container) rows and for prefix-reuse shapes (300 + 333, 64 + 511, 257 + 1, 1 + 255 + 257: the grid is
  anchored at each call). A negative control shows the digest does change when the chunk grid moves.
- VRAM (MEASURED 2026-09-30, device 1, Huihui trellis mix4.5m, `--max-ctx 131072`, vision auto, prompt
  checkpoint on, 8k prompt; `E:\models\r4dx\chunk\implement\mem`): the 256-row activation buffers, the
  DFlash feature buffer and the 224 MiB arena (96 MiB with `=0`) cost 0.13 to 0.20 GiB more per Model.
  Used VRAM at the end of an `r4dx-cli` run, default vs `R4DX_PREFILL_CHUNK=0`: plain 21.86 vs 21.73
  GiB, `--dflash` 23.96 vs 23.76, `--mtp 3` 22.59 vs 22.41 (`arena+scratch` in the load line 0.2312 vs
  0.1062 GiB; a load without the vision tower shows 0.21875 vs 0.09375). `r4dx-server --max-ctx 131072
  --dflash` (the largest setup) loads with 7.71 GiB free of 31.86 (7.91 with `=0`), serves an 8k
  request, and its greedy answer, `draft_n` (21) and `draft_n_accepted` (5) equal the kill-switch
  run's. Every buffer is allocated at load; nothing grows during a request.
- **MTP and DFlash** run the wide path, bit-identical, by working in the 64-row slices of a super-chunk:
  the MTP head primes its KV per slice with exactly the pair of `PrimeKv` calls a 64-row chunk makes
  (a 1-row boundary call seeded by the previous slice's last hidden row, then 63 within-slice rows; the
  pinned host staging of `MtpHead` has 256 rows so the slices' back-to-back calls stay disjoint), and the
  DFlash feature capture (a 256-row buffer) is handed to `SetDflashCaptureObserver` observers and the
  drafter's `InjectFeatures` (`max_inject_rows` 64) slice by slice, each injection synchronized as the
  64-row run does. `Prefill`'s `on_chunk_captured` callback runs once per 64-row slice with
  `DflashFeatureBuffer()` / `DflashFeatureRows()` showing that slice. `test_prefill_chunk_identity`
  compares the MTP KV, the DFlash ring, the captured feature bytes and the speculative rounds' tokens.
- **TP = 2** runs the wide path too. The all-reduce mailbox holds 64 rows (`kMaxAllReduceBytes`); a
  256-row row-parallel activation is all-reduced by `TpComm::AllReduceSumBf16Rows` as four ordinary
  64-row calls (each element is the sum of the same two ranks' values, so slicing changes no byte, and
  each call keeps the 500 ms spin timeout and the abort protocol untouched; both ranks make the same
  calls). `TpWarmup` runs a 256-row prefill first, so the M = 256 GEMM's first launch at the shard
  shapes and the sliced all-reduces are warm before the first request; the submission bounding's unit
  (`Model::RunChunk`) is a quarter of its layers for a 256-row chunk, so the GPU time between forced
  submissions on the display card is what it was. `test_prefill_chunk_identity` compares TpModel in
  emulate mode (both ranks on one device, the real trellis container included) 64-row against 256-row:
  logits, tokens and the per-rank all-reduce call counts are equal.
- **Validation of the default-on change** (MEASURED 2026-09-30, branch `chunk-default`, clean build in a
  new build directory, HIP device 1 and TP = 2 as stated; logs and scratch under
  `E:\models\r4dx\chunk\gates`, nothing else changed on disk):

  | check | result |
  |---|---|
  | dense KL, 12 segments (prose / code / recall x 8k / 32k / 64k / 128k), default (no env) and `R4DX_PREFILL_CHUNK=0` | 12 of 12 `logprobs.f16` sha256-identical to `pflash\baseline\kl-dense-huihui`, both runs (whole-run wall, which includes the 256 teacher-forced decode rows of every segment: 926 s default vs 1109 s off; 8k prefix prefill 5.31 vs 7.27 s, 32k 26.60 vs 34.85 s) |
  | `g6_validate.ps1` (validate_dflash, validate_spec_sampling, three smokes) | 5 of 5 steps exit 0 (smoke PASS counts 217 / 168 / 170, FAIL 0) |
  | `kl_rung4.ps1` | mean KL 0.00788, top-1 95.70%, all four segments byte-identical to `huihui\kl\rt-mix45m` |
  | `tp1_identity.ps1` against the frozen `tp1-1099446` binaries | rows 1-6, 8, 9 EQUAL (text, token ids, `[stats]` lines; row 6 bf16 and w4a16 logprobs); row 7 SKIP (the vision test image is absent from this tree) |
  | `ctest -LE tp2gpu`, no env | 93 of 93 passed in 16.5 min: 73 ran, 20 skipped for gitignored goldens / fixtures (convert_trellis_import, test_kernel_bandwidth, test_rope_neox, test_topk16, test_dflash_attn, test_dflash_conv, test_rmsnorm_plain, test_trellis_decode / input / gemm, test_gdn_layer, test_final_lm_head, test_dflash_draft, test_trellis_linear, test_attn_layer, test_mrope_attn_layer, test_preprocess, test_position_ids, test_vision_index, test_vision_tower); `test_prefill_chunk` and `test_prefill_chunk_identity` (15 configurations) among the passes |
  | `r4dx-cli`, 8k prompts (needle, CWE, code-QA), default vs `=0`: plain, `--mtp 3`, `--dflash --dflash-k 7` | text, token ids and the `[stats] mtp / dflash` lines identical in all 9 pairs (e.g. CWE `--dflash`: 7 rounds, 49 drafted, 15 accepted, both) |
  | cold TTFT, TP = 1, prefill seconds (one discarded warm-up) | 8k: OFF 7.656 / 7.664, ON 5.711 / 5.699 = **1.343x** (pairs 1.341, 1.345); 32k: OFF 35.186, ON 26.667 = **1.319x** |
  | TP = 2 (both GPUs): `ctest -L tp2gpu` | 2 of 2 passed (`test_tp_allreduce_2gpu`, `test_tp_real_vs_emulation`); `tdr_check.ps1`: no TDR over all the TP = 2 runs |
  | TP = 2: KL prose_8k, greedy on an 8k needle and CWE prompt | logprobs sha256 identical with and without the chunk; both greedy outputs (text and ids) byte-identical |
  | TP = 2 cold TTFT, 8k (one discarded warm-up) | OFF 5.382 / 5.400, ON 3.982 / 3.985 = **1.353x** |

  TP = 2 with `--mtp` / `--dflash` was validated with both ranks emulated on one device
  (`test_prefill_chunk_identity`, `smoke_tp2_emulate_dflash`), not on the two cards: only the
  transport differs, and its calls are the ordinary 64-row ones.
- **Image prompts** run the wide path too: `PrefillMultimodal` walks the same grid. The image splice and
  the 3-axis rope rows are functions of the absolute position (the rope buffers are sized for 256
  rows), so a super-chunk that straddles an image run splices and ropes what four 64-row chunks do, and
  the text-only continuations of an image conversation are wide as well. `test_prefill_chunk_identity`
  feeds synthetic image rows (the tree has no vision test image; row 7 of `tp1_identity` skips): an
  image mid-prompt, at position 0, and a conversation (image prompt, text `Prefill`, text-only
  `PrefillMultimodal`, decode), plain / `--mtp 3` / `--dflash`, state digests bit-identical.
- Falls back to 64 rows, with a stderr line at load, only for a quant2 (rotated) container (its residual
  rotation and Hadamard epilogues were not validated at 256 rows).