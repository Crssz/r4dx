# Prefill: long-prompt TTFT

**Goal:** cut cold time-to-first-token for long prompts (8k to 128k) on Qwen3.8-27B. The model is a
hybrid of 48 GDN and 16 full-attention layers, with fp8 paged KV, head 256 and gqa 6. Work runs
through `qwen38-27b-trellis-mix45m.r4dx` on 2x R9700.

- **M0:** measure. A long-context eval kit, a profile of where prefill time goes, and dense accuracy
  baselines.
- **M1:** lossless attention parallelism (below).
- **M2:** opt-in lossy modes, gated on the M0 KL harness.

The kit and its commands are in [`tools/prefill/README.md`](../tools/prefill/README.md). Raw outputs
are in `D:\models\r4dx\prefill-m0\` (`profile\results.json`, `baseline\`) and are never committed.

## M0: baseline and profile (dense prefill, 64-row chunks)

### Cold TTFT

`ttft_cli.ps1` measures prefill seconds. Each run is a fresh `r4dx-cli` process.

| length | tokens | TP=1 s (2 runs) | TP=1 tok/s | TP=2 s | TP=2 tok/s | TP=2 speedup |
|---|---|---|---|---|---|---|
| 8k | 8145 | 7.23 / 7.28 | 1123 | - | - | - |
| 32k | 32623 | 38.41 / 38.56 | 848 | 33.10 / 33.58 | 979 | 1.15x |
| 64k | 65529 | 103.03 / 102.94 | 636 | - | - | - |
| 128k | 130884 | 309.21 / 309.05 | 423 | 293.36 (1 run) | 446 | 1.05x |

- **TP=1:** HIP device 1, binary built at 60e6fae. K4m gives 37.6 s at 32k and 304.3 s at 128k.
- **TP=2:** `--tp 2 --tp-mode real` on both GPUs, with the dense pre-M1 binary. The `git` field of
  that jsonl is only the worktree HEAD label, which says 4b04ece.
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

- **Task set:** 154 RULER-style items from 8k to 128k, on mix45m at TP=1 with the 60e6fae binaries.
  - The per-task scores for each length are in `baseline\tasks\mix45m\summary.json`.
  - TODO: copy the table here.
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
- **`R4DX_PREFILL_SPLITKV`:** `0` or `1` means never split (the kill switch; checked bit-identical to
  the pre-M1 build at 32k). `N` forces N segments on every prompt-prefill call, capped at 32.

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
