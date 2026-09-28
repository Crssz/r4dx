# Prefill M1: split-KV prefill attention

Long-prompt prefill was bounded by one kernel. Prefill runs in 64-row chunks (`Model::max_chunk_`).
Each full-attention layer makes one `r4d_attn_prefill_h256_gqa6_fp8kv` call per chunk. That call
launches `ceil(q_len/64) x kv_heads` workgroups: 4 at TP=1 and 2 per rank at TP=2. The device holds
32 of these workgroups at once (768 threads and 54 KB of LDS each), so 1/8 of it was used. As a
result, attention grew from 0.6 ms per chunk at depth 0 to 188 ms at 128k (prefill M0,
`tools/prefill`).

M1 is lossless attention parallelism. Nothing else in the chunk changes.

## Design choice

| Option | What it changes | Verdict |
|---|---|---|
| (a) split-KV prefill attention | The KV tile range of one call is cut into S segments (grid x = q_blocks x S). Each segment writes fp32 partials, and a fixed-order merge follows. | **Chosen.** Only the attention call changes. Every invariant sized by `max_chunk_` stays as it is: activation buffers, `MtpHead::kMaxPrime`, DFlash feature buffers, GDN prewarm, the 96 MiB arena, `T <= 64` in `AttentionLayer`, `ApplyLinear`'s 64-row tiles, and the TP all-reduce channel sizes. Measured at the kernel: S=8 matches a 512-row call. |
| (b) group several chunks' attention into one call | Linears and GDN run in 64-row chunks, and attention runs over up to 512 rows at once. | Not possible without reordering layers. Layer L+1's input for chunk c needs layer L's attention output for chunk c. Batching attention across chunks means running every layer over all grouped chunks together, which is (c). |
| (c) raise `max_chunk_` to 256 or 512 | Everything. | The largest blast radius. Every buffer and invariant listed under (a) grows, and the skinny-GEMM, trellis and verify-window paths are all tuned for M <= 64. (a) gets the attention gain without any of that. |

## Mechanism (libr4d branch `prefill`)

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

## Split law (r4dx, `attention_layer.hpp`)

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

## Lossless: definition and evidence

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

## Speed

Kernel, q_len 64, ms per call (`tool_attn_prefill_bench --splitkv`, merge included):

| depth | 4 KV heads, unsplit | 4 KV heads, S=8 | 2 KV heads (TP=2 rank), unsplit | 2 KV heads, S=16 |
|---|---|---|---|---|
| 8192 | 0.685 | 0.152 (4.5x) | 0.726 | 0.089 (8.2x) |
| 32768 | 2.74 | 0.518 (5.3x) | 2.75 | 0.272 (10.1x) |
| 65536 | 5.58 | 0.974 (5.7x) | 5.72 | 0.485 (11.8x) |
| 122880 | 10.52 | 1.75 (6.0x) | 10.71 | 0.929 (11.5x) |

End to end at TP=1, prefix prefill: 32k goes from 38.3 s to 29.3 s (1.31x), and 128k from 299.6 s
to 141.6 s (2.12x). TP=2 was not run in M1: this milestone ran on HIP device 1 only.
