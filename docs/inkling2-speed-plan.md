# Inkling-2-Mini decode and prefill speed-up plan

Status: plan only, nothing implemented (written 2026-11-22). Base: branch `inkling2` at `f3e58ef` (worktree
`r4dx-inkling2` next to the main checkout). Model semantics and the KL gate are in [inkling2-plan.md](inkling2-plan.md) and
[inkling2.md](inkling2.md); read those first.

## 0. Rules for whoever runs this

- **Ask the user before every GPU run** (tests, benches, KL dumps, CLI runs) and give them the exact command. CPU builds and
  CPU tests need no approval.
- **Never overlap a model load with another bulk disk job** (DirectStorage hard-crashed the machine on 2026-11-22). If they
  must overlap, set `R4DX_DIRECT_STORAGE=0`.
- GPU work runs on **device 1** (`$env:HIP_VISIBLE_DEVICES='1'`). TP=1 uses 31.45 GiB at `--max-ctx 8192`, so keep 8192
  for every TP=1 run, and do not grow per-token workspace by more than a few hundred MiB.
- One stage per commit (or a small series). Each stage passes its correctness gate (section 4) before the next one starts.
- Kill switch for every behaviour change (env var `R4DX_INK_<NAME>=0` returns the old path), so a regression can be
  bisected without a rebuild. Remove switches only after the stage has been KL-gated.

## 1. Where we are and where the ceiling is

Measured in milestone 1 ([inkling2.md](inkling2.md) "Speed"):

| | TP=1 (device 1) | TP=2 |
|---|---|---|
| decode, CLI short prompt | **23.6 tok/s (42 ms/token)** | 20.0 tok/s |
| prefill, 25-token prompt | 218 tok/s | 264 tok/s |
| teacher-forced, chat corpus | 43.1 ms/row | 59.3 ms/row |

Byte budget per decoded token (from inkling2-plan.md 3.4; looped layers re-read their weights each visit):

| item | weights read per token |
|---|---|
| routed experts: 48 visits x 8 experts x 7.96 M | 3.06 B |
| attention, router, controllers: 48 visits x about 27 M | about 1.3 B |
| lm_head (200064 x 2304) | 0.46 B |
| **total** | **about 4.8 B weights, about 5.0 GB at MXFP8 (1.03 B/weight)** |

At the R9700's about 640 GB/s, the bandwidth ceiling is about **128 tok/s** at short context. LFM2 w8a16 reached about 67 %
of its ceiling (171 tok/s), so a realistic target is about **80 tok/s at TP=1**. Qwen3.8-27B trellis runs 37.4 tok/s, so
that would be about 2x faster.

**Targets** (TP=1, device 1, `--max-ctx 8192`, greedy, short prompt, 256 generated tokens):
- after S1: at least 40 tok/s;
- after S1-S5: at least 70 tok/s (stretch 85);
- prefill at a 2048-token prompt: measure first (S0), then at least 3x the S0 number after S2 + S3.

## 2. Why decode is slow today (read from `src/model/inkling2_model.cpp`)

Per layer visit (48 per token), the T = 1 path does:

1. **Message gates** (visits in passes 2 and 3): an `attnres` launch and 4 x `MsgGate` (3 launches each: prep, dense
   `r4dx_inkling2_dense_bf16` down to rank 64, apply).
2. **AttnRes + norms**: `attnres` plus 1 to 3 `rmsnorm_plain_f32in_bf16`.
3. **Attention** (`Attention()`): q GEMM, `head_clip`, k GEMM, v GEMM, k/v conv (full layers add 2 more `dense_bf16`
   launches for the dynamic conv weights), 2 head ops into the paged cache, rel GEMM, `head_clip`, `rel_logits` (writes
   `T x heads x 1024` fp32), sink `dense_bf16`, **`r4dx_inkling2_attn_ref_bf16kv`** (the reference kernel: one workgroup per
   (row, head), fp32 two-pass softmax), gate GEMM, sigmoid-mul, o GEMM.
4. **Post-attention conv**, **AttnRes + norm** for the MLP.
5. **MoE** (`Moe()`, T = 1 branch): router, then **`hipMemcpyAsync` of the 8 expert ids + `hipStreamSynchronize`**
   (`moe.readback`), then for each of the 8 experts a gate_up GEMM, a softcap-SwiGLU and a down GEMM (24 launches), then
   `r4dx_lfm2_moe_combine_bf16`.
6. **Post-MLP conv**.

So each token pays:
- **48 host syncs.** Each one drains the stream, so the GPU sits idle while the host enqueues the next visit's launches.
  The roughly 5 us per-launch cost is then on the critical path instead of hidden behind GPU work.
- **About 50-65 launches per visit, about 2500-3000 per token.** At about 5 us each that is 12-15 ms of the 42 ms, even
  with no syncs.
- A reference attention kernel, and a fully materialized `T x 24 x 1024` fp32 relative bias, read back by attention.
- MXFP8 GEMMs on **unmeasured fallback tunings**: `src/model/gemm_tuning_table_mxfp8.inc` is a one-row placeholder, so
  every linear takes `Mxfp8FallbackTuning` in `src/model/linear.cpp`.
- TP=2 adds 2 host-staged bf16 all-reduces per visit (96 per token, no P2P), which is why it is slower than TP=1.

## 3. Stages

Order: S0 first, then S1 (the biggest win). After that, S2-S6 are mostly independent; run them in the listed order unless
the S0/S1 profile says otherwise. S7 (TP=2) is optional.

### S0. Baseline profile (GPU, approval)

- Run `r4dx-cli ... --profile --profile-token 32` (`Inkling2Model::DecodeStepProfiled`) and the prefill profile flag from
  `src/cli/cli_args.h` (hipEvent spans per op family, `gemm:` share split, `r4dx_kernel_launches`). Record the decode
  profile at a short prompt and at about 4k context, and the prefill profile at 2048 tokens.
- Record plain decode tok/s (256 tokens, 3 runs, median) and prefill tok/s at 2048 and 8192 tokens.
- Write the table into section 6 of this doc. Every later stage reports against it.
- Done when: the baseline table exists, with launches/token, host enqueue ms, GPU-sum ms and wall ms.

### S1. Device-indexed grouped MoE decode, no host sync (biggest win)

The model: `r4dx_gemm_w8a16_moe_decode` in `src/kernels/include/r4dx/kernels/w8a16_gemm.h` (docs/w8a16.md 4.4), which LFM2
uses. The MXFP8 GEMM (`src/kernels/src/mxfp8_gemm.hip`) is already a copy-adapt of the w8a16 m64 kernel, so the grouped
variant is a copy-adapt as well. Only the scale changes: E8M0 per 32-K group instead of an f16 per 64/128 group.

1. **Kernel** `r4dx_gemm_mxfp8_moe_decode(a_bf16, wq_ptrs, ws_ptrs, ids, c, n_pairs, a_row_group, n_experts, K, N, WV, SK,
   NPW, NT, stream)` in `mxfp8_gemm.h/.hip`: grid.z = pair, expert = `ids[p]` read on the device, weight pointers from
   uint64 device tables, and the same per-pair contract as the w8a16 one. **It must be bit-identical** to
   `r4dx_gemm_mxfp8_nt` at M = 1, MB = 1, same SK, per pair.
2. **Activation**: run `r4dx_inkling2_swiglu_softcap_bf16` once over all 8 pairs (`rows = 8`; it already takes rows).
3. **Pointer tables**: in `Inkling2Container` (`src/model/inkling2_container.cpp`), build per-layer device tables for
   gate_up and down (wq and ws), the way `lfm2_container.cpp` does (`ptr_table_w8_wq` / `ptr_table_w8_ws`). Under TP the
   tables point at the rank's shard.
4. **Model**: in `Inkling2Model::Moe`, the T = 1 branch becomes router, grouped gate_up (n_pairs 8, a_row_group 8),
   softcap-SwiGLU over 8 rows, grouped down (a_row_group 1), and combine. No `moe.readback`, no host loop.
   `SetRouterTrace` keeps its sync (debug only).
5. **Tests**: `tests/kernels/test_mxfp8_gemm.cpp` gets a grouped-vs-per-pair bit-exactness test (random ids including
   repeats and out-of-range ids, the real Inkling shapes gate_up N 2304 x K 2304 and down N 2304 x K 1152, and the tiny shapes).
   `test_inkling2_tiny_e2e` must stay byte-identical.
- Gate: **bit-exact**. The greedy output and the teacher-forced dump are byte-identical to the S0 run, so no KL rerun is
  needed (verify with a byte compare of one teacher-forced dump).
- Expected: removes 48 syncs and about 16 launches per visit (about 770 per token). Estimate 23.6 -> 40-50 tok/s.

### S2. Device-indexed grouped MoE prefill

The model: `r4dx_gemm_w8a16_moe_prefill` (same header). Today the T > 1 branch syncs once per visit to read the expert
offsets, then loops over all 128 experts on the host (up to 256 GEMM launches plus 128 activations per visit).

1. Kernel `r4dx_gemm_mxfp8_moe_prefill(a_bf16, wq_ptrs, ws_ptrs, expert_offsets, c, n_pairs, n_experts, K, N, WV, SK, NPW,
   NT, stream)`, driven by the device offsets from `r4dx_lfm2_moe_sort`. It must be bit-identical to the per-expert
   `r4dx_gemm_mxfp8_nt` calls at the same SK.
2. Activation over all `T * top_k` sorted rows in one launch.
3. Remove the offsets readback from `Moe()`. Keep the coverage check (`off[E] == Pn`) behind a debug env var only, since it
   needs a sync.
- Gate: bit-exact against S1 (prefill logits and the teacher-forced dump byte-identical).
- Expected: the largest prefill win. Measure at 2048 and 8192.

### S3. Fused attention kernel (h128, GQA 3, relative bias, sink, sliding window, paged bf16 KV)

Replace `r4dx_inkling2_attn_ref_bf16kv`. Decode (T = 1) first, then prefill.

1. **Decode kernel**: one workgroup per (kv head, KV split). The 3 q heads of a kv head share each K/V tile read. Online
   softmax in fp32, split-K over context for long context with a small combine pass, and the sink as an extra logit with a
   zero value row.
2. **Relative bias on the fly**: do not materialize `rel` (`T x 24 x 1024` fp32). Compute the bias for distance d as
   `r[t, h, :] . rel_proj[:, d]` (d_rel 16) inside the kernel. `rel_proj` is `[16, 1024]`, small enough for LDS, which removes
   the `rel_logits` launch and its buffer. Keep `r4dx_inkling2_rel_logits_f32` for the reference path.
3. **Window**: sliding layers read the 1536-position ring through the existing block table. Full layers read all positions.
4. **Prefill kernel**: tile over query rows (WMMA for Q.K^T and P.V), same bias/sink/window rules. It can come later than
   the decode kernel.
5. **Tests**: `tests/kernels/test_inkling2_kernels.cpp` compares against the CPU spec (`inkling2_ref.hpp` Attention) and the
   reference kernel at the real shapes. Contexts: 1, 17, 1023, 1024, 1025, 1536, 4096 and 9000 (past the 8192 log-scaling
   switch, which is untested on real weights). Window on/off, sink on/off. Use tolerances, not bit equality.
- Gate: **not bit-exact**, so a full **KL gate rerun at TP=1** is required (section 4). The headroom is small: chat-ALL is
  0.00573 against a limit of 0.00882, the tool group 0.00740 against 0.00820, and routing is fragile in passes 2-3. Keep
  fp32 accumulation and one bf16 rounding at the output, like the reference.
- Expected: matters more as context grows. Measure decode at a short prompt and at 4k/8k.

### S4. Launch-count cuts (fusions)

Start with the ones that save the most launches per visit, and check each against the S0/S1 profile.

1. **One GEMM for the shared-input linears.** q, gate and rel (and the sink) always read the same `xq`. k and v also read
   `xq` only in sliding layers on visits **without messages** (`xk = xv = xq` in `RunVisit`: the pre-loop layers, pass 1 of
   the loop and the post-loop layers). On the 24 message visits (loop passes 2 and 3) each of q, k and v gets its own
   message-gated input. Store ONE concatenated weight per layer, rows in the order `q, gate, rel, k, v`
   (N = 3072 + 3072 + 384 + 1024 + 1024 = 8576), with no duplicate copy (TP=1 has no VRAM to spare). The MXFP8 layout is
   ordered by 16-row tile first (`mxfp8_gemm.h` byte order), so any 16-aligned row range is a contiguous sub-weight: point
   `wq`/`ws` at its first tile. Then:
   - sliding layers, message-free visits: one GEMM over all 8576 rows on `xq`;
   - message visits, and every full-layer visit: one GEMM over the first 6528 rows (`q|gate|rel`) on `xq`, plus k and v
     as separate GEMMs on their own row slices and their own inputs.
   Consumers read column slices of the output with the row stride (as Qwen's `BuildGdnAb` in docs/perf.md does).
   `head_clip` already takes a row stride; `sigmoid_gate_mul_bf16`, `sconv_static_f32` and `sconv_dynamic_f32` do not, so
   give them one (or apply the concatenation only at T = 1, where the stride does not matter). Each
   MXFP8 row is independent and the per-row result depends only on SK, so this is **bit-exact** when SK is unchanged (check
   `Mxfp8FallbackTuning` and pin SK if needed). Under TP=2 the concatenation is per rank, over the rank's shards (rel is
   replicated, inkling2-plan.md 11.2 item 5).
2. **Sink and the dynamic-conv projections** (`dense_bf16` with N = 24 and N = 32) can join the same concatenated GEMM if
   stored MXFP8. That changes their numerics (bf16 to MXFP8), so it needs the KL gate. Otherwise fuse them into one
   `dense_bf16` launch.
3. **MsgGate** (12 launches per looped visit): fuse prep + rank-64 down + apply, or at least batch the 4 gates of a visit
   into one launch each step (same input layout, 4 weight sets).
4. **Elementwise fusions**: `head_clip` into the q-slice consumer, `sigmoid_gate_mul` into the o GEMM's A-load or into
   attention's output, and the rmsnorms into the preceding `attnres`.
5. Report launches/token after each fusion.
- Gate: bit-exact items only need byte compares. Anything that changes rounding needs the KL gate (they can share one rerun
  at the end of S4).

### S5. HIP graph for the decode step

Precedent: `CaptureDecodeGraph` in `src/model/qwen4exp_stage.cpp` (capture, instantiate, replay, plus a fuse-mask check).
Needs S1 (no host syncs inside the step). Then:

1. Every position-dependent kernel argument (`pos0`, `ctx`, ring slot) must come from a **device scalar**, not a kernel
   argument baked at capture time. That means a device-pointer variant of the head ops, the conv state writes, the paged
   writes and attention.
2. Arena allocations inside the step must land at the same addresses on every step (they do if `arena_.Reset()` runs at the
   same point, but verify).
3. Capture once per (layout, TP rank) after warm-up and replay per token. Fall back to eager when capture fails, and add a
   kill switch.
- Gate: bit-exact against eager (graph replay launches the same kernels).
- Expected: removes the host enqueue time that is left. Use the S0 `host_enqueue_ms` to predict the gain.

### S6. MXFP8 GEMM tuning table

`gemm_tuning_table_mxfp8.inc` is a placeholder, so every shape uses the unverified fallback. Bench `r4dx_gemm_mxfp8_nt` (and
the S1/S2 grouped kernels) on the real shapes at M = 1, 8, 16, 64 and 256, and write measured rows:

| linear | N | K |
|---|---|---|
| expert gate_up | 2304 | 2304 |
| expert down | 2304 | 1152 |
| q / gate | 3072 | 2304 |
| k / v | 1024 | 2304 |
| o | 2304 | 3072 |
| rel | 384 | 2304 |
| S4 concatenation and its prefix | 8576 (`q,gate,rel,k,v`), 6528 (`q,gate,rel`) | 2304 |
| lm_head | 200064 (TP=1), 100032 (TP=2) | 2304 |

Copy the w8a16 table's row format and its `W8a16RowFits` rules (`Mxfp8RowFits` already exists). A tuning row can change SK
and therefore the bytes, so a new row that changes SK on a gated linear needs a KL rerun (or keep SK fixed and tune only
WV/NPW/NT/MB, which are bit-neutral per the header).

### S7 (optional). TP=2 decode

TP=2 is slower than TP=1 today because of 96 host-staged bf16 all-reduces per token. Its KL gate also misses by a small
margin, likely because of the bf16 all-reduce (the fp32 all-reduce follow-up in [inkling2.md](inkling2.md)). TP=2 matters
for long context (131072 only fits there). Options, cheapest first:
1. an fp32 all-reduce (fixes the gate miss; speed-neutral or slightly worse);
2. measure the per-call cost of `AllReduceSumBf16Rows` at H = 2304, M = 1, and see whether S1 + S5 alone bring TP=2 above
   TP=1;
3. expert parallel instead of tensor parallel for the MoE (each card holds 64 experts and only the routed rows cross), which
   removes one of the two reductions per visit. This is a larger design change: write it up before building it.

## 4. Correctness gates (every stage)

1. CPU: `.\tests\run_tests.ps1` (or `ctest -L inkling2` CPU tests). No approval needed.
2. GPU unit tests: `ctest -L inkling2_gpu` on device 1 (approval).
3. Tiny end-to-end: `test_inkling2_tiny_e2e` at TP=1 (and TP=2 emulate when TP code changed).
4. Bit-exact stages (S1, S2, S4.1, S5): a byte compare of one teacher-forced dump and of a 256-token greedy CLI output
   against the previous stage. If the bytes differ, the stage is not bit-exact: find out why before going on.
5. Non-bit-exact stages (S3, S4.2-4, S6 if SK changes): the full KL gate at TP=1, with the invocation of
   [inkling2-plan.md](inkling2-plan.md) 11.1 (`tool_teacher_forced_logprobs --layout mxfp8 --max-ctx 8192`, then
   `kl_report.py --gate inkling2-fp32`, epsilon 0.005, against the saved truth and noise in
   `E:\models\r4dx\inkling2\kl\fp32\`). Every group must PASS. Record the numbers in section 6.
6. Server smoke: `tools\server\smoke.ps1 -Dialect inkling2` at the end of S1, S3 and S5.

## 5. Out of scope here (separate milestones)

- **DSpark/MTP speculative decoding** (`mtp.safetensors`). After S1-S5 this is the largest remaining decode lever (Qwen goes
  from 37 to 123 tok/s with DFlash2). Note that the DSpark draft reads fixed layer visits.
- **fp8 KV**: frees memory at TP=1 (the 131072 context does not fit today), not a speed item at short context.
- Vision, varloop checkpoints.

## 6. Results log

Fill in as stages land (date, commit, device, prompt length, tok/s, launches/token, KL chat-ALL / worst group).

| stage | commit | decode tok/s (short) | decode tok/s (4k) | prefill tok/s (2k) | launches/token | KL chat-ALL / worst | notes |
|---|---|---|---|---|---|---|---|
| M1 (baseline from inkling2.md) | f3e58ef | 23.6 | | | | 0.00573 / tool 0.00740 | 25-token prefill 218 tok/s |
| S0 (baseline profile, 2026-11-23) | 088fa74 | 25.31 (med of 25.45/25.31/24.79) | prof wall 119.8 ms (attn.core 37.6 ms) | 242.8 (2922 tok, prof wall 12.04 s) | 2693 | (same as M1) | dev 1, `--max-ctx 8192`, 31.45 GiB; short prof: gpu_sum 69.87 ms, host_enq 91.28 ms, wall 94.53 ms (MoE 51.7 ms = 74%: router 14.9, readback 6.7, gate_up 13.9, act 4.8, down 10.1, combine 1.3); 2k prefill: 195,918 launches (attn.core 3.36 s, MoE gate_up/down/act 3.79 s across 176k calls) |
| S1 (grouped MXFP8 MoE decode + fast router, 2026-11-23) | a2138d2 | **50.66** (med of 50.79/50.66/50.33) | prof wall 66.0 ms (pre-wave32 topk) | (unchanged) | 1733 | **bit-exact** vs S0 (`en_email` SHA256 + greedy 256 identical) | 2.0x over S0 (clears >=40 target); `R4DX_INK_MOE_DECODE=0` switch; short prof: gpu_sum 20.49 ms, host_enq 1.86 ms, wall 26.46 ms (moe.readback 0, gate_up 3.51, act 0.06, down 2.08, router 1.56); tf `en_email` 44.2 -> 22.4 ms/row; VRAM 31.45 GiB |
| S2 (grouped MXFP8 MoE prefill, 2026-11-23) | 9b410a1 | 51.11 | | **395.0** (2922 tok, prof wall 7.40 s) | 1733 (prefill 21,324 vs 195,918) | **bit-exact** vs S1 (`en_email` tail-16 SHA256 + greedy 256 identical) | `R4DX_INK_MOE_PREFILL=0` switch; 2k prefill 1.63x faster (MoE gate_up 1.74->0.71s, down 1.59->0.38s, act 0.45->0.22s, readback 0); `attn.core` is now 46.2% (3.31s) + `attn.rel_sink` 7.6% (0.55s) |
| S3 (fused h128 attention + on-the-fly bias, 2026-11-23) | pending | **53.88** (med of 53.66/53.88/54.36) | prof wall **29.1 ms** (attn.core **2.63 ms** vs 35.9 ms) | **626.4** (2922 tok, prof wall 4.66 s) | 1685 (short) / 1733 (4k) | **0.00567 / 97.73%** (chat-thai 0.00631 / 97.35%, 0.0036 pt below unrounded 97.351%; user accepted) | `R4DX_INK_ATTN_FUSED=0` switch; 128-key 2-pass LDS decode + LDS-tiled prefill; 4k attn.core 13.6x faster, 2k prefill 2.58x over S0 (`attn.rel_sink` 547->49ms, `attn.core` 3.31->1.01s); tf 20.6 ms/row; beats M1 on all KL metrics and overall top-1 |
