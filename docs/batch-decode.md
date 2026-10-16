# Batched decode (`--batch N`)

Decode N sequences at once: one forward pass per token for all of them, so the weight stream (the cost that sets
the single-sequence 37 tok/s) is paid once per step instead of once per sequence. The server serves up to N
requests concurrently; `--tp 2` and the hybrid mode (`--tp 2 --pp 2`) support it.

> **Validation status.** Written in a session with no GPU and no ROCm. What has actually run: the HIP-free pieces
> (`test_batch_plan_cpu`, `test_batch_executor`, `test_batch_port`, the `--batch` argument tests -- all built and
> run, the threaded ones under ThreadSanitizer and AddressSanitizer too) and a syntax-only compile of every touched
> `.cpp` against a stub of the HIP runtime header. **No kernel has launched.** The GPU tests
> (`test_batch_decode`, `test_batch_decode_tp`, `test_batch_decode_tp_hybrid`, `test_engine_batch`) are written,
> compile-checked the same way, and have never met a GPU or the tokenizer directory. Section 8 says how to run them
> and what to read into a failure. No throughput number in this document is measured.

## 1. Scope

| | |
|---|---|
| Flags | `--batch N` (0 = off, the default, byte-for-byte the one-request-at-a-time server; N in [2, 16], at most 8 with `--tp 2`: a TP step is never split into submission units, docs/tp.md N80), `--batch-ctx N` (tokens per slot, a multiple of 16, default 32768) |
| Works with | `--tp 1`, `--pp 2` (decode runs on the decode card), `--tp 2`, `--tp 2 --pp 2` (hybrid), every `--layout`, images (a slot carries its conversation's mrope delta), sampled and greedy requests, streaming, tools, thinking |
| Not with | `--mtp`, `--dflash` (a batched step decodes one plain token per sequence; refused at startup), Gemma 4 (`BatchSlots() == 0`; refused at startup) |
| Changes under `--batch` | the prompt checkpoint is off, and a request never reuses another's prefix: every request prefills its whole prompt (a request owns a slot, not the model) |
| Unchanged | `Prefill`, `DecodeStep*`, `Reset()`, the single-sequence KV and GDN state, every kernel's launch for a non-batch caller |

## 2. Design

A **batch slot** is one sequence's decode state: in every full-attention layer `batch_ctx` tokens of one shared fp8
`PagedKvCache` (slot `s` owns cache positions `[s * batch_ctx, (s + 1) * batch_ctx)`), and in every GDN layer physical
slot `s + 1` of a `GdnStateManager` of its own (physical slot 0 is reserved, `gdn_state.h`). The slots live next to
the single-sequence state and never alias it, so `Model::Reset()` of the single-sequence state cannot touch a slot
some other request is decoding in.

```
Prefill(prompt)         the usual single-sequence prefill (hybrid pipeline, int8 chunks, images: whatever the TextModel does)
BatchImport(slot)       copy that state into the slot: per attention layer the first ceil(pos / 16) KV blocks (one contiguous
                        D2D copy, the cache layouts match), per GDN layer the fp32 recurrent state and the conv history
DecodeBatch(rows)       row r = (slot, the token to feed, sampling params, rng): one forward over all rows, one next token per row
BatchRelease(slot)      host bookkeeping; the next import overwrites everything
```

Prefilling into the single-sequence state and *copying* (rather than prefilling into a slot directly) keeps every
existing prefill path -- the 256-row int8 chunks, split-KV, images, the PP-2 pipeline and the hybrid reshard --
exactly as it is. The copy is a few MB of GDN state and `pos * 32 KiB` of KV (TP=1), against a prefill that took
seconds.

`batch_plan.h` (HIP-free, tested on the host) holds the bookkeeping: `SlotTable`, the per-step metadata layout and
its builder, the validation, the lockstep fingerprint and the memory formulas.

## 3. One batched step

`Model::DecodeBatch` is `VerifyWindow`'s shape with the rows belonging to different sequences: one embedding gather
of `T = rows` tokens, the 64 layers, the final norm and lm_head over `T` rows, one argmax launch over all rows.
Everything row-independent -- RMSNorms, every projection (`ApplyLinear` at M = T), the gate, the residual, the MLP,
the all-reduces -- sees all rows in one launch. Two things are per sequence.

One async H2D copy (2 KiB, pinned host to a persistent device array, the way `step_meta.h` does it) stages the whole
step's metadata: token ids, rope positions, KV slot mapping, `seqused_k`, the GDN `cu` / `cache_idx` / `sidx`, and
(only when a row has an image in its conversation) the 3-axis rope rows.

### 3.1 GDN layers

The libr4d GDN decode kernels already take `N` sequences (`cu_seqlens`, per-sequence `cache_idx` and state indices
-- `r4d_gdn_conv_update_w4_h128_bf16`: one workgroup per sequence; `r4d_gdn_recurrent_update_*`: grid `(N, H, ...)`).
`GdnLayerParams::num_seqs > 0` makes `GdnLayer::Forward` pass `N = num_seqs` and the three per-step device arrays
instead of the single-sequence ones the `GdnControlCache` serves (which is also why a batch step needs nothing from
that cache, and so nothing to prewarm before the TP warm-up freezes it). The batched call is a plain decode: window 1,
no write-once state, no `num_accepted`.

### 3.2 Attention layers

`AttentionLayer::Forward` takes an `AttnBatchView`. Rows are `T` sequences with one query each; the KV write goes to
`slot_mapping[r] = slot * batch_ctx + position` (so the fused precore launch, which uses one array for the rope
position and the slot, is not taken; the unfused chain with separate arrays is). The attention core then runs **one
plain `num_seqs = 1`, `q_len = 1` decode launch per row**, against that row's slice of the cache's identity block table
(`block_table + slot * blocks_per_slot`) with `max_ctx` set to the single-sequence cache's capacity, so the
split-KV law picks the split count (hence the reduction order) a lone decode of this Model takes.

That is deliberately the conservative choice: a row's attention output is the single-sequence decode's, launch for
launch, which is what makes the equality claim in section 6 a statement about kernels that already run, not about a
new launch shape. The price is `16 x rows` core launches per step instead of 16. The alternative -- one launch with
`num_seqs = rows` and a per-step block table -- is listed in section 9.

### 3.3 Sampling

Each row carries its own parameters and generator. All rows are argmaxed in one launch pair (`r4dx_argmax_rows_f32`,
each row's answer the one-workgroup kernel's, bit for bit). A sampled row additionally gets a device row summary at
its own `1 / temperature` (`LaunchRowSummaries` with `out_row0`, one call per row into consecutive summary rows) and is
resolved exactly as `DecodeStepSampled` resolves a row: exactly one draw from the row's rng, `SampleFromSummary`, and
when the summary cannot prove the answer the row's full fp32 logits are fetched from this step's logits buffer and the
canonical sampler runs with the same draw. A temperature too small for a summary (`kMinSummaryTemperature`) takes the
full-row path directly. A greedy row draws nothing.

## 4. Memory

Per slot, at `--tp 1` (the 27B: 16 full-attention layers x 4 KV heads x 256, 48 GDN layers):

| | |
|---|---|
| KV | `batch_ctx x 32 KiB` (K and V, fp8) |
| GDN state | 48 layers x (3 MiB fp32 recurrent + a 60 KiB conv line) ~ 147 MiB, plus one more reserved slot per layer |

so `--batch 4 --batch-ctx 32768` is 4 GiB of KV + about 0.7 GiB of GDN state. Under `--tp 2` each rank holds half of
the KV (2 KV heads) and half of the GDN state. The load log prints the total (`batched decode: N slots x M tokens,
X GiB`); `batch::BatchKvBytes` / `BatchGdnBytes` are the formulas. A prompt longer than `batch_ctx - 1` tokens is
answered `400` (`exceeds the batch context --batch-ctx`), and `max_tokens` is clamped to what the slot has left.

The slots are allocated at load, so the hybrid mode's stage-KV planning (`--hybrid-ctx auto`) already sees the
memory they took. Size `--batch-ctx` first, then check the hybrid's reported stage capacity.

## 5. Tensor parallel and the hybrid mode

`TpModel::BatchImport` and `DecodeBatch` are collective commands like any forward call (`RunCollective`: state
check, per-rank closure, failure -> `kNeedsRecovery`). Every rank holds the same slots over its own shards.

* **Lockstep.** A step checks `{kind, rows, StepFingerprint(rows, tokens, positions), per-row mode mask}` across
  ranks before its all-reduces are enqueued (`TpComm::CheckLockstep`), the way `VerifyWindow` does.
* **Merges.** The `T` greedy pairs and the summarised rows' summaries cross the host in ONE all-gather
  (`MergeShardResults`); an unresolved sampled row is gathered full-width (`GatherVocabRow`). All ranks reach the
  same token because they hold identical rng copies: `TpModel::DecodeBatch` hands each rank a copy of every sampled
  row's generator, compares the copies afterwards (`RequireRngsEqual`) and gives the caller's generators rank 0's.
* **Warm-up.** `Model::TpWarmup` (and, at TP=1, the end of `Model::Load`) runs `BatchWarmup`: every row count `1..N` greedy, then one
  all-sampled step, so no kernel's first use and no first merge happens inside a request.
* **Hybrid.** The PP-2 prefill reshards its state into the two rank Models at the end of the call. `BatchImport` runs
  after that, on the ranks, so a slot receives the same state a plain `--tp 2` prefill would have left: **the slot never
  knows which prefill built it.** A batched step runs on the ranks alone and leaves the single-sequence state where it
  was, so the hybrid's stage mirrors (`NoteTpMoved`) have nothing to be told. Stage Models are loaded with
  `batch_slots = 0`. Because each request resets the single-sequence state first, the hybrid never sees a warm
  continuation under `--batch`; its warm gather is simply not used.
* **`--pp 2` alone.** `PpModel` forwards to the decode Model (stage B), where the pipelined prefill leaves the state.

## 6. Numerics: what is claimed

For one sequence, a batched step is meant to produce **the same token** the single-sequence step produces, for greedy
rows exactly and for sampled rows with the same draw. The argument, kernel by kernel:

| Part | Why a row's bits do not depend on the other rows |
|---|---|
| Embedding, RMSNorm, residual, rope, KV write, gate | row-independent kernels (the existing 64-row verify window relies on the same) |
| Attention pre-core (split_qg, q/k norm, rope, KV write) | a batched step runs the five-launch chain (the KV slot differs from the rope position, which the fused `r4dx_attn_precore_bf16` cannot express); the single-sequence decode takes the fused launch, documented as the same bytes wherever the norm took its vector path (`kernels.h`, `r4dx_attn_precore_supported`) -- head_dim 256, which is always |
| Projections, MLP, lm_head | the skinny GEMMs reduce each output column over K independently of M; the comment in `gdn_layer.cpp` states this for the bf16 GEMM and `ApplyLinear` slices a 256-row call into 64-row launches on the same premise |
| GDN conv / recurrent update | one workgroup per sequence; the `N`-dependent launch geometry (channel grouping, v-row splits) only regroups per-channel work |
| Attention core | the single-sequence decode launch, once per row, same `max_ctx` hence same split count |
| Argmax | the multi-row kernel is documented bit-identical to the one-workgroup kernel |
| TP all-reduce | exact adds of two partials in rank order, independent of the row count |

What this does **not** prove: that the GEMM tuning table (`gemm_tuning_table*.inc`, keyed by M) picks a kernel at
M = 3 whose reduction order equals the M = 1 kernel's. If it does not, a row's logits differ in the last bits and a
near-tie in the argmax can flip. `test_batch_decode` compares token chains exactly and, on a mismatch, prints the
first diverging step and the alone run's top-2 logit gap there: a gap of a few bf16 ulps is that effect, a large gap
is a bug. `R4DX_DECODE_LEGACY` knobs are untouched by this feature.

The single-sequence state is not read by a batched step and is not written by one.

## 7. Serving

### 7.1 Threads

```
HTTP threads --Submit--> BoundedQueue --Pop--> request thread 0..N-1 (Engine::BatchWorkerLoop, slot = thread index)
                                                     |   each runs the ordinary Engine::RunRequest
                                                     v   through its own BatchPort (a TextModel)
                                            BatchExecutor (ONE thread, owns the TextModel)
                                               jobs:  Reset+Prefill+BatchImport, EncodeImages, BatchRelease
                                               steps: rows from several requests -> ONE TextModel::DecodeBatch
```

`RunRequest` is unchanged except for three substitutions: `model_` becomes the request's port, the prefix state
becomes a per-request one that always says "reset" (an empty `PrefixState` matches any prompt, so it is
`Invalidate()`d up front), and the TP group's health is read through the executor (it belongs to the thread that owns
the model). Tool parsing, stop strings, reasoning splitting, streaming and the request log are exactly the single-
sequence code, per request thread.

`TextModel` is single-caller (`TpModel` is facade-thread-only, `PpModel` binds a device per call, HIP's current device
is per thread), which is why every model call funnels through one thread; the executor is that thread, as the old
worker thread was.

### 7.2 What a request does (`BatchPort`)

`Reset()` takes the **primary lock**: one request at a time owns the single-sequence state between its `Reset()` and
the end of its prefill. `Prefill` / `PrefillMultimodal` is ONE executor job -- the real prefill, then
`BatchImport(slot)` -- after which the lock is released and the slot *joins* the decoding set. Each decode call
(`DecodeStepGreedy`, `DecodeStepGreedyOverlap`, `DecodeStepSampled`) posts one row and blocks for its token;
the `Overlap` form runs the caller's host work (decode, stop-string scan, streaming) while the step is on the device and
rethrows a callback exception only after the step has finished. The port's destructor releases the slot and the primary
lock however the request ended. Speculative rounds, the full-logits `DecodeStep` and the checkpoint calls throw (batch
mode never reaches them).

### 7.3 Scheduling, and what a prefill costs the others

The executor alternates: after a decode step, one waiting job runs, then steps resume; with no steps waiting, jobs run
back to back. **A prefill is a stall for every decoding request**, for as long as the prefill takes (a long prompt:
seconds). This is *serialized prefill*, the simplest correct policy: the batched decode never waits on a half-built
slot, and the prefill paths stay the existing, validated ones. Chunked prefill interleaved with decode steps would
bound the stall and is a follow-up.

Before each step the executor waits up to `EngineOptions::batch_gather` (2 ms, not a CLI flag) for the sequences that
are still between tokens, so a step starts as soon as every decoding sequence has posted its row, or when the window
closes. "Decoding" means joined (prefill imported) and not yet left. A request that ends on its first token leaves
without ever posting.

### 7.4 Failure

A `DecodeBatch` that throws fails every row of that step (`500` for each request in it); `Model::DecodeBatch`
releases those slots first (their state is unknown), the executor keeps running and the other requests -- those
still in prefill, or not in this step -- continue. Under TP the group is then `kNeedsRecovery` until a request's
`Reset()` recovers it; requests that were mid-decode fail at their next step with the state error. A prefill failure
fails only its request. A failed `BatchImport` leaves its slot free on every rank (`TpModel::BatchImport` releases it everywhere). Input errors (a slot that is not live, a repeated slot) throw before anything is enqueued;
under TP they still make the group `kNeedsRecovery` because the validation runs inside the collective -- the engine
never produces them.

## 8. Tests and how to run them

| Test | Needs | Status |
|---|---|---|
| `test_batch_plan_cpu` | nothing | built and run |
| `test_server_args` (`TestBatchFlags`) | nothing | built and run |
| `test_batch_executor` | nothing (real threads, fake decode) | built and run, also under TSan / ASan, 25x at -O2 |
| `test_batch_port` | nothing (real executor + threads, `FakeBatchModel`) | built and run with a stubbed HIP header (the test never calls HIP), TSan / ASan / -O2 x30 |
| `test_engine_batch` | tokenizer dir (`R4DX_TOKENIZER_MODEL_DIR`), links the server | compile-checked only; 8 requests through 3 slots equal the one-at-a-time texts, steps really batch, one model thread, no interleaved prefill, slots released, `400` for an oversized prompt, `500` for the rows of a failed step |
| `test_batch_decode` | 4-layer container, GPU | compile-checked only; chains alone == batched (lockstep, irregular schedule, sampled mix), state untouched, slot reuse, refusals |
| `test_batch_decode_tp` (`--rig emulate`) | 4-layer container, GPU | compile-checked only |
| `test_batch_decode_tp_hybrid` (`--rig hybrid`) | 4-layer container, both GPUs | compile-checked only; each prefill must run the pipeline (`HybridStats::pipelined_calls`) |

Run order for the first GPU session: `test_batch_decode` (single device, exercises every new kernel path: the batched
GDN launches, the per-row attention launches, the KV and state copies, the meta layout), then
`test_batch_decode_tp` and the hybrid rig, then `r4dx-server --batch 4` with `tools/server/smoke.ps1` (which does not
know about `--batch` yet: send N concurrent requests and compare each answer with a `--batch 0` server's, greedy).
A first failure is most likely one of: the meta offsets (`batch_plan.h` pins them on the host, but not their use on
the device), an argument-order slip in a launch I could only compile against stubs, or the GEMM-tuning effect of
section 6.

## 9. Not done, and follow-ups

* **One attention launch for all rows.** Needs a per-step block table (row `r` -> slot `s_r`'s blocks, `rows x
  blocks_per_slot` int32) or rows packed as slots `0..rows-1`; then `num_seqs = rows` replaces the per-row loop. Worth
  measuring the loop's cost first.
* **Chunked prefill interleaved with decode** (section 7.3), and prefilling *into* a slot to drop the import copy.
* **Prefix reuse across requests** in batch mode (a slot could keep its sequence after the request ends).
* **Speculative decoding** (MTP / DFlash2) per sequence: the write-once GDN log is single-sequence by construction.
* **`r4dx-cli`**: no batch mode; the server is the only consumer. No `--batch` in `smoke.ps1`.
* **TP input validation on the facade** so a caller error does not need a group recovery (section 7.4).
* Throughput and latency numbers: none measured.
