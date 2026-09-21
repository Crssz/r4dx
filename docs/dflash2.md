# DFlash2 drafter -- Python reference, conventions, and r4dx integration plan

Status: **Python/CPU/fp32 reference implementation done and golden-fixture-tested** (this
document + `tools/reference/dflash2_ref.py` + `tools/reference/gguf_min.py` +
`tools/reference/dflash2_selftest.py` + `tools/reference/golden_out/dflash2/*`). No C++/HIP code
exists yet -- this is Task A2 (the exact-reference + fixtures task); the GGUF reader and draft
container (a separate, concurrently-developed task) live under `src/convert` and are **not**
touched by anything in this document or its accompanying tools.

Everything below was resolved by reading the ROCmFPX reference implementation line-by-line, not
guessed, against the ROCmFPX checkout on this machine (a sibling checkout next to this repo's own
worktrees, not a path this document hardcodes -- see the standing rule against committing
`C:\Users\<name>\...` paths): `src\models\dflash.cpp` (`load_arch_hparams`/
`load_arch_tensors`/`build_dflash2_conv`/`build_dflash2_selector`/`graph<false>`, lines 1-840),
`common\speculative.cpp`
(`common_speculative_impl_draft_dflash::process`/`draft`, lines 909-1274, and the verify-region KV
rollback, lines ~1520-1573), `conversion\qwen.py`'s `DFlashModel`
converter (lines 640-773), `gguf-py\gguf\{constants.py,gguf_writer.py}`
(exact GGUF metadata key strings), `src\llama-model.cpp` (n_rot default,
rope type selection, SWA mask semantics, `llama_hparams::is_masked_swa`), and this repo's own
`third_party/libr4d/r4d.h:116-125` + `r4d_dflash_conv_body.h` (the fused conv kernel's own spec
comment, which independently confirms the conv formula derived from the ggml graph). That reference
code is read-only and is never copied verbatim into anything committed here; this document and
`tools/reference/dflash2_ref.py` describe/reimplement the math, they do not vendor the source.

## 1. What DFlash2 is

DFlash2 is a **block-diffusion self-speculative drafter**: instead of an autoregressive draft head
(DFlash1/EAGLE-style, one token at a time), a small (~1.9B param) 5-layer transformer takes a
"noise block" -- the last committed real token followed by `block_size - 1` copies of a special
`<mask>` token -- and produces up to `block_size - 1` draft tokens for the *whole block at once* via
non-causal attention plus a learned **selector lattice** that greedily chains the most likely
token at each block position conditioned on its predecessor. It never runs its own embedding table
or LM head: both are the TARGET's (`Qwen3.8-27B`'s `embed_tokens`/`lm_head`), gathered/matmul'd
directly. Its own attention layers read a **separate KV cache** that is *not* keyed by draft
tokens at all -- it is continuously fed the TARGET's own mid-stack residual-stream features (5
fixed layers' worth, concatenated) for every real committed token, via a small encoder (`fc` +
`output_norm_enc`) and a per-layer K/V projection. This is why the container calls the two GGUF
batch types "embd" (feature injection) and "token" (noise-block drafting) -- `dflash.cpp`'s
`graph<false>` is genuinely two different forward passes selected by `ubatch.embd != nullptr`.

## 2. Resolved conventions (compact table for the container-format reviewer)

| Question | Resolution | Settled by |
|---|---|---|
| `n_rot` (no `dflash.rope.dimension_count` key) | **128 (full `head_dim`)** | `src/llama-model.cpp:1343-1346`: `hparams.n_rot_full = hparams.n_embd_head_k_full` (=128) as the default, only overridden if the (absent) key exists. |
| RoPE pairing | **GPT-NeoX split-half**: pair `(i, i+64)` for `i` in `[0,64)`, single scalar (temporal) position per token, `theta=1e7` | M-RoPE sections `[64,0,0,0]` with `n_rot/2=64` puts every one of the 64 frequency pairs in section 0 (temporal); sections 1-3 (height/width/extra) get zero pairs. `ggml_rope_multi`'s per-section assignment plus NeoX-style `(i, i+n_rot/2)` pairing (the same pairing `LLAMA_ROPE_TYPE_MROPE`/`NEOX` both use) therefore degenerates to plain single-position NeoX rope over the full 128 dims. Confirmed by `src/llama-model.cpp:3258-3264` (`LLM_ARCH_DFLASH` returns `LLAMA_ROPE_TYPE_MROPE` whenever any section is nonzero) and `qwen.py:726-730` (`add_rope_dimension_sections([head_dim // 2, 0, 0, 0])` -- the converter's own comment: "the draft ropes on the temporal dim only"). |
| RoPE base | `1e7` | `dflash.rope.freq_base` GGUF key, read directly (verified against the real file: `10000000.0`). |
| q_norm/k_norm before or after RoPE | **Before** | `dflash.cpp:731-735`: `Qcur = build_norm(...); Kcur = build_norm(...); Qcur = build_rope(Qcur, ...); Kcur = build_rope(Kcur, ...)` -- norm strictly precedes rope, both in the draft block and in the injection path (`Kcur = build_norm(...); Kcur = build_rope(...)`, line 631-632). |
| RMSNorm convention | **Plain** `x * rsqrt(mean(x^2)+eps) * w` -- **not** the target's zero-centered `(1+w)` `Qwen3_5RMSNorm` convention `docs/architecture.md` documents for the TARGET model | Every norm site in `dflash.cpp` (`attn_norm`/`ffn_norm`/`output_norm`/`output_norm_enc`/`attn_q_norm`/`attn_k_norm`, injection and draft alike) calls `build_norm(cur, w, NULL, LLM_NORM_RMS, il)`, ggml's plain `ggml_rms_norm` + `ggml_mul`, no offset. DFlash2 is a llama.cpp-native GGUF checkpoint (z-lab/Qwen3.8-27B-DFlash2), not the HF Qwen3_5 target checkpoint -- the two models' norm conventions are independent. |
| SWA visibility rule | **Visible iff `query_pos - key_pos < sliding_window (2048)`**, both directions (the draft's own block is fully non-causal, so a key in the query's future is never separately masked) | `attention.causal=false` (GGUF key, confirmed `False` in the real file) skips llama-graph.cpp's "mask future tokens" branch entirely; `llama_hparams::is_masked_swa` (`LLAMA_SWA_TYPE_STANDARD`) masks iff `p1 - p0 >= n_swa` where `p1`=query pos, `p0`=key pos (`llama-graph.cpp:417-439`) -- everything else (including a "future" key relative to the query) stays visible. The block's own 8 positions are always mutually visible (max offset 7 << 2048); only injected-cache positions older than `query_pos - 2048` are ever excluded (exercised by fixture B, N=2100). |
| Conv delta layout | **group fastest, then tap, then side** (column index `c = side*(kernel_size*n_groups) + tap*n_groups + group`) | `dflash.cpp:432`: `ggml_reshape_4d(ctx0, dynamic, n_groups, kernel_size, 2, n_tokens)` -- ggml's `ne[0]` (fastest/contiguous) is `n_groups`. |
| Conv base layout | `base[side][tap][channel]` (channel fastest in the raw GGUF tensor, `ne=[H, kernel, 2]`; this reader's `numpy_shape` reverses that to `(2, kernel, H)`, i.e. `base[side, tap, :]`) | `dflash.cpp:235-238`: `create_tensor(..., { n_embd, kernel, 2 })` (ggml `ne` order); `r4d_dflash_conv_body.h`'s header comment independently states the same `base[side][tap]` indexing for its fused kernel. |
| Conv formula | `out[t,c] = (base[0,c]+dyn[t,0,g])*h[t,c] + (base[1,c]+dyn[t,1,g])*h[t-1,c]*[t>=1]`, `g=c//16` | `third_party/libr4d/r4d_dflash_conv_body.h:6-9` (exact header comment) and independently re-derived from `build_dflash2_conv` (`dflash.cpp:405-471`) -- both agree bit-for-bit. |
| Shift-by-one / block-start masking | `tap`'s contribution is zeroed for local block position `t < tap` (`t` is the position *within this one block*, 0-based; a fresh block always starts the shift at zero, no cross-round leakage) | `r4d_dflash_conv_body.h:97`: `if ((t & blockmask) >= tap)`; degenerates to plain `t >= tap` when the kernel is given exactly one block of `T == block_size` tokens (this reference's `dflash2_conv` never batches multiple blocks, so `t & blockmask == t` always). |
| Selector score formula | `score[a,b] = sum_r successor[cand[t][b], r] * predecessor[P[a], r] * gate[t][r] + unary[t][b]` | Re-derived from `build_dflash2_selector`'s `score_run` (`dflash.cpp:510-540`: `cond = predecessor * gate` (broadcast), `score = mul_mat(successor, cond)` -> ggml convention `C[b,a] = sum_r successor[r,b]*cond[r,a]`, i.e. `score[a,b]` in row-major terms after the `[top_k*top_k]` flatten order `a*top_k+b`) and cross-checked against the CPU consumer `speculative.cpp:1236-1237` (`scores = row + top_k + predecessor*top_k`, i.e. row-major `a`-major, `b`-minor -- matches). |
| Predecessor set `P` | `t=1`: `P = {anchor_id}` (the block's own committed seed token, a single id, not a candidate set); `t>=2`: `P = cand[t-1]` (previous position's own top-k ids) | `dflash.cpp:546-558` (`build_dflash2_selector`): position 1's `score_run(1, 1, anchor_ids)` reads `tokens`' own anchor row; positions `2..block_size-1`'s `score_run(2, block_size-2, prev_ids)` reads `cand_blk` at the previous position. |
| Greedy walk indexing | `predecessor` (an index into the CURRENT `P`, initialized 0) selects `score`'s row; `argmax` over that row picks `b`; emit `cand[t][b]`; `predecessor <- b` (now an index into the NEW `P = cand[t]` for the next iteration) | `speculative.cpp:1234-1251`, transcribed verbatim into `tools/reference/dflash2_ref.py::draft_round`'s walk loop. |
| Early stop (`p_min`) | `prob = 1 / sum(exp(scores_row - scores_row.max()))` (softmax of the arg-maxed row, evaluated **at** the argmax); if `prob < p_min`, stop **without** emitting that position's token (nothing after it drafts either) | `speculative.cpp:1241-1249`. |
| Block/anchor/rollback lifecycle | See section 5 below. | `speculative.cpp:909-1274` (`draft()`); the DRIVER (not the DFlash2 impl itself) explicitly invalidates the draft-region KV every round -- `examples/speculative-simple/speculative-simple.cpp:207-213`/`281`/`335` call `llama_memory_seq_rm(ctx_dft, seq_id, ckpt.pos_max+1, -1)` right after `common_speculative_draft()` and before target verify; `tools/server/server-context.cpp:3106`/`4004`/`4051` do the same. This is a DIFFERENT call site from the DSV4/DSpark chained-head rollback at `1553`/`1641` -- see section 5. |
| Target feature capture point | Residual stream **entering** target decoder layer `L` (0-based HF index) for `L` in `{6,20,34,48,62}` == the **output** of layers `{5,19,33,47,61}` | `qwen.py:707-710`: GGUF `target_layers = [i+1 for i in HF target_layer_ids]`; `dflash.cpp:1046-1047`/`speculative.cpp:1141`: `llama_set_embeddings_layer_inp(ctx_tgt, target_layer_ids[k], true)` extracts the **input** to that GGUF-indexed layer, i.e. the residual stream as it enters layer `L`. |
| Injection math | `g = rmsnorm_enc(fc(features))` (shared across all 5 draft layers); per layer `il`: `K = rope(k_norm(Wk_il @ g))`, `V = Wv_il @ g` (raw, no norm, no rope), written at the feature's own absolute position | `dflash.cpp:606-669`. No `Wq` in this path (only K/V are ever written; there is no "injection attention", just a KV cache fill). |

## 3. RoPE, worked out in full

`n_rot = 128 = head_dim`, `theta = 1e7`. For a vector `x` of length 128 at absolute position `p`
(a single integer -- there is no separate height/width position for DFlash2, only the target's
"temporal" position, since sections `[64,0,0,0]` route every frequency pair to section 0):

```
half = 64
inv_freq[i] = theta ** (-(2*i) / 128),   i in [0, 64)
angle[i]    = p * inv_freq[i]
x1 = x[0:64]; x2 = x[64:128]
out[0:64]   = x1*cos(angle) - x2*sin(angle)
out[64:128] = x2*cos(angle) + x1*sin(angle)
```

Applied identically to Q and K (never V), per head, both in the injection path (K only, at the
feature's absolute position) and in the draft block (Q and K, at `n + t` where `n` = number of
positions already injected and `t` = the token's 0-based position within the block).

## 4. Forward pass (as implemented in `tools/reference/dflash2_ref.py`)

### 4.1 Feature encoding + KV injection (`encode_features` + `inject`)

For every already-committed real token, the caller (eventually r4dx's own target-side prefill/
verify loop) captures the target's residual stream entering layers `[6,20,34,48,62]` and
concatenates them into one `[25600]` feature vector per position:

```
g = rmsnorm(fc(features), output_norm_enc, eps)              # [N, 5120], shared across all 5 layers
for layer il in 0..4:
    K = rope(rmsnorm(Wk_il @ g, attn_k_norm_il, eps), pos, theta=1e7, n_rot=128)   # [N, 8, 128]
    V = Wv_il @ g                                                                  # [N, 8, 128], raw
    append (K, V, pos) to layer il's own KV cache
```

### 4.2 Draft block forward (`draft_round`)

```
tokens  = [id_last, <mask>, <mask>, ..., <mask>]        # block_size (8) tokens
x       = target.embed_tokens[tokens]                    # [8, 5120] -- TARGET's embedding rows
pos     = [n, n+1, ..., n+7]                              # n = positions already injected

for layer il in 0..4:
    h    = rmsnorm(x, attn_norm)
    dyn  = attn_conv_proj @ h                             # [8, 1280], computed ONCE per sub-block
    h    = dflash2_conv(h, dyn, attn_conv_base, side=0)
    q,k,v = Wq@h, Wk@h, Wv@h                              # [8,32,128], [8,8,128], [8,8,128]
    q,k  = rmsnorm(q, attn_q_norm), rmsnorm(k, attn_k_norm)
    q,k  = rope(q, pos), rope(k, pos)
    o    = attention_non_causal_swa(q, [cache_k_il; k], [cache_v_il; v], scale=1/sqrt(128), gqa=32/8)
    o    = Wo @ o
    o    = dflash2_conv(o, dyn, attn_conv_base, side=1)   # SAME dyn as side=0, computed pre-conv
    x    = x + o
    h2   = rmsnorm(x, ffn_norm)
    dyn2 = ffn_conv_proj @ h2
    h2   = dflash2_conv(h2, dyn2, ffn_conv_base, side=0)
    f    = down( silu(gate@h2) * (up@h2) )
    f    = dflash2_conv(f, dyn2, ffn_conv_base, side=1)
    x    = x + f

x_final = rmsnorm(x, output_norm)                          # res->t_embd, ALSO the selector gate input
logits  = target.lm_head @ x_final                          # [8, vocab]
cand, unary = top16(logits)                                 # per position
gate    = selector_hidden @ x_final                          # [8, 256]
```

### 4.3 Selector walk

```
P = {anchor_id}; pred_idx = 0; tokens = []
for t in 1..7:
    succ  = selector_successor[cand[t]]                     # [16, 256]
    pred  = selector_predecessor[P]                          # [len(P), 256]
    cond  = pred * gate[t]                                   # broadcast over rank
    score = cond @ succ.T + unary[t]                          # [len(P), 16] == score[a,b]
    b = argmax(score[pred_idx])
    if p_min > 0 and softmax_prob_at(score[pred_idx], b) < p_min: break
    tokens.append(cand[t][b]); pred_idx = b; P = cand[t]
```

Up to `block_size - 1 = 7` tokens per round.

**`n_min` (not modeled by `dflash2_ref.py`'s `draft_round`, but load-bearing in the reference):**
`speculative.cpp:1254-1256` discards the ENTIRE draft (`result.clear()`) if fewer than
`params.n_min` tokens survived the walk above -- so a short walk (e.g. cut off early by `p_min`)
isn't necessarily used as-is; the caller may throw it away wholesale rather than verify a too-short
block. `draft_round` itself still returns whatever the walk produced (used as-is by
`dflash2_selftest.py`'s fixture C, which is deliberately measuring the walk/early-stop behavior in
isolation); a port's driver loop must add the `n_min` discard-if-too-short policy on top before
feeding a draft to target verify.

## 5. Block/anchor/rollback lifecycle (multi-round, for the eventual r4dx integration)

1. **Prefill / every accepted real token**: capture the 5 target features, encode (`encode_features`),
   `inject` into the draft's own per-layer KV cache at that token's absolute sequence position.
   This grows monotonically and in lockstep with the real sequence -- exactly analogous to how
   `docs/mtp.md`'s MTP head's own KV cache is kept (see "Design" there for the general pattern this
   mirrors: a self-speculative head's context must be a REAL running cache, not a per-round scratch
   buffer, or its drafts degrade to near-uncorrelated noise past the first few tokens -- MTP's
   original 0-1.2%-acceptance incident was exactly this bug).
2. **Draft round**: build the noise block `[id_last, <mask>*7]` at positions `n..n+7` (`n` = number
   of positions injected so far), run `draft_round`, get up to 7 draft tokens.
3. **Verify**: feed `id_last` + the drafted tokens through the TARGET (not this document's scope --
   r4dx's own `Model::VerifyWindow`, `docs/mtp.md`, already does the analogous thing for MTP) and
   accept the longest greedy-matching prefix, `a` tokens (`0 <= a <= 7`). Before this step, the
   reference driver applies an `n_min` gate (section 4.3): if the walk produced fewer than
   `params.n_min` tokens, the WHOLE draft is discarded and verify runs on zero draft tokens instead
   -- a port needs this same discard-if-too-short policy, not just the walk itself.
4. **Commit**: reference behavior, corrected -- `process()` (`speculative.cpp:1139-1164`) actually
   injects target features for **every row of the verify batch, including rejected draft
   positions**, not just the accepted prefix; the driver then calls
   `llama_memory_seq_rm(ctx_dft, seq_id, ckpt.pos_max+1, -1)` immediately after drafting and again
   after verify (`speculative-simple.cpp:207-213`/`281`/`335`, `server-context.cpp:3106`/`4004`/
   `4051`) to explicitly delete every draft-cache cell at or beyond the last-known-good checkpoint
   position, INCLUDING the just-injected-but-rejected rows -- because llama.cpp's KV cache is
   append-only (new cells are appended, never overwritten by position), so a stale rejected row
   would otherwise sit there forever and get read by a future round's attention over the sliding
   window. Net effect for a port: only the ACCEPTED tokens' features end up durably in the cache
   after a round, but that is enforced by an explicit delete of the rejected rows, not by never
   injecting them.
   A physical ring-buffer implementation (fixed-size, position-indexed, unlike llama.cpp's
   append-only cache) can skip the explicit delete IF AND ONLY IF it upholds the invariant llama.cpp
   itself cannot rely on: every round starts at a strictly higher `n_new = n + a >= n + 1` (the
   fixed 8-wide block guarantees at least the anchor position advances even when `a=0`), so the
   NEXT round's own speculative write at `n_new..n_new+7` always physically overwrites any stale
   rejected-row bytes in the ring before anything reads them as committed data -- "self-correcting
   via position overwrite", the same invariant `docs/mtp.md`'s own KV cache documents for its
   analogous case. If r4dx's draft KV ring ever stops being strictly position-indexed and
   monotonically advancing (e.g. multiple in-flight rounds, non-monotonic retries), this invariant
   breaks and an explicit rollback becomes mandatory again, mirroring llama.cpp's own `seq_rm` calls.

   **Relaxed 2026-09-21: monotonic, but no longer contiguous.** `r4dx::model::DflashDraft` now keeps
   a second counter next to `InjectedCount()`: `ValidFrom()`, the first position whose ring contents
   are valid. **The visible store is the contiguous run `[ValidFrom(), InjectedCount())` intersected
   with the sliding window**, and that lower bound is passed to `r4dx_dflash_attn_bf16` as its
   `store_begin` argument (section 6b). `InjectFeatures(rows, start_pos)` accepts any
   `start_pos >= InjectedCount()`:
   * `start_pos == InjectedCount()` -- the ordinary append. `ValidFrom()` does not move; everything
     above behaves exactly as before, and `ValidFrom()` is 0 for the whole life of a drafter that is
     only ever fed this way (the CLI, every test, every warm server session).
   * `start_pos > InjectedCount()` -- a **gap**. The driver stopped feeding this drafter for a while
     and resumed higher up; `ValidFrom()` and `InjectedCount()` both jump to `start_pos` before the
     rows are written. The skipped positions' bytes are stale, and are simply never read again --
     the identical "self-correcting via position overwrite" argument, with an explicit lower bound
     in place of the implicit 0, so no clear and no rollback is needed for them either.
   * `start_pos < InjectedCount()` still throws. That is the rollback direction this whole section
     exists to rule out.

   The only production driver that opens a gap is `r4dx-server`: `Model::SetDflashInjectionEnabled`
   (`src/model/model.h`) lets a request turn the target-feature capture and the injection off
   together, and `Engine::RunRequest` turns them off for every `temperature>0` request, which will
   never draft. The next greedy request re-enables them, its first `RunChunk` injects at
   `start_pos = pos_ > InjectedCount()`, and that round drafts from a cold ring -- correct by
   construction (verification still runs the real model over every candidate), just with lower
   acceptance until the ring refills. See `docs/server.md` for the measured numbers on both sides of
   that trade. After ANY injection `InjectedCount() == Model::pos_` holds again, which is what
   `Model::DecodeStepDflashGreedy` checks on entry; it refuses to run at all while injection is
   disabled, because it would otherwise build its block at the drafter's stale frontier.
5. **New anchor**: `id_last_new` = the token sampled at the last accepted row (either the last
   accepted draft, or a "bonus"/corrected token beyond it, exactly as `docs/mtp.md`'s own verify
   step produces) -- becomes the next round's block position 0.

## 6. Cost model on the R9700 (from the task's own projection, restated for the integration plan)

Draft weights ~1.9B params; at w4 (a future r4dx quantized layout for this drafter, not yet built)
~0.95 GB/round read (~1.6 ms) + target `lm_head` over 8 rows (~1.2 ms) + small kernels; verifying 8
rows costs about one normal decode step (~26 ms). ROCmFPX measured **84% acceptance and 120 tok/s**
against 35 tok/s plain decode on this exact card and draft model. r4dx's own MTP head (a much
smaller, purpose-specific module, `docs/mtp.md`) currently reaches 68 tok/s at its best `K`. DFlash2
is projected to be the single largest remaining speedup lever in the roadmap (see
`docs/status.md`'s "Next milestone" list, which names it explicitly).

## 6a. Implementation (Milestone 5 B1, GPU stage, 2026-09-20)

**Target feature capture (item 1 of `docs/status.md`'s Milestone 5 GPU-stage list) is DONE and
tested on real hardware; the rest of B1 (draft module, new kernels, GPU-side tests, anchor dump,
perf measurement) is NOT done this pass -- see `docs/status.md`'s Milestone 5 section for the full
"what shipped / what's left" accounting and why.**

**B2 update (2026-09-20)**: a follow-on pass attempted task B2 ("wire the drafter into generation,
prove it lossless, measure it") and confirmed this section's own gate still holds -- every B2 item
except the unrelated item 8 (`--mtp-draft-head` default flip) needs the draft module and three
kernels below to exist first, and they still do not. See `docs/status.md`'s dated "task B2" section
(above the B1 section in that file) for exactly what B2 did and did not attempt and why.

**Review fix pass (2026-09-20)**: an adversarial review of B1+B2's combined diff confirmed the
blocker above (the drafter does not exist -- nothing new here) and additionally found the shipped
feature-capture hook itself had three lifecycle/coverage gaps, all fixed this pass, real hardware,
HIP device 1:
1. **`Model::Reset()` did not invalidate `dflash_feature_rows_`** -- a capture attached across a
   `Reset()` (e.g. a server prefix-mismatch or `--chat` turn boundary) would let a caller read the
   PREVIOUS sequence's stale captured rows. Fixed: `Reset()` now zeroes `dflash_feature_rows_` too,
   alongside its existing `mtp_seed_valid_`/`mtp_num_accepted_valid_` invalidations.
2. **The `VerifyWindow` capture call site had zero test coverage** (only the `RunChunk`/`Prefill`
   half was tested) -- fixed: `tests/model/test_dflash_feature_capture.cpp`'s new Part 3 attaches a
   capture on an MTP-enabled `Model` (`D:/models/r4dx/qwen38-27b-l4-allmtp.r4dx`, `--mtp 3`), drives
   one `VerifyWindow` call, and asserts `DflashFeatureRows()==candidates.size()` and bit-exactness
   against `EmbeddingGatherHost` -- **[PASS]**, real hardware.
3. **Multi-chunk `Prefill()` silently discarded every chunk's captured rows except the last** --
   `dflash_features_dev_` is sized `max_chunk_` rows (not `max_ctx` -- a `max_ctx`-sized buffer
   would cost gigabytes of VRAM per attached target layer at `max_ctx=262144`), and every internal
   `RunChunk` call rewrote it starting at row 0, so a prompt over 64 tokens lost all but the tail
   chunk's features. Fixed: `Model::Prefill` gained an optional `on_chunk_captured` drain callback,
   invoked once per internal `RunChunk` call (in order) before the next chunk overwrites the buffer
   -- a caller that needs every prefilled position's features (this drafter does) passes the
   callback and drains `DflashFeatureBuffer()`/`DflashFeatureRows()` inside it (e.g. injecting that
   chunk's rows into the draft's own KV store). `tests/model/test_dflash_feature_capture.cpp`'s new
   Part 4 exercises a 130-token prompt (3 `RunChunk` calls: 64/64/2) and confirms all 130 rows are
   recoverable bit-exact via the callback -- **[PASS]**, real hardware. Callers that pass no
   callback (the default, `nullptr`) get byte-identical behavior to before this fix.

Also fixed as part of the same pass: `ModelOptions::mtp_draft_reduced_vocab`'s struct-literal
default flipped `true`->`false` to match the CLI/server flag default flip B2 already made (was
previously inconsistent -- see `src/model/model.h`'s own updated comment); the matched-K=3
`--mtp-draft-head` measurement was redone directly (the previous `65.02-67.34 tok/s` citation
mixed two different, non-matched-K sweeps) -- see `docs/mtp.md`'s "Flag default flipped" section
for the real matched-K numbers. None of these fixes touch the drafter itself; the blocker above is
unchanged.

`Model` (`src/model/model.h`/`model.cpp`) gained:

- `Model::AttachDflashFeatureCapture(std::vector<int64_t> target_layers)` /
  `DetachDflashFeatureCapture()` / `DflashFeatureCaptureAttached()` -- `target_layers` is the same
  0-based layer-INPUT-index convention this doc's section 7 (below) and
  `DflashDraftWeights::Config().target_layers` already use (e.g. `{6,20,34,48,62}` for the real
  container). Attaching (re)allocates a `[max_chunk_, target_layers.size()*hidden]` bf16 device
  buffer sized for the largest possible call (a 64-row prefill chunk); detaching frees it and
  restores byte-identical pre-B1 behavior.
- `Model::DflashFeatureBuffer()` / `DflashFeatureRows()` / `DflashFeatureCols()` -- the captured
  buffer plus the row count of whichever `RunChunk`/`VerifyWindow` call most recently ran (valid
  until the next such call, same lifetime convention as `mtp_last_hidden_`).

**Where it hooks in**: one call site inserted at the top of each layer iteration in BOTH
`Model::RunChunk`'s layer loop (covers prefill chunks AND plain decode, since `DecodeStep` funnels
through `RunChunk` with `T=1`) and `Model::VerifyWindow`'s layer loop (covers MTP-style verify
rounds, `T<=mtp_draft_k_+1<=8`) -- placed BEFORE that iteration's `GdnLayer`/`AttentionLayer`
`::Forward` call mutates `cur` in place, so `cur` at that point is exactly the residual stream AS
IT ENTERS layer `i`. A free function `CaptureDflashLayerInput` (anonymous namespace, `model.cpp`)
does the actual copy: a no-op (`target_layers.empty()` check, nothing else) unless `i` is in
`target_layers`, in which case one `hipMemcpy2DAsync` (device-to-device, no host sync) strides the
`[T, hidden]` `cur` slab into that target layer's column of the `[T, num_target_layers*hidden]`
destination. `Model::DecodeStepProfiled`/`PrefillProfiled` (the diagnostic-only profiling paths,
never used by the production draft/verify/commit round loop) were deliberately NOT hooked -- out of
this pass's scope, and orthogonal to what a real DFlash2 round needs.

**Why `hipMemcpy2DAsync` and not a new kernel**: it is the plain HIP runtime's strided
device-to-device copy -- no new kernel to write, tune, or unit-test, and (deliberately) invisible to
`r4dx::kernels::r4dx_kernel_launch_counter_get()`, which only counts `r4dx`-owned kernel launches
(see kernels.h's own comment). That means the "prove nothing extra happens" bar from
`docs/status.md`'s task item 1 is met two ways: (a) when no capture is attached, the hot path is
unchanged (a single `.empty()` branch, no allocation, no copy -- literally the pre-B1 code plus one
cheap check) and (b) even WITH a capture attached, the r4dx-owned launch counter's delta across a
`Prefill()` call is unaffected (verified below), because the mechanism used adds no r4dx-owned
kernel launches at all.

**Tests**: `tests/model/test_dflash_feature_capture.cpp` (new, registered in
`tests/model/CMakeLists.txt`'s `R4DX_MODEL_TESTS`, same SKIP-if-missing-container convention as
`test_forward_smoke.cpp`), against the real 4-layer `bf16` test container, HIP device 1:
1. Attaches capture at `{0, 2}`, prefills 10 tokens, and checks column 0 (layer 0's input) is
   BIT-EXACT against an independently-computed ground truth: `r4dx::kernels::EmbeddingGatherHost`
   (the same host-side embedding gather `RunChunk` itself uses) called directly against the
   container's own embedding table -- since layer 0's input is, by construction, exactly the raw
   token embedding. Column 1 (layer 2's input) is checked to be populated and to differ from
   column 0 (the simplest available cross-check that the per-target-layer column-offset arithmetic
   is not aliasing two different layers onto the same column).
2. Loads two fresh `Model`s from the same options, one with a capture attached (`{0,1,2,3}`, every
   layer of the 4-layer test container) and one without; asserts the r4dx-owned kernel launch count
   delta across each one's `Prefill()` call is IDENTICAL, and that the returned logits are
   bit-identical -- i.e. attaching a capture perturbs neither the launch count nor the actual
   forward pass's arithmetic.

Measured on real hardware (HIP device 1, `bf16` 4-layer test container): all checks above pass
(`[PASS]` lines in `build/logs/`, not committed -- gitignored scratch); r4dx-owned launch count was
19 both with and without a capture attached, for a 10-token single-chunk prefill.

**Superseded (2026-09-20, stage S1)**: this section previously closed by listing "the three NEW
kernels" as not done. They are done -- see section 6b below, which specifies and links all five
device pieces, each with its own green ctest on HIP device 1.

**Superseded again (2026-09-20, stage S2)**: the remaining B1 list this section then carried -- the
draft MODULE (encoder, per-layer KV injection ring, the block-diffusion layer stack, the selector
walk), the device-upload path, the fixture A/B/C draft-round test, the anchor dump and the isolated
cost measurement -- is done. See section 6c. The `TODO(dflash2-forward)` marker in
`src/model/dflash_draft_weights.h` is closed by `src/model/dflash_draft.cpp`'s `Load()`, which is
exactly the `ToDevice()`-style walk that header predicted, reusing the `SafetensorsReader` the
class already holds. What is left for stage S3 is the DRIVER: wiring the drafter into generation
(prefill injection, draft/verify/commit round loop, CLI/server flags), proving it lossless against
greedy decode, and the end-to-end tok/s measurement.

## 6b. Kernels (Milestone 5 stage S1, 2026-09-20)

The five device-side primitives the DFlash2 draft forward needs that neither r4dx nor
`third_party/libr4d` already provided. All five are declared in
`src/kernels/include/r4dx/kernels/kernels.h`, implemented in `src/kernels/src/r4dx_kernels.hip`,
and launched/counted exactly like every other r4dx-owned kernel. **No model wiring is part of this
stage** -- these are standalone, individually tested pieces.

Every test runs its CPU-reference checks unconditionally and then compares against
`tools/reference/golden_out/dflash2/fixture_a`, returning 77 (CTest SKIPPED) if that directory is
absent -- so a missing fixture can never hide a real regression. Regenerate with
`<reference venv>/python.exe tools/reference/dflash2_ref.py --gen-fixtures all --seed 0`.

| Entry point | Contract | Test |
|---|---|---|
| `r4dx_rope_neox_bf16` | In-place NeoX split-half rotation over ALL `head_dim` dims (n_rot == head_dim, so no pass-through tail), pair `(i, i+head_dim/2)`, theta 1e7, per-row absolute positions as an int32 DEVICE array. fp32 math, RTNE to bf16. Q and K are independently optional (`ptr=0` **and** the matching head count `0`) because the injection path ropes K only -- there is no `Wq` in it at all. | `tests/kernels/test_rope_neox.cpp` |
| `r4dx_topk16_f32` | `[rows, vocab]` fp32 -> ids int32 `[rows,16]` + vals fp32 `[rows,16]`, descending by value, ties deterministically to the LOWER id. One workgroup per row: per-thread register top-16 over a grid-strided slice, then an 8-round pairwise LDS merge under the same total order, so the result is the exact global top-16 regardless of thread scheduling. | `tests/kernels/test_topk16.cpp` |
| `r4dx_dflash_attn_bf16` | The draft block's attention. Visible keys for query row `t` (abs position `q = n_injected + t`): injected-store positions `p` in `[max(store_begin, q-window+1), n_injected-1]` read at `slot = p % slots`, **plus all `T` block keys unconditionally** (the block is non-causal; `attention.causal=false`). `store_begin` (added 2026-09-21) is the first position whose ring contents are valid, i.e. the visible store is the CONTIGUOUS run `[store_begin, n_injected)` intersected with the window; `0` is the original behaviour and the value every caller passed before the injection gap existed. Precondition `0 <= store_begin <= n_injected` (throws). Slot mapping stays `p % slots` over true absolute positions and rope positions stay absolute, so nothing else about the ring geometry changes. fp32 two-pass max-subtracted softmax, bf16 out, GQA `q-head h -> kv-head h/(heads_q/heads_kv)`. The block's own K/V is scratch and is never written to the store -- which is why the ring needs no rollback (section 5). | `tests/kernels/test_dflash_attn.cpp` |
| `r4dx_dflash_conv_bf16` | Thin wrapper over libr4d's `r4d_dflash_conv_t2_g16_bf16` (taps 2, group 16). The whole point is the address arithmetic: `delta = dyn + side*taps*NG` while `dpitch` stays the **full** `2*taps*NG` row pitch (delta is a SLICE, not a compacted copy -- `r4d.h:118`); `base = base + side*taps*H` over `[2(side)][taps][H]`; `block_size` is the power of two `>= T`, so `(t & blockmask) >= tap` degenerates to `t >= tap` for one block starting at row 0. `out` must not alias `x`. Deliberately does **not** bump the r4dx launch counter -- the launch is libr4d's. | `tests/kernels/test_dflash_conv.cpp` |
| `r4dx_rmsnorm_plain_bf16` | `out = x * rsqrt(mean(x^2)+eps) * w`, i.e. the PLAIN weight form -- **not** `r4dx_rmsnorm_bf16`'s `(1+w)` Qwen3.5 convention (see this doc's "RMSNorm convention" row). Output bf16 or fp32 (`out_fp32`); in-place supported. Nothing in this repo or in libr4d computed this form before. | `tests/kernels/test_rmsnorm_plain.cpp` |

**Fixture arrays added for these tests.** `tools/reference/dflash2_ref.py` now also dumps, for
layer 0 of fixture A: `attn_q_prerope_l0`/`attn_k_prerope_l0` (so a rope port has a real
input/position/output triple instead of only a post-rope tensor), `attn_q_l0`/`attn_k_l0`/
`attn_v_l0`/`attn_out_l0` (a complete standalone input/output pair for the attention kernel,
together with `injected_k_l0`/`injected_v_l0` and `n_injected`), `attn_conv_x_l0` +
`attn_o_preconv_l0` + `attn_dyn_l0` + `attn_conv_base_l0` (the conv's two INPUTS plus dyn and base
-- the pre-existing `attn_conv_in_l0`/`attn_conv_out_l0` are both conv OUTPUTS, so neither alone
could drive the kernel), and `output_norm_w`. `dflash2_selftest.py` gates every one of them for
bit-exact determinism, so a regenerated fixture that silently changed any of them fails there
first.

**Measured on HIP device 1** (gfx1201, all five green):

- rope: vs CPU reference, `pos <= 4096` norm_rel 4.2e-5 (Q) / 3.9e-6 (K); `pos <= 300000` 4.6e-4 /
  4.4e-4; position 0 is the bit-exact identity; Q-only and K-only calls match the combined call
  byte for byte. Fixture A (positions 40..47) norm_rel 2.1e-3 (Q) / 2.2e-3 (K).
- top-16: **exact** id and value match against the CPU reference on random data at V in
  {4096, 248320}, on tie-heavy inputs (12 distinct levels, all-identical, all `-inf`), and across
  rows 1..8; **exact** against fixture A's `cand`/`unary` (0/128 mismatches).
- attention: vs CPU fp64 reference over `n_injected` in {0, 3, 40, 2047, 2048, 2049, 2100, 5000} x
  `T` in {1, 4, 8} -- norm_rel 1.63e-3..1.69e-3 throughout, i.e. pinned at the bf16 output quantum
  (2^-9 = 1.95e-3) with no drift at the ring wrap or the window clip. Fixture A norm_rel 3.4e-3.
  The `store_begin > 0` sweep added 2026-09-21 (`n_injected` in {40, 2100, 5000} x `store_begin` in
  {n-1, n-17, n} x `T` in {1, 8}, 18 cases) scores 1.63e-3..1.74e-3 on the same gate, with every
  ring slot the kernel would read at `store_begin=0` but must not read at that `store_begin`
  deliberately filled with 1e4 -- finite rather than NaN on purpose, because a junk key that is
  actually read takes over the softmax completely and moves the output by orders of magnitude
  instead of by a tolerance. `store_begin == n_injected` (nothing in the store visible) is included.
  The `store_begin=0` rows above are unchanged to the printed digit, i.e. the parameter is a
  measured no-op at 0.
- conv: vs CPU reference at `T` in {1, 5, 8} x both sides, norm_rel 1.64e-3..1.67e-3. Fixture A
  side 0 norm_rel 2.6e-3, side 1 2.4e-3.
- plain rmsnorm: vs CPU fp64 reference at hidden in {128, 5120, 17408} x rows in {1, 8, 40} --
  bf16 out norm_rel ~1.7e-3, fp32 out ~5e-8. In-place is bit-identical to out-of-place. Fixture A
  `x_post_ffn_l4 -> x_final_normed` norm_rel 2.6e-3. The `(1+w)` kernel on the same inputs scores
  norm_rel 1.15 -- the explicit assertion that the two forms are not interchangeable.

**On tolerances.** These kernels are bf16-out, so the per-element error floor for any correct
implementation is one bf16 step (2^-9 relative, and 3.1e-2 absolute once an output reaches
magnitude 4-8). The gates are therefore `norm_rel` plus, where it is meaningful, `max_abs`;
`max_rel` is printed but never gated, because a rotation or a softmax drives some outputs to near
zero by cancellation, where the smallest possible disagreement reads as `max_rel ~ 1`. The gates
are still sharp: a wrong pairing, theta, mask, head mapping, tap, side or group index moves
`norm_rel` to O(0.1..1), two to three orders of magnitude over threshold.

**Open, project-wide, found by this stage and NOT decided here.** Every `src/kernels` entry point
is `extern "C"` and there are 13 sites in `r4dx_kernels.hip` that report a violated precondition (
and `R4DX_HIP_CHECK`, a failed HIP call) by `throw std::runtime_error`. The project compiles with
CMake's Windows default `/EHsc`, whose `c` means "assume every `extern "C"` function is nothrow" --
so the caller emits no unwind edge and such a throw does not unwind at all: the process dies
immediately with exit code `0xC0000409`, no message, and neither `catch (const std::exception&)`
nor `catch (...)` runs. The kernels object itself emits the correct MSVC EH ABI
(`_CxxThrowException` / `__CxxFrameHandler3`), so the defect is purely the caller-side nothrow
assumption. `tests/kernels/CMakeLists.txt` sets `/EHc-` on `test_dflash_attn.cpp` and
`test_dflash_conv.cpp` so their precondition checks can catch, and those two tests are the
regression gate. **`r4dx-cli`, `r4dx-server` and `src/model` are NOT fixed** -- they still call the
same entry points under `/EHsc`, so a HIP error raised inside `src/kernels` kills them rather than
unwinding. Whether to move the whole project to `/EHs` is a cross-cutting decision, not one a
kernel stage should make unilaterally.

## 6c. Implementation: the draft module (Milestone 5 stage S2, 2026-09-20)

`src/model/dflash_draft.{h,cpp}` -- `r4dx::model::DflashDraft`, the drafter itself, on device. It is
the structural analogue of `MtpHead` (`docs/mtp.md`) for the other self-speculation family: it owns
its weights, its KV ring and its scratch, never touches the target `Model`'s state, and hands back
token ids that `Model::VerifyWindow` verifies exactly as it does MTP's.

### Module layout

| Piece | Where |
|---|---|
| `DflashDraft::Load(DflashDraftOptions)` | Opens the draft container through the existing `DflashDraftWeights` (the A1 CPU loader -- its `TODO(dflash2-forward)` device-upload gap is what this closes) and uploads every linear through the SAME `QuantLinear`/`ApplyLinear` path the main container uses. All four layouts load (`bf16`/`w4a16`/`w4a8`/`mxfp4`); `bf16` is the exact-arithmetic reference. |
| `InjectFeatures(stream, arena, features_dev, rows, start_pos)` | `g = rmsnorm_plain(fc(features))` once, then per layer `K = rope(k_norm(Wk g))` at the position's ABSOLUTE index and `V = Wv g` raw, written into the ring at `slot = pos % 2048`. `rows <= 64` (a prefill chunk's width). |
| `DraftRound(stream, arena, anchor_id, k, p_min, n_min, embed, lm_head, trace, device_ms)` | The 8-wide noise block through 5 layers, final plain rmsnorm, the TARGET lm_head, top-16, the selector-gate GEMM, one readback, then the host lattice walk. |
| `MakeTargetEmbeddingProvider` / `MakeTargetLmHeadProvider` | The two injectable providers, built from the target `Container`. The drafter has no embedding table and no lm_head of its own -- there is deliberately no `dflash.embed_tokens`/`dflash.lm_head` tensor in the container at all. |

**Why the providers are injectable and not just `const Container&`.** The embedding provider is
handed BOTH the host and device id arrays, so it works whether or not the target's embedding table
has a VRAM mirror (`Container::EmbedTokensDeviceResident()`) -- both paths are exercised. The
lm_head provider runs the BARE lm_head GEMM + `bf16->fp32` widen, deliberately NOT
`FinalLmHead::Forward`, which would apply the target's own `text.final_norm` on top of the
drafter's `dflash.output_norm` (section 4.2 applies exactly one norm). And injecting them is what
lets `tests/model/test_dflash_draft.cpp` swap the real 27B target for the fixtures' synthetic
4096-row one without a second code path.

### Buffers

| Buffer | Size (real drafter) |
|---|---|
| Draft KV ring `k_store_`/`v_store_`, `[5 layers][2048 slots][8 kv heads][128]` bf16 each | 21 MB + 21 MB |
| Injection scratch (`g`, per-layer K/V, positions), sized `max_inject_rows = 64` | ~0.9 MB |
| Draft-block activations (8 rows: `x`, `h`, `conv`, `proj`, `xf`, `dyn`, q/k/v/attn, gate/up/fused/act) | ~2.3 MB |
| `logits_dev_`, `[8, vocab]` fp32 | 7.9 MB |
| Selector codebooks, HOST bf16 `[248320][256]` x2 | 254 MB host, 0 VRAM |
| Weights (`bf16` / `w4a16` container) | 3.33 GB / 0.89 GB device, plus the `fc` encoder |

`slots == sliding_window == 2048` exactly: a slot is reused only once its previous occupant has
aged out of every possible query's visible range, which is also `r4dx_dflash_attn_bf16`'s own
`window <= slots` precondition (section 6b).

### Host vs device split

Everything except the selector lattice runs on the GPU. One `DraftRound` issues its whole layer
stack asynchronously and then performs **exactly one** device->host copy -- a single ~5 KB staging
blob whose three slices are written in place by their producers (`cand` and `unary` by
`r4dx_topk16_f32`, `gate` by the selector-hidden GEMM) -- followed by **exactly one** stream
synchronize. The lattice walk then runs on the host in fp32.

The two `[vocab][selector_rank]` codebooks stay HOST-resident (bf16, 127 MB each) rather than being
mirrored to VRAM: the walk is inherently sequential across block positions (position `t`'s
`pred_idx` is an index into position `t-1`'s candidate list), so a device gather would be gathered
only to be copied straight back. On the hot path only the ONE row the walk actually reads is
materialised (256 multiplies + 16 dot products of length 256 per position, ~31k FMA per round);
the full `[|P|, 16]` score matrix is built only when a caller passes a `DflashRoundTrace`, which
tests do so they can compare against the fixtures' own `score_t{1..7}`.

Measured host cost of the whole non-device part of a round (walk + ~80 kernel launches + the sync):
**0.32-0.41 ms**, i.e. wall 6.18 ms vs device 5.77 ms for the `w4a16` drafter.

### The store invariant

**The block's own K/V is scratch and is NEVER written to the ring.** `r4dx_dflash_attn_bf16` takes
the 8 block keys/values as separate operands and the ring as another pair, and only
`InjectFeatures()` ever writes the ring -- only for positions the target has already committed.
That is precisely why a partially-rejected verify round needs no rollback (section 5): nothing
speculative was ever stored. `InjectFeatures` additionally enforces `start_pos >= InjectedCount()`
and throws otherwise, which upholds the second half of that argument (monotonic, so a stale byte is
always physically overwritten before it can be read as committed data). Since 2026-09-21 a
`start_pos` strictly ABOVE the frontier is allowed too and moves `ValidFrom()` up -- the visible
store is `[ValidFrom(), InjectedCount())`, not `[0, InjectedCount())` (section 5's "Relaxed" note).
Both facts are stated as comments at the two call sites in `dflash_draft.cpp`.

### The per-chunk capture observer

`Model::SetDflashCaptureObserver(std::function<void(const uint16_t* features, int64_t rows,
int64_t start_pos)>)` is invoked at the end of **every** `RunChunk` call -- every prefill chunk AND
every plain decode step, in order, after that call's captured rows are complete on the device and
before any later call overwrites `dflash_features_dev_` at row 0. This generalises the narrower
`Prefill(..., on_chunk_captured)` drain hook section 6a added: `dflash_features_dev_` is sized
`max_chunk_` rows, not `max_ctx` (a `max_ctx`-sized capture would cost gigabytes of VRAM per
attached target layer), so without it a >64-token prompt silently loses every chunk's features but
the last, and a plain decode step's single captured row is likewise lost.

`VerifyWindow` deliberately does NOT invoke it: a verify window's rows are CANDIDATES, and only the
accepted prefix may ever be injected (section 5 step 4) -- only the driver knows how many that is,
so it reads `DflashFeatureBuffer()`/`DflashFeatureRows()` itself after `VerifyWindow` returns.

Also generalised this stage so a DFlash2 run needs no MTP head anywhere: `ModelOptions` gained
`dflash_draft_k`, `Model` gained `draft_window_ = 1 + max(mtp_draft_k, dflash_draft_k)`, and
`VerifyWindow`'s "MTP is not enabled" throw became a `draft_window_ > 1` check. The verify scratch
was renamed `mtp_logits_dev_`/`mtp_argmax_dev_` -> `verify_logits_dev_`/`verify_argmax_dev_` to
match. MTP behaviour at a given `mtp_draft_k` is byte-identical (`test_mtp` unchanged and green).
`Model::CommitVerifiedWindow(n)` exposes the pos_/acceptance-count commit
`DecodeStepMtpGreedy` does inline, for a drafter that owns its own round loop.

### Measured fixture tolerances (`tests/model/test_dflash_draft.cpp`, HIP device 1)

Against the `bf16` draft container and the reference's synthetic 4096-vocab target, `RelL2` (never
per-element `max_rel` -- see section 6b's note on cancellation):

| Intermediate | Fixture A (N=40) | Fixture B (N=2100) |
|---|---|---|
| `g` (encoder) | 3.25e-3 | -- |
| injected K / V, per layer | 3.50e-3..3.97e-3 / 3.05e-3..4.00e-3 | -- |
| `x_post_attn` l0..l4 | 8.19e-3 -> 2.00e-2 | -- |
| `x_post_ffn` l0..l4 | 1.41e-2 -> 2.65e-2 | -- |
| `x_final_normed` | 3.69e-2 | -- |
| `cand`, `unary` | **bit-exact** | **bit-exact** |
| `gate` | 3.33e-2 | 1.20e-1 |
| `score_t{1..7}` | 5.68e-3..1.27e-2 | 5.68e-3..4.68e-2 |
| drafted chain | **exact (7 tokens)** | **exact (7 tokens)** |

Fixture C (`p_min=0.3` on A's state): chain **exact**, 5 tokens, a strict prefix of A's 7, stopped
by `p_min`; the `n_min` discard throws the whole draft away as specified.

Two things about how that table is produced, because they are load-bearing:

1. **The exactness gate substitutes the fixture's own `logits` for the target lm_head.** The
   fixtures' target is synthetic -- `lm_head = RandomState(0).randn(4096, 5120) * 0.02` on a
   ~unit-RMS hidden state -- so a row's 4096 logits are near-Gaussian and the gap between the 16th
   and 17th order statistics is ~0.05 sigma, while this path's own logits agree to 3.7e-2 relative.
   The top-16 SET and especially its ORDER therefore reshuffle, and the walk's `pred_idx` is an
   index INTO the previous position's candidate list, so one reordering silently re-points the next
   position's whole score row. That is a property of the synthetic target, not of the drafter.
   Feeding the reference's exact logits leaves everything this module owns under test (top-16,
   the gate GEMM, the codebook trilinear form, the walk, `p_min`, `n_min`) and makes the score
   matrices column-aligned with the fixture's so they can be compared at all.
2. **The full end-to-end path is still run and reported**, right after, through a real bf16 lm_head
   GEMM: fixture A logits `RelL2` 3.72e-2, cand set-overlap 120/128, agreeing chain prefix 4/7 --
   with the per-position selector margins printed so a reader can see the flips are near-ties.

Fixture B's `gate` gate is looser (2.5e-1) for a measured reason: at `n_injected=2100` the block's
attention averages over ~2048 visible keys instead of 40, and a diffuse softmax's output is a
heavily cancelling sum, so the same per-element bf16 input error lands ~3x larger relative. It is
not the windowing -- `r4dx_dflash_attn_bf16` was swept against a CPU fp64 reference at
`n_injected` in {0,3,40,2047,2048,2049,2100,5000} and stayed pinned at 1.6e-3 throughout (section
6b) -- and the properties that WOULD catch a windowing bug (`cand`/`unary` bit-exactness, the
drafted chain) are gated exactly for B like every other fixture, and pass.

**`w4a16` draft container, same fixture A inputs** (reported, not gated): the drift profile is
smooth, not a step -- `g` 1.02e-1, injected K 8.21e-2..1.15e-1, `x_post_ffn` l0..l4
4.42e-1/3.49e-1/3.19e-1/3.30e-1/4.38e-1, `x_final` 7.02e-1, cand set-overlap 32/128. A w4a16
PACKING bug would show as a step change at one stage; this is ordinary 4-bit noise entering at the
`fc` encoder (K=25600 in 4 bits) and staying flat thereafter. **Do not read that 7.02e-1 as "w4a16
is unusable"**: the fixtures' features are i.i.d. unit Gaussians with no structure at all, which is
close to the worst case for a low-bit projection. On REAL captured features both containers draft
the IDENTICAL chain (next subsection).

**How the synthetic inputs are reconstructed.** Two of the fixtures' inputs are deliberately not
dumped (section 9): the synthetic target's `[4096, 5120]` embedding table and lm_head, and fixture
B's `[2100, 25600]` features. `tests/model/numpy_legacy_rng.hpp` reimplements
`numpy.random.RandomState(seed).randn` bit-exactly (`mt19937_seed`/`mt19937_next_double`/
`legacy_gauss`, the polar method -- NOT `Generator`'s Ziggurat) so the test rebuilds them in memory
exactly as `dflash2_selftest.py` does. It is validated before it is trusted: Part 0 of the test
regenerates fixture A's own dumped `features.npy` and asserts BIT-EXACT equality (0/1,024,000
mismatches), and fixture B re-checks its first 40 rows against A's since it is the same stream.

### Anchor check on real inputs

`tests/model/tool_dflash_probe.cpp` (built, deliberately not `add_test()`-registered -- same
convention as `tool_hseed_drift`/`tool_vocab_calib`) drives the REAL 64-layer `w4a16`
`qwen38-27b-v3.r4dx` target with the real tokenizer, captures the residual stream entering
`{6,20,34,48,62}` through the per-chunk observer, injects it, drafts, and writes
`features.bin` + `manifest.json` (schema `dflash2_real_dump_v1`) plus its own
`x_final_normed.bin`/`gate.bin`/`cand.bin`/`unary.bin` and drafted chain.
`tools/reference/dflash2_ref.py --real <that directory>` now accepts that raw-blob form alongside
the `.npz` of section 8, redoes the round in fp32 numpy from the original Q8_0 GGUF against the
checkpoint's own fp32 embedding/lm_head, and prints a per-tensor comparison plus a hybrid-walk
attribution.

Two short prompts, real hardware:

| | prompt 0 ("...the capital of Germany is") | prompt 1 (a `fibonacci` body) |
|---|---|---|
| anchor | 19241 (` Berlin`) | 73111 (` fibonacci`) |
| r4dx chain | `[13,271,760,6511,314,11751,369]` | `[1393,12,16,8,478,73111,1393]` = `(n-1) + fibonacci(n` |
| reference chain | `[13,1061,369,264,3750,6511,314]` | identical |
| `x_final_normed` RelL2 | **1.09e-2** | **1.34e-2** |
| `gate` RelL2 | 9.02e-3 | 9.55e-3 |
| `cand` exact-position / set-overlap | 72/128, 119/128 | 78/128, 125/128 |

Prompt 1's chain is **identical**. Prompt 0's differs, and the divergence is localised, not
hand-waved: the two hybrid walks the reference now prints show that the port's own `cand`/`unary`
combined with the REFERENCE's `gate` reproduce the port's chain exactly, and the reference's
`cand`/`unary` combined with the PORT's `gate` reproduce the reference's chain exactly. The
drafter's decision-relevant output is therefore equivalent on both sides; the whole difference is
`cand`, which comes from the **target lm_head** -- r4dx reads the container's 4-bit `w4a16` head,
the reference the checkpoint's fp32 one. `cand[2]`'s top two entries (`1061` and `271`) are simply
swapped between them. There is nothing in the drafter to fix; a drafter output agreeing to ~1% on
real activations is the anchor this check exists to establish.

That both the `bf16` and the `w4a16` draft containers produce the IDENTICAL chain on both real
prompts is the other half of the w4a16 story above: on structured, real features the 4-bit drafter
tracks the bf16 one exactly, whatever the synthetic fixture's unstructured Gaussians suggest.

### Isolated cost (HIP device 1, `n_injected=512`, 50 rounds, one process at a time)

| Draft container | `DraftRound` device | `DraftRound` wall | `InjectFeatures(64 rows)` device / wall |
|---|---|---|---|
| `bf16` (3.58 GB) | **9.696 ms** (9.500-9.865) | **10.021 ms** (9.927-10.436) | 2.000 / 2.087 ms |
| `w4a16` (1.13 GB) | **5.770 ms** (5.704-5.852) | **6.179 ms** (6.091-6.771) | 0.799 / 0.875 ms |

Per-round weight traffic is 4.01 GB for `bf16` (5 x 665.9 MB of layer weights + the target's 676 MB
4-bit lm_head) and 1.56 GB for `w4a16`. Fitting `t = c + bytes/BW` to the two device figures gives
**BW = 622 GB/s** -- essentially this card's peak -- and **c = 3.26 ms** of launch-bound remainder
over ~80 kernel launches at M=8 (~41 us each, the Windows/WDDM launch overhead `docs/r9700.md`'s
host-overhead work already documents). So the memory-bound part is already running at the roof and
the obvious S3 lever is launch count, not bandwidth. VRAM with the target loaded at `--max-ctx
8192`: 20.83 GiB (`bf16` drafter) / 18.39 GiB (`w4a16`).

Projecting against the cost model: a round is draft + verify(<=8 rows, ~26 ms) + inject(accepted
rows only, well under the 64-row figure above), i.e. ~32.5 ms with the `w4a16` drafter -- so
3.1 accepted tokens per round already clears 95 tok/s and 5.9 (the 84% ROCmFPX figure) would reach
~180. Measuring that end to end is S3's job, not this stage's.

**A measurement trap worth recording.** `hipEvent` timing of a `DraftRound` is only valid if the
closing `hipEventRecord` precedes the round's final small async D2H into PINNED host memory: that
copy can be serviced by the SDMA/blit engine rather than the compute queue, and an event recorded
behind it carries that queue's timestamp instead. With the record one line later the same code
reported 0.03-0.27 ms (with an occasional correct sample at exactly the wall figure) for a round
whose wall clock is 6-10 ms and whose weight read alone is 1.5-4 GB -- an impossible 16-50 TB/s.
The `device_ms` out-parameter on `DraftRound` exists so callers get this right by construction.

## 7. Integration plan for r4dx (items 1-4 BUILT, see section 6c; item 5 is stage S3's)

1. **Target-side feature capture -- DONE, see section 6a above** (`Model::AttachDflashFeatureCapture`,
   2026-09-20). During both prefill/decode and `VerifyWindow`, the residual stream entering layers
   `{6,20,34,48,62}` (0-based HF indexing, i.e. the OUTPUT of layers `{5,19,33,47,61}`) is captured
   for every position, analogous to how `docs/mtp.md`'s `h_seed` is already captured off
   `Model::RunChunk`.
2. **Draft KV ring -- DONE, see section 6c** (`DflashDraft::k_store_`/`v_store_`): per-layer
   (5 layers), per-head (8 kv heads), 128-dim, sized to `sliding_window = 2048` positions -- 21 MB
   each, no fp8 paging needed.
3. **Injection -- DONE, see section 6c** (`DflashDraft::InjectFeatures`, driven off
   `Model::SetDflashCaptureObserver` for prefill/decode chunks and off the driver for a verify
   round's accepted prefix). It did NOT end up sharing a helper with `MtpHead::PrimeKv` as this
   item speculated: the two have different inputs (target FEATURES vs an `(h_i, t_{i+1})` pair),
   different arithmetic (an encoder + K/V only, no attention at all vs a full decoder sublayer) and
   different stores, so the only thing they would have shared is the word "prime".
4. **Draft loop -- DONE, see section 6c** (`DflashDraft::DraftRound`). The selector walk did stay on
   the host as predicted, and is measured at well under 0.4 ms including every kernel launch and the
   round's one synchronize.
5. **Verify + commit -- the verify half is ready, the driver is stage S3's.** `Model::VerifyWindow`
   was generalised this stage so it no longer requires an MTP head (`ModelOptions::dflash_draft_k`,
   `Model::DraftWindow()`), and `Model::CommitVerifiedWindow` exposes the pos_/acceptance-count
   commit. What remains is the round loop itself: draft -> verify -> accept the longest greedy
   prefix -> inject the accepted rows' captured features -> new anchor, plus the CLI/server flags
   and the losslessness proof. DFlash2 remains a drop-in alternative DRAFT SOURCE, not a different
   verify contract.

## 7a. Stage S3: the driver, wired and measured (2026-09-20)

**Item 1 (flags) -- DONE.** `--dflash <container>` / `--dflash-k N` (1..7, default 7) /
`--dflash-p-min F` (default 0) / `--dflash-n-min N` (default 0) on both `r4dx-cli` and
`r4dx-server` (`src/cli/cli_args.h`, `src/server/server_args.h`); `--dflash` with `--mtp > 0` is a
`CliUsageError`/`ServerUsageError` at parse time. `ModelOptions` gained `dflash_container` (empty
disables it, matching every other opt-in flag in this codebase); `Model::Load` peeks the
container's own `__metadata__.dflash2.layout` field (NOT `ModelOptions::layout`, which is the
TARGET body's layout -- the two are independent, docs/container-format.md), validates it via
`LayoutFromName` (throws on garbage), checks `dflash_draft_k <= block_size-1`, and prints a load
line (`[r4dx::model::Model] DFlash2 drafter loaded: <path> (layout=..., block_size=8,
target_layers=[6,20,34,48,62], ...)`). The VRAM breakdown gained its own
`DFlash2 drafter VRAM: X GiB (weights+kv_ring+draft_scratch; already included in kv+gdn_state
above)` line -- measured 0.95-1.05 GiB depending on `--dflash-k` (w4a16 draft) or ~1.9-2.0 GiB
(bf16 draft), bracketing `DflashDraft::Load` with its own `hipMemGetInfo` snapshot pair.

**Item 2 (prefill injection) -- DONE, and turned out to need no new driver code at all.**
`Model` now owns an `optional<DflashDraft> dflash_` internally when loaded with a non-empty
`dflash_container`. `RunChunk` (which both `Prefill`'s per-chunk loop and every plain
`DecodeStep`/`DecodeStepGreedy` call funnel through) calls `dflash_->InjectFeatures(...)` directly,
once per call, right where `SetDflashCaptureObserver`'s external hook already fires for a
caller-owned drafter -- this is a SEPARATE, simpler path from that external-observer mechanism
(which stays exactly as section 6c documented it, for a driver that owns its OWN `DflashDraft`
object, e.g. `tests/model/tool_dflash_probe.cpp`): a Model-owned drafter is fed via a plain member
call, not a captured closure, specifically because a closure capturing `this`/`&m` inside
`Model::Load`'s own local variable would dangle across the NRVO-not-guaranteed move on return.
`VerifyWindow` deliberately still never auto-injects (its rows are unverified candidates); only
`DecodeStepDflashGreedy` (item 3) injects the accepted prefix, explicitly, after it knows how many
rows that is.

**Item 3 (round loop) -- DONE.** `Model::DecodeStepDflashGreedy(token_id, k, p_min, n_min,
walk_len_out=nullptr)` returns the exact same round-vector contract as `DecodeStepMtpGreedy`
(`0..k` accepted drafts + one correction/bonus token), so `mtp_round.hpp`'s `ProcessMtpRound` (already
speculation-family-agnostic) and `src/cli/main.cpp`'s/`src/server/engine.cpp`'s round loops plug in
with the same shape as the MTP branch, just gated on `!args.dflash.empty()` /
`!opts_.model_opts.dflash_container.empty()` instead of `args.mtp > 0`. Sequence: `DraftRound`
anchored at `token_id` -> candidates `[token_id, d1..dm]` -> `VerifyWindow` (which, as a side effect
of the SAME per-layer capture hook, also refills `DflashFeatureBuffer()`/`DflashFeatureRows()` with
this window's own captured target features) -> greedy longest-accepted-prefix `a` (anchor always
accepted) -> `InjectFeatures` the accepted prefix's own just-captured rows (`0..a-1`, contiguous,
no offset needed) at the drafter's current `InjectedCount()` -> `CommitVerifiedWindow(a)` (advances
`pos_`, threads the GDN acceptance count -- the exact non-MTP factoring `CommitVerifiedWindow`
exists for). `Model::Reset()` also resets the drafter's own ring (`dflash_->Reset()`, a cheap
host-only counter zero, same self-correcting-via-position-overwrite argument as everything else this
project resets this way).

**Server prefix-reuse decision (item 3's own ask, "pick one, justify"):** re-inject on every prefix
hit, not a forced full re-prefill -- and this requires NO new server code, because it falls out of
the design directly: `dflash_`'s `InjectedCount()` and `Model::pos_` are two counters on the SAME
persistent `Model` object that ALWAYS advance together, by construction, on every code path that
touches either (`RunChunk`'s inline auto-inject advances both by the same `T`; `DecodeStepDflashGreedy`
advances both by the same `a`). The server's `PrefixState::Extend()` prefix-hit path does not call
`Model::Reset()` at all, so neither counter is touched -- both simply stay exactly where they were,
still in lockstep. A prefix MISS (not an extension) DOES call `Reset()`, which zeroes both together.
There is therefore no window in which the ring could hold stale/mismatched injected positions
relative to `pos_`. `tests/model/test_dflash_e2e.cpp`'s `CheckChatMultiTurnMidRoundStop` (below)
is the concrete regression test for the surrounding "committed vs displayed" bookkeeping this
reasoning depends on.

**Bug found and fixed while validating item 4 (see below): a leaked, un-`Reset()` `arena_` after
`RunChunk`'s own inline `InjectFeatures` call.** Every OTHER terminal user of `Model::arena_` resets
it before returning; the new inline auto-inject call was the one exception, leaving the next
`RunChunk` call's own per-layer loop starting from a non-zero, dirty offset instead of the clean one
every other code path (including every `--mtp 0`/`--mtp N` baseline) always starts from. Fixed by
adding one `arena_.Reset()` call. **This fix did NOT, on its own, explain the losslessness
mismatches `validate_dflash.ps1` found** (same output hash before and after) -- see item 4's own
finding below for the actual explanation. Kept anyway: leaving `arena_` dirty across calls is a
real, if apparently latent, correctness hazard independent of whether THIS specific mismatch traced
back to it.

**Bug found in `tests/model/test_mtp.cpp` (out of this stage's own scope -- NOT fixed here, flagged
as a background task): `CheckChatMultiTurnMidRoundStop`'s `stop_after = cumulative + round.size() -
1` formula can mathematically never produce a genuine "committed > displayed" gap** (`ProcessMtpRound`'s
`committed` is unconditionally `round.size()-1` elements; with `max_tokens_remaining ==
round.size()-1`, `displayed` pushes exactly the same `round.size()-1` elements, so the two are
always equal). This has apparently gone unnoticed because the 4-layer container that test runs
against never returns a round with `size>=2` within its own first 8 rounds, so the function always
takes its own (correct, non-failing) "cannot force a mid-round stop; skipping" early return instead
of ever reaching the broken arithmetic. `tests/model/test_dflash_e2e.cpp`'s own copy of this check
uses the corrected `cumulative + round.size() - 2` formula and DOES land a genuine mid-round stop
(verified: "displayed=6 committed=8 (2 committed-but-undisplayed token(s))" on a real run).

**Item 4 (lossless gate) -- `tools/validate_dflash.ps1`, run for real (2026-09-21, Integrate stage);
result: PASSED WITH WARNINGS, exit 0, every cell now accounted for by evidence.** Modeled on
`tools/validate_fusion.ps1`. For every (target layout, prompt) pair it also runs an `--mtp 7`
CONTROL (no `--dflash` at all) against the same prompt, so a `--dflash` vs `--mtp 0` mismatch is
checked against the pre-existing, dflash-uninvolved MTP path on the SAME prompt/layout rather than
assumed to be "the known mechanism" from a layout name.

**Correction (review finding, 2026-09-21, blocker): the previous revision of this section reported
"4 of 9 cells mismatch" and called the gate's run "PASS" without noting it exited 1** (the script's
own `--mtp 7`-only control had no way to close the mxfp4/long row, and a hard FAIL with no override
cannot be called a pass). Both are fixed: `validate_dflash.ps1` gained a second, "grouping control"
(a DIFFERENT draft container at the same target/layout/prompt -- see the script's own doc comment)
for exactly the case where the `--mtp 7` control does not reproduce a mismatch, and a fresh run on
this corrected tree found **5 of 9 cells mismatch** (not 4), all 5 now resolved to WARN (not a hard
FAIL), so the script exits 0:

| Layout | short (~20 tok) | medium (~250 tok, 4 prefill chunks) | long (~3500 tok, 55 chunks, exceeds the drafter's own 2048-token sliding window) |
|---|---|---|---|
| w4a16 | OK (byte-identical) | MISMATCH -- `--mtp 7` control diverges from `--mtp 0` with the EXACT SAME SHA-256 as `--dflash` (direct confirmation) | OK |
| w4a8 | OK | MISMATCH -- control also diverges, but a DIFFERENT hash than `--dflash` (same mechanism class, argued not proven) | MISMATCH -- control diverges with the EXACT SAME SHA-256 as `--dflash` (direct confirmation) |
| mxfp4 | OK | MISMATCH -- control also diverges, but a DIFFERENT hash than `--dflash` (same mechanism class, argued not proven) | MISMATCH -- `--mtp` at K=1..7 ALL match `--mtp 0` on this prompt (the `--mtp 7` control does NOT reproduce it), but the GROUPING control (`--dflash` against the mxfp4 draft container instead of w4a16, same target/layout/prompt) DOES match `--mtp 0` exactly -- resolved below, not by the primary control |

Of the 5 mismatching cells, **2 (w4a16/medium, w4a8/long) are DIRECTLY confirmed** by an
identical-SHA `--mtp 7` control; **2 (w4a8/medium, mxfp4/medium) are same-mechanism-class but NOT
directly proven** (the control diverges from `--mtp 0` too, but not to the identical text `--dflash`
produced); and **1 (mxfp4/long) is resolved by the new grouping control**, not the primary one --
see below. This corrects the previous revision's "four of the five are DIRECTLY confirmed" claim,
which counted the 2 same-class-but-unproven cells as direct confirmations; `docs/status.md`'s "3 of
4" count is consistent with the 2-direct/2-class-only split once mxfp4/long (a 5th cell, then
unresolved) is counted separately, as it is here.

Every mismatch is a single coherent word/phrase substitution (e.g. "which states that" vs "which
describes how", "repeated identically" vs "repeated verbatim") -- never a dropped/duplicated token,
never garbage. Some are DIRECTLY confirmed (not merely argued) to be the documented
batched-verify-reduction-order mechanism docs/mtp.md already describes for MTP, because the
`--mtp 7` control reproduces the identical divergence (same SHA-256) on the same prompt/layout with
ZERO DFlash2 code involved -- a finding this stage adds to that mechanism's own record: it is NOT
mxfp4-specific (docs/mtp.md previously only documented it there), w4a16 and w4a8 show it too.

**Correction (review finding, 2026-09-21): the previous paragraph here wrongly explained why
DFlash2 diverges more often than MTP as "DFlash2's own verify batch size varies round-to-round ...
while MTP's is fixed at `k+1` every round".** That is factually wrong at the shipped defaults
(`--dflash-p-min 0`, `--dflash-n-min 0`): `SelectorWalk`'s loop (`src/model/dflash_draft.cpp`) only
ever breaks on the `k` cap or on `p_min`, so with `p_min<=0` it always emits exactly
`min(k, block_size-1)` tokens and the verify window is exactly `k+1` rows every single round --
IDENTICAL to MTP's. The shipped per-request stats prove it directly: `drafted=128` over `rounds=32`
at k=4 and `drafted=217` over `rounds=31` at k=7, i.e. exactly k tokens drafted per round, in every
run made. The real differentiator between DFlash2 and MTP is WHICH tokens fill an identically-sized
verify window, not the window's width -- DFlash2's selector walk and MTP's own head propose
different candidate sequences from the same context, and a batched-GEMM reduction-order effect that
is borderline for one candidate sequence need not be borderline for another. This is also why
changing the DRAFT CONTAINER (a pure re-quantization of the encoder/layers that changes which
tokens get drafted, not the verify mechanism or window width) can flip a specific mismatch from
diverging to matching, and is the basis of `tools/validate_dflash.ps1`'s own "grouping control"
(below).

**The mxfp4/long cell -- RESOLVED (2026-09-21, Integrate stage), by experiment, not by falling back
to the same explanation without evidence.** `--mtp` at every one of K=1,2,3,4,5,6,7 reproduces
`--mtp 0` byte-for-byte on this exact prompt/layout (checked directly, not assumed), yet `--dflash`
(w4a16 draft) diverges by one word ("identically" vs "verbatim") on the same prompt -- so the
`--mtp 7` control experiment that confirmed the other four cells does NOT confirm this one. Localized
with a battery of controls that each vary exactly one thing:
- `--dflash-k` 1, 2, 3 all reproduce `--mtp 0` byte-for-byte; `--dflash-k` 4, 5, 6, 7 ALL flip the
  same one word, deterministically across repeats -- i.e. the divergence appears at a specific verify
  window WIDTH, not randomly.
- `--dflash-p-min 0.5` and `--dflash-p-min 0.9` at k=7 (early-stopping the selector walk before it
  reaches that width on this prompt) both go back to matching `--mtp 0` exactly.
- The **mxfp4 draft container** at k=7 matches `--mtp 0` exactly, while the **w4a16 and bf16** draft
  containers at k=7 both diverge -- re-quantizing the draft body (which changes WHICH tokens the
  selector walk drafts, not the verify mechanism or the window's width) flips the result. A
  bookkeeping/lifecycle bug cannot be switched off by changing the draft container's own numeric
  precision; a verify-window grouping effect can, because a different draft container proposes a
  different candidate sequence into the identical-width, identical-code verify window.
- `tools/validate_dflash.ps1`'s own new "grouping control" (added this stage specifically for this
  finding) automates exactly the third point: re-running `--dflash` with the mxfp4 draft container
  in place of w4a16, same target/layout/prompt. It matches `--mtp 0` exactly, confirming the above.

**Conclusion: this is the same verify-window-grouping mechanism as the other 4 mismatching cells,
confirmed by a control the primary `--mtp 7` check cannot express** (varying `--dflash-k`/
`--dflash-p-min`/the draft container all change which candidate tokens fill the verify window,
without touching DFlash2's own bookkeeping) -- not a DFlash2-specific correctness bug. It is not
"proven" in the identical-SHA sense the other two direct-confirmation cells have (there is no
zero-DFlash-code control that reproduces this exact byte pattern, because MTP's own fixed-K
schedule never happens to visit this precise window geometry), but the battery above rules out a
bookkeeping/lifecycle explanation specifically, which is the failure mode item 4 exists to catch.

**Item 5 (anchor check against real captured features) -- DONE (Integrate stage, 2026-09-21).**
Stage S2 confirmed the drafter's own math against the Python reference on two real prompts using a
SEPARATELY-owned `DflashDraft` fed via `SetDflashCaptureObserver` (an external-observer harness).
This left the WIRED path -- `ModelOptions::dflash_container`, `Model`'s own internal `dflash_`,
`Model::Prefill`'s auto-inject, `Model::DecodeStepDflashGreedy` -- never checked against the Python
reference at all, which is the one path a real `r4dx-cli`/`r4dx-server` generation loop actually
takes and the one thing that could catch "wired up but silently drafting from the wrong features".

Closed by: (1) `Model::DecodeStepDflashGreedy` gained two optional out-params, `trace_out`
(forwarded straight to `DflashDraft::DraftRound`'s own `trace`) and `drafted_tokens_out` (the raw
pre-verify chain, distinct from the method's own post-verify return value) -- diagnostics-only,
zero cost when null, existing call sites unaffected (both are trailing defaulted params); (2)
`tests/model/tool_dflash_probe.cpp` gained a `--wired` mode that loads exactly one `Model` with
`dflash_container` set (no separately-owned `DflashDraft`, so there is never a second ~GB-scale
object competing for VRAM with the wired one), drains the model's own public
`DflashFeatureBuffer()`/`DflashFeatureRows()` via `Prefill`'s `on_chunk_captured` callback for the
dump, and calls `DecodeStepDflashGreedy` with both new out-params. Dumps to
`build/logs/dflash_probe/wired_promptN`.

**A real bug was found and fixed while building this: `RunWired`'s per-chunk callback dereferenced
`Model::DflashFeatureBuffer()` directly from host code.** That pointer is a DEVICE pointer
(`dflash_features_dev_` lives on the GPU) -- the original S2 harness always `hipMemcpy`'d it to a
host staging buffer first; the new wired-mode callback initially skipped that copy and read GPU
memory from the CPU, crashing with `STATUS_ACCESS_VIOLATION` (0xC0000005) partway through the first
prompt. Found by running it (not by inspection) and fixed by adding the same D2H `hipMemcpy` the
original harness already does.

**Result, both real prompts, w4a16 draft container (matches the S3 driver's own default choice):**

| Prompt | x_final_normed RelL2 | gate RelL2 | cand (exact/overlap of 128) | chain vs reference |
|---|---|---|---|---|
| "capital of France/Germany" | 1.568e-01 | 1.413e-01 | 50/117 | DIFFERENT -- attributed to the container's 4-bit lm_head precision (hybrid walk: port cand + reference gate reproduces the PORT's chain exactly) |
| "def fibonacci(n)" | 2.057e-01 | 1.551e-01 | 50/113 | IDENTICAL |

**Same check with the bf16 draft container (the S2 anchor check's own default), for direct
comparability against the previously-recorded numbers:** x_final_normed RelL2 **1.094e-02** and
**1.338e-02** on the two prompts -- these are the EXACT SAME figures the ROUND stage's own S3
report already recorded from the UNWIRED (`SetDflashCaptureObserver`) harness ("the drafter's own
x_final_normed agrees with the fp32 reference to 1.09e-2 and 1.34e-2 on the two prompts"), to 4
significant figures. **This is the actual point of item 5's check**: the wired driver reproduces
the previously-validated isolated harness's numbers exactly, on both draft containers, including
the same divergence pattern (prompt 0 differs by lm_head precision only, prompt 1 is bit-exact) --
i.e. the wiring itself introduced no drafting-from-the-wrong-features bug. The higher RelL2 with the
w4a16 draft (0.157/0.206) vs bf16 (0.0109/0.0134) is the drafter's own known 4-bit-noise-at-the-fc-
encoder effect (S2's own finding on synthetic fixtures, 7.02e-1), not a wiring defect -- and, as S2
also found, it does not change the drafted chain on real features (both draft containers draft the
IDENTICAL 7-token raw chain on both prompts: `[13,271,760,6511,314,11751,369]` and
`[1393,12,16,8,478,73111,1393]`).

**Item 6 (measurement matrix) -- PARTIAL, real numbers only, no fabricated rows.** Standard haiku
prompt, `--max-ctx 2048`, greedy, HIP device 1, one process at a time:

| Target layout | Draft layout | K | Decode tok/s (two runs) | Acceptance | Tok/round |
|---|---|---|---|---|---|
| w4a16 | w4a16 | 7 | 74.33, 74.77 | 24.4% | 2.68 |
| w4a16 | bf16  | 7 | 68.43, 68.42 | 25.7% | 2.77 |
| w4a16 | w4a16 | 3 | 71.57 (once) | 46.7% | 2.37 |
| w4a16 | w4a16 | 4 | 77.05 (once) | 40.6% | 2.59 |
| w4a16 | w4a16 | 5 | 75.63 (once) | 32.5% | 2.59 |
| w4a16 | w4a16 | 6 | 73.92 (once) | 27.1% | 2.59 |
| w4a8  | w4a16 | 4 | 64.75 (once) | 33.1% | 2.30 |
| mxfp4 | w4a16 | 4 | 58.64 (once) | 31.8% | 2.24 |

VRAM (w4a16 target + w4a16 draft, K=7): 18.13-18.19 GiB. Best observed on the standard prompt:
**w4a16/w4a16 at K=4, 77.05 tok/s (40.6% acceptance, 2.59 tok/round)** -- beats the plain
(`--mtp 0`) w4a16 baseline of 38.98 tok/s by ~2.0x, but is still below M4's own MTP headline
(68.73 tok/s at its OWN best K) by comparison to THIS run's plain baseline multiplier, and well
below ROCmFPX's 120 tok/s / 84% acceptance on this same card/draft.

**Integrate stage (2026-09-21) additions -- headline re-confirmed, one real gap closed, several
still open (review finding: this must be stated plainly, not narrowed silently).**
- **Headline re-measured after this stage's own fixes** (RunChunk stream-sync, `Model::Load`
  hidden_size check): w4a16/w4a16 K=4, standard prompt, twice: **77.19, 77.26 tok/s** -- <0.3% from
  the pre-fix 77.05/77.06/77.16, confirming neither fix moved decode throughput (the stream-sync
  fix only touches `RunChunk`'s dflash branch, which `DecodeStepDflashGreedy`'s own decode-loop
  injection path never calls). See `docs/perf.md`'s Milestone 5 S3 section for the full writeup.
- **NEW: the ~400-token code prompt, `--max-tokens 256`, real container, twice each --
  `docs/perf.md` has the full table.** Headline: **DFlash2 K=7 reaches 106.93/106.82 tok/s (44.2%
  acceptance) vs MTP K=3's 72.36/72.42 tok/s (53.7%) vs plain's 38.71** -- DFlash2 clearly BEATS MTP
  on this prompt, the opposite ranking from the standard haiku prompt. This corrects the previous
  revision of this section's implicit "DFlash2 is in MTP's range but behind it" framing to
  **prompt-dependent, not a fixed ranking** -- see `docs/perf.md` for the corrected conclusion.
- **Item 7's tok/s parity is CLOSED** (see below) -- moved out of "NOT measured".
- **STILL NOT measured, same reason as before (time budget, not a discovered blocker)**: the
  `p_min` sweep ({0, 0.3, 0.5}); the w4a8/mxfp4 DRAFT-container legs (only w4a16 and bf16 drafts
  have ever been tried); an explicit prefill-with/without-`--dflash` A/B; the one long-context point
  (`--max-ctx 32768`, ~30k real prefilled tokens); and doubling every row of the ORIGINAL haiku-
  prompt table above (K=3/5/6 and the w4a8/mxfp4 target rows are each still a single run -- only
  K=4/K=7/the new code-prompt table are doubled). The acceptance ceiling on the haiku prompt
  (24-47% depending on K, well under ROCmFPX's 84%) is still the more likely place to look before
  assuming a bandwidth/kernel-count problem, and the code-prompt result above (44-69% acceptance on
  a more templated prompt) is consistent with that, but NEITHER was root-caused this stage either.

**Item 7 (server passthrough + smoke) -- flags DONE, end-to-end verified, tok/s-parity CLOSED
(Integrate stage, 2026-09-21).** `--dflash`/`--dflash-k`/`--dflash-p-min`/`--dflash-n-min` pass
through `src/server/server_args.h` -> `EngineOptions` -> `Model::Load` exactly like the CLI;
`engine.cpp`'s `RunRequest` gained a `use_dflash` branch structurally identical to its existing
`use_mtp` one. `tools/server/smoke.ps1` gained `-Dflash <path>` and a `" dflash: "` log-line check.
Run for real against the real 64-layer container + the real w4a16 draft: **all 28 checks passed**,
including
streaming, non-streaming, tool-call round trip readiness (`-ToolRoundTrip` not passed this run but
the flag path is unchanged), the prefix-reuse-does-not-reload check, and the new dflash-path-taken
check.

**Tok/s parity -- CLOSED (Integrate stage, review's own measurement, recorded here rather than
re-run since neither this stage's fix touches the server's request-routing path):**
`r4dx-server --dflash --dflash-k 4` reported **77.54, 77.62 tok/s** on the standard prompt across
two requests, vs the CLI's own **77.06, 77.16 tok/s** (pre-fix measurement; post-fix CLI is
77.19/77.26, see item 6 above) -- within 1%, PASSING item 7's own "must track the CLI within a few
percent" bar.

**Build/test**: full `ctest` **49 registered (was 48), 45 passed, 4 skipped** (the same
pre-existing missing-golden-data skips: `test_kernel_bandwidth`, `test_gdn_layer`,
`test_final_lm_head`, `test_attn_layer`), 0 failed, ~293s, HIP device 1. New:
`tests/model/test_dflash_e2e.cpp` (the mid-round-stop regression test above), `tests/cli/test_args.cpp`
and `tests/server/test_server_args.cpp` both gained `TestDflashFlags`.

**Integrate stage (2026-09-21) closing note on item 6.** The final confirmation sweep (each layout's
own best DFlash `K` vs. its own best MTP `K`, standard + code prompt, twice each) is in
`docs/perf.md`'s top section and `docs/status.md`'s "Milestone 5: done" entry, not repeated here.
Headline: DFlash2 beats MTP on **all three layouts on the code prompt** (w4a16 +30.7%, w4a8 +24.6%,
mxfp4 +24.9%) and on **w4a16/w4a8 on the standard prompt** (+12.2%/+12.3%); MTP still leads on
**mxfp4/standard** (DFlash2 -9.8%) -- the one cell carried into Milestone 6 as an open, not
root-caused, gap. The `p_min` sweep and the w4a8/mxfp4 DRAFT containers remain unmeasured.

## 8. `--real` mode input schema

`tools/reference/dflash2_ref.py --real <path> --target-dir <dir>` accepts EITHER of two forms.

**Form 2 (added stage S2, what the C++ probe writes): a DIRECTORY** holding `features.bin` (raw
little-endian float32, C order) plus a `manifest.json` with `schema: "dflash2_real_dump_v1"`,
`features_file`, `features_dtype`, `features_shape`, `anchor_id`, `n_injected`, and optionally
`positions`. A C++ producer has no npz writer and adding one there would mean implementing zip; a
raw blob plus the shape is the whole difference. When the directory ALSO contains the port's own
`x_final_normed.bin` / `gate.bin` / `cand.bin` / `unary.bin` and a `drafted_tokens` array in the
manifest, `--real` additionally prints the per-tensor comparison and hybrid-walk attribution
section 6c's "Anchor check on real inputs" describes.

**Form 1 (original): an `.npz`** with:

| Key | Shape | dtype | Meaning |
|---|---|---|---|
| `features` | `[N, 25600]` | float32 | Concatenated target residual-stream features at layers `[6,20,34,48,62]`, one row per already-committed position, in absolute-position order. |
| `anchor_id` | scalar | int | The last committed real token id (the block's position-0 anchor). |
| `n_injected` | scalar (optional) | int | Defaults to `features.shape[0]` if omitted. |
| `positions` | `[N]` (optional) | int64 | Absolute sequence positions for `features`; defaults to `arange(n_injected)` if omitted (i.e. assumes a fresh sequence starting at position 0). |

Output: the drafted token ids and each block position's top-16 candidate ids, printed to stdout
(this mode is a manual inspection tool for whoever dumps real captured features from a later GPU
stage, not a golden-fixture generator).

## 9. Golden fixtures (`tools/reference/golden_out/dflash2/`)

All seed 0, deterministic, fp32, generated by `dflash2_ref.py --gen-fixtures {A,B,C,all}` and
verified bit-identical by `dflash2_selftest.py`. Total directory size ~9 MB (budget: 60 MB).

**Not committed, by design**: `tools/reference/golden_out/dflash2/` matches
`tools/reference/.gitignore`'s pre-existing `golden_out*/` pattern (the same one
`layer_golden.py`'s own dumps use), and stays that way -- these fixtures are exactly reproducible
from `(weights, seed=0)`, so shipping ~9 MB of binary `.npy` in a public repo buys nothing a
regeneration step doesn't. `dflash2_selftest.py` therefore does not hard-fail when a fixture
directory is missing: each of the three test functions calls `_ensure_fixture(name, weights)`
first, which regenerates that fixture (via `gen_fixture_{a,b,c}`) exactly once if its
`manifest.json` isn't present, then proceeds with the normal bit-identical checks against the
freshly-written files. A clean checkout therefore runs the suggested `ctest` line below with no
manual fixture-generation step.

- **Fixture A** (`fixture_a/`): synthetic vocab 4096, `N=40` injected positions of seeded-random
  features (`~N(0,1)` per element, i.e. ~unit RMS per channel like a real residual stream), a
  seeded-random anchor token, **real draft weights** (the actual Q8_0 GGUF, dequantized to fp32).
  Every intermediate the task asked for is dumped: `features`, `g_encoded`, per-layer
  `injected_k_l{0..4}`/`injected_v_l{0..4}` (post-norm, post-rope K; raw V), per-layer
  `x_post_attn_l{0..4}`/`x_post_ffn_l{0..4}`, layer-0's `attn_conv_in_l0`/`attn_conv_out_l0`,
  `x_final_normed`, `logits`, `cand`, `unary`, `gate`, per-position `score_t{1..7}` matrices, and
  `drafted_tokens`. Also `logits_bf16_weights`/`drafted_tokens_bf16_weights`: the same round run
  against every linear weight (not norms/tables) rounded fp32->bf16->fp32 first (round-to-nearest-
  even, the exact bit trick `third_party/libr4d/r4d_dflash_conv_body.h`'s `r4d_f2bf` uses). **This
  is NOT a bit-exact target for a real bf16 GEMM path** -- only the WEIGHTS are rounded here; every
  downstream op still runs this reference's own numerics, which accumulate in float64
  (`dflash2_ref.py`'s `rmsnorm` casts to float64 internally; `attention_gqa`'s two einsums run in
  float64), while an actual bf16 GEMM accumulates in fp32 with a different reduction order and
  cannot reproduce those exact bits. Treat `logits_bf16_weights`/`drafted_tokens_bf16_weights` as a
  determinism/regression gate (did the weight-rounding path change) plus a rough precision
  expectation, not a bit-exact target; a porter should expect the real GEMM's logits to differ from
  this fixture by a small but nonzero max-abs error (not measured here -- no vectorized bf16-GEMM
  reference exists in this repo to quantify it against). The only true bit-level anchor for the
  eventual C++/HIP port is a real ROCmFPX `graph<false>` capture compared via this file's own
  `--real <npz>` path once a GPU is free for that capture; nothing in this task compared against
  ROCmFPX's actual intermediates, so `dflash2_selftest.py` proves this Python is internally
  deterministic, not that it agrees with the reference implementation's actual numbers.
  - **The full `[4096,5120]` synthetic embedding table and lm_head are NOT dumped** (each is
    ~80 MB fp32, alone far over this directory's 60 MB budget). They are exactly reproducible from
    `(hidden=5120, vocab=4096, seed=0)` via `SyntheticTarget.__init__`
    (`RandomState(seed).randn(vocab,hidden)*0.02` for the embedding table, then the SAME
    `RandomState` instance drawing the lm_head immediately after) -- `manifest.json`'s
    `target_provider` field records this formula, and `dflash2_selftest.py` reconstructs both
    in-memory rather than loading them from disk. Only the 2 rows actually read (`anchor_id`,
    `mask_id`) are dumped, as `embedding_rows_used`/`embedding_rows_used_ids`.
- **Fixture B** (`fixture_b/`): fixture A's exact setup but `N=2100` injected positions, to exercise
  the 2048-position sliding-window truncation during the draft block's attention over the injected
  cache. Per task item 3, only `logits`/`cand`/`unary`/`gate`/`score_t{1..7}`/`drafted_tokens` are
  saved (no `features`/embedding/lm_head dump -- all reproducible from `seed=0` the same way as A).
- **Fixture C** (`fixture_c/`): fixture A's exact setup with `p_min=0.3`, to exercise the selector's
  early stop. Drafts 5 tokens instead of A's 7 (`p_min=0` never stops early) -- the first 5 tokens
  are bit-identical to fixture A's first 5, confirming the walk itself is unperturbed by the early
  stop and only the truncation point changes.

`manifest.json` in each fixture directory records every array's shape/dtype/file plus the
seed/vocab/anchor/mask/block_size provenance needed to reproduce it independently.

## 10. `dflash2_selftest.py`

Re-derives every one of the three fixtures from their stored/documented inputs (loading the real
Q8_0 GGUF fresh each time -- no golden weight dump needed) and asserts **bit-identical** (`np.
array_equal`, not a tolerance) agreement against the stored arrays for fixture A's full
intermediate set and B/C's smaller output set. Runs in ~20s on CPU, no checkpoint needed. **This is
a determinism gate on `dflash2_ref.py` itself (same code, same inputs, same bits every run) -- it
does not by itself prove agreement with ROCmFPX's actual `graph<false>` output**; that anchor is
the `--real <npz>` path (section 8) against a real GPU capture, not yet done. An
optional `--with-real-sanity` pass (task item 5's suggested check; off by default since it needs
the real 27B checkpoint) loads the real target's embedding/lm_head via
`tools/reference/common.py`'s `ShardIndex` (streaming: only the needed embedding rows + one
`lm_head.weight` tensor, never a full shard sweep) and asserts (a) every drafted token id is a
valid vocab id and (b) the selector's chained walk differs from plain per-position argmax at least
once -- both passed in ~6.5s on this checkoint (well within CPU time/RAM budget, so this was NOT
skipped).

**Suggested ctest registration** (the Integrate stage owns `tests/CMakeLists.txt`, not this task):

```cmake
add_test(NAME reference_dflash2
         COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tools/reference/dflash2_selftest.py)
```

## 11. Files

- `tools/reference/gguf_min.py` -- from-scratch GGUF v3 reader + Q8_0 dequantizer (numpy/struct
  only, no `gguf-py` import). Verified against the real 81-tensor, 48-metadata-key DFlash2
  container end to end (every shape/metadata value in this document's tables was read through this
  parser, not assumed).
- `tools/reference/dflash2_ref.py` -- the reference forward pass, fixture generator, and `--real`
  inspection mode.
- `tools/reference/dflash2_selftest.py` -- the determinism gate described in section 10.
- `tools/reference/golden_out/dflash2/{fixture_a,fixture_b,fixture_c}/` -- the golden fixtures
  (section 9).
