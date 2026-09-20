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

## 7. Integration plan for r4dx (not built by this task -- for whoever picks up the C++ port)

1. **Target-side feature capture**: during both prefill and `VerifyWindow`, capture the residual
   stream entering layers `{6,20,34,48,62}` (0-based HF indexing, i.e. the OUTPUT of layers
   `{5,19,33,47,61}`) for every position, analogous to how `docs/mtp.md`'s `h_seed` is already
   captured off `Model::RunChunk`.
2. **Draft KV ring**: a per-layer (5 layers), per-head (8 kv heads), 128-dim ring sized to
   `sliding_window = 2048` positions -- much smaller than the backbone's own KV cache, and (unlike
   the backbone) never needs fp8 paging since it is tiny.
3. **Injection**: after every prefill chunk and every verify round's accepted prefix, run the
   encoder (`fc` + `output_norm_enc`, a single small GEMM+norm) once, then 5 tiny K/V GEMMs (one
   per draft layer) writing into the ring at the real absolute position -- mirrors
   `MtpHead::PrimeKv`'s call pattern (`docs/mtp.md`) closely enough that the two should likely share
   a "keep an auxiliary head's KV in lockstep with the backbone" helper.
4. **Draft loop**: build the 8-token noise block, run the 5-layer non-causal SWA attention forward
   (all fp8/bf16-quantizable GEMMs; the dynamic conv is small elementwise work, a good
   `r4dx`-owned-kernel candidate akin to `r4dx_rope_partial_mrope_bf16`), then the selector walk
   (score matrices are `16x16` at most -- trivial to run on host, per token position, rather than
   as a device kernel, unless profiling later shows otherwise).
5. **Verify + commit**: reuses the SAME verify/accept/commit machinery `docs/mtp.md` already
   documents (longest-greedy-matching-prefix acceptance, corrected/bonus token) -- DFlash2 is a
   drop-in alternative DRAFT SOURCE, not a different verify contract.

## 8. `--real` mode npz schema

`tools/reference/dflash2_ref.py --real <path.npz> --target-dir <dir>` expects:

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
