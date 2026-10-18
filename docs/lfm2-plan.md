# LFM2.5-24B-A2B support for r4dx -- design and task plan

Branch `lfm2` (from `main` at `7e280b0`). Written 2026-10-14. **Design only**: no source file was edited, no GPU
workload or ctest was run, nothing was downloaded or installed while writing it. Structure follows
`docs/gemma4-plan.md` (the precedent for a second model family behind `TextModel`). Where it says MUST, do exactly
that; anything tagged **UNVERIFIED** is inferred or read from a summary and is settled by the task named next to it
before the code that depends on it is written. Section 12 lists every UNVERIFIED item, section 11 the open questions.

## Summary

`LiquidAI/LFM2.5-24B-A2B` (`lfm2_moe`, 24.0 B total / 2.45 B active) is a 40-layer hybrid: 30 gated short-convolution
layers (kernel 3, no activation) and 10 full-attention GQA layers (head_dim 64), each followed by an FFN that is dense
SwiGLU in layers 0-1 and a 64-expert top-4 sigmoid-routed MoE in layers 2-39 (no shared expert). It becomes a third
`TextModel` family (`Lfm2Model`, `Lfm2TpModel`) with its own container arch; the Qwen hot path and the Gemma path are
not touched. bf16 TP=2 is the reference-exact baseline (48 GB does not fit one 32 GB card); the shipping format is a
trellis mix of about 4.5 bpw (about 14 GB) on ONE card, built with the Gemma tooling ported to a `lfm2_moe` arch entry.
First milestone is text only: forward pass, tokenizer, chat template, thinking split, Pythonic tool-call parser,
server smoke. No drafter, no vision. Every GPU step is a command handed to the user (GPU runs need approval).

## 1. Agreed decisions (user, 2026-10-14; fixed)

1. **Side model.** Separate `Lfm2Model` / `Lfm2TpModel` behind `TextModel` (like Gemma). Qwen path untouched.
2. **Weights.** bf16 TP=2 first as the reference-exact baseline; then trellis mix about 4.5 bpw on one card as the
   shipping format, reusing the Gemma trellis / Hessian / oracle / mix tooling, **unrotated** (Gemma's final choice,
   docs/gemma4-plan.md 9.13) unless a reason is found (none found; section 7.1).
3. **First milestone scope.** Text forward + tokenizer + chat template + thinking split + Pythonic tool-call parser +
   server smoke. No drafter / MTP, no vision.
4. **MoE.** A FRESH implementation sized for this model. The deleted Qwen3.8 `q38-*` branches are not resurrected.
5. **Gate.** fp32 CPU truth; per segment group `KL(fp32||r4dx) <= 1.5 x KL(fp32||HF-bf16) + eps`, top-1 >= HF-bf16
   minus 1 pt, on the repo KL corpus chat-templated; trellis judged on the increment over r4dx bf16 (section 6.4).

## 2. Model facts

### 2.1 Verified (read locally; sources in brackets)

`config.json`, `generation_config.json`, `chat_template.jinja` were read from `E:\models\LiquidAI\LFM2.5-24B-A2B\`
(the download is in progress: no safetensors and no tokenizer.json there yet). The code semantics were read from
`D:\venvs\r4dx-gemma-ref\Lib\site-packages\transformers\models\lfm2_moe\modeling_lfm2_moe.py` (transformers 5.18.0;
the checkpoint was saved with 5.12.1).

| Item | Value |
|---|---|
| model_type / class | `lfm2_moe` / `Lfm2MoeForCausalLM` |
| layers / hidden / vocab / max_pos | 40 / 2048 / 128000 / 131072 (rope has NO scaling: `rope_type default`) |
| layer_types | `full_attention` at 2, 6, 10, 14, 18, 22, 26, 30, 34, 38; `conv` elsewhere (30 conv + 10 attention) |
| attention | 32 q heads, 8 kv heads (GQA 4), `head_dim` NOT in config -> `hidden/heads` = **64**, scale `64^-0.5`, no bias |
| QK norm | `q_layernorm`, `k_layernorm`: `Lfm2MoeRMSNorm(64)` per head, BEFORE rope; rope = `rotate_half` pairs `(i, i+32)`, theta **1e7**, full rotary (64 dims); cos/sin are rounded to the activation dtype in HF |
| RMSNorm | `weight * (x * rsqrt(mean(x^2) + eps)).to(input_dtype)` (plain weight, NOT 1+w; fp32 compute, bf16 rounding BEFORE the weight multiply in HF), `norm_eps` **1e-5** |
| decoder layer | `h += mixer(operator_norm(h))`; `h += ffn(ffn_norm(h))`; after layer 39 `embedding_norm` (the FINAL norm, a misleading name), tied lm_head, no logit softcap, no embedding scale |
| ShortConv | `in_proj` H->3H (no bias; `conv_bias` false), chunk order **B, C, x**; `h = B*x`; depthwise causal `Conv1d` kernel `conv_L_cache` = **3**, groups = H, weight `[H,1,3]`, no bias, **no activation**; `y = C * conv(h)`; `out_proj` H->H. State = last 2 values of `h` per channel (zeros at start) |
| dense FFN (layers 0, 1) | `feed_forward.{w1,w3,w2}`, `w2(silu(w1 x) * w3 x)`, I = 11776 |
| MoE FFN (layers 2-39) | 64 experts, top-4, `moe_intermediate_size` 1536, no shared expert; router `gate.weight [64,2048]`: `s = sigmoid(x @ W^T)`; selection by top-4 of `s + expert_bias` (`expert_bias` fp32 buffer `[64]`, `use_expert_bias` true); weights = the UNBIASED `s` of the selected experts, divided by `(sum + 1e-6)` (`norm_topk_prob` true), times `routed_scaling_factor` (**1.0**); experts: `gate_up_proj [E, 2I, H]` (first half = gate), `down_proj [E, H, I]`, `silu(gate) * up`; outputs accumulated per token (`index_add_`) |
| HF router precision | bf16 matmul, bf16 sigmoid, bf16 weights; near-ties and top-4 flips differ between bf16 and fp32 (a noise source for the gate; r4dx computes the router in fp32) |
| ids | bos 124894 (`<|startoftext|>`), eos 124900 (`<|im_end|>`; `generation_config` eos list is `[124900]`), pad 124893 |
| dtype | bf16 (+ F32 `expert_bias`), `tie_word_embeddings` true (no `lm_head` tensor), transformers 5.12.1 |
| generation defaults | `do_sample` true, temperature 0.1, top_k 50, repetition_penalty 1.05 (`generation_config.json`); model card: always thinks |
| HF state_dict names (from the modeling code) | `model.embed_tokens.weight`, `model.embedding_norm.weight`, `model.layers.N.{operator_norm,ffn_norm}.weight`; attention `self_attn.{q,k,v,out}_proj.weight`, `self_attn.{q,k}_layernorm.weight`; conv `conv.conv.weight`, `conv.in_proj.weight`, `conv.out_proj.weight`; dense `feed_forward.{w1,w2,w3}.weight`; MoE `feed_forward.gate.weight`, `feed_forward.expert_bias`, `feed_forward.experts.{gate_up_proj,down_proj}` (3-D, no `.weight` suffix) |

Chat template (verified by reading it): `<|startoftext|>` then `<|im_start|>role\n...<|im_end|>\n`; a leading system message
is merged with `List of tools: [<tojson tool>, ...]` into the system prompt; assistant history keeps
`<think>reasoning</think>` only for assistant messages AFTER the last user message (or `preserve_thinking`), otherwise
strips it (also strips a `</think>` prefix found in content); assistant tool calls render as
`<|tool_call_start|>[name(a='x', b=1), ...]<|tool_call_end|>` (strings single-quoted with `\\ \' \n \r` escaped;
lists/dicts via `tojson`; **a string `arguments` value raises** -- the server must pass a mapping); role `tool` is
rendered as a plain turn; `add_generation_prompt` appends `<|im_start|>assistant\n<think>` (no trailing newline, no
thinking switch in the template); `{% generation %}` tags appear (minja already treats them as a no-op, minja.hpp:2606);
`CONTINUE_FINAL_MESSAGE_TAG ` handling.

### 2.2 Sizes (computed from the config)

Per layer: attention 10.5 M params, ShortConv 16.8 M, dense FFN 72.4 M, one expert 9.44 M (MoE layer 604 M = 64 experts),
router 0.13 M. Totals: experts 38 x 604 M = **22.95 B (95.7 %)**, mixers 0.61 B, dense FFN 0.14 B, embedding 0.26 B =
23.97 B. Active per token: 38 x 4 x 9.44 M = 1.43 B experts + 0.61 B mixers + 0.14 B dense + 0.26 B head = 2.45 B.
bf16 = 47.9 GB. Trellis 4.5 bpw: experts 12.9 GB, rest 0.57 GB, embedding bf16 0.52 GB, lm_head w4a16 0.15 GB = about 14 GB.
KV (bf16, 10 layers x 8 heads x 64 x K,V): 2048 B/token/layer = 20 KB/token = **2.5 GiB at 131072**.
ShortConv state: 30 x 2 x 2048 x 2 B = 245 KB (a sequence's whole recurrent state).

## 3. Design: text model

### 3.1 Dispatch, config, TextModel fit

- `src/model/arch.h/.cpp`: `Arch::kLfm2Moe`; `ArchName` becomes a switch (`"lfm2_moe"`); `DetectArchFromMetadata` accepts
  `model_arch == "lfm2_moe"` and `model_config.model_type`/`architectures` naming `lfm2_moe` / `Lfm2MoeForCausalLM`;
  `ResolveContainerMaxCtx`: default 131072 (the config value), no extended opt-in (no rope scaling exists beyond it;
  an explicit larger value throws). The Qwen/Gemma branches stay byte-identical (existing arch tests extended).
- `src/model/lfm2_config.h` (header-only, HIP-free, strict `FromJson` like `GemmaConfig`): geometry above, `layer_types`
  (asserted: 10 attention layers at 2 mod 4, `num_dense_layers` 2, `num_experts` 64, top-k 4, experts' I 1536, conv kernel
  3), `head_dim = hidden / heads` when absent (VERIFIED), flags `router_fp32`, `norm_topk_prob`, `routed_scaling_factor`;
  `Shard(world, rank)`: q heads and kv heads / world (gqa stays 4), conv channels / world, `intermediate_size` and
  `moe_intermediate_size` / world, vocab slice (128000 % 32 == 0 -> tie allowed); K constraints per rank checked
  (attn-o K 1024, conv out K 1024, dense w2 K 5888 = 23 x 256, expert down K 768 = 3 x 256; all pass the K % 256 rule that
  Gemma M1-2 relaxed; K % 512 fails for 5888 and 768 -- the relaxed `FallbackTuning` is a prerequisite). `ToModelConfig()`
  fills the generic fields (GDN fields 0, `arch` = kLfm2Moe) so `TextModel::Config()` works.
- `LoadTextModel` (`tp_model.cpp`) dispatches `kLfm2Moe` to `LoadLfm2TextModel` (new `lfm2_local_text_model.cpp`,
  mirrors `gemma_local_text_model.cpp`): world 1 -> `Lfm2LocalTextModel`, world 2 -> `Lfm2TpModel`. TextModel map:
  Prefill / DecodeStep / Greedy / Sampled / Reset / Save+RestoreCheckpoint / `*Profiled` implemented; `PrefillMultimodal`
  with spans throws, `HasVision`/`MtpEnabled`/`DflashEnabled` false, MTP/DFlash decode throw `TpUnsupportedError`.
- Why not a flag in `Model`: same reasons as Gemma (GDN, mrope, MTP, 64-row slicing everywhere). `Lfm2Model` is
  deliberately given the SAME public surface as `GemmaModel` (minus vision, features, DFlash) so `Lfm2TpModel` can be a
  copy-adapt of `gemma_tp_model.{h,cpp}` (rank threads, `TpGroup`, watchdog, recovery); templating the facade over the
  engine type is a later dedupe (open question 3).

### 3.2 Layer graph (`Lfm2Model::RunChunk`, T <= 256 rows, one sequence)

```
embed gather (device table, bf16, NO scale) -> h
for i in 0..39:
  x = operator_norm(h)                                   r4dx_rmsnorm_plain_bf16 (hidden 2048, eps 1e-5)
  conv layer:   bcx = in_proj(x)  [T, 6144]; y = shortconv(bcx, state_i, w_i)   [T, 2048]; m = out_proj(y)
  attn layer:   q,k,v = proj(x); q,k = QKnorm(q,k) (rows T*H, hidden 64); rope; KV write; attention; m = out_proj(o)
  [TP] all-reduce(m);  h += m
  x = ffn_norm(h)
  dense (i<2):  g = gate_up(x) [T, 2*11776]; a = silu_mul(g); f = down(a)
  MoE   (i>=2): router(x) -> ids/w; routed experts (3.5); f = sum_j w_j * expert_{ids_j}(x)
  [TP] all-reduce(f);  h += f
h = embedding_norm(h)  -> tied lm_head -> fp32 logits (no softcap)
```

Residual stream: **bf16 first**, reusing the existing bf16 kernels (`r4dx_residual_add_bf16`, plain rmsnorm). The fp32
residual that Gemma needed (massive activations, docs/gemma4-plan.md 3.2.1) is a contingency, `R4DX_LFM2_RESID=fp32`, built
only if the L1 gate misses while the L0 noise measurement shows the HF-bf16 yardstick is stable (task L1-17). Rationale:
LFM has no sandwich norms and no `layer_scalar`; the fp32-residual kernels would be new (add + norm fusion), so they are
not free. Norm kernels compute in fp32 and round once (closer to the fp32 truth than HF's double rounding; the gate is
against the truth).

### 3.3 ShortConv: kernels and state lifecycle

Decode and chunked prefill are the same math (no recurrence through the output, only through `h = B*x`), so ONE kernel
serves both. New r4dx-owned kernel (`src/kernels`, `lfm2_kernels.hip` / `lfm2_kernels.h`, plain C ABI like
`gemma_kernels.h`):

`r4dx_lfm2_shortconv_bf16(bcx, state, weight, out, T, H, bcx_row_stride, stream)`:
`bcx [T, 3H]` bf16 (B | C | x), `weight [H, 3]` bf16 (taps oldest..newest), `state [2, H]` bf16 (`state[0]` = h at t-2,
`state[1]` = h at t-1), `out [T, H]` bf16. Per element: `h_t = bf16(B_t * x_t)`; `acc = w0*h_{t-2} + w1*h_{t-1} + w2*h_t`
in fp32 over bf16 operands, rounded once to bf16 (HF `F.conv1d` bf16); `out = bf16(C_t * bf16(acc))`. Fusing the two
gates and the conv into one pass is cheap and removes two [T,H] round trips; `in_proj` / `out_proj` stay ordinary GEMMs
(`ApplyLinear`), so the trellis layout needs nothing special. Grid: channels x time tiles (each thread owns a channel and
a run of time steps, keeping the 2 previous `h` in registers; the 2 halo values of a tile come from `bcx` or `state`).
State update runs in stream order after the reads (a 2-column micro-kernel, or ping-pong buffers): for `T >= 2` the new
state is the last two `h` of the chunk, for `T == 1` it shifts. Tests (CPU reference in a header-only
`lfm2_ref.hpp`, GPU test compares): T = 1, 2, 3, 63, 64, 255, 256; a 600-row sequence fed as 256 + 256 + 88 equals one
pass; decode after prefill; zero state start.

State lifecycle (`Lfm2ConvState`, device, bf16 `[n_conv_layers=30][2][H_local]`, 245 KB, 123 KB per rank under TP):
- Reset: zeros (equals HF's zero causal padding). A new request that is not a prefix extension resets it with the KV.
- `SaveCheckpoint` D2D-copies the state into a spare (needs `prompt_checkpoint`); `RestoreCheckpoint` copies it back and
  sets `pos_` to the checkpoint position. KV needs no copy (positional overwrite, as Gemma full layers). Net memory: +245 KB.
  The recurrent state cannot be rewound like a KV prefix, so the server's `PrefixState` semantics are exactly GDN's
  (docs/server.md "Prefix cache"): reuse `fed()` when the new prompt strictly extends it, else restore the prompt checkpoint
  and feed the tail, else reset. `prefix_state.h` needs no change; the engine's `ckpt_back` for LFM is the token count of the
  generation suffix `<think>` (1 token if it is a single added token, UNVERIFIED) because a replayed history drops reasoning
  that precedes the last user turn (template rule above), so the state at the end of the prompt minus the suffix is the
  last shareable one.
- Speculative rollback is out of scope (no drafter). If it is ever needed, the state only needs the last 2 `h` of each
  verified row, which the kernel already computes.

### 3.4 Attention (10 layers)

- Geometry per layer: 32 q heads / 8 kv heads x 64, GQA 4. libr4d has only `h256_gqa6` and `h256_gqa2` attention
  (`third_party/libr4d/r4d.h`); there is no h64 kernel. **L1 uses the r4dx-owned stage-0 reference kernel**
  (`r4dx_gemma_attn_ref_bf16kv`, `attn_ref.h`: generic head_dim in [8, 512], any GQA, `window` 0 = full causal,
  `scale` argument = 0.125) -- exact, slow at long context. **L4-1 (done, 2026-10-18) replaced it on the LFM2 path** with the
  libr4d `h64_gqa4` bf16-KV unit (`r4d_attn_paged_h64_gqa4.hip`: the gqa6 prefill / decode templates instantiated at head_dim
  64, GQA 4, KVP 1); the reference kernel stays as the fallback for any other head geometry and as the test oracle. For scale,
  the reference kernel re-reads K/V per q head: about 1 GB per layer per token at 131072 context.
- **KV dtype: bf16 for all 10 layers** (2.5 GiB at 131072, 1.25 GiB per rank). fp8 would save 1.25 GiB of a budget
  that has room on the trellis card and none of the Gemma full-layer risks apply, but it needs a calibration port and a KL
  A/B; deferred (open question 7). Reuse `GemmaKvCache::Contiguous(kv_heads, 64, max_ctx, kBf16)` and
  `r4dx_kv_write_paged_*` bf16 variant as `GemmaAttnLayer` already does.
- New thin `Lfm2AttnLayer` (`lfm2_attention_layer.hpp`, in the style of `gemma_attention_layer.hpp`, without k_eq_v,
  v_norm, windows, rotation): `q,k,v` projections (separate; no output gate), `r4dx_rmsnorm_plain_bf16` over rows `T * heads`
  with hidden 64 for q and k, `r4dx_rope_neox_bf16` (rotate_half; head_dim 64, theta 1e7; the kernel is documented
  generic -- a 64-wide golden pins it, L1-9), KV write, reference attention, `out_proj`. `GemmaAttnLayer` is not reused as is.
- Positions are plain 1-D; no mrope.

### 3.5 MoE (fresh; 38 layers, 64 experts, top-4, I = 1536)

Router kernel (`r4dx_lfm2_router_f32`): one workgroup per token row; 64 dot products of length 2048 (bf16 x bf16, fp32
accumulate), `s = sigmoid(logit)` in fp32, `score = s + expert_bias[e]` (fp32), top-4 by score (ties -> lower expert
index), weights `s[sel] / (sum(s[sel]) + 1e-6) * routed_scaling_factor` in fp32. Output: `ids int32 [T,4]`, `w fp32 [T,4]`.
Replicated on both TP ranks from identical bits, so both ranks pick identical experts. This deliberately computes the
router in fp32 (HF does it in bf16); a router golden (ids exact vs the fp32 CPU reference, weights rel 1e-3) pins it, and
the per-layer id agreement vs the truth on the corpus is reported (proposal: >= 99 % of (token, slot) pairs).

Expert weights: **per-expert linears** `moe.experts.{e}.gate_up.{layout}` (N = 2I = 3072, K = 2048, gate rows first) and
`moe.experts.{e}.down.{layout}` (N = 2048, K = 1536): each is an ordinary `QuantLinear`, so bf16, trellis (own `suh`/`svh`
sign vectors and per-linear KB) and w4a16 all work with no new layout, and the per-layer expert table (a device array of
the 64 weight pointers) is built at load for the L4 kernels.

Staging (correctness first, then speed; the staged design is what keeps the KL gate independent of perf kernels):
- **L1 decode (M = 1):** router kernel, then a host readback of `ids/w` (16 B + 16 B, one sync per MoE layer, 38 per token),
  then for each of the 4 selected experts: `ApplyLinear(gate_up)` -> `r4dx_silu_mul_bf16` -> `ApplyLinear(down)`
  -> `r4dx_lfm2_moe_combine` (fp32 weighted sum over the 4 slot outputs in fixed slot order, one rounding to bf16).
  Existing m64 GEMMs, existing tuning fallback. Slow but exact.
- **L1 prefill (chunk of T <= 256 rows):** router; device counting sort of the T x 4 (token, slot) pairs by expert
  (64-bucket histogram + exclusive scan, one block; deterministic order within an expert = token order); `expert_counts[64]`
  D2H once per layer (256 B, one sync per layer per chunk, 38 per chunk); per non-empty expert: gather its rows, `ApplyLinear`
  gate_up (M = c_e, sliced by the existing <= 64-row loop), silu_mul, down, write the `[c_e, 2048]` result into a
  `[T, 4, 2048]` slot buffer at (token, slot); then the same deterministic combine kernel. No atomics, so the result does
  not depend on scheduling. Per chunk about 38 x 64 x 2 launches plus the 12 GB (per rank at TP=2) weight read: tens of ms
  per layer-set, acceptable for L1.
- **L4 decode:** device-indexed grouped kernels: (a) router -> (b) gate_up GEMV over (4 slots x N tiles) reading
  `ids[slot]` from device memory and the expert base from the pointer table, with the silu_mul epilogue -> (c) down GEMV
  accumulating the weighted sum over the 4 slots in fp32 in one kernel. Target 40-60 us per MoE layer at M = 1 (the order of
  magnitude measured for the abandoned Qwen3.8 MoE work: 43-75 us/layer, from memory notes, not a spec). bf16 first (r4dx-owned);
  the trellis variant either extends libr4d's trellis m64 kernel with an indirect weight pointer (needs the Qwen ISA-diff guard) or
  keeps the host-sync path if the measured sync cost is below 15 % of a decode step (decision at L4-3 from L3 numbers).
- **L4 prefill:** one grouped GEMM over the sorted rows with a device `expert_offsets[65]` (no host sync, no per-expert launches),
  M-tile 16/32/64 per expert, split by rows when an expert holds more than 64 rows; bigger prefill chunks for MoE layers
  (1024 rows read each expert once; the dense 256-row chunk size is not sacred here) are measured.

Numerics inside an expert follow HF: gate and up are bf16 GEMM outputs, `silu(g) * u` (single-rounding `r4dx_silu_mul_bf16`
is within noise of HF's two roundings), down is a bf16 GEMM output; the router-weighted sum and the cross-expert accumulation
are fp32 (HF accumulates in bf16 with `index_add_`; the truth is fp32). Dense layers 0-1 use `mlp.gate_up` / `mlp.down`
exactly like the Qwen `Mlp` shape (SiLU), reusing `Mlp`'s kernels without the class if its API is Qwen-coupled.

### 3.6 TP=2 split

Rule: **everything is split inside a layer; two all-reduces per layer (80 per token), both on the bf16 [T, 2048]
sublayer output, residual and norms replicated**, exactly the Gemma structure (docs/tp.md; measured all-reduce about 7.5 us
for 10 KiB, 13 us for 80 KiB; here a decode row is 4 KiB).

| Part | Split |
|---|---|
| attention | q heads 32 -> 16, kv heads 8 -> 4 per rank (gqa 4), q/k/v column-parallel, `out_proj` row-parallel (K 1024); q/k layernorm weights `[64]` replicated |
| ShortConv | **by channel**, which is exact for a depthwise conv: `in_proj` is a THREE-segment column split (rank r takes channels `[r*1024, (r+1)*1024)` of each of B, C, x -- the existing `gate_up` rule is two segments, so the rule gains a segment count), `conv.weight` rows split, state `[2][1024]` per rank, `out_proj` row-parallel (K 1024) |
| dense FFN | `w1|w3` column-parallel (two segments, N 5888 each), `w2` row-parallel (K 5888) |
| MoE experts | **TP-split EVERY expert** (gate and up halves 1536 -> 768 per rank, down K 1536 -> 768); router and `expert_bias` replicated; one all-reduce on the weighted sum |
| embedding / head | table replicated (gather), tied lm_head = the rank's vocab slice rows `[r*64000, (r+1)*64000)`, the existing vocab-split merges (`tp_vocab.h`) |

Why TP-split experts and not expert-parallel (32/32): (1) **decode is memory-bound and EP is imbalanced**: with top-4 of 64
split 32/32 the critical-path rank reads on average 2.75 experts per token (P(2,2) 37.5 %, P(3,1) 50 %, P(4,0) 12.5 %) versus
exactly 2.0 expert-equivalents per rank for TP-split, i.e. about 27 % more bytes on the slow rank, plus a variable-length
kernel launch per rank; (2) prefill reads every expert on both layouts so the byte volume is identical (about 0.6 GB per rank
per MoE layer), and TP-split keeps per-expert M equal on both ranks; (3) it reuses the dense gate_up/down rules and the single
all-reduce per layer unchanged, so `tp::RuleFor` needs only the per-expert names; (4) the expert linear shapes that result
(N 1536 K 2048; N 2048 K 768) are all multiples of 128 (trellis-legal) and of 256. EP's advantage is bigger per-rank
GEMMs, irrelevant for M <= 16 rows per expert; its disadvantage is the per-token imbalance. The shipping format is TP=1 where
the question disappears. **Memory consequence**: both layouts give about 25 GB per rank at bf16 (section 8); revisit only if
device 0 (desktop, about 4.6 GB taken) cannot hold its share (risk R1).

## 4. Converter and container

Same safetensors-compatible shell as `docs/container-format.md`; `__metadata__` gains `"model_arch": "lfm2_moe"`,
`"norm_kind": "plain"`; `model_config` is the verbatim config.json. Container names (proposal; `{L}` = the layout suffix the
existing converter appends):

| Container tensor | HF source | Notes |
|---|---|---|
| `text.embed_tokens` | `model.embed_tokens.weight` | bf16 `[128000, 2048]`, no scale |
| `text.final_norm` | `model.embedding_norm.weight` | bf16 `[2048]` |
| `lm_head.{L}` | tied to the embedding | written untied as Gemma does; bf16 in the bf16 container, w4a16 g32 in the trellis one |
| `text.layers.{i}.operator_norm`, `.ffn_norm` | same | bf16 `[2048]` |
| `text.layers.{i}.attn.{q,k,v,o}.{L}` | `self_attn.{q,k,v,out}_proj` | no gate, no bias |
| `text.layers.{i}.attn.{q_norm,k_norm}` | `self_attn.{q,k}_layernorm` | bf16 `[64]` |
| `text.layers.{i}.conv.in_proj.{L}`, `conv.out_proj.{L}` | `conv.in_proj`, `conv.out_proj` | in_proj N 6144 (B,C,x order kept) |
| `text.layers.{i}.conv.weight` | `conv.conv.weight` `[2048,1,3]` | squeezed to bf16 `[2048, 3]` (passthrough, never quantized) |
| `text.layers.{i}.mlp.gate_up.{L}`, `mlp.down.{L}` (i < 2) | `feed_forward.{w1;w3}`, `w2` | gate rows first |
| `text.layers.{i}.moe.router` | `feed_forward.gate.weight` | bf16 `[64, 2048]` passthrough |
| `text.layers.{i}.moe.expert_bias` | `feed_forward.expert_bias` | **fp32 `[64]`** (stored widened like Gemma's `layer_scalar`: fp32 `[1, 64]` if the writer wants 2-D) |
| `text.layers.{i}.moe.experts.{e}.gate_up.{L}`, `.down.{L}` | slices `[e]` of `experts.gate_up_proj` / `down_proj` | per-expert linears (3.5) |

The on-disk HF key layout is **UNVERIFIED** until the header is dumped (L0-3): transformers 5 may serialize the fused 3-D
tensors, or the older per-expert `experts.{e}.w1/w2/w3.weight`. The converter's LFM layout handles both (fused: slice;
per-expert: concatenate `w1` then `w3` into gate_up), and a coverage audit (as the Gemma branch has: every checkpoint name
must be consumed or allow-listed) fails the run on any unconsumed tensor. Per-expert linears mean about 38 x 64 x 2 = 4864
linears plus their trellis side tensors (about 20 000 header entries, a few MB of JSON; `ReadContainerMetadata`'s 1 GiB header
guard is not a concern).

Code structure: `src/convert/lfm2_layout.{hpp,cpp}` (`AddLfm2TextStack(Lfm2Kit&, ...)` with the same
std::function-handle pattern as `gemma_layout.*`; `RunConvert` dispatches on `model_type`, the Qwen/Gemma branches are
untouched), `ShardedModel` already reads a sharded checkpoint (10 shards + index). `--kv-calib` not needed (bf16 KV);
`--mtp`, vision, audio rejected for this arch. Tests: `tests/convert/test_lfm2_layout.cpp` on a synthetic tiny checkpoint
(`tools/convert_ref/make_tiny_lfm2.py`: 4 layers = conv dense, conv dense, attention MoE, conv MoE; hidden 256, 8 experts,
top-2 so the real naming and both expert key styles are exercised) and a tiny two-layer `lfm2-l6` style test container
(layers 0-5 of the real model, about 7 GB bf16, fits one card) for single-device GPU goldens.
Trellis import: the oracle manifest key for an expert is the HF base name plus an expert suffix
(`...feed_forward.experts.gate_up_proj[e]`); `trellis_import.hpp` maps it to `moe.experts.{e}.gate_up`; a missing expert is
refused before an output file exists (same rule as the Gemma missing-linear check).

## 5. Tokenizer, chat template, tool parser, server

### 5.1 Tokenizer

`tokenizer.json` is 17,905,598 bytes (Git LFS; not downloaded yet), `tokenizer_class` = `TokenizersBackend` (transformers
5 Rust backend), bos `<|startoftext|>`, eos `<|im_end|>`, pad `<|pad|>`. Its `normalizer`, `pre_tokenizer`, `decoder`,
`post_processor`, merge count and added-token ids are **UNVERIFIED** (L0-4 reads the head of the file, CPU only). Expectation
(NOT assumed in code): a byte-level BPE like LFM2's family, pre-tokenizer `Sequence[Split(regex), ByteLevel]`. The in-house
`kByteLevelRegex` loader (`bpe_tokenizer.cpp`) accepts exactly two regexes (`kQwen2SplitExpected`, `kQwen35SplitExpected`) and
throws on anything else, and its regex is dispatched to a vendored llama.cpp splitter by trigger string. Likely new work:
(a) accept the LFM regex if it differs (the classic divergence is digit grouping, `\p{N}{1,3}` vs `\p{N}`, and `\p{M}`
handling), reusing the llama.cpp unicode splitter if the pattern already has a dispatcher, else a new trigger + validation;
(b) NFC normalizer if declared (the existing KNOWN GAP: not applied; `allow_unimplemented_normalizer` must stay false and
the golden set NFC-only); (c) `keep_special_on_decode` for `<|tool_call_start|>`, `<|tool_call_end|>` and, if the think
tags are flagged `special`, `<think>`/`</think>`; (d) bos on raw prompts per the post-processor (the template emits
`<|startoftext|>` as text, so chat prompts are encoded with `parse_special = true` and must not get a second BOS).
Validation like Gemma: `tools/tok_ref/gen_golden_lfm2.py` -> `tests/tokenizer/golden_lfm2.json` (English, code, JSON, Thai,
CJK, Arabic, emoji, digit runs of 1..9, newline/space runs, tabs, `\r\n`, non-NFC input, special tokens with `parse_special`
both ways, a 20 KB line, decode cases, chat cases); raw `tokenizers` ids and `AutoTokenizer` ids MUST agree, asserted by
`check_hf_equiv.py` over the KL corpus plus fuzz. The tokenizer directory is the model directory itself
(`E:\models\LiquidAI\LFM2.5-24B-A2B`: tokenizer.json, tokenizer_config.json, chat_template.jinja, generation_config.json).

### 5.2 Chat template

minja loads `{% generation %}` already. UNVERIFIED until L0-6 renders it: the `.split("</think>")[-1] | trim`,
`.endswith`, string slicing `content[:n]`, `message.get`, `is mapping/iterable/string` tests, `namespace`, `default`,
`tojson` key order/spacing for tool definitions, and the `raise_exception` path. Any unsupported construct gets a small
patch in `third_party/minja/minja.hpp` with a note in `third_party/VERSIONS.md` (Gemma precedent M1-8), polyfills OFF for this
family (they would rewrite `tool_calls` into JSON content and destroy the Pythonic format). Server-side: OpenAI
`tool_calls[].function.arguments` is a JSON STRING and this template raises on strings, so the message normalizer parses it
to a mapping before rendering (the Qwen path may already do this; verify). The `{% generation %}` markers do not change the
output.

### 5.3 Dialect, thinking split, tool parser

- `DialectKind::kLfm2`, `Lfm2Dialect()` in `src/server/dialect.{h,cpp}`: reasoning opens IN THE PROMPT with `<think>`
  (no newline; `prompt_reasoning_open = "<think>"`) and closes with `</think>` (Qwen shape, `ReasoningSyntax` already
  supports an in-prompt opener, so `StartState::kInReasoning`); the model ALWAYS thinks, there is no thinking-off prompt:
  `enable_thinking` / `reasoning_effort` are ignored (documented; `usage.reasoning_tokens` always reported); tool delimiters
  `<|tool_call_start|>` / `<|tool_call_end|>`; `chat_template_polyfills = false`; `bos_on_raw_prompt` per 5.1; eos ids `[124900]`;
  `CheckpointSuffix = "<think>"`; `--dialect {auto|qwen35|gemma4|lfm2}`, auto = `tokenizer_config.json` bos `<|startoftext|>`
  plus the `<|tool_call_start|>` token in the vocab, cross-checked with the container arch (`CheckDialectAgainstArch` keys
  on the arch string: add `lfm2`).
- `lfm2_tool_call_parser.cpp`: recursive descent over `<|tool_call_start|>[ call (, call)* ]<|tool_call_end|>`,
  `call := NAME "(" [ IDENT "=" value ("," IDENT "=" value)* ] ")"`, values: Python strings (single or double quoted with
  `\\ \' \" \n \r \t`), ints, floats, `True/False/None`, lists, dicts with string keys, AND JSON literals
  (`true/false/null`, double-quoted strings) because the template renders nested lists/dicts with `tojson`; the model may also
  emit a JSON tool-call list when the system prompt asks for it (a different dialect knob, not parsed in M1). Output reuses
  `ParsedToolCall` (arguments as a JSON object string). Malformed spans (positional args, unbalanced brackets, truncated by
  `max_tokens`) degrade to content with `had_malformed_call = true`. `ToolStreamGate` takes the 19-byte opener (holdback 18).
  Ground truth from HF: the golden generator records the HF-rendered history strings and a set of Pythonic reply vectors
  (nested args, escapes, unicode, empty args, multiple calls).
- Engine: all edits behind the dialect (Gemma precedent M1-14); the `ckpt_back`/`ReasoningLocator` bookkeeping already
  generalized for Gemma is reused; stop on `<|im_end|>`; a stop after a parsed call gives `finish_reason = "tool_calls"`.

### 5.4 Sampling: repetition penalty

The model card and `generation_config.json` recommend temperature 0.1, top_k 50, **repetition_penalty 1.05**. r4dx has
no repetition penalty (`SampleParams` has temperature/top_k/top_p/min_p/seed; the request parser does not read it). Needed
for fidelity of agent behaviour, not for the gate. Design: `SampleParams::repetition_penalty = 1.0` (default = exact no-op,
Qwen/Gemma bit-identical, verified by their identity rows), HF semantics (`logit > 0 ? logit / p : logit * p` for every distinct
token in prompt + generated text), applied BEFORE the temperature/top-k pipeline. The summary-sampler path sees only a
device top-64 + logsumexp, so the penalty must be applied to the logits before `r4dx_topk_lse_f32`:
`r4dx_apply_repetition_penalty_f32(logits, seen_ids, n_seen, p)` over a device list of distinct seen ids that the host
maintains incrementally (upload deltas only). With the penalty on, greedy also changes (HF applies it in greedy decoding).
`/v1/chat/completions` reads `repetition_penalty` (extension field, as vLLM does), and the dialect supplies defaults only when
the request omits them (open question 6).

## 6. Reference tooling, goldens, KL gate

### 6.1 Environment (specify, do not create)

`D:\venvs\r4dx-gemma-ref` already has transformers 5.18.0, torch 2.9.1+rocmsdk, tokenizers 0.23.2, safetensors 0.8.0,
accelerate 1.15.0, numpy 2.5.3, and its `transformers\models\lfm2_moe\` package (config, modeling, modular) exists; the
checkpoint was written by 5.12.1 and the model card requires `transformers>=5.0.0`. So **no new venv**: L0-2 only runs
`python -c "from transformers import Lfm2MoeForCausalLM, Lfm2MoeConfig"` and the tiny random-config forward in that venv.
Python312 (transformers 5.5.0, `models\lfm2_moe` also present) is the fallback. Oracle runs that need torch >= 2.13
`cholesky_ex` use `E:\models\r4dx\huihui\venv-rocm10` as for Gemma (docs/gemma4-plan.md 12).

### 6.2 Reference package `tools/reference/lfm2/`

Mirror of `tools/reference/gemma/`: `arch.py` (names, per-layer type), `common_lfm2.py` (`DEFAULT_MODEL_DIR` =
`E:\models\LiquidAI\LFM2.5-24B-A2B`, `R4DX_MODEL_DIR` override, `R4DX_REF_ALLOWED_DEVICES`), `ref.py` (`Lfm2Reference`: the HF
`Lfm2MoeDecoderLayer` loop with explicit layer hooks, streaming one layer at a time, plus an fp32 twin), `layer_golden_lfm2.py`,
`full_logits_lfm2_cpu.py` (`--variant truth|bf16sdpa`, in kl_report.py's shared dump format), `greedy_smoke.py`,
`dump_header.py`, `gen_samples.py` (after L1), `router_golden.py`, `test_*_tiny.py`. `arch_table.py` gains an `lfm2_moe` `ArchSpec`
(section 7.1). Shared `kl_report.py`, `make_tokens_json.py`, `hessian_capture.py`, `trellis_quant.py` are parameterized, not forked.

**fp32 truth does not fit in RAM as a resident model** (24 B x 4 B = 96 GB vs 64 GB host RAM, and the "ONE big host-memory
job" rule). So the truth, like Gemma's, is **layer-streamed on CPU**: read one layer's bf16 tensors (about 1.2 GB), upcast to
fp32 (about 2.4 GB), run ALL segments through it in lockstep, free it, checkpoint the hidden states per layer for
resumption (`_ckpt`). Peak host memory about 6-8 GB plus the hidden states of the corpus (10 segments x about 1k tokens x 2048 x
4 B, tens of MB). Truth = bf16 weights upcast, fp32 activations, eager attention, **fp32 rope tables, fp32 router/sigmoid/top-k,
fp32 expert math**, fp32 final norm and lm_head. `bf16sdpa` = stock HF numerics (bf16 weights/activations, sdpa, bf16 router,
bf16 rope tables) = the NOISE TERM. Both are CPU only (`HIP_VISIBLE_DEVICES=-1`, asserted). Expected cost: dominated by the 48 GB
weight read per variant plus fp32 expert GEMMs for about 10k tokens (about 50 TFLOP in total), minutes to tens of minutes
(UNVERIFIED; measure at L0-8). `kl_report.py` gains `--gate lfm2-fp32` (the `gemma-fp32` rule with this plan's epsilon).

### 6.3 Goldens (rung 3, per layer)

Fixed-seed random inputs `[T = 64, 2048]` with perturbed norm weights (Gemma practice), prefill then 4 decode steps, for:
`layer_000_conv_dense` (ShortConv + dense SwiGLU), `layer_002_attn_moe` (attention + router + experts), `layer_003_conv_moe`,
plus unit goldens: `shortconv_state` (chunk boundary at 256, T = 300), `router` (ids + weights incl. a forced near-tie),
`qk_norm_rope` (head_dim 64, theta 1e7, positions up to 131071), `final_norm`. Intermediates recorded: in_proj output split into
B/C/x, `h = B*x`, conv output, `y`, out_proj; q/k raw/normed/roped, attention out; router logits, sigmoid, selected ids, weights;
per-expert gate/up/act/down outputs for the selected experts; the combined FFN output. All CPU-producible (a layer is small),
so goldens do not need a GPU; the r4dx side is a GPU test.

### 6.4 KL gate (proposals; exact numbers to be fixed after L0-8 measures the noise term)

Corpus: new `tools/reference/kl_corpus/chat_lfm2.json` (+ `build_chat_lfm2.py`) built from the SAME 10 conversations as
`chat_gemma.json` (prompts and answers imported from `build_chat_gemma.py`), rendered with the LFM template and the tokenizer of
the model dir; scoring on assistant-turn tokens only via `score_mask`. Because the model always thinks, each scored span is
`{thought}</think>{answer}<|im_end|>` after the generation prompt `...<|im_start|>assistant\n<think>` (the prompt part is
asserted equal to `apply_chat_template(add_generation_prompt=True)`); thoughts are short original author-written ones (the 2
existing thinking examples keep theirs); 2 more segments with a Pythonic tool call and a following tool result (group `tool`).
`tokens_lfm2.json` (raw segments, BOS explicit per the post-processor finding, truncated to 512 for reporting, no pass/fail) and
`tokens_lfm2_long.json` (about 6k tokens: many 256-row chunk boundaries for conv state and attention). Open question 5 covers
model-generated transcripts as an extra set.

Gate per group g in {english, thai, code, tool, chat-ALL} (all must pass; thai kept as a multilingual regression canary):
- `KL(fp32||r4dx) <= 1.5 x KL(fp32||HF-bf16-sdpa) + 0.005` (mean over scored rows; the Gemma epsilon; the Qwen3.8 run used 0.01;
  LFM's router near-ties may raise the HF-bf16 term, which loosens the rule automatically -- do not loosen epsilon above 0.01
  without asking), median and p99 and the worst rows reported;
- `top1(fp32, r4dx) >= top1(fp32, HF-bf16-sdpa) - 1.0 pt`.
- Also reported: router id agreement vs truth per layer, and KL split by whether a row's token had any expert flip vs the
  truth (diagnostic, no pass/fail).
- **Trellis (L3):** per group `KL(fp32||trellis) - KL(fp32||r4dx-bf16) <= 0.01` (Gemma/Qwen precedent), plus the absolute rule
  above reported, plus a rare-expert report (KL on rows whose selected experts include one of the 10 % least-used per layer in
  the calibration profile).
- TP identity: TP=2 is the only bf16 mode, so there is no bf16 TP=1 comparison; trellis TP=1 vs trellis TP=2 (optional, both fit)
  mean KL <= 3 x noise like Gemma. `tool_teacher_forced_logprobs.exe` needs only the arch dispatch.
- Greedy smoke: 64 tokens on 3 prompts plus 1 tool-call prompt, HF greedy vs r4dx, common prefix >= 16 tokens or divergence at a
  recorded near-tie.

## 7. Trellis pipeline

Status 2026-10-16 (L3): **done**, with a changed outcome: the 4.5-bit mix fails the gate and the shipped format is the K5 hybrid; see the Outcome paragraph under L3 in section 9 and [lfm2.md](lfm2.md).

Reuse list: `trellis_quant.py`, `hessian_capture.py`, `gemma_mix.py`/`sens_*.py` patterns, `rotation_oracle.py` (unused: unrotated),
`trellis_import.hpp`, `tools/gemma/trellis_pipeline.ps1` (copied to `tools/lfm2/trellis_pipeline.ps1`, same stage names:
`hessian | oracle-k4 | oracle-k5 | mix | convert | gate`, a stage without `-Run` only prints its command and refuses GPU work).

### 7.1 Arch entry and decisions

- `arch_table.py` `ArchSpec` `lfm2_moe`: taps `in` = `operator_norm` output feeding `q,k,v` or ShortConv `in_proj`; `out` =
  input of `out_proj` / `self_attn.out_proj`; `mlp_in` = `ffn_norm` output feeding dense `w1,w3` OR (routed subset) each
  expert's `gate_up`; `mlp_mid` = input of `w2` OR each expert's `down` (routed subset); `lm_head` = post-`embedding_norm` hidden.
  Linears: `attn.{q,k,v,o}`, `conv.{in_proj,out_proj}`, `mlp.{gate_up,down}`, `moe.experts.{e}.{gate_up,down}`;
  `conv.weight`, router, norms, expert_bias stay bf16/fp32 passthrough. Quantizer core unchanged (every K and N is a multiple of 128).
- **Unrotated** (decision 2): the rotation fold needs `diag(w_norm)` merged into the following linears, and Gemma failed it on
  large pre-norm scales; LFM's `operator_norm`/`ffn_norm` scale magnitudes are UNVERIFIED (L0-5 records min/max|w|), and with no
  post-norms the fold would be cheaper than Gemma's, but there is no evidence it is needed. Unrotated is the default and only path
  built; revisit only if the gate misses and the norm-weight survey shows benign scales.
- lm_head tied: bf16 table kept for the embedding gather; lm_head w4a16 g32 with LDLQ against the captured `lm_head.hess`
  (Gemma recipe), or bf16 if the increment prefers it (0.5 GB either way fits).
- Mix: unit = one QUANT GROUP per layer role (mixer in, mixer out, ffn-in = `gate_up` of ALL experts of that layer, ffn-down = all
  experts), promoted K4 -> K5 whole (single KB per group keeps the per-layer expert table uniform for the L4 kernels); attention,
  ShortConv and dense linears (4 % of the weights) go straight to K5 (cost about +0.06 GB); the experts' average is then about 4.48 to
  hit 4.5 bpw overall. Ranking: first `prior x proxy gap` like `gemma_mix.py` (layer priors: layers 2 and 3, last 2 layers, the 10 attention
  layers), data-driven proxies from the K4/K5 oracle runs; an empirical per-layer sweep (Gemma's `sens_sweep.py`) only if the first mix
  misses the gate, because 40 container builds are expensive (open question 8). Expert-granular KB is a possible later refinement.

### 7.2 Calibration data

Generator: **the r4dx bf16 TP=2 engine itself** once L1/L2 pass (HF generation of a 24 B model is impractical on 2 x 32 GB without
the offload tricks the Gemma plan used for a 12 B one). `gen_samples.py` drives `r4dx-server` over the 390 prompts of
`tools/quant2/corpus_v2_prompts.json` plus agentic/tool-call prompts and multilingual prompts, natural thinking, thinking+answer kept,
token-id JSONL (the existing `tokens` gen format; no re-templating). Because rare experts need more rows: **target >= 1.5 M tokens**
(about 90 k tokens reach an average expert per layer; a 10 %-frequency expert still sees about 9 k, versus the rows gate below), a 60/40
mix of model-generated text and local raw text (code, prose, multilingual, JSON/tool schemas), all disjoint from the KL corpus
(word 6-gram check as `build_chat_gemma.py::check_disjoint`). Sampling for generation: temperature and top_p are open question 9.

### 7.3 Hessians for 64 x 38 experts: the rare-expert risk

A Hessian per expert linear is `X_e^T X_e` over only the tokens routed to expert e (gate/up share one `K = 2048` input Hessian, down
has `K = 1536`). Rows gate (Gemma rule: rows >= 4 x K): gate_up needs >= 8,192 routed rows, down >= 6,144, **per expert per layer**.
With top-4 of 64 an average expert gets N/16 rows, so N = 1.5 M gives about 94 k average and about 9 k for a 10 %-frequency expert;
but frequency is skewed and domain-specific experts (code, a language, tool syntax) may be 50x rarer. The aux-loss-free `expert_bias`
balancing helps but does not remove skew. Mitigations (all in the capture script, per-expert row counts recorded in `hessian.json`):
(1) report rows per (layer, expert) before any oracle run and fail the stage if the fraction of experts below the gate exceeds a
threshold (proposal 2 %); (2) **shrinkage** for experts below the gate: `H_e' = (n_e H_e + n_0 H_pool)/(n_e + n_0)` with `H_pool` the layer's
pooled Hessian of ALL tokens entering the experts (routing-agnostic, computed in the same pass; `n_0` = the gate's rows), which degrades
to a generic Hessian rather than a rank-deficient one; (3) targeted calibration text for the weakest experts found by (1) (a second
generation round, prompts picked to hit them); (4) rare experts promoted to K5 inside a group is NOT possible with one KB per group, so
the fallback for a hopeless expert is to keep the whole layer's group at K5. The rare-expert KL report in 6.4 is the arbiter. The Qwen3.8
precedent (memory notes) shows the 4-bit MoE experts landed near the bf16 floor while the dense hyper-connection layers were the problem;
that is encouraging but not proof for this model.

Hessian size: per expert `2048^2 x 4 B + 1536^2 x 4 B` = 26 MB, x 64 x 38 = 63 GB, plus mixers and dense about 3 GB: **about 66 GB**
fp32 full matrices (about 33 GB if only triangles are stored; `--dry-run` reports the real figure), written to
`D:\r4dx-scratch\lfm2\hessian-v1` (D: has 426 GB free; E: is for models). Capture streams layers: one layer's bf16 weights (1.2 GB) on
the GPU, the corpus activations at layer boundaries (1.5 M x 2048 x 4 B = 12 GB, host or spilled to disk in chunks), per-layer Hessians
accumulated on the device (1.7 GB for the 64 experts).

### 7.4 Oracle and container

`quantize-model --arch lfm2_moe --K 4` and `--K 5`, `--hessian-basis matched`, no rotation, then `mix --bpw 4.5`, then
`r4dx-convert --trellis-from <mix> --trellis-verify full --no-bf16 --lm-head w4a16 --w4a16-group-rule "^lm_head$=32" --ldlq "^lm_head$"
--hessian-dir <hessian-v1>` into `E:\models\r4dx\lfm2\lfm2-24b-a2b-trellis-mix45.r4dx`. Oracle cost is about 2.2 x Gemma's (24 B vs 10.9 B
quantized weights; UNVERIFIED, measure on the first 4 layers), two passes. Tuning rows for the new shapes (`gemm_tuning_table_trellis*.inc`):
expert N 3072 K 2048, N 2048 K 1536 (and TP=2 N 1536 K 2048, N 2048 K 768 only if a TP=2 trellis is ever wanted), mixers N 6144 K 2048,
N 2048 K 2048, N 512 K 2048, dense N 23552 K 2048, N 2048 K 11776; `PlanTrellisM256` must be checked per shape (a failed plan silently
falls back to 64-row slicing). `test_trellis_input` gains K = 1536 and 11776.

## 8. Budgets

| Resource | Budget |
|---|---|
| bf16 checkpoint | `E:\models\LiquidAI\LFM2.5-24B-A2B` (47.9 GB, 10 safetensors shards; downloading, user-owned: nothing here touches it) |
| r4dx outputs | `E:\models\r4dx\lfm2\`: bf16 container about 48 GB (tied head written once as bf16: +0.5 GB), trellis container about 14 GB, oracle K4 about 12 GB + K5 about 14 GB, KL dumps (about 260 MB per 1k-token segment, 5 dumps: truth, bf16sdpa, r4dx-bf16, r4dx-trellis, TP2) about 8 GB. About 100 GB total on E: (659 GB free) |
| scratch | Hessians about 66 GB on `D:\r4dx-scratch\lfm2\hessian-v1` (D: 426 GB free). Keep >= 30 GB free on every drive; stages check before writing |
| host RAM 64 GB | **ONE big host-memory Python job at a time** (truth CPU run, Hessian capture, oracle, sample merge are each such a job; the 2026-10-05 OOM hung the machine). Streaming layer loaders keep peaks at 6-20 GB; the container loader mmaps/streams tensors (a 48 GB container is read through the page cache, never held privately) |
| VRAM, bf16 TP=2, per rank | experts 22.95 GB + mixers/dense 0.75 GB + embedding 0.52 GB (replicated) + head slice 0.26 GB + KV 1.25 GiB (131072) + activations/arena about 0.4 GB = about 25 GB (23.3 GiB). Device 0 also drives the desktop (about 4.6 GB): about 2 GB margin (risk R1) |
| VRAM, trellis TP=1 | about 14 GB weights + 2.7 GB KV + conv state + scratch = about 17.5 GB on a 32 GB card: room for the bf16 embedding, bigger prefill chunks, and a later drafter |

## 9. Task list

Legend: **[CPU]** authoring and tests need no GPU. **[GPU]** is a command handed to the user, run from the main session after
approval (every GPU workload, including every `tests/kernels/*` test, which calls `hipSetDevice`). **[CPU-heavy]** is a long
CPU/RAM job (user/main session starts it; one at a time). Build dir `build\win-hip`. Worktrees: one per lane (below).

### Parallel lanes (each in its own worktree off `lfm2`; merge back to `lfm2` before the dependent task)

| Lane | Worktree / branch | Tasks | Blocks |
|---|---|---|---|
| A refs | `r4dx-lfm2-ref` / `lfm2-ref` | L0-2..L0-9, L1-14 | gate scripts, goldens, calibration |
| B tokenizer + server | `r4dx-lfm2-tok` / `lfm2-tok` | L0-4, L2-1..L2-8 (all CPU) | L2 smoke |
| C config + converter + container | `r4dx-lfm2-conv` / `lfm2-conv` | L1-1..L1-5 | L1-8 (model) |
| D kernels | `r4dx-lfm2-kern` / `lfm2-kern` | L1-6, L1-7 (code CPU, tests GPU) | L1-8 |
| E model + TP | `r4dx-lfm2-model` / `lfm2-model` (starts when C+D merge) | L1-8..L1-13 | gates |
| F quant tools | `r4dx-lfm2-quant` / `lfm2-quant` | L3-1, L3-2, L3-3 (CPU authoring) | L3 runs |

A, B, C, D, F can all start on day 1 (F's authoring needs only the arch table and a tiny checkpoint). Only one lane at a time may run a
[CPU-heavy] job. CMake edits (arch.h, CMakeLists) are the merge hot spots: each lane touches only its own `CMakeLists` blocks.

### L0: setup and reference data

- **L0-1 Branch and plan [CPU].** `lfm2` from main, this document. Done.
- **L0-2 Environment check [CPU].** In `D:\venvs\r4dx-gemma-ref`: import `Lfm2MoeForCausalLM`, run a tiny random-config forward vs a manual
  layer loop. Done: imports, no package installed. Lane A.
- **L0-3 Header dump and name audit [CPU].** After the user's download finishes: `tools/reference/lfm2/dump_header.py` reads the safetensors
  headers/index only; records tensor names, dtypes (the F32 tensors), shapes, shard sizes; settles fused vs per-expert expert keys, the
  presence of `lm_head`, `conv.conv.weight` shape. Output `tensor_names.json`. Deps: download. Lane A.
- **L0-4 Tokenizer inspection [CPU].** Read `tokenizer.json` head (normalizer, pre_tokenizer, decoder, post_processor, added tokens with ids
  and `special` flags) and `tokenizer_config.json`; write `docs/lfm2-semantics.md` section "tokenizer". Settles 5.1 items (BOS behaviour, the
  `<think>` token id and whether it is special, regex). Deps: download. Lane B.
- **L0-5 Semantics doc [CPU].** `docs/lfm2-semantics.md`: each fact of section 2 with its source line, plus norm-weight survey (min/max |w|
  per layer, decides 7.1), router sigmoid statistics (margin between 4th and 5th score; bias magnitudes). Deps: L0-3. Lane A.
- **L0-6 Template render check [CPU].** Render `chat_template.jinja` in minja for roles, tools, tool calls (all arg types), history thinking
  strip, `preserve_thinking`; list unsupported constructs. Deps: L0-4. Lane B.
- **L0-7 KL corpus [CPU].** `build_chat_lfm2.py` -> `chat_lfm2.json`, `tokens_lfm2.json`, `tokens_lfm2_long.json`; ids equal `AutoTokenizer`.
  Deps: L0-4. Lane A.
- **L0-8 fp32 truth and bf16sdpa dumps [CPU-heavy].** `full_logits_lfm2_cpu.py --variant truth` then `--variant bf16sdpa` (streaming, section 6.2),
  into `E:\models\r4dx\lfm2\kl\fp32\{truth,bf16sdpa}`; produces `noise_term.json` (the numbers that fix the gate in 6.4). Deps: L0-3, L0-7. Lane A.
- **L0-9 Per-layer goldens [CPU].** `layer_golden_lfm2.py` + `router_golden.py` (section 6.3), manifest with tolerances. Deps: L0-5. Lane A.

### L1: bf16 TP=2 text path + gate

**Foundations (CPU)**
- **L1-1 `Lfm2Config`, `Arch::kLfm2Moe`, dispatch hook.** Files: `src/model/{lfm2_config.h,arch.h,arch.cpp}`, `tests/model/test_lfm2_config.cpp`
  (real config parses; layer pattern asserted; shard rules incl. K % 256 legality). `LoadLfm2TextModel` is a throwing placeholder. Deps: L0-5. Lane C.
- **L1-2 Converter LFM branch.** `src/convert/lfm2_layout.{hpp,cpp}`, `RunConvert` dispatch, coverage audit, `tools/convert_ref/make_tiny_lfm2.py`,
  `tests/convert/test_lfm2_layout.cpp` (fused and per-expert key styles, fp32 `expert_bias`, squeezed conv weight). Deps: L0-3. Lane C.
- **L1-3 `Lfm2Container` loader.** `src/model/lfm2_container.{h,cpp}` (reuse `container_load_util`, `shard_loader.h`): loads the tiny fixture, builds
  the per-layer expert pointer tables. Deps: L1-1, L1-2. Lane C.
- **L1-4 TP shard rules.** `tp_shard.cpp` `RuleFor` for `lfm2_moe` (three-segment `conv.in_proj`, per-expert gate_up/down, replicated router/bias/norms,
  vocab-slice head), `tests/model/test_lfm2_tp_shard.cpp` (real 40-layer config x 2 ranks vs `Lfm2Config::Shard`; algebra check that column-parallel
  conv channels concatenate and row-parallel partial sums add up). Deps: L1-1. Lane C.
- **L1-5 Bf16 test containers [CPU, converter run].** Tiny fixture container, and `lfm2-l6.r4dx` (layers 0-5 of the real checkpoint, about 7 GB) once the download exists.
  Deps: L1-2, L0-3. Lane C.

**Kernels (code CPU, tests GPU)**
- **L1-6 Kernels + CPU references.** `src/kernels/{include/r4dx/kernels/lfm2_kernels.h,src/lfm2_kernels.hip}` and header-only `lfm2_ref.hpp`:
  shortconv (+ state update), router, expert counting sort/gather, MoE combine, optional head_dim-64 variants if the existing rope/rmsnorm kernels
  reject 64. Deps: L0-5. Lane D.
- **L1-7 Kernel tests [GPU].** `ctest -R "lfm2_"` (shortconv sizes and chunk equivalence, router ids/weights vs the CPU reference incl. ties, sort/combine
  determinism, rope64, qk-rmsnorm64). Command: `$env:HIP_VISIBLE_DEVICES='1'; ctest --test-dir build\win-hip -R lfm2_ --output-on-failure`. Deps: L1-6. Lane D.

**Model**
- **L1-8 `Lfm2AttnLayer`, `Lfm2MoeFfn`, `Lfm2ConvLayer`, `Lfm2Model`, `Lfm2LocalTextModel`.** Single-device engine with the staged MoE (host-sync decode, sorted
  prefill), conv state + checkpoint, bf16 KV, reference attention, bf16 residual; `Lfm2Model` has `GemmaModel`'s public surface (no vision/features/DFlash). Files:
  `src/model/{lfm2_model,lfm2_layers,lfm2_local_text_model}.{h,cpp}`, `src/model/attention/include/r4dx/model/attention/lfm2_attention_layer.hpp`.
  Build-only guard: the Qwen/Gemma binaries are unchanged (the existing ISA-diff check). Deps: L1-3, L1-6. Lane E.
- **L1-9 Rung-3 goldens in C++ [GPU].** `tests/model/test_lfm2_layers.cpp` against the L0-9 goldens (layers 0, 2, 3; conv state carry; router). Command:
  `ctest -R lfm2_layer`. Single device, `lfm2-l6` container. Deps: L1-8, L1-5, L0-9. Lane E.
- **L1-10 `Lfm2TpModel` + warm-up.** Copy-adapt `gemma_tp_model.{h,cpp}` (rank threads, `TpGroup`, watchdog, recovery); `Lfm2Model` rank options
  (`tp_world/rank/comm`, all-reduce of the two bf16 sublayer outputs, vocab-split head); `TpWarmup` covers a 256-row chunk, a 64-row prefill, greedy/sampled/full-logits steps,
  and a MoE prefill that touches all experts. Deps: L1-4, L1-8. Lane E.
- **L1-11 bf16 container [CPU-heavy].** `r4dx-convert --input E:\models\LiquidAI\LFM2.5-24B-A2B --output E:\models\r4dx\lfm2\lfm2-24b-a2b-bf16.r4dx` (about 48 GB write, converter
  streams per tensor; check free space on E: first). Deps: L1-2, download. Lane E.
- **L1-12 bf16 TP=2 KL gate [GPU].** `$env:HIP_VISIBLE_DEVICES` UNSET: `tool_teacher_forced_logprobs.exe --model <bf16>.r4dx --layout bf16 --tokens
  tools\reference\kl_corpus\chat_lfm2.json --tp 2 --tp-mode real --max-ctx 8192 --out-dir E:\models\r4dx\lfm2\kl\r4dx-bf16`, then `kl_report.py --gate lfm2-fp32 --truth-dir ...\truth --noise-dir ...\bf16sdpa
  --test-dir ...\r4dx-bf16 --tokens chat_lfm2.json`, the long segment for chunk-boundary / conv-state checks. Gate: section 6.4. Deps: L1-10, L1-11, L0-8. Lane E.
- **L1-13 Greedy smoke + determinism [GPU].** r4dx greedy output vs the L1-14 reference; a 1,000-token generation twice (determinism). `--tp-mode emulate` is not available at bf16
  (2 x 24 GB on one card). Deps: L1-12, L1-14.
- **L1-14 Greedy-smoke reference [CPU-heavy].** A cache-less streaming-layer greedy of 16-24 tokens x 4 prompts (3 plain + 1 tool-call) is feasible on CPU (each token re-reads 48 GB, so batch all
  prompts in lockstep); the 64-token variant of the Gemma gate is replaced by "common prefix >= 16 tokens or near-tie" (6.4). If that proves too slow, a GPU run on both devices with
  `accelerate` `device_map` is the alternative (needs approval). Lane A.
- **L1-15 Frozen TP=2 baseline.** `tools/tp/lfm2_tp_identity.ps1`, `r4dx-baselines\lfm2-tp2-<commit>` for later byte-identity guards. Deps: L1-12.
- **L1-16 Qwen/Gemma regression [GPU].** The existing identity rows (Qwen rows 1, 3; Gemma TP1) on the merged `lfm2` branch: the shared-code edits (arch.h/cpp, `tp_model.cpp`
  dispatch, `tp_shard.cpp`, `SampleParams`) are the only places they could change. Deps: L1-10.
- **L1-17 Contingency: fp32 residual.** Only if L1-12 misses: `R4DX_LFM2_RESID=fp32` with `r4dx_rmsnorm_plain_f32in_bf16` and a fused add+norm f32 kernel (the Gemma `*_f32res` kernels are
  the template). Decision rule in 3.2.

### L2: server, tools, end-to-end

- **L2-1 Tokenizer backend changes [CPU].** Per L0-4 findings (regex, normalizer, keep-special, bos). Files `src/tokenizer/{tokenizer.h,bpe_tokenizer.cpp}`; tests `tokenizer_golden_lfm2`
  (parametrized `golden_test.cpp`, env `R4DX_LFM2_TOKENIZER_DIR`, skip 77) with `tools/tok_ref/gen_golden_lfm2.py`, `check_hf_equiv.py`; Qwen and Gemma goldens unchanged. Lane B.
- **L2-2 minja patches [CPU].** Only what L0-6 lists; `third_party/VERSIONS.md` notes; `test_minja_lfm2.cpp` (every chat case equals the HF-rendered string and ids). Lane B.
- **L2-3 `Lfm2Dialect`, `--dialect lfm2`, arch cross-check [CPU].** `src/server/dialect.{h,cpp}`, `tests/server/test_dialect.cpp` rows; `CheckDialectAgainstArch` knows `lfm2`. Lane B.
- **L2-4 Pythonic tool-call parser + stream gate [CPU].** `src/server/lfm2_tool_call_parser.cpp`, `tool_stream_gate` opener 19, tests `test_lfm2_tool_call_parser` at every byte split. Lane B.
- **L2-5 Engine/sink wiring [CPU].** `OnStart` reasoning-in-prompt with `<think>`, `ckpt_back` from the dialect suffix, EOS list, tool arguments as mapping, 400 on a template error, default
  tokenizer dir = the container's model dir, CLI. `test_engine_recovery` with an LFM fake model. Deps: L2-1..L2-4, L1-1. Lane B.
- **L2-6 Repetition penalty [CPU + GPU test].** `SampleParams::repetition_penalty`, request parsing, `r4dx_apply_repetition_penalty_f32`, host seen-set; the default-off path is bit-identical for
  Qwen/Gemma (identity rows). GPU test `ctest -R rep_penalty`. Lane B (kernel by lane D).
- **L2-7 Server smoke [GPU].** `powershell tools\server\smoke.ps1 -Model <bf16>.r4dx -Tp 2 -Dialect lfm2 -Port 8091` (LFM variant: reasoning split streaming and non-streaming, a Pythonic tool call round trip with
  a tool result turn, prefix-cache hit on a second turn via the checkpoint path, client abort, TP fault recovery as the existing smoke does). Deps: L2-5, L1-12.
- **L2-8 Docs.** `docs/status.md` / README row, `docs/server.md` LFM dialect section. Done when L2-7 passes.

### L3: trellis on one card

Status 2026-10-16: L3-1..L3-9 **done**.

**Outcome.** The 4.5-bit mix (4.5085 bpw, 13.4 GB) failed the KL gate (ALL +.0222 over bf16; english +.0245, thai +.0291), as did all-K5 (5.02 bpw, 14.8 GB, ALL +.0156) and K5 experts with bf16 non-expert (15.8 GB, ALL +.0100, english/thai fail). The shipped format is a hybrid: K5 experts, bf16 non-expert linears and bf16 experts of layers 2-5, 18.89 GB, which passes every group (ALL +.0055; top-1 96.13 %). K6 was not usable (only KB 4 and 5 are instantiated). Quantization damage is broad and concentrated in the early layers' experts, since HF bf16 itself keeps the same top-4 expert set for only 87 % of tokens. Numbers, layer sweep and reproduction commands: [lfm2.md](lfm2.md).

- **L3-1 Arch table + Hessian capture port [CPU].** `arch_table.py` `lfm2_moe`, `hessian_capture.py --arch lfm2_moe` (per-expert routed Hessians + pooled Hessian + row counts, `--dry-run` lists
  tap groups, files and the size in GiB), `gen_samples.py`. Tests on the tiny checkpoint (`reference_arch_lfm2`, tiny pipeline). Lane F.
- **L3-2 Oracle port [CPU].** `trellis_quant.py --arch lfm2_moe` (per-expert linears, `quantize-model --dry-run`), `trellis_import.hpp` expert-suffix keys, `convert_lfm2_trellis` C++ test (CPU, plain). Lane F.
- **L3-3 Mix tool [CPU].** `lfm2_mix.py` (section 7.1 groups, priors, K5-floor for non-expert linears, budget test on the real shapes `reference_lfm2_mix`). Lane F.
- **L3-4 Calibration generation [GPU].** `gen_samples.py` through `r4dx-server --tp 2` bf16, both GPUs, one prompt shard; resumable, merge on CPU with `--require-complete`. Output `E:\models\r4dx\lfm2\corpus\samples.jsonl`
  (>= 1.5 M tokens). Deps: L2-7, L3-1.
- **L3-5 Hessian capture [GPU].** `$env:HIP_VISIBLE_DEVICES='1'; python tools\reference\hessian_capture.py --arch lfm2_moe --model-dir E:\models\LiquidAI\LFM2.5-24B-A2B --out-dir D:\r4dx-scratch\lfm2\hessian-v1 --gen-file
  ...\samples.jsonl --seq-len 4096`. Check free space and run the row-count report first (7.3). Deps: L3-4. One big host job.
- **L3-6 Oracle K4 and K5 [GPU].** `quantize-model --arch lfm2_moe --K 4|5 --hessian-basis matched` (the torch >= 2.13 venv), after a 4-layer timing probe. Deps: L3-5, L3-2. One big host job at a time.
- **L3-7 Mix + convert [CPU].** `trellis_quant.py mix --bpw 4.5 --ranking auto`, `r4dx-convert ... --trellis-from` (section 7.4), `decode_bytes.py`. Deps: L3-6, L3-3.
- **L3-8 Trellis loader + TP=1 [GPU].** `Lfm2Container` accepts the trellis layout (per-expert `QuantLinear` trellis); trellis input tests for K 1536 / 11776; tuning rows. Command `ctest -R trellis`, then the
  gate: `$env:HIP_VISIBLE_DEVICES='1'; tool_teacher_forced_logprobs.exe --model <trellis>.r4dx --layout trellis --tokens chat_lfm2.json ...` and `kl_report.py --gate lfm2-fp32 --base-dir <r4dx-bf16 dump>`
  (increment <= 0.01 per group; rare-expert report). Server smoke at TP=1 (`-Tp 1`). Deps: L3-7, L1-12.
- **L3-9 Mix repair loop.** If the increment misses: promote more groups to K5 / the empirical sweep / extra calibration for the weakest experts (7.3), re-gate. Deps: L3-8.

### L4: performance and hardening

- **L4-1 libr4d `h64_gqa4` attention.** **Done, 2026-10-18.** New unit `third_party/libr4d/r4d_attn_paged_h64_gqa4.hip` (`R4D_VERSION` 0.7.0; registry rows, the `h64_gqa4` geometry, `r4dx::core::r4d` wrappers): split-KV decode (q_len <= 16, 32-key tiles, split law aiming at 1024 workgroups, max 256 splits), prefill (32 query rows per workgroup, 64-key tiles, gqa6 option set minus the wide K fetch) and split-KV prefill (`Lfm2PrefillSplits`: from 2048 tokens of context, units x splits about 128, >= 8 tiles per segment, <= 32). The Qwen attention ISA guard (`check_qwen_attn_isa`) still passes. `Lfm2AttnLayer::Forward` takes the device `seqused_k` (`Lfm2Model::seqused_dev_`). Tests: `test_attn_h64_gqa4` (decode / prefill / split-KV vs the reference kernel on the same bf16 cache, rel L2 1e-3..2.3e-3, deterministic, refusals), the LFM2 suites incl. the real-weights `test_lfm2_layers` (layer 2 mixer unchanged at 6.9e-3). Benchmark `tool_attn_h64_bench` (per layer, device 1, reference -> h64_gqa4): decode q_len 1 at 2k / 8k / 32k / 128k: 479 -> 15 us, 2.5 ms -> 31 us, 9.0 ms -> 61 us, 34.9 ms -> 454 us; 64-row prefill chunk: 4.2 ms -> 40 us, 20 ms -> 81 us, 75 ms -> 245 us, 307 ms -> 841 us. End to end `w8a16` TP=1: decode 176 tok/s at short context, a 7239-token prompt prefills at 3847 tok/s and decodes at 164 tok/s. **Follow-ups (same day):** (1) KL gate on the real model, `w8a16` TP=1 with the new attention: **PASS**, ALL +.00333 over r4dx bf16 (code +.0035, english +.0035, thai +.0040, tool +.0002; top-1 95.80 %), the same as before (+.0033); the 12 chat segments are < 600 tokens, so split-KV prefill is not exercised by the gate. (2) `smoke.ps1 -Tp 2 -Dialect lfm2` on the `w8a16` container: all checks pass, TP=2 decode about 249 tok/s, no TDR. (3) The HIP 719 fault the bench hit with the reference kernel at power-of-two contexts was a real bug in `attn_ref.hip`: for head_dim < 256 the PV loop read `vb + d0` for every thread (d0 up to 255) past the cache end, which faults when the allocation ends on a page boundary; fixed with `if (d0 < hd)` (the result was discarded anyway; no numeric change). (4) Decode split-count sweep (2k..128k, splits 4..256): the law is within about 15 % of the best fixed count at every context, so it is unchanged; the 8192 slowdown seen earlier was run-to-run variation of the same configuration (28-68 us; the 33 MB cache sits in and out of the GPU cache), not the law. Not done: the 128k needle run and a long-context KL (L4-7).
- **L4-2 Grouped MoE decode (bf16 + w8a16) [GPU].** **Done, 2026-10-18.** `r4dx_lfm2_gemm_bf16_moe_decode` (`lfm2_kernels.hip`) and `r4dx_gemm_w8a16_moe_decode` (`w8a16_gemm.hip`) plus the 32-wave dual-expert `r4dx_lfm2_router_f32`: zero host syncs per token, 861 -> 519 launches/token, byte-for-bit identical to the staged `kPerSlot` path (`test_lfm2_moe`, `test_w8a16_gemm`, `test_lfm2_layers`, `test_lfm2_tiny_w8a16`, `test_lfm2_tiny_e2e`). Decode speed: `w8a16` TP=1 **88.9 -> 171.2 tok/s** (5.84 ms/tok; MoE `gate_up` + `silu_mul` + `down` ~136 us/layer profiled, ~85 us/layer unprofiled), `w8a16` TP=2 **99.1 -> 208.9 tok/s** (4.79 ms/tok), trellis hybrid TP=1 **61 -> 72.7 tok/s**.
- **L4-3 Trellis MoE decode.** **Done, 2026-10-18.** Profiled on `lfm2-trellis-hybrid.r4dx` (layers 2-5 bf16 experts on `kGroupedDecode`, layers 6-39 K5 trellis experts on `kPerSlot`): the 34 remaining `moe.readback` syncs are 7.9 % of `gpu_sum` (~7-11 % of the 13.76 ms step), below section 3.5's 15 % threshold, and `w8a16` TP=1 (`171.2 tok/s`, ALL `+.0033`) is 2.35x faster and lower-KL than trellis (`72.7 tok/s`, ALL `+.0055`). Decision: keep the staged path for trellis experts; no trellis kernel change in `third_party/libr4d` (the L4-1 attention unit is a separate addition).
- **L4-4 Prefill grouped GEMM and chunk size [GPU].** Expert-offset grouped GEMM; chunk 256 vs 512 vs 1024 TTFT for 2k/8k prompts. Deps: L1-12.
- **L4-5 Fusions.** Fused q/k norm + rope + KV write, fused conv gate in_proj epilogue if profiling says so, device repetition penalty. Deps: L4-1.
- **L4-6 Tuning rows.** `tune_gemm.py` + `tool_trellis_gemm_bench` for the shapes of 7.4 and the TP=2 shapes. [GPU] Deps: L3-8.
- **L4-7 Long context + soak.** 128k needle, 60-minute soak at TP=1 trellis, `Lfm2TpModel` fault injection. [GPU] Deps: L4-1. **Partly done, 2026-10-18:** needle on `w8a16` TP=1 passes at 32,113 tokens (prefill 3736 tok/s, decode 149 tok/s) and 101,783 tokens (prefill 3344 tok/s, decode 110 tok/s, 28.5 GiB); not done: a long-context KL, the soak, the fault injection.
- **Proposed perf targets (from section 2.2 sizes and Gemma's measured effective rate of about 300 GB/s on one card, UNVERIFIED):** trellis TP=1 decode >= 120 tok/s (bytes per token about 1.4 GB, ceiling about 200), bf16 TP=2
  decode >= 60 tok/s (about 1.2 GB per rank), 8k TTFT measured and reported, not gated.

## 10. Risks

- **R1 VRAM on device 0 at bf16 TP=2** (about 25 GB of 27 GB usable, section 8). Mitigations: embedding replicated only on rank 1 (gather via host for rank 0) or an uneven expert split; bf16 is a baseline, not the shipping format.
- **R2 Rare-expert Hessians** (7.3) and the **expert-granularity** of the mix. Gate: rare-expert KL report.
- **R3 Router near-ties.** HF bf16 selects experts from bf16 sigmoids; r4dx selects in fp32. Flips are legitimate noise relative to the truth, but trellis perturbs the router INPUT and flips more; watch the id-agreement report.
- **R4 Tokenizer divergence** (digit grouping, `\p{M}`, NFC) is silent mis-tokenization; mitigated by the golden + fuzz equivalence gate (L2-1) before any KL number is trusted.
- **R5 Reference attention speed** at long context: resolved by L4-1 (the h64_gqa4 kernels; the reference kernel remains only as fallback and oracle).
- **R6 Host-sync MoE** costs 38 syncs per decode token until L4-2/L4-3; L1-L3 numbers will understate performance.
- **R7 Shared-code edits** (arch.h/cpp, `tp_model.cpp`, `tp_shard.cpp`, `SampleParams`, `dialect.*`) can disturb Qwen/Gemma; L1-16 is mandatory before merge.
- **R8 Host memory** (2026-10-05 OOM): one big host job at a time, free-RAM check before each.
- **R9 Template always thinks.** No thinking-off mode; tool calls and short answers pay for reasoning tokens. Out of scope; recorded.

## 11. Open questions (for the user; none block L0)

1. **Gate epsilon.** `+ 0.005` (Gemma) or `+ 0.01` (Qwen3.8)? The plan proposes 0.005, to be re-derived from the measured noise term at L0-8; do not exceed 0.01 without asking.
2. **Truth-run cost.** The layer-streamed fp32 CPU truth/bf16sdpa runs read 48 GB twice and compute about 50 TFLOP; acceptable on the CPU (hours?) or should L0-8 wait for a GPU window with the checkpoint split over both cards?
3. **`Lfm2TpModel` as a copy of `GemmaTpModel`** (recommended, zero risk to Gemma) or a template shared facade (touches Gemma TP; needs the Gemma TP identity rows re-run)?
4. **Embedding on device 0** (R1): accept the ~2 GB margin at bf16 TP=2, or pre-plan an uneven split?
5. **KL corpus thoughts:** are author-written short thoughts acceptable for the gate, with model-generated greedy transcripts (from r4dx bf16 TP=2 after L1) added as a non-gating second set?
6. **Sampling defaults:** should the server apply generation_config defaults (temp 0.1, top_k 50, repetition 1.05) when a request omits them, per dialect, or leave server defaults and only support the fields?
7. **KV dtype:** bf16 for all 10 layers (proposed); is an fp8 KV study wanted before long-context shipping (saves 1.25 GiB at 131072)?
8. **Mix search budget:** first mix by prior x proxy gap with whole-group K4/K5, empirical per-layer sweep only on a miss (proposed); or a sweep up front?
9. **Calibration sampling:** temperature/top_p for generating the calibration transcripts (greedy-like 0.1 matches deployment; 0.7 widens expert coverage). Proposed 0.7 / top_p 0.95 / top_k 50 for generation, 0.1 not used.
10. **Logit mask:** the embedding has 128000 rows but the tokenizer likely has about 125k ids; mask logits above the real vocab in the sampler, or leave as HF does?
11. **Rotation:** confirm unrotated trellis (decision 2) without a fallback study unless the first gate misses.
12. **Download tooling:** the user's HF download shows `.incomplete` shards; L0-3/L0-4 wait for it. Confirm nothing in this plan should fetch files itself.

## 12. UNVERIFIED facts (settle by the named task before dependent code)

| # | Item | Settled by |
|---|---|---|
| U1 | `tokenizer.json` content: normalizer, pre-tokenizer regex (digit grouping, `\p{M}`), decoder, post-processor (automatic BOS?), merges, added-token ids and `special` flags | L0-4 |
| U2 | `<think>` / `</think>` are single added tokens (and not `special`), `<|tool_call_start|>/<|tool_call_end|>` ids and flags | L0-4 |
| U3 | `tokenizer_config.json` details beyond bos `<|startoftext|>`, eos `<|im_end|>`, pad `<|pad|>` (read from a web summary), and that eos 124900 == `<|im_end|>` | L0-4 |
| U4 | On-disk safetensors key layout of experts (fused `experts.gate_up_proj` vs per-expert `w1/w2/w3`), which tensors are F32, `lm_head` absence, shard sizes (10 shards claimed) | L0-3 |
| U5 | minja renders the full chat template (string `.split`, slicing, `.endswith`, `tojson` key order vs HF) | L0-6 |
| U6 | `r4dx_rope_neox_bf16`, `r4dx_rmsnorm_plain_bf16`, `r4dx_kv_write_paged_*` bf16 accept head_dim 64 (documented generic, never run at 64) | L1-7 |
| U7 | Norm-weight magnitudes (decides the rotation question), router score margins | L0-5 |
| U8 | HF-bf16 vs fp32-truth noise term for this model (sets the gate; router flips may dominate) | L0-8 |
| U9 | Memory-bandwidth-based perf targets (Gemma's effective about 300 GB/s assumed) | L4 measurements |
| U10 | Raw-text BOS rule (`tok("hi")` adds BOS?) for `tokens_lfm2.json` | L0-4 |
| U11 | Oracle and Hessian runtimes (2.2 x Gemma estimate) and the exact Hessian disk size | L3-1 `--dry-run`, L3-6 probe |
| U12 | Real vocab size of the tokenizer vs 128000 embedding rows | L0-4 |

Verified in the repo while writing: `TextModel` surface and `Arch` dispatch (`text_model.h`, `arch.h`); libr4d has only `h256_gqa6` / `h256_gqa2`
attention (`r4d.h`); the r4dx-owned reference attention is generic in head_dim and window (`attn_ref.h`); `GemmaKvCache::Contiguous`, `r4dx_rope_neox_bf16`,
`r4dx_rmsnorm_plain_bf16`, `r4dx_silu_mul_bf16`, `r4dx_residual_add_bf16` exist; `SampleParams` has no repetition penalty; `PrefixState` is
single-slot fed() plus one prompt checkpoint; minja has a `{% generation %}` no-op; the transformers 5.18.0 `lfm2_moe` package is installed in `D:\venvs\r4dx-gemma-ref`.
