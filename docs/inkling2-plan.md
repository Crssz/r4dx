# Inkling-2-Mini (inkling2_loop) support for r4dx: design and task plan

Branch `inkling2` (from `main` at `42e8a54`), worktree `C:\Users\pay20\dev\r4dx-inkling2`. Written 2026-11-22.
**Design only so far**: no source file was edited, no GPU workload was run, and no weights were downloaded. Only the
small metadata files (config, tokenizer_config, chat template, index, hf_quant_config) and the two safetensors
*headers* (an HTTP range request for the first 148 KB, no tensor bytes) were read. Structure follows
`docs/lfm2-plan.md` and `docs/nanbeige-plan.md`. Section 9 lists the decisions the user still has to make.

Target: `OpenUncensored/Inkling-2-Mini-Uncensored` (Apache-2.0). It is `thinkingmachines/Inkling-2-Mini` with
directional ablation applied to `attn.wo_ud` on layers 4-23 only. Config, tensor names, shapes and dtypes are byte-identical
to the base model's, so this port also runs the base checkpoint (the converter does not care which one it is given).

## Summary

`inkling2_loop` is a **new family**: none of the existing archs fits (section 2). It is a 26.0 B-total / ~3.4 B-active
looped MoE with 24 stored layers. Layers 6-17 run 3 times per token with shared weights, so a token makes **48 layer
visits**. There is **no RoPE**: position comes from short causal convolutions on K/V plus a learned
hidden-state-conditioned relative bias (extent 1024). Other features: per-token per-head attention sinks, sigmoid-gated
output projection, L2-clipped Q/K, value RMS-norm, block attention residuals (Kimi "AttnRes" style) instead of a plain
residual stream, cross-pass message gates in the loop, 128-expert top-8 sigmoid MoE with a soft-capped SwiGLU, and a final
logit softcap of 30. Experts ship **pre-quantized** (NVFP4 in layers 0-19, MXFP8 in layers 20-23). Everything else is
bf16/f32.

Path (decided, section 0): a new side family behind `TextModel` (`Inkling2Model`, `Arch::kInkling2Loop`), text only,
TP=1 and TP=2. All linears use a new **MXFP8** layout (e4m3 + E8M0 per 32, W8A16 kernels copy-adapted from w8a16), about
27 GB. It is gated by teacher-forced KL against an fp32 CPU truth. Vision (23-layer ViT) and the DSpark/MTP drafter
(`mtp.safetensors`) are later milestones.

## 0. Agreed decisions (user, 2026-11-22; fixed)

1. **Weights: MXFP8 for all linears first** (user: "i have amd card so all mxfp8 first"). New layout `mxfp8`:
   e4m3 weights plus one E8M0 scale per 32 along K, run as W8A16. The e4m3 values are decoded to f16 in registers; there is
   no activation quantization. The MXFP8 experts of layers 20-23 are copied **bit-exact** from the checkpoint. The NVFP4
   experts of layers 0-19 are dequantized to f32 and re-encoded to MXFP8 (section 4.1; lossy, relative error at most 2^-4
   per element). The bf16 non-expert linears and the lm_head are MXFP8 by default; `--keep-bf16 <regex>` keeps any of them
   bf16 (LFM2 precedent, the first lever if the gate misses). The embedding table stays bf16. A native NVFP4 layout is NOT
   in milestone 1.
2. **Scope: text only, TP=1 and TP=2** (`Inkling2LocalTextModel` and `Inkling2TpModel`). No vision, no DSpark/MTP drafter.
3. **Downloads approved**: checkpoint (without `mtp.safetensors`) to `E:\models\OpenUncensored\Inkling-2-Mini-Uncensored`,
   reference venv `D:\venvs\r4dx-inkling-ref` (Python 3.12, torch 2.9.1 CPU, transformers at main `2f1bc5e`, which contains
   the inkling2_loop merge `046bdb8`).
4. **Context: default `--max-ctx` 131072**, explicit values up to 1048576 allowed.
5. KV: bf16 (section 3.4); fp8 is a later opt-in. Gate: section 7 (eps 0.005, LFM2/Nanbeige rule).

Memory at TP=1 (device 1): experts 24.46 B x 1.03125 B = 25.2 GB, other linears 0.65 B -> 0.67 GB, lm_head 0.48 GB,
embedding bf16 0.93 GB: **27.3 GB (25.4 GiB)**, plus 4 GiB of KV at 131072. That is tight on a 32 GB card: if it does not fit,
the CLI says so, and `--max-ctx 65536` is the documented fallback. At TP=2, each rank holds about 13.2 GiB plus 2 GiB of KV.

## 1. Sources read

| Source | What |
|---|---|
| `config.json`, `generation_config.json`, `tokenizer_config.json`, `special_tokens_map.json`, `chat_template.jinja`, `model.safetensors.index.json`, `hf_quant_config.json` | fetched to the session scratchpad (`inkling_meta/`) |
| safetensors headers of `model-00001-of-00001.safetensors` (148,408 B) and `mtp.safetensors` (5,664 B) | range request: every tensor name, dtype and shape (section 3.2) |
| first 40 KB of `tokenizer.json` | normalizer, pre-tokenizer, decoder, model type |
| transformers PR huggingface/transformers#51050 (merged 2026-11-21, merge commit `046bdb8`): `modeling_inkling2_loop.py`, `modular_inkling2_loop.py`, `configuration_inkling2_loop.py`, the `inkling2*` entries of `conversion_mapping.py`, and `Inkling2LoopDequantizeExperts` / `Inkling2LoopMergeMessageQueries` / `Inkling2UnpadVocab` in `core_model_loading.py` | **the semantics in section 3 are read from this code**, not from the model card |
| model card | claims only (26 B / 3.4 B active, 1M context, SGLang support, HF dequantizes experts to bf16 ~53 GB) |

The installed reference venv (`D:\venvs\r4dx-gemma-ref`, transformers 5.18.0) predates the merge and has no
`inkling2_loop`. The reference needs a transformers build at or after `046bdb8` (decision D4).

## 2. Mapping onto the existing archs

| Existing arch | Shared with Inkling | Why it cannot be reused |
|---|---|---|
| `lfm2_moe` | short causal conv, sigmoid router with selection bias, normalized top-k times a scale | no attention-residual blocks, RoPE attention, conv is a gated mixer not a K/V/output filter, 64 experts top-4, plain SwiGLU |
| `qwen3_5_moe` (Ornith) | 128+-expert top-8 MoE, grouped MoE decode infra | GDN + RoPE, softmax router, shared expert |
| `gemma4_unified` | sliding + full layer mix, logit softcap | RoPE, sandwich norms, dense GeGLU |
| `llama` (Nanbeige) | GQA h128 attention units, w8a16 TP path | dense, RoPE |
| `qwen4_exp` | hyper-connection-like multi-stream residual, 512-expert MoE | GDN/QSA, different residual math |

Verdict: **new `Arch::kInkling2Loop`**, side model like LFM2/Nanbeige (`Inkling2Model`, `Inkling2LocalTextModel`; TP facade
only if decision D2 asks for it). The pieces that carry over are copy-adapt, not shared classes: the LFM2 sigmoid-router and
per-expert QuantLinear layout (`lfm2_layout.cpp`, `lfm2_layers.cpp`), the w8a16 kernels and loader (`docs/w8a16.md`), the
`r4dx_gemma_attn_ref_bf16kv` reference attention as the stage-0 oracle, the LFM2/Nanbeige reference + KL tooling, and the
Qwen BPE machinery.

## 3. Model facts

### 3.1 Geometry (config.json, verified)

| Item | Value |
|---|---|
| model_type / class | `inkling2_loop` / `Inkling2LoopForConditionalGeneration`; text sub-config `inkling2_loop_text` |
| hidden / stored layers / layer visits | 2304 / 24 / 48 (`loop_config`: core 6..17, 3 passes, gate rank 64, `is_varloop` false) |
| layer types | full attention (`hybrid`) at 5, 11, 17, 23 (`(i+1) % 6 == 0`); sliding window 1024 (`hybrid_sliding`) elsewhere (matches `local_layer_ids`) |
| attention | 24 q heads, 8 kv heads (GQA 3), head_dim 128 (same for sliding: `swa_*`), scale 128^-0.5, no biases, **no RoPE** |
| relative bias | `d_rel` 16, extent 1024 for both layer kinds, `rel_l2_clip_threshold` 8.0 |
| Q/K clip | `qk_l2_clip_threshold` 22.6274 (= sqrt 512); log scaling for full layers: alpha 0.1, floor 8192 |
| short conv | kernel 4 (`sconv_kernel_size`): K/V (static in sliding layers, dynamic in full layers), post-attention and post-MLP (2304 ch) |
| attention residual | `n_layers_per_attn_res_block` 6 (block boundaries at layers 0, 6, 12, 18) |
| MoE | 128 routed experts, top-8, 0 shared, `moe_intermediate_size` 1152, router `sigmoid`, `route_scale` 8.0, `use_gate_bias` (selection bias) |
| activation | `softcap(g, 4) * sigmoid(g) * softcap(u, 25)` with `softcap(x, c) = c * tanh(x / c)` |
| norms | RMSNorm eps 1e-6, weight is plain `w` (not 1+w); several unweighted RMSNorms (section 3.3) |
| embedding / head | `embed` [201024, 2304] bf16 followed by a weighted RMSNorm (`embed_norm`); **untied** `unembed` [201024, 2304], only the first `unpadded_vocab_size` = **200058** rows are used; final logit softcap 30 |
| context | `model_max_length` 1048576 |
| ids | eos 200006 `<|content_model_end_sampling|>`, pad 199999 `<|endoftext|>`; image placeholder 200054 |
| generation_config | `do_sample` true, temperature 1.0 (no top-k/top-p given) |

### 3.2 Checkpoint tensors (from the safetensors header, verified)

One shard, 19,526,019,168 bytes (header says `format: pt`). Per layer `model.llm.layers.{L}.`:

| Tensor | dtype / shape | Layers |
|---|---|---|
| `attn.wq_du`, `attn.gate_attn` | BF16 [3072, 2304] | all |
| `attn.wk_dv`, `attn.wv_dv` | BF16 [1024, 2304] | all |
| `attn.wo_ud` | BF16 [2304, 3072] | all (the abliterated tensor) |
| `attn.wr_du` | BF16 [384, 2304] (24 heads x 16) | all |
| `attn.attn_sink_proj` | BF16 [24, 2304] | all |
| `attn.q_l2_clip.scale` / `attn.rel_l2_clip.scale` | BF16 [24, 128] / [24, 16] | all |
| `attn.rel_logits_proj.proj` | BF16 [16, 1024] | all |
| `attn.{k,v}_sconv.weight` | BF16 [1024, 1, 4] | sliding (20) |
| `attn.{k,v}_sconv.static_w` / `.weight_proj` | BF16 [4, 1024] / [32, 2304] (8 heads x 4 taps) | full (5, 11, 17, 23) |
| `attn_norm` | BF16 [2304] | sliding |
| `{q,k,v}_input_norm` | BF16 [2304] | full |
| `mlp_norm` | BF16 [2304] | all |
| `attn_sconv.weight`, `mlp_sconv.weight` | BF16 [2304, 1, 4] | all |
| `controllers.{attn,k,v,mlp}.down_weight` / `.forget_weight` / `.gain_weight` | BF16 [64, 4608] / [2304, 64] / [2304, 64] | loop layers 6-17 |
| `mlp.gate.weight` / `mlp.gate.bias` | BF16 [128, 2304] / F32 [128] | all |
| `mlp.experts.w13_weight` | **U8 [128, 2304, 1152]** (NVFP4, 2 codes per byte) in 0-19; **F8_E4M3 [128, 2304, 2304]** (MXFP8) in 20-23 | all |
| `mlp.experts.w13_weight_scale` | F8_E4M3 [128, 2304, 144] (1 per 16) in 0-19; U8 [128, 2304, 72] (E8M0, 1 per 32) in 20-23 | all |
| `mlp.experts.w2_weight` / `_scale` | U8 [128, 2304, 576] / F8_E4M3 [.., 72] in 0-19; F8_E4M3 [128, 2304, 1152] / U8 [.., 36] in 20-23 | all |
| `mlp.experts.w{13,2}_weight_amax` / `_outer_scale_denominator` | F32 [128] / F32 [1] | 0-19 only |

Non-layer: `model.llm.embed.weight`, `embed_norm.weight`, `norm.weight`, `unembed.weight` (bf16);
`attn_res.attn_queries.{0..22}` (F32 [2304], or **[3, 2304] for the full layers**: index i belongs to layer i+1);
`attn_res.mlp_queries.{0..23}` (F32 [2304]); `attn_res.output_query` (F32 [2304]);
`middle_cycle_cross_pass_queries` and `middle_cycle_cross_pass_kv_queries` (F32 [24, 2304] each: [q, mlp] and [k, v] pairs for
the 12 loop layers); `model.visual.*` (ViT, not used in milestone 1). `mtp.safetensors` (636 MB) holds the DSpark drafter
(3 layers, a Markov head, `target_layer_ids` [5, 23, 41, 48] = layer **visit** indices); not used in milestone 1.

Dequantization (from `Inkling2LoopDequantizeExperts`):
- NVFP4: byte b holds element 2j in the LOW nibble and 2j+1 in the HIGH nibble; E2M1 table
  `{0, .5, 1, 1.5, 2, 3, 4, 6, -0, -.5, -1, -1.5, -2, -3, -4, -6}`; value = `e2m1 * f32(scale_e4m3[row, j/16]) * amax[e] / denom`.
- MXFP8: value = `f32(e4m3) * 2^(scale_u8 - 127)` per 32 columns.
- `w13` rows are **interleaved** gate/up: the HF mapping applies `Interleave(dim=1)` to make `gate_up_proj` = [gate; up]
  halves. The exact row order (even = gate, odd = up, or the reverse) is **UNVERIFIED** until I0-3 reads `Interleave` and a
  tiny dequant golden pins it.

### 3.3 Forward pass (from `modeling_inkling2_loop.py`)

Notation: `rms(x)` = unweighted RMSNorm (fp32, eps 1e-6; the attention-residual norms use the default eps 1e-6 too),
`RMS_w(x)` = weighted. `sconv_s(x)` = depthwise causal conv, 4 taps, weight [C, 4] (tap 3 = current token, PyTorch
cross-correlation with left padding 3), output `(conv(x) + x) / sqrt(5)`, fp32. `sconv_d(x, z)` = dynamic version: taps
`static_w[lag, c] + (z @ weight_proj^T)[head(c), lag]` with `lag 0` = current token, output `(sum + x) / sqrt(9)`.
`clip(v, t)` = `v * min(1, t / (||v|| + 1e-6))` per head vector. `logscale(pos)` = `1 + 0.1 * ln(max((pos + 1) / 8192, 1))`.

**Attention residual (AttnRes).** The model keeps a list `B` of block states (each [T, 2304]) and a partial state `h`.
`res(q, B, h)` = softmax over the stack `S = [B..., h]` of `(q . rms(S_i)) / sqrt(2304)`, returning `sum_i p_i rms(S_i)` (fp32).
Note that it mixes the **normalized** states. A layer with several queries (full layers: 3) returns one mix per query.

**Decoder layer i** (inputs `h`, `B`):
```
x_in   = h
if i > 0:  a = res(attn_query[i], B, h)     # full layers: aq, ak, av (3 queries)
else:      a = h                            # layer 0 only
sliding:   xq = xk = xv = RMS_w(attn_norm, a)
full:      xq = RMS_w(q_input_norm, aq), xk = RMS_w(k_input_norm, ak), xv = RMS_w(v_input_norm, av)
loop pass >= 2: before those norms, a (or aq/ak/av) is mixed with the pass messages m_q/m_k/m_v by the message gate (below)
q  = clip(scale_q * (xq @ Wq^T), 22.63)              [* logscale(pos) in full layers]   per head [24, 128]
k  = clip(sconv(xk @ Wk^T), 22.63)                    sconv_s (sliding) or sconv_d(., xk) (full), then per head [8, 128]
v  = rms(sconv(xv @ Wv^T))                            per head, unweighted
r  = clip(scale_r * (xq @ Wr^T), 8.0) [* logscale]   per head [24, 16];  rel[t, head, d] = r[t, head] @ proj[:, d], d in [0, 1024)
s  = xq @ Wsink^T                                     [T, 24]: one extra logit per (query, head) with value 0
logits(t, j) = (q_t . k_j) / sqrt(128) + (0 <= t - j < 1024 ? rel[t, head, t - j] : 0); sliding layers also mask t - j >= 1024
o  = softmax([logits, s]) without the sink column, times V; o = o * sigmoid(xq @ Wgate^T);  o = o @ Wo^T
o  = sconv_s(o)                                       post_attention_convolution, 2304 ch, fp32
if i % 6 == 0:  B = B + [x_in];  h' = o              # block boundary: the incoming partial becomes a block
else:           h' = x_in + o
y  = res(mlp_query[i], B, h')                         [loop pass >= 2: y = gate_mlp(y, m_mlp)]
f  = MoE(RMS_w(mlp_norm, y));  f = sconv_s(f)         post_mlp_convolution
return h' + f
```
**MoE.** Router in fp32: `logits = x @ Wg^T` (bf16 weight upcast), `s = sigmoid(logits)`, top-8 of `s + bias`, weights =
`s[sel] / (sum + 1e-20) * 8.0`. Expert: `gu = x @ W13^T` [2304] -> gate / up halves -> `act` above -> `@ W2^T`; the weighted
sum is accumulated in fp32.

**Model loop.** `h = RMS_w(embed_norm, embed[ids])`, `B = []`. The model loop also appends `h` to `B` at every boundary
layer, the same value the layer appends internally. Then:
```
layers 0..5                     -> h_pre, B_pre (2 blocks: embed, L0..5 partial)
pass 1: layers 6..17            -> B gains 2 blocks (at L6: h_pre, at L12)
pass p = 2, 3: prev = (B_prev[3:] = [block appended at L12 of pass p-1], h_prev = pass p-1 output)
   h, B = h_pre, B_pre;  for loop layer j = 0..11: m[j] = res(msg_query[j], prev) (4 queries: q, k, v, mlp)
   run layers 6..17 with messages, using their OWN KV/conv state copies (visit index 24 + 12 (p-2) + j)
layers 18..23                   -> h
h = RMS_w(norm, res(output_query, B, h));  logits = 30 * tanh((h @ unembed[:200058]^T) / 30)
```
**Message gate** `gate(x, m)`: `g = [rms(x), rms(m)] @ down^T` (rank 64); `x' = (1 - tanh(g @ forget^T)) * x + tanh(g @ gain^T) * m`.

Caches per sequence: 48 layer visits, each with its own K/V (sliding: ring of 1024 positions; full: all positions) and 4 conv
states (k, v, post-attn, post-MLP; 3 previous inputs each, fp32). **K/V are cached after conv and clip/norm**, so decode
reads them as-is.

### 3.4 Sizes

Experts: 24 x 128 x (2304 x 2304 + 2304 x 1152) = **24.46 B (94 %)**. Attention + router + controllers about 0.65 B. Embed and
unembed 0.46 B each. Total 26.0 B. Active per token: 48 visits x 8 x 7.96 M = 3.06 B of experts + about 1.3 B of non-expert
work (per visit) + head, about 3.4 B on the card's claim (weights re-read per visit, so decode **bytes** count the looped
layers 3x).

| Layout | Experts | Rest | Total | Fits |
|---|---|---|---|---|
| bf16 (dequantized) | 48.9 GB | 2.9 GB | **51.8 GB** | TP=2 only, about 26 GB per rank (device 0 has about 27 GB free: too tight) |
| **mxfp8 everything, embed bf16 (chosen)** | 25.2 GB | 2.1 GB | **27.3 GB** | TP=1 on device 1 (tight with 4 GiB KV), TP=2 about 13.7 GB per rank |
| w8a16 g128 everything, embed bf16 | 24.8 GB | 2.1 GB | **26.9 GB** | TP=1 on device 1 |
| native NVFP4 + MXFP8 experts, w8a16 rest | 15.7 GB | 2.1 GB | **17.8 GB** | TP=1 on either card |

KV, bf16: 8 full-attention visits (layers 5, 23 once; 11, 17 three times) x 8 heads x 128 x 2 x 2 B = **32 KiB/token** (4 GiB
at 131072, 32 GiB at 1M). Sliding: 40 visits x 1024 x 4 KiB = 160 MiB. Conv states: 48 visits x (1024 + 1024 + 2304 + 2304) x 3 x 4 B
= 3.8 MB.

## 4. Converter (`src/convert/`)

- `inkling2_layout.{hpp,cpp}`: `AddInkling2TextStack` in the `llama_layout` / `lfm2_layout` style; `RunConvert` dispatches on
  `model_type == "inkling2_loop"`. `__metadata__`: `model_arch = "inkling2_loop"`, `model_config` = the verbatim config.json.
- **New source dtypes** (`tensor_codec.hpp` today rejects all but BF16/F32): an expert decoder `inkling2_expert_codec.hpp` that
  reads U8 / F8_E4M3 tensors **only through the expert path** (so no other arch starts accepting them) and produces f32 per
  expert slice [rows, cols] using the section 3.2 rules, one expert at a time (8.5 M floats = 34 MB). F8_E4M3 decode is
  table-based (256 entries, subnormals honoured; NaN codes rejected).
- Container names (proposal):

| Container | Source | Layout |
|---|---|---|
| `text.embed_tokens`, `text.embed_norm` | `embed.weight` (all 201024 rows), `embed_norm` | bf16 |
| `lm_head.{L}` | `unembed.weight[:200058]` | mxfp8 (bf16 option) |
| `text.final_norm`, `text.final_query` | `norm`, `attn_res.output_query` | bf16 / f32 |
| `text.layers.{i}.attn_query` | `attn_res.attn_queries.{i-1}` (i >= 1) | f32 [1 or 3, 2304] |
| `text.layers.{i}.mlp_query` | `attn_res.mlp_queries.{i}` | f32 |
| `text.layers.{i}.attn.{q,k,v,gate,o,rel}.{L}`, `attn.sink` | `wq_du, wk_dv, wv_dv, gate_attn, wo_ud, wr_du`, `attn_sink_proj` | mxfp8 or bf16 (sink [24, 2304] bf16) |
| `text.layers.{i}.attn.{q_scale,rel_scale,rel_proj}` | `q_l2_clip.scale`, `rel_l2_clip.scale`, `rel_logits_proj.proj` | bf16, passthrough |
| `text.layers.{i}.attn.{k,v}_conv` (sliding) / `{k,v}_conv_static` + `{k,v}_conv_proj` (full) | `{k,v}_sconv.*` | conv weight squeezed to [C, 4] (f32 on load); `weight_proj` bf16 |
| `text.layers.{i}.{attn_norm \| q_norm,k_norm,v_norm}`, `mlp_norm`, `attn_conv`, `mlp_conv` | same | bf16 |
| `text.layers.{i}.msg.{q,k,v,mlp}.{down,forget,gain}` (6-17) | `controllers.{attn,k,v,mlp}.*` | bf16 (forget/gain used in f32, as HF keeps them f32) |
| `text.msg_queries` | the two `middle_cycle_cross_pass_*` merged to [12, 4, 2304] in order q, k, v, mlp | f32 |
| `text.layers.{i}.moe.router`, `moe.bias` | `mlp.gate.weight`, `mlp.gate.bias` | bf16 / f32 |
| `text.layers.{i}.moe.experts.{e}.gate_up.{L}`, `.down.{L}` | slices of `w13` (de-interleaved to [gate; up]) / `w2` | mxfp8 (bit-exact copy in 20-23, re-encoded from NVFP4 in 0-19) |

### 4.1 MXFP8 layout (`mxfp8`, `Layout::kMxfp8`)

- Per linear [N, K] (K % 32 == 0; every Inkling K is: 2304, 3072, 1152, and 576 per rank at TP=2):
  `{name}.mxfp8.weight` U8 [N, K] (OCP e4m3fn codes: bias 7, max 448, no inf, 0x7F/0xFF = NaN, never written) and
  `{name}.mxfp8.scale` U8 [N, K/32] (E8M0: value 2^(s - 127); s = 255 = NaN, never written). The container `quant.mxfp8`
  metadata records `{"group": 32, "elem": "e4m3fn", "scale": "e8m0"}`; the loader refuses the `mxfp8` layout on a container
  without it (w8a16 precedent).
- Encoder (f32 source: bf16 linears, or NVFP4 dequantized in f32): per row and per 32-column group, `amax = max |v|`;
  `e = amax == 0 ? -127 : ceil(log2(amax / 448))`, clamped to [-127, 127]; elements `q = RNE_e4m3(v / 2^e)` with saturation at
  448. Choosing `ceil` means no element is ever clipped (the OCP `floor(log2 amax) - 8` rule can clip up to 512/448). The MXFP8
  source tensors of layers 20-23 are copied unchanged, after the same gate/up de-interleave of rows, which keeps the
  32-groups intact because they run along K.
- Kernels (`src/kernels/src/mxfp8_gemm.hip`, own object, libr4d untouched): copy-adapt of the w8a16 GEMM/GEMV (`docs/w8a16.md`).
  The int8 -> f16 step becomes an exact e4m3 -> f16 decode (every e4m3 value, subnormals included, is a normal f16). The
  E8M0 scale is applied to fp32 partial sums once per 32-K group (2 WMMA K-steps), because 2^e can leave the f16 range.
  Decode (M = 1 to 16) is a GEMV, prefill M up to 256 is the m64-tiled GEMM, and a CPU reference sits in
  `mxfp8_ref.hpp`. The bit-exact goldens include subnormals, 448, -0, and scales 2^-127 / 2^127.
- `QuantLinear` gains the `kMxfp8` branch (`ApplyLinear` dispatch), with per-expert linears like LFM2's.

### 4.2 Converter rest

- Coverage audit: every checkpoint name is consumed or allow-listed (`model.visual.*` allow-listed with a log line in milestone 1);
  any unconsumed tensor fails the run before the output file exists.
- `--layouts bf16` writes dequantized experts in bf16 (52 GB container) for a TP=2 bf16 run only if D2 asks for it.
- Tests (CPU, no weights): `tests/convert/test_inkling2_layout.cpp` on a synthetic tiny checkpoint written by
  `tools/convert_ref/make_tiny_inkling2.py` (hidden 64, 6 + loop layers, 8 experts top-2, BOTH expert encodings so NVFP4 and
  MXFP8 decode are covered, a full layer with `[3, H]` attn query), plus `test_inkling2_expert_codec.cpp` (NVFP4 nibble order,
  E2M1 table, e4m3 subnormals, E8M0, amax/denominator, de-interleave), with goldens from a numpy port of the HF dequant op.

## 5. Model (`src/model/`)

- `arch.h/.cpp`: `Arch::kInkling2Loop`, `"inkling2_loop"`; detection by `model_arch` or `model_type`/`architectures`;
  `ResolveContainerMaxCtx`: default 131072 (KV budget, as Nanbeige caps by KV), explicit up to 1048576 allowed.
- `inkling2_config.h` (header-only, strict `FromJson`, asserts the section 3.1 invariants: loop bounds multiples of 6, full
  layers at `(i+1) % 6 == 0`, `d_rel * heads == wr rows`, GQA integer). `ToModelConfig()` for `TextModel::Config()`.
- `inkling2_container.{h,cpp}`: loads the names above; per-expert `QuantLinear` (mxfp8) like LFM2; device expert pointer tables.
- `inkling2_model.{h,cpp}` (`RunChunk`, T <= 256 rows): the section 3.3 graph with a **visit table** of 48 entries (stored layer,
  pass, cache slot). The residual `h`, block states and messages are **fp32** (they are small: at most 5 blocks x [T, 2304]).
  HF rounds `h` to bf16 at each layer end; the fp32 truth does not.
- New kernels (`src/kernels/inkling2_kernels.{hip,h}`, plain C ABI, CPU reference in `inkling2_ref.hpp`):
  `attnres` (softmax over at most 6 normalized states, 1-3 queries), `sconv_static` / `sconv_dynamic` (state in/out, chunked
  prefill = decode math), `head_clip` (scale, clip, logscale, unweighted head rms), `rel_logits` ([T, 24, 16] x [16, 1024]),
  `sigmoid_gate_mul`, `msg_gate`, router `sigmoid_topk_bias` (128 -> 8, fp32, ties to the lower index), `softcap_swiglu`, logit
  softcap fused into the final f32 logits.
- Attention, stage 0: extend the r4dx reference attention (`attn_ref.h`, generic head_dim/GQA/window) with two optional inputs:
  `rel [T, H, 1024]` f32 bias and `sink [T, H]` f32 logits. It is exact but slow at long context. Stage 1 (perf): a fused
  h128/GQA3 unit with bias + sink, decode first.
- MoE stage 0: LFM2's staged path (router, one host sync per MoE visit, per-expert mxfp8 GEMV/GEMM, deterministic fp32
  combine). That is 48 syncs per token: correct but slow. Grouped decode (Ornith-style) is stage 1.
- `LoadTextModel` dispatch, `Inkling2LocalTextModel` (TP=1): Prefill / DecodeStep / Greedy / Sampled / Reset /
  Save+RestoreCheckpoint. `HasVision`, MTP and DFlash are false and throw.
- `Inkling2TpModel` (TP=2, copy-adapt of `lfm2_tp_model` / `llama_tp_model`: rank threads, `TpGroup`, watchdog). Split:

| Part | Split |
|---|---|
| attention | by head: q 24 -> 12, kv 8 -> 4 (GQA 3 kept); `wq`, `gate_attn`, `wr` (16 rows per head), `sink` column-parallel by head; `wk`, `wv` and the K/V convs (channels 1024 -> 512, `weight_proj` rows 32 -> 16, `static_w` columns) by kv head; `q_scale`/`rel_scale` rows by head; `rel_proj` replicated; `wo` row-parallel (K 1536) -> all-reduce |
| post-attn / post-MLP conv, norms, AttnRes, message gates, router | replicated on the full 2304 (after the all-reduce); identical bits on both ranks |
| MoE experts | TP-split every expert (LFM2 rule): gate and up halves 1152 -> 576 per rank, `down` K 576 (18 groups of 32) -> one all-reduce on the fp32-combined sum, rounded once |
| embedding / head | table replicated; `lm_head` vocab split over the 200058 used rows (100029 per rank), merged by `tp_vocab.h` |

  That gives two all-reduces per layer visit, 96 per token (about 0.7 ms at the measured 7.5 us each). TP=2 work is covered by
  the user's standing approval for both GPUs, but each run is still announced with its command (task rule).
- Tests (CPU): config parse, visit table, cache slot mapping, every kernel against `inkling2_ref.hpp` (GPU tests, written
  but run only with approval), and a tiny end-to-end CPU reference vs HF on the tiny checkpoint (`test_inkling2_ref_tiny.py`).

## 6. Tokenizer, chat dialect, server

- Tokenizer: byte-level BPE (`ignore_merges` true, no normalizer, byte-level decoder) with the **o200k pre-tokenizer regex**
  (`[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|...)?|...|\p{N}{1,3}|...`).
  `bpe_tokenizer.cpp` accepts only the Qwen2/Qwen3.5/Llama3 regexes today, so this needs a new hand-written splitter
  (`kO200kSplitExpected`), with goldens from `tokenizers` (`tools/tok_ref/gen_golden_inkling2.py`: English, code, Thai, CJK,
  Arabic, emoji, contractions in both cases, digit runs 1..9, whitespace and newline runs, special tokens). 60 added tokens
  199998-200057, all `special`.
- Chat template (read in full): role tokens `<|message_user|>`, `<|message_model|>`, `<|message_system|>`, `<|message_tool|>`;
  every part is `ROLE <|content_text|> text <|end_message|>`; reasoning is `<|message_model|><|content_thinking|> ... <|end_message|>`;
  a thinking-effort system message `Thinking effort level: N` (default 0.9; `reasoning_effort` none/minimal/low/medium/high/xhigh/max
  or a float in [0, 0.99]) is placed before the first non-system message, or at the end; tools are declared as one system
  message `tool_declare<|content_xml|>` + sorted compact JSON; a tool call is
  `<|message_model|>NAME<|content_invoke_tool_json|>{"name":...,"args":{...}}<|end_message|>` (string arguments raise, so the
  server must pass a parsed object); an assistant turn ends with `<|content_model_end_sampling|>` (= eos). **There is no
  generation prompt**: the template appends nothing for `add_generation_prompt`, so the model emits its own
  `<|message_model|>` header. The minja check covers `dictsort(case_sensitive=true)`, `tojson(sort_keys=true, separators=...)`,
  `namespace`, `{% set %}...{% endset %}` blocks, `{% generation %}`, and `| float`.
- `DialectKind::kInkling2` in `src/server/dialect.cpp`: the stream parser is a small state machine over the special tokens
  (header, content kind, end_message). The content kinds are thinking -> `reasoning_content`, text -> `content`, and
  invoke_tool_json -> `tool_calls` (JSON object, `args` -> `arguments`). Stop on 200006. `reasoning_effort` is mapped from the
  request (default 0.9; `enable_thinking=false` -> effort 0). Checkpoint suffix: none (the template ends at the prompt).
  `--dialect inkling2`, with auto-detection by `<|message_model|>` in the vocab plus the container arch.

## 7. Validation gate (`docs/validation.md` style)

- Truth: fp32 CPU, layer-streamed in **visit order** (48 visits; per visit, the stored layer is read and its experts are
  dequantized to f32, about 4 GB, then freed). Noise term: stock HF numerics (bf16, eager attention; this model has no
  sdpa/flash path) on CPU. Both live in `tools/reference/inkling2/` (copy of the LFM2/Nanbeige package; `ref.py` drives the HF
  `Inkling2Loop*DecoderLayer` modules directly, with the cache-offset bookkeeping of the HF loop).
- Corpus: `tools/reference/kl_corpus/chat_inkling2.json` from the same 10 conversations as the earlier ports (plus a tool turn),
  rendered with this template; scored on assistant tokens. A long set (about 6k tokens) crosses the 1024 sliding / relative
  extent and many chunk boundaries.
- Rule (LFM2 / Nanbeige precedent): per group, `KL(fp32 || r4dx) <= 1.5 x KL(fp32 || HF-bf16) + 0.005` and
  `top1 >= top1(HF-bf16) - 1 pt`, after the noise term is measured. Also reported: router id agreement per visit.
- Greedy smoke: 64 tokens on 3 prompts + 1 tool prompt, r4dx vs HF greedy, common prefix >= 16 or a recorded near-tie.
- Also reported for the first container: the KL of an all-bf16-non-expert variant (`--keep-bf16` on everything except
  experts), so the cost of MXFP8 on the non-expert linears is known.

## 8. Tasks (each GPU step is a command handed to the user)

| ID | Task | Needs |
|---|---|---|
| I0-1 | Download the checkpoint (19.5 GB model + 26 MB tokenizer.json + 3.4 MB tiktoken; skip mtp 636 MB) to `E:\models\OpenUncensored\Inkling-2-Mini-Uncensored` | approval (D4) |
| I0-2 | Reference venv with transformers >= `046bdb8` (`D:\venvs\r4dx-inkling-ref`, torch CPU wheel or the existing ROCm torch) | approval (D4) |
| I0-3 | Tiny checkpoint generator + HF tiny forward + dequant golden (pins the interleave order) | CPU |
| I1-1..4 | arch/config, expert codec, layout + audit, CPU tests | CPU |
| I2-1..6 | kernels + CPU refs, container, model graph, local text model, CLI wiring | CPU build; GPU tests on approval |
| I3-1..3 | o200k splitter + goldens, chat template check, dialect + server | CPU |
| I4-1 | fp32 truth + HF-bf16 noise dumps (CPU, hours) | CPU, one big host job |
| I4-2 | convert the mxfp8 container; teacher-forced dump on device 1; KL gate; greedy smoke; server smoke | GPU (approval) |
| I5 | native NVFP4 layout (optional, smaller) | later |
| I6 | perf: fused bias+sink attention, grouped MoE decode, fewer syncs | later |

## 9. Decisions for the user (answered 2026-11-22: section 0; the options below are kept for the record)

- **D1 Weight layout.** Answer: MXFP8 for all, first. (a) w8a16 experts dequantized from NVFP4/MXFP8, 27 GB, TP=1 on device 1, existing kernels
  (recommended first); (b) native NVFP4/MXFP8 kernels, 18 GB, lossless to the checkpoint, new GEMV/GEMM kernels first;
  (c) bf16 TP=2, 52 GB, too tight on device 0. Recommendation: (a) now, (b) as milestone 2.
- **D2 TP.** TP=1 only (recommended: a 3.4 B-active model gains little from TP=2 over a non-P2P link, and 27 GB fits device 1),
  or TP=2 too.
- **D3 Scope.** Text only (recommended); vision and the DSpark drafter later.
- **D4 Downloads.** OK to download the 19.5 GB checkpoint + tokenizer files, and to create a venv with transformers from git
  (main at or after `046bdb8`)?
- **D5 Context.** Default `--max-ctx` 131072 (bf16 KV 4 GiB) with an explicit opt-in up to 1M (32 GiB KV, does not fit next to
  the weights), or default 32768.
- **D6 KV dtype.** bf16 (recommended; Nanbeige's fp8 KV failed its gate) or fp8 e4m3 as an opt-in later.

## 10. UNVERIFIED items

1. `w13` interleave order (I0-3).
2. Whether HF's eager path (the noise term) differs from the "original" implementation that SGLang runs; the gate's truth is
   the HF semantics read above.
3. minja support for every construct in the template (I3-2).
4. Whether `embed` rows 200058..201023 are ever indexed (pad/unused ids); they are kept.
5. The conv state for padding tokens (`apply_mask_to_padding_states`): irrelevant for single-sequence r4dx.
