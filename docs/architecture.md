# r4dx architecture

## Module map

```
src/core/       device/stream/buffer plumbing: HIP device/stream lifetime, device buffer RAII,
                the paged KV cache allocator, the GDN state allocator.
src/kernels/    r4dx-owned HIP kernels -- the glue r4d does not provide (see "Own kernels" below).
src/model/      the layer graph: embedding -> N decoder layers -> final norm -> lm_head, built
                from a parsed r4dx container + config_model.
src/tokenizer/  BPE tokenizer + chat template application (minja).
src/vision/     the vision tower (docs/vision.md). TWO targets on purpose: r4dx_vision is the
                CPU-only host half (image decode, the exact Qwen2VL preprocessing pipeline, the
                mrope/pos-embed/rope/cu_seqlens index math -- no HIP, so its tests run anywhere),
                and r4dx_vision_tower is the device half (the container's 333 vision.* tensors in
                VRAM, plus Qwen3_5VisionModel's forward on top of them).
src/convert/    HF checkpoint (safetensors + config.json) -> r4dx container (docs/container-format.md).
src/server/     OpenAI-compatible chat completions API (cpp-httplib + nlohmann/json), streaming.
src/cli/        single-process text-generation entry point.
third_party/    r4d_core (the libr4d submodule, static lib) + vendored header-only deps.
```

`r4d_core` sits under `src/model` and `src/kernels`: the layer graph calls straight into the C ABI
declared in `third_party/libr4d/r4d.h` for attention, GDN, and every GEMM; `src/kernels` supplies
everything r4d does not -- there is no third layer of abstraction between a decoder layer and a
kernel launch.

## Forward pass, one text token step

```
token id
  -> embedding lookup (host table, text.embed_tokens, bf16)           [src/kernels: embedding]
  -> upload to device
  -> for each of 64 layers:
       residual = x
       x = rmsnorm(x, input_layernorm)                                [src/kernels: rmsnorm]
       if layer is GDN (48 of 64):
         x = gdn_layer(x)            -- see "GDN layer" below
       else:                          (16 of 64, full attention)
         x = attn_layer(x)           -- see "Attention layer" below
       x = residual + x                                               [src/kernels: residual add]
       residual = x
       x = rmsnorm(x, post_attention_layernorm)                       [src/kernels: rmsnorm]
       gate_up = gemm(x, mlp.gate_up.{layout})           -- r4d_gemm_*_nt_m64 family
       h = silu(gate_up[:intermediate]) * gate_up[intermediate:]      [src/kernels: silu_mul]
       x = gemm(h, mlp.down.{layout})
       x = residual + x                                               [src/kernels: residual add]
  -> x = rmsnorm(x, text.final_norm)
  -> logits = gemm(x, lm_head.{layout})
  -> sample (argmax or top-k/top-p)                                   [src/kernels: sampling]
```

At M<=64 (the whole non-chunked-prefill decode/small-batch band) every GEMM above is one of the
skinny `r4d_gemm_*_nt_m64` kernels; the model graph picks WV/SK/MB per shape once at layer-build
time (mirroring libr4d's own `build-win/check_gemm_bf16.py` tuning) rather than per call.

### Attention layer (16 of 64 layers)

```
qg = gemm(x, attn.qg.{layout})                 -- fused q_proj + output gate, per-head-interleaved
                                                   rows (docs/container-format.md "attn.qg")
q, gate = split(qg, head_dim, head_dim) per head
k = gemm(x, attn.k.{layout});  v = gemm(x, attn.v.{layout})
q = rmsnorm_zero_centered(q, q_norm);  k = rmsnorm_zero_centered(k, k_norm)   -- head_dim rows,
                                          NOT the GDN q/k l2norm (different op, see GDN below)
q, k = rope(q, k, cos, sin)                    -- partial_rotary_factor 0.25 of head_dim (256),
                                                   so only the first 64 of 256 dims rotate;
                                                   mrope_interleaved=true, sections [11,11,10]
                                                   over (temporal, height, width) position ids.
                                                   Text-only: all three streams == the token
                                                   position. With an image in the prompt they
                                                   DIVERGE, and so does the rope position from
                                                   the KV slot index -- docs/vision.md
                                                   "Text-side splicing"
                                                                        [src/kernels: rope]
o = r4d_attn_{prefill,decode}_h256_gqa6_{fp8kv,bf16kv}(q, k_cache, v_cache, ...)
                                                -- kv cache write happens BEFORE this call
                                                                        [src/kernels: kv cache write]
o = o * sigmoid(gate)                          -- the output gate, applied post-attention
                                                   (modeling_qwen3_5.py Qwen3_5Attention.forward:818)
out = gemm(o, attn.o.{layout})
```

`head_dim=256`, `gqa=6` (`num_attention_heads=24 / num_key_value_heads=4 = 6`), paged block size
16 -- exactly what `r4d_attn_dims()` reports (verified by `tests/smoke_r4d.cpp`). Prefill/chunked
prefill use the query-tiled kernel; decode (and speculative-verify windows up to `q_len*gqa<=64`)
use the split-KV kernel.

### GDN layer (48 of 64 layers -- Gated DeltaNet)

Math, from `Qwen3_5GatedDeltaNet.forward` (`modeling_qwen3_5.py:550-663`) and the kernels that
implement each step (`third_party/libr4d/r4d.h`):

```
mixed_qkv = gemm(x, gdn.in_proj_qkv.{layout})      -- [2*key_dim + value_dim], key_dim=2048 (16
                                                        heads * 128), value_dim=6144 (48*128)
mixed_qkv = causal_conv1d(mixed_qkv, conv1d_weight, width=4, silu)   -- depthwise, causal, per
                                                        channel; state cached per sequence
q, k, v = split(mixed_qkv, key_dim, key_dim, value_dim)
q, k = l2norm(q, k)                                -- per-head l2 normalize (NOT the attention
                                                        layer's RMSNorm -- this is
                                                        use_qk_l2norm_in_kernel in the reference)
q, k = repeat_interleave(q, k, num_v_heads // num_k_heads)   -- 48/16 = 3x, GQA-style broadening
b = gemm(x, gdn.in_proj_b);  a = gemm(x, gdn.in_proj_a)
beta = sigmoid(b)
g = -exp(A_log) * softplus(a + dt_bias)            -- per-(token, v_head) gate, cumsum'd per chunk
z = gemm(x, gdn.in_proj_z.{layout})                 -- output gate, applied by the norm below
                                                        (R1, docs/r9700.md: quantized like
                                                        in_proj_qkv/out_proj since this pass;
                                                        in_proj_a/in_proj_b above stay bf16)

-- prefill / chunked prefill (chunk 64):
A = r4d_gdn_kkt_solve_k128_c64_bf16(k, beta, g, cu, ...)     -- (I + strict_lower(diag(beta) K K^T
                                                                 e^{g_i-g_j}))^-1, gram in fp32,
                                                                 never reaches HBM
o, h_t = r4d_gdn_chunk_scan_k128_v128_c64_bf16(q, k, v, A, g, beta, h0, cu, ...)
                                                    -- WY recompute + state recurrence + output,
                                                       one kernel; conv/kkt prep is
                                                       r4d_gdn_conv_prep_w4_h128_bf16 in libr4d's
                                                       fuller build (this repo drives conv +
                                                       l2norm + gating with r4dx's own rope-style
                                                       kernels + r4d_gdn_kkt_solve until the prep
                                                       kernel is wired in)

-- decode (one candidate token, or a speculative window):
o = r4d_gdn_recurrent_update_k128_v128_bf16_fp32state(q, k, v, a, b, A_log, dt_bias, state, ...)
                                                    -- gating + l2norm + delta-rule update +
                                                       output against the paged fp32 state, one
                                                       state write per candidate token

out_core = r4d_gdn_gated_rmsnorm_h128_bf16(o, z, norm_weight, eps=1e-6, act=silu)
                                                    -- out = rms(o) . w . silu(z)
                                                       (Qwen3_5RMSNormGated; decode instead folds
                                                       this into the recurrent kernel's epilogue)
out = gemm(out_core, gdn.out_proj.{layout})
```

`head_k=128, head_v=128, chunk=64` -- `r4d_gdn_dims()` (verified by `tests/smoke_r4d.cpp`).
`num_k_heads=16, num_v_heads=48` (`linear_num_key_heads`, `linear_num_value_heads`), so every K/Q
head serves 3 V heads. GDN state is fp32 (`mamba_ssm_dtype: float32`), shape `[N, H, V, K]`
per-sequence recurrent state plus a `[N, conv_dim, kernel-1]` conv state.

### Interim chunked prefill

Until the dedicated 4-bit WMMA prefill kernel exists, a prompt is chunked into <=64-row pieces and
each chunk runs through the same skinny `r4d_gemm_*_nt_m64` GEMMs decode uses -- correct, not
prefill-throughput-optimal. Attention prefill already has a dedicated query-tiled kernel
(`r4d_attn_prefill_h256_gqa6_*`); it is only the *GEMM* side (qkv/gate_up/down/lm_head projections)
that takes the interim path.

## Own kernels (`src/kernels`)

The text-side set (all implemented; the "not yet implemented" this heading used to carry is long
stale):

| Kernel | Purpose |
|---|---|
| `rmsnorm` | `input_layernorm` / `post_attention_layernorm` / `text.final_norm` (zero-centered weight, `(1+w)`, per `Qwen3_5RMSNorm`) |
| `residual_add` | post-attention and post-MLP residual sum |
| `rope_partial_mrope` | partial rotary (first 25% of `head_dim`=256, i.e. 64 dims), NeoX half-split pairing, `theta=1e7`. The **text-only** entry point: all three (t,h,w) streams equal the token position, so the section split is a no-op and it takes one `int32[tokens]` array |
| `rope_partial_mrope3` | the **multimodal** counterpart: identical rotation, but each of the 32 frequency bins draws its position from one of three `int32[3, tokens]` rows. The bin -> stream assignment is `Qwen3_5TextRotaryEmbedding.recomposition_frequencies` with sections `[11,11,10]` -- the height stream overwrites bins `range(1, 3*sec_h, 3)` and width `range(2, 3*sec_w, 3)`, the rest stay temporal. With these sections that reduces to `bin % 3`, but the bounds are carried explicitly (a section vector where a slice does not reach its last in-range bin would diverge). Three identical rows make it **bit-identical** to `rope_partial_mrope`, which is what lets every text-only caller keep the cheaper entry point. See docs/vision.md "Text-side splicing" |
| `silu_mul` | MLP's `silu(gate) * up` |
| `quant_act_fp8_mxfp4a8` | per-row e4m3 activation quant + f32 scale, feeding `r4d_gemm_mxfp4a8_nt_m64` (parallels `r4d_quant_act_i8` for the int8 path, which r4d already provides) |
| `kv_write_paged_fp8_hnd` | write K/V into the fp8 e4m3 paged HND cache at decode/prefill time, applying the static per-(layer,head) descale |
| `embedding_lookup` | `text.embed_tokens` gather (host table; device-side gather for batches) |
| `argmax_sample` | greedy argmax now; top-k/top-p later |

Plus the vision tower's own five (docs/vision.md "The device kernels"), which exist because nothing
on the text side has the right shape: `layernorm` (mean-subtracted, with bias -- `rmsnorm` is wrong
for it in three ways), `bias_add`, `gelu_tanh` and `gelu_erf` (the encoder MLP and the merger use
DIFFERENT GELUs), `vision_qkv_rope` (fused qkv split + full-head axial rope) and
`vision_pos_embed` (the 4-tap learned-grid gather).

## Vision tower (`src/vision`, shipped -- full detail in docs/vision.md)

Architecturally unrelated to everything above: a different attention kernel, a real mean-subtracted
`nn.LayerNorm` instead of RMSNorm, a plain GELU MLP instead of SwiGLU, an axial 2-D rope over the
full head instead of a partial interleaved mrope, and a learned+interpolated position embedding with
no text-side analogue. It is bf16 end to end -- `vision.*` is passthrough in the container, never
quantized (docs/container-format.md), so there is one numeric path, not the text side's
mxfp4/w4a16/w4a8 fan-out.

```
image bytes
  -> decode (stb_image) -> smart_resize -> torch's uint8 antialias resampler   [host, bit-exact]
  -> rescale/normalize -> patchify to [num_patches, 1536] in 2x2 block order   [host]
  -> patch embed: [1536 -> 1152] GEMM + bias                       [r4d_gemm_bf16_nt_m64 + r4dx_bias_add_bf16]
  -> + learned 48x48 position grid, 4-tap bilinear                 [r4dx_vision_pos_embed_bf16]
  -> for each of 27 blocks:
       h = h + proj(attn(LayerNorm(h)))    LayerNorm  [r4dx_layernorm_bf16]
                                           qkv+rope   [GEMM + bias + r4dx_vision_qkv_rope_bf16]
                                           attention  [r4d_attn_vit_h72_bf16, per-image cu_seqlens]
       h = h + fc2(gelu_tanh(fc1(LayerNorm(h))))      [GEMM + bias + r4dx_gelu_tanh_bf16]
  -> merger: LayerNorm(1152) -> view[-1, 4608] -> fc1 -> GELU(erf) -> fc2      [r4dx_gelu_erf_bf16]
  -> [num_merged_tokens, 5120] bf16, ready to splice into the text embedding sequence
```

`Container` owns the loaded weights, `Model` owns the tower, `Model::EncodeImages` is the entry
point, and `--vision {auto|on|off}` decides whether the 0.9 GiB is paid for at all. The splicing of
those merged rows into the text sequence at the `248056` placeholder positions is the one piece
that is still unwritten.

## fp8 KV paging

Layout matches `R4DArgs.kv` exactly (`r4d.h:44-58`): `(num_blocks, kv_heads, block_size=16,
2*head_dim)`, fp8 e4m3, K then V per slot, strides in **elements** of that dtype
(`kv_block_stride` between blocks, `kv_head_stride` between heads inside a block). Descales are
static per-(layer, head): `text.layers.{i}.attn.k_descale` / `.v_descale`,
`fp32[kv_heads]`, calibration placeholder `1.0` (docs/container-format.md). Default max context
262144 tokens = 16384 blocks/sequence at block size 16 (raised from 131072 -- "Long-context
validation", docs/status.md/docs/perf.md; corrected here per review finding, 2026-09-20).

## GDN state

Two pieces of per-sequence state, both allocated by `src/core`:

- **Recurrent state**: `fp32[H=48, V=128, K=128]` (`h0`/`ht` in `r4d_gdn_chunk_scan_*`'s
  signature, `[N, H, V, K]` across sequences), read/written every chunk in prefill and every token
  in decode (`r4d_gdn_recurrent_update_*`'s `state` argument, paged by `ssm_state_indices`).
- **Conv state**: rolling buffer of `conv_kernel_size - 1 (=3)` + speculative-window entries per
  channel (`conv_dim = 2*key_dim + value_dim = 10240`), read/written by
  `r4d_gdn_conv_update_w4_h128_bf16` at decode and by the conv-prep path at prefill.

## Docs cross-reference

- `docs/container-format.md`: tensor names, quantized layouts, permutations -- the contract this
  forward pass reads against.
- `third_party/libr4d/r4d.h`: every kernel signature and shape comment cited above.
- `transformers/models/qwen3_5/modeling_qwen3_5.py` (read-only reference venv): the exact GDN and
  attention math this document paraphrases.
