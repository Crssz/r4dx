# r4dx architecture

## Module map

```
src/core/       device/stream/buffer plumbing: HIP device/stream lifetime, device buffer RAII,
                the paged KV cache allocator, the GDN state allocator.
src/kernels/    r4dx-owned HIP kernels -- the glue r4d does not provide (see "Own kernels" below).
src/model/      the layer graph: embedding -> N decoder layers -> final norm -> lm_head, built
                from a parsed r4dx container + config_model.
src/tokenizer/  BPE tokenizer + chat template application (minja).
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
                                                   over (temporal, height, width) position ids
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

## Own kernels (`src/kernels`, required, not yet implemented)

| Kernel | Purpose |
|---|---|
| `rmsnorm` | `input_layernorm` / `post_attention_layernorm` / `text.final_norm` (zero-centered weight, `(1+w)`, per `Qwen3_5RMSNorm`) |
| `residual_add` | post-attention and post-MLP residual sum |
| `rope_partial_mrope` | partial rotary (first 25% of `head_dim`=256, i.e. 64 dims), interleaved mrope with sections `[11,11,10]` over (t,h,w) position ids, `theta=1e7` |
| `silu_mul` | MLP's `silu(gate) * up` |
| `quant_act_fp8_mxfp4a8` | per-row e4m3 activation quant + f32 scale, feeding `r4d_gemm_mxfp4a8_nt_m64` (parallels `r4d_quant_act_i8` for the int8 path, which r4d already provides) |
| `kv_write_paged_fp8_hnd` | write K/V into the fp8 e4m3 paged HND cache at decode/prefill time, applying the static per-(layer,head) descale |
| `embedding_lookup` | `text.embed_tokens` gather (host table; device-side gather for batches) |
| `argmax_sample` | greedy argmax now; top-k/top-p later |

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
