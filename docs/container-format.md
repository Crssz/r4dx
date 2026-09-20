# r4dx weight container format

Contract between `src/convert` (writer) and `src/model` (reader). Any change here must be made in
both, or the loader silently reads garbage.

## Container shell: safetensors-compatible

Byte-identical to the [safetensors](https://github.com/huggingface/safetensors) file shape, so
generic safetensors tooling can still open an r4dx file for inspection:

```
[8 bytes]  N, little-endian uint64: length of the JSON header that follows
[N bytes]  JSON header (utf-8), see below
[...]      raw tensor data, concatenated, each tensor at the byte offsets its header entry gives
```

The JSON header is a flat object. Every key except `__metadata__` is a tensor name mapping to:

```json
{ "dtype": "U8", "shape": [rows, cols], "data_offsets": [start, end] }
```

`dtype` is always `U8` (uint8) for every tensor in this container, quantized or not -- r4dx does
its own dtype interpretation from `__metadata__` rather than trusting safetensors' own dtype enum,
because several of the layouts below (packed int4, e2m1, per-fragment-permuted) have no
safetensors dtype to name. A bf16 tensor is stored as `U8` with `shape = [..., 2]` (a trailing
byte-pair axis) and reconstituted by the loader from `__metadata__`.

## `__metadata__`

```json
{
  "r4dx_format_version": "1",
  "model_id": "Qwen/Qwen3.8-27B",
  "config_sha256": "<sha256 of the source config.json, hex>",
  "produced_by": "r4dx-convert <git rev>",
  "quant_summary": {
    "text.layers.*.mlp.gate_up": "mxfp4|w4a16|w4a8 (all three present; pick at load time)",
    "text.layers.*.attn.qg|k|v|o": "bf16",
    "lm_head": "mxfp4|w4a16|w4a8|bf16 (all four present)"
  },
  "model_config": { /* verbatim copy of the source config.json */ }
}
```

`config_sha256` lets the loader assert it is reading the container the converter actually built
for -- a stale container next to a bumped `config.json` fails loudly instead of loading a subtly
wrong RoPE base or head count.

## Tensor naming

```
text.embed_tokens                              bf16, host-resident
text.layers.{i}.input_layernorm                bf16  [hidden]
text.layers.{i}.attn.qg.{layout}                see "Attention" below (full-attention layers only)
text.layers.{i}.attn.k.{layout}                 [kv_heads*head_dim, hidden]  (full-attention layers only; R1)
text.layers.{i}.attn.v.{layout}                 [kv_heads*head_dim, hidden]  (full-attention layers only; R1)
text.layers.{i}.attn.o.{layout}                 [hidden, num_heads*head_dim]
text.layers.{i}.attn.q_norm                     bf16  [head_dim]
text.layers.{i}.attn.k_norm                     bf16  [head_dim]
text.layers.{i}.attn.k_descale                  fp32  [kv_heads]           (per-head amax/448.0 when --kv-calib is given; 1.0 placeholder otherwise)
text.layers.{i}.attn.v_descale                  fp32  [kv_heads]           (per-head amax/448.0 when --kv-calib is given; 1.0 placeholder otherwise)
text.layers.{i}.gdn.in_proj_qkv.{layout}        [2*key_dim + value_dim, hidden]     (gdn layers only)
text.layers.{i}.gdn.in_proj_z.{layout}          [value_dim, hidden]  (R1)
text.layers.{i}.gdn.in_proj_b                   bf16  [num_v_heads, hidden]
text.layers.{i}.gdn.in_proj_a                   bf16  [num_v_heads, hidden]
text.layers.{i}.gdn.conv1d_weight               bf16  [conv_dim, 1, kernel=4]
text.layers.{i}.gdn.A_log                       fp32  [num_v_heads]
text.layers.{i}.gdn.dt_bias                     fp32  [num_v_heads]
text.layers.{i}.gdn.norm_weight                 bf16  [head_v_dim]         (Qwen3_5RMSNormGated)
text.layers.{i}.gdn.out_proj.{layout}           [hidden, value_dim]
text.layers.{i}.post_attention_layernorm        bf16  [hidden]
text.layers.{i}.mlp.gate_up.{layout}            [2*intermediate, hidden]  (fused, see below)
text.layers.{i}.mlp.down.{layout}               [hidden, intermediate]
text.final_norm                                 bf16  [hidden]
lm_head.{layout}                                [vocab, hidden]           (all four layouts, incl. bf16)
mtp.*                                            same tensor set as one text layer, prefixed mtp.
vision.*                                         bf16 passthrough, HF parameter names preserved
```

`{layout}` is one of `mxfp4`, `w4a16`, `w4a8`, `bf16`. Every linear that participates in the A/B
(everything the milestone list benchmarks -- attention qg/k/v/o, GDN in/out proj, MLP gate_up/down,
lm_head) is stored in **all three quantized layouts plus bf16**, so the server can switch layouts
with a flag rather than a re-convert. `text.embed_tokens`, `text.*_norm`, `*.A_log`, `*.dt_bias`,
`*_descale`, `gdn.in_proj_a`/`gdn.in_proj_b`, `gdn.conv1d_weight`, and everything under `vision.*`
have exactly one layout (bf16 or fp32) and drop the `.{layout}` suffix.

**R1 (docs/r9700.md) note**: `attn.k`/`attn.v`/`gdn.in_proj_z` joined this multi-layout family in
this pass -- they used to be single-layout bf16 tensors with no `.{layout}` suffix at all (3.4
GB/token, 20.6% of every token, moved bf16 in every layout regardless of `--layout`). A container
converted before this pass still has the OLD bare form (`text.layers.{i}.attn.k`, no suffix);
`src/model/container.cpp`'s `LoadQuantLinearWithFallback` reads either form, falling back
requested-layout -> bf16 -> bare in that order, so old containers keep loading unmodified.
**`mtp.attn.k`/`mtp.attn.v` are deliberately NOT part of this change** and still write the old bare
bf16 form -- the MTP head stays bf16-only per this pass's task brief.

### Fused projections

- **`mlp.gate_up`**: `gate_proj` and `up_proj` concatenated on the output (row) axis --
  `[2*intermediate, hidden]`, rows `[0:intermediate)` = gate, `[intermediate:2*intermediate)` = up
  (`Qwen3_5MLP.forward`, `modeling_qwen3_5.py`: `down_proj(act(gate_proj(x)) * up_proj(x))`).
- **`attn.qg`** (full-attention layers only): `q_proj` is fused with the output gate
  (`attn_output_gate=true`). `Qwen3_5Attention.__init__` builds it as
  `nn.Linear(hidden, num_heads * head_dim * 2)`, and `forward` reshapes to
  `[*, num_heads, 2*head_dim]` then `torch.chunk(..., 2, dim=-1)` (`modeling_qwen3_5.py:761-790`).
  So row order is **per-head-interleaved**, not query-block-then-gate-block: for head `h`, rows
  `[h*2*head_dim : h*2*head_dim + head_dim)` are query, `[h*2*head_dim + head_dim : (h+1)*2*head_dim)`
  are gate. The gate is applied *after* attention and *before* `o_proj`
  (`attn_output * sigmoid(gate)`, line 818) -- it is not part of the attention math itself, so the
  loader splits `qg` back into a `[num_heads*head_dim, hidden]` query matrix and a
  `[num_heads*head_dim, hidden]` gate matrix at load time rather than the kernel seeing a fused
  tensor.
- **`gdn.in_proj_qkv`**: this checkpoint (`transformers` 5.17.0's `Qwen3_5GatedDeltaNet`) keeps
  `in_proj_qkv` (query+key+value, `[2*key_dim + value_dim, hidden]`), `in_proj_z`, `in_proj_b`,
  `in_proj_a` as four **separate** linears (`modeling_qwen3_5.py:544-547`) -- there is no fused
  `qkvz`/`ba` pair to reverse-engineer here; the container mirrors that 1:1. `in_proj_qkv`'s row
  order is `[key_dim) query, [key_dim:2*key_dim) key, [2*key_dim:) value` (the `torch.split` at
  line 603-611), each further reshaped to `[heads, head_k_dim or head_v_dim]` by the kernel side,
  not the converter.
- **`lm_head`**: `tie_word_embeddings=false`, so this is its own tensor, not a view of
  `text.embed_tokens`.

### Quantized layout tensors

Every `{layout}` variant of a linear `W [N, K]` (`N` = out features, `K` = in features) stores:

**`bf16`**: `<name>.bf16.w` -- `[N, K]` uint8 pairs (no permutation, row-major, K-contiguous).

**`w4a16`** (`r4d_gemm_w4a16_nt_m64`, f16 activation): asymmetric per-output-channel,
per-group-of-128-K quantization, `w ~= scale * (q - zero)`, `q` in `0..15`.
  - `<name>.w4a16.wq` -- `uint8[N * K / 2]`, **pre-permuted into the WMMA fragment order** so a
    wave's 32 lanes read 512 contiguous bytes for a (n-tile, k-step): lane `l`'s dword holds
    element `e` of `W[n0 + (l&15)][16*ks + 8*(e>>2) + 4*(l>>4) + (e&3)]`, dword nibble `2e` (e<4)
    / `2(e-4)+1` (e>=4) (`r4d_gemm_w4a16_nt_m64.hip` "LAYOUT" comment, using the
    `r4d_gdn_wmma.h` fragment map `idx = lane%16, k = 8*(e>>2) + 4*(lane>>4) + (e&3)`).
  - `<name>.w4a16.wsz` -- `uint32[N * K / 128]`, one dword per `(row, group)`: low 16 bits = f16
    `scale`, high 16 bits = f16 of `-(1024 + zero)` (ready for `v_pk_add_f16` against the
    `0x6400 | q` widened weight nibble) (`r4d_gemm_w4a16_nt_m64.hip:56-59`, `r4d.h` `r4d_gemm_w4a16_nt_m64_group()` = 128).
  - Group size: `r4d_gemm_w4a16_nt_m64_group()` (128, `R4D_GEMM_W4_GROUP`).

**`w4a8`** (`r4d_gemm_w4a8_nt_m64`, int8 activation): **byte-for-byte the same `wq` as `w4a16`**
  (`r4d_registry.hip`: "SHARED byte for byte with gemm_w4a16_nt_m64"), plus its own scale tensor:
  - `<name>.w4a8.wq` -- identical bytes to `<name>.w4a16.wq` (the converter writes it once and
    hardlinks/duplicates rather than re-deriving).
  - `<name>.w4a8.ws` -- `uint16[N * K / 128]` f16 scale only (no zero point -- signed 4-bit codes,
    symmetric).
  - Activation-side: `quant_act_i8` (`r4d_quant_act_i8`) produces a per-row int8 activation plus an
    f32 per-row scale at *inference* time, in the A-fragment byte order the kernel expects; nothing
    from this is stored in the container (activations are never static).
  - Group size: `R4D_GEMM_W4A8_GROUP=128` (the build flag third_party/CMakeLists.txt passes).

**`mxfp4`** (`r4d_gemm_mxfp4a8_nt_m64`, OCP MXFP4 weight + fp8 activation):
  - `<name>.mxfp4.wq` -- `uint8[N * K / 2]` packed e2m1 (two 4-bit floats per byte), permuted by
    `mxfp4_layout.py::permute_w`: checkpoint-order `[N, K/2]` reshaped to `[N/16, 16, K/16, 8]`,
    and for fragment slot `l` (0..31) with `r = l&15, h = l>>4`, `out[nt, ks, l, :] = w[nt, r, ks, 4h:4h+4]`
    -- one lane's 4-byte (8-element) slice per (n-tile, k-step), 32 lanes = 128 contiguous bytes
    per wave read (`mxfp4_layout.py:11-24`).
  - `<name>.mxfp4.ws` -- one E8M0 exponent byte per 32 K, laid out `[K/32][N]` (`r4d_gemm_mxfp4a8_nt_m64`
    shape comment in `r4d_registry.hip:266-267`).
  - `<name>.mxfp4.wref` -- `int8[N]`, one reference exponent per output row that the per-group E8M0
    scale is folded against (same source comment: "plus a per-row reference exponent").
  - Group size: `r4d_gemm_mxfp4a8_nt_m64_group()` (32).
  - Activation-side: `e4m3` with an f32 per-row scale, produced at inference time (not stored).

## KV descale tables

`text.layers.{i}.attn.k_descale` / `.v_descale`, `fp32[kv_heads]`, one scalar per KV head, feeding
`R4DArgs.k_descale` / `.v_descale` (`r4d.h:49-50`). `r4dx-convert --kv-calib <json>` fills these from
a `tools/reference/kv_calibrate.py` calibration run: `descale[head] = amax[head] / 448.0` (OCP e4m3fn
max), per `r4dx_convert::ResolveKvDescale` (`src/convert/include/r4dx_convert/kv_calib.hpp`). A layer
missing from the calibration JSON, or omitting `--kv-calib` entirely, falls back to the placeholder
`1.0`; the container format does not change either way, only the values. The real 64-layer container
(`D:\models\r4dx\qwen38-27b.r4dx`) was converted with `--kv-calib` covering all 16 full-attention
layers.

## `vision.*` and `mtp.*`

`vision.*` mirrors the HF `Qwen3_5VisionModel` parameter names 1:1, all `bf16`, no quantization
and no permutation -- the vision tower runs on `r4d_attn_vit_h72_bf16` in bf16 end to end for now.

`mtp.*` has the exact tensor set of one `text.layers.{i}` entry (its own attention or GDN block per
`mtp_num_hidden_layers=1`, MLP, norms), just prefixed `mtp.` instead of `text.layers.{i}.`.

### `mtp.draft_head.*` (OPTIONAL, docs/r9700.md R9 "reduced-vocab draft head")

```
mtp.draft_head.lm_head.{layout}     [draft_vocab_size, hidden]  (same {layout} family as lm_head)
mtp.draft_head.vocab_ids            raw int32[draft_vocab_size]: subset index -> real vocab id
```

Present only when the container was converted with `--draft-vocab-ids <json>` (`src/convert/main.cpp`);
absent from every container converted without that flag, including every container that predates
this feature. `mtp.draft_head.lm_head` is a plain row-slice of `lm_head.weight` (same `K`=`hidden`,
`N`=`draft_vocab_size` rows instead of the full vocab), planned/emitted through the exact same
`PlanLinearLayouts`/`EmitLinearLayouts` helpers every other quantized linear uses, so it carries the
same `{layout}` family (`mxfp4`/`w4a16`/`w4a8`/`bf16`) and the same `mtp_head_layout` load-time
selection as `mtp.attn.qg/o` and `mtp.mlp.gate_up/down`. `mtp.draft_head.vocab_ids` is a raw on-disk
int32 array (no dtype conversion, no permutation) -- element `i` is the REAL vocabulary id that
subset-local index `i` represents; `src/model/mtp_head.cpp`'s `r4dx_gather_i32` kernel is the only
consumer, mapping a subset-local argmax back to a real id entirely on-device.

**This tensor pair is used ONLY by `MtpHead::Draft` (drafting).** `Model::VerifyWindow` always reads
the real, full-vocab `lm_head.{layout}` tensor regardless of whether `mtp.draft_head.*` is present --
this is what keeps the technique lossless (docs/mtp.md's "reduced-vocab draft head" section has the
full argument): a draft token the reduced head's own subset could not represent is simply a rejected
draft, identical in effect to a wrong full-vocab-head guess, never a wrong ACCEPTED token.
`src/model/container.cpp`'s loader probes for `mtp.draft_head.vocab_ids` the same way it probes for
`mtp.norm` to decide `HasMtp()` -- absent means `MtpWeights::HasDraftHead()` is false and
`MtpHead::Draft` falls back to the full-vocab head unconditionally, so an old container (or any
container converted without `--draft-vocab-ids`) loads and behaves exactly as before this feature
existed.

## Provenance

- `r4d.h` (this repo's `third_party/libr4d/r4d.h`): every kernel's parameter comment, cited above
  by line range against the `windows-llp64` checkout at commit `7675605`.
- `third_party/libr4d/r4d_gemm_w4a16_nt_m64.hip`, `r4d_gemm_mxfp4a8_nt_m64.hip`,
  `r4d_gdn_wmma.h`, `r4d_registry.hip`, `mxfp4_layout.py`: exact permutation math.
- `transformers/models/qwen3_5/modeling_qwen3_5.py` (transformers 5.17.0, read-only reference venv
  at `C:\Users\user\dev\vLLM_for_AMD\.venv-rocm10`): `Qwen3_5MLP.forward` (gate_up fusion),
  `Qwen3_5Attention.__init__`/`forward` (q/gate fusion), `Qwen3_5GatedDeltaNet.__init__`/`forward`
  (GDN projection layout, conv/gating math).
