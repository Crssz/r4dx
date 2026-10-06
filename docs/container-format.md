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
    "text.layers.*.mlp.gate_up": "w4a16|trellis (and bf16 when requested; pick at load time)",
    "text.layers.*.attn.qg|k|v|o": "bf16",
    "lm_head": "w4a16|bf16"
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
rotation.signs                                  fp32  [hidden]             (rotated containers only, see "Residual rotation")
rotation.mix5                                   fp32  [5, 5]               (rotated containers only)
rotation.had_{down,o,gdn_out}_signs             fp32  [K of that linear]   (q2ab containers only)
```

`{layout}` is one of `w4a16`, `bf16`, or -- for the 400 decoder body linears of a
trellis container only -- `trellis` (see "Trellis body layout" below). A converter run stores every
linear that participates in the A/B (attention qg/k/v/o, GDN in/out proj, MLP gate_up/down, lm_head)
in each layout it was asked for, so the server can switch layouts with a flag rather than a
re-convert. (Older containers may also carry `w4a8` and `mxfp4` tensors; this build neither writes
nor reads them, see "Retired layouts" below.) `text.embed_tokens`, `text.*_norm`, `*.A_log`, `*.dt_bias`,
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
  (`attn_output * sigmoid(gate)`, line 818) -- it is not part of the attention math itself. The
  loader keeps `qg` fused: `Container::Load` reads it as one `[2*num_heads*head_dim, hidden]` linear
  (`src/model/container.cpp`, the `attn.qg` `LoadQuantLinearWithFallback` call), and
  `AttentionLayer::Forward` runs it as one GEMM. The split happens on that GEMM's output
  activations, per token, in `r4dx_model_attn_split_qg_bf16` (`src/model/attention/`), which
  de-interleaves each head's `2*head_dim` outputs into separate query and gate buffers. No weight is
  split or re-laid-out at load time.
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

A linear may legitimately carry **only** `.bf16.w` while its neighbours carry a quantized layout:
that is what `r4dx-convert --keep-bf16 <regex>` produces (docs/validation.md "Milestone 11 /
sensitivity"). `Container::Load` resolves every quantized body linear through
`LoadQuantLinearWithFallback`, which loads the requested layout when present, falls back to
`.bf16.w` when it is not, and reports the per-container fallback count on stderr. So a mixed-layout
container is a supported on-disk state, not a malformed one -- the run that made it is recorded in
`__metadata__.r4dx_convert_run.keep_bf16` (the pattern), `.keep_bf16_linears` (the resolved names)
and `.keep_bf16_extra_bytes` (the signed byte delta against the layouts they replaced).

**`w4a16`** (`r4d_gemm_w4a16_nt_m64`, f16 activation): asymmetric per-output-channel,
per-group-of-`g`-K quantization, `w ~= scale * (q - zero)`, `q` in `0..15`. `g` is **64** by default
(32 for a tensor packed with a group rule, e.g. the LM head), see docs/build-windows.md "w4a16 group
size" -- and the value a container was packed with is recorded in `__metadata__.quant.w4a16.group`
and checked against the kernel's own `r4d_gemm_w4a16_nt_m64_group()` at load. Older containers
(group 128) are refused. Everything below says 128 where it means `g`; at `g = 64` every `/ 128`
becomes `/ 64` and the weight costs 4.5 bits (`4 + 32/g`), which is what every container packed by
this build carries.
  - `<name>.w4a16.wq` -- `uint8[N * K / 2]`, **pre-permuted into the WMMA fragment order** so a
    wave's 32 lanes read 512 contiguous bytes for a (n-tile, k-step): lane `l`'s dword holds
    element `e` of `W[n0 + (l&15)][16*ks + 8*(e>>2) + 4*(l>>4) + (e&3)]`, dword nibble `2e` (e<4)
    / `2(e-4)+1` (e>=4) (`r4d_gemm_w4a16_nt_m64.hip` "LAYOUT" comment, using the
    `r4d_gdn_wmma.h` fragment map `idx = lane%16, k = 8*(e>>2) + 4*(lane>>4) + (e&3)`).
  - `<name>.w4a16.wsz` -- `uint32[N * K / g]`, one dword per `(row, group)`, ordered by 16-row
    n-tile, then group, then row within the tile (dword `(t * (K/g) + gi) * 16 + r` holds row
    `16t + r`, group `gi`; `PackW4A16Scales` in `quant_int4.hpp`): low 16 bits = f16
    `scale`, high 16 bits = f16 of `-(1024 + zero)` (ready for `v_pk_add_f16` against the
    `0x6400 | q` widened weight nibble) (`r4d_gemm_w4a16_nt_m64.hip:56-59`, `r4d.h` `r4d_gemm_w4a16_nt_m64_group()`).
    `wq` is unaffected by `g` -- only the number of `(scale, zero)` dwords changes.
  - Group size `g`: `r4d_gemm_w4a16_nt_m64_group()` (`R4D_GEMM_W4_GROUP`, 64; the kernel also
    serves 32 per tensor). Recorded per container in `__metadata__.quant.w4a16.group`; a container whose group
    differs from the loading binary's kernel is REFUSED at `Container::Load`
    (`CheckW4a16Group`) rather than read at the wrong stride -- but only when the load actually
    selects `w4a16` for the body, the lm head or the MTP head. The `quant` block is written
    unconditionally, so a container's recorded w4a16 group says nothing about whether it holds a
    `.w4a16.*` tensor, and a `--layout bf16` run reads none.
  - **Per-tensor groups** (docs/quant2.md section 5, Q3; `r4dx-convert --w4a16-group-rule
    "<regex>=<g>"`, `src/convert/include/r4dx_convert/w4a16_groups.hpp`). `quant.w4a16.group` stays
    the container's DEFAULT group -- the one it was packed with and the one a loading binary's
    `r4d_gemm_w4a16_nt_m64_group()` must equal. A linear packed at another `g` in {32, 64, 128}
    differs in exactly two places:
    - its scale tensor is **`<name>.w4a16.wsz.g<g>`** (e.g. `text.layers.5.mlp.down.w4a16.wsz.g32`),
      `uint32[N * K / g]` in the same tile/group/row order as above, instead of `<name>.w4a16.wsz`
      (`<name>.w4a16.wq` keeps its name and its bytes' layout: the 64-K packed block does not
      depend on the group);
    - its base is listed in the optional **`__metadata__.quant.w4a16.groups`**, an object
      `{"<container base>": g, ...}` naming ONLY the linears whose group differs from the default.
      No key, or an empty object, means every w4a16 linear is at the default -- a container
      converted without the flag has neither the key nor a renamed tensor, and is byte-identical
      to one converted before per-tensor groups existed.

    The loader resolves each w4a16 linear's group from the map (default otherwise), reads the
    matching scale name, and dispatches `r4d_gemm_w4a16_nt_m64_g` for a non-default group (the
    default keeps the existing entry). The default-group check applies to the unmapped linears; a
    mapped one needs `r4d_gemm_w4a16_nt_m64_has_group(g)`. Shape rules: `K % g == 0`, `K % 64 == 0`
    (32 is half a packed block), `N % 16 == 0` -- the converter refuses anything else while
    planning. At `g = 32` a weight costs 5 bits (`4 + 32/32`).

    The rename is the guard against **binaries that predate per-tensor groups**: they never read the
    map, look for the bare `.w4a16.wsz`, and so cannot read the scales at the default stride. Their
    `LoadQuantLinearWithFallback` (requested layout -> `.bf16.w` -> bare name) then throws "no tensor
    found ... in any known on-disk form" -- but only if the linear has no `.bf16.w` to fall back to;
    with one it would load the bf16 copy instead (right numbers, wrong layout and memory, one
    fallback-count line in the log). So `r4dx-convert` refuses a non-default group on any linear that
    also keeps a `.bf16.w` (convert with `--no-bf16` and an `--lm-head` spec without bf16, as the
    production recipe already does). `r4dx_format_version` is not bumped: no reader checks it.
    `__metadata__.r4dx_convert_run` records `w4a16_group_rules` (the rules as given; first match
    wins), `w4a16_groups` (the resolved map, identical to `quant.w4a16.groups`) and
    `w4a16_group_extra_bytes` (the signed `.wsz` byte delta against packing those linears at the
    default) -- all three only when rules were given. The DFlash2 drafter container never carries
    a map (`--w4a16-group-rule` is refused with `--dflash-gguf`).

### Retired layouts

The two layouts below (`w4a8` and `mxfp4`) are documented as the record of their on-disk format only.
`r4dx-convert` no longer writes them, this build has no kernel for them, and the loader ignores their
tensors in older containers (a `--layout` naming one is refused by name).

**`w4a8`** (`r4d_gemm_w4a8_nt_m64`, int8 activation): the same nibble *permutation* as `w4a16`
  (`r4d_registry.hip`: "SHARED byte for byte with gemm_w4a16_nt_m64"), but its own separately
  quantized tensor -- see the note below:
  - `<name>.w4a8.wq` -- `uint8[N * K / 2]`, `PackW4Nibbles`'d exactly like `w4a16.wq`, but from a
    SYMMETRIC quantization whose integer zero is pinned to 8. `r4d_gemm_w4a8_nt_m64`'s `dequant8()`
    has no zero-point input at all (it reads the stored nibble as a two's-complement signed nibble),
    so sharing one `wq` with `w4a16` would only be exact if `w4a16`'s per-group zero happened to be
    8 everywhere -- and this converter deliberately computes a free per-group zero for `w4a16`,
    because it is measurably more accurate. So r4dx-convert emits **two separate `wq` tensors**:
    `w4a16.wq` with a free zero and `w4a8.wq` with the zero pinned to 8. They are the same size and
    the same layout; only the codes differ. (`src/convert/include/r4dx_convert/quant_int4.hpp`'s
    header comment derives this from the two kernels' `dequant()` / `dequant8()`.)
  - `<name>.w4a8.ws` -- `uint32[N * K / 128]`, one dword per `(row, group)` in the same
    tile/group/row order and 4-byte stride as `w4a16.wsz`: low 16 bits = f16 `scale`, high 16 bits
    zero and never read (no zero point -- signed 4-bit codes, symmetric). Only the scale is
    *content*, but the stride is a dword: the kernel reads `Ws` through an `unsigned*` and masks
    `& 0xFFFF` (`r4d_gemm_w4a8_nt_m64.hip`), `PackW4A8Scales` (`quant_int4.hpp`) writes
    `uint32_t`, and `LoadQuantLinear` uploads it with `UploadRawU32` (`src/model/container.cpp`).
    A reader that takes it as `uint16[N * K / 128]` reads half the tensor, at the wrong stride.
  - Activation-side: `quant_act_i8` (`r4d_quant_act_i8`) produces a per-row int8 activation plus an
    f32 per-row scale at *inference* time, in the A-fragment byte order the kernel expects; nothing
    from this is stored in the container (activations are never static).
  - Group size: `R4D_GEMM_W4A8_GROUP=128` (the build flag third_party/CMakeLists.txt passed; the
    kernel's own default was 256). w4a8 and w4a16 had independent groups and independent converter
    constants (`kW4A8Group`, `kW4A16Group`), each asserted against its own kernel export.

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

### How the quantized values are chosen

**The byte layout above does not depend on this section.** Everything here is about *which* `q` /
`scale` / `zero` (or E8M0 exponent) values get written into those exact bytes, and a loader cannot
tell the two modes apart except by reading `r4dx_convert_run.quant_values` out of the container
metadata. `r4dx-convert --quant {rtn,search}` picks between them; **`rtn` is the default**, so a
converter command that predates these flags still produces the same bytes it always did. (A
container built before the flags existed has no `quant_values`/`imatrix` key at all; that absence
means `rtn`.) `--ldlq <regex>` (below) is a third, per-linear choice layered on top: the linears
listed in `r4dx_convert_run.ldlq_linears` were rounded by LDLQ, every other one by
`quant_values`. (`ldlq` = `"none"`, or no `ldlq` key at all in a container that predates the
flag, means no linear was.)

**`rtn`** -- the original quantizer, and what every container built before this flag existed
contains. One grid per `(row, group)` straight from the data's extremes, then round-to-nearest:
`w4a16` takes `scale = (max-min)/15`, `zero = round(-min/scale)` clipped to `0..15`; `w4a8` takes
`scale = amax/7` with the zero pinned to 8; `mxfp4` takes `exponent = ceil(log2(amax/6))`, i.e. it
always rounds the exponent up so nothing clips.

**`search`** -- the same byte layout, but each `(row, group)` keeps the candidate that minimizes

```
E(scale, zero) = sum_k  wt_k * (x_k - scale * (q_k - zero))^2,
                 q_k = clamp(round_half_away_from_zero(x_k / scale) + zero, 0, 15)
```

over 21 candidate scales spanning `0.85x .. 1.15x` of the `rtn` grid (step 10 is exactly `1.0x`, so
the `rtn` grid is itself a candidate) crossed with the three integer zeros `round(-min/scale) + {-1,
0, +1}` clamped to `0..15`, modelled on llama.cpp's `make_qkx2_quants`. The winner then gets one
weighted least-squares refit of the float scale, `scale* = sum_k wt_k x_k (q_k-z) / sum_k wt_k
(q_k-z)^2`, with the codes held fixed, kept only if it lowers the error. `w4a8` runs the same sweep
with the zero held at 8 (its kernel has no zero-point input); `mxfp4` instead tries the round-up
exponent and that exponent minus one and keeps the lower-error one. The `rtn` candidate is scored
first and later candidates must win **strictly**, so the search's error is never above `rtn`'s --
`tests/convert/test_quant_search.cpp` gates that group by group.

That guarantee is about `E(scale, zero)` for one `(row, group)`, and nothing more. It does **not**
imply a better model: `--quant search` without `--imatrix` lowers this objective on every group and
still lands at mean KL 0.0713 / top-1 87.71% against `rtn`'s 0.0724 / 88.4% on the real checkpoint
(`docs/validation.md` "Milestone 10"). Only the importance weighting turns the per-group win into a
model-level one.

`wt_k` is `1` unless `--imatrix <npz>` is given, in which case it is input channel `k`'s mean
activation energy from `tools/reference/imatrix_capture.py`'s importance matrix, keyed by these same
container base names. A linear absent from the `.npz` falls back to `wt_k = 1` with a warning and an
end-of-run coverage line.

The reference implementations are `src/convert/include/r4dx_convert/quant_search.hpp` and
`tools/convert_ref/w4_ref.py` / `mxfp4_ref.py`, which agree **bit-for-bit**
(`tools/convert_ref/selftest_compare.py` diffs the real CLI's output against the Python for all
three modes). That is only possible because the evaluation order is pinned: float32 throughout for
`w4a16`/`w4a8` (float64 for `mxfp4`), error accumulated sequentially over `k`, and no
floating-point contraction.

#### `ldlq` -- error-feedback rounding against the input Hessian

**Same bytes, same packers, same kernels** -- LDLQ changes only which `q` / `scale` / `zero` (or
E8M0 exponent) values are written, exactly like `search` does, and only for the linears
`r4dx-convert --ldlq <regex>` selects (`regex_search` over the container base names, like
`--keep-bf16`, which wins where both match). Design and gates: `docs/quant2.md` section 2.

`rtn` and `search` both choose every `(row, group)` in isolation and minimize the error in the
*weights*. What the model feels is the error in the *outputs*, `(W - Wq) x`, and input channels are
correlated. LDLQ (the GPTQ algorithm; LDLQ is QuIP's name for the same recursion) minimizes the
proxy `tr((W - Wq) H (W - Wq)^T)` with `H = E[x x^T]`, the linear's input second moment over a
calibration corpus, by rounding one input column `k` at a time and feeding that column's rounding
error forward onto the columns not yet rounded, weighted so that the later columns absorb it where
the inputs make that possible:

```
H     <- H + damp * mean(diag H) * I          (--ldlq-damp, default 0.01; retried at 10x, 100x)
U     =  upper Cholesky factor of H^-1        (H^-1 = U^T U, computed without forming H^-1)
for k in 0..K-1:
  if k starts a group: pick the group's (scale, zero) from the CURRENT (already error-updated)
                       weights with the `search` grid, weighted by diag(H), without the refit
  q[:, k] = round per layout (the same clamp/round `search` uses)
  e       = (W[:, k] - dequant(q[:, k])) / U[k][k]
  W[:, j] -= e * U[k][j]   for every j > k
```

The last line is applied lazily -- within a 128-column block immediately, to the columns after the
block as one matrix product per block -- which is the same arithmetic reordered, not an
approximation. The refit `search` ends with is skipped because it would fit the scale to codes that
error feedback then changes. Per layout the only differences are the ones `search` already has:
`w4a16` has a free zero, `w4a8` pins it at 8, `mxfp4` chooses between the round-up exponent and
that exponent minus one and then encodes E2M1. The implementation
(`src/convert/include/r4dx_convert/quant_ldlq.hpp`, CPU, deterministic: the output bytes do not
depend on `--threads`) requires `K` to be a multiple of 128.

It needs the Hessians. `tools/reference/hessian_capture.py` runs the bf16 checkpoint over a
calibration corpus and writes a directory -- one `.hess` file per distinct linear input (the packed
upper triangle of `H` in float32, with `K`, the token count and the trace for an integrity check)
and a `hessian.json` that maps each container base name to its file (linears reading the same input,
such as `gdn.in_proj_qkv` and `gdn.in_proj_z`, share one). `r4dx-convert --hessian-dir` points at
that directory; a selected linear missing from `hessian.json`, or with a different `K`, fails the
conversion before any tensor is written. The container records the pattern (`ldlq`), `ldlq_damp`,
`hessian_dir`, the sha256 of the `hessian.json` it used (`hessian_manifest_sha256`) and the
resolved `ldlq_linears` list in `r4dx_convert_run`.

`hessian.json` may also carry `"rms_keys"`: for each norm-fed in-projection, the Hessian of the norm's
weightless output `rms(x)`. Only a rotated conversion reads it (see "Residual rotation" below). Each
entry must name the listed rms twin of the base's `"keys"` file (`L03.in.hess` -> `L03.in.rms.hess`),
with the same `K` and, when both list them, the same `rows`. Any other entry is refused when the
manifest is read. An rms file with a zero diagonal is refused when it is factored.

## Residual rotation (`__metadata__.rotation`, `rotation.*`)

`r4dx-convert --rotate {none,q2a,q2ab} [--rotation-seed <u64>]` (docs/quant2.md sections 3-4;
`src/convert/include/r4dx_convert/rotation.hpp`). **`none` is the default and changes nothing**: no
fold, no `rotation.*` tensor, no `rotation` / `rotate` metadata key -- a container converted without
the flag is byte-identical to one converted before it existed (checked on the real checkpoint,
`--layers 4 --mtp on`, bf16 and `--quant search --imatrix` w4a16/w4a8/mxfp4: sha256-identical
files). Everything below applies only to a container whose `__metadata__` has a `rotation` key.

**The contract with the loader.** A rotated container's text-layer weights are only correct together
with the runtime's online ops; read as an ordinary container they produce garbage *with no error*.
So a binary MUST refuse a `rotation.kind` it does not implement (anything but `q2a` / `q2ab` today),
and MUST NOT apply any rotation op to a container without the key. Note that binaries built before
this section existed never look at `__metadata__.rotation` at all.

```json
"rotation": { "kind": "q2a" | "q2ab", "seed": 1592598565, "hidden": 5120, "block": 1024,
              "had": { "down": 512, "o": 256, "gdn_out": 128 } }      // "had": q2ab only
```

Residual rows are row vectors `x` (`hidden = 5120 = 5 x 1024`); `Q` is orthogonal:

```
x Q    y = x * d                               d = rotation.signs (+-1)
       y[b*1024 : (b+1)*1024] = FWHT(block) / 32    for b in 0..4   (natural/Sylvester order)
       z[b*1024 + i] = sum_c y[c*1024 + i] * R[c][b]                 R = rotation.mix5, R[c][b] at [c][b]
x Q^T  y[c*1024 + i] = sum_b x[b*1024 + i] * R[c][b];  FWHT / 32 per block;  * d
```

`FWHT` is the unnormalized Walsh-Hadamard transform, `H[i][j] = (-1)^popcount(i & j)`, so
`FWHT / sqrt(B)` is its own inverse. For `q2ab`, the input `h` of three out-projections is also
rotated online by a block Hadamard `h Hb := (h * s)`, then `FWHT / sqrt(B)` on each contiguous block
of `B`:

| linear | K (TP=1) | B | `s` |
|---|---|---|---|
| `mlp.down` | 17408 (`intermediate_size`) | 512 | `rotation.had_down_signs` |
| `attn.o` | 6144 (`num_attention_heads * head_dim`, one head per block) | 256 | `rotation.had_o_signs` |
| `gdn.out_proj` | 6144 (`linear_num_value_heads * linear_value_head_dim`, one head per block) | 128 | `rotation.had_gdn_out_signs` |

All five `rotation.*` tensors are fp32, stored like every other single-layout tensor (`U8`, trailing
element-width axis of 4, no `.{layout}` suffix): `rotation.signs` `[5120, 4]`, `rotation.mix5`
`[5, 5, 4]` (row-major `R[c][b]`), `rotation.had_*_signs` `[K, 4]`. Every value of a sign vector is
exactly `+1.0f` or `-1.0f`. The runtime reads these tensors and never regenerates them. Under TP the
Hadamard sign vectors are sliced with the same K range as their linear's columns (every rank's K is a
whole number of blocks), and `signs` / `mix5` are replicated.

**What is folded (fp32, before any quantization; text layers `0 .. layers_converted-1` only):**

| tensor | stored as |
|---|---|
| `text.layers.{i}.input_layernorm.rotated`, `.post_attention_layernorm.rotated` | **all zeros** (bf16 `+0.0`, same shape): the zero-centred norm then computes `rms(x) * 1`, which commutes with `Q`. **Renamed** from the bare names on purpose: a binary predating `__metadata__.rotation` fails on the missing bare tensor instead of running rotated weights in the unrotated basis |
| in-projections `gdn.in_proj_qkv/z/a/b`, `attn.qg/k/v` (norm = that layer's `input_layernorm`), `mlp.gate_up` (norm = `post_attention_layernorm`) | `W' = W diag(1 + w_norm) Q`: every row `r -> (r * (1 + w)) Q` |
| out-projections `gdn.out_proj`, `attn.o`, `mlp.down` | `W' = Q^T W`: every column, taken as a row, `-> c Q` |
| `q2ab` additionally, those three | `W'' = W' Hb`: every row `-> row Hb` (the runtime feeds `h Hb`) |

Every layout of a folded linear -- bf16, the quantized ones, a `--keep-bf16` bf16-only one, and the
bf16-only `gdn.in_proj_a/b` -- is produced from the folded fp32, rounded once. The fold is per row on
the K side and per column on the N side, so the output-axis fusions (`mlp.gate_up`'s gate|up rows,
`attn.qg`'s per-head query/gate interleave) are unaffected. **Untouched, byte-identical to an
unrotated container:** `text.embed_tokens`, `text.final_norm`, `lm_head`, all of `mtp.*` (including
the draft head), `vision.*`, and inside the text layers `gdn.conv1d_weight`, `A_log`, `dt_bias`,
`gdn.norm_weight`, `attn.q_norm/k_norm` and the KV descales. The runtime therefore applies `x Q` to
the stack's input rows (after the embedding gather and vision splice), `x Q^T` to its output rows
(before `final_norm` and anything else that reads the pre-final-norm residual: MTP priming, verify
windows), and `x Q^T` to DFlash2 feature captures -- the MTP head, the DFlash2 drafter and the vision
tower all run un-rotated.

**Generation** (converter only). One splitmix64 stream seeded with `seed`, drawn in this order:
`rotation.signs` (5120 draws, sign = the draw's top bit, set -> `-1`); then 25 standard normals by
Box-Muller from 13 pairs of draws (`u1 = ((a >> 11) + 1) * 2^-53`, `u2 = (b >> 11) * 2^-53`,
`r cos(2 pi u2)`, `r sin(2 pi u2)`, `r = sqrt(-2 ln u1)`; the last sine is dropped), filled row-major
into a 5x5 matrix whose rows are orthonormalized in order by modified Gram-Schmidt (two passes, fp64)
and stored as `R` in fp32; then, for `q2ab`, `had_down_signs`, `had_o_signs`, `had_gdn_out_signs` by
top bit. `q2a` and `q2ab` of the same seed share `signs` and `mix5`. Default seed `0x5EED2025`.

**Choosing the quantized values on a rotated container.** `--ldlq` rounds a folded linear against its
input Hessian carried into the new basis, computed exactly from the captured (un-rotated) Hessians:

- **An in-projection with an rms Hessian.** When `hessian.json`'s `"rms_keys"` lists the
  in-projection, `--ldlq` uses `Q^T H_rms Q`. `H_rms` is captured on the norm's weightless output
  `rms(x)`, and the folded linear's input is exactly `rms(x) Q`. This needs no division, so it is
  defined even where `(1 + w) == 0`.
- **An in-projection without one.** `--ldlq` divides the norm out of the post-norm capture:
  `Q^T D^-1 H D^-1 Q`, with `D = diag(1 + w_norm)`. The conversion is refused during planning if
  any `|1 + w| < 1e-3`, and the message says to capture the rms Hessians with
  `hessian_capture.py --rms-only`.
- **A `q2ab` out-projection.** `--ldlq` uses `Hb^T H Hb`.
- **An out-projection's N-side `Q^T`** does not change the linear's input, so its Hessian is
  unchanged.

Then `tr(W' H' W'^T) = tr(W H W^T)`: the proxy of the same error is the same. `r4dx_convert_run`
lists the in-projections that used `H_rms` as `ldlq_rms_linears`, rotated containers only. See
`docs/quant2.md` 3.2 for why the division path is not good enough. `--imatrix`
vectors, which are only `diag(H)`, are carried over under their own diagonal model instead --
`diag(M^T diag(v) M)`, which after a Hadamard is a per-block constant -- so on rotated linears they
carry little information; `--ldlq` is the tool there.

`r4dx_convert_run` records `rotate` and `rotation_seed` (and `imatrix_rotated` when `--imatrix` was
given); `quant_summary` gains a note for the zeroed norms and the `rotation.*` tensors.
`tests/convert/test_rotation.cpp` (ctest `convert_rotation`) gates the generation, `Q` / `Hb` against
their closed forms and for orthogonality, the fold identities on a miniature layer, and the Hessian
change of basis against a direct capture on the folded input.

### Any hidden size, and the Gemma 4 variant (option A)

`hidden` is no longer fixed at 5120. `block = ChooseRotationBlock(hidden)` (largest power of two
dividing `hidden`, capped at 1024) and `nblk = hidden / block`: 5120 -> 1024 x 5 (byte-identical to
the Qwen containers above, pinned by `convert_rotation`'s golden hash), Gemma 4's 3840 -> 256 x 15.
`__metadata__.rotation` gains `"nblk"` **only when it is not 5**, and the mixing matrix is
`rotation.mix5` `[5, 5]` for `nblk == 5` and `rotation.mix` `[nblk, nblk]` otherwise.

A Gemma 4 container (`model_arch` `gemma4_unified`, plain-weight norms) differs in three ways:

- `input_layernorm` and `pre_feedforward_layernorm` fold as `W' = W diag(w) Q` (norm offset 0, not
  `1 + w`); the folded norm tensors are then stored as ones (plain `rms(x) * 1`, which commutes with `Q`; naming of
  those tensors is the Gemma converter branch's call).
- `post_attention_layernorm` / `post_feedforward_layernorm` do **not** fold (`Q^T` does not commute with
  the channelwise weight). `attn.o` and `mlp.down` get only `W Hb` on the K side (`LinearFold::kHadOnly`);
  their outputs stay in the original basis and the runtime computes
  `r' = r' + Q( post_norm(sublayer_out) )` (`r4dx_post_rmsnorm_rotate_add_bf16`). The metadata says so
  with `"out_fold": "had_only"`; a runtime that does not implement it must refuse the container.
- Hadamard sites: `had` is `{down: 512, o: 256, o_full: 256}` (no `gdn_out`), tensors
  `rotation.had_down_signs` `[15360]`, `rotation.had_o_signs` `[4096]` (sliding layers' o_proj) and
  `rotation.had_o_full_signs` `[8192]` (full layers'). Draw order after `signs` / `mix`: down, o, o_full.

## Gemma 4 container (`model_arch` `gemma4_unified`)

Written by `r4dx-convert` when `config.json` says `model_type` `gemma4_unified` (`src/convert/gemma_layout.{hpp,cpp}`;
`tests/convert/test_gemma_layout.cpp`, ctest `convert_gemma_layout`). The checkpoint is one `model.safetensors`
(no index). `__metadata__` gains `"model_arch": "gemma4_unified"` and `"norm_kind": "plain"` (written only for
Gemma; `model::DetectArch` reads them, falling back to `model_config.model_type`); `model_config` is the verbatim
`config.json`; `model_id` is the checkpoint directory's name. No `mtp.*` (`--mtp on` is refused).

| Container tensor | HF source | Notes |
|---|---|---|
| `text.embed_tokens` | `model.language_model.embed_tokens.weight` | bf16 `[vocab, hidden]`; the `sqrt(hidden)` scale is a runtime op, never folded |
| `text.final_norm` | `model.language_model.norm.weight` | bf16 `[hidden]`, a plain weight (`x * w`) |
| `lm_head.{layout}` | the same embedding (tied) | written untied, like Qwen's, in the `--lm-head` layouts |
| `text.layers.{i}.input_layernorm`, `post_attention_layernorm`, `pre_feedforward_layernorm`, `post_feedforward_layernorm` | same names | bf16 `[hidden]`, raw. Under `--rotate` the first and third are folded away and stored as bf16 **ones** as `<name>.rotated` |
| `text.layers.{i}.attn.{q,k,o}.{layout}`, `attn.v.{layout}` | `self_attn.{q,k,o,v}_proj.weight` | `q` has no output gate. **No `attn.v` on full layers** (V is the raw `k_proj` output, `attention_k_eq_v`) |
| `text.layers.{i}.attn.{q_norm,k_norm}` | `self_attn.{q,k}_norm.weight` | bf16 `[head_dim]`: 256 sliding, 512 full; `v_norm` has no weight and stores nothing |
| `text.layers.{i}.attn.{k,v}_descale` | `--kv-calib` (else 1.0) | fp32 `[kv_heads of the layer]`: **8 on sliding layers, 1 on full layers** |
| `text.layers.{i}.mlp.gate_up.{layout}`, `mlp.down.{layout}` | `mlp.{gate,up,down}_proj.weight` | gate rows first |
| `text.layers.{i}.layer_scalar` | `layer_scalar` | the `[1]` bf16 buffer widened to fp32 `[1, 4]` (applied once, after the MLP residual add) |
| `vision.vision_embedder.*`, `vision.embed_vision.*`, `audio.embed_audio.*` | `model.vision_embedder.*`, `model.embed_vision.*`, `model.embed_audio.*` | bf16 passthrough, `model.` stripped; only with `--vision on` / `--audio on` (both default off) |

Shapes come from the safetensors headers and are cross-checked against `text_config` before anything is written. The
converter's **coverage audit**: every checkpoint tensor must be consumed by the table above or allow-listed (layers
past `--layers`; vision / audio tensors while their flag is off). Anything else -- a tensor the converter does not
know, a stray `lm_head.weight` -- fails the run before the header is written. `--trellis-from`,
`--record-reuse-guard` and `--reuse-tensors-from` are refused for Gemma for now.

## Trellis body layout (`__metadata__.quant.trellis`, `<base>.trellis.*`)

`r4dx-convert --trellis-from <oracle dir>` (docs/trellis-kernel.md sections 2-3; the format itself
is EXL3/QTIP's, docs/trellis.md). The converter **imports** the bits that
`tools/reference/trellis_quant.py quantize-model` (one rate) or `mix` (EXL3's 4/5 allocation) wrote,
pinned by the manifest's sha256 (`--trellis-manifest-sha256`) and checked by a full CPU
reconstruction (`--trellis-verify full`, the only mode a release converter accepts) before the file
gets its final name. It never re-encodes. A container converted without the flag has no
`quant.trellis` key and no `.trellis.*` tensor, and is byte-identical to one converted before it
existed.

**Which linears.** All 400 decoder linears (HF tensors), which are 336 container bases:
`gdn.in_proj_qkv`, `gdn.in_proj_z`, `gdn.out_proj` (48 GDN layers), `attn.qg`, `attn.k`, `attn.v`,
`attn.o` (16 attention layers), `mlp.gate_up` (gate and up, two parts) and `mlp.down` (64 layers).
A trellis base carries **only** its three `.trellis.*` tensors: no `.w4a16.*` and no `.bf16.w`
(a `--keep-bf16` base instead carries only `.bf16.w`). Everything else follows the flags as in any
container; the shipped recipe writes `lm_head` as w4a16 g32 (`--lm-head w4a16 --w4a16-group-rule
"^lm_head$=32"`), the MTP head's `qg`/`o`/`gate_up`/`down` as w4a16 g64, and `mtp.attn.k/v`,
`mtp.fc`, `gdn.in_proj_a/b`, the conv, norms, embeddings and `vision.*` as bf16/fp32 -- 611
tensors byte-identical to `qwen38-27b-q2ab_hv2_q3.r4dx`'s (trellis-kernel.md 10.3).

| tensor | shape (U8) | content |
|---|---|---|
| `<base>.trellis.w` | `[N*K*KB/8]`, 1-D | uint32 little-endian ring words, `8*KB` per 16x16 tile, in the pair grid below |
| `<base>.trellis.suh` | `[P*K, 2]` | fp16 input scale with the input Hadamard's signs folded in; `P` = 2 for `mlp.gate_up` (gate's `K` values, then up's), 1 otherwise |
| `<base>.trellis.svh` | `[N, 2]` | fp16 output scale with the output Hadamard's signs folded in, in container row order (gate rows, then up rows) |

`KB` is the linear's trellis bits per weight (4 or 5). **Pair grid:** word `w` (`0 <= w < 8*KB`) of
tile `(tn, tk)` -- HF rows `16tn..16tn+15`, columns `16tk..16tk+15` -- is at uint32 index

```
idx(tn, tk, w) = (((tn >> 1) * (K/16) + tk) * 2 + (tn & 1)) * 8*KB + w
```

a whole-tile permutation of the oracle's `[k/16][n/16][8KB]` order. Inside a tile the words are the
oracle's exactly: the stream's bit `S[32w]` at bit 31, positions in EXL3's tensor-core order, a
16-bit tail-biting state, EXL3's `mul1` codebook. The linear computes, per 128-block of `K` and of
`N` (`H` = the 128-point Sylvester Hadamard scaled by `1/sqrt(128)`, `D` = the decoded `[N, K]`
matrix):

```
y = svh * H_N( D . H_K( suh * x ) )          (per part for gate_up: part p uses suh[p*K : (p+1)*K])
```

The runtime applies `H_K(suh * x)` as the GEMM's input transform (f16, times `2^prescale_log2`) and
`H_N` then `svh` (times `2^-prescale_log2`) in the GEMM epilogue, in fp32 with one bf16 rounding.

```json
"quant": { "trellis": {
  "format": "r4dx-trellis", "version": 1, "codebook": "mul1",
  "codebook_consts": {"mult": "0x83dcd12d", "k_inv_f16": "0x1eee", "k_bias_f16": "0xc931"},
  "state_bits": 16, "tail_biting": true, "position_order": "exl3-tensor-core",
  "bitstream": "ring-u32-msb-first", "tile_grid": "n32-pairs-k-major",
  "hadamard": {"block": 128, "order": "sylvester-natural", "scale": "1/sqrt(128)",
               "input": "x*suh then H", "output": "H then *svh"},
  "prescale_log2": 0,
  "linears": { "text.layers.0.gdn.in_proj_qkv": {"bits": 5},
               "text.layers.0.mlp.gate_up": {"bits": 5, "parts": [17408, 17408]}, ... } } }
```

- `bits` is per linear; `gate_up`'s two parts always share it. `parts` appears only on a
  multi-part linear. A linear may carry its own `prescale_log2` (the container-wide one is the
  default; both in [-16, 16]; 0 in every shipped container).
- `quant_summary` names the body as e.g. `"body: trellis mul1 KB=4 x200 + KB=5 x200 (imported)"`
  (HF tensors per rate).
- `r4dx_convert_run.trellis` records the import: `manifest`, `manifest_sha256`, `manifest_form`
  (`quantize-model` / `mix`), `hessian_basis` (only `matched` is imported unless
  `--trellis-allow-basis exl3`), `hessian_manifest_sha256`, `code_sha256`, the oracle's `recipe`,
  `allocation_sources` (a mix), `hf_tensors_per_K`, `kept_bf16`, every source `files` entry with its
  sha256, and `verify: {mode, tolerance, result, worst, worst_tensor, checked, failed}`. `result` is
  written as a fixed-width "pending" placeholder and patched in place once the reconstruction check
  has passed.

**The contract with the loader** (`src/model/trellis_meta.h`; `Container::Load` and
`Container::LoadShard`). A trellis-aware binary refuses: any unknown `format`, `version`,
`codebook`, `position_order`, `bitstream`, `tile_grid` or `hadamard` value; a `bits` the kernel does
not instantiate; a `linears` field other than `bits`, `parts`, `prescale_log2`; a `.trellis.*`
tensor without a `linears` entry (or in a container without the block) and an entry without its
three tensors; byte sizes other than `N*K*bits/8`, `P*K*2`, `N*2`; any `K`, part or `N` (and, under
TP, any rank range) not a multiple of 128; `rotation` together with `trellis` (a Qwen container; a gemma4_unified one may carry both, docs/gemma4-plan.md 10.2); a `verify` record
that is missing or does not pass; and **any `--layout` other than `trellis`** on such a container
(and `--layout trellis` on one without the block). A trellis `--lm-head`/MTP-head layout request is
mapped to w4a16. A binary that predates the format finds neither `.w4a16.*` nor `.bf16.w` for a body
linear and throws "no tensor found" -- that, not `r4dx_format_version` (not bumped), is the guard.

**Tensor parallelism** (TP = 2): column-parallel linears (`gate_up`, `qkv`, `z`, `qg`, `k`, `v`) cut
`.trellis.w` by pair rows and `svh` by rank rows and replicate `suh`; row-parallel ones (`down`, `o`,
`out_proj`) cut `.trellis.w` per pair row by the rank's K range and `suh` by that range and replicate
`svh`. Every rank range of this model is 128-aligned.

**Sizes** (`tools/quant2/decode_bytes.py`: every text-layer weight plus `lm_head`):
`qwen38-27b-trellis-k4m.r4dx` (KB = 4 everywhere) 15.66 GiB on disk, 12.130 GiB decode;
`qwen38-27b-trellis-mix45m.r4dx` (KB = 5 on 168 of the 336 bases, 200 of 400 HF tensors) 17.08 GiB,
13.546 GiB; `qwen38-27b-q2ab_hv2_q3.r4dx` for comparison 13.680 GiB decode. The recipe and the
measured quality and speed are in docs/quant2.md section 7 ("Trellis").

## KV descale tables

`text.layers.{i}.attn.k_descale` / `.v_descale`, `fp32[kv_heads]`, one scalar per KV head, feeding
`R4DArgs.k_descale` / `.v_descale` (`r4d.h:49-50`). `r4dx-convert --kv-calib <json>` fills these from
a calibration run: `descale[head] = amax[head] / 448.0` (OCP e4m3fn
max), per `r4dx_convert::ResolveKvDescale` (`src/convert/include/r4dx_convert/kv_calib.hpp`). A layer
missing from the calibration JSON, or omitting `--kv-calib` entirely, falls back to the placeholder
`1.0`; the container format does not change either way, only the values.

Two producers write that JSON, in the same layout: `tools/reference/kv_calibrate.py` (a
**prototype** -- raw embeddings into one layer, every preceding layer skipped) and
`tools/reference/kv_calibrate_full.py` (the real one -- the whole 64-layer stack over a 6-file
calibration corpus, all 16 full-attention layers in one file, at
`E:\models\r4dx\qwen38-27b.kvcalib-full.json`). Use the latter. The real 64-layer container
(`E:\models\r4dx\qwen38-27b.r4dx`) was converted with `--kv-calib` covering all 16 full-attention
layers, but from the **prototype's** numbers, which run 1.4-3.3x low on `k_amax` and up to 8.8x low
on `v_amax` in the back half of the stack -- it needs re-converting against the full-forward JSON
before its fp8 KV cache means anything (see `tools/reference/README.md`, "kv_calibrate_full.py").

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
same `{layout}` family (`w4a16`/`bf16`) and the same `mtp_head_layout` load-time
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

## DFlash2 draft container (Milestone 5 groundwork)

A DFlash2 speculative-decoding draft model (background: `docs/dflash2.md`, the "DFlash2 assessment"
section of `docs/mtp.md`, `docs/status.md`) converts from its own GGUF v3 source (`E:\models\Qwen3.8-27B-DFlash2\Qwen3.8-27B-DFlash2-Q8_0.gguf`)
into its OWN r4dx container -- a **separate file** from the main text-model container, never mixed
into `text.*`/`vision.*`/`mtp.*`, so every existing loader/container is unaffected by this section.
Written by `r4dx-convert --dflash-gguf <gguf> --out <container> --layout {w4a16,bf16}`
(`src/convert/main.cpp`'s `RunDflashConvert`, `src/convert/include/r4dx_convert/gguf_reader.hpp` +
`dflash2_container.hpp`). Same safetensors-shaped shell as every other r4dx container (8-byte
header length + JSON header + raw tensor bytes, every tensor `dtype: "U8"`).

### `__metadata__`

```json
{
  "r4dx_format_version": "1",
  "container_kind": "dflash2_draft",
  "model_id": "z-lab/Qwen3.8-27B-DFlash2",
  "source_gguf": { "filename": "Qwen3.8-27B-DFlash2-Q8_0.gguf", "sha256_first_1mib": "<hex>" },
  "produced_by": "r4dx-convert --dflash-gguf ...",
  "dflash2": {
    "hidden_size": 5120, "block_count": 5, "feed_forward_length": 17408,
    "attention": { "head_count": 32, "head_count_kv": 8, "key_length": 128, "value_length": 128,
                   "causal": false, "rms_eps": 1e-6, "sliding_window": 2048,
                   "sliding_window_pattern": [true, true, true, true, true] },
    "rope": { "freq_base": 1e7, "dimension_sections": [64, 0, 0, 0], "n_rot": 128,
              "pairing": "neox_split_half" },
    "block_size": 8, "conv_kernel_size": 2, "conv_group_size": 16,
    "selector_rank": 256, "selector_top_k": 16,
    "target_layers": [6, 20, 34, 48, 62],
    "context_length": 262144, "mask_token_id": 248070, "vocab_size": 248320,
    "source_file_type": 7, "layout": "w4a16"
  },
  "quant": { /* same shape BuildQuantMetadata() emits for the main container -- see below */ }
}
```

`container_kind` is a new field absent from every pre-existing text-model container -- a loader
that checks it can refuse to open a draft container as a body model (or vice versa) instead of
misreading its tensor set; a loader that doesn't check it simply never looks, so nothing existing
breaks. `target_layers` is stored **exactly as the GGUF gives it**: 0-based indices into the
TARGET model's own decoder-layer stack, each meaning "the residual stream as it ENTERS target
layer L" (== the OUTPUT of target layer L-1) -- for the real container, `[6,20,34,48,62]` = the
outputs of target layers `[5,19,33,47,61]`. **`n_rot` = 128 (the FULL `key_length`/head_dim), not a
partial rotary factor**: the real GGUF carries no `dflash.rope.dimension_count` key, and
`llama-model.cpp`'s generic hparam load defaults `n_rot_full` to `n_embd_head_k_full` whenever that
key is absent (verified by reading that file, not assumed) -- confirmed absent from the real
file's 48-key metadata dump. `dimension_sections=[64,0,0,0]` (sum 64) is the M-RoPE PAIR-count
split across (temporal, height, width, extra); `rotated_dims = 2*sum(sections) = 128 = n_rot`, so
this is a full rotation with only the temporal (sequential-position) section active -- DFlash2's
draft attention needs no 2D/3D image/video position awareness, unlike the main text model's
partial (0.25 factor, 3-section) M-RoPE. **Pairing is `neox_split_half`, NOT the main model's
interleaved M-RoPE**: `llama-model.cpp` returns `LLAMA_ROPE_TYPE_MROPE` (not `IMROPE`) for
`LLM_ARCH_DFLASH`, and ggml's MROPE dispatch (`ggml-cpu/ops.cpp`'s `rotate_pairs`, stride
`n_dims/2`) pairs `src[ic]` with `src[ic + n_dims/2]` -- GPT-NeoX split-half pairing `(i, i+64)`
across all 128 dims. The interleaved scheme only fires on ggml's separate `is_imrope` branch,
which `LLM_ARCH_DFLASH` never takes; do not assume the main text model's interleaved convention
carries over here. `vocab_size` has no scalar GGUF key of its own (the draft
carries no `tokenizer.ggml.tokens` array -- it has no embedding table or lm_head of its own, see
below) and is derived from `selector_predecessor.weight`'s own on-disk shape instead.

### Tensor naming

```
dflash.fc.{layout}                                [hidden, len(target_layers)*hidden]  (GGUF fc.weight)
dflash.enc_output_norm                            f32  [hidden]                 (GGUF enc.output_norm.weight)
dflash.output_norm                                f32  [hidden]                 (GGUF output_norm.weight)
dflash.selector.hidden.{layout}                   [selector_rank, hidden]       (GGUF selector_hidden.weight)
dflash.selector.predecessor                       bf16 [vocab, selector_rank]   row-gather codebook, NOT a GEMM
dflash.selector.successor                         bf16 [vocab, selector_rank]   row-gather codebook, NOT a GEMM
dflash.layers.{i}.input_layernorm                 f32  [hidden]                 (GGUF blk.{i}.attn_norm.weight)
dflash.layers.{i}.self_attn.q_proj.{layout}       [head_count*key_length, hidden]
dflash.layers.{i}.self_attn.k_proj.{layout}       [head_count_kv*key_length, hidden]
dflash.layers.{i}.self_attn.v_proj.{layout}       [head_count_kv*value_length, hidden]
dflash.layers.{i}.self_attn.o_proj.{layout}       [hidden, head_count*value_length]
dflash.layers.{i}.self_attn.q_norm                f32  [key_length]
dflash.layers.{i}.self_attn.k_norm                f32  [key_length]
dflash.layers.{i}.self_attn.conv.base             bf16 [2 sides, 2 taps, hidden]  (GGUF blk.{i}.attn_conv_base)
dflash.layers.{i}.self_attn.conv.proj.{layout}    [2*conv_kernel_size*(hidden/conv_group_size), hidden]
dflash.layers.{i}.post_attention_layernorm        f32  [hidden]                 (GGUF blk.{i}.ffn_norm.weight)
dflash.layers.{i}.mlp.gate_proj.{layout}          [feed_forward_length, hidden]
dflash.layers.{i}.mlp.up_proj.{layout}            [feed_forward_length, hidden]
dflash.layers.{i}.mlp.down_proj.{layout}          [hidden, feed_forward_length]
dflash.layers.{i}.mlp.conv.base                   bf16 [2 sides, 2 taps, hidden]  (GGUF blk.{i}.ffn_conv_base)
dflash.layers.{i}.mlp.conv.proj.{layout}          [2*conv_kernel_size*(hidden/conv_group_size), hidden]
```

`dflash.fc`'s K is `len(target_layers) * hidden` (`dflash.cpp`'s
`n_embd_inp_enc_impl = target_layer_ids.size() * hparams.n_embd`) -- for the real checkpoint this
equals `block_count * hidden` only because it happens to have 5 target-feature taps and 5 draft
layers; derive the encoder input width from `target_layers`' length, not from `block_count`.

Naming mirrors two existing conventions at once: `layers.{i}.input_layernorm` /
`post_attention_layernorm` matches the main container's `text.layers.{i}.*` norm names; `self_attn.
q_proj`/`mlp.gate_proj` etc. use HF-linear-style names per the task brief, mapped 1:1 from the
GGUF's own `blk.{i}.*` names (cross-checked against the read-only
`C:\Users\user\dev\ROCmFPX\gguf-py\gguf\constants.py`'s `TENSOR_NAMES` table for
`MODEL_TENSOR.DFLASH_*`, not imported). **`ffn_gate`/`ffn_up` are kept as two separate tensors**
(`mlp.gate_proj`/`mlp.up_proj`), unlike the main text-model container's fused `mlp.gate_up` --
the GGUF source never fuses them, and fusing here would be an unrequested extra permutation with
no test coverage; a future pass MAY fuse them the same way if a measured perf reason appears.
**The draft has no `embed_tokens`/`lm_head` of its own** (task A1's own note: it reuses the
TARGET model's embedding table and lm_head) -- there is deliberately no `dflash.embed_tokens` or
`dflash.lm_head` tensor in this container; the loader must be handed the target `Container`'s
`EmbedTokensHost()`/`EmbedTokensDevice()` and `LmHead()` instead (see "Loader" below).

### Orientation

Every GGUF linear's `ne` is `[K, N]` (`ne[0]` = in/fastest-varying, `ne[1]` = out) -- e.g.
`fc.weight` `ne=[25600,5120]` (`K=25600` in, `N=5120` out), `blk.0.attn_q.weight` `ne=[5120,4096]`
(`K=5120` in, `N=4096` out; confirmed directly against the real file's own tensor-info dump, not
assumed). GGUF stores a tensor's bytes in `ne`-order (`ne[1]` outer, `ne[0]` inner), which for a 2D
tensor is **already** row-major `[N,K]` with row = output feature -- byte-identical to the
orientation `r4dx_convert::PlanLinearLayouts`/`EmitLinearLayouts` (the SAME packers the main HF
converter uses) already expect. So `N=ne[1]`, `K=ne[0]`, and the flattened
`GgufReader::DequantToF32` output feeds those packers with **no transpose or permutation** --
verified by `tests/convert/test_dflash_container.cpp`'s orientation check (container bf16 bytes
bit-identical to `GgufReader::DequantToBf16` of the same source tensor).

`selector.predecessor`/`selector.successor`: GGUF `ne=[selector_rank, vocab]` flattens to
`[vocab][selector_rank]` row-major (vocab outer, rank inner) -- already exactly the
`[vocab, selector_rank]` row-gather layout the container wants, so these are a straight
`DequantToBf16` + raw write, same as any other bf16 passthrough tensor, with NO GEMM packing
(`selector_hidden` IS a GEMM -- `hidden -> selector_rank` -- and goes through the normal linear
packers; only the two codebooks are row-gather).

Note the container's *declared* tensor shape (the `shape` field the safetensors-style header
records) is row-major, slowest-axis-first -- the REVERSE of GGUF's fastest-first `ne` -- for every
bf16/f32 passthrough tensor here, even though the on-disk BYTES are copied verbatim with no
permutation. `conv.base`'s declared shape is therefore `[2 sides, 2 taps, hidden, <width>]` and
`selector.predecessor`/`successor`'s is `[vocab, selector_rank, <width>]`, matching the table
above; declaring `ne` itself (fastest-first) as the shape is a defect (`PlanDflash2Bf16`/
`PlanDflash2F32` reverse `ne` before recording it, see their own comments).

`conv.base`: GGUF `ne=[hidden,2,2]` flattens to `[side][tap][hidden]` (hidden innermost) --
exactly `third_party/libr4d/r4d_dflash_conv_body.h`'s expected per-side `[taps,H]` slice
(`base[0,h]`/`base[1,h]` in that header's own formula), so this is also a straight `DequantToBf16`
+ raw write with no reshaping; a per-side pointer for the libr4d kernel call is
`conv_base_ptr + side*(conv_kernel_size*hidden)`.

`conv.proj`: the GGUF weight itself needs no permutation either (it's a normal linear, `K=hidden`
in, `N=2*conv_kernel_size*(hidden/conv_group_size)` out) -- the "group index fastest, then tap,
then side" structure the reference (`ROCmFPX/src/models/dflash.cpp`) describes is how the
CONSUMER reshapes this GEMM's flat `N`-wide OUTPUT per token at inference time, not a permutation
the converter applies to the weight's rows. Documented here so the loader's/forward-pass author's
row-order assumption is written down once: output index `r` in `[0, N)` decomposes as
`side = r / (conv_kernel_size * n_groups)`, `tap = (r / n_groups) % conv_kernel_size`,
`group = r % n_groups`, where `n_groups = hidden / conv_group_size`.

### Loader (CPU-only stage; GPU upload deferred)

`src/model/dflash_draft_weights.h`/`.cpp` (`r4dx::model::DflashDraftWeights`) opens a
`dflash2_draft` container and exposes its metadata + tensor directory (name, shape, dtype) via the
same `r4dx_convert::SafetensorsReader` the main `Container` reuses for tensor data -- no HIP call
anywhere in this stage, so it is fully CPU-unit-testable
(`tests/model/test_dflash_draft_weights.cpp`). Uploading tensors to device memory (through the
existing `QuantLinear`/`DeviceBuffer` path) and the DFlash2 forward pass itself are explicitly
**not** implemented yet -- see that header's own `// TODO(dflash2-forward)` markers.

## Provenance

- `r4d.h` (this repo's `third_party/libr4d/r4d.h`): every kernel's parameter comment, cited above
  by line range against the `windows-llp64` checkout at commit `7675605`.
- `third_party/libr4d/r4d_gemm_w4a16_nt_m64.hip`, `r4d_gdn_wmma.h`, `r4d_registry.hip`: exact
  permutation math.
- `transformers/models/qwen3_5/modeling_qwen3_5.py` (transformers 5.17.0, in the read-only reference
  venv of the time, since deleted): `Qwen3_5MLP.forward` (gate_up fusion),
  `Qwen3_5Attention.__init__`/`forward` (q/gate fusion), `Qwen3_5GatedDeltaNet.__init__`/`forward`
  (GDN projection layout, conv/gating math).
