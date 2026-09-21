# Vision tower

Status as of 2026-09-21: **the tower runs on the GPU.** The host half (`src/vision/{image_decode,
preprocess,position_ids,vision_index}`) produces `pixel_values`, the pos-embed interpolation taps,
the vision rope positions and `cu_seqlens`, all bit-exact against the real `transformers`
processor/helpers; the device half (`src/vision/{vision_weights,vision_tower}`, plus five new
kernels in `src/kernels`) loads the container's 333 `vision.*` tensors and runs patch embed ->
position embedding -> 27 encoder blocks over `r4d_attn_vit_h72_bf16` -> the 2x2 patch merger,
validated against every tensor `tools/reference/vision_golden.py` dumps for all three of its cases.
`Model::EncodeImages` is the entry point and `--vision {auto|on|off}` / `--image-max-pixels N` are
the flags on both binaries.

Status as of 2026-09-22: **the model answers questions about a picture.** The merged rows are
spliced into the text embedding sequence at the `248056` placeholder positions, 3-axis (t,h,w)
mrope position ids reach every rope call site in the decode stack (prefill, plain decode, MTP
verify/draft, DFlash2 injection/draft), and the rope delta is carried so that rope position and KV
slot index stay correctly divergent for the rest of the conversation.
`Model::PrefillMultimodal(tokens, spans)` is the entry point; see "Text-side splicing" for the
design and "Splicing pass: what was measured" for the numbers.

Status as of 2026-09-22 (stage 5): **the user-facing surface is wired up.** `--image <path>`
(repeatable) on `r4dx-cli`, one or more `/image <path>` lines in `--chat`, and OpenAI-shaped
`image_url`/`input_image` content parts on the server's `/v1/chat/completions` -- all going through
the one shared `r4dx::vision::ExpandImagePlaceholders` (`src/vision/image_prompt.h`) that turns the
chat template's single `<|image_pad|>` marker per image into that image's real merged-token-count
run. See "User-facing wiring: --image and image_url" below for the design and the real-hardware
numbers (including the OCR check actually reading a rendered string back off a real image).
`tests/vision/tool_vision_chat` remains the lower-level driver these two production entry points
now sit beside, not on top of. See "What's done / what's not" at the end for the fuller accounting.

For the rest of this document, the design contract, the measured evidence behind it and the
implementation status are interleaved per section.

## Why this is its own document, not a section of docs/architecture.md

The vision tower is architecturally unrelated to the GDN/attention decoder stack
docs/architecture.md describes (different attention kernel, different norm, different rope, a
learned+interpolated position embedding with no text-side analogue, a completely different
preprocessing pipeline). docs/architecture.md's "Vision tower" section now names the shipped
components and points here; this file remains the place where the numbers and the derivations live.

## Model facts (real checkpoint, `C:\AI\models\Qwen3.8-27B\config.json`'s `vision_config`, confirmed
by loading the real config with the reference venv's `transformers` 5.17.0)

```
depth                    27
hidden_size              1152
num_heads                16          (head_dim = 1152/16 = 72)
hidden_act               gelu_pytorch_tanh
intermediate_size        4304
in_channels               3
patch_size                16
temporal_patch_size        2
spatial_merge_size          2
out_hidden_size          5120        (== text hidden_size, the merger's output feeds straight into
                                       the text embedding sequence, no extra projection)
num_position_embeddings  2304        (48x48 learned position-embedding grid, 48 = sqrt(2304))
rope_theta              10000.0      (vision's own axial rope base -- DIFFERENT from the text side's
                                       1e7 `rope_theta` in docs/architecture.md)
image_token_id          248056       (config.json top level, not vision_config)
video_token_id          248057
vision_start_token_id   248053
vision_end_token_id     248054
```

`third_party/libr4d/r4d.h`'s `r4d_attn_vit_h72_bf16` (dense, non-causal, per-image `cu_seqlens`,
bf16 throughout, head_dim 72 native) is the one r4d kernel this tower needs beyond the GEMM/norm/
activation primitives `src/kernels`/`third_party/libr4d` already provide for the text side.
`docs/container-format.md`'s `vision.*` -- confirmed present in the real checkpoint, `model.visual.*`
prefix, 333 tensors (patch_embed.proj.{weight,bias}, pos_embed.weight, 27x
blocks.{i}.{norm1,attn.qkv,attn.proj,norm2,mlp.linear_fc1,mlp.linear_fc2}.{weight,bias}, merger.
{norm,linear_fc1,linear_fc2}.{weight,bias}) -- is bf16 passthrough, **never quantized**, so there is
exactly one numeric path to validate here, not the text side's mxfp4/w4a16/w4a8 fan-out.

## Forward pass (from `transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5VisionModel`, read
directly from the reference venv, not paraphrased from memory)

### 1. Preprocessing (host-side -- **implemented**: `src/vision/{image_decode,preprocess}.{h,cpp}`)

Real checkpoint's `preprocessor_config.json` names `image_processor_type:
Qwen2VLImageProcessorFast`, but that name is only a deprecated alias in `transformers` 5.17.0
("the `Fast` suffix for image processors has been removed", it warns, and hands back the plain
class) -- there is no distinct fast processor to compare against.
`AutoImageProcessor` resolves that string to
`transformers.models.qwen2_vl.image_processing_qwen2_vl.Qwen2VLImageProcessor`, whose MRO is
`(Qwen2VLImageProcessor, TorchvisionBackend, BaseImageProcessor, ...)`: the slow numpy/PIL
processor of older releases is gone and the surviving class *is* the torchvision-backed one. The
rest of the config:
`size: {shortest_edge: 65536, longest_edge: 16777216}` (in **pixels**, i.e. `h*w` bounds, not a
side length), `patch_size: 16`, `temporal_patch_size: 2`, `merge_size: 2`, `image_mean: [0.5,0.5,
0.5]`, `image_std: [0.5,0.5,0.5]` (i.e. normalize to roughly `[-1, 1]`, not ImageNet stats). The
pipeline, read from `transformers.models.qwen2_vl.image_processing_qwen2_vl` and
`transformers.image_processing_backends.TorchvisionBackend` (the base class above it), is:

1. **RGB conversion.** `do_convert_rgb=True` -> PIL's `Image.convert("RGB")`, which for an RGBA
   input **drops the alpha channel** -- it does *not* composite over white or any other
   background. Greyscale is replicated across all three channels. `stb_image` with `req_comp=3`
   does exactly the same two things, which is why `src/vision/image_decode.cpp` needs no alpha
   handling of its own. This is the one claim here that is a reading of someone else's source
   rather than arithmetic, so it has its own fixture: `vision_golden.py` writes an RGBA PNG whose
   alpha sweeps 0..255 (including fully transparent columns, whose RGB would be unrecoverable if
   anything composited them) plus a greyscale PNG, and dumps PIL's own `convert("RGB")` output for
   both. `stb_image` matches **byte for byte** (0 differing bytes). The other three formats
   `image_decode.h` claims are covered the same way (`make_format_fixtures` writes one picture as
   BMP, GIF and JPEG and dumps PIL's decode of each): BMP and GIF match **exactly**, JPEG to
   **max 2/255** on 435 of 6720 bytes -- not a bug, just stb_image's IDCT against libjpeg's, which
   is why that one case is the only tolerance in the whole preprocessing test.
2. **smart_resize.** Both sides rounded to a multiple of `patch_size * merge_size = 32`, then
   clamped into `[shortest_edge, longest_edge]` *total pixels* (a real image is nowhere near the
   16 M ceiling; the floor is what tiny images hit). The rounding is Python's `round()`, i.e.
   **half-to-even** -- `48/32 = 1.5` rounds up to 2 but `80/32 = 2.5` rounds down to 2, and a
   `floor(x + 0.5)` implementation gets the second one wrong.
3. **Resampling** -- see the subsection below; this is the step the earlier version of this
   document got wrong.
4. **Rescale + normalize, fused.** `_fuse_mean_std_and_rescale_factor` folds `rescale_factor` into
   the mean/std, so the byte never passes through a `[0,1]` intermediate: the result is
   `(u8 - 127.5) / 127.5` in fp32, a subtraction and a **division** (not a multiply by a
   precomputed reciprocal -- that is an fp32 ulp off on ~95% of pixels).
5. **Patchify.** Reshape into `(grid_t=1 for a still image, grid_h, grid_w, in_channels,
   temporal_patch_size, patch_size, patch_size)` and flatten to `[num_patches,
   in_channels*temporal_patch_size*patch_size^2 = 1536]`, with the single frame duplicated across
   `temporal_patch_size=2` and the rows **already reordered into 2x2 spatial-merge-block-major
   order** (patch index `((block_row*blocks_w + block_col)*merge + in_row)*merge + in_col`, NOT
   raster h/w order). That reordering is what lets the merger later do a plain
   `.view(-1, hidden_size*merge^2)` with no explicit permute (see "4. Merger" below).

`image_grid_thw = [t, h_patches, w_patches]` (one row per image) is the shape contract every
downstream step needs.

`tests/vision/test_preprocess.cpp` asserts **bit equality** against `vision_golden.py`'s
`pixel_values` -- not a tolerance, because every step above is integer-deterministic. Measured, on
all three golden cases (5,443,584 fp32 values total): `max_abs = 0`, `mean_abs = 0`, 0 differing
values.

#### Resampling (the part a "looks like an image" test never catches)

`Qwen2VLImageProcessor.resample` is `PILImageResampling.BICUBIC` (a **class attribute** -- this
checkpoint's `preprocessor_config.json` has `"resample": null`, so reading the JSON alone tells you
nothing and the value has to come from the class), and `TorchvisionBackend.resize` calls
`tvF.resize(..., antialias=True)` on the **uint8** tensor. That dispatches to torch's uint8
antialias kernel, which is **not** a float bicubic and is emphatically not the "bilinear resize" an
earlier version of this document claimed.

None of that is inferred: instrumenting `TorchvisionBackend.resize` during a real
`AutoImageProcessor` call on a 613x409 image reports, verbatim, `in_dtype=torch.uint8
in_device=cpu in_shape=[1, 3, 409, 613] -> out_dtype=torch.uint8 out_shape=[1, 3, 416, 608],
resample=3 (BICUBIC), antialias=True`. So the tensor really is uint8 on both sides of the call --
no float cast happens anywhere in the resize -- and the target size really is `smart_resize`'s.

Its four load-bearing properties:

- **Pillow's cubic kernel with `a = -0.5`**, not the `a = -0.75` of torch's *non*-antialiased
  bicubic (aten's `HelperInterpCubic::aa_filter` cites Pillow's `Resample.c` directly).
- **Antialiasing widens the kernel support by the downscale factor** when `scale >= 1`
  (`support = 2*scale`, `invscale = 1/scale`); upscaling keeps `support = 2`, `invscale = 1`.
- **int16 fixed-point weights.** The shift is the largest `p < 22` for which every scaled weight
  still fits an int16; accumulation adds `1 << (p-1)` and arithmetic-shifts right by `p`.
- **Two separable passes, horizontal first, with a uint8 intermediate.** The horizontal pass'
  output is rounded and clamped back to uint8 before the vertical pass runs, so the passes do not
  commute and bicubic's negative-lobe overshoot is clipped mid-way.

Each of those matters numerically, not academically. Measured against the reference on random-noise
images (the worst case for any rounding disagreement), changing exactly one property at a time and
re-measuring on all four size pairs below:

| deviation from the reference algorithm | worst max-abs error | share of bytes differing |
|---|---|---|
| `a = -0.75` instead of Pillow's `-0.5` | 25/255 | 72% - 91% |
| float intermediate instead of uint8 between the passes | 25/255 | 11% - 23% |
| vertical pass first instead of horizontal | 25/255 | 15% - 32% |

(The per-pair spread is real and worth reading: the strong-downscale pair, 1280x960 -> 320x256, is
the most forgiving of all three mistakes -- 6/255 on 72%, 1/255 on 11%, 1/255 on 15% -- because
averaging ~4x4 source pixels per output pixel washes out kernel and rounding differences. The
moderate resizes a real attachment actually gets are where the error is worst, so a "tested it on a
thumbnail" check would badly understate all three.)

Reproducing all four properties (`src/vision/preprocess.cpp`'s `BuildAxisWeights`/`ResamplePass`)
gives **zero** differing bytes out of **1,898,496** across four random-noise size pairs -- 640x416
-> 608x384 (both axes down), 613x409 -> 608x416 (the real `smart_resize` pair: width down, height
up), 140x100 -> 288x224 (both up) and 1280x960 -> 320x256 (strong downscale) -- and zero across
both golden images. Those four pairs are not a one-off measurement: `vision_golden.py`'s
`make_resampler_fixtures` saves each noise input as a PNG and dumps the reference's own
`tvF.resize` output, and `tests/vision/test_preprocess.cpp`'s `CheckResamplerOnNoise` asserts byte
equality on every run, so a future "harmless" change to the resampler fails ctest instead of
quietly degrading OCR-type accuracy.

Note that the 448x448 golden case never resizes at all (448 is already a multiple of 32, and
torchvision returns the input unchanged when the target size matches) -- which is exactly why
`vision_golden.py` also produces a 613x409 -> 608x416 case, and why the noise fixtures cover the
upscale and strong-downscale regimes neither photograph reaches.

### 2. Patch embedding (**implemented**: `VisionTower::Encode` step 1)

`Qwen3_5VisionPatchEmbed`: a `Conv3d(3, 1152, kernel=[2,16,16], stride=[2,16,16])` -- but since
`pixel_values` already arrives pre-patchified/flattened (`[num_patches, 1536]`), this Conv3d is
mathematically **a single dense `[1536 -> 1152]` matmul + bias per patch row**, not a real
convolution. `r4dx` implements it as a `r4d_gemm_bf16_nt_m64` call against
`vision.patch_embed.proj.weight` read as `[1152, 1536]` (the checkpoint's `[1152, 3, 2, 16, 16]`
Conv3d weight IS that matrix, byte for byte, in the row-major order the GEMM wants), exactly like
every other linear in this codebase -- no Conv3d kernel was ported.

One precision detail that is easy to get backwards: the reference casts `pixel_values` to the
projection's dtype **before** the matmul (`hidden_states.to(dtype=target_dtype)` in
`Qwen3_5VisionPatchEmbed.forward`), so the fp32 -> bf16 rounding happens on the INPUT, not on the
GEMM's output alone. `VisionTower::Encode` rounds on the host for the same reason, then uploads
bf16 (which also halves the H2D traffic). Measured against the golden's `patch_embed_out`:
`rel_l2 = 2.26e-05`, `max_abs = 1` bf16 ulp.

### 3. Position embedding (learned, bilinearly interpolated -- no text-side analogue)

A learned `[2304, 1152]` embedding table (`num_grid_per_side = 48 = sqrt(2304)`), resampled to each
image's actual `(h_patches, w_patches)` grid via 4-tap bilinear interpolation
(`align_corners=True`) computed by `transformers.vision_utils.
get_vision_interpolation_indices_and_weights` -- **not** a fixed sinusoidal table like the text
side's rope. Per patch: gather 4 rows of the table at `interp_indices[patch]` and weighted-sum by
`interp_weights[patch]`, in the same spatial-merge-block-major patch order preprocessing already
emits. Added to the patch-embed output before the encoder blocks.

`align_corners=True` means the closed form `src = index * (side - 1) / max(size - 1, 1)`, evaluated
in **fp32** (the reference casts the index to float32 first), with the two taps `floor(src)` and
`floor(src)+1` clamped to `[0, side-1]` and weights `max(1 - |src - floor(src) - k|, 0)`; the 2-D
taps are the outer product of the two axes'. The per-patch `(row, col)` fed into it is *decoded*
from the flat block-major patch index rather than iterated, because the flat order is not raster:
`in_col = i % merge`, `in_row = (i / merge) % merge`, `block_col = (i / merge^2) % blocks_w`,
`block_row = i / (merge^2 * blocks_w)`.

**Implemented**: `src/vision/vision_index.cpp`'s `BuildPosEmbedInterpolation` (host, the index
math) and `r4dx_vision_pos_embed_bf16` (device, the gather + weighted sum + residual add).
`vision_golden.py` dumps `interp_indices`, `interp_weights` *and* the resulting `pos_embeds`, plus
the raw `pos_embed_table` (so the test can gather from the real learned table without needing a
converted container). `tests/vision/test_vision_index.cpp` measures: indices and weights match
**exactly**, and the gathered position embedding matches to `max_abs = 0` on all three cases. On
device, against the same golden through the real container's table: `pos_embeds` matches to
`max_abs = 0` (i.e. **exactly**, in fp32) on all three cases.

The one thing the device kernel has to get right beyond the arithmetic is the ORDER of the last
two roundings. The reference line is

```
hidden_states = hidden_states + pos_embeds.to(hidden_states.dtype)
```

-- the fp32 weighted sum is rounded to bf16 **first**, and only then added to the bf16 patch-embed
output. Adding in fp32 and rounding once afterwards is a different tensor (the two differ by up to
half a bf16 ulp on every element), so `r4dx_vision_pos_embed_bf16` reproduces the reference's order
explicitly, and `tests/kernels/test_vision_kernels.cpp` checks that specific ordering against a CPU
reference that also rounds twice.

### 4. 27 encoder blocks (`Qwen3_5VisionBlock`, pre-norm, residual)

```
h = h + attn(LayerNorm(h), cu_seqlens, rope_cos_sin)     -- norm1, NOT RMSNorm (real nn.LayerNorm,
                                                             mean-subtracted, eps=1e-6)
h = h + mlp(LayerNorm(h))                                -- norm2
```

`cu_seqlens` itself is `BuildCuSeqlens` in `src/vision/vision_index.cpp` (prefix sums of `h*w`, one
segment per frame): exact match against the golden, including the two-segment two-image case. Note
which reference function that is: `Qwen3_5VisionModel.forward` calls
`get_vision_attention_seqlens(grid_thw, config)`, a wrapper that forwards `merge_temporal=False`
(one segment per **frame**, the qwen2_vl convention -- `True` would make a whole clip one segment,
which is a different model family's rule) to `get_vision_cu_seqlens` and adds a `max_seqlen` that
is `None` unless flash-attention is requested. `vision_golden.py` dumps the wrapper's output, i.e.
the tensor the model actually hands its blocks, and asserts it still equals the bare
`get_vision_cu_seqlens(grid_thw)` whose rule `BuildCuSeqlens` implements -- so if a future
`transformers` teaches that wrapper a new rule, the golden run fails loudly instead of the C++
silently diverging.

`attn` (`Qwen3_5VisionAttention`): fused `qkv = Linear(1152, 1152*3)` (bias=True, unlike the text
side's `attn.qg`/`attn.k`/`attn.v` which are bias-free), split into `q,k,v [num_patches, 16, 72]`,
apply axial rope (below) to q/k only, then **dense non-causal attention scoped per-image by
`cu_seqlens`** (one segment per image; for a batch of images/video frames, `cu_seqlens` has one
entry per frame after `Qwen3_5VisionRotaryEmbedding`'s temporal handling) -- this is exactly
`r4d_attn_vit_h72_bf16`'s contract (`third_party/libr4d/r4d.h`: `q,k,v,o [total_tokens, heads, 72]`
contiguous, `cu_seqlens[num_seqs+1]`, `scale = head_dim**-0.5`). Output projection `proj = Linear
(1152, 1152)` (bias=True).

Axial rope (`Qwen3_5VisionRotaryEmbedding`): `inv_freq` over `head_dim/4 = 18` frequencies (not
`head_dim/2` like the text side, because h and w each get half of the frequencies), `theta=10000`.
For each patch's `(h_pos, w_pos)` (block-major order, from `get_vision_position_ids`, same file):
`freq_h = h_pos * inv_freq`, `freq_w = w_pos * inv_freq`, concatenated `[freq_h, freq_w]` (36 dims),
then duplicated to fill the full 72-dim head (`cat([freqs_hw, freqs_hw])`) -- **the full head_dim
rotates, no partial_rotary_factor** (unlike the text side's 0.25). `cos`/`sin` computed and applied
in fp32 (`apply_rotary_pos_emb_vision` explicitly upcasts), same rotate-half convention as the text
side's rope kernel. **Implemented**: `src/vision/vision_index.cpp`'s
`BuildVisionRopePositionIds` (the `(h,w)` pairs, exact match against the golden) and
`BuildVisionRopeCosSin` (the table, `max_abs` 1.9e-6 vs the golden -- the fp32-vs-double
disagreement in `theta^(-2i/36)` multiplied by a position index up to ~40).

**A trap in the reference harness, not in the model**: `Qwen3_5VisionRotaryEmbedding` registers
`inv_freq` as a non-persistent `nn.Buffer`, and `module.to(dtype=torch.bfloat16)` casts buffers
too. A naive bf16 golden run therefore rounds the frequency table to bf16's ~3 significant decimal
digits (`1/theta^(2/36)` becomes 0.597656 instead of 0.599484) -- which defeats the rope module's
own explicit fp32 intent (its forward wraps the cos/sin computation in
`maybe_autocast(enabled=False)` and `.float()`s both operands), and is worth up to **0.047
absolute on cos/sin** at a 38-wide patch grid, an order of magnitude more than any bf16 matmul
noise downstream of it. `vision_golden.py` restores the table by recomputing it through the rope
class's own `compute_axial_rope_parameters`, because r4dx's rope kernels compute `inv_freq` in
fp32 on device (`src/kernels/src/r4dx_kernels.hip`) and a golden that baked in the bf16
quantization would make every block output unmatchable for the right implementation. Anyone
diffing r4dx against a stock `from_pretrained(dtype=bfloat16)` HF run should expect this
difference and not chase it as a bug in r4dx.

`mlp` (`Qwen3_5VisionMLP`): `Linear(1152,4304,bias=True) -> gelu_pytorch_tanh -> Linear(4304,1152,
bias=True)`. Not SwiGLU like the text side's MLP -- plain GELU-MLP, both linears carry bias.

### 5. Merger (`Qwen3_5VisionPatchMerger`) -- image tokens -> text hidden size

```
x = LayerNorm(1152)(h).view(-1, 1152*2*2=4608)    -- relies on preprocessing's block-major patch
                                                       order (step 1) so this view is a pure reshape,
                                                       no permute
x = Linear(4608,4608,bias=True)(x)
x = GELU(x)                                        -- plain GELU here, not gelu_pytorch_tanh
x = Linear(4608, 5120, bias=True)(x)                -- 5120 == text hidden_size
```

`num_merged_tokens = num_patches / 4` (784 patches -> 196 merged tokens for the 28x28-patch square
test image, 988 -> 247 for the 26x38 non-square one). These `[5120]`-dim rows are exactly what gets
spliced into the text token sequence at the image's placeholder positions -- still unwritten, see
below.

## Text-side splicing (`Qwen3_5Model.get_rope_index`)

Read directly (not paraphrased) from `modeling_qwen3_5.py` (~line 1410-1469): the text sequence's
mrope `position_ids` (`[3, batch, seq_len]`, sections `[11,11,10]` per docs/architecture.md) are
built by walking the token sequence's `mm_token_type_ids` (0=text, 1=image, 2=video) in contiguous
runs:

- A **text run** of length `L` starting at `current_pos` gets `position_ids[:, start:start+L] =
  current_pos + arange(L)` broadcast identically across all 3 (t,h,w) rows (exactly
  `layer_golden.py`'s `mrope_position_ids` fallback for pure text).
- An **image run** gets `Qwen3_5Model.get_vision_position_ids(current_pos, grid_thw, 1,
  spatial_merge_size)` -- **a different function from the vision-tower-internal
  `vision_utils.get_vision_position_ids`** used in step 3 above (that one returns 2-axis (h,w) pairs
  for the vision rope; this one returns 3-axis (t,h,w) triples offset by `current_pos`, one triple
  per **merged** token, i.e. 196 triples for this golden's test image, not 784) -- the image's
  merged tokens occupy `t=current_pos` (constant) while `h`/`w` range over the merged
  `(h_patches/2, w_patches/2)` grid, each offset by `current_pos`.
- After an image run, `current_pos += max(grid_thw.h, grid_thw.w) // spatial_merge_size` (the
  larger spatial dimension's merged extent, NOT the token count) -- so the next text run's position
  resumes right after the image's spatial footprint in mrope-position-space, not after its 196-token
  footprint in sequence-space. This is the standard Qwen2-VL-family mrope advancement rule and is
  the single easiest place to get an off-by-one that silently produces plausible-looking garbage
  (the class of bug docs/validation.md rung 5 exists to catch).
- The decode side needs `mrope_position_deltas = position_ids.max() + 1 - seq_len`: decode step
  `i` after the prompt uses position `seq_len + i + delta` on all three rows. For a 209-token
  prompt containing one 28x28 image that delta is **-182** -- i.e. the first generated token sits
  at position 27, not 209. This is invisible to any check that only inspects the prompt's own
  position rows, and it is the value that decides whether the first token after an image lands
  anywhere sensible.

**Implemented**: `src/vision/position_ids.cpp`'s `BuildMropePositionIds` (plus
`MmTokenTypeIdsFromTokens`, which derives the type ids from the token ids the way the server path
will). `tools/reference/rope_index_golden.py` dumps the reference's own output for five cases --
text only, text + 1 image, text + 2 images with different grids, image at position 0, image last --
by calling the real `get_rope_index`/`get_vision_position_ids` bound to a config-only shim (no
weights, no GPU, under a second). `tests/vision/test_position_ids.cpp` compares every position of
every row and the delta: **0 mismatches** on all five cases.

### Splicing and the 3-axis rope through the whole decode stack (**implemented**, 2026-09-22)

`Model::PrefillMultimodal(tokens, images)` is the entry point. An `ImageSpan` is `{offset, tokens,
grid, embeds}` -- the offset of the placeholder run within THIS call's token vector, its length,
the image's PATCH grid (pre-merge, because that is what the advance rule reads) and a device
pointer at that image's rows of `EncodeImages`' output. The call:

1. **Validates the spans against the tokens.** Sorted, non-overlapping, in range, backed by real
   device rows, `grid.MergedTokenCount(merge) == tokens`, and every token under a span really is
   the container's own `image_token_id` (`Container::ImageTokenId()`, read from the TOP level of
   `model_config` -- it sits next to `text_config`/`vision_config`, not inside either). That last
   check is the one that earns its keep: the spans are the contract, and a caller whose offsets
   have drifted from its own rendered prompt would otherwise splice real embeddings over real
   text and produce fluent nonsense.
2. **Splices.** In `RunChunk`, right after the embedding gather, each span's intersection with the
   current chunk is one D2D `hipMemcpyAsync` -- both sides are contiguous `[rows, hidden]` bf16
   (the placeholder run is contiguous in the prompt, the merger's rows are contiguous in
   `EncodeImages`' output), so there is no kernel and no per-row loop. Chunked prefill needs no
   special case: a span that straddles a 64-token boundary simply intersects two chunks.
3. **Feeds 3-axis positions.** `BuildMropePositionIds` gained `(seq_start, mrope_start)` so it can
   walk a CONTINUATION block, and its returned `mrope_position_delta` is now stated as "what a
   token at absolute sequence index `s` adds to get its mrope position", which is the same number
   for a whole-prompt build and stays meaningful for every later token of the conversation.
4. **Records the delta.** `Model::MropeDelta()` / `MropeActive()`. From then on every later step --
   plain decode, MTP verify, MTP draft, DFlash2 injection and draft blocks -- ropes at `sequence
   index + delta` while its KV slot stays the sequence index.

#### Rope position vs KV slot: the split this stage introduces

Before this stage `Model::attn_positions_` was BOTH the rope position ids and the KV slot_mapping,
and model.h said so explicitly ("`positions[t] = pos_ + t` doubles as both"). With an image in the
prompt those two stop being the same number, in two different ways at once: image tokens carry
distinct `(t,h,w)`, and every token AFTER an image sits at `index + delta` (delta **negative**).

The split is deliberately one-sided. `attn_positions_` keeps its original meaning and its original
value -- it is the paged cache's slot index and the attention kernel's sequence index, both of
which must stay the plain running token count. Only the rope call changes: `AttentionLayer::
Forward` gained a trailing `const int32_t* rope_pos3` (device `int32[3, T]`, t row then h then w)
which, when non-null, routes to `r4dx_rope_partial_mrope3_bf16` instead of the single-row entry
point. `nullptr` is the default and is what every text-only caller passes, so a text-only run is
unchanged instruction for instruction, not merely numerically.

`Model::RopePositionsForChunk(start, T)` is the ONE helper every rope call site goes through. It
returns `nullptr` outright when `mrope_active_` is false -- which is what makes "text-only is
unchanged" structural rather than a claim -- and otherwise fills `[3, T]` from two sources: the
block table `PrefillMultimodal` is currently feeding (the only place the three axes genuinely
differ), or `s + mrope_delta_` on all three rows for everything else. There is no full-sequence
position table: `mrope_delta_` is by construction constant over every text run after the last
image, so a table would be 3 MiB of host memory restating one integer.

Every rope call site found and routed through it (`codegraph`/grep over `r4dx_rope_*` call sites in
`src/model`):

| call site | what changed |
|---|---|
| `Model::RunChunk` (prefill + plain decode) | uploads `[3, T]` once per chunk next to `attn_positions_`, passes it to every full-attention layer |
| `Model::VerifyWindow` (MTP and DFlash2 verify) | same, for the candidate rows -- always past the prompt, so all three rows are `pos_ + t + delta`, but they still have to be BUILT because `attn_positions_` is the slot mapping |
| `Model::DecodeStepProfiled` / `PrefillProfiled` | same, so `--profile` measures the path real decode takes |
| `MtpHead::Draft` | new `(mrope_active, mrope_delta)` pair; its own `positions_dev_` stays the sequence index (it is the MTP cache's slot mapping), a new step-major `rope3_dev_` carries the rope value. Step-major (`3*step+axis`, not `axis*max_draft+step`) because each step ropes exactly one token and the kernel's row stride is `tokens` == 1, so `rope3_dev_ + 3*step` is a valid compact `[3,1]` with no per-step upload |
| `MtpHead::PrimeKv` | new `rope3_host` (`int32[3, n]`). Unlike Draft's, THESE positions are inside the prompt and can land on image rows, so a scalar delta is not enough and `Model::RopePositionsHost` hands over the real rows |
| `DflashDraft::InjectFeatures` | new `rope_t_host` (`int32[rows]`) -- see below |
| `DflashDraft::DraftRound` | `pos_host_` (which feeds `r4dx_rope_neox_bf16` and nothing else) becomes `n_injected_ + t + rope_delta_` |

GDN layers have no rope at all, so `src/model/gdn_layer.cpp` is untouched.

One refactor came with it: `MakeAttnConfig` (`src/model/attn_config.h`) replaces five hand-written
copies of the same seven `AttnConfig` assignments (RunChunk, VerifyWindow, DecodeStepProfiled,
PrefillProfiled, `MtpHead`'s constructor). That duplication was survivable while the field list was
fixed; adding `mrope_section_{t,h,w}` to it made a forgotten site compile fine and silently rope
image tokens with the default split.

#### DFlash2: rope on the temporal axis, ring slots in sequence space

DFlash2's own M-RoPE sections are the degenerate `[64,0,0,0]` (docs/dflash2.md "RoPE"), i.e. every
frequency pair reads the TEMPORAL position and there is no h/w stream to carry. So the drafter's
rope positions are the mrope `t` row.

Its **ring slot** stays `(sequence position) % 2048`, and that is not a shortcut -- it is forced.
An image's merged tokens all share ONE temporal position (`t = current_pos`, constant across all
196 of them), so keying the ring on the rope position would collide ~196 rows onto one slot and
destroy the store. The SWA visibility rule (`query_pos - key_pos < sliding_window`) likewise stays
in sequence space, where "196 tokens ago" still means 196. This is the decision the task brief
flagged, resolved the way it suggested: ring slot = KV/sequence index, rope = mrope temporal
position. Nothing in the llama.cpp-derived reference contradicts it, and the reference's own cache
is position-keyed by the mrope position only because llama.cpp's KV cache is append-only rather
than a fixed-size physical ring.

`InjectFeatures` therefore takes an optional host `rope_t_host[rows]` and falls back to
`start_pos + t + RopeDelta()` when it is null. The fallback is exactly right for every injection
at or past the prompt's end (all text, one scalar describes them), and only `Model::RunChunk`'s own
injection -- which can straddle an image run -- passes the real rows.

#### What `Model::Reset()` has to clear, and why it is different from the caches

The KV caches, MTP's cache and DFlash2's ring all self-correct by position overwrite and need no
explicit reset (model.h's own `Reset()` comment argues this at length). The mrope state does not:
`mrope_delta_`, `mrope_active_` and the pending block describe the CONVERSATION, not bytes at a
position, and a stale delta would rope the next conversation's every token at the wrong position
while looking entirely healthy. `Reset()` clears all of them and puts `DflashDraft::SetRopeDelta`
back to 0 in the same breath.

#### Prefix reuse across a turn that contained an image

Every image placeholder is the SAME token id. Two requests carrying two COMPLETELY DIFFERENT
pictures at the same position therefore produce **byte-identical token sequences**, so
`PrefixState::Extend`'s token comparison stopped being sufficient the moment images existed:
reusing the prefix would keep turn 1's image in the KV cache while the client believes it sent a
new one, with nothing downstream able to detect it.

`PrefixState` now carries a per-image `ImageKey{content_hash, grid_thw, token_offset}` list
alongside `fed_`. A prefix is reusable only when this request's images START with exactly the ones
already fed; a new image whose offset lands inside the already-fed prefix is caller-bookkeeping
disagreement and is refused rather than guessed at. `Clear()`/`Invalidate()` drop the list with the
tokens. A caller with no images passes an empty vector and gets the original behaviour unchanged.
`tests/server/test_prefix_state.cpp`'s `TestImageAwarePrefixReuse` covers all six cases, including
an explicit assertion that the two turns' tokens really do match in the different-picture case.

Model-side, the delta simply survives: turn 2 re-prefills only the new tail through
`PrefillMultimodal(tail, {})`, whose positions continue at `index + delta` because `mrope_delta_`
is Model state that only `Reset()` clears. Measured end to end (`tool_vision_chat --turn2`): turn 1
"How many black circles are in this image?" -> `5`, then 331 of 356 tokens reused and 25 fed, turn
2 "Now multiply that number by three and give only the result." -> `15`, with `delta=-280` on both
turns.

Reuse itself is best-effort, and for a reason that has nothing to do with images: the server
re-tokenizes the client's own replayed `assistant` text, and this tokenizer's decode->encode round
trip is not the identity for every string (measured: a long Japanese or Thai answer does not
survive it, while ASCII, em/en dashes, emoji, Korean and a short Japanese sentence do). A miss
degrades gracefully to a full re-prefill and a full re-encode. docs/server.md's "Prefix cache,
image-aware" carries the measurements and the smoke case that pins it.

## Load policy (`--vision {auto|on|off}`)

The 333 `vision.*` tensors are **0.9154 GiB of VRAM, measured** -- so loading them
unconditionally would tax every text-only run. `ModelOptions::vision` is therefore a three-way
choice, exposed identically on `r4dx-cli` and `r4dx-server`:

| value | behaviour |
|---|---|
| `auto` (default) | load the tower **iff the container carries `vision.*` tensors**. A text-only container (the 4-layer test container, any `--language-model-only` convert) loads exactly as it did before this milestone -- no warning, no extra byte. |
| `on` | load it, and **throw at startup** if the container has none, so a deployment that means to serve images finds out immediately rather than at the first request. |
| `off` | never load it, even from a vision-capable container. The escape hatch for a VRAM-constrained text-only run on the real container. |

`Container::Load` takes `load_vision` and defaults it to **false**; the `auto` policy lives one
level up in `Model::Load`, which is the only layer that knows whether the caller asked for vision
at all. The probe is one representative tensor (`vision.patch_embed.proj.weight`), the same way
`mtp.*` is probed by `mtp.norm` rather than by a metadata field -- so it stays correct against a
hand-built or metadata-stripped container.

**Measured VRAM delta** (real `qwen38-27b-v3.r4dx`, `--layout w4a16`, `--max-ctx 2048`, HIP device
1, `Model::Load`'s own breakdown line):

```
--vision off : weights=15.5076 GiB, kv+gdn_state=0.40625 GiB, arena+scratch=0.09375 GiB, free=15.6918 GiB
--vision on  : weights=16.4230 GiB, kv+gdn_state=0.34375 GiB, arena+scratch=0.10620 GiB, free=14.8264 GiB
             + vision tower VRAM: 0.858177 GiB (already included in weights above)
```

i.e. **0.9154 GiB** of real driver-reported delta against **0.8582 GiB** of tensor bytes. The
0.0572 GiB difference is hipMalloc's own per-allocation rounding across 333 separate allocations
(the tower has a lot of small tensors -- 4 per block are `[1152]` or `[4304]` bias/norm vectors);
packing them into one arena would reclaim it and is the obvious follow-up if that 57 MiB ever
matters.

**Text-only output is unchanged**, checked rather than asserted: the same greedy prompt
(`--layout w4a16 --temperature 0 --max-tokens 48`) produces the **identical SHA-256** under
`--vision off`, `--vision auto` and `--vision on` on the real container
(`9C7D3E676CE50CCBD9385DDC982D6F0EDCA7637044594076D6E75EF3859D421D` for all three). Nothing in the
text path reads the tower: `Model` holds it in an `std::optional` that only `EncodeImages` touches.

## The device kernels (`src/kernels`, five new entry points)

Every linear in the tower is a plain bf16 GEMM (`r4d_gemm_bf16_nt_m64`) and its attention is
`r4d_attn_vit_h72_bf16`, both of which already existed. What did not exist is the norm, the two
activations, the axial rope and the learned-grid gather:

| entry point | what it is, and why it could not reuse something | validated by |
|---|---|---|
| `r4dx_layernorm_bf16` | A real `nn.LayerNorm`: mean-subtracted, biased variance, weight **and** bias. `r4dx_rmsnorm_bf16` is wrong for it in three separate ways (no mean subtraction, no bias, and a `1 + weight` convention on top). Two reduction passes, not the `E[x^2] - E[x]^2` identity: by block 26 the residual stream's mean is large relative to its variance and the identity cancels catastrophically in fp32. | CPU fp64 reference at 4 shapes (`max_rel` 3.77e-3 .. 3.89e-3, i.e. one bf16 ulp) + a shift-invariance check that an RMSNorm would fail by **150x** |
| `r4dx_bias_add_bf16` | The `+ b` half of an `nn.Linear` whose matmul half the GEMM did. | 4 shapes in place, `max_rel` 3.89e-3 |
| `r4dx_gelu_tanh_bf16` | `gelu_pytorch_tanh`, the encoder MLP's activation (`vision_config.hidden_act`). | fp64 reference, `max_rel` 3.86e-3 |
| `r4dx_gelu_erf_bf16` | The exact erf GELU, the **merger's** activation (`nn.GELU()`, `approximate='none'`). The two are separate entry points, not one with a flag, because they agree to 4.7e-4 -- inside every downstream tolerance -- so calling the wrong one is otherwise invisible. The test asserts they really do differ. | fp64 reference, `max_rel` 3.81e-3 |
| `r4dx_vision_qkv_rope_bf16` | Splits the fused `[tokens, 3456]` qkv into the three contiguous `[tokens, 16, 72]` tensors `r4d_attn_vit_h72_bf16` wants, and rotates q/k by the axial rope in fp32 over the **full** 72-dim head (rotate-half, no `partial_rotary_factor`). One launch instead of a split + two ropes. | CPU reference, q/k `max_rel` 3.89e-3, v **bit-exact** (0/226944), plus an explicit check that >90% of the head's TOP half moved -- a rope that only rotated the first 18 or 36 dims would leave it untouched |
| `r4dx_vision_pos_embed_bf16` | The 4-tap gather + fp32 weighted sum + the `.to(bf16)`-then-add residual add. | fp32 sum `max_abs` **0.0** vs a CPU reference; the residual add 3.89e-3; out-of-range indices clamp to row 0 rather than faulting |

`tests/kernels/test_vision_kernels.cpp` is the CPU-reference test (always on, no golden data, no
container); `tests/vision/test_vision_tower.cpp` is the golden comparison.

Every number above is one bf16 ulp (`2^-8 = 3.906e-3` relative), i.e. these kernels are as exact as
a bf16 output allows.

**One deliberate, quantified precision difference from the reference**: `nn.Linear` adds its bias
inside the GEMM's fp32 accumulator and rounds to bf16 **once**; r4dx's GEMM writes bf16 and then
`r4dx_bias_add_bf16` rounds **again**. That is one extra bf16 rounding (~`2^-9` relative, RMS
~1.1e-3) per linear. Avoiding it would need an fp32-output GEMM, which `third_party/libr4d` does not
have; the measured cost is far below the band the section below establishes, so it was priced
rather than engineered around.

## GEMM tuning: why `src/model/linear.cpp`'s table does not apply here

`PickTuning`/`FallbackTuning` in `src/model/linear.cpp` require `K % 512 == 0`, which every text-side
K satisfies (5120 / 17408 / 6144). The tower's do not: `patch_dim = 1536` is fine, `hidden = 1152`
and `merger_hidden = 4608` are fine, but **`intermediate_size = 4304` is `16 * 269` with 269 prime**,
so `mlp.linear_fc2`'s K admits `SK = 1` and nothing else. `src/vision/vision_tower.cpp` therefore
picks its own (`WV`, `SK`, `MB`): `SK` = the largest of {4, 2, 1} that divides `K/16`, then

```
                 448      1024     1536     (square image side, encode ms, tower only, best of 3)
WV=4  MB=1      28.1     151.2    399.8     <- shipped
WV=2  MB=1      29.0     156.4    409.7
WV=8  MB=1      30.3     161.6    421.0
WV=4  MB=4      35.0     183.3    464.1
```

(That sweep was run before the scratch-sizing fix below, so its absolute numbers are ~1% above the
final table's; the ranking is what it was measuring and is unaffected.)

`MB > 1` loses because it amortises the weight fragment over more row tiles at the cost of `grid.y`
blocks, and at `M <= 64` (`mtiles <= 4`) this tower is block-starved long before it is
weight-bandwidth-bound -- `N = 1152` is only 72 column tiles, so `grid.x` is 18 at `WV = 4`.

Every linear in the tower -- patch embed `[1536 -> 1152]`, the fused qkv `[1152 -> 3456]`, proj
`[1152 -> 1152]`, mlp fc1 `[1152 -> 4304]` and fc2 `[4304 -> 1152]`, merger fc1 `[4608 -> 4608]`
and fc2 `[4608 -> 5120]` -- goes through `r4d_gemm_bf16_nt_m64`, the same entry point
`src/model/linear.cpp`'s `ApplyLinear` and `DflashDraft` use for their bf16 weights, chunked to 64
rows for the same reason (that is the kernel's own `M` cap). `r4d_gemm_bf16_nt_m16` is the other
bf16 entry point in the family and is not used: it keeps `M` scalar accumulators per output column
and cannot run at all above `M = 16`, so it would triple the launch count at these shapes for no
benefit.

Non-multiple-of-64 row counts work and are exercised by every case: 784 patches is 12 full 64-row
GEMM launches plus one of 16, 988 is 15 + 28, and the merger's 196/247/443 merged rows are all
partial too. `r4d_gemm_bf16_nt_m64` clamps fragment rows past `M` rather than masking them, so a
partial chunk re-reads the last valid row and stores nothing for it. `N` never needs to be a
multiple of anything either -- the kernel masks the store with `n < N` -- though every `N` here
happens to be a multiple of 16 anyway (4304 = 269 x 16).

## Large images: chunking, scratch and the `--image-max-pixels` cap

The attention kernel needs every row of q/k/v at once, so those and the residual stream are sized
`[total_patches, 1152]`; everything else -- the patch-embed input, the fused qkv, the MLP's
`[rows, 4304]` and the merger's `[rows, 4608]` -- is computed in **1024-row chunks**
(`VisionTower::kDefaultRowChunk`, a multiple of 64 so it never splits a GEMM sub-chunk). That caps
the widest intermediate at ~8.8 MiB regardless of image size and is what keeps a 1536x1536 image at
148 MiB of scratch instead of ~310 MiB. Chunking changes buffer sizes only: every op between chunk
boundaries is row-independent, so a chunked encode and an unchunked one are bit-identical (which is
what lets `tests/vision/test_vision_tower.cpp` force `row_chunk = total_patches` when it needs a
traced intermediate whole).

**Measured** (`tests/vision/tool_vision_bench.exe`, HIP device 1, real container, best of 3 after a
discarded warm-up, `encode_ms` = H2D of `pixel_values` through the merger's last GEMM):

*Tower only (no text model loaded):*

| image | grid | patches | merged tokens | encode | scratch | VRAM used |
|---|---|---|---|---|---|---|
| 448x448 | 28x28 | 784 | 196 | **27.7 ms** | 26.9 MiB | 0.904 GiB |
| 1024x1024 | 64x64 | 4096 | 1024 | **149.5 ms** | 77.4 MiB | 0.963 GiB |
| 1536x1536 | 96x96 | 9216 | 2304 | **396.7 ms** | 147.8 MiB | 1.049 GiB |
| 2048x2048 | 128x128 | 16384 | 4096 | **858.8 ms** | 246.5 MiB | 1.194 GiB |

*Next to the loaded 27B model (`--layout w4a16`, `--max-ctx 262144` -- the shipped default, so the
KV cache is at its full 262144-token allocation), which is the question that actually matters:*

| image | encode | scratch | free VRAM after |
|---|---|---|---|
| 448x448 | 28.0 ms | 26.9 MiB | 6.850 GiB |
| 1024x1024 | 151.3 ms | 77.4 MiB | 6.791 GiB |
| 1536x1536 | 399.7 ms | 147.8 MiB | 6.706 GiB |
| 2048x2048 | 861.5 ms | 246.5 MiB | **6.560 GiB** |

So 1536x1536 encodes without OOM next to the full model at the full KV allocation with 6.7 GiB
still free, and so does 2048x2048. The encode cost is unchanged by the model being resident, as
expected -- nothing is shared. (Earlier measurements in this milestone's own history quote ~22%
larger scratch: an earlier revision reserved the trace-only fp32 `pos_embeds` buffer
unconditionally. `EnsureScratch` now reserves it only under a trace, which is where it is used.)

Where the time goes: at 448 the four GEMMs per block dominate (0.65 TFLOP of GEMM against 0.08 TFLOP
of attention, so ~23 TFLOP/s effective on the skinny `M<=64` path); at 1536 attention is quadratic
and takes over (10.6 TFLOP of attention against 7.6 TFLOP of GEMM, ~45 TFLOP/s combined). The
obvious lever for the small-image case is the same one docs/status.md's R10/P9 already names for the
text side: a prefill GEMM that is not capped at 64 rows.

### `--image-max-pixels`

Both binaries take `--image-max-pixels N` (`r4dx::vision::MakeImageProcessorConfig`), which
substitutes `N` for the checkpoint's own `preprocessor_config.json` ceiling. **It downsizes, it does
not reject**: `smart_resize` already contains the "too many pixels" branch (scale both sides by
`sqrt(h*w / max_pixels)`, floor each to a multiple of 32), so a large attachment still answers, just
at a coarser patch grid. `0` means "the checkpoint's own 16777216-pixel ceiling", and a value below
`min_pixels` clamps up to the floor rather than fighting the floor branch.

**Default: 1048576** (1024x1024 -> a 64x64 patch grid -> 1024 merged tokens). Chosen from the table
above, not rounded to a nice number: the tower's attention is quadratic in the patch count, so
going from 1024 to 1536 costs 2.6x the encode time and from 1536 to 2048 another 2.2x, while the
marginal detail does not scale that way; 151 ms is a defensible per-image cost on a serving path
where a 128-token completion takes ~3 s, and 862 ms is not. Raise it explicitly for an OCR-type
workload -- the scratch grows linearly and 2048x2048 still fits next to the loaded 27B with 6.5 GiB
free.

`tests/vision/test_preprocess.cpp`'s `CheckImageMaxPixelsCap` pins the behaviour end to end: a
2048x2048 image is a 128x128 grid (16384 patches) uncapped and a 64x64 grid (4096 patches) under the
default cap, with a real `pixel_values` tensor of exactly the capped size -- downsized, not rejected.

## Why the deep-block disagreement is not an r4dx error

Measured against the committed golden, the tower's agreement DEGRADES with depth: `rel_l2` is
4.4e-3 at `block_00_output` and 9.4e-2 at `block_26_output` on the square image. That is above the
1e-2 band this milestone set out to hold, so it was chased rather than waved away, in two steps.

**Step 1 -- per-block localization.** `VisionTower::Encode` takes a `VisionPreBlock` hook that
overwrites the residual stream before each block, so every block can be run from the REFERENCE's own
input and its OWN error measured with no inherited error in it. Result (square image, all 27
blocks):

```
block 00 own rel_l2=4.44e-03    block 09 own rel_l2=2.32e-03    block 18 own rel_l2=2.09e-03
block 01 own rel_l2=7.93e-03    block 10 own rel_l2=1.87e-03    block 19 own rel_l2=2.48e-03
block 02 own rel_l2=1.18e-02    block 11 own rel_l2=1.93e-03    block 20 own rel_l2=2.41e-03
...                             ...                              block 26 own rel_l2=3.73e-03
```

Uniform, 1.9e-3 .. 1.2e-2, with **no outlier block**. A wrong constant, a wrong activation, a
transposed index or a mis-scoped attention would show up as one block an order of magnitude worse
than its neighbours; nothing does. (The early blocks read slightly worse only because the relative
L2's denominator is small there: blocks 0-2 peak at |8|-|14|, while from block 9 on the residual
stream carries "massive activation" channels of |458| and up that pass through unchanged and
dominate the norm.)

**Step 2 -- what the reference itself costs.** `transformers`' `eager_attention_forward` computes
`torch.matmul(q, k^T)` in **bf16**, so the attention scores are rounded to bf16 before the
`* scaling`, rounded again after it, and the softmax probabilities are rounded to bf16 again before
the `P @ V` matmul. Both SDPA and `r4d_attn_vit_h72_bf16` keep all three in fp32/f16 -- r4dx is
strictly MORE accurate than the golden it is measured against. `vision_golden.py` gained
`--attn-impl {eager,sdpa}` (default `eager`, so every committed golden is unchanged) so that cost
could be measured instead of asserted. Regenerating the square case with SDPA into a scratch
directory and diffing:

| tensor | ref eager vs ref **sdpa** | r4dx vs ref eager | r4dx vs ref sdpa |
|---|---|---|---|
| `block_00_output` | 2.84e-3 | 4.44e-3 | 4.33e-3 |
| `block_08_output` | 2.67e-2 | 3.06e-2 | 2.30e-2 |
| `block_18_output` | 3.67e-2 | 4.02e-2 | 3.07e-2 |
| `block_26_output` | 7.75e-2 | 9.36e-2 | 5.64e-2 |
| `merger_output` | **6.74e-2** | **6.73e-2** | **5.35e-2** |

The reference's own two attention implementations disagree with each other by essentially exactly
as much as r4dx disagrees with either, and r4dx is **closer to the fp32-accumulating one**, which is
what its own kernel is. The disagreement is the tower's bf16 conditioning -- 27 blocks each
injecting ~2e-3 of legitimate rounding into a residual stream whose last blocks amplify it ~2x
apiece -- not an r4dx defect. A tighter tolerance at depth would not measure correctness; it would
measure which bf16 attention kernel the golden happened to be generated with.

**Tolerance policy, from that evidence** (`tests/vision/test_vision_tower.cpp`):

- fp32 index tables (`pos_embeds`, `rope_cos`/`rope_sin`): 1e-6 relative / 1e-5 absolute. Measured:
  `pos_embeds` exact, cos/sin 1.8e-6 .. 1.9e-6 absolute.
- the front end, block 0's whole internal chain and `block_00_output`: **1e-2**. Nothing has
  amplified yet, so a real bug cannot hide under bf16 noise; measured worst 4.4e-3.
- blocks 1..26, `last_hidden_state`, `merger_output`: **0.12**, from the table above.
- the real depth regression detector is the per-block localization pass, bounded at **1.5e-2**
  (measured worst 1.18e-2 vs eager, 7.1e-3 vs sdpa).

To reproduce the sdpa column:

```
python tools/reference/vision_golden.py --device cuda --attn-impl sdpa --cases vision_tower --out-dir <scratch>
$env:R4DX_VISION_GOLDEN_DIR = '<scratch>'; build\win-hip\tests\vision\test_vision_tower.exe
```

## Measured, per tensor, against the committed (eager) golden

Square 448x448 image, 784 patches, 196 merged tokens -- the full front end and block-0 chain:

```
patch_embed_out        rel_l2=2.26e-05   max_abs=3.91e-03  (1 bf16 ulp)
pos_embeds             rel_l2=0.00e+00   max_abs=0.00e+00  (exact, fp32)
block_input            rel_l2=2.43e-05   max_abs=3.91e-03  (1 ulp)
rope_cos / rope_sin    max_abs=1.85e-06 / 1.70e-06
block0_norm1_out       rel_l2=6.14e-05   max_abs=1.56e-02  (2 ulp)
block0_attn_qkv_raw    rel_l2=2.34e-03   max_abs=3.13e-02  (1 ulp)
attn_q_post_rope       rel_l2=2.37e-03   attn_k_post_rope  rel_l2=2.90e-03
block0_attn_proj_out   rel_l2=2.30e-03   <- the attention kernel did NOT amplify: same order as qkv
block0_norm2_out       rel_l2=4.80e-03
block0_mlp_fc1_out     rel_l2=2.32e-03   block0_mlp_fc2_out  rel_l2=3.29e-03
block_00_output        rel_l2=4.44e-03
block_26_output        rel_l2=9.36e-02   merger_output  rel_l2=6.73e-02   (see the section above)
```

Non-square 613x409 -> 608x416 (988 patches, 247 merged tokens) and the two-image batch (1772
patches, 2 `cu_seqlens` segments, 443 merged tokens) carry the front end and the merger only, since
the golden does not dump their per-block outputs:

```
non-square    patch_embed_out 2.83e-05   pos_embeds 0.0   last_hidden_state 5.59e-02   merger_output 5.67e-02
two images    patch_embed_out 2.60e-05   pos_embeds 0.0   last_hidden_state 6.93e-02   merger_output 5.41e-02
```

The two-image case is the only one that exercises per-image attention scoping (`num_segments = 2`,
`max_seqlen = 988`); it lands in the same band as the single-image cases, which is what it would not
do if `cu_seqlens` were being ignored or mis-ordered.

## The host-side module (`src/vision/**`)

CPU-only and dependency-light on purpose: no HIP, no `r4dx_core`, no `r4dx_model`, so
`tests/vision/**` are ordinary always-on unit tests (no GPU, no container) and a future
server/CLI path can preprocess an attachment without a device context. The only third-party
dependency is the vendored `third_party/stb/stb_image.h`.

| file | contents |
|---|---|
| `image_decode.{h,cpp}` | `DecodeImageBytes`/`DecodeImageFile` -> 8-bit interleaved RGB. The single TU that instantiates `stb_image` (`STBI_ONLY_{JPEG,PNG,BMP,GIF}`; WEBP has no stb decoder and is rejected with stb's own message). |
| `preprocess.{h,cpp}` | `ImageProcessorConfig` (this checkpoint's `preprocessor_config.json` as defaults), `SmartResize`, `ResizeU8` (the uint8 antialias resampler above), `PreprocessImages` -> `pixel_values` + `grid_thw`. |
| `position_ids.{h,cpp}` | `BuildMropePositionIds` (text-side `get_rope_index`), `MmTokenTypeIdsFromTokens`. |
| `vision_index.{h,cpp}` | `BuildPosEmbedInterpolation`, `BuildVisionRopePositionIds`, `BuildCuSeqlens`, `BuildVisionRopeCosSin`. |

The DEVICE half is a second target, `r4dx_vision_tower`, deliberately separate so the CPU-only
property above survives -- `tests/vision`'s three preprocessing/index tests still link only
`r4dx_vision` and still run on a machine with no GPU:

| file | contents |
|---|---|
| `vision_weights.{h,cpp}` | `VisionConfig` (from the container's `model_config.vision_config`), `VisionLinear`/`VisionLayerNorm`/`VisionBlockWeights`/`VisionWeights`, `HasVisionTensors`, `LoadVisionWeights`. Validates every tensor's element count against the config and rejects a head_dim other than 72 at load time (`r4d_attn_vit_h72_bf16` is compiled for 72 only) rather than at the first launch. |
| `vision_tower.{h,cpp}` | `VisionTower::Encode` -- the whole forward, the GEMM tuning, the row chunking, `PlanScratchBytes`, plus the two diagnostic hooks (`VisionTrace`, `VisionPreBlock`) the golden test drives. Holds the stream and the arena only, NOT the weights: a `Model` is moved at least once on its way out of `Model::Load`, which would leave a cached `const VisionWeights*` pointing at the moved-from container, so the weights are passed per call. |

`Container` owns the loaded weights (`Container::Vision()`); `Model` owns the tower and exposes
`HasVision()` / `EncodeImages()`. The dependency edge runs `src/model -> src/vision`, never back.

Video (`mm_token_type_id == 2`) is **rejected with an exception** rather than half-implemented:
r4dx has no video path and there is no golden here to test the reference's t-splitting rule
against. `GridThw::t` is carried through the index math (frames repeat the h/w indices, and
`cu_seqlens` gets one segment per frame) but every golden uses `t == 1`.

## Splicing pass: what was measured (2026-09-22)

Every number below is from a real run on HIP device 1 against the real
`qwen38-27b-v3.r4dx` container at `--layout w4a16`, unless it says otherwise.

### The kernel (`tests/kernels/test_rope_mrope3`)

`r4dx_rope_partial_mrope3_bf16` against a CPU fp32 reference that replays
`recomposition_frequencies`' two python slice assignments LITERALLY rather than re-deriving the
predicate the kernel itself uses:

```
text + 4x5 image + text          sections=[11,11,10] max_rel  q=6.0976e-03 k=0.0000e+00 tail=ok
long-context, axes far apart     sections=[11,11,10] norm_rel q=3.5213e-04 k=4.1258e-04 tail=ok
non-default sections [16,10,6]   sections=[16,10,6]  max_rel  q=7.1942e-03 k=5.8824e-03 tail=ok
[16,10,6]  axis table differs from bin%3 on 5/32 bins
[11,11,10] axis table differs from bin%3 on 0/32 bins
identical-rows vs single-row kernel: q 0/178176 differ, k 0/29696 differ
sections that do not sum to rotary_dim/2 throw: yes
```

Two of those lines are the point of the test rather than decoration. The `[16,10,6]` case exists
because this model's own `[11,11,10]` makes the reference's slice assignment **coincide exactly**
with `bin % 3` (0/32 bins differ) -- so a kernel that implemented the easy version would pass every
realistic case and still be wrong; `[16,10,6]` differs on 5 bins and catches it. And the
identical-rows case is the whole basis of the "text-only is unchanged" argument: with three equal
position rows the multimodal kernel reproduces the single-row one **bit for bit** (0 of 207,872
elements differ), so the choice of entry point is a performance decision with no numeric content.

The long-context case uses a norm-relative metric at 5e-2, for the same reason
`tests/kernels/test_rope.cpp`'s own long-context case does (near position 262143 `sincosf`'s
accuracy, not the stream selection, dominates a per-element maximum).

### Layer level (`tests/model/attention/test_mrope_attn_layer`)

A full bf16 27B reference does not fit on this card, so one decoder layer is the largest piece of
the real stack that can be compared numerically at all.
`tools/reference/mrope_layer_golden.py` runs layer 3 -- real weights, real
`Qwen3_5DecoderLayer`, eager attention -- over a 63-token prompt containing a 10x16-patch image
(40 merged tokens), with position ids from the unmodified `Qwen3_5Model.get_rope_index`:

```
total=67 prefill=63 decode=4; 39 rows have distinct (t,h,w), 59 rows rope off their sequence index
prompt carries 40 image-placeholder rows
prefill  T=63 start_pos= 0 norm rel err=1.2235e-02 (PASS, bound 2e-2)
decode   T= 4 start_pos=63 norm rel err=1.5754e-02 (PASS, bound 2e-2)
control (rope at the KV slot index, no mrope): norm rel err=5.4774e-02
```

The first line is asserted, not printed for colour: if the golden's rows did NOT diverge across
axes and from the sequence index, a passing tolerance would prove nothing. The tolerance is the
repo's standard 2e-2 `bf16_matmul_rel_err`, identical to `test_attn_layer.cpp`'s and for the
identical reason (this path writes through the fp8 e4m3 paged KV cache; the golden ran bf16 K/V).

The control run is the same call with `rope_pos3 = nullptr`, i.e. roping at the KV slot index the
way the pre-vision path does. It lands at 5.48e-2, **4.5x** the correct run -- so the test can
actually distinguish a correct 3-axis rope from no mrope at all, which a tolerance number alone
never demonstrates. The gap is bounded rather than enormous because the prompt's first seven tokens
DO sit at their own sequence index and rope identically either way, and only 64 of each head's 256
dims rotate. The test asserts the control exceeds both 2x the absolute tolerance and 3x whatever
the correct run measured.

`--decode-len 4` continues the same KV cache at `sequence index + delta`, which is the decode-side
half of the contract.

### Position ids for the engine's REAL rendered prompt

`tests/vision/test_position_ids` already compared `BuildMropePositionIds` against the reference's
`get_rope_index` on five synthetic cases (0 mismatches). This pass adds two things it could not
cover:

- **Continuation splits.** For each of those five cases, every legal split point is built as two
  blocks -- the second with `(seq_start, mrope_start)` carried from the first's own delta, exactly
  as `PrefillMultimodal` does for turn 2 of a chat -- and must reproduce the whole-sequence rows
  and delta. Measured: **0 mismatches** over 16/13/19/11/13 splits respectively.
- **The engine's own rows, for a prompt that came out of the real chat template and tokenizer.**
  `tool_vision_chat --dump-prompt` writes what `Model::PrefillMultimodal` actually handed the rope
  kernel (`rope_rows_out`, not a re-derivation), and `rope_index_golden.py --verify-prompt` checks
  it against the unmodified reference. On a two-image prompt (a 30x40-patch chart and the 28x28
  synthetic image, 517 tokens after placeholder expansion):
  `all 1551 position ids match the reference exactly, delta=-462`.

  This is the check a synthetic fixture structurally cannot do: an off-by-one in the
  `<|image_pad|>` expansion or in the span offsets lives between the template and the model, not
  inside `BuildMropePositionIds`.

### End to end: does the model actually see the picture

`tests/vision/tool_vision_chat` (built, deliberately never `add_test()`'d -- same convention as
`tool_vision_bench`), greedy decoding throughout, real container. Prompts and answers verbatim:

| image | prompt | answer | verdict |
|---|---|---|---|
| `vision_test_image.png` (the golden's synthetic gradient + checkerboard + circle), 28x28 grid | "Describe this image in two sentences: the colour gradient, any repeating pattern, and any shape you see." | "The image features a smooth, multi-colored gradient background that transitions from blue and purple at the top to green and yellow at the bottom, overlaid with a subtle repeating checkerboard pattern. Centered in the composition is a large, semi-transparent circle that blends with the underlying gradient, creating a soft, glowing effect." | all three features correct |
| three bars: red 100px, green 250px, blue 350px on white with black axes, 30x40 grid | "This is a bar chart. How many bars are there, what colour is each one, and which bar is tallest? Answer briefly." | "There are 3 bars. The colors are red, green, and blue. The tallest bar is the blue one." | count, colours and ordering all correct |
| five black circles on white, 30x40 grid | "How many black circles are in this image? Answer with just the number." | "5" | correct |
| `R4DX 7391` rendered at 160pt on 1280x320, 20x80 grid | "What text is written in this image? Answer with the exact characters." | " R4DX7391" | exact (the space between words is not reproduced) |

The three non-synthetic images are rendered locally with `System.Drawing` from PowerShell (no
network), which is what makes them checkable: the expected answer is known by construction rather
than by a human judging a photograph.

**Resolution, not a positional bug.** The OCR case was run at three sizes and the failure mode is
monotone in the patch grid, which is what distinguishes "the model cannot resolve the glyphs" from
"the rows are spliced at the wrong positions" (the latter degrades into fluent nonsense, not into
one wrong digit):

| render | grid | merged tokens | answer |
|---|---|---|---|
| 640x320, 54pt, two lines | 20x40 | 200 | `HELLO` / `RDX 791` |
| 1280x640, 108pt, two lines | 40x80 | 800 | `HELLO R4DX 7301` |
| 1280x320, 160pt, one line | 20x80 | 400 | ` R4DX7391` (exact) |

### Speculative decode with an image in the prompt

Same image, same prompt, greedy, `--max-tokens 120`. The text baseline is the same question asked
without a picture, so the acceptance numbers are comparable:

| run | tokens/round | drafts accepted | decode tok/s | output |
|---|---|---|---|---|
| image, plain decode | -- | -- | 39.3 | (the description above) |
| image, `--mtp 3` | 2.91 | 41/66 (62.1%) | 86.0 | **identical to plain decode** |
| image, `--dflash k=7` | 3.14 | 44/147 (29.9%) | 88.3 | **identical to plain decode** |
| text-only, `--mtp 3` | 2.21 | 28/72 (38.9%) | 65.4 | -- |
| text-only, `--dflash k=7` | 2.52 | 31/147 (21.1%) | 72.4 | -- |

Acceptance with an image is not merely "in the same range" as text -- it is **higher** on both
families (2.91 vs 2.21 tokens/round for MTP, 3.14 vs 2.52 for DFlash2), which is the expected
direction: an image description is more predictable text than an open-ended explanation. Both
speculative outputs are byte-identical to plain decode on this prompt, so the documented
batched-verify numerics class did not trigger here.

The DFlash2 drafter needed no special handling for image rows beyond the position split: its
feature injection consumes the target's residual stream, which by that point already carries the
spliced embeddings, and its ring is keyed on the sequence position, which image rows advance
normally.

### Text-only is unchanged, checked against a build of commit `20cdee3`

Rather than compare against a SHA recorded in a doc, commit `20cdee3` (the last commit before this
milestone's vision work) was exported to a scratch tree, built, and run side by side with this
tree's binary on the same prompt (`--layout w4a16 --temperature 0 --max-tokens 64 --max-ctx 8192`,
real container):

| configuration | SHA-256 of raw stdout | identical |
|---|---|---|
| plain (`--mtp 0`) | `28EFFB5B41D7336556C5A08EF91A43E3DE0C56A0D395BBBD66B5C2ABC393381C` | **yes** |
| `--mtp 3` | `F590905B4CD01B56D92E20B2AD139B534B81735BD5E2D0EED03B11690F24ADA0` | **yes** |
| `--dflash <w4a16 draft> --dflash-k 7` | `28EFFB5B41D7336556C5A08EF91A43E3DE0C56A0D395BBBD66B5C2ABC393381C` | **yes** |

`tools/validate_dflash.ps1 -AllowBatchedVerifyDivergence` on this tree: **4 byte-identical, 5
accepted as the known batched-verify divergence** -- the same 4/5 split, on the same
(layout, prompt) cells, that docs/status.md's own table records as the pre-existing baseline
(w4a16 medium; w4a8 medium + long; mxfp4 medium + long). No cell moved.

## User-facing wiring: --image and image_url (2026-09-22, stage 5)

Everything above this section produces embeddings and splices them; this section is what a real
user or client actually types. Full request/response contract, error cases and limits: `docs/
server.md`'s "Images" section -- this section covers the design decisions and the real-hardware
numbers.

**One shared expansion routine.** `src/vision/image_prompt.h`'s `ExpandImagePlaceholders` is the
ONE place that turns a chat-template-rendered token sequence's single `<|image_pad|>` marker per
image into that image's real `grid.MergedTokenCount(merge_size)`-token run, deliberately kept
independent of `r4dx::model::Model` (`ImagePlaceholderSpan` mirrors `Model::ImageSpan`'s fields
rather than including model.h, so `src/vision` keeps its own "no r4dx_model" property) -- both
`r4dx-cli`'s `--image` path (`src/cli/main.cpp`) and the server's image content-part path
(`src/server/engine.cpp`) call it, rather than each re-deriving the same offset arithmetic.

**CLI (`src/cli/cli_args.h`/`main.cpp`).** `--image <path>` (repeatable) attaches to the next user
turn -- the one-shot `--prompt` itself, or (in `--chat`) whichever line is typed first; every later
`--chat` turn attaches images via one or more leading `"/image <path>"` REPL lines. `--stats` gains
an `[stats] image: ...` line (encode ms, then spliced image token count). Multi-turn state: one
`ImageBatch` (grid list + device embeddings) per turn that ever attached an image, kept alive for
the whole process -- a later turn's chat-template re-render still carries every earlier turn's own
`"image"` content part and has to re-expand its placeholder to keep the running `fed_tokens`
comparison correct, but must not re-encode it; only a span whose offset lands at or past the
already-fed prefix boundary is ever handed to `PrefillMultimodal`.

**Server (`src/server/openai_types.h`/`.cpp`, `engine.cpp`).** Every validation and the actual
image decode/preprocess (base64, `stb_image`, `smart_resize`) happen at PARSE time in
`openai_types.cpp` -- pure host C++, no HIP, no `Model` -- so `tests/server/test_openai_types.cpp`
exercises the whole accept/reject surface with no container and no GPU. `Engine::RunRequest` then
does only the two things that need a live `Model`: (a) refuse with `400` up front if any image
appeared but `Model::HasVision()` is false, and (b) after the prefix-reuse decision, call
`Model::EncodeImages` ONLY for the images whose placeholder span lands in the newly-fed tail --
`timings.image_n`/`image_ms` count exactly that, so a turn that reuses an earlier turn's own image
carries no `image_n` key at all. Prefix-cache image-awareness (`PrefixState::ImageKey`, built in
stage 4) is now fed from real request data: a 64-bit FNV-1a hash of each image's raw (post-base64,
pre-decode) bytes, computed once in `openai_types.cpp` and threaded through `Extend`/`Commit`.

**Real-hardware verification** (`tools/server/smoke.ps1 -Vision`, real container
`qwen38-27b-v3.r4dx`, `w4a16`, HIP device 1, synthetic PNGs generated in-script with
System.Drawing, no fixtures committed):

```
[PASS] vision: describe request returns 200
[PASS] vision: describe usage.prompt_tokens (86) reflects real spliced image tokens (>50)
[PASS] vision: describe timings.image_n == 1        [PASS] vision: describe timings.image_ms > 0
[PASS] vision: OCR response contains the rendered string 'R4DXVSN9' (got 'R4DXVSN9')
[PASS] vision: two-images timings.image_n == 2
[PASS] vision: image + tools returns 200            [PASS] vision: image + thinking produces reasoning_content
[PASS] vision: streaming image request produces non-empty streamed content
[PASS] vision multi-turn: turn 2 timings carries NO image_n (the image was NOT re-encoded)
[PASS] vision multi-turn: turn 2 timings.prompt_n (24) < usage.prompt_tokens (121) -- only the new tail was prefilled
[PASS] vision multi-turn (different image): prefix NOT reused -- timings.prompt_n (121) == usage.prompt_tokens (121)
[PASS] vision bad input: unsupported format (webp) / corrupt data / 9 images -- all 400
[PASS] vision: /v1/models architecture.input_modalities / capabilities contain 'image'
```

The OCR line is the one worth reading twice: greedy, real container, a freshly-rendered
`"R4DXVSN9"` PNG the script generates at runtime, and the model reads it back **exactly**. Every
check in the `-Vision` suite passed (see `docs/server.md`'s "Images" section for the full list);
the one thing worth recording for a future reader is that the FIRST version of the multi-turn-reuse
check used a full descriptive sentence as the turn-1 assistant reply and failed both of its own
assertions -- not an image-handling bug, but the same generation-round-trips-through-re-
tokenization risk any prefix-caching server has with client-resent text (docs/server.md's own
"reasoning_content" section notes the analogous risk for a replayed `<think>` block). Switching
that one check's prompt to elicit a short, simple yes/no answer removed the confound and the
image-specific mechanism it was built to test passed cleanly; this is a note about the SMOKE
SCRIPT's own prompt choice, not a limitation of `PrefixState::ImageKey` itself, whose text-free
identity comparison (content hash + grid + token offset) never depended on generation text at all.

**Default (non-`-Vision`) run, 4-layer text-only container**: a well-formed local image against a
container with no vision tower is a clean `400` naming the real reason --
`"this model/container has no vision tower (loaded without vision.* tensors, or started with
--vision off)"` -- verified on this same run, not merely asserted.

**Full `ctest`** after this stage's changes: **61 registered (up from 59: `TestImageFlag` in
`test_cli_args`, plus this stage's new `test_openai_types` cases run inside the existing binary,
not as new registered targets), 60 passed, 1 skipped (`test_kernel_bandwidth`, gitignored golden
absent), 0 failed, 639.10 s**, HIP device 1 -- identical pass/skip counts to every prior stage's
baseline plus the one new registered CLI test, no regression.

## What's done / what's not (2026-09-21, device-half pass)

**Done and measured this pass** (every number above is from a real run on HIP device 1; the
sections they live in carry the detail):

- **The tower runs on the GPU.** `src/vision/vision_weights.{h,cpp}` (the 333 `vision.*` tensors,
  0.8582 GiB, config-validated) and `src/vision/vision_tower.{h,cpp}` (patch embed -> pos embed ->
  27 blocks over `r4d_attn_vit_h72_bf16` -> merger), reached through `Model::EncodeImages`.
- **Five new device kernels**, each against a CPU reference at one bf16 ulp
  (`tests/kernels/test_vision_kernels.cpp`): `r4dx_layernorm_bf16`, `r4dx_bias_add_bf16`,
  `r4dx_gelu_tanh_bf16`, `r4dx_gelu_erf_bf16`, `r4dx_vision_qkv_rope_bf16`,
  `r4dx_vision_pos_embed_bf16`.
- **`r4d_attn_vit_h72_bf16` needed no submodule change.** It was already in
  `third_party/CMakeLists.txt`'s unit list and already compiled correctly for this Windows LLP64
  build; `block0_attn_proj_out` (downstream of it) matches at `rel_l2 = 2.30e-3`, the same order as
  its own input, i.e. the attention step amplifies nothing.
- **56 tensors compared across all three golden cases** (`tests/vision/test_vision_tower.cpp`),
  including every `block_00..26_output`, the whole block-0 internal chain, and the two-segment
  two-image case. Plus a per-block localization pass that runs every block from the reference's
  own input.
- **The deep-block disagreement was root-caused, not waved away**: it is the reference's own bf16
  attention scores. r4dx sits INSIDE the spread between transformers' eager and SDPA attention and
  is closer to the fp32-accumulating one. See "Why the deep-block disagreement is not an r4dx
  error"; `vision_golden.py` gained `--attn-impl {eager,sdpa}` and `--cases` so it is reproducible.
- **VRAM**: 0.9154 GiB measured delta (0.8582 GiB of tensors + hipMalloc rounding across 333
  allocations), behind `--vision {auto|on|off}`.
- **Perf and scratch**: 27.7 / 149.5 / 396.7 / 858.8 ms and 26.9 / 77.4 / 147.8 / 246.5 MiB for
  448 / 1024 / 1536 / 2048-pixel square images; 1536x1536 and 2048x2048 both encode next to the
  loaded 27B at the full 262144-token KV allocation with 6.5 GiB free.
- **`--image-max-pixels`** on both binaries, defaulting to 1048576 from those measurements, which
  downsizes through `smart_resize` rather than rejecting.
- **Text-only output is byte-identical** with the tower loaded, not loaded, or forced
  (`--vision off` / `auto` / `on` produce the same SHA-256).

**Full `ctest`, run three times end to end on this tree** (59 registered, up from 57 -- this pass
adds `test_vision_kernels` and `test_vision_tower`):

| run | result | note |
|---|---|---|
| 1 | **58 passed, 1 skipped (`test_kernel_bandwidth`, gitignored golden absent), 0 failed, 657.10 s** | before the BOM/scratch cleanups below |
| 2 | 57 passed, 1 skipped, **1 FAILED -- `test_mtp`, `Exit code 0xc0000409`**, 652.47 s | see below |
| 3 (final tree) | **58 passed, 1 skipped, 0 failed, 659.68 s** | |

Run 2's failure is reported rather than re-rolled away. It is the **same intermittent Windows
fail-fast (`STATUS_STACK_BUFFER_OVERRUN`) this milestone has already recorded twice** -- once in
`test_mtp` and once in `test_forward_smoke`, both on the host half's own passes, both passing on
the runs either side and 3/3 standalone afterwards (see "Earlier pass" below). Here it died 344.25 s
into `test_mtp`, after the second bf16 model's VRAM breakdown and a `verify-vs-sequential` line,
i.e. deep in the run and nowhere near load; the identical test passed at 352.93 s in run 1 and
360.85 s in run 3 on binaries either side of it. `test_mtp` links no vision code -- `r4dx_model`
does link `r4dx_vision_tower`, but the only vision work a `test_mtp` run performs is one
`SafetensorsReader::Has("vision.patch_embed.proj.weight")` probe at container load, which returns
false for the 4-layer test container and allocates nothing. This remains a pre-existing, low-rate
box-level fail-fast in long real-container GPU test runs -- now observed three times in two
different tests -- and it deserves a debugger rather than a re-run.

**Not done -- everything downstream of the merger:**  *(superseded by the 2026-09-22 splicing pass
below; kept for the record of what that pass inherited)*

- **No splicing.** Merged image rows are not written into the text embedding sequence at the
  `248056` placeholder positions, and `Model`/`Engine` do not thread `mrope_position_delta`
  through decode. `Model::EncodeImages` hands back the rows; nothing consumes them yet.
- ~~No `--image` CLI flag, no server `image_url` wiring~~ **DONE, 2026-09-22 (stage 5) -- see
  "User-facing wiring: --image and image_url" above.**
- ~~No end-to-end "describes a real image correctly" check (docs/validation.md rung 5)~~ **DONE,
  2026-09-22 -- four by-construction cases (OCR string, known-colour chart, countable objects, the
  tower golden's own synthetic image), all correct; see "End to end: does the model actually see
  the picture" above and docs/validation.md's "Rung 5 for images".** Still open: no non-synthetic
  photograph run through `vision_golden.py` -- every image used is synthetic or locally rendered.
- ~~No prefill-with-image tok/s measurement (there is no prefill-with-image path yet).~~ **DONE,
  2026-09-22 -- docs/perf.md's "Vision" section: encode 32.2/155.4/408.7 ms at 196/1024/2304 image
  tokens, and image tokens prefill at the same ~1000 tok/s text tokens do.**
- No video path, deliberately (see above).

**Known limitations of what shipped:**

1. **The 57 MiB of hipMalloc rounding** across 333 separate small allocations (0.9154 GiB measured
   vs 0.8582 GiB of tensor bytes). One arena for the whole tower would reclaim it.
2. **The bias is rounded to bf16 twice** (GEMM output, then `r4dx_bias_add_bf16`) where
   `nn.Linear` rounds once, because `third_party/libr4d` has no fp32-output GEMM. ~1 extra bf16
   ulp per linear, priced in the section above.
3. **The GEMM path is capped at 64 rows** (`r4d_gemm_bf16_nt_m64`), so a 448x448 image makes 13
   launches per linear x 4 linears x 27 blocks. That is the same R10/P9 prefill-GEMM gap
   docs/status.md already names for the text side, and it is what dominates small-image encode
   time.
4. **No fixed golden for an image with `t > 1`** (video). `GridThw::t` is carried through the index
   math and `cu_seqlens` gets one segment per frame, but nothing tests it.

**Recommended next steps, in order** *(as written on 2026-09-21; item 1 was done on 2026-09-22 --
see "Splicing pass: what was measured" -- and item 3 was done for synthetic and locally-rendered
images, though not yet for a real photograph)*: (1) splicing + mrope wiring into `Model`'s
prefill, using `BuildMropePositionIds`' output and its delta, with `Model::EncodeImages`' rows
written over the `248056` placeholder embeddings; (2) `--image` on `r4dx-cli` and `image_url` on
the server, both feeding the already-built `ImageProcessorConfig`; (3) the rung-5 end-to-end
description check on a real photograph; (4) prefill-with-image throughput.

## Earlier pass: what the host half established (2026-09-21, host-half pass)

**Done and measured (host half):**

- Read every architecture/preprocessing/mrope-splicing source above directly from the reference
  venv's `transformers` 5.17.0 -- `Qwen3_5VisionModel`, `Qwen3_5VisionAttention`,
  `Qwen3_5VisionPatchEmbed`, `Qwen3_5VisionPatchMerger`, `Qwen3_5VisionRotaryEmbedding`,
  `transformers.vision_utils.*`, `Qwen3_5Model.get_rope_index`, and (this pass)
  `Qwen2VLImageProcessor` + `TorchvisionBackend` + torch's own uint8 antialias kernel -- not
  inferred or guessed.
- Confirmed the real checkpoint carries all 333 `model.visual.*` weights and that its
  `vision_config` matches "Model facts" exactly.
- **Corrected three things this document and its golden previously got wrong about their own
  evidence**, by instrumenting `TorchvisionBackend.resize` during a real `AutoImageProcessor`
  call, by re-measuring each resampler deviation on all four noise pairs instead of one, and by
  re-reading `Qwen3_5VisionModel.forward`'s own call list: (a) `Qwen2VLImageProcessorFast` is a
  DEPRECATED ALIAS in `transformers` 5.17.0, not a class of its own -- touching the name warns
  that "the `Fast` suffix for image processors has been removed" and hands back
  `Qwen2VLImageProcessor`, which is also what `AutoImageProcessor` resolves that config string to
  and which inherits `TorchvisionBackend` directly (the first wording of this bullet said the name
  "does not exist", which the 2026-09-22 review pass disproved against the reference venv); (b) the cost of keeping a **float** intermediate between the two resampler passes was
  stated here as "up to 2/255 on 0.5% of pixels", which understated it by an order of magnitude in
  both axes -- it is **up to 25/255 on 11-23%** of bytes (the table under "Resampling" now carries
  the re-measured range for all three deviations); (c) `vision_golden.py` dumped `cu_seqlens` from
  `get_vision_cu_seqlens`, which is *not* the function the model calls -- the model calls
  `get_vision_attention_seqlens`. The two agree for this config, but the golden now takes the
  model's own path and asserts the equality, so the C++ is validated against the tensor the blocks
  actually receive rather than a lookalike.
- **Fixed a dangling-mmap bug in `tools/reference/common.py`** (previous pass; `ShardIndex`'s
  tensors aliased the `safe_open` mmap and outlived it). Retained here because every script in
  that directory depends on it.
- `tools/reference/vision_golden.py`: three cases (square / non-square-with-resize / two-image),
  real weights, real processor, regenerated in full on HIP device 1 this pass (`weights=
  safetensors missing_keys=0`). Adds `pixel_values`, `image_grid_thw`, `interp_indices`,
  `interp_weights`, `pos_embeds`, `vision_position_ids`, `cu_seqlens`, `patch_embed_out`,
  `rope_cos`, `rope_sin`, `block_input`, `pos_embed_table`, `rope_inv_freq` to the
  per-block/merger tensors it already produced, plus three CPU-only fixture groups that give the
  preprocessing claims their own tests: `make_channel_conversion_fixtures` (alpha-ramp RGBA +
  greyscale), `make_resampler_fixtures` (four random-noise resize pairs) and
  `make_format_fixtures` (BMP/GIF/JPEG).
- `tools/reference/rope_index_golden.py`: five text-side position-id cases from the real
  `get_rope_index`, no weights and no GPU (re-run this pass; 5/5 cases written in under a second).
- `src/vision/**` + `tests/vision/**` as described above. Numbers, copied from the real test
  binaries' own stdout, not from memory:
  - preprocessing: `max_abs = 0`, `mean_abs = 0`, `0 / 5,443,584` values differing, across all
    three cases (i.e. bit-exact, including the resized non-square image and the two-image
    concatenation);
  - text-side mrope position ids: 0 mismatches over all 3 rows of all 5 cases, deltas exact
    (`0 / -182 / -410 / -228 / -182`);
  - pos-embed interpolation indices and weights: exact; the gathered `pos_embeds`: `max_abs = 0`;
  - `cu_seqlens`, vision rope `(h,w)` ids: exact, including the two-segment case;
  - vision rope cos/sin: `max_abs` 1.9e-6 (fp32 `theta^(-2i/36)` vs double, times a position
    index up to ~40);
  - RGB conversion vs PIL's own `convert("RGB")`, on an alpha-ramp RGBA PNG and a greyscale PNG:
    0 differing bytes;
  - the resampler alone, against the reference's own `tvF.resize`, on four random-noise size
    pairs: **0 differing bytes out of 1,898,496**, covering downscale-both, the real
    `smart_resize` mixed pair, upscale-both and a 4x downscale;
  - BMP and GIF decode vs PIL: 0 differing bytes; JPEG: `max_abs = 2`, 435 of 6720 bytes (the one
    tolerance in the file, and it is a libjpeg-vs-stb IDCT difference, not a pipeline one).
- Full `ctest`, run twice end to end on this tree. **Run 1: 57 registered, 56 passed, 1 skipped
  (`test_kernel_bandwidth`, gitignored golden absent), 0 failed, 639.55 s.** **Run 2: 56 passed,
  1 skipped, 1 FAILED -- `test_forward_smoke`, `Exit code 0xc0000409`, 611.14 s.** Both runs'
  three vision tests passed (0.10 s / 0.01 s / 0.03 s), and so did every other pre-existing test
  in both.

  That failure is reported rather than re-rolled away, so here is everything known about it. It is
  the **same intermittent Windows fail-fast (`STATUS_STACK_BUFFER_OVERRUN`) this milestone has now
  seen in two different long real-container GPU tests**: `test_mtp` died with it once on an
  earlier full-suite run and passed on the runs either side (345 s, 344 s, then 343 s in run 2),
  and `test_forward_smoke` passed run 1 (43.41 s) and died 19.23 s into run 2, part-way through
  its third `bf16` model instance -- after that instance's VRAM breakdown printed, so during load
  or forward, not at teardown. Re-run on its own immediately afterwards it passed **3 times out of
  3** (48.18 s, 40.52 s, 39.59 s).

  It is not a regression from this pass, and that is checkable rather than asserted:
  `r4dx_vision` is a leaf static library that **nothing but `tests/vision/**` links** (`git grep
  r4dx_vision` finds only its own `src/vision/CMakeLists.txt` and `tests/vision/CMakeLists.txt`),
  and this pass changed no file under `src/core`, `src/kernels`, `src/model`, `src/tokenizer`,
  `src/convert` or `third_party` -- so the binary `test_forward_smoke` runs is byte-identical to
  the one that has always run it. What this pass did change is how *long* a full suite spends with
  a GPU context alive, by adding three (sub-second, CPU-only) tests -- which is not a plausible
  cause either. The honest summary is that this box has a pre-existing, low-rate fail-fast in
  long real-container GPU test runs that has now been observed twice, in two different tests, and
  that a later milestone should chase with a debugger rather than a re-run.

