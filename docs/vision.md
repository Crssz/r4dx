# Vision tower

Status as of 2026-09-20: **investigation + golden reference done; no C++ implemented yet.** This
document is the design contract for whoever implements `src/model`'s vision tower next -- it exists
so that pass can start writing kernels/layer code against a precise, measured spec instead of
re-deriving it from `transformers` source. See "What's done / what's not" at the end for the exact
handoff state.

## Why this is its own document, not a section of docs/architecture.md

The vision tower is architecturally unrelated to the GDN/attention decoder stack
docs/architecture.md describes (different attention kernel, different norm, different rope, a
learned+interpolated position embedding with no text-side analogue, a completely different
preprocessing pipeline). It gets folded into docs/architecture.md once real code exists to
document; until then this file is the spec, not a description of shipped code.

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

### 1. Preprocessing (host-side, this is `stb_image`'s job, task item 2)

Real checkpoint's `preprocessor_config.json`: `image_processor_type: Qwen2VLImageProcessorFast`,
`size: {shortest_edge: 65536, longest_edge: 16777216}` (in **pixels**, i.e. `h*w` bounds, not a
side length), `patch_size: 16`, `temporal_patch_size: 2`, `merge_size: 2`, `image_mean: [0.5,0.5,
0.5]`, `image_std: [0.5,0.5,0.5]` (i.e. normalize to roughly `[-1, 1]`, not ImageNet stats). Qwen2-VL
family algorithm (read from `transformers.models.qwen2_vl.image_processing_qwen2_vl`, not
re-derived): resize so both height and width are a multiple of `patch_size * merge_size = 32`,
choosing the largest such resize whose total pixel count stays within `[shortest_edge,
longest_edge]` (a single real image is nowhere near the 16M-pixel ceiling in practice; the floor
matters more for tiny images), bilinear resize a rescale to `[0,1]`, normalize with mean/std 0.5,
then patchify: reshape into `(grid_t=1 for a static image (each frame duplicated to fill
temporal_patch_size=2), grid_h/patch_size, grid_w/patch_size, in_channels, temporal_patch_size,
patch_size, patch_size)` and flatten to `[num_patches, in_channels*temporal_patch_size*patch_size^2
= 1536]` **already reordered into 2x2 spatial-merge-block-major order** (patch `[br, in_row]` where
row = block_row*merge + in_row, NOT raster h/w order) -- this reordering is what lets the merger
later do a plain `.view(-1, hidden_size*merge^2)` with no explicit permute (see "4. Merger" below).
**r4dx's stb_image-based preprocessing must reproduce this exact patchify+reorder+normalize, not
just visually resemble it** -- validate against `vision_golden.py`'s `pixel_values` tensor
(`[num_patches, 1536]` fp32), not by eye. `image_grid_thw = [t, h_patches, w_patches]` (one row per
image) is the shape contract every downstream step needs.

### 2. Patch embedding

`Qwen3_5VisionPatchEmbed`: a `Conv3d(3, 1152, kernel=[2,16,16], stride=[2,16,16])` -- but since
`pixel_values` already arrives pre-patchified/flattened (`[num_patches, 1536]`), this Conv3d is
mathematically **a single dense `[1536 -> 1152]` matmul + bias per patch row**, not a real
convolution -- `r4dx` should implement it as `r4d_gemm_bf16_nt_m64`-family GEMM against
`vision.patch_embed.proj.weight` reshaped to `[1152, 1536]`, exactly like every other linear in this
codebase, not port a Conv3d kernel.

### 3. Position embedding (learned, bilinearly interpolated -- no text-side analogue)

A learned `[2304, 1152]` embedding table (`num_grid_per_side = 48 = sqrt(2304)`), resampled to each
image's actual `(h_patches, w_patches)` grid via 4-tap bilinear interpolation
(`align_corners=True`) computed by `transformers.vision_utils.
get_vision_interpolation_indices_and_weights` -- **not** a fixed sinusoidal table like the text
side's rope. Per patch: gather 4 rows of the table at `interp_indices[patch]` and weighted-sum by
`interp_weights[patch]`, in the same spatial-merge-block-major patch order preprocessing already
emits. Added to the patch-embed output before the encoder blocks. `vision_golden.py` does not dump
`interp_indices`/`interp_weights` themselves (recorded as a follow-up: r4dx's own implementation
should recompute them from `grid_thw`, the same way the reference does, rather than have this golden
bake in a redundant copy -- flag if that turns out to be insufficient once someone writes the C++).

### 4. 27 encoder blocks (`Qwen3_5VisionBlock`, pre-norm, residual)

```
h = h + attn(LayerNorm(h), cu_seqlens, rope_cos_sin)     -- norm1, NOT RMSNorm (real nn.LayerNorm,
                                                             mean-subtracted, eps=1e-6)
h = h + mlp(LayerNorm(h))                                -- norm2
```

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
side's rope kernel.

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

`num_merged_tokens = num_patches / 4` (784 patches -> 196 merged tokens for `vision_golden.py`'s
28x28-patch test image). These 196 `[5120]`-dim rows are exactly what gets spliced into the text
token sequence at the image's placeholder positions (task item 3).

## Text-side splicing (`Qwen3_5Model.get_rope_index`, task item 3's "correct mrope handling")

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
  (the class of bug docs/validation.md rung 5 exists to catch) -- validate this arithmetic against a
  short synthetic token sequence with a known image span before trusting any end-to-end output.

`r4dx`'s existing `rope_partial_mrope` kernel (docs/architecture.md) already implements the
`[11,11,10]`-section interleaved mrope math for **text** positions; what's new for step 3's task is
(a) computing the above three-case position-id assignment (a host-side/CPU bookkeeping problem, not
a new kernel) and (b) the image token id (`248056`) placeholder-splicing itself -- replacing each
image-placeholder token's embedding row with the corresponding merger output row instead of an
`embed_tokens` lookup, at the right sequence offset.

## What's done / what's not (2026-09-20 pass)

**Done, real hardware, HIP device 1:**

- Read every architecture/preprocessing/mrope-splicing source above directly from the reference
  venv's `transformers` 5.17.0 (`Qwen3_5VisionModel`, `Qwen3_5VisionAttention`,
  `Qwen3_5VisionPatchEmbed`, `Qwen3_5VisionPatchMerger`, `Qwen3_5VisionRotaryEmbedding`,
  `transformers.vision_utils.{get_vision_position_ids,get_vision_interpolation_indices_and_weights,
  get_vision_attention_seqlens}`, `Qwen3_5Model.get_rope_index`), not inferred or guessed.
- Confirmed the real checkpoint (`C:\AI\models\Qwen3.8-27B`) carries all 333 real `model.visual.*`
  weights (not a shard gap) and that its `vision_config` matches this document's "Model facts"
  table exactly.
- **Found and fixed a real, reproducible bug in shared reference-tooling infrastructure**:
  `tools/reference/common.py`'s `ShardIndex.get_tensor`/`get_row_slice` returned a tensor whose
  storage aliases the `safe_open` context manager's own mmap; once the `with` block exits and
  unmaps the file, that storage is dangling. Reading a handful of tensors (as `layer_golden.py`'s
  ~14-20-tensor-per-component text-layer goldens do) never surfaced this; a tight loop over the
  vision tower's 333 weights reproducibly crashed the Python interpreter with an access violation
  (`0xC0000005`) when loading all of them in one `load_state_dict` call -- and, more dangerously,
  is exactly the shape of bug that can silently read garbage instead of crashing on a different
  allocator/timing (this is a Python-side memory-safety bug in tooling every reference-golden
  script in this repo depends on, not a modeling bug). Fixed with a `.clone()` after `get_tensor`/
  the slice (see `common.py`'s inline comment for the full mechanism) -- verified the fix by
  reproducing the crash, bisecting it down to `load_state_dict` on the full 333-tensor dict (not
  reading tensors one at a time, and not the reads themselves), and confirming a clean run after
  the fix, three times, at both `--device cpu` and `--device cuda`.
- Wrote `tools/reference/vision_golden.py` (mirrors `layer_golden.py`'s structure/CLI/manifest
  conventions), covering: real preprocessing (`transformers.AutoImageProcessor` +
  `Qwen2VLImageProcessor`, the checkpoint's actual configured processor) of a deterministic
  synthetic 448x448 test image (saved to `<out-dir>/vision_test_image.png` for reproducibility and
  later rung-5 use -- gradient + checkerboard + circle, chosen to exercise both smoothly-varying and
  sharp-edge content, not flat noise); real weights end to end (`weights_source=safetensors`,
  0 missing keys); every encoder block's output (27 tensors); block 0's full internal chain
  (norm1/qkv-raw/post-rope-q-k/proj-out/norm2/mlp-fc1/mlp-fc2, mirroring `layer_golden.py`'s
  per-kernel granularity for the text side); the merger output (`[196, 5120]` for this test image).
  Ran for real on HIP device 1 (bf16, matching the container's actual dtype): 39 tensors, all
  finite (no NaN/Inf), shapes exactly matching this document's math (784 patches -> 196 merged
  tokens at a 28x28-patch grid), `tools/reference/golden_out/vision_tower.safetensors` +
  `vision_manifest.json`. This is docs/validation.md rung 3's ground truth for the vision tower,
  the same status GDN/full-attention layers already had before `src/model` existed for them.

**Not done, explicitly, per this task's own "finish in the listed order and report exactly where you
stopped" instruction** (task items 2-5 in full):

- **No C++ implemented.** `src/model` has no vision module, no patch-embed GEMM wiring, no
  `r4d_attn_vit_h72_bf16` call site, no merger, no position-embedding interpolation code, no
  `stb_image`-based preprocessing (`third_party` vendors `stb_image` per README.md but nothing in
  `src/` includes it yet). This is a large, multi-subsystem C++/HIP implementation (image
  preprocessing exactly matching the reference's resize/patchify/normalize; a 27-layer encoder
  reusing this codebase's GEMM/LayerNorm/GELU primitives plus one new kernel call site; the
  bilinear position-embedding interpolation, which has no existing analogue anywhere in this
  codebase; the merger; then the text-side splicing/mrope bookkeeping) that genuinely needs its own
  dedicated implementation pass with real on-hardware iteration against the goldens above, the same
  class of "correctly scope, don't half-build" judgment this project's own R10 pass (docs/status.md)
  applied to the prefill GEMM kernel. Shipping a partially-wired, unvalidated tower would be strictly
  worse than the current state (a precise spec + real goldens, nothing claimed as working that
  isn't) per this task's own explicit guidance.
- No `--image` CLI flag, no server `image_url` wiring (still correctly rejected with `400`,
  `docs/server.md`'s existing deferred-features note is accurate and unchanged).
- No golden-vs-r4dx tolerance validation (nothing in `src/model` yet to validate against).
- No end-to-end "describes a real image correctly" check.
- No perf measurement (image encode time, prefill-with-image tok/s, VRAM delta) -- there is nothing
  to measure yet.
- Not attempted: a real (non-synthetic) photograph through `vision_golden.py` -- the synthetic test
  image is sufficient for rung-3 per-layer numeric validation (which only needs real, structured
  pixel data, not a particular photo's content) but rung 5's "describes a real image correctly"
  check will need an actual photograph once the C++ side exists to run it through.
- Not attempted: multi-image or video input (`vision_golden.py` covers exactly the one-image case;
  `get_vision_position_ids`'s temporal handling and `cu_seqlens`'s multi-segment case are read and
  documented above but not exercised against a second golden).

**Recommended next steps, in order** (matches this document's own section order): (1) `stb_image`
preprocessing, validated against `vision_golden.py`'s `pixel_values` tensor byte-for-byte-in-spirit
(within a resize/rounding tolerance to be determined empirically, since bilinear resize + this
exact patch-order reshuffle has no existing r4dx precedent to inherit a tolerance from); (2) patch
embed + position-embedding interpolation, validated against `patch_embed`-plus-`pos_embed`
(currently not separately dumped -- add `patch_embed_out` and `pos_embeds` as their own golden
tensors, an easy follow-up to `vision_golden.py`, before or during that implementation work) --
i.e. block 0's input; (3) one encoder block, validated against `block_00_output` and the
block0_* internal-chain tensors; (4) all 27 blocks + merger, validated against `block_26_output`/
`merger_output`; (5) text splicing + mrope, validated against a short synthetic prompt with a known
image span (no golden for this exists yet -- would need a small addition to `layer_golden.py` or a
new joint script, since it requires the FULL model, text+vision together); (6) CLI/server wiring;
(7) end-to-end description + perf measurement.
