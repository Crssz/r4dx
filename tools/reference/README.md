# tools/reference

Python reference/validation tooling for r4dx. These scripts don't touch the C++ engine at all --
they run the *real* HF `transformers` implementation of Qwen3_5 (read-only, from the reference
venv) and dump its numbers to disk, so `src/model`'s tests have a ground truth to diff against.
See `docs/validation.md` for where this fits in the overall validation ladder.

Everything here is read-only against:

- `C:\Users\user\dev\vLLM_for_AMD\.venv-rocm10` -- the reference `transformers` 5.17.0 /
  `torch` 2.13.0+rocm10.0.0 install. **Run every script in this directory with that venv's
  `python.exe`.** Never `pip`/`uv install` into it -- these scripts only import from it.
- `C:\AI\models\Qwen3.8-27B` -- the Qwen3.8-27B checkpoint. Shards may still be downloading; every
  tensor read from it is allowed to fail and falls back to deterministic seeded random init
  instead (see "Weight fallback" below). As of this writing all 18 shards are fully downloaded and
  every script below loads real weights.

Nothing in this directory writes outside the `--out-dir`/`--out` path you pass it (default:
a `golden_out*`/`kv_calibrate_out` subdirectory of `tools/reference/` itself, gitignored -- see
`tools/reference/.gitignore`).

## GPU rule

Per this repo's rule (only HIP device 1, the headless R9700, may be used), every script's
`--device cuda` path refuses to run unless `$env:HIP_VISIBLE_DEVICES` is exactly `'1'`:

```powershell
$env:HIP_VISIBLE_DEVICES = '1'
```

Set that once per shell before any `--device cuda` invocation. `--device cpu` always works (no
GPU rule to enforce, no device to serialize against) and is what `tests/reference/test_manifest.py`
uses.

If you run both scripts in the same shell, run them one after another, not in parallel -- this
repo's GPU tests are serialized by convention (one process on device 1 at a time).

## layer_golden.py

```powershell
$env:HIP_VISIBLE_DEVICES = '1'
C:\Users\user\dev\vLLM_for_AMD\.venv-rocm10\Scripts\python.exe tools\reference\layer_golden.py `
    --device cuda --out-dir tools\reference\golden_out
```

Builds, one at a time, straight from `transformers`' `modeling_qwen3_5.py`:

- **`layer_000_gdn`**: layer 0, a GDN (linear-attention) layer, real weights.
- **`layer_003_full_attention`**: layer 3, a full-attention layer, real weights.
- **`final_norm_lm_head`**: `text.final_norm` + a `--tiny-vocab` (default 256) row-slice of
  `lm_head.weight` (never materializes the full `[248320, 5120]` matrix -- reads only the rows it
  needs via `safetensors`' `get_slice`).
- **`mtp`**: the MTP block's inner decoder layer (`mtp.layers.0.*`, real weights -- confirmed by
  inspection to be one `full_attention`-type `Qwen3_5DecoderLayer`, no `linear_attn.*` keys), plus
  the four un-exercised fusion tensors (`mtp.fc.weight`, `mtp.norm.weight`,
  `mtp.pre_fc_norm_{embedding,hidden}.weight`) dumped raw. **transformers 5.17.0 has no MTP
  forward implementation to run** (`Qwen3_5PreTrainedModel._keys_to_ignore_on_load_unexpected =
  [r"^mtp.*"]`), so the fc/embedding-fusion math itself is not exercised here -- see the `mtp`
  component's `architecture_assumption` field in the manifest and docs/validation.md.

Each component runs a fixed-seed random `[T=64, hidden=5120]` hidden-state input through a
**prefill** call (no cache) and a **decode** call (`T=4` new tokens, continuing the prefill's
`Cache`), and dumps every intermediate the task cares about. Output:
`<out-dir>/<component>.safetensors` (one file per component) + `<out-dir>/manifest.json` (shapes,
dtypes, weight provenance, per-component status, and the tolerance table `src/model`'s tests should
use).

For a **GDN (linear-attention) layer**, captured via monkeypatching the module-level free functions
`Qwen3_5GatedDeltaNet.forward` calls into (see `GdnCapture` in the script) plus forward hooks:

- `gdn_q` / `gdn_k` / `gdn_v` / `gdn_g` / `gdn_beta`: the RAW tensors as passed to/from
  `torch_{chunk,recurrent}_gated_delta_rule` -- **pre-l2norm, pre-(1/sqrt(K))-scale,
  POST-repeat_interleave** (GQA-expanded from `Hg=16` up to `H=48` heads for q/k), and `gdn_g` is
  **per-token**, not chunk-summed. (An earlier version of this README and the script's own
  docstring incorrectly called these "post-l2norm/repeat-interleave" -- l2norm and scaling happen
  *inside* the callee, after the point these are captured.)
- `gdn_q_l2` / `gdn_k_l2` / `gdn_g_chunk_cumsum` / `gdn_core_attn_out`: the **r4d-shaped** goldens --
  l2-normed (but NOT scaled by `1/sqrt(K)`, which `r4d_gdn_chunk_scan_k128_v128_c64_bf16` takes as
  its own `scale` argument) q/k de-interleaved back to `[T, Hg=16, K=128]`, g cumulatively summed
  within each 64-token chunk `[T, H=48]`, and the chunked-scan/recurrent-rule's raw `[T, H=48,
  V=128]` output before the gated RMSNorm -- exactly the layouts `r4d_gdn_conv_prep_w4_h128_bf16`
  emits and `r4d_gdn_kkt_solve_k128_c64_bf16` / `r4d_gdn_chunk_scan_k128_v128_c64_bf16` consume/
  produce (see `r4d.h`).
- `gdn_conv_out_fn_full` / `gdn_conv_out_fn_trimmed`: the causal-conv1d output. `_full` is the raw,
  still-cache-prefixed return value of `causal_conv1d_fn` (which does NOT trim to just the new
  tokens); `_trimmed` replicates the extra slice `Qwen3_5GatedDeltaNet.forward` itself applies
  afterward (`mixed_qkv[:, :, -seq_len:]`) and has exactly `prefill_len`/`decode_len` columns for
  its stage -- **only `_trimmed` is the correct golden shape** for
  `r4d_gdn_conv_update_w4_h128_bf16`'s output. (A prior version of this tool recorded only the
  untrimmed tensor, under a stage tag that didn't even track which conv function ran -- fixed.)
- `gdn_z` / `gdn_a_raw` / `gdn_b_raw`: raw `in_proj_z`/`in_proj_a`/`in_proj_b` outputs (the gate and
  the pre-sigmoid/pre-softplus `a`/`b` that feed `r4d_gdn_recurrent_update_k128_v128_bf16_fp32state`
  alongside `A_log`/`dt_bias`).
- `gdn_gated_norm_out` / `gdn_out_proj_out`: the gated-RMSNorm output (golden for
  `r4d_gdn_gated_rmsnorm_h128_bf16`) and the post-`out_proj` tensor.

For a **full-attention layer** (also used by `mtp`'s inner decoder layer, which is
`full_attention`-typed), captured via `nn.Module` forward hooks plus a monkeypatch of
`apply_rotary_pos_emb` (same technique as `kv_calibrate.py`):

- `attn_qg_raw` / `attn_k_raw` / `attn_v_raw`: raw `q_proj`/`k_proj`/`v_proj` outputs.
- `attn_q_normed` / `attn_k_normed`: **pre-rope** (q_norm/k_norm are applied before rope in
  `Qwen3_5Attention.forward`) -- despite the name, these are NOT post-rope.
- `attn_q_post_rope` / `attn_k_post_rope`: the actual **post-rope** q/k (shape `[heads, T,
  head_dim]`, post-transpose) -- what the fp8 paged KV cache stores for K. (A prior version of this
  tool didn't capture this at all, despite the README claiming it did.)
- `attn_gate_sigmoid`: `sigmoid(gate)` from the fused `q_proj` output's second half.
- `attn_post_gate`: the tensor right before `o_proj` (i.e. `attn_output * sigmoid(gate)`), to
  isolate the `attn_output_gate` path.
- `attention_output`: the layer's `self_attn` module output (post-`o_proj`).

**Weight fallback**: for any component, if the checkpoint doesn't have the tensors this script
needs (shard not downloaded, or -- for `mtp` -- genuinely absent), the module is left at its
PyTorch-default seeded random init (`torch.manual_seed(seed)` right before construction) instead
of crashing. The manifest's `weights_source` field says which happened
(`"safetensors"` or `"random_init(seed=...)"`) -- check it before trusting a component's numbers
as a real-weights golden.

**CPU / smoke mode**: `--device cpu --force-random-init` skips the checkpoint entirely (uses
`transformers`' own default `Qwen3_5TextConfig()`, smaller dims) -- this is what
`tests/reference/test_manifest.py` drives, no checkpoint or GPU needed. Whenever a component falls
back to random init, every `Qwen3_5RMSNorm`/`Qwen3_5RMSNormGated` weight under it is perturbed with
small seeded noise (`manifest.json`'s `norm_weights_perturbed` / `final_norm_weights_perturbed`
fields say so) -- `transformers`' own default init for these weights is exactly zero, which makes
the norm's `output * (1.0 + weight)` scale identically `1.0` and can't catch a missing/incorrect
`(1 + w)` term, a classic port bug for this norm convention. Without the perturbation, the
random-init smoke path would silently never exercise that scale.

Runtime: ~20-30s on HIP device 1 (real weights, default `--prefill-len 64 --decode-len 4
--tiny-vocab 256`) for all four components combined; well under a second in `--device cpu
--force-random-init` smoke mode.

Key options (see `--help` for the rest): `--gdn-layer` / `--attn-layer` (default 0 / 3),
`--prefill-len` / `--decode-len` (default 64 / 4), `--tiny-vocab` (default 256), `--seed` (default
1234), `--skip-mtp`.

## vision_golden.py

```powershell
$env:HIP_VISIBLE_DEVICES = '1'
C:\Users\user\dev\vLLM_for_AMD\.venv-rocm10\Scripts\python.exe tools\reference\vision_golden.py `
    --device cuda --out-dir tools\reference\golden_out
```

The vision-tower analogue of `layer_golden.py` -- see `docs/vision.md` for the full architecture
spec this script validates against. Builds `Qwen3_5VisionModel` directly from `modeling_qwen3_5.py`,
loads real `model.visual.*` weights (333 tensors, all present in the real checkpoint), preprocesses
real images with the checkpoint's own configured `transformers.AutoImageProcessor` (the config
names `Qwen2VLImageProcessorFast`; in `transformers` 5.17.0 that resolves to
`Qwen2VLImageProcessor`, which *is* the torchvision-backed one), and runs **three** cases, each
its own `.safetensors` file and its
own component in `<out-dir>/vision_manifest.json`:

| component | image(s) | grid | what it is for |
|---|---|---|---|
| `vision_tower` | synthetic 448x448 | 28x28 | the deep case: every encoder block's output (27) + block 0's full internal chain (norm1/qkv-raw/post-rope-q-k/proj-out/norm2/mlp-fc1/mlp-fc2). Also carries the learned `pos_embed_table` and `rope_inv_freq`. `smart_resize` is the identity here, so this case does **not** exercise the resampler. |
| `vision_tower_nonsquare` | synthetic 613x409 (WxH) -> 608x416 | 26x38 | the one that does: width down-sampled, height up-sampled, neither input side a multiple of 32, `h != w` afterwards. |
| `vision_tower_two_image` | both, in one processor call and one forward | 28x28 + 26x38 | `cu_seqlens` with two segments, concatenated `pixel_values`, per-image restart of the pos-embed/rope index math. |

Every case dumps the front-end tensors a from-scratch implementation has to match one at a time
before any block output can: `pixel_values`, `image_grid_thw`, `interp_indices`/`interp_weights`
(the 4-tap position-embedding gather), `pos_embeds`, `vision_position_ids`, `cu_seqlens`,
`patch_embed_out`, `rope_cos`/`rope_sin`, `block_input`, `last_hidden_state`, `merger_output`.

`--image`/`--image2 <path>` use real photos; without them the two deterministic synthetic test
images are generated and saved to `<out-dir>/vision_test_image.png` and
`<out-dir>/vision_test_image_nonsquare.png` -- `tests/vision/test_preprocess.cpp` decodes those
exact files, so regenerating the goldens with different `--image` arguments regenerates the images
the test reads too.

The `vision_tower` component additionally carries three small CPU-only fixture groups, each pinning
a preprocessing claim that the whole-pipeline `pixel_values` comparison would detect but not
localize:

| fixture group | files written | what it pins |
|---|---|---|
| `make_channel_conversion_fixtures` | `vision_test_image_rgba.png`, `vision_test_image_grey.png` (+ `rgba_pil_rgb`, `grey_pil_rgb`) | alpha is **dropped, not composited**, and greyscale is replicated -- the behaviour `src/vision`'s `stb_image` path relies on. |
| `make_resampler_fixtures` | `vision_resize_noise_{0..3}.png` (+ `resize_noise_{i}_out`) | the uint8 antialias resampler alone, on random noise (the worst case for any rounding disagreement) at four size pairs: both axes down, the real `smart_resize` mixed pair, both axes up, and a 4x downscale -- the two photographs exercise one resize between them and no upscale at all. |
| `make_format_fixtures` | `vision_test_image_format.{bmp,gif,jpg}` (+ `bmp_pil_rgb`, `gif_pil_rgb`, `jpeg_pil_rgb`) | the three non-PNG decoders `image_decode.h` claims. BMP/GIF must match PIL exactly; JPEG only within a small bound (stb's IDCT is not libjpeg's). |

The manifest keeps `layer_golden.py`'s tolerance-table convention, in a separate file since this
rung has no text-layer components to share a manifest with; each fixture group records its own
file/tensor pairs under the `vision_tower` component (`channel_conversion`, `resampler_fixtures`,
`format_fixtures`).

**One deliberate divergence from a plain `.to(dtype=bfloat16)` run**: that cast also rounds the
rope module's non-persistent `inv_freq` buffer to bf16, which the script undoes by recomputing it
in fp32 through the rope class's own `compute_axial_rope_parameters`. See `docs/vision.md` "Rope"
for why (and for the 0.047 absolute cos/sin error the bf16 buffer is worth).

**Found and fixed while building this script**: `common.py`'s `ShardIndex.get_tensor`/
`get_row_slice` had a real dangling-mmap bug (see `common.py`'s inline comment and `docs/vision.md`)
that a tight loop over ~300+ tensors reproducibly turns into a Python interpreter crash (access
violation) -- `layer_golden.py`'s/`kv_calibrate.py`'s much smaller per-component tensor counts never
triggered it. Fixed with `.clone()`; every script in this directory benefits, no other script's own
code changed.

**`--attn-impl {eager,sdpa}` and `--cases <names>`** (added with the device half, 2026-09-21).
`eager` is the default, so every committed golden is byte-identical to before these options
existed. They exist for one measurement: transformers' `eager_attention_forward` computes
`torch.matmul(q, k^T)` in **bf16**, rounding the attention scores to bf16 before the `* scaling`,
again after it, and the softmax probabilities to bf16 before the `P @ V` matmul; SDPA and
`r4d_attn_vit_h72_bf16` keep all three in fp32/f16. Regenerating one case through SDPA into a
scratch directory and diffing it against the committed golden is how `docs/vision.md` shows that
the reference's own two implementations disagree with each other by as much as r4dx disagrees with
either -- i.e. that the deep-block disagreement is the tower's bf16 conditioning, not an r4dx
error:

```powershell
$env:HIP_VISIBLE_DEVICES = '1'
<venv>\Scripts\python.exe tools\reference\vision_golden.py `
    --device cuda --attn-impl sdpa --cases vision_tower --out-dir <scratch>
$env:R4DX_VISION_GOLDEN_DIR = '<scratch>'
build\win-hip\tests\vision\test_vision_tower.exe
```

Runtime: ~40-60s on HIP device 1 (real weights, three cases, 27 blocks each); ~20s for one case.

## rope_index_golden.py

```powershell
C:\Users\user\dev\vLLM_for_AMD\.venv-rocm10\Scripts\python.exe tools\reference\rope_index_golden.py `
    --out-dir tools\reference\golden_out
```

The text-side counterpart of `vision_golden.py`: the mrope `position_ids` a prompt containing
images gets, i.e. `Qwen3_5Model.get_rope_index` + `Qwen3_5Model.get_vision_position_ids`, called as
the real unmodified methods bound to a shim that carries only `.config` (those two touch nothing
else on the model). **No weights, no GPU, no images** -- `get_rope_index` consumes `input_ids` for
shape, `mm_token_type_ids` and `image_grid_thw` -- so this is the cheapest script in this
directory, under a second on any machine.

Five synthetic cases, each dumping `input_ids`, `mm_token_type_ids`, `image_grid_thw`,
`position_ids` `[3, 1, seq]` and `mrope_position_deltas`: `text_only`, `text_one_image`,
`text_two_images_different_grids`, `image_first` (no leading text run), `image_last` (no trailing
one). Output: `<out-dir>/rope_index.safetensors` + `<out-dir>/rope_index_manifest.json`, whose
`semantics` block states the three position rules in prose. Consumed by
`tests/vision/test_position_ids.cpp`.

**`--verify-prompt <dump.json>`** (added with the splicing stage, 2026-09-22) checks the ENGINE's
own rope rows for a REAL rendered prompt against the same unmodified `get_rope_index`, instead of
regenerating the golden. The input is what `tests/vision/tool_vision_chat --dump-prompt` writes:
the tokens the engine actually fed (after the processor's `<|image_pad|>` expansion), the derived
`mm_token_type_ids`, the image grids, and `engine_position_ids` -- the `[3, seq]` rows
`Model::PrefillMultimodal` really handed the rope kernel, not a re-derivation. The five cases above
are synthetic; this is the same comparison against a prompt that came out of the real chat
template and the real tokenizer, where an off-by-one in the placeholder expansion or the span
offsets lives and a synthetic fixture cannot reach:

```powershell
$env:HIP_VISIBLE_DEVICES = '1'
build\win-hip\tests\vision\tool_vision_chat.exe --layout w4a16 --image pic.png `
    --prompt "Describe this image." --dump-prompt dump.json
C:\Users\user\dev\vLLM_for_AMD\.venv-rocm10\Scripts\python.exe tools\reference\rope_index_golden.py `
    --verify-prompt dump.json
```

## mrope_layer_golden.py

```powershell
$env:HIP_VISIBLE_DEVICES = '1'
C:\Users\user\dev\vLLM_for_AMD\.venv-rocm10\Scripts\python.exe tools\reference\mrope_layer_golden.py `
    --device cuda --out-dir tools\reference\golden_out
```

ONE real full-attention decoder layer (layer 3 by default -- `layer_golden.py`'s own attention
case, so the two are comparable), real weights, driven by the 3-AXIS mrope position ids
`get_rope_index` produces for a prompt containing a 10x16-patch image. `layer_golden.py`'s own
`mrope_position_ids` collapses (t,h,w) to one sequential index, which is correct for text but
means that golden passes identically whether the rope kernel selects a per-bin position stream or
ignores the h/w rows entirely; this one cannot. A full bf16 27B reference does not fit on this
card, so one layer is the largest piece of the real stack that can be compared numerically at all.

The grid is deliberately non-square (h and w cannot be swapped without changing the answer) and
`max(h,w)//merge = 8` differs from the 40-token merged count, so the post-image advance rule is
exercised. The image rows' hidden states are scaled 2x relative to the text rows so that a splice
writing the right rows at the wrong offset moves the output measurably. Dumps prefill and decode
`hidden_states` / `attention_output` / `layer_output`, post-rope q/k, the `[3, total]` position
rows and the token/type/grid arrays. Output:
`<out-dir>/mrope_layer_003.safetensors` + `<out-dir>/mrope_layer_manifest.json`. Consumed by
`tests/model/attention/test_mrope_attn_layer.cpp`.

## kv_calibrate.py

```powershell
$env:HIP_VISIBLE_DEVICES = '1'
C:\Users\user\dev\vLLM_for_AMD\.venv-rocm10\Scripts\python.exe tools\reference\kv_calibrate.py `
    --device cuda --layer 3 --out tools\reference\kv_calibrate_out\kv_descale.json
```

Prototype of the static fp8 KV descale calibration from `docs/container-format.md` ("KV descale
tables"). Tokenizes a calibration text (real tokenizer, via `AutoTokenizer` -- this only *uses*
the tokenizer to build input ids, it doesn't touch anything under `src/tokenizer` or
`tools/reference/tok_golden.py`, which belong to the tokenizer agent), gathers the calibration
tokens' `text.embed_tokens` rows (real weights, row-gathered so the full `[248320, 5120]` table is
never materialized), runs them through `--layer`'s `input_layernorm` + `self_attn` (real weights),
and records the per-kv-head `amax` of K (**post-rope** -- captured by monkeypatching
`apply_rotary_pos_emb`, matching exactly what the paged fp8 cache would store) and V (no rope
applied to V; captured via a forward hook on `v_proj`). `--layer` must be a `full_attention` layer
(0-indexed `3, 7, 11, ...`; the script asserts this and explains why if you pick a GDN layer --
r4d's paged fp8 KV cache is attention-only).

**Calibration corpus**: `--calib-text-file <path>` (default: `tools/reference/calib.txt` next to
this script if present, else the old built-in `DEFAULT_CALIBRATION_TEXT` paragraph). `calib.txt` is
a deliberately mixed English-prose / source-code / Thai-prose corpus (a few hundred tokens after
repetition to `--num-tokens`) -- a single-register corpus (e.g. English news prose alone) tends to
under-estimate the true per-head amax against the token distributions (code, Thai, mixed-script)
this engine will actually see, which would make the calibrated descale too small and clip outlier
activations at inference time. The output JSON records `calib_text_source` (which file was used)
and `calib_text_sha256` for provenance.

Output JSON: `{"<layer_idx>": {"k_amax": [4 floats], "v_amax": [4 floats], ...}}` plus provenance
fields (`weights_source`, `embed_source`, `config_sha256`, `torch_dtype`, `device`,
`calib_text_source`, `calib_text_sha256`, etc. -- `torch_dtype`/`device` matter because `amax` is
computed in bf16 on `--device cuda` and fp32 on `--device cpu`, materially different numbers) and
two prose fields baked into every run's output: `"caveat"` (this is a **prototype** -- it feeds the
calibration tokens' raw embeddings straight into one layer, skipping every preceding layer's
transform, so the hidden-state distribution isn't what that layer truly sees mid-stack) and
`"converter_consumption"` (exactly how `src/convert` turns `k_amax`/`v_amax` into
`text.layers.{i}.attn.k_descale`/`.v_descale`).

**Descale convention (exact)**: `descale = amax / 448.0`, where `448.0` is the OCP e4m3fn finite
max magnitude and `amax` is the per-kv-head absolute-max over the calibration set (K: post-rope;
V: raw `v_proj` output, no rope). This matches `src/kernels/src/r4dx_kernels.hip`'s
`r4dx_kv_write_paged_fp8_hnd`, which writes `stored_fp8 = fp8e4m3(real_bf16_value / descale[head])`
-- i.e. it *divides* by descale before the fp8 cast -- so the inverse, `dequant = fp8_value *
descale[head]`, is exactly what the attention kernel must (and does) use to reconstruct the real
value for QK^T / PV. `r4dx-convert --kv-calib <json>` (`src/convert/main.cpp`) applies this same
`amax / 448.0` formula per kv head when it fills `text.layers.{i}.attn.k_descale` / `.v_descale`,
replacing `docs/container-format.md`'s placeholder `1.0`; a layer missing from the calibration JSON
(or with a `k_amax`/`v_amax` of the wrong length) falls back to `1.0` with a `WARNING` printed to
stderr, not a hard error. The resulting per-layer `fp32[kv_heads]` vector must still be broadcast to
`r4d.h`'s runtime `(num_seqs, kv_heads)` shape, one identical row per sequence, by whatever in
`src/model` builds `R4DArgs.k_descale`/`.v_descale` -- calibration is not per-sequence.

Running the script again with the same `--out` path **merges** the new layer's result into the
existing file instead of overwriting it, so calibrating all 16 full-attention layers is a matter of
running `--layer 3`, `--layer 7`, ... `--layer 63` in sequence against the same `--out`. A
`--calib-text-file` argument (default `tools/reference/calib.txt`) picks the calibration corpus; see
above.

Runtime: ~10s on HIP device 1 (default `--num-tokens 256`); a few seconds on CPU.

## Shared file format (the KL comparison)

Milestone 9 measures how far the quantized r4dx engine has drifted from the original bf16
checkpoint, by scoring both on the same token sequences and reporting the KL divergence per
position. Two *independent* producers write the same two file kinds, and `kl_report.py` pairs them:

| producer | source tag | what it runs |
|---|---|---|
| `tools/reference/full_logits_golden.py` | `"reference"` | the original bf16 HF checkpoint through `transformers`, streamed layer by layer |
| `tests/model/tool_teacher_forced_logprobs` | `"r4dx"` | the r4dx C++ engine, teacher-forced over the same ids |

**Tokens file** (`kl_corpus/tokens.json`, committed):

```json
{"tokenizer": "C:\\AI\\models\\Qwen3.8-27B",
 "segments": [{"name": "english_prose", "token_ids": [8241, 264, ...]}, ...]}
```

Ids come from the checkpoint's own `AutoTokenizer` with `add_special_tokens=False` on the raw file
text -- **no chat template, no BOS, no EOS**. Each segment is evaluated independently from a fresh
context (position 0).

**Log-prob file**, one pair per segment, in the producer's `--out-dir`:

- `<segment>.logprobs.f16` -- raw little-endian float16, row-major `[T-1, V]`, where `T =
  len(token_ids)` and `V` is the **lm_head's own row count** (248320 for this checkpoint --
  `full_logits_golden.py` reads it off `lm_head.weight`'s shape rather than trusting
  `config.vocab_size`, and prints a note if the two disagree). Row `i` is
  `log_softmax(logits at position i)` = `log p(next token | token_ids[0..i])`, computed in **fp32**
  before the fp16 cast. The last position is not written: it has no next token inside the segment
  to be scored against (`full_logits_golden.py` still reports its top-5, which is what validation
  (i) below compares).
- `<segment>.meta.json` -- `{"T", "V", "dtype": "float16", "rows": T-1, "source", "layout" or
  "torch_dtype", "sha256_of_token_ids_json", ...}`.

**The `-1e4` clamp.** fp16's most negative finite value is about `-65504`, so a log-probability
below that would become `-inf` on the cast and poison every downstream sum. Both producers floor
log-probs at `-1e4` *before* the cast. `exp(-1e4)` is exactly `0.0` even in fp64, so those entries
carry zero weight in `sum_v p_ref * (logp_ref - logq_test)` and no KL total moves materially; the
only thing the clamp can hide is *how* impossible an already-impossible token was.

**`sha256_of_token_ids_json`** is SHA-256, lowercase hex, of the **compact** JSON array of that
segment's ids -- exactly what
`hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode("utf-8")).hexdigest()` produces, i.e.
sha256 of the bytes `[1,2,3]`. `make_tokens_json.token_ids_sha256`,
`full_logits_golden.token_ids_sha256`, `kl_report.token_ids_sha256` and the C++
`TokenIdsSha256` in `tests/model/teacher_forced.h` all implement this one formula; `kl_report.py`
recomputes it from the tokens file and refuses to pair two dumps whose sidecars disagree with it
(`--allow-mismatch` downgrades that to a warning).

## make_tokens_json.py

```powershell
C:\Users\user\dev\vLLM_for_AMD\.venv-rocm10\Scripts\python.exe tools\reference\make_tokens_json.py `
    --corpus-dir tools\reference\kl_corpus --max-tokens 1024 `
    --out tools\reference\kl_corpus\tokens.json
```

Turns a directory of `.txt` files into the tokens file above -- one segment per file, named after
the file's stem, truncated to exactly `--max-tokens` ids, and **failing loudly** if any file
tokenizes to fewer than `--min-tokens` (default: `--max-tokens`) rather than silently emitting a
short segment. `--file NAME=PATH` (repeatable) replaces the directory scan when you want an ad-hoc
corpus. No GPU, no weights: only the tokenizer files under `--model-dir` are read.

## kl_corpus/

The **held-out** corpus the KL report is measured on. Deliberately disjoint from
`tools/reference/calib.txt`, which is what calibrated the fp8 KV descales -- measuring drift on the
same text that chose the quantization constants would flatter the result. Four files, each
comfortably over 1024 tokens, all four truncated to exactly 1024 in `tokens.json`:

| segment | file | tokens before truncation | what it is |
|---|---|---|---|
| `english_prose` | `english_prose.txt` | 1386 | original technical-explanatory English prose (written for this corpus: an essay on the memory hierarchy). No third-party or licensed text. |
| `cpp_source` | `cpp_source.txt` | 4124 | a verbatim excerpt of this repo's own `src/model/model.cpp` (first 260 lines) |
| `python_source` | `python_source.txt` | 3531 | a verbatim excerpt of this repo's own `tools/reference/layer_golden.py` (first 230 lines) |
| `thai_prose` | `thai_prose.txt` | 2325 | original Thai prose (written for this corpus: everyday and technical topics) |

Nothing here is downloaded, scraped or licensed: two files are this repository's own source and two
were written for this purpose. All four are UTF-8 **without** a BOM and LF-only, and
`kl_corpus/.gitattributes` marks the whole directory `-text` so git performs no end-of-line
conversion on it. Both of those matter because the corpus is tokenized byte for byte: a BOM
tokenizes to a real token and would shift every id by one position, and under this repo's
`core.autocrlf=true` a Windows checkout would otherwise hand back CRLF files whose ids no longer
match the committed `tokens.json`.

`tokens.json` is committed (44 KiB, deterministic given the tokenizer); the `.logprobs.f16` /
`.meta.json` outputs are not -- see `tools/reference/.gitignore`, which ignores `kl_out*/` for the
same reason it ignores `golden_out*/`.

## full_logits_golden.py

```powershell
$env:HIP_VISIBLE_DEVICES = '1'
C:\Users\user\dev\vLLM_for_AMD\.venv-rocm10\Scripts\python.exe tools\reference\full_logits_golden.py `
    --device cuda --tokens tools\reference\kl_corpus\tokens.json `
    --out-dir tools\reference\kl_out\ref
```

Runs the **original bf16 checkpoint end to end** -- all 64 decoder layers, the final norm and the
full `[248320, 5120]` lm_head -- over each segment of the tokens file, and writes the shared format
above with `"source": "reference"`.

**How a 51.7 GiB checkpoint runs on a 32 GiB card (peak 1.56 GiB).** Nothing is ever fully
resident. The model skeleton is built on the `meta` device (no storage at all), and then
`Qwen3_5TextModel.forward` -- transformers' own, unmodified, so the embedding path, the mask
builders, `Qwen3_5TextRotaryEmbedding`, the layer ordering and the final norm are all the real
thing -- walks a `LazyDecoderLayers` object in place of `self.layers`. Its `__getitem__(slice)`
returns a generator that materializes one `Qwen3_5DecoderLayer`'s real weights from the safetensors
shards onto HIP device 1 (`load_state_dict(..., assign=True)` onto a meta-constructed module, so
the ~750 MiB of parameters are never randomly initialized first), yields it to the forward loop,
and drops the last reference to it the moment the loop asks for the next one. Only the `[T, 5120]`
bf16 hidden state survives from one layer to the next. `embed_tokens` is never built or called:
the sequence's rows are gathered straight out of the shard's mmap with `get_slice` and handed in as
`inputs_embeds`. The lm_head is streamed the same way, `--lm-head-chunk` vocabulary rows at a time
(default 32768 rows = 320 MiB bf16), accumulating into one `[T, 248320]` fp32 buffer; the per-chunk
matmul is done in bf16 exactly as `nn.Linear` does it inside the real
`Qwen3_5ForConditionalGeneration.forward`, so only the accumulator is wider than the reference's.
`log_softmax` then runs in fp32 over `--row-block` sequence rows at a time, gets the `-1e4` floor,
and is cast to fp16 and appended to the output file.

**`--impl model` vs `--impl manual`.** Two independent implementations of the *composition* around
the decoder layers exist in this file, and `--cross-check` runs both:

- `model` (default): transformers' `Qwen3_5TextModel.forward` as described above.
- `manual`: a hand-rolled loop in this file with its own partial-rope cos/sin derived from the
  config (`rope_theta=1e7`, `partial_rotary_factor=0.25` -> rotary dim 64 of head_dim 256), its own
  additive causal mask, its own zero-centered RMSNorm (`x/rms * (1 + w)` in fp32), and its own
  layer ordering. Only the `Qwen3_5DecoderLayer` modules themselves are shared.

**rope / mrope.** This corpus is text-only, so `get_rope_index`'s three mrope axes (t, h, w) all
carry the same sequential index. `Qwen3_5TextModel.forward` builds exactly that itself when
`position_ids is None` (`arange(T).view(1,1,-1).expand(4, B, -1)`; row 0 is the text index used for
masking, rows 1..3 go to the rope module), and
`Qwen3_5TextRotaryEmbedding.recomposition_frequencies` then interleaves the h and w sections out of
tensors that are elementwise identical to the t tensor -- so the recomposition is the identity and
this reduces to plain rope over the first 64 dims. `--impl manual` computes that reduced form
directly from the config, which is why the two agreeing is a real check on the claim and not a
tautology. (Image-carrying prompts are a different story and are covered by
`rope_index_golden.py` / `mrope_layer_golden.py`, not here.)

**Attention** is `eager` (`common.load_text_config` forces it), same as every other golden in this
directory: at T=2048 with 24 query heads that is a `[1, 24, 2048, 2048]` score matrix, ~400 MiB in
fp32 inside the softmax, which fits comfortably. No cache is used anywhere -- every segment is one
full-sequence causal prefill from position 0.

### Validation

Three checks, all run on the real checkpoint on HIP device 1. Reproduce with:

```powershell
$env:HIP_VISIBLE_DEVICES = '1'
C:\Users\user\dev\vLLM_for_AMD\.venv-rocm10\Scripts\python.exe tools\reference\full_logits_golden.py `
    --device cuda --segment english_prose --cross-check 48 --selfcheck-greedy 32 `
    --selfcheck-prefix 48 --skip-segments --out-dir tools\reference\kl_out\validation
```

**(i) 48-token prompt, last-position argmax and top-5.** The task's intended cross-check was
`tools/reference/first_token.py`, which loads the whole model through
`AutoModelForCausalLM.from_pretrained`. That is not runnable on this machine: the bf16 checkpoint
is 51.7 GiB and the box has 63.5 GiB of RAM total (42.6 GiB free at the time of this run), so a CPU
load thrashes, and the `device_map="auto"` path would have to put ~20 GiB on device 1, far over
this task's ~12 GiB reference-side budget while another stage shares the card. `first_token.py`'s
own docstring already records that it was never executed for this reason. The cross-check actually
run instead is the two-composition one described above: the same 48 tokens through `--impl model`
and `--impl manual`, whose top-5 ids **and full 248320-way logit vectors** must agree. They agree
exactly -- max absolute logit difference `0.0000e+00`, i.e. bit-identical, which also means the
manual rope/mask/norm derivation reproduces transformers' own bit for bit:

```
[full_logits] cross-check: 48 tokens through --impl model and --impl manual
  [model] 41.0s top-5:
    #1: id=    198 logit= 21.2500 prob= 0.6909 piece='\n'
    #2: id=    264 logit= 19.6250 prob= 0.1360 piece=' a'
    #3: id=   1379 logit= 18.3750 prob= 0.0390 piece=' most'
    #4: id=   1603 logit= 17.8750 prob= 0.0236 piece=' good'
    #5: id=    279 logit= 17.6250 prob= 0.0184 piece=' the'
  [manual] 28.4s top-5: (identical ids/logits/probs)
  top-5 ids agree: True  (model=[198, 264, 1379, 1603, 279] manual=[198, 264, 1379, 1603, 279])
  max |logit difference| over the whole 248320-way vocabulary: 0.0000e+00
```

**(ii) Greedy self-consistency.** 32 tokens are generated greedily by the streaming forward (one
full 64-layer pass per token, ~26 s each -- slow by design), then the whole 80-token sequence is
re-fed as one fixed context and the argmax at every generated position must reproduce the token
that was generated there. This is what catches an off-by-one in the position / rope / mask wiring
that a single forward pass cannot see. The generated text is printed, and reading it is itself part
of the check -- a stack with a subtly wrong layer order still produces *some* argmax at every step,
but it does not produce fluent English:

```
  step 32/32: id=    557 ' was' (28.3s)
  re-fed the 80-token sequence in 28.2s: 32/32 positions reproduce their token
  generated text: '\nmost of the history of computing, that was the right mental model. The CPU was the bottleneck,\nand the number of instructions it had to execute was'
```

The 48-token prefix is the opening of `kl_corpus/english_prose.txt`, which is an essay about the
memory hierarchy -- the continuation is on topic *and* argues the same point, which no amount of
coincidence produces from a mis-wired stack. The whole validation run (one cross-check pair plus 33
streaming passes) takes ~16 minutes.

**(iii) `logsumexp` of every written row within 1e-2 of 0.** Measured on the **fp16 values actually
written**, not on the fp32 intermediate, and reported per segment as `max_abs_logsumexp_fp16` in
the sidecar. On the four 1024-token segments the worst row was `1.65e-03`.

Gate run (2026-09-22, HIP device 1, shared with another stage):

```
[full_logits] skeleton ready in 0.5s: 64 layers, hidden=5120, V=248320, impl=model
[full_logits] cpp_source: T=1024 V=248320 stack=37.7s lm_head=1.6s peak=1.56GiB |logsumexp|max=1.06e-03 ppl=3.802 -> cpp_source.logprobs.f16 (484.5 MiB)
[full_logits] english_prose: T=1024 V=248320 stack=26.4s lm_head=1.4s peak=1.56GiB |logsumexp|max=8.80e-04 ppl=3.761 -> english_prose.logprobs.f16 (484.5 MiB)
[full_logits] python_source: T=1024 V=248320 stack=24.9s lm_head=1.5s peak=1.56GiB |logsumexp|max=1.65e-03 ppl=4.606 -> python_source.logprobs.f16 (484.5 MiB)
[full_logits] thai_prose: T=1024 V=248320 stack=26.4s lm_head=1.5s peak=1.56GiB |logsumexp|max=9.45e-04 ppl=15.589 -> thai_prose.logprobs.f16 (484.5 MiB)
```

Runtime: **~28 s per 1024-token segment** (~26 s of that is streaming 48 GiB of layer weights off
disk; the first segment of a run pays a cold page cache and costs ~38 s), ~2 min for all four.
Peak VRAM **1.563 GiB allocated / 3.424 GiB reserved** (the allocator's high-water mark; the
difference is caching, not live tensors) -- the 27B bf16 model is 51.7 GiB on disk.
Each segment's output is 484.5 MiB, so a four-segment run writes ~1.9 GiB per side.

The bf16 reference's own perplexities on this corpus (a useful sanity floor for the engine side --
r4dx should land near these, not far below them): C++ 3.802, English 3.761, Python 4.606, Thai
15.589.

Key options: `--segment NAME` (repeatable/comma-separated), `--max-tokens N` (truncate every
segment), `--impl {model,manual}`, `--lm-head-chunk`, `--row-block`, `--top-k`, `--skip-segments`
(validations only), and `--debug-max-layers N`, which runs only the first N layers for timing
experiments and prints a loud warning that the outputs are **not** a valid golden.

## kl_report.py

```powershell
C:\Users\user\dev\vLLM_for_AMD\.venv-rocm10\Scripts\python.exe tools\reference\kl_report.py `
    --ref-dir tools\reference\kl_out\ref --test-dir tools\reference\kl_out\r4dx `
    --tokens tools\reference\kl_corpus\tokens.json --out tools\reference\kl_out\kl.json
```

Pairs the two dumps and reports, per segment and overall: mean / median / 99th-percentile / max KL
in nats (with the position and token where the max occurs), top-1 agreement %, top-5 containment %
(is the *reference's* argmax inside the *test's* top-5), each model's perplexity over the corpus,
and the number of positions whose KL exceeds 1 nat. A markdown table goes to stdout; `--out` also
writes the whole thing as JSON.

Everything is computed in fp64, per position, over the **full** vocabulary:

```
KL(P_ref || Q_test) = sum_v p_ref[v] * (logp_ref[v] - logq_test[v])
```

Neither `[1023, 248320]` fp16 file (485 MiB each) is loaded whole, let alone widened to fp64 whole:
`numpy.memmap` plus a `--row-chunk` (default 32 rows = 2 x 60 MiB of fp64 scratch) walks the two in
lockstep. A wrong-sized `.logprobs.f16` is a hard error, not a misread: the file's byte length must
be exactly `rows * V * 2`.

Because the inputs are fp16, `sum_v exp(logp_ref[v])` is only approximately 1; the report prints
the worst per-segment deviation as a diagnostic (expect ~1e-3, and suspect a format or clamp bug if
it is ~1).

**`--self-test`** (no files, no GPU, no checkpoint) checks the arithmetic against a closed form:

```
p = [1/2, 1/4, 1/8, 1/8], q = [1/4, 1/4, 1/4, 1/4]  ->  KL(p||q) = (1/4) ln 2 = 0.17328679514
```

first through `chunk_stats` directly in fp64 (must match to ~1e-16), then through the *whole file
pipeline* -- the pair is written out as real `.logprobs.f16` + `.meta.json` files in a temp
directory and read back through `compare_segment`, where it only has to hold to fp16 precision
(~1e-4 on the KL, ~1e-3 relative on the perplexity). A third case with genuinely different argmaxes
exercises the top-1 / top-5 wiring, and a fourth truncates one file to confirm the size check
fires:

```
[self-test] 1. chunk_stats fp64 KL     = 0.173286795140 (analytic 0.173286795140, max err 2.776e-17)
[self-test]    NLL_ref wiring          max err 0.000e+00
[self-test]    asymmetric pair KL      = 1.167546089433 (analytic 1.167546089433), top1_agree=False (expect False), top5_contain=True (expect True, V=4 < 5)
[self-test] 2. end-to-end file KL     = 0.173376598 (analytic 0.173286795, err 8.980e-05, fp16 inputs)
[self-test]    ppl_ref               = 4.759100090 (analytic 4.756828460, rel err 4.776e-04)
[self-test] 3. truncated file detected: ValueError
[self-test] PASS
```

**Identity smoke check on real data.** Passing the same directory as both `--ref-dir` and
`--test-dir` must produce exactly zero KL and 100% agreement across all four 1024-token segments,
and the perplexities it reports must equal the ones `full_logits_golden.py` computed independently
in its own sidecars (3.802 / 3.761 / 4.606 / 15.589). It does, in ~21 s for 4 x 2 x 485 MiB:

```
| segment | rows | mean KL | median KL | p99 KL | max KL | max @ | top-1 | top-5 | ppl ref | ppl test | KL>1 |
| cpp_source | 1023 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | pos 0 (tok 328) | 100.00% | 100.00% | 3.802 | 3.802 | 0 |
| **ALL** | 4092 | **0.00000** | 0.00000 | 0.00000 | 0.00000 | cpp_source pos 0 | **100.00%** | 100.00% | 5.661 | 5.661 | 0 |
```

The same run's fp16 round-trip diagnostic (`1.06e-03`, `8.80e-04`, `1.65e-03`, `9.44e-04`) matches
each segment's `max_abs_logsumexp_fp16` sidecar field to the digit, which is what it should be:
both are measuring the same fp16 rounding of the same rows from two different directions.

## tok_golden.py

Not this component's -- owned by the tokenizer agent. Don't add it here.

## Tests

`tests/reference/test_manifest.py` is CPU-only, needs no checkpoint, and has **no pytest
dependency** (a plain script with bare asserts and a `__main__` entry, since the reference venv --
the only python on this machine with `torch`+`transformers` importable together -- doesn't have
`pytest` installed and is read-only). Run it directly:

```powershell
C:\Users\user\dev\vLLM_for_AMD\.venv-rocm10\Scripts\python.exe tests\reference\test_manifest.py
```

It's also registered as ctest test `reference_manifest` in `tests/CMakeLists.txt`, so
`.\tests\run_tests.ps1`'s `ctest --preset win-hip` picks it up automatically alongside the C++
tests once the rest of the project configures cleanly. It checks every component ran `status=ok`
with honest `random_init` provenance, that each `.safetensors` file was written and non-empty, the
full expected tensor-name set per component (including the r4d-shaped GDN tensors and post-rope
attention tensors above -- a regression that silently drops one now fails this test instead of only
being noticed by whoever diffs against the missing tensor), that the decode GDN conv output's
trimmed length matches `--decode-len`, and that the manifest's tolerance table is well-formed.
