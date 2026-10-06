# Gemma 4 (gemma4_unified) semantics, verified against the HF source

Task M0-5 of `docs/gemma4-plan.md`. Every item the plan tagged UNVERIFIED (section 3.1 and the
scattered tags in sections 2, 4, 5) is settled here from the installed source, with file:line.

- Source: `transformers 5.18.0` in `D:\venvs\r4dx-gemma-ref` (torch 2.9.1+rocmsdk20260116, the same wheels as
  Python312). Paths below are relative to `Lib\site-packages\transformers\`. `M` = `models\gemma4_unified\modeling_gemma4_unified.py`,
  `MU` = `masking_utils.py`, `RU` = `modeling_rope_utils.py`, `CFG` = `models\gemma4_unified\configuration_gemma4_unified.py`,
  `IMG` = `models\gemma4_unified\image_processing_gemma4_unified.py`, `PROC` = `models\gemma4_unified\processing_gemma4_unified.py`.
- Checkpoint facts come from `tools/reference/gemma/tensor_names.json` (safetensors header of the Huihui file,
  read both over HTTP and from the downloaded file; identical) and `E:\models\Huihui-gemma-4-12B-it-abliterated\config.json`.
- Mask / rope / embed-scale claims are backed by a CPU run of `tools/reference/gemma/rope_masks_golden.py`, which
  writes `tools/reference/golden_out/gemma/mask_semantics.json` (tiny 2-layer config, window 4, 8 tokens; the 12B rope
  tables). A copy is force-added to git because it is 30 KB and deterministic.

Status key: **VERIFIED** (read in source, and run where noted), **CORRECTS PLAN** (plan text was wrong).

## 1. Items from plan section 3.1 / M0-5

| # | Item | Status | Finding | Evidence |
|---|---|---|---|---|
| 1 | `v_norm` scope | VERIFIED | `v_norm` (RMSNorm, **no weight**) is applied to V on **every** layer (sliding and full), after V is chosen. Sliding: V = `v_proj(x)`. Full (`attention_k_eq_v`, no `v_proj`): V = the raw `k_proj(x)` output, **before** `k_norm` and rope (K gets `k_norm` + rope as a separate, non-in-place tensor). No `v_norm` tensor exists in the checkpoint. | `M:377` (module), `M:386-388` (`v_proj` None when `attention_k_eq_v and not sliding`), `M:422-423` (V = `v_proj` or the same `key_states`), `M:425-427` (K path), `M:429` (v_norm unconditional); header: layer 5 has no `v_proj`/`v_norm` |
| 2 | attention `scaling = 1.0` | VERIFIED | `self.scaling = 1.0` for all layers, passed to the kernel as `scaling=`. No attention logit softcap (`softcap` is not passed; default None). | `M:357`, `M:441-451`, `M:308-339` |
| 3 | Embed scale dtype | VERIFIED | `embed_scale = hidden_size**0.5` held as an fp32 tensor and cast with `.to(weight.dtype)` before the multiply, so in bf16 it is **62.0** (fp32 value 61.9677). The multiply happens in bf16 on the gathered row. | `M:528-539`, `M:590-591`; golden JSON `rope_and_scale.embed_scale` = f32 61.967735, bf16 62.0 |
| 4 | Sliding mask boundary | VERIFIED | `kv_idx > q_idx - sliding_window` AND causal `kv_idx <= q_idx`: a query sees `sliding_window` keys **including itself** (distances 0..1023 for 1024). Golden: window 4, query 7 sees keys 4..7. (`use_bidirectional_attention == "all"` would halve the window; the checkpoint uses `"vision"`.) | `MU:94-103`, `MU:78-82`, `MU:136-140`; `CFG:153-155`; golden `sliding_boundary` |
| 5 | Proportional rope layout | VERIFIED | Full layer: `inv_freq` has length `head_dim/2 = 256`; `rope_angles = int(0.25*512//2) = 64`; `inv_freq[i] = 1/1e6^(2i/512)` for i < 64 (the exponent denominator is the **full** head_dim 512, not 128) and **zeros** for the other 192; cos/sin are `cat(freqs, freqs)` and the rotation is rotate-half, so pairs are `(i, i+256)` for i < 64 and every other dim has cos 1 / sin 0 (identity). Sliding: default rope, theta 1e4, head_dim 256, 128 freqs, pairs `(i, i+128)`. `attention_scaling` = 1.0 for both. Checked numerically against an explicit pair loop (max abs err 3.6e-5 at position 1000, fp32 angle noise). | `RU:193-266` (esp. 246-263), `M:185-264` (cos/sin build 259-262), `M:267-293` (rotate_half / apply); golden `rope_and_scale` (`full_inv_freq_matches_formula`, `full_rope_pairs_...max_abs_err`, `sliding_inv_freq_matches_formula`) |
| 6 | Image-token bidirectional mask, per layer type | VERIFIED, with an HF inconsistency to decide on (see section 2) | Intended and `generate()` semantics: **sliding layers only**; full layers stay purely causal. A direct `Gemma4UnifiedModel.forward()` call with `mm_token_type_ids` and no mask dict applies the block overlay to **both** layer types. | `M:1205-1265` (docstring `M:1213-1220`: "Unlike Gemma 3 ... Gemma 4 explicitly disables bidirectional attention on global attention layers"), `M:1375-1404`, `M:1105-1126`; golden `generate_path` vs `forward_path` |
| 7 | 1-D image positions | VERIFIED | `position_ids = arange(T) + past_len` over the whole expanded sequence (boi, image soft tokens, eoi count as ordinary positions). No mrope, no per-image 2-D rope; the 2-D position only enters through the vision embedder's `pos_embedding`. | `M:1100-1103`, `M:626-629`, `M:651` |
| 8 | `layer_types` | VERIFIED | Explicit list in `config.json`: `full_attention` at 5, 11, 17, 23, 29, 35, 41, 47, `sliding_attention` elsewhere (40 + 8). HF default is the same 5:1 rule `(i+1) % 6 == 0`, and the last layer is forced to full. Header agrees: `v_proj` is absent at exactly those 8 layers. Assert the pattern at load. | `CFG:157-168`; `config.json` `text_config.layer_types`; `tensor_names.json` `summary.v_proj_absent_layers`; golden JSON `bos_and_layer_types.pattern_is_5_to_1_full_at_5_11_to_47` true |
| 9 | rope scaling / max_position | VERIFIED | No scaling anywhere: `rope_parameters` has no `factor`/yarn keys (factor defaults to 1.0) and `dynamic_rope_update` only reacts to `dynamic*`/`longrope` rope types. `max_position_embeddings` is used only to set `max_seq_len_cached`/`original_max_seq_len`; it does not change `inv_freq`. So 131072 (Huihui) vs 262144 (google) is a **metadata-only** difference: running at 262144 needs no rope change in r4dx. (It is the only difference between the two config.json files, see section 4.) | `RU:34-130` (122-127), `RU:241,265`, `M:189-190`, `config_diff.json` |
| 10 | BOS on raw prompts | **CORRECTS PLAN** | HF does **not** add BOS: `tok("hi").input_ids == [2202]`. The `tokenizers` post-processor is a `TemplateProcessing` with only the `$A` sequence and no special tokens, and `tokenizer_config.json` has no `add_bos_token`. BOS only appears when the text contains `<bos>`: the chat template emits `<bos>` as text and `tok("<bos>hi")` gives `[2, 2202]` (no double BOS when the rendered template is tokenized, `add_special_tokens` either way). So `bos_on_raw_prompt()` for HF parity is **false**; if r4dx prepends BOS to raw `/v1/completions` prompts it is a deliberate deviation. KL tokens must add BOS=2 explicitly on both sides (plan decision 8). | `tokenizer.json` `post_processor`; `tokenizer_config.json`; golden JSON `bos_and_layer_types` (`hi_default` [2202], `bos_literal_hi` [2, 2202], `raw_adds_bos` false) |
| 11 | `layer_scalar` placement and shape | VERIFIED, **CORRECTS PLAN** (shape) | A per-layer buffer of shape **`[1]`**, stored **BF16** in the checkpoint (plan said fp32 `[1,4]`, UNVERIFIED). Applied **once per layer**, in place, after the MLP residual add: `h *= layer_scalar`. The attention half is not scaled (the post-norm kernel must get 1.0 there). In bf16: `bf16(residual + bf16(post_norm(y)))` then `* layer_scalar` in bf16. | `M:490`, `M:514-524`; header `layer_scalar` x48, all `[1]` BF16 |
| 12 | RMSNorm form | VERIFIED | `x * (mean(x^2)+eps)^-0.5` in fp32, then `* weight.float()`, then **one** cast to the input dtype (not (1+w); the rounding point is after the weight multiply). eps 1e-6. `with_scale=False` variants (v_norm, multimodal pre-projection norm) skip the weight. | `M:164-182` |
| 13 | GeGLU | VERIFIED | `down(act(gate(x)) * up(x))`, `act = gelu_pytorch_tanh` (from `hidden_activation`), intermediate 15360, no bias. No double-wide MLP and no KV sharing in this checkpoint (`use_double_wide_mlp` false, `num_kv_shared_layers` 0). | `M:458-474`, `M:362-367` |
| 14 | Final logit softcap | VERIFIED, **flag** | `tanh(logits/30)*30` is applied to the **lm_head output in its own dtype**: with a bf16 model that is bf16 logits, softcapped in bf16 (bf16 spacing near |x|=30 is 0.125). The plan's reference tool says "softcap in fp32 before log_softmax": that is more accurate than HF and is **not** what HF computes. `full_logits_gemma.py` must pick one explicitly (recommend: compute lm_head in fp32 accumulate, softcap in fp32, and also record the HF-faithful bf16 variant so the noise floor is known). | `M:1352-1358` |
| 15 | Tied lm_head | VERIFIED | `lm_head.weight` tied to `model.language_model.embed_tokens.weight`; the checkpoint has **no** `lm_head` tensor. | `M:1275`; header `lm_head_present: false` |
| 16 | GQA geometry per layer type | VERIFIED | Sliding: 16 q / 8 kv, head_dim 256 (groups 2). Full: 16 q / 1 kv (`num_global_key_value_heads`, applied because `attention_k_eq_v`), head_dim 512 (`global_head_dim`), groups 16. `is_causal` true for `use_bidirectional_attention == "vision"`. | `M:353-359`, `CFG:177-190`; header shapes (layer 0 k_proj 2048x3840, layer 5 k_proj 512x3840, q_proj 8192x3840) |

## 2. Image mask: the one decision this leaves

Facts (golden `mask_semantics.json`, one image at positions 2..5, window 4; `1` = attend, row = query):

```
generate() path                     direct Model.forward() path
sliding          full               sliding          full
10000000         10000000           10000000         10000000
11000000         11000000           11000000         11000000
11111100         11100000           11111100         11111100   <- query 2 sees keys 3..5 (future image keys)
11111100         11110000           11111100         11111100
01111100         11111000           01111100         11111100
00111100         11111100           00111100         11111100
00011110         11111110           00011110         11111110
00001111         11111111           00001111         11111111
```

- Block ids come from `mm_token_type_ids` (1 = image, 2 = video): a block is a maximal run of such tokens. boi/eoi
  are type 0, so each image is its own block and a block never reaches the boi/eoi tokens; two images do not see each
  other's future tokens (golden `two_images_do_not_see_each_other_ahead`). Audio tokens are type 3, never a block:
  audio stays causal (golden `audio_block_stays_causal`). `M:862-871`, `processing_utils.py:926-956`.
- Sliding layer rule on the generate path is `AND(window, OR(causal, same-block))`; on the direct-forward path it is
  `OR(AND(window, causal), same-block)`. They are identical here because a block (<= 280 tokens) is far inside the
  window (1024).
- At decode time the mask is plain causal: `prepare_inputs_for_generation` drops `mm_token_type_ids` after the first
  iteration (`M:1406-1414`).
- **Decision recorded for M2:** implement **sliding only** (`klimit_ext` on the sliding kernel; full layers stay causal).
  This is what `generate()` does and what the HF docstring states; the rung-3/M2 reference run must therefore pass the
  mask dict from `Gemma4UnifiedForConditionalGeneration.create_masks_for_generate` (or call `generate()`), not rely on
  a bare `forward()` with `mm_token_type_ids`, which differs on the 8 full layers. If an M2 real-image comparison is
  ambiguous, the full-layer variant is a one-line mask switch.
- Consequence for the kernel plan: the `klimit_ext` hook is needed only in the h256 sliding kernel; the h512 full
  kernel needs no bidirectional support. The 288-row image chunk still applies (keys are written for the whole block
  before attention).

## 3. Vision, audio, processor facts (plan section 2 and M2/M3 inputs)

All VERIFIED; they match plan section 2 unless noted.

- **Image preprocess** (`IMG:54-105`, `IMG:214-362`): bicubic, antialiased resize (`IMG:264-269`) to the largest
  `(h, w)` multiples of 48 with `h*w <= max_soft_tokens*9*256 = 645120` (280 -> max_patches 2520); rescale 1/255, no
  mean/std (`IMG:216-223`); 16 px patches, 3x3 merged into 48x48x3 **HWC** rasters of 6912 values (`IMG:108-119`,
  `IMG:139-210`; vertical index is the major axis inside a merged patch); merged patches are in row-major order of the
  merged grid, positions are `(x, y)` in merged-grid units; padded to 280 patches with position `(-1,-1)`; the model
  drops padded patches before scattering (`M:929-935`). Soft tokens = `(h/48)*(w/48)` (`PROC:252-254`).
- **Embedder** (`M:764-826`): pixel values cast to the weight dtype (bf16) -> `patch_ln1` LayerNorm(6912) -> `patch_dense`
  Linear(6912->3840, bias) -> `patch_ln2` LayerNorm(3840) -> `+ pos_embedding[x, 0] + pos_embedding[y, 1]` (table
  `[1120, 2, 3840]`, summed over the two axes, padding masked) -> `pos_norm` LayerNorm(3840) -> `multimodal_embedder`:
  RMSNorm(3840, no weight, eps 1e-6) -> `embedding_projection` Linear(3840->3840, no bias) (`M:829-859`). LayerNorms are
  torch defaults (eps 1e-5, with weight and bias).
- **Splice** (`M:1022-1053`): placeholder ids are replaced by pad (0) for the gather, then overwritten via
  `masked_scatter` with the **unscaled** embedder output (text rows carry the `sqrt(3840)` scale, image/audio rows do not).
- **Prompt expansion** (`PROC:179-181`): `<|image>` + N x `<|image|>` + `<image|>`. Audio (`PROC:202-211`, `PROC:295-315`):
  `<|audio>` + N x `<|audio|>` + `<audio|>` with `N = ceil(samples / 640)`.
- **Audio** (`feature_extraction_gemma4_unified.py:67-93`, `M:1152-1176`, `M:829-859`): zero-pad the 16 kHz waveform to a
  multiple of 640 on the right, reshape to `[n, 640]`, RMSNorm(640, no weight, eps 1e-6) -> Linear(640->3840, no bias);
  every frame is a valid token; no scale. The 750 cap is only a processor-level `audio_seq_length` upper bound
  (`PROC:99-102`), not something the model enforces. Audio attends causally (see section 2).
- **Checkpoint names vs module names.** On disk: `model.vision_embedder.{patch_ln1,patch_dense,patch_ln2,pos_embedding,pos_norm}`,
  `model.embed_vision.embedding_projection.weight` [3840,3840], `model.embed_audio.embedding_projection.weight` [3840,640].
  HF renames them on load (`conversion_mapping.py:285-299`) to `embed_vision.*` and
  `embed_vision.multimodal_embedder.embedding_projection`. Converter prefixes (plan 4.2) should key off the **on-disk**
  names: `model.vision_embedder.*` and `model.embed_vision.*` -> `vision.*`, `model.embed_audio.*` -> `audio.*`.

## 4. Checkpoint header and config diff (M0-4)

Full data: `tools/reference/gemma/tensor_names.json`, `config_diff.json`.

- 677 tensors, all BF16, one `model.safetensors` (23,919,549,376 bytes; sha256 `2dd13d86...c7e723` matches the HF LFS
  hash), no `__metadata__`, header 88,928 bytes. No `lm_head`. No `v_norm` tensor.
- Text: `model.language_model.{embed_tokens.weight [262144,3840], norm.weight [3840]}` and, per layer,
  `input_layernorm`, `post_attention_layernorm`, `pre_feedforward_layernorm`, `post_feedforward_layernorm` ([3840]),
  `layer_scalar` ([1]), `mlp.{gate,up}_proj` [15360,3840], `mlp.down_proj` [3840,15360], `self_attn.{q,k,o}_proj`,
  `self_attn.{q,k}_norm.weight` ([head_dim]), and `self_attn.v_proj` only on the 40 sliding layers.
  Sliding: q 4096x3840, k/v 2048x3840, o 3840x4096, norms [256]. Full: q 8192x3840, k 512x3840, o 3840x8192, norms [512].
- Non-text: 9 `model.vision_embedder.*` tensors, `model.embed_vision.embedding_projection.weight`,
  `model.embed_audio.embedding_projection.weight` (names and shapes in section 3).
- **Huihui vs current google/gemma-4-12B-it `config.json`: exactly one difference**,
  `text_config.max_position_embeddings` (google 262144, Huihui 131072). Huihui and google `generation_config.json` and
  `processor_config.json` are byte-identical; `tokenizer_config.json` (2102 vs 3089 bytes) and `chat_template.jinja`
  (17466 vs 18683 bytes) differ, which is why the assembled tokenizer dir uses google's (plan 5.1/5.2).
  `config.json` `eos_token_id` is `[1, 106]`; `generation_config.json` gives `[1, 106, 50]` (plan section 9.4: use the latter).
  Note also `eoa_token_index: 258883` (sic) in the top-level config.

## 5. Corrections to the plan text (for whoever edits it next)

**Status: all seven are applied to `docs/gemma4-plan.md`** (branch g4-convert-trellis): the 4.2 `layer_scalar`
and vision/audio rows, 5.2 / 5.3 `bos_on_raw_prompt` (false), 3.1 and section 7 item 3 (image mask: sliding
only), 6.3 `full_logits_gemma.py` (softcap dtype), the tensor-name bullet, the 2.1 `max_position_embeddings`
note and the rope denominator. The plan's 3.1 "UNVERIFIED" list is now the settled list.

1. `layer_scalar` is `[1]` BF16, not fp32 `[1,4]` (4.2 table).
2. Raw HF tokenization does not add BOS (`bos_on_raw_prompt` for parity = false); section 5.2/5.3 and the dialect table say true.
3. Image bidirectional attention: sliding only (settled, with the direct-forward caveat above).
4. HF applies the final softcap in the logits dtype (bf16), not fp32 (6.3 `full_logits_gemma.py`).
5. Tensor prefixes confirmed: `model.vision_embedder.*`, `model.embed_vision.embedding_projection`, `model.embed_audio.embedding_projection`
   (plan said "maybe"); no `lm_head`.
6. `max_position_embeddings` is metadata only (no rope scaling), so 262144 opt-in is a runtime/KV-sizing flag, not a numerics change.
7. Plan item "rope: pairs i, i+256 for i<64" is confirmed; note the frequency denominator is 512, not 128.

