# Image input for Ornith-1.5-122B (`qwen3_5_moe`)

Written 2026-10-23, branch `q35moe-vision` (from `qwen35moe` at `0d8b2e1`). Everything below was built and measured on this machine; numbers are
quoted from the runs named next to them. Plan item: `docs/qwen35moe-plan.md` M6 / section 9 P3.

## What was needed (and what was not)

The vision tower of Ornith is **the 27B's tower**: `Qwen3_5MoeVision*` in transformers 5.18 is line-for-line identical to `Qwen3_5Vision*`
(checked by diffing the class sources), the 333 `model.visual.*` tensors have the same names and shapes, `vision_config` is the same
(depth 27, hidden 1152, 16 heads, intermediate 4304, patch 16 x temporal 2, 2x2 merger, learned 48 x 48 position grid, rope 1e4) except
`out_hidden_size` 3072 (the merger's `linear_fc2` is `[3072, 4608]`), and `preprocessor_config.json` is the same (65536 .. 16777216 pixels,
mean / std 0.5, bicubic). `get_rope_index` is unchanged too. So `src/vision` (host preprocessing, `VisionTower`, `BuildMropePositionIds`)
is reused **unchanged**; the work is the converter flag, a small owner class, and the text-side integration that the dense `Model` already has
and `Qwen35MoeModel` did not:

| piece | where | what |
|---|---|---|
| converter | `src/convert/main.cpp` | `--vision on` for `qwen3_5_moe`: copies `model.visual.*` as `vision.*` bf16 (default stays off: a container converted without the flag is byte-identical to before) |
| tower owner | `src/model/qwen35moe_vision.{h,cpp}` (new) | `Qwen35MoeVision`: `VisionWeights` + `VisionTower` on one device, `Encode` returns HOST bf16 rows `[tokens, 3072]` |
| model | `qwen35moe_model.{h,cpp}` | `EncodeImages`, `PrefillMultimodal`, `--vision auto|on|off` option, mrope state (`mrope_active_`, delta, block rows) through decode, verify windows, checkpoints and the DFlash2 drafter's `SetRopeDelta` |
| stage | `qwen35moe_stage.{h,cpp}` (~20 lines) | `Qwen35MoeChunk::rope_pos3` (host `[3, rows]`) is uploaded once per chunk and passed to every attention layer as `rope_pos3` |
| TextModel | `qwen35moe_local_text_model.{h,cpp}` | `HasVision / ImageTokenId / VisionMergeSize / EncodeImages / PrefillMultimodal` |

The CLI (`--image`, `--vision`, `--image-max-pixels`) and the server (`image_url` content parts, prefix reuse with image keys,
`/v1/models` modalities) needed **no change**: they go through `TextModel`.

### How an image enters the text model

* **Embeddings.** The model's embedding gather is a host gather into a pinned chunk buffer (`RunChunk`). `PrefillMultimodal` keeps the
  call's image spans; for every chunk that overlaps a span the placeholder rows are overwritten in that buffer with the tower's rows (a
  `memcpy`, no kernel). Hence the tower returns host rows and may live on any card.
* **Positions.** The attention layers rope with the 3-axis interleaved mrope `[11, 11, 10]`. `positions` doubles as the KV slot (slot ==
  position), so it cannot carry the shifted rope positions; after an image **every chunk** (prefill, one-token decode, verify window) carries
  explicit `[3, T]` rows: inside the call's block the rows of `BuildMropePositionIds` (the larger merged side advances the position, not the
  token count), everywhere else `sequence index + mrope_delta` on all three axes. The delta is kept across `Reset` (cleared), the prompt
  checkpoint (saved and restored) and the DFlash2 drafter (`SetRopeDelta`).
* **GDN layers** (36 of 48) have no positions and no change: the spliced rows flow through them in order.
* A text-only conversation never sets `mrope_active_`: no rows are uploaded, the attention layers take the same fused path as before.
  After an image the decode path takes the unfused rope path (as the dense model does), which costs nothing measurable here.

## Container

`E:\models\r4dx\ornith-1.5-122b-trellis-mix-kv-vision.r4dx` (57.43 GB = the 56.53 GB kv container + 0.90 GB of `vision.*`), converted with
the flags of the pipeline's convert stage (`tools/quant2/qwen35moe/pipeline.ps1`, `Stage-Convert`) plus `--vision on`:

```
r4dx-convert --input E:\models\r4dx\Ornith-1.5-122B --output ...-kv-vision.r4dx --trellis-from E:\models\r4dx\q35moe\mix `
  --trellis-manifest-sha256 6fe385b70d8efeca8b3de12a18839c54ab9357dff82de69139c2e65a4cbad37c --trellis-verify full --layouts w4a16 `
  --no-bf16 --lm-head w4a16 --w4a16-group-rule '^lm_head$=32' --threads 32 --kv-calib E:\models\r4dx\q35moe\kvcalib.json --vision on
```

322 s of CPU (37020 / 37020 trellis tensors verified). **Checked against the existing kv container:** the same 74800 non-vision tensors
with identical names, shapes and offsets, the whole text data region byte-identical (56.5 GB compared), only `__metadata__.r4dx_convert_run`
differs. Both older containers are kept. `model_config.vision_config` and `image_token_id` (248056) were already in the metadata.

## Placement and memory

The weights fill both cards (25.93 + 25.57 GiB), so the tower (0.841 GiB of weights, measured by the loader; scratch 0.04-0.35 GiB depending
on the image, `docs/vision.md` "Large images") goes on the **last stage's card (HIP device 1, headless)**, not on the desktop card;
`R4DX_Q35_VISION_DEVICE` overrides. With the tower, a 1M-pixel image and `--max-ctx 8192`, the CLI reports 26.85-27.03 GiB used (one run
each, `--stats`), i.e. it fits next to the weights with the same ~5 GiB headroom the text-only run has, less the tower. `--vision off`
never loads it; `--vision auto` (default) loads it iff the container has `vision.*`; `--vision on` throws without. The tower stays
resident (load-on-demand was not needed: 0.84 GiB). Not supported yet: `--tp 2` (auto resolves to text-only there, `--vision on` throws a
clear message); the rank-0-tower design of `docs/tp.md` 8.3 applies, since the rows are already host rows.

## Validation

### 1. The tower against HF (`tools/reference/vision_golden.py --moe`, real Ornith weights, bf16, CPU; `tests/vision/test_vision_tower`
with `R4DX_VISION_CONTAINER` / `R4DX_VISION_GOLDEN_DIR`)

Three cases (448x448; 613x409 -> 608x416; both images in one call). Against HF's **sdpa** attention (fp32-accumulating scores, the numerics of
`r4d_attn_vit_h72_bf16`) everything passes the 27B's bands: front end 1-3 bf16 ulp, block 0's whole chain <= 7e-3, blocks 1-26 rel-L2 1.2e-2 ..
5.7e-2, block 26 / last hidden state 9.2e-2, merger output **6.9e-2** (non-square 7.5e-2, two images 7.2e-2); per-block localization (each
block run from the reference's own input) worst single block **8.4e-3** (block 2), 3.5e-3 at block 26. Against HF's **eager** attention
(scores rounded to bf16, the committed 27B golden's flavour) the same run lands at block 26 **1.28e-1** (band 0.12) and merger 1.16e-1, per-block
worst 1.54e-2 (band 1.5e-2): marginally above the 27B-tuned bands for one tensor and one block, for the reason `docs/vision.md` "Why the
deep-block disagreement is not an r4dx error" gives (the eager reference itself sits 7-8e-2 from the sdpa one at block 26); r4dx is nearer
the fp32-accumulating reference. No tower code changed, so no band was changed either: the committed test is unmodified besides the container
override. Fixture dirs: `E:\models\r4dx\q35moe-vision\golden_out` (eager), `golden_out_sdpa`.

### 2. The text integration on the tiny model (`test_model_q35moe_vision`, ctest label `q35moe`, 4 s)

`tools/reference/qwen35moe/tiny_vision.py` builds the tiny text stack (4 layers, 16 experts) **with the real tower** and HF goldens with images
(`one_image`: 64 image tokens, `two_images`: 64 + 70 tokens across three prefill chunks), bf16 container `tiny-bf16-vision.r4dx` (a trellis tiny
net is no reference: random weights do not survive 3.4 bits). HF's tower rows are spliced so the text side is judged alone:

| check | result |
|---|---|
| tower alone vs HF (our rows) | rel-L2 0.103 / 0.105 (band 0.12) |
| layer 0..3 residual, norm, logits of every row (incl. image rows) | mean/rms 0.008 / 0.016 / 0.026 / 0.037 at layers 0-3, logits 0.037 (the tiny net's text-only goldens run at the same level; bound 0.15) |
| last-row argmax, mrope delta | equal to HF's in both cases (-56, -116) |
| greedy continuation after the image, teacher-forced with HF's tokens | 12 / 12 and 11 / 12 (the one miss is a 0.19-logit near-tie of our own logits) |
| prompt in two calls (text, then the image part); one image per call | same argmax and delta (continuation blocks, call-relative offsets) |
| text only with a tower loaded vs without | prefill logits **bit-identical**, 8 decode steps identical |

### 3. The real model, end to end (`tools/qwen35moe_vision_e2e.ps1`, `r4dx-cli`, both cards, greedy, `--think off`, `--image-max-pixels 1000000`)

Images are the two in the model repo's own `assets/` (no download).

* `ornith_logo.png` (3648 x 1046): *"This image features a cute cartoon bird character with a cream-colored face, black-rimmed glasses, rosy cheeks, and a small
  orange beak, set against a dark circular background. The bird has a distinctive yellow crest on its head and appears to be wearing a dark hoodie with white
  drawstrings. To the right of the character, the word "Ornith" is displayed in large, bold, brown letters against a black background."* (the PNG's transparent
  background decodes to black, as in HF's `convert("RGB")`). 928 image tokens, encode 149 ms, prefill 958 tok at 514 tok/s, decode 66.3 tok/s.
* `ornith_122b_eval.png` (the benchmark chart): title and the first model's scores transcribed correctly (TerminalBench 2.1 81.2, CyberGym 80, NL2Repo 51.2,
  SWE-bench Pro 76.9, HLE 37.6, WideSearch 75.2, BrowseComp 78.5 ...); 925 image tokens, prefill 966 tok at 540 tok/s, decode 64.2-64.7 tok/s.
* **Against HF on the same images** (`tools/reference/qwen35moe/vision_check.py`: HF tower on HF's preprocessing, HF decoder layers streamed from the shards, one
  teacher-forced pass over the prompt plus r4dx's own generated tokens, `--dump-token-ids`): HF's argmax equals the token r4dx emitted in **94.4 %** of the
  positions for the logo description (HF top-5 100 %, mean log-prob of r4dx's tokens -0.215) and **98.8 %** for the chart (-0.025). r4dx runs 3.4-bit weights, so
  a perfect match is not expected (Rung 4: 88 % top-1 on text). Controls run in the same pass: with the image rows **not** spliced the agreement falls to
  **64.0 % / 63.7 %** (top-5 83 % / 81 %), so the splice is what the numbers measure; with plain 1-D positions instead of mrope 92.1 % / 99.4 %: on these two
  short answers the 3-axis rope changes little (position delta of 56-60 on a 928-token image), the tiny-model test is the one that pins the rope (its argmax and
  the two-call and image-per-call cases only agree with the golden under the right rows).
* **Text only unchanged:** the same haiku prompt on the vision container and on the old kv container produces **byte-identical** output (SHA-256 `EF9B95F4...`);
  decode 58.0 vs 48.0 tok/s in these two single 95-token runs (the order and cold caches dominate; the image runs above sit at 64-66), so no slowdown is claimed or seen.
* **DFlash2 with an image** (`--dflash ornith-1.5-122b-dflash2-w4a16.r4dx --dflash-k 11 --dflash-p-min 0.8`): the chart answer is **byte-identical** to the plain run
  (same SHA-256 `3F590D36...`), 105.9 tok/s vs 64.7 (83.2 % acceptance, 4.0 tokens per round): the drafter ropes at index + delta, the verify windows take explicit rows.
* **Server**: `tools/server/smoke.ps1 -Model ...-kv-vision.r4dx -Layout trellis -Layers -1 -Vision`: **191 checks pass**, 1 fails and is not about images
  (`--prompt-checkpoint (default on): the load log reports the checkpoint's VRAM`: the log line this check greps for is not printed by the qwen3_5_moe loader). The
  vision section passes whole: describe, OCR of a rendered string (`R4DXVSN9` read back exactly), two images in one request, image + tools, image + thinking,
  streaming, multi-turn prefix reuse without re-encoding, a different image invalidating the prefix, bad inputs (webp, corrupt data, 9 images, remote URL) all 400,
  `/v1/models` input modalities.

### Regressions

`ctest -L q35moe` (11 tests): 10 pass. `test_model_q35moe_tp_facade` has one failing check, `--dflash under --tp 2 is refused by name`: the test still expects the
refusal that DFlash2-under-TP (P6) removed; it is independent of this branch (no vision code is on that path) and left alone.

## Limits and follow-ups

* `--tp 2` with images: refused (see above). The tower would run on rank 0 and the facade copy the host rows to both ranks (`ImageRows` host path, `docs/tp.md` 8.3).
* Video input (`t > 1`) is not supported anywhere in r4dx.
* The image token budget is the CLI / server's `--image-max-pixels` (default 1048576): ~1000 merged tokens; a 2k x 2k image at the 16.7M-pixel ceiling would be ~16k tokens
  and a long prefill at ~500 tok/s.
* Prefill of image rows runs the same 256-row MoE chunks as text (the host-routed MoE; the prefill-speed track will speed images up as it speeds text up).

## Reproduce

```
D:\venvs\r4dx-qwen4-ref\Scripts\python.exe tools\reference\qwen35moe\tiny_vision.py attach      # tiny text stack + the real tower
D:\venvs\r4dx-qwen4-ref\Scripts\python.exe tools\reference\qwen35moe\tiny_vision.py golden      # HF goldens with images (CPU, ~1 min)
r4dx-convert --input E:\models\r4dx\tiny-q35moe-vision\hf --output E:\models\r4dx\tiny-q35moe-vision\tiny-bf16-vision.r4dx --layouts bf16 --lm-head bf16 --vision on
tools\with_lock.ps1 -Name gpu -Run "build\win-hip\tests\model\qwen35moe\test_model_q35moe_vision.exe"
tools\with_lock.ps1 -Name gpu -Run "powershell -File tools\qwen35moe_vision_e2e.ps1"           # the CLI cases (logo, chart, text, chart + DFlash2)
# HF check of a generated answer (HIP_VISIBLE_DEVICES=1, ~4 min):
#   vision_check.py --tokens <e2e\logo.tokens.json> --prompt-len <prefill N tok> --image <png> --max-pixels 1000000
```
