# Gemma 4 audio input (M3)

gemma4_unified has **no audio tower**. Audio is raw waveform frames projected into the text space, so the whole
path is host-side arithmetic plus a splice. Every statement below was checked against transformers 5.18
(`Gemma4UnifiedAudioFeatureExtractor`, `Gemma4UnifiedProcessor`, `Gemma4UnifiedMultimodalEmbedder`, the chat
template) by `tools/reference/gemma/audio_golden_gemma.py` (CPU only; output `golden_out/gemma/audio_semantics.json`
and `audio_golden.safetensors`, gitignored).

## Semantics (verified)

| Step | Rule |
|---|---|
| Input | 16 kHz mono float waveform, raw samples (no normalisation, no clipping). |
| Frames | right zero-pad to a multiple of 640, reshape `[n, 640]`; `n = ceil(samples / 640)` (40 ms per token). All `n` tokens are valid (the padding sits inside the last frame). |
| Embedder | frames cast to **bf16 first** (the embedder casts to `embedding_projection.weight.dtype`), `RMSNorm(640, no weight, eps 1e-6)` (fp32 math, bf16 result), `Linear(640 -> 3840, no bias)` (fp32 accumulate, bf16 result). Tensor `model.embed_audio.embedding_projection.weight` bf16 `[3840, 640]`; container name `audio.embed_audio.embedding_projection.weight` (`[3840, 640, 2]` bytes). Skipping the bf16 cast changes the output (checked). |
| Scale | none. The rows overwrite the scaled embedding gather at the `<\|audio\|>` positions; only text rows carry `sqrt(3840) = 62.0`. |
| Prompt | the chat template emits **one** `<\|audio\|>` (258881) per `audio` content part (the chat template matches ONLY `type == "audio"`; HF's `apply_chat_template` normalizes OpenAI `input_audio` to it, so the r4dx engine maps `input_audio` to `audio` in the template message JSON via `kAudioTemplatePartType`, covered by `test_audio_template`); the processor expands it to `<\|audio>` (boa 256000) + `n` x `<\|audio\|>` + `<audio\|>` (eoa 258883). Checked on the real tokenizer dir: template ids + manual expansion == processor ids. Example, thinking off: `<bos><\|turn>user\nTranscribe:<\|audio\|><turn\|>\n<\|turn>model\n<\|channel>thought\n<channel\|>`. |
| Attention | audio tokens are `mm_token_type_ids` 3, never a vision block: plain causal (+ the sliding window) on sliding and full layers. Verified at mask level (generate-path masks) and end to end on a tiny random model: perturbing audio frame k leaves every earlier position's hidden state bit-identical. |
| Positions | plain 1D rope positions, one per token (no 2D image-style positions). |
| 750 cap | `audio_seq_length = 750` is only a processor upper bound; HF does **not** enforce it (480001 samples -> 751 tokens expand fine). r4dx enforces it as a 400 (30 s). |

## r4dx implementation

* **Converter** (`r4dx-convert --audio on`, default off): bf16 passthrough of the one tensor as
  `audio.embed_audio.embedding_projection.weight`; the coverage audit allow-lists it when off. Already covered by
  `convert_gemma_layout` (byte-exact passthrough, off by default, refusal when the checkpoint has none).
* **`src/audio`** (CPU-only, no HIP): `audio_frames.h` (constants, `NumAudioTokens`, `FrameWaveform`,
  `ExpandAudioPlaceholders`), `audio_wav.{h,cpp}` (RIFF/WAVE: PCM 8/16/24/32, float 32/64, extensible, any channel
  count -> mean mono; streamed data length clamped; does not resample), `audio_embed.{h,cpp}` (`AudioEmbedder`, a
  multi-threaded CPU RMSNorm + Linear, bit-deterministic for any thread count, ~1.8 GFLOP for 30 s) and
  `audio_weights.cpp` (`ContainerHasAudio`, `LoadAudioEmbedder`: reads the one 4.9 MB tensor from the container).
* **Model**: `GemmaModel::PrefillAudio(token_ids, spans)` splices host bf16 rows (`AudioRowSpan`) over the scaled
  gather inside `RunChunk`, before the residual rotation (so a rotated container sees them in the rotated basis, like
  text rows); a span may straddle prefill chunks and a continuation call may start inside a span (offset 0, rows
  pointer advanced). `TextModel` gained non-pure `HasAudio() / EncodeAudio() / PrefillAudio()` (Qwen's models keep
  the throwing defaults, so Qwen behaviour is unchanged); `GemmaLocalTextModel` implements them and
  `LoadGemmaTextModel` loads the embedder when the container carries the tensor. No VRAM, no kernel.
* **Server**: `{"type":"input_audio","input_audio":{"data":"<base64 wav>","format":"wav"}}` (`data` may carry a
  `data:audio/wav;base64,` prefix). Decoded at parse time (`openai_types.cpp`). 400 for: any format but wav,
  malformed base64, not-a-WAV (an mp3 labelled wav), unsupported encoding / bit depth, a sample rate other than
  **16000** (not resampled; the message gives the ffmpeg command), more than 30 s, more than 4 clips per request, a
  request mixing images and audio (until the Gemma vision splice lands, M2), a Qwen container or a Gemma container
  converted without `--audio on`. `/v1/models` lists `audio` in `modalities` / `capabilities` when the embedder is
  present. Prefix reuse keys audio like images (`ImageKey`, `grid_t = -1`, `grid_h = tokens`, FNV-1a over the decoded
  samples), so two different clips of the same length never share a cached prefix; a clip already in the reused prefix
  is not re-embedded. Why refuse non-16 kHz instead of resampling: a naive resampler silently changes what the model
  hears, and a proper polyphase one is a dependency for a one-line `ffmpeg -ar 16000 -ac 1`.

## Tests

CPU (no GPU, no container; `HIP_VISIBLE_DEVICES=-1`):

```
D:\venvs\r4dx-gemma-ref\Scripts\python.exe tools\reference\gemma\audio_golden_gemma.py   # reads 4.9 MB of the checkpoint
cmake --build build\win-hip --target test_audio test_audio_golden test_openai_types -j 4
ctest --test-dir build\win-hip -R "^test_audio$|^test_audio_golden$|^test_openai_types$|^convert_gemma_layout$"
```

* `test_audio`: framing / token counts, expansion + every refusal, WAV decode (all encodings, odd chunks, streamed,
  refusals), the embedder against a double-precision recipe (1 bf16 ulp), thread-count determinism, row
  independence, `LoadAudioEmbedder` on a hand-built container.
* `test_audio_golden` (exit 77 without the golden): the prompt layout reproduces the HF processor's for every
  recorded case; the embedder with the **real** weight matches HF's rows (n = 1, 2, 12, 750). Not bit-exact by
  construction: torch's fp32 reduction order for `mean(x^2)` differs by ~1e-7, which flips the bf16 rounding of one
  normed element in ~3% of rows (those rows move by a few bf16 ulps, visible only on near-zero outputs); the test
  bounds that (<= 0.5% of elements beyond 1 ulp + 1e-5, max abs diff <= 0.25, <= 5% of rows flipped). Observed:
  0 / 0 / 16 / 26274 differing elements of 3840 / 7680 / 46080 / 2.88M, 0.3% beyond tolerance, max 0.0625, 21 of 750
  rows.
* `test_openai_types`: the `input_audio` accept and reject matrix above.

GPU (user-run; **not run by the author**):

```
r4dx-convert --model-dir D:\models\Huihui-gemma-4-12B-it-abliterated --out D:\models\r4dx\huihui-gemma\bf16-audio.r4dx --audio on
$env:HIP_VISIBLE_DEVICES='1'; ctest --test-dir build\win-hip -R gemma_audio --output-on-failure
```

(`R4DX_GEMMA_AUDIO_CONTAINER` overrides the container.) `test_gemma_audio` checks finite non-flat logits for a
synthetic 1 s tone plus 16 greedy ids, that a different waveform changes the logits and the same one is bit-identical,
a 750-token clip, a prefill split inside the span against the one-call prefill, and that a bad span is refused. The
end-to-end text check is the server: `r4dx-server --model <container>` and POST a chat request with a speech WAV as an
`input_audio` part ("Transcribe the audio."). The quality gate against HF (greedy transcript agreement on a few clips
via `Gemma4UnifiedForConditionalGeneration.generate` on a GPU the user frees) is the M3-2 gate and is still open.

## Not done

* `r4dx-cli --audio` (the CLI has no Gemma multimodal path yet; the server is the entry point).
* Audio together with images in one request (needs the Gemma vision splice, M2, to own a single multimodal prefill).
* Under TP (`GemmaTpModel`, M1b): the rows are host bf16 and would be spliced on every rank like Qwen's TP image rows.
* Resampling, mp3/ogg/flac decoding, streaming audio.
