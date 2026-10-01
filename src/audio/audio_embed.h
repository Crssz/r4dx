// src/audio/audio_embed.h -- the Gemma 4 audio embedder, on the CPU (docs/gemma4-audio.md).
//
//   frames [n, 640] f32  ->  bf16  ->  RMSNorm(640, no weight, eps 1e-6; fp32 math, bf16 result)
//                         ->  Linear(640 -> 3840, no bias; fp32 accumulate, bf16 result)   [n, 3840] bf16
//
// This is Gemma4UnifiedMultimodalEmbedder with the checkpoint's audio weight. The waveform is cast to bf16
// BEFORE the norm (the embedder casts its input to embedding_projection.weight's dtype) -- skipping that step
// changes the output, which the golden checks. The rows are NOT scaled by sqrt(3840): they overwrite the
// scaled embedding gather at the <|audio|> positions (docs/gemma4-semantics.md section 3).
//
// 750 rows x 640 x 3840 is 1.8 GFLOP: a few hundred ms on a few CPU threads, once per clip, so there is no
// device kernel. The result is host bf16, spliced into the residual by the model (GemmaModel's row-splice
// hook, which converts to fp32 for the default fp32 residual).
//
// HIP-free; the weight comes from a container (LoadAudioEmbedder, audio_weights.cpp) or from raw bf16 bytes.
#pragma once

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace r4dx::audio {

constexpr int kEmbedHidden = 3840;  // text hidden size

// bf16 <-> f32 (round-to-nearest-even; NaN stays NaN), the same conversion torch uses.
inline uint16_t F32ToBf16(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  if ((u & 0x7FFFFFFFu) > 0x7F800000u) return static_cast<uint16_t>((u >> 16) | 0x40u);  // NaN
  u += 0x7FFFu + ((u >> 16) & 1u);
  return static_cast<uint16_t>(u >> 16);
}
inline float Bf16ToF32(uint16_t b) {
  const uint32_t u = static_cast<uint32_t>(b) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

class AudioEmbedder {
 public:
  // `weight_bf16`: [out_dim, in_dim] row-major bf16 (the checkpoint's embedding_projection.weight, the
  // container's `audio.embed_audio.embedding_projection.weight` bytes). Defaults are the 12B's shapes.
  AudioEmbedder(const std::vector<uint16_t>& weight_bf16, int in_dim = 640, int out_dim = kEmbedHidden,
                float eps = 1e-6f);

  int InDim() const { return in_; }
  int OutDim() const { return out_; }

  // `frames`: [n, in_dim] f32 (audio::FrameWaveform). Returns [n, out_dim] bf16. `threads` <= 0: hardware
  // concurrency, capped at 8. Bit-deterministic for any thread count (each row is computed by one thread).
  std::vector<uint16_t> Embed(const float* frames, int64_t n, int threads = 0) const;

 private:
  int in_, out_;
  float eps_;
  std::vector<float> w_;  // bf16 weight widened to f32, [out, in]
};

// Reads `audio.embed_audio.embedding_projection.weight` from a gemma4_unified .r4dx container (header
// parse + a read of that one 4.9 MB tensor; the container is memory-mapped by the shared SafetensorsReader,
// but only these pages are touched) and checks its shape [3840, 640, 2]. Throws std::runtime_error naming the
// container when it was converted without `--audio on`.
bool ContainerHasAudio(const std::string& container_path);
AudioEmbedder LoadAudioEmbedder(const std::string& container_path);

// The converter's tensor name for the audio projection.
inline const char* AudioWeightName() { return "audio.embed_audio.embedding_projection.weight"; }

}  // namespace r4dx::audio
