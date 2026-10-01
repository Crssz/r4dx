// src/audio/audio_frames.h -- the host half of Gemma 4 (gemma4_unified) audio input (docs/gemma4-audio.md).
//
// gemma4_unified has no audio tower: a 16 kHz mono float waveform is right-zero-padded to a multiple of 640
// samples, reshaped to [n, 640] (n = ceil(samples / 640), 40 ms per token) and every frame becomes one soft
// token via RMSNorm(640, no weight) -> Linear(640 -> 3840) (audio_embed.h). Verified against transformers'
// Gemma4UnifiedAudioFeatureExtractor / Gemma4UnifiedProcessor by tools/reference/gemma/audio_golden_gemma.py.
//
// Header-only, no HIP, no r4dx_model dependency (like src/vision/image_prompt.h).
#pragma once

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace r4dx::audio {

constexpr int kSampleRate = 16000;
constexpr int kSamplesPerToken = 640;   // audio_samples_per_token == audio_embed_dim
constexpr int kMaxAudioTokens = 750;    // processor audio_seq_length: 30 s. HF never enforces it; r4dx does.
constexpr int64_t kMaxAudioSamples = int64_t{kMaxAudioTokens} * kSamplesPerToken;  // 480000

// Token ids (config.json: audio_token_id / boa_token_id / eoa_token_index; the tokenizer's <|audio|> /
// <|audio> / <audio|>).
constexpr int32_t kBoaTokenId = 256000;
constexpr int32_t kAudioTokenId = 258881;
constexpr int32_t kEoaTokenId = 258883;

// ceil(samples / 640): Gemma4UnifiedProcessor._compute_audio_num_tokens.
inline int64_t NumAudioTokens(int64_t samples) {
  return (samples + kSamplesPerToken - 1) / kSamplesPerToken;
}

// Right zero-pad to a multiple of 640 and return the [n, 640] frames, row-major (the feature extractor's
// `input_features`). An empty waveform is an error (zero tokens cannot be spliced).
inline std::vector<float> FrameWaveform(const std::vector<float>& samples, int64_t* tokens_out = nullptr) {
  if (samples.empty()) throw std::runtime_error("FrameWaveform: empty waveform");
  const int64_t n = NumAudioTokens(static_cast<int64_t>(samples.size()));
  std::vector<float> frames(static_cast<size_t>(n) * kSamplesPerToken, 0.0f);
  std::copy(samples.begin(), samples.end(), frames.begin());
  if (tokens_out != nullptr) *tokens_out = n;
  return frames;
}

// One audio occurrence in an expanded prompt: `offset` is the index of the first `<|audio|>` soft token
// (the one after <|audio>), `tokens` the run length. Mirrors Model's ImageSpan offset contract (relative to
// the expanded token vector).
struct AudioPlaceholderSpan {
  int64_t offset = 0;
  int64_t tokens = 0;
};

struct ExpandedAudioPrompt {
  std::vector<int32_t> tokens;
  std::vector<AudioPlaceholderSpan> spans;
};

// `raw_tokens`: the chat-template-rendered, tokenized prompt, carrying exactly one `<|audio|>` (258881) per
// audio part (the template emits that single placeholder, never <|audio>/<audio|>). `audio_tokens[i]` is the
// soft-token count of the i-th audio in order. Each placeholder becomes <|audio> + n x <|audio|> + <audio|>
// (Gemma4UnifiedProcessor.replace_audio_token). Throws on a count mismatch either way, and on a count outside
// [1, kMaxAudioTokens].
inline ExpandedAudioPrompt ExpandAudioPlaceholders(const std::vector<int32_t>& raw_tokens,
                                                   const std::vector<int64_t>& audio_tokens,
                                                   int32_t audio_id = kAudioTokenId, int32_t boa_id = kBoaTokenId,
                                                   int32_t eoa_id = kEoaTokenId) {
  ExpandedAudioPrompt out;
  out.tokens.reserve(raw_tokens.size());
  size_t next = 0;
  for (const int32_t id : raw_tokens) {
    if (id != audio_id) {
      out.tokens.push_back(id);
      continue;
    }
    if (next >= audio_tokens.size()) {
      throw std::runtime_error("rendered prompt has more audio placeholders than audio clips were supplied");
    }
    const int64_t n = audio_tokens[next++];
    if (n < 1 || n > kMaxAudioTokens) {
      throw std::runtime_error("audio clip token count " + std::to_string(n) + " is outside [1, " +
                               std::to_string(kMaxAudioTokens) + "]");
    }
    out.tokens.push_back(boa_id);
    AudioPlaceholderSpan sp;
    sp.offset = static_cast<int64_t>(out.tokens.size());
    sp.tokens = n;
    out.spans.push_back(sp);
    out.tokens.insert(out.tokens.end(), static_cast<size_t>(n), audio_id);
    out.tokens.push_back(eoa_id);
  }
  if (next != audio_tokens.size()) {
    throw std::runtime_error("rendered prompt has fewer audio placeholders than audio clips were supplied");
  }
  return out;
}

}  // namespace r4dx::audio
