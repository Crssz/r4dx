// src/audio/audio_wav.h -- RIFF/WAVE decode for the server's `input_audio` content parts (docs/gemma4-audio.md).
//
// Accepts PCM 8/16/24/32-bit integer and IEEE float 32/64 (also inside WAVE_FORMAT_EXTENSIBLE), any channel
// count (downmixed to mono by the arithmetic mean), and returns float samples in [-1, 1]. It does NOT
// resample: the caller compares `sample_rate` with audio::kSampleRate and refuses anything else (the server
// answers 400 -- a naive resampler would silently change what the model hears; see docs/gemma4-audio.md).
// Throws std::runtime_error with a message fit for a 400 body on malformed or unsupported input.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace r4dx::audio {

struct WavAudio {
  int sample_rate = 0;
  int channels = 0;           // channels in the FILE (samples below are already mono)
  int bits_per_sample = 0;
  bool is_float = false;
  std::vector<float> samples;  // mono, [-1, 1] for integer PCM (full scale = 2^(bits-1))
};

WavAudio DecodeWav(const uint8_t* data, size_t size);

}  // namespace r4dx::audio
