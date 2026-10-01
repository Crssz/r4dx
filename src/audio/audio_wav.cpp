#include "audio_wav.h"

#include <cstring>
#include <stdexcept>
#include <string>

namespace r4dx::audio {

namespace {

uint32_t Le32(const uint8_t* p) {
  return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) | (static_cast<uint32_t>(p[2]) << 16) |
         (static_cast<uint32_t>(p[3]) << 24);
}
uint16_t Le16(const uint8_t* p) { return static_cast<uint16_t>(p[0] | (p[1] << 8)); }

[[noreturn]] void Fail(const std::string& why) { throw std::runtime_error("wav: " + why); }

constexpr uint16_t kFormatPcm = 1, kFormatFloat = 3, kFormatExtensible = 0xFFFE;

}  // namespace

WavAudio DecodeWav(const uint8_t* data, size_t size) {
  if (size < 12 || std::memcmp(data, "RIFF", 4) != 0 || std::memcmp(data + 8, "WAVE", 4) != 0) {
    Fail("not a RIFF/WAVE file (expected audio/wav; mp3, ogg, flac, ... are not supported)");
  }
  bool have_fmt = false;
  uint16_t format = 0, channels = 0, bits = 0, block_align = 0;
  uint32_t rate = 0;
  const uint8_t* pcm = nullptr;
  size_t pcm_bytes = 0;
  size_t pos = 12;
  while (pos + 8 <= size) {
    const uint8_t* id = data + pos;
    size_t len = Le32(data + pos + 4);
    const size_t body = pos + 8;
    const size_t avail = size - body;
    if (std::memcmp(id, "fmt ", 4) == 0) {
      if (len < 16 || len > avail) Fail("truncated fmt chunk");
      format = Le16(data + body);
      channels = Le16(data + body + 2);
      rate = Le32(data + body + 4);
      block_align = Le16(data + body + 12);
      bits = Le16(data + body + 14);
      if (format == kFormatExtensible) {
        if (len < 26) Fail("truncated WAVE_FORMAT_EXTENSIBLE fmt chunk");
        format = Le16(data + body + 24);  // first two bytes of the sub-format GUID are the real format tag
      }
      have_fmt = true;
    } else if (std::memcmp(id, "data", 4) == 0) {
      // A streamed file may carry 0 / 0xFFFFFFFF as the data length: take what is there.
      if (len > avail) len = avail;
      pcm = data + body;
      pcm_bytes = len;
      if (have_fmt) break;
    }
    const size_t next = body + len + (len & 1);
    if (next <= pos || next > size) break;
    pos = next;
  }
  if (!have_fmt) Fail("no fmt chunk");
  if (pcm == nullptr) Fail("no data chunk");
  if (channels < 1 || channels > 64) Fail("bad channel count " + std::to_string(channels));
  if (rate == 0) Fail("zero sample rate");
  const bool is_float = format == kFormatFloat;
  if (format != kFormatPcm && !is_float) {
    Fail("unsupported encoding (format tag " + std::to_string(format) + "); only PCM and IEEE float WAV are supported");
  }
  if (is_float ? (bits != 32 && bits != 64) : (bits != 8 && bits != 16 && bits != 24 && bits != 32)) {
    Fail("unsupported bit depth " + std::to_string(bits) + (is_float ? " (float)" : " (PCM)"));
  }
  const size_t bytes_per_sample = bits / 8;
  if (block_align != bytes_per_sample * channels) Fail("inconsistent block alignment");
  const size_t frames = pcm_bytes / block_align;
  if (frames == 0) Fail("no audio samples");

  WavAudio out;
  out.sample_rate = static_cast<int>(rate);
  out.channels = channels;
  out.bits_per_sample = bits;
  out.is_float = is_float;
  out.samples.resize(frames);
  for (size_t f = 0; f < frames; ++f) {
    double acc = 0.0;
    for (uint16_t c = 0; c < channels; ++c) {
      const uint8_t* p = pcm + f * block_align + c * bytes_per_sample;
      double v;
      if (is_float) {
        if (bits == 32) {
          const uint32_t u = Le32(p);
          float x;
          std::memcpy(&x, &u, 4);
          v = x;
        } else {
          uint64_t u = static_cast<uint64_t>(Le32(p)) | (static_cast<uint64_t>(Le32(p + 4)) << 32);
          double x;
          std::memcpy(&x, &u, 8);
          v = x;
        }
      } else if (bits == 8) {
        v = (static_cast<int>(p[0]) - 128) / 128.0;  // 8-bit WAV is unsigned
      } else if (bits == 16) {
        v = static_cast<int16_t>(Le16(p)) / 32768.0;
      } else if (bits == 24) {
        int32_t s = static_cast<int32_t>((static_cast<uint32_t>(p[0]) << 8) | (static_cast<uint32_t>(p[1]) << 16) |
                                         (static_cast<uint32_t>(p[2]) << 24)) >>
                    8;
        v = s / 8388608.0;
      } else {
        v = static_cast<int32_t>(Le32(p)) / 2147483648.0;
      }
      acc += v;
    }
    out.samples[f] = static_cast<float>(acc / channels);
  }
  return out;
}

}  // namespace r4dx::audio
