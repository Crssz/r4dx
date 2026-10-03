// CPU-only unit test for src/audio (docs/gemma4-audio.md): framing / token counts, placeholder expansion,
// WAV decode, and the CPU embedder against a double-precision recipe. No GPU, no container, no golden file.
// The HF-processor comparison lives in test_audio_golden.cpp.
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "audio_embed.h"
#include "audio_frames.h"
#include "audio_wav.h"

namespace {

int g_fail = 0;
#define CHECK(cond, msg)                                                          \
  do {                                                                            \
    if (!(cond)) {                                                                \
      std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, msg);                   \
      ++g_fail;                                                                   \
    }                                                                             \
  } while (0)

using namespace r4dx::audio;

template <class Fn>
bool Throws(Fn&& fn, const char* needle = nullptr) {
  try {
    fn();
  } catch (const std::exception& e) {
    return needle == nullptr || std::string(e.what()).find(needle) != std::string::npos;
  }
  return false;
}

void Put16(std::vector<uint8_t>& b, uint16_t v) { b.push_back(v & 0xFF); b.push_back(v >> 8); }
void Put32(std::vector<uint8_t>& b, uint32_t v) { for (int i = 0; i < 4; ++i) b.push_back((v >> (8 * i)) & 0xFF); }
void PutTag(std::vector<uint8_t>& b, const char* t) { b.insert(b.end(), t, t + 4); }

// A WAV with `payload` as the data chunk. extensible: WAVE_FORMAT_EXTENSIBLE with `format` as the sub-format.
std::vector<uint8_t> MakeWav(uint16_t format, int channels, int rate, int bits, const std::vector<uint8_t>& payload,
                             bool extensible = false, bool odd_chunk_first = false) {
  std::vector<uint8_t> b;
  PutTag(b, "RIFF");
  Put32(b, 0);  // patched below
  PutTag(b, "WAVE");
  if (odd_chunk_first) {  // an odd-sized unknown chunk (padded to even) before fmt
    PutTag(b, "LIST");
    Put32(b, 3);
    b.insert(b.end(), {1, 2, 3, 0});
  }
  PutTag(b, "fmt ");
  Put32(b, extensible ? 40 : 16);
  Put16(b, extensible ? 0xFFFE : format);
  Put16(b, static_cast<uint16_t>(channels));
  Put32(b, static_cast<uint32_t>(rate));
  Put32(b, static_cast<uint32_t>(rate * channels * bits / 8));
  Put16(b, static_cast<uint16_t>(channels * bits / 8));
  Put16(b, static_cast<uint16_t>(bits));
  if (extensible) {
    Put16(b, 22);
    Put16(b, static_cast<uint16_t>(bits));
    Put32(b, 0);
    Put16(b, format);  // sub-format GUID starts with the format tag
    b.insert(b.end(), {0, 0, 0, 0, 0x10, 0, 0x80, 0, 0, 0xAA, 0, 0x38, 0x9B, 0x71});
  }
  PutTag(b, "data");
  Put32(b, static_cast<uint32_t>(payload.size()));
  b.insert(b.end(), payload.begin(), payload.end());
  const uint32_t riff = static_cast<uint32_t>(b.size() - 8);
  std::memcpy(b.data() + 4, &riff, 4);
  return b;
}

void TestFraming() {
  CHECK(NumAudioTokens(1) == 1 && NumAudioTokens(639) == 1 && NumAudioTokens(640) == 1 &&
            NumAudioTokens(641) == 2 && NumAudioTokens(480000) == 750 && NumAudioTokens(480001) == 751,
        "NumAudioTokens = ceil(samples / 640)");
  CHECK(kMaxAudioSamples == 480000, "750 tokens == 30 s at 16 kHz");
  std::vector<float> x(641);
  for (size_t i = 0; i < x.size(); ++i) x[i] = static_cast<float>(i + 1);
  int64_t n = 0;
  const std::vector<float> f = FrameWaveform(x, &n);
  CHECK(n == 2 && f.size() == 1280, "641 samples -> [2, 640]");
  CHECK(f[640] == 641.0f && f[641] == 0.0f && f[1279] == 0.0f && f[639] == 640.0f, "right zero pad, row-major");
  CHECK(Throws([] { FrameWaveform({}); }, "empty"), "empty waveform refused");
  const std::vector<float> one = FrameWaveform({0.5f}, &n);
  CHECK(n == 1 && one.size() == 640 && one[0] == 0.5f && one[1] == 0.0f, "1 sample -> one padded frame");
}

void TestExpansion() {
  const std::vector<int32_t> raw = {2, 105, 7, kAudioTokenId, 9, kAudioTokenId, 11};
  const ExpandedAudioPrompt e = ExpandAudioPlaceholders(raw, {3, 1});
  const std::vector<int32_t> want = {2, 105, 7, kBoaTokenId, kAudioTokenId, kAudioTokenId, kAudioTokenId, kEoaTokenId,
                                     9, kBoaTokenId, kAudioTokenId, kEoaTokenId, 11};
  CHECK(e.tokens == want, "<|audio|> -> <|audio> + n x <|audio|> + <audio|>");
  CHECK(e.spans.size() == 2 && e.spans[0].offset == 4 && e.spans[0].tokens == 3 && e.spans[1].offset == 10 &&
            e.spans[1].tokens == 1,
        "spans point at the soft tokens, not at boa");
  CHECK(Throws([&] { ExpandAudioPlaceholders(raw, {3}); }, "more audio placeholders"), "too few clips refused");
  CHECK(Throws([&] { ExpandAudioPlaceholders(raw, {3, 1, 2}); }, "fewer audio placeholders"), "too many clips refused");
  CHECK(Throws([&] { ExpandAudioPlaceholders({kAudioTokenId}, {751}); }, "outside"), "751 tokens refused");
  CHECK(Throws([&] { ExpandAudioPlaceholders({kAudioTokenId}, {0}); }, "outside"), "0 tokens refused");
  CHECK(ExpandAudioPlaceholders({kAudioTokenId}, {750}).tokens.size() == 752, "750 tokens accepted");
  CHECK(ExpandAudioPlaceholders({1, 2, 3}, {}).tokens == std::vector<int32_t>({1, 2, 3}), "no audio: untouched");
}

void TestWav() {
  // PCM16 mono.
  std::vector<uint8_t> p;
  const int16_t s16[] = {0, 16384, -16384, 32767, -32768};
  for (int16_t v : s16) Put16(p, static_cast<uint16_t>(v));
  WavAudio a = DecodeWav(MakeWav(1, 1, 16000, 16, p).data(), MakeWav(1, 1, 16000, 16, p).size());
  CHECK(a.sample_rate == 16000 && a.channels == 1 && a.samples.size() == 5, "pcm16 header");
  CHECK(a.samples[1] == 0.5f && a.samples[2] == -0.5f && a.samples[4] == -1.0f &&
            std::fabs(a.samples[3] - 32767.0f / 32768.0f) < 1e-7f,
        "pcm16 scaling by 2^15");

  // PCM16 stereo (L = 1/2, R = -1/2 -> 0; L = R = 1/4 -> 1/4), 44.1 kHz is decoded as-is (the caller refuses it).
  std::vector<uint8_t> st;
  for (int16_t v : {int16_t{16384}, int16_t{-16384}, int16_t{8192}, int16_t{8192}}) Put16(st, static_cast<uint16_t>(v));
  const auto stw = MakeWav(1, 2, 44100, 16, st);
  a = DecodeWav(stw.data(), stw.size());
  CHECK(a.sample_rate == 44100 && a.channels == 2 && a.samples.size() == 2 && a.samples[0] == 0.0f &&
            a.samples[1] == 0.25f,
        "stereo is averaged to mono; the rate is reported, not converted");

  // float32 and float64.
  std::vector<uint8_t> f32;
  for (float v : {0.25f, -0.75f}) { uint32_t u; std::memcpy(&u, &v, 4); Put32(f32, u); }
  const auto f32w = MakeWav(3, 1, 16000, 32, f32);
  a = DecodeWav(f32w.data(), f32w.size());
  CHECK(a.is_float && a.samples.size() == 2 && a.samples[0] == 0.25f && a.samples[1] == -0.75f, "float32");
  std::vector<uint8_t> f64;
  { double v = 0.125; uint64_t u; std::memcpy(&u, &v, 8); Put32(f64, static_cast<uint32_t>(u)); Put32(f64, static_cast<uint32_t>(u >> 32)); }
  const auto f64w = MakeWav(3, 1, 16000, 64, f64);
  a = DecodeWav(f64w.data(), f64w.size());
  CHECK(a.samples.size() == 1 && a.samples[0] == 0.125f, "float64");

  // 24-bit and 32-bit and 8-bit PCM.
  std::vector<uint8_t> p24 = {0x00, 0x00, 0x40, /* +0.5 */ 0x00, 0x00, 0xC0 /* -0.5 */};
  const auto w24 = MakeWav(1, 1, 16000, 24, p24);
  a = DecodeWav(w24.data(), w24.size());
  CHECK(a.samples.size() == 2 && a.samples[0] == 0.5f && a.samples[1] == -0.5f, "pcm24");
  std::vector<uint8_t> p32;
  Put32(p32, 0x40000000u);
  Put32(p32, 0x80000000u);
  const auto w32 = MakeWav(1, 1, 16000, 32, p32);
  a = DecodeWav(w32.data(), w32.size());
  CHECK(a.samples.size() == 2 && a.samples[0] == 0.5f && a.samples[1] == -1.0f, "pcm32");
  const auto w8 = MakeWav(1, 1, 16000, 8, {128, 192, 0});
  a = DecodeWav(w8.data(), w8.size());
  CHECK(a.samples.size() == 3 && a.samples[0] == 0.0f && a.samples[1] == 0.5f && a.samples[2] == -1.0f, "pcm8 is unsigned");

  // WAVE_FORMAT_EXTENSIBLE carrying float, and an odd chunk before fmt.
  const auto ext = MakeWav(3, 1, 16000, 32, f32, /*extensible=*/true, /*odd_chunk_first=*/true);
  a = DecodeWav(ext.data(), ext.size());
  CHECK(a.is_float && a.samples.size() == 2 && a.samples[1] == -0.75f, "extensible float + odd chunk padding");

  // A streamed file: data length 0xFFFFFFFF takes what is there.
  auto streamed = MakeWav(1, 1, 16000, 16, p);
  const uint32_t big = 0xFFFFFFFFu;
  std::memcpy(streamed.data() + streamed.size() - p.size() - 4, &big, 4);
  a = DecodeWav(streamed.data(), streamed.size());
  CHECK(a.samples.size() == 5, "data length beyond EOF is clamped");

  // Refusals.
  const std::vector<uint8_t> junk = {'I', 'D', '3', 4, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  CHECK(Throws([&] { DecodeWav(junk.data(), junk.size()); }, "not a RIFF/WAVE"), "mp3/ID3 refused");
  const auto adpcm = MakeWav(2, 1, 16000, 4, {0, 0});
  CHECK(Throws([&] { DecodeWav(adpcm.data(), adpcm.size()); }, "unsupported encoding"), "ADPCM refused");
  const auto empty = MakeWav(1, 1, 16000, 16, {});
  CHECK(Throws([&] { DecodeWav(empty.data(), empty.size()); }, "no audio samples"), "empty data refused");
  auto cut = MakeWav(1, 1, 16000, 16, p);
  cut.resize(30);
  CHECK(Throws([&] { DecodeWav(cut.data(), cut.size()); }), "truncated header refused");
  const auto bad_depth = MakeWav(1, 1, 16000, 12, {0, 0});
  CHECK(Throws([&] { DecodeWav(bad_depth.data(), bad_depth.size()); }, "bit depth"), "12-bit refused");
}

// Plain double-precision recipe of the embedder, with the same bf16 rounding points.
std::vector<float> RefEmbed(const std::vector<float>& frames, int64_t n, const std::vector<uint16_t>& w, int in, int out) {
  std::vector<float> res(static_cast<size_t>(n) * out);
  for (int64_t r = 0; r < n; ++r) {
    std::vector<double> x(in);
    double ss = 0;
    for (int k = 0; k < in; ++k) {
      x[k] = Bf16ToF32(F32ToBf16(frames[r * in + k]));
      ss += x[k] * x[k];
    }
    const double scale = std::pow(ss / in + 1e-6, -0.5);
    for (int k = 0; k < in; ++k) x[k] = Bf16ToF32(F32ToBf16(static_cast<float>(x[k] * scale)));
    for (int j = 0; j < out; ++j) {
      double acc = 0;
      for (int k = 0; k < in; ++k) acc += x[k] * Bf16ToF32(w[static_cast<size_t>(j) * in + k]);
      res[r * out + j] = static_cast<float>(acc);
    }
  }
  return res;
}

void TestEmbedder() {
  CHECK(F32ToBf16(1.0f) == 0x3F80 && F32ToBf16(-2.0f) == 0xC000, "bf16 exact values");
  CHECK(F32ToBf16(1.00390625f) == 0x3F80, "bf16 ties to even (down)");
  CHECK(F32ToBf16(1.01171875f) == 0x3F82, "bf16 ties to even (up)");
  CHECK(std::isnan(Bf16ToF32(F32ToBf16(std::nanf("")))), "bf16 NaN");

  const int in = 640, out = 96;  // a narrower out dim keeps the test quick; the recipe is dimension-agnostic
  std::mt19937 rng(5);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::vector<uint16_t> w(static_cast<size_t>(in) * out);
  for (auto& v : w) v = F32ToBf16(nd(rng) * 0.03f);
  const AudioEmbedder emb(w, in, out);

  const int64_t n = 9;
  std::vector<float> frames(static_cast<size_t>(n) * in);
  for (auto& v : frames) v = nd(rng) * 0.2f;
  for (int k = 0; k < in; ++k) frames[static_cast<size_t>(4) * in + k] = 0.0f;  // an all-zero (silent) frame
  const std::vector<uint16_t> y1 = emb.Embed(frames.data(), n, 1);
  const std::vector<uint16_t> y5 = emb.Embed(frames.data(), n, 5);
  CHECK(y1 == y5, "bit-identical for 1 and 5 threads");

  const std::vector<float> ref = RefEmbed(frames, n, w, in, out);
  int bad = 0;
  for (size_t i = 0; i < ref.size(); ++i) {
    const float got = Bf16ToF32(y1[i]);
    const float tol = std::fabs(ref[i]) * 0.0078125f + 1e-6f;  // 1 bf16 ulp (the fp32 vs double accumulate order)
    if (std::fabs(got - ref[i]) > tol) ++bad;
  }
  CHECK(bad == 0, "embedder == double recipe within 1 bf16 ulp");
  bool silent_zero = true;
  for (int j = 0; j < out; ++j) silent_zero = silent_zero && Bf16ToF32(y1[static_cast<size_t>(4) * out + j]) == 0.0f;
  CHECK(silent_zero, "an all-zero frame embeds to exactly zero (eps keeps the norm finite)");

  // Rows are independent: perturbing frame 7 changes row 7 and no other row (the embedder cannot leak audio
  // information backwards; the model's causality is separately held by the attention masks).
  std::vector<float> f2 = frames;
  f2[static_cast<size_t>(7) * in + 3] += 1.0f;
  const std::vector<uint16_t> y2 = emb.Embed(f2.data(), n, 3);
  bool others_same = true, row7_changed = false;
  for (int64_t r = 0; r < n; ++r) {
    const bool same = std::equal(y1.begin() + r * out, y1.begin() + (r + 1) * out, y2.begin() + r * out);
    if (r == 7) row7_changed = !same;
    else others_same = others_same && same;
  }
  CHECK(others_same && row7_changed, "rows are independent");
  CHECK(Throws([&] { AudioEmbedder(std::vector<uint16_t>(10), in, out); }, "expected"), "wrong weight size refused");
  CHECK(emb.Embed(frames.data(), 0).empty(), "zero rows");
}

// A minimal container: one safetensors file holding only the audio projection in the converter's layout
// (bf16 passthrough = U8 [rows, cols, 2]) -- LoadAudioEmbedder reads exactly that tensor.
std::string WriteTinyContainer(const std::string& name, const std::vector<int64_t>& shape, const std::vector<uint16_t>& w,
                               const std::string& tensor_name = AudioWeightName()) {
  const std::filesystem::path path = std::filesystem::temp_directory_path() / name;
  const size_t bytes = w.size() * 2;
  std::string hdr = "{\"" + tensor_name + "\":{\"dtype\":\"U8\",\"shape\":[";
  for (size_t i = 0; i < shape.size(); ++i) hdr += (i ? "," : "") + std::to_string(shape[i]);
  hdr += "],\"data_offsets\":[0," + std::to_string(bytes) + "]},\"__metadata__\":{}}";
  while (hdr.size() % 8 != 0) hdr += ' ';
  std::ofstream f(path, std::ios::binary);
  const uint64_t n = hdr.size();
  f.write(reinterpret_cast<const char*>(&n), 8);
  f.write(hdr.data(), static_cast<std::streamsize>(hdr.size()));
  f.write(reinterpret_cast<const char*>(w.data()), static_cast<std::streamsize>(bytes));
  return path.string();
}

void TestContainerLoad() {
  std::mt19937 rng(11);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::vector<uint16_t> w(static_cast<size_t>(kEmbedHidden) * kSamplesPerToken);
  for (auto& v : w) v = F32ToBf16(nd(rng) * 0.03f);
  const std::string good = WriteTinyContainer("r4dx_test_audio_good.safetensors", {kEmbedHidden, kSamplesPerToken, 2}, w);
  CHECK(ContainerHasAudio(good), "ContainerHasAudio: present");
  const AudioEmbedder loaded = LoadAudioEmbedder(good);
  const AudioEmbedder direct(w);
  std::vector<float> frames(static_cast<size_t>(2) * kSamplesPerToken);
  for (auto& v : frames) v = nd(rng) * 0.1f;
  CHECK(loaded.Embed(frames.data(), 2) == direct.Embed(frames.data(), 2), "LoadAudioEmbedder == the embedder built from the same bytes");

  const std::string none = WriteTinyContainer("r4dx_test_audio_none.safetensors", {4, 2}, std::vector<uint16_t>(4), "text.final_norm");
  CHECK(!ContainerHasAudio(none), "ContainerHasAudio: absent");
  CHECK(Throws([&] { LoadAudioEmbedder(none); }, "--audio on"), "a container without the tensor names the --audio on flag");
  const std::string bad = WriteTinyContainer("r4dx_test_audio_bad.safetensors", {8, 4, 2}, std::vector<uint16_t>(32));
  CHECK(Throws([&] { LoadAudioEmbedder(bad); }, "unexpected shape"), "a wrong-shaped tensor is refused");
  std::error_code ec;
  for (const std::string& p : {good, none, bad}) std::filesystem::remove(p, ec);
}
}  // namespace

int main() {
  TestFraming();
  TestExpansion();
  TestWav();
  TestEmbedder();
  TestContainerLoad();
  if (g_fail != 0) {
    std::printf("test_audio: %d failure(s)\n", g_fail);
    return 1;
  }
  std::printf("test_audio: all checks passed\n");
  return 0;
}
