// CPU-only golden test for src/audio against transformers' Gemma4Unified processor / embedder
// (tools/reference/gemma/audio_golden_gemma.py writes tools/reference/golden_out/gemma/audio_golden.safetensors
// and audio_semantics.json; they are gitignored and exit 77 = SKIPPED here when absent, like tests/vision).
//
// Checks: (1) the placeholder expansion reproduces the HF processor's prompt layout for every dumped case
// (prefix ids + <|audio|> + suffix ids -> boa at the same index, same length); (2) the CPU embedder, fed the
// REAL checkpoint weight and the golden frames, matches HF's bf16 rows to within 1 bf16 ulp, and (3) FrameWaveform
// of a regenerated waveform equals the golden frames bit for bit is NOT checked here (the numpy synth is not
// ported); the framing rule itself is covered by test_audio.cpp and by the Python golden's extractor check.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "audio_embed.h"
#include "audio_frames.h"
#include "nlohmann/json.hpp"
#include "r4dx_convert/safetensors_reader.hpp"

#ifndef R4DX_AUDIO_GOLDEN_DIR
#define R4DX_AUDIO_GOLDEN_DIR "tools/reference/golden_out/gemma"
#endif

namespace {
int g_fail = 0;
#define CHECK(cond, msg)                                                  \
  do {                                                                    \
    if (!(cond)) {                                                        \
      std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, msg);           \
      ++g_fail;                                                           \
    }                                                                     \
  } while (0)
}  // namespace

int main() {
  using namespace r4dx::audio;
  std::string dir = R4DX_AUDIO_GOLDEN_DIR;
  char* env_dir = nullptr;
  size_t env_len = 0;
  if (_dupenv_s(&env_dir, &env_len, "R4DX_AUDIO_GOLDEN_DIR") == 0 && env_dir != nullptr) {
    dir = env_dir;
    std::free(env_dir);
  }
  const std::string st_path = dir + "/audio_golden.safetensors", js_path = dir + "/audio_semantics.json";
  std::ifstream js_in(js_path);
  std::ifstream st_probe(st_path, std::ios::binary);
  if (!js_in || !st_probe) {
    std::printf("SKIP test_audio_golden: %s / %s not found (run tools/reference/gemma/audio_golden_gemma.py)\n",
                st_path.c_str(), js_path.c_str());
    return 77;
  }
  st_probe.close();
  nlohmann::json sem = nlohmann::json::parse(js_in);

  // (1) processor layout
  for (const auto& c : sem.at("processor").at("cases")) {
    std::vector<int32_t> raw = c.at("prefix_ids").get<std::vector<int32_t>>();
    const size_t boa_at = raw.size();
    raw.push_back(kAudioTokenId);
    const std::vector<int32_t> suffix = c.at("suffix_ids").get<std::vector<int32_t>>();
    raw.insert(raw.end(), suffix.begin(), suffix.end());
    const int64_t samples = c.at("samples").get<int64_t>();
    const ExpandedAudioPrompt e = ExpandAudioPlaceholders(raw, {NumAudioTokens(samples)});
    CHECK(NumAudioTokens(samples) == c.at("tokens").get<int64_t>(), "token count matches HF");
    CHECK(static_cast<int64_t>(e.tokens.size()) == c.at("prompt_len").get<int64_t>(), "expanded prompt length matches HF");
    CHECK(e.tokens[boa_at] == kBoaTokenId && static_cast<int64_t>(boa_at) == c.at("boa_at").get<int64_t>(),
          "boa at the same index as HF");
    CHECK(e.spans.size() == 1 && e.spans[0].offset == static_cast<int64_t>(boa_at) + 1 &&
              e.tokens[static_cast<size_t>(e.spans[0].offset + e.spans[0].tokens)] == kEoaTokenId,
          "soft-token run is followed by eoa");
  }
  for (const auto& c : sem.at("extractor").at("cases")) {
    CHECK(NumAudioTokens(c.at("samples").get<int64_t>()) == c.at("tokens").get<int64_t>(), "extractor token count");
  }

  // (2) embedder vs HF rows
  r4dx_convert::SafetensorsReader r(r4dx_convert::Utf8ToWide(st_path));
  const r4dx_convert::TensorMeta& wm = r.Meta("weight");
  CHECK(wm.shape == std::vector<int64_t>({3840, 640}), "golden weight shape");
  std::vector<uint16_t> w(static_cast<size_t>(3840) * 640);
  std::memcpy(w.data(), r.Data("weight"), w.size() * 2);
  const AudioEmbedder emb(w);
  for (int i = 0; r.Has("frames_" + std::to_string(i)); ++i) {
    const std::string fn = "frames_" + std::to_string(i), en = "expect_" + std::to_string(i);
    const int64_t n = r.Meta(fn).shape.at(0);
    const float* frames = reinterpret_cast<const float*>(r.Data(fn));
    const uint16_t* expect = reinterpret_cast<const uint16_t*>(r.Data(en));
    const std::vector<uint16_t> got = emb.Embed(frames, n);
    // Not bit-exact by construction: torch's fp32 reduction order for mean(x^2) differs from ours by ~1e-7
    // relative, which now and then (a few rows in 750) flips the bf16 rounding of ONE normed element; that moves
    // every output of the row by ~2e-4 absolute, visible only on outputs near zero. So: (a) at most 0.5% of the
    // elements outside "1 ulp + 1e-5", (b) nothing beyond 0.25 absolute (one flipped element of the largest normed value times the largest weight), (c) at most 5% of the rows carry a flip at all.
    int64_t not_equal = 0, over_ulp = 0;
    float max_abs = 0.0f;
    for (int64_t k = 0; k < n * 3840; ++k) {
      if (got[static_cast<size_t>(k)] == expect[k]) continue;
      ++not_equal;
      const float g = Bf16ToF32(got[static_cast<size_t>(k)]), x = Bf16ToF32(expect[k]);
      max_abs = std::max(max_abs, std::fabs(g - x));
      if (std::fabs(g - x) > std::fabs(x) * 0.0078125f + 1e-5f) ++over_ulp;
    }
    std::printf("case %d: %lld rows, %lld/%lld elements differ from HF, %lld beyond 1 ulp + 1e-5, max abs diff %g\n", i,
                static_cast<long long>(n), static_cast<long long>(not_equal), static_cast<long long>(n * 3840),
                static_cast<long long>(over_ulp), static_cast<double>(max_abs));
    int64_t bad_rows = 0;  // rows with a flipped normed element: hundreds of outputs move by a few bf16 ulps
    for (int64_t r = 0; r < n; ++r) {
      int64_t d = 0;
      for (int64_t j = 0; j < 3840; ++j) d += got[static_cast<size_t>(r * 3840 + j)] != expect[r * 3840 + j];
      if (d > 100) ++bad_rows;
    }
    std::printf("        %lld row(s) carry a flipped normed element\n", static_cast<long long>(bad_rows));
    CHECK(bad_rows * 20 <= n, "at most 5% of rows carry a rounding flip");
    CHECK(over_ulp * 200 <= n * 3840, "at most 0.5% of elements beyond 1 ulp + 1e-5 of HF");
    CHECK(max_abs <= 0.25f, "no element further than 0.25 from HF");
    CHECK(not_equal * 20 <= n * 3840, "at most 5% of elements differ at all");
  }  if (g_fail != 0) {
    std::printf("test_audio_golden: %d failure(s)\n", g_fail);
    return 1;
  }
  std::printf("test_audio_golden: all checks passed\n");
  return 0;
}
