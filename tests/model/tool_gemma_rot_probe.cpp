// tests/model/tool_gemma_rot_probe.cpp -- per-layer probe of a ROTATED Gemma container against the unrotated one
// (debugging tool, GPU, one device). Pass A loads the unrotated container and walks the first T tokens of a
// chat_gemma.json segment through every layer with DebugLayerForward, recording each layer's input and output
// (original basis). Pass B loads the rotated container and replays every layer on pass A's recorded input
// (DebugLayerForward is basis-agnostic), printing rel L2(B, A) per layer: a layer whose math is wrong under the
// rotation stands out, independent of upstream error.
//
//   $env:HIP_VISIBLE_DEVICES='1'; tool_gemma_rot_probe.exe <plain.r4dx> <rotated.r4dx> <chat_gemma.json> [segment] [T]
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>

#include "gemma_model.h"
#include "nlohmann/json.hpp"
#include "r4dx/core/dtype.hpp"
#include "r4dx/core/error.hpp"
#include "test_common.h"

using namespace r4dx;
using namespace r4dx::model;

namespace {

std::vector<uint16_t> EmbedRows(GemmaModel& m, const std::vector<int32_t>& ids) {
  const GemmaConfig& cfg = m.Config();
  const int64_t hidden = cfg.hidden_size;
  std::vector<uint16_t> out(ids.size() * static_cast<size_t>(hidden));
  const uint16_t* table = m.GetContainer().EmbedTokensDevice().data();
  for (size_t t = 0; t < ids.size(); ++t) {
    R4DX_HIP_CHECK(hipMemcpy(out.data() + t * hidden, table + static_cast<int64_t>(ids[t]) * hidden, hidden * 2,
                             hipMemcpyDeviceToHost));
  }
  const float scale = static_cast<float>(cfg.EmbedScale());
  for (auto& v : out) v = core::FloatToBf16(core::Bf16ToFloat(v) * scale);  // HF: bf16 table * bf16(sqrt(hidden))
  return out;
}

double RelL2(const std::vector<uint16_t>& b, const std::vector<uint16_t>& a) {
  double num = 0, den = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    const double x = core::Bf16ToFloat(a[i]), y = core::Bf16ToFloat(b[i]);
    num += (y - x) * (y - x);
    den += x * x;
  }
  return std::sqrt(num / (den + 1e-30));
}

GemmaModel LoadBf16(const std::string& path) {
  GemmaModelOptions o;
  o.container_path = path;
  o.layout = Layout::kBf16;
  o.max_ctx = 1024;
  o.kv = GemmaKvMode::kBf16;
  return GemmaModel::Load(o);
}

}  // namespace

int main(int argc, char** argv) {
  return r4dx_test::RunGuardedMain("tool_gemma_rot_probe", [&]() -> int {
    if (argc < 4) {
      std::fprintf(stderr, "usage: tool_gemma_rot_probe <plain.r4dx> <rotated.r4dx> <chat_gemma.json> [segment] [T]\n");
      return 2;
    }
    const std::string seg = argc > 4 ? argv[4] : "en_tides";
    const int64_t T = argc > 5 ? std::stoll(argv[5]) : 64;
    std::ifstream f(argv[3]);
    const nlohmann::json doc = nlohmann::json::parse(f);
    std::vector<int32_t> ids;
    for (const auto& s : doc.at("segments"))
      if (s.at("name") == seg) ids = s.at("token_ids").get<std::vector<int32_t>>();
    if (static_cast<int64_t>(ids.size()) < T) throw std::runtime_error("segment too short / not found: " + seg);
    ids.resize(static_cast<size_t>(T));

    std::vector<std::vector<uint16_t>> xin, xout;
    int64_t layers = 0;
    {
      GemmaModel a = LoadBf16(argv[1]);
      layers = a.GetContainer().NumLoadedLayers();
      std::vector<uint16_t> x = EmbedRows(a, ids);
      const char* dump = std::getenv("R4DX_PROBE_DUMP_DIR");  // write each layer's input as raw bf16 [T, hidden]
      for (int64_t i = 0; i < layers; ++i) {
        if (dump != nullptr && *dump != '\0') {
          std::ofstream o(std::string(dump) + "/xin_L" + std::to_string(i) + ".bf16", std::ios::binary);
          o.write(reinterpret_cast<const char*>(x.data()), static_cast<std::streamsize>(x.size() * 2));
        }
        xin.push_back(x);
        x = a.DebugLayerForward(i, x, T, 0);
        xout.push_back(x);
      }
    }
    GemmaModel b = LoadBf16(argv[2]);
    std::printf("layer  type     rel L2(rotated, plain) on the plain input   (|out| rms)\n");
    for (int64_t i = 0; i < layers; ++i) {
      const std::vector<uint16_t> y = b.DebugLayerForward(i, xin[static_cast<size_t>(i)], T, 0);
      double ss = 0;
      for (uint16_t v : xout[static_cast<size_t>(i)]) ss += core::Bf16ToFloat(v) * core::Bf16ToFloat(v);
      std::printf("%5lld  %-7s  %.3e   (%.2f)\n", static_cast<long long>(i), b.Config().IsFullLayer(i) ? "full" : "sliding",
                  RelL2(y, xout[static_cast<size_t>(i)]), std::sqrt(ss / xout[static_cast<size_t>(i)].size()));
    }
    return 0;
  });
}
