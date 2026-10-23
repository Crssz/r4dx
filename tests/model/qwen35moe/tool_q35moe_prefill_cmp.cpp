// tool_q35moe_prefill_cmp: the device-side prefill MoE against the host-routed MoE on the REAL container, layer by layer.
//
// One full Prefill of the first N tokens of a segment records every layer's residual input (host-routed MoE, capture on). Then each layer
// alone (RunLayerIsolated over that residual, so both paths start from identical bits) runs once host-routed and once grouped; moe_out and the
// router's top-k ids are compared. A layer where the paths part by more than rounding noise shows up as a row of the table.
//
//   tool_q35moe_prefill_cmp --model <container> --tokens <tokens.json> [--segment name] [--count 512] [--layers a,b]
//   env R4DX_Q35_PREFILL_SKG="gu,down" picks the grouped path's split-group counts (default 1,1).
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <set>
#include <string>
#include <vector>

#include "q35moe_model_util.h"

using namespace q35m;

int main(int argc, char** argv) {
  try {
    Unbuffer();
    std::string model_path, tokens_path, segment;
    int64_t count = 512;
    std::set<int> only;
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
      if (a == "--model") model_path = next();
      else if (a == "--tokens") tokens_path = next();
      else if (a == "--segment") segment = next();
      else if (a == "--count") count = std::stoll(next());
      else if (a == "--layers") {
        std::string s = next();
        size_t p = 0;
        while (p < s.size()) {
          size_t q = s.find(',', p);
          if (q == std::string::npos) q = s.size();
          only.insert(std::stoi(s.substr(p, q - p)));
          p = q + 1;
        }
      }
    }
    if (model_path.empty() || tokens_path.empty()) {
      std::fprintf(stderr, "usage: tool_q35moe_prefill_cmp --model <container> --tokens <tokens.json> [--segment name] [--count N] [--layers a,b]\n");
      return 2;
    }
    std::ifstream f(tokens_path, std::ios::binary);
    const nlohmann::json j = nlohmann::json::parse(std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>()));
    const nlohmann::json* seg = nullptr;
    for (const auto& s : j.at("segments")) {
      if (segment.empty() || s.at("name").get<std::string>() == segment) {
        seg = &s;
        break;
      }
    }
    if (seg == nullptr) throw std::runtime_error("no such segment");
    std::vector<int32_t> ids = seg->at("token_ids").get<std::vector<int32_t>>();
    if (static_cast<int64_t>(ids.size()) > count) ids.resize(static_cast<size_t>(count));
    std::printf("segment %s: %zu tokens\n", seg->at("name").get<std::string>().c_str(), ids.size());

    Qwen35MoeModelOptions o;
    o.container_path = model_path;
    o.max_ctx = static_cast<int64_t>(ids.size()) + 64;
    o.grouped_decode = 1;
    auto model = Qwen35MoeModel::Load(o);
    const int64_t layers = model->NumLoadedLayers();
    const int64_t hidden = model->Config().hidden_size;
    std::printf("%lld layers, %lld take the grouped MoE\n", static_cast<long long>(layers), static_cast<long long>(model->GroupedDecodeLayers()));

    CaptureStore pre;
    model->SetPrefillMoe(false);
    model->SetGroupedDecode(true, true);
    model->Reset();
    model->SetCapture(pre.Fn());
    model->Prefill(ids);
    model->SetCapture(nullptr);
    std::printf("%-6s %10s %12s %12s %12s %10s\n", "layer", "ids differ", "elems diff %", "mean|e|/rms", "max|e|/rms", "gu/down kb");
    for (int layer = 0; layer < static_cast<int>(layers); ++layer) {
      if (!only.empty() && only.count(layer) == 0) continue;
      const std::string pfx = "layer" + std::to_string(layer) + ".";
      const std::vector<uint16_t>& resid = pre.Get(pfx + "residual_in").bf16;
      std::vector<uint16_t> out[2];
      std::vector<int32_t> topk[2];
      for (int mode = 0; mode < 2; ++mode) {
        model->SetPrefillMoe(mode == 1);
        CaptureStore store;
        model->SetCapture(store.Fn());
        model->RunLayerIsolated(ids, layer, resid.data());
        model->SetCapture(nullptr);
        out[mode] = store.Get(pfx + "moe_out").bf16;
        topk[mode] = store.Get(pfx + "topk_ids").i32;
      }
      const int64_t n = static_cast<int64_t>(out[0].size());
      double sq = 0, sum = 0, mx = 0;
      int64_t differing = 0;
      for (int64_t i = 0; i < n; ++i) {
        const double hv = Bf(out[0][static_cast<size_t>(i)]), gv = Bf(out[1][static_cast<size_t>(i)]);
        sq += hv * hv;
        sum += std::fabs(gv - hv);
        mx = std::max(mx, std::fabs(gv - hv));
        differing += out[0][static_cast<size_t>(i)] != out[1][static_cast<size_t>(i)] ? 1 : 0;
      }
      int64_t id_diff = 0;
      const size_t k = static_cast<size_t>(model->Config().num_experts_per_tok);
      for (size_t r = 0; r + k <= topk[0].size() && r + k <= topk[1].size(); r += k) {
        std::set<int> a(topk[0].begin() + r, topk[0].begin() + r + k), b(topk[1].begin() + r, topk[1].begin() + r + k);
        id_diff += a != b ? 1 : 0;
      }
      const double rms = std::sqrt(sq / static_cast<double>(n));
      std::printf("%-6d %10lld %12.4f %12.6f %12.4f\n", layer, static_cast<long long>(id_diff), 100.0 * static_cast<double>(differing) / static_cast<double>(n),
                  sum / static_cast<double>(n) / rms, mx / rms);
    }
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "EXCEPTION: %s\n", e.what());
    return 2;
  }
}