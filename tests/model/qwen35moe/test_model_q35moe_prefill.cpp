// The device-side prefill MoE (Qwen35PrefillMoe) against the host-routed MoE on the TRELLIS tiny container (hidden 3072, expert width 1024,
// top-8 of 16 experts, 4 layers; gate_up 3 or 4 bits per layer, down 4).
//
//  1. layer level: each of the first rows of a prompt's real residual goes through ONE layer (RunLayerIsolated, so both paths start from
//     the same input) in a chunk of T rows, host-routed and grouped, and moe_out is compared element by element;
//  2. whole prompt: Prefill + a greedy continuation with the prefill MoE off and on; last-row logits and tokens compared.
//
//   R4DX_Q35MOE_TRELLIS      the trellis container (default E:\models\r4dx\q35moe-tiny\tiny-mix.r4dx)
//   R4DX_Q35MOE_TINY_DIR     prompt source: golden_fp8\T{T}.safetensors input_ids (default E:\models\r4dx\tiny-q35moe)
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "q35moe_model_util.h"

using namespace q35m;

namespace {

int g_failures = 0;
void Check(bool cond, const std::string& what) {
  if (!cond) {
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++g_failures;
  }
}

std::string TrellisPath() {
  const char* e = std::getenv("R4DX_Q35MOE_TRELLIS");
  return e != nullptr && e[0] != 0 ? e : std::string("E:\\models\\r4dx\\q35moe-tiny\\tiny-mix.r4dx");
}

using Clock = std::chrono::steady_clock;

}  // namespace

int RunTest();
int main() {
  try {
    return RunTest();
  } catch (const std::exception& e) {
    std::fprintf(stderr, "EXCEPTION: %s\n", e.what());
    return 2;
  }
}

int RunTest() {
  Unbuffer();
  const std::string path = TrellisPath();
  if (!Exists(path)) {
    std::printf("SKIP: %s missing\n", path.c_str());
    return kSkip;
  }
  if (VisibleDevices() < 1) {
    std::printf("SKIP: no HIP device\n");
    return kSkip;
  }
  const int prompts[] = {7, 70, 300};
  for (int T : prompts) {
    if (!Exists(GoldenPath(T))) {
      std::printf("SKIP: %s missing (prompt ids)\n", GoldenPath(T).c_str());
      return kSkip;
    }
  }
  Qwen35MoeModelOptions o;
  o.container_path = path;
  o.max_ctx = 4096;
  o.devices = {TestDevice(), TestDevice()};
  o.grouped_decode = 1;
  auto model = Qwen35MoeModel::Load(o);
  const int64_t layers = model->NumLoadedLayers();
  const int64_t hidden = model->Config().hidden_size;
  std::printf("trellis container %s: %lld layer(s), %lld take the grouped MoE\n", path.c_str(), static_cast<long long>(layers),
              static_cast<long long>(model->GroupedDecodeLayers()));
  Check(model->GroupedDecodeLayers() == layers, "every layer of the tiny trellis container must qualify");

  // ---- 1. layer level -----------------------------------------------------------------------------------------------------------
  for (int T : prompts) {
    const Golden g(T, GoldenPath(T));
    CaptureStore pre;
    model->SetPrefillMoe(false);
    model->SetGroupedDecode(true, true);
    model->Reset();
    model->SetCapture(pre.Fn());
    model->Prefill(g.ids());
    model->SetCapture(nullptr);
    for (int layer = 0; layer < static_cast<int>(layers); ++layer) {
      const std::string pfx = "layer" + std::to_string(layer) + ".";
      const std::vector<uint16_t>& resid = pre.Get(pfx + "residual_in").bf16;
      std::vector<uint16_t> out[2];
      for (int mode = 0; mode < 2; ++mode) {
        model->SetPrefillMoe(mode == 1);
        CaptureStore store;
        model->SetCapture(store.Fn());
        model->RunLayerIsolated(g.ids(), layer, resid.data());
        model->SetCapture(nullptr);
        out[mode] = store.Get(pfx + "moe_out").bf16;
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
        if (!std::isfinite(gv)) Check(false, "non-finite grouped moe_out");
      }
      const double rms = std::sqrt(sq / static_cast<double>(n));
      std::printf("   T=%3d layer %d moe_out: %.3f%% of %lld elements differ, mean|err|/rms %.5f, max|err|/rms %.4f\n", T, layer,
                  100.0 * static_cast<double>(differing) / static_cast<double>(n), static_cast<long long>(n), sum / static_cast<double>(n) / rms,
                  mx / rms);
      Check(sum / static_cast<double>(n) / rms <= 0.01, "T" + std::to_string(T) + " layer " + std::to_string(layer) +
                                                            ": the prefill MoE differs from the host-routed one beyond bf16 noise");
    }
  }

  // ---- 2. whole prompt -----------------------------------------------------------------------------------------------------------
  model->SetGroupedDecode(true, false);
  for (int T : prompts) {
    const Golden g(T, GoldenPath(T));
    std::vector<float> lg[2];
    std::vector<int32_t> toks[2];
    double ms[2] = {0, 0};
    for (int mode = 0; mode < 2; ++mode) {
      model->SetPrefillMoe(mode == 1);
      model->Reset();
      const auto t0 = Clock::now();
      lg[mode] = model->Prefill(g.ids());
      ms[mode] = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
      int32_t tok = static_cast<int32_t>(ArgMax(lg[mode].data(), static_cast<int64_t>(lg[mode].size())));
      toks[mode].push_back(tok);
      for (int i = 0; i < 24; ++i) {
        tok = model->DecodeStepGreedy(tok);
        toks[mode].push_back(tok);
      }
    }
    const FStats fs = CompareF32(lg[1].data(), lg[0].data(), static_cast<int64_t>(lg[0].size()));
    size_t agree = 0;
    while (agree < toks[0].size() && toks[0][agree] == toks[1][agree]) ++agree;
    std::printf("   T=%3d prefill: last-row logits mean|err|/rms %.5f max %.4f; continuation %zu of %zu tokens identical; prefill %.1f ms host-routed, %.1f ms grouped\n",
                T, fs.mean_norm, fs.max_norm, agree, toks[0].size(), ms[0], ms[1]);
    Check(!fs.nan, "NaN logits");
    Check(fs.mean_norm <= 0.03, "T" + std::to_string(T) + ": last-row logits drift above the bf16-noise bound");
    Check(toks[0][0] == toks[1][0], "T" + std::to_string(T) + ": the first token differs");
  }
  model->SetPrefillMoe(true);
  std::printf("\n%s (%d failure(s))\n", g_failures == 0 ? "PASS" : "FAIL", g_failures);
  return g_failures == 0 ? 0 : 1;
}
