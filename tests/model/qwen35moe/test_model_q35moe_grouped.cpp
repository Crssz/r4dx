// The on-device grouped MoE decode (Qwen35GroupedMoeDecode) against the host-routed MoE, end to end on a TRELLIS tiny container
// (hidden 3072, expert width 1024, top-8 of 16 experts, 4 layers: gate_up at 3 or 4 bits per layer, down at 4).
//
// For each prompt: Prefill (always host-routed: more than one row), then a free-running greedy continuation twice on the same
// model, once with SetGroupedDecode(false) and once with true. The continuations must be IDENTICAL token for token; a teacher-forced
// run (the host-routed tokens fed to both) additionally compares the logits row by row, and a layer-level section runs real MoE inputs
// (70 residual rows x 4 layers, one row per call) through both paths and compares moe_out element by element (expected bit-exact
// bar a handful of elements: the paths round alike). The decode wall time per token of both is
// printed. A bf16 container (no trellis experts) must never take the grouped path.
//
//   R4DX_Q35MOE_TRELLIS      the trellis container (default E:\models\r4dx\q35moe-tiny\tiny-mix.r4dx)
//   R4DX_Q35MOE_TINY_DIR     prompt source: golden_fp8\T{T}.safetensors input_ids (default E:\models\r4dx\tiny-q35moe)
//   R4DX_Q35_STEPS           decode steps per run (default 48)
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <functional>
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

struct Run {
  std::vector<int32_t> tokens;
  std::vector<std::vector<float>> logits;  // the logits row behind tokens[i]
  double ms_per_token = 0;
};

// Prefill `ids`, then `steps` decode steps. forced == nullptr: free run (each step feeds the previous argmax); otherwise step i
// feeds forced[i].
Run Generate(Qwen35MoeModel& m, bool grouped, const std::vector<int32_t>& ids, int steps, const std::vector<int32_t>* forced) {
  m.SetGroupedDecode(grouped);
  m.Reset();
  Run r;
  std::vector<float> lg = m.Prefill(ids);
  int32_t tok = static_cast<int32_t>(ArgMax(lg.data(), static_cast<int64_t>(lg.size())));
  r.tokens.push_back(tok);
  r.logits.push_back(lg);
  const auto t0 = Clock::now();
  for (int i = 0; i < steps; ++i) {
    const int32_t in = forced != nullptr ? (*forced)[static_cast<size_t>(i)] : tok;
    lg = m.DecodeStep(in);
    tok = static_cast<int32_t>(ArgMax(lg.data(), static_cast<int64_t>(lg.size())));
    r.tokens.push_back(tok);
    r.logits.push_back(lg);
  }
  r.ms_per_token = std::chrono::duration<double, std::milli>(Clock::now() - t0).count() / steps;
  return r;
}

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
  const int steps = std::getenv("R4DX_Q35_STEPS") != nullptr ? std::atoi(std::getenv("R4DX_Q35_STEPS")) : 48;
  const int prompts[] = {1, 7, 70, 300};
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
  model->SetPrefillMoe(false);  // this test is about the decode MoE: the prompts always prefill host-routed (the prefill MoE has its own test)
  const int64_t layers = model->NumLoadedLayers();
  std::printf("trellis container %s on device %d: %lld layer(s), %lld take the grouped decode MoE\n", path.c_str(), TestDevice(),
              static_cast<long long>(layers), static_cast<long long>(model->GroupedDecodeLayers()));
  Check(model->GroupedDecodeLayers() == layers, "every layer of the tiny trellis container must qualify for the grouped decode MoE");

  for (int T : prompts) {
    const Golden g(T, GoldenPath(T));
    std::printf("\n-- prompt T=%d, %d decode steps\n", T, steps);
    const Run host = Generate(*model, false, g.ids(), steps, nullptr);
    const Run grp = Generate(*model, true, g.ids(), steps, nullptr);
    size_t agree = 0;
    while (agree < host.tokens.size() && host.tokens[agree] == grp.tokens[agree]) ++agree;
    std::printf("   free run: %zu of %zu tokens identical (host-routed vs grouped)\n", agree, host.tokens.size());
    // The grouped path is built to round like the host-routed one (same split-group order, silu rounding and combine order), so the
    // greedy continuations must agree on every token even though this random-weight net is noise-amplifying (HF's own bf16 run leaves
    // its fp32 run within a few tokens).
    Check(agree == host.tokens.size(), "T" + std::to_string(T) + ": the grouped free run leaves the host-routed one at token " + std::to_string(agree));

    // teacher forced: both consume the host-routed tokens, so the logits rows are comparable even after a near-tie flip
    std::vector<int32_t> forced(host.tokens.begin(), host.tokens.begin() + steps);
    const Run hf = Generate(*model, false, g.ids(), steps, &forced);
    const Run gf = Generate(*model, true, g.ids(), steps, &forced);
    double worst_max = 0, worst_mean = 0, sum_mean = 0;
    int argmax_diff = 0;
    for (size_t i = 1; i < hf.logits.size(); ++i) {  // row 0 is the prefill's (identical code)
      const FStats fs = CompareF32(gf.logits[i].data(), hf.logits[i].data(), static_cast<int64_t>(hf.logits[i].size()));
      worst_max = std::max(worst_max, fs.max_norm);
      worst_mean = std::max(worst_mean, fs.mean_norm);
      sum_mean += fs.mean_norm;
      if (hf.tokens[i] != gf.tokens[i]) {
        ++argmax_diff;
        std::vector<float> sorted = hf.logits[i];
        std::partial_sort(sorted.begin(), sorted.begin() + 2, sorted.end(), std::greater<float>());
        const double margin = sorted[0] - sorted[1];
        Check(margin <= 0.126, "T" + std::to_string(T) + " step " + std::to_string(i) + ": the argmax flips although the host margin (" + std::to_string(margin) +
                                   ") exceeds two bf16 steps of a top logit");
        std::printf("   step %zu: argmax flips (host margin top1-top2 = %.4g, logits rms %.4g, row max|err| %.4g)\n", i, margin, fs.rms_ref,
                    fs.max_abs);
      }
      Check(!fs.nan, "T" + std::to_string(T) + " step " + std::to_string(i) + ": NaN / Inf logits");
    }
    std::printf("   teacher forced: logits max|err|/rms(ref) worst %.4f, mean|err|/rms(ref) worst %.5f avg %.5f; argmax differs in %d of %zu steps\n",
                worst_max, worst_mean, sum_mean / static_cast<double>(hf.logits.size() - 1), argmax_diff, hf.logits.size() - 1);
    Check(worst_mean <= 0.03, "T" + std::to_string(T) + ": teacher-forced logits drift above the bf16-noise bound (mean/rms " +
                                  std::to_string(worst_mean) + " > 0.03)");
    Check(argmax_diff == 0, "T" + std::to_string(T) + ": teacher-forced argmax flips in " + std::to_string(argmax_diff) + " of " + std::to_string(steps) + " steps");
    std::printf("   decode wall per token: host-routed %.2f ms, grouped %.2f ms (tiny model, 4 layers, incl. logits readback)\n",
                host.ms_per_token, grp.ms_per_token);
  }

  // ---- layer level: the same MoE input through both paths ---------------------------------------------------------------------
  // The decode logits above carry the chaotic amplification of this random-weight net, so judge the MoE block itself: take the real
  // per-layer residual of a 70-token prompt, run each row alone through ONE layer (RunLayerIsolated, one row = a decode-shaped MoE
  // call) once host-routed and once grouped, and compare the layer's moe_out bit for bit.
  {
    const int T = 70;
    const Golden g(T, GoldenPath(T));
    CaptureStore pre;
    model->SetGroupedDecode(false);
    model->Reset();
    model->SetCapture(pre.Fn());
    model->Prefill(g.ids());
    model->SetCapture(nullptr);
    const int64_t hidden = model->Config().hidden_size;
    int64_t total = 0, differing = 0;
    double max_rel = 0;  // max |grouped - host| / rms(host row)
    for (int layer = 0; layer < static_cast<int>(layers); ++layer) {
      const std::string pfx = "layer" + std::to_string(layer) + ".";
      const std::vector<uint16_t>& resid = pre.Get(pfx + "residual_in").bf16;
      int64_t l_total = 0, l_diff = 0;
      double l_rel = 0;
      for (int r = 0; r < T; ++r) {
        std::vector<uint16_t> out[2];
        for (int mode = 0; mode < 2; ++mode) {
          model->SetGroupedDecode(mode == 1, /*with_capture=*/true);
          CaptureStore store;
          model->SetCapture(store.Fn());
          model->RunLayerIsolated({g.ids()[static_cast<size_t>(r)]}, layer, resid.data() + static_cast<size_t>(r) * hidden);
          model->SetCapture(nullptr);
          out[mode] = store.Get(pfx + "moe_out").bf16;
        }
        double sq = 0;
        for (int64_t d = 0; d < hidden; ++d) sq += static_cast<double>(Bf(out[0][static_cast<size_t>(d)])) * Bf(out[0][static_cast<size_t>(d)]);
        const double rms = std::sqrt(sq / static_cast<double>(hidden));
        for (int64_t d = 0; d < hidden; ++d) {
          const double hv = Bf(out[0][static_cast<size_t>(d)]), gv = Bf(out[1][static_cast<size_t>(d)]);
          ++l_total;
          if (out[0][static_cast<size_t>(d)] != out[1][static_cast<size_t>(d)]) {
            ++l_diff;
            if (rms > 0) l_rel = std::max(l_rel, std::fabs(gv - hv) / rms);
          }
        }
      }
      std::printf("   layer %d moe_out, %d single-row calls: %lld of %lld elements differ (%.4f%%), max |diff| / rms(row) %.5f\n", layer, T,
                  static_cast<long long>(l_diff), static_cast<long long>(l_total), 100.0 * static_cast<double>(l_diff) / static_cast<double>(l_total),
                  l_rel);
      total += l_total;
      differing += l_diff;
      max_rel = std::max(max_rel, l_rel);
    }
    model->SetGroupedDecode(true);
    model->Reset();
    // Same GEMV split-group order, same silu rounding, same combine order: the two paths are meant to be bit-identical.
    Check(static_cast<double>(differing) <= 1e-4 * static_cast<double>(total), "more than 0.01% of the MoE output elements differ between the paths");
    Check(max_rel <= 0.03, "a MoE output element differs by more than 3% of the row rms (more than one bf16 step of the largest element)");
  }

  // A bf16 container has no trellis experts: the grouped path must stay off.
  if (Exists(ContainerPath("bf16"))) {
    auto bf = LoadTiny("bf16", false, {}, 0, 1024);
    std::printf("\nbf16 container: %lld layer(s) take the grouped decode MoE\n", static_cast<long long>(bf->GroupedDecodeLayers()));
    Check(bf->GroupedDecodeLayers() == 0, "a bf16 container must not take the grouped decode MoE");
  }

  std::printf("\n%s (%d failure(s))\n", g_failures == 0 ? "PASS" : "FAIL", g_failures);
  return g_failures == 0 ? 0 : 1;
}
