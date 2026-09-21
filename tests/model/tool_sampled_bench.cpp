// tests/model/tool_sampled_bench.cpp -- Milestone 6 stage S2 item 4: what a PLAIN SAMPLED decode
// token costs, before and after the device row summary (docs/sampling.md section 10).
//
//   BEFORE: Model::DecodeStep (full fp32 logits D2H, ~993 KB/token) + the full-vocab canonical
//           sampler on the host (an O(vocab) pass, plus a partial selection when a filter is set).
//           This is exactly what src/server and src/cli do today for a temperature > 0 request.
//   AFTER:  Model::DecodeStepSampled -- r4dx_topk_lse_f32 on device, a 516-byte D2H, and a host
//           sampler that touches 64 entries (falling back to the full row only when the summary
//           cannot prove the answer, which this tool counts and reports).
//
// Both paths emit the SAME token for the same seed (tests/model/test_mtp.cpp's
// CheckSampledSummaryPathExact proves it on this very model), so this is a pure cost comparison on
// one trajectory, not an A/B of two different generations.
//
// Same "built, but never add_test()'d" convention as tool_hseed_drift/tool_dflash_probe: it prints
// numbers for a human and docs/sampling.md to read, it asserts no pass/fail contract. Run manually:
//   $env:HIP_VISIBLE_DEVICES='1'; .\build\win-hip\tests\model\tool_sampled_bench.exe
// Optional argv: <container> <layout> <tokens>.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "model.h"
#include "test_common.h"

using r4dx::model::Layout;
using r4dx::model::LayoutFromName;
using r4dx::model::LayoutName;
using r4dx::model::Model;
using r4dx::model::ModelOptions;

namespace {

std::vector<int32_t> MakePromptTokens(int n) {
  std::vector<int32_t> ids(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) ids[static_cast<size_t>(i)] = 100 + (i * 41) % 5000;
  return ids;
}

struct Result {
  double ms_per_token = 0.0;
  int64_t fallbacks = 0;
  std::vector<int32_t> tokens;
};

// The pre-S2 path: full logits D2H + the full-vocab canonical sampler.
Result RunFullVocab(Model& m, const std::vector<int32_t>& prompt,
                     const r4dx::kernels::SampleParams& params, uint64_t seed, int tokens) {
  m.Reset();
  std::mt19937_64 rng = r4dx::kernels::MakeRng(seed);
  const int64_t vocab = m.Config().vocab_size;
  std::vector<float> row = m.Prefill(prompt);
  int32_t tok = r4dx::kernels::SampleCanonical(row.data(), vocab, params,
                                                r4dx::kernels::DrawUniform01(rng));
  Result r;
  r.tokens.push_back(tok);
  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < tokens; ++i) {
    row = m.DecodeStep(tok);
    tok = r4dx::kernels::SampleCanonical(row.data(), vocab, params,
                                          r4dx::kernels::DrawUniform01(rng));
    r.tokens.push_back(tok);
  }
  const auto t1 = std::chrono::steady_clock::now();
  r.ms_per_token = std::chrono::duration<double, std::milli>(t1 - t0).count() / tokens;
  return r;
}

// The S2 path: device row summary + the exact summary sampler.
Result RunSummary(Model& m, const std::vector<int32_t>& prompt,
                   const r4dx::kernels::SampleParams& params, uint64_t seed, int tokens) {
  m.Reset();
  std::mt19937_64 rng = r4dx::kernels::MakeRng(seed);
  const std::vector<float> l0 = m.Prefill(prompt);
  int32_t tok = r4dx::kernels::SampleCanonical(l0.data(), m.Config().vocab_size, params,
                                                r4dx::kernels::DrawUniform01(rng));
  Result r;
  r.tokens.push_back(tok);
  const int64_t fallbacks_before = m.SampledFallbackRows();
  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < tokens; ++i) {
    tok = m.DecodeStepSampled(tok, params, rng);
    r.tokens.push_back(tok);
  }
  const auto t1 = std::chrono::steady_clock::now();
  r.ms_per_token = std::chrono::duration<double, std::milli>(t1 - t0).count() / tokens;
  r.fallbacks = m.SampledFallbackRows() - fallbacks_before;
  return r;
}

void Report(const char* config_name, const r4dx::kernels::SampleParams& params, Model& m,
             const std::vector<int32_t>& prompt, int tokens) {
  constexpr uint64_t kSeed = 20260921u;
  // docs/perf.md's measurement rule: every figure twice, both reported when they differ by >3%.
  for (int pass = 0; pass < 2; ++pass) {
    const Result before = RunFullVocab(m, prompt, params, kSeed, tokens);
    const Result after = RunSummary(m, prompt, params, kSeed, tokens);
    const bool same = (before.tokens == after.tokens);
    std::printf(
        "%-28s pass %d: full-vocab %7.3f ms/token (%6.2f tok/s) | summary %7.3f ms/token "
        "(%6.2f tok/s) | speedup %5.2fx | fallbacks %lld/%d | same tokens: %s\n",
        config_name, pass + 1, before.ms_per_token, 1000.0 / before.ms_per_token,
        after.ms_per_token, 1000.0 / after.ms_per_token, before.ms_per_token / after.ms_per_token,
        static_cast<long long>(after.fallbacks), tokens, same ? "yes" : "NO");
    std::fflush(stdout);
  }
}

}  // namespace

int main(int argc, char** argv) {
  const std::string container =
      argc > 1 ? argv[1] : std::string("D:/models/r4dx/qwen38-27b-v3.r4dx");
  const std::string layout_name = argc > 2 ? argv[2] : std::string("w4a16");
  const int tokens = argc > 3 ? std::atoi(argv[3]) : 64;
  if (!r4dx_test::FileExists(container.c_str())) {
    std::fprintf(stderr, "[tool_sampled_bench] container not found: %s\n", container.c_str());
    return 77;
  }

  ModelOptions opts;
  opts.container_path = container;
  opts.layout = LayoutFromName(layout_name);
  opts.max_ctx = 2048;
  Model m = Model::Load(opts);
  const std::vector<int32_t> prompt = MakePromptTokens(64);

  std::printf("[tool_sampled_bench] container=%s layout=%s tokens=%d vocab=%lld\n",
              container.c_str(), LayoutName(opts.layout), tokens,
              static_cast<long long>(m.Config().vocab_size));

  {  // The stage's own configuration: a typical chat request.
    r4dx::kernels::SampleParams p;
    p.temperature = 0.7f;
    p.top_k = 20;
    p.top_p = 0.8f;
    Report("T0.7 top_k=20 top_p=0.8", p, m, prompt, tokens);
  }
  {  // Pure temperature: no candidate-set filter, so the summary's top-64 does not close the set on
     // its own and this is the configuration that pays the fallback.
    r4dx::kernels::SampleParams p;
    p.temperature = 1.0f;
    Report("T1.0 (pure temperature)", p, m, prompt, tokens);
  }
  {  // For reference: what a GREEDY token costs on the same model (DecodeStepGreedy's 4-byte D2H).
    r4dx::kernels::SampleParams p;
    p.temperature = 0.8f;
    p.min_p = 0.05f;
    Report("T0.8 min_p=0.05", p, m, prompt, tokens);
  }
  return 0;
}
