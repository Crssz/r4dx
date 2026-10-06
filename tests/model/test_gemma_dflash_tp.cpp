// tests/model/test_gemma_dflash_tp.cpp -- Gemma 4 DFlash under tensor parallelism (docs/gemma4-plan.md, docs/tp.md Gemma
// section). TWO-GPU test, opt-in exactly like test_tp_real_vs_emulation: SKIP (77) unless R4DX_TP2GPU=1, two visible HIP
// devices and both containers. WRITTEN, NOT RUN by its author (no GPU work was allowed); run it with
// `tests\run_tests.ps1 -TwoGpu` (LABEL tp2gpu) only when both GPUs are free.
//
// Four passes over the same prompt (the first kPrompt tokens of the first tools/reference/kl_corpus/tokens_gemma.json
// segment), each generating kGen greedy tokens through a TextModel from LoadTextModel:
//   A  TP=2 real, plain DecodeStepGreedy                      -> the TP=2 ground truth
//   B  TP=2 real, DecodeStepDflashGreedy rounds (k = 7)       -> must be BYTE-IDENTICAL to A (hard gate; the verify window's
//                                                                row GEMMs differ in shape from the single-row decode, so a
//                                                                mismatch is first read as a near-tie flip: tools/gemma_dflash_ab.ps1's
//                                                                rule -- a coherent continuation is a tie, dropped/duplicated
//                                                                tokens are bookkeeping)
//   C  TP=1, DecodeStepDflashGreedy rounds                    -> reported against B (divergence position) and the acceptance
//                                                                gate: tokens/round of B within 5% of C's (the drafter, the
//                                                                features and the merges are replicated / exact, so only the
//                                                                numerics of the vocab-split GEMM may move it)
//   D  TP=2 real, lifecycle: a sampled request takes one plain step (walk_len 0); with injection off DecodeStepDflashGreedy
//      throws and plain decode still works; Reset() + injection on drafts again and still equals A's prefix.
// Env: R4DX_GEMMA_DFLASH_TARGET (default <R4DX_MODELS_ROOT>\r4dx\huihui-gemma\bf16.r4dx), R4DX_GEMMA_DFLASH_DRAFTER (default
// <R4DX_MODELS_ROOT>\r4dx\gemma4-12b-dflash-bf16.r4dx), R4DX_GEMMA_DFLASH_LAYOUT (default bf16; trellis for a trellis container).
#include <hip/hip_runtime.h>
#include "r4dx/models_root.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "model.h"
#include "nlohmann/json.hpp"
#include "test_common.h"
#include "text_model.h"

using namespace r4dx::model;
using nlohmann::json;

#ifndef R4DX_SOURCE_DIR_STR
#define R4DX_SOURCE_DIR_STR "."
#endif

namespace {

int g_failures = 0;
void Check(bool ok, const std::string& what) {
  std::printf("%s: %s\n", ok ? "PASS" : "FAIL", what.c_str());
  if (!ok) ++g_failures;
}

std::string Env(const char* name, const char* def) {
  const char* v = std::getenv(name);
  return (v != nullptr && *v != '\0') ? std::string(v) : std::string(def);
}

constexpr int64_t kPrompt = 256, kGen = 192, kK = 7;

int32_t ArgmaxOf(const std::vector<float>& v) {
  return static_cast<int32_t>(std::max_element(v.begin(), v.end()) - v.begin());  // lowest index on ties
}

struct Run {
  std::vector<int32_t> tokens;  // kGen generated tokens (the first comes from the prefill logits)
  int64_t rounds = 0;
  double TokensPerRound() const { return rounds > 0 ? static_cast<double>(tokens.size() - 1) / static_cast<double>(rounds) : 0.0; }
};

Run Generate(TextModel& m, const std::vector<int32_t>& prompt, bool dflash) {
  Run r;
  m.Reset();
  r.tokens.push_back(ArgmaxOf(m.Prefill(prompt)));
  while (static_cast<int64_t>(r.tokens.size()) < kGen) {
    const int32_t last = r.tokens.back();
    if (dflash) {
      int64_t walk = 0;
      const std::vector<int32_t> got = m.DecodeStepDflashGreedy(last, kK, /*p_min=*/0.0f, /*n_min=*/0, &walk);
      ++r.rounds;
      for (int32_t t : got) {
        if (static_cast<int64_t>(r.tokens.size()) < kGen) r.tokens.push_back(t);
      }
    } else {
      r.tokens.push_back(m.DecodeStepGreedy(last));
    }
  }
  return r;
}

int64_t FirstDiff(const std::vector<int32_t>& a, const std::vector<int32_t>& b) {
  const size_t n = std::min(a.size(), b.size());
  for (size_t i = 0; i < n; ++i) {
    if (a[i] != b[i]) return static_cast<int64_t>(i);
  }
  return a.size() == b.size() ? -1 : static_cast<int64_t>(n);
}

ModelOptions Options(const std::string& target, const std::string& drafter, const std::string& layout, bool with_dflash) {
  ModelOptions o;
  o.container_path = target;
  o.layout = LayoutFromName(layout);
  o.max_ctx = 1024;
  o.vision = ModelOptions::VisionMode::kOff;
  if (with_dflash) {
    o.dflash_container = drafter;
    o.dflash_draft_k = kK;
  }
  return o;
}

TpOptions Tp2() {
  TpOptions t;
  t.world = 2;
  t.mode = TpOptions::Mode::kReal;
  return t;
}

std::vector<int32_t> LoadPrompt() {
  const std::string path = std::string(R4DX_SOURCE_DIR_STR) + "/tools/reference/kl_corpus/tokens_gemma.json";
  std::ifstream f(path);
  if (!f) throw std::runtime_error("missing " + path);
  const json j = json::parse(f);
  std::vector<int32_t> ids = j.at("segments").at(0).at("token_ids").get<std::vector<int32_t>>();
  if (static_cast<int64_t>(ids.size()) > kPrompt) ids.resize(static_cast<size_t>(kPrompt));
  return ids;
}

}  // namespace

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  if (Env("R4DX_TP2GPU", "") != "1") {
    std::printf("SKIP: two-GPU test; run tests\\run_tests.ps1 -TwoGpu\n");
    return 77;
  }
  const std::string target = Env("R4DX_GEMMA_DFLASH_TARGET", r4dx::ModelsPath("r4dx/huihui-gemma/bf16.r4dx").c_str());
  const std::string drafter = Env("R4DX_GEMMA_DFLASH_DRAFTER", r4dx::ModelsPath("r4dx/gemma4-12b-dflash-bf16.r4dx").c_str());
  const std::string layout = Env("R4DX_GEMMA_DFLASH_LAYOUT", "bf16");
  if (!r4dx_test::FileExists(target)) return r4dx_test::SkipMissing(target);
  if (!r4dx_test::FileExists(drafter)) return r4dx_test::SkipMissing(drafter);
  int visible = 0;
  if (hipGetDeviceCount(&visible) != hipSuccess || visible < 2) {
    std::printf("SKIP: needs two visible HIP devices (have %d)\n", visible);
    return 77;
  }
  try {
    const std::vector<int32_t> prompt = LoadPrompt();
    Run a, b, c;
    {  // A and B and D share one TP=2 group (the DFlash-capable one: plain decode on it is the unchanged TP=2 path).
      std::unique_ptr<TextModel> m = LoadTextModel(Options(target, drafter, layout, true), Tp2());
      Check(m->DflashEnabled() && m->TpWorld() == 2, "TP=2 group reports DflashEnabled()");
      a = Generate(*m, prompt, /*dflash=*/false);
      b = Generate(*m, prompt, /*dflash=*/true);
      const int64_t d = FirstDiff(a.tokens, b.tokens);
      Check(d < 0, "TP=2 DFlash greedy output byte-identical to TP=2 plain greedy (first diff at " + std::to_string(d) + ")");
      std::printf("[tp2] dflash: %.3f tokens/round over %lld rounds\n", b.TokensPerRound(), static_cast<long long>(b.rounds));

      // D: lifecycle.
      m->Reset();
      (void)m->Prefill(prompt);
      r4dx::kernels::SampleParams sp;
      sp.temperature = 0.8f;
      std::mt19937_64 rng(1234);
      int64_t walk = -1;
      const std::vector<int32_t> s = m->DecodeStepDflashSampled(a.tokens[0], kK, 0.0f, 0, sp, rng, &walk);
      Check(s.size() == 1 && walk == 0, "sampled request on TP=2 takes one plain step (walk_len 0)");
      m->Reset();
      m->SetDflashInjectionEnabled(false);
      (void)m->Prefill(prompt);
      (void)m->DecodeStepGreedy(a.tokens[0]);
      bool threw = false;
      try {
        (void)m->DecodeStepDflashGreedy(a.tokens[1], kK, 0.0f, 0);
      } catch (const std::exception&) {
        threw = true;
      }
      Check(threw, "injection off: DecodeStepDflashGreedy refuses (the drafter frontier would lag)");
      const TpDiagnostics* diag = dynamic_cast<const TpDiagnostics*>(m.get());
      Check(diag != nullptr && diag->GroupHealth() == TpDiagnostics::Health::kNeedsRecovery,
            "the refusal is a failed collective command: group kNeedsRecovery (Reset() below recovers it)");
      m->SetDflashInjectionEnabled(true);
      const Run again = Generate(*m, prompt, /*dflash=*/true);  // Generate() Reset()s: the policy is re-applied on every rank
      Check(FirstDiff(a.tokens, again.tokens) < 0, "Reset() + injection on: DFlash rounds work again and equal TP=2 plain greedy");
    }
    {  // C: TP=1 DFlash (one GPU; device = the last visible ordinal, the compute card)
      (void)hipSetDevice(visible - 1);
      TpOptions t1;
      t1.world = 1;
      std::unique_ptr<TextModel> m = LoadTextModel(Options(target, drafter, layout, true), t1);
      c = Generate(*m, prompt, /*dflash=*/true);
    }
    const int64_t d_bc = FirstDiff(b.tokens, c.tokens);
    std::printf("[report] TP=2 DFlash vs TP=1 DFlash: %s (first diff at %lld of %lld)\n", d_bc < 0 ? "identical" : "DIFFERENT",
                static_cast<long long>(d_bc), static_cast<long long>(kGen));
    const double tb = b.TokensPerRound(), tc = c.TokensPerRound();
    std::printf("[report] tokens/round: TP=2 %.3f, TP=1 %.3f\n", tb, tc);
    Check(tc > 0.0 && std::abs(tb - tc) <= 0.05 * tc, "TP=2 acceptance (tokens/round) within 5% of TP=1");
  } catch (const std::exception& e) {
    std::fprintf(stderr, "test_gemma_dflash_tp: exception: %s\n", e.what());
    return 1;
  }
  if (g_failures > 0) {
    std::fprintf(stderr, "%d check(s) FAILED\n", g_failures);
    return 1;
  }
  std::printf("ALL PASS\n");
  return 0;
}
