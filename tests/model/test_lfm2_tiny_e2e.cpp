// tests/model/test_lfm2_tiny_e2e.cpp -- docs/lfm2-plan.md L1-8 / L1-10: the whole LFM2 stack end to end on the TINY
// synthetic lfm2_moe checkpoint (tools/convert_ref/make_tiny_lfm2.py: hidden 512, 4 layers = conv+dense, conv+dense,
// attention+MoE, conv+MoE, 8 experts top-2, vocab 64), at TP=1 (Lfm2LocalTextModel) and TP=2 (Lfm2TpModel, --tp-mode
// emulate: both ranks on one device), against tools/reference/lfm2/tiny_e2e_golden.py's Lfm2Reference logits of every row.
//
// GPU test (one device; NOT RUN by its author). SKIP 77 while the fixture is missing. The fixture directory
// (R4DX_LFM2_TINY_DIR, default <models root>/r4dx/lfm2/tiny_e2e) holds tiny.r4dx + e2e_golden.safetensors/.json; it is
// rebuilt on the CPU with the three commands in tiny_e2e_golden.py's docstring.
//
// Per TP mode:
//   1. Feed: Prefill [0,1), Prefill [1,280) (crosses the 256-row chunk boundary inside one call), then DecodeStep for
//      positions 280..299. Rows 0, 279, 280..299 vs the fp32 truth: mean KL(truth||r4dx) <= 1.5 x mean KL(truth||HF-bf16)
//      over the same rows + 2e-3 (the plan-6.4 rule with a tiny-model epsilon), every row KL < 0.25, top-1 agreement
//      >= HF-bf16's - 2 rows.
//   2. Determinism: Reset and the same feed again -> identical bytes.
//   3. Checkpoint (the ShortConv state D2D copy): Prefill [0,256), SaveCheckpoint, Prefill [256,280) + one decode step,
//      RestoreCheckpoint, the same again -> identical bytes; PositionCount follows.
//   4. DecodeStepGreedy == argmax of DecodeStep's row in the same state; DecodeStepSampled returns an id in [0, V).
// Across modes: TP=2 rows vs TP=1 rows mean KL < 2e-3 (TP sums the row-parallel partials in another order).
// R4DX_LFM2_TINY_TP_REAL=1 additionally runs TP=2 --tp-mode real (needs HIP_VISIBLE_DEVICES=0,1 / unset).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "model.h"  // ModelOptions
#include "r4dx/models_root.h"
#include "r4dx_convert/safetensors_reader.hpp"
#include "test_common.h"
#include "text_model.h"

using namespace r4dx;
using namespace r4dx::model;

namespace {

int g_failures = 0;
void Check(bool ok, const std::string& what) {
  std::printf("%s: %s\n", ok ? "PASS" : "FAIL", what.c_str());
  if (!ok) ++g_failures;
}

std::string EnvOr(const char* name, const std::string& fallback) {
  const char* v = std::getenv(name);
  return v != nullptr && *v != '\0' ? std::string(v) : fallback;
}

std::vector<double> LogSoftmax(const float* x, int64_t V) {
  double mx = -1e300;
  for (int64_t i = 0; i < V; ++i) mx = std::max(mx, static_cast<double>(x[i]));
  double s = 0;
  for (int64_t i = 0; i < V; ++i) s += std::exp(static_cast<double>(x[i]) - mx);
  const double lse = mx + std::log(s);
  std::vector<double> out(static_cast<size_t>(V));
  for (int64_t i = 0; i < V; ++i) out[static_cast<size_t>(i)] = static_cast<double>(x[i]) - lse;
  return out;
}

double Kl(const float* p_logits, const float* q_logits, int64_t V) {  // KL(p || q)
  const std::vector<double> lp = LogSoftmax(p_logits, V), lq = LogSoftmax(q_logits, V);
  double kl = 0;
  for (int64_t i = 0; i < V; ++i) kl += std::exp(lp[static_cast<size_t>(i)]) * (lp[static_cast<size_t>(i)] - lq[static_cast<size_t>(i)]);
  return kl;
}

int64_t Argmax(const float* x, int64_t V) { return std::max_element(x, x + V) - x; }

struct Golden {
  std::vector<int32_t> ids;
  std::vector<float> truth, hf;  // [N, V]
  int64_t N = 0, V = 0;
  const float* Truth(int64_t row) const { return truth.data() + row * V; }
  const float* Hf(int64_t row) const { return hf.data() + row * V; }
};

constexpr int64_t kPrefillEnd = 280;  // rows [0, 280) prefilled, [280, N) decoded

// The feed of check 1: returns {row -> logits} for rows 0, 279, 280..N-1 (in that order).
std::vector<std::vector<float>> Feed(TextModel& m, const Golden& g, std::vector<int64_t>* rows) {
  std::vector<std::vector<float>> out;
  rows->clear();
  out.push_back(m.Prefill({g.ids[0]}));
  rows->push_back(0);
  out.push_back(m.Prefill(std::vector<int32_t>(g.ids.begin() + 1, g.ids.begin() + kPrefillEnd)));
  rows->push_back(kPrefillEnd - 1);
  for (int64_t p = kPrefillEnd; p < g.N; ++p) {
    out.push_back(m.DecodeStep(g.ids[static_cast<size_t>(p)]));
    rows->push_back(p);
  }
  return out;
}

std::vector<std::vector<float>> RunMode(TextModel& m, const Golden& g, const std::string& tag) {
  std::printf("---- %s ----\n", tag.c_str());
  const int64_t V = g.V;
  // 1. vs the truth
  m.Reset();
  std::vector<int64_t> rows;
  const std::vector<std::vector<float>> got = Feed(m, g, &rows);
  Check(m.PositionCount() == g.N, tag + ": PositionCount after the feed == " + std::to_string(g.N));
  double kl_r = 0, kl_h = 0, kl_max = 0;
  int top_r = 0, top_h = 0;
  bool shape_ok = true;
  for (size_t i = 0; i < rows.size(); ++i) {
    if (static_cast<int64_t>(got[i].size()) != V) {
      shape_ok = false;
      continue;
    }
    const int64_t r = rows[i];
    const double kr = Kl(g.Truth(r), got[i].data(), V), kh = Kl(g.Truth(r), g.Hf(r), V);
    kl_r += kr;
    kl_h += kh;
    kl_max = std::max(kl_max, kr);
    top_r += Argmax(got[i].data(), V) == Argmax(g.Truth(r), V) ? 1 : 0;
    top_h += Argmax(g.Hf(r), V) == Argmax(g.Truth(r), V) ? 1 : 0;
  }
  Check(shape_ok, tag + ": every returned logits row has vocab_size entries");
  const double n = static_cast<double>(rows.size());
  kl_r /= n;
  kl_h /= n;
  std::printf("  %s: %zu rows, mean KL(truth||r4dx) %.3e, mean KL(truth||HF-bf16) %.3e, worst r4dx row %.3e, top-1 %d vs HF %d\n",
              tag.c_str(), rows.size(), kl_r, kl_h, kl_max, top_r, top_h);
  Check(kl_r <= 1.5 * kl_h + 2e-3, tag + ": mean KL(truth||r4dx) <= 1.5 x KL(truth||HF-bf16) + 2e-3");
  Check(kl_max < 0.25, tag + ": every row KL(truth||r4dx) < 0.25");
  Check(top_r >= top_h - 2, tag + ": top-1 agreement with the truth >= HF-bf16's - 2 rows");

  // 2. determinism
  m.Reset();
  std::vector<int64_t> rows2;
  const std::vector<std::vector<float>> again = Feed(m, g, &rows2);
  Check(again == got, tag + ": the same feed after Reset() gives identical bytes");

  // 3. checkpoint / restore of the ShortConv state
  m.Reset();
  (void)m.Prefill(std::vector<int32_t>(g.ids.begin(), g.ids.begin() + 256));
  m.SaveCheckpoint();
  const std::vector<int32_t> tail(g.ids.begin() + 256, g.ids.begin() + kPrefillEnd);
  const std::vector<float> a1 = m.Prefill(tail);
  const std::vector<float> b1 = m.DecodeStep(g.ids[kPrefillEnd]);
  m.RestoreCheckpoint();
  Check(m.PositionCount() == 256, tag + ": RestoreCheckpoint rewinds PositionCount to 256");
  const std::vector<float> a2 = m.Prefill(tail);
  const std::vector<float> b2 = m.DecodeStep(g.ids[kPrefillEnd]);
  Check(a1 == a2 && b1 == b2, tag + ": RestoreCheckpoint + the same tail reproduces the logits bit for bit");
  // The 256 + 24 split vs the feed's 1 + 256 + 23 split: same math, other GEMM row slicing -> close, not equal.
  const double kl_split = Kl(got[1].data(), a1.data(), V);
  std::printf("  %s: row 279 via 256+24 vs 1+256+23 chunks: KL %.3e\n", tag.c_str(), kl_split);
  Check(kl_split < 1e-3, tag + ": a different prefill chunking gives the same row 279 (KL < 1e-3)");

  // 4. greedy / sampled
  m.Reset();
  (void)m.Prefill(std::vector<int32_t>(g.ids.begin(), g.ids.begin() + kPrefillEnd));
  const int32_t greedy = m.DecodeStepGreedy(g.ids[kPrefillEnd]);
  m.Reset();
  (void)m.Prefill(std::vector<int32_t>(g.ids.begin(), g.ids.begin() + kPrefillEnd));
  const std::vector<float> full = m.DecodeStep(g.ids[kPrefillEnd]);
  Check(greedy == static_cast<int32_t>(Argmax(full.data(), V)), tag + ": DecodeStepGreedy == argmax(DecodeStep) in the same state");
  kernels::SampleParams sp;
  sp.temperature = 1.0f;
  std::mt19937_64 rng(1234);
  const int32_t sampled = m.DecodeStepSampled(g.ids[kPrefillEnd + 1], sp, rng);
  Check(sampled >= 0 && sampled < V, tag + ": DecodeStepSampled returns an id in [0, vocab)");
  m.Reset();
  return got;
}

}  // namespace

int main() {
  return r4dx_test::RunGuardedMain("test_lfm2_tiny_e2e", []() -> int {
    const std::string dir = EnvOr("R4DX_LFM2_TINY_DIR", r4dx::ModelsPath("r4dx/lfm2/tiny_e2e"));
    const std::string container = dir + "/tiny.r4dx";
    const std::string golden = dir + "/e2e_golden.safetensors";
    if (!r4dx_test::FileExists(container)) return r4dx_test::SkipMissing(container + " (tiny_e2e_golden.py docstring)");
    if (!r4dx_test::FileExists(golden)) return r4dx_test::SkipMissing(golden);

    Golden g;
    {
      const r4dx_convert::SafetensorsReader r(r4dx_convert::Utf8ToWide(golden));
      const auto& im = r.Meta("token_ids");
      if (im.dtype != "I32") throw std::runtime_error("token_ids must be I32");
      g.N = im.ElemCount();
      g.ids.resize(static_cast<size_t>(g.N));
      std::memcpy(g.ids.data(), r.Data("token_ids"), g.ids.size() * sizeof(int32_t));
      g.truth = r4dx_test::ReadGoldenAsFloat(r, "logits_truth");
      g.hf = r4dx_test::ReadGoldenAsFloat(r, "logits_hf_bf16");
      g.V = static_cast<int64_t>(g.truth.size()) / g.N;
    }
    if (g.N <= kPrefillEnd + 2) throw std::runtime_error("the golden needs more than 282 tokens");

    ModelOptions o;
    o.container_path = container;
    o.prompt_checkpoint = true;

    std::vector<std::vector<float>> tp1;
    {
      TpOptions tp;  // world 1
      std::unique_ptr<TextModel> m = LoadTextModel(o, tp);
      Check(m->Config().arch == Arch::kLfm2Moe, "LoadTextModel dispatches the tiny lfm2_moe container to the LFM2 engine");
      Check(m->Config().vocab_size == g.V, "the model's vocab_size matches the golden's columns");
      Check(!m->HasVision() && !m->MtpEnabled() && !m->DflashEnabled(), "TP=1: no vision / MTP / DFlash");
      tp1 = RunMode(*m, g, "TP=1");
    }
    const auto cross = [&](const std::vector<std::vector<float>>& tp2, const std::string& tag) {
      double kl = 0;
      for (size_t i = 0; i < tp1.size(); ++i) kl += Kl(tp1[i].data(), tp2[i].data(), g.V);
      kl /= static_cast<double>(tp1.size());
      std::printf("  %s vs TP=1: mean KL %.3e\n", tag.c_str(), kl);
      Check(kl < 2e-3, tag + " rows agree with the TP=1 rows (mean KL < 2e-3)");
    };
    {
      TpOptions tp;
      tp.world = 2;
      tp.mode = TpOptions::Mode::kEmulate;
      std::unique_ptr<TextModel> m = LoadTextModel(o, tp);
      Check(m->TpWorld() == 2 && m->Config().arch == Arch::kLfm2Moe, "TP=2 emulate: an LFM2 tensor-parallel model");
      cross(RunMode(*m, g, "TP=2 emulate"), "TP=2 emulate");
    }
    if (EnvOr("R4DX_LFM2_TINY_TP_REAL", "0") == "1") {
      TpOptions tp;
      tp.world = 2;
      tp.mode = TpOptions::Mode::kReal;
      std::unique_ptr<TextModel> m = LoadTextModel(o, tp);
      cross(RunMode(*m, g, "TP=2 real"), "TP=2 real");
    }
    if (g_failures != 0) {
      std::printf("%d check(s) FAILED\n", g_failures);
      return 1;
    }
    std::printf("all checks passed\n");
    return 0;
  });
}
