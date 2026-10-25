// tests/model/test_llama_tiny_e2e.cpp -- docs/nanbeige-plan.md N1-9 / N1-11: the whole dense Llama stack end to end on the
// TINY synthetic llama checkpoint (tools/convert_ref/make_tiny_llama.py: hidden 1792, 28 q / 4 kv heads of 128, FFN 512,
// vocab 512, 3 layers, untied head, rope theta 1.2e8), at TP=1 (LlamaLocalTextModel) and TP=2 (LlamaTpModel, --tp-mode
// emulate: both ranks on one device; --tp-mode real with R4DX_LLAMA_TINY_TP_REAL=1), against
// tools/reference/llama/tiny_e2e_golden.py's stock-HF logits of every row.
//
// GPU test (one device; two with the real mode). SKIP 77 while the fixture is missing. The fixture directory
// (R4DX_LLAMA_TINY_DIR, default <models root>/r4dx/nanbeige/tiny_e2e) holds tiny.r4dx, tiny-w8a16.r4dx and
// e2e_golden.safetensors/.json; it is rebuilt on the CPU with the commands in tiny_e2e_golden.py's docstring.
//
// Per (container, TP mode):
//   1. Feed: Prefill [0,1), Prefill [1,280) (crosses the 256-row chunk boundary inside one call), then DecodeStep for
//      positions 280..299. Rows 0, 279, 280..299 vs the fp32 truth: mean KL(truth||r4dx) <= 1.5 x mean KL(truth||HF-bf16)
//      over the same rows + 2e-3 (the plan-6.3 rule with a tiny-model epsilon), every row KL < 1.0 (HF-bf16's own worst
//      row is 0.43 on this random model), top-1 agreement >= HF-bf16's - 2 rows.
//   2. Determinism: Reset and the same feed again -> identical bytes.
//   3. Checkpoint (positional): Prefill [0,256), SaveCheckpoint, Prefill [256,280) + one decode step, RestoreCheckpoint,
//      the same again -> identical bytes; PositionCount follows; a 256+24 chunking of row 279 is close to 1+256+23.
//   4. DecodeStepGreedy == argmax of DecodeStep's row in the same state; DecodeStepSampled returns an id in [0, V).
// Across modes: TP=2 rows vs TP=1 rows of the same container, mean KL <= 3 x noise. w8a16 vs bf16 (TP=1): mean KL <= noise.
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
#include "llama_model.h"
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

std::vector<int64_t> FeedRows(const Golden& g) {
  std::vector<int64_t> rows = {0, kPrefillEnd - 1};
  for (int64_t p = kPrefillEnd; p < g.N; ++p) rows.push_back(p);
  return rows;
}

// The feed of check 1: logits for rows 0, 279, 280..N-1 (in that order).
std::vector<std::vector<float>> Feed(TextModel& m, const Golden& g) {
  std::vector<std::vector<float>> out;
  out.push_back(m.Prefill({g.ids[0]}));
  out.push_back(m.Prefill(std::vector<int32_t>(g.ids.begin() + 1, g.ids.begin() + kPrefillEnd)));
  for (int64_t p = kPrefillEnd; p < g.N; ++p) out.push_back(m.DecodeStep(g.ids[static_cast<size_t>(p)]));
  return out;
}

double NoiseKl(const Golden& g) {
  double kl = 0;
  const std::vector<int64_t> rows = FeedRows(g);
  for (int64_t r : rows) kl += Kl(g.Truth(r), g.Hf(r), g.V);
  return kl / static_cast<double>(rows.size());
}

double MeanKl(const std::vector<std::vector<float>>& a, const std::vector<std::vector<float>>& b, int64_t V) {
  double kl = 0;
  for (size_t i = 0; i < a.size(); ++i) kl += Kl(a[i].data(), b[i].data(), V);
  return kl / static_cast<double>(a.size());
}

std::vector<std::vector<float>> RunMode(TextModel& m, const Golden& g, const std::string& tag) {
  std::printf("---- %s ----\n", tag.c_str());
  const int64_t V = g.V;
  // 1. vs the truth
  m.Reset();
  const std::vector<int64_t> rows = FeedRows(g);
  const std::vector<std::vector<float>> got = Feed(m, g);
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
  Check(kl_max < 1.0, tag + ": every row KL(truth||r4dx) < 1.0");
  Check(top_r >= top_h - 2, tag + ": top-1 agreement with the truth >= HF-bf16's - 2 rows");

  // 2. determinism
  m.Reset();
  const std::vector<std::vector<float>> again = Feed(m, g);
  Check(again == got, tag + ": the same feed after Reset() gives identical bytes");

  // 3. checkpoint / restore (positional)
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
  const double kl_split = Kl(got[1].data(), a1.data(), V);
  std::printf("  %s: row 279 via 256+24 vs 1+256+23 chunks: KL %.3e\n", tag.c_str(), kl_split);
  Check(kl_split < 1e-2, tag + ": a different prefill chunking gives nearly the same row 279 (KL < 1e-2)");

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
  return r4dx_test::RunGuardedMain("test_llama_tiny_e2e", []() -> int {
    const std::string dir = EnvOr("R4DX_LLAMA_TINY_DIR", r4dx::ModelsPath("r4dx/nanbeige/tiny_e2e"));
    const std::string golden = dir + "/e2e_golden.safetensors";
    if (!r4dx_test::FileExists(dir + "/tiny.r4dx")) return r4dx_test::SkipMissing(dir + "/tiny.r4dx (tiny_e2e_golden.py docstring)");
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
    const double noise = NoiseKl(g);

    std::vector<std::vector<float>> bf16_tp1;
    for (const char* name : {"tiny.r4dx", "tiny-w8a16.r4dx"}) {
      const std::string container = dir + "/" + name;
      if (!r4dx_test::FileExists(container)) {
        std::printf("SKIP: %s not present\n", container.c_str());
        continue;
      }
      const bool w8 = std::string(name) == "tiny-w8a16.r4dx";
      ModelOptions o;
      o.container_path = container;
      o.prompt_checkpoint = true;
      o.layout = w8 ? Layout::kW8a16 : Layout::kBf16;
      const std::string fmt = w8 ? "w8a16" : "bf16";

      std::vector<std::vector<float>> tp1;
      {
        TpOptions tp;  // world 1
        std::unique_ptr<TextModel> m = LoadTextModel(o, tp);
        Check(m->Config().arch == Arch::kLlama, fmt + ": LoadTextModel dispatches the tiny llama container to the Llama engine");
        Check(m->Config().vocab_size == g.V, fmt + ": the model's vocab_size matches the golden's columns");
        Check(!m->HasVision() && !m->MtpEnabled() && !m->DflashEnabled(), fmt + " TP=1: no vision / MTP / DFlash");
        tp1 = RunMode(*m, g, fmt + " TP=1");
      }
      if (w8 && !bf16_tp1.empty()) {
        const double kl = MeanKl(bf16_tp1, tp1, g.V);
        std::printf("  w8a16 vs bf16 (TP=1): mean KL %.3e (noise %.3e)\n", kl, noise);
        Check(kl <= noise, "w8a16 TP=1 rows agree with the bf16 TP=1 rows (mean KL <= noise)");
      }
      if (!w8) bf16_tp1 = tp1;
      const auto cross = [&](const std::vector<std::vector<float>>& tp2, const std::string& tag) {
        const double kl = MeanKl(tp1, tp2, g.V);
        std::printf("  %s vs TP=1: mean KL %.3e (3x noise %.3e)\n", tag.c_str(), kl, 3.0 * noise);
        Check(kl <= 3.0 * noise, tag + " rows agree with the TP=1 rows (mean KL <= 3 x noise)");
      };
      {
        TpOptions tp;
        tp.world = 2;
        tp.mode = TpOptions::Mode::kEmulate;
        std::unique_ptr<TextModel> m = LoadTextModel(o, tp);
        Check(m->TpWorld() == 2 && m->Config().arch == Arch::kLlama, fmt + " TP=2 emulate: a Llama tensor-parallel model");
        cross(RunMode(*m, g, fmt + " TP=2 emulate"), fmt + " TP=2 emulate");
      }
      if (EnvOr("R4DX_LLAMA_TINY_TP_REAL", "0") == "1") {
        TpOptions tp;
        tp.world = 2;
        tp.mode = TpOptions::Mode::kReal;
        std::unique_ptr<TextModel> m = LoadTextModel(o, tp);
        cross(RunMode(*m, g, fmt + " TP=2 real"), fmt + " TP=2 real");
      }
    }
    {
      // The engine directly: a layer's debug taps (shape and the residual identity h_out = mid + mlp).
      LlamaModelOptions lo;
      lo.container_path = dir + "/tiny.r4dx";
      lo.max_ctx = 512;
      LlamaModel lm = LlamaModel::Load(lo);
      const int64_t H = lm.Config().hidden_size, T = 8;
      std::mt19937 rng(77);
      std::normal_distribution<float> nd(0.0f, 1.0f);
      std::vector<uint16_t> x(static_cast<size_t>(T * H));
      for (auto& v : x) v = core::FloatToBf16(nd(rng));
      LlamaLayerDebug d;
      const std::vector<uint16_t> y = lm.DebugLayerForward(1, x, T, 0, &d);
      bool sizes = y.size() == x.size() && d.input_norm_out.size() == x.size() && d.attn_out.size() == x.size() &&
                   d.mid_residual.size() == x.size() && d.post_norm_out.size() == x.size() && d.mlp_out.size() == x.size();
      Check(sizes, "DebugLayerForward: every tap is [T, hidden]");
      double worst = 0;
      for (size_t i = 0; sizes && i < y.size(); ++i) {
        const double want = static_cast<double>(core::Bf16ToFloat(d.mid_residual[i])) + core::Bf16ToFloat(d.mlp_out[i]);
        worst = std::max(worst, std::fabs(want - core::Bf16ToFloat(y[i])) / std::max(1.0, std::fabs(want)));
      }
      Check(sizes && worst < 1e-2, "DebugLayerForward: out == mid_residual + mlp_out (bf16 rounding, rel " + std::to_string(worst) + ")");
    }
    if (g_failures != 0) {
      std::printf("%d check(s) FAILED\n", g_failures);
      return 1;
    }
    std::printf("all checks passed\n");
    return 0;
  });
}
