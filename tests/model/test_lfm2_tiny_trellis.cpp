// tests/model/test_lfm2_tiny_trellis.cpp -- docs/lfm2-plan.md L3-8: the LFM2 engine on a TRELLIS container at TP=1, end
// to end, on the TINY synthetic lfm2_moe checkpoint (the tiny_e2e one: hidden 512, 4 layers = conv+dense, conv+dense,
// attention+MoE, conv+MoE, 8 experts top-2, vocab 64), quantized at KB=4 by the CPU oracle and converted with
// r4dx-convert --trellis-from: every attn q/k/v/o, ShortConv in_proj/out_proj, dense gate_up (2 parts) / down and EVERY
// expert's gate_up (2 parts) / down is trellis; lm_head / embedding / router / expert_bias / conv taps / norms bf16.
//
// GPU test (one device; NOT RUN by its author). SKIP 77 while a fixture is missing:
//   R4DX_LFM2_TINY_TRELLIS_DIR (default <models root>/r4dx/lfm2/tiny_trellis): tiny_trellis.r4dx + trellis_golden.safetensors
//   R4DX_LFM2_TINY_DIR         (default <models root>/r4dx/lfm2/tiny_e2e):     tiny.r4dx + e2e_golden.safetensors
// Both are rebuilt on the CPU with the commands in tools/reference/lfm2/tiny_trellis_golden.py's docstring (and
// tiny_e2e_golden.py's for the bf16 one).
//
// The gate's truth is `logits_truth_q`: the fp32 reference forward on the checkpoint whose linears are the oracle's
// RECONSTRUCTIONS (exactly what the container decodes to), so the quantization error is not inside the tolerance:
//   1. Feed (as test_lfm2_tiny_e2e): Prefill [0,1), Prefill [1,280) (crosses the 256-row chunk), DecodeStep 280..299.
//      mean KL(truth_q||r4dx) <= 1.5 x mean KL(truth_q||HF-bf16_q) + 2e-3, every row < 0.25, top-1 >= HF-bf16_q's - 2.
//   2. The quantization effect, against the UNQUANTIZED fp32 truth of e2e_golden: mean KL(truth||r4dx trellis) <= 1.5 x
//      the CPU's mean KL(truth||truth_q) + 5e-3 (a loose sanity bound: the trellis body is the model the oracle made);
//      and (informative) the trellis rows vs the bf16 container's rows from the same feed.
//   3. Determinism: Reset and the same feed -> identical bytes (the per-linear ticket protocol of the split-group GEMMs).
//   4. Checkpoint / restore of the ShortConv state reproduces the logits bit for bit.
//   5. The MoE paths agree on trellis experts: Lfm2Model::DebugFfnForward on layer 2 (attention + MoE) and layer 3,
//      T = 8 rows, kPerSlot (M = 1 GEMMs per pair) vs kSorted (M = the expert's row count <= 16): identical bytes (a chunk
//      of M <= 16 rows takes the M = 1 tuning whole -- linear.h PickTuning -- and the trellis input transform is per row),
//      and both give the same router ids.
//   6. DecodeStepGreedy == argmax(DecodeStep) in the same state; DecodeStepSampled returns an id in [0, V).
//   7. Refusals: --tp 2 on the trellis container (TP=1 only, before any device work); --layout bf16 on it.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "lfm2_model.h"
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
  std::vector<float> truth, truth_q, hf_q;  // [N, V]
  int64_t N = 0, V = 0;
  const float* Truth(int64_t r) const { return truth.data() + r * V; }
  const float* TruthQ(int64_t r) const { return truth_q.data() + r * V; }
  const float* HfQ(int64_t r) const { return hf_q.data() + r * V; }
};

std::vector<int32_t> ReadIds(const r4dx_convert::SafetensorsReader& r) {
  const auto& im = r.Meta("token_ids");
  if (im.dtype != "I32") throw std::runtime_error("token_ids must be I32");
  std::vector<int32_t> ids(static_cast<size_t>(im.ElemCount()));
  std::memcpy(ids.data(), r.Data("token_ids"), ids.size() * sizeof(int32_t));
  return ids;
}

constexpr int64_t kPrefillEnd = 280;

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

ModelOptions Opts(const std::string& container, Layout layout) {
  ModelOptions o;
  o.container_path = container;
  o.layout = layout;
  o.prompt_checkpoint = true;
  return o;
}

template <class Fn>
bool ThrowsWith(Fn&& fn, const std::string& needle, std::string* msg) {
  try {
    fn();
  } catch (const std::exception& e) {
    *msg = e.what();
    return msg->find(needle) != std::string::npos;
  }
  *msg = "(no exception)";
  return false;
}

}  // namespace

int main() {
  return r4dx_test::RunGuardedMain("test_lfm2_tiny_trellis", []() -> int {
    const std::string tdir = EnvOr("R4DX_LFM2_TINY_TRELLIS_DIR", r4dx::ModelsPath("r4dx/lfm2/tiny_trellis"));
    const std::string bdir = EnvOr("R4DX_LFM2_TINY_DIR", r4dx::ModelsPath("r4dx/lfm2/tiny_e2e"));
    const std::string tcont = tdir + "/tiny_trellis.r4dx", tgold = tdir + "/trellis_golden.safetensors";
    const std::string bcont = bdir + "/tiny.r4dx", bgold = bdir + "/e2e_golden.safetensors";
    for (const std::string& f : {tcont, tgold, bcont, bgold}) {
      if (!r4dx_test::FileExists(f)) return r4dx_test::SkipMissing(f + " (tiny_trellis_golden.py docstring)");
    }

    Golden g;
    {
      const r4dx_convert::SafetensorsReader rt(r4dx_convert::Utf8ToWide(tgold));
      const r4dx_convert::SafetensorsReader rb(r4dx_convert::Utf8ToWide(bgold));
      g.ids = ReadIds(rt);
      Check(g.ids == ReadIds(rb), "the trellis golden's token ids are the tiny_e2e golden's");
      g.N = static_cast<int64_t>(g.ids.size());
      g.truth_q = r4dx_test::ReadGoldenAsFloat(rt, "logits_truth_q");
      g.hf_q = r4dx_test::ReadGoldenAsFloat(rt, "logits_hf_bf16_q");
      g.truth = r4dx_test::ReadGoldenAsFloat(rb, "logits_truth");
      g.V = static_cast<int64_t>(g.truth_q.size()) / g.N;
      if (static_cast<int64_t>(g.truth.size()) != g.N * g.V) throw std::runtime_error("golden shapes disagree");
    }
    if (g.N <= kPrefillEnd + 2) throw std::runtime_error("the golden needs more than 282 tokens");
    const int64_t V = g.V;

    // CPU: the container is what this test means.
    const Lfm2ContainerInfo info = Lfm2Container::Inspect(tcont);
    Check(info.has_trellis, "Inspect: the tiny container carries __metadata__.quant.trellis");

    // ---- 1-4, 6: the whole stack through LoadTextModel --layout trellis, TP=1 ----
    std::vector<std::vector<float>> got;
    std::vector<int64_t> rows;
    {
      TpOptions tp;  // world 1
      std::unique_ptr<TextModel> m = LoadTextModel(Opts(tcont, Layout::kTrellis), tp);
      Check(m->Config().arch == Arch::kLfm2Moe && m->Config().vocab_size == V,
            "LoadTextModel --layout trellis: the LFM2 engine, vocab matches the golden");
      m->Reset();
      got = Feed(*m, g, &rows);
      Check(m->PositionCount() == g.N, "PositionCount after the feed == " + std::to_string(g.N));
      double kl_r = 0, kl_h = 0, kl_max = 0, kl_q = 0, kl_tq = 0;
      int top_r = 0, top_h = 0;
      bool shape_ok = true;
      for (size_t i = 0; i < rows.size(); ++i) {
        if (static_cast<int64_t>(got[i].size()) != V) {
          shape_ok = false;
          continue;
        }
        const int64_t r = rows[i];
        const double kr = Kl(g.TruthQ(r), got[i].data(), V);
        kl_r += kr;
        kl_h += Kl(g.TruthQ(r), g.HfQ(r), V);
        kl_max = std::max(kl_max, kr);
        kl_q += Kl(g.Truth(r), got[i].data(), V);
        kl_tq += Kl(g.Truth(r), g.TruthQ(r), V);
        top_r += Argmax(got[i].data(), V) == Argmax(g.TruthQ(r), V) ? 1 : 0;
        top_h += Argmax(g.HfQ(r), V) == Argmax(g.TruthQ(r), V) ? 1 : 0;
      }
      Check(shape_ok, "every returned logits row has vocab_size entries");
      const double n = static_cast<double>(rows.size());
      kl_r /= n;
      kl_h /= n;
      kl_q /= n;
      kl_tq /= n;
      std::printf("  %zu rows: mean KL(truth_q||r4dx) %.3e, mean KL(truth_q||HF-bf16_q) %.3e, worst row %.3e, top-1 %d vs HF %d\n",
                  rows.size(), kl_r, kl_h, kl_max, top_r, top_h);
      std::printf("  quantization: mean KL(truth||r4dx trellis) %.3e vs the CPU's KL(truth||truth_q) %.3e\n", kl_q, kl_tq);
      Check(kl_r <= 1.5 * kl_h + 2e-3, "mean KL(truth_q||r4dx) <= 1.5 x KL(truth_q||HF-bf16_q) + 2e-3");
      Check(kl_max < 0.25, "every row KL(truth_q||r4dx) < 0.25");
      Check(top_r >= top_h - 2, "top-1 agreement with truth_q >= HF-bf16_q's - 2 rows");
      Check(kl_q <= 1.5 * kl_tq + 5e-3, "vs the unquantized truth: mean KL <= 1.5 x KL(truth||truth_q) + 5e-3");

      // 3. determinism
      m->Reset();
      std::vector<int64_t> rows2;
      Check(Feed(*m, g, &rows2) == got, "the same feed after Reset() gives identical bytes");

      // 4. checkpoint / restore
      m->Reset();
      (void)m->Prefill(std::vector<int32_t>(g.ids.begin(), g.ids.begin() + 256));
      m->SaveCheckpoint();
      const std::vector<int32_t> tail(g.ids.begin() + 256, g.ids.begin() + kPrefillEnd);
      const std::vector<float> a1 = m->Prefill(tail);
      const std::vector<float> b1 = m->DecodeStep(g.ids[kPrefillEnd]);
      m->RestoreCheckpoint();
      const std::vector<float> a2 = m->Prefill(tail);
      const std::vector<float> b2 = m->DecodeStep(g.ids[kPrefillEnd]);
      Check(a1 == a2 && b1 == b2, "RestoreCheckpoint + the same tail reproduces the logits bit for bit");
      const double kl_split = Kl(got[1].data(), a1.data(), V);
      std::printf("  row 279 via 256+24 vs 1+256+23 chunks: KL %.3e\n", kl_split);
      Check(kl_split < 1e-3, "a different prefill chunking gives the same row 279 (KL < 1e-3)");

      // 6. greedy / sampled
      m->Reset();
      (void)m->Prefill(std::vector<int32_t>(g.ids.begin(), g.ids.begin() + kPrefillEnd));
      const int32_t greedy = m->DecodeStepGreedy(g.ids[kPrefillEnd]);
      m->Reset();
      (void)m->Prefill(std::vector<int32_t>(g.ids.begin(), g.ids.begin() + kPrefillEnd));
      const std::vector<float> full = m->DecodeStep(g.ids[kPrefillEnd]);
      Check(greedy == static_cast<int32_t>(Argmax(full.data(), V)), "DecodeStepGreedy == argmax(DecodeStep) in the same state");
      kernels::SampleParams sp;
      sp.temperature = 1.0f;
      std::mt19937_64 rng(1234);
      const int32_t sampled = m->DecodeStepSampled(g.ids[kPrefillEnd + 1], sp, rng);
      Check(sampled >= 0 && sampled < V, "DecodeStepSampled returns an id in [0, vocab)");
    }

    // 2 (informative): the bf16 container's rows from the same feed.
    {
      TpOptions tp;
      std::unique_ptr<TextModel> mb = LoadTextModel(Opts(bcont, Layout::kBf16), tp);
      std::vector<int64_t> rows_b;
      const std::vector<std::vector<float>> bf = Feed(*mb, g, &rows_b);
      double kl = 0;
      int agree = 0;
      for (size_t i = 0; i < bf.size() && i < got.size(); ++i) {
        kl += Kl(bf[i].data(), got[i].data(), V);
        agree += Argmax(bf[i].data(), V) == Argmax(got[i].data(), V) ? 1 : 0;
      }
      std::printf("  (informative) r4dx bf16 vs r4dx trellis rows: mean KL %.3e, top-1 agreement %d / %zu\n",
                  kl / static_cast<double>(bf.size()), agree, bf.size());
    }

    // ---- 5: the MoE paths on trellis experts (Lfm2Model directly) ----
    {
      Lfm2ModelOptions lo;
      lo.container_path = tcont;
      lo.layout = Layout::kTrellis;
      lo.max_ctx = 512;
      Lfm2Model lm = Lfm2Model::Load(lo);
      const int64_t H = lm.Config().hidden_size, T = 8;
      std::mt19937 rng(77);
      std::normal_distribution<float> nd(0.0f, 1.0f);
      std::vector<uint16_t> x(static_cast<size_t>(T * H));
      for (auto& v : x) v = core::FloatToBf16(nd(rng));
      for (int64_t layer : {int64_t{2}, int64_t{3}}) {
        if (!lm.Config().IsMoeLayer(layer)) {
          Check(false, "layer " + std::to_string(layer) + " is an MoE layer of the tiny model");
          continue;
        }
        Lfm2MoeDebug d1, d2;
        const std::vector<uint16_t> y1 = lm.DebugFfnForward(layer, x, T, Lfm2MoePath::kPerSlot, &d1);
        const std::vector<uint16_t> y2 = lm.DebugFfnForward(layer, x, T, Lfm2MoePath::kSorted, &d2);
        Check(d1.ids == d2.ids, "layer " + std::to_string(layer) + ": both MoE paths route the same experts");
        size_t diff = 0;
        for (size_t i = 0; i < y1.size(); ++i) diff += y1[i] != y2[i] ? 1 : 0;
        Check(diff == 0, "layer " + std::to_string(layer) + ": trellis experts, per-slot (M=1) == sorted (M<=16) bytes (" +
                             std::to_string(diff) + " of " + std::to_string(y1.size()) + " differ)");
        bool finite = true;
        for (uint16_t v : y1) finite = finite && std::isfinite(core::Bf16ToFloat(v));
        Check(finite, "layer " + std::to_string(layer) + ": the trellis MoE output is finite");
      }
    }

    // ---- 7: refusals ----
    {
      std::string msg;
      TpOptions tp2;
      tp2.world = 2;
      tp2.mode = TpOptions::Mode::kEmulate;
      Check(ThrowsWith([&] { (void)LoadTextModel(Opts(tcont, Layout::kTrellis), tp2); }, "TP=1 only", &msg),
            "--tp 2 on the trellis container is refused (TP=1 only): " + msg);
      TpOptions tp1;
      Check(ThrowsWith([&] { (void)LoadTextModel(Opts(tcont, Layout::kBf16), tp1); }, "--layout trellis", &msg),
            "--layout bf16 on the trellis container is refused: " + msg);
    }

    if (g_failures != 0) {
      std::printf("%d check(s) FAILED\n", g_failures);
      return 1;
    }
    std::printf("all checks passed\n");
    return 0;
  });
}
