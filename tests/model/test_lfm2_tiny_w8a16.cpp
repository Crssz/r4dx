// tests/model/test_lfm2_tiny_w8a16.cpp -- docs/w8a16.md 7.3 `lfm2_tiny_w8a16`: the LFM2 engine on a W8A16 container at TP=1,
// end to end, on the TINY synthetic lfm2_moe checkpoint (the tiny_e2e one: hidden 512, 4 layers = conv+dense, conv+dense,
// attention+MoE, conv+MoE, 8 experts top-2, vocab 64) converted by lane W1 with
//   r4dx-convert --layouts w8a16 --no-bf16 --lm-head w8a16     (every body linear, every expert and the head w8a16, g128)
//
// GPU test (one device; NOT RUN by its author). SKIP 77 while a fixture is missing:
//   R4DX_LFM2_TINY_W8A16_DIR (default <models root>/r4dx/lfm2/tiny_w8a16): tiny_w8a16.r4dx
//   R4DX_LFM2_TINY_DIR       (default <models root>/r4dx/lfm2/tiny_e2e):   tiny.r4dx + e2e_golden.safetensors
// Optional: <R4DX_LFM2_TINY_W8A16_DIR>/w8a16_golden.safetensors (R4DX_LFM2_TINY_W8A16_GOLDEN overrides the path), lane W4's
// tools/reference/lfm2/tiny_w8a16_golden.py output with `logits_truth_q` (the fp32 reference forward on the DEQUANTIZED
// weights) and `logits_hf_bf16_q`; when present, check 1b gates against it like test_lfm2_tiny_trellis does.
//
//   1. Feed (as test_lfm2_tiny_e2e): Prefill [0,1), Prefill [1,280) (crosses the 256-row chunk), DecodeStep 280..299, against
//      the UNQUANTIZED fp32 truth of e2e_golden with a tolerance for 8-bit weights: mean KL(truth||r4dx w8a16) <= 1.5 x mean
//      KL(truth||HF-bf16) + 5e-3 (the bf16 gate's epsilon 2e-3 widened by 3e-3: RTN int8 g128's relative weight error is
//      about 0.007, under 2x bf16's 0.004), every row < 0.25, top-1 >= HF-bf16's - 3.
//      1b (only with the w8a16 golden): mean KL(truth_q||r4dx) <= 1.5 x KL(truth_q||HF-bf16_q) + 2e-3, every row < 0.25,
//      top-1 >= HF-bf16_q's - 2 (the quantization error is then outside the tolerance, as for trellis).
//      (informative) the r4dx bf16 container's rows from the same feed vs the w8a16 rows.
//   2. Determinism: Reset and the same feed -> identical bytes.
//   3. Checkpoint / restore of the ShortConv state reproduces the logits bit for bit; a different prefill chunking gives
//      the same row 279 (KL < 1e-3).
//   4. The MoE paths agree on w8a16 experts: Lfm2Model::DebugFfnForward on layers 2 and 3, T = 8 rows, kPerSlot (M = 1 GEMMs
//      per pair) vs kSorted (M = the expert's row count <= 16): identical bytes (a chunk of M <= 16 rows takes the M = 1
//      tuning, and a w8a16 row's bits depend on SK alone), and both route the same experts.
//   5. DecodeStepGreedy == argmax(DecodeStep) in the same state. NO sampled path: the tiny vocab (64) is under the top-k
//      kernel's minimum (a known limitation of the sampler on vocab <= 64, independent of w8a16), so DecodeStepSampled is
//      not exercised here; the real-model smoke covers it.
//   6. The loaded container: every body linear, expert and the head is kW8a16 at group 128 (0 fallbacks).
//   7. Refusals (docs/w8a16.md 6.3): (1) --layout w8a16 on the bf16 tiny.r4dx (no quant.w8a16) -- before any upload;
//      (2) a doctored copy whose quant.w8a16.scale_log2_bias is 13, and one whose version is 2; (3) a doctored copy that
//      declares group 64 while every .ws holds g128 scales (each .ws is then the wrong size), naming the tensor;
//      the v1 TP rule (--tp 2 --layout w8a16, refused before any device work); (5) --layout bf16 on this --no-bf16
//      container: "no tensor found ... in any known on-disk form".
//   8. MIXED container (docs/w8a16.md 5.2 fallback A: non-expert linears bf16, experts w8a16), only when this build has
//      r4dx-convert (R4DX_CONVERT_EXE) and the tiny checkpoint (R4DX_LFM2_TINY_DIR/ckpt) exists: converted into a temp
//      directory with --keep-bf16 '^(?!.*\.moe\.experts\.)', loaded with --layout w8a16 -- the experts are kW8a16, every
//      non-expert linear and the head kBf16 (the per-linear fallback) -- and its feed passes check 1's gate.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "lfm2_model.h"
#include "model.h"  // ModelOptions
#include "nlohmann/json.hpp"
#include "r4dx/models_root.h"
#include "r4dx_convert/safetensors_reader.hpp"
#include "test_common.h"
#include "text_model.h"

using namespace r4dx;
using namespace r4dx::model;
namespace fs = std::filesystem;
using nlohmann::json;

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
  std::vector<float> truth, hf;          // e2e_golden [N, V]
  std::vector<float> truth_q, hf_q;      // the optional w8a16 golden [N, V]
  int64_t N = 0, V = 0;
  const float* Truth(int64_t r) const { return truth.data() + r * V; }
  const float* Hf(int64_t r) const { return hf.data() + r * V; }
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

// Check 1's gate against the unquantized truth. Returns the mean KL.
double GateVsTruth(const Golden& g, const std::vector<std::vector<float>>& got, const std::vector<int64_t>& rows,
                   const std::string& tag) {
  const int64_t V = g.V;
  double kl_r = 0, kl_h = 0, kl_max = 0;
  int top_r = 0, top_h = 0;
  bool shape_ok = true;
  for (size_t i = 0; i < rows.size(); ++i) {
    if (static_cast<int64_t>(got[i].size()) != V) {
      shape_ok = false;
      continue;
    }
    const int64_t r = rows[i];
    const double kr = Kl(g.Truth(r), got[i].data(), V);
    kl_r += kr;
    kl_h += Kl(g.Truth(r), g.Hf(r), V);
    kl_max = std::max(kl_max, kr);
    top_r += Argmax(got[i].data(), V) == Argmax(g.Truth(r), V) ? 1 : 0;
    top_h += Argmax(g.Hf(r), V) == Argmax(g.Truth(r), V) ? 1 : 0;
  }
  Check(shape_ok, tag + ": every returned logits row has vocab_size entries");
  const double n = static_cast<double>(rows.size());
  kl_r /= n;
  kl_h /= n;
  std::printf("  %s: %zu rows, mean KL(truth||r4dx) %.3e, mean KL(truth||HF-bf16) %.3e, worst row %.3e, top-1 %d vs HF %d\n",
              tag.c_str(), rows.size(), kl_r, kl_h, kl_max, top_r, top_h);
  Check(kl_r <= 1.5 * kl_h + 5e-3, tag + ": mean KL(truth||r4dx) <= 1.5 x KL(truth||HF-bf16) + 5e-3 (8-bit tolerance)");
  Check(kl_max < 0.25, tag + ": every row KL(truth||r4dx) < 0.25");
  Check(top_r >= top_h - 3, tag + ": top-1 agreement with the truth >= HF-bf16's - 3 rows");
  return kl_r;
}

std::string ReadWhole(const fs::path& p) {
  std::ifstream f(p, std::ios::binary);
  if (!f) throw std::runtime_error("cannot read " + p.string());
  return std::string(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}

// A copy of the container at `src` whose header JSON went through `edit` (the safetensors shell: u64 length, the JSON
// padded with spaces to 8 bytes, then the data, whose offsets are relative to the data start and so unchanged).
template <class Edit>
std::string DoctoredCopy(const std::string& src, const fs::path& dst, Edit&& edit) {
  const std::string bytes = ReadWhole(src);
  uint64_t hlen = 0;
  std::memcpy(&hlen, bytes.data(), 8);
  json h = json::parse(bytes.substr(8, static_cast<size_t>(hlen)));
  edit(h);
  std::string hj = h.dump();
  while (hj.size() % 8 != 0) hj.push_back(' ');
  const uint64_t nlen = hj.size();
  std::ofstream f(dst, std::ios::binary | std::ios::trunc);
  f.write(reinterpret_cast<const char*>(&nlen), 8);
  f.write(hj.data(), static_cast<std::streamsize>(hj.size()));
  f.write(bytes.data() + 8 + hlen, static_cast<std::streamsize>(bytes.size() - 8 - hlen));
  if (!f) throw std::runtime_error("cannot write " + dst.string());
  return dst.string();
}

// Check 6 / 8: the layout of every linear the container loaded.
struct LayoutCount {
  int w8 = 0, bf16 = 0, other = 0;
  int expert_w8 = 0, expert_total = 0, nonexpert_bf16 = 0, nonexpert_total = 0;
  bool all_g128 = true;
};
LayoutCount CountLayouts(const Lfm2Container& c) {
  LayoutCount n;
  const auto one = [&](const QuantLinear& q, bool expert) {
    if (q.layout == Layout::kW8a16) {
      ++n.w8;
      n.all_g128 = n.all_g128 && q.w8a16_group == 128;
    } else if (q.layout == Layout::kBf16) {
      ++n.bf16;
    } else {
      ++n.other;
    }
    (expert ? n.expert_total : n.nonexpert_total) += 1;
    if (expert && q.layout == Layout::kW8a16) ++n.expert_w8;
    if (!expert && q.layout == Layout::kBf16) ++n.nonexpert_bf16;
  };
  for (int64_t i = 0; i < c.NumLoadedLayers(); ++i) {
    const Lfm2LayerWeights& lw = c.Layer(i);
    if (lw.attn_layer) {
      for (const QuantLinear* q : {&lw.attn.q, &lw.attn.k, &lw.attn.v, &lw.attn.o}) one(*q, false);
    } else {
      one(lw.conv.in_proj, false);
      one(lw.conv.out_proj, false);
    }
    if (lw.dense) {
      one(lw.mlp.gate_up, false);
      one(lw.mlp.down, false);
    } else {
      for (const QuantLinear& q : lw.moe.gate_up) one(q, true);
      for (const QuantLinear& q : lw.moe.down) one(q, true);
    }
  }
  // The head counts in w8 / bf16 / other and all_g128, not in the expert / non-expert split (a --keep-bf16 regex may or
  // may not cover it).
  const int nb = n.nonexpert_bf16, nt = n.nonexpert_total;
  one(c.LmHead(), false);
  n.nonexpert_bf16 = nb;
  n.nonexpert_total = nt;
  return n;
}

}  // namespace

int main() {
  return r4dx_test::RunGuardedMain("test_lfm2_tiny_w8a16", []() -> int {
    const std::string wdir = EnvOr("R4DX_LFM2_TINY_W8A16_DIR", r4dx::ModelsPath("r4dx/lfm2/tiny_w8a16"));
    const std::string bdir = EnvOr("R4DX_LFM2_TINY_DIR", r4dx::ModelsPath("r4dx/lfm2/tiny_e2e"));
    const std::string wcont = wdir + "/tiny_w8a16.r4dx";
    const std::string bcont = bdir + "/tiny.r4dx", bgold = bdir + "/e2e_golden.safetensors";
    for (const std::string& f : {wcont, bcont, bgold}) {
      if (!r4dx_test::FileExists(f)) return r4dx_test::SkipMissing(f + " (lane W1's tiny w8a16 container / tiny_e2e_golden.py)");
    }
    const std::string wgold = EnvOr("R4DX_LFM2_TINY_W8A16_GOLDEN", wdir + "/w8a16_golden.safetensors");
    const bool have_q = r4dx_test::FileExists(wgold);

    Golden g;
    {
      const r4dx_convert::SafetensorsReader rb(r4dx_convert::Utf8ToWide(bgold));
      g.ids = ReadIds(rb);
      g.N = static_cast<int64_t>(g.ids.size());
      g.truth = r4dx_test::ReadGoldenAsFloat(rb, "logits_truth");
      g.hf = r4dx_test::ReadGoldenAsFloat(rb, "logits_hf_bf16");
      g.V = static_cast<int64_t>(g.truth.size()) / g.N;
      if (have_q) {
        const r4dx_convert::SafetensorsReader rq(r4dx_convert::Utf8ToWide(wgold));
        Check(ReadIds(rq) == g.ids, "the w8a16 golden's token ids are the tiny_e2e golden's");
        g.truth_q = r4dx_test::ReadGoldenAsFloat(rq, "logits_truth_q");
        g.hf_q = r4dx_test::ReadGoldenAsFloat(rq, "logits_hf_bf16_q");
        if (static_cast<int64_t>(g.truth_q.size()) != g.N * g.V) throw std::runtime_error("w8a16 golden shape disagrees");
      } else {
        std::printf("  (no %s: check 1b skipped; check 1 gates against the unquantized truth)\n", wgold.c_str());
      }
    }
    if (g.N <= kPrefillEnd + 2) throw std::runtime_error("the golden needs more than 282 tokens");
    const int64_t V = g.V;

    const Lfm2ContainerInfo info = Lfm2Container::Inspect(wcont);
    Check(info.has_w8a16 && !info.has_trellis, "Inspect: the tiny container carries __metadata__.quant.w8a16");

    // ---- 1-3, 5: the whole stack through LoadTextModel --layout w8a16, TP=1 ----
    std::vector<std::vector<float>> got;
    std::vector<int64_t> rows;
    {
      TpOptions tp;  // world 1
      std::unique_ptr<TextModel> m = LoadTextModel(Opts(wcont, Layout::kW8a16), tp);
      Check(m->Config().arch == Arch::kLfm2Moe && m->Config().vocab_size == V,
            "LoadTextModel --layout w8a16: the LFM2 engine, vocab matches the golden");
      m->Reset();
      got = Feed(*m, g, &rows);
      Check(m->PositionCount() == g.N, "PositionCount after the feed == " + std::to_string(g.N));
      (void)GateVsTruth(g, got, rows, "w8a16 TP=1");
      if (have_q) {
        double kl_r = 0, kl_h = 0, kl_max = 0;
        int top_r = 0, top_h = 0;
        for (size_t i = 0; i < rows.size(); ++i) {
          const int64_t r = rows[i];
          const double kr = Kl(g.TruthQ(r), got[i].data(), V);
          kl_r += kr;
          kl_h += Kl(g.TruthQ(r), g.HfQ(r), V);
          kl_max = std::max(kl_max, kr);
          top_r += Argmax(got[i].data(), V) == Argmax(g.TruthQ(r), V) ? 1 : 0;
          top_h += Argmax(g.HfQ(r), V) == Argmax(g.TruthQ(r), V) ? 1 : 0;
        }
        kl_r /= static_cast<double>(rows.size());
        kl_h /= static_cast<double>(rows.size());
        std::printf("  1b: mean KL(truth_q||r4dx) %.3e, mean KL(truth_q||HF-bf16_q) %.3e, worst %.3e, top-1 %d vs %d\n", kl_r,
                    kl_h, kl_max, top_r, top_h);
        Check(kl_r <= 1.5 * kl_h + 2e-3, "1b: mean KL(truth_q||r4dx) <= 1.5 x KL(truth_q||HF-bf16_q) + 2e-3");
        Check(kl_max < 0.25, "1b: every row KL(truth_q||r4dx) < 0.25");
        Check(top_r >= top_h - 2, "1b: top-1 agreement with truth_q >= HF-bf16_q's - 2 rows");
      }

      // 2. determinism
      m->Reset();
      std::vector<int64_t> rows2;
      Check(Feed(*m, g, &rows2) == got, "the same feed after Reset() gives identical bytes");

      // 3. checkpoint / restore
      m->Reset();
      (void)m->Prefill(std::vector<int32_t>(g.ids.begin(), g.ids.begin() + 256));
      m->SaveCheckpoint();
      const std::vector<int32_t> tail(g.ids.begin() + 256, g.ids.begin() + kPrefillEnd);
      const std::vector<float> a1 = m->Prefill(tail);
      const std::vector<float> b1 = m->DecodeStep(g.ids[kPrefillEnd]);
      m->RestoreCheckpoint();
      Check(m->PositionCount() == 256, "RestoreCheckpoint rewinds PositionCount to 256");
      const std::vector<float> a2 = m->Prefill(tail);
      const std::vector<float> b2 = m->DecodeStep(g.ids[kPrefillEnd]);
      Check(a1 == a2 && b1 == b2, "RestoreCheckpoint + the same tail reproduces the logits bit for bit");
      const double kl_split = Kl(got[1].data(), a1.data(), V);
      std::printf("  row 279 via 256+24 vs 1+256+23 chunks: KL %.3e\n", kl_split);
      Check(kl_split < 1e-3, "a different prefill chunking gives the same row 279 (KL < 1e-3)");

      // 5. greedy (no sampled path: vocab 64, see the header)
      m->Reset();
      (void)m->Prefill(std::vector<int32_t>(g.ids.begin(), g.ids.begin() + kPrefillEnd));
      const int32_t greedy = m->DecodeStepGreedy(g.ids[kPrefillEnd]);
      m->Reset();
      (void)m->Prefill(std::vector<int32_t>(g.ids.begin(), g.ids.begin() + kPrefillEnd));
      const std::vector<float> full = m->DecodeStep(g.ids[kPrefillEnd]);
      Check(greedy == static_cast<int32_t>(Argmax(full.data(), V)), "DecodeStepGreedy == argmax(DecodeStep) in the same state");
    }

    // (informative) the bf16 container's rows from the same feed.
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
      std::printf("  (informative) r4dx bf16 vs r4dx w8a16 rows: mean KL %.3e, top-1 agreement %d / %zu\n",
                  kl / static_cast<double>(bf.size()), agree, bf.size());
    }

    // ---- 4, 6: Lfm2Model directly ----
    {
      Lfm2ModelOptions lo;
      lo.container_path = wcont;
      lo.layout = Layout::kW8a16;
      lo.max_ctx = 512;
      Lfm2Model lm = Lfm2Model::Load(lo);
      const LayoutCount n = CountLayouts(lm.GetContainer());
      Check(n.w8 > 0 && n.bf16 == 0 && n.other == 0 && n.all_g128,
            "every body linear, expert and the head loaded as w8a16 g128 (" + std::to_string(n.w8) + " w8a16, " +
                std::to_string(n.bf16) + " bf16 fallbacks)");
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
        Check(diff == 0, "layer " + std::to_string(layer) + ": w8a16 experts, per-slot (M=1) == sorted (M<=16) bytes (" +
                             std::to_string(diff) + " of " + std::to_string(y1.size()) + " differ)");
        bool finite = true;
        for (uint16_t v : y1) finite = finite && std::isfinite(core::Bf16ToFloat(v));
        Check(finite, "layer " + std::to_string(layer) + ": the w8a16 MoE output is finite");
      }
    }

    // ---- 7: refusals ----
    {
      std::string msg;
      TpOptions tp1;
      Check(ThrowsWith([&] { (void)LoadTextModel(Opts(bcont, Layout::kW8a16), tp1); }, "needs a w8a16 container", &msg),
            "(1) --layout w8a16 on the bf16 container (no quant.w8a16) is refused: " + msg);
      const fs::path tmp = fs::temp_directory_path() / "r4dx_test_lfm2_tiny_w8a16";
      fs::create_directories(tmp);
      const std::string bias13 = DoctoredCopy(wcont, tmp / "bias13.r4dx", [](json& h) {
        h["__metadata__"]["quant"]["w8a16"]["scale_log2_bias"] = 13;
      });
      Check(ThrowsWith([&] { (void)LoadTextModel(Opts(bias13, Layout::kW8a16), tp1); }, "scale_log2_bias", &msg),
            "(2) a container with scale_log2_bias 13 is refused: " + msg);
      const std::string v2 = DoctoredCopy(wcont, tmp / "version2.r4dx", [](json& h) {
        h["__metadata__"]["quant"]["w8a16"]["version"] = 2;
      });
      Check(ThrowsWith([&] { (void)LoadTextModel(Opts(v2, Layout::kW8a16), tp1); }, "\"version\"", &msg),
            "(2) a container with quant.w8a16.version 2 is refused: " + msg);
      const std::string g64 = DoctoredCopy(wcont, tmp / "group64.r4dx", [](json& h) {
        h["__metadata__"]["quant"]["w8a16"]["group"] = 64;
      });
      Check(ThrowsWith([&] { (void)LoadTextModel(Opts(g64, Layout::kW8a16), tp1); }, ".w8a16.ws' is", &msg),
            "(3) a .ws whose size is not N*K/G*2 (g128 scales under a g64 header) is refused, naming the tensor: " + msg);
      TpOptions tp2;
      tp2.world = 2;
      tp2.mode = TpOptions::Mode::kEmulate;
      Check(ThrowsWith([&] { (void)LoadTextModel(Opts(wcont, Layout::kW8a16), tp2); }, "TP=1 only", &msg),
            "v1: --tp 2 --layout w8a16 is refused (TP=1 only): " + msg);
      Check(ThrowsWith([&] { (void)LoadTextModel(Opts(wcont, Layout::kBf16), tp1); }, "in any known on-disk form", &msg),
            "(5) --layout bf16 on the --no-bf16 w8a16 container is refused: " + msg);
      std::error_code ec;
      fs::remove_all(tmp, ec);
    }

    // ---- 8: a MIXED container (bf16 non-expert linears + w8a16 experts) ----
#ifdef R4DX_CONVERT_EXE
    {
      const std::string ckpt = bdir + "/ckpt";
      if (!fs::exists(ckpt)) {
        std::printf("  SKIP check 8: %s absent (the tiny checkpoint)\n", ckpt.c_str());
      } else {
        const fs::path tmp = fs::temp_directory_path() / "r4dx_test_lfm2_tiny_w8a16_mixed";
        fs::create_directories(tmp);
        const std::string mixed = (tmp / "tiny_w8a16_mixed.r4dx").string();
        const std::string cmd = std::string("\"\"") + R4DX_CONVERT_EXE + "\" --input \"" + ckpt + "\" --output \"" + mixed +
                                "\" --threads 2 --layouts w8a16 --no-bf16 --lm-head w8a16 --keep-bf16 \"^(?!.*\\.moe\\.experts\\.)\" > \"" +
                                (tmp / "convert.log").string() + "\" 2>&1\"";
        const int rc = std::system(cmd.c_str());
        Check(rc == 0 && fs::exists(mixed), "r4dx-convert wrote the mixed tiny container (rc " + std::to_string(rc) + ")");
        if (rc == 0 && fs::exists(mixed)) {
          {
            Lfm2ModelOptions lo;
            lo.container_path = mixed;
            lo.layout = Layout::kW8a16;
            lo.max_ctx = 512;
            Lfm2Model lm = Lfm2Model::Load(lo);
            const LayoutCount n = CountLayouts(lm.GetContainer());
            Check(n.expert_total > 0 && n.expert_w8 == n.expert_total, "mixed: every expert loaded as w8a16 (" +
                                                                           std::to_string(n.expert_w8) + " / " +
                                                                           std::to_string(n.expert_total) + ")");
            Check(n.nonexpert_bf16 == n.nonexpert_total, "mixed: every non-expert body linear fell back to bf16 (" +
                                                             std::to_string(n.nonexpert_bf16) + " / " +
                                                             std::to_string(n.nonexpert_total) + ")");
          }
          TpOptions tp;
          std::unique_ptr<TextModel> m = LoadTextModel(Opts(mixed, Layout::kW8a16), tp);
          std::vector<int64_t> rows_m;
          const std::vector<std::vector<float>> mg = Feed(*m, g, &rows_m);
          (void)GateVsTruth(g, mg, rows_m, "mixed w8a16 experts + bf16 rest");
        }
        std::error_code ec;
        fs::remove_all(tmp, ec);
      }
    }
#else
    std::printf("  SKIP check 8: this build has no r4dx-convert (R4DX_BUILD_CONVERT off)\n");
#endif

    if (g_failures != 0) {
      std::printf("%d check(s) FAILED\n", g_failures);
      return 1;
    }
    std::printf("all checks passed\n");
    return 0;
  });
}
