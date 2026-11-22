// tests/model/test_inkling2_mxfp8.cpp -- CPU-only (no HIP call). docs/inkling2-plan.md 4.1 / 5: the loader half of the
// mxfp8 layout.
//   * mxfp8_meta.h: ParseMxfp8Metadata accepts exactly the version-1 block and refuses every other version / group / elem /
//     scale / fragment permutation and any unknown field; RefuseMxfp8Choice (--layout mxfp8 without the block);
//     CheckMxfp8MetaAgainst (the kernel's group); Mxfp8ShapeProblem (N % 16, K % 32, the two tensors' byte sizes);
//   * Layout::kMxfp8 == 6 and its name round-trip;
//   * the TP=2 byte plans of an mxfp8 linear (tp::PlanRows / PlanCols over tp::Part::kW8Wq at group 32 and
//     tp::Part::kMxfp8Ws): for the inkling2 shard rules -- a two-segment row split (expert gate_up = [gate; up]), a
//     one-segment row split (q / k / v / gate / lm_head) and a K split (attn.o, expert down) -- the gathered bytes equal the
//     converter's packers run on the sliced codes / scales, and decode to the slice of the full weight; ranges that are
//     not whole 16-row tiles / 32-K groups are refused.
#include <cstdint>
#include <cstdio>
#include <functional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "mxfp8_meta.h"
#include "nlohmann/json.hpp"
#include "quant_linear.h"
#include "r4dx_convert/mxfp8_format.hpp"
#include "r4dx_convert/quant_mxfp8.hpp"
#include "tp/tp_shard.h"

using nlohmann::json;
namespace tp = r4dx::model::tp;
using r4dx::model::Layout;

namespace {

int g_failures = 0;
void Check(bool cond, const std::string& what) {
  std::printf("%s: %s\n", cond ? "PASS" : "FAIL", what.c_str());
  if (!cond) ++g_failures;
}
template <class F>
bool Throws(F&& f, const std::string& substr = "") {
  try {
    f();
  } catch (const std::exception& e) {
    return substr.empty() || std::string(e.what()).find(substr) != std::string::npos;
  }
  return false;
}

json Block() {
  namespace f = r4dx_convert::mxfp8;
  return {{"version", f::kMetaVersion},
          {"group", f::kGroup},
          {"elem", f::kMetaElem},
          {"scale", f::kMetaScale},
          {"fragment_permutation", f::kMetaFragmentPermutation}};
}
json Meta(const json& block) { return {{"quant", {{"mxfp8", block}}}, {"model_arch", "inkling2_loop"}}; }

void TestMeta() {
  using r4dx::model::Mxfp8FormatMismatch;
  using r4dx::model::ParseMxfp8Metadata;
  const auto spec = ParseMxfp8Metadata(Meta(Block()), "x.r4dx");
  Check(spec && spec->version == 1 && spec->group == 32, "the version-1 block parses (group 32)");
  Check(!ParseMxfp8Metadata(json{{"model_arch", "inkling2_loop"}}, "x") && !ParseMxfp8Metadata(json{{"quant", json::object()}}, "x"),
        "no quant.mxfp8 -> not an mxfp8 container");
  const auto refused = [&](const std::function<void(json&)>& edit, const std::string& sub, const std::string& what) {
    json b = Block();
    edit(b);
    bool typed = false;
    try {
      (void)ParseMxfp8Metadata(Meta(b), "x.r4dx");
    } catch (const Mxfp8FormatMismatch& e) {
      typed = std::string(e.what()).find(sub) != std::string::npos;
    } catch (...) {
    }
    Check(typed, "quant.mxfp8 refused (Mxfp8FormatMismatch): " + what);
  };
  refused([](json& b) { b["version"] = 2; }, "version", "version 2");
  refused([](json& b) { b.erase("version"); }, "version", "no version");
  refused([](json& b) { b["group"] = 16; }, "group", "group 16");
  refused([](json& b) { b["group"] = "32"; }, "group", "a string group");
  refused([](json& b) { b["elem"] = "e5m2"; }, "elem", "elem e5m2");
  refused([](json& b) { b["scale"] = "fp8"; }, "scale", "scale fp8");
  refused([](json& b) { b["fragment_permutation"] = "row_major"; }, "fragment_permutation", "another fragment order");
  refused([](json& b) { b["bias"] = 0; }, "bias", "an unknown field");
  {
    bool typed = false;
    try {
      (void)ParseMxfp8Metadata(json{{"quant", {{"mxfp8", 1}}}}, "x");
    } catch (const Mxfp8FormatMismatch&) {
      typed = true;
    }
    Check(typed, "quant.mxfp8 that is not an object refused");
  }
  Check(Throws([] { r4dx::model::RefuseMxfp8Choice(std::nullopt, true, "bf16.r4dx"); }, "needs an mxfp8 container"),
        "--layout mxfp8 on a container without the block refused before any upload");
  Check(!Throws([] { r4dx::model::RefuseMxfp8Choice(std::nullopt, false, "bf16.r4dx"); }),
        "no refusal when mxfp8 was not requested");
  Check(!Throws([] { r4dx::model::CheckMxfp8MetaAgainst(32, 32, "x"); }) &&
            Throws([] { r4dx::model::CheckMxfp8MetaAgainst(32, 64, "x"); }, "group"),
        "the kernel group check (32 == 32 ok, 32 vs 64 refused)");
  using r4dx::model::Mxfp8ShapeProblem;
  Check(Mxfp8ShapeProblem("a", 64, 128, 64 * 128, 64 * 4).empty(), "shape rules: [64, 128] with N*K / N*K/32 bytes holds");
  Check(!Mxfp8ShapeProblem("a", 24, 128, 24 * 128, 24 * 4).empty(), "shape rules: N 24 (not a multiple of 16) refused");
  Check(!Mxfp8ShapeProblem("a", 64, 48, 64 * 48, 64).empty(), "shape rules: K 48 (not a multiple of 32) refused");
  Check(Mxfp8ShapeProblem("a", 64, 128, 64 * 128 - 1, 64 * 4).find(".mxfp8.wq") != std::string::npos,
        "shape rules: a short .mxfp8.wq named");
  Check(Mxfp8ShapeProblem("a", 64, 128, 64 * 128, 64 * 4 + 16).find(".mxfp8.ws") != std::string::npos,
        "shape rules: a long .mxfp8.ws named");
  Check(static_cast<int>(Layout::kMxfp8) == 6 && std::string(r4dx::model::LayoutName(Layout::kMxfp8)) == "mxfp8" &&
            r4dx::model::LayoutFromName("mxfp8") == Layout::kMxfp8,
        "Layout::kMxfp8 == 6, named mxfp8 both ways");
}

// One mxfp8 linear W[N, K] as the converter writes it.
struct Linear {
  int N = 0, K = 0;
  std::vector<float> w;
  std::vector<uint8_t> codes, scales;  // unpacked [N, K] / [N, K/32]
  std::vector<uint8_t> wq, ws;         // packed (on-disk)
};
Linear Make(int N, int K, uint32_t seed) {
  Linear l;
  l.N = N;
  l.K = K;
  std::mt19937 rng(seed);
  std::normal_distribution<float> d(0.0f, 1.0f);
  l.w.resize(static_cast<size_t>(N) * K);
  for (size_t i = 0; i < l.w.size(); ++i) l.w[i] = d(rng) * std::ldexp(1.0f, static_cast<int>(i % 7) - 3);
  r4dx_convert::QuantizeMxfp8(l.w.data(), N, K, 1, l.codes, l.scales);
  l.wq = r4dx_convert::PackMxfp8Codes(l.codes, N, K, 1);
  l.ws = r4dx_convert::PackMxfp8Scales(l.scales, N, K);
  return l;
}
// The packed form of rows `rows` / columns [c0, c0 + nc) of `l`.
void Expected(const Linear& l, const std::vector<tp::Range>& rows, tp::Range cols, std::vector<uint8_t>* wq,
              std::vector<uint8_t>* ws, int* n_out, int* k_out) {
  std::vector<uint8_t> c, s;
  int n = 0;
  for (const tp::Range& r : rows) {
    for (int64_t i = r.begin; i < r.begin + r.count; ++i) {
      for (int64_t k = cols.begin; k < cols.begin + cols.count; ++k) c.push_back(l.codes[static_cast<size_t>(i * l.K + k)]);
      for (int64_t g = cols.begin / 32; g < (cols.begin + cols.count) / 32; ++g)
        s.push_back(l.scales[static_cast<size_t>(i * (l.K / 32) + g)]);
      ++n;
    }
  }
  const int k = static_cast<int>(cols.count);
  *wq = r4dx_convert::PackMxfp8Codes(c, n, k, 1);
  *ws = r4dx_convert::PackMxfp8Scales(s, n, k);
  *n_out = n;
  *k_out = k;
}

tp::PartShape Shape(tp::Part p, int N, int K) {
  tp::PartShape s;
  s.part = p;
  s.N = N;
  s.K = K;
  s.group = 32;
  return s;
}

void TestPlans() {
  struct Case {
    std::string what;
    int N, K;
    tp::ShardRule rule;
  };
  const auto rows_rule = [](std::vector<int64_t> segs) {
    tp::ShardRule r;
    r.split = tp::Split::kRows;
    int64_t b = 0;
    for (int64_t n : segs) {
      r.segments.push_back({b, n});
      b += n;
    }
    return r;
  };
  tp::ShardRule cols_rule;
  cols_rule.split = tp::Split::kCols;
  // proxies of the real shapes with the same tile / group arithmetic: gate_up [2 x 96, 128] (rank rows 48 + 48),
  // q [192, 128] (rank 96 rows), down [128, 128] (rank K 64 = 2 groups), o [64, 192] (rank K 96 = 3 groups)
  std::vector<Case> cases;
  cases.push_back({"expert gate_up [gate; up] two-segment row split", 192, 128, rows_rule({96, 96})});
  cases.push_back({"q / k / v / gate / lm_head one-segment row split", 192, 128, rows_rule({192})});
  tp::ShardRule down = cols_rule;
  down.k_total = 128;
  cases.push_back({"expert down K split", 128, 128, down});
  tp::ShardRule o = cols_rule;
  o.k_total = 192;
  cases.push_back({"attn.o K split (odd group count per rank)", 64, 192, o});
  uint32_t seed = 1;
  for (const Case& c : cases) {
    const Linear l = Make(c.N, c.K, seed++);
    for (int rank = 0; rank < 2; ++rank) {
      std::vector<tp::Range> rows{{0, c.N}};
      tp::Range cols{0, c.K};
      std::vector<tp::ByteRun> pq, ps;
      if (c.rule.split == tp::Split::kRows) {
        rows = tp::RankRows(c.rule, 2, rank);
        pq = tp::PlanRows(Shape(tp::Part::kW8Wq, c.N, c.K), rows);
        ps = tp::PlanRows(Shape(tp::Part::kMxfp8Ws, c.N, c.K), rows);
      } else {
        cols = tp::RankCols(c.rule, 2, rank);
        pq = tp::PlanCols(Shape(tp::Part::kW8Wq, c.N, c.K), cols);
        ps = tp::PlanCols(Shape(tp::Part::kMxfp8Ws, c.N, c.K), cols);
      }
      const std::vector<uint8_t> gq = tp::Gather(l.wq.data(), l.wq.size(), pq), gs = tp::Gather(l.ws.data(), l.ws.size(), ps);
      std::vector<uint8_t> eq, es;
      int n = 0, k = 0;
      Expected(l, rows, cols, &eq, &es, &n, &k);
      Check(gq == eq && gs == es, c.what + " rank " + std::to_string(rank) + ": gathered wq / ws == the packers on the slice");
      // and it decodes to the slice of the full weight's decode
      const std::vector<float> full = r4dx_convert::DequantMxfp8(l.wq, l.ws, c.N, c.K);
      const std::vector<float> part = r4dx_convert::DequantMxfp8(gq, gs, n, k);
      bool same = static_cast<int>(part.size()) == n * k;
      int64_t row = 0;
      for (const tp::Range& r : rows)
        for (int64_t i = r.begin; same && i < r.begin + r.count; ++i, ++row)
          for (int64_t kk = 0; same && kk < k; ++kk)
            same = part[static_cast<size_t>(row * k + kk)] == full[static_cast<size_t>(i * c.K + cols.begin + kk)];
      Check(same, c.what + " rank " + std::to_string(rank) + ": the shard decodes to the slice of the full weight");
    }
  }
  // misaligned ranges are refused
  Check(Throws([] { (void)tp::PlanRows(Shape(tp::Part::kMxfp8Ws, 64, 128), {{8, 16}}); }),
        "a row range that is not whole 16-row tiles is refused (ws)");
  Check(Throws([] { (void)tp::PlanRows(Shape(tp::Part::kW8Wq, 64, 128), {{0, 24}}); }),
        "a row count that is not whole 16-row tiles is refused (wq)");
  Check(Throws([] { (void)tp::PlanCols(Shape(tp::Part::kMxfp8Ws, 64, 128), {16, 64}); }),
        "a K range that is not whole 32-K groups is refused (ws)");
  Check(Throws([] { (void)tp::PlanCols(Shape(tp::Part::kW8Wq, 64, 128), {0, 48}); }),
        "a K count that is not whole 32-K groups is refused (wq)");
  // the tiny fixture's expert down at TP=2 (K 32 -> 16 per rank) cannot be an mxfp8 shard: the tiny TP=2 run is bf16
  Check(Throws([] { (void)tp::PlanCols(Shape(tp::Part::kW8Wq, 64, 32), {0, 16}); }),
        "a 16-wide K shard (the tiny fixture's down at TP=2) is refused");
}

}  // namespace

int main() {
  try {
    TestMeta();
    TestPlans();
  } catch (const std::exception& e) {
    std::fprintf(stderr, "FAIL: unexpected exception: %s\n", e.what());
    return 1;
  }
  if (g_failures != 0) {
    std::printf("%d check(s) FAILED\n", g_failures);
    return 1;
  }
  std::printf("all checks passed\n");
  return 0;
}
