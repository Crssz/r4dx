// tests/model/test_lfm2_tp_shard.cpp -- CPU-only (no GPU, no container, no HIP). docs/lfm2-plan.md L1-4.
//
// The LFM2 tensor-parallel sharding contract on the real LFM2.5-24B-A2B config (values inlined so the test is hermetic):
//   A. tp::RuleFor's lfm2_moe table for every tensor of every one of the 40 layers: which split, which segments, which K --
//      attention q / k / v column-parallel and out_proj row-parallel (K 1024 per rank), conv.in_proj a THREE-segment column
//      split (B | C | x), conv.weight rows, conv.out_proj row-parallel, dense gate_up (two segments) / down, the router and
//      expert_bias and every norm replicated, EVERY expert's gate_up / down split, the tied head a vocab slice;
//   B. consistency with Lfm2Config::Shard: the rows / K every rank's slice gets are exactly the per-rank shapes the kernels
//      are sized for;
//   C. the PHYSICAL plans (bf16, w4a16 wq / wsz at groups 32 and 64, trellis KB 4 / 5, the conv taps as a 6-byte-row elem
//      part): each rank's runs are whole 128-blocks / 16-row tiles and the two ranks' byte runs partition the part exactly;
//   D. the algebra of TP=2 on a tiny LFM2-shaped model with real bf16 bytes: the three-segment in_proj columns concatenate
//      per channel, a depthwise conv + gating on a rank's channels equals the same channels of the unsharded one (so the
//      row-parallel out_proj partial sums add up to the unsharded product), the dense and per-expert SwiGLU with the
//      gate_up / down rules gives the unsharded output as a sum of two partials;
//   E. refusals: unknown names, a conv tensor on an attention layer and the reverse, a dense mlp on an MoE layer and the
//      reverse, an expert past num_experts, mtp.* / rotation.* on LFM2, a layer out of range, a still-suffixed name, an
//      already-sharded config; and the Qwen / Gemma tables unchanged.
// What it cannot show (GPU): that the kernels and the all-reduce agree with it.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "lfm2_config.h"
#include "model_config.h"
#include "nlohmann/json.hpp"
#include "tp/tp_shard.h"

using nlohmann::json;
using r4dx::model::Lfm2Config;
using r4dx::model::ModelConfig;
namespace tp = r4dx::model::tp;

namespace {

int g_failures = 0;

void Check(bool cond, const std::string& what) {
  if (!cond) {
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++g_failures;
  } else {
    std::printf("PASS: %s\n", what.c_str());
  }
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

json RealConfig() {
  json types = json::array();
  for (int i = 0; i < 40; ++i) types.push_back(i % 4 == 2 ? "full_attention" : "conv");
  return {{"conv_L_cache", 3},          {"conv_bias", false},       {"hidden_size", 2048},
          {"intermediate_size", 11776}, {"layer_types", types},      {"max_position_embeddings", 131072},
          {"model_type", "lfm2_moe"},   {"moe_intermediate_size", 1536}, {"norm_eps", 1e-05},
          {"norm_topk_prob", true},     {"num_attention_heads", 32}, {"num_dense_layers", 2},
          {"num_experts", 64},          {"num_experts_per_tok", 4},  {"num_hidden_layers", 40},
          {"num_key_value_heads", 8},   {"rope_parameters", {{"rope_theta", 1e7}, {"rope_type", "default"}}},
          {"routed_scaling_factor", 1.0}, {"tie_word_embeddings", true}, {"use_expert_bias", true},
          {"vocab_size", 128000}};
}

std::string L(int i, const std::string& tail) { return "text.layers." + std::to_string(i) + "." + tail; }
std::string Ex(int i, int e, const char* part) { return L(i, "moe.experts." + std::to_string(e) + "." + part); }

struct Slice {
  tp::Split split;
  int64_t rows = 0, k = 0, first_row = 0, first_k = 0;
  std::vector<tp::Range> ranges;
};
Slice SliceOf(const tp::ShardRule& r, int rank, int64_t N, int64_t K) {
  Slice s;
  s.split = r.split;
  s.rows = N;
  s.k = K;
  if (r.split == tp::Split::kRows) {
    s.ranges = tp::RankRows(r, 2, rank);
    s.rows = 0;
    for (const tp::Range& g : s.ranges) s.rows += g.count;
    s.first_row = s.ranges.front().begin;
  } else if (r.split == tp::Split::kCols) {
    const tp::Range c = tp::RankCols(r, 2, rank);
    s.k = c.count;
    s.first_k = c.begin;
  }
  return s;
}

// ---- A + B -------------------------------------------------------------------------------------------------------------

void TestRealRules() {
  const Lfm2Config g = Lfm2Config::FromJson(RealConfig());
  const ModelConfig gm = g.ToModelConfig();
  const Lfm2Config r0 = Lfm2Config::Shard(g, 2, 0), r1 = Lfm2Config::Shard(g, 2, 1);
  const int64_t H = g.hidden_size;
  bool all = true;
  int tensors = 0;
  auto expect = [&](bool cond, const std::string& what) {
    if (!cond) std::fprintf(stderr, "  rule mismatch: %s\n", what.c_str());
    all = all && cond;
  };
  for (int i = 0; i < 40; ++i) {
    for (const char* n : {"operator_norm", "ffn_norm"}) {
      expect(tp::RuleFor(L(i, n), gm).split == tp::Split::kReplicate, L(i, n));
      ++tensors;
    }
    if (g.IsAttentionLayer(i)) {
      const tp::ShardRule q = tp::RuleFor(L(i, "attn.q"), gm), k = tp::RuleFor(L(i, "attn.k"), gm), v = tp::RuleFor(L(i, "attn.v"), gm),
                          o = tp::RuleFor(L(i, "attn.o"), gm);
      tensors += 4;
      expect(q.split == tp::Split::kRows && q.segments.size() == 1 && q.segments[0].rows == 2048, L(i, "attn.q"));
      expect(k.split == tp::Split::kRows && v.split == tp::Split::kRows && k.segments.size() == 1 && k.segments[0].rows == 512, L(i, "attn.k/v"));
      expect(o.split == tp::Split::kCols && o.k_total == 2048, L(i, "attn.o"));
      for (int rk = 0; rk < 2; ++rk) {
        const Lfm2Config& r = rk == 0 ? r0 : r1;
        const Slice sq = SliceOf(q, rk, g.QDim(), H), sk = SliceOf(k, rk, g.KvDim(), H), sv = SliceOf(v, rk, g.KvDim(), H),
                    so = SliceOf(o, rk, H, g.QDim());
        expect(sq.rows == r.QDim() && sq.first_row == rk * 1024, "q rank slice");
        expect(sk.rows == r.KvDim() && sv.rows == r.KvDim() && sk.first_row == rk * 256, "k/v rank slice");
        expect(so.k == r.QDim() && so.k == 1024 && so.first_k == rk * 1024, "o rank K");
      }
      for (const char* n : {"attn.q_norm", "attn.k_norm"}) {
        expect(tp::RuleFor(L(i, n), gm).split == tp::Split::kReplicate, L(i, n));
        ++tensors;
      }
      expect(Throws([&] { (void)tp::RuleFor(L(i, "conv.in_proj"), gm); }, "ShortConv tensor on an attention layer"), "conv on attn layer");
    } else {
      const tp::ShardRule in = tp::RuleFor(L(i, "conv.in_proj"), gm), w = tp::RuleFor(L(i, "conv.weight"), gm),
                          out = tp::RuleFor(L(i, "conv.out_proj"), gm);
      tensors += 3;
      // THREE segments B | C | x of H rows each; rank r takes channels [r*1024, (r+1)*1024) of each.
      expect(in.split == tp::Split::kRows && in.segments.size() == 3 && in.segments[0].rows == 2048 && in.segments[1].begin == 2048 &&
                 in.segments[2].begin == 4096 && in.segments[2].rows == 2048,
             L(i, "conv.in_proj segments"));
      for (int rk = 0; rk < 2; ++rk) {
        const std::vector<tp::Range> rr = tp::RankRows(in, 2, rk);
        expect(rr.size() == 3, "in_proj ranges");
        for (int s = 0; s < 3 && rr.size() == 3; ++s)
          expect(rr[static_cast<size_t>(s)].begin == s * 2048 + rk * 1024 && rr[static_cast<size_t>(s)].count == 1024, "in_proj rank range");
        const Slice sw = SliceOf(w, rk, g.conv_dim, 3), so = SliceOf(out, rk, H, g.conv_dim);
        expect(w.split == tp::Split::kRows && w.segments.size() == 1 && w.segments[0].rows == 2048 && sw.rows == (rk == 0 ? r0 : r1).conv_dim &&
                   sw.first_row == rk * 1024,
               "conv.weight rank rows");
        expect(out.split == tp::Split::kCols && out.k_total == 2048 && so.k == 1024 && so.first_k == rk * 1024, "conv.out_proj rank K");
      }
      expect(Throws([&] { (void)tp::RuleFor(L(i, "attn.q"), gm); }, "attention tensor on a ShortConv layer"), "attn on conv layer");
    }
    if (g.IsDenseLayer(i)) {
      const tp::ShardRule gu = tp::RuleFor(L(i, "mlp.gate_up"), gm), dn = tp::RuleFor(L(i, "mlp.down"), gm);
      tensors += 2;
      expect(gu.split == tp::Split::kRows && gu.segments.size() == 2 && gu.segments[0].rows == 11776 && gu.segments[1].begin == 11776, L(i, "mlp.gate_up"));
      for (int rk = 0; rk < 2; ++rk) {
        const std::vector<tp::Range> rr = tp::RankRows(gu, 2, rk);
        expect(rr.size() == 2 && rr[0].begin == rk * 5888 && rr[0].count == 5888 && rr[1].begin == 11776 + rk * 5888 && rr[1].count == 5888, "dense gate_up ranges");
        const tp::Range c = tp::RankCols(dn, 2, rk);
        expect(c.begin == rk * 5888 && c.count == 5888 && c.count == (rk == 0 ? r0 : r1).intermediate_size, "dense down K");
      }
      expect(dn.split == tp::Split::kCols && dn.k_total == 11776, L(i, "mlp.down"));
      expect(Throws([&] { (void)tp::RuleFor(L(i, "moe.router"), gm); }, "MoE tensor on a dense"), "moe on dense layer");
    } else {
      expect(tp::RuleFor(L(i, "moe.router"), gm).split == tp::Split::kReplicate &&
                 tp::RuleFor(L(i, "moe.expert_bias"), gm).split == tp::Split::kReplicate,
             L(i, "router / expert_bias replicate"));
      tensors += 2;
      for (int e = 0; e < 64; ++e) {
        const tp::ShardRule gu = tp::RuleFor(Ex(i, e, "gate_up"), gm), dn = tp::RuleFor(Ex(i, e, "down"), gm);
        tensors += 2;
        // gate_up: two segments (gate | up), each 1536 rows split 768 / 768; down: K 1536 -> 768 per rank.
        expect(gu.split == tp::Split::kRows && gu.segments.size() == 2 && gu.segments[0].rows == 1536 && gu.segments[1].begin == 1536, Ex(i, e, "gate_up"));
        expect(dn.split == tp::Split::kCols && dn.k_total == 1536, Ex(i, e, "down"));
        for (int rk = 0; rk < 2; ++rk) {
          const std::vector<tp::Range> rr = tp::RankRows(gu, 2, rk);
          const Lfm2Config& r = rk == 0 ? r0 : r1;
          expect(rr.size() == 2 && rr[0].begin == rk * 768 && rr[0].count == 768 && rr[1].begin == 1536 + rk * 768 && rr[1].count == 768 &&
                     rr[0].count == r.moe_intermediate_size,
                 "expert gate_up ranges");
          const tp::Range c = tp::RankCols(dn, 2, rk);
          expect(c.begin == rk * 768 && c.count == 768 && c.count == r.moe_intermediate_size, "expert down K");
        }
      }
      expect(Throws([&] { (void)tp::RuleFor(L(i, "mlp.gate_up"), gm); }, "dense-FFN tensor on an MoE"), "dense mlp on moe layer");
      expect(Throws([&] { (void)tp::RuleFor(Ex(i, 64, "down"), gm); }, "expert 64 of a 64-expert"), "expert 64 out of range");
    }
  }
  Check(all, "A/B: every LFM2 layer tensor's rule and per-rank slice match Lfm2Config::Shard (" + std::to_string(tensors) +
                 " tensors, 40 layers, 38 x 64 experts)");

  const tp::ShardRule emb = tp::RuleFor("text.embed_tokens", gm), head = tp::RuleFor("lm_head", gm);
  Check(emb.split == tp::Split::kReplicate && tp::RuleFor("text.final_norm", gm).split == tp::Split::kReplicate, "A: embed_tokens and final_norm replicate");
  const std::vector<tp::Range> h0 = tp::RankRows(head, 2, 0), h1 = tp::RankRows(head, 2, 1);
  Check(head.split == tp::Split::kRows && h0.size() == 1 && h0[0].begin == 0 && h0[0].count == 64000 && h1[0].begin == 64000 && h1[0].count == 64000 &&
            r0.VocabShardSize() == 64000 && r1.VocabShardBegin() == 64000,
        "A/B: the tied head is a vocab slice: rank r = rows [r*64000, (r+1)*64000)");
}

void TestRefusals() {
  const Lfm2Config g = Lfm2Config::FromJson(RealConfig());
  const ModelConfig gm = g.ToModelConfig();
  Check(Throws([&] { (void)tp::RuleFor("text.layers.40.attn.q", gm); }, "layer 40"), "E: layer 40 of a 40-layer model");
  Check(Throws([&] { (void)tp::RuleFor(L(2, "attn.qg"), gm); }, "unknown tensor"), "E: a Qwen name (attn.qg) on an LFM2 config is refused, not replicated");
  Check(Throws([&] { (void)tp::RuleFor(L(0, "gdn.out_proj"), gm); }, "unknown tensor"), "E: a GDN tensor is refused");
  Check(Throws([&] { (void)tp::RuleFor(L(2, "attn.q.bf16.w"), gm); }, "unknown tensor"), "E: a still-suffixed name is refused");
  Check(Throws([&] { (void)tp::RuleFor(Ex(3, 5, "gate_up.bf16.w"), gm); }, "unknown tensor"), "E: a still-suffixed expert name is refused");
  Check(Throws([&] { (void)tp::RuleFor(L(2, "attn.o_norm"), gm); }, "unknown tensor"), "E: an unknown suffix is refused");
  Check(Throws([&] { (void)tp::RuleFor(L(3, "moe.experts.x.down"), gm); }, "unknown tensor"), "E: a non-numeric expert index is refused");
  Check(Throws([&] { (void)tp::RuleFor(L(3, "moe.experts.1.up"), gm); }, "unknown tensor"), "E: an unknown expert tensor (up) is refused");
  Check(Throws([&] { (void)tp::RuleFor("mtp.fc", gm); }, "unknown tensor"), "E: mtp.* is refused on LFM2");
  Check(Throws([&] { (void)tp::RuleFor("rotation.signs", gm); }, "unknown tensor"), "E: rotation.* is refused on LFM2");
  Check(Throws([&] { (void)tp::RuleFor(L(3, "input_layernorm"), gm); }, "unknown tensor"), "E: Qwen's input_layernorm is not an LFM2 name (operator_norm is)");
  Check(Throws([&] { (void)tp::RuleFor("text.layers.0.attn.q", Lfm2Config::Shard(g, 2, 0).ToModelConfig()); }, "unsharded"), "E: RuleFor needs the UNSHARDED config");

  // The Qwen table is unchanged by the LFM2 one: attn.qg is still the q|gate rule there, and mtp.fc still replicates.
  ModelConfig q;
  q.arch = r4dx::model::Arch::kQwen35;
  q.num_hidden_layers = 4;
  q.layer_types = {"linear_attention", "linear_attention", "linear_attention", "full_attention"};
  q.num_attention_heads = 8;
  q.head_dim = 16;
  q.num_key_value_heads = 2;
  q.intermediate_size = 64;
  q.vocab_size = 64;
  Check(tp::RuleFor("text.layers.3.attn.qg", q).split == tp::Split::kRows && tp::RuleFor("mtp.fc", q).split == tp::Split::kReplicate &&
            tp::RuleFor("rotation.signs", q).split == tp::Split::kReplicate,
        "E: the Qwen table (attn.qg, mtp.fc, rotation.signs) is unchanged");
  Check(Throws([&] { (void)tp::RuleFor("text.layers.3.moe.router", q); }, "unknown tensor"), "E: Qwen has no moe.router");
}

// ---- C: physical plans -----------------------------------------------------------------------------------------------------

bool Partitions(std::vector<tp::ByteRun> a, const std::vector<tp::ByteRun>& b, size_t total) {
  a.insert(a.end(), b.begin(), b.end());
  std::sort(a.begin(), a.end(), [](const tp::ByteRun& x, const tp::ByteRun& y) { return x.src_off < y.src_off; });
  size_t at = 0;
  for (const tp::ByteRun& r : a) {
    if (r.src_off != at) return false;
    at += r.bytes;
  }
  return at == total;
}

struct Lin {
  std::string base;
  int64_t N, K;
  bool rows;
};

void TestPhysicalPlans() {
  const Lfm2Config g = Lfm2Config::FromJson(RealConfig());
  const ModelConfig gm = g.ToModelConfig();
  const int64_t H = g.hidden_size;
  std::vector<Lin> lins = {{L(0, "conv.in_proj"), 3 * H, H, true},     {L(0, "conv.out_proj"), H, H, false},
                           {L(0, "mlp.gate_up"), 2 * g.intermediate_size, H, true}, {L(0, "mlp.down"), H, g.intermediate_size, false},
                           {L(2, "attn.q"), g.QDim(), H, true},          {L(2, "attn.k"), g.KvDim(), H, true},
                           {L(2, "attn.v"), g.KvDim(), H, true},         {L(2, "attn.o"), H, g.QDim(), false},
                           {Ex(2, 0, "gate_up"), g.ExpertGateUpRows(), H, true}, {Ex(2, 63, "down"), H, g.moe_intermediate_size, false},
                           {"lm_head", g.vocab_size, H, true}};
  bool ok = true;
  int plans = 0;
  for (const Lin& l : lins) {
    const tp::ShardRule rule = tp::RuleFor(l.base, gm);
    struct P {
      tp::Part part;
      int group, rate;
      size_t total;
    };
    std::vector<P> parts = {{tp::Part::kBf16, 0, 0, static_cast<size_t>(l.N * l.K * 2)},
                            {tp::Part::kW4Wq, 0, 0, static_cast<size_t>(l.N * l.K / 2)},
                            {tp::Part::kW4a16Wsz, 32, 0, static_cast<size_t>(l.N * (l.K / 32) * 4)},
                            {tp::Part::kW4a16Wsz, 64, 0, static_cast<size_t>(l.N * (l.K / 64) * 4)},
                            {tp::Part::kW8Wq, 128, 0, static_cast<size_t>(l.N * l.K)},
                            {tp::Part::kW8a16Ws, 128, 0, static_cast<size_t>(l.N * (l.K / 128) * 2)},
                            {tp::Part::kW8a16Ws, 64, 0, static_cast<size_t>(l.N * (l.K / 64) * 2)},
                            {tp::Part::kTrellisW, 0, 4, static_cast<size_t>(l.N * l.K * 4 / 8)},
                            {tp::Part::kTrellisW, 0, 5, static_cast<size_t>(l.N * l.K * 5 / 8)}};
    if (l.base == "lm_head") parts.resize(7);
    for (const P& p : parts) {
      tp::PartShape sh;
      sh.part = p.part;
      sh.N = l.N;
      sh.K = l.K;
      sh.group = p.group;
      sh.rate = p.rate;
      std::vector<tp::ByteRun> a, b;
      try {
        if (l.rows) {
          a = tp::PlanRows(sh, tp::RankRows(rule, 2, 0));
          b = tp::PlanRows(sh, tp::RankRows(rule, 2, 1));
        } else {
          a = tp::PlanCols(sh, tp::RankCols(rule, 2, 0));
          b = tp::PlanCols(sh, tp::RankCols(rule, 2, 1));
        }
      } catch (const std::exception& e) {
        std::fprintf(stderr, "  plan %s part %d group %d rate %d threw: %s\n", l.base.c_str(), static_cast<int>(p.part), p.group, p.rate, e.what());
        ok = false;
        continue;
      }
      size_t ba = 0, bb = 0;
      for (const auto& r : a) ba += r.bytes;
      for (const auto& r : b) bb += r.bytes;
      const bool part_ok = ba == bb && ba == p.total / 2 && Partitions(a, b, p.total);
      if (!part_ok) std::fprintf(stderr, "  plan %s part %d group %d rate %d: bytes %zu / %zu of %zu\n", l.base.c_str(), static_cast<int>(p.part), p.group, p.rate, ba, bb, p.total);
      ok = ok && part_ok;
      ++plans;
    }
  }
  Check(ok, "C: bf16 / w4a16 (g32, g64) / w8a16 (wq, ws g128, g64) / trellis (KB4, KB5) plans of every distinct LFM2 linear: the two ranks partition the part (" +
                std::to_string(plans) + " plans)");

  // conv.weight is a 1-D-per-channel elem part: 6 bytes per channel row (3 bf16 taps); the ranks' channel blocks partition it.
  {
    const tp::ShardRule w = tp::RuleFor(L(0, "conv.weight"), gm);
    tp::PartShape sh;
    sh.part = tp::Part::kElem;
    sh.N = g.conv_dim;
    sh.row_bytes = 3 * 2;
    const auto a = tp::PlanRows(sh, tp::RankRows(w, 2, 0)), b = tp::PlanRows(sh, tp::RankRows(w, 2, 1));
    Check(a.size() == 1 && b.size() == 1 && a[0].src_off == 0 && a[0].bytes == 1024 * 6 && b[0].src_off == 1024 * 6 && Partitions(a, b, 2048 * 6),
          "C: conv.weight [2048, 3] bf16: rank r's channels are one contiguous byte run of 6144 bytes");
  }
  bool align = true;
  for (const Lin& l : lins) {
    const tp::ShardRule rule = tp::RuleFor(l.base, gm);
    for (int rk = 0; rk < 2; ++rk) {
      if (l.rows) {
        for (const tp::Range& r : tp::RankRows(rule, 2, rk)) align = align && r.begin % 128 == 0 && r.count % 128 == 0;
      } else {
        const tp::Range c = tp::RankCols(rule, 2, rk);
        align = align && c.begin % 128 == 0 && c.count % 128 == 0;
      }
    }
  }
  Check(align, "C: every rank row / K range of every LFM2 linear is a whole number of 128-blocks (trellis-legal)");
}

// ---- D: the algebra of the sharded forward on a tiny LFM2-shaped model ---------------------------------------------------------

uint16_t Bf16Of(int v) {
  const float f = static_cast<float>(v);
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return static_cast<uint16_t>(u >> 16);
}
float FloatOf(uint16_t b) {
  const uint32_t u = static_cast<uint32_t>(b) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

struct Mat {
  int64_t N, K;
  std::vector<uint8_t> bytes;  // bf16 row-major [N, K]
  Mat(int64_t n, int64_t k, uint32_t seed) : N(n), K(k), bytes(static_cast<size_t>(n * k * 2)) {
    uint32_t s = seed;
    for (int64_t i = 0; i < n * k; ++i) {
      s = s * 1664525u + 1013904223u;
      const uint16_t v = Bf16Of(static_cast<int>((s >> 24) % 7) - 3);
      std::memcpy(&bytes[static_cast<size_t>(i) * 2], &v, 2);
    }
  }
  double At(int64_t n, int64_t k) const {
    uint16_t v;
    std::memcpy(&v, &bytes[static_cast<size_t>(n * K + k) * 2], 2);
    return FloatOf(v);
  }
};
Mat SliceMat(const Mat& m, const tp::ShardRule& rule, int rank) {
  tp::PartShape sh;
  sh.part = tp::Part::kBf16;
  sh.N = m.N;
  sh.K = m.K;
  if (rule.split == tp::Split::kRows) {
    const auto runs = tp::PlanRows(sh, tp::RankRows(rule, 2, rank));
    Mat s(0, m.K, 0);
    s.bytes = tp::Gather(m.bytes.data(), m.bytes.size(), runs);
    s.N = static_cast<int64_t>(s.bytes.size()) / 2 / m.K;
    return s;
  }
  const tp::Range c = tp::RankCols(rule, 2, rank);
  const auto runs = tp::PlanCols(sh, c);
  Mat s(0, c.count, 0);
  s.bytes = tp::Gather(m.bytes.data(), m.bytes.size(), runs);
  s.N = m.N;
  return s;
}
std::vector<double> Gemv(const Mat& w, const std::vector<double>& x) {
  std::vector<double> y(static_cast<size_t>(w.N), 0.0);
  for (int64_t n = 0; n < w.N; ++n) {
    double a = 0;
    for (int64_t k = 0; k < w.K; ++k) a += w.At(n, k) * x[static_cast<size_t>(k)];
    y[static_cast<size_t>(n)] = a;
  }
  return y;
}
double Silu(double x) { return x / (1.0 + std::exp(-x)); }

// y = W_down (silu(gate x) * (up x)) for a gate_up [2I, H] / down [H, I] pair, optionally one rank's slice of both.
std::vector<double> SwiGlu(const Mat& gate_up, const Mat& down, const std::vector<double>& x) {
  const std::vector<double> gu = Gemv(gate_up, x);
  const int64_t half = gate_up.N / 2;
  std::vector<double> a(static_cast<size_t>(half));
  for (int64_t i = 0; i < half; ++i) a[static_cast<size_t>(i)] = Silu(gu[static_cast<size_t>(i)]) * gu[static_cast<size_t>(half + i)];
  return Gemv(down, a);
}

void TestShardedAlgebra() {
  // A tiny LFM2-shaped config: 4 layers (conv dense, conv dense, attention MoE, conv MoE), hidden 16, 4 heads x 4, 2 kv,
  // dense I 24, experts 4 of 8, vocab 32.
  ModelConfig c;
  c.arch = r4dx::model::Arch::kLfm2Moe;
  c.hidden_size = 16;
  c.num_hidden_layers = 4;
  c.layer_types = {"conv", "conv", "full_attention", "conv"};
  c.num_attention_heads = 4;
  c.num_key_value_heads = 2;
  c.head_dim = 4;
  c.intermediate_size = 24;
  c.moe_intermediate_size = 8;
  c.num_experts = 4;
  c.num_dense_layers = 2;
  c.vocab_size = 32;
  const int64_t H = c.hidden_size;
  bool ok = true;

  // ShortConv: in_proj [3H, H] -> (B, C, x); h = B * x; conv over time (depthwise, causal, kernel 3); y = C * conv(h);
  // out_proj [H, H]. Sharded by channel: rank r holds channels [r*H/2, (r+1)*H/2) of each segment and of the taps.
  {
    const Mat in_proj(3 * H, H, 11), taps(H, 3, 12), out_proj(H, H, 13);
    const int T = 5;
    std::vector<std::vector<double>> xs(T, std::vector<double>(static_cast<size_t>(H)));
    for (int t = 0; t < T; ++t)
      for (int64_t k = 0; k < H; ++k) xs[static_cast<size_t>(t)][static_cast<size_t>(k)] = static_cast<double>((t * 7 + k * 3) % 5) - 2.0;
    const tp::ShardRule rin = tp::RuleFor(L(0, "conv.in_proj"), c), rtap = tp::RuleFor(L(0, "conv.weight"), c), rout = tp::RuleFor(L(0, "conv.out_proj"), c);

    // The conv core on `ch` channels, given the in_proj / taps / out_proj of those channels (unsharded: all H).
    auto run = [&](const Mat& in, const Mat& tp_taps, const Mat& out) {
      const int64_t ch = in.N / 3;
      std::vector<std::vector<double>> h(T, std::vector<double>(static_cast<size_t>(ch))), Cg(T, std::vector<double>(static_cast<size_t>(ch)));
      for (int t = 0; t < T; ++t) {
        const std::vector<double> p = Gemv(in, xs[static_cast<size_t>(t)]);
        for (int64_t j = 0; j < ch; ++j) {
          const double B = p[static_cast<size_t>(j)], Cc = p[static_cast<size_t>(ch + j)], X = p[static_cast<size_t>(2 * ch + j)];
          h[static_cast<size_t>(t)][static_cast<size_t>(j)] = B * X;
          Cg[static_cast<size_t>(t)][static_cast<size_t>(j)] = Cc;
        }
      }
      std::vector<std::vector<double>> ys(T);
      for (int t = 0; t < T; ++t) {
        std::vector<double> z(static_cast<size_t>(ch));
        for (int64_t j = 0; j < ch; ++j) {
          double acc = 0;
          for (int k = 0; k < 3; ++k) {  // tap k multiplies h[t - 2 + k] (zeros before the start)
            const int tt = t - 2 + k;
            if (tt >= 0) acc += tp_taps.At(j, k) * h[static_cast<size_t>(tt)][static_cast<size_t>(j)];
          }
          z[static_cast<size_t>(j)] = Cg[static_cast<size_t>(t)][static_cast<size_t>(j)] * acc;
        }
        ys[static_cast<size_t>(t)] = Gemv(out, z);  // out_proj over THIS rank's K slice: a partial sum
      }
      return ys;
    };
    const auto full = run(in_proj, taps, out_proj);
    const Mat in0 = SliceMat(in_proj, rin, 0), in1 = SliceMat(in_proj, rin, 1), t0 = SliceMat(taps, rtap, 0), t1 = SliceMat(taps, rtap, 1),
              o0 = SliceMat(out_proj, rout, 0), o1 = SliceMat(out_proj, rout, 1);
    ok = ok && in0.N == 3 * H / 2 && in1.N == 3 * H / 2 && t0.N == H / 2 && o0.K == H / 2 && o0.N == H;
    const auto p0 = run(in0, t0, o0), p1 = run(in1, t1, o1);
    for (int t = 0; t < T; ++t)
      for (int64_t k = 0; k < H; ++k)
        ok = ok && std::fabs(p0[static_cast<size_t>(t)][static_cast<size_t>(k)] + p1[static_cast<size_t>(t)][static_cast<size_t>(k)] -
                             full[static_cast<size_t>(t)][static_cast<size_t>(k)]) < 1e-9;
  }
  Check(ok, "D: ShortConv by channel: in_proj (B|C|x three-segment) + taps + out_proj partials add up to the unsharded conv layer");

  // Dense SwiGLU and one MoE expert: sharded gate_up / down partials add up.
  {
    const std::vector<double> x = {1, -2, 3, 0, 1, 2, -1, 1, 0, -3, 2, 1, 1, -1, 2, 0};
    struct Case {
      const char* what;
      std::string gu, dn;
      int64_t I;
    };
    for (const Case& cs : {Case{"dense", L(0, "mlp.gate_up"), L(0, "mlp.down"), c.intermediate_size},
                           Case{"expert 3", Ex(2, 3, "gate_up"), Ex(2, 3, "down"), c.moe_intermediate_size}}) {
      const Mat gate_up(2 * cs.I, H, 21), down(H, cs.I, 22);
      const tp::ShardRule rgu = tp::RuleFor(cs.gu, c), rdn = tp::RuleFor(cs.dn, c);
      const std::vector<double> full = SwiGlu(gate_up, down, x);
      const Mat g0 = SliceMat(gate_up, rgu, 0), g1 = SliceMat(gate_up, rgu, 1), d0 = SliceMat(down, rdn, 0), d1 = SliceMat(down, rdn, 1);
      bool part_ok = g0.N == cs.I && g1.N == cs.I && d0.K == cs.I / 2 && d1.K == cs.I / 2;
      const std::vector<double> p0 = SwiGlu(g0, d0, x), p1 = SwiGlu(g1, d1, x);
      for (int64_t k = 0; k < H; ++k) part_ok = part_ok && std::fabs(p0[static_cast<size_t>(k)] + p1[static_cast<size_t>(k)] - full[static_cast<size_t>(k)]) < 1e-9;
      Check(part_ok, std::string("D: ") + cs.what + " SwiGLU: the two ranks' gate_up (gate|up halves) / down partials add up to the unsharded FFN");
    }
  }

  // Attention: q heads and kv heads split in contiguous blocks keep the GQA grouping (each rank's q heads read its own kv heads).
  {
    const tp::ShardRule rq = tp::RuleFor(L(2, "attn.q"), c), rk = tp::RuleFor(L(2, "attn.k"), c), ro = tp::RuleFor(L(2, "attn.o"), c);
    bool attn_ok = true;
    const int64_t gqa = c.num_attention_heads / c.num_key_value_heads;
    for (int rank = 0; rank < 2; ++rank) {
      const tp::Range q = tp::RankRows(rq, 2, rank)[0], k = tp::RankRows(rk, 2, rank)[0], o = tp::RankCols(ro, 2, rank);
      const int64_t qh0 = q.begin / c.head_dim, qh1 = (q.begin + q.count) / c.head_dim, kh0 = k.begin / c.head_dim, kh1 = (k.begin + k.count) / c.head_dim;
      attn_ok = attn_ok && qh0 / gqa == kh0 && (qh1 - 1) / gqa == kh1 - 1 && (qh1 - qh0) == gqa * (kh1 - kh0) && o.begin == q.begin && o.count == q.count;
    }
    Check(attn_ok, "D: attention: rank r's q heads attend only its own kv heads (contiguous blocks, GQA kept) and out_proj K == its q rows");
  }
  // Head: the vocab slices concatenate.
  {
    const tp::ShardRule head = tp::RuleFor("lm_head", c);
    const Mat w(c.vocab_size, H, 31);
    const std::vector<double> x(static_cast<size_t>(H), 1.0);
    const std::vector<double> full = Gemv(w, x), a = Gemv(SliceMat(w, head, 0), x), b = Gemv(SliceMat(w, head, 1), x);
    bool head_ok = a.size() == 16 && b.size() == 16;
    for (size_t i = 0; head_ok && i < 16; ++i) head_ok = a[i] == full[i] && b[i] == full[16 + i];
    Check(head_ok, "D: the tied head's two vocab slices concatenate to the full logits");
  }
}

}  // namespace

int main() {
  TestRealRules();
  TestRefusals();
  TestPhysicalPlans();
  TestShardedAlgebra();
  std::printf("\n%s: %d failure(s)\n", g_failures == 0 ? "OK" : "FAILED", g_failures);
  return g_failures == 0 ? 0 : 1;
}
