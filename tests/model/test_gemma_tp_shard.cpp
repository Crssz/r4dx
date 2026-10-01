// tests/model/test_gemma_tp_shard.cpp -- CPU-only (no GPU, no container, no HIP). docs/gemma4-plan.md M1b-1.
//
// The Gemma 4 tensor-parallel sharding contract, on the real Huihui-gemma-4-12B config (values inlined so the test is
// hermetic) and on a tiny synthetic one:
//   A. tp::RuleFor's Gemma table for every tensor of every one of the 48 layers (sliding and full), the tied head, the
//      embedding, the norms, layer_scalar and the five rotation tensors: which split, which segments, which K;
//   B. consistency with GemmaConfig::Shard: the rows / K every rank's slice gets are exactly the per-rank shapes the
//      kernels are sized for (q 2048 / 4096, sliding k, v 1024, the full k REPLICATED at 512, o K 2048 / 4096, gate_up 7680
//      per segment, down K 7680, lm_head 131072 rows, the three Hadamard sign vectors);
//   C. the PHYSICAL plans (bf16, w4a16 wq / wsz at groups 64 and 128, trellis KB 4 / 5): each rank's runs are whole
//      128-blocks / 16-row tiles, and the two ranks' byte runs partition the full part exactly (no gap, no overlap);
//   D. the algebra of TP=2 on a tiny model with real packed bytes: column-parallel q / gate_up outputs concatenate, the
//      row-parallel o / down partial sums add up to the unsharded product, with the GeGLU applied per rank on its own
//      intermediate channels -- i.e. the sharded forward IS the unsharded one up to the all-reduce's addition order;
//   E. refusals: unknown names, a v_proj on a full layer, a GDN / Qwen tensor on a Gemma config, a layer out of range, a
//      still-suffixed name, and an already-sharded config.
// What it cannot show (GPU, `gemma_tp_identity.ps1` / M1b-2): that the kernels and the all-reduce agree with it.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "gemma_config.h"
#include "model_config.h"
#include "nlohmann/json.hpp"
#include "rotation_meta.h"
#include "tp/tp_shard.h"

using nlohmann::json;
using r4dx::model::GemmaConfig;
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

// Huihui-gemma-4-12B-it-abliterated config.json "text_config" (same values as test_gemma_config.cpp).
json RealTextConfig() {
  json t;
  t["attention_bias"] = false;
  t["attention_k_eq_v"] = true;
  t["bos_token_id"] = 2;
  t["enable_moe_block"] = false;
  t["final_logit_softcapping"] = 30.0;
  t["global_head_dim"] = 512;
  t["head_dim"] = 256;
  t["hidden_activation"] = "gelu_pytorch_tanh";
  t["hidden_size"] = 3840;
  t["hidden_size_per_layer_input"] = 0;
  t["intermediate_size"] = 15360;
  json types = json::array();
  for (int i = 0; i < 48; ++i) types.push_back((i + 1) % 6 == 0 ? "full_attention" : "sliding_attention");
  t["layer_types"] = types;
  t["max_position_embeddings"] = 131072;
  t["model_type"] = "gemma4_unified_text";
  t["num_attention_heads"] = 16;
  t["num_global_key_value_heads"] = 1;
  t["num_hidden_layers"] = 48;
  t["num_key_value_heads"] = 8;
  t["num_kv_shared_layers"] = 0;
  t["pad_token_id"] = 0;
  t["rms_norm_eps"] = 1e-06;
  t["rope_parameters"] = {
      {"full_attention", {{"partial_rotary_factor", 0.25}, {"rope_theta", 1000000.0}, {"rope_type", "proportional"}}},
      {"sliding_attention", {{"rope_theta", 10000.0}, {"rope_type", "default"}}}};
  t["sliding_window"] = 1024;
  t["tie_word_embeddings"] = true;
  t["use_bidirectional_attention"] = "vision";
  t["use_double_wide_mlp"] = false;
  t["vocab_size"] = 262144;
  return t;
}

std::string L(int i, const char* tail) { return "text.layers." + std::to_string(i) + "." + tail; }

// Rows / K a rank's slice of `base` has, from the rule alone (replicated: the full N, K passed in).
struct Slice {
  tp::Split split;
  int64_t rows = 0;      // kRows: sum of the rank's row ranges; else the full N
  int64_t k = 0;         // kCols: the rank's K; else the full K
  int64_t first_row = 0;  // kRows: first row of the first range
  int64_t first_k = 0;    // kCols: first column
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

void TestRealRules() {
  const GemmaConfig g = GemmaConfig::FromJson(RealTextConfig());
  const ModelConfig gm = g.ToModelConfig();
  const GemmaConfig r0 = GemmaConfig::Shard(g, 2, 0), r1 = GemmaConfig::Shard(g, 2, 1);
  const int64_t H = g.hidden_size;
  bool all = true;
  int tensors = 0;
  for (int i = 0; i < 48; ++i) {
    const bool full = g.IsFullLayer(i);
    const int64_t qd = g.QDim(i), kvd = g.KvDim(i);
    const tp::ShardRule q = tp::RuleFor(L(i, "attn.q"), gm), k = tp::RuleFor(L(i, "attn.k"), gm),
                        o = tp::RuleFor(L(i, "attn.o"), gm), gu = tp::RuleFor(L(i, "mlp.gate_up"), gm),
                        dn = tp::RuleFor(L(i, "mlp.down"), gm), kd = tp::RuleFor(L(i, "attn.k_descale"), gm),
                        vd = tp::RuleFor(L(i, "attn.v_descale"), gm);
    tensors += 7;
    // q: column-parallel, one segment, heads split 8 / 8.
    all = all && q.split == tp::Split::kRows && q.segments.size() == 1 && q.segments[0].rows == qd;
    for (int rk = 0; rk < 2; ++rk) {
      const Slice s = SliceOf(q, rk, qd, H);
      all = all && s.rows == qd / 2 && s.first_row == rk * qd / 2 && s.rows == (rk == 0 ? r0 : r1).QDim(i);
    }
    // k (and v): sliding split 4 / 4 heads; the full layer's single kv head replicates.
    if (full) {
      all = all && k.split == tp::Split::kReplicate && kd.split == tp::Split::kReplicate &&
            vd.split == tp::Split::kReplicate;
      all = all && Throws([&] { (void)tp::RuleFor(L(i, "attn.v"), gm); }, "k_eq_v");
    } else {
      const tp::ShardRule v = tp::RuleFor(L(i, "attn.v"), gm);
      ++tensors;
      all = all && k.split == tp::Split::kRows && v.split == tp::Split::kRows && kd.split == tp::Split::kRows &&
            vd.split == tp::Split::kRows;
      for (int rk = 0; rk < 2; ++rk) {
        const Slice sk = SliceOf(k, rk, kvd, H), sv = SliceOf(v, rk, kvd, H), sd = SliceOf(kd, rk, 8, 1);
        all = all && sk.rows == (rk == 0 ? r0 : r1).KvDim(i) && sv.rows == sk.rows && sd.rows == 4 &&
              sk.rows == kvd / 2 && sk.first_row == rk * kvd / 2;
      }
    }
    // o: row-parallel, K = the rank's heads * head_dim.
    all = all && o.split == tp::Split::kCols && o.k_total == qd;
    for (int rk = 0; rk < 2; ++rk) {
      const Slice s = SliceOf(o, rk, H, qd);
      all = all && s.k == (rk == 0 ? r0 : r1).QDim(i) && s.first_k == rk * qd / 2;
    }
    // gate_up: two segments (gate | up), each split 7680 / 7680; down: K 7680 per rank.
    all = all && gu.split == tp::Split::kRows && gu.segments.size() == 2 && gu.segments[0].rows == 15360 &&
          gu.segments[1].begin == 15360;
    for (int rk = 0; rk < 2; ++rk) {
      const std::vector<tp::Range> rr = tp::RankRows(gu, 2, rk);
      all = all && rr.size() == 2 && rr[0].begin == rk * 7680 && rr[0].count == 7680 && rr[1].begin == 15360 + rk * 7680 &&
            rr[1].count == 7680;
      const tp::Range c = tp::RankCols(dn, 2, rk);
      all = all && c.begin == rk * 7680 && c.count == 7680 && c.count == (rk == 0 ? r0 : r1).intermediate_size;
    }
    // norms, scalars, q/k norm weights replicate on every layer.
    for (const char* n : {"input_layernorm", "input_layernorm.rotated", "post_attention_layernorm",
                          "pre_feedforward_layernorm", "pre_feedforward_layernorm.rotated",
                          "post_feedforward_layernorm", "layer_scalar", "attn.q_norm", "attn.k_norm"}) {
      all = all && tp::RuleFor(L(i, n), gm).split == tp::Split::kReplicate;
      ++tensors;
    }
  }
  Check(all, "A/B: every Gemma layer tensor's rule and per-rank slice match GemmaConfig::Shard (" +
                 std::to_string(tensors) + " tensors, 48 layers)");

  // Embedding, head, final norm.
  const tp::ShardRule emb = tp::RuleFor("text.embed_tokens", gm), head = tp::RuleFor("lm_head", gm);
  Check(emb.split == tp::Split::kReplicate && tp::RuleFor("text.final_norm", gm).split == tp::Split::kReplicate,
        "A: embed_tokens and final_norm replicate");
  const std::vector<tp::Range> h0 = tp::RankRows(head, 2, 0), h1 = tp::RankRows(head, 2, 1);
  Check(head.split == tp::Split::kRows && h0.size() == 1 && h0[0].begin == 0 && h0[0].count == 131072 &&
            h1[0].begin == 131072 && h1[0].count == 131072 && r0.VocabShardSize() == 131072 && r1.VocabShardBegin() == 131072,
        "A/B: the tied head is a vocab slice: rank r = rows [r*131072, (r+1)*131072)");

  // Rotation tensors (option A, q2ab with o_full): signs / mix replicate; the Hadamard sign vectors split with their K.
  r4dx::model::RotationSpec spec;
  spec.kind = r4dx::model::RotationKind::kQ2ab;
  spec.hidden = 3840;
  spec.block = 256;
  spec.nblk = 15;
  spec.post_norm_rotate = true;
  spec.has_gdn_out = false;
  spec.has_o_full = true;
  const int64_t o_full_global = g.num_attention_heads * g.head_dim_full;
  const int64_t o_full_local = r0.num_attention_heads * r0.head_dim_full;
  const auto want_g = r4dx::model::RotationTensors(spec, gm, o_full_global);
  const auto want_0 = r4dx::model::RotationTensors(spec, r0.ToModelConfig(), o_full_local);
  const auto want_1 = r4dx::model::RotationTensors(spec, r1.ToModelConfig(), o_full_local);
  bool rot_ok = want_g.size() == 5 && want_0.size() == 5 && want_1.size() == 5;  // signs, mix, had_down, had_o, had_o_full
  for (size_t t = 0; rot_ok && t < want_g.size(); ++t) {
    const std::string name = want_g[t].name;
    const tp::ShardRule rr = tp::RuleFor(name, gm);
    for (int rk = 0; rk < 2; ++rk) {
      const int64_t want = (rk == 0 ? want_0 : want_1)[t].elems;
      int64_t got;
      if (rr.split == tp::Split::kReplicate) {
        got = want_g[t].elems;
      } else {
        got = 0;
        for (const tp::Range& rg : tp::RankRows(rr, 2, rk)) got += rg.count;
      }
      rot_ok = rot_ok && got == want;
    }
  }
  Check(rot_ok, "A/B: rotation tensors: signs/mix replicate, had_down 7680 / had_o 2048 / had_o_full 4096 per rank");
  // A Hadamard sign vector's rank range is exactly its linear's K range (the head / intermediate-channel blocks line up).
  const tp::Range down_k = tp::RankCols(tp::RuleFor(L(0, "mlp.down"), gm), 2, 1);
  const tp::Range down_s = tp::RankRows(tp::RuleFor("rotation.had_down_signs", gm), 2, 1)[0];
  const tp::Range o_k = tp::RankCols(tp::RuleFor(L(5, "attn.o"), gm), 2, 1);
  const tp::Range o_s = tp::RankRows(tp::RuleFor("rotation.had_o_full_signs", gm), 2, 1)[0];
  const tp::Range os_k = tp::RankCols(tp::RuleFor(L(0, "attn.o"), gm), 2, 1);
  const tp::Range os_s = tp::RankRows(tp::RuleFor("rotation.had_o_signs", gm), 2, 1)[0];
  Check(down_k.begin == down_s.begin && down_k.count == down_s.count && o_k.begin == o_s.begin && o_k.count == o_s.count &&
            os_k.begin == os_s.begin && os_k.count == os_s.count,
        "A: each had_*_signs rank range equals its linear's K range (down, o sliding, o full)");
  Check(down_k.count % 512 == 0 && o_k.count % 256 == 0 && os_k.count % 256 == 0,
        "B: the rank K ranges hold whole Hadamard blocks (512 / 256)");
}

void TestRefusals() {
  const GemmaConfig g = GemmaConfig::FromJson(RealTextConfig());
  const ModelConfig gm = g.ToModelConfig();
  Check(Throws([&] { (void)tp::RuleFor("text.layers.48.attn.q", gm); }, "layer 48"), "E: layer 48 of a 48-layer model");
  Check(Throws([&] { (void)tp::RuleFor("text.layers.0.attn.qg", gm); }, "unknown tensor"),
        "E: a Qwen name (attn.qg) on a Gemma config is refused, not replicated");
  Check(Throws([&] { (void)tp::RuleFor("text.layers.0.gdn.out_proj", gm); }, "unknown tensor"), "E: a GDN tensor is refused");
  Check(Throws([&] { (void)tp::RuleFor("text.layers.0.attn.q.bf16.w", gm); }, "unknown tensor"),
        "E: a still-suffixed name is refused");
  Check(Throws([&] { (void)tp::RuleFor("text.layers.0.attn.o_norm", gm); }, "unknown tensor"), "E: an unknown suffix is refused");
  // A Qwen-shaped config is untouched by the Gemma table: attn.qg is still the q|gate rule there.
  ModelConfig q;
  q.arch = r4dx::model::Arch::kQwen35;
  q.num_hidden_layers = 4;
  q.layer_types = {"linear_attention", "linear_attention", "linear_attention", "full_attention"};
  q.num_attention_heads = 8;
  q.head_dim = 16;
  q.num_key_value_heads = 2;
  q.intermediate_size = 64;
  q.vocab_size = 64;
  Check(tp::RuleFor("text.layers.3.attn.qg", q).split == tp::Split::kRows, "E: Qwen's attn.qg rule is unchanged");
  Check(Throws([&] { (void)tp::RuleFor("text.layers.3.attn.q", q); }, "unknown tensor"), "E: Qwen has no attn.q");
  // GemmaConfig::Shard refusals that guard the plans above.
  Check(Throws([&] { (void)GemmaConfig::Shard(GemmaConfig::Shard(g, 2, 0), 2, 0); }, "already a rank shard"),
        "E: a shard of a shard is refused");
  Check(Throws([&] { (void)tp::RuleFor("text.layers.0.attn.q", GemmaConfig::Shard(g, 2, 0).ToModelConfig()); }, "unsharded"),
        "E: RuleFor needs the UNSHARDED config");
}

// ---- C: physical plans ----------------------------------------------------------------------------------------------

// The byte runs of the two ranks partition [0, total): sorted, no overlap, no gap.
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
  bool rows;  // column-parallel (kRows) or row-parallel (kCols)
};

void TestPhysicalPlans() {
  const GemmaConfig g = GemmaConfig::FromJson(RealTextConfig());
  const ModelConfig gm = g.ToModelConfig();
  const int64_t H = g.hidden_size;
  std::vector<Lin> lins;
  for (int i : {0, 5}) {  // one sliding, one full layer: every distinct shape
    lins.push_back({L(i, "attn.q"), g.QDim(i), H, true});
    if (!g.IsFullLayer(i)) {
      lins.push_back({L(i, "attn.k"), g.KvDim(i), H, true});
      lins.push_back({L(i, "attn.v"), g.KvDim(i), H, true});
    }
    lins.push_back({L(i, "attn.o"), H, g.QDim(i), false});
    lins.push_back({L(i, "mlp.gate_up"), 2 * g.intermediate_size, H, true});
    lins.push_back({L(i, "mlp.down"), H, g.intermediate_size, false});
  }
  lins.push_back({"lm_head", g.vocab_size, H, true});
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
                            {tp::Part::kW4a16Wsz, 64, 0, static_cast<size_t>(l.N * (l.K / 64) * 4)},
                            {tp::Part::kW4a16Wsz, 128, 0, static_cast<size_t>(l.N * (l.K / 128) * 4)},
                            {tp::Part::kTrellisW, 0, 4, static_cast<size_t>(l.N * l.K * 4 / 8)},
                            {tp::Part::kTrellisW, 0, 5, static_cast<size_t>(l.N * l.K * 5 / 8)}};
    if (l.base == "lm_head") parts.resize(4);  // heads are never trellis (bf16 / w4a16 only)
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
        std::fprintf(stderr, "  plan %s part %d group %d rate %d threw: %s\n", l.base.c_str(), static_cast<int>(p.part),
                     p.group, p.rate, e.what());
        ok = false;
        continue;
      }
      size_t ba = 0, bb = 0;
      for (const auto& r : a) ba += r.bytes;
      for (const auto& r : b) bb += r.bytes;
      const bool part_ok = ba == bb && ba == p.total / 2 && Partitions(a, b, p.total);
      if (!part_ok) std::fprintf(stderr, "  plan %s part %d group %d rate %d: bytes %zu / %zu of %zu\n", l.base.c_str(),
                                 static_cast<int>(p.part), p.group, p.rate, ba, bb, p.total);
      ok = ok && part_ok;
      ++plans;
    }
  }
  Check(ok, "C: bf16 / w4a16 (g64, g128) / trellis (KB4, KB5) plans of every Gemma linear: the two ranks partition the part (" +
                std::to_string(plans) + " plans)");
  // The full layers' k_proj (512 x 3840) is replicated: both ranks load it whole.
  const tp::ShardRule kf = tp::RuleFor(L(5, "attn.k"), gm);
  Check(kf.split == tp::Split::kReplicate, "C: the full layer's k_proj loads whole on both ranks");
  // Trellis alignment of every rank dimension (docs/trellis-kernel.md 2.4): rows and K in whole 128-blocks.
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
  Check(align, "C: every rank row / K range is a whole number of 128-blocks");
}

// ---- D: the algebra of the sharded forward on a tiny model -----------------------------------------------------------

uint16_t Bf16Of(int v) {  // a small integer is exact in bf16
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
      const uint16_t v = Bf16Of(static_cast<int>((s >> 24) % 9) - 4);
      std::memcpy(&bytes[static_cast<size_t>(i) * 2], &v, 2);
    }
  }
  double At(int64_t n, int64_t k) const {
    uint16_t v;
    std::memcpy(&v, &bytes[static_cast<size_t>(n * K + k) * 2], 2);
    return FloatOf(v);
  }
};
// A rank's slice of `m` by `rule` as a Mat.
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
std::vector<double> Gemv(const Mat& w, const std::vector<double>& x) {  // y = W x
  std::vector<double> y(static_cast<size_t>(w.N), 0.0);
  for (int64_t n = 0; n < w.N; ++n) {
    double a = 0;
    for (int64_t k = 0; k < w.K; ++k) a += w.At(n, k) * x[static_cast<size_t>(k)];
    y[static_cast<size_t>(n)] = a;
  }
  return y;
}
double GeluTanh(double x) {
  return 0.5 * x * (1.0 + std::tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)));
}

void TestShardedAlgebra() {
  // A tiny Gemma-shaped config: 6 layers (full at 5), 4 heads, sliding kv 2 x hd 8, full hd 16 x 1 kv head, I = 24.
  ModelConfig c;
  c.arch = r4dx::model::Arch::kGemma4;
  c.hidden_size = 16;
  c.num_hidden_layers = 6;
  for (int i = 0; i < 6; ++i) c.layer_types.push_back(i == 5 ? "full_attention" : "sliding_attention");
  c.num_attention_heads = 4;
  c.num_key_value_heads = 2;
  c.head_dim = 8;
  c.global_head_dim = 16;
  c.intermediate_size = 24;
  c.vocab_size = 32;
  const int64_t H = c.hidden_size;
  bool ok = true;
  for (int layer : {0, 5}) {
    const bool full = layer == 5;
    const int64_t hd = full ? c.global_head_dim : c.head_dim;
    const int64_t qd = c.num_attention_heads * hd, kvd = (full ? 1 : c.num_key_value_heads) * hd;
    const Mat wq(qd, H, 1), wk(kvd, H, 2), wo(H, qd, 3), wgu(2 * c.intermediate_size, H, 4), wdn(H, c.intermediate_size, 5);
    std::vector<double> x(static_cast<size_t>(H));
    for (int64_t i = 0; i < H; ++i) x[static_cast<size_t>(i)] = static_cast<double>((i * 7) % 5) - 2.0;

    // q: the ranks' outputs concatenate to the full q (heads 0..1 | 2..3).
    const std::vector<double> q_full = Gemv(wq, x);
    const tp::ShardRule rq = tp::RuleFor(L(layer, "attn.q"), c);
    std::vector<double> q_cat;
    for (int rk = 0; rk < 2; ++rk) {
      const std::vector<double> y = Gemv(SliceMat(wq, rq, rk), x);
      q_cat.insert(q_cat.end(), y.begin(), y.end());
    }
    ok = ok && q_cat == q_full;
    // k: sliding concatenates (kv head 0 | 1); the full layer's single head is the same bytes on both ranks.
    const tp::ShardRule rk_ = tp::RuleFor(L(layer, "attn.k"), c);
    const std::vector<double> k_full = Gemv(wk, x);
    if (full) {
      ok = ok && rk_.split == tp::Split::kReplicate;
    } else {
      std::vector<double> k_cat;
      for (int rk = 0; rk < 2; ++rk) {
        const std::vector<double> y = Gemv(SliceMat(wk, rk_, rk), x);
        k_cat.insert(k_cat.end(), y.begin(), y.end());
      }
      ok = ok && k_cat == k_full;
    }
    // o: partial sums over the rank's heads add up to the full o_proj of the full attention output.
    std::vector<double> attn(static_cast<size_t>(qd));
    for (int64_t i = 0; i < qd; ++i) attn[static_cast<size_t>(i)] = static_cast<double>((i * 3) % 7) - 3.0;
    const std::vector<double> o_full = Gemv(wo, attn);
    const tp::ShardRule ro = tp::RuleFor(L(layer, "attn.o"), c);
    std::vector<double> o_sum(static_cast<size_t>(H), 0.0);
    for (int rk = 0; rk < 2; ++rk) {
      const tp::Range kr = tp::RankCols(ro, 2, rk);
      const std::vector<double> xs(attn.begin() + kr.begin, attn.begin() + kr.begin + kr.count);
      const std::vector<double> y = Gemv(SliceMat(wo, ro, rk), xs);
      for (size_t i = 0; i < y.size(); ++i) o_sum[i] += y[i];
    }
    ok = ok && o_sum == o_full;
    // MLP: gate_up column-parallel (gate_r | up_r), GeGLU on the rank's own channels, down row-parallel; the sum of the
    // ranks' down outputs equals the unsharded MLP.
    const int64_t I = c.intermediate_size;
    const std::vector<double> gu = Gemv(wgu, x);
    std::vector<double> h_full(static_cast<size_t>(I));
    for (int64_t i = 0; i < I; ++i) h_full[static_cast<size_t>(i)] = GeluTanh(gu[static_cast<size_t>(i)]) * gu[static_cast<size_t>(I + i)];
    const std::vector<double> mlp_full = Gemv(wdn, h_full);
    const tp::ShardRule rg = tp::RuleFor(L(layer, "mlp.gate_up"), c), rd = tp::RuleFor(L(layer, "mlp.down"), c);
    std::vector<double> mlp_sum(static_cast<size_t>(H), 0.0);
    for (int rk = 0; rk < 2; ++rk) {
      const std::vector<double> y = Gemv(SliceMat(wgu, rg, rk), x);  // [gate_r (I/2) | up_r (I/2)]
      const int64_t half = I / 2;
      std::vector<double> h(static_cast<size_t>(half));
      for (int64_t i = 0; i < half; ++i) h[static_cast<size_t>(i)] = GeluTanh(y[static_cast<size_t>(i)]) * y[static_cast<size_t>(half + i)];
      const std::vector<double> d = Gemv(SliceMat(wdn, rd, rk), h);
      for (size_t i = 0; i < d.size(); ++i) mlp_sum[i] += d[i];
    }
    for (size_t i = 0; i < mlp_sum.size(); ++i) ok = ok && std::fabs(mlp_sum[i] - mlp_full[i]) <= 1e-9 * (1.0 + std::fabs(mlp_full[i]));
  }
  // The vocab-split head: concatenated shard logits equal the full row, global ids = rank * V/2 + local id.
  const Mat head(c.vocab_size, H, 9);
  std::vector<double> x(static_cast<size_t>(H));
  for (int64_t i = 0; i < H; ++i) x[static_cast<size_t>(i)] = static_cast<double>((i * 5) % 3) - 1.0;
  const std::vector<double> full = Gemv(head, x);
  const tp::ShardRule rh = tp::RuleFor("lm_head", c);
  std::vector<double> cat;
  for (int rk = 0; rk < 2; ++rk) {
    const std::vector<double> y = Gemv(SliceMat(head, rh, rk), x);
    cat.insert(cat.end(), y.begin(), y.end());
  }
  ok = ok && cat == full;
  Check(ok, "D: tiny Gemma (sliding + full layer): q / k / lm_head concatenate, o and GeGLU-MLP partial sums add up exactly");
}

}  // namespace

int main() {
  TestRealRules();
  TestRefusals();
  TestPhysicalPlans();
  TestShardedAlgebra();
  if (g_failures != 0) {
    std::fprintf(stderr, "%d check(s) FAILED\n", g_failures);
    return 1;
  }
  std::printf("test_gemma_tp_shard: all checks passed\n");
  return 0;
}
