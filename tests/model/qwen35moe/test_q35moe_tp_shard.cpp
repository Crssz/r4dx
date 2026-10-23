// tests/model/qwen35moe/test_q35moe_tp_shard.cpp -- CPU-only (no GPU, no HIP call). docs/qwen35moe-tp-design.md P0 / gate G1.
//
// The --tp 2 sharding contract of Ornith-1.5-122B (qwen3_5_moe) before any rank code exists:
//   A. Qwen35MoeConfig::Shard on the real text_config (values inlined, hermetic): heads 32 -> 16, kv 2 -> 1 (GQA 16 kept), GDN
//      key / value heads 16 / 64 -> 8 / 32 (ratio 4 kept), moe 1024 -> 512, shared 1024 -> 512, vocab stays global; every
//      refusal (world 3, rank 2, a shard of a shard, a kv-head count that does not halve, a 64-wide expert that breaks the
//      128-block rule, ...) names its field;
//   B. tp::RuleFor's qwen3_5_moe table for every tensor of every one of the 48 layers x 256 experts, against the shapes
//      Shard() gives the kernels (the docs table 2.2), and its refusals (mlp.* on an MoE layer, a GDN tensor on an attention
//      layer, expert 256, mtp.* / rotation.*, a still-suffixed name, an already-sharded config);
//   C. the byte plans of the trellis classes at KB 3 / 4 / 5 on synthetic bytes: the two ranks' word runs partition the part
//      exactly, the expert stack strides are w_per_rank, sign vectors split as documented;
//   D. [real container, SKIP when absent] G1 on E:\models\r4dx\ornith-1.5-122b-trellis-mix-kv.r4dx (read-only mmap):
//        D1 every tensor name of the container classifies (an unclassified name throws) and the per-rank bytes are summed;
//        D2 for one GDN layer, one attention layer and several experts of KB 3 / 4 / 5 layers: decode(Gather(plan(rank))) ==
//           slice(decode(full)) bit for bit for every trellis linear of the table, both ranks (trellis_ref's CPU decoder);
//        D3 the linear around the decode: column-parallel ranks reproduce their output rows bit for bit (fp64 LinearExact),
//           row-parallel ranks' partials add up to the full output;
//        D4 the bf16 shared expert and the small raw tensors (conv1d_weight, in_proj_a, descales) slice as the rules say.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "../../kernels/trellis_ref.hpp"
#include "nlohmann/json.hpp"
#include "qwen35moe_config.h"
#include "qwen35moe_tp_shard.h"
#include "r4dx/core/dtype.hpp"
#include "r4dx/models_root.h"
#include "r4dx_convert/safetensors_reader.hpp"
#include "tp/tp_shard.h"
#include "trellis_meta.h"

using nlohmann::json;
using r4dx::model::ModelConfig;
using r4dx::model::Qwen35MoeConfig;
namespace tp = r4dx::model::tp;
namespace q35tp = r4dx::model::qwen35moe_tp;

namespace {

int g_failures = 0;
int g_skipped = 0;

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

json TextConfig() {
  json layers = json::array();
  for (int i = 0; i < 48; ++i) layers.push_back(i % 4 == 3 ? "full_attention" : "linear_attention");
  return json{{"attn_output_gate", true},
              {"eos_token_id", 248044},
              {"head_dim", 256},
              {"hidden_size", 3072},
              {"layer_types", layers},
              {"linear_conv_kernel_dim", 4},
              {"linear_key_head_dim", 128},
              {"linear_num_key_heads", 16},
              {"linear_num_value_heads", 64},
              {"linear_value_head_dim", 128},
              {"max_position_embeddings", 262144},
              {"mlp_only_layers", json::array()},
              {"model_type", "qwen3_5_moe_text"},
              {"moe_intermediate_size", 1024},
              {"norm_topk_prob", true},
              {"num_attention_heads", 32},
              {"num_experts", 256},
              {"num_experts_per_tok", 8},
              {"num_hidden_layers", 48},
              {"num_key_value_heads", 2},
              {"partial_rotary_factor", 0.25},
              {"rms_norm_eps", 1e-6},
              {"rope_parameters",
               {{"mrope_interleaved", true},
                {"mrope_section", {11, 11, 10}},
                {"partial_rotary_factor", 0.25},
                {"rope_theta", 10000000.0},
                {"rope_type", "default"}}},
              {"shared_expert_intermediate_size", 1024},
              {"tie_word_embeddings", false},
              {"vocab_size", 248320}};
}

std::string L(int i, const std::string& tail) { return "text.layers." + std::to_string(i) + "." + tail; }
std::string Ex(int i, int e, const char* part) { return L(i, "moe.experts." + std::to_string(e) + "." + part); }

struct Slice {
  tp::Split split = tp::Split::kReplicate;
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

// ---- A ----------------------------------------------------------------------------------------------------------------

void TestShardConfig() {
  const Qwen35MoeConfig g = Qwen35MoeConfig::FromJson(TextConfig());
  const Qwen35MoeConfig r0 = Qwen35MoeConfig::Shard(g, 2, 0), r1 = Qwen35MoeConfig::Shard(g, 2, 1);
  for (const Qwen35MoeConfig* r : {&r0, &r1}) {
    Check(r->tp_world == 2 && r->IsShard() && r->tp_rank == (r == &r0 ? 0 : 1), "Shard: tp_world / tp_rank set");
    Check(r->num_attention_heads == 16 && r->num_key_value_heads == 1 && r->AttnGqa() == 16 && g.AttnGqa() == 16,
          "Shard: 16 q heads, ONE kv head, GQA 16 kept");
    Check(r->linear_num_key_heads == 8 && r->linear_num_value_heads == 32 && r->GqaRepeats() == 4,
          "Shard: GDN key / value heads 8 / 32, ratio 4 kept");
    Check(r->KeyDim() == 1024 && r->ValueDim() == 4096 && r->ConvDim() == 6144, "Shard: key dim 1024, value dim 4096, conv channels 6144");
    Check(r->moe_intermediate_size == 512 && r->shared_expert_intermediate_size == 512, "Shard: expert and shared width 512");
    Check(r->hidden_size == 3072 && r->num_experts == 256 && r->num_experts_per_tok == 8 && r->vocab_size == 248320,
          "Shard: hidden, experts, top-k and the GLOBAL vocab unchanged");
    Check(r->VocabShardSize() == 124160 && r->VocabShardSize() % 16 == 0, "Shard: vocab slice 124160 = 7760 tiles");
  }
  Check(r1.VocabShardBegin() == 124160 && r0.VocabShardBegin() == 0, "Shard: vocab slice begin");
  const Qwen35MoeConfig w1 = Qwen35MoeConfig::Shard(g, 1, 0);
  Check(w1.tp_world == 1 && w1.num_attention_heads == 32 && w1.moe_intermediate_size == 1024 && !w1.IsShard(),
        "Shard at world 1 accepts the config and changes nothing");
  Check(r0.layer_types == g.layer_types && r0.NumGdnLayers() == 36 && r0.NumAttnLayers() == 12, "Shard: layer pattern unchanged");

  Check(Throws([&] { (void)Qwen35MoeConfig::Shard(g, 3, 0); }, "world must be 1 or 2"), "refusal: world 3");
  Check(Throws([&] { (void)Qwen35MoeConfig::Shard(g, 2, 2); }, "rank must be in [0, world)"), "refusal: rank 2 of 2");
  Check(Throws([&] { (void)Qwen35MoeConfig::Shard(r0, 2, 0); }, "already a rank shard"), "refusal: a shard of a shard");
  auto bad = [&](const char* key, int64_t v, const char* expect, const std::string& what) {
    json t = TextConfig();
    t[key] = v;
    // keep FromJson happy where a ratio must hold; a refusal from FromJson is its own failure and shows as a mismatch.
    Check(Throws([&] { (void)Qwen35MoeConfig::Shard(Qwen35MoeConfig::FromJson(t), 2, 0); }, expect), what);
  };
  bad("num_key_value_heads", 1, "num_key_value_heads", "refusal: a single kv head does not halve");
  bad("moe_intermediate_size", 192, "expert", "refusal: 192-wide experts (96 per rank) break the 128-block rule");
  bad("moe_intermediate_size", 1030, "moe_intermediate_size", "refusal: odd expert width");
  bad("shared_expert_intermediate_size", 1030, "shared_expert_intermediate_size", "refusal: odd shared width");
  bad("vocab_size", 248328, "vocab_size", "refusal: vocab not a whole number of 16-row tiles per rank");
  bad("linear_num_value_heads", 62, "linear_num_value_heads", "refusal: GDN value heads that change the ratio / do not divide into the key heads");
}

// ---- B ----------------------------------------------------------------------------------------------------------------

void TestRules() {
  const Qwen35MoeConfig g = Qwen35MoeConfig::FromJson(TextConfig());
  const ModelConfig gm = g.ToModelConfig();
  const Qwen35MoeConfig r0 = Qwen35MoeConfig::Shard(g, 2, 0), r1 = Qwen35MoeConfig::Shard(g, 2, 1);
  const int64_t H = g.hidden_size;
  const int64_t kv = g.num_key_value_heads * g.head_dim, q = g.num_attention_heads * g.head_dim;
  bool all = true;
  int tensors = 0;
  auto expect = [&](bool cond, const std::string& what) {
    if (!cond) std::fprintf(stderr, "  rule mismatch: %s\n", what.c_str());
    all = all && cond;
  };
  for (int i = 0; i < 48; ++i) {
    for (const char* n : {"input_layernorm", "post_attention_layernorm"}) {
      expect(tp::RuleFor(L(i, n), gm).split == tp::Split::kReplicate, L(i, n));
      ++tensors;
    }
    if (g.IsGdnLayer(i)) {
      const tp::ShardRule qkv = tp::RuleFor(L(i, "gdn.in_proj_qkv"), gm), cv = tp::RuleFor(L(i, "gdn.conv1d_weight"), gm),
                          z = tp::RuleFor(L(i, "gdn.in_proj_z"), gm), out = tp::RuleFor(L(i, "gdn.out_proj"), gm),
                          a = tp::RuleFor(L(i, "gdn.in_proj_a"), gm), b = tp::RuleFor(L(i, "gdn.in_proj_b"), gm),
                          alog = tp::RuleFor(L(i, "gdn.A_log"), gm), dt = tp::RuleFor(L(i, "gdn.dt_bias"), gm),
                          nw = tp::RuleFor(L(i, "gdn.norm_weight"), gm);
      tensors += 9;
      expect(qkv.split == tp::Split::kRows && qkv.segments.size() == 3 && qkv.segments[0].rows == 2048 && qkv.segments[1].begin == 2048 &&
                 qkv.segments[2].begin == 4096 && qkv.segments[2].rows == 8192,
             L(i, "in_proj_qkv segments q | k | v"));
      expect(cv.split == tp::Split::kRows && cv.segments.size() == 3 && cv.segments[2].rows == 8192, L(i, "conv1d_weight segments"));
      expect(z.split == tp::Split::kRows && z.segments.size() == 1 && z.segments[0].rows == 8192, L(i, "in_proj_z"));
      expect(out.split == tp::Split::kCols && out.k_total == 8192, L(i, "out_proj"));
      expect(a.split == tp::Split::kRows && b.split == tp::Split::kRows && alog.split == tp::Split::kRows && dt.split == tp::Split::kRows &&
                 a.segments.size() == 1 && a.segments[0].rows == 64,
             L(i, "in_proj_a/b, A_log, dt_bias rows {64}"));
      expect(nw.split == tp::Split::kReplicate, L(i, "norm_weight"));
      for (int rk = 0; rk < 2; ++rk) {
        const Qwen35MoeConfig& r = rk == 0 ? r0 : r1;
        const Slice sq = SliceOf(qkv, rk, g.ConvDim(), H), sz = SliceOf(z, rk, g.ValueDim(), H), so = SliceOf(out, rk, H, g.ValueDim()),
                    sa = SliceOf(a, rk, g.linear_num_value_heads, H);
        expect(sq.rows == r.ConvDim() && sq.ranges.size() == 3 && sq.ranges[0].begin == rk * 1024 && sq.ranges[1].begin == 2048 + rk * 1024 &&
                   sq.ranges[2].begin == 4096 + rk * 4096 && sq.ranges[2].count == 4096,
               "in_proj_qkv rank ranges");
        expect(sz.rows == r.ValueDim() && sz.first_row == rk * 4096, "in_proj_z rank rows");
        expect(so.k == r.ValueDim() && so.first_k == rk * 4096, "out_proj rank K");
        expect(sa.rows == r.linear_num_value_heads && sa.first_row == rk * 32, "in_proj_a rank rows");
      }
      expect(Throws([&] { (void)tp::RuleFor(L(i, "attn.o"), gm); }, "full-attention tensor on a GDN layer") ||
                 Throws([&] { (void)tp::RuleFor(L(i, "attn.o"), gm); }, "attention tensor on a GDN"),
             "attention tensor on a GDN layer refused");
    } else {
      const tp::ShardRule qg = tp::RuleFor(L(i, "attn.qg"), gm), k = tp::RuleFor(L(i, "attn.k"), gm), v = tp::RuleFor(L(i, "attn.v"), gm),
                          o = tp::RuleFor(L(i, "attn.o"), gm), kd = tp::RuleFor(L(i, "attn.k_descale"), gm),
                          vd = tp::RuleFor(L(i, "attn.v_descale"), gm);
      tensors += 6;
      expect(qg.split == tp::Split::kRows && qg.segments.size() == 1 && qg.segments[0].rows == 2 * q, L(i, "attn.qg"));
      expect(k.split == tp::Split::kRows && v.split == tp::Split::kRows && k.segments[0].rows == kv, L(i, "attn.k/v"));
      expect(o.split == tp::Split::kCols && o.k_total == q, L(i, "attn.o"));
      expect(kd.split == tp::Split::kRows && vd.split == tp::Split::kRows && kd.segments[0].rows == 2, L(i, "k/v_descale {2}"));
      for (int rk = 0; rk < 2; ++rk) {
        const Qwen35MoeConfig& r = rk == 0 ? r0 : r1;
        const Slice sqg = SliceOf(qg, rk, 2 * q, H), sk = SliceOf(k, rk, kv, H), so = SliceOf(o, rk, H, q), sd = SliceOf(kd, rk, 2, 1);
        expect(sqg.rows == 2 * r.num_attention_heads * r.head_dim && sqg.first_row == rk * 8192, "qg rank rows = 16 whole heads");
        expect(sk.rows == r.num_key_value_heads * r.head_dim && sk.rows == 256 && sk.first_row == rk * 256, "k/v rank rows = one kv head");
        expect(so.k == r.num_attention_heads * r.head_dim && so.k == 4096 && so.first_k == rk * 4096, "attn.o rank K = 16 heads x 256");
        expect(sd.rows == 1 && sd.first_row == rk, "descale 1 + 1");
      }
      for (const char* n : {"attn.q_norm", "attn.k_norm"}) {
        expect(tp::RuleFor(L(i, n), gm).split == tp::Split::kReplicate, L(i, n));
        ++tensors;
      }
      expect(Throws([&] { (void)tp::RuleFor(L(i, "gdn.out_proj"), gm); }, "GDN tensor on a full-attention"), "GDN tensor on an attention layer refused");
    }
    // the MoE every layer carries
    expect(tp::RuleFor(L(i, "moe.router"), gm).split == tp::Split::kReplicate &&
               tp::RuleFor(L(i, "moe.shared_gate"), gm).split == tp::Split::kReplicate,
           L(i, "router / shared_gate replicate"));
    const tp::ShardRule sg = tp::RuleFor(L(i, "moe.shared.gate"), gm), su = tp::RuleFor(L(i, "moe.shared.up"), gm),
                        sd = tp::RuleFor(L(i, "moe.shared.down"), gm);
    tensors += 5;
    expect(sg.split == tp::Split::kRows && su.split == tp::Split::kRows && sg.segments.size() == 1 && sg.segments[0].rows == 1024, L(i, "shared gate/up"));
    expect(sd.split == tp::Split::kCols && sd.k_total == 1024, L(i, "shared down"));
    for (int rk = 0; rk < 2; ++rk) {
      const Qwen35MoeConfig& r = rk == 0 ? r0 : r1;
      expect(SliceOf(sg, rk, 1024, H).rows == r.shared_expert_intermediate_size && SliceOf(sd, rk, H, 1024).k == r.shared_expert_intermediate_size,
             "shared rank slice");
    }
    for (int e = 0; e < 256; ++e) {
      const tp::ShardRule gu = tp::RuleFor(Ex(i, e, "gate_up"), gm), dn = tp::RuleFor(Ex(i, e, "down"), gm);
      tensors += 2;
      expect(gu.split == tp::Split::kRows && gu.segments.size() == 2 && gu.segments[0].rows == 1024 && gu.segments[1].begin == 1024,
             Ex(i, e, "gate_up"));
      expect(dn.split == tp::Split::kCols && dn.k_total == 1024, Ex(i, e, "down"));
      for (int rk = 0; rk < 2; ++rk) {
        const std::vector<tp::Range> rr = tp::RankRows(gu, 2, rk);
        const Qwen35MoeConfig& r = rk == 0 ? r0 : r1;
        expect(rr.size() == 2 && rr[0].begin == rk * 512 && rr[0].count == 512 && rr[1].begin == 1024 + rk * 512 && rr[1].count == 512 &&
                   rr[0].count == r.moe_intermediate_size,
               "expert gate_up ranges");
        const tp::Range c = tp::RankCols(dn, 2, rk);
        expect(c.begin == rk * 512 && c.count == 512 && c.count == r.moe_intermediate_size, "expert down K");
      }
    }
    expect(Throws([&] { (void)tp::RuleFor(L(i, "mlp.gate_up"), gm); }, "unknown tensor"), L(i, "mlp.gate_up on an MoE-only arch refused"));
    expect(Throws([&] { (void)tp::RuleFor(Ex(i, 256, "down"), gm); }, "expert 256 of a 256-expert"), "expert 256 out of range");
  }
  Check(all, "B: every tensor's rule and per-rank slice of the 48 layers x 256 experts matches Qwen35MoeConfig::Shard (" +
                 std::to_string(tensors) + " tensors)");

  Check(tp::RuleFor("lm_head", gm).split == tp::Split::kRows && tp::RuleFor("lm_head", gm).segments[0].rows == 248320 &&
            tp::RuleFor("text.embed_tokens", gm).split == tp::Split::kReplicate && tp::RuleFor("text.final_norm", gm).split == tp::Split::kReplicate,
        "lm_head vocab rows, embedding and final norm replicated");
  Check(tp::RuleFor("vision.blocks.0.attn.qkv", gm).split == tp::Split::kRank0Only && tp::RuleFor("dflash.fc", gm).split == tp::Split::kReplicate,
        "vision.* rank 0 only, dflash.* replicated (whole-prefix rules unchanged)");
  Check(Throws([&] { (void)tp::RuleFor("mtp.fc", gm); }, "unknown tensor"), "mtp.* refused on qwen3_5_moe");
  Check(Throws([&] { (void)tp::RuleFor("rotation.signs", gm); }, "unknown tensor"), "rotation.* refused on qwen3_5_moe");
  Check(Throws([&] { (void)tp::RuleFor(L(0, "input_layernorm.rotated"), gm); }, "unknown tensor"), ".rotated norm refused on qwen3_5_moe");
  Check(Throws([&] { (void)tp::RuleFor(L(0, "gdn.out_proj.trellis.w"), gm); }, "unknown tensor"), "a still-suffixed name refused");
  Check(Throws([&] { (void)tp::RuleFor(L(48, "moe.router"), gm); }, "layer 48"), "layer 48 refused");
  Check(Throws([&] { (void)tp::RuleFor(L(0, "moe.bogus"), gm); }, "unknown tensor"), "an unknown moe name refused");
  ModelConfig sharded = gm;
  sharded.tp_world = 2;
  Check(Throws([&] { (void)tp::RuleFor(L(0, "moe.router"), sharded); }, "unsharded config"), "an already-sharded config refused");
  // The dense Qwen table is unchanged: a dense 27B-shaped config still classifies mlp.*.
  ModelConfig dense;
  dense.num_hidden_layers = 4;
  dense.layer_types = {"linear_attention", "linear_attention", "linear_attention", "full_attention"};
  dense.intermediate_size = 17408;
  Check(tp::RuleFor(L(0, "mlp.down"), dense).split == tp::Split::kCols && tp::RuleFor(L(0, "mlp.gate_up"), dense).segments.size() == 2,
        "the dense Qwen3.5 table is unchanged");
}

// ---- C ----------------------------------------------------------------------------------------------------------------

// Do `a` and `b` partition [0, total) exactly (every byte once)?
bool Partition(std::vector<tp::ByteRun> runs, size_t total) {
  std::sort(runs.begin(), runs.end(), [](const tp::ByteRun& x, const tp::ByteRun& y) { return x.src_off < y.src_off; });
  size_t pos = 0;
  for (const tp::ByteRun& r : runs) {
    if (r.src_off != pos) return false;
    pos += r.bytes;
  }
  return pos == total;
}

void TestPlans() {
  const Qwen35MoeConfig g = Qwen35MoeConfig::FromJson(TextConfig());
  const ModelConfig gm = g.ToModelConfig();
  for (int kb : {3, 4, 5}) {
    for (const bool gate_up : {true, false}) {
      const q35tp::ExpertClassPlan p0 = q35tp::PlanExpertClass(gm, gate_up, kb, 2, 0), p1 = q35tp::PlanExpertClass(gm, gate_up, kb, 2, 1);
      const int64_t N = gate_up ? 2048 : 3072, K = gate_up ? 3072 : 1024;
      const size_t whole = static_cast<size_t>(N * K * kb / 8);
      std::vector<tp::ByteRun> all = p0.slice.w;
      all.insert(all.end(), p1.slice.w.begin(), p1.slice.w.end());
      const std::string tag = std::string(gate_up ? "gate_up" : "down") + " KB " + std::to_string(kb);
      Check(Partition(all, whole), "plan: the two ranks' word runs partition the expert's .trellis.w (" + tag + ")");
      Check(p0.slice.w_bytes == whole / 2 && p1.slice.w_bytes == whole / 2 && p0.w_per_rank == whole / 8 && p0.StackWords() == 256 * whole / 8,
            "plan: w_per_rank = half the expert, stack of 256 (" + tag + ")");
      if (gate_up) {
        Check(p0.slice.N == 1024 && p0.slice.K == 3072 && p0.slice.part_n[0] == 512 && p0.slice.part_n[1] == 512 && p0.slice.parts == 2 &&
                  p0.slice.w.size() == 2 && p0.slice.suh_elems == 6144 && p0.slice.svh_elems == 1024 && p1.slice.svh[0].src_off == 1024,
              "plan: gate_up rank = rows 512 + 512, both suh whole (6144), svh cut (1024) (" + tag + ")");
        Check(p1.slice.svh.size() == 2 && p1.slice.svh[0].src_off == 512 * 2 && p1.slice.svh[1].src_off == 1536 * 2,
              "plan: gate_up svh runs follow the segments (" + tag + ")");
      } else {
        Check(p0.slice.N == 3072 && p0.slice.K == 512 && p0.slice.part_n[0] == 3072 && p0.slice.part_n[1] == 0 && p0.slice.parts == 1 &&
                  p0.slice.w.size() == 96 && p0.slice.suh_elems == 512 && p0.slice.svh_elems == 3072,
              "plan: down rank = 96 runs of K tiles, suh cut (512), svh whole (3072) (" + tag + ")");
        Check(p1.slice.suh.size() == 1 && p1.slice.suh[0].src_off == 512 * 2, "plan: down rank 1 suh = K [512, 1024) (" + tag + ")");
      }
    }
  }
  // world 1 is the whole linear.
  const q35tp::ExpertClassPlan w1 = q35tp::PlanExpertClass(gm, true, 3, 1, 0);
  Check(w1.slice.w.size() == 1 && w1.slice.w_bytes == 2048u * 3072u * 3 / 8 && w1.slice.part_n[0] == 1024, "plan: world 1 = the whole expert");
  // The refusals ShardLoader::TrellisSlice makes.
  Check(Throws([&] {
          tp::ShardRule bad = tp::RuleFor(Ex(0, 0, "down"), gm);
          bad.k_total = 1024 + 64;  // 568 per rank: not a 128-block
          (void)q35tp::PlanTrellisLinear(bad, 3072, 1024 + 64, 3, {}, 2, 0);
        },
              "128"),
        "plan: a K range that is not whole 128-blocks is refused");
  Check(Throws([&] { (void)q35tp::PlanTrellisLinear(tp::RuleFor(Ex(0, 0, "down"), gm), 3072, 1024, 3, {1536, 1536}, 2, 0); }, "two input transforms"),
        "plan: a row-parallel linear with two input transforms is refused");
  Check(Throws([&] { (void)q35tp::PlanTrellisLinear(tp::RuleFor(Ex(0, 0, "gate_up"), gm), 2048, 3072, 3, {2048 - 128, 128}, 2, 0); }, "part"),
        "plan: a rank range crossing / emptying a part boundary is refused");
}

// ---- D (real container) -----------------------------------------------------------------------------------------------

// Decode a pair-grid trellis part of logical [N, K] into f16 Q[K][N].
std::vector<uint16_t> DecodeBytes(const uint8_t* bytes, size_t nbytes, int64_t K, int64_t N, int kb) {
  if (nbytes != static_cast<size_t>(N * K * kb / 8)) throw std::runtime_error("DecodeBytes: wrong byte count");
  std::vector<uint32_t> grid(nbytes / 4);
  std::memcpy(grid.data(), bytes, nbytes);
  return trellis_ref::DecodePairGrid(grid, K, N, kb);
}

std::vector<float> Widen(const std::vector<uint8_t>& f16bytes) {
  std::vector<float> out(f16bytes.size() / 2);
  for (size_t i = 0; i < out.size(); ++i) {
    uint16_t h;
    std::memcpy(&h, f16bytes.data() + 2 * i, 2);
    out[i] = r4dx::core::F16ToFloat(h);
  }
  return out;
}

struct RealCtx {
  r4dx_convert::SafetensorsReader* reader = nullptr;
  const ModelConfig* gm = nullptr;
  r4dx::model::TrellisSpec spec;
  int64_t mismatches = 0;
};

// One trellis linear `base` [N, K] of the real container, both ranks: D2 (bit-exact Q slices) and D3 (the linear).
void CheckTrellisLinear(RealCtx& c, const std::string& base, int64_t N, int64_t K, bool linear_level, std::string* line) {
  const r4dx::model::TrellisLinearSpec* ts = c.spec.Find(base);
  if (ts == nullptr) throw std::runtime_error("no quant.trellis entry for " + base);
  const int kb = ts->bits;
  const tp::ShardRule rule = tp::RuleFor(base, *c.gm);
  const std::string wn = base + ".trellis.w", un = base + ".trellis.suh", vn = base + ".trellis.svh";
  const auto bytes_of = [&](const std::string& n) {
    const auto& m = c.reader->Meta(n);
    return std::make_pair(c.reader->Data(n), static_cast<size_t>(m.end - m.begin));
  };
  const auto [wp, wb] = bytes_of(wn);
  const std::vector<uint16_t> Q = DecodeBytes(wp, wb, K, N, kb);
  const auto [up, ub] = bytes_of(un);
  const auto [vp, vb] = bytes_of(vn);
  const std::vector<uint8_t> suh_full(up, up + ub), svh_full(vp, vp + vb);
  const std::vector<float> suh_f = Widen(suh_full), svh_f = Widen(svh_full);
  const int nparts = ts->Parts();

  // a deterministic bf16 input
  std::vector<uint16_t> x(static_cast<size_t>(K));
  uint32_t s = 12345u;
  for (auto& v : x) {
    s = s * 1664525u + 1013904223u;
    v = r4dx::core::FloatToBf16(static_cast<float>(static_cast<int32_t>(s >> 8) % 2001 - 1000) / 500.0f);
  }
  std::vector<double> y_full;
  if (linear_level) {
    std::vector<const float*> sp;
    for (int p = 0; p < nparts; ++p) sp.push_back(suh_f.data() + static_cast<size_t>(p) * K);
    y_full = trellis_ref::LinearExact(x, 1, K, N, nparts > 1 ? ts->parts[0] : N, sp, Q, svh_f.data());
  }
  std::vector<double> y_sum(static_cast<size_t>(N), 0.0);
  double ymax = 0;
  for (double v : y_full) ymax = std::max(ymax, std::fabs(v));
  bool q_ok = true, lin_ok = true;
  std::vector<std::vector<double>> rank_y(2);
  for (int rk = 0; rk < 2; ++rk) {
    const q35tp::TrellisSlicePlan p = q35tp::PlanTrellisLinear(rule, N, K, kb, ts->parts, 2, rk);
    const std::vector<uint8_t> wr = tp::Gather(wp, wb, p.w);
    const std::vector<uint16_t> Qr = DecodeBytes(wr.data(), wr.size(), p.K, p.N, kb);
    // expected Q slice: columns (rows of the linear) / K rows of the full decode
    std::vector<int64_t> cols;  // global linear row of each rank column
    int64_t k0 = 0;
    if (rule.split == tp::Split::kRows) {
      for (const tp::Range& r : tp::RankRows(rule, 2, rk))
        for (int64_t n = r.begin; n < r.begin + r.count; ++n) cols.push_back(n);
    } else {
      for (int64_t n = 0; n < N; ++n) cols.push_back(n);
      if (rule.split == tp::Split::kCols) k0 = tp::RankCols(rule, 2, rk).begin;
    }
    for (int64_t kk = 0; kk < p.K && q_ok; ++kk)
      for (int64_t j = 0; j < p.N; ++j)
        if (Qr[static_cast<size_t>(kk * p.N + j)] != Q[static_cast<size_t>((k0 + kk) * N + cols[static_cast<size_t>(j)])]) {
          q_ok = false;
          break;
        }
    if (!linear_level) continue;
    const std::vector<float> suh_r = Widen(tp::Gather(up, ub, p.suh)), svh_r = Widen(tp::Gather(vp, vb, p.svh));
    std::vector<uint16_t> xr(x.begin() + k0, x.begin() + k0 + p.K);
    std::vector<const float*> sp;
    for (int pi = 0; pi < p.parts; ++pi) sp.push_back(suh_r.data() + static_cast<size_t>(pi) * p.K);
    rank_y[rk] = trellis_ref::LinearExact(xr, 1, p.K, p.N, p.part_n[0], sp, Qr, svh_r.data());
    if (rule.split == tp::Split::kRows) {
      for (int64_t j = 0; j < p.N; ++j)
        if (rank_y[rk][static_cast<size_t>(j)] != y_full[static_cast<size_t>(cols[static_cast<size_t>(j)])]) lin_ok = false;
    } else if (rule.split == tp::Split::kCols) {
      for (int64_t n = 0; n < N; ++n) y_sum[static_cast<size_t>(n)] += rank_y[rk][static_cast<size_t>(n)];
    }
  }
  if (linear_level && rule.split == tp::Split::kCols) {
    for (int64_t n = 0; n < N; ++n)
      if (std::fabs(y_sum[static_cast<size_t>(n)] - y_full[static_cast<size_t>(n)]) > 1e-10 * std::max(ymax, 1e-30)) lin_ok = false;
  }
  if (!q_ok) ++c.mismatches;
  if (!lin_ok) ++c.mismatches;
  char buf[200];
  std::snprintf(buf, sizeof buf, "%s [%lld, %lld] KB %d %s: Q slices %s%s", base.c_str(), static_cast<long long>(N), static_cast<long long>(K), kb,
                rule.split == tp::Split::kRows ? "rows" : "cols", q_ok ? "bit-exact" : "MISMATCH",
                !linear_level ? "" : (lin_ok ? (rule.split == tp::Split::kRows ? ", linear bit-exact" : ", partials add up") : ", LINEAR MISMATCH"));
  *line = buf;
}

// base name + on-disk tail of a tensor name; false for a name that is already a base ("text.embed_tokens", a bare norm).
std::string BaseOf(const std::string& name, std::string* tail) {
  for (const char* t : {".trellis.w", ".trellis.suh", ".trellis.svh", ".bf16.w", ".w4a16.wq"}) {
    const size_t n = std::strlen(t);
    if (name.size() > n && name.compare(name.size() - n, n, t) == 0) {
      *tail = t;
      return name.substr(0, name.size() - n);
    }
  }
  const size_t p = name.find(".w4a16.wsz.g");
  if (p != std::string::npos) {
    *tail = name.substr(p);
    return name.substr(0, p);
  }
  tail->clear();
  return name;
}

void TestRealContainer() {
  const std::string path = r4dx::ModelsPath("r4dx/ornith-1.5-122b-trellis-mix-kv.r4dx");
  if (!std::filesystem::exists(path)) {
    std::printf("SKIP: %s not found (real-container G1 checks)\n", path.c_str());
    ++g_skipped;
    return;
  }
  r4dx_convert::SafetensorsReader reader(r4dx_convert::Utf8ToWide(path));
  const json header = json::parse(reader.HeaderJson());
  const json& metadata = header.at("__metadata__");
  const Qwen35MoeConfig g = Qwen35MoeConfig::FromJson(metadata.at("model_config"));
  const ModelConfig gm = g.ToModelConfig();
  RealCtx ctx;
  ctx.reader = &reader;
  ctx.gm = &gm;
  ctx.spec = *r4dx::model::ParseTrellisMetadata(metadata, path);
  const std::vector<std::string> names = reader.Names();

  // ---- D1: every name classifies; per-rank bytes ------------------------------------------------------------------------
  int classified = 0, unclassified = 0;
  std::string first_bad;
  std::map<std::string, int> by_split;
  double rank_bytes[2] = {0, 0};      // on-disk bytes
  double rank_device[2] = {0, 0};     // device bytes (sign vectors widened to fp32, norm_weight to fp32)
  double replicated_device = 0;
  for (const std::string& name : names) {
    if (name == "__metadata__") continue;
    std::string tail;
    const std::string base = BaseOf(name, &tail);
    if (base == "text.embed_tokens") continue;  // pinned host memory, not VRAM
    tp::ShardRule rule;
    try {
      rule = tp::RuleFor(base, gm);
    } catch (const std::exception& e) {
      ++unclassified;
      if (first_bad.empty()) first_bad = name + ": " + e.what();
      continue;
    }
    ++classified;
    by_split[rule.split == tp::Split::kReplicate ? "replicate" : rule.split == tp::Split::kRows ? "rows" : rule.split == tp::Split::kCols ? "cols" : "rank0"]++;
    const auto& m = reader.Meta(name);
    const size_t span = static_cast<size_t>(m.end - m.begin);
    const bool widen = tail == ".trellis.suh" || tail == ".trellis.svh" || name.size() > 12 && name.compare(name.size() - 12, 12, "norm_weight") == 0;
    for (int rk = 0; rk < 2; ++rk) {
      size_t b = span;
      if (tail == ".trellis.w" || tail == ".trellis.suh" || tail == ".trellis.svh") {
        const r4dx::model::TrellisLinearSpec* ts = ctx.spec.Find(base);
        const int64_t K = tail == ".trellis.suh" ? static_cast<int64_t>(span / 2 / ts->Parts()) : 0;
        (void)K;
        // N, K from the byte counts: w = N*K*kb/8, svh = N fp16, suh = parts*K fp16.
        const int64_t N = static_cast<int64_t>(reader.Meta(base + ".trellis.svh").end - reader.Meta(base + ".trellis.svh").begin) / 2;
        const int64_t Kk = static_cast<int64_t>(reader.Meta(base + ".trellis.suh").end - reader.Meta(base + ".trellis.suh").begin) / 2 / ts->Parts();
        const q35tp::TrellisSlicePlan p = q35tp::PlanTrellisLinear(rule, N, Kk, ts->bits, ts->parts, 2, rk);
        b = tail == ".trellis.w" ? p.w_bytes : tail == ".trellis.suh" ? p.suh_elems * 2 : p.svh_elems * 2;
      } else if (tail == ".bf16.w") {
        const auto& sh = m.shape;
        tp::PartShape ps;
        ps.part = tp::Part::kBf16;
        ps.N = sh[0];
        ps.K = sh[1];
        if (rule.split == tp::Split::kRows) {
          size_t t = 0;
          for (const tp::ByteRun& r : tp::PlanRows(ps, tp::RankRows(rule, 2, rk))) t += r.bytes;
          b = t;
        } else if (rule.split == tp::Split::kCols) {
          size_t t = 0;
          for (const tp::ByteRun& r : tp::PlanCols(ps, tp::RankCols(rule, 2, rk))) t += r.bytes;
          b = t;
        }
      } else if (tail == ".w4a16.wq" || tail.rfind(".w4a16.wsz.g", 0) == 0) {
        tp::PartShape ps;
        const int64_t N = g.vocab_size, K = g.hidden_size;
        ps.N = N;
        ps.K = K;
        if (tail == ".w4a16.wq") {
          ps.part = tp::Part::kW4Wq;
        } else {
          ps.part = tp::Part::kW4a16Wsz;
          ps.group = std::stoi(tail.substr(std::strlen(".w4a16.wsz.g")));
        }
        size_t t = 0;
        for (const tp::ByteRun& r : tp::PlanRows(ps, tp::RankRows(rule, 2, rk))) t += r.bytes;
        b = t;
      } else if (rule.split == tp::Split::kRows) {  // raw tensors: rows of span / total rows bytes
        int64_t rows = 0;
        for (const tp::Segment& sg : rule.segments) rows += sg.rows;
        tp::PartShape ps;
        ps.part = tp::Part::kElem;
        ps.N = rows;
        ps.row_bytes = static_cast<int64_t>(span) / rows;
        size_t t = 0;
        for (const tp::ByteRun& r : tp::PlanRows(ps, tp::RankRows(rule, 2, rk))) t += r.bytes;
        b = t;
      }
      rank_bytes[rk] += static_cast<double>(b);
      rank_device[rk] += static_cast<double>(widen ? b * 2 : b);
    }
    if (rule.split == tp::Split::kReplicate) replicated_device += static_cast<double>(widen ? span * 2 : span);
  }
  Check(unclassified == 0, "D1: every tensor name of the real container classifies (" + std::to_string(classified) + " tensors; rows " +
                               std::to_string(by_split["rows"]) + ", cols " + std::to_string(by_split["cols"]) + ", replicate " +
                               std::to_string(by_split["replicate"]) + ")" + (first_bad.empty() ? "" : " first failure: " + first_bad));
  const double gib = 1024.0 * 1024.0 * 1024.0;
  std::printf("INFO: per-rank bytes  rank 0 disk %.3f GiB device %.3f GiB | rank 1 disk %.3f GiB device %.3f GiB | replicated device %.3f GiB\n",
              rank_bytes[0] / gib, rank_device[0] / gib, rank_bytes[1] / gib, rank_device[1] / gib, replicated_device / gib);
  Check(rank_bytes[0] == rank_bytes[1], "D1: both ranks hold the same number of bytes");
  Check(std::fabs(rank_device[0] / gib - 25.997) < 0.30, "D1: per-rank device weights ~ 25.997 GiB (design section 5)");

  // ---- D2 / D3: real bytes ----------------------------------------------------------------------------------------------
  // layers by expert KB
  int first_gu[6] = {-1, -1, -1, -1, -1, -1}, first_dn[6] = {-1, -1, -1, -1, -1, -1};
  std::map<int, int> gu_hist, dn_hist;
  for (int i = 0; i < 48; ++i) {
    const auto* gu = ctx.spec.Find(Ex(i, 0, "gate_up"));
    const auto* dn = ctx.spec.Find(Ex(i, 0, "down"));
    if (gu != nullptr) {
      ++gu_hist[gu->bits];
      if (first_gu[gu->bits] < 0) first_gu[gu->bits] = i;
    }
    if (dn != nullptr) {
      ++dn_hist[dn->bits];
      if (first_dn[dn->bits] < 0) first_dn[dn->bits] = i;
    }
  }
  std::string hist = "INFO: expert KB per layer  gate_up";
  for (auto& kv : gu_hist) hist += " KB" + std::to_string(kv.first) + "x" + std::to_string(kv.second);
  hist += "  down";
  for (auto& kv : dn_hist) hist += " KB" + std::to_string(kv.first) + "x" + std::to_string(kv.second);
  std::printf("%s\n", hist.c_str());

  std::vector<std::string> lines;
  std::string line;
  const auto run = [&](const std::string& base, int64_t N, int64_t K, bool lin = true) {
    CheckTrellisLinear(ctx, base, N, K, lin, &line);
    lines.push_back(line);
  };
  // one GDN layer and one attention layer
  run(L(0, "gdn.in_proj_qkv"), 12288, 3072);
  run(L(0, "gdn.in_proj_z"), 8192, 3072);
  run(L(0, "gdn.out_proj"), 3072, 8192);
  run(L(3, "attn.qg"), 16384, 3072);
  run(L(3, "attn.k"), 512, 3072);
  run(L(3, "attn.v"), 512, 3072);
  run(L(3, "attn.o"), 3072, 8192);
  // a layer of each mixer KB in the mix, if the container has other KBs on mixers
  std::set<int> mixer_kbs;
  for (const char* sfx : {"gdn.in_proj_qkv", "gdn.in_proj_z", "gdn.out_proj", "attn.qg", "attn.k", "attn.v", "attn.o"})
    for (int i = 0; i < 48; ++i)
      if (const auto* t = ctx.spec.Find(L(i, sfx))) mixer_kbs.insert(t->bits);
  std::string mk = "INFO: mixer KBs in the container:";
  for (int b : mixer_kbs) mk += " " + std::to_string(b);
  std::printf("%s\n", mk.c_str());
  for (int b : mixer_kbs) {
    if (b == 3 || b == 4 || b == 5) {
      // a layer other than 0 / 3 whose out_proj or o has this KB (Q-level only; the linear check is on layers 0 and 3)
      for (int i = 1; i < 48; ++i) {
        const std::string base = L(i, ctx.gm->layer_types[static_cast<size_t>(i)] == "full_attention" ? "attn.o" : "gdn.out_proj");
        const auto* t = ctx.spec.Find(base);
        if (t != nullptr && t->bits == b && i != 3) {
          run(base, 3072, 8192, false);
          break;
        }
      }
    }
  }
  // experts: several per KB class (gate_up and down), first / middle / last expert
  for (int b : {3, 4, 5}) {
    for (const bool gate_up : {true, false}) {
      const int layer = gate_up ? first_gu[b] : first_dn[b];
      if (layer < 0) continue;
      for (int e : {0, 101, 255}) run(Ex(layer, e, gate_up ? "gate_up" : "down"), gate_up ? 2048 : 3072, gate_up ? 3072 : 1024);
    }
  }
  for (const std::string& l : lines) std::printf("INFO: %s\n", l.c_str());
  Check(ctx.mismatches == 0, "D2/D3: decode(Gather(plan(rank))) == slice(decode(full)) bit for bit, and the sliced linears reproduce / add up to the full output (" +
                                 std::to_string(lines.size()) + " real linears, both ranks)");

  // negative control: rank 1's bytes against rank 0's expected slice must NOT match (the comparison above can fail).
  {
    const std::string base = Ex(0, 7, "down");
    const tp::ShardRule rule = tp::RuleFor(base, gm);
    const int kb = ctx.spec.Find(base)->bits;
    const auto& m = reader.Meta(base + ".trellis.w");
    const uint8_t* wp = reader.Data(base + ".trellis.w");
    const size_t wb = static_cast<size_t>(m.end - m.begin);
    const std::vector<uint16_t> Q = DecodeBytes(wp, wb, 1024, 3072, kb);
    const q35tp::TrellisSlicePlan p1 = q35tp::PlanTrellisLinear(rule, 3072, 1024, kb, {}, 2, 1);
    const std::vector<uint8_t> w1 = tp::Gather(wp, wb, p1.w);
    const std::vector<uint16_t> Q1 = DecodeBytes(w1.data(), w1.size(), p1.K, p1.N, kb);
    bool same_as_rank0 = true, same_as_rank1 = true;
    for (int64_t kk = 0; kk < p1.K; ++kk)
      for (int64_t n = 0; n < p1.N; ++n) {
        same_as_rank0 = same_as_rank0 && Q1[static_cast<size_t>(kk * p1.N + n)] == Q[static_cast<size_t>(kk * 3072 + n)];
        same_as_rank1 = same_as_rank1 && Q1[static_cast<size_t>(kk * p1.N + n)] == Q[static_cast<size_t>((512 + kk) * 3072 + n)];
      }
    Check(!same_as_rank0 && same_as_rank1, "D2 negative control: rank 1's slice equals K rows [512, 1024) of the full decode and differs from rows [0, 512)");
  }

  // ---- D4: bf16 shared expert + raw tensors -----------------------------------------------------------------------------
  bool d4 = true;
  {
    const std::string sd = L(5, "moe.shared.down"), sg = L(5, "moe.shared.gate");
    const auto& md = reader.Meta(sd + ".bf16.w");
    const uint8_t* dp = reader.Data(sd + ".bf16.w");
    tp::PartShape ps;
    ps.part = tp::Part::kBf16;
    ps.N = 3072;
    ps.K = 1024;
    for (int rk = 0; rk < 2; ++rk) {
      const std::vector<uint8_t> got = tp::Gather(dp, static_cast<size_t>(md.end - md.begin), tp::PlanCols(ps, tp::RankCols(tp::RuleFor(sd, gm), 2, rk)));
      for (int64_t n = 0; n < 3072 && d4; ++n)
        d4 = std::memcmp(got.data() + static_cast<size_t>(n) * 512 * 2, dp + (static_cast<size_t>(n) * 1024 + static_cast<size_t>(rk) * 512) * 2, 512 * 2) == 0;
    }
    const auto& mg = reader.Meta(sg + ".bf16.w");
    const uint8_t* gp = reader.Data(sg + ".bf16.w");
    ps.N = 1024;
    ps.K = 3072;
    for (int rk = 0; rk < 2; ++rk) {
      const std::vector<uint8_t> got = tp::Gather(gp, static_cast<size_t>(mg.end - mg.begin), tp::PlanRows(ps, tp::RankRows(tp::RuleFor(sg, gm), 2, rk)));
      d4 = d4 && got.size() == 512u * 3072u * 2 && std::memcmp(got.data(), gp + static_cast<size_t>(rk) * 512 * 3072 * 2, got.size()) == 0;
    }
    // conv1d_weight [12288][4] bf16 = 8-byte rows, three segments 2048 | 2048 | 8192
    const std::string cv = L(0, "gdn.conv1d_weight");
    const uint8_t* cp = reader.Data(cv);
    const size_t cb = static_cast<size_t>(reader.Meta(cv).end - reader.Meta(cv).begin);
    tp::PartShape es;
    es.part = tp::Part::kElem;
    es.N = 12288;
    es.row_bytes = 8;
    for (int rk = 0; rk < 2; ++rk) {
      const std::vector<uint8_t> got = tp::Gather(cp, cb, tp::PlanRows(es, tp::RankRows(tp::RuleFor(cv, gm), 2, rk)));
      std::vector<uint8_t> want;
      for (const auto& seg : {std::make_pair(0, 2048), std::make_pair(2048, 2048), std::make_pair(4096, 8192)}) {
        const int64_t half = seg.second / 2;
        const uint8_t* s = cp + static_cast<size_t>(seg.first + rk * half) * 8;
        want.insert(want.end(), s, s + static_cast<size_t>(half) * 8);
      }
      d4 = d4 && got == want;
    }
    // k_descale [2] fp32 -> one fp32 per rank
    const std::string kd = L(3, "attn.k_descale");
    es.N = 2;
    es.row_bytes = 4;
    for (int rk = 0; rk < 2; ++rk) {
      const std::vector<uint8_t> got = tp::Gather(reader.Data(kd), 8, tp::PlanRows(es, tp::RankRows(tp::RuleFor(kd, gm), 2, rk)));
      d4 = d4 && got.size() == 4 && std::memcmp(got.data(), reader.Data(kd) + 4 * rk, 4) == 0;
    }
  }
  Check(d4, "D4: bf16 shared down (K 512), shared gate (rows 512), conv1d_weight (3 segments), k_descale (1 + 1) slice as the rules say");
}

}  // namespace

int main() {
  try {
    TestShardConfig();
    TestRules();
    TestPlans();
    TestRealContainer();
  } catch (const std::exception& e) {
    std::fprintf(stderr, "FAIL: uncaught exception: %s\n", e.what());
    return 1;
  }
  if (g_failures != 0) {
    std::fprintf(stderr, "%d check(s) FAILED\n", g_failures);
    return 1;
  }
  std::printf("ALL PASS%s\n", g_skipped != 0 ? " (real-container checks skipped)" : "");
  return 0;
}
