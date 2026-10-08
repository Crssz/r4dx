// tests/model/test_reshard_plan_cpu.cpp -- CPU-only checks of src/model/reshard_plan.h, the exact state mapping of the hybrid
// mode (docs/pp-tp2-hybrid.md 3): the full-head live state of a pipeline stage <-> TP=2 rank r's head-split image.
//   * the mapping is DERIVED from tp::RuleFor / tp::RankRows (the tables the weight loader cuts the weights with) and equals
//     the design's numbers on the real 27B config; every rule that owns a head or a channel is asserted equal to the state
//     mapping, so a change to one shard rule cannot silently diverge from the state;
//   * the copy ranges and sizes per direction for (split k, p0, n_end) reproduce the design's crossing-size table, the MTP
//     primed-block rule and the ring chunking;
//   * ReshardRef / GatherRef (the host reference) round-trip random images, are hand-checked on a tiny model, and agree
//     byte for byte with a host executor of the plan's strided copies (the on-device copies' shape), including pitched conv
//     lines and partial block ranges;
//   * NEGATIVE CONTROLS: swapped head halves, a dropped conv segment, a skipped MTP half, an off-by-one block, a perturbed
//     rule each break an observable.
// No HIP call, no container; always runs.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "reshard_plan.h"

using namespace r4dx::model;
using namespace r4dx::model::hybrid;

namespace {

int g_fails = 0;
void Check(bool ok, const char* what) {
  if (!ok) {
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++g_fails;
  }
}

// ---- configs ----------------------------------------------------------------------------------------------------------
// The 27B's config (tests/model/test_tp_shard.cpp's RealConfig).
ModelConfig RealConfig() {
  ModelConfig c;
  c.hidden_size = 5120;
  c.num_hidden_layers = 64;
  for (int i = 0; i < 64; ++i) c.layer_types.push_back(i % 4 == 3 ? "full_attention" : "linear_attention");
  c.num_attention_heads = 24;
  c.num_key_value_heads = 4;
  c.head_dim = 256;
  c.intermediate_size = 17408;
  c.linear_num_key_heads = 16;
  c.linear_num_value_heads = 48;
  c.linear_key_head_dim = 128;
  c.linear_value_head_dim = 128;
  c.vocab_size = 248320;
  return c;
}
// A small model with the same structure (4 kv heads, GQA 2 / 2, 3 conv segments, 8 layers: attention at 3 and 7) for random data.
ModelConfig TinyConfig() {
  ModelConfig c;
  c.hidden_size = 32;
  c.num_hidden_layers = 8;
  for (int i = 0; i < 8; ++i) c.layer_types.push_back(i % 4 == 3 ? "full_attention" : "linear_attention");
  c.num_attention_heads = 8;
  c.num_key_value_heads = 4;
  c.head_dim = 4;
  c.intermediate_size = 16;
  c.linear_num_key_heads = 4;
  c.linear_num_value_heads = 8;
  c.linear_key_head_dim = 4;
  c.linear_value_head_dim = 4;
  c.vocab_size = 64;
  return c;
}
// The smallest one a human can check by hand: 4 layers (attention at 3), 4 kv heads of head_dim 1, 2 key heads / 4 value heads of 1.
ModelConfig HandConfig() {
  ModelConfig c;
  c.hidden_size = 8;
  c.num_hidden_layers = 4;
  for (int i = 0; i < 4; ++i) c.layer_types.push_back(i % 4 == 3 ? "full_attention" : "linear_attention");
  c.num_attention_heads = 8;
  c.num_key_value_heads = 4;
  c.head_dim = 1;
  c.intermediate_size = 8;
  c.linear_num_key_heads = 2;
  c.linear_num_value_heads = 4;
  c.linear_key_head_dim = 1;
  c.linear_value_head_dim = 1;
  c.vocab_size = 8;
  return c;
}

bool Same(const std::vector<tp::Range>& a, const std::vector<tp::Range>& b) {
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); ++i) {
    if (a[i].begin != b[i].begin || a[i].count != b[i].count) return false;
  }
  return true;
}
std::vector<tp::Range> Rows(const ModelConfig& g, const std::string& name, int rank) {
  return tp::RankRows(tp::RuleFor(name, g), 2, rank);
}
constexpr uint64_t kMiB = 1ull << 20;
uint64_t MiBr(uint64_t bytes) { return (bytes + (kMiB / 2)) / kMiB; }  // rounded to the nearest MiB (the design's table)

// ---- the real mapping: closed form against the rule tables, rule by rule --------------------------------------------------
void RealMapping() {
  const ModelConfig cfg = RealConfig();
  const StateGeometry g = StateGeometry::FromRules(cfg);
  Check(VerifyAgainstRules(g, cfg).empty(), "FromRules' geometry equals every rule that owns a head or a channel");
  Check(g.num_layers == 64 && g.attn_layers.size() == 16 && g.gdn_layers.size() == 48 && g.attn_layers.front() == 3 && g.attn_layers.back() == 63,
        "16 attention layers (3, 7, ..., 63) and 48 GDN layers");
  Check(g.KvTokenHeadBytes() == 512 && g.KvHeadBlockBytes() == 8192 && g.KvBlockBytesFull() == 32768 && g.KvBlockBytesRank(0) == 16384 &&
            g.KvBlockBytesRank(1) == 16384,
        "KV: 512 B per (token, head), 8 KiB per (block, head), 32 KiB per full block, 16 KiB per rank block");
  Check(g.RecurrentBytesFull() == 3 * static_cast<int64_t>(kMiB) && g.RecurrentBytesRank(0) == 3 * static_cast<int64_t>(kMiB) / 2 &&
            g.RecurrentBytesRank(1) == 3 * static_cast<int64_t>(kMiB) / 2,
        "GDN recurrent slot: 3 MiB full, 1.5 MiB per rank");
  Check(g.conv_dim_full == 10240 && g.conv_live == 3 && g.ConvChannelsRank(0) == 5120 && g.ConvChannelsRank(1) == 5120, "conv: 10240 channels, 3 live entries, 5120 per rank");
  for (int r = 0; r < 2; ++r) {
    const RankMap& m = g.ranks[static_cast<size_t>(r)];
    Check(m.kv_head_begin == 2 * r && m.kv_heads == 2, "rank r holds KV heads {2r, 2r+1}");
    Check(m.v_head_begin == 24 * r && m.v_heads == 24, "rank r holds GDN v-heads [24r, 24r+24)");
    Check(m.conv.size() == 3, "three conv segments");
    if (m.conv.size() == 3) {
      Check(m.conv[0].full_begin == 1024 * r && m.conv[0].count == 1024 && m.conv[0].rank_begin == 0, "q segment [1024r,+1024) -> local [0,1024)");
      Check(m.conv[1].full_begin == 2048 + 1024 * r && m.conv[1].count == 1024 && m.conv[1].rank_begin == 1024,
            "k segment [2048+1024r,+1024) -> local [1024,2048)");
      Check(m.conv[2].full_begin == 4096 + 3072 * r && m.conv[2].count == 3072 && m.conv[2].rank_begin == 2048,
            "v segment [4096+3072r,+3072) -> local [2048,5120)");
    }
  }

  // MAPPING == WEIGHT MAPPING, for every rule that owns a head or a channel, against the DESIGN'S numbers (not through the
  // geometry): rank r of the loader's tables must be exactly these ranges.
  const std::string gdn = "text.layers.0.gdn.", att = "text.layers.3.attn.";
  for (int r = 0; r < 2; ++r) {
    const std::vector<tp::Range> conv = {{1024 * r, 1024}, {2048 + 1024 * r, 1024}, {4096 + 3072 * r, 3072}};
    Check(Same(Rows(cfg, gdn + "conv1d_weight", r), conv), "rule gdn.conv1d_weight == the state's conv channel segments");
    Check(Same(Rows(cfg, gdn + "in_proj_qkv", r), conv), "rule gdn.in_proj_qkv == the state's conv channel segments");
    for (const char* t : {"A_log", "dt_bias", "in_proj_a", "in_proj_b"}) {
      Check(Same(Rows(cfg, gdn + t, r), {{24 * r, 24}}), (std::string("rule gdn.") + t + " == the state's v-heads [24r,+24)").c_str());
    }
    Check(Same(Rows(cfg, gdn + "in_proj_z", r), {{24 * r * 128, 24 * 128}}), "rule gdn.in_proj_z == v-heads x 128 rows");
    const tp::Range oc = tp::RankCols(tp::RuleFor(gdn + "out_proj", cfg), 2, r);
    Check(oc.begin == 24 * r * 128 && oc.count == 24 * 128, "rule gdn.out_proj (K split) == v-heads x 128");
    for (const std::string& p : {att, std::string("mtp.attn.")}) {
      for (const char* t : {"k", "v"}) {
        Check(Same(Rows(cfg, p + t, r), {{2 * r * 256, 2 * 256}}), (p + t + " == the state's KV heads {2r,2r+1} x head_dim").c_str());
      }
      for (const char* t : {"k_descale", "v_descale"}) {
        Check(Same(Rows(cfg, p + t, r), {{2 * r, 2}}), (p + t + " == the state's KV heads {2r,2r+1}").c_str());
      }
      Check(Same(Rows(cfg, p + "qg", r), {{2 * 12 * r * 256, 2 * 12 * 256}}), (p + "qg == q heads 12r..12r+11 (the GQA group of the KV heads)").c_str());
      const tp::Range o = tp::RankCols(tp::RuleFor(p + "o", cfg), 2, r);
      Check(o.begin == 12 * r * 256 && o.count == 12 * 256, (p + "o == q heads 12r..12r+11").c_str());
    }
  }
  // The rule table's split KINDS for every per-layer tensor: a rule that changes kind (or a new row split on a state-owning
  // tensor) must be seen here. Replicated / MLP tensors own no state.
  struct Kind {
    const char* name;
    tp::Split split;
    bool owns_state;
  };
  const Kind kinds[] = {
      {"gdn.in_proj_qkv", tp::Split::kRows, true}, {"gdn.conv1d_weight", tp::Split::kRows, true}, {"gdn.in_proj_z", tp::Split::kRows, true},
      {"gdn.in_proj_a", tp::Split::kRows, true},   {"gdn.in_proj_b", tp::Split::kRows, true},     {"gdn.A_log", tp::Split::kRows, true},
      {"gdn.dt_bias", tp::Split::kRows, true},     {"gdn.out_proj", tp::Split::kCols, true},      {"gdn.norm_weight", tp::Split::kReplicate, false},
      {"mlp.gate_up", tp::Split::kRows, false},    {"mlp.down", tp::Split::kCols, false},         {"input_layernorm", tp::Split::kReplicate, false},
  };
  for (const Kind& k : kinds) {
    Check(tp::RuleFor("text.layers.0." + std::string(k.name), cfg).split == k.split, (std::string("rule kind of ") + k.name).c_str());
  }
  const Kind akinds[] = {
      {"attn.qg", tp::Split::kRows, true},  {"attn.k", tp::Split::kRows, true},        {"attn.v", tp::Split::kRows, true},
      {"attn.k_descale", tp::Split::kRows, true}, {"attn.v_descale", tp::Split::kRows, true}, {"attn.o", tp::Split::kCols, true},
      {"attn.q_norm", tp::Split::kReplicate, false}, {"attn.k_norm", tp::Split::kReplicate, false},
  };
  for (const Kind& k : akinds) {
    Check(tp::RuleFor("text.layers.3." + std::string(k.name), cfg).split == k.split, (std::string("rule kind of ") + k.name).c_str());
  }
  // per-rank head sums
  Check(g.ranks[0].kv_heads + g.ranks[1].kv_heads == 4 && g.ranks[0].v_heads + g.ranks[1].v_heads == 48, "the ranks partition the heads");
}

// ---- sizes per direction: the design's table ----------------------------------------------------------------------------
void Sizes() {
  const ModelConfig cfg = RealConfig();
  const StateGeometry g = StateGeometry::FromRules(cfg);
  const PlanParams p35 = PlanParams::ForSplit(g, 35);
  Check(p35.stage[kStageA].attn.size() == 8 && p35.stage[kStageA].attn.back() == 31 && p35.stage[kStageA].gdn.size() == 27,
        "k = 35: stage A owns attention 3..31 (8) and 27 GDN layers; layer 35 is stage B's");
  Check(p35.stage[kStageB].attn.size() == 8 && p35.stage[kStageB].attn.front() == 35 && p35.stage[kStageB].gdn.size() == 21,
        "k = 35: stage B owns attention 35..63 (8) and 21 GDN layers");
  const PlanParams p32 = PlanParams::ForSplit(g, 32);
  Check(p32.stage[kStageA].gdn.size() == 24 && p32.stage[kStageB].gdn.size() == 24 && p32.stage[kStageA].attn.size() == 8 && p32.stage[kStageB].attn.size() == 8,
        "k = 32: 8 attention + 24 GDN layers on each stage");

  // The design table (section 3), a COLD call (p0 = 0), k = 35: KV 64 / 256 / 1024 MiB, GDN 41 (X->Y) / 32 (Y->X), MTP 8 / 32 / 128
  // (Y->X only), X->Y 105 / 297 / 1065, Y->X 104 / 320 / 1184.
  struct Row {
    int64_t n_end;
    uint64_t kv, mtp, xy, yx;
  };
  const Row rows[] = {{8192, 64, 8, 105, 104}, {32768, 256, 32, 297, 320}, {131072, 1024, 128, 1065, 1184}};
  for (const Row& r : rows) {
    const ReshardPlan plan = ScatterPlan(g, p35, 0, r.n_end);
    const PlanTotals t = Totals(plan);
    Check(MiBr(t.CrossKind(kStageA, StateKind::kKv)) == r.kv && MiBr(t.CrossKind(kStageB, StateKind::kKv)) == r.kv,
          "scatter: KV crossing per direction (8 layers x 16 KiB per block x blocks)");
    Check(MiBr(t.CrossKind(kStageA, StateKind::kGdnRecurrent) + t.CrossKind(kStageA, StateKind::kGdnConv)) == 41, "scatter: GDN X->Y = 27 x 1.53 MiB = 41 MiB");
    Check(MiBr(t.CrossKind(kStageB, StateKind::kGdnRecurrent) + t.CrossKind(kStageB, StateKind::kGdnConv)) == 32, "scatter: GDN Y->X = 21 x 1.53 MiB = 32 MiB");
    Check(MiBr(t.CrossKind(kStageB, StateKind::kMtpKv)) == r.mtp && t.CrossKind(kStageA, StateKind::kMtpKv) == 0, "scatter: MTP KV crosses Y->X only");
    Check(MiBr(t.CrossFrom(kStageA)) == r.xy, "scatter: X->Y total");
    Check(MiBr(t.CrossFrom(kStageB)) == r.yx, "scatter: Y->X total");
    std::fprintf(stderr, "  n_end %6lld  X->Y %.2f MiB  Y->X %.2f MiB  (local %.2f MiB)\n", static_cast<long long>(r.n_end),
                 static_cast<double>(t.CrossFrom(kStageA)) / static_cast<double>(kMiB), static_cast<double>(t.CrossFrom(kStageB)) / static_cast<double>(kMiB),
                 static_cast<double>(t.Local()) / static_cast<double>(kMiB));
  }
  {  // exact figures at 8k
    const PlanTotals t = Totals(ScatterPlan(g, p35, 0, 8192));
    Check(t.CrossKind(kStageA, StateKind::kKv) == 8ull * 512 * 16384, "8k: KV per direction = 8 layers x 512 blocks x 16 KiB exactly");
    Check(t.CrossKind(kStageA, StateKind::kGdnRecurrent) == 27ull * 1572864 && t.CrossKind(kStageA, StateKind::kGdnConv) == 27ull * 5120 * 6,
          "8k: GDN X->Y = 27 x (1.5 MiB recurrent half + 5120 channels x 6 B)");
    Check(t.CrossKind(kStageB, StateKind::kMtpKv) == 512ull * 16384, "8k: MTP KV Y->X = 512 blocks x 16 KiB");
  }
  {  // the local halves are D2D: stage A's own rank (rank 1) and stage B's own rank (rank 0) never cross
    const ReshardPlan plan = ScatterPlan(g, p35, 0, 8192);
    bool ok = true;
    for (const CopyOp& op : plan.ops) {
      if (op.local != (op.rank == RankOfStage(op.stage))) ok = false;
    }
    Check(ok, "an op is local exactly when its rank is the stage's own card (stage A = rank 1, stage B = rank 0)");
    const PlanTotals t = Totals(plan);
    Check(t.Local() > 0 && t.local[static_cast<int>(StateKind::kMtpKv)] == 512ull * 16384, "the MTP head's own half (rank 0) is a D2D copy");
  }
  {  // warm scatter: only the new rows (and the MTP boundary block)
    const PlanTotals t = Totals(ScatterPlan(g, p35, 1000, 1300));  // rows [1000,1300): blocks 62 .. 81 = 20 blocks
    Check(t.CrossKind(kStageA, StateKind::kKv) == 8ull * 20 * 16384, "warm scatter [1000,1300): layer KV blocks 62..81");
    Check(t.CrossKind(kStageB, StateKind::kMtpKv) == 20ull * 16384, "warm scatter: MTP blocks start at block(999) = 62, also 62..81");
    const PlanTotals t2 = Totals(ScatterPlan(g, p35, 1024, 1300));  // p0 on a block boundary: layer KV blocks 64..81, MTP 63..81
    Check(t2.CrossKind(kStageA, StateKind::kKv) == 8ull * 18 * 16384 && t2.CrossKind(kStageB, StateKind::kMtpKv) == 19ull * 16384,
          "p0 = 1024 (a block boundary): layer KV from block 64, MTP from block 63 (the previous one)");
  }

  // Run shapes: one hipMemcpy2D per (layer, rank) for KV: spitch 32 KiB, dpitch 16 KiB, width 16 KiB, height = blocks, offset r*16 KiB.
  {
    const ReshardPlan plan = ScatterPlan(g, p35, 0, 8192);
    const CopyOp* kv0 = nullptr;
    const CopyOp* kv1 = nullptr;
    for (const CopyOp& op : plan.ops) {
      if (op.kind == StateKind::kKv && op.layer == 3) (op.rank == 0 ? kv0 : kv1) = &op;
    }
    Check(kv0 != nullptr && kv1 != nullptr, "layer 3 has a KV op per rank");
    if (kv0 != nullptr && kv1 != nullptr) {
      const Run2D &a = kv0->runs.at(0), &b = kv1->runs.at(0);
      Check(a.full_off == 0 && b.full_off == 16384, "KV: rank r reads at byte offset r x 16 KiB in every block");
      Check(a.full_pitch == 32768 && a.rank_pitch == 16384 && a.width == 16384 && a.height == 512 && b.height == 512, "KV: spitch 32 KiB, dpitch 16 KiB, width 16 KiB, height = blocks");
      Check(a.rank_off == 0 && b.rank_off == 0, "KV: the rank's image starts at its block 0");
    }
    const CopyOp* conv1 = nullptr;
    const CopyOp* rec1 = nullptr;
    for (const CopyOp& op : plan.ops) {
      if (op.layer == 0 && op.rank == 1 && op.kind == StateKind::kGdnConv) conv1 = &op;
      if (op.layer == 0 && op.rank == 1 && op.kind == StateKind::kGdnRecurrent) rec1 = &op;
    }
    Check(conv1 != nullptr && conv1->runs.size() == 3, "conv: three segment runs per (layer, rank)");
    if (conv1 != nullptr && conv1->runs.size() == 3) {
      Check(conv1->runs[0].full_off == 1024 * 6 && conv1->runs[0].height == 1024 && conv1->runs[0].rank_off == 0, "rank 1 conv q: global channel 1024 -> local 0");
      Check(conv1->runs[1].full_off == (2048 + 1024) * 6 && conv1->runs[1].rank_off == 1024 * 6, "rank 1 conv k: global 3072 -> local 1024");
      Check(conv1->runs[2].full_off == (4096 + 3072) * 6 && conv1->runs[2].rank_off == 2048 * 6 && conv1->runs[2].height == 3072, "rank 1 conv v: global 7168 -> local 2048");
      Check(conv1->runs[0].width == 6 && conv1->runs[0].full_pitch == 6, "conv: 3 live bf16 entries (6 B) per channel line");
    }
    Check(rec1 != nullptr && rec1->runs.at(0).full_off == 24 * 65536 && rec1->runs.at(0).width == 24 * 65536 && rec1->runs.at(0).rank_off == 0,
          "recurrent: rank 1 = v-heads 24..47 = bytes [1.5 MiB, 3 MiB) of the live slot, contiguous");
  }
  {  // a speculating rank's conv lines are longer (2 + window); the pitch is a parameter of the run, not of the state
    PlanParams p = p35;
    p.rank_conv_pitch = 5;
    const ReshardPlan plan = ScatterPlan(g, p, 0, 16);
    for (const CopyOp& op : plan.ops) {
      if (op.kind == StateKind::kGdnConv && op.rank == 1) {
        Check(op.runs[1].rank_pitch == 10 && op.runs[1].rank_off == 1024 * 10 && op.runs[1].full_pitch == 6 && op.runs[1].width == 6, "pitched rank conv line (window 3): 10 B pitch, 6 B live");
        break;
      }
    }
  }

  // The warm gather from a tracker plan: 300 rows behind the position, both stages stale.
  {
    TpMasterTracker t;
    t.AfterPipelined(10000);
    t.TpOnly(10000);
    const TpMasterTracker::SyncPlan sp = t.PlanSync(10300);
    const ReshardPlan plan = GatherPlan(g, p35, sp);
    const PlanTotals tot = Totals(plan);
    // rows [10000, 10300): blocks 625 .. 643 = 19 blocks; per direction the remote half
    Check(tot.CrossKind(kStageB, StateKind::kKv) == 8ull * 19 * 16384 && tot.CrossKind(kStageA, StateKind::kKv) == 8ull * 19 * 16384,
          "gather: 19 blocks of KV per stage layer, the remote head half only");
    Check(MiBr(tot.CrossKind(kStageB, StateKind::kGdnRecurrent) + tot.CrossKind(kStageB, StateKind::kGdnConv)) == 41,
          "gather: stage A <- rank 0's GDN half over PCIe, Y->X, 27 layers = 41 MiB");
    Check(MiBr(tot.CrossKind(kStageA, StateKind::kGdnRecurrent) + tot.CrossKind(kStageA, StateKind::kGdnConv)) == 32,
          "gather: stage B <- rank 1's GDN half over PCIe, X->Y, 21 layers = 32 MiB");
    Check(tot.CrossKind(kStageA, StateKind::kMtpKv) == 16384 && tot.local[static_cast<int>(StateKind::kMtpKv)] == 16384,
          "gather: one MTP block (row p0-1) into stage B, 16 KiB local + 16 KiB from the other card");
    // a call that needs nothing but the MTP block
    TpMasterTracker t2;
    t2.AfterPipelined(10000);
    const ReshardPlan quiet = GatherPlan(g, p35, t2.PlanSync(10000));
    const PlanTotals qt = Totals(quiet);
    Check(qt.CrossFrom(kStageB) == 0 && qt.CrossFrom(kStageA) == 16384 && quiet.ops.size() == 2, "a quiet warm gather is just the MTP block");
  }
}

void Ring() {
  const std::vector<RingPiece> p = ChunkRing(1065ull * kMiB + 300000);  // ~ the 128k X->Y total
  Check(p.size() == 9 && p[0].bytes == kRingPieceBytes && p[7].bytes == kRingPieceBytes && p[8].bytes == 41ull * kMiB + 300000,
        "9 pieces for 1065 MiB (8 full 128 MiB pieces and a tail)");
  uint64_t sum = 0;
  bool contiguous = true;
  for (size_t i = 0; i < p.size(); ++i) {
    sum += p[i].bytes;
    if (p[i].offset != i * kRingPieceBytes || p[i].slot != static_cast<int>(i % 3)) contiguous = false;
  }
  Check(sum == 1065ull * kMiB + 300000 && contiguous, "pieces tile the stream with no gap and rotate through the 3 slots");
  Check(ChunkRing(0).empty(), "an empty stream has no pieces");
  Check(ChunkRing(kRingPieceBytes).size() == 1 && ChunkRing(kRingPieceBytes + 1).size() == 2 && ChunkRing(3 * kRingPieceBytes).size() == 3, "exact multiples and one-over");
  const std::vector<RingPiece> q = ChunkRing(10, 4, 2);
  Check(q.size() == 3 && q[2].bytes == 2 && q[2].offset == 8 && q[2].slot == 0 && q[1].slot == 1, "custom piece / slot counts");
  bool threw = false;
  try {
    ChunkRing(10, 0, 3);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  Check(threw, "a zero piece size is refused");
  // the real plan's stream
  const StateGeometry g = StateGeometry::FromRules(RealConfig());
  const ReshardPlan plan = ScatterPlan(g, PlanParams::ForSplit(g, 35), 0, 131072);
  const std::vector<RingPiece> xy = RingPiecesFrom(plan, kStageA), yx = RingPiecesFrom(plan, kStageB);
  Check(xy.size() == 9 && yx.size() == 10, "128k: 9 pieces X->Y (1065 MiB), 10 pieces Y->X (1184 MiB)");
}

// ---- helpers for the data tests ---------------------------------------------------------------------------------------
std::mt19937_64 g_rng(20261008);
std::vector<uint8_t> RandBytes(size_t n) {
  std::vector<uint8_t> v(n);
  for (uint8_t& b : v) b = static_cast<uint8_t>(g_rng());
  return v;
}
std::vector<uint16_t> RandU16(size_t n) {
  std::vector<uint16_t> v(n);
  for (uint16_t& b : v) b = static_cast<uint16_t>(g_rng());
  return v;
}
LiveImage RandomImage(const StateGeometry& g, int64_t blocks) {
  LiveImage x;
  for (const int64_t l : g.attn_layers) x.kv[l] = RandBytes(static_cast<size_t>(blocks * g.KvBlockBytesFull()));
  x.mtp_kv = RandBytes(static_cast<size_t>(blocks * g.KvBlockBytesFull()));
  for (const int64_t l : g.gdn_layers) {
    x.gdn_rec[l] = RandBytes(static_cast<size_t>(g.RecurrentBytesFull()));
    x.gdn_conv[l] = RandU16(static_cast<size_t>(g.conv_dim_full * g.conv_live));
  }
  return x;
}

void RoundTrip() {
  const ModelConfig cfg = TinyConfig();
  const StateGeometry g = StateGeometry::FromRules(cfg, /*block_tokens=*/4);
  for (int trial = 0; trial < 50; ++trial) {
    const LiveImage x = RandomImage(g, 1 + trial % 7);
    const LiveImage r0 = ReshardRef(g, x, 0), r1 = ReshardRef(g, x, 1);
    Check(GatherRef(g, r0, r1) == x, "round trip: gather(reshard(x, 0), reshard(x, 1)) == x on random data");
    // sizes: each rank holds exactly half
    bool halves = true;
    for (const auto& [l, bytes] : x.kv) halves = halves && r0.kv.at(l).size() * 2 == bytes.size() && r1.kv.at(l).size() * 2 == bytes.size();
    for (const auto& [l, bytes] : x.gdn_rec) halves = halves && r0.gdn_rec.at(l).size() * 2 == bytes.size();
    for (const auto& [l, c] : x.gdn_conv) halves = halves && r0.gdn_conv.at(l).size() * 2 == c.size();
    halves = halves && r0.mtp_kv.size() * 2 == x.mtp_kv.size();
    Check(halves, "every rank image is half the full image");
    Check(r0 != r1, "the two ranks hold different halves");
  }
  // the ranks must carry the same layers
  LiveImage x = RandomImage(g, 2);
  LiveImage a = ReshardRef(g, x, 0), b = ReshardRef(g, x, 1);
  b.kv.erase(b.kv.begin());
  bool threw = false;
  try {
    (void)GatherRef(g, a, b);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  Check(threw, "gather refuses two ranks that carry different layers");
  threw = false;
  x.gdn_rec.begin()->second.pop_back();
  try {
    (void)ReshardRef(g, x, 0);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  Check(threw, "reshard refuses a recurrent image of the wrong size");
}

// By hand: 4 kv heads of 1 head_dim (2 B per token-head), 2-token blocks (4 B per head-block, 16 B per block); 4 v-heads of 1 x 1 fp32
// (4 B per head); 2 key heads -> 8 conv channels, 3 live entries each.
void HandChecked() {
  const ModelConfig cfg = HandConfig();
  const StateGeometry g = StateGeometry::FromRules(cfg, /*block_tokens=*/2);
  Check(g.KvBlockBytesFull() == 16 && g.KvBlockBytesRank(0) == 8 && g.RecurrentBytesFull() == 16 && g.conv_dim_full == 8 && g.gqa_repeats == 2,
        "hand model: 16 B full KV block, 8 B rank block, 16 B recurrent slot, 8 conv channels, GQA 2");
  Check(g.ranks[0].v_head_begin == 0 && g.ranks[0].v_heads == 2 && g.ranks[1].v_head_begin == 2 && g.ranks[1].v_heads == 2, "hand model: v-heads [0,2) / [2,4)");
  Check(g.ranks[1].conv.size() == 3 && g.ranks[1].conv[0].full_begin == 1 && g.ranks[1].conv[1].full_begin == 3 && g.ranks[1].conv[2].full_begin == 6 &&
            g.ranks[1].conv[2].count == 2 && g.ranks[1].conv[2].rank_begin == 2,
        "hand model: rank 1 takes channels q {1}, k {3}, v {6, 7} -> local 0, 1, 2, 3");
  LiveImage x;
  x.kv[3].resize(32);  // two blocks
  for (size_t i = 0; i < 32; ++i) x.kv[3][i] = static_cast<uint8_t>(i);
  for (const int64_t l : {0, 1, 2}) {
    x.gdn_rec[l].resize(16);
    for (size_t i = 0; i < 16; ++i) x.gdn_rec[l][i] = static_cast<uint8_t>(100 + i);
    x.gdn_conv[l].resize(24);
    for (size_t c = 0; c < 8; ++c) {
      for (size_t e = 0; e < 3; ++e) x.gdn_conv[l][c * 3 + e] = static_cast<uint16_t>(c * 10 + e);
    }
  }
  x.mtp_kv = x.kv[3];
  const LiveImage r0 = ReshardRef(g, x, 0), r1 = ReshardRef(g, x, 1);
  // block 0 is bytes 0..15: head0 = 0..3, head1 = 4..7, head2 = 8..11, head3 = 12..15; block 1 = 16..31.
  Check(r0.kv.at(3) == std::vector<uint8_t>({0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23}), "hand: rank 0 holds heads 0,1 of each block");
  Check(r1.kv.at(3) == std::vector<uint8_t>({8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31}), "hand: rank 1 holds heads 2,3 of each block");
  Check(r1.mtp_kv == r1.kv.at(3), "hand: the MTP head KV splits the same way");
  Check(r0.gdn_rec.at(1) == std::vector<uint8_t>({100, 101, 102, 103, 104, 105, 106, 107}) &&
            r1.gdn_rec.at(1) == std::vector<uint8_t>({108, 109, 110, 111, 112, 113, 114, 115}),
        "hand: v-heads 0,1 / 2,3 are the two halves of the recurrent slot");
  Check(r0.gdn_conv.at(2) == std::vector<uint16_t>({0, 1, 2, 20, 21, 22, 40, 41, 42, 50, 51, 52}), "hand: rank 0 conv channels {0, 2, 4, 5}");
  Check(r1.gdn_conv.at(2) == std::vector<uint16_t>({10, 11, 12, 30, 31, 32, 60, 61, 62, 70, 71, 72}), "hand: rank 1 conv channels {1, 3, 6, 7}");
  Check(GatherRef(g, r0, r1) == x, "hand: gather restores the image");

  // Plan ops on the hand model (k = 2: stage A = layers 0, 1; stage B = layers 2, 3 + MTP), rows [0, 2) = block 0.
  const PlanParams p = PlanParams::ForSplit(g, 2);
  const ReshardPlan plan = ScatterPlan(g, p, 0, 2);
  Check(p.stage[0].gdn.size() == 2 && p.stage[0].attn.empty() && p.stage[1].gdn.size() == 1 && p.stage[1].attn == std::vector<int64_t>({3}), "hand: layer ownership");
  Check(plan.ops.size() == 16, "hand: stage A 8 ops (2 GDN layers x 2 ranks x recurrent+conv), stage B 4 GDN + 2 KV + 2 MTP");
  for (const CopyOp& op : plan.ops) {
    if (op.kind == StateKind::kKv && op.layer == 3 && op.rank == 1) {
      const Run2D& r = op.runs.at(0);
      Check(r.full_off == 8 && r.full_pitch == 16 && r.rank_off == 0 && r.rank_pitch == 8 && r.width == 8 && r.height == 1 && !op.local,
            "hand: stage B -> rank 1 KV: bytes [8,16) of block 0, via the host (rank 1 is the other card)");
    }
    if (op.kind == StateKind::kKv && op.layer == 3 && op.rank == 0) Check(op.local, "hand: stage B -> rank 0 is local");
  }
}

// ---- a host executor of the plan's strided copies, against the reference ----------------------------------------------------
struct Side {  // the buffers of one holder (a stage or a rank), in DEVICE layout: conv lines are pitched, KV in whole blocks
  std::map<int64_t, std::vector<uint8_t>> kv, rec, conv;
  std::vector<uint8_t> mtp;
};
std::vector<uint8_t> Pitch(const std::vector<uint16_t>& compact, int64_t channels, int64_t live, int64_t pitch, uint16_t fill) {
  std::vector<uint16_t> out(static_cast<size_t>(channels * pitch), fill);
  for (int64_t c = 0; c < channels; ++c) {
    for (int64_t e = 0; e < live; ++e) out[static_cast<size_t>(c * pitch + e)] = compact[static_cast<size_t>(c * live + e)];
  }
  std::vector<uint8_t> bytes(out.size() * 2);
  std::memcpy(bytes.data(), out.data(), bytes.size());
  return bytes;
}
std::vector<uint16_t> Unpitch(const std::vector<uint8_t>& bytes, int64_t channels, int64_t live, int64_t pitch) {
  std::vector<uint16_t> all(bytes.size() / 2);
  std::memcpy(all.data(), bytes.data(), bytes.size());
  std::vector<uint16_t> out(static_cast<size_t>(channels * live));
  for (int64_t c = 0; c < channels; ++c) {
    for (int64_t e = 0; e < live; ++e) out[static_cast<size_t>(c * live + e)] = all[static_cast<size_t>(c * pitch + e)];
  }
  return out;
}
// The device-layout side for the layers in `keep` (nullptr = all), conv lines `pitch` entries long, non-live entries `fill`.
Side MakeSide(const StateGeometry& g, const LiveImage& img, int64_t pitch, uint16_t fill, const std::vector<int64_t>* keep, bool with_mtp, int rank) {
  Side s;
  const auto wanted = [&](int64_t l) {
    if (keep == nullptr) return true;
    for (const int64_t k : *keep) {
      if (k == l) return true;
    }
    return false;
  };
  for (const auto& [l, b] : img.kv) {
    if (wanted(l)) s.kv[l] = b;
  }
  for (const auto& [l, b] : img.gdn_rec) {
    if (wanted(l)) s.rec[l] = b;
  }
  const int64_t channels = rank < 0 ? g.conv_dim_full : g.ConvChannelsRank(rank);
  for (const auto& [l, c] : img.gdn_conv) {
    if (wanted(l)) s.conv[l] = Pitch(c, channels, g.conv_live, pitch, fill);
  }
  if (with_mtp) s.mtp = img.mtp_kv;
  return s;
}
void Apply(const ReshardPlan& plan, std::vector<Side>* stage, std::vector<Side>* rank) {
  for (const CopyOp& op : plan.ops) {
    Side& st = (*stage)[static_cast<size_t>(op.stage)];
    Side& rk = (*rank)[static_cast<size_t>(op.rank)];
    std::vector<uint8_t>* full = nullptr;
    std::vector<uint8_t>* rnk = nullptr;
    switch (op.kind) {
      case StateKind::kKv: full = &st.kv.at(op.layer); rnk = &rk.kv.at(op.layer); break;
      case StateKind::kMtpKv: full = &st.mtp; rnk = &rk.mtp; break;
      case StateKind::kGdnRecurrent: full = &st.rec.at(op.layer); rnk = &rk.rec.at(op.layer); break;
      case StateKind::kGdnConv: full = &st.conv.at(op.layer); rnk = &rk.conv.at(op.layer); break;
    }
    for (const Run2D& r : op.runs) {
      for (uint64_t i = 0; i < r.height; ++i) {
        const uint64_t fo = r.full_off + i * r.full_pitch, ro = r.rank_off + i * r.rank_pitch;
        if (fo + r.width > full->size() || ro + r.width > rnk->size()) throw std::out_of_range("plan run reaches past its buffer");
        if (plan.dir == Dir::kStageToRank) {
          std::memcpy(rnk->data() + ro, full->data() + fo, r.width);
        } else {
          std::memcpy(full->data() + fo, rnk->data() + ro, r.width);
        }
      }
    }
  }
}
// bytes of `over` in blocks `r`, the rest from `base`.
std::vector<uint8_t> Blend(std::vector<uint8_t> base, const std::vector<uint8_t>& over, size_t block_bytes, const BlockRange& r) {
  for (int64_t b = r.first; b <= r.last; ++b) {
    std::memcpy(base.data() + static_cast<size_t>(b) * block_bytes, over.data() + static_cast<size_t>(b) * block_bytes, block_bytes);
  }
  return base;
}

constexpr int64_t kStagePitch = 3, kRankPitch = 5;
constexpr uint16_t kFill = 0xBEEF;

struct Harness {
  StateGeometry g;
  PlanParams p;
  int64_t blocks;
  LiveImage x, y;  // x = the new truth, y = the stale content of the destination
  std::vector<Side> stage, rank;
  Harness(const ModelConfig& cfg, int64_t split, int64_t blocks_in) : g(StateGeometry::FromRules(cfg, 4)), p(PlanParams::ForSplit(g, split)), blocks(blocks_in) {
    p.stage_conv_pitch = kStagePitch;
    p.rank_conv_pitch = kRankPitch;
    x = RandomImage(g, blocks);
    y = RandomImage(g, blocks);
  }
  // scatter: the stages hold x, the ranks hold y
  void SetupScatter() {
    stage.clear();
    rank.clear();
    for (int s = 0; s < 2; ++s) {
      std::vector<int64_t> keep = p.stage[s].attn;
      keep.insert(keep.end(), p.stage[s].gdn.begin(), p.stage[s].gdn.end());
      stage.push_back(MakeSide(g, x, kStagePitch, kFill, &keep, s == kStageB, -1));
    }
    for (int r = 0; r < 2; ++r) rank.push_back(MakeSide(g, ReshardRef(g, y, r), kRankPitch, kFill, nullptr, true, r));
  }
  // gather: the ranks hold x, the stages hold y
  void SetupGather() {
    stage.clear();
    rank.clear();
    for (int s = 0; s < 2; ++s) {
      std::vector<int64_t> keep = p.stage[s].attn;
      keep.insert(keep.end(), p.stage[s].gdn.begin(), p.stage[s].gdn.end());
      stage.push_back(MakeSide(g, y, kStagePitch, kFill, &keep, s == kStageB, -1));
    }
    for (int r = 0; r < 2; ++r) rank.push_back(MakeSide(g, ReshardRef(g, x, r), kRankPitch, kFill, nullptr, true, r));
  }
  uint16_t ConvLive(const Side& s, int64_t layer, int64_t channel, int64_t e, int64_t pitch) const {
    uint16_t v;
    std::memcpy(&v, s.conv.at(layer).data() + static_cast<size_t>((channel * pitch + e) * 2), 2);
    return v;
  }
};

// Every rank buffer equals ReshardRef of (x on the blocks that were copied, y elsewhere); GDN wholly x; pad entries untouched.
bool RankMatches(const Harness& h, const BlockRange& kvb, const BlockRange& mtpb) {
  const size_t rb = static_cast<size_t>(h.g.KvBlockBytesRank(0));
  for (int r = 0; r < 2; ++r) {
    const LiveImage xr = ReshardRef(h.g, h.x, r), yr = ReshardRef(h.g, h.y, r);
    const Side& s = h.rank[static_cast<size_t>(r)];
    for (const int64_t l : h.g.attn_layers) {
      if (s.kv.at(l) != Blend(yr.kv.at(l), xr.kv.at(l), rb, kvb)) return false;
    }
    if (s.mtp != Blend(yr.mtp_kv, xr.mtp_kv, rb, mtpb)) return false;
    for (const int64_t l : h.g.gdn_layers) {
      if (s.rec.at(l) != xr.gdn_rec.at(l)) return false;
      if (Unpitch(s.conv.at(l), h.g.ConvChannelsRank(r), h.g.conv_live, kRankPitch) != xr.gdn_conv.at(l)) return false;
      for (int64_t c = 0; c < h.g.ConvChannelsRank(r); ++c) {
        for (int64_t e = h.g.conv_live; e < kRankPitch; ++e) {
          if (h.ConvLive(s, l, c, e, kRankPitch) != kFill) return false;  // the non-live entries of the speculating line are not state
        }
      }
    }
  }
  return true;
}

void ExecutorVsReference() {
  const ModelConfig cfg = TinyConfig();
  for (const int64_t split : {1, 3, 4, 5, 7}) {
    Harness h(cfg, split, 6);  // 6 blocks of 4 rows = 24 rows
    // full scatter of the whole image
    h.SetupScatter();
    ReshardPlan plan = ScatterPlan(h.g, h.p, 0, 24);
    Apply(plan, &h.stage, &h.rank);
    Check(RankMatches(h, BlockRange{0, 5}, BlockRange{0, 5}), "executor: a full scatter equals ReshardRef(x) on both ranks, pad entries untouched");
    // a warm scatter: rows [9, 14) of a 4-row-block image: layer KV blocks 2..3
    h.SetupScatter();
    plan = ScatterPlan(h.g, h.p, 9, 14);
    Apply(plan, &h.stage, &h.rank);
    Check(RankMatches(h, KvScatterBlocks(9, 14, 4), MtpScatterBlocks(9, 14, 4)), "executor: a warm scatter changes exactly the planned blocks (layer KV 2..3, MTP 2..3)");
    // p0 on a block boundary: the MTP range starts one block earlier than the layer range
    h.SetupScatter();
    plan = ScatterPlan(h.g, h.p, 8, 14);
    Apply(plan, &h.stage, &h.rank);
    Check(RankMatches(h, BlockRange{2, 3}, BlockRange{1, 3}), "executor: p0 = 8 = 2 blocks: layer KV blocks 2..3, MTP blocks 1..3 (row 7 is the boundary)");
    Check(!RankMatches(h, BlockRange{2, 3}, BlockRange{2, 3}), "NEGATIVE CONTROL: the executor result is NOT the layer-KV block range for the MTP head");

    // gather: stale rows [0, 14) (tracker plan after a TP-only stretch from 0) and the MTP block of row 13
    TpMasterTracker t(4, true);
    t.TpOnly(0);
    const TpMasterTracker::SyncPlan sp = t.PlanSync(14);
    h.SetupGather();
    plan = GatherPlan(h.g, h.p, sp);
    Apply(plan, &h.stage, &h.rank);
    bool ok = true;
    const size_t fb = static_cast<size_t>(h.g.KvBlockBytesFull());
    for (int s = 0; s < 2; ++s) {
      const Side& st = h.stage[static_cast<size_t>(s)];
      for (const int64_t l : h.p.stage[s].attn) ok = ok && st.kv.at(l) == Blend(h.y.kv.at(l), h.x.kv.at(l), fb, BlockRange{0, 3});
      for (const int64_t l : h.p.stage[s].gdn) {
        ok = ok && st.rec.at(l) == h.x.gdn_rec.at(l);
        ok = ok && Unpitch(st.conv.at(l), h.g.conv_dim_full, h.g.conv_live, kStagePitch) == h.x.gdn_conv.at(l);
      }
    }
    ok = ok && h.stage[kStageB].mtp == Blend(h.y.mtp_kv, h.x.mtp_kv, fb, BlockRange{3, 3}) && h.stage[kStageA].mtp.empty();
    Check(ok, "executor: the warm gather (rows [0,14), GDN, MTP block 3) rebuilds the stages' full-head state from the two ranks");
  }
}

// ---- negative controls -------------------------------------------------------------------------------------------------
void NegativeControls() {
  const ModelConfig cfg = TinyConfig();
  const StateGeometry good = StateGeometry::FromRules(cfg, 4);
  const LiveImage x = RandomImage(good, 3);
  const LiveImage ref1 = ReshardRef(good, x, 1);

  {  // (1) swap the head halves
    StateGeometry bad = good;
    std::swap(bad.ranks[0].kv_head_begin, bad.ranks[1].kv_head_begin);
    Check(!VerifyAgainstRules(bad, cfg).empty(), "NEGATIVE CONTROL: swapped KV head halves disagree with the attn.k / v / qg / o rules");
    Check(ReshardRef(bad, x, 1) != ref1, "NEGATIVE CONTROL: ... and change the rank image");
    StateGeometry bad2 = good;
    std::swap(bad2.ranks[0].v_head_begin, bad2.ranks[1].v_head_begin);
    const std::string why = VerifyAgainstRules(bad2, cfg);
    Check(why.find("state mapping != weight mapping") != std::string::npos, "NEGATIVE CONTROL: swapped GDN v-head halves are reported as a state != weight mismatch");
    std::fprintf(stderr, "  (reported: %s)\n", why.c_str());
  }
  {  // (2) drop a conv segment
    StateGeometry bad = good;
    bad.ranks[1].conv.pop_back();
    Check(!VerifyAgainstRules(bad, cfg).empty(), "NEGATIVE CONTROL: a dropped conv segment disagrees with conv1d_weight / in_proj_qkv");
    const LiveImage r0 = ReshardRef(good, x, 0);
    bool lost = true;
    try {
      lost = !(GatherRef(bad, r0, ReshardRef(bad, x, 1)) == x);  // the v channels of rank 1 never come back
    } catch (const std::exception&) {
      lost = true;
    }
    Check(lost, "NEGATIVE CONTROL: ... and the round trip loses the channels");
    StateGeometry bad2 = good;
    bad2.ranks[0].conv[1].full_begin += 1;  // off by one channel
    Check(!VerifyAgainstRules(bad2, cfg).empty(), "NEGATIVE CONTROL: a conv segment shifted by one channel is caught");
  }
  {  // (3) skip the MTP half
    Harness h(cfg, 3, 3);
    h.SetupScatter();
    ReshardPlan plan = ScatterPlan(h.g, h.p, 0, 12);
    ReshardPlan cut;
    cut.dir = plan.dir;
    for (const CopyOp& op : plan.ops) {
      if (!(op.kind == StateKind::kMtpKv && op.rank == 1)) cut.ops.push_back(op);
    }
    Apply(cut, &h.stage, &h.rank);
    Check(!RankMatches(h, BlockRange{0, 2}, BlockRange{0, 2}), "NEGATIVE CONTROL: skipping the MTP half leaves rank 1's MTP KV stale");
  }
  {  // (4) off-by-one block
    Harness h(cfg, 3, 3);
    h.SetupScatter();
    ReshardPlan plan = ScatterPlan(h.g, h.p, 0, 12);
    for (CopyOp& op : plan.ops) {
      if (op.kind == StateKind::kKv) {
        op.runs[0].full_off += op.runs[0].full_pitch;  // starts one block late
        op.runs[0].rank_off += op.runs[0].rank_pitch;
        op.runs[0].height -= 1;
      }
    }
    Apply(plan, &h.stage, &h.rank);
    Check(!RankMatches(h, BlockRange{0, 2}, BlockRange{0, 2}), "NEGATIVE CONTROL: an off-by-one first block leaves block 0 stale");
  }
  {  // (5) the wrong rank mapping: every op delivers rank r's half into the OTHER rank's buffer (sizes are symmetric, the bytes are not)
    Harness h(cfg, 3, 3);
    h.SetupScatter();
    ReshardPlan wrong = ScatterPlan(h.g, h.p, 0, 12);
    for (CopyOp& op : wrong.ops) op.rank = 1 - op.rank;
    Apply(wrong, &h.stage, &h.rank);
    Check(!RankMatches(h, BlockRange{0, 2}, BlockRange{0, 2}), "NEGATIVE CONTROL: a wrong stage/rank mapping puts each half on the wrong rank");
    const StateGeometry g = StateGeometry::FromRules(RealConfig());
    const PlanTotals a = Totals(ScatterPlan(g, PlanParams::ForSplit(g, 35), 0, 8192));
    Check(a.CrossFrom(kStageA) != a.CrossFrom(kStageB), "the two directions carry different amounts (so a swapped direction is visible in the sizes)");
  }
  {  // (6) a plan for the wrong split moves the wrong layers
    const StateGeometry g = StateGeometry::FromRules(RealConfig());
    const PlanTotals a = Totals(ScatterPlan(g, PlanParams::ForSplit(g, 35), 0, 8192)), b = Totals(ScatterPlan(g, PlanParams::ForSplit(g, 32), 0, 8192));
    Check(a.CrossFrom(kStageA) != b.CrossFrom(kStageA), "NEGATIVE CONTROL: k = 32 vs 35 changes the GDN layers each direction carries (24 vs 27)");
  }
  {  // refusals
    bool threw = false;
    try {
      (void)LayersOfStage(good, 0, 0);
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    try {
      (void)LayersOfStage(good, good.num_layers, 1);
      threw = false;
    } catch (const std::invalid_argument&) {
    }
    Check(threw, "a split outside [1, N-1] is refused");
    ModelConfig shard = cfg;
    shard.tp_world = 2;
    threw = false;
    try {
      (void)StateGeometry::FromRules(shard);
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    Check(threw, "FromRules refuses a rank-local config");
    ModelConfig gemma = cfg;
    gemma.arch = Arch::kGemma4;
    threw = false;
    try {
      (void)StateGeometry::FromRules(gemma);
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    Check(threw, "FromRules refuses a model of another family");
    threw = false;
    try {
      (void)ScatterPlan(good, PlanParams::ForSplit(good, 3), 5, 5);
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    Check(threw, "an empty call has no scatter plan");
  }
}

}  // namespace

int main() {
  RealMapping();
  Sizes();
  Ring();
  RoundTrip();
  HandChecked();
  ExecutorVsReference();
  NegativeControls();
  if (g_fails != 0) {
    std::fprintf(stderr, "test_reshard_plan_cpu: %d FAILED\n", g_fails);
    return 1;
  }
  std::fprintf(stderr, "test_reshard_plan_cpu: PASS\n");
  return 0;
}
