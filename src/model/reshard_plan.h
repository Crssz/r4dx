// HIP-free arithmetic of the hybrid mode's reshard (docs/pp-tp2-hybrid.md 3): the exact mapping between the FULL-head live
// state a pipeline stage holds (the TP=1 layout) and TP=2 rank r's head-split image of it, the copy ranges and sizes of
// each direction, the ring chunking of the host hop and a host reference (ReshardRef / GatherRef) the on-device copies are
// checked against. Header-only so a CPU test (tests/model/test_reshard_plan_cpu.cpp) runs it without a device.
//
// The mapping is DERIVED, not restated: StateGeometry::FromRules reads the head and channel ownership out of the same
// tp::RuleFor / tp::RankRows tables the weight loader (shard_loader.h) cuts the weights with, and VerifyAgainstRules
// re-checks it against EVERY rule that owns a head or a channel (conv1d_weight / in_proj_qkv / in_proj_z / A_log /
// dt_bias / in_proj_a / in_proj_b / out_proj, attn.k / v / their descales / qg / o, on a body layer and on the MTP head), so
// a change to one shard rule that is not made to the others is an error at the first FromRules, not a silent divergence
// between the weights and the state.
//
// Names: stage 0 = A = layers [0, k) = card X (the desktop card, HIP device 0) = TP rank 1; stage 1 = B = layers [k, N) +
// the MTP head = card Y (the headless card) = TP rank 0 (docs/pp-tp2-hybrid.md, names).
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "hybrid_sync.h"
#include "model_config.h"
#include "tp/tp_shard.h"

namespace r4dx::model::hybrid {

constexpr int kWorld = 2;
constexpr int kStageA = 0;
constexpr int kStageB = 1;
// The two placements already agree (docs/tp.md 9.2, docs/pp-prefill.md 4.2): stage B is rank 0, stage A is rank 1.
inline int RankOfStage(int stage) { return 1 - stage; }
inline int StageOfRank(int rank) { return 1 - rank; }

// ---- the geometry -----------------------------------------------------------------------------------------------------
struct ConvSegment {
  int64_t full_begin = 0;  // first global conv channel of the run
  int64_t rank_begin = 0;  // first local channel (the runs are concatenated in segment order)
  int64_t count = 0;
};
struct RankMap {
  int64_t kv_head_begin = 0, kv_heads = 0;  // global KV heads [begin, begin + heads)
  int64_t v_head_begin = 0, v_heads = 0;    // global GDN value heads
  std::vector<ConvSegment> conv;            // q, k, v channel runs
};

struct StateGeometry {
  int world = kWorld;
  int64_t num_layers = 0;
  std::vector<int64_t> attn_layers, gdn_layers;  // global layer indices, ascending
  // KV: one layer's cache is [block][heads][block_tokens][2 * head_dim] fp8, head-major inside a block.
  int64_t block_tokens = 16;
  int64_t kv_heads_full = 0, head_dim = 0;
  // GDN: the recurrent slot is [v_heads][V][K] fp32; a conv line holds `conv_live` live bf16 entries per channel.
  int64_t v_heads_full = 0, v_head_dim = 0, k_head_dim = 0;
  int64_t key_heads_full = 0, key_head_dim = 0, gqa_repeats = 1;  // q/k channel geometry (GqaRepeats)
  int64_t conv_dim_full = 0, conv_live = 0;
  int64_t attn_gqa = 1;  // q heads per kv head
  std::vector<RankMap> ranks;

  int64_t KvTokenHeadBytes() const { return 2 * head_dim; }  // K and V rows, fp8
  int64_t KvHeadBlockBytes() const { return block_tokens * KvTokenHeadBytes(); }
  int64_t KvBlockBytesFull() const { return kv_heads_full * KvHeadBlockBytes(); }
  int64_t KvBlockBytesRank(int rank) const { return ranks.at(static_cast<size_t>(rank)).kv_heads * KvHeadBlockBytes(); }
  int64_t RecurrentHeadBytes() const { return v_head_dim * k_head_dim * 4; }
  int64_t RecurrentBytesFull() const { return v_heads_full * RecurrentHeadBytes(); }
  int64_t RecurrentBytesRank(int rank) const { return ranks.at(static_cast<size_t>(rank)).v_heads * RecurrentHeadBytes(); }
  int64_t ConvLiveBytesPerChannel() const { return conv_live * 2; }
  int64_t ConvChannelsRank(int rank) const {
    int64_t n = 0;
    for (const ConvSegment& s : ranks.at(static_cast<size_t>(rank)).conv) n += s.count;
    return n;
  }

  // Derives the geometry from the loader's rule tables. `global` is the UNSHARDED config. Throws std::invalid_argument for a
  // model the hybrid does not serve and std::logic_error when the rules disagree with each other (VerifyAgainstRules).
  static StateGeometry FromRules(const ModelConfig& global, int64_t block_tokens = 16);
};

namespace detail {

inline std::string LayerName(int64_t layer, const char* tail) { return "text.layers." + std::to_string(layer) + "." + tail; }

inline std::vector<tp::Range> RowsOf(const std::string& name, const ModelConfig& g, int rank) {
  return tp::RankRows(tp::RuleFor(name, g), kWorld, rank);
}

inline std::string Fmt(const std::vector<tp::Range>& v) {
  std::string s;
  for (const tp::Range& r : v) s += "[" + std::to_string(r.begin) + ",+" + std::to_string(r.count) + ")";
  return s.empty() ? "[]" : s;
}

inline bool Same(const std::vector<tp::Range>& a, const std::vector<tp::Range>& b) {
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); ++i) {
    if (a[i].begin != b[i].begin || a[i].count != b[i].count) return false;
  }
  return true;
}

}  // namespace detail

// Re-derives, from every rule that owns a head or a channel, what rank r holds and compares it with the geometry. Returns ""
// when the state mapping equals the weight mapping for all of them, else the first disagreement. FromRules runs it on what
// it derived; the test runs it on perturbed geometries (the negative controls).
inline std::string VerifyAgainstRules(const StateGeometry& s, const ModelConfig& g);

inline StateGeometry StateGeometry::FromRules(const ModelConfig& g, int64_t block_tokens) {
  if (g.arch != Arch::kQwen35) throw std::invalid_argument("hybrid::StateGeometry: only the Qwen3.5 family has this state layout");
  if (g.tp_world != 1) throw std::invalid_argument("hybrid::StateGeometry: needs the unsharded config (tp_world 1)");
  if (block_tokens < 1) throw std::invalid_argument("hybrid::StateGeometry: block_tokens must be >= 1");
  StateGeometry s;
  s.num_layers = g.num_hidden_layers;
  s.block_tokens = block_tokens;
  s.kv_heads_full = g.num_key_value_heads;
  s.head_dim = g.head_dim;
  s.attn_gqa = g.AttnGqa();
  s.v_heads_full = g.linear_num_value_heads;
  s.v_head_dim = g.linear_value_head_dim;
  s.k_head_dim = g.linear_key_head_dim;
  s.key_heads_full = g.linear_num_key_heads;
  s.key_head_dim = g.linear_key_head_dim;
  s.gqa_repeats = g.GqaRepeats();
  s.conv_dim_full = g.ConvDim();
  s.conv_live = g.linear_conv_kernel_dim - 1;
  for (int64_t i = 0; i < g.num_hidden_layers; ++i) (g.IsGdnLayer(i) ? s.gdn_layers : s.attn_layers).push_back(i);
  if (s.attn_layers.empty() || s.gdn_layers.empty()) {
    throw std::invalid_argument("hybrid::StateGeometry: needs both full-attention and GDN layers");
  }
  const int64_t gdn0 = s.gdn_layers.front(), attn0 = s.attn_layers.front();
  for (int r = 0; r < kWorld; ++r) {
    RankMap m;
    // KV heads: attn.k's rows are kv_heads x head_dim, split into `world` equal contiguous parts.
    const std::vector<tp::Range> k = detail::RowsOf(detail::LayerName(attn0, "attn.k"), g, r);
    if (k.size() != 1 || k[0].begin % s.head_dim != 0 || k[0].count % s.head_dim != 0) {
      throw std::logic_error("hybrid::StateGeometry: attn.k does not split on whole KV heads");
    }
    m.kv_head_begin = k[0].begin / s.head_dim;
    m.kv_heads = k[0].count / s.head_dim;
    // GDN value heads: A_log has one row per v-head.
    const std::vector<tp::Range> a = detail::RowsOf(detail::LayerName(gdn0, "gdn.A_log"), g, r);
    if (a.size() != 1) throw std::logic_error("hybrid::StateGeometry: gdn.A_log is not one contiguous run of v-heads");
    m.v_head_begin = a[0].begin;
    m.v_heads = a[0].count;
    // Conv channels: the conv1d weight's rows are the q | k | v channels, one segment each.
    int64_t at = 0;
    for (const tp::Range& c : detail::RowsOf(detail::LayerName(gdn0, "gdn.conv1d_weight"), g, r)) {
      m.conv.push_back({c.begin, at, c.count});
      at += c.count;
    }
    s.ranks.push_back(m);
  }
  // The ranks must partition the heads and the channels exactly (a gap or an overlap would lose or double state).
  int64_t kv = 0, vh = 0, ch = 0;
  for (const RankMap& m : s.ranks) {
    kv += m.kv_heads;
    vh += m.v_heads;
    for (const ConvSegment& c : m.conv) ch += c.count;
  }
  if (kv != s.kv_heads_full || vh != s.v_heads_full || ch != s.conv_dim_full) {
    throw std::logic_error("hybrid::StateGeometry: the ranks' heads / channels do not add up to the model's");
  }
  const std::string why = VerifyAgainstRules(s, g);
  if (!why.empty()) throw std::logic_error("hybrid::StateGeometry: " + why);
  return s;
}

inline std::string VerifyAgainstRules(const StateGeometry& s, const ModelConfig& g) {
  using detail::Fmt;
  using detail::LayerName;
  using detail::RowsOf;
  using detail::Same;
  const auto bad = [](const std::string& tensor, int rank, const std::string& got, const std::string& want) {
    return "state mapping != weight mapping: " + tensor + " rank " + std::to_string(rank) + " is " + got +
           ", the state expects " + want;
  };
  const int64_t gdn0 = s.gdn_layers.front(), attn0 = s.attn_layers.front();
  for (int r = 0; r < kWorld; ++r) {
    const RankMap& m = s.ranks.at(static_cast<size_t>(r));
    const std::vector<tp::Range> vheads = {{m.v_head_begin, m.v_heads}};
    std::vector<tp::Range> conv;
    for (const ConvSegment& c : m.conv) conv.push_back({c.full_begin, c.count});
    // GDN
    for (const char* t : {"gdn.conv1d_weight", "gdn.in_proj_qkv"}) {
      const std::vector<tp::Range> got = RowsOf(LayerName(gdn0, t), g, r);
      if (!Same(got, conv)) return bad(LayerName(gdn0, t), r, Fmt(got), Fmt(conv));
    }
    for (const char* t : {"gdn.A_log", "gdn.dt_bias", "gdn.in_proj_a", "gdn.in_proj_b"}) {
      const std::vector<tp::Range> got = RowsOf(LayerName(gdn0, t), g, r);
      if (!Same(got, vheads)) return bad(LayerName(gdn0, t), r, Fmt(got), Fmt(vheads));
    }
    const std::vector<tp::Range> zrows = {{m.v_head_begin * s.v_head_dim, m.v_heads * s.v_head_dim}};
    const std::vector<tp::Range> z = RowsOf(LayerName(gdn0, "gdn.in_proj_z"), g, r);
    if (!Same(z, zrows)) return bad(LayerName(gdn0, "gdn.in_proj_z"), r, Fmt(z), Fmt(zrows));
    const tp::Range oc = tp::RankCols(tp::RuleFor(LayerName(gdn0, "gdn.out_proj"), g), kWorld, r);
    if (oc.begin != zrows[0].begin || oc.count != zrows[0].count) {
      const std::vector<tp::Range> got = {oc};
      return bad(LayerName(gdn0, "gdn.out_proj") + " cols", r, Fmt(got), Fmt(zrows));
    }
    // The conv channels must be the ones of THESE v-heads: the v segment is the v-heads' channels, the q / k segments those
    // of the key heads they read (v-head h reads key head h / GqaRepeats).
    if (m.conv.size() != 3) return "state mapping: the conv1d rule is not the three q | k | v segments";
    const int64_t key_dim = s.key_heads_full * s.key_head_dim;
    if (m.v_head_begin % s.gqa_repeats != 0 || m.v_heads % s.gqa_repeats != 0) {
      return "state mapping: a rank's v-heads do not cover whole key heads";
    }
    const int64_t kq = (m.v_head_begin / s.gqa_repeats) * s.key_head_dim, kn = (m.v_heads / s.gqa_repeats) * s.key_head_dim;
    const std::vector<tp::Range> want_conv = {
        {kq, kn}, {key_dim + kq, kn}, {2 * key_dim + m.v_head_begin * s.v_head_dim, m.v_heads * s.v_head_dim}};
    if (!Same(conv, want_conv)) return bad(LayerName(gdn0, "gdn.conv1d_weight") + " (vs the v-heads)", r, Fmt(conv), Fmt(want_conv));
    // Attention (a body layer and the MTP head share the rules)
    const std::vector<tp::Range> kvrows = {{m.kv_head_begin * s.head_dim, m.kv_heads * s.head_dim}};
    const std::vector<tp::Range> kvh = {{m.kv_head_begin, m.kv_heads}};
    const int64_t q0 = m.kv_head_begin * s.attn_gqa, qn = m.kv_heads * s.attn_gqa;  // q heads that read these kv heads
    const std::vector<tp::Range> qgrows = {{2 * q0 * s.head_dim, 2 * qn * s.head_dim}};
    for (const std::string& p : {LayerName(attn0, "attn."), std::string("mtp.attn.")}) {
      for (const char* t : {"k", "v"}) {
        const std::vector<tp::Range> got = RowsOf(p + t, g, r);
        if (!Same(got, kvrows)) return bad(p + t, r, Fmt(got), Fmt(kvrows));
      }
      for (const char* t : {"k_descale", "v_descale"}) {
        const std::vector<tp::Range> got = RowsOf(p + t, g, r);
        if (!Same(got, kvh)) return bad(p + t, r, Fmt(got), Fmt(kvh));
      }
      const std::vector<tp::Range> qg = RowsOf(p + "qg", g, r);
      if (!Same(qg, qgrows)) return bad(p + "qg", r, Fmt(qg), Fmt(qgrows));
      const tp::Range o = tp::RankCols(tp::RuleFor(p + "o", g), kWorld, r);
      if (o.begin != q0 * s.head_dim || o.count != qn * s.head_dim) {
        const std::vector<tp::Range> got = {o}, want = {{q0 * s.head_dim, qn * s.head_dim}};
        return bad(p + "o cols", r, Fmt(got), Fmt(want));
      }
    }
  }
  return "";
}

// ---- who owns which layers --------------------------------------------------------------------------------------------
struct StageLayers {
  std::vector<int64_t> attn, gdn;  // global layer indices the stage keeps state for
};
// Stage A = layers [0, split), stage B = [split, N): a layer's state lives only on the stage that runs it (layer `split`
// itself is stage B's; stage A loads it as the norm-only layer and keeps no state for it).
inline StageLayers LayersOfStage(const StateGeometry& g, int64_t split, int stage) {
  if (split < 1 || split >= g.num_layers) throw std::invalid_argument("hybrid::LayersOfStage: split outside [1, N-1]");
  if (stage != kStageA && stage != kStageB) throw std::invalid_argument("hybrid::LayersOfStage: stage must be 0 or 1");
  StageLayers out;
  const auto mine = [&](int64_t l) { return stage == kStageA ? l < split : l >= split; };
  for (const int64_t l : g.attn_layers) {
    if (mine(l)) out.attn.push_back(l);
  }
  for (const int64_t l : g.gdn_layers) {
    if (mine(l)) out.gdn.push_back(l);
  }
  return out;
}

// ---- copy ops ---------------------------------------------------------------------------------------------------------
enum class StateKind { kKv = 0, kMtpKv = 1, kGdnRecurrent = 2, kGdnConv = 3 };
enum class Dir { kStageToRank, kRankToStage };  // scatter (the end of a pipelined call) / gather (before a warm one)

// One strided copy between the full-head image (a stage) and a rank image: `height` rows of `width` bytes, the rows `*_pitch`
// apart, starting at `*_off` into the layer's buffer (the live recurrent slot for GDN). hipMemcpy2D's shape.
struct Run2D {
  uint64_t full_off = 0, full_pitch = 0;
  uint64_t rank_off = 0, rank_pitch = 0;
  uint64_t width = 0, height = 1;
  uint64_t Bytes() const { return width * height; }
};
struct CopyOp {
  StateKind kind = StateKind::kKv;
  int64_t layer = -1;  // body layer; -1 for the MTP head
  int stage = 0;       // holder of the full-head side
  int rank = 0;        // the TP rank on the other side
  bool local = false;  // same card: a D2D copy; else D2H on one card + H2D on the other through the host ring
  std::vector<Run2D> runs;  // one run (KV, recurrent), three (the conv segments)
  uint64_t Bytes() const {
    uint64_t n = 0;
    for (const Run2D& r : runs) n += r.Bytes();
    return n;
  }
};
struct ReshardPlan {
  Dir dir = Dir::kStageToRank;
  std::vector<CopyOp> ops;
};

// Per-layer call parameters. The conv line pitch (entries per channel) differs between the models: a stage Model's
// state_len_max is conv_width - 1 = 3, a speculating TP rank's is conv_width - 2 + window.
struct PlanParams {
  StageLayers stage[2];
  int64_t stage_conv_pitch = 3, rank_conv_pitch = 3;
  // Stage B's conv line pitch when it differs from stage A's `stage_conv_pitch` (0 = the same). Stage B carries the MTP head, so it is
  // loaded with mtp_draft_k > 0 and its GDN state is sized for the verify window (conv_width - 2 + window entries per channel, like a
  // speculating rank), while stage A, which has no head, keeps the compact conv_width - 1 (Model::ReshardConvPitch has the Model's).
  int64_t stage_b_conv_pitch = 0;
  int64_t StageConvPitch(int stage) const { return stage == kStageB && stage_b_conv_pitch > 0 ? stage_b_conv_pitch : stage_conv_pitch; }
  bool mtp = true;  // the MTP head KV is part of the state (stage B only)
  // `rank_conv_pitch` is the TP rank Model's conv line pitch (state_len_max): conv_width - 2 + window for a speculating rank, so
  // it is the caller's to give (a default of the compact conv_live would silently mis-address every conv run of a real rank).
  // The stage Model's pitch is conv_live (its state_len_max is conv_width - 1).
  static PlanParams ForSplit(const StateGeometry& g, int64_t split, int64_t rank_conv_pitch, bool mtp = true) {
    if (rank_conv_pitch < g.conv_live) {
      throw std::invalid_argument("PlanParams::ForSplit: rank_conv_pitch " + std::to_string(rank_conv_pitch) +
                                  " is below the " + std::to_string(g.conv_live) + " live conv entries per channel");
    }
    PlanParams p;
    p.stage[kStageA] = LayersOfStage(g, split, kStageA);
    p.stage[kStageB] = LayersOfStage(g, split, kStageB);
    p.stage_conv_pitch = g.conv_live;
    p.rank_conv_pitch = rank_conv_pitch;
    p.mtp = mtp;
    return p;
  }
};

namespace detail {

inline CopyOp KvOp(const StateGeometry& g, StateKind kind, int64_t layer, int stage, int rank, const BlockRange& b) {
  const RankMap& m = g.ranks.at(static_cast<size_t>(rank));
  const uint64_t full_block = static_cast<uint64_t>(g.KvBlockBytesFull()), rank_block = static_cast<uint64_t>(g.KvBlockBytesRank(rank));
  CopyOp op;
  op.kind = kind;
  op.layer = layer;
  op.stage = stage;
  op.rank = rank;
  op.local = RankOfStage(stage) == rank;
  Run2D r;
  r.full_off = static_cast<uint64_t>(b.first) * full_block + static_cast<uint64_t>(m.kv_head_begin * g.KvHeadBlockBytes());
  r.full_pitch = full_block;
  r.rank_off = static_cast<uint64_t>(b.first) * rank_block;
  r.rank_pitch = rank_block;
  r.width = rank_block;
  r.height = static_cast<uint64_t>(b.Blocks());
  op.runs.push_back(r);
  return op;
}

inline CopyOp RecurrentOp(const StateGeometry& g, int64_t layer, int stage, int rank) {
  const RankMap& m = g.ranks.at(static_cast<size_t>(rank));
  CopyOp op;
  op.kind = StateKind::kGdnRecurrent;
  op.layer = layer;
  op.stage = stage;
  op.rank = rank;
  op.local = RankOfStage(stage) == rank;
  Run2D r;
  r.width = static_cast<uint64_t>(m.v_heads * g.RecurrentHeadBytes());
  r.full_off = static_cast<uint64_t>(m.v_head_begin * g.RecurrentHeadBytes());
  r.full_pitch = r.rank_pitch = r.width;
  r.rank_off = 0;
  op.runs.push_back(r);
  return op;
}

inline CopyOp ConvOp(const StateGeometry& g, const PlanParams& p, int64_t layer, int stage, int rank) {
  const RankMap& m = g.ranks.at(static_cast<size_t>(rank));
  CopyOp op;
  op.kind = StateKind::kGdnConv;
  op.layer = layer;
  op.stage = stage;
  op.rank = rank;
  op.local = RankOfStage(stage) == rank;
  const uint64_t fp = static_cast<uint64_t>(p.StageConvPitch(stage)) * 2, rp = static_cast<uint64_t>(p.rank_conv_pitch) * 2;
  for (const ConvSegment& c : m.conv) {
    Run2D r;
    r.full_off = static_cast<uint64_t>(c.full_begin) * fp;
    r.full_pitch = fp;
    r.rank_off = static_cast<uint64_t>(c.rank_begin) * rp;
    r.rank_pitch = rp;
    r.width = static_cast<uint64_t>(g.ConvLiveBytesPerChannel());
    r.height = static_cast<uint64_t>(c.count);
    op.runs.push_back(r);
  }
  return op;
}

inline void AddGdn(ReshardPlan* plan, const StateGeometry& g, const PlanParams& p, int stage) {
  for (const int64_t l : p.stage[stage].gdn) {
    for (int r = 0; r < kWorld; ++r) {
      plan->ops.push_back(RecurrentOp(g, l, stage, r));
      plan->ops.push_back(ConvOp(g, p, l, stage, r));
    }
  }
}

inline void AddKv(ReshardPlan* plan, const StateGeometry& g, const PlanParams& p, int stage, const BlockRange& b) {
  if (b.Blocks() == 0) return;
  for (const int64_t l : p.stage[stage].attn) {
    for (int r = 0; r < kWorld; ++r) plan->ops.push_back(KvOp(g, StateKind::kKv, l, stage, r, b));
  }
}

inline void AddMtp(ReshardPlan* plan, const StateGeometry& g, const PlanParams& p, const BlockRange& b) {
  if (!p.mtp || b.Blocks() == 0) return;
  for (int r = 0; r < kWorld; ++r) plan->ops.push_back(KvOp(g, StateKind::kMtpKv, -1, kStageB, r, b));
}

}  // namespace detail

// The end of a pipelined call over rows [p0, n_end): every stage's layer KV blocks the call wrote, the MTP head's primed
// blocks (stage B), and every layer's GDN state, stage -> both ranks (the rank on the stage's own card gets its half D2D, the
// other rank's half goes through the host).
inline ReshardPlan ScatterPlan(const StateGeometry& g, const PlanParams& p, int64_t p0, int64_t n_end) {
  if (p0 < 0 || n_end <= p0) throw std::invalid_argument("hybrid::ScatterPlan: needs 0 <= p0 < n_end");
  ReshardPlan plan;
  plan.dir = Dir::kStageToRank;
  for (int s = 0; s < 2; ++s) {
    detail::AddKv(&plan, g, p, s, KvScatterBlocks(p0, n_end, g.block_tokens));
    detail::AddGdn(&plan, g, p, s);
  }
  detail::AddMtp(&plan, g, p, MtpScatterBlocks(p0, n_end, g.block_tokens));
  return plan;
}

// Before a warm pipelined call: what the tracker says is stale on each stage (rows [kv_row0, kv_row1) of its attention
// layers, its GDN state) and the one MTP block of row p0-1 for stage B, rank shards -> stage.
inline ReshardPlan GatherPlan(const StateGeometry& g, const PlanParams& p, const TpMasterTracker::SyncPlan& sync) {
  ReshardPlan plan;
  plan.dir = Dir::kRankToStage;
  for (int s = 0; s < 2; ++s) {
    const TpMasterTracker::StageSync& st = sync.stage[s];
    detail::AddKv(&plan, g, p, s, BlocksOfRows(st.kv_row0, st.kv_row1, g.block_tokens));
    if (st.gdn) detail::AddGdn(&plan, g, p, s);
  }
  if (sync.mtp_block >= 0) detail::AddMtp(&plan, g, p, BlockRange{sync.mtp_block, sync.mtp_block});
  return plan;
}

// The card a copy's bytes leave from, and the totals per direction. `cross` = through the host (PCIe), `local` = D2D.
inline int SourceStage(const CopyOp& op, Dir d) { return d == Dir::kStageToRank ? op.stage : StageOfRank(op.rank); }
struct PlanTotals {
  uint64_t cross[2][4] = {};  // [source stage][StateKind]
  uint64_t local[4] = {};     // [StateKind]
  uint64_t CrossFrom(int stage) const {
    uint64_t n = 0;
    for (int k = 0; k < 4; ++k) n += cross[stage][k];
    return n;
  }
  uint64_t CrossKind(int stage, StateKind k) const { return cross[stage][static_cast<int>(k)]; }
  uint64_t Local() const {
    uint64_t n = 0;
    for (int k = 0; k < 4; ++k) n += local[k];
    return n;
  }
};
inline PlanTotals Totals(const ReshardPlan& plan) {
  PlanTotals t;
  for (const CopyOp& op : plan.ops) {
    const int k = static_cast<int>(op.kind);
    if (op.local) {
      t.local[k] += op.Bytes();
    } else {
      t.cross[SourceStage(op, plan.dir)][k] += op.Bytes();
    }
  }
  return t;
}

// ---- the host ring ----------------------------------------------------------------------------------------------------
// The bytes that cross out of one card are one stream (the plan's cross ops in order) cut into pieces of at most
// kRingPieceBytes, each through one of kRingSlots pinned slots round robin: D2H of piece i on the source card overlaps H2D of
// piece i-1 on the other (docs/pp-tp2-hybrid.md 2: 3 x 128 MiB per direction).
constexpr uint64_t kRingPieceBytes = 128ull << 20;
constexpr int kRingSlots = 3;
struct RingPiece {
  uint64_t offset = 0, bytes = 0;
  int slot = 0;
};
inline std::vector<RingPiece> ChunkRing(uint64_t total, uint64_t piece_bytes = kRingPieceBytes, int slots = kRingSlots) {
  if (piece_bytes == 0 || slots < 1) throw std::invalid_argument("hybrid::ChunkRing: piece_bytes and slots must be positive");
  std::vector<RingPiece> out;
  for (uint64_t at = 0; at < total; at += piece_bytes) {
    out.push_back({at, std::min(piece_bytes, total - at), static_cast<int>(out.size() % static_cast<size_t>(slots))});
  }
  return out;
}
inline std::vector<RingPiece> RingPiecesFrom(const ReshardPlan& plan, int source_stage) {
  return ChunkRing(Totals(plan).CrossFrom(source_stage));
}

// ---- the host reference -----------------------------------------------------------------------------------------------
// The live state of one Model (full-head) or of one TP rank (its head-split image), as host bytes. KV images are whole blocks
// [block][heads][block_tokens][2 * head_dim]; the recurrent image is the live slot [v_heads][V][K] fp32; the conv image is
// compact: [channels][conv_live] bf16 (the pitch of the device line is a copy-op matter, not part of the state).
struct LiveImage {
  std::map<int64_t, std::vector<uint8_t>> kv;        // attention layer -> bytes
  std::vector<uint8_t> mtp_kv;                       // the MTP head's KV
  std::map<int64_t, std::vector<uint8_t>> gdn_rec;   // GDN layer -> bytes
  std::map<int64_t, std::vector<uint16_t>> gdn_conv; // GDN layer -> [channels][conv_live]
  bool operator==(const LiveImage& o) const {
    return kv == o.kv && mtp_kv == o.mtp_kv && gdn_rec == o.gdn_rec && gdn_conv == o.gdn_conv;
  }
  bool operator!=(const LiveImage& o) const { return !(*this == o); }
};

namespace detail {

// KV image: full <-> rank, one head at a time.
inline std::vector<uint8_t> KvToRank(const StateGeometry& g, const std::vector<uint8_t>& in, int rank) {
  const RankMap& m = g.ranks.at(static_cast<size_t>(rank));
  const size_t hb = static_cast<size_t>(g.KvHeadBlockBytes()), fb = static_cast<size_t>(g.KvBlockBytesFull());
  if (in.size() % fb != 0) throw std::invalid_argument("hybrid::ReshardRef: a KV image is not whole blocks");
  const size_t blocks = in.size() / fb, H = static_cast<size_t>(g.kv_heads_full), h0 = static_cast<size_t>(m.kv_head_begin);
  const size_t hr = static_cast<size_t>(m.kv_heads);
  std::vector<uint8_t> out(blocks * hr * hb);
  for (size_t b = 0; b < blocks; ++b) {
    for (size_t h = 0; h < hr; ++h) std::memcpy(out.data() + (b * hr + h) * hb, in.data() + (b * H + h0 + h) * hb, hb);
  }
  return out;
}
inline void KvFromRank(const StateGeometry& g, const std::vector<uint8_t>& in, int rank, std::vector<uint8_t>* full) {
  const RankMap& m = g.ranks.at(static_cast<size_t>(rank));
  const size_t hb = static_cast<size_t>(g.KvHeadBlockBytes());
  const size_t hr = static_cast<size_t>(m.kv_heads), H = static_cast<size_t>(g.kv_heads_full), h0 = static_cast<size_t>(m.kv_head_begin);
  if (in.size() % (hr * hb) != 0) throw std::invalid_argument("hybrid::GatherRef: a rank KV image is not whole blocks");
  const size_t blocks = in.size() / (hr * hb);
  if (full->empty()) full->assign(blocks * H * hb, 0);
  if (full->size() != blocks * H * hb) throw std::invalid_argument("hybrid::GatherRef: the two ranks' KV images differ in size");
  for (size_t b = 0; b < blocks; ++b) {
    for (size_t h = 0; h < hr; ++h) std::memcpy(full->data() + (b * H + h0 + h) * hb, in.data() + (b * hr + h) * hb, hb);
  }
}

}  // namespace detail

// Rank `rank`'s image of a full-head live state. Layers present in `full` are present in the result.
inline LiveImage ReshardRef(const StateGeometry& g, const LiveImage& full, int rank) {
  const RankMap& m = g.ranks.at(static_cast<size_t>(rank));
  LiveImage out;
  for (const auto& [layer, bytes] : full.kv) out.kv[layer] = detail::KvToRank(g, bytes, rank);
  if (!full.mtp_kv.empty()) out.mtp_kv = detail::KvToRank(g, full.mtp_kv, rank);
  for (const auto& [layer, bytes] : full.gdn_rec) {
    if (bytes.size() != static_cast<size_t>(g.RecurrentBytesFull())) {
      throw std::invalid_argument("hybrid::ReshardRef: a recurrent image is not [v_heads][V][K] fp32");
    }
    const size_t hb = static_cast<size_t>(g.RecurrentHeadBytes());
    out.gdn_rec[layer].assign(bytes.begin() + static_cast<std::ptrdiff_t>(static_cast<size_t>(m.v_head_begin) * hb),
                              bytes.begin() + static_cast<std::ptrdiff_t>(static_cast<size_t>(m.v_head_begin + m.v_heads) * hb));
  }
  const size_t live = static_cast<size_t>(g.conv_live);
  for (const auto& [layer, conv] : full.gdn_conv) {
    if (conv.size() != static_cast<size_t>(g.conv_dim_full) * live) {
      throw std::invalid_argument("hybrid::ReshardRef: a conv image is not [conv_dim][live] bf16");
    }
    std::vector<uint16_t>& o = out.gdn_conv[layer];
    o.assign(static_cast<size_t>(g.ConvChannelsRank(rank)) * live, 0);
    for (const ConvSegment& c : m.conv) {
      for (int64_t ch = 0; ch < c.count; ++ch) {
        for (size_t e = 0; e < live; ++e) {
          o[static_cast<size_t>(c.rank_begin + ch) * live + e] = conv[static_cast<size_t>(c.full_begin + ch) * live + e];
        }
      }
    }
  }
  return out;
}

// The inverse: the full-head image from the two ranks' (rank 0, rank 1). Both must carry the same layers.
inline LiveImage GatherRef(const StateGeometry& g, const LiveImage& rank0, const LiveImage& rank1) {
  const LiveImage* r[2] = {&rank0, &rank1};
  LiveImage out;
  const auto same_keys = [](const auto& a, const auto& b) {
    if (a.size() != b.size()) return false;
    auto i = a.begin();
    auto j = b.begin();
    for (; i != a.end(); ++i, ++j) {
      if (i->first != j->first) return false;
    }
    return true;
  };
  if (!same_keys(rank0.kv, rank1.kv) || !same_keys(rank0.gdn_rec, rank1.gdn_rec) || !same_keys(rank0.gdn_conv, rank1.gdn_conv) ||
      rank0.mtp_kv.empty() != rank1.mtp_kv.empty()) {
    throw std::invalid_argument("hybrid::GatherRef: the two ranks carry different layers");
  }
  for (int k = 0; k < kWorld; ++k) {
    for (const auto& [layer, bytes] : r[k]->kv) detail::KvFromRank(g, bytes, k, &out.kv[layer]);
    if (!r[k]->mtp_kv.empty()) detail::KvFromRank(g, r[k]->mtp_kv, k, &out.mtp_kv);
  }
  const size_t live = static_cast<size_t>(g.conv_live), hb = static_cast<size_t>(g.RecurrentHeadBytes());
  for (int k = 0; k < kWorld; ++k) {
    const RankMap& m = g.ranks.at(static_cast<size_t>(k));
    for (const auto& [layer, bytes] : r[k]->gdn_rec) {
      if (bytes.size() != static_cast<size_t>(m.v_heads) * hb) throw std::invalid_argument("hybrid::GatherRef: a rank recurrent image has the wrong size");
      std::vector<uint8_t>& full = out.gdn_rec[layer];
      if (full.empty()) full.assign(static_cast<size_t>(g.RecurrentBytesFull()), 0);
      std::memcpy(full.data() + static_cast<size_t>(m.v_head_begin) * hb, bytes.data(), bytes.size());
    }
    for (const auto& [layer, conv] : r[k]->gdn_conv) {
      if (conv.size() != static_cast<size_t>(g.ConvChannelsRank(k)) * live) throw std::invalid_argument("hybrid::GatherRef: a rank conv image has the wrong size");
      std::vector<uint16_t>& full = out.gdn_conv[layer];
      if (full.empty()) full.assign(static_cast<size_t>(g.conv_dim_full) * live, 0);
      for (const ConvSegment& c : m.conv) {
        for (int64_t ch = 0; ch < c.count; ++ch) {
          for (size_t e = 0; e < live; ++e) {
            full[static_cast<size_t>(c.full_begin + ch) * live + e] = conv[static_cast<size_t>(c.rank_begin + ch) * live + e];
          }
        }
      }
    }
  }
  return out;
}

}  // namespace r4dx::model::hybrid
