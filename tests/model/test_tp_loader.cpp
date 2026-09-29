// tests/model/test_tp_loader.cpp -- the tensor-parallel sharded Container::Load (docs/tp.md 5.1,
// 10.1 "test_tp_loader", phase P2a), on the real 4-layer container qwen38-27b-l4-allmtp.r4dx (every
// layout, GDN and full-attention layers, the MTP head).
//
// For each layout in {bf16, w4a16}, both ranks of world 2 are loaded side by side
// (Container::Load(path, ContainerLoadOptions{tp_world 2, tp_rank r})) and checked:
//   * every uploaded device buffer, read back, equals tp::Gather(tp::Plan*(...)) of the file's own
//     bytes for that rank -- which test_tp_shard proved equals pack(slice(W)) (docs/tp.md 10.2);
//   * every QuantLinear's N/K is the rank shape, checked against the literal numbers of docs/tp.md
//     4.2's tables (not re-derived from the rules), and every raw tensor has its rank size;
//   * the loaded form (requested layout, .bf16.w, bare) is the one the TP=1 fallback chain picks;
//   * Config() is ModelConfig::Shard(GlobalConfig(), 2, r) and GlobalConfig() the file's config;
//   * bf16 layout only: the two ranks' slices REASSEMBLE every sharded tensor of the file exactly --
//     every global row (column-parallel) or every row's column range (row-parallel) is mapped to its
//     (rank, local position) by logical index arithmetic alone and compared, so the check does not
//     lean on the byte planners it would otherwise be testing with themselves;
//   * EmbedTokensHost() is the same pointer on both ranks, the shared pinned copy, and holds the
//     file's bytes; the device mirror follows embed_device_resident_decided (and holds the file's
//     bytes when present).
// Plus the load-time refusals: TP-only options at tp_world 1, vision weights on rank 1.
// And the trellis cases (docs/trellis-kernel.md 2.4, 5.5; CheckTrellisTp below) on tests/convert's
// tiny trellis containers (FIXTURES_REQUIRED trellis_tiny): both ranks' words, suh, svh and part
// widths against the TP=1 load of the same file.
//
// GPU test on HIP device 1 (ctest sets HIP_VISIBLE_DEVICES=1); SKIPs (77) when neither the 4-layer
// container nor the tiny trellis ones are present. Peak VRAM ~7 GiB (both bf16 ranks at once, no
// embedding mirror).
#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "container.h"
#include "model_config.h"
#include "r4dx/core/error.hpp"
#include "r4dx_convert/safetensors_reader.hpp"
#include "test_common.h"
#include "tp/tp_shard.h"
#include "trellis_ref.hpp"  // tests/kernels: PairGridIndex (the trellis cases)
#include "w4a16_group_meta.h"

using r4dx::model::Container;
using r4dx::model::ContainerLoadOptions;
using r4dx::model::Layout;
using r4dx::model::LayoutName;
using r4dx::model::ModelConfig;
using r4dx::model::QuantLinear;
namespace tp = r4dx::model::tp;

namespace {

const char* kContainerPath = r4dx_test::ContainerPath("D:/models/r4dx/qwen38-27b-l4-allmtp.r4dx");
constexpr int kWorld = 2;
constexpr int64_t kLayers = 4;

struct Checker {
  int checks = 0;
  int failures = 0;
  void Expect(bool ok, const std::string& what) {
    ++checks;
    if (ok) return;
    ++failures;
    if (failures <= 40) std::fprintf(stderr, "[FAIL] %s\n", what.c_str());
  }
};

template <class T>
std::vector<uint8_t> DeviceBytes(const r4dx::core::DeviceBuffer<T>& b) {
  std::vector<uint8_t> out(b.bytes());
  if (!out.empty()) {
    R4DX_HIP_CHECK(hipMemcpy(out.data(), b.data(), b.bytes(), hipMemcpyDeviceToHost));
  }
  return out;
}

// The file side: bytes of a whole tensor and the rank's expected slice of it.
class FileView {
 public:
  explicit FileView(const std::string& path)
      : r_(r4dx_convert::Utf8ToWide(path)) {}
  const r4dx_convert::SafetensorsReader& Reader() const { return r_; }
  bool Has(const std::string& name) const { return r_.Has(name); }
  const uint8_t* Data(const std::string& name) const { return r_.Data(name); }
  size_t Span(const std::string& name) const {
    const auto& m = r_.Meta(name);
    return static_cast<size_t>(m.end - m.begin);
  }
  std::vector<uint8_t> Gather(const std::string& name, const std::vector<tp::ByteRun>& runs) const {
    return tp::Gather(Data(name), Span(name), runs);
  }
  std::vector<uint8_t> Whole(const std::string& name) const {
    return std::vector<uint8_t>(Data(name), Data(name) + Span(name));
  }

 private:
  r4dx_convert::SafetensorsReader r_;
};

// The rank's expected slice of a raw (layout-less) tensor under its rule.
std::vector<uint8_t> ExpectedRaw(const FileView& f, const std::string& name, const ModelConfig& g,
                                 int rank) {
  const tp::ShardRule rule = tp::RuleFor(name, g);
  if (rule.split == tp::Split::kReplicate) return f.Whole(name);
  int64_t rows = 0;
  for (const auto& s : rule.segments) rows += s.rows;
  tp::PartShape sh;
  sh.part = tp::Part::kElem;
  sh.N = rows;
  sh.row_bytes = static_cast<int64_t>(f.Span(name)) / rows;
  return f.Gather(name, tp::PlanRows(sh, tp::RankRows(rule, kWorld, rank)));
}

struct LoadedPair {
  std::unique_ptr<Container> rank[kWorld];
};

// ---- per-rank checks: every buffer == Gather(Plan(file)) ----------------------------------------

class RankChecker {
 public:
  // `groups`: the container's w4a16 groups (quant2 Q3) -- each linear's wsz name and slice group.
  RankChecker(const FileView& f, const ModelConfig& global, int rank,
              const r4dx::model::W4a16Groups& groups, Layout requested, Checker& ck)
      : f_(f), g_(global), rank_(rank), groups_(groups), requested_(requested), ck_(ck) {}

  template <class T>
  void Raw(const std::string& name, const r4dx::core::DeviceBuffer<T>& buf, size_t rank_elems) {
    ck_.Expect(buf.size() == rank_elems, Where(name) + ": " + std::to_string(buf.size()) +
                                             " elements, expected " + std::to_string(rank_elems));
    ck_.Expect(DeviceBytes(buf) == ExpectedRaw(f_, name, g_, rank_), Where(name) + ": bytes");
  }

  // gdn.norm_weight: bf16 on disk, widened to fp32 on device, replicated.
  void Widened(const std::string& name, const r4dx::core::DeviceBuffer<float>& buf) {
    const auto* src = reinterpret_cast<const uint16_t*>(f_.Data(name));
    const size_t n = f_.Span(name) / 2;
    std::vector<float> want(n);
    for (size_t i = 0; i < n; ++i) want[i] = r4dx::core::Bf16ToFloat(src[i]);
    const std::vector<float> got = buf.CopyToHost();
    ck_.Expect(got == want, Where(name) + ": widened values");
  }

  // `force_bf16`: mtp.attn.k/v, which the loader always requests as bf16.
  void Lin(const std::string& base, const QuantLinear& q, int64_t N, int64_t K, int64_t rank_n,
           int64_t rank_k, bool force_bf16 = false) {
    const Layout req = force_bf16 ? Layout::kBf16 : requested_;
    const Layout want_form = HasForm(base, req) ? req : Layout::kBf16;
    ck_.Expect(q.layout == want_form, Where(base) + ": loaded as " + LayoutName(q.layout) +
                                          ", expected " + LayoutName(want_form));
    ck_.Expect(q.N == rank_n && q.K == rank_k,
               Where(base) + ": rank shape [" + std::to_string(q.N) + ", " + std::to_string(q.K) +
                   "], expected [" + std::to_string(rank_n) + ", " + std::to_string(rank_k) + "]");
    const tp::ShardRule rule = tp::RuleFor(base, g_);
    auto part = [&](const std::string& name, tp::Part p, int group, const std::vector<uint8_t>& got) {
      tp::PartShape sh;
      sh.part = p;
      sh.N = N;
      sh.K = K;
      sh.group = group;
      std::vector<tp::ByteRun> runs;
      if (rule.split == tp::Split::kRows) {
        runs = tp::PlanRows(sh, tp::RankRows(rule, kWorld, rank_));
      } else if (rule.split == tp::Split::kCols) {
        runs = tp::PlanCols(sh, tp::RankCols(rule, kWorld, rank_));
      } else {
        runs = {tp::ByteRun{0, f_.Span(name)}};
      }
      ck_.Expect(got == f_.Gather(name, runs), Where(name) + ": bytes");
    };
    switch (q.layout) {
      case Layout::kBf16: {
        const std::string name = f_.Has(base + ".bf16.w") ? base + ".bf16.w" : base;
        part(name, tp::Part::kBf16, 0, DeviceBytes(q.bf16_w));
        break;
      }
      case Layout::kW4a16:
        part(base + ".w4a16.wq", tp::Part::kW4Wq, 0, DeviceBytes(q.wq));
        part(groups_.WszName(base), tp::Part::kW4a16Wsz, groups_.GroupFor(base),
             DeviceBytes(q.w4a16_wsz));
        ck_.Expect(q.w4a16_group == groups_.QuantLinearGroup(base),
                   Where(base) + ": w4a16_group " + std::to_string(q.w4a16_group));
        break;
      case Layout::kTrellis:  // never requested here; CheckTrellisTp covers the trellis slices
        ck_.Expect(false, Where(base) + ": loaded as trellis from a non-trellis container");
        break;
    }
  }

 private:
  bool HasForm(const std::string& base, Layout l) const {
    switch (l) {
      case Layout::kBf16: return f_.Has(base + ".bf16.w");
      case Layout::kW4a16: return f_.Has(base + ".w4a16.wq") && f_.Has(groups_.WszName(base));
      case Layout::kTrellis:
        return false;
    }
    return false;
  }
  std::string Where(const std::string& name) const {
    return std::string(LayoutName(requested_)) + " rank " + std::to_string(rank_) + " " + name;
  }

  const FileView& f_;
  const ModelConfig& g_;
  int rank_;
  const r4dx::model::W4a16Groups& groups_;
  Layout requested_;
  Checker& ck_;
};

// Rank shapes from docs/tp.md 4.2 at TP=2 (v6 / 27B, which the 4-layer container shares).
constexpr int64_t kHidden = 5120;

void CheckRank(const Container& c, const FileView& f, const ModelConfig& g, int rank,
               const r4dx::model::W4a16Groups& groups, Layout layout, Checker& ck) {
  RankChecker rc(f, g, rank, groups, layout, ck);
  ck.Expect(c.NumLoadedLayers() == kLayers, "layer count");
  for (int64_t i = 0; i < kLayers; ++i) {
    const std::string b = "text.layers." + std::to_string(i) + ".";
    const auto& lw = c.Layer(i);
    rc.Raw(b + "input_layernorm", lw.input_layernorm, kHidden);
    rc.Raw(b + "post_attention_layernorm", lw.post_attention_layernorm, kHidden);
    if (g.IsGdnLayer(i)) {
      ck.Expect(lw.gdn.has_value() && !lw.attn.has_value(), b + ": GDN layer");
      const auto& gw = *lw.gdn;
      rc.Lin(b + "gdn.in_proj_qkv", gw.in_proj_qkv, 10240, kHidden, 5120, kHidden);
      rc.Lin(b + "gdn.in_proj_z", gw.in_proj_z, 6144, kHidden, 3072, kHidden);
      rc.Raw(b + "gdn.in_proj_a", gw.in_proj_a, 24 * kHidden);
      rc.Raw(b + "gdn.in_proj_b", gw.in_proj_b, 24 * kHidden);
      rc.Raw(b + "gdn.conv1d_weight", gw.conv1d_weight, 5120 * 4);
      rc.Raw(b + "gdn.A_log", gw.A_log, 24);
      rc.Raw(b + "gdn.dt_bias", gw.dt_bias, 24);
      rc.Widened(b + "gdn.norm_weight", gw.norm_weight);
      rc.Lin(b + "gdn.out_proj", gw.out_proj, kHidden, 6144, kHidden, 3072);
    } else {
      ck.Expect(lw.attn.has_value() && !lw.gdn.has_value(), b + ": attention layer");
      const auto& a = *lw.attn;
      rc.Lin(b + "attn.qg", a.qg, 12288, kHidden, 6144, kHidden);
      rc.Lin(b + "attn.k", a.k, 1024, kHidden, 512, kHidden);
      rc.Lin(b + "attn.v", a.v, 1024, kHidden, 512, kHidden);
      rc.Lin(b + "attn.o", a.o, kHidden, 6144, kHidden, 3072);
      rc.Raw(b + "attn.q_norm", a.q_norm, 256);
      rc.Raw(b + "attn.k_norm", a.k_norm, 256);
      rc.Raw(b + "attn.k_descale", a.k_descale, 2);
      rc.Raw(b + "attn.v_descale", a.v_descale, 2);
    }
    rc.Lin(b + "mlp.gate_up", lw.mlp.gate_up, 34816, kHidden, 17408, kHidden);
    rc.Lin(b + "mlp.down", lw.mlp.down, kHidden, 17408, kHidden, 8704);
  }
  rc.Raw("text.final_norm", c.FinalNorm(), kHidden);
  rc.Lin("lm_head", c.LmHead(), 248320, kHidden, 124160, kHidden);

  ck.Expect(c.HasMtp(), "mtp head loaded");
  if (c.HasMtp()) {
    const auto& mw = c.Mtp();
    const auto& a = *mw.layer.attn;
    rc.Raw("mtp.input_layernorm", mw.layer.input_layernorm, kHidden);
    rc.Raw("mtp.post_attention_layernorm", mw.layer.post_attention_layernorm, kHidden);
    rc.Lin("mtp.attn.qg", a.qg, 12288, kHidden, 6144, kHidden);
    rc.Lin("mtp.attn.k", a.k, 1024, kHidden, 512, kHidden, /*force_bf16=*/true);
    rc.Lin("mtp.attn.v", a.v, 1024, kHidden, 512, kHidden, /*force_bf16=*/true);
    rc.Lin("mtp.attn.o", a.o, kHidden, 6144, kHidden, 3072);
    rc.Raw("mtp.attn.q_norm", a.q_norm, 256);
    rc.Raw("mtp.attn.k_norm", a.k_norm, 256);
    rc.Raw("mtp.attn.k_descale", a.k_descale, 2);
    rc.Raw("mtp.attn.v_descale", a.v_descale, 2);
    rc.Lin("mtp.mlp.gate_up", mw.layer.mlp.gate_up, 34816, kHidden, 17408, kHidden);
    rc.Lin("mtp.mlp.down", mw.layer.mlp.down, kHidden, 17408, kHidden, 8704);
    rc.Raw("mtp.fc", mw.fc, kHidden * 2 * kHidden);  // replicated
    rc.Raw("mtp.norm", mw.norm, kHidden);
    rc.Raw("mtp.pre_fc_norm_hidden", mw.pre_fc_norm_hidden, kHidden);
    rc.Raw("mtp.pre_fc_norm_embedding", mw.pre_fc_norm_embedding, kHidden);
  }

  // Config() is the rank shard of GlobalConfig(), docs/tp.md 3.1's numbers.
  const ModelConfig& rc_cfg = c.Config();
  const ModelConfig& gc = c.GlobalConfig();
  ck.Expect(gc.tp_world == 1 && gc.num_attention_heads == 24 && gc.num_key_value_heads == 4 &&
                gc.intermediate_size == 17408 && gc.linear_num_key_heads == 16 &&
                gc.linear_num_value_heads == 48 && gc.vocab_size == 248320,
            "GlobalConfig() is the container's own config");
  ck.Expect(rc_cfg.tp_world == kWorld && rc_cfg.tp_rank == rank &&
                rc_cfg.num_attention_heads == 12 && rc_cfg.num_key_value_heads == 2 &&
                rc_cfg.intermediate_size == 8704 && rc_cfg.linear_num_key_heads == 8 &&
                rc_cfg.linear_num_value_heads == 24 && rc_cfg.vocab_size == 248320 &&
                rc_cfg.hidden_size == kHidden && rc_cfg.VocabShardBegin() == 124160 * rank,
            "Config() is the rank " + std::to_string(rank) + " shard");
  ck.Expect(!c.HasVision() && !c.HasVisionConfig(), "no vision on a text-only container");
}

// ---- bf16 reassembly: rank 0 and rank 1 slices rebuild every sharded tensor ---------------------

// Column-parallel (row split): every GLOBAL row g of the file is found in exactly one rank's buffer
// at the position logical index arithmetic says -- segment s holds rows [begin, begin+rows), rank r
// its part [begin + r*part, +part), laid out in segment order.
bool ReassembleRows(const tp::ShardRule& rule, const std::vector<uint8_t>* ranks,
                    const uint8_t* file, size_t file_bytes) {
  int64_t total = 0;
  for (const auto& s : rule.segments) total += s.rows;
  const size_t row_bytes = file_bytes / static_cast<size_t>(total);
  for (int r = 0; r < kWorld; ++r) {
    if (ranks[r].size() * kWorld != file_bytes) return false;
  }
  int64_t local_base = 0;  // rows of earlier segments in each rank's buffer
  for (const auto& s : rule.segments) {
    const int64_t part = s.rows / kWorld;
    for (int64_t gr = s.begin; gr < s.begin + s.rows; ++gr) {
      const int r = static_cast<int>((gr - s.begin) / part);
      const int64_t local = local_base + (gr - s.begin - r * part);
      if (std::memcmp(ranks[r].data() + static_cast<size_t>(local) * row_bytes,
                      file + static_cast<size_t>(gr) * row_bytes, row_bytes) != 0) {
        return false;
      }
    }
    local_base += part;
  }
  return true;
}

// Row-parallel (K split) of a bf16 [N, K] matrix: row n of the file is rank 0's row n followed by
// rank 1's row n.
bool ReassembleCols(int64_t N, int64_t K, const std::vector<uint8_t>* ranks, const uint8_t* file) {
  const int64_t k = K / kWorld;
  for (int r = 0; r < kWorld; ++r) {
    if (ranks[r].size() != static_cast<size_t>(N * k * 2)) return false;
  }
  for (int64_t n = 0; n < N; ++n) {
    for (int r = 0; r < kWorld; ++r) {
      if (std::memcmp(ranks[r].data() + static_cast<size_t>(n * k * 2),
                      file + static_cast<size_t>((n * K + r * k) * 2),
                      static_cast<size_t>(k * 2)) != 0) {
        return false;
      }
    }
  }
  return true;
}

void CheckBf16Reassembly(const LoadedPair& p, const FileView& f, const ModelConfig& g,
                         Checker& ck) {
  int sharded = 0;
  // `name` is the on-disk tensor (for a linear, its .bf16.w or bare form), `base` its rule key.
  auto check = [&](const std::string& name, const std::string& base,
                   const std::vector<uint8_t> (&ranks)[kWorld], int64_t N, int64_t K) {
    const tp::ShardRule rule = tp::RuleFor(base, g);
    if (rule.split == tp::Split::kReplicate) return;
    ++sharded;
    const bool ok = rule.split == tp::Split::kRows
                        ? ReassembleRows(rule, ranks, f.Data(name), f.Span(name))
                        : ReassembleCols(N, K, ranks, f.Data(name));
    ck.Expect(ok, "bf16 reassembly of " + name);
  };
  auto lin = [&](const std::string& base, const QuantLinear& q0, const QuantLinear& q1, int64_t N,
                 int64_t K) {
    if (q0.layout != Layout::kBf16) {
      ck.Expect(false, "bf16 reassembly: " + base + " is not bf16");
      return;
    }
    const std::string name = f.Has(base + ".bf16.w") ? base + ".bf16.w" : base;
    const std::vector<uint8_t> ranks[kWorld] = {DeviceBytes(q0.bf16_w), DeviceBytes(q1.bf16_w)};
    check(name, base, ranks, N, K);
  };
  auto raw = [&](const std::string& name, const auto& b0, const auto& b1) {
    const std::vector<uint8_t> ranks[kWorld] = {DeviceBytes(b0), DeviceBytes(b1)};
    check(name, name, ranks, 0, 0);
  };

  const Container& c0 = *p.rank[0];
  const Container& c1 = *p.rank[1];
  for (int64_t i = 0; i < kLayers; ++i) {
    const std::string b = "text.layers." + std::to_string(i) + ".";
    const auto& l0 = c0.Layer(i);
    const auto& l1 = c1.Layer(i);
    if (g.IsGdnLayer(i)) {
      lin(b + "gdn.in_proj_qkv", l0.gdn->in_proj_qkv, l1.gdn->in_proj_qkv, 10240, kHidden);
      lin(b + "gdn.in_proj_z", l0.gdn->in_proj_z, l1.gdn->in_proj_z, 6144, kHidden);
      lin(b + "gdn.out_proj", l0.gdn->out_proj, l1.gdn->out_proj, kHidden, 6144);
      raw(b + "gdn.in_proj_a", l0.gdn->in_proj_a, l1.gdn->in_proj_a);
      raw(b + "gdn.in_proj_b", l0.gdn->in_proj_b, l1.gdn->in_proj_b);
      raw(b + "gdn.conv1d_weight", l0.gdn->conv1d_weight, l1.gdn->conv1d_weight);
      raw(b + "gdn.A_log", l0.gdn->A_log, l1.gdn->A_log);
      raw(b + "gdn.dt_bias", l0.gdn->dt_bias, l1.gdn->dt_bias);
    } else {
      lin(b + "attn.qg", l0.attn->qg, l1.attn->qg, 12288, kHidden);
      lin(b + "attn.k", l0.attn->k, l1.attn->k, 1024, kHidden);
      lin(b + "attn.v", l0.attn->v, l1.attn->v, 1024, kHidden);
      lin(b + "attn.o", l0.attn->o, l1.attn->o, kHidden, 6144);
      raw(b + "attn.k_descale", l0.attn->k_descale, l1.attn->k_descale);
      raw(b + "attn.v_descale", l0.attn->v_descale, l1.attn->v_descale);
    }
    lin(b + "mlp.gate_up", l0.mlp.gate_up, l1.mlp.gate_up, 34816, kHidden);
    lin(b + "mlp.down", l0.mlp.down, l1.mlp.down, kHidden, 17408);
  }
  lin("lm_head", c0.LmHead(), c1.LmHead(), 248320, kHidden);
  const auto& m0 = *c0.Mtp().layer.attn;
  const auto& m1 = *c1.Mtp().layer.attn;
  lin("mtp.attn.qg", m0.qg, m1.qg, 12288, kHidden);
  lin("mtp.attn.k", m0.k, m1.k, 1024, kHidden);
  lin("mtp.attn.v", m0.v, m1.v, 1024, kHidden);
  lin("mtp.attn.o", m0.o, m1.o, kHidden, 6144);
  raw("mtp.attn.k_descale", m0.k_descale, m1.k_descale);
  raw("mtp.attn.v_descale", m0.v_descale, m1.v_descale);
  lin("mtp.mlp.gate_up", c0.Mtp().layer.mlp.gate_up, c1.Mtp().layer.mlp.gate_up, 34816, kHidden);
  lin("mtp.mlp.down", c0.Mtp().layer.mlp.down, c1.Mtp().layer.mlp.down, kHidden, 17408);
  // 3 GDN layers x 8 + 1 attention layer x 6 + 4 x 2 MLP + lm_head + MTP 8 = 47 sharded tensors.
  ck.Expect(sharded == 47, "bf16 reassembly covered " + std::to_string(sharded) +
                               " sharded tensors, expected 47");
}

// ---- embeddings -----------------------------------------------------------------------------

void CheckEmbed(const LoadedPair& p, const FileView& f,
                const std::shared_ptr<const r4dx::core::PinnedBuffer<uint16_t>>& shared,
                int decided, bool check_host_bytes, Checker& ck) {
  const std::string tag = "embed (decided " + std::to_string(decided) + ")";
  for (int r = 0; r < kWorld; ++r) {
    ck.Expect(p.rank[r]->EmbedTokensHost() == shared->data(),
              tag + ": rank " + std::to_string(r) + " host pointer is the shared copy");
    ck.Expect(p.rank[r]->EmbedTokensDeviceResident() == (decided == 1),
              tag + ": rank " + std::to_string(r) + " device mirror follows the decision");
  }
  const size_t bytes = f.Span("text.embed_tokens");
  if (check_host_bytes) {
    ck.Expect(shared->bytes() == bytes &&
                  std::memcmp(shared->data(), f.Data("text.embed_tokens"), bytes) == 0,
              tag + ": shared host copy holds the file's bytes");
  }
  if (decided == 1) {
    for (int r = 0; r < kWorld; ++r) {
      std::vector<uint8_t> dev(bytes);
      R4DX_HIP_CHECK(hipMemcpy(dev.data(), p.rank[r]->EmbedTokensDevice(), bytes,
                               hipMemcpyDeviceToHost));
      ck.Expect(std::memcmp(dev.data(), f.Data("text.embed_tokens"), bytes) == 0,
                tag + ": rank " + std::to_string(r) + " device mirror holds the file's bytes");
    }
  }
}

// ---- refusals -------------------------------------------------------------------------------

template <class Fn>
bool ThrowsInvalidArgument(Fn&& fn) {
  try {
    fn();
  } catch (const std::invalid_argument&) {
    return true;
  } catch (...) {
    return false;
  }
  return false;
}

void CheckRefusals(const std::shared_ptr<const r4dx::core::PinnedBuffer<uint16_t>>& shared,
                   Checker& ck) {
  ck.Expect(ThrowsInvalidArgument([&] {
              ContainerLoadOptions o;
              o.shared_embed_host = shared;  // TP-only, at tp_world 1
              (void)Container::Load(kContainerPath, o);
            }),
            "tp_world 1 refuses shared_embed_host");
  ck.Expect(ThrowsInvalidArgument([&] {
              ContainerLoadOptions o;
              o.embed_device_resident_decided = 1;
              (void)Container::Load(kContainerPath, o);
            }),
            "tp_world 1 refuses embed_device_resident_decided");
  ck.Expect(ThrowsInvalidArgument([&] {
              ContainerLoadOptions o;
              o.tp_world = 2;
              o.tp_rank = 1;
              o.load_vision = true;  // vision weights are rank-0-only
              (void)Container::Load(kContainerPath, o);
            }),
            "rank 1 refuses load_vision");
  ck.Expect(ThrowsInvalidArgument([&] {
              ContainerLoadOptions o;
              o.tp_world = 2;
              o.tp_rank = 2;
              (void)Container::Load(kContainerPath, o);
            }),
            "tp_rank out of range");
}

// ---- trellis (docs/trellis-kernel.md 2.4, 5.5, 6) -----------------------------------------------
// tests/convert's tiny trellis containers (FIXTURES_REQUIRED trellis_tiny) loaded whole (TP = 1)
// and as both ranks of TP = 2. Every rank's trellis buffers must be the slice of the TP = 1 buffers
// its rows or K range select, derived by index arithmetic alone -- not through the byte planners:
//   words: each tile (tn, tk) of the rank's own [N, K] pair grid (trellis_ref::PairGridIndex) holds
//     the 8*KB words of the global tile it stands for (the rank's row map for a column-parallel
//     linear, k0/16 + tk for a row-parallel one);
//   suh: column-parallel replicated (every part), row-parallel the rank's K range;
//   svh: column-parallel the rank's rows, row-parallel replicated;
// plus the rank shape (the literal numbers of the tiny config: hidden 256, 2 key heads and 8 value
// heads of 128, 4 heads of 256 and 2 KV heads, intermediate 1024), the rank-local part widths of
// mlp.gate_up (512 + 512), rate, prescale and tickets. Returns the containers checked (0: absent).
const char* const kTrellisTiny[] = {R4DX_TRELLIS_TINY_DIR "/tiny_k4.r4dx",
                                    R4DX_TRELLIS_TINY_DIR "/tiny_mix.r4dx"};

int CheckTrellisTp(Checker& ck) {
  int checked = 0;
  for (const char* path : kTrellisTiny) {
    if (!r4dx_test::FileExists(path)) continue;
    ++checked;
    ContainerLoadOptions o1;
    o1.layout = o1.lm_head_layout = o1.mtp_head_layout = Layout::kTrellis;
    o1.embed_device_resident = false;
    const Container full = Container::Load(path, o1);
    const ModelConfig& g = full.GlobalConfig();
    const auto shared = Container::LoadEmbedTokensHost(path);
    std::unique_ptr<Container> rank[kWorld];
    for (int r = 0; r < kWorld; ++r) {
      ContainerLoadOptions o = o1;
      o.tp_world = kWorld;
      o.tp_rank = r;
      o.embed_device_resident_decided = 0;
      o.shared_embed_host = shared;
      rank[r] = std::make_unique<Container>(Container::Load(path, o));
    }
    const int before = ck.failures;
    // {base, full linear, rank linears, rank N, rank K}
    struct Lin {
      std::string base;
      const QuantLinear* full;
      const QuantLinear* rank[kWorld];
      int64_t rank_n, rank_k;
    };
    std::vector<Lin> lins;
    for (int64_t i = 0; i < full.NumLoadedLayers(); ++i) {
      const std::string b = "text.layers." + std::to_string(i) + ".";
      const auto& f = full.Layer(i);
      const auto& r0 = rank[0]->Layer(i);
      const auto& r1 = rank[1]->Layer(i);
      if (g.IsGdnLayer(i)) {
        lins.push_back({b + "gdn.in_proj_qkv", &f.gdn->in_proj_qkv,
                        {&r0.gdn->in_proj_qkv, &r1.gdn->in_proj_qkv}, 768, 256});
        lins.push_back({b + "gdn.in_proj_z", &f.gdn->in_proj_z,
                        {&r0.gdn->in_proj_z, &r1.gdn->in_proj_z}, 512, 256});
        lins.push_back({b + "gdn.out_proj", &f.gdn->out_proj,
                        {&r0.gdn->out_proj, &r1.gdn->out_proj}, 256, 512});
      } else {
        lins.push_back({b + "attn.qg", &f.attn->qg, {&r0.attn->qg, &r1.attn->qg}, 1024, 256});
        lins.push_back({b + "attn.k", &f.attn->k, {&r0.attn->k, &r1.attn->k}, 256, 256});
        lins.push_back({b + "attn.v", &f.attn->v, {&r0.attn->v, &r1.attn->v}, 256, 256});
        lins.push_back({b + "attn.o", &f.attn->o, {&r0.attn->o, &r1.attn->o}, 256, 512});
      }
      lins.push_back({b + "mlp.gate_up", &f.mlp.gate_up, {&r0.mlp.gate_up, &r1.mlp.gate_up}, 1024,
                      256});
      lins.push_back({b + "mlp.down", &f.mlp.down, {&r0.mlp.down, &r1.mlp.down}, 256, 512});
    }
    for (const Lin& l : lins) {
      const QuantLinear& q = *l.full;
      const std::string what = std::string(path) + " " + l.base;
      ck.Expect(q.layout == Layout::kTrellis, what + ": TP=1 loads trellis");
      if (q.layout != Layout::kTrellis) continue;
      const int kb = q.trellis_bits;
      const int64_t N = q.N, K = q.K;
      const std::vector<uint32_t> fw = q.trellis_w.CopyToHost();
      const std::vector<float> fsuh = q.trellis_suh.CopyToHost(), fsvh = q.trellis_svh.CopyToHost();
      const tp::ShardRule rule = tp::RuleFor(l.base, g);
      for (int r = 0; r < kWorld; ++r) {
        const QuantLinear& s = *l.rank[r];
        const std::string w = what + " rank " + std::to_string(r);
        ck.Expect(s.layout == Layout::kTrellis && s.trellis_bits == kb &&
                      s.trellis_prescale_log2 == q.trellis_prescale_log2 &&
                      s.trellis_parts == q.trellis_parts && s.trellis_tickets != nullptr,
                  w + ": rate / prescale / parts / tickets");
        ck.Expect(s.N == l.rank_n && s.K == l.rank_k,
                  w + ": rank shape [" + std::to_string(s.N) + ", " + std::to_string(s.K) + "]");
        if (s.layout != Layout::kTrellis || s.N != l.rank_n || s.K != l.rank_k) continue;
        // The rank's global rows (column-parallel) or K range (row-parallel), by the rule's logic.
        std::vector<int64_t> row_of;  // local row -> global row
        int64_t k0 = 0;
        if (rule.split == tp::Split::kRows) {
          for (const auto& seg : rule.segments) {
            const int64_t part = seg.rows / kWorld;
            for (int64_t i = 0; i < part; ++i) row_of.push_back(seg.begin + r * part + i);
          }
        } else {
          for (int64_t n = 0; n < N; ++n) row_of.push_back(n);
          k0 = r * (K / kWorld);
        }
        const std::vector<uint32_t> sw = s.trellis_w.CopyToHost();
        bool words = sw.size() == static_cast<size_t>(s.N * s.K * kb / 32);
        for (int64_t tn = 0; words && tn < s.N / 16; ++tn) {
          const int64_t gtn = row_of[static_cast<size_t>(16 * tn)] / 16;
          for (int64_t tk = 0; words && tk < s.K / 16; ++tk) {
            const size_t li = trellis_ref::PairGridIndex(s.K, kb, tn, tk);
            const size_t gi = trellis_ref::PairGridIndex(K, kb, gtn, k0 / 16 + tk);
            words = std::memcmp(&sw[li], &fw[gi], static_cast<size_t>(8 * kb) * 4) == 0;
          }
        }
        ck.Expect(words, w + ": every tile's words are its global tile's");
        std::vector<float> want_suh, want_svh;
        if (rule.split == tp::Split::kRows) {
          want_suh = fsuh;
          for (int64_t row : row_of) want_svh.push_back(fsvh[static_cast<size_t>(row)]);
        } else {
          want_suh.assign(fsuh.begin() + k0, fsuh.begin() + k0 + s.K);
          want_svh = fsvh;
        }
        ck.Expect(s.trellis_suh.CopyToHost() == want_suh, w + ": suh slice");
        ck.Expect(s.trellis_svh.CopyToHost() == want_svh, w + ": svh slice");
        const bool two = q.trellis_parts == 2;
        ck.Expect(s.trellis_part_n[0] == (two ? q.trellis_part_n[0] / kWorld : s.N) &&
                      s.trellis_part_n[1] == (two ? q.trellis_part_n[1] / kWorld : 0),
                  w + ": rank-local part widths " + std::to_string(s.trellis_part_n[0]) + " + " +
                      std::to_string(s.trellis_part_n[1]));
      }
    }
    ck.Expect(lins.size() == 11, std::string(path) + ": 11 trellis linears checked");
    std::printf("test_tp_loader: trellis %s both ranks checked (%d failure(s))\n", path,
                ck.failures - before);
    std::fflush(stdout);
  }
  return checked;
}

// CheckTrellisTp, run after the w4a16 block and with its throws caught: a trellis failure is counted
// as one, never allowed to cost the w4a16 TP-loader checks their run. Returns the containers checked
// (a throwing run counts as one).
int CheckTrellisTpGuarded(Checker& ck) {
  try {
    return CheckTrellisTp(ck);
  } catch (const std::exception& e) {
    ++ck.checks;
    ++ck.failures;
    std::fprintf(stderr, "[FAIL] test_tp_loader: the trellis cases threw: %s\n", e.what());
    return 1;
  }
}

int RunTest() {
  if (!r4dx_test::FileExists(kContainerPath)) {
    Checker trellis_ck;
    const int trellis_checked = CheckTrellisTpGuarded(trellis_ck);
    if (trellis_checked == 0) return r4dx_test::SkipMissing(kContainerPath);
    std::printf("test_tp_loader: %s absent -- trellis cases only: %d/%d checks passed\n",
                kContainerPath, trellis_ck.checks - trellis_ck.failures, trellis_ck.checks);
    return trellis_ck.failures == 0 ? 0 : 1;
  }
  const FileView f(kContainerPath);

  // The container's own config and w4a16 group, parsed independently of the loader under test by
  // a TP=1 metadata-only route: ModelConfig::FromJson on the header's text_config.
  ModelConfig global;
  r4dx::model::W4a16Groups w4a16_groups;  // default 128 and no map unless the header says so
  {
    std::FILE* fp = std::fopen(kContainerPath, "rb");
    if (fp == nullptr) throw std::runtime_error("cannot open container");
    uint64_t header_len = 0;
    if (std::fread(&header_len, 8, 1, fp) != 1) throw std::runtime_error("short header");
    std::string header(static_cast<size_t>(header_len), '\0');
    if (std::fread(header.data(), 1, header.size(), fp) != header.size()) {
      throw std::runtime_error("short header");
    }
    std::fclose(fp);
    const nlohmann::json meta = nlohmann::json::parse(header).at("__metadata__");
    const nlohmann::json& mc = meta.at("model_config");
    global = ModelConfig::FromJson(mc.contains("text_config") ? mc.at("text_config") : mc);
    // quant2 Q3: the default group and any per-tensor map (test_w4a16_group_meta covers the parse).
    w4a16_groups = r4dx::model::ParseW4a16Groups(meta, kContainerPath);
  }

  const auto shared = Container::LoadEmbedTokensHost(kContainerPath);
  Checker ck;
  CheckRefusals(shared, ck);

  const Layout layouts[] = {Layout::kBf16, Layout::kW4a16};
  for (Layout layout : layouts) {
    // Exercise both embedding-mirror decisions: mirrored on the w4a16 pass (the production
    // layout), host-only on the others (which also keeps the bf16 pass's peak VRAM down).
    const int decided = layout == Layout::kW4a16 ? 1 : 0;
    LoadedPair p;
    for (int r = 0; r < kWorld; ++r) {
      ContainerLoadOptions o;
      o.layout = o.lm_head_layout = o.mtp_head_layout = layout;
      o.layer_limit = kLayers;
      o.embed_device_resident_decided = decided;
      o.tp_world = kWorld;
      o.tp_rank = r;
      o.shared_embed_host = shared;
      p.rank[r] = std::make_unique<Container>(Container::Load(kContainerPath, o));
    }
    const int before = ck.failures;
    for (int r = 0; r < kWorld; ++r) CheckRank(*p.rank[r], f, global, r, w4a16_groups, layout, ck);
    CheckEmbed(p, f, shared, decided, /*check_host_bytes=*/layout == Layout::kBf16, ck);
    if (layout == Layout::kBf16) CheckBf16Reassembly(p, f, global, ck);
    std::printf("test_tp_loader: %-5s both ranks checked (%d failure(s))\n", LayoutName(layout),
                ck.failures - before);
    std::fflush(stdout);
  }

  Checker trellis_ck;
  const int trellis_checked = CheckTrellisTpGuarded(trellis_ck);
  ck.checks += trellis_ck.checks;
  ck.failures += trellis_ck.failures;
  std::printf("test_tp_loader: %d/%d checks passed (%d trellis container(s))\n",
              ck.checks - ck.failures, ck.checks, trellis_checked);
  return ck.failures == 0 ? 0 : 1;
}

}  // namespace

int main() { return r4dx_test::RunGuardedMain("test_tp_loader", [] { return RunTest(); }); }
