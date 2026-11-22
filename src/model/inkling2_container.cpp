#include "inkling2_container.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>

#include "arch.h"
#include "container_load_util.h"
#include "r4dx/core/direct_storage.hpp"
#include "r4dx/core/dtype.hpp"
#include "tp/tp_shard.h"

namespace r4dx::model {

using namespace container_util;

namespace {

const std::string kWho = "r4dx::model::Inkling2Container: ";

std::string LayerBase(int64_t i) { return "text.layers." + std::to_string(i) + "."; }

// The bytes of a container tensor, checked against the element count and width it must have.
const uint8_t* Bytes(const SafetensorsReader& r, const std::string& name, int64_t elems, int64_t width) {
  if (!r.Has(name)) throw std::runtime_error(kWho + "the container has no tensor '" + name + "'");
  const auto& m = r.Meta(name);
  const uint64_t span = m.end - m.begin;
  if (span != static_cast<uint64_t>(elems) * static_cast<uint64_t>(width))
    throw std::runtime_error(kWho + "'" + name + "' is " + std::to_string(span) + " bytes, the model needs " +
                             std::to_string(elems) + " x " + std::to_string(width));
  return r.Data(name);
}

std::vector<uint16_t> HostBf16(const SafetensorsReader& r, const std::string& name, int64_t elems) {
  std::vector<uint16_t> v(static_cast<size_t>(elems));
  std::memcpy(v.data(), Bytes(r, name, elems, 2), v.size() * 2);
  return v;
}
std::vector<float> HostBf16AsF32(const SafetensorsReader& r, const std::string& name, int64_t elems) {
  const std::vector<uint16_t> b = HostBf16(r, name, elems);
  std::vector<float> f(b.size());
  for (size_t i = 0; i < b.size(); ++i) f[i] = core::Bf16ToFloat(b[i]);
  return f;
}
std::vector<float> HostF32(const SafetensorsReader& r, const std::string& name, int64_t elems) {
  std::vector<float> v(static_cast<size_t>(elems));
  std::memcpy(v.data(), Bytes(r, name, elems, 4), v.size() * 4);
  return v;
}

// Rows [r0, r0 + n) of a row-major [rows, row_elems] array.
template <class T>
std::vector<T> Rows(const std::vector<T>& v, int64_t row_elems, int64_t r0, int64_t n) {
  return std::vector<T>(v.begin() + r0 * row_elems, v.begin() + (r0 + n) * row_elems);
}
// Columns [c0, c0 + n) of a row-major [rows, cols] array.
template <class T>
std::vector<T> Cols(const std::vector<T>& v, int64_t rows, int64_t cols, int64_t c0, int64_t n) {
  std::vector<T> out(static_cast<size_t>(rows * n));
  for (int64_t i = 0; i < rows; ++i)
    std::copy(v.begin() + i * cols + c0, v.begin() + i * cols + c0 + n, out.begin() + i * n);
  return out;
}

template <class T>
core::DeviceBuffer<T> Upload(const std::vector<T>& v) {
  core::DeviceBuffer<T> b(v.size());
  b.CopyFromHost(v.data(), v.size());
  return b;
}

// One linear: TP=1 (or a replicated one) through the shared fallback chain; a split one cut by `rule` with the same
// requested -> .bf16.w -> bare chain as ShardLoader::Linear (whose rules come from tp::RuleFor; inkling2's are local).
QuantLinear LoadLinear(const SafetensorsReader& r, const LinearLoadMeta& meta, const std::string& base, Layout requested,
                       int64_t N, int64_t K, const tp::ShardRule& rule, int world, int rank, int* fallbacks) {
  if (world == 1 || rule.split == tp::Split::kReplicate)
    return LoadQuantLinearWithFallback(r, meta, base, requested, N, K, fallbacks);
  Layout form = requested;
  bool bare = false;
  if (!HasLayout(r, meta, base, requested)) {
    if (fallbacks != nullptr) ++*fallbacks;
    form = Layout::kBf16;
    if (!HasLayout(r, meta, base, Layout::kBf16)) {
      if (!r.Has(base)) throw std::runtime_error(kWho + "no tensor found for '" + base + "' (requested layout, bf16, or bare)");
      bare = true;
    }
  }
  QuantLinear q;
  q.layout = form;
  q.N = N;
  q.K = K;
  std::vector<tp::Range> rows;
  tp::Range cols{0, K};
  if (rule.split == tp::Split::kRows) {
    rows = tp::RankRows(rule, world, rank);
    q.N = 0;
    for (const tp::Range& x : rows) q.N += x.count;
  } else if (rule.split == tp::Split::kCols) {
    cols = tp::RankCols(rule, world, rank);
    q.K = cols.count;
  } else {
    throw std::logic_error(kWho + "'" + base + "': unsupported shard rule");
  }
  const auto part = [&](const std::string& name, tp::Part p, int group) {
    tp::PartShape s;
    s.part = p;
    s.N = N;
    s.K = K;
    s.group = group;
    const auto& m = r.Meta(name);
    const uint64_t span = m.end - m.begin;
    const std::vector<tp::ByteRun> runs = rule.split == tp::Split::kRows ? tp::PlanRows(s, rows) : tp::PlanCols(s, cols);
    return tp::Gather(r.Data(name), static_cast<size_t>(span), runs);
  };
  const auto up_u8 = [](const std::vector<uint8_t>& b) { return Upload(b); };
  const auto up_u16 = [](const std::vector<uint8_t>& b) {
    std::vector<uint16_t> v(b.size() / 2);
    std::memcpy(v.data(), b.data(), v.size() * 2);
    return Upload(v);
  };
  switch (form) {
    case Layout::kBf16: {
      const std::string name = bare ? base : base + ".bf16.w";
      (void)Bytes(r, name, N * K, 2);
      q.bf16_w = up_u16(part(name, tp::Part::kBf16, 0));
      break;
    }
    case Layout::kMxfp8:
      if (!meta.mxfp8) throw std::logic_error(kWho + "'" + base + "' read as mxfp8 without a checked quant.mxfp8");
      CheckMxfp8Shape(r, base, N, K);
      q.mx_wq = up_u8(part(base + r4dx_convert::mxfp8::kWqSuffix, tp::Part::kW8Wq, 32));
      q.mx_ws = up_u8(part(base + r4dx_convert::mxfp8::kWsSuffix, tp::Part::kMxfp8Ws, 32));
      break;
    default:
      throw std::invalid_argument(kWho + "'" + base + "': layout " + LayoutName(form) +
                                  " is not supported for inkling2_loop (mxfp8 or bf16)");
  }
  return q;
}

tp::ShardRule RowRule(std::vector<int64_t> segments) {
  tp::ShardRule r;
  r.split = tp::Split::kRows;
  int64_t b = 0;
  for (int64_t n : segments) {
    r.segments.push_back({b, n});
    b += n;
  }
  return r;
}
tp::ShardRule ColRule(int64_t k) {
  tp::ShardRule r;
  r.split = tp::Split::kCols;
  r.k_total = k;
  return r;
}
tp::ShardRule Replicated() { return tp::ShardRule{}; }

}  // namespace

Inkling2Config Inkling2Container::InspectConfig(const std::string& path, nlohmann::json* metadata_out) {
  // DetectArch takes the path, not a json: r4dx_model_arch is built against third_party's nlohmann and this TU may see
  // another copy (the ROCm include directory's), so no json crosses that library boundary.
  if (DetectArch(path) != Arch::kInkling2Loop) throw std::runtime_error(kWho + path + " is not an inkling2_loop container");
  const nlohmann::json meta = ReadMetadata(path);
  if (!meta.is_object()) throw std::runtime_error(kWho + path + " is not an r4dx container (no __metadata__)");
  if (!meta.contains("model_config")) throw std::runtime_error(kWho + path + " has no __metadata__.model_config");
  Inkling2Config c = Inkling2Config::FromModelConfig(meta.at("model_config"));
  if (meta.contains("inkling2") && meta.at("inkling2").is_object()) {
    const nlohmann::json& ik = meta.at("inkling2");
    if (ik.contains("lm_head_rows") && ik.at("lm_head_rows").get<int64_t>() != c.unpadded_vocab_size)
      throw std::runtime_error(kWho + path + ": __metadata__.inkling2.lm_head_rows != the config's unpadded_vocab_size");
    if (ik.contains("lm_head_rows_stored") && ik.at("lm_head_rows_stored").get<int64_t>() != c.HeadRowsStored())
      throw std::runtime_error(kWho + path + ": __metadata__.inkling2.lm_head_rows_stored != round16(unpadded_vocab_size)");
  }
  if (metadata_out != nullptr) *metadata_out = meta;
  return c;
}

Inkling2Container Inkling2Container::Load(const std::string& path, const Inkling2LoadOptions& o) {
  const core::DirectLoadScope direct_load;
  if (o.tp_world < 1 || o.tp_world > 2 || o.tp_rank < 0 || o.tp_rank >= o.tp_world)
    throw std::invalid_argument(kWho + "need tp_world in {1, 2} and 0 <= tp_rank < tp_world");
  if (o.layout != Layout::kMxfp8 && o.layout != Layout::kBf16)
    throw std::invalid_argument(kWho + "--layout " + LayoutName(o.layout) +
                                " is not supported for inkling2_loop (use --layout mxfp8, or bf16 for a bf16 container)");
  Inkling2Container c;
  nlohmann::json metadata;
  c.global_ = InspectConfig(path, &metadata);
  c.local_ = o.tp_world > 1 ? Inkling2Config::Shard(c.global_, o.tp_world, o.tp_rank) : c.global_;
  c.generic_ = c.global_.ToModelConfig();
  c.model_id_ = metadata.value("model_id", std::string("inkling2_loop"));
  const Inkling2Config& g = c.global_;
  const Inkling2Config& l = c.local_;
  const int W = o.tp_world, R = o.tp_rank;

  LinearLoadMeta meta;
  meta.path = path;
  meta.mxfp8 = CheckMxfp8Choice(metadata, path, o.layout, o.layout);
  SafetensorsReader r(Utf8ToWide(path));

  const int64_t H = g.hidden_size, hd = g.head_dim, I = g.moe_intermediate_size;
  const int64_t hl = l.num_attention_heads, kvl = l.num_key_value_heads;
  const int64_t h0 = R * hl, kv0 = R * kvl;  // the rank's first q head / kv head
  int fb = 0;
  const auto lin = [&](const std::string& base, int64_t N, int64_t K, const tp::ShardRule& rule) {
    return LoadLinear(r, meta, base, o.layout, N, K, rule, W, R, &fb);
  };

  c.embed_tokens_ = Upload(HostBf16(r, "text.embed_tokens", g.vocab_size * H));
  c.embed_norm_ = Upload(HostBf16(r, "text.embed_norm", H));
  c.final_norm_ = Upload(HostBf16(r, "text.final_norm", H));
  c.final_query_ = Upload(HostF32(r, "text.final_query", H));
  c.msg_queries_ = Upload(HostF32(r, "text.msg_queries", g.LoopLayers() * 4 * H));

  for (int64_t i = 0; i < g.num_hidden_layers; ++i) {
    const std::string b = LayerBase(i);
    Inkling2LayerWeights w;
    w.full = g.IsFull(i);
    w.loop = g.IsLoop(i);
    w.n_attn_queries = i == 0 ? 0 : (w.full ? 3 : 1);
    if (w.n_attn_queries > 0) w.attn_query = Upload(HostF32(r, b + "attn_query", w.n_attn_queries * H));
    w.mlp_query = Upload(HostF32(r, b + "mlp_query", H));
    if (w.full) {
      w.q_norm = Upload(HostBf16(r, b + "q_norm", H));
      w.k_norm = Upload(HostBf16(r, b + "k_norm", H));
      w.v_norm = Upload(HostBf16(r, b + "v_norm", H));
    } else {
      w.attn_norm = Upload(HostBf16(r, b + "attn_norm", H));
    }
    w.mlp_norm = Upload(HostBf16(r, b + "mlp_norm", H));
    w.attn_conv = Upload(HostBf16AsF32(r, b + "attn_conv", H * 4));
    w.mlp_conv = Upload(HostBf16AsF32(r, b + "mlp_conv", H * 4));

    Inkling2AttnWeights& a = w.attn;
    a.q = lin(b + "attn.q", g.QDim(), H, RowRule({g.QDim()}));
    a.gate = lin(b + "attn.gate", g.QDim(), H, RowRule({g.QDim()}));
    a.k = lin(b + "attn.k", g.KvDim(), H, RowRule({g.KvDim()}));
    a.v = lin(b + "attn.v", g.KvDim(), H, RowRule({g.KvDim()}));
    a.o = lin(b + "attn.o", H, g.QDim(), ColRule(g.QDim()));
    a.rel = lin(b + "attn.rel", g.num_attention_heads * g.d_rel, H, Replicated());
    // S4.1: concatenate [q, gate, rel, k, v] into one contiguous MXFP8 buffer (zero duplicate VRAM).
    const bool concat_qkv = [] {
      const char* e = std::getenv("R4DX_INK_CONCAT_QKV");
      return !(e != nullptr && (std::strcmp(e, "0") == 0 || std::strcmp(e, "off") == 0));
    }();
    if (concat_qkv && a.q.layout == Layout::kMxfp8 && a.gate.layout == Layout::kMxfp8 && a.rel.layout == Layout::kMxfp8 &&
        a.k.layout == Layout::kMxfp8 && a.v.layout == Layout::kMxfp8 && a.q.K == H && a.gate.K == H && a.rel.K == H &&
        a.k.K == H && a.v.K == H && (a.q.N % 16) == 0 && (a.gate.N % 16) == 0 && (a.rel.N % 16) == 0 &&
        (a.k.N % 16) == 0 && (a.v.N % 16) == 0) {
      QuantLinear* parts[5] = {&a.q, &a.gate, &a.rel, &a.k, &a.v};
      int64_t* row0s[5] = {&a.q_row0, &a.gate_row0, &a.rel_row0, &a.k_row0, &a.v_row0};
      int64_t total_n = 0;
      for (int p = 0; p < 5; ++p) {
        *row0s[p] = total_n;
        total_n += parts[p]->N;
      }
      a.qgr_n = a.q.N + a.gate.N + a.rel.N;
      a.qgrkv_n = total_n;
      a.qgrkv_wq = core::DeviceBuffer<uint8_t>(static_cast<size_t>(total_n * H));
      a.qgrkv_ws = core::DeviceBuffer<uint8_t>(static_cast<size_t>(total_n * (H / 32)));
      for (int p = 0; p < 5; ++p) {
        R4DX_HIP_CHECK(hipMemcpy(a.qgrkv_wq.data() + static_cast<size_t>(*row0s[p] * H), parts[p]->mx_wq.data(),
                                 parts[p]->mx_wq.bytes(), hipMemcpyDeviceToDevice));
        R4DX_HIP_CHECK(hipMemcpy(a.qgrkv_ws.data() + static_cast<size_t>(*row0s[p] * (H / 32)), parts[p]->mx_ws.data(),
                                 parts[p]->mx_ws.bytes(), hipMemcpyDeviceToDevice));
        parts[p]->mx_wq = {};
        parts[p]->mx_ws = {};
      }
    }
    a.sink = Upload(Rows(HostBf16(r, b + "attn.sink", g.num_attention_heads * H), H, h0, hl));
    a.q_scale = Upload(Rows(HostBf16AsF32(r, b + "attn.q_scale", g.num_attention_heads * hd), hd, h0, hl));
    a.rel_scale = Upload(Rows(HostBf16AsF32(r, b + "attn.rel_scale", g.num_attention_heads * g.d_rel), g.d_rel, h0, hl));
    a.rel_proj = Upload(HostBf16AsF32(r, b + "attn.rel_proj", g.d_rel * g.RelExtent(i)));
    if (w.full) {
      a.k_conv_static = Upload(Cols(HostBf16AsF32(r, b + "attn.k_conv_static", 4 * g.KvDim()), 4, g.KvDim(), kv0 * hd, kvl * hd));
      a.v_conv_static = Upload(Cols(HostBf16AsF32(r, b + "attn.v_conv_static", 4 * g.KvDim()), 4, g.KvDim(), kv0 * hd, kvl * hd));
      a.k_conv_proj = Upload(Rows(HostBf16(r, b + "attn.k_conv_proj", g.num_key_value_heads * 4 * H), 4 * H, kv0, kvl));
      a.v_conv_proj = Upload(Rows(HostBf16(r, b + "attn.v_conv_proj", g.num_key_value_heads * 4 * H), 4 * H, kv0, kvl));
    } else {
      a.k_conv = Upload(Rows(HostBf16AsF32(r, b + "attn.k_conv", g.KvDim() * 4), 4, kv0 * hd, kvl * hd));
      a.v_conv = Upload(Rows(HostBf16AsF32(r, b + "attn.v_conv", g.KvDim() * 4), 4, kv0 * hd, kvl * hd));
    }
    if (w.loop) {
      const char* names[4] = {"q", "k", "v", "mlp"};
      for (int m = 0; m < 4; ++m) {
        const std::string mb = b + "msg." + names[m] + ".";
        w.msg[m].down = Upload(HostBf16(r, mb + "down", g.gate_rank * 2 * H));
        w.msg[m].forget = Upload(HostBf16AsF32(r, mb + "forget", H * g.gate_rank));
        w.msg[m].gain = Upload(HostBf16AsF32(r, mb + "gain", H * g.gate_rank));
      }
    }
    w.router = Upload(HostBf16(r, b + "moe.router", g.num_experts * H));
    w.bias = Upload(HostF32(r, b + "moe.bias", g.num_experts));
    w.gate_up.reserve(static_cast<size_t>(g.num_experts));
    w.down.reserve(static_cast<size_t>(g.num_experts));
    for (int64_t e = 0; e < g.num_experts; ++e) {
      const std::string eb = b + "moe.experts." + std::to_string(e) + ".";
      w.gate_up.push_back(lin(eb + "gate_up", 2 * I, H, RowRule({I, I})));
      w.down.push_back(lin(eb + "down", H, I, ColRule(I)));
    }
    // Device pointer tables when every expert of the layer has one layout.
    bool all_mx = true, all_bf = true;
    for (size_t e = 0; e < w.gate_up.size(); ++e) {
      all_mx = all_mx && w.gate_up[e].layout == Layout::kMxfp8 && w.down[e].layout == Layout::kMxfp8;
      all_bf = all_bf && w.gate_up[e].layout == Layout::kBf16 && w.down[e].layout == Layout::kBf16;
    }
    if (all_mx || all_bf) {
      std::vector<uint64_t> gu0, gu1, dn0, dn1;
      for (size_t e = 0; e < w.gate_up.size(); ++e) {
        gu0.push_back(reinterpret_cast<uint64_t>(all_mx ? static_cast<const void*>(w.gate_up[e].mx_wq.data())
                                                        : static_cast<const void*>(w.gate_up[e].bf16_w.data())));
        dn0.push_back(reinterpret_cast<uint64_t>(all_mx ? static_cast<const void*>(w.down[e].mx_wq.data())
                                                        : static_cast<const void*>(w.down[e].bf16_w.data())));
        if (all_mx) {
          gu1.push_back(reinterpret_cast<uint64_t>(w.gate_up[e].mx_ws.data()));
          dn1.push_back(reinterpret_cast<uint64_t>(w.down[e].mx_ws.data()));
        }
      }
      w.gate_up_ptrs0 = Upload(gu0);
      w.down_ptrs0 = Upload(dn0);
      if (all_mx) {
        w.gate_up_ptrs1 = Upload(gu1);
        w.down_ptrs1 = Upload(dn1);
      }
    }
    c.layers_.push_back(std::move(w));
  }
  const int fb_body = fb;
  c.lm_head_ = lin("lm_head", g.HeadRowsStored(), H, RowRule({g.HeadRowsStored()}));
  if (c.lm_head_.N != l.HeadRowsLocal())
    throw std::logic_error(kWho + "lm_head has " + std::to_string(c.lm_head_.N) + " rows on this rank, expected " +
                           std::to_string(l.HeadRowsLocal()));
  const bool bf16_head = TakeMxfp8Bf16Head(meta, c.lm_head_, fb_body, &fb);
  c.fallbacks_ = fb;
  if (o.tp_rank == 0) {
    if (meta.mxfp8) LogMxfp8(*meta.mxfp8, path);
    if (bf16_head) std::fprintf(stderr, "r4dx: %s stores lm_head as bf16 only -- loaded bf16\n", path.c_str());
    if (fb > 0)
      std::fprintf(stderr, "r4dx: %d linear(s) in %s do not carry the requested layout (%s) and were loaded as bf16\n", fb,
                   path.c_str(), LayoutName(o.layout));
  }
  return c;
}

}  // namespace r4dx::model
