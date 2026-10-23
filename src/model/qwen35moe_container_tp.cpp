// Qwen35MoeContainer::LoadShard -- the tensor-parallel (`--tp 2`) loader of a qwen3_5_moe container (docs/qwen35moe-tp-design.md 1.3,
// 2.3). Every rank gathers its slice of every tensor from the one container (tp::RuleFor + the byte plans of tp_shard.h; no per-rank file, no
// re-pack): the mixers, the router (replicated), the bf16 shared expert and lm_head go through ShardLoader exactly as the dense
// Container::LoadShard does; the routed experts keep LoadExpertClass's layout -- ONE stacked buffer per (layer, class) that the grouped decode
// kernels index with `expert * w_per` -- at the RANK's per-expert size, each expert's slice gathered into the stack by
// qwen35moe_tp::PlanTrellisLinear (the same arithmetic tests/model/qwen35moe/test_q35moe_tp_shard.cpp proves on the shipped container's bytes).
//
// A separate translation unit on purpose: the TP=1 loader (qwen35moe_container.cpp) is byte-identical to before.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>

#include "container_load_util.h"
#include "qwen35moe_container.h"
#include "qwen35moe_tp_shard.h"
#include "r4dx/core/direct_storage.hpp"
#include "r4dx/core/dtype.hpp"
#include "r4dx/core/error.hpp"
#include "r4dx/core/pinned_buffer.hpp"
#include "r4dx/core/r4d.hpp"
#include "shard_loader.h"

namespace r4dx::model {

namespace q35tp = qwen35moe_tp;

using namespace container_util;
using qwen4exp::ExpertClassStorage;
using qwen4exp::ExpertLinears;
using qwen4exp::MoeWeights;

namespace {

const std::string kWho = "r4dx::model::Qwen35MoeContainer(TP): ";

struct DeviceScope {
  int prev = -1;
  bool active = false;
  explicit DeviceScope(int dev) {
    if (dev < 0) return;
    R4DX_HIP_CHECK(hipGetDevice(&prev));
    R4DX_HIP_CHECK(hipSetDevice(dev));
    active = true;
  }
  ~DeviceScope() {
    if (active) static_cast<void>(hipSetDevice(prev));
  }
  DeviceScope(const DeviceScope&) = delete;
  DeviceScope& operator=(const DeviceScope&) = delete;
};

std::string ExpertBase(int64_t layer, int64_t e, const char* cls) {
  return "text.layers." + std::to_string(layer) + ".moe.experts." + std::to_string(e) + "." + cls;
}

uint64_t SpanOf(const SafetensorsReader& r, const std::string& name) {
  const auto& m = r.Meta(name);
  return m.end - m.begin;
}

void CopyToDevice(void* dst, const void* src, size_t bytes) {
  if (bytes != 0) R4DX_HIP_CHECK(hipMemcpy(dst, src, bytes, hipMemcpyHostToDevice));
}

// fp16 bytes -> fp32 floats appended at `dst`.
void WidenF16(const uint8_t* src, size_t elems, float* dst) {
  for (size_t i = 0; i < elems; ++i) {
    uint16_t h;
    std::memcpy(&h, src + 2 * i, 2);
    dst[i] = core::F16ToFloat(h);
  }
}

// The bytes of `plan` out of tensor `name`, appended at `dst` (which must have room); returns the bytes written.
size_t GatherInto(const SafetensorsReader& r, const std::string& name, const std::vector<tp::ByteRun>& plan, uint8_t* dst) {
  const uint8_t* src = r.Data(name);
  const uint64_t span = SpanOf(r, name);
  size_t o = 0;
  for (const tp::ByteRun& run : plan) {
    if (run.src_off > span || run.bytes > span - run.src_off) {
      throw std::out_of_range(kWho + "a byte run of '" + name + "' reaches past its " + std::to_string(span) + " bytes");
    }
    std::memcpy(dst + o, src + run.src_off, run.bytes);
    o += run.bytes;
  }
  return o;
}

size_t RunsBytes(const std::vector<tp::ByteRun>& runs) {
  size_t t = 0;
  for (const tp::ByteRun& r : runs) t += r.bytes;
  return t;
}

// One (layer, class) of routed experts for this rank: the stacked buffers at the rank's per-expert size, and per-expert views.
// `N` / `K` are the GLOBAL logical shape of one expert's linear.
void LoadExpertClassSharded(const SafetensorsReader& r, const LinearLoadMeta& meta, const ModelConfig& gm, Layout requested,
                            const Qwen35MoeConfig& c, int world, int rank, int64_t layer, const char* cls, int64_t N, int64_t K,
                            ExpertClassStorage& st, std::vector<ExpertLinears>& experts, QuantLinear ExpertLinears::*member,
                            int* fallbacks) {
  const int64_t E = c.num_experts;
  const std::string base0 = ExpertBase(layer, 0, cls);
  const tp::ShardRule rule = tp::RuleFor(base0, gm);
  const bool gate_up = std::strcmp(cls, "gate_up") == 0;

  // The layout this class resolves to: the requested one if expert 0 carries it, else bf16 (the loader's three-tier chain; a class is uniform).
  Layout layout = requested;
  bool bare = false;
  if (!HasLayout(r, meta, base0, requested)) {
    layout = Layout::kBf16;
    if (!HasLayout(r, meta, base0, Layout::kBf16)) {
      if (!r.Has(base0)) throw std::runtime_error(kWho + "no tensor found for '" + base0 + "'");
      bare = true;
    }
  }
  const auto name_of = [&](int64_t e, const char* tail) { return ExpertBase(layer, e, cls) + tail; };
  st.layout = layout;
  st.experts = E;
  st.bits = 0;

  // the rank's slice shape
  int64_t rn = N, rk = K;
  std::vector<tp::Range> rows;
  tp::Range cols{0, K};
  if (rule.split == tp::Split::kRows) {
    rows = tp::RankRows(rule, world, rank);
    rn = 0;
    for (const tp::Range& g : rows) rn += g.count;
  } else {
    cols = tp::RankCols(rule, world, rank);
    rk = cols.count;
  }
  st.N = rn;
  st.K = rk;
  const auto plan_of = [&](const tp::PartShape& sh) {
    return rule.split == tp::Split::kRows ? tp::PlanRows(sh, rows) : tp::PlanCols(sh, cols);
  };
  const auto shape = [&](tp::Part p, int group) {
    tp::PartShape s;
    s.part = p;
    s.N = N;
    s.K = K;
    s.group = group;
    return s;
  };

  switch (layout) {
    case Layout::kBf16: {
      const std::vector<tp::ByteRun> plan = plan_of(shape(tp::Part::kBf16, 0));
      const size_t per = RunsBytes(plan);  // bytes of one expert's slice
      std::vector<uint8_t> host(static_cast<size_t>(E) * per);
      for (int64_t e = 0; e < E; ++e) {
        const std::string name = bare ? ExpertBase(layer, e, cls) : name_of(e, ".bf16.w");
        if (SpanOf(r, name) != static_cast<uint64_t>(N) * static_cast<uint64_t>(K) * 2) {
          throw std::runtime_error(kWho + "tensor '" + name + "' has the wrong size for a bf16 [" + std::to_string(N) + ", " +
                                   std::to_string(K) + "]");
        }
        GatherInto(r, name, plan, host.data() + static_cast<size_t>(e) * per);
        if (!bare && fallbacks != nullptr && layout != requested) ++*fallbacks;
      }
      st.bf16_w = core::DeviceBuffer<uint16_t>(host.size() / 2);
      CopyToDevice(st.bf16_w.data(), host.data(), host.size());
      const size_t nk = per / 2;
      for (int64_t e = 0; e < E; ++e) {
        QuantLinear q;
        q.layout = Layout::kBf16;
        q.N = rn;
        q.K = rk;
        q.bf16_w = core::DeviceBuffer<uint16_t>::View(st.bf16_w.data() + static_cast<size_t>(e) * nk, nk);
        experts[static_cast<size_t>(e)].*member = std::move(q);
      }
      break;
    }
    case Layout::kW4a16: {
      const int group = meta.w4a16.Resolve(base0);
      const std::string wsz0 = meta.w4a16.groups.WszName(base0);
      CheckW4a16Shape(r, base0, wsz0, N, K, meta.w4a16.KernelGroup(base0));
      const int kgroup = meta.w4a16.KernelGroup(base0);
      const std::vector<tp::ByteRun> pq = plan_of(shape(tp::Part::kW4Wq, 0)), pz = plan_of(shape(tp::Part::kW4a16Wsz, kgroup));
      const size_t perq = RunsBytes(pq), perz = RunsBytes(pz);
      std::vector<uint8_t> hq(static_cast<size_t>(E) * perq), hz(static_cast<size_t>(E) * perz);
      for (int64_t e = 0; e < E; ++e) {
        const std::string base = ExpertBase(layer, e, cls);
        GatherInto(r, base + ".w4a16.wq", pq, hq.data() + static_cast<size_t>(e) * perq);
        GatherInto(r, meta.w4a16.groups.WszName(base), pz, hz.data() + static_cast<size_t>(e) * perz);
      }
      st.wq = core::DeviceBuffer<uint8_t>(hq.size());
      st.wsz = core::DeviceBuffer<uint32_t>(hz.size() / 4);
      CopyToDevice(st.wq.data(), hq.data(), hq.size());
      CopyToDevice(st.wsz.data(), hz.data(), hz.size());
      for (int64_t e = 0; e < E; ++e) {
        QuantLinear q;
        q.layout = Layout::kW4a16;
        q.N = rn;
        q.K = rk;
        q.w4a16_group = group;
        q.wq = core::DeviceBuffer<uint8_t>::View(st.wq.data() + static_cast<size_t>(e) * perq, perq);
        q.w4a16_wsz = core::DeviceBuffer<uint32_t>::View(st.wsz.data() + static_cast<size_t>(e) * (perz / 4), perz / 4);
        experts[static_cast<size_t>(e)].*member = std::move(q);
      }
      break;
    }
    case Layout::kW8a16:
      throw std::runtime_error(kWho + "layer " + std::to_string(layer) + " class " + cls + ": w8a16 is not supported for qwen3_5_moe experts");
    case Layout::kTrellis: {
      const TrellisLinearSpec& ts0 = TrellisFor(meta, base0);
      const int bits = ts0.bits;
      st.bits = bits;
      const q35tp::TrellisSlicePlan p0 =
          q35tp::PlanTrellisLinear(rule, N, K, bits, ts0.parts, world, rank);  // identical for every expert of the class
      const size_t w_per = p0.w_bytes / 4, suh_per = p0.suh_elems, svh_per = p0.svh_elems;
      std::vector<uint8_t> hw(static_cast<size_t>(E) * p0.w_bytes);
      std::vector<float> hsuh(static_cast<size_t>(E) * suh_per), hsvh(static_cast<size_t>(E) * svh_per);
      std::vector<uint8_t> tmp;
      for (int64_t e = 0; e < E; ++e) {
        const std::string base = ExpertBase(layer, e, cls);
        const TrellisLinearSpec& ts = TrellisFor(meta, base);
        CheckTrellisShape(r, base, ts, N, K, meta.path);
        if (ts.bits != bits || ts.parts != ts0.parts) {
          throw std::runtime_error(kWho + "expert " + std::to_string(e) + " of layer " + std::to_string(layer) + " class " + cls +
                                   " has another KB / parts than expert 0 of its class");
        }
        const q35tp::TrellisSlicePlan p = q35tp::PlanTrellisLinear(rule, N, K, bits, ts.parts, world, rank);
        GatherInto(r, base + ".trellis.w", p.w, hw.data() + static_cast<size_t>(e) * p0.w_bytes);
        tmp.resize(p.suh_elems * 2);
        GatherInto(r, base + ".trellis.suh", p.suh, tmp.data());
        WidenF16(tmp.data(), p.suh_elems, hsuh.data() + static_cast<size_t>(e) * suh_per);
        tmp.resize(p.svh_elems * 2);
        GatherInto(r, base + ".trellis.svh", p.svh, tmp.data());
        WidenF16(tmp.data(), p.svh_elems, hsvh.data() + static_cast<size_t>(e) * svh_per);
      }
      st.trellis_w = core::DeviceBuffer<uint32_t>(static_cast<size_t>(E) * w_per);
      st.trellis_suh = core::DeviceBuffer<float>(hsuh.size());
      st.trellis_svh = core::DeviceBuffer<float>(hsvh.size());
      CopyToDevice(st.trellis_w.data(), hw.data(), hw.size());
      CopyToDevice(st.trellis_suh.data(), hsuh.data(), hsuh.size() * sizeof(float));
      CopyToDevice(st.trellis_svh.data(), hsvh.data(), hsvh.size() * sizeof(float));
      for (int64_t e = 0; e < E; ++e) {
        QuantLinear q;
        q.layout = Layout::kTrellis;
        q.N = p0.N;
        q.K = p0.K;
        SetTrellisFields(q, ts0);
        q.trellis_tp_shard = true;
        q.trellis_part_n[0] = p0.part_n[0];
        q.trellis_part_n[1] = p0.part_n[1];
        q.trellis_w = core::DeviceBuffer<uint32_t>::View(st.trellis_w.data() + static_cast<size_t>(e) * w_per, w_per);
        q.trellis_suh = core::DeviceBuffer<float>::View(st.trellis_suh.data() + static_cast<size_t>(e) * suh_per, suh_per);
        q.trellis_svh = core::DeviceBuffer<float>::View(st.trellis_svh.data() + static_cast<size_t>(e) * svh_per, svh_per);
        experts[static_cast<size_t>(e)].*member = std::move(q);
      }
      break;
    }
  }
  (void)gate_up;
}

}  // namespace

std::shared_ptr<const core::PinnedBuffer<uint16_t>> Qwen35MoeContainer::LoadEmbeddingHost(const std::string& path) {
  SafetensorsReader reader(Utf8ToWide(path));
  const int64_t n = ElemCountBySize(reader, "text.embed_tokens", 2);
  auto buf = std::make_shared<core::PinnedBuffer<uint16_t>>(static_cast<size_t>(n), hipHostMallocPortable);
  std::memcpy(buf->data(), reader.Data("text.embed_tokens"), static_cast<size_t>(n) * 2);
  return buf;
}

Qwen35MoeContainer Qwen35MoeContainer::LoadShard(const std::string& path, const Qwen35MoeLoadOptions& o) {
  const core::DirectLoadScope direct_load;
  if (o.tp_world != 2 && o.tp_world != 1) throw std::invalid_argument(kWho + "tp_world must be 1 or 2");
  if (o.tp_rank < 0 || o.tp_rank >= o.tp_world) throw std::invalid_argument(kWho + "tp_rank out of range");
  Qwen35MoeContainer c;
  c.info_ = Inspect(path, o.layout, o.lm_head_layout);
  const Qwen35MoeConfig& cfg = c.info_.config;  // GLOBAL
  c.local_config_ = Qwen35MoeConfig::Shard(cfg, o.tp_world, o.tp_rank);
  const int64_t last = o.layer_last < 0 ? c.info_.layers_present : o.layer_last;
  if (o.layer_first < 0 || o.layer_first > last || last > c.info_.layers_present) {
    throw std::invalid_argument(kWho + "layer range [" + std::to_string(o.layer_first) + ", " + std::to_string(last) +
                                ") is outside the " + std::to_string(c.info_.layers_present) + " layers of " + path);
  }
  DeviceScope scope(o.device);
  R4DX_HIP_CHECK(hipGetDevice(&c.device_));

  SafetensorsReader reader(Utf8ToWide(path));
  LinearLoadMeta meta;
  meta.path = path;
  meta.trellis = ParseTrellisMetadata(c.info_.metadata, path);
  CheckTrellisChoice(meta.trellis, /*rotated=*/false, c.info_.body_layout, path);
  meta.w4a16 = CheckQuantGroups(c.info_.metadata, path, c.info_.body_layout, c.info_.head_layout, Layout::kBf16);
  CheckW4a16GroupTensors(reader, meta.w4a16.groups, path);
  CheckTrellisTensors(reader, meta.trellis, path);
  const Layout body = c.info_.body_layout;
  const ModelConfig& gm = c.info_.model_config;  // the unsharded generic config tp::RuleFor classifies against
  ShardLoader L(reader, gm, o.tp_world, o.tp_rank, meta);
  int fallbacks = 0;

  const int64_t H = cfg.hidden_size, I = cfg.moe_intermediate_size, Si = cfg.shared_expert_intermediate_size;
  const int64_t value_dim = cfg.ValueDim();
  const int64_t conv_dim = cfg.ConvDim();
  const int64_t qdim = cfg.num_attention_heads * cfg.head_dim, kvdim = cfg.num_key_value_heads * cfg.head_dim;

  c.first_layer_ = o.layer_first;
  c.layers_.reserve(static_cast<size_t>(last - o.layer_first));
  for (int64_t i = o.layer_first; i < last; ++i) {
    const std::string b = "text.layers." + std::to_string(i) + ".";
    qwen35moe::LayerWeights lw;
    lw.full_attention = !cfg.IsGdnLayer(i);
    lw.input_layernorm = L.Raw<uint16_t>(b + "input_layernorm");
    lw.post_attention_layernorm = L.Raw<uint16_t>(b + "post_attention_layernorm");
    if (lw.full_attention) {
      qwen35moe::AttnWeights a;
      a.qg = L.Linear(b + "attn.qg", body, 2 * qdim, H, &fallbacks);
      a.k = L.Linear(b + "attn.k", body, kvdim, H, &fallbacks);
      a.v = L.Linear(b + "attn.v", body, kvdim, H, &fallbacks);
      a.o = L.Linear(b + "attn.o", body, H, qdim, &fallbacks);
      a.q_norm = L.Raw<uint16_t>(b + "attn.q_norm");
      a.k_norm = L.Raw<uint16_t>(b + "attn.k_norm");
      a.k_descale = L.Raw<float>(b + "attn.k_descale");
      a.v_descale = L.Raw<float>(b + "attn.v_descale");
      lw.attn = std::move(a);
    } else {
      qwen4exp::GdnWeights g;
      g.in_proj_qkv = L.Linear(b + "gdn.in_proj_qkv", body, conv_dim, H, &fallbacks);
      g.in_proj_z = L.Linear(b + "gdn.in_proj_z", body, value_dim, H, &fallbacks);
      g.in_proj_a = L.Raw<uint16_t>(b + "gdn.in_proj_a");
      g.in_proj_b = L.Raw<uint16_t>(b + "gdn.in_proj_b");
      g.conv1d_weight = L.Raw<uint16_t>(b + "gdn.conv1d_weight");
      g.A_log = L.Raw<float>(b + "gdn.A_log");
      g.dt_bias = L.Raw<float>(b + "gdn.dt_bias");
      g.norm_weight = L.WidenedF32(b + "gdn.norm_weight");
      g.out_proj = L.Linear(b + "gdn.out_proj", body, H, value_dim, &fallbacks);
      lw.gdn = std::move(g);
    }
    MoeWeights& m = lw.moe;
    m.router = L.Raw<uint16_t>(b + "moe.router");
    m.shared_gate = L.Raw<uint16_t>(b + "moe.shared_gate");
    m.shared.gate = L.Linear(b + "moe.shared.gate", Layout::kBf16, Si, H, &fallbacks);
    m.shared.up = L.Linear(b + "moe.shared.up", Layout::kBf16, Si, H, &fallbacks);
    m.shared.down = L.Linear(b + "moe.shared.down", Layout::kBf16, H, Si, &fallbacks);
    m.experts.resize(static_cast<size_t>(cfg.num_experts));
    LoadExpertClassSharded(reader, meta, gm, body, cfg, o.tp_world, o.tp_rank, i, "gate_up", 2 * I, H, m.gate_up_store, m.experts,
                           &ExpertLinears::gate_up, &fallbacks);
    LoadExpertClassSharded(reader, meta, gm, body, cfg, o.tp_world, o.tp_rank, i, "down", H, I, m.down_store, m.experts,
                           &ExpertLinears::down, &fallbacks);
    c.layers_.push_back(std::move(lw));
  }

  if (o.load_head) {
    c.final_norm_ = L.Raw<uint16_t>("text.final_norm");
    c.lm_head_ = L.Linear("lm_head", c.info_.head_layout, cfg.vocab_size, H, &fallbacks);  // this rank's vocab rows
    c.has_head_ = true;
  }
  if (o.load_embed) {
    if (o.shared_embed_host) {
      const int64_t n = ElemCountBySize(reader, "text.embed_tokens", 2);
      if (o.shared_embed_host->size() != static_cast<size_t>(n)) {
        throw std::invalid_argument(kWho + "shared_embed_host holds " + std::to_string(o.shared_embed_host->size()) +
                                    " elements, text.embed_tokens has " + std::to_string(n));
      }
      c.embed_host_ = o.shared_embed_host;
    } else {
      const int64_t n = ElemCountBySize(reader, "text.embed_tokens", 2);
      auto buf = std::make_shared<core::PinnedBuffer<uint16_t>>(static_cast<size_t>(n), hipHostMallocPortable);
      std::memcpy(buf->data(), reader.Data("text.embed_tokens"), static_cast<size_t>(n) * 2);
      c.embed_host_ = std::move(buf);
    }
  }
  if (fallbacks > 0) {
    std::fprintf(stderr, "r4dx: %d linear(s) in %s do not carry the requested layout and were loaded as bf16\n", fallbacks, path.c_str());
  }
  if (meta.trellis) c.AssignTrellisTickets();
  const ShardLoadStats& st = L.Stats();
  std::fprintf(stderr, "[r4dx::model::Qwen35MoeContainer] rank %d/%d: %d tensors sliced, %d replicated, %.3f GiB uploaded by the shard loader\n",
               o.tp_rank, o.tp_world, st.sharded, st.replicated, static_cast<double>(st.uploaded_bytes) / (1024.0 * 1024.0 * 1024.0));
  return c;
}

}  // namespace r4dx::model
