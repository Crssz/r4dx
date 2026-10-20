#include "qwen35moe_container.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <exception>
#include <stdexcept>

#include "container_load_util.h"
#include "r4dx/core/direct_storage.hpp"
#include "r4dx/core/error.hpp"
#include "r4dx/core/r4d.hpp"  // GemmTrellisTicketsBytes / GemmTrellisZeroTickets

namespace r4dx::model {

using namespace container_util;
using qwen4exp::ExpertClassStorage;
using qwen4exp::ExpertLinears;
using qwen4exp::MoeWeights;

namespace {

const std::string kWho = "r4dx::model::Qwen35MoeContainer: ";

// ---- name / size helpers (the same tiers and rules as qwen4exp_container.cpp) ---------------------------------------

std::string LayerPrefix(int64_t i) { return "text.layers." + std::to_string(i) + "."; }

uint64_t SpanOf(const SafetensorsReader& r, const std::string& name) {
  const auto& m = r.Meta(name);
  return m.end - m.begin;
}

int64_t TicketBytes(int64_t N) { return static_cast<int64_t>(core::r4d::GemmTrellisTicketsBytes(static_cast<int>(N))); }

// The on-disk form a linear resolves to: the requested layout, its bf16 twin, or the bare tensor (the three tiers
// of LoadQuantLinearWithFallback, whose rules this mirrors exactly).
struct Tier {
  enum Kind { kLayout, kBf16W, kBare } kind = kLayout;
  Layout layout = Layout::kBf16;
  int bits = 0;   // trellis: the linear's KB
  int group = 0;  // w4a16: QuantLinear::w4a16_group
  bool SameAs(const Tier& o) const {
    return kind == o.kind && layout == o.layout && bits == o.bits && group == o.group;
  }
};

Tier ResolveTier(const SafetensorsReader& r, const LinearLoadMeta& meta, const std::string& base, Layout requested) {
  Tier t;
  if (HasLayout(r, meta, base, requested)) {
    t.kind = Tier::kLayout;
    t.layout = requested;
    if (requested == Layout::kTrellis) t.bits = TrellisFor(meta, base).bits;
    if (requested == Layout::kW4a16) t.group = meta.w4a16.Resolve(base);
    return t;
  }
  if (requested == Layout::kW4a16 && meta.w4a16.groups.Mapped(base)) {
    throw std::runtime_error(kWho + "'" + base + "' is listed in __metadata__.quant.w4a16.groups but '" +
                             meta.w4a16.groups.WszName(base) + "' is missing; not falling back");
  }
  t.layout = Layout::kBf16;
  if (HasLayout(r, meta, base, Layout::kBf16)) {
    t.kind = Tier::kBf16W;
    return t;
  }
  if (r.Has(base)) {
    t.kind = Tier::kBare;
    return t;
  }
  throw std::runtime_error(kWho + "no tensor found for '" + base + "' in any known on-disk form (requested " +
                           LayoutName(requested) + ", bf16, or bare)");
}

struct Bytes {
  int64_t device = 0;   // what the upload allocates (sign vectors widened fp16 -> fp32)
  int64_t tickets = 0;  // this linear's ticket slice (trellis)
};

Bytes LinearBytes(const SafetensorsReader& r, const LinearLoadMeta& meta, const std::string& base, const Tier& t,
                  int64_t N, int64_t K) {
  Bytes b;
  const uint64_t nk = static_cast<uint64_t>(N) * static_cast<uint64_t>(K);
  if (t.kind != Tier::kLayout || t.layout == Layout::kBf16) {
    const std::string name = t.kind == Tier::kBare ? base : base + ".bf16.w";
    if (SpanOf(r, name) != nk * 2) {
      throw std::runtime_error(kWho + "tensor '" + name + "' is " + std::to_string(SpanOf(r, name)) +
                               " bytes, a bf16 [" + std::to_string(N) + ", " + std::to_string(K) + "] needs " +
                               std::to_string(nk * 2));
    }
    b.device = static_cast<int64_t>(nk * 2);
    return b;
  }
  if (t.layout == Layout::kW4a16) {
    const std::string wsz = meta.w4a16.groups.WszName(base);
    CheckW4a16Shape(r, base, wsz, N, K, meta.w4a16.KernelGroup(base));
    b.device = static_cast<int64_t>(SpanOf(r, base + ".w4a16.wq") + SpanOf(r, wsz));
    return b;
  }
  const TrellisLinearSpec& ts = TrellisFor(meta, base);
  CheckTrellisShape(r, base, ts, N, K, meta.path);
  b.device = static_cast<int64_t>(SpanOf(r, base + ".trellis.w") + 2 * SpanOf(r, base + ".trellis.suh") +
                                  2 * SpanOf(r, base + ".trellis.svh"));
  b.tickets = TicketBytes(N);
  return b;
}

LinearLoadMeta MakeMeta(const SafetensorsReader& reader, const nlohmann::json& metadata, const std::string& path,
                        Layout body, Layout head) {
  LinearLoadMeta meta;
  meta.path = path;
  meta.trellis = ParseTrellisMetadata(metadata, path);
  CheckTrellisChoice(meta.trellis, /*rotated=*/false, body, path);
  meta.w4a16 = CheckQuantGroups(metadata, path, body, head, Layout::kBf16);
  CheckW4a16GroupTensors(reader, meta.w4a16.groups, path);
  CheckTrellisTensors(reader, meta.trellis, path);
  return meta;
}

std::string ExpertBase(int64_t layer, int64_t e, const char* cls) {
  return LayerPrefix(layer) + "moe.experts." + std::to_string(e) + "." + cls;
}

// ---- Inspect's table ------------------------------------------------------------------------------------------------

struct Problems {
  std::vector<std::string> list;
  template <class Fn>
  void Guard(Fn&& fn) {
    try {
      fn();
    } catch (const std::exception& e) {
      list.push_back(e.what());
    }
  }
};

// A raw tensor of `elems` elements of `elem_bytes` each; returns its device bytes (`device_elem_bytes` per element,
// differing from the disk width for the widened ones).
int64_t Raw(const SafetensorsReader& r, Problems& bad, const std::string& name, int64_t elems, int64_t elem_bytes,
            int64_t device_elem_bytes = 0) {
  if (!r.Has(name)) {
    bad.list.push_back("missing tensor '" + name + "'");
    return 0;
  }
  if (SpanOf(r, name) != static_cast<uint64_t>(elems * elem_bytes)) {
    bad.list.push_back("tensor '" + name + "' is " + std::to_string(SpanOf(r, name)) + " bytes, expected " +
                       std::to_string(elems * elem_bytes) + " (" + std::to_string(elems) + " x " +
                       std::to_string(elem_bytes) + ")");
    return 0;
  }
  return elems * (device_elem_bytes != 0 ? device_elem_bytes : elem_bytes);
}

struct ExpertClassInfo {
  int64_t device = 0, tickets = 0;
  int bits = 0;
};

ExpertClassInfo InspectExpertClass(const SafetensorsReader& r, const LinearLoadMeta& meta, Layout requested,
                                   const Qwen35MoeConfig& c, int64_t layer, const char* cls, int64_t N, int64_t K,
                                   Problems& bad) {
  ExpertClassInfo out;
  const bool gate_up = std::strcmp(cls, "gate_up") == 0;
  Tier first;
  for (int64_t e = 0; e < c.num_experts; ++e) {
    const std::string base = ExpertBase(layer, e, cls);
    const size_t problems_before = bad.list.size();
    bad.Guard([&] {
      const Tier t = ResolveTier(r, meta, base, requested);
      if (e == 0) {
        first = t;
      } else if (!t.SameAs(first)) {
        throw std::runtime_error(kWho + "expert " + std::to_string(e) + " of layer " + std::to_string(layer) + " class " +
                                 cls + " resolves to a different layout / KB / group than expert 0 (a class must be " +
                                 "uniform per layer: one launch config serves every expert)");
      }
      if (t.kind == Tier::kLayout && t.layout == Layout::kTrellis) {
        const TrellisLinearSpec& ts = TrellisFor(meta, base);
        const bool parts_ok = gate_up ? ts.parts == std::vector<int64_t>{c.moe_intermediate_size, c.moe_intermediate_size}
                                      : ts.parts.empty();
        if (!parts_ok) {
          throw std::runtime_error(kWho + "trellis linear '" + base + "' has " + std::to_string(ts.Parts()) +
                                   " part(s); gate_up needs [" + std::to_string(c.moe_intermediate_size) + ", " +
                                   std::to_string(c.moe_intermediate_size) + "], down a single part");
        }
      }
      const Bytes b = LinearBytes(r, meta, base, t, N, K);
      out.device += b.device;
      out.tickets += b.tickets;
    });
    if (bad.list.size() > problems_before) break;  // one problem per class is enough to refuse the container
  }
  out.bits = first.kind == Tier::kLayout && first.layout == Layout::kTrellis ? first.bits : 0;
  return out;
}

}  // namespace

// ---- Info -----------------------------------------------------------------------------------------------------------

std::vector<int64_t> Qwen35MoeContainerInfo::PerLayerDeviceBytes() const {
  std::vector<int64_t> v;
  v.reserve(layer_bytes.size());
  for (const Qwen35MoeLayerBytes& b : layer_bytes) v.push_back(b.Total());
  return v;
}

int64_t Qwen35MoeContainerInfo::TotalDeviceBytes() const {
  int64_t t = TailDeviceBytes();
  for (const Qwen35MoeLayerBytes& b : layer_bytes) t += b.Total();
  return t;
}

// ---- Inspect --------------------------------------------------------------------------------------------------------

Qwen35MoeContainerInfo Qwen35MoeContainer::Inspect(const std::string& path, std::optional<Layout> body,
                                                   std::optional<Layout> head) {
  SafetensorsReader reader(Utf8ToWide(path));
  const nlohmann::json header = nlohmann::json::parse(reader.HeaderJson());
  if (!header.contains("__metadata__")) throw std::runtime_error(kWho + path + " has no __metadata__");
  const nlohmann::json& metadata = header.at("__metadata__");
  const std::string who = kWho + path + ": ";
  if (!metadata.is_object()) throw std::runtime_error(who + "__metadata__ is not an object");
  if (metadata.value("model_arch", std::string()) != "qwen3_5_moe") {
    throw std::runtime_error(who + "__metadata__.model_arch is '" + metadata.value("model_arch", std::string()) +
                             "', not qwen3_5_moe");
  }
  if (!metadata.contains("model_config")) throw std::runtime_error(who + "no __metadata__.model_config");

  Qwen35MoeContainerInfo info;
  info.path = path;
  info.metadata = metadata;
  info.model_id = metadata.value("model_id", std::string());
  info.config_sha256 = metadata.value("config_sha256", std::string());
  info.config = Qwen35MoeConfig::FromJson(metadata.at("model_config"));
  info.model_config = info.config.ToModelConfig();
  info.tensor_names = reader.Names();
  std::sort(info.tensor_names.begin(), info.tensor_names.end());

  const Qwen35MoeConfig& c = info.config;
  info.layers_present = c.num_hidden_layers;
  if (metadata.contains("layers_present")) {
    const nlohmann::json& lp = metadata.at("layers_present");
    if (!lp.is_number_integer() || lp.get<int64_t>() < 1 || lp.get<int64_t>() > c.num_hidden_layers) {
      throw std::runtime_error(who + "__metadata__.layers_present is " + lp.dump() + ", want an integer in [1, " +
                               std::to_string(c.num_hidden_layers) + "]");
    }
    info.layers_present = lp.get<int64_t>();
  }
  info.trellis = ParseTrellisMetadata(metadata, path);
  info.has_trellis = info.trellis.has_value();
  info.body_layout = body ? *body : (info.has_trellis ? Layout::kTrellis : Layout::kBf16);
  info.head_layout = head ? (*head == Layout::kTrellis ? Layout::kW4a16 : *head)
                          : (reader.Has("lm_head.w4a16.wq") ? Layout::kW4a16 : Layout::kBf16);
  const LinearLoadMeta meta = MakeMeta(reader, metadata, path, info.body_layout, info.head_layout);

  const int64_t H = c.hidden_size;
  const int64_t I = c.moe_intermediate_size, Si = c.shared_expert_intermediate_size, E = c.num_experts;
  const int64_t value_dim = c.ValueDim();
  const int64_t conv_dim = c.ConvDim();
  const int64_t qdim = c.num_attention_heads * c.head_dim, kvdim = c.num_key_value_heads * c.head_dim;

  Problems bad;
  const auto linear = [&](const std::string& base, Layout lay, int64_t N, int64_t K, int64_t* device, int64_t* tickets) {
    bad.Guard([&] {
      const Bytes b = LinearBytes(reader, meta, base, ResolveTier(reader, meta, base, lay), N, K);
      *device += b.device;
      *tickets += b.tickets;
    });
  };

  info.embed_host_bytes = Raw(reader, bad, "text.embed_tokens", c.vocab_size * H, 2);
  info.final_norm_bytes = Raw(reader, bad, "text.final_norm", H, 2);
  {
    int64_t tk = 0;
    linear("lm_head", info.head_layout, c.vocab_size, H, &info.lm_head_bytes, &tk);
  }

  info.gate_up_kb.assign(static_cast<size_t>(info.layers_present), 0);
  info.down_kb.assign(static_cast<size_t>(info.layers_present), 0);
  info.layer_bytes.assign(static_cast<size_t>(info.layers_present), Qwen35MoeLayerBytes());
  for (int64_t i = 0; i < info.layers_present; ++i) {
    const std::string b = LayerPrefix(i);
    Qwen35MoeLayerBytes& lb = info.layer_bytes[static_cast<size_t>(i)];
    lb.norms = Raw(reader, bad, b + "input_layernorm", H, 2) + Raw(reader, bad, b + "post_attention_layernorm", H, 2);
    if (!c.IsGdnLayer(i)) {
      linear(b + "attn.qg", info.body_layout, 2 * qdim, H, &lb.mixer, &lb.tickets);
      linear(b + "attn.k", info.body_layout, kvdim, H, &lb.mixer, &lb.tickets);
      linear(b + "attn.v", info.body_layout, kvdim, H, &lb.mixer, &lb.tickets);
      linear(b + "attn.o", info.body_layout, H, qdim, &lb.mixer, &lb.tickets);
      lb.mixer += Raw(reader, bad, b + "attn.q_norm", c.head_dim, 2);
      lb.mixer += Raw(reader, bad, b + "attn.k_norm", c.head_dim, 2);
      lb.mixer += Raw(reader, bad, b + "attn.k_descale", c.num_key_value_heads, 4);
      lb.mixer += Raw(reader, bad, b + "attn.v_descale", c.num_key_value_heads, 4);
    } else {
      linear(b + "gdn.in_proj_qkv", info.body_layout, conv_dim, H, &lb.mixer, &lb.tickets);
      linear(b + "gdn.in_proj_z", info.body_layout, value_dim, H, &lb.mixer, &lb.tickets);
      lb.mixer += Raw(reader, bad, b + "gdn.in_proj_a", c.linear_num_value_heads * H, 2);
      lb.mixer += Raw(reader, bad, b + "gdn.in_proj_b", c.linear_num_value_heads * H, 2);
      lb.mixer += Raw(reader, bad, b + "gdn.conv1d_weight", conv_dim * c.linear_conv_kernel_dim, 2);
      lb.mixer += Raw(reader, bad, b + "gdn.A_log", c.linear_num_value_heads, 4);
      lb.mixer += Raw(reader, bad, b + "gdn.dt_bias", c.linear_num_value_heads, 4);
      lb.mixer += Raw(reader, bad, b + "gdn.norm_weight", c.linear_value_head_dim, 2, 4);  // widened to fp32
      linear(b + "gdn.out_proj", info.body_layout, H, value_dim, &lb.mixer, &lb.tickets);
    }
    lb.dense += Raw(reader, bad, b + "moe.router", E * H, 2);
    lb.dense += Raw(reader, bad, b + "moe.shared_gate", H, 2);
    int64_t unused_tk = 0;
    linear(b + "moe.shared.gate", Layout::kBf16, Si, H, &lb.dense, &unused_tk);
    linear(b + "moe.shared.up", Layout::kBf16, Si, H, &lb.dense, &unused_tk);
    linear(b + "moe.shared.down", Layout::kBf16, H, Si, &lb.dense, &unused_tk);
    const ExpertClassInfo gu = InspectExpertClass(reader, meta, info.body_layout, c, i, "gate_up", 2 * I, H, bad);
    const ExpertClassInfo dn = InspectExpertClass(reader, meta, info.body_layout, c, i, "down", H, I, bad);
    lb.experts = gu.device + dn.device;
    lb.tickets += gu.tickets + dn.tickets;
    info.gate_up_kb[static_cast<size_t>(i)] = gu.bits;
    info.down_kb[static_cast<size_t>(i)] = dn.bits;
  }
  if (!bad.list.empty()) {
    std::string msg = who + std::to_string(bad.list.size()) + " problem(s) in the tensor table:";
    for (size_t j = 0; j < bad.list.size() && j < 20; ++j) msg += "\n  - " + bad.list[j];
    if (bad.list.size() > 20) msg += "\n  ... and " + std::to_string(bad.list.size() - 20) + " more";
    throw std::runtime_error(msg);
  }
  return info;
}

// ---- Load -----------------------------------------------------------------------------------------------------------

namespace {

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

void CopyToDevice(void* dst, const void* src, size_t bytes) {
  if (bytes != 0) R4DX_HIP_CHECK(hipMemcpy(dst, src, bytes, hipMemcpyHostToDevice));
}

// fp16 bytes -> fp32 into `staging`, then to `dst` (the trellis sign vectors' device form).
void CopyWidenedF16(float* dst, const uint8_t* src, size_t elems, std::vector<float>& staging) {
  staging.resize(elems);
  for (size_t i = 0; i < elems; ++i) {
    uint16_t h;
    std::memcpy(&h, src + 2 * i, 2);
    staging[i] = core::F16ToFloat(h);
  }
  CopyToDevice(dst, staging.data(), elems * sizeof(float));
}

// One (layer, class) of routed experts: ONE buffer per tensor kind holding all `E` experts back to back, and
// per-expert QuantLinear views into it. `member` selects gate_up / down of ExpertLinears.
void LoadExpertClass(const SafetensorsReader& r, const LinearLoadMeta& meta, Layout requested, const Qwen35MoeConfig& c,
                     int64_t layer, const char* cls, int64_t N, int64_t K, ExpertClassStorage& st,
                     std::vector<ExpertLinears>& experts, QuantLinear ExpertLinears::*member, int* fallbacks) {
  const int64_t E = c.num_experts;
  const Tier first = ResolveTier(r, meta, ExpertBase(layer, 0, cls), requested);
  st.layout = first.layout;
  st.N = N;
  st.K = K;
  st.experts = E;
  st.bits = first.layout == Layout::kTrellis ? first.bits : 0;
  const size_t nk = static_cast<size_t>(N) * static_cast<size_t>(K);
  std::vector<float> staging;

  const auto base_of = [&](int64_t e) { return ExpertBase(layer, e, cls); };
  const auto tier_of = [&](int64_t e) {
    const Tier t = ResolveTier(r, meta, base_of(e), requested);
    if (!t.SameAs(first)) {
      throw std::runtime_error(kWho + "expert " + std::to_string(e) + " of layer " + std::to_string(layer) + " class " +
                               cls + " resolves to a different layout / KB / group than expert 0");
    }
    if (t.kind != Tier::kLayout && fallbacks != nullptr) ++*fallbacks;
    return t;
  };

  switch (first.layout) {
    case Layout::kBf16: {
      st.bf16_w = core::DeviceBuffer<uint16_t>(static_cast<size_t>(E) * nk);
      for (int64_t e = 0; e < E; ++e) {
        const Tier t = tier_of(e);
        const std::string name = t.kind == Tier::kBare ? base_of(e) : base_of(e) + ".bf16.w";
        static_cast<void>(LinearBytes(r, meta, base_of(e), t, N, K));  // checks the tensor's size
        uint16_t* dst = st.bf16_w.data() + static_cast<size_t>(e) * nk;
        CopyToDevice(dst, r.Data(name), nk * 2);
        QuantLinear q;
        q.layout = Layout::kBf16;
        q.N = N;
        q.K = K;
        q.bf16_w = core::DeviceBuffer<uint16_t>::View(dst, nk);
        experts[static_cast<size_t>(e)].*member = std::move(q);
      }
      break;
    }
    case Layout::kW4a16: {
      const std::string wsz0 = meta.w4a16.groups.WszName(base_of(0));
      const size_t wq_per = nk / 2;
      const size_t wsz_per = static_cast<size_t>(SpanOf(r, wsz0) / 4);
      st.wq = core::DeviceBuffer<uint8_t>(static_cast<size_t>(E) * wq_per);
      st.wsz = core::DeviceBuffer<uint32_t>(static_cast<size_t>(E) * wsz_per);
      for (int64_t e = 0; e < E; ++e) {
        const Tier t = tier_of(e);
        const std::string base = base_of(e);
        const std::string wsz = meta.w4a16.groups.WszName(base);
        CheckW4a16Shape(r, base, wsz, N, K, meta.w4a16.KernelGroup(base));
        uint8_t* dq = st.wq.data() + static_cast<size_t>(e) * wq_per;
        uint32_t* ds = st.wsz.data() + static_cast<size_t>(e) * wsz_per;
        CopyToDevice(dq, r.Data(base + ".w4a16.wq"), wq_per);
        CopyToDevice(ds, r.Data(wsz), wsz_per * 4);
        QuantLinear q;
        q.layout = Layout::kW4a16;
        q.N = N;
        q.K = K;
        q.w4a16_group = t.group;
        q.wq = core::DeviceBuffer<uint8_t>::View(dq, wq_per);
        q.w4a16_wsz = core::DeviceBuffer<uint32_t>::View(ds, wsz_per);
        experts[static_cast<size_t>(e)].*member = std::move(q);
      }
      break;
    }
    case Layout::kW8a16:
      throw std::runtime_error(kWho + "layer " + std::to_string(layer) + " class " + cls +
                               ": --layout w8a16 is not supported for qwen3_5_moe experts (use w4a16, bf16 or trellis)");
    case Layout::kTrellis: {
      const TrellisLinearSpec& ts0 = TrellisFor(meta, base_of(0));
      const size_t parts = static_cast<size_t>(ts0.Parts());
      const size_t w_per = nk * static_cast<size_t>(first.bits) / 32;
      const size_t suh_per = parts * static_cast<size_t>(K), svh_per = static_cast<size_t>(N);
      st.trellis_w = core::DeviceBuffer<uint32_t>(static_cast<size_t>(E) * w_per);
      st.trellis_suh = core::DeviceBuffer<float>(static_cast<size_t>(E) * suh_per);
      st.trellis_svh = core::DeviceBuffer<float>(static_cast<size_t>(E) * svh_per);
      for (int64_t e = 0; e < E; ++e) {
        tier_of(e);
        const std::string base = base_of(e);
        const TrellisLinearSpec& ts = TrellisFor(meta, base);
        CheckTrellisShape(r, base, ts, N, K, meta.path);
        if (ts.Parts() != ts0.Parts() || ts.parts != ts0.parts) {
          throw std::runtime_error(kWho + "trellis linear '" + base + "' has other parts than expert 0 of its class");
        }
        uint32_t* dw = st.trellis_w.data() + static_cast<size_t>(e) * w_per;
        float* dsuh = st.trellis_suh.data() + static_cast<size_t>(e) * suh_per;
        float* dsvh = st.trellis_svh.data() + static_cast<size_t>(e) * svh_per;
        CopyToDevice(dw, r.Data(base + ".trellis.w"), w_per * 4);
        CopyWidenedF16(dsuh, r.Data(base + ".trellis.suh"), suh_per, staging);
        CopyWidenedF16(dsvh, r.Data(base + ".trellis.svh"), svh_per, staging);
        QuantLinear q;
        q.layout = Layout::kTrellis;
        q.N = N;
        q.K = K;
        SetTrellisFields(q, ts);
        q.trellis_part_n[0] = ts.parts.empty() ? N : ts.parts[0];
        q.trellis_part_n[1] = ts.parts.empty() ? 0 : ts.parts[1];
        q.trellis_w = core::DeviceBuffer<uint32_t>::View(dw, w_per);
        q.trellis_suh = core::DeviceBuffer<float>::View(dsuh, suh_per);
        q.trellis_svh = core::DeviceBuffer<float>::View(dsvh, svh_per);
        experts[static_cast<size_t>(e)].*member = std::move(q);
      }
      break;
    }
  }
}

}  // namespace

Qwen35MoeContainer Qwen35MoeContainer::Load(const std::string& path, const Qwen35MoeLoadOptions& o) {
  const core::DirectLoadScope direct_load;  // docs/direct-storage.md
  Qwen35MoeContainer c;
  c.info_ = Inspect(path, o.layout, o.lm_head_layout);
  const Qwen35MoeConfig& cfg = c.info_.config;
  const int64_t last = o.layer_last < 0 ? c.info_.layers_present : o.layer_last;
  if (o.layer_first < 0 || o.layer_first > last || last > c.info_.layers_present) {
    const bool prefix = c.info_.layers_present < cfg.num_hidden_layers;
    throw std::invalid_argument(kWho + "layer range [" + std::to_string(o.layer_first) + ", " + std::to_string(last) +
                                ") is outside the " + std::to_string(c.info_.layers_present) + " layers " +
                                (prefix ? "present in the prefix container (config: " + std::to_string(cfg.num_hidden_layers) +
                                              ") "
                                        : std::string("of ")) +
                                path);
  }
  DeviceScope scope(o.device);
  R4DX_HIP_CHECK(hipGetDevice(&c.device_));

  SafetensorsReader reader(Utf8ToWide(path));
  const LinearLoadMeta meta = MakeMeta(reader, c.info_.metadata, path, c.info_.body_layout, c.info_.head_layout);
  if (meta.w4a16.check_default_per_linear) LogW4a16Groups(meta.w4a16.groups, path);
  const Layout body = c.info_.body_layout;
  int fallbacks = 0;
  const auto linear = [&](const std::string& base, int64_t N, int64_t K) {
    return LoadQuantLinearWithFallback(reader, meta, base, body, N, K, &fallbacks);
  };
  const auto raw_u16 = [&](const std::string& name) { return UploadRawU16(reader, name); };

  const int64_t H = cfg.hidden_size, I = cfg.moe_intermediate_size, Si = cfg.shared_expert_intermediate_size;
  const int64_t value_dim = cfg.ValueDim();
  const int64_t conv_dim = cfg.ConvDim();
  const int64_t qdim = cfg.num_attention_heads * cfg.head_dim, kvdim = cfg.num_key_value_heads * cfg.head_dim;

  c.first_layer_ = o.layer_first;
  c.layers_.reserve(static_cast<size_t>(last - o.layer_first));
  for (int64_t i = o.layer_first; i < last; ++i) {
    const std::string b = LayerPrefix(i);
    qwen35moe::LayerWeights lw;
    lw.full_attention = !cfg.IsGdnLayer(i);
    lw.input_layernorm = raw_u16(b + "input_layernorm");
    lw.post_attention_layernorm = raw_u16(b + "post_attention_layernorm");
    if (lw.full_attention) {
      qwen35moe::AttnWeights a;
      a.qg = linear(b + "attn.qg", 2 * qdim, H);
      a.k = linear(b + "attn.k", kvdim, H);
      a.v = linear(b + "attn.v", kvdim, H);
      a.o = linear(b + "attn.o", H, qdim);
      a.q_norm = raw_u16(b + "attn.q_norm");
      a.k_norm = raw_u16(b + "attn.k_norm");
      a.k_descale = UploadRawF32(reader, b + "attn.k_descale");
      a.v_descale = UploadRawF32(reader, b + "attn.v_descale");
      lw.attn = std::move(a);
    } else {
      qwen4exp::GdnWeights g;
      g.in_proj_qkv = linear(b + "gdn.in_proj_qkv", conv_dim, H);
      g.in_proj_z = linear(b + "gdn.in_proj_z", value_dim, H);
      g.in_proj_a = raw_u16(b + "gdn.in_proj_a");
      g.in_proj_b = raw_u16(b + "gdn.in_proj_b");
      g.conv1d_weight = raw_u16(b + "gdn.conv1d_weight");
      g.A_log = UploadRawF32(reader, b + "gdn.A_log");
      g.dt_bias = UploadRawF32(reader, b + "gdn.dt_bias");
      g.norm_weight = UploadWidenedF32(reader, b + "gdn.norm_weight");
      g.out_proj = linear(b + "gdn.out_proj", H, value_dim);
      lw.gdn = std::move(g);
    }
    MoeWeights& m = lw.moe;
    m.router = raw_u16(b + "moe.router");
    m.shared_gate = raw_u16(b + "moe.shared_gate");
    m.shared.gate = LoadQuantLinearWithFallback(reader, meta, b + "moe.shared.gate", Layout::kBf16, Si, H, &fallbacks);
    m.shared.up = LoadQuantLinearWithFallback(reader, meta, b + "moe.shared.up", Layout::kBf16, Si, H, &fallbacks);
    m.shared.down = LoadQuantLinearWithFallback(reader, meta, b + "moe.shared.down", Layout::kBf16, H, Si, &fallbacks);
    m.experts.resize(static_cast<size_t>(cfg.num_experts));
    LoadExpertClass(reader, meta, body, cfg, i, "gate_up", 2 * I, H, m.gate_up_store, m.experts,
                    &ExpertLinears::gate_up, &fallbacks);
    LoadExpertClass(reader, meta, body, cfg, i, "down", H, I, m.down_store, m.experts, &ExpertLinears::down, &fallbacks);
    c.layers_.push_back(std::move(lw));
  }

  if (o.load_head) {
    c.final_norm_ = raw_u16("text.final_norm");
    c.lm_head_ = LoadQuantLinearWithFallback(reader, meta, "lm_head", c.info_.head_layout, cfg.vocab_size, H, &fallbacks);
    c.has_head_ = true;
  }
  if (o.load_embed) {
    const int64_t n = ElemCountBySize(reader, "text.embed_tokens", 2);
    auto buf = std::make_shared<core::PinnedBuffer<uint16_t>>(static_cast<size_t>(n), hipHostMallocPortable);
    std::memcpy(buf->data(), reader.Data("text.embed_tokens"), static_cast<size_t>(n) * 2);
    c.embed_host_ = std::move(buf);
  }
  if (fallbacks > 0) {
    std::fprintf(stderr, "r4dx: %d linear(s) in %s do not carry the requested layout and were loaded as bf16\n",
                 fallbacks, path.c_str());
  }
  if (meta.trellis) c.AssignTrellisTickets();
  return c;
}

// ---- members --------------------------------------------------------------------------------------------------------

template <class Fn>
void Qwen35MoeContainer::ForEachLinear(Fn&& fn) {
  for (qwen35moe::LayerWeights& lw : layers_) {
    if (lw.attn) {
      fn(lw.attn->qg);
      fn(lw.attn->k);
      fn(lw.attn->v);
      fn(lw.attn->o);
    }
    if (lw.gdn) {
      fn(lw.gdn->in_proj_qkv);
      fn(lw.gdn->in_proj_z);
      fn(lw.gdn->out_proj);
    }
    for (ExpertLinears& e : lw.moe.experts) {
      fn(e.gate_up);
      fn(e.down);
    }
  }
}

void Qwen35MoeContainer::AssignTrellisTickets() {
  size_t total = 0;
  int64_t n = 0;
  ForEachLinear([&](QuantLinear& q) {
    if (q.layout != Layout::kTrellis) return;
    total += core::r4d::GemmTrellisTicketsBytes(static_cast<int>(q.N)) / sizeof(uint32_t);
    ++n;
  });
  trellis_linears_ = n;
  trellis_tickets_ = core::DeviceBuffer<uint32_t>(total);
  trellis_tickets_.Zero();
  size_t off = 0;
  ForEachLinear([&](QuantLinear& q) {
    if (q.layout != Layout::kTrellis) return;
    q.trellis_tickets = trellis_tickets_.data() + off;
    off += core::r4d::GemmTrellisTicketsBytes(static_cast<int>(q.N)) / sizeof(uint32_t);
  });
}

void Qwen35MoeContainer::ZeroTrellisTickets(hipStream_t stream) {
  if (trellis_tickets_.empty()) return;
  core::r4d::GemmTrellisZeroTickets(trellis_tickets_.data(), trellis_tickets_.bytes(), stream);
}

template <class Fn>
void Qwen35MoeContainer::ForEachBuffer(Fn&& fn) const {
  const auto linear = [&](const QuantLinear& q) {
    fn(q.bf16_w);
    fn(q.wq);
    fn(q.w4a16_wsz);
    fn(q.trellis_w);
    fn(q.trellis_suh);
    fn(q.trellis_svh);
    fn(q.trellis_wscale);
    fn(q.trellis_i8_sw);
    fn(q.trellis_i8_swc);
  };
  const auto store = [&](const ExpertClassStorage& s) {
    fn(s.bf16_w);
    fn(s.wq);
    fn(s.wsz);
    fn(s.trellis_w);
    fn(s.trellis_suh);
    fn(s.trellis_svh);
  };
  for (const qwen35moe::LayerWeights& lw : layers_) {
    fn(lw.input_layernorm);
    fn(lw.post_attention_layernorm);
    if (lw.attn) {
      const qwen35moe::AttnWeights& a = *lw.attn;
      linear(a.qg);
      linear(a.k);
      linear(a.v);
      linear(a.o);
      fn(a.q_norm);
      fn(a.k_norm);
      fn(a.k_descale);
      fn(a.v_descale);
    }
    if (lw.gdn) {
      const qwen4exp::GdnWeights& g = *lw.gdn;
      linear(g.in_proj_qkv);
      linear(g.in_proj_z);
      fn(g.in_proj_a);
      fn(g.in_proj_b);
      fn(g.conv1d_weight);
      fn(g.A_log);
      fn(g.dt_bias);
      fn(g.norm_weight);
      linear(g.out_proj);
    }
    const MoeWeights& m = lw.moe;
    fn(m.router);
    fn(m.shared_gate);
    linear(m.shared.gate);
    linear(m.shared.up);
    linear(m.shared.down);
    store(m.gate_up_store);
    store(m.down_store);
    for (const ExpertLinears& e : m.experts) {  // views: owned() is false, so they add nothing
      linear(e.gate_up);
      linear(e.down);
    }
  }
  if (has_head_) {
    fn(final_norm_);
    linear(lm_head_);
  }
  fn(trellis_tickets_);
}

int64_t Qwen35MoeContainer::DeviceBytes() const {
  int64_t total = 0;
  ForEachBuffer([&](const auto& b) {
    if (b.owned()) total += static_cast<int64_t>(b.bytes());
  });
  return total;
}

const qwen35moe::LayerWeights& Qwen35MoeContainer::Layer(int64_t i) const {
  if (i < first_layer_ || i >= EndLayer()) {
    throw std::out_of_range(kWho + "layer " + std::to_string(i) + " is outside the loaded range [" +
                            std::to_string(first_layer_) + ", " + std::to_string(EndLayer()) + ")");
  }
  return layers_[static_cast<size_t>(i - first_layer_)];
}

const uint16_t* Qwen35MoeContainer::EmbedTokensHost() const {
  if (!embed_host_) throw std::logic_error(kWho + "text.embed_tokens was not loaded (load_embed = false)");
  return embed_host_->data();
}

const core::DeviceBuffer<uint16_t>& Qwen35MoeContainer::FinalNorm() const {
  if (!has_head_) throw std::logic_error(kWho + "the head was not loaded (load_head = false)");
  return final_norm_;
}

const QuantLinear& Qwen35MoeContainer::LmHead() const {
  if (!has_head_) throw std::logic_error(kWho + "the head was not loaded (load_head = false)");
  return lm_head_;
}

}  // namespace r4dx::model
