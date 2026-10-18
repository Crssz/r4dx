#include "qwen4exp_container.h"

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
using qwen4exp::HcWeights;
using qwen4exp::MoeWeights;
using qwen4exp::PleWeights;
using qwen4exp::QsaWeights;

namespace {

const std::string kWho = "r4dx::model::Qwen4ExpContainer: ";

// ---- name / size helpers ----------------------------------------------------------------------------------------

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

// Checks `base`'s tensors at tier `t` against [N, K] (container_util's own checks for w4a16 / trellis) and returns
// the device bytes an upload of it allocates.
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

// ---- metadata ---------------------------------------------------------------------------------------------------

std::optional<Qwen4ExpNgramRef> ParseNgramRef(const nlohmann::json& metadata, const std::string& path) {
  if (!metadata.contains("ngram_sidecar") || metadata.at("ngram_sidecar").is_null()) return std::nullopt;
  const nlohmann::json& n = metadata.at("ngram_sidecar");
  const std::string who = kWho + path + ": __metadata__.ngram_sidecar ";
  if (!n.is_object()) throw std::runtime_error(who + "is not an object");
  Qwen4ExpNgramRef ref;
  const char* path_key = n.contains("path") ? "path" : (n.contains("file") ? "file" : nullptr);
  if (path_key == nullptr || !n.at(path_key).is_string() || n.at(path_key).get<std::string>().empty()) {
    throw std::runtime_error(who + "has no non-empty string 'path'");
  }
  ref.path = n.at(path_key).get<std::string>();
  if (!n.contains("sha256") || !n.at("sha256").is_string() || n.at("sha256").get<std::string>().size() != 64) {
    throw std::runtime_error(who + "has no 64-character 'sha256'");
  }
  ref.sha256 = n.at("sha256").get<std::string>();
  if (!n.contains("rows") || !n.at("rows").is_number_integer() || n.at("rows").get<int64_t>() <= 0) {
    throw std::runtime_error(who + "has no positive integer 'rows'");
  }
  ref.rows = n.at("rows").get<int64_t>();
  if (n.contains("row_bytes")) {
    if (!n.at("row_bytes").is_number_integer() || n.at("row_bytes").get<int64_t>() <= 0) {
      throw std::runtime_error(who + "'row_bytes' is not a positive integer");
    }
    ref.row_bytes = n.at("row_bytes").get<int64_t>();
  }
  const size_t slash = path.find_last_of("/\\");
  const bool absolute = ref.path.size() > 1 && (ref.path[1] == ':' || ref.path[0] == '/' || ref.path[0] == '\\');
  ref.resolved_path = absolute || slash == std::string::npos ? ref.path : path.substr(0, slash + 1) + ref.path;
  return ref;
}

// ---- Inspect's table --------------------------------------------------------------------------------------------

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
                                   const Qwen4ExpConfig& c, int64_t layer, const char* cls, int64_t N, int64_t K,
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

// ---- Info --------------------------------------------------------------------------------------------------------

std::vector<int64_t> Qwen4ExpContainerInfo::PerLayerDeviceBytes() const {
  std::vector<int64_t> v;
  v.reserve(layer_bytes.size());
  for (const Qwen4ExpLayerBytes& b : layer_bytes) v.push_back(b.Total());
  return v;
}

int64_t Qwen4ExpContainerInfo::TotalDeviceBytes() const {
  int64_t t = TailDeviceBytes();
  for (const Qwen4ExpLayerBytes& b : layer_bytes) t += b.Total();
  return t;
}

Qwen4ExpSplit ChooseSplit(const std::vector<int64_t>& per_layer_bytes, int64_t head_bytes, int64_t tail_bytes,
                          int64_t cap_bytes) {
  const std::string who = "r4dx::model::ChooseSplit: ";
  const int64_t L = static_cast<int64_t>(per_layer_bytes.size());
  if (L < 2) throw std::invalid_argument(who + "need at least two layers to split, got " + std::to_string(L));
  if (head_bytes < 0 || tail_bytes < 0 || cap_bytes <= 0) {
    throw std::invalid_argument(who + "head_bytes, tail_bytes must be >= 0 and cap_bytes > 0");
  }
  int64_t total = 0;
  for (int64_t b : per_layer_bytes) {
    if (b < 0) throw std::invalid_argument(who + "a layer has negative bytes");
    total += b;
  }
  Qwen4ExpSplit best;
  int64_t best_diff = 0;
  int64_t below = 0;
  for (int64_t k = 1; k < L; ++k) {
    below += per_layer_bytes[static_cast<size_t>(k - 1)];
    const int64_t d0 = head_bytes + below, d1 = tail_bytes + (total - below);
    if (d0 > cap_bytes || d1 > cap_bytes) continue;
    const int64_t diff = d0 > d1 ? d0 - d1 : d1 - d0;
    if (best.k == 0 || diff < best_diff) {
      best = {k, d0, d1};
      best_diff = diff;
    }
  }
  if (best.k == 0) {
    throw std::runtime_error(who + "no layer boundary keeps both cards within " + std::to_string(cap_bytes) +
                             " bytes: " + std::to_string(head_bytes + tail_bytes + total) + " bytes of weights (" +
                             std::to_string(total) + " in " + std::to_string(L) + " layers, head " +
                             std::to_string(head_bytes) + ", tail " + std::to_string(tail_bytes) + ")");
  }
  return best;
}

// ---- Inspect -----------------------------------------------------------------------------------------------------

Qwen4ExpContainerInfo Qwen4ExpContainer::Inspect(const std::string& path, std::optional<Layout> body,
                                                 std::optional<Layout> head) {
  SafetensorsReader reader(Utf8ToWide(path));
  const nlohmann::json header = nlohmann::json::parse(reader.HeaderJson());
  if (!header.contains("__metadata__")) throw std::runtime_error(kWho + path + " has no __metadata__");
  const nlohmann::json& metadata = header.at("__metadata__");
  const std::string who = kWho + path + ": ";
  if (!metadata.is_object()) throw std::runtime_error(who + "__metadata__ is not an object");
  if (metadata.value("model_arch", std::string()) != "qwen4_exp") {
    throw std::runtime_error(who + "__metadata__.model_arch is '" + metadata.value("model_arch", std::string()) +
                             "', not qwen4_exp");
  }
  if (!metadata.contains("model_config")) throw std::runtime_error(who + "no __metadata__.model_config");

  Qwen4ExpContainerInfo info;
  info.path = path;
  info.metadata = metadata;
  info.model_id = metadata.value("model_id", std::string());
  info.config_sha256 = metadata.value("config_sha256", std::string());
  info.config = Qwen4ExpConfig::FromJson(metadata.at("model_config"));
  info.model_config = info.config.ToModelConfig();
  info.ngram = ParseNgramRef(metadata, path);
  info.tensor_names = reader.Names();
  std::sort(info.tensor_names.begin(), info.tensor_names.end());

  const Qwen4ExpConfig& c = info.config;
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

  const int64_t H = c.hidden_size, S = c.hc_count * H, lr = c.hc_lowrank;
  const int64_t I = c.moe_intermediate_size, Si = c.shared_expert_intermediate_size, E = c.num_experts;
  const int64_t key_dim = c.linear_num_key_heads * c.linear_key_head_dim;
  const int64_t value_dim = c.linear_num_value_heads * c.linear_value_head_dim;
  const int64_t conv_dim = 2 * key_dim + value_dim;
  const int64_t qdim = c.num_attention_heads * c.head_dim, kvdim = c.num_key_value_heads * c.head_dim;
  const int64_t idx_rows = (c.indexer_n_heads + c.indexer_kv_heads) * c.indexer_head_dim;

  Problems bad;
  const auto linear = [&](const std::string& base, Layout lay, int64_t N, int64_t K, int64_t* device, int64_t* tickets) {
    bad.Guard([&] {
      const Bytes b = LinearBytes(reader, meta, base, ResolveTier(reader, meta, base, lay), N, K);
      *device += b.device;
      *tickets += b.tickets;
    });
  };
  const auto hc = [&](const std::string& b, bool inject) {
    int64_t n = 0;
    n += Raw(reader, bad, b + "norm", S, 2);
    n += Raw(reader, bad, b + "down", lr * S, 2);
    n += Raw(reader, bad, b + "up", S * lr, 2);
    if (inject) n += Raw(reader, bad, b + "inject", c.hc_count * S, 2);
    return n;
  };

  info.embed_host_bytes = Raw(reader, bad, "text.embed_tokens", c.vocab_size * H, 2);
  info.hc_final_bytes = hc("text.hc_final.", false);
  {
    int64_t tk = 0;
    linear("lm_head", info.head_layout, c.vocab_size, H, &info.lm_head_bytes, &tk);
  }

  info.gate_up_kb.assign(static_cast<size_t>(info.layers_present), 0);
  info.down_kb.assign(static_cast<size_t>(info.layers_present), 0);
  info.layer_bytes.assign(static_cast<size_t>(info.layers_present), Qwen4ExpLayerBytes());
  for (int64_t i = 0; i < info.layers_present; ++i) {
    const std::string b = LayerPrefix(i);
    Qwen4ExpLayerBytes& lb = info.layer_bytes[static_cast<size_t>(i)];
    lb.hc = hc(b + "hc.attn.", true) + hc(b + "hc.mlp.", true);
    if (c.IsQsaLayer(i)) {
      linear(b + "attn.qg", info.body_layout, 2 * qdim, H, &lb.mixer, &lb.tickets);
      linear(b + "attn.k", info.body_layout, kvdim, H, &lb.mixer, &lb.tickets);
      linear(b + "attn.v", info.body_layout, kvdim, H, &lb.mixer, &lb.tickets);
      linear(b + "attn.o", info.body_layout, H, qdim, &lb.mixer, &lb.tickets);
      lb.mixer += Raw(reader, bad, b + "attn.q_norm", c.head_dim, 2);
      lb.mixer += Raw(reader, bad, b + "attn.k_norm", c.head_dim, 2);
      lb.mixer += Raw(reader, bad, b + "attn.k_descale", c.num_key_value_heads, 4);
      lb.mixer += Raw(reader, bad, b + "attn.v_descale", c.num_key_value_heads, 4);
      lb.mixer += Raw(reader, bad, b + "attn.indexer.qk", idx_rows * H, 2);
      lb.mixer += Raw(reader, bad, b + "attn.indexer.q_norm", c.indexer_head_dim, 2);
      lb.mixer += Raw(reader, bad, b + "attn.indexer.k_norm", c.indexer_head_dim, 2);
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
    if (c.IsPleLayer(i)) {
      lb.ple += Raw(reader, bad, b + "ple.key_proj", S * H, 2);
      lb.ple += Raw(reader, bad, b + "ple.value_proj", H * H, 2);
      lb.ple += Raw(reader, bad, b + "ple.norm_key", S, 2);
      lb.ple += Raw(reader, bad, b + "ple.norm_query", S, 2);
      lb.ple += Raw(reader, bad, b + "ple.norm_conv", S, 2);
      lb.ple += Raw(reader, bad, b + "ple.conv1d", S * c.ple_conv_kernel_size, 2);
    } else if (reader.Has(b + "ple.key_proj")) {
      bad.list.push_back("layer " + std::to_string(i) + " carries ple.* but is not a PLE layer of the config");
    }
  }
  if (!bad.list.empty()) {
    std::string msg = who + std::to_string(bad.list.size()) + " problem(s) in the tensor table:";
    for (size_t j = 0; j < bad.list.size() && j < 20; ++j) msg += "\n  - " + bad.list[j];
    if (bad.list.size() > 20) msg += "\n  ... and " + std::to_string(bad.list.size() - 20) + " more";
    throw std::runtime_error(msg);
  }
  return info;
}

// ---- Load --------------------------------------------------------------------------------------------------------

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
// per-expert QuantLinear views into it (plan 4.11 item 5). `member` selects gate_up / down of ExpertLinears.
void LoadExpertClass(const SafetensorsReader& r, const LinearLoadMeta& meta, Layout requested, const Qwen4ExpConfig& c,
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
      // W8A16 (docs/w8a16.md) is wired for LFM2 only; qwen4_exp's grouped expert storage has no int8 form.
      throw std::runtime_error(kWho + "layer " + std::to_string(layer) + " class " + cls +
                               ": --layout w8a16 is not supported for qwen4_exp experts (use w4a16, bf16 or trellis)");
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

Qwen4ExpContainer Qwen4ExpContainer::Load(const std::string& path, const Qwen4ExpLoadOptions& o) {
  const core::DirectLoadScope direct_load;  // docs/direct-storage.md
  Qwen4ExpContainer c;
  c.info_ = Inspect(path, o.layout, o.lm_head_layout);
  const Qwen4ExpConfig& cfg = c.info_.config;
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
  const auto load_hc = [&](const std::string& b, bool inject) {
    HcWeights h;
    h.norm = raw_u16(b + "norm");
    h.down = raw_u16(b + "down");
    h.up = raw_u16(b + "up");
    if (inject) h.inject = raw_u16(b + "inject");
    return h;
  };

  const int64_t H = cfg.hidden_size, I = cfg.moe_intermediate_size, Si = cfg.shared_expert_intermediate_size;
  const int64_t key_dim = cfg.linear_num_key_heads * cfg.linear_key_head_dim;
  const int64_t value_dim = cfg.linear_num_value_heads * cfg.linear_value_head_dim;
  const int64_t conv_dim = 2 * key_dim + value_dim;
  const int64_t qdim = cfg.num_attention_heads * cfg.head_dim, kvdim = cfg.num_key_value_heads * cfg.head_dim;

  c.first_layer_ = o.layer_first;
  c.layers_.reserve(static_cast<size_t>(last - o.layer_first));
  for (int64_t i = o.layer_first; i < last; ++i) {
    const std::string b = LayerPrefix(i);
    qwen4exp::LayerWeights lw;
    lw.qsa = cfg.IsQsaLayer(i);
    lw.hc_attn = load_hc(b + "hc.attn.", true);
    lw.hc_mlp = load_hc(b + "hc.mlp.", true);
    if (lw.qsa) {
      QsaWeights a;
      a.qg = linear(b + "attn.qg", 2 * qdim, H);
      a.k = linear(b + "attn.k", kvdim, H);
      a.v = linear(b + "attn.v", kvdim, H);
      a.o = linear(b + "attn.o", H, qdim);
      a.q_norm = raw_u16(b + "attn.q_norm");
      a.k_norm = raw_u16(b + "attn.k_norm");
      a.k_descale = UploadRawF32(reader, b + "attn.k_descale");
      a.v_descale = UploadRawF32(reader, b + "attn.v_descale");
      a.indexer_qk = raw_u16(b + "attn.indexer.qk");
      a.indexer_q_norm = raw_u16(b + "attn.indexer.q_norm");
      a.indexer_k_norm = raw_u16(b + "attn.indexer.k_norm");
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
    if (cfg.IsPleLayer(i)) {
      PleWeights p;
      p.key_proj = raw_u16(b + "ple.key_proj");
      p.value_proj = raw_u16(b + "ple.value_proj");
      p.norm_key = raw_u16(b + "ple.norm_key");
      p.norm_query = raw_u16(b + "ple.norm_query");
      p.norm_conv = raw_u16(b + "ple.norm_conv");
      p.conv1d = raw_u16(b + "ple.conv1d");
      lw.ple = std::move(p);
    }
    c.layers_.push_back(std::move(lw));
  }

  if (o.load_head) {
    c.hc_final_ = load_hc("text.hc_final.", false);
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

// ---- members -----------------------------------------------------------------------------------------------------

template <class Fn>
void Qwen4ExpContainer::ForEachLinear(Fn&& fn) {
  for (qwen4exp::LayerWeights& lw : layers_) {
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

void Qwen4ExpContainer::AssignTrellisTickets() {
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

void Qwen4ExpContainer::ZeroTrellisTickets(hipStream_t stream) {
  if (trellis_tickets_.empty()) return;
  core::r4d::GemmTrellisZeroTickets(trellis_tickets_.data(), trellis_tickets_.bytes(), stream);
}

template <class Fn>
void Qwen4ExpContainer::ForEachBuffer(Fn&& fn) const {
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
  const auto hc = [&](const HcWeights& h) {
    fn(h.norm);
    fn(h.down);
    fn(h.up);
    fn(h.inject);
  };
  const auto store = [&](const ExpertClassStorage& s) {
    fn(s.bf16_w);
    fn(s.wq);
    fn(s.wsz);
    fn(s.trellis_w);
    fn(s.trellis_suh);
    fn(s.trellis_svh);
  };
  for (const qwen4exp::LayerWeights& lw : layers_) {
    hc(lw.hc_attn);
    hc(lw.hc_mlp);
    if (lw.attn) {
      const QsaWeights& a = *lw.attn;
      linear(a.qg);
      linear(a.k);
      linear(a.v);
      linear(a.o);
      fn(a.q_norm);
      fn(a.k_norm);
      fn(a.k_descale);
      fn(a.v_descale);
      fn(a.indexer_qk);
      fn(a.indexer_q_norm);
      fn(a.indexer_k_norm);
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
    if (lw.ple) {
      const PleWeights& p = *lw.ple;
      fn(p.key_proj);
      fn(p.value_proj);
      fn(p.norm_key);
      fn(p.norm_query);
      fn(p.norm_conv);
      fn(p.conv1d);
    }
  }
  if (has_head_) {
    hc(hc_final_);
    linear(lm_head_);
  }
  fn(trellis_tickets_);
}

int64_t Qwen4ExpContainer::DeviceBytes() const {
  int64_t total = 0;
  ForEachBuffer([&](const auto& b) {
    if (b.owned()) total += static_cast<int64_t>(b.bytes());
  });
  return total;
}

const qwen4exp::LayerWeights& Qwen4ExpContainer::Layer(int64_t i) const {
  if (i < first_layer_ || i >= EndLayer()) {
    throw std::out_of_range(kWho + "layer " + std::to_string(i) + " is outside the loaded range [" +
                            std::to_string(first_layer_) + ", " + std::to_string(EndLayer()) + ")");
  }
  return layers_[static_cast<size_t>(i - first_layer_)];
}

const uint16_t* Qwen4ExpContainer::EmbedTokensHost() const {
  if (!embed_host_) throw std::logic_error(kWho + "text.embed_tokens was not loaded (load_embed = false)");
  return embed_host_->data();
}

const qwen4exp::HcWeights& Qwen4ExpContainer::HcFinal() const {
  if (!has_head_) throw std::logic_error(kWho + "the head was not loaded (load_head = false)");
  return hc_final_;
}

const QuantLinear& Qwen4ExpContainer::LmHead() const {
  if (!has_head_) throw std::logic_error(kWho + "the head was not loaded (load_head = false)");
  return lm_head_;
}

}  // namespace r4dx::model
