#include "awq_source.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>

#include "r4dx/core/dtype.hpp"

namespace r4dx_convert::awq {

namespace {

const std::string kPacked = ".weight_packed";
const std::string kScale = ".weight_scale";
const std::string kZero = ".weight_zero_point";
const std::string kShape = ".weight_shape";
const std::string kWeight = ".weight";

bool EndsWith(const std::string& s, const std::string& suffix) {
  return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

uint32_t Load32(const uint8_t* p) {
  uint32_t v;
  std::memcpy(&v, p, 4);
  return v;
}

uint16_t Load16(const uint8_t* p) {
  uint16_t v;
  std::memcpy(&v, p, 2);
  return v;
}

std::string ShapeStr(const std::vector<int64_t>& s) { return nlohmann::json(s).dump(); }

[[noreturn]] void Bad(const std::string& what) {
  throw std::runtime_error("r4dx-convert (pack-quantized source): " + what);
}

// `j[key]` must be absent or null (the field's default).
void RequireNullField(const nlohmann::json& j, const char* key, const std::string& where) {
  if (j.contains(key) && !j.at(key).is_null())
    Bad(where + "." + key + " is " + j.at(key).dump() + "; only the default (null) is supported");
}

}  // namespace

bool PackQuantizedScheme(const nlohmann::json& config, PackScheme* out) {
  const auto it = config.find("quantization_config");
  if (it == config.end() || !it->is_object()) return false;
  const nlohmann::json& q = *it;
  if (!q.contains("format") || !q.at("format").is_string() || q.at("format").get<std::string>() != "pack-quantized")
    return false;
  const std::string who = "quantization_config";
  if (q.contains("quant_method") && q.at("quant_method") != "compressed-tensors")
    Bad(who + ".quant_method is " + q.at("quant_method").dump() + ", not \"compressed-tensors\"");
  if (!q.contains("config_groups") || !q.at("config_groups").is_object() || q.at("config_groups").size() != 1)
    Bad(who + ".config_groups must hold exactly one group (this converter dequantizes one scheme for every "
              "packed linear)");
  const nlohmann::json& g = *q.at("config_groups").begin();
  if (g.contains("format") && g.at("format") != "pack-quantized")
    Bad(who + ".config_groups.*.format is " + g.at("format").dump() + ", not \"pack-quantized\"");
  RequireNullField(g, "input_activations", who + ".config_groups.*");
  RequireNullField(g, "output_activations", who + ".config_groups.*");
  if (!g.contains("weights") || !g.at("weights").is_object())
    Bad(who + ".config_groups.*.weights is missing");
  const nlohmann::json& w = g.at("weights");
  const std::string ww = who + ".config_groups.*.weights";
  auto want = [&](const char* key, const nlohmann::json& value) {
    if (!w.contains(key) || w.at(key) != value)
      Bad(ww + "." + key + " is " + (w.contains(key) ? w.at(key).dump() : std::string("missing")) +
          ", this converter handles " + value.dump() + " only");
  };
  want("num_bits", 4);
  want("group_size", 128);
  want("strategy", "group");
  want("symmetric", false);
  want("type", "int");
  if (w.contains("dynamic") && w.at("dynamic") != false) Bad(ww + ".dynamic is " + w.at("dynamic").dump());
  RequireNullField(w, "actorder", ww);
  RequireNullField(w, "block_structure", ww);
  if (out) {
    out->num_bits = 4;
    out->group_size = 128;
  }
  return true;
}

void DequantRows(const uint8_t* packed, const uint8_t* scale, const uint8_t* zp, int64_t N, int64_t K,
                 int group, int64_t row0, int64_t nrows, uint16_t* out) {
  if (group <= 0 || group % 8 != 0 || K % group != 0)
    throw std::invalid_argument("awq::DequantRows: K=" + std::to_string(K) + " is not a multiple of group " +
                                std::to_string(group) + " (a multiple of 8)");
  if (row0 < 0 || nrows < 0 || row0 + nrows > N)
    throw std::invalid_argument("awq::DequantRows: rows [" + std::to_string(row0) + ", " +
                                std::to_string(row0 + nrows) + ") outside N=" + std::to_string(N));
  const int64_t groups = K / group;
  const int64_t words_per_group = group / 8;
  for (int64_t r = 0; r < nrows; ++r) {
    const int64_t n = row0 + r;
    const uint8_t* prow = packed + static_cast<uint64_t>(n) * static_cast<uint64_t>(K / 8) * 4;
    const int shift = static_cast<int>(n & 7) * 4;
    uint16_t* o = out + static_cast<uint64_t>(r) * static_cast<uint64_t>(K);
    for (int64_t gi = 0; gi < groups; ++gi) {
      const float s = r4dx::core::Bf16ToFloat(Load16(scale + (static_cast<uint64_t>(n) * groups + gi) * 2));
      const int z = static_cast<int>((Load32(zp + (static_cast<uint64_t>(n >> 3) * groups + gi) * 4) >> shift) & 15u);
      const uint8_t* pw = prow + static_cast<uint64_t>(gi * words_per_group) * 4;
      for (int64_t c = 0; c < words_per_group; ++c) {
        const uint32_t word = Load32(pw + c * 4);
        for (int j = 0; j < 8; ++j) {
          const int q = static_cast<int>((word >> (4 * j)) & 15u);
          *o++ = r4dx::core::FloatToBf16(static_cast<float>(q - z) * s);
        }
      }
    }
  }
}

std::shared_ptr<AwqOverlay> AwqOverlay::Create(ShardedModel& model, const PackScheme& scheme) {
  if (scheme.num_bits != 4 || scheme.group_size != 128)
    Bad("only int4 / group 128 is supported (got " + std::to_string(scheme.num_bits) + " bits, group " +
        std::to_string(scheme.group_size) + ")");
  std::shared_ptr<AwqOverlay> self(new AwqOverlay(model, scheme));
  const int64_t G = scheme.group_size;
  const std::vector<std::string> names = model.AllNames();
  for (const std::string& name : names) {
    if (!EndsWith(name, kPacked)) continue;
    const std::string base = name.substr(0, name.size() - kPacked.size());
    for (const std::string* suffix : {&kScale, &kZero, &kShape})
      if (!model.Has(base + *suffix))
        Bad("'" + name + "' has no companion '" + base + *suffix + "'");
    if (model.Has(base + kWeight)) Bad("'" + base + kWeight + "' exists next to '" + name + "'");

    const TensorMeta& pm = model.RawMeta(base + kPacked);
    const TensorMeta& sm = model.RawMeta(base + kScale);
    const TensorMeta& zm = model.RawMeta(base + kZero);
    const TensorMeta& hm = model.RawMeta(base + kShape);
    auto check = [&](const std::string& tensor, const TensorMeta& m, const char* dtype,
                     const std::vector<int64_t>& shape) {
      if (m.dtype != dtype || m.shape != shape)
        Bad("'" + tensor + "' is " + m.dtype + " " + ShapeStr(m.shape) + ", expected " + dtype + " " +
            ShapeStr(shape));
    };
    if (pm.dtype != "I32" || pm.shape.size() != 2)
      Bad("'" + name + "' is " + pm.dtype + " " + ShapeStr(pm.shape) + ", expected I32 [N, K/8]");
    const int64_t N = pm.shape[0], K = pm.shape[1] * 8;
    if (K % G != 0)
      Bad("'" + name + "': K = " + std::to_string(K) + " is not a multiple of the group size " + std::to_string(G));
    check(base + kScale, sm, "BF16", {N, K / G});
    check(base + kZero, zm, "I32", {(N + 7) / 8, K / G});
    check(base + kShape, hm, "I64", {2});
    int64_t shape[2];
    std::memcpy(shape, model.RawData(base + kShape), sizeof(shape));
    if (shape[0] != N || shape[1] != K)
      Bad("'" + base + kShape + "' says [" + std::to_string(shape[0]) + ", " + std::to_string(shape[1]) +
          "] but weight_packed implies [" + std::to_string(N) + ", " + std::to_string(K) + "]");
    Linear l;
    l.N = N;
    l.K = K;
    l.meta.dtype = "BF16";
    l.meta.shape = {N, K};
    l.meta.begin = 0;
    l.meta.end = static_cast<uint64_t>(N) * static_cast<uint64_t>(K) * 2;
    self->linears_.emplace(base + kWeight, std::move(l));
  }
  return self;
}

bool AwqOverlay::Provides(const std::string& name) const { return linears_.count(name) != 0; }

const AwqOverlay::Linear& AwqOverlay::Find(const std::string& name, std::string* base) const {
  const auto it = linears_.find(name);
  if (it == linears_.end()) throw std::runtime_error("AwqOverlay: no virtual tensor " + name);
  if (base) *base = name.substr(0, name.size() - kWeight.size());
  return it->second;
}

const TensorMeta& AwqOverlay::Meta(const std::string& name) const { return Find(name, nullptr).meta; }

void AwqOverlay::Dequant(const std::string& base, const Linear& l, int64_t row0, int64_t nrows,
                         uint8_t* out) const {
  DequantRows(model_.RawData(base + kPacked), model_.RawData(base + kScale), model_.RawData(base + kZero), l.N,
              l.K, scheme_.group_size, row0, nrows, reinterpret_cast<uint16_t*>(out));
}

const uint8_t* AwqOverlay::Data(const std::string& name) const {
  std::string base;
  const Linear& l = Find(name, &base);
  thread_local std::vector<uint8_t> buffer;
  buffer.resize(static_cast<size_t>(l.meta.end));
  Dequant(base, l, 0, l.N, buffer.data());
  return buffer.data();
}

const uint8_t* AwqOverlay::DataRows(const std::string& name, int64_t row0, int64_t nrows,
                                    std::vector<uint8_t>& scratch) const {
  std::string base;
  const Linear& l = Find(name, &base);
  if (row0 < 0 || nrows < 0 || row0 > l.N || nrows > l.N - row0)
    throw std::invalid_argument("awq::DataRows: '" + name + "' rows [" + std::to_string(row0) + ", +" +
                                std::to_string(nrows) + ") outside N=" + std::to_string(l.N));
  scratch.resize(static_cast<size_t>(nrows) * static_cast<size_t>(l.K) * 2);
  Dequant(base, l, row0, nrows, scratch.data());
  return scratch.data();
}

void AwqOverlay::RewriteNames(std::vector<std::string>& names) const {
  const auto component_of_linear = [&](const std::string& n) {
    for (const std::string* suffix : {&kPacked, &kScale, &kZero, &kShape})
      if (EndsWith(n, *suffix)) return linears_.count(n.substr(0, n.size() - suffix->size()) + kWeight) != 0;
    return false;
  };
  names.erase(std::remove_if(names.begin(), names.end(), component_of_linear), names.end());
  names.reserve(names.size() + linears_.size());
  for (const auto& kv : linears_) names.push_back(kv.first);
}

size_t InstallAwqSource(ShardedModel& model, const nlohmann::json& config) {
  PackScheme scheme;
  if (!PackQuantizedScheme(config, &scheme)) return 0;
  const std::shared_ptr<AwqOverlay> overlay = AwqOverlay::Create(model, scheme);
  model.SetOverlay(overlay);
  return overlay->LinearCount();
}

}  // namespace r4dx_convert::awq
