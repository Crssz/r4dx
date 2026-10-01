// container_load_util.cpp -- the helpers declared in container_load_util.h, moved verbatim from
// container.cpp's anonymous namespace (docs/gemma4-plan.md M1-19). Nothing here changed but its linkage.
#include "container_load_util.h"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <map>

#include "r4dx/core/error.hpp"
#include "r4dx/core/r4d.hpp"

namespace r4dx::model::container_util {
// Reads just the safetensors-shell header (8-byte length + JSON) to pull out `__metadata__`.
// r4dx_convert::SafetensorsReader (reused below for the mmap'd tensor DATA reads) parses the same
// header but discards __metadata__ -- this is the small, deliberate amount of independent
// re-parsing docs/container-format.md's writer/reader contract implies (see container.h's file
// comment), not a second copy of the mmap machinery.
nlohmann::json ReadMetadata(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("r4dx::model::Container: cannot open " + path);
  uint64_t header_len = 0;
  f.read(reinterpret_cast<char*>(&header_len), 8);
  if (!f) throw std::runtime_error("r4dx::model::Container: " + path + " too small for a header");
  std::string header_json(static_cast<size_t>(header_len), '\0');
  f.read(header_json.data(), static_cast<std::streamsize>(header_len));
  if (!f) throw std::runtime_error("r4dx::model::Container: " + path + " header truncated");
  nlohmann::json header = nlohmann::json::parse(header_json);
  if (!header.contains("__metadata__")) {
    throw std::runtime_error("r4dx::model::Container: " + path + " has no __metadata__");
  }
  return header.at("__metadata__");
}

W4a16LoadGroups CheckQuantGroups(const nlohmann::json& metadata, const std::string& path,
                                 Layout layout, Layout lm_head_layout, Layout mtp_head_layout) {
  W4a16LoadGroups w;
  w.groups = ParseW4a16Groups(metadata, path);
  w.path = path;
  if (layout != Layout::kW4a16 && lm_head_layout != Layout::kW4a16 &&
      mtp_head_layout != Layout::kW4a16) {
    return w;
  }
  if (w.groups.mapped.empty()) {
    if (w.groups.default_recorded) CheckW4a16Group(w.groups.default_group, path);
    return w;
  }
  for (const auto& [base, group] : w.groups.mapped) CheckW4a16MappedGroup(group, base, path);
  w.check_default_per_linear = true;
  return w;
}

// docs/trellis-kernel.md 2.5, the refusals that need only the metadata and the requested body
// layout -- run by both loaders right after the parse, before any upload.
void CheckTrellisChoice(const std::optional<TrellisSpec>& trellis, bool rotated, Layout layout,
                        const std::string& path) {
  if (trellis && rotated) {
    throw std::runtime_error("r4dx::model::Container: " + path +
                             " carries both __metadata__.rotation and __metadata__.quant.trellis; "
                             "the two are mutually exclusive (docs/trellis-kernel.md 3.4)");
  }
  if (trellis && layout != Layout::kTrellis) {
    throw std::runtime_error("r4dx::model::Container: " + path + " -- this container's body is "
                             "trellis: run with --layout trellis (requested '" +
                             std::string(LayoutName(layout)) + "')");
  }
  if (!trellis && layout == Layout::kTrellis) {
    throw std::runtime_error("r4dx::model::Container: --layout trellis needs a trellis container "
                             "(__metadata__.quant.trellis), and " + path + " has none");
  }
}

// The trellis tensor suffixes (docs/trellis-kernel.md 2.1).
constexpr const char* kTrellisSuffixes[] = {".trellis.w", ".trellis.suh", ".trellis.svh"};

// docs/trellis-kernel.md 2.5, the refusals that need the tensor directory: every `.trellis.*`
// tensor belongs to a `linears` entry, every entry has its three tensors, and every entry's rate is
// one this build's kernel instantiates. A container without the block may carry no `.trellis.*`
// tensor at all. Checked once per load, before any upload.
void CheckTrellisTensors(const SafetensorsReader& r, const std::optional<TrellisSpec>& trellis,
                         const std::string& path) {
  for (const std::string& name : r.Names()) {
    for (const char* suffix : kTrellisSuffixes) {
      const size_t n = std::strlen(suffix);
      if (name.size() <= n || name.compare(name.size() - n, n, suffix) != 0) continue;
      const std::string base = name.substr(0, name.size() - n);
      if (!trellis) {
        throw std::runtime_error("r4dx::model::Container: " + path + " carries '" + name +
                                 "' but no __metadata__.quant.trellis to read it with");
      }
      if (trellis->Find(base) == nullptr) {
        throw std::runtime_error("r4dx::model::Container: " + path + " carries '" + name +
                                 "' but __metadata__.quant.trellis.linears has no '" + base + "'");
      }
    }
  }
  if (!trellis) return;
  for (const auto& [base, spec] : trellis->linears) {
    for (const char* suffix : kTrellisSuffixes) {
      if (!r.Has(base + suffix)) {
        throw std::runtime_error("r4dx::model::Container: " + path +
                                 " __metadata__.quant.trellis.linears lists '" + base +
                                 "' but the container has no '" + base + suffix + "'");
      }
    }
    CheckTrellisRate(spec.bits, base, path);
  }
}

// One trellis linear's entry, which CheckTrellisTensors has matched with its tensors.
const TrellisLinearSpec& TrellisFor(const LinearLoadMeta& meta, const std::string& base) {
  const TrellisLinearSpec* t = meta.trellis ? meta.trellis->Find(base) : nullptr;
  if (t == nullptr) {
    throw std::logic_error("r4dx::model::Container: '" + base + "' is not a trellis linear of " +
                           meta.path);
  }
  return *t;
}

// docs/trellis-kernel.md 2.5: the GLOBAL shape rules of one trellis linear [N, K] -- K, N and every
// part whole 128-blocks, the parts summing to N -- and its three tensors' byte sizes (N*K*bits/8,
// P*K*2, N*2), checked before a byte of it is uploaded, by both loaders.
void CheckTrellisShape(const SafetensorsReader& r, const std::string& base,
                       const TrellisLinearSpec& t, int64_t N, int64_t K, const std::string& path) {
  const auto fail = [&](const std::string& why) {
    throw std::runtime_error("r4dx::model::Container: trellis linear '" + base + "' [" +
                             std::to_string(N) + ", " + std::to_string(K) + "] at " +
                             std::to_string(t.bits) + " bits in " + path + ": " + why);
  };
  if (N <= 0 || K <= 0 || N % kTrellisBlock != 0 || K % kTrellisBlock != 0) {
    fail("K and N must be multiples of 128");
  }
  if (!t.parts.empty()) {
    int64_t sum = 0;
    for (int64_t p : t.parts) sum += p;
    if (sum != N) fail("its parts sum to " + std::to_string(sum) + " rows, not N");
  }
  const auto span = [&](const char* suffix) {
    const auto& m = r.Meta(base + suffix);
    return static_cast<uint64_t>(m.end - m.begin);
  };
  const uint64_t n = static_cast<uint64_t>(N), k = static_cast<uint64_t>(K);
  const uint64_t want_w = n * k * static_cast<uint64_t>(t.bits) / 8;
  const uint64_t want_suh = static_cast<uint64_t>(t.Parts()) * k * 2;
  if (span(".trellis.w") != want_w) {
    fail("'.trellis.w' is " + std::to_string(span(".trellis.w")) + " bytes, expected " +
         std::to_string(want_w));
  }
  if (span(".trellis.suh") != want_suh) {
    fail("'.trellis.suh' is " + std::to_string(span(".trellis.suh")) + " bytes, expected " +
         std::to_string(want_suh));
  }
  if (span(".trellis.svh") != n * 2) {
    fail("'.trellis.svh' is " + std::to_string(span(".trellis.svh")) + " bytes, expected " +
         std::to_string(n * 2));
  }
}

// The fields of a trellis QuantLinear that do not depend on a rank's slice.
void SetTrellisFields(QuantLinear& q, const TrellisLinearSpec& t) {
  q.trellis_bits = t.bits;
  q.trellis_prescale_log2 = t.prescale_log2;
  q.trellis_parts = t.Parts();
}

void LogTrellis(const TrellisSpec& spec, const std::string& path) {
  std::map<int, int> per_rate;
  int two_part = 0;
  for (const auto& kv : spec.linears) {
    ++per_rate[kv.second.bits];
    if (kv.second.Parts() > 1) ++two_part;
  }
  std::string detail;
  for (const auto& kv : per_rate) {
    detail += (detail.empty() ? "" : ", ") + std::string("KB=") + std::to_string(kv.first) + " x" +
              std::to_string(kv.second);
  }
  std::fprintf(stderr,
               "r4dx: %s is a trellis container: %zu body linears (%s; %d with two input "
               "transforms), prescale 2^%d, heads w4a16/bf16\n",
               path.c_str(), spec.linears.size(), detail.c_str(), two_part, spec.prescale_log2);
}

// A trellis container converted with `--lm-head bf16` (A2's twin, docs/trellis-kernel.md 1) stores
// lm_head as bf16 only, so the w4a16 head a trellis load asks for falls back by design -- not a
// `--keep-bf16` linear or an old container, which the loaders' generic fallback line names. Called
// right after the head's load with the fallback count from before it: such a head is taken out of
// that count (true, and the caller says so on its own line).
bool TakeTrellisBf16Head(const LinearLoadMeta& meta, const QuantLinear& head, int fallbacks_before,
                         int* fallbacks) {
  if (!meta.trellis || *fallbacks == fallbacks_before || head.layout != Layout::kBf16) return false;
  --*fallbacks;
  return true;
}

void LogTrellisBf16Head(const std::string& path) {
  std::fprintf(stderr,
               "r4dx: %s stores lm_head as bf16 only (a trellis container converted with "
               "--lm-head bf16) -- loaded as bf16\n",
               path.c_str());
}

int64_t ElemCountBySize(const SafetensorsReader& r, const std::string& name, int64_t elem_bytes) {
  const auto& m = r.Meta(name);
  const uint64_t span = m.end - m.begin;
  if (span % static_cast<uint64_t>(elem_bytes) != 0) {
    throw std::runtime_error("r4dx::model::Container: tensor '" + name +
                              "' byte span is not a multiple of " + std::to_string(elem_bytes));
  }
  return static_cast<int64_t>(span / static_cast<uint64_t>(elem_bytes));
}

core::DeviceBuffer<uint16_t> UploadRawU16(const SafetensorsReader& r, const std::string& name) {
  const int64_t n = ElemCountBySize(r, name, 2);
  core::DeviceBuffer<uint16_t> buf(static_cast<size_t>(n));
  buf.CopyFromHost(reinterpret_cast<const uint16_t*>(r.Data(name)), static_cast<size_t>(n));
  return buf;
}

core::DeviceBuffer<float> UploadRawF32(const SafetensorsReader& r, const std::string& name) {
  const int64_t n = ElemCountBySize(r, name, 4);
  core::DeviceBuffer<float> buf(static_cast<size_t>(n));
  buf.CopyFromHost(reinterpret_cast<const float*>(r.Data(name)), static_cast<size_t>(n));
  return buf;
}

core::DeviceBuffer<uint8_t> UploadRawU8(const SafetensorsReader& r, const std::string& name) {
  const int64_t n = ElemCountBySize(r, name, 1);
  core::DeviceBuffer<uint8_t> buf(static_cast<size_t>(n));
  buf.CopyFromHost(reinterpret_cast<const uint8_t*>(r.Data(name)), static_cast<size_t>(n));
  return buf;
}

core::DeviceBuffer<uint32_t> UploadRawU32(const SafetensorsReader& r, const std::string& name) {
  const int64_t n = ElemCountBySize(r, name, 4);
  core::DeviceBuffer<uint32_t> buf(static_cast<size_t>(n));
  buf.CopyFromHost(reinterpret_cast<const uint32_t*>(r.Data(name)), static_cast<size_t>(n));
  return buf;
}

core::DeviceBuffer<int32_t> UploadRawI32(const SafetensorsReader& r, const std::string& name) {
  const int64_t n = ElemCountBySize(r, name, 4);
  core::DeviceBuffer<int32_t> buf(static_cast<size_t>(n));
  buf.CopyFromHost(reinterpret_cast<const int32_t*>(r.Data(name)), static_cast<size_t>(n));
  return buf;
}

// gdn.norm_weight is stored bf16 (docs/container-format.md), but both consumers
// (r4d_gdn_gated_rmsnorm_h128_bf16, r4d_gdn_recurrent_update_*) take a float* -- widen on load
// once rather than every layer call.
core::DeviceBuffer<float> UploadWidenedF32(const SafetensorsReader& r, const std::string& name) {
  const int64_t n = ElemCountBySize(r, name, 2);
  const auto* src = reinterpret_cast<const uint16_t*>(r.Data(name));
  std::vector<float> host(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) host[static_cast<size_t>(i)] = core::Bf16ToFloat(src[static_cast<size_t>(i)]);
  core::DeviceBuffer<float> buf(static_cast<size_t>(n));
  buf.CopyFromHost(host);
  return buf;
}

// Trellis suh / svh (docs/trellis-kernel.md 2.1, 5.1): fp16 on disk, fp32 on device (the input
// transform and the GEMM epilogue both multiply in fp32), from `bytes` of fp16 at `src`.
core::DeviceBuffer<float> UploadWidenedF16Bytes(const uint8_t* src, size_t bytes) {
  const size_t n = bytes / 2;
  std::vector<float> host(n);
  for (size_t i = 0; i < n; ++i) {
    uint16_t h;
    std::memcpy(&h, src + 2 * i, 2);
    host[i] = core::F16ToFloat(h);
  }
  core::DeviceBuffer<float> buf(n);
  buf.CopyFromHost(host);
  return buf;
}

core::DeviceBuffer<float> UploadWidenedF16(const SafetensorsReader& r, const std::string& name) {
  const int64_t n = ElemCountBySize(r, name, 2);
  return UploadWidenedF16Bytes(r.Data(name), static_cast<size_t>(n) * 2);
}

// quant2 Q3: the byte sizes a w4a16 linear [N, K] at `group` (the group the kernel will read it at,
// W4a16LoadGroups::KernelGroup) must have on disk -- `.w4a16.wq`
// N*K/2, the scales N*(K/group) dwords -- plus the kernel's shape rules at that group (K % group,
// K % 64 for the packed block, N % 16). The tensor-parallel loader checks every part's size
// against its [N, K] already (ShardLoader::Part); this is the TP=1 path's counterpart, so a scale
// tensor of the wrong group can never be read at the wrong stride.
void CheckW4a16Shape(const SafetensorsReader& r, const std::string& base, const std::string& wsz,
                     int64_t N, int64_t K, int group) {
  const auto fail = [&](const std::string& why) {
    throw std::runtime_error("r4dx::model::Container: w4a16 linear '" + base + "' [" +
                             std::to_string(N) + ", " + std::to_string(K) + "] at group " +
                             std::to_string(group) + ": " + why);
  };
  if (group <= 0 || K % group != 0 || K % 64 != 0 || N % 16 != 0) {
    fail("K must be a multiple of the group and of 64, N of 16");
  }
  const uint64_t wq_bytes = r.Meta(base + ".w4a16.wq").end - r.Meta(base + ".w4a16.wq").begin;
  const uint64_t wsz_bytes = r.Meta(wsz).end - r.Meta(wsz).begin;
  const uint64_t nk = static_cast<uint64_t>(N) * static_cast<uint64_t>(K);
  if (wq_bytes != nk / 2) {
    fail("'" + base + ".w4a16.wq' is " + std::to_string(wq_bytes) + " bytes, expected " +
         std::to_string(nk / 2));
  }
  if (wsz_bytes != nk / static_cast<uint64_t>(group) * 4) {
    fail("'" + wsz + "' is " + std::to_string(wsz_bytes) + " bytes, expected " +
         std::to_string(nk / static_cast<uint64_t>(group) * 4));
  }
}

// quant2 Q3: every base the map names must be a w4a16 linear written at its mapped group -- its
// `.w4a16.wq` and `.w4a16.wsz.g<g>` present, and NO bare `.w4a16.wsz` (which would make the
// container readable at the default stride by a binary that ignores the map). Checked once per
// load, before any upload, whatever layout the load selects: a map naming a tensor the container
// does not have is a converter bug, not a layout choice.
void CheckW4a16GroupTensors(const SafetensorsReader& r, const W4a16Groups& groups,
                            const std::string& path) {
  for (const auto& [base, group] : groups.mapped) {
    const std::string wsz = groups.WszName(base);
    if (!r.Has(base + ".w4a16.wq") || !r.Has(wsz)) {
      throw std::runtime_error("r4dx::model::Container: " + path +
                               " __metadata__.quant.w4a16.groups lists '" + base + "' at group " +
                               std::to_string(group) + " but the container has no '" + base +
                               ".w4a16.wq' + '" + wsz + "'");
    }
    if (r.Has(base + ".w4a16.wsz")) {
      throw std::runtime_error("r4dx::model::Container: " + path + " lists '" + base +
                               "' at w4a16 group " + std::to_string(group) +
                               " but also carries the default-group '" + base + ".w4a16.wsz'");
    }
  }
}

// One stderr line when a load that selects w4a16 meets a per-tensor group map; silent otherwise, so
// a container without a map logs exactly what it always did.
void LogW4a16Groups(const W4a16Groups& groups, const std::string& path) {
  std::map<int, int> per_group;
  for (const auto& kv : groups.mapped) ++per_group[kv.second];
  std::string detail;
  for (const auto& kv : per_group) {
    detail += (detail.empty() ? "" : ", ") + std::string("g") + std::to_string(kv.first) + " x" +
              std::to_string(kv.second);
  }
  std::fprintf(stderr,
               "r4dx: %s packs %zu w4a16 linear(s) at a per-tensor group (%s; default g%d)\n",
               path.c_str(), groups.mapped.size(), detail.c_str(), groups.default_group);
}

QuantLinear LoadQuantLinear(const SafetensorsReader& r, const LinearLoadMeta& meta,
                             const std::string& base, Layout layout, int64_t N, int64_t K) {
  const W4a16LoadGroups& w4a16 = meta.w4a16;
  QuantLinear q;
  q.layout = layout;
  q.N = N;
  q.K = K;
  switch (layout) {
    case Layout::kBf16:
      q.bf16_w = UploadRawU16(r, base + ".bf16.w");
      break;
    case Layout::kW4a16: {
      // quant2 Q3: the linear's own group and scale name (the bare `.w4a16.wsz` unless mapped).
      q.w4a16_group = w4a16.Resolve(base);
      const std::string wsz = w4a16.groups.WszName(base);
      CheckW4a16Shape(r, base, wsz, N, K, w4a16.KernelGroup(base));
      q.wq = UploadRawU8(r, base + ".w4a16.wq");
      q.w4a16_wsz = UploadRawU32(r, wsz);
      break;
    }
    case Layout::kTrellis: {
      // docs/trellis-kernel.md 5.1: the pair-grid words as stored, suh / svh widened to fp32.
      const TrellisLinearSpec& t = TrellisFor(meta, base);
      CheckTrellisShape(r, base, t, N, K, meta.path);
      SetTrellisFields(q, t);
      q.trellis_part_n[0] = t.parts.empty() ? N : t.parts[0];
      q.trellis_part_n[1] = t.parts.empty() ? 0 : t.parts[1];
      q.trellis_w = UploadRawU32(r, base + ".trellis.w");
      q.trellis_suh = UploadWidenedF16(r, base + ".trellis.suh");
      q.trellis_svh = UploadWidenedF16(r, base + ".trellis.svh");
      break;
    }
  }
  return q;
}

// True iff `r` carries every tensor `LoadQuantLinear(r, meta, base, layout, ...)` would read
// (for w4a16, the scale tensor of `base`'s own group: W4a16Groups::WszName; for trellis, a
// `linears` entry as well -- a base without one, a `--keep-bf16` linear, takes the bf16 fallback).
bool HasLayout(const SafetensorsReader& r, const LinearLoadMeta& meta, const std::string& base,
               Layout layout) {
  const W4a16Groups& groups = meta.w4a16.groups;
  switch (layout) {
    case Layout::kBf16: return r.Has(base + ".bf16.w");
    case Layout::kW4a16: return r.Has(base + ".w4a16.wq") && r.Has(groups.WszName(base));
    case Layout::kTrellis:
      return meta.trellis && meta.trellis->Find(base) != nullptr && r.Has(base + ".trellis.w") &&
             r.Has(base + ".trellis.suh") && r.Has(base + ".trellis.svh");
  }
  return false;
}

// R1 (docs/r9700.md): gdn.in_proj_z and attn.k/v join the quantized-linear family (previously
// bf16-only, no `.{layout}` suffix at all). Three tiers, in order:
//   1. `base` carries the requested `layout` -- load it normally (the common case for any
//      container converted with the new converter and --layouts including this layout).
//   2. `base` carries `.bf16.w` but not the requested layout (e.g. --layouts omitted this
//      quantized form, or the requested layout is bf16 itself) -- fall back to bf16 rather than
//      throwing, exactly like docs/container-format.md's other multi-layout linears already do
//      when a caller requests a layout the container didn't bake in.
//   3. `base` is a bare single tensor with no `.{layout}` suffix at all -- the OLD, pre-R1
//      on-disk form these three tensors used to have exclusively (every container converted
//      before this pass). Old containers keep working unmodified (task requirement).
//
// Milestone 11 (docs/validation.md "Milestone 11 / sensitivity"): EVERY quantized body linear now
// loads through this, not just the three R1 tensors. Tier 2 is what makes r4dx-convert's
// `--keep-bf16 <regex>` work -- that flag writes a matched linear as `<base>.bf16.w` and nothing
// else, so the container is quantized everywhere except the tensor class under test and this
// function is the only thing that has to notice. Before, those call sites used LoadQuantLinear
// directly and a bf16-only base was a hard "tensor not found" throw. `fallbacks`, when non-null, is
// incremented once per linear that did NOT have the requested layout, so Load() can report the
// count instead of falling back silently -- a container that accidentally quantized nothing and one
// that deliberately kept one class in bf16 must not look the same in a log.
//
// quant2 Q3 (docs/quant2.md section 5.1): a base listed in __metadata__.quant.w4a16.groups that is
// requested as w4a16 never falls back -- the container promised that linear at its own group, and
// quietly serving a bf16 copy (or anything else) instead would hide a broken container.
// CheckW4a16GroupTensors has already refused a map entry without its tensors, so this is the
// second line, not the first.
QuantLinear LoadQuantLinearWithFallback(const SafetensorsReader& r, const LinearLoadMeta& meta,
                                         const std::string& base, Layout requested, int64_t N,
                                         int64_t K, int* fallbacks) {
  const W4a16LoadGroups& w4a16 = meta.w4a16;
  if (HasLayout(r, meta, base, requested)) {
    return LoadQuantLinear(r, meta, base, requested, N, K);
  }
  if (requested == Layout::kW4a16 && w4a16.groups.Mapped(base)) {
    throw std::runtime_error("r4dx::model::Container: '" + base + "' is listed in " +
                             "__metadata__.quant.w4a16.groups at group " +
                             std::to_string(w4a16.groups.GroupFor(base)) + " but '" +
                             w4a16.groups.WszName(base) + "' is missing; not falling back");
  }
  if (fallbacks != nullptr) ++*fallbacks;
  if (HasLayout(r, meta, base, Layout::kBf16)) {
    return LoadQuantLinear(r, meta, base, Layout::kBf16, N, K);
  }
  if (r.Has(base)) {
    QuantLinear q;
    q.layout = Layout::kBf16;
    q.N = N;
    q.K = K;
    q.bf16_w = UploadRawU16(r, base);
    return q;
  }
  throw std::runtime_error("r4dx::model::Container: no tensor found for '" + base +
                            "' in any known on-disk form (requested layout, bf16, or bare)");
}

void LogRotation(const RotationSpec& spec, const std::string& path) {
  std::fprintf(stderr,
               "r4dx: %s is a quant2 %s rotated container (seed 0x%llx): online Q / Q^T at the "
               "layer-stack boundary%s\n",
               path.c_str(), RotationKindName(spec.kind),
               static_cast<unsigned long long>(spec.seed),
               spec.Hadamard() ? ", Hadamard on the mlp.down / attn.o / gdn.out_proj inputs" : "");
}

}  // namespace r4dx::model::container_util

