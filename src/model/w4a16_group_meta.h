// r4dx::model per-tensor w4a16 groups -- the loader half of `__metadata__.quant.w4a16.groups`
// (docs/quant2.md sections 5 and 5.1; docs/container-format.md, w4a16, "Per-tensor groups"). The
// converter half is src/convert/include/r4dx_convert/w4a16_groups.hpp (`--w4a16-group-rule`).
//
// HIP-free and header-only on purpose, like rotation_meta.h: tests/model/test_w4a16_group_meta.cpp
// checks the parse, every refusal, the scale-tensor names and the tensor-parallel slices without a
// device, and Container::Load / Container::LoadShard (src/model/container.cpp) are the only
// production callers. Nothing here touches tensor bytes or asks the kernel anything: whether this
// binary's r4d_gemm_w4a16_nt_m64_g can run a mapped group is CheckW4a16MappedGroup's job
// (quant_linear.h), called by the loader only when the load actually selects w4a16.
//
// The contract, in the loader's words:
//   - `quant.w4a16.group` is the container's DEFAULT group (128 when the `quant` block predates the
//     recorded group, CheckQuantGroups' historical rule). Every w4a16 linear NOT named in the map
//     is at that group and keeps the historical scale tensor `<base>.w4a16.wsz`.
//   - `quant.w4a16.groups` (optional) = {"<container base>": g, ...} names ONLY the linears whose
//     group differs from the default; g is 32, 64 or 128. Such a linear's scales are
//     `<base>.w4a16.wsz.g<g>` and its `.w4a16.wq` is unchanged.
//   - No key, or an empty object: the container is exactly what it was before per-tensor groups, and
//     every call site below degrades to the pre-Q3 name and group.
#pragma once

#include <map>
#include <stdexcept>
#include <string>

#include "nlohmann/json.hpp"

namespace r4dx::model {

// The groups a map entry may name: the ones r4d_gemm_w4a16_nt_m64_g instantiates (r4d.h). The
// kernel's own has_group() is still asked at load time (CheckW4a16MappedGroup); this is the
// format's rule, checked on every load whether or not it reads w4a16.
inline bool IsW4a16MapGroup(int g) { return g == 32 || g == 64 || g == 128; }

// `<base>.w4a16.wsz` at the container's default group, `<base>.w4a16.wsz.g<group>` at any other --
// byte for byte the converter's W4a16WszName (linear_layouts.hpp), with the container's recorded
// default in place of the converter's build constant.
inline std::string W4a16WszName(const std::string& base, int group, int default_group) {
  return group == default_group ? base + ".w4a16.wsz"
                                : base + ".w4a16.wsz.g" + std::to_string(group);
}

struct W4a16Groups {
  int default_group = 128;        // quant.w4a16.group; 128 for a container without it
  bool default_recorded = false;  // quant.w4a16.group was present
  // quant.w4a16.groups: base -> group, never containing the default (ParseW4a16Groups refuses it).
  std::map<std::string, int> mapped;

  bool Mapped(const std::string& base) const { return mapped.count(base) != 0; }
  int GroupFor(const std::string& base) const {
    const auto it = mapped.find(base);
    return it == mapped.end() ? default_group : it->second;
  }
  std::string WszName(const std::string& base) const {
    return W4a16WszName(base, GroupFor(base), default_group);
  }
  // QuantLinear::w4a16_group for a linear loaded from this container: its own group when mapped, 0
  // ("this build's default") otherwise -- the loader has already checked that the container's
  // default IS this build's default (CheckW4a16Group) before an unmapped w4a16 linear is read.
  int QuantLinearGroup(const std::string& base) const {
    const auto it = mapped.find(base);
    return it == mapped.end() ? 0 : it->second;
  }
};

// `metadata`: the container's whole `__metadata__`. Never consults the kernel and never throws for
// a container without a map, so an existing container parses to {group, {}} exactly as the old
// inline read did. Throws std::runtime_error naming `path` when:
//   - `groups` is present but not a JSON object, or present without `group` (a map only means
//     something against the default it was written next to);
//   - an entry's key is empty, or its value is not an integer, not 32/64/128, or equal to the
//     default (the converter lists only the linears that DIFFER; a default-valued entry would
//     make the wsz name ambiguous -- W4a16WszName would say bare, the converter wrote bare too,
//     but a hand-edited map claiming it is a converter bug worth refusing).
inline W4a16Groups ParseW4a16Groups(const nlohmann::json& metadata, const std::string& path) {
  W4a16Groups g;
  if (!metadata.contains("quant")) return g;
  const nlohmann::json& quant = metadata.at("quant");
  if (!quant.is_object() || !quant.contains("w4a16")) return g;
  const nlohmann::json& w = quant.at("w4a16");
  if (!w.is_object()) return g;
  if (w.contains("group")) {
    g.default_group = w.at("group").get<int>();  // the historical read (CheckQuantGroups, LoadShard)
    g.default_recorded = true;
  }
  if (!w.contains("groups")) return g;
  const auto fail = [&](const std::string& why) {
    throw std::runtime_error("r4dx::model::Container: " + path + " __metadata__.quant.w4a16.groups " +
                             why);
  };
  const nlohmann::json& map = w.at("groups");
  if (!map.is_object()) fail("is not an object of {\"<container base>\": group}");
  if (!map.empty() && !g.default_recorded) {
    fail("is present without quant.w4a16.group, the default group it is relative to");
  }
  for (auto it = map.begin(); it != map.end(); ++it) {
    const std::string& base = it.key();
    if (base.empty()) fail("has an empty base name");
    if (!it.value().is_number_integer()) fail("entry '" + base + "' is not an integer group");
    const int group = it.value().get<int>();
    if (!IsW4a16MapGroup(group)) {
      fail("entry '" + base + "' = " + std::to_string(group) +
           " is not a per-tensor group this format has (32, 64, 128)");
    }
    if (group == g.default_group) {
      fail("entry '" + base + "' = " + std::to_string(group) +
           " equals the default group; the map lists only linears whose group differs");
    }
    g.mapped.emplace(base, group);
  }
  return g;
}

}  // namespace r4dx::model
