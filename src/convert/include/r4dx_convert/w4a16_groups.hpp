// r4dx_convert::W4a16GroupRules -- r4dx-convert's `--w4a16-group-rule "<regex>=<g>"` (repeatable),
// the converter half of docs/quant2.md section 5 (Q3, per-tensor w4a16 group).
//
// Every linear the run writes in w4a16 gets the build default group (kW4A16Group, which is what
// __metadata__.quant.w4a16.group records) unless a rule selects another one for its CONTAINER BASE
// NAME. Matching is std::regex_search, ECMAScript, over the base name -- exactly like --keep-bf16 and
// --ldlq, so "mlp\.down$" selects every layer's down projection and
// "^text\.layers\.([0-9]|[12][0-9]|3[01])\.mlp\.down$" the first 32 layers'. Rules are tried in the
// order given and the FIRST match wins, so a narrow exception goes before a broad rule:
//   --w4a16-group-rule "^text\.layers\.63\.mlp\.down$=64" --w4a16-group-rule "mlp\.down$=32"
// keeps layer 63 at 64 (a rule may name the default group for exactly this) and moves every other
// down projection to 32. An unmatched linear keeps the default.
//
// What a non-default group changes on disk (docs/container-format.md, w4a16, "Per-tensor groups"):
//   - `<base>.w4a16.wsz` is written as `<base>.w4a16.wsz.g<g>` (W4a16WszName), N*K/g dwords;
//   - `<base>` is listed in __metadata__.quant.w4a16.groups = {"<base>": g, ...}, which names ONLY
//     the linears whose group differs from the default (an empty map is not written at all);
//   - `.w4a16.wq` is unchanged (its 64-K packed blocks do not depend on the group).
// A container converted without the flag is byte-identical to one converted before it existed.
//
// Failure modes, all deliberate:
//   - a malformed rule, an invalid regex or a group other than 32/64 is an argument error before
//     any shard is read;
//   - a linear whose K is not a multiple of its group and of 64 (the packed block), or whose N is
//     not a multiple of 16, is a PLANNING error naming the rule -- before the header is written;
//   - a non-default group on a linear that ALSO keeps a `.bf16.w` is a planning error. A binary that
//     predates per-tensor groups resolves a w4a16 load through LoadQuantLinearWithFallback
//     (src/model/container.cpp): requested layout, then `.bf16.w`, then the bare name. Missing the
//     bare `.w4a16.wsz`, it would quietly load the bf16 companion -- right numbers, wrong layout,
//     twice the VRAM, and nothing in the log but a fallback count -- instead of refusing the
//     container. Without the bf16 companion it throws "no tensor found", which is the contract.
//     So a grouped conversion must be `--no-bf16` (the v6 recipe) with an `--lm-head` spec without
//     bf16, or keep that linear out of the rules;
//   - a rule that decides no w4a16 linear warns (stderr) and converts normally, like --keep-bf16:
//     a sweep scripted over a candidate list must not die on a class this checkpoint lacks;
//   - with --dflash-gguf the flag is an argument error (the drafter loader reads one group).
// --keep-bf16 wins over a rule (a kept linear has no w4a16 layout); --ldlq composes (LDLQ rounds at
// the linear's own group -- 32 and 64 both divide its 128-column block).
#pragma once

#include <cctype>
#include <cstdint>
#include <map>
#include <ostream>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"
#include "r4dx_convert/linear_layouts.hpp"
#include "r4dx_convert/quant_int4.hpp"

namespace r4dx_convert {

struct W4a16GroupRule {
  std::string spec;     // as given on the command line, "<regex>=<g>"
  std::string pattern;  // the regex half
  int group = 0;
  std::regex re;
};

// "<regex>=<g>": split at the LAST '=', so a regex may itself contain '=' (the group never does).
inline W4a16GroupRule ParseW4a16GroupRule(const std::string& spec) {
  const size_t eq = spec.rfind('=');
  const std::string usage = "--w4a16-group-rule: expected \"<ECMAScript regex>=<group>\" with group "
                            "32 or 64, got '" + spec + "'";
  if (eq == std::string::npos || eq == 0 || eq + 1 >= spec.size()) throw std::runtime_error(usage);
  W4a16GroupRule r;
  r.spec = spec;
  r.pattern = spec.substr(0, eq);
  const std::string g = spec.substr(eq + 1);
  for (char c : g) {
    if (!std::isdigit(static_cast<unsigned char>(c))) throw std::runtime_error(usage);
  }
  if (g.size() > 4) throw std::runtime_error(usage);
  r.group = std::stoi(g);
  if (!IsW4A16GroupSupported(r.group)) {
    throw std::runtime_error("--w4a16-group-rule '" + spec + "': group " + g +
                             " is not one r4d_gemm_w4a16_nt_m64_g instantiates (32, 64)");
  }
  try {
    r.re = std::regex(r.pattern, std::regex::ECMAScript);
  } catch (const std::regex_error& e) {
    throw std::runtime_error("--w4a16-group-rule: invalid ECMAScript regex '" + r.pattern +
                             "': " + e.what());
  }
  return r;
}

class W4a16GroupRules {
 public:
  // `specs` empty => disabled: Apply() returns its argument unchanged, nothing is logged or recorded.
  explicit W4a16GroupRules(const std::vector<std::string>& specs) {
    for (const auto& s : specs) rules_.push_back(ParseW4a16GroupRule(s));
    decided_.assign(rules_.size(), 0);
  }

  bool Enabled() const { return !rules_.empty(); }
  const std::vector<W4a16GroupRule>& Rules() const { return rules_; }

  // Index of the first rule whose regex matches `container_base`, or -1.
  int MatchIndex(const std::string& container_base) const {
    for (size_t i = 0; i < rules_.size(); ++i) {
      if (std::regex_search(container_base, rules_[i].re)) return static_cast<int>(i);
    }
    return -1;
  }

  // The group `container_base` gets: the first matching rule's, else the build default.
  int GroupFor(const std::string& container_base) const {
    const int i = MatchIndex(container_base);
    return i < 0 ? kW4A16Group : rules_[static_cast<size_t>(i)].group;
  }

  // `requested` with its w4a16 group resolved for `container_base`. Pure, so the caller resolves it
  // ONCE (add_linear) and hands the same LayoutSet to --keep-bf16's accounting, --ldlq's plan check,
  // PlanLinearLayouts and EmitLinearLayouts -- the plan and emit passes cannot disagree.
  LayoutSet Apply(const std::string& container_base, LayoutSet requested) const {
    if (requested.w4a16) requested.w4a16_group = GroupFor(container_base);
    return requested;
  }

  // Planning pass, once per linear, with the LayoutSet it is actually written in (after
  // --keep-bf16). Validates the shape against the group, applies the bf16-companion guard (header
  // comment) and records the base when its group is not the default. Throws on any problem.
  void Plan(const std::string& container_base, int64_t N, int64_t K, const LayoutSet& ls) {
    const int i = MatchIndex(container_base);
    if (i < 0) return;
    if (!ls.w4a16) {  // --keep-bf16 won, or this linear has no w4a16 layout in this run
      ++skipped_;
      return;
    }
    ++decided_[static_cast<size_t>(i)];
    const W4a16GroupRule& r = rules_[static_cast<size_t>(i)];
    const int g = ls.w4a16_group;
    if (g != r.group) {
      throw std::logic_error("W4a16GroupRules::Plan: '" + container_base + "' carries group " +
                             std::to_string(g) + " but rule '" + r.spec + "' says " +
                             std::to_string(r.group) + " (LayoutSet not resolved by Apply)");
    }
    if (g == kW4A16Group) return;
    auto fail = [&](const std::string& why) {
      throw std::runtime_error("--w4a16-group-rule '" + r.spec + "' selects '" + container_base +
                               "' [" + std::to_string(N) + "," + std::to_string(K) + "]: " + why);
    };
    if (K % g != 0) fail("K is not a multiple of the group " + std::to_string(g));
    if (K % 64 != 0) fail("K is not a multiple of the kernel's 64-K packed block");
    if (N % 16 != 0) fail("N is not a multiple of 16");
    if (ls.bf16) {
      fail("it also keeps a .bf16.w, which a binary that predates per-tensor w4a16 groups would "
           "silently load instead of refusing the container -- convert with --no-bf16 (and an "
           "--lm-head spec without bf16), or narrow the rule");
    }
    const uint64_t nk = static_cast<uint64_t>(N) * static_cast<uint64_t>(K);
    extra_bytes_ += static_cast<int64_t>(nk / static_cast<uint64_t>(g) * 4) -
                    static_cast<int64_t>(nk / static_cast<uint64_t>(kW4A16Group) * 4);
    groups_[container_base] = g;
  }

  // The linears written at a non-default group, base -> group: __metadata__.quant.w4a16.groups.
  const std::map<std::string, int>& Groups() const { return groups_; }
  nlohmann::json GroupsJson() const {
    nlohmann::json j = nlohmann::json::object();
    for (const auto& kv : groups_) j[kv.first] = kv.second;
    return j;
  }
  nlohmann::json RulesJson() const {
    nlohmann::json j = nlohmann::json::array();
    for (const auto& r : rules_) j.push_back(r.spec);
    return j;
  }
  // Signed .w4a16.wsz byte delta of the whole selection against the default group (positive = the
  // container grew). .w4a16.wq does not depend on the group, so this is the whole w4a16 delta.
  int64_t ExtraBytes() const { return extra_bytes_; }

  // One line per rule after the planning pass, before the long emit pass. A rule that decided no
  // w4a16 linear -- matched nothing, or only bases an earlier rule already took -- is a WARNING on
  // stderr, not an error (header comment).
  void Report(std::ostream& log, std::ostream& warn) const {
    if (!Enabled()) return;
    for (size_t i = 0; i < rules_.size(); ++i) {
      if (decided_[i] == 0) {
        warn << "[r4dx-convert] WARNING: --w4a16-group-rule '" << rules_[i].spec
             << "' decided no w4a16 linear (matched none, or only bases an earlier rule took)\n";
      } else {
        log << "[r4dx-convert] w4a16-group-rule '" << rules_[i].spec << "': " << decided_[i]
            << " linear(s) at group " << rules_[i].group << "\n";
      }
    }
    std::map<int, int> per_group;
    for (const auto& kv : groups_) ++per_group[kv.second];
    log << "[r4dx-convert] w4a16 groups: default " << kW4A16Group << ", " << groups_.size()
        << " linear(s) at another group";
    for (const auto& kv : per_group) log << " (g" << kv.first << " x" << kv.second << ")";
    log << ", .w4a16.wsz " << (extra_bytes_ >= 0 ? "+" : "") << extra_bytes_ << " B ("
        << static_cast<double>(extra_bytes_) / (1024.0 * 1024.0 * 1024.0) << " GiB) vs default";
    if (skipped_ > 0) log << "; " << skipped_ << " other match(es) have no w4a16 layout";
    log << "\n";
  }

 private:
  std::vector<W4a16GroupRule> rules_;
  std::vector<int64_t> decided_;  // per rule: w4a16 linears it was the first match for
  std::map<std::string, int> groups_;
  int64_t skipped_ = 0;
  int64_t extra_bytes_ = 0;
};

}  // namespace r4dx_convert
