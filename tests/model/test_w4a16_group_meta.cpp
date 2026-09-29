// tests/model/test_w4a16_group_meta.cpp -- CPU-only (no GPU, no container). docs/quant2.md
// section 5.1 (Q3, per-tensor w4a16 groups).
//
// The host half of the runtime's per-tensor group contract, src/model/w4a16_group_meta.h, exactly
// as Container::Load and the tensor-parallel ShardLoader use it:
//   1. No map (every container converted without --w4a16-group-rule, and every container predating
//      the `quant` block): the default group as the old inline read found it, every scale tensor
//      under its historical bare name, every QuantLinear at group 0 (the historical kernel entry).
//   2. A map: each mapped base resolves to its own group and to `<base>.w4a16.wsz.g<g>`; the
//      unmapped ones keep the default and the bare name.
//   3. Every refusal: a map that is not an object, a map without its default, an empty key, a
//      non-integer / unsupported / default-valued group.
//   4. Writer/reader agreement: r4dx-convert's W4a16GroupRules (src/convert w4a16_groups.hpp) resolve,
//      plan and serialize a map that this parser reads back entry for entry, and the converter's
//      W4a16WszName and the loader's name the same tensor at every group.
//   5. TP=2: for every real linear, each group's wsz slice (the PartShape ShardLoader::Linear builds
//      from GroupFor) is legal on both ranks, exactly rank_N * (rank_K / g) dwords, and each rank's
//      K is a legal kernel K at that group under FallbackTuning's SK=4 -- the tuning a non-default
//      group gets (linear.cpp). The byte-exactness of those slices against the converter's packers
//      is test_tp_shard's (its variants include g32).
#include <algorithm>
#include <cstdio>
#include <functional>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

#include "model_config.h"
#include "nlohmann/json.hpp"
#include "r4dx_convert/linear_layouts.hpp"
#include "r4dx_convert/quant_int4.hpp"
#include "r4dx_convert/w4a16_groups.hpp"
#include "tp/tp_shard.h"
#include "w4a16_group_meta.h"

using namespace r4dx::model;

namespace {

int g_failures = 0;
int g_checks = 0;

void Check(bool cond, const std::string& what) {
  ++g_checks;
  if (!cond) {
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++g_failures;
  }
}

// The real v6 text config (test_tp_shard.cpp's RealConfig).
ModelConfig RealConfig() {
  ModelConfig c;
  c.hidden_size = 5120;
  c.num_hidden_layers = 64;
  for (int i = 0; i < 64; ++i) {
    c.layer_types.push_back(i % 4 == 3 ? "full_attention" : "linear_attention");
  }
  c.num_attention_heads = 24;
  c.num_key_value_heads = 4;
  c.head_dim = 256;
  c.intermediate_size = 17408;
  c.linear_num_key_heads = 16;
  c.linear_num_value_heads = 48;
  c.linear_key_head_dim = 128;
  c.linear_value_head_dim = 128;
  c.vocab_size = 248320;
  return c;
}

bool Refuses(const char* text) {
  try {
    ParseW4a16Groups(nlohmann::json::parse(text), "fixture.r4dx");
  } catch (const std::runtime_error&) {
    return true;
  }
  return false;
}

// ---- 1. no map ----------------------------------------------------------------------------------

void TestNoMap() {
  // A container predating the `quant` block: group 128 by construction, not recorded.
  {
    const W4a16Groups g = ParseW4a16Groups(nlohmann::json::parse(R"({"model_id": "x"})"), "p");
    Check(g.default_group == 128 && !g.default_recorded && g.mapped.empty(),
          "no quant block: default 128, not recorded, no map");
    Check(g.WszName("text.layers.0.mlp.down") == "text.layers.0.mlp.down.w4a16.wsz",
          "no quant block: the bare wsz name");
    Check(g.QuantLinearGroup("text.layers.0.mlp.down") == 0 && !g.Mapped("lm_head"),
          "no quant block: every QuantLinear at group 0 (the build default)");
  }
  // What src/convert/main.cpp's BuildQuantMetadata writes without --w4a16-group-rule (text, so the
  // JSON number types are what a container actually carries).
  for (int d : {64, 128}) {
    const std::string text = R"({"quant": {"w4a16": {"group": )" + std::to_string(d) +
                             R"(, "kind": "asym"}}})";
    const W4a16Groups g = ParseW4a16Groups(nlohmann::json::parse(text), "p");
    Check(g.default_group == d && g.default_recorded && g.mapped.empty(),
          "converter quant block, default " + std::to_string(d) + ": parsed, no map");
    Check(g.GroupFor("lm_head") == d && g.WszName("lm_head") == "lm_head.w4a16.wsz" &&
              g.QuantLinearGroup("lm_head") == 0,
          "default " + std::to_string(d) + ": lm_head at the default, bare name, group 0");
  }
  // An empty map is the same as none (the converter never writes one, but it means nothing).
  {
    const W4a16Groups g = ParseW4a16Groups(
        nlohmann::json::parse(R"({"quant": {"w4a16": {"group": 64, "groups": {}}}})"), "p");
    Check(g.default_group == 64 && g.mapped.empty(), "an empty groups object: no map");
  }
  // `quant` without w4a16, and a w4a16 block without a group: the old read's defaults.
  {
    const W4a16Groups a =
        ParseW4a16Groups(nlohmann::json::parse(R"({"quant": {"other": {"group": 128}}})"), "p");
    const W4a16Groups b =
        ParseW4a16Groups(nlohmann::json::parse(R"({"quant": {"w4a16": {"kind": "asym"}}})"), "p");
    Check(a.default_group == 128 && !a.default_recorded && b.default_group == 128 &&
              !b.default_recorded,
          "quant without a w4a16 group: 128, not recorded");
  }
}

// ---- 2. a map -----------------------------------------------------------------------------------

void TestMap() {
  const char* text = R"({"quant": {"w4a16": {"group": 64, "groups": {
      "text.layers.0.mlp.down": 32, "text.layers.3.attn.o": 32, "lm_head": 32,
      "mtp.draft_head.lm_head": 32}}}})";
  const W4a16Groups g = ParseW4a16Groups(nlohmann::json::parse(text), "p");
  Check(g.default_group == 64 && g.mapped.size() == 4, "map: default 64, four entries");
  Check(g.GroupFor("text.layers.0.mlp.down") == 32 &&
            g.WszName("text.layers.0.mlp.down") == "text.layers.0.mlp.down.w4a16.wsz.g32" &&
            g.QuantLinearGroup("text.layers.0.mlp.down") == 32,
        "map: mlp.down at 32, .w4a16.wsz.g32, QuantLinear group 32");
  Check(g.GroupFor("lm_head") == 32 && g.WszName("lm_head") == "lm_head.w4a16.wsz.g32" &&
            g.QuantLinearGroup("lm_head") == 32,
        "map: lm_head at 32, .w4a16.wsz.g32");
  Check(g.WszName("mtp.draft_head.lm_head") == "mtp.draft_head.lm_head.w4a16.wsz.g32",
        "map: the MTP draft head is an ordinary base");
  // Unmapped neighbours -- including a base that merely CONTAINS a mapped one -- keep the default.
  for (const char* base : {"text.layers.1.mlp.down", "text.layers.0.mlp.gate_up",
                           "text.layers.10.mlp.down", "mtp.mlp.down"}) {
    Check(!g.Mapped(base) && g.GroupFor(base) == 64 &&
              g.WszName(base) == std::string(base) + ".w4a16.wsz" && g.QuantLinearGroup(base) == 0,
          std::string("map: unmapped '") + base + "' at the default, bare name, group 0");
  }
  // The name function on its own: bare exactly at the container's default.
  Check(W4a16WszName("b", 64, 64) == "b.w4a16.wsz" && W4a16WszName("b", 128, 64) ==
                                                           "b.w4a16.wsz.g128" &&
            W4a16WszName("b", 64, 128) == "b.w4a16.wsz.g64" && W4a16WszName("b", 128, 128) ==
                                                                   "b.w4a16.wsz",
        "W4a16WszName: bare at the default, .g<g> otherwise, for both build defaults");
}

// ---- 3. refusals --------------------------------------------------------------------------------

void TestRefusals() {
  struct Case {
    const char* what;
    const char* text;
  };
  const Case cases[] = {
      {"groups is an array", R"({"quant": {"w4a16": {"group": 64, "groups": [32]}}})"},
      {"groups is a number", R"({"quant": {"w4a16": {"group": 64, "groups": 32}}})"},
      {"groups is a string", R"({"quant": {"w4a16": {"group": 64, "groups": "a=32"}}})"},
      {"groups without the default", R"({"quant": {"w4a16": {"groups": {"lm_head": 32}}}})"},
      {"an empty base", R"({"quant": {"w4a16": {"group": 64, "groups": {"": 32}}}})"},
      {"a string group", R"({"quant": {"w4a16": {"group": 64, "groups": {"lm_head": "32"}}}})"},
      {"a float group", R"({"quant": {"w4a16": {"group": 64, "groups": {"lm_head": 32.0}}}})"},
      {"a null group", R"({"quant": {"w4a16": {"group": 64, "groups": {"lm_head": null}}}})"},
      {"group 16", R"({"quant": {"w4a16": {"group": 64, "groups": {"lm_head": 16}}}})"},
      {"group 48", R"({"quant": {"w4a16": {"group": 64, "groups": {"lm_head": 48}}}})"},
      {"group 96", R"({"quant": {"w4a16": {"group": 64, "groups": {"lm_head": 96}}}})"},
      {"group 128", R"({"quant": {"w4a16": {"group": 64, "groups": {"lm_head": 128}}}})"},
      {"group 256", R"({"quant": {"w4a16": {"group": 64, "groups": {"lm_head": 256}}}})"},
      {"group 0", R"({"quant": {"w4a16": {"group": 64, "groups": {"lm_head": 0}}}})"},
      {"group -32", R"({"quant": {"w4a16": {"group": 64, "groups": {"lm_head": -32}}}})"},
      {"an entry equal to the default (64)",
       R"({"quant": {"w4a16": {"group": 64, "groups": {"lm_head": 64}}}})"},
      {"an entry equal to the default (128)",
       R"({"quant": {"w4a16": {"group": 128, "groups": {"a": 32, "lm_head": 128}}}})"},
  };
  int refused = 0;
  for (const Case& c : cases) {
    const bool r = Refuses(c.text);
    Check(r, std::string("refuses ") + c.what);
    refused += r ? 1 : 0;
  }
  std::printf("PASS-count: %d/%zu malformed maps refused\n", refused, std::size(cases));
  // The error names the container and the offending entry.
  try {
    ParseW4a16Groups(nlohmann::json::parse(
                         R"({"quant": {"w4a16": {"group": 64, "groups": {"lm_head": 48}}}})"),
                     "D:/x.r4dx");
    Check(false, "group 48 throws");
  } catch (const std::runtime_error& e) {
    const std::string m = e.what();
    Check(m.find("D:/x.r4dx") != std::string::npos && m.find("lm_head") != std::string::npos &&
              m.find("48") != std::string::npos,
          "the refusal names the path, the base and the group: " + m);
  }
}

// ---- 4. writer/reader agreement -----------------------------------------------------------------

void TestConverterRoundTrip() {
  using r4dx_convert::kW4A16Group;
  using r4dx_convert::LayoutSet;
  for (int g : {32, 64}) {
    Check(r4dx_convert::W4a16WszName("text.layers.7.mlp.down", g) ==
              W4a16WszName("text.layers.7.mlp.down", g, kW4A16Group),
          "converter and loader name the g" + std::to_string(g) + " scale tensor alike");
  }
  // A v6-style run (--no-bf16): a narrow exception before a broad rule, a class rule, lm_head.
  const std::vector<std::string> specs = {
      R"(^text\.layers\.63\.mlp\.down$=)" + std::to_string(kW4A16Group),
      R"(^text\.layers\.([0-9]|[12][0-9]|3[01])\.mlp\.down$=32)",
      R"(mlp\.down$=64)",
      R"(^lm_head$=32)",
  };
  r4dx_convert::W4a16GroupRules rules(specs);
  const ModelConfig cfg = RealConfig();
  const int64_t H = cfg.hidden_size, I = cfg.intermediate_size;
  std::vector<std::string> bases;
  for (int i = 0; i < 64; ++i) bases.push_back("text.layers." + std::to_string(i) + ".mlp.down");
  bases.push_back("lm_head");
  bases.push_back("text.layers.0.mlp.gate_up");
  for (const std::string& b : bases) {
    LayoutSet requested;
    requested.w4a16 = true;
    requested.bf16 = false;
    const LayoutSet ls = rules.Apply(b, requested);
    const bool down = b.find("mlp.down") != std::string::npos;
    const int64_t N = b == "lm_head" ? cfg.vocab_size : (down ? H : 2 * I);
    const int64_t K = down ? I : H;
    rules.Plan(b, N, K, ls);
  }
  // Serialize exactly as BuildQuantMetadata does and read it back through the loader's parser.
  nlohmann::json meta;
  meta["quant"]["w4a16"] = {{"group", kW4A16Group}, {"kind", "asym"}};
  if (!rules.GroupsJson().empty()) meta["quant"]["w4a16"]["groups"] = rules.GroupsJson();
  const W4a16Groups g =
      ParseW4a16Groups(nlohmann::json::parse(meta.dump()), "roundtrip.r4dx");
  Check(g.mapped == rules.Groups(), "the converter's map reads back entry for entry");
  bool all = true;
  for (const std::string& b : bases) all = all && g.GroupFor(b) == rules.GroupFor(b);
  Check(all, "every base resolves to the group the converter packed it at");
  Check(g.GroupFor("text.layers.5.mlp.down") == 32 && g.GroupFor("text.layers.40.mlp.down") == 64 &&
            g.GroupFor("text.layers.63.mlp.down") == kW4A16Group && !g.Mapped("text.layers.63.mlp.down") &&
            g.GroupFor("lm_head") == 32 && !g.Mapped("text.layers.0.mlp.gate_up"),
        "first match wins; a rule naming the default leaves its base out of the map");
  const size_t want_mapped = 32 + 1;  // layers 0-31 at g32, lm_head; layers 32-63 are the default
  Check(g.mapped.size() == want_mapped, "map size " + std::to_string(g.mapped.size()) + " == " +
                                            std::to_string(want_mapped));
}

// ---- 5. TP=2 slices per group -------------------------------------------------------------------

void TestTpSlices() {
  using namespace r4dx::model::tp;
  const ModelConfig cfg = RealConfig();
  struct Linear {
    const char* base;
    int64_t N, K;
  };
  const Linear linears[] = {
      {"text.layers.0.gdn.in_proj_qkv", 10240, 5120}, {"text.layers.0.gdn.in_proj_z", 6144, 5120},
      {"text.layers.0.gdn.out_proj", 5120, 6144},     {"text.layers.0.mlp.gate_up", 34816, 5120},
      {"text.layers.0.mlp.down", 5120, 17408},        {"text.layers.3.attn.qg", 12288, 5120},
      {"text.layers.3.attn.o", 5120, 6144},           {"lm_head", 248320, 5120},
      {"mtp.attn.qg", 12288, 5120},                   {"mtp.attn.o", 5120, 6144},
      {"mtp.mlp.gate_up", 34816, 5120},               {"mtp.mlp.down", 5120, 17408},
  };
  int plans = 0;
  for (int gsel : {32, 64}) {
    // One container mapping every one of these linears to gsel (unless gsel is the default).
    nlohmann::json meta;
    meta["quant"]["w4a16"]["group"] = r4dx_convert::kW4A16Group;
    if (gsel != r4dx_convert::kW4A16Group) {
      for (const Linear& l : linears) meta["quant"]["w4a16"]["groups"][l.base] = gsel;
    }
    const W4a16Groups groups = ParseW4a16Groups(meta, "tp.r4dx");
    for (const Linear& l : linears) {
      const int g = groups.GroupFor(l.base);
      Check(g == gsel, std::string(l.base) + ": resolves to g" + std::to_string(gsel));
      const ShardRule rule = RuleFor(l.base, cfg);
      for (int r = 0; r < 2; ++r) {
        const std::string tag = std::string(l.base) + " g" + std::to_string(g) + " rank " +
                                std::to_string(r) + " (" + groups.WszName(l.base) + ")";
        int64_t rn = l.N, rk = l.K;
        std::vector<ByteRun> runs;
        PartShape shape;
        shape.part = Part::kW4a16Wsz;
        shape.N = l.N;
        shape.K = l.K;
        shape.group = g;
        try {
          if (rule.split == Split::kRows) {
            const std::vector<Range> rows = RankRows(rule, 2, r);
            rn = 0;
            for (const Range& x : rows) rn += x.count;
            runs = PlanRows(shape, rows);
          } else {
            const Range cols = RankCols(rule, 2, r);
            rk = cols.count;
            runs = PlanCols(shape, cols);
            // wq's own slice must plan too (64-K blocks): the two parts cut the same K range.
            PartShape wq = shape;
            wq.part = Part::kW4Wq;
            wq.group = 0;
            (void)PlanCols(wq, cols);
          }
        } catch (const std::exception& e) {
          Check(false, tag + ": plan throws: " + e.what());
          continue;
        }
        size_t total = 0;
        for (const ByteRun& b : runs) total += b.bytes;
        Check(static_cast<int64_t>(total) == rn * (rk / g) * 4,
              tag + ": " + std::to_string(total) + " B == rank_N * (rank_K / g) * 4");
        // The rank's GEMM: K % (SK * max(g, 64)) at FallbackTuning's SK = 4.
        Check(rk % (4 * std::max(g, 64)) == 0,
              tag + ": rank K " + std::to_string(rk) + " is launchable at SK=4");
        ++plans;
      }
    }
  }
  std::printf("PASS-count: %d (linear, group, rank) wsz slices planned at their own group\n",
              plans);
}

}  // namespace

int main() {
  TestNoMap();
  TestMap();
  TestRefusals();
  TestConverterRoundTrip();
  TestTpSlices();
  std::printf("test_w4a16_group_meta: %d/%d checks passed\n", g_checks - g_failures, g_checks);
  return g_failures == 0 ? 0 : 1;
}
