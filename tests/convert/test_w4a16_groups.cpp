// tests/convert/test_w4a16_groups.cpp -- r4dx-convert's `--w4a16-group-rule "<regex>=<g>"`
// (src/convert/include/r4dx_convert/w4a16_groups.hpp, docs/quant2.md section 5 "Q3"): the converter
// half of per-tensor w4a16 groups. CPU-only, no checkpoint, no GPU. Gates:
//
//   1. PARSING: "<regex>=<g>" split at the last '=', g in {32, 64, 128}; everything else -- no '=',
//      an empty half, a non-numeric or unsupported group, an invalid regex -- throws naming the flag.
//   2. SELECTION: regex_search over the real container base names, FIRST matching rule wins, a rule
//      may name the default group (an exception ahead of a broad rule), unmatched -> the default.
//   3. PLANNING: only non-default groups enter the map; K % g, K % 64 and N % 16 are checked with
//      the rule named; the bf16-companion guard (a binary that predates per-tensor groups would load
//      the `.bf16.w` instead of refusing); --keep-bf16'd linears are skipped; the wsz byte delta.
//   4. ACCOUNTING: LinearLayoutBytes == ContainerWriter::PlannedDataBytes for mixed groups, and
//      --keep-bf16's "would have been" bytes follow the linear's resolved group.
//   5. EMISSION: `<base>.w4a16.wsz.g<g>` at a non-default group, the bare name at the default, with
//      exactly the quantizer+packer bytes at that group; the other layouts untouched; LDLQ at g32 and
//      g128 through EmitLinearLayouts equals QuantizeInt4AsymmetricLdlq at that group.
//   6. THE EXE (when r4dx-convert is built): --selftest without rules reproduces the pre-Q3
//      containers byte for byte (sha256 of the whole file, golden values from the converter as of
//      commit 95cc327 at the default group 64); with rules it writes the renamed tensor, the
//      __metadata__.quant.w4a16.groups map and the r4dx_convert_run record; the guard, a bad group
//      and --dflash-gguf are refused with a nonzero exit.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"
#include "r4dx_convert/container_writer.hpp"
#include "r4dx_convert/keep_bf16.hpp"
#include "r4dx_convert/linear_layouts.hpp"
#include "r4dx_convert/quant_int4.hpp"
#include "r4dx_convert/quant_ldlq.hpp"
#include "r4dx_convert/sha256.hpp"
#include "r4dx_convert/w4a16_groups.hpp"

using namespace r4dx_convert;

namespace {

int g_failures = 0;

bool Check(const std::string& label, bool cond) {
  std::printf("%-72s %s\n", label.c_str(), cond ? "OK" : "FAIL");
  if (!cond) ++g_failures;
  return cond;
}

template <typename Fn>
std::string ThrowMessage(Fn&& fn) {
  try {
    fn();
  } catch (const std::exception& e) {
    return e.what();
  }
  return std::string();
}

bool Has(const std::string& s, const std::string& sub) { return s.find(sub) != std::string::npos; }

LayoutSet Set(bool w4a16, bool bf16, int group = kW4A16Group) {
  LayoutSet ls;
  ls.w4a16 = w4a16;
  ls.bf16 = bf16;
  ls.w4a16_group = group;
  return ls;
}

// The one non-default group every test below can rely on, whatever R4DX_W4A16_GROUP this build is.
constexpr int kOther = kW4A16Group == 32 ? 64 : 32;

std::vector<std::string> RealBaseNames() {
  return {
      "text.layers.0.gdn.in_proj_qkv", "text.layers.0.gdn.in_proj_z", "text.layers.0.gdn.out_proj",
      "text.layers.0.mlp.gate_up",      "text.layers.0.mlp.down",
      "text.layers.3.attn.qg",          "text.layers.3.attn.o",       "text.layers.3.mlp.down",
      "text.layers.31.mlp.down",        "text.layers.32.mlp.down",    "text.layers.63.mlp.down",
      "mtp.mlp.down",                   "mtp.draft_head.lm_head",     "lm_head",
  };
}

std::vector<float> RandomNormal(size_t n, uint32_t seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> d(0.0f, 1.0f);
  std::vector<float> v(n);
  for (auto& x : v) x = d(rng);
  return v;
}

std::string TempPath(const std::string& leaf) {
  const char* t = std::getenv("TEMP");
  return std::string(t ? t : ".") + "\\" + leaf;
}

// A just-written container: header JSON + each tensor's raw bytes (safetensors shell, see
// docs/container-format.md).
struct Container {
  bool ok = false;
  nlohmann::json header;
  std::map<std::string, std::string> tensors;
  std::string file;  // whole file, for the sha256
};

Container ReadContainer(const std::string& path) {
  Container c;
  std::ifstream f(path, std::ios::binary);
  if (!f) return c;
  std::ostringstream ss;
  ss << f.rdbuf();
  c.file = ss.str();
  if (c.file.size() < 8) return c;
  uint64_t n = 0;
  std::memcpy(&n, c.file.data(), 8);
  if (8 + n > c.file.size()) return c;
  c.header = nlohmann::json::parse(c.file.substr(8, static_cast<size_t>(n)));
  const size_t data0 = 8 + static_cast<size_t>(n);
  for (auto it = c.header.begin(); it != c.header.end(); ++it) {
    if (it.key() == "__metadata__") continue;
    const auto off = it.value().at("data_offsets");
    const size_t a = off[0].get<size_t>(), b = off[1].get<size_t>();
    c.tensors[it.key()] = c.file.substr(data0 + a, b - a);
  }
  c.ok = true;
  return c;
}

// sha256 over every tensor (name, size, bytes; in name order), independent of __metadata__: the
// (a) goldens below are digests of the tensors a build from before the mxfp4 / w4a8 layouts were
// cut (bcebb21) wrote with the same arguments.
std::string TensorDigest(const Container& c) {
  std::string all;
  for (const auto& kv : c.tensors) {
    all += kv.first;
    all.push_back('\0');
    all += std::to_string(kv.second.size());
    all.push_back('\0');
    all += kv.second;
  }
  return Sha256Hex(all);
}

std::vector<std::string> Names(const Container& c) {
  std::vector<std::string> v;
  for (const auto& kv : c.tensors) v.push_back(kv.first);
  return v;
}

std::string Bytes(const std::vector<uint32_t>& v) {
  return std::string(reinterpret_cast<const char*>(v.data()), v.size() * 4);
}

// ---- 1. parsing ----------------------------------------------------------------------------------

void TestParsing() {
  std::printf("---- 1. parsing ----\n");
  {
    const W4a16GroupRule r = ParseW4a16GroupRule("mlp\\.down$=32");
    Check("'mlp\\.down$=32' -> pattern 'mlp\\.down$', group 32",
          r.pattern == "mlp\\.down$" && r.group == 32 && r.spec == "mlp\\.down$=32");
  }
  {
    const W4a16GroupRule r = ParseW4a16GroupRule("a=b=128");
    Check("split at the LAST '=': 'a=b=128' -> pattern 'a=b', group 128",
          r.pattern == "a=b" && r.group == 128);
  }
  for (int g : {32, 64, 128}) {
    Check("group " + std::to_string(g) + " accepted",
          ParseW4a16GroupRule("x=" + std::to_string(g)).group == g);
  }
  const char* bad_syntax[] = {"mlp", "=32", "mlp=", "mlp=3x2", "mlp=-32", "mlp= 32"};
  for (const char* s : bad_syntax) {
    const std::string m = ThrowMessage([&] { ParseW4a16GroupRule(s); });
    Check(std::string("malformed '") + s + "' throws naming the flag",
          Has(m, "--w4a16-group-rule") && Has(m, "<group>"));
  }
  for (const char* s : {"mlp=16", "mlp=48", "mlp=96", "mlp=256", "mlp=0"}) {
    const std::string m = ThrowMessage([&] { ParseW4a16GroupRule(s); });
    Check(std::string("unsupported group '") + s + "' throws naming 32, 64, 128",
          Has(m, "--w4a16-group-rule") && Has(m, "32, 64, 128"));
  }
  {
    const std::string m = ThrowMessage([] { ParseW4a16GroupRule("text\\.layers\\.[0-7=32"); });
    Check("invalid regex throws naming the flag and the pattern",
          Has(m, "--w4a16-group-rule") && Has(m, "invalid ECMAScript regex") && Has(m, "[0-7"));
    std::printf("  %s\n", m.c_str());
  }
  {
    // One bad rule poisons the whole list (the CLI parses every rule before converting).
    const std::string m = ThrowMessage([] { W4a16GroupRules({"mlp=32", "attn=48"}); });
    Check("a bad rule anywhere in the list throws", Has(m, "attn=48"));
  }
  {
    W4a16GroupRules off({});
    Check("no rules -> disabled", !off.Enabled());
    const LayoutSet in = Set(true, false);
    const LayoutSet out = off.Apply("text.layers.0.mlp.down", in);
    Check("disabled: Apply leaves the default group", out.w4a16_group == kW4A16Group);
    std::ostringstream log, warn;
    off.Report(log, warn);
    Check("disabled: reports nothing", log.str().empty() && warn.str().empty());
    Check("disabled: empty map, zero bytes", off.GroupsJson().empty() && off.ExtraBytes() == 0);
  }
}

// ---- 2. selection --------------------------------------------------------------------------------

void TestSelection() {
  std::printf("---- 2. selection (first match wins) ----\n");
  const W4a16GroupRules rules({"^text\\.layers\\.63\\.mlp\\.down$=" + std::to_string(kW4A16Group),
                               "^text\\.layers\\.([0-9]|[12][0-9]|3[01])\\.mlp\\.down$=32",
                               "mlp\\.down$=128", "^lm_head$=128"});
  const std::map<std::string, int> want = {
      {"text.layers.0.gdn.in_proj_qkv", kW4A16Group}, {"text.layers.0.gdn.in_proj_z", kW4A16Group},
      {"text.layers.0.gdn.out_proj", kW4A16Group},    {"text.layers.0.mlp.gate_up", kW4A16Group},
      {"text.layers.0.mlp.down", 32},                 {"text.layers.3.attn.qg", kW4A16Group},
      {"text.layers.3.attn.o", kW4A16Group},          {"text.layers.3.mlp.down", 32},
      {"text.layers.31.mlp.down", 32},                {"text.layers.32.mlp.down", 128},
      {"text.layers.63.mlp.down", kW4A16Group},       {"mtp.mlp.down", 128},
      {"mtp.draft_head.lm_head", kW4A16Group},        {"lm_head", 128},
  };
  bool all = true;
  for (const auto& name : RealBaseNames()) {
    const int g = rules.GroupFor(name);
    if (g != want.at(name)) {
      std::printf("  %s: got %d want %d\n", name.c_str(), g, want.at(name));
      all = false;
    }
  }
  Check("depth-half rules + a default-group exception resolve as written", all);
  Check("MatchIndex: layer 63's exception is rule 0, not the broad rule 2",
        rules.MatchIndex("text.layers.63.mlp.down") == 0);
  Check("MatchIndex: unmatched -> -1", rules.MatchIndex("text.layers.3.attn.o") == -1);
  // Order matters: the broad rule first shadows the narrow one.
  const W4a16GroupRules shadowed({"mlp\\.down$=128", "^text\\.layers\\.0\\.mlp\\.down$=32"});
  Check("broad rule first shadows a later narrow rule",
        shadowed.GroupFor("text.layers.0.mlp.down") == 128);
  // Apply touches only a w4a16 LayoutSet.
  const LayoutSet no_w4a16 = rules.Apply("text.layers.0.mlp.down", Set(false, true));
  Check("Apply on a set without w4a16 keeps the default group",
        no_w4a16.w4a16_group == kW4A16Group);
  const LayoutSet with = rules.Apply("text.layers.0.mlp.down", Set(true, false));
  Check("Apply sets the rule's group and nothing else",
        with.w4a16_group == 32 && with.w4a16 && !with.bf16);
}

// ---- 3. planning ---------------------------------------------------------------------------------

void TestPlanning() {
  std::printf("---- 3. planning ----\n");
  W4a16GroupRules rules({"mlp\\.down$=" + std::to_string(kOther), "attn\\.o$=128",
                         "gdn\\.out_proj$=" + std::to_string(kW4A16Group), "nonesuch$=32"});
  const int N = 64, K = 512;
  auto plan = [&](const std::string& base, int64_t n, int64_t k, bool bf16, bool w4a16 = true) {
    const LayoutSet ls = rules.Apply(base, Set(w4a16, bf16));
    rules.Plan(base, n, k, ls);
  };
  plan("text.layers.0.mlp.down", N, K, false);
  plan("text.layers.1.mlp.down", N, K, false);
  plan("text.layers.3.attn.o", N, K, false);
  plan("text.layers.0.gdn.out_proj", N, K, false);  // matched, default group: no map entry
  plan("text.layers.0.gdn.in_proj_z", N, K, true);   // unmatched: bf16 is fine
  plan("text.layers.2.mlp.down", N, K, false, /*w4a16=*/false);  // e.g. --keep-bf16'd: skipped

  std::map<std::string, int> want = {{"text.layers.0.mlp.down", kOther},
                                     {"text.layers.1.mlp.down", kOther}};
  if (kW4A16Group != 128) want["text.layers.3.attn.o"] = 128;
  Check("map holds exactly the non-default linears", rules.Groups() == want);
  const nlohmann::json gj = rules.GroupsJson();
  Check("GroupsJson mirrors the map", gj.size() == want.size() &&
                                          gj.at("text.layers.0.mlp.down").get<int>() == kOther);
  const int64_t nk = static_cast<int64_t>(N) * K;
  int64_t extra = 2 * (nk / kOther * 4 - nk / kW4A16Group * 4);
  if (kW4A16Group != 128) extra += nk / 128 * 4 - nk / kW4A16Group * 4;
  Check("ExtraBytes = sum of per-linear wsz deltas vs the default", rules.ExtraBytes() == extra);
  Check("RulesJson keeps the rules in order", rules.RulesJson().size() == 4 &&
                                                  rules.RulesJson()[1].get<std::string>() ==
                                                      "attn\\.o$=128");
  {
    std::ostringstream log, warn;
    rules.Report(log, warn);
    Check("rule that decided nothing -> WARNING naming it",
          Has(warn.str(), "WARNING") && Has(warn.str(), "nonesuch$=32"));
    Check("summary line counts the skipped (no-w4a16) match", Has(log.str(), "1 other match"));
    std::printf("%s%s", log.str().c_str(), warn.str().c_str());
  }

  // The guard: a non-default group next to a .bf16.w.
  {
    W4a16GroupRules r({"mlp\\.down$=" + std::to_string(kOther)});
    const LayoutSet ls = r.Apply("text.layers.0.mlp.down", Set(true, true));
    const std::string m = ThrowMessage([&] { r.Plan("text.layers.0.mlp.down", N, K, ls); });
    Check("non-default group + bf16 companion is refused at planning",
          Has(m, "--w4a16-group-rule") && Has(m, "text.layers.0.mlp.down") && Has(m, "--no-bf16"));
    std::printf("  %s\n", m.c_str());
    // ...but a rule naming the DEFAULT group writes the historical names, so bf16 is harmless.
    W4a16GroupRules d({"mlp\\.down$=" + std::to_string(kW4A16Group)});
    const LayoutSet lsd = d.Apply("text.layers.0.mlp.down", Set(true, true));
    Check("default-group rule + bf16 companion is accepted",
          ThrowMessage([&] { d.Plan("text.layers.0.mlp.down", N, K, lsd); }).empty());
  }
  // Shape checks name the rule. K = 96 is a multiple of 32 but not of the 64-K packed block.
  {
    W4a16GroupRules r({"x$=32", "y$=128"});
    const std::string m1 = ThrowMessage([&] { r.Plan("t.x", 64, 96, r.Apply("t.x", Set(true, false))); });
    Check("g32 with K % 64 != 0 is refused, naming the rule and the packed block",
          Has(m1, "'x$=32'") && Has(m1, "64-K packed block"));
    const std::string m2 = ThrowMessage([&] { r.Plan("t.y", 64, 192, r.Apply("t.y", Set(true, false))); });
    Check("g128 with K % 128 != 0 is refused", Has(m2, "'y$=128'") && Has(m2, "group 128"));
    const std::string m3 = ThrowMessage([&] { r.Plan("t.x", 40, 256, r.Apply("t.x", Set(true, false))); });
    Check("N % 16 != 0 is refused", Has(m3, "N is not a multiple of 16"));
    // PlanLinearLayouts is the backstop for any caller that bypasses the rules.
    ContainerWriter w;
    const std::string m4 =
        ThrowMessage([&] { PlanLinearLayouts(w, "t", 64, 96, Set(true, false, 32)); });
    Check("PlanLinearLayouts refuses g32 at K % 64 != 0 too", Has(m4, "not divisible by 64"));
    const std::string m5 =
        ThrowMessage([&] { PlanLinearLayouts(w, "t", 64, 512, Set(true, false, 48)); });
    Check("PlanLinearLayouts refuses an uninstantiated group", Has(m5, "32, 64, 128"));
  }
}

// ---- 4. accounting -------------------------------------------------------------------------------

void TestAccounting() {
  std::printf("---- 4. byte accounting with mixed groups ----\n");
  const int N = 64, K = 512;
  bool all = true;
  for (int g : {32, 64, 128}) {
    const LayoutSet sets[] = {Set(true, false, g), Set(true, true, g)};
    for (const LayoutSet& ls : sets) {
      ContainerWriter w;
      PlanLinearLayouts(w, "t", N, K, ls);
      if (w.PlannedDataBytes() != LinearLayoutBytes(N, K, ls)) {
        std::printf("  g%d: planned=%llu predicted=%llu\n", g,
                    static_cast<unsigned long long>(w.PlannedDataBytes()),
                    static_cast<unsigned long long>(LinearLayoutBytes(N, K, ls)));
        all = false;
      }
    }
  }
  Check("LinearLayoutBytes == PlannedDataBytes at g32/g64/g128 (6 sets)", all);
  // A whole mixed-group "model" planned into ONE writer: the sum still agrees.
  {
    ContainerWriter w;
    uint64_t predicted = 0;
    const int groups[] = {32, 64, 128, kW4A16Group};
    for (int i = 0; i < 4; ++i) {
      const LayoutSet ls = Set(true, false, groups[i]);
      PlanLinearLayouts(w, "l" + std::to_string(i), N, K, ls);
      predicted += LinearLayoutBytes(N, K, ls);
    }
    Check("mixed-group container: PlannedDataBytes == sum of LinearLayoutBytes",
          w.PlannedDataBytes() == predicted);
  }
  const uint64_t nk = static_cast<uint64_t>(N) * K;
  Check("w4a16 at g32 = 4.0 + 1.0 bits/weight",
        LinearLayoutBytes(N, K, Set(true, false, 32)) == nk / 2 + nk / 8);
  Check("w4a16 at g128 = 4.0 + 0.25 bits/weight",
        LinearLayoutBytes(N, K, Set(true, false, 128)) == nk / 2 + nk / 32);
  // --keep-bf16's "would have been" is priced at the linear's resolved group.
  {
    W4a16GroupRules rules({"attn\\.o$=32"});
    const LayoutSet requested = rules.Apply("text.layers.3.attn.o", Set(true, false));
    KeepBf16Selector sel("attn\\.o$");
    std::ostringstream log;
    sel.Record("text.layers.3.attn.o", N, K, requested, log);
    const int64_t want = static_cast<int64_t>(nk * 2) - static_cast<int64_t>(nk / 2 + nk / 8);
    Check("keep-bf16 ExtraBytes against a g32 linear is bf16 - (wq + N*K/32 dwords)",
          sel.ExtraBytes() == want);
    KeepBf16Selector sel2("attn\\.o$");
    sel2.Record("text.layers.3.attn.o", N, K, Set(true, false, 128), log);
    Check("...and differs from the same linear at g128 by exactly the wsz delta",
          sel2.ExtraBytes() - sel.ExtraBytes() == static_cast<int64_t>(nk / 8 - nk / 32));
  }
}

// ---- 5. emission ---------------------------------------------------------------------------------

void TestEmission() {
  std::printf("---- 5. emission ----\n");
  Check("W4a16WszName at the default group is the historical name",
        W4a16WszName("t", kW4A16Group) == "t.w4a16.wsz");
  Check("W4a16WszName at another group carries .g<g>",
        W4a16WszName("t", 32) == (kW4A16Group == 32 ? "t.w4a16.wsz" : "t.w4a16.wsz.g32") &&
            W4a16WszName("t.x", 128) ==
                (kW4A16Group == 128 ? "t.x.w4a16.wsz" : "t.x.w4a16.wsz.g128"));

  const int N = 48, K = 256;  // 3 row tiles; K a multiple of 32, 64 and 128
  const std::vector<float> w = RandomNormal(static_cast<size_t>(N) * K, 17);

  for (int g : {32, 64, 128}) {
    const std::string tag = " g" + std::to_string(g);
    const std::string path = TempPath("r4dx_test_w4a16_groups_g.r4dx");
    const LayoutSet ls = Set(true, false, g);
    {
      ContainerWriter writer;
      PlanLinearLayouts(writer, "t", N, K, ls);
      writer.FinalizeHeader(path, nlohmann::json{{"r4dx_format_version", "1"}});
      EmitLinearLayouts(writer, "t", w, N, K, ls, 3);
      writer.Finish();
    }
    const Container c = ReadContainer(path);
    const std::string wsz_name = W4a16WszName("t", g);
    const std::vector<std::string> want_names = {"t.w4a16.wq", wsz_name};
    std::vector<std::string> sorted_want = want_names;
    std::sort(sorted_want.begin(), sorted_want.end());
    Check("tensor set" + tag + " (wsz named " + wsz_name + ")", c.ok && Names(c) == sorted_want);
    if (!c.ok || !c.tensors.count(wsz_name)) continue;

    std::vector<uint8_t> q, zero;
    std::vector<float> scale;
    QuantizeInt4Asymmetric(w.data(), N, K, g, 3, q, scale, zero);
    Check("w4a16.wq" + tag + " == PackW4Nibbles(QuantizeInt4Asymmetric(g))",
          c.tensors.at("t.w4a16.wq") == Bytes(PackW4Nibbles(q, N, K, 3)));
    Check("wsz" + tag + " == PackW4A16Scales(g), N*K/g dwords",
          c.tensors.at(wsz_name) == Bytes(PackW4A16Scales(scale, zero, N, K, g)) &&
              c.tensors.at(wsz_name).size() == static_cast<size_t>(N) * K / g * 4);
    std::remove(path.c_str());
  }

  // LDLQ through EmitLinearLayouts at g32 and g128: the linear's own group reaches the quantizer.
  {
    // H = X^T X / rows on AR(1)-correlated inputs, so error feedback has something to act on.
    const int rows = 1024;
    std::vector<float> X = RandomNormal(static_cast<size_t>(rows) * K, 23);
    for (int r = 0; r < rows; ++r)
      for (int k = 1; k < K; ++k)
        X[static_cast<size_t>(r) * K + k] += 0.8f * X[static_cast<size_t>(r) * K + k - 1];
    std::vector<float> H(static_cast<size_t>(K) * K);
    for (int i = 0; i < K; ++i) {
      for (int j = i; j < K; ++j) {
        double s = 0.0;
        for (int r = 0; r < rows; ++r)
          s += static_cast<double>(X[static_cast<size_t>(r) * K + i]) * X[static_cast<size_t>(r) * K + j];
        H[static_cast<size_t>(i) * K + j] = H[static_cast<size_t>(j) * K + i] =
            static_cast<float>(s / rows);
      }
    }
    const LdlqFactor f = FactorHessian(H, K, 0.01f, 4);
    for (int g : {32, 128}) {
      const std::string tag = " g" + std::to_string(g);
      const std::string path = TempPath("r4dx_test_w4a16_groups_ldlq.r4dx");
      const LayoutSet ls = Set(true, false, g);
      QuantOptions opts;
      opts.ldlq = &f;
      {
        ContainerWriter writer;
        PlanLinearLayouts(writer, "t", N, K, ls);
        writer.FinalizeHeader(path, nlohmann::json{{"r4dx_format_version", "1"}});
        EmitLinearLayouts(writer, "t", w, N, K, ls, 3, opts);
        writer.Finish();
      }
      const Container c = ReadContainer(path);
      std::vector<uint8_t> q, zero;
      std::vector<float> scale;
      QuantizeInt4AsymmetricLdlq(w.data(), N, K, g, f, 3, q, scale, zero);
      const std::string wsz_name = W4a16WszName("t", g);
      Check("LDLQ" + tag + ": wq == QuantizeInt4AsymmetricLdlq(g) packed",
            c.ok && c.tensors.count("t.w4a16.wq") &&
                c.tensors.at("t.w4a16.wq") == Bytes(PackW4Nibbles(q, N, K, 3)));
      Check("LDLQ" + tag + ": " + wsz_name + " == its scales packed at g",
            c.ok && c.tensors.count(wsz_name) &&
                c.tensors.at(wsz_name) == Bytes(PackW4A16Scales(scale, zero, N, K, g)));
      std::remove(path.c_str());
    }
    std::vector<uint8_t> q32, z32, q128, z128;
    std::vector<float> s32, s128;
    QuantizeInt4AsymmetricLdlq(w.data(), N, K, 32, f, 3, q32, s32, z32);
    QuantizeInt4AsymmetricLdlq(w.data(), N, K, 128, f, 3, q128, s128, z128);
    Check("LDLQ g32 and g128 produce different codes and N*K/g scales",
          q32 != q128 && s32.size() == static_cast<size_t>(N) * K / 32 &&
              s128.size() == static_cast<size_t>(N) * K / 128);
  }
}

// ---- 6. the exe ----------------------------------------------------------------------------------

#if defined(R4DX_CONVERT_EXE)

// cmd.exe strips the outermost pair of quotes of a command that starts with one, so wrap it whole.
int Run(const std::string& args, const std::string& log) {
  const std::string cmd = "\"\"" + std::string(R4DX_CONVERT_EXE) + "\" " + args + " > \"" + log +
                          "\" 2>&1\"";
  return std::system(cmd.c_str());
}

std::string ReadText(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

void TestExe(const std::string& fixtures) {
  std::printf("---- 6. r4dx-convert --selftest ----\n");
  const std::string input = fixtures + "/input.safetensors";
  const std::string out = TempPath("r4dx_test_w4a16_groups_exe.r4dx");
  const std::string log = TempPath("r4dx_test_w4a16_groups_exe.log");
  const std::string base_args =
      "--selftest --selftest-input \"" + input + "\" --selftest-output \"" + out + "\" --threads 3";

  // A hessian dir whose one key is "selftest" (K = 256, the fixture's), from fixtures/hess_small.
  const std::string hdir = TempPath("r4dx_test_w4a16_groups_hess");
  {
    std::filesystem::create_directories(hdir);
    const std::string hess = ReadText(fixtures + "/hess_small/bc.hess");
    std::ofstream(hdir + "\\bc.hess", std::ios::binary).write(hess.data(),
                                                             static_cast<std::streamsize>(hess.size()));
    nlohmann::json m = nlohmann::json::parse(ReadText(fixtures + "/hess_small/hessian.json"));
    nlohmann::json j = {{"format", m.at("format")},
                        {"version", m.at("version")},
                        {"files", {{"bc.hess", m.at("files").at("bc.hess")}}},
                        {"keys", {{"selftest", "bc.hess"}}}};
    std::ofstream(hdir + "\\hessian.json") << j.dump(2);
  }

  // (a) No rules: the whole file is what the converter wrote before per-tensor groups existed.
  // TensorDigest of `r4dx-convert --selftest ... --threads 3 <args>` on fixtures/input.safetensors,
  // from the build of commit bcebb21 (R4DX_W4A16_GROUP=64, run with the equivalent w4a16-only
  // arguments: that build's default also wrote the retired layouts). The earlier whole-file sha256s
  // were taken at 95cc327, before per-tensor groups were written.
  if (kW4A16Group == 64) {
    const struct {
      const char* name;
      std::string args;
      const char* sha;
    } golden[] = {
        {"default layouts", "", "c93f078b596882914184d475367150ff3410055f84deaa80da0b34dbd931b796"},
        {"--layouts w4a16", "--layouts w4a16", "c93f078b596882914184d475367150ff3410055f84deaa80da0b34dbd931b796"},
        {"--quant search", "--quant search", "f5524e9f2f2dc1c76732925cb6956a2dd4a86751a8df73d15611b6f26ea536e8"},
        {"--keep-bf16 selftest", "--keep-bf16 selftest", "774dbead1bd3f01391c96d87b1ca2eb02a1e806dec176af777148a85293917ed"},
        {"--ldlq .* (w4a16)",
         "--layouts w4a16 --hessian-dir \"" + hdir + "\" --ldlq .*",
         "ce402c80fe09529c535ebf5f3c21dc8702928a818c718475f8e7a7ca9f6a2851"},
    };
    for (const auto& gcase : golden) {
      const int rc = Run(base_args + " " + gcase.args, log);
      const Container c = ReadContainer(out);
      const std::string sha = c.ok ? TensorDigest(c) : std::string("(unreadable)");
      Check(std::string("(a) no rules, ") + gcase.name + ": byte-identical to pre-Q3",
            rc == 0 && sha == gcase.sha);
      if (sha != gcase.sha) std::printf("  rc=%d sha=%s\n%s", rc, sha.c_str(), ReadText(log).c_str());
    }
  } else {
    std::printf("SKIP (a) pre-Q3 golden sha256s were taken at group 64; this build is %d\n",
                kW4A16Group);
  }

  // (b) A rule at a non-default group, --no-bf16: renamed wsz, the map, the run record.
  Run(base_args + " --layouts w4a16 --no-bf16", log);
  const Container plain = ReadContainer(out);
  const std::string rule = "^selftest$=" + std::to_string(kOther);
  {
    const int rc = Run(base_args + " --layouts w4a16 --no-bf16 --w4a16-group-rule \"" +
                           rule + "\"",
                       log);
    const Container c = ReadContainer(out);
    const std::string wsz = "selftest.w4a16.wsz.g" + std::to_string(kOther);
    Check("(b) rule: exit 0", rc == 0);
    if (rc != 0) std::printf("%s", ReadText(log).c_str());
    Check("(b) rule: " + wsz + " written, bare selftest.w4a16.wsz absent",
          c.ok && c.tensors.count(wsz) && !c.tensors.count("selftest.w4a16.wsz"));
    if (c.ok && c.tensors.count(wsz)) {
      Check("(b) rule: wsz holds N*K/g dwords", c.tensors.at(wsz).size() == 32u * 256 / kOther * 4);
    }
    const nlohmann::json md = c.ok ? c.header.at("__metadata__") : nlohmann::json::object();
    Check("(b) quant.w4a16.group stays the default",
          md.contains("quant") && md["quant"]["w4a16"]["group"].get<int>() == kW4A16Group);
    Check("(b) quant.w4a16.groups == {\"selftest\": g}",
          md.contains("quant") && md["quant"]["w4a16"].contains("groups") &&
              md["quant"]["w4a16"]["groups"] == nlohmann::json{{"selftest", kOther}});
    Check("(b) r4dx_convert_run records the rules and the map",
          md.contains("r4dx_convert_run") &&
              md["r4dx_convert_run"]["w4a16_group_rules"] == nlohmann::json::array({rule}) &&
              md["r4dx_convert_run"]["w4a16_groups"] == nlohmann::json{{"selftest", kOther}} &&
              md["r4dx_convert_run"]["w4a16_group_extra_bytes"].get<int64_t>() ==
                  static_cast<int64_t>(32 * 256 / kOther * 4) -
                      static_cast<int64_t>(32 * 256 / kW4A16Group * 4));
    Check("(b) no-rule run has no groups map and no run record",
          plain.ok && !plain.header["__metadata__"]["quant"]["w4a16"].contains("groups") &&
              !plain.header["__metadata__"].contains("r4dx_convert_run"));
  }
  // (c) A rule naming the default group: historical tensor name, no map, same tensor bytes.
  {
    const int rc = Run(base_args + " --layouts w4a16 --no-bf16 --w4a16-group-rule "
                                   "\"^selftest$=" + std::to_string(kW4A16Group) + "\"",
                       log);
    const Container c = ReadContainer(out);
    Check("(c) default-group rule: tensors byte-identical to the no-rule run, no map",
          rc == 0 && c.ok && plain.ok && c.tensors == plain.tensors &&
              !c.header["__metadata__"]["quant"]["w4a16"].contains("groups"));
  }
  // (d) LDLQ + a g32 / g128 rule end to end.
  for (int g : {32, 128}) {
    if (g == kW4A16Group) continue;
    const int rc = Run(base_args + " --layouts w4a16 --no-bf16 --hessian-dir \"" + hdir +
                           "\" --ldlq .* --w4a16-group-rule \"selftest=" + std::to_string(g) + "\"",
                       log);
    const Container c = ReadContainer(out);
    Check("(d) --ldlq with a g" + std::to_string(g) + " rule: exit 0, renamed wsz",
          rc == 0 && c.ok && c.tensors.count("selftest.w4a16.wsz.g" + std::to_string(g)));
    if (rc != 0) std::printf("%s", ReadText(log).c_str());
  }
  // (e) Refusals.
  {
    int rc = Run(base_args + " --layouts w4a16 --w4a16-group-rule \"" + rule + "\"", log);
    Check("(e) non-default group with the bf16 companion: nonzero exit, names --no-bf16",
          rc != 0 && Has(ReadText(log), "--no-bf16"));
    rc = Run(base_args + " --layouts w4a16 --no-bf16 --w4a16-group-rule \"selftest=48\"", log);
    Check("(e) group 48: nonzero exit", rc != 0 && Has(ReadText(log), "32, 64, 128"));
    rc = Run("--dflash-gguf \"" + fixtures + "/dflash_mini.gguf\" --out \"" + out +
                 "\" --w4a16-group-rule \"x=32\"",
             log);
    Check("(e) with --dflash-gguf: nonzero exit",
          rc != 0 && Has(ReadText(log), "does not apply to --dflash-gguf"));
    // A rule that matches nothing converts normally, with a warning.
    rc = Run(base_args + " --layouts w4a16 --no-bf16 --w4a16-group-rule \"nonesuch=32\"", log);
    Check("(e) rule matching nothing: exit 0 with a WARNING",
          rc == 0 && Has(ReadText(log), "WARNING: --w4a16-group-rule 'nonesuch=32'"));
  }
  std::remove(out.c_str());
  std::remove(log.c_str());
  std::error_code ec;
  std::filesystem::remove_all(hdir, ec);
}

#endif  // R4DX_CONVERT_EXE

}  // namespace

int main() {
  TestParsing();
  TestSelection();
  TestPlanning();
  TestAccounting();
  TestEmission();
#if defined(R4DX_CONVERT_EXE)
  TestExe(R4DX_CONVERT_FIXTURES_DIR);
#else
  std::printf("SKIP 6. r4dx-convert is not part of this build (R4DX_BUILD_CONVERT off)\n");
#endif
  if (g_failures != 0) {
    std::printf("FAIL (%d check(s))\n", g_failures);
    return 1;
  }
  std::printf("PASS\n");
  return 0;
}
