// tests/model/test_rotation_meta.cpp -- CPU-only (no GPU, no container). docs/quant2.md section 3.1.
//
// The host half of the runtime's quant2 contract, src/model/rotation_meta.h, plus the tensor-parallel
// slicing of the rotation.* tensors (src/model/tp/tp_shard.cpp's RuleFor), exactly as Container::Load
// uses them:
//   1. No `rotation` key -> nullopt: an unrotated container gets no rotation op anywhere.
//   2. The converter's two real metadata blocks (q2a, q2ab, parsed from TEXT so the JSON number
//      types are what a container actually carries) parse, with the right tensor list and lengths.
//   3. Every refusal: unknown kinds (including the design draft's "hadamard1024x5"), missing or
//      mistyped fields, geometry other than the contract's, a model shape the kernels cannot serve.
//   4. TP=2: signs/mix5 replicate; each q2ab sign vector's rank range is EXACTLY the K range of the
//      linear whose input it rotates, a whole number of Hadamard blocks, and as long as
//      RotationTensors says the rank config needs (what LoadRotationWeights checks after the upload).
#include <cstdio>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include "model_config.h"
#include "nlohmann/json.hpp"
#include "rotation_meta.h"
#include "tp/tp_shard.h"

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

// What src/convert/main.cpp's RotationSource::Metadata() writes, as the container stores it (text).
const char* kQ2a = R"({"rotation": {"kind": "q2a", "seed": 1592598565, "hidden": 5120,
                                    "block": 1024}, "quant": {}})";
const char* kQ2ab = R"({"rotation": {"kind": "q2ab", "seed": 18446744073709551615, "hidden": 5120,
                                     "block": 1024, "had": {"down": 512, "o": 256,
                                                            "gdn_out": 128}}})";

bool Refuses(const nlohmann::json& metadata, const ModelConfig& g) {
  try {
    ParseRotationMetadata(metadata, g, "fixture.r4dx");
  } catch (const std::runtime_error&) {
    return true;
  }
  return false;
}

void TestParse() {
  const ModelConfig g = RealConfig();

  // 1. unrotated
  Check(!ParseRotationMetadata(nlohmann::json::parse(R"({"model_id": "x", "quant": {}})"), g, "a"),
        "no rotation key -> nullopt");
  Check(!ParseRotationMetadata(nlohmann::json::object(), g, "a"), "empty metadata -> nullopt");

  // 2. the two real blocks
  const auto a = ParseRotationMetadata(nlohmann::json::parse(kQ2a), g, "a");
  Check(a.has_value() && a->kind == RotationKind::kQ2a && !a->Hadamard() && a->seed == 1592598565u,
        "q2a parses (kind, seed)");
  if (a) {
    const auto t = RotationTensors(*a, g);
    Check(t.size() == 2 && std::string(t[0].name) == "rotation.signs" && t[0].elems == 5120 &&
              std::string(t[1].name) == "rotation.mix5" && t[1].elems == 25,
          "q2a needs rotation.signs [5120] and rotation.mix5 [25] only");
  }
  const auto ab = ParseRotationMetadata(nlohmann::json::parse(kQ2ab), g, "a");
  Check(ab.has_value() && ab->kind == RotationKind::kQ2ab && ab->Hadamard() &&
            ab->seed == 18446744073709551615ull,
        "q2ab parses, full-range u64 seed");
  if (ab) {
    const auto t = RotationTensors(*ab, g);
    Check(t.size() == 5 && std::string(t[2].name) == "rotation.had_down_signs" &&
              t[2].elems == 17408 && std::string(t[3].name) == "rotation.had_o_signs" &&
              t[3].elems == 6144 && std::string(t[4].name) == "rotation.had_gdn_out_signs" &&
              t[4].elems == 6144,
          "q2ab adds had_down [17408], had_o [6144], had_gdn_out [6144]");
  }
  // A seed built in C++ from a signed literal is a signed json number; still accepted.
  nlohmann::json signed_seed = nlohmann::json::parse(kQ2a);
  signed_seed["rotation"]["seed"] = 7;
  Check(!Refuses(signed_seed, g), "a non-negative signed-typed seed is accepted");

  // 3. refusals
  const auto with = [&](const char* base, const std::function<void(nlohmann::json&)>& edit) {
    nlohmann::json j = nlohmann::json::parse(base);
    edit(j["rotation"]);
    return j;
  };
  struct Case {
    const char* what;
    nlohmann::json metadata;
  };
  const std::vector<Case> refused = {
      {"kind hadamard1024x5 (the design draft's name)",
       with(kQ2a, [](nlohmann::json& r) { r["kind"] = "hadamard1024x5"; })},
      {"kind q2b", with(kQ2a, [](nlohmann::json& r) { r["kind"] = "q2b"; })},
      {"kind none", with(kQ2a, [](nlohmann::json& r) { r["kind"] = "none"; })},
      {"kind Q2A (case matters)", with(kQ2a, [](nlohmann::json& r) { r["kind"] = "Q2A"; })},
      {"no kind", with(kQ2a, [](nlohmann::json& r) { r.erase("kind"); })},
      {"numeric kind", with(kQ2a, [](nlohmann::json& r) { r["kind"] = 1; })},
      {"rotation is a string", nlohmann::json::parse(R"({"rotation": "q2a"})")},
      {"rotation is null", nlohmann::json::parse(R"({"rotation": null})")},
      {"no seed", with(kQ2a, [](nlohmann::json& r) { r.erase("seed"); })},
      {"negative seed", with(kQ2a, [](nlohmann::json& r) { r["seed"] = -1; })},
      {"string seed", with(kQ2a, [](nlohmann::json& r) { r["seed"] = "0x5EED2025"; })},
      {"no hidden", with(kQ2a, [](nlohmann::json& r) { r.erase("hidden"); })},
      {"hidden 4096", with(kQ2a, [](nlohmann::json& r) { r["hidden"] = 4096; })},
      {"block 512", with(kQ2a, [](nlohmann::json& r) { r["block"] = 512; })},
      {"no block", with(kQ2a, [](nlohmann::json& r) { r.erase("block"); })},
      {"q2a with a had block",
       with(kQ2a, [](nlohmann::json& r) { r["had"] = {{"down", 512}, {"o", 256}, {"gdn_out", 128}}; })},
      {"q2ab without had", with(kQ2ab, [](nlohmann::json& r) { r.erase("had"); })},
      {"q2ab had.down 256", with(kQ2ab, [](nlohmann::json& r) { r["had"]["down"] = 256; })},
      {"q2ab had.o 128", with(kQ2ab, [](nlohmann::json& r) { r["had"]["o"] = 128; })},
      {"q2ab had.gdn_out 256", with(kQ2ab, [](nlohmann::json& r) { r["had"]["gdn_out"] = 256; })},
      {"q2ab no had.gdn_out", with(kQ2ab, [](nlohmann::json& r) { r["had"].erase("gdn_out"); })},
  };
  int n = 0;
  for (const Case& c : refused) {
    const bool t = Refuses(c.metadata, g);
    Check(t, std::string("refuses: ") + c.what);
    n += t ? 1 : 0;
  }
  std::printf("PASS: %d/%zu malformed / unknown rotation blocks refused\n", n, refused.size());

  // Model shapes the contract does not cover.
  ModelConfig small = g;
  small.hidden_size = 4096;
  Check(Refuses(nlohmann::json::parse(kQ2a), small), "refuses a model whose hidden is not 5120");
  ModelConfig hd = g;
  hd.head_dim = 128;
  Check(Refuses(nlohmann::json::parse(kQ2ab), hd), "q2ab refuses head_dim != 256 (one o block/head)");
  Check(!Refuses(nlohmann::json::parse(kQ2a), hd), "q2a does not care about head_dim");
  ModelConfig vd = g;
  vd.linear_value_head_dim = 256;
  Check(Refuses(nlohmann::json::parse(kQ2ab), vd), "q2ab refuses linear_value_head_dim != 128");
  ModelConfig im = g;
  im.intermediate_size = 17408 + 256;
  Check(Refuses(nlohmann::json::parse(kQ2ab), im), "q2ab refuses intermediate % 512 != 0");

  // The error names the container and says why.
  try {
    ParseRotationMetadata(with(kQ2a, [](nlohmann::json& r) { r["kind"] = "zzz"; }), g, "C:/m.r4dx");
    Check(false, "unknown kind throws");
  } catch (const std::runtime_error& e) {
    const std::string w = e.what();
    Check(w.find("C:/m.r4dx") != std::string::npos && w.find("zzz") != std::string::npos,
          "the unknown-kind error names the path and the kind: " + w);
  }
}

void TestTpSlices() {
  using namespace r4dx::model::tp;
  const ModelConfig g = RealConfig();
  Check(RuleFor(kRotationSigns, g).split == Split::kReplicate &&
            RuleFor(kRotationMix5, g).split == Split::kReplicate,
        "rotation.signs / rotation.mix5 replicate");
  const auto spec = ParseRotationMetadata(nlohmann::json::parse(kQ2ab), g, "a");
  if (!spec) return;
  struct Pair {
    const char* signs;
    const char* linear;
    int64_t block;
  };
  const Pair pairs[] = {{kRotationHadDownSigns, "text.layers.0.mlp.down", kHadDownBlock},
                        {kRotationHadOSigns, "text.layers.3.attn.o", kHadOBlock},
                        {kRotationHadGdnOutSigns, "text.layers.0.gdn.out_proj", kHadGdnOutBlock}};
  for (int r = 0; r < 2; ++r) {
    const ModelConfig local = ModelConfig::Shard(g, 2, r);
    const auto want_local = RotationTensors(*spec, local);
    for (size_t i = 0; i < 3; ++i) {
      const Pair& p = pairs[i];
      const std::string what = "rank " + std::to_string(r) + " " + p.signs;
      const ShardRule rule = RuleFor(p.signs, g);
      Check(rule.split == Split::kRows && rule.segments.size() == 1, what + " is a 1-segment row split");
      const std::vector<Range> rows = RankRows(rule, 2, r);
      const Range cols = RankCols(RuleFor(p.linear, g), 2, r);
      Check(rows.size() == 1 && rows[0].begin == cols.begin && rows[0].count == cols.count,
            what + " rank range == " + p.linear + "'s rank K range");
      Check(cols.begin % p.block == 0 && cols.count % p.block == 0,
            what + ": whole Hadamard blocks per rank");
      Check(rows[0].count == want_local[2 + i].elems,
            what + " slice length == RotationTensors(rank config)");
    }
    Check(want_local[0].elems == 5120 && want_local[1].elems == 25,
          "rank " + std::to_string(r) + ": signs / mix5 keep their full length");
  }
}

}  // namespace

int main() {
  TestParse();
  TestTpSlices();
  std::printf("%d checks, %d failures\n", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}
