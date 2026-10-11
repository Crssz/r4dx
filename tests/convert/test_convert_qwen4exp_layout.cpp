// tests/convert/test_convert_qwen4exp_layout.cpp -- CPU-only (ctest `convert_qwen4exp_layout`), plan M1-2.
//
// qwen4exp_layout.cpp over the REAL checkpoint's geometry (the 48-layer / 512-expert config and the tensor
// names + shapes of the Swift-1.5-Qwen3.8-Flash-Next AWQ checkpoint, in their virtual bf16 form), driven
// through a recording Kit with no weight bytes:
//   * every non-skipped tensor is consumed, vision / mtp / n-gram tensors are skipped on purpose, and the
//     container names, roles, descale sites and expert fusion are exactly plan 4.11 item 3 (the expected
//     lists are written out below independently of the layout's own loops);
//   * the coverage audit names a tensor that is neither consumed nor skipped, --layers limits it, and a
//     missing tensor / wrong shape / mixed layer kinds / a tied head fail before anything is registered;
//   * the per-(layer, class) trellis KB rule names its offenders;
//   * the n-gram sidecar header reader (a synthetic file in a temp dir) and its refusals.
// Prints the seconds the 49,152-expert-base registration + audit takes (no weights, no I/O).
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "nlohmann/json.hpp"
#include <sstream>
#include "r4dx_convert/trellis_import.hpp"
#include "qwen4exp_layout.hpp"

namespace fs = std::filesystem;
namespace qw = r4dx_convert::qwen4exp;
using nlohmann::json;

namespace {

int g_failures = 0;

void Check(bool cond, const std::string& what) {
  if (!cond) {
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++g_failures;
  } else {
    std::printf("PASS: %s\n", what.c_str());
  }
}

template <class F>
bool Throws(F&& f, const std::string& substr = "") {
  try {
    f();
  } catch (const std::exception& e) {
    const bool ok = substr.empty() || std::string(e.what()).find(substr) != std::string::npos;
    if (!ok) std::fprintf(stderr, "  (threw, but not with '%s': %s)\n", substr.c_str(), e.what());
    return ok;
  }
  return false;
}

// ---- the real config ---------------------------------------------------------------------------------------

json RealConfig() {
  json types = json::array();
  for (int i = 0; i < 48; ++i) types.push_back((i + 1) % 4 == 0 ? "full_attention" : "linear_attention");
  return {{"model_type", "qwen4_exp"},
          {"tie_word_embeddings", false},
          {"text_config",
           {{"hidden_size", 2560},
            {"num_hidden_layers", 48},
            {"vocab_size", 248320},
            {"layer_types", types},
            {"num_attention_heads", 24},
            {"num_key_value_heads", 2},
            {"head_dim", 256},
            {"num_experts", 512},
            {"moe_intermediate_size", 640},
            {"shared_expert_intermediate_size", 640},
            {"hc_count", 4},
            {"hc_lowrank", 320},
            {"linear_num_key_heads", 16},
            {"linear_num_value_heads", 48},
            {"linear_key_head_dim", 128},
            {"linear_value_head_dim", 128},
            {"linear_conv_kernel_dim", 4},
            {"indexer_n_heads", 4},
            {"indexer_kv_heads", 1},
            {"indexer_head_dim", 128},
            {"ple_embed_dim", 2560},
            {"ple_conv_kernel_size", 4},
            {"ple_layer_ids", {2}},
            {"eos_token_id", 248044},
            {"tie_word_embeddings", false}}}};
}

using Shapes = std::map<std::string, std::vector<int64_t>>;

// The checkpoint's tensor names and shapes (virtual bf16 `.weight` form for the AWQ linears), written
// out literally from the real header (docs/qwen4exp-plan.md 2.1-2.5, the index of rev 598afc96).
Shapes RealCheckpoint(int experts = 512) {
  Shapes s;
  const int64_t H = 2560, W = 10240, V = 248320;
  const std::string P = "model.language_model.layers.";
  s["model.language_model.embed_tokens.weight"] = {V, H};
  s["lm_head.weight"] = {V, H};
  s["model.language_model.hyper_connection_mixer.hc_norm.weight"] = {W};
  s["model.language_model.hyper_connection_mixer.input_mix_weight_down.weight"] = {320, W};
  s["model.language_model.hyper_connection_mixer.input_mix_weight_up.weight"] = {W, 320};
  for (int i = 0; i < 48; ++i) {
    const std::string p = P + std::to_string(i) + ".";
    for (const char* blk : {"attn_hyper_connection.", "mlp_hyper_connection."}) {
      s[p + blk + "hc_norm.weight"] = {W};
      s[p + blk + "input_mix_weight_down.weight"] = {320, W};
      s[p + blk + "input_mix_weight_up.weight"] = {W, 320};
      s[p + blk + "block_inject_weight.weight"] = {4, W};
    }
    if ((i + 1) % 4 == 0) {
      s[p + "self_attn.q_proj.weight"] = {12288, H};
      s[p + "self_attn.k_proj.weight"] = {512, H};
      s[p + "self_attn.v_proj.weight"] = {512, H};
      s[p + "self_attn.o_proj.weight"] = {H, 6144};
      s[p + "self_attn.q_norm.weight"] = {256};
      s[p + "self_attn.k_norm.weight"] = {256};
      s[p + "self_attn.indexer.index_qk_proj.weight"] = {640, H};
      s[p + "self_attn.indexer.q_layernorm.weight"] = {128};
      s[p + "self_attn.indexer.k_layernorm.weight"] = {128};
    } else {
      s[p + "linear_attn.in_proj_qkv.weight"] = {10240, H};
      s[p + "linear_attn.in_proj_z.weight"] = {6144, H};
      s[p + "linear_attn.out_proj.weight"] = {H, 6144};
      s[p + "linear_attn.in_proj_a.weight"] = {48, H};
      s[p + "linear_attn.in_proj_b.weight"] = {48, H};
      s[p + "linear_attn.conv1d.weight"] = {10240, 1, 4};
      s[p + "linear_attn.A_log"] = {48};
      s[p + "linear_attn.dt_bias"] = {48};
      s[p + "linear_attn.norm.weight"] = {128};
    }
    if (i == 1) {
      s[p + "ple.key_proj.weight"] = {W, H};
      s[p + "ple.value_proj.weight"] = {H, H};
      s[p + "ple.norm_key.weight"] = {W};
      s[p + "ple.norm_query.weight"] = {W};
      s[p + "ple.norm_conv.weight"] = {W};
      s[p + "ple.conv1d.weight"] = {W, 1, 4};
      s[p + "ple.ple_embedding.layer_multipliers"] = {3};
      s[p + "ple.ple_embedding.ngram_heads_offsets"] = {16};
      s[p + "ple.ple_embedding.ngram_heads_vocab_sizes"] = {16};
      for (int sh = 0; sh < 128; ++sh)
        s[p + "ple.ple_embedding.ngram_embedding.shard_" + std::to_string(sh) + ".weight"] = {2500012, 160};
    }
    s[p + "mlp.gate.weight"] = {experts, H};
    s[p + "mlp.shared_expert.gate_proj.weight"] = {640, H};
    s[p + "mlp.shared_expert.up_proj.weight"] = {640, H};
    s[p + "mlp.shared_expert.down_proj.weight"] = {H, 640};
    s[p + "mlp.shared_expert_gate.weight"] = {1, H};
    for (int e = 0; e < experts; ++e) {
      const std::string q = p + "mlp.experts." + std::to_string(e) + ".";
      s[q + "gate_proj.weight"] = {640, H};
      s[q + "up_proj.weight"] = {640, H};
      s[q + "down_proj.weight"] = {H, 640};
    }
  }
  // vision and MTP: skipped on purpose
  s["model.visual.patch_embed.proj.weight"] = {1152, 3, 2, 16, 16};
  s["model.visual.blocks.0.attn.qkv.weight"] = {3456, 1152};
  s["mtp.fc_hidden.weight"] = {H, 2 * H};
  s["mtp.layers.0.mlp.experts.down_proj"] = {512, H, 640};
  s["mtp.layers.0.self_attn.q_proj.weight"] = {12288, H};
  return s;
}

// The container tensors plan 4.11 item 3 lists for the real model, as (name -> kind), kind one of
// bf16 | fp32 | linear | linear_bf16only | linear_head | descale_k | descale_v.
std::map<std::string, std::string> ExpectedContainer(int experts = 512) {
  std::map<std::string, std::string> c;
  c["text.embed_tokens"] = "bf16";
  c["lm_head"] = "linear_head";
  for (const char* n : {"norm", "down", "up"}) c[std::string("text.hc_final.") + n] = "bf16";
  for (int i = 0; i < 48; ++i) {
    const std::string b = "text.layers." + std::to_string(i) + ".";
    for (const char* blk : {"attn", "mlp"})
      for (const char* n : {"norm", "down", "up", "inject"}) c[b + "hc." + blk + "." + n] = "bf16";
    c[b + "moe.router"] = "bf16";
    for (const char* n : {"gate", "up", "down"}) c[b + "moe.shared." + n] = "linear_bf16only";
    c[b + "moe.shared_gate"] = "bf16";
    for (int e = 0; e < experts; ++e) {
      c[b + "moe.experts." + std::to_string(e) + ".gate_up"] = "linear";
      c[b + "moe.experts." + std::to_string(e) + ".down"] = "linear";
    }
    if ((i + 1) % 4 == 0) {
      for (const char* n : {"attn.qg", "attn.k", "attn.v", "attn.o"}) c[b + n] = "linear";
      for (const char* n : {"attn.q_norm", "attn.k_norm", "attn.indexer.qk", "attn.indexer.q_norm",
                            "attn.indexer.k_norm"})
        c[b + n] = "bf16";
      c[b + "attn.k_descale"] = "descale_k";
      c[b + "attn.v_descale"] = "descale_v";
    } else {
      for (const char* n : {"gdn.in_proj_qkv", "gdn.in_proj_z", "gdn.out_proj"}) c[b + n] = "linear";
      for (const char* n : {"gdn.in_proj_a", "gdn.in_proj_b", "gdn.conv1d_weight", "gdn.norm_weight"}) c[b + n] = "bf16";
      c[b + "gdn.A_log"] = "fp32";
      c[b + "gdn.dt_bias"] = "fp32";
    }
    if (i == 1)
      for (const char* n : {"ple.key_proj", "ple.value_proj", "ple.norm_key", "ple.norm_query", "ple.norm_conv",
                            "ple.conv1d"})
        c[b + n] = "bf16";
  }
  return c;
}

struct Recorder {
  Shapes shapes;
  std::map<std::string, std::string> kind;  // container name -> kind
  std::map<std::string, std::string> hf_of;  // container name -> first HF name
  struct Linear {
    std::vector<std::string> hf;
    qw::LinearRole role;
  };
  std::map<std::string, Linear> linears;
  struct Descale {
    int n_kv, layer;
    std::string kind;
  };
  std::map<std::string, Descale> descales;
  size_t calls = 0;

  qw::Kit MakeKit() {
    qw::Kit kit;
    kit.all_names = [this]() {
      std::vector<std::string> n;
      n.reserve(shapes.size());
      for (const auto& kv : shapes) n.push_back(kv.first);
      return n;
    };
    kit.has = [this](const std::string& n) { return shapes.count(n) != 0; };
    kit.shape = [this](const std::string& n) { return shapes.at(n); };
    kit.add_bf16 = [this](const std::string& hf, const std::string& name) { Add(name, "bf16", hf); };
    kit.add_fp32_widen = [this](const std::string& hf, const std::string& name) { Add(name, "fp32", hf); };
    kit.add_descale = [this](const std::string& name, int n_kv, int layer, const char* k) {
      Add(name, std::string("descale_") + k, "");
      descales[name] = {n_kv, layer, k};
    };
    kit.add_linear = [this](const std::vector<std::string>& hf, const std::string& base, qw::LinearRole role) {
      Add(base, role == qw::LinearRole::kHead ? "linear_head" : role == qw::LinearRole::kBf16Only ? "linear_bf16only" : "linear",
          hf[0]);
      linears[base] = {hf, role};
    };
    return kit;
  }
  void Add(const std::string& name, const std::string& k, const std::string& hf) {
    ++calls;
    if (!kind.emplace(name, k).second) throw std::runtime_error("registered twice: " + name);
    hf_of[name] = hf;
  }
};

void TestLayoutRealGeometry() {
  std::printf("---- plan 4.11 item 3 over the real geometry ----\n");
  const qw::TextShape s = qw::ParseTextConfig(RealConfig());
  Check(s.hidden == 2560 && s.layers_total == 48 && s.num_experts == 512 && s.StreamWidth() == 10240 && s.QkvRows() == 10240,
        "ParseTextConfig: hidden 2560, 48 layers, 512 experts, stream 4 x 2560, qkv rows 10240");
  Check(s.HasPle(1) && !s.HasPle(0) && !s.HasPle(2) && s.ple_layers.size() == 1, "ple_layer_ids [2] (1-based) -> layer index 1");
  Check(s.IsFull(3) && s.IsFull(47) && !s.IsFull(0) && !s.IsFull(44), "QSA layers are 3, 7, ..., 47");

  Recorder rec;
  rec.shapes = RealCheckpoint();
  qw::Kit kit = rec.MakeKit();
  const auto t0 = std::chrono::steady_clock::now();
  const std::unordered_set<std::string> consumed = qw::AddTextStack(kit, s, 48);
  const auto t1 = std::chrono::steady_clock::now();
  const std::vector<std::string> unconsumed = qw::UnconsumedTensors(kit.all_names(), consumed, 48);
  const auto t2 = std::chrono::steady_clock::now();
  std::printf("INFO: 512 experts x 48 layers = %zu expert linears registered (%zu kit calls) in %.3f s; coverage audit over %zu names in %.3f s\n",
              static_cast<size_t>(512) * 48 * 2, rec.calls, std::chrono::duration<double>(t1 - t0).count(),
              rec.shapes.size(), std::chrono::duration<double>(t2 - t1).count());
  Check(std::chrono::duration<double>(t2 - t0).count() < 30.0, "registration + audit of 49,152 expert bases takes seconds, not minutes");

  Check(unconsumed.empty(), "the coverage audit finds nothing left over (" + std::to_string(unconsumed.size()) + " unconsumed)");
  if (!unconsumed.empty()) std::fprintf(stderr, "  first unconsumed: %s\n", unconsumed.front().c_str());
  size_t skipped = 0, non_skipped = 0;
  for (const auto& kv : rec.shapes) (qw::IsSkipped(kv.first) ? skipped : non_skipped)++;
  Check(consumed.size() == non_skipped && skipped == 5 + 3 + 128,
        "consumed = every non-skipped tensor (" + std::to_string(non_skipped) + "); skipped = vision + mtp + 131 n-gram tensors (" +
            std::to_string(skipped) + ")");

  const auto expected = ExpectedContainer();
  Check(rec.kind == expected, "the container tensor set and kinds are exactly plan 4.11 item 3's (" + std::to_string(expected.size()) +
                                  " names)");
  if (rec.kind != expected) {
    for (const auto& kv : expected)
      if (!rec.kind.count(kv.first) || rec.kind.at(kv.first) != kv.second) std::fprintf(stderr, "  expected but missing/different: %s\n", kv.first.c_str());
    for (const auto& kv : rec.kind)
      if (!expected.count(kv.first)) std::fprintf(stderr, "  unexpected: %s\n", kv.first.c_str());
  }

  // experts: gate_up is ONE two-part linear (gate rows, then up rows), down one part, role body
  const auto& gu = rec.linears.at("text.layers.5.moe.experts.17.gate_up");
  Check(gu.hf.size() == 2 && gu.hf[0] == "model.language_model.layers.5.mlp.experts.17.gate_proj.weight" &&
            gu.hf[1] == "model.language_model.layers.5.mlp.experts.17.up_proj.weight" && gu.role == qw::LinearRole::kBody,
        "expert gate_up = {gate_proj, up_proj} of that expert, role body");
  const auto& dn = rec.linears.at("text.layers.5.moe.experts.17.down");
  Check(dn.hf == std::vector<std::string>{"model.language_model.layers.5.mlp.experts.17.down_proj.weight"} &&
            dn.role == qw::LinearRole::kBody,
        "expert down = down_proj, role body");
  Check(rec.linears.at("text.layers.9.moe.shared.up").role == qw::LinearRole::kBf16Only &&
            rec.linears.at("text.layers.9.moe.shared.up").hf[0] == "model.language_model.layers.9.mlp.shared_expert.up_proj.weight",
        "shared experts are bf16-only linears");
  Check(rec.linears.at("lm_head").role == qw::LinearRole::kHead && rec.linears.at("lm_head").hf[0] == "lm_head.weight",
        "lm_head is the head role, from lm_head.weight (untied)");
  Check(rec.linears.at("text.layers.3.attn.qg").hf[0] == "model.language_model.layers.3.self_attn.q_proj.weight",
        "attn.qg is q_proj as it is (per-head [q | gate] rows)");
  Check(rec.hf_of.at("text.layers.1.ple.conv1d") == "model.language_model.layers.1.ple.conv1d.weight" &&
            rec.hf_of.at("text.layers.2.hc.mlp.inject") == "model.language_model.layers.2.mlp_hyper_connection.block_inject_weight.weight" &&
            rec.hf_of.at("text.hc_final.up") == "model.language_model.hyper_connection_mixer.input_mix_weight_up.weight" &&
            rec.hf_of.at("text.layers.3.attn.indexer.q_norm") == "model.language_model.layers.3.self_attn.indexer.q_layernorm.weight" &&
            rec.hf_of.at("text.layers.0.gdn.conv1d_weight") == "model.language_model.layers.0.linear_attn.conv1d.weight" &&
            rec.hf_of.at("text.layers.0.moe.shared_gate") == "model.language_model.layers.0.mlp.shared_expert_gate.weight" &&
            rec.hf_of.at("text.layers.0.moe.router") == "model.language_model.layers.0.mlp.gate.weight",
        "HF -> container name map for the hc / ple / indexer / gdn / router tensors");
  bool descales_ok = rec.descales.size() == 24;
  for (int i = 3; i < 48; i += 4) {
    for (const char* k : {"k", "v"}) {
      const auto it = rec.descales.find("text.layers." + std::to_string(i) + ".attn." + k + "_descale");
      descales_ok = descales_ok && it != rec.descales.end() && it->second.n_kv == 2 && it->second.layer == i && it->second.kind == k;
    }
  }
  Check(descales_ok, "k/v descale tables for exactly the 12 QSA layers, [2 kv heads], keyed by the checkpoint layer index");

  // --layers 2: the first two layers; the audit lets layers >= 2 go but not an unknown global tensor
  {
    Recorder r2;
    r2.shapes = RealCheckpoint();
    qw::Kit k2 = r2.MakeKit();
    const auto c2 = qw::AddTextStack(k2, s, 2);
    Check(qw::UnconsumedTensors(k2.all_names(), c2, 2).empty(), "--layers 2: layers 2.. are skipped by request, nothing else is left over");
    r2.shapes["model.language_model.surprise.weight"] = {3};
    const auto left = qw::UnconsumedTensors(r2.MakeKit().all_names(), c2, 2);
    Check(left == std::vector<std::string>{"model.language_model.surprise.weight"}, "--layers 2: an unknown global tensor is still reported");
  }
}

void TestLayoutRefusals() {
  std::printf("---- refusals ----\n");
  const qw::TextShape s = qw::ParseTextConfig(RealConfig());
  {
    Recorder rec;
    rec.shapes = RealCheckpoint(4);  // 4 experts: only to keep these cases quick; the config is read with 512
    qw::TextShape s4 = s;
    s4.num_experts = 4;
    qw::Kit kit = rec.MakeKit();
    const auto consumed = qw::AddTextStack(kit, s4, 48);
    rec.shapes["model.language_model.layers.7.mlp.experts.4.gate_proj.weight"] = {640, 2560};
    rec.shapes["model.visual.merger.norm.weight"] = {1152};
    rec.shapes["layers.0.unknown"] = {1};
    const auto left = qw::UnconsumedTensors(kit.all_names(), consumed, 48);
    Check(left == std::vector<std::string>{"layers.0.unknown", "model.language_model.layers.7.mlp.experts.4.gate_proj.weight"},
          "the audit names an expert beyond num_experts and an unknown tensor, and not the visual one");
  }
  auto fresh = [&](auto mutate, int experts = 2) {
    Recorder rec;
    rec.shapes = RealCheckpoint(experts);
    mutate(rec.shapes);
    qw::TextShape s2 = s;
    s2.num_experts = experts;
    qw::Kit kit = rec.MakeKit();
    qw::AddTextStack(kit, s2, 48);
  };
  Check(Throws([&] { fresh([](Shapes& m) { m.erase("model.language_model.layers.11.mlp.gate.weight"); }); },
               "no tensor 'model.language_model.layers.11.mlp.gate.weight'"),
        "a missing router fails and names it");
  Check(Throws([&] { fresh([](Shapes& m) { m["model.language_model.layers.4.linear_attn.in_proj_z.weight"] = {6144, 2048}; }); },
               "layers.4.linear_attn.in_proj_z.weight' has shape [6144,2048], config says [6144,2560]"),
        "a wrong shape fails and names the tensor and both shapes");
  Check(Throws([&] { fresh([](Shapes& m) { m["model.language_model.layers.3.linear_attn.in_proj_qkv.weight"] = {10240, 2560}; }); },
               "layer 3 is full_attention in the config but the checkpoint has linear_attn tensors"),
        "a GDN tensor on a full-attention layer is refused");
  Check(Throws([&] { fresh([](Shapes& m) { m["model.language_model.layers.2.self_attn.q_proj.weight"] = {12288, 2560}; }); },
               "layer 2 is linear_attention in the config but the checkpoint has self_attn tensors"),
        "a self_attn tensor on a GDN layer is refused");
  Check(Throws([&] { fresh([](Shapes& m) { m["model.language_model.layers.1.ple.key_proj.weight"] = {10240, 1280}; }); }, "ple.key_proj.weight"),
        "a wrong PLE shape fails");
  {
    Recorder rec;
    rec.shapes = RealCheckpoint(2);
    qw::TextShape s2 = s;
    s2.num_experts = 2;
    qw::Kit kit = rec.MakeKit();
    Check(Throws([&] { qw::AddTextStack(kit, s2, 49); }, "--layers 49 is outside [0, 48]"), "--layers beyond the model is refused");
  }
  json bad = RealConfig();
  bad["text_config"]["tie_word_embeddings"] = true;
  Check(Throws([&] { qw::ParseTextConfig(bad); }, "tie_word_embeddings is true"), "a tied head is refused (the layout writes lm_head.weight)");
  bad = RealConfig();
  bad["text_config"]["ple_layer_ids"] = {49};
  Check(Throws([&] { qw::ParseTextConfig(bad); }, "ple_layer_ids has 49"), "a PLE layer outside the model is refused");
  bad = RealConfig();
  bad["text_config"]["layer_types"][5] = "sliding_attention";
  Check(Throws([&] { qw::ParseTextConfig(bad); }, "layer_types has 'sliding_attention'"), "an unknown layer type is refused");
  bad = RealConfig();
  bad["text_config"].erase("hc_lowrank");
  Check(Throws([&] { qw::ParseTextConfig(bad); }, "has no 'hc_lowrank'"), "a missing config key is named");
  bad = RealConfig();
  bad["text_config"]["layer_types"].erase(bad["text_config"]["layer_types"].begin());
  Check(Throws([&] { qw::ParseTextConfig(bad); }, "layer_types length 47"), "layer_types shorter than num_hidden_layers is refused");
}

void TestExpertKbRule() {
  std::printf("---- per-(layer, class) KB ----\n");
  std::unordered_map<std::string, int> kb;
  for (int l = 0; l < 4; ++l)
    for (int e = 0; e < 8; ++e) {
      const std::string b = "text.layers." + std::to_string(l) + ".moe.experts." + std::to_string(e) + ".";
      kb[b + "gate_up"] = l < 2 ? 3 : 4;
      kb[b + "down"] = l == 0 ? 5 : 3;
    }
  kb["text.layers.0.attn.qg"] = 4;  // not an expert base: ignored
  Check(!Throws([&] { qw::CheckExpertKbUniform(kb); }), "one KB per layer and class, differing across layers and classes, is accepted");
  Check(!Throws([&] { qw::CheckExpertKbUniform({}); }), "an empty map (no trellis) is accepted");
  auto bad = kb;
  bad["text.layers.2.moe.experts.5.gate_up"] = 5;
  Check(Throws([&] { qw::CheckExpertKbUniform(bad); }, "layer 2 gate_up:"), "one expert at another KB fails, naming the layer and class");
  Check(Throws([&] { qw::CheckExpertKbUniform(bad); }, "text.layers.2.moe.experts.5.gate_up"), "... and the offending base");
  Check(Throws([&] { qw::CheckExpertKbUniform(bad); }, "KB 4 (text.layers.2.moe.experts.0.gate_up)"), "... and a base of the majority KB");
  bad = kb;
  bad["text.layers.1.moe.experts.7.down"] = 0;
  Check(Throws([&] { qw::CheckExpertKbUniform(bad); }, "not imported (text.layers.1.moe.experts.7.down)"),
        "an expert that is not imported (--keep-bf16) next to trellis ones fails");
  bad = kb;
  bad["text.layers.3.moe.experts.0.down"] = 4;
  bad["text.layers.1.moe.experts.0.gate_up"] = 4;
  Check(Throws([&] { qw::CheckExpertKbUniform(bad); }, "2 (layer, class) group(s)"), "every disagreeing group is counted");
}

// ---- the sidecar ----------------------------------------------------------------------------------------

struct TempDir {
  fs::path p;
  TempDir() {
    p = fs::temp_directory_path() / ("r4dx_q4exp_layout_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(p);
  }
  ~TempDir() {
    std::error_code ec;
    fs::remove_all(p, ec);
  }
};

std::vector<uint8_t> SidecarHeader(uint64_t rows, uint32_t row_bytes, int64_t eos, uint32_t version = 1, const char* magic = "R4DXNGRM") {
  std::vector<uint8_t> h(4096, 0);
  size_t off = 0;
  auto put = [&](const void* p, size_t n) {
    std::memcpy(h.data() + off, p, n);
    off += n;
  };
  put(magic, 8);
  put(&version, 4);
  put(&rows, 8);
  put(&row_bytes, 4);
  const uint32_t heads = 16, head_dim = row_bytes / 2, ngram = 3, hpn = 8;
  put(&heads, 4);
  put(&head_dim, 4);
  put(&ngram, 4);
  put(&hpn, 4);
  put(&eos, 8);
  const uint64_t seed = 99;
  put(&seed, 8);
  for (int64_t m : {7, 11, 13}) put(&m, 8);
  for (uint64_t i = 0; i < 16; ++i) {
    const uint64_t o = i * 100;
    put(&o, 8);
  }
  for (uint64_t i = 0; i < 16; ++i) {
    const uint64_t pr = 101 + i;
    put(&pr, 8);
  }
  put(&rows, 8);  // padded_rows
  for (int i = 0; i < 32; ++i) {
    const uint8_t b = static_cast<uint8_t>(0xA0 + i);
    put(&b, 1);
  }
  return h;
}

void WriteFile(const fs::path& p, const std::vector<uint8_t>& head, size_t body_bytes) {
  std::ofstream f(p, std::ios::binary | std::ios::trunc);
  f.write(reinterpret_cast<const char*>(head.data()), static_cast<std::streamsize>(head.size()));
  const std::vector<char> zeros(body_bytes, 0);
  f.write(zeros.data(), static_cast<std::streamsize>(zeros.size()));
}

void TestSidecar() {
  std::printf("---- n-gram sidecar header ----\n");
  const qw::TextShape s = qw::ParseTextConfig(RealConfig());
  const json cfg = RealConfig();
  TempDir t;
  const fs::path ok = t.p / "ok.r4dx-ngram";
  WriteFile(ok, SidecarHeader(50, 320, 248044), 50 * 320);
  const qw::NgramSidecar n = qw::ReadNgramSidecar(ok.string(), s, cfg);
  Check(n.rows == 50 && n.row_bytes == 320 && n.heads == 16 && n.head_dim == 160 && n.eos_token_id == 248044 && n.seed == 99 &&
            n.multipliers == std::vector<int64_t>{7, 11, 13} && n.head_offsets[3] == 300 && n.head_primes[15] == 116 && n.padded_rows == 50,
        "every header field is read at its offset");
  std::string want_sha;
  for (int i = 0; i < 32; ++i) {
    char b[3];
    std::snprintf(b, sizeof b, "%02x", 0xA0 + i);
    want_sha += b;
  }
  Check(n.sha256 == want_sha, "the body sha256 is the header's 32 bytes in lowercase hex (not rehashed)");
  Check(fs::path(n.path).is_absolute() && fs::path(n.path).filename() == "ok.r4dx-ngram", "the path is absolute");
  const json j = n.ToJson();
  Check(j.at("sha256") == want_sha && j.at("rows") == 50 && j.at("row_bytes") == 320 && j.at("path").is_string(), "ToJson carries path, sha256, rows, row_bytes");
  {  // a relative path is made absolute
    const fs::path cwd = fs::current_path();
    fs::current_path(t.p);
    const qw::NgramSidecar r = qw::ReadNgramSidecar("ok.r4dx-ngram", s, cfg);
    fs::current_path(cwd);
    Check(fs::path(r.path).is_absolute() && fs::equivalent(r.path, ok), "a relative --ngram-sidecar resolves to the absolute path");
  }
  auto bad = [&](const char* name, const std::vector<uint8_t>& head, size_t body, const std::string& want) {
    const fs::path p = t.p / name;
    WriteFile(p, head, body);
    Check(Throws([&] { qw::ReadNgramSidecar(p.string(), s, cfg); }, want), std::string("refused: ") + want);
  };
  bad("magic", SidecarHeader(50, 320, 248044, 1, "R4DXXXXX"), 50 * 320, "bad magic");
  bad("version", SidecarHeader(50, 320, 248044, 2), 50 * 320, "version 2");
  bad("trunc", SidecarHeader(50, 320, 248044), 49 * 320, "truncated or still being written");
  bad("long", SidecarHeader(50, 320, 248044), 51 * 320, "truncated or still being written");
  bad("rowbytes", SidecarHeader(50, 640, 248044), 50 * 640, "row_bytes 640 x 16 heads != 2 x ple_embed_dim 2560");
  bad("eos", SidecarHeader(50, 320, 7), 50 * 320, "eos id 7 != the config's eos_token_id 248044");
  {
    auto h = SidecarHeader(50, 320, 248044);
    h[4000] = 1;
    bad("padding", h, 50 * 320, "padding is not zero");
  }
  {
    const fs::path p = t.p / "short";
    WriteFile(p, std::vector<uint8_t>(100, 0), 0);
    Check(Throws([&] { qw::ReadNgramSidecar(p.string(), s, cfg); }, "shorter than its 4096-byte header"), "refused: a file shorter than the header");
  }
  Check(Throws([&] { qw::ReadNgramSidecar((t.p / "nope").string(), s, cfg); }, "cannot stat"), "refused: a missing file");
}

// 49,152 expert bases (73,728 HF records) through the converter's TrellisSource (manifest parse, Resolve per base,
// the coverage check, quant.trellis.linears): the pieces whose cost could be quadratic in the base count.
void TestTrellisScale() {
  std::printf("---- --trellis-from at 49,152 expert bases ----\n");
  using clk = std::chrono::steady_clock;
  TempDir t;
  const std::string config_text = RealConfig().dump();
  json tensors = json::object();
  const std::string sha(64, 'a');
  auto rec = [&](int64_t n, int64_t k, int kb) {
    return json{{"encoding", "trellis-exl3"}, {"K", static_cast<double>(kb)}, {"codebook", "mul1"}, {"k", k}, {"n", n},
                {"shape_hf", {n, k}}, {"words_shape", {k / 16, n / 16, 8 * kb}}, {"file", "L.safetensors"},
                {"file_sha256", sha}, {"rel_weight_err", 0.1}, {"hessian_basis", "matched"}};
  };
  std::vector<std::pair<std::string, std::vector<std::string>>> bases;
  for (int l = 0; l < 48; ++l)
    for (int e = 0; e < 512; ++e) {
      const std::string h = "model.language_model.layers." + std::to_string(l) + ".mlp.experts." + std::to_string(e) + ".";
      const int kb = l < 24 ? 3 : 4;
      tensors[h + "gate_proj.weight"] = rec(640, 2560, kb);
      tensors[h + "up_proj.weight"] = rec(640, 2560, kb);
      tensors[h + "down_proj.weight"] = rec(2560, 640, kb);
      const std::string b = "text.layers." + std::to_string(l) + ".moe.experts." + std::to_string(e) + ".";
      bases.push_back({b + "gate_up", {h + "gate_proj.weight", h + "up_proj.weight"}});
      bases.push_back({b + "down", {h + "down_proj.weight"}});
    }
  json man = {{"format", "r4dx-weights-override"}, {"version", 1}, {"encoding", "trellis-exl3"}, {"complete", true},
              {"missing", json::array()}, {"missing_count", 0}, {"bpw_target", nullptr}, {"K_uniform", nullptr},
              {"config_sha256", r4dx_convert::Sha256Hex(config_text)}, {"hessian_basis", "matched"}, {"codebook", "mul1"},
              {"allocation", {{"rule", "synthetic"}, {"sources", json::object()}, {"source_code_sha256", json::object()}}},
              {"tensors", tensors}};
  {
    std::ofstream f(t.p / "weights_override.json");
    f << man.dump();
  }
  r4dx_convert::trellis::TrellisOptions opt;
  opt.from = t.p.string();
  const auto t0 = clk::now();
  r4dx_convert::trellis::TrellisSource src(opt, config_text);
  const auto t1 = clk::now();
  std::ostringstream log;
  size_t imported = 0;
  std::unordered_map<std::string, int> kb;
  for (const auto& [base, hf] : bases) {
    r4dx_convert::LayoutSet ls;
    if (src.Resolve(base, hf, false, &ls, log)) ++imported;
    kb[base] = ls.trellis_bits;
  }
  const auto t2 = clk::now();
  std::ostringstream warn;
  src.CheckCoverage(log, warn);
  qw::CheckExpertKbUniform(kb);
  const auto t3 = clk::now();
  const std::string qm = src.QuantMetadata().dump();
  const auto t4 = clk::now();
  auto s = [](clk::time_point a, clk::time_point b) { return std::chrono::duration<double>(b - a).count(); };
  std::printf("INFO: manifest of %zu records (%.1f MB) parsed in %.2f s; Resolve x %zu %.2f s; CheckCoverage + KB rule %.2f s; QuantMetadata %.2f s (%.1f MB)\n",
              tensors.size(), static_cast<double>(man.dump().size()) / 1e6, s(t0, t1), bases.size(), s(t1, t2), s(t2, t3), s(t3, t4),
              static_cast<double>(qm.size()) / 1e6);
  Check(imported == bases.size() && warn.str().empty(), "all 49,152 expert bases resolve against the manifest, nothing unused");
  Check(s(t0, t4) < 60.0, "manifest, Resolve, coverage and metadata for 49,152 bases take seconds (linear, not quadratic)");
}

}  // namespace

int main() {
  try {
    TestLayoutRealGeometry();
    TestLayoutRefusals();
    TestExpertKbRule();
    TestSidecar();
    TestTrellisScale();
  } catch (const std::exception& e) {
    std::fprintf(stderr, "FAIL: exception: %s\n", e.what());
    return 1;
  }
  std::printf("%s (%d failure(s))\n", g_failures == 0 ? "ALL PASSED" : "FAILED", g_failures);
  return g_failures == 0 ? 0 : 1;
}
