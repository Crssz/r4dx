// tests/model/test_qwen35moe_config.cpp -- CPU-only (no GPU, no container). docs/qwen35moe-plan.md M1.
//
//   * Qwen35MoeConfig::FromJson on the Ornith-1.5-122B config.json (values inlined, layer_types generated from
//     full_attention_interval 4 so the test is hermetic; if E:\models\r4dx\Ornith-1.5-122B\config.json exists it is
//     parsed too and must agree), full config and bare text_config;
//   * 36 GDN : 12 full-attention layers, derived helpers, the routed-expert parameter count (116.0 B);
//   * every refusal names its cause;
//   * DetectArch / DetectArchFromMetadata / ArchName for qwen3_5_moe, the other arches unchanged.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>

#include "arch.h"
#include "nlohmann/json.hpp"
#include "qwen35moe_config.h"

using nlohmann::json;
using r4dx::model::Arch;
using r4dx::model::Qwen35MoeConfig;

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
    return substr.empty() || std::string(e.what()).find(substr) != std::string::npos;
  }
  return false;
}

json TextConfig() {
  json layers = json::array();
  for (int i = 0; i < 48; ++i) layers.push_back(i % 4 == 3 ? "full_attention" : "linear_attention");
  return json{{"attn_output_gate", true},
              {"eos_token_id", 248044},
              {"full_attention_interval", 4},
              {"head_dim", 256},
              {"hidden_size", 3072},
              {"layer_types", layers},
              {"linear_conv_kernel_dim", 4},
              {"linear_key_head_dim", 128},
              {"linear_num_key_heads", 16},
              {"linear_num_value_heads", 64},
              {"linear_value_head_dim", 128},
              {"max_position_embeddings", 262144},
              {"mlp_only_layers", json::array()},
              {"model_type", "qwen3_5_moe_text"},
              {"moe_intermediate_size", 1024},
              {"mtp_num_hidden_layers", 1},
              {"num_attention_heads", 32},
              {"num_experts", 256},
              {"num_experts_per_tok", 8},
              {"num_hidden_layers", 48},
              {"num_key_value_heads", 2},
              {"partial_rotary_factor", 0.25},
              {"rms_norm_eps", 1e-06},
              {"rope_parameters",
               {{"mrope_interleaved", true},
                {"mrope_section", {11, 11, 10}},
                {"partial_rotary_factor", 0.25},
                {"rope_theta", 10000000},
                {"rope_type", "default"}}},
              {"shared_expert_intermediate_size", 1024},
              {"tie_word_embeddings", false},
              {"vocab_size", 248320}};
}

json FullConfig() {
  return json{{"architectures", {"Qwen3_5MoeForConditionalGeneration"}},
              {"model_type", "qwen3_5_moe"},
              {"text_config", TextConfig()},
              {"vision_config",
               {{"depth", 27}, {"hidden_size", 1152}, {"out_hidden_size", 3072}, {"model_type", "qwen3_5_moe_vision"}}}};
}

void CheckGeometry(const Qwen35MoeConfig& c, const std::string& tag) {
  Check(c.hidden_size == 3072 && c.num_hidden_layers == 48 && c.vocab_size == 248320, tag + ": hidden / layers / vocab");
  Check(c.NumGdnLayers() == 36 && c.NumAttnLayers() == 12, tag + ": 36 GDN + 12 full attention");
  Check(c.IsGdnLayer(0) && c.IsGdnLayer(2) && !c.IsGdnLayer(3) && !c.IsGdnLayer(47) && c.IsGdnLayer(44), tag + ": layer pattern");
  Check(c.num_experts == 256 && c.num_experts_per_tok == 8 && c.moe_intermediate_size == 1024 &&
            c.shared_expert_intermediate_size == 1024,
        tag + ": MoE geometry");
  Check(c.KeyDim() == 2048 && c.ValueDim() == 8192 && c.ConvDim() == 12288 && c.GqaRepeats() == 4, tag + ": GDN dims");
  Check(c.RotaryDim() == 64 && c.AttnGqa() == 16, tag + ": attention dims");
  Check(c.mrope_section == std::vector<int64_t>({11, 11, 10}) && c.mrope_interleaved, tag + ": mrope");
  Check(c.RoutedExpertParams() == 48LL * 256 * 3 * 3072 * 1024, tag + ": routed expert parameters (116.0 B)");
  Check(c.mtp_num_hidden_layers == 1 && c.eos_token_id == 248044 && !c.tie_word_embeddings, tag + ": mtp / eos / untied head");
  const auto m = c.ToModelConfig();
  Check(m.arch == Arch::kQwen35Moe && m.num_experts == 256 && m.moe_intermediate_size == 1024 && m.mtp_num_hidden_layers == 0 &&
            m.hidden_size == 3072 && m.layer_types.size() == 48,
        tag + ": ToModelConfig");
}

}  // namespace

int main() {
  {
    const Qwen35MoeConfig c = Qwen35MoeConfig::FromJson(FullConfig());
    CheckGeometry(c, "full config");
    Check(c.vision_depth == 27 && c.vision_hidden_size == 1152 && c.vision_out_hidden_size == 3072, "vision_config read");
    CheckGeometry(Qwen35MoeConfig::FromJson(TextConfig()), "bare text_config");
  }
  {
    const char* path = "E:\\models\\r4dx\\Ornith-1.5-122B\\config.json";
    if (std::filesystem::exists(path)) {
      std::ifstream f(path, std::ios::binary);
      const std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
      CheckGeometry(Qwen35MoeConfig::FromJson(json::parse(s)), "downloaded config.json");
      Check(r4dx::model::DetectArchFromMetadata(json{{"model_config", json::parse(s)}}) == Arch::kQwen35Moe,
            "downloaded config.json detects as qwen3_5_moe");
    } else {
      std::printf("SKIP: %s not present\n", path);
    }
  }

  // ---- refusals ----
  {
    json j = FullConfig();
    j["model_type"] = "qwen4_exp";
    Check(Throws([&] { Qwen35MoeConfig::FromJson(j); }, "want \"qwen3_5_moe\""), "refuse wrong model_type");
    j = FullConfig();
    j["text_config"].erase("num_experts");
    Check(Throws([&] { Qwen35MoeConfig::FromJson(j); }, "num_experts"), "refuse missing num_experts");
    j = FullConfig();
    j["text_config"]["layer_types"].push_back("full_attention");
    Check(Throws([&] { Qwen35MoeConfig::FromJson(j); }, "layer_types has 49"), "refuse layer_types length mismatch");
    j = FullConfig();
    j["text_config"]["layer_types"][0] = "sliding_attention";
    Check(Throws([&] { Qwen35MoeConfig::FromJson(j); }, "unknown layer type"), "refuse unknown layer type");
    j = FullConfig();
    j["text_config"]["num_experts_per_tok"] = 300;
    Check(Throws([&] { Qwen35MoeConfig::FromJson(j); }, "exceeds num_experts"), "refuse top-k > experts");
    j = FullConfig();
    j["text_config"]["mlp_only_layers"] = {0};
    Check(Throws([&] { Qwen35MoeConfig::FromJson(j); }, "mlp_only_layers"), "refuse dense layers in the MoE stack");
    j = FullConfig();
    j["text_config"]["rope_parameters"]["mrope_section"] = {11, 11, 11};
    Check(Throws([&] { Qwen35MoeConfig::FromJson(j); }, "mrope_section sums"), "refuse mrope_section that misses rotary_dim/2");
    j = FullConfig();
    j["text_config"]["linear_num_value_heads"] = 60;
    Check(Throws([&] { Qwen35MoeConfig::FromJson(j); }, "not a multiple of linear_num_key_heads"), "refuse GDN head ratio");
  }

  // ---- arch detection ----
  Check(std::string(r4dx::model::ArchName(Arch::kQwen35Moe)) == "qwen3_5_moe", "ArchName(kQwen35Moe)");
  Check(r4dx::model::DetectArchFromMetadata({{"model_arch", "qwen3_5_moe"}}) == Arch::kQwen35Moe, "metadata: model_arch qwen3_5_moe");
  Check(r4dx::model::DetectArchFromMetadata({{"model_config", FullConfig()}}) == Arch::kQwen35Moe, "metadata: model_config model_type");
  Check(r4dx::model::DetectArchFromMetadata({{"model_config", json{{"architectures", {"Qwen3_5MoeForConditionalGeneration"}}}}}) ==
            Arch::kQwen35Moe,
        "metadata: architectures only");
  Check(r4dx::model::DetectArchFromMetadata({{"model_config", json{{"model_type", "qwen3_5"}}}}) == Arch::kQwen35,
        "dense qwen3_5 still kQwen35");
  Check(r4dx::model::DetectArchFromMetadata({{"model_arch", "qwen4_exp"}}) == Arch::kQwen4Exp, "qwen4_exp unchanged");
  Check(r4dx::model::DetectArchFromMetadata(json::object()) == Arch::kQwen35, "no metadata: kQwen35");
  Check(r4dx::model::ResolveContainerMaxCtx("no/such/file.r4dx", Arch::kQwen35Moe, true, 8192, false) == 8192,
        "max ctx: qwen3_5_moe request untouched");

  std::printf("%s (%d failure%s)\n", g_failures == 0 ? "ALL PASS" : "FAILED", g_failures, g_failures == 1 ? "" : "s");
  return g_failures == 0 ? 0 : 1;
}
