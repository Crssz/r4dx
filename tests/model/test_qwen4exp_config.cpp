// tests/model/test_qwen4exp_config.cpp -- CPU-only (no GPU, no container). docs/qwen4exp-plan.md M1-1.
//
//   * Qwen4ExpConfig::FromJson on the real Swift-1.5-Qwen3.8-Flash-Next-W4A16-AWQ config.json (revision
//     598afc96128f76a67117caf878f538d34f6f1e02; values inlined so the test is hermetic, the quantization
//     and vision blocks left out; if <R4DX_MODELS_ROOT>\Swift-1.5-Qwen3.8-Flash-Next-W4A16-AWQ\config.json
//     exists it is parsed too and must agree), both the full config and the bare text_config;
//   * the 3 GDN : 1 QSA layer pattern, ple_layer_ids (1-indexed) and their 0-based layer indices;
//   * the derived helpers: 16 n-gram heads of dim 160, 512 QSA blocks, dense == QSA up to 2051 tokens;
//   * every refusal names its cause; the three HF defaults (norm_topk_prob, seed, output_gate_type);
//   * DetectArch / DetectArchFromMetadata / ArchName for qwen4_exp, with the qwen3_5 and gemma4_unified
//     outcomes unchanged; ResolveContainerMaxCtx leaves a qwen4_exp request untouched.
#include <cstdint>
#include "r4dx/models_root.h"
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "arch.h"
#include "nlohmann/json.hpp"
#include "qwen4exp_config.h"

using nlohmann::json;
using r4dx::model::Arch;
using r4dx::model::Qwen4ExpConfig;

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

// Swift-1.5-Qwen3.8-Flash-Next-W4A16-AWQ config.json (transformers 5.8.0.dev0), without
// quantization_config / vision_config.
json RealModelConfig() {
  return json::parse(R"json({
  "architectures": ["Qwen4ExpForConditionalGeneration"],
  "image_token_id": 248056,
  "language_model_only": false,
  "model_type": "qwen4_exp",
  "text_config": {
    "attention_bias": false,
    "attention_dropout": 0.0,
    "bos_token_id": 248044,
    "dtype": "bfloat16",
    "eos_token_id": 248044,
    "full_attention_interval": 4,
    "hc_count": 4,
    "hc_lowrank": 320,
    "head_dim": 256,
    "heads_per_ngram": 8,
    "hidden_act": "silu",
    "hidden_size": 2560,
    "indexer_budget": 2048,
    "indexer_compress_ratio": 4,
    "indexer_head_dim": 128,
    "indexer_kv_heads": 1,
    "indexer_n_heads": 4,
    "initializer_range": 0.02,
    "layer_types": [
      "linear_attention", "linear_attention", "linear_attention", "full_attention",
      "linear_attention", "linear_attention", "linear_attention", "full_attention",
      "linear_attention", "linear_attention", "linear_attention", "full_attention",
      "linear_attention", "linear_attention", "linear_attention", "full_attention",
      "linear_attention", "linear_attention", "linear_attention", "full_attention",
      "linear_attention", "linear_attention", "linear_attention", "full_attention",
      "linear_attention", "linear_attention", "linear_attention", "full_attention",
      "linear_attention", "linear_attention", "linear_attention", "full_attention",
      "linear_attention", "linear_attention", "linear_attention", "full_attention",
      "linear_attention", "linear_attention", "linear_attention", "full_attention",
      "linear_attention", "linear_attention", "linear_attention", "full_attention",
      "linear_attention", "linear_attention", "linear_attention", "full_attention"
    ],
    "linear_conv_kernel_dim": 4,
    "linear_key_head_dim": 128,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 48,
    "linear_value_head_dim": 128,
    "make_ngram_vocab_size_divisible_by": 128,
    "mamba_ssm_dtype": "float32",
    "max_position_embeddings": 262144,
    "model_type": "qwen4_exp_text",
    "moe_intermediate_size": 640,
    "mtp": {
      "hybrid": true,
      "layer_types": ["full_attention"],
      "mtp_use_hidden_state_from_layer": null,
      "num_hidden_layers": 1,
      "rope_theta": 10000000
    },
    "mtp_num_hidden_layers": 1,
    "mtp_use_dedicated_embeddings": false,
    "ngram_size": 3,
    "ngram_vocab_size_base": 20000000,
    "num_attention_heads": 24,
    "num_experts": 512,
    "num_experts_per_tok": 10,
    "num_hidden_layers": 48,
    "num_key_value_heads": 2,
    "output_gate_type": "sigmoid",
    "output_router_logits": false,
    "pad_token_id": null,
    "partial_rotary_factor": 0.25,
    "ple_conv_kernel_size": 4,
    "ple_embed_dim": 2560,
    "ple_layer_ids": [2],
    "rms_norm_eps": 1e-06,
    "rope_parameters": {
      "mrope_interleaved": true,
      "mrope_section": [11, 11, 10],
      "partial_rotary_factor": 0.25,
      "rope_theta": 10000000,
      "rope_type": "default"
    },
    "router_aux_loss_coef": 0.001,
    "shared_expert_intermediate_size": 640,
    "split_ngram_parts": 128,
    "tie_word_embeddings": false,
    "use_cache": true,
    "vocab_size": 248320
  },
  "tie_word_embeddings": false,
  "transformers_version": "5.8.0.dev0",
  "video_token_id": 248057,
  "vision_end_token_id": 248054,
  "vision_start_token_id": 248053
})json");
}

json RealTextConfig() { return RealModelConfig().at("text_config"); }

bool SameGeometry(const Qwen4ExpConfig& a, const Qwen4ExpConfig& b) {
  return a.hidden_size == b.hidden_size && a.num_hidden_layers == b.num_hidden_layers && a.vocab_size == b.vocab_size &&
         a.layer_types == b.layer_types && a.linear_num_value_heads == b.linear_num_value_heads &&
         a.output_gate_type == b.output_gate_type && a.head_dim == b.head_dim && a.rope_theta == b.rope_theta &&
         a.mrope_section == b.mrope_section && a.num_experts == b.num_experts && a.hc_lowrank == b.hc_lowrank &&
         a.ple_layer_ids == b.ple_layer_ids && a.ple_embed_dim == b.ple_embed_dim &&
         a.split_ngram_parts == b.split_ngram_parts && a.indexer_budget == b.indexer_budget &&
         a.eos_token_id == b.eos_token_id && a.max_position_embeddings == b.max_position_embeddings;
}

void TestRealConfig() {
  const Qwen4ExpConfig c = Qwen4ExpConfig::FromJson(RealModelConfig());
  Check(c.hidden_size == 2560 && c.num_hidden_layers == 48 && c.vocab_size == 248320, "real: hidden/layers/vocab");
  Check(c.rms_norm_eps == 1e-6 && !c.tie_word_embeddings && c.max_position_embeddings == 262144,
        "real: eps 1e-6, untied head, max_position_embeddings 262144");
  Check(c.eos_token_id == 248044, "real: eos 248044");
  Check(c.linear_num_key_heads == 16 && c.linear_num_value_heads == 48 && c.linear_key_head_dim == 128 &&
            c.linear_value_head_dim == 128 && c.linear_conv_kernel_dim == 4,
        "real: GDN 16 k / 48 v heads of 128, conv 4");
  Check(c.output_gate_type == Qwen4ExpConfig::OutputGate::kSigmoid, "real: GDN gate is sigmoid (not hidden_act silu)");
  Check(c.num_attention_heads == 24 && c.num_key_value_heads == 2 && c.head_dim == 256, "real: attention 24 q / 2 kv x 256");
  Check(c.partial_rotary_factor == 0.25 && c.rope_theta == 1.0e7 && c.RotaryDim() == 64, "real: partial rotary 0.25, theta 1e7");
  Check(c.mrope_section == std::vector<int64_t>{11, 11, 10} && c.mrope_interleaved, "real: interleaved mrope [11, 11, 10]");
  Check(c.num_experts == 512 && c.num_experts_per_tok == 10 && c.moe_intermediate_size == 640 &&
            c.shared_expert_intermediate_size == 640 && c.norm_topk_prob,
        "real: MoE 512 / top-10 / 640, shared 640, norm_topk_prob default true");
  Check(c.hc_count == 4 && c.hc_lowrank == 320, "real: hyper-connections 4 x lowrank 320");
  Check(c.ple_layer_ids == std::vector<int64_t>{2} && c.ple_layer_indices == std::vector<int64_t>{1},
        "real: ple_layer_ids [2] (1-indexed) is layer index 1");
  Check(c.IsPleLayer(1) && !c.IsPleLayer(2) && !c.IsPleLayer(0), "real: IsPleLayer(1) only");
  Check(c.ple_embed_dim == 2560 && c.ple_conv_kernel_size == 4 && c.ngram_size == 3 && c.heads_per_ngram == 8,
        "real: PLE dim 2560, conv 4, 3-grams, 8 heads per order");
  Check(c.ngram_vocab_size_base == 20000000 && c.make_ngram_vocab_size_divisible_by == 128 && c.split_ngram_parts == 128,
        "real: n-gram vocab base 2e7, divisible by 128, 128 parts");
  Check(c.seed == 1234, "real: seed absent -> HF default 1234");
  Check(c.indexer_n_heads == 4 && c.indexer_kv_heads == 1 && c.indexer_head_dim == 128 && c.indexer_budget == 2048 &&
            c.indexer_compress_ratio == 4,
        "real: indexer 4 q / 1 k heads of 128, budget 2048, ratio 4");

  // Derived helpers (plan 2.3, 2.4).
  Check(c.NgramHeads() == 16 && c.NgramHeadDim() == 160, "derived: 16 n-gram heads of dim 160");
  Check(c.BlockTopK() == 512, "derived: BlockTopK 2048 / 4 = 512");
  Check(c.DenseExactMaxTokens() == 2051, "derived: dense == QSA up to 512 * 4 + 3 = 2051 tokens");

  // Layer pattern: QSA at 3, 7, ..., 47.
  Check(c.NumQsaLayers() == 12 && c.NumGdnLayers() == 36, "pattern: 12 QSA + 36 GDN");
  bool pattern = true;
  for (int i = 0; i < 48; ++i) pattern = pattern && (c.IsQsaLayer(i) == (i % 4 == 3)) && (c.IsGdnLayer(i) != c.IsQsaLayer(i));
  Check(pattern, "pattern: QSA exactly at i % 4 == 3 (HF's full_attention)");
  Check(Throws([&] { c.IsQsaLayer(48); }) && Throws([&] { c.IsQsaLayer(-1); }), "IsQsaLayer out of range throws");

  // Bare text_config, and HF's post-init spelling "indexed_attention".
  Check(SameGeometry(Qwen4ExpConfig::FromJson(RealTextConfig()), c), "FromJson accepts the bare text_config");
  json indexed = RealTextConfig();
  for (auto& lt : indexed["layer_types"])
    if (lt == "full_attention") lt = "indexed_attention";
  Check(SameGeometry(Qwen4ExpConfig::FromJson(indexed), c), "layer type 'indexed_attention' == 'full_attention' (QSA)");

  // The real file, when present: must parse and agree with the inlined copy.
  const std::string path = r4dx::ModelsPath("Swift-1.5-Qwen3.8-Flash-Next-W4A16-AWQ/config.json");
  std::ifstream f(path);
  if (f) {
    const json real = json::parse(f);
    Check(SameGeometry(Qwen4ExpConfig::FromJson(real), c), "real config.json on disk agrees with the inlined copy (" + path + ")");
    Check(r4dx::model::DetectArchFromMetadata(json{{"model_config", real}}) == Arch::kQwen4Exp,
          "real config.json as model_config: kQwen4Exp");
  } else {
    std::printf("SKIP: no config.json at %s\n", path.c_str());
  }
}

void TestDefaults() {
  const json base = RealTextConfig();
  json t = base;
  t["seed"] = 7;
  t["norm_topk_prob"] = false;
  const Qwen4ExpConfig c = Qwen4ExpConfig::FromJson(t);
  Check(c.seed == 7 && !c.norm_topk_prob, "explicit seed / norm_topk_prob are read");
  json no_gate = base;
  no_gate.erase("output_gate_type");
  Check(Qwen4ExpConfig::FromJson(no_gate).output_gate_type == Qwen4ExpConfig::OutputGate::kSilu,
        "output_gate_type absent -> hidden_act (silu), as HF");
  no_gate.erase("hidden_act");
  Check(Throws([&] { Qwen4ExpConfig::FromJson(no_gate); }, "hidden_act"),
        "output_gate_type and hidden_act both absent throws, naming hidden_act");
  json eos_list = base;
  eos_list["eos_token_id"] = {248044, 248046};
  Check(Qwen4ExpConfig::FromJson(eos_list).eos_token_id == 248044, "eos_token_id list -> its first id, as HF");
}

void TestRefusals() {
  const json base = RealTextConfig();
  auto with = [&](const char* key, json v) {
    json t = base;
    t[key] = std::move(v);
    return t;
  };
  auto without = [&](const char* key) {
    json t = base;
    t.erase(key);
    return t;
  };
  for (const char* k :
       {"hidden_size", "num_hidden_layers", "vocab_size", "rms_norm_eps", "layer_types", "linear_num_key_heads",
        "linear_num_value_heads", "linear_key_head_dim", "linear_value_head_dim", "linear_conv_kernel_dim",
        "num_attention_heads", "num_key_value_heads", "head_dim", "rope_parameters", "num_experts",
        "num_experts_per_tok", "moe_intermediate_size", "shared_expert_intermediate_size", "hc_count", "hc_lowrank",
        "ple_layer_ids", "ple_embed_dim", "ple_conv_kernel_size", "ngram_size", "heads_per_ngram",
        "ngram_vocab_size_base", "make_ngram_vocab_size_divisible_by", "split_ngram_parts", "indexer_n_heads",
        "indexer_kv_heads", "indexer_head_dim", "indexer_budget", "indexer_compress_ratio", "eos_token_id",
        "max_position_embeddings", "tie_word_embeddings"}) {
    Check(Throws([&] { Qwen4ExpConfig::FromJson(without(k)); }, std::string("missing '") + k + "'"),
          std::string("missing '") + k + "' throws, naming it");
  }
  for (const char* k : {"rope_theta", "partial_rotary_factor", "mrope_section", "mrope_interleaved"}) {
    json t = base;
    t["rope_parameters"].erase(k);
    Check(Throws([&] { Qwen4ExpConfig::FromJson(t); }, std::string("rope_parameters missing '") + k + "'"),
          std::string("missing rope_parameters.") + k + " throws, naming it");
  }
  Check(Throws([&] { Qwen4ExpConfig::FromJson(json::array()); }, "not an object"), "non-object config throws");
  Check(Throws([&] { Qwen4ExpConfig::FromJson(json{{"text_config", json::array()}}); }, "text_config is not an object"),
        "non-object text_config throws");

  // Layer pattern.
  json short_types = base;
  short_types["layer_types"].erase(47);
  Check(Throws([&] { Qwen4ExpConfig::FromJson(short_types); }, "layer_types length 47 does not match num_hidden_layers 48"),
        "layer_types: wrong length throws");
  json moved = base;
  moved["layer_types"][3] = "linear_attention";
  moved["layer_types"][2] = "full_attention";
  Check(Throws([&] { Qwen4ExpConfig::FromJson(moved); }, "layer_types[2]"), "layer_types: a moved QSA layer throws");
  json unknown = base;
  unknown["layer_types"][0] = "sliding_attention";
  Check(Throws([&] { Qwen4ExpConfig::FromJson(unknown); }, "unknown layer type 'sliding_attention'"),
        "layer_types: unknown type throws");

  // The asserted invariants.
  Check(Throws([&] { Qwen4ExpConfig::FromJson(with("hc_count", 3)); }, "hc_count must be 4"), "hc_count != 4 throws");
  Check(Throws([&] { Qwen4ExpConfig::FromJson(with("indexer_kv_heads", 2)); }, "indexer_kv_heads must be 1"),
        "indexer_kv_heads != 1 throws");
  Check(Throws([&] { Qwen4ExpConfig::FromJson(with("ngram_size", 2)); }, "ngram_size must be 3"), "ngram_size != 3 throws");
  Check(Throws([&] { Qwen4ExpConfig::FromJson(with("output_gate_type", "gelu")); }, "unsupported output_gate_type 'gelu'"),
        "output_gate_type not sigmoid/silu throws");
  Check(Qwen4ExpConfig::FromJson(with("output_gate_type", "silu")).output_gate_type == Qwen4ExpConfig::OutputGate::kSilu,
        "output_gate_type silu is accepted");
  Check(Throws([&] { Qwen4ExpConfig::FromJson(with("ple_embed_dim", 2568)); }, "is not divisible by (ngram_size - 1) * heads_per_ngram = 16"),
        "ple_embed_dim % 16 != 0 throws");
  Check(Throws([&] { Qwen4ExpConfig::FromJson(with("ple_layer_ids", {0})); }, "outside the 1-indexed range"),
        "ple_layer_ids 0 throws (1-indexed)");
  Check(Throws([&] { Qwen4ExpConfig::FromJson(with("ple_layer_ids", {49})); }, "outside the 1-indexed range"),
        "ple_layer_ids past the last layer throws");
  Check(Throws([&] { Qwen4ExpConfig::FromJson(with("ple_layer_ids", {4})); }, "not a linear_attention layer"),
        "ple_layer_ids on a QSA layer throws (HF refuses it too)");
  Check(Throws([&] { Qwen4ExpConfig::FromJson(with("ple_layer_ids", {2, 2})); }, "twice"), "duplicate ple_layer_ids throws");
  Check(Throws([&] { Qwen4ExpConfig::FromJson(with("indexer_budget", 2050)); }, "not divisible by indexer_compress_ratio"),
        "indexer_budget % compress_ratio != 0 throws");
  Check(Throws([&] { Qwen4ExpConfig::FromJson(with("num_key_value_heads", 5)); }, "multiple of num_key_value_heads"),
        "q heads not a multiple of kv heads throws");
  Check(Throws([&] { Qwen4ExpConfig::FromJson(with("hidden_size", 0)); }, "'hidden_size' must be > 0"), "hidden_size 0 throws");
  Check(Throws([&] { Qwen4ExpConfig::FromJson(with("hidden_size", "2560")); }, "'hidden_size' is not an integer"),
        "a string hidden_size throws");
  Check(Throws([&] { Qwen4ExpConfig::FromJson(with("model_type", "qwen3_5_text")); }, "text_config.model_type"),
        "a foreign text model_type throws");
  json foreign = RealModelConfig();
  foreign["model_type"] = "qwen3_5_moe";
  Check(Throws([&] { Qwen4ExpConfig::FromJson(foreign); }, "model_type \"qwen3_5_moe\""), "a foreign top-level model_type throws");
  json yarn = base;
  yarn["rope_parameters"]["rope_type"] = "yarn";
  Check(Throws([&] { Qwen4ExpConfig::FromJson(yarn); }, "rope_type"), "a non-default rope_type throws");
  json mrope = base;
  mrope["rope_parameters"]["mrope_section"] = {11, 11, 11};
  Check(Throws([&] { Qwen4ExpConfig::FromJson(mrope); }, "mrope_section sums to 33"), "mrope_section not matching the rotary width throws");
}

// A safetensors shell: 8-byte little-endian header length, then the JSON header.
std::string WriteShell(const std::string& name, const json& header) {
  const std::filesystem::path p = std::filesystem::temp_directory_path() / name;
  const std::string h = header.dump();
  const uint64_t len = h.size();
  std::ofstream f(p, std::ios::binary | std::ios::trunc);
  f.write(reinterpret_cast<const char*>(&len), 8);
  f.write(h.data(), static_cast<std::streamsize>(h.size()));
  return p.string();
}

void TestDetectArch() {
  using r4dx::model::ArchName;
  using r4dx::model::DetectArch;
  using r4dx::model::DetectArchFromMetadata;

  Check(std::string(ArchName(Arch::kQwen35)) == "qwen3_5" && std::string(ArchName(Arch::kGemma4)) == "gemma4_unified" &&
            std::string(ArchName(Arch::kQwen4Exp)) == "qwen4_exp",
        "ArchName spellings for all three arches");

  // qwen4_exp, three ways.
  Check(DetectArchFromMetadata(json{{"model_arch", "qwen4_exp"}}) == Arch::kQwen4Exp, "metadata: model_arch qwen4_exp");
  Check(DetectArchFromMetadata(json{{"model_config", RealModelConfig()}}) == Arch::kQwen4Exp,
        "metadata: model_config.model_type qwen4_exp, no model_arch key");
  Check(DetectArchFromMetadata(json{{"model_config", {{"architectures", {"Qwen4ExpForConditionalGeneration"}}}}}) ==
            Arch::kQwen4Exp,
        "metadata: model_config.architectures names Qwen4ExpForConditionalGeneration");
  Check(DetectArchFromMetadata(json{{"model_arch", "qwen4_exp"}, {"model_config", RealModelConfig()}}) == Arch::kQwen4Exp,
        "metadata: model_arch and model_config agree");

  // Unknown model_arch still throws and names the known ones.
  Check(Throws([&] { DetectArchFromMetadata(json{{"model_arch", "llama"}}); }, "unknown __metadata__.model_arch"),
        "metadata: unknown model_arch throws, not Qwen");
  Check(Throws([&] { DetectArchFromMetadata(json{{"model_arch", "qwen4"}}); }, "'qwen4_exp'"),
        "metadata: the unknown-arch error lists qwen4_exp");

  // The existing outcomes, unchanged.
  Check(DetectArchFromMetadata(json{{"model_arch", "qwen3_5"}}) == Arch::kQwen35, "metadata: model_arch qwen3_5");
  Check(DetectArchFromMetadata(json{{"model_arch", "gemma4_unified"}}) == Arch::kGemma4, "metadata: model_arch gemma4_unified");
  Check(DetectArchFromMetadata(json{{"model_config", {{"model_type", "gemma4_unified"}, {"text_config", json::object()}}}}) ==
            Arch::kGemma4,
        "metadata: model_config.model_type gemma4_unified");
  Check(DetectArchFromMetadata(json{{"model_config", {{"architectures", {"Gemma4UnifiedForConditionalGeneration"}}}}}) ==
            Arch::kGemma4,
        "metadata: model_config.architectures names Gemma 4");
  Check(DetectArchFromMetadata(json{{"model_config", {{"model_type", "qwen3_5"}, {"text_config", json::object()}}}}) ==
            Arch::kQwen35,
        "metadata: a Qwen3.5 model_config is kQwen35");
  Check(DetectArchFromMetadata(json{{"model_config", {{"model_type", "qwen3_5_moe"},
                                                      {"architectures", {"Qwen3_5MoeForConditionalGeneration"}}}}}) ==
            Arch::kQwen35,
        "metadata: a Qwen3.5-MoE model_config is kQwen35");
  Check(DetectArchFromMetadata(json::object()) == Arch::kQwen35, "metadata: empty (selftest container) is kQwen35");
  Check(DetectArchFromMetadata(json::array()) == Arch::kQwen35, "metadata: non-object is kQwen35");

  json q4_hdr = {{"__metadata__", {{"model_arch", "qwen4_exp"}, {"model_config", RealModelConfig()}}},
                 {"text.final_norm", {{"dtype", "U8"}, {"shape", {2}}, {"data_offsets", {0, 2}}}}};
  const std::string q4 = WriteShell("r4dx_arch_qwen4exp.r4dx", q4_hdr);
  Check(DetectArch(q4) == Arch::kQwen4Exp, "file: qwen4_exp container header");
  json qwen_hdr = {{"__metadata__", {{"model_config", {{"model_type", "qwen3_5"}, {"text_config", json::object()}}}}}};
  Check(DetectArch(WriteShell("r4dx_arch_qwen_q4test.r4dx", qwen_hdr)) == Arch::kQwen35, "file: Qwen container header");
  Check(DetectArch((std::filesystem::temp_directory_path() / "r4dx_arch_q4_missing.r4dx").string()) == Arch::kQwen35,
        "file: missing file is kQwen35");

  // ResolveContainerMaxCtx: a qwen4_exp request passes through untouched, the file is not read.
  using r4dx::model::ResolveContainerMaxCtx;
  Check(ResolveContainerMaxCtx(q4, Arch::kQwen4Exp, false, 262144, false) == 262144, "max ctx: qwen4_exp default untouched");
  Check(ResolveContainerMaxCtx("no/such/file.r4dx", Arch::kQwen4Exp, true, 8192, false) == 8192,
        "max ctx: qwen4_exp explicit untouched, never reads the file");
}

}  // namespace

int main() {
  TestRealConfig();
  TestDefaults();
  TestRefusals();
  TestDetectArch();
  if (g_failures != 0) {
    std::fprintf(stderr, "%d check(s) FAILED\n", g_failures);
    return 1;
  }
  std::printf("test_qwen4exp_config: all checks passed\n");
  return 0;
}
