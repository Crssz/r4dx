// tests/model/test_lfm2_config.cpp -- CPU-only (no GPU, no container, no HIP). docs/lfm2-plan.md L1-1.
//
//   * Lfm2Config::FromJson on the real LiquidAI/LFM2.5-24B-A2B config.json (values inlined so the test is hermetic):
//     every field, the layer pattern (attention at 2, 6, ..., 38; dense layers 0-1), the derived shapes, and every
//     refusal;
//   * Lfm2Config::Shard at TP=2: the per-rank shapes (heads 16 / kv 4, conv 1024, dense 5888, experts 768, vocab
//     64000) and the K % 256 / rows % 16 legality, plus its refusals;
//   * ToModelConfig carries arch kLfm2Moe and the MoE geometry tp::RuleFor reads;
//   * DetectArchFromMetadata / DetectArch / ResolveContainerMaxCtx for lfm2_moe, and the Qwen / Gemma answers unchanged.
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "arch.h"
#include "lfm2_config.h"
#include "model_config.h"
#include "nlohmann/json.hpp"

using nlohmann::json;
using r4dx::model::Arch;
using r4dx::model::Lfm2Config;
using r4dx::model::ModelConfig;

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

// LiquidAI/LFM2.5-24B-A2B config.json (read from E:\models\LiquidAI\LFM2.5-24B-A2B, 2026-10-14).
json RealConfig() {
  json types = json::array();
  for (int i = 0; i < 40; ++i) types.push_back(i % 4 == 2 ? "full_attention" : "conv");
  return {{"architectures", {"Lfm2MoeForCausalLM"}},
          {"bos_token_id", 124894},
          {"conv_L_cache", 3},
          {"conv_bias", false},
          {"dtype", "bfloat16"},
          {"eos_token_id", 124900},
          {"hidden_size", 2048},
          {"initializer_range", 0.02},
          {"intermediate_size", 11776},
          {"layer_types", types},
          {"max_position_embeddings", 131072},
          {"model_type", "lfm2_moe"},
          {"moe_intermediate_size", 1536},
          {"norm_eps", 1e-05},
          {"norm_topk_prob", true},
          {"num_attention_heads", 32},
          {"num_dense_layers", 2},
          {"num_experts", 64},
          {"num_experts_per_tok", 4},
          {"num_hidden_layers", 40},
          {"num_key_value_heads", 8},
          {"pad_token_id", 124893},
          {"rope_parameters", {{"rope_theta", 10000000.0}, {"rope_type", "default"}}},
          {"routed_scaling_factor", 1.0},
          {"tie_word_embeddings", true},
          {"transformers_version", "5.12.1"},
          {"use_cache", true},
          {"use_expert_bias", true},
          {"vocab_size", 128000}};
}

void TestRealConfig() {
  const Lfm2Config c = Lfm2Config::FromJson(RealConfig());
  Check(c.hidden_size == 2048 && c.num_hidden_layers == 40 && c.num_attention_heads == 32 && c.num_key_value_heads == 8 &&
            c.head_dim == 64 && c.Gqa() == 4,
        "geometry: 40 layers, hidden 2048, 32 q heads / 8 kv heads, head_dim 64 derived (no key in the config)");
  Check(c.intermediate_size == 11776 && c.moe_intermediate_size == 1536 && c.num_experts == 64 && c.num_experts_per_tok == 4 &&
            c.num_dense_layers == 2 && c.routed_scaling_factor == 1.0 && c.norm_topk_prob && c.use_expert_bias && c.router_fp32,
        "FFN: dense I 11776 (layers 0-1), 64 experts of 1536, top-4, routed_scaling_factor 1.0, expert_bias, fp32 router");
  Check(c.conv_l_cache == 3 && c.conv_dim == 2048 && c.ConvInProjRows() == 6144 && c.ExpertGateUpRows() == 3072,
        "ShortConv: kernel 3, 2048 channels, in_proj N 6144; expert gate_up N 3072");
  Check(c.vocab_size == 128000 && c.norm_eps == 1e-5 && c.rope_theta == 1e7 && c.tie_word_embeddings && c.max_position_embeddings == 131072 &&
            c.bos_token_id == 124894 && c.eos_token_id == 124900 && c.pad_token_id == 124893,
        "vocab 128000, eps 1e-5, rope theta 1e7, tied head, max_pos 131072, ids 124894 / 124900 / 124893");
  Check(c.NumAttentionLayers() == 10 && c.NumConvLayers() == 30 && c.NumMoeLayers() == 38, "10 attention + 30 conv layers, 38 MoE layers");
  bool pattern = true;
  for (int i = 0; i < 40; ++i)
    pattern = pattern && c.IsAttentionLayer(i) == (i % 4 == 2) && c.IsConvLayer(i) == (i % 4 != 2) && c.IsDenseLayer(i) == (i < 2) &&
              c.IsMoeLayer(i) == (i >= 2);
  Check(pattern, "layer pattern: attention at 2, 6, ..., 38; dense FFN on 0-1, MoE on 2-39");
  Check(Throws([&] { c.IsAttentionLayer(40); }, "out of range"), "IsAttentionLayer(40) is out of range");
  Check(c.QDim() == 2048 && c.KvDim() == 512, "QDim 2048, KvDim 512");

  // The generic view.
  const ModelConfig m = c.ToModelConfig();
  Check(m.arch == Arch::kLfm2Moe && m.hidden_size == 2048 && m.num_hidden_layers == 40 && m.layer_types.size() == 40 &&
            m.num_attention_heads == 32 && m.num_key_value_heads == 8 && m.head_dim == 64 && m.intermediate_size == 11776 &&
            m.moe_intermediate_size == 1536 && m.num_experts == 64 && m.num_dense_layers == 2 && m.vocab_size == 128000 &&
            m.tie_word_embeddings && m.linear_num_key_heads == 0 && m.linear_num_value_heads == 0 && m.tp_world == 1,
        "ToModelConfig: arch kLfm2Moe, the MoE geometry, GDN fields 0");
  Check(Lfm2Config::FromModelConfig(RealConfig()).hidden_size == 2048 &&
            Lfm2Config::FromModelConfig(json{{"text_config", RealConfig()}}).num_experts == 64,
        "FromModelConfig: flat config.json (and a nested text_config)");

  // Context.
  Check(c.ResolveMaxCtx(0) == 131072 && c.ResolveMaxCtx(-1) == 131072 && c.ResolveMaxCtx(8192) == 8192 && c.ResolveMaxCtx(131072) == 131072,
        "ResolveMaxCtx: default 131072, any explicit value up to it");
  Check(Throws([&] { c.ResolveMaxCtx(131073); }, "no rope scaling"), "ResolveMaxCtx: above 131072 throws (no extended context)");
}

void TestRefusals() {
  auto bad = [&](const char* key, json value, const std::string& needle, const std::string& what) {
    json c = RealConfig();
    c[key] = std::move(value);
    Check(Throws([&] { (void)Lfm2Config::FromJson(c); }, needle), what);
  };
  auto missing = [&](const char* key) {
    json c = RealConfig();
    c.erase(key);
    Check(Throws([&] { (void)Lfm2Config::FromJson(c); }, key), std::string("a missing '") + key + "' names the key");
  };
  bad("model_type", "gemma4_unified_text", "model_type", "refuses another model_type");
  bad("tie_word_embeddings", false, "tie_word_embeddings", "refuses an untied head");
  bad("conv_bias", true, "conv_bias", "refuses conv_bias");
  bad("conv_L_cache", 4, "conv_L_cache", "refuses a conv kernel other than 3");
  bad("use_expert_bias", false, "use_expert_bias", "refuses use_expert_bias false");
  bad("norm_topk_prob", false, "norm_topk_prob", "refuses norm_topk_prob false");
  bad("rope_parameters", json{{"rope_theta", 1e7}, {"rope_type", "yarn"}}, "rope_type", "refuses a scaled rope");
  bad("rope_scaling", json{{"rope_type", "linear"}, {"factor", 2.0}}, "rope_scaling", "refuses rope_scaling");
  bad("num_experts_per_tok", 65, "num_experts_per_tok", "refuses top-k > experts");
  bad("num_dense_layers", 41, "num_dense_layers", "refuses num_dense_layers > layers");
  bad("num_key_value_heads", 7, "multiple", "refuses heads not a multiple of kv heads");
  {
    json c = RealConfig();
    c["layer_types"][2] = "conv";
    Check(Throws([&] { (void)Lfm2Config::FromJson(c); }, "pattern"), "refuses a layer_types list off the i % 4 == 2 pattern");
    c = RealConfig();
    c["layer_types"][0] = "linear_attention";
    Check(Throws([&] { (void)Lfm2Config::FromJson(c); }, "unknown layer type"), "refuses an unknown layer type");
    c = RealConfig();
    c["layer_types"].push_back("conv");
    Check(Throws([&] { (void)Lfm2Config::FromJson(c); }, "does not match"), "refuses a layer_types length mismatch");
    c = RealConfig();
    c["head_dim"] = 64;
    Check(Lfm2Config::FromJson(c).head_dim == 64, "an explicit head_dim key is honoured");
    c["head_dim"] = 63;
    Check(Throws([&] { (void)Lfm2Config::FromJson(c); }, "even"), "refuses an odd head_dim");
  }
  for (const char* k : {"hidden_size", "layer_types", "num_experts", "moe_intermediate_size", "norm_eps", "conv_L_cache", "rope_parameters",
                        "vocab_size", "use_expert_bias", "num_dense_layers"})
    missing(k);
  Check(Throws([&] { (void)Lfm2Config::FromJson(json::array()); }, "not an object"), "refuses a non-object config");
}

void TestShard() {
  const Lfm2Config g = Lfm2Config::FromJson(RealConfig());
  for (int rank = 0; rank < 2; ++rank) {
    const Lfm2Config r = Lfm2Config::Shard(g, 2, rank);
    Check(r.tp_world == 2 && r.tp_rank == rank && r.num_attention_heads == 16 && r.num_key_value_heads == 4 && r.Gqa() == 4 &&
              r.intermediate_size == 5888 && r.moe_intermediate_size == 768 && r.conv_dim == 1024 && r.hidden_size == 2048 &&
              r.vocab_size == 128000 && r.VocabShardSize() == 64000 && r.VocabShardBegin() == rank * 64000 && r.IsShard(),
          "Shard(2, " + std::to_string(rank) + "): heads 16 / kv 4 (GQA 4), dense 5888, experts 768, conv 1024, vocab slice 64000");
    // The per-rank K the kernels see: attn.o 1024, conv out 1024, dense w2 5888 = 23 x 256, expert down 768 = 3 x 256.
    Check(r.QDim() == 1024 && r.QDim() % 256 == 0 && r.conv_dim % 256 == 0 && r.intermediate_size % 256 == 0 && r.moe_intermediate_size % 256 == 0 &&
              r.intermediate_size % 512 != 0 && r.moe_intermediate_size % 512 != 0,
          "Shard(2, " + std::to_string(rank) + "): every row-parallel K is a multiple of 256 (5888 and 768 are NOT multiples of 512)");
    // The expert linears at TP=2: gate_up N 1536 K 2048, down N 2048 K 768: all whole 128-blocks (trellis-legal).
    Check(r.ExpertGateUpRows() == 1536 && r.ExpertGateUpRows() % 128 == 0 && r.moe_intermediate_size % 128 == 0 && r.conv_dim % 128 == 0 &&
              r.intermediate_size % 128 == 0,
          "Shard(2, " + std::to_string(rank) + "): expert / conv / dense rank shapes are whole 128-blocks");
    const ModelConfig m = r.ToModelConfig();
    Check(m.tp_world == 2 && m.tp_rank == rank && m.moe_intermediate_size == 768 && m.num_experts == 64, "Shard: ToModelConfig carries the rank");
  }
  const Lfm2Config w1 = Lfm2Config::Shard(g, 1, 0);
  Check(w1.num_attention_heads == 32 && w1.intermediate_size == 11776 && w1.conv_dim == 2048 && !w1.IsShard() == true,
        "Shard(1, 0) is the global shape (and legal)");
  Check(Throws([&] { (void)Lfm2Config::Shard(g, 3, 0); }, "world"), "Shard: world 3 is refused");
  Check(Throws([&] { (void)Lfm2Config::Shard(g, 2, 2); }, "rank"), "Shard: rank 2 of 2 is refused");
  Check(Throws([&] { (void)Lfm2Config::Shard(Lfm2Config::Shard(g, 2, 0), 2, 0); }, "already a rank shard"), "Shard: a shard of a shard is refused");
  {
    Lfm2Config bad = g;
    bad.moe_intermediate_size = 1000;  // 500 per rank: not a multiple of 16 / 256
    Check(Throws([&] { (void)Lfm2Config::Shard(bad, 2, 0); }, "moe_intermediate_size"), "Shard: an expert width that does not split legally names the field");
    bad = g;
    bad.intermediate_size = 11264 + 128;  // /2 = 5696 = 22.25 x 256
    Check(Throws([&] { (void)Lfm2Config::Shard(bad, 2, 0); }, "dense w2"), "Shard: a dense width with K % 256 != 0 per rank names the K");
    bad = g;
    bad.num_key_value_heads = 1;
    bad.num_attention_heads = 32;
    Check(Throws([&] { (void)Lfm2Config::Shard(bad, 2, 0); }, "num_key_value_heads"), "Shard: 1 kv head cannot split 2 ways");
    bad = g;
    bad.vocab_size = 128001;
    Check(Throws([&] { (void)Lfm2Config::Shard(bad, 2, 0); }, "vocab_size"), "Shard: a vocab that does not split into 16-row tiles is refused");
  }
}

// ---- arch dispatch ----------------------------------------------------------------------------------------------

std::string WriteShell(const std::string& name, const json& header) {
  const std::filesystem::path p = std::filesystem::temp_directory_path() / name;
  const std::string h = header.dump();
  const uint64_t len = h.size();
  std::ofstream f(p, std::ios::binary | std::ios::trunc);
  f.write(reinterpret_cast<const char*>(&len), 8);
  f.write(h.data(), static_cast<std::streamsize>(h.size()));
  return p.string();
}

void TestArch() {
  using r4dx::model::DetectArch;
  using r4dx::model::DetectArchFromMetadata;
  using r4dx::model::ResolveContainerMaxCtx;

  Check(std::string(r4dx::model::ArchName(Arch::kLfm2Moe)) == "lfm2_moe" && std::string(r4dx::model::ArchName(Arch::kGemma4)) == "gemma4_unified" &&
            std::string(r4dx::model::ArchName(Arch::kQwen35)) == "qwen3_5",
        "ArchName: lfm2_moe / gemma4_unified / qwen3_5 (Qwen and Gemma spellings unchanged)");
  Check(DetectArchFromMetadata(json{{"model_arch", "lfm2_moe"}}) == Arch::kLfm2Moe, "metadata: model_arch lfm2_moe");
  Check(DetectArchFromMetadata(json{{"model_arch", "gemma4_unified"}}) == Arch::kGemma4 && DetectArchFromMetadata(json{{"model_arch", "qwen3_5"}}) == Arch::kQwen35,
        "metadata: gemma4_unified / qwen3_5 unchanged");
  Check(Throws([&] { DetectArchFromMetadata(json{{"model_arch", "llama"}}); }, "lfm2_moe"), "metadata: an unknown model_arch throws, naming lfm2_moe among the known");
  Check(DetectArchFromMetadata(json{{"model_config", RealConfig()}}) == Arch::kLfm2Moe, "metadata: flat model_config.model_type lfm2_moe, no model_arch key");
  Check(DetectArchFromMetadata(json{{"model_config", {{"architectures", {"Lfm2MoeForCausalLM"}}}}}) == Arch::kLfm2Moe,
        "metadata: model_config.architectures names Lfm2MoeForCausalLM");
  Check(DetectArchFromMetadata(json{{"model_config", {{"model_type", "qwen3_5"}, {"text_config", json::object()}}}}) == Arch::kQwen35 &&
            DetectArchFromMetadata(json{{"model_config", {{"architectures", {"Gemma4UnifiedForConditionalGeneration"}}}}}) == Arch::kGemma4,
        "metadata: Qwen / Gemma model_configs still resolve to Qwen / Gemma");

  json lfm_hdr = {{"__metadata__", {{"model_arch", "lfm2_moe"}, {"model_config", RealConfig()}}},
                  {"text.final_norm", {{"dtype", "U8"}, {"shape", {2}}, {"data_offsets", {0, 2}}}}};
  const std::string f = WriteShell("r4dx_lfm2_arch.r4dx", lfm_hdr);
  Check(DetectArch(f) == Arch::kLfm2Moe, "file: LFM2 container header");

  // max_ctx: default = the config's 131072 (the CLI / server default 262144 is NOT used), no extended-context opt-in.
  Check(ResolveContainerMaxCtx(f, Arch::kLfm2Moe, false, 262144, false) == 131072, "lfm2: no --max-ctx -> 131072");
  Check(ResolveContainerMaxCtx(f, Arch::kLfm2Moe, false, 262144, true) == 131072, "lfm2: --extended-ctx does not extend (no rope scaling)");
  Check(ResolveContainerMaxCtx(f, Arch::kLfm2Moe, true, 8192, false) == 8192, "lfm2: an explicit smaller value");
  Check(Throws([&] { ResolveContainerMaxCtx(f, Arch::kLfm2Moe, true, 262144, false); }, "no rope scaling"), "lfm2: 262144 is refused");
  Check(Throws([&] { ResolveContainerMaxCtx(f, Arch::kLfm2Moe, true, 200000, true); }, "no rope scaling"), "lfm2: an explicit value above 131072 is refused even with --extended-ctx");
  Check(Throws([&] { ResolveContainerMaxCtx(WriteShell("r4dx_lfm2_nocfg.r4dx", json::object()), Arch::kLfm2Moe, false, 0, false); }, "model_config"),
        "lfm2: a container without model_config is refused");
  Check(ResolveContainerMaxCtx("no/such/file.r4dx", Arch::kQwen35, false, 262144, true) == 262144, "qwen: still never reads the file");
}

}  // namespace

int main() {
  TestRealConfig();
  TestRefusals();
  TestShard();
  TestArch();
  std::printf("\n%s: %d failure(s)\n", g_failures == 0 ? "OK" : "FAILED", g_failures);
  return g_failures == 0 ? 0 : 1;
}
