// tests/model/test_gemma_config.cpp -- CPU-only (no GPU, no container). docs/gemma4-plan.md M1-1, M1-24.
//
//   * GemmaConfig::FromJson on the real Huihui-gemma-4-12B-it-abliterated text_config (values inlined
//     so the test is hermetic; if <R4DX_MODELS_ROOT>\Huihui-gemma-4-12B-it-abliterated\config.json exists it
//     is parsed too and must agree), through FromModelConfig's nesting as well;
//   * the 5:1 layer pattern is asserted (a swapped layer, a wrong length, an unknown type all throw);
//   * every refusal of an unsupported variant names its field;
//   * derived per-layer geometry (sliding vs full), rotary angles, the bf16 embed scale (62.0);
//   * ResolveMaxCtx: default 131072, 262144 opt-in only;
//   * GemmaConfig::Shard at TP=2 (heads, sliding kv, intermediate, replicated full kv, tied head) and
//     its legality rules;
//   * ToModelConfig / ModelConfig::arch (what TextModel::Config() exposes), Qwen's default kQwen35;
//   * DetectArch / DetectArchFromMetadata on synthetic safetensors-shell headers.
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
#include "gemma_config.h"
#include "model_config.h"
#include "nlohmann/json.hpp"

using nlohmann::json;
using r4dx::model::Arch;
using r4dx::model::GemmaConfig;
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

// Huihui-gemma-4-12B-it-abliterated config.json, "text_config" (transformers 5.10.0.dev0).
json RealTextConfig() {
  json t;
  t["attention_bias"] = false;
  t["attention_dropout"] = 0.0;
  t["attention_k_eq_v"] = true;
  t["bos_token_id"] = 2;
  t["enable_moe_block"] = false;
  t["eos_token_id"] = 1;
  t["final_logit_softcapping"] = 30.0;
  t["global_head_dim"] = 512;
  t["head_dim"] = 256;
  t["hidden_activation"] = "gelu_pytorch_tanh";
  t["hidden_size"] = 3840;
  t["hidden_size_per_layer_input"] = 0;
  t["intermediate_size"] = 15360;
  json types = json::array();
  for (int i = 0; i < 48; ++i) types.push_back((i + 1) % 6 == 0 ? "full_attention" : "sliding_attention");
  t["layer_types"] = types;
  t["max_position_embeddings"] = 131072;
  t["model_type"] = "gemma4_unified_text";
  t["moe_intermediate_size"] = nullptr;
  t["num_attention_heads"] = 16;
  t["num_experts"] = nullptr;
  t["num_global_key_value_heads"] = 1;
  t["num_hidden_layers"] = 48;
  t["num_key_value_heads"] = 8;
  t["num_kv_shared_layers"] = 0;
  t["pad_token_id"] = 0;
  t["rms_norm_eps"] = 1e-06;
  t["rope_parameters"] = {
      {"full_attention", {{"partial_rotary_factor", 0.25}, {"rope_theta", 1000000.0}, {"rope_type", "proportional"}}},
      {"sliding_attention", {{"rope_theta", 10000.0}, {"rope_type", "default"}}}};
  t["sliding_window"] = 1024;
  t["tie_word_embeddings"] = true;
  t["top_k_experts"] = nullptr;
  t["use_bidirectional_attention"] = "vision";
  t["use_cache"] = true;
  t["use_double_wide_mlp"] = false;
  t["vocab_size"] = 262144;
  t["vocab_size_per_layer_input"] = 262144;
  return t;
}

json RealModelConfig() {
  json m;
  m["architectures"] = {"Gemma4UnifiedForConditionalGeneration"};
  m["model_type"] = "gemma4_unified";
  m["eos_token_id"] = {1, 106};
  m["image_token_id"] = 258880;
  m["text_config"] = RealTextConfig();
  return m;
}

void TestRealConfig() {
  const GemmaConfig c = GemmaConfig::FromJson(RealTextConfig());
  Check(c.hidden_size == 3840 && c.num_hidden_layers == 48 && c.num_attention_heads == 16, "real: hidden/layers/heads");
  Check(c.num_kv_heads_sliding == 8 && c.num_kv_heads_full == 1, "real: kv heads 8 sliding / 1 full");
  Check(c.head_dim_sliding == 256 && c.head_dim_full == 512, "real: head dims 256 / 512");
  Check(c.sliding_window == 1024 && c.intermediate_size == 15360 && c.vocab_size == 262144, "real: window/mlp/vocab");
  Check(c.max_position_embeddings == 131072, "real: max_position_embeddings 131072 (Huihui, not google's 262144)");
  Check(c.rope_theta_sliding == 1.0e4 && c.rope_theta_full == 1.0e6 && c.partial_rotary_factor_full == 0.25, "real: rope");
  Check(c.rms_norm_eps == 1e-6 && c.final_logit_softcapping == 30.0, "real: eps and softcap");
  Check(c.tie_word_embeddings && c.attention_k_eq_v && c.v_norm_all_layers && c.embed_scale_bf16, "real: flags");
  Check(c.bos_token_id == 2 && c.pad_token_id == 0, "real: bos/pad");
  Check(c.use_bidirectional_attention == "vision", "real: use_bidirectional_attention vision");

  // 5:1 pattern: full at 5, 11, ..., 47.
  Check(c.NumFullLayers() == 8 && c.NumSlidingLayers() == 40, "pattern: 8 full + 40 sliding");
  bool pattern = true;
  for (int i = 0; i < 48; ++i) pattern = pattern && (c.IsFullLayer(i) == (i % 6 == 5));
  Check(pattern, "pattern: full exactly at i % 6 == 5");
  Check(Throws([&] { c.IsFullLayer(48); }), "IsFullLayer out of range throws");

  // Per-layer geometry (semantics 1.16; header shapes: layer 0 k 2048x3840, layer 5 k 512x3840, q 8192x3840).
  Check(c.HeadDim(0) == 256 && c.HeadDim(5) == 512, "geometry: head_dim per layer type");
  Check(c.NumKvHeads(0) == 8 && c.NumKvHeads(5) == 1, "geometry: kv heads per layer type");
  Check(c.Gqa(0) == 2 && c.Gqa(5) == 16, "geometry: GQA 2 sliding / 16 full");
  Check(c.QDim(0) == 4096 && c.QDim(5) == 8192, "geometry: q_proj rows / o_proj K 4096 / 8192");
  Check(c.KvDim(0) == 2048 && c.KvDim(5) == 512, "geometry: k_proj rows 2048 / 512");
  Check(c.RopeTheta(0) == 1.0e4 && c.RopeTheta(5) == 1.0e6, "geometry: rope theta per layer type");
  Check(c.RotaryAngles(0) == 128 && c.RotaryAngles(5) == 64, "geometry: rope angles 128 sliding / 64 full (semantics 1.5)");
  Check(c.EmbedScale() == 62.0f, "embed scale: sqrt(3840) in bf16 is 62.0 (semantics 1.3)");
  GemmaConfig f32 = c;
  f32.embed_scale_bf16 = false;
  Check(f32.EmbedScale() > 61.96f && f32.EmbedScale() < 61.97f, "embed scale: fp32 61.9677 without the bf16 flag");

  // Through the container's nesting, and a bare text_config.
  const GemmaConfig n = GemmaConfig::FromModelConfig(RealModelConfig());
  Check(n.hidden_size == 3840 && n.layer_types == c.layer_types, "FromModelConfig reads the nested text_config");
  Check(GemmaConfig::FromModelConfig(RealTextConfig()).hidden_size == 3840, "FromModelConfig accepts a bare text_config");

  // Context: default 131072, 262144 opt-in (docs/gemma4-plan.md 9.1).
  Check(c.ResolveMaxCtx(0, false) == 131072, "max ctx: default is 131072");
  Check(c.ResolveMaxCtx(8192, false) == 8192, "max ctx: smaller request honoured");
  Check(c.ResolveMaxCtx(131072, false) == 131072, "max ctx: exactly native is fine");
  Check(Throws([&] { c.ResolveMaxCtx(262144, false); }, "opt-in"), "max ctx: 262144 refused without the opt-in");
  Check(c.ResolveMaxCtx(262144, true) == 262144, "max ctx: 262144 with the opt-in");
  Check(Throws([&] { c.ResolveMaxCtx(262145, true); }, "supported maximum"), "max ctx: above 262144 refused");

  // The real file, when present: must parse and match the inlined copy field for field.
  const std::string path = r4dx::ModelsPath("Huihui-gemma-4-12B-it-abliterated/config.json");
  std::ifstream f(path);
  if (f) {
    const json real = json::parse(f);
    const GemmaConfig r = GemmaConfig::FromModelConfig(real);
    Check(r.layer_types == c.layer_types && r.hidden_size == c.hidden_size && r.sliding_window == c.sliding_window &&
              r.max_position_embeddings == c.max_position_embeddings && r.rope_theta_full == c.rope_theta_full &&
              r.final_logit_softcapping == c.final_logit_softcapping,
          "real config.json on disk agrees with the inlined copy (" + path + ")");
    Check(r.ToModelConfig().arch == Arch::kGemma4, "real config.json: arch kGemma4");
  } else {
    std::printf("SKIP: no config.json at %s\n", path.c_str());
  }
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
  for (const char* k : {"hidden_size", "num_hidden_layers", "layer_types", "num_attention_heads", "num_key_value_heads",
                        "num_global_key_value_heads", "head_dim", "global_head_dim", "sliding_window",
                        "intermediate_size", "vocab_size", "max_position_embeddings", "rms_norm_eps",
                        "tie_word_embeddings", "attention_k_eq_v", "hidden_activation", "rope_parameters"}) {
    Check(Throws([&] { GemmaConfig::FromJson(without(k)); }, k), std::string("missing '") + k + "' throws, naming it");
  }
  Check(Throws([&] { GemmaConfig::FromJson(json::array()); }, "not an object"), "non-object text_config throws");

  // The 5:1 pattern.
  json swapped = base;
  swapped["layer_types"][5] = "sliding_attention";
  swapped["layer_types"][4] = "full_attention";
  Check(Throws([&] { GemmaConfig::FromJson(swapped); }, "5 sliding : 1 full"), "layer_types: a moved full layer throws");
  json short_types = base;
  short_types["layer_types"].erase(47);
  Check(Throws([&] { GemmaConfig::FromJson(short_types); }, "does not match num_hidden_layers"),
        "layer_types: wrong length throws");
  json bad_type = base;
  bad_type["layer_types"][0] = "linear_attention";
  Check(Throws([&] { GemmaConfig::FromJson(bad_type); }, "unknown layer type"), "layer_types: unknown type throws");

  // Unsupported variants.
  Check(Throws([&] { GemmaConfig::FromJson(with("enable_moe_block", true)); }, "enable_moe_block"), "refuses MoE");
  Check(Throws([&] { GemmaConfig::FromJson(with("hidden_size_per_layer_input", 256)); }, "per-layer inputs"),
        "refuses per-layer inputs");
  Check(Throws([&] { GemmaConfig::FromJson(with("num_kv_shared_layers", 4)); }, "shared KV"), "refuses shared KV");
  Check(Throws([&] { GemmaConfig::FromJson(with("use_double_wide_mlp", true)); }, "double_wide"), "refuses double-wide MLP");
  Check(Throws([&] { GemmaConfig::FromJson(with("attention_bias", true)); }, "attention_bias"), "refuses attention bias");
  Check(Throws([&] { GemmaConfig::FromJson(with("tie_word_embeddings", false)); }, "tie_word_embeddings"),
        "refuses an untied head");
  Check(Throws([&] { GemmaConfig::FromJson(with("attention_k_eq_v", false)); }, "attention_k_eq_v"), "refuses k != v");
  Check(Throws([&] { GemmaConfig::FromJson(with("hidden_activation", "silu")); }, "hidden_activation"),
        "refuses a non-gelu-tanh activation");
  Check(Throws([&] { GemmaConfig::FromJson(with("use_bidirectional_attention", "all")); }, "use_bidirectional"),
        "refuses bidirectional 'all'");
  Check(Throws([&] { GemmaConfig::FromJson(with("model_type", "qwen3_5_text")); }, "model_type"),
        "refuses a foreign text model_type");
  json yarn = base;
  yarn["rope_parameters"]["full_attention"]["rope_type"] = "yarn";
  Check(Throws([&] { GemmaConfig::FromJson(yarn); }, "rope_type"), "refuses a non-proportional full rope");
  json odd_rot = base;
  odd_rot["rope_parameters"]["full_attention"]["partial_rotary_factor"] = 0.3;
  Check(Throws([&] { GemmaConfig::FromJson(odd_rot); }, "even integer"), "refuses a rotary width that cannot pair");
  Check(Throws([&] { GemmaConfig::FromJson(with("num_attention_heads", 12)); }, "multiple"),
        "refuses heads not a multiple of the kv heads");
}

void TestShard() {
  const GemmaConfig g = GemmaConfig::FromJson(RealTextConfig());
  for (int rank = 0; rank < 2; ++rank) {
    const GemmaConfig r = GemmaConfig::Shard(g, 2, rank);
    const std::string tag = "shard r" + std::to_string(rank) + ": ";
    Check(r.tp_world == 2 && r.tp_rank == rank && r.IsShard(), tag + "world/rank/IsShard");
    Check(r.num_attention_heads == 8, tag + "heads 16 -> 8");
    Check(r.num_kv_heads_sliding == 4 && r.Gqa(0) == 2, tag + "sliding kv 8 -> 4, GQA stays 2");
    Check(r.num_kv_heads_full == 1 && r.Gqa(5) == 8, tag + "full kv replicated (1), per-rank GQA 8");
    Check(r.intermediate_size == 7680 && r.intermediate_size % 512 == 0, tag + "intermediate 15360 -> 7680 (15 x 512)");
    Check(r.QDim(0) == 2048 && r.QDim(5) == 4096, tag + "o_proj K per rank 2048 / 4096");
    Check(r.vocab_size == 262144 && r.VocabShardSize() == 131072 && r.VocabShardBegin() == rank * 131072,
          tag + "vocab stays GLOBAL, tied head slice 131072");
    Check(r.tie_word_embeddings, tag + "tied embedding allowed");
    Check(r.hidden_size == 3840 && r.head_dim_full == 512 && r.layer_types == g.layer_types, tag + "the rest copied");
  }
  const GemmaConfig w1 = GemmaConfig::Shard(g, 1, 0);
  Check(!w1.IsShard() && w1.num_attention_heads == 16 && w1.num_kv_heads_sliding == 8 && w1.intermediate_size == 15360,
        "shard world 1 is the global config");

  Check(Throws([&] { GemmaConfig::Shard(g, 3, 0); }, "world must be 1 or 2"), "shard: world 3 refused");
  Check(Throws([&] { GemmaConfig::Shard(g, 2, 2); }, "rank must be"), "shard: rank out of range refused");
  Check(Throws([&] { GemmaConfig::Shard(GemmaConfig::Shard(g, 2, 0), 2, 1); }, "already a rank shard"),
        "shard: double sharding refused");
  GemmaConfig odd = g;
  odd.intermediate_size = 15360 + 16;  // not divisible by 2*512
  Check(Throws([&] { GemmaConfig::Shard(odd, 2, 0); }, "mlp.down"), "shard: down K % 512 enforced");
  GemmaConfig kv = g;
  kv.num_kv_heads_sliding = 3;
  kv.num_attention_heads = 18;
  Check(Throws([&] { GemmaConfig::Shard(kv, 2, 0); }, "num_kv_heads_sliding"), "shard: sliding kv must divide");
  GemmaConfig vocab = g;
  vocab.vocab_size = 262144 + 8;
  Check(Throws([&] { GemmaConfig::Shard(vocab, 2, 0); }, "vocab_size"), "shard: vocab % (16 * world) enforced");
  GemmaConfig narrow = g;
  narrow.head_dim_sliding = 128;  // 8 heads x 128 = 1024 K per rank: still % 512, but 256 would not be
  Check(!Throws([&] { GemmaConfig::Shard(narrow, 2, 0); }), "shard: 1024 rank K is legal");
  narrow.head_dim_sliding = 96;   // 8 x 96 = 768, not a multiple of 512
  Check(Throws([&] { GemmaConfig::Shard(narrow, 2, 0); }, "attn.o sliding"), "shard: o_proj rank K % 512 enforced");
}

void TestArchOnConfig() {
  const GemmaConfig g = GemmaConfig::FromJson(RealTextConfig());
  const ModelConfig m = g.ToModelConfig();
  Check(m.arch == Arch::kGemma4, "ToModelConfig: arch kGemma4 (TextModel::Config().arch)");
  Check(m.hidden_size == 3840 && m.num_hidden_layers == 48 && m.vocab_size == 262144 && m.tie_word_embeddings,
        "ToModelConfig: the fields the CLI and server read");
  Check(m.layer_types.size() == 48 && !m.IsGdnLayer(0) && !m.IsGdnLayer(5), "ToModelConfig: no GDN layers");
  Check(m.linear_num_key_heads == 0 && m.linear_num_value_heads == 0 && m.mtp_num_hidden_layers == 0 &&
            !m.attn_output_gate,
        "ToModelConfig: GDN / MTP / output-gate fields are 0");
  Check(m.num_attention_heads == 16 && m.num_key_value_heads == 8 && m.head_dim == 256, "ToModelConfig: sliding geometry");
  Check(m.global_head_dim == 512 && ModelConfig{}.global_head_dim == 0,
        "ToModelConfig: global_head_dim 512 (tp::RuleFor's rotation.had_o_full_signs K = heads x it); Qwen's is 0");
  const ModelConfig r = GemmaConfig::Shard(g, 2, 1).ToModelConfig();
  Check(r.tp_world == 2 && r.tp_rank == 1 && r.arch == Arch::kGemma4, "ToModelConfig of a shard carries tp_world/rank");

  // Qwen is untouched: a default / parsed ModelConfig stays kQwen35.
  Check(ModelConfig{}.arch == Arch::kQwen35, "ModelConfig default arch is kQwen35");
  json q;
  q["hidden_size"] = 512;
  q["num_hidden_layers"] = 1;
  q["layer_types"] = {"full_attention"};
  q["num_attention_heads"] = 4;
  q["num_key_value_heads"] = 2;
  q["head_dim"] = 128;
  q["intermediate_size"] = 1024;
  q["linear_num_key_heads"] = 2;
  q["linear_num_value_heads"] = 4;
  q["linear_key_head_dim"] = 64;
  q["linear_value_head_dim"] = 64;
  q["rms_norm_eps"] = 1e-6;
  q["vocab_size"] = 1024;
  Check(ModelConfig::FromJson(q).arch == Arch::kQwen35, "ModelConfig::FromJson (Qwen) keeps kQwen35");
  Check(std::string(r4dx::model::ArchName(Arch::kGemma4)) == "gemma4_unified" &&
            std::string(r4dx::model::ArchName(Arch::kQwen35)) == "qwen3_5",
        "ArchName spellings");
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
  using r4dx::model::DetectArch;
  using r4dx::model::DetectArchFromMetadata;

  Check(DetectArchFromMetadata(json{{"model_arch", "gemma4_unified"}}) == Arch::kGemma4, "metadata: model_arch gemma4_unified");
  Check(DetectArchFromMetadata(json{{"model_arch", "qwen3_5"}}) == Arch::kQwen35, "metadata: model_arch qwen3_5");
  Check(Throws([&] { DetectArchFromMetadata(json{{"model_arch", "llama"}}); }, "unknown __metadata__.model_arch"),
        "metadata: unknown model_arch throws, not Qwen");
  Check(DetectArchFromMetadata(json{{"model_config", RealModelConfig()}}) == Arch::kGemma4,
        "metadata: model_config.model_type gemma4_unified, no model_arch key");
  Check(DetectArchFromMetadata(json{{"model_config", {{"architectures", {"Gemma4UnifiedForConditionalGeneration"}}}}}) ==
            Arch::kGemma4,
        "metadata: model_config.architectures names Gemma 4");
  Check(DetectArchFromMetadata(json{{"model_config", {{"model_type", "qwen3_5"}, {"text_config", json::object()}}}}) ==
            Arch::kQwen35,
        "metadata: a Qwen model_config is kQwen35");
  Check(DetectArchFromMetadata(json::object()) == Arch::kQwen35, "metadata: empty (selftest container) is kQwen35");
  Check(DetectArchFromMetadata(json::array()) == Arch::kQwen35, "metadata: non-object is kQwen35");

  json gemma_hdr = {{"__metadata__", {{"model_arch", "gemma4_unified"}, {"model_config", RealModelConfig()}}},
                    {"text.final_norm", {{"dtype", "U8"}, {"shape", {2}}, {"data_offsets", {0, 2}}}}};
  Check(DetectArch(WriteShell("r4dx_arch_gemma.r4dx", gemma_hdr)) == Arch::kGemma4, "file: Gemma container header");
  json qwen_hdr = {{"__metadata__", {{"model_config", {{"model_type", "qwen3_5"}, {"text_config", json::object()}}}}}};
  Check(DetectArch(WriteShell("r4dx_arch_qwen.r4dx", qwen_hdr)) == Arch::kQwen35, "file: Qwen container header");
  Check(DetectArch(WriteShell("r4dx_arch_nometa.r4dx", json::object())) == Arch::kQwen35, "file: no __metadata__ is kQwen35");
  Check(DetectArch((std::filesystem::temp_directory_path() / "r4dx_arch_missing_file.r4dx").string()) == Arch::kQwen35,
        "file: missing file is kQwen35 (the Qwen loader reports it)");
  const std::filesystem::path junk = std::filesystem::temp_directory_path() / "r4dx_arch_junk.r4dx";
  {
    std::ofstream f(junk, std::ios::binary | std::ios::trunc);
    f << "not a safetensors file at all, just text";
  }
  Check(DetectArch(junk.string()) == Arch::kQwen35, "file: junk (absurd header length) is kQwen35");
}

// ResolveContainerMaxCtx (task M1-14): the max_ctx r4dx-cli / r4dx-server hand ModelOptions. Qwen
// containers are untouched; a Gemma container defaults to 131072, with 262144 an explicit opt-in.
void TestResolveContainerMaxCtx() {
  using r4dx::model::ReadContainerMetadata;
  using r4dx::model::ResolveContainerMaxCtx;
  json gemma_hdr = {{"__metadata__", {{"model_arch", "gemma4_unified"}, {"model_config", RealModelConfig()}}},
                    {"text.final_norm", {{"dtype", "U8"}, {"shape", {2}}, {"data_offsets", {0, 2}}}}};
  const std::string g = WriteShell("r4dx_ctx_gemma.r4dx", gemma_hdr);
  json qwen_hdr = {{"__metadata__", {{"model_config", {{"model_type", "qwen3_5"}, {"text_config", json::object()}}}}}};
  const std::string q = WriteShell("r4dx_ctx_qwen.r4dx", qwen_hdr);

  Check(ReadContainerMetadata(g).is_object() && ReadContainerMetadata(g).contains("model_config"),
        "ReadContainerMetadata returns __metadata__");
  Check(ReadContainerMetadata((std::filesystem::temp_directory_path() / "r4dx_ctx_missing.r4dx").string()).is_null(),
        "ReadContainerMetadata: a missing file is null");

  // Qwen: whatever the caller passed, unchanged (262144 default, an explicit value, no opt-in needed).
  Check(ResolveContainerMaxCtx(q, Arch::kQwen35, false, 262144, false) == 262144, "qwen: default untouched");
  Check(ResolveContainerMaxCtx(q, Arch::kQwen35, true, 4096, false) == 4096, "qwen: explicit untouched");
  Check(ResolveContainerMaxCtx("no/such/file.r4dx", Arch::kQwen35, false, 262144, true) == 262144,
        "qwen: never reads the file");

  // Gemma.
  Check(ResolveContainerMaxCtx(g, Arch::kGemma4, false, 262144, false) == 131072,
        "gemma: no --max-ctx -> the checkpoint's 131072 (the CLI/server default 262144 is NOT used)");
  Check(ResolveContainerMaxCtx(g, Arch::kGemma4, true, 8192, false) == 8192, "gemma: explicit smaller value");
  Check(ResolveContainerMaxCtx(g, Arch::kGemma4, false, 262144, true) == 262144, "gemma: --extended-ctx alone -> 262144");
  Check(ResolveContainerMaxCtx(g, Arch::kGemma4, true, 200000, true) == 200000, "gemma: explicit value above 131072 with the opt-in");
  Check(Throws([&] { ResolveContainerMaxCtx(g, Arch::kGemma4, true, 262144, false); }, "opt-in"),
        "gemma: 262144 without the opt-in is refused");
  Check(Throws([&] { ResolveContainerMaxCtx(g, Arch::kGemma4, true, 300000, true); }, "supported maximum"),
        "gemma: above 262144 is refused even with the opt-in");
  Check(Throws([&] { ResolveContainerMaxCtx(WriteShell("r4dx_ctx_nocfg.r4dx", json::object()), Arch::kGemma4, false, 0, false); },
               "model_config"),
        "gemma: a container without model_config is refused");
}

}  // namespace

int main() {
  TestResolveContainerMaxCtx();
  TestRealConfig();
  TestRefusals();
  TestShard();
  TestArchOnConfig();
  TestDetectArch();
  if (g_failures != 0) {
    std::fprintf(stderr, "%d check(s) FAILED\n", g_failures);
    return 1;
  }
  std::printf("test_gemma_config: all checks passed\n");
  return 0;
}
