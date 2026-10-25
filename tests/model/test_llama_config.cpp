// tests/model/test_llama_config.cpp -- CPU-only (no GPU, no container, no HIP). docs/nanbeige-plan.md N1-1 / N1-2.
//
//   * LlamaConfig::FromJson on the real Nanbeige/Nanbeige4.3-33B config.json (values inlined so the test is hermetic):
//     every field, the derived shapes, both rope spellings, and every refusal;
//   * LlamaConfig::Shard at TP=2: the per-rank shapes (heads 28 / kv 4, GQA 7, FFN 11008, vocab 83456) and the
//     K % 256 / rows % 16 legality, plus its refusals;
//   * ToModelConfig carries arch kLlama;
//   * DetectArchFromMetadata / DetectArch / ResolveContainerMaxCtx for llama, and the other arches unchanged.
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "arch.h"
#include "llama_config.h"
#include "model_config.h"
#include "nlohmann/json.hpp"

using nlohmann::json;
using r4dx::model::Arch;
using r4dx::model::LlamaConfig;
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

// Nanbeige/Nanbeige4.3-33B config.json (read from E:\models\Nanbeige\Nanbeige4.3-33B, 2026-10-25).
json RealConfig() {
  return {{"architectures", {"LlamaForCausalLM"}},
          {"attention_bias", false},
          {"attention_dropout", 0.0},
          {"bos_token_id", 166100},
          {"embd_pdrop", 0.0},
          {"eos_token_id", 166101},
          {"ffn_hidden_size", 22016},
          {"head_dim", 128},
          {"hidden_act", "silu"},
          {"hidden_size", 7168},
          {"initializer_range", 0.02},
          {"intermediate_size", 22016},
          {"kv_channels", 128},
          {"max_position_embeddings", 524288},
          {"mlp_bias", false},
          {"model_type", "llama"},
          {"num_attention_heads", 56},
          {"num_hidden_layers", 52},
          {"num_key_value_heads", 8},
          {"num_query_groups", 8},
          {"pad_token_id", 0},
          {"pretraining_tp", 1},
          {"resid_pdrop", 0.0},
          {"rms_norm_eps", 1e-05},
          {"rope_scaling", nullptr},
          {"rope_theta", 120000000},
          {"rotary_base", 120000000},
          {"seq_length", 524288},
          {"tie_word_embeddings", false},
          {"torch_dtype", "bfloat16"},
          {"transformers_version", "4.51.1"},
          {"use_cache", false},
          {"vocab_size", 166912}};
}

void TestRealConfig() {
  const LlamaConfig c = LlamaConfig::FromJson(RealConfig());
  Check(c.hidden_size == 7168 && c.num_hidden_layers == 52 && c.num_attention_heads == 56 && c.num_key_value_heads == 8 &&
            c.head_dim == 128 && c.Gqa() == 7,
        "geometry: 52 layers, hidden 7168, 56 q heads / 8 kv heads (GQA 7), head_dim 128");
  Check(c.intermediate_size == 22016 && c.GateUpRows() == 44032, "MLP: I 22016, gate_up N 44032");
  Check(c.vocab_size == 166912 && c.rms_norm_eps == 1e-5 && c.rope_theta == 1.2e8 && !c.tie_word_embeddings &&
            c.max_position_embeddings == 524288 && c.bos_token_id == 166100 && c.eos_token_id == 166101 && c.pad_token_id == 0,
        "vocab 166912, eps 1e-5, rope theta 1.2e8 (an integer in the json), untied head, max_pos 524288, ids 166100 / 166101 / 0");
  Check(c.QDim() == 7168 && c.KvDim() == 1024, "QDim 7168, KvDim 1024");
  const std::vector<std::string> types = c.LayerTypes();
  bool all_full = types.size() == 52;
  for (const auto& t : types) all_full = all_full && t == "full_attention";
  Check(all_full, "LayerTypes: 52 x full_attention");

  const ModelConfig m = c.ToModelConfig();
  Check(m.arch == Arch::kLlama && m.hidden_size == 7168 && m.num_hidden_layers == 52 && m.layer_types.size() == 52 &&
            m.num_attention_heads == 56 && m.num_key_value_heads == 8 && m.head_dim == 128 && m.intermediate_size == 22016 &&
            m.num_experts == 0 && m.moe_intermediate_size == 0 && m.vocab_size == 166912 && !m.tie_word_embeddings &&
            !m.attn_output_gate && m.partial_rotary_factor == 1.0 && m.linear_num_key_heads == 0 && m.mtp_num_hidden_layers == 0 &&
            m.tp_world == 1,
        "ToModelConfig: arch kLlama, dense geometry, GDN / MoE / MTP fields 0");
  Check(LlamaConfig::FromModelConfig(RealConfig()).hidden_size == 7168 &&
            LlamaConfig::FromModelConfig(json{{"text_config", RealConfig()}}).num_hidden_layers == 52,
        "FromModelConfig: flat config.json (and a nested text_config)");

  Check(c.ResolveMaxCtx(0) == 32768 && c.ResolveMaxCtx(-1) == 32768 && c.ResolveMaxCtx(8192) == 8192 &&
            c.ResolveMaxCtx(131072) == 131072 && c.ResolveMaxCtx(524288) == 524288,
        "ResolveMaxCtx: default 32768 (KV-bound), any explicit value up to 524288");
  Check(Throws([&] { c.ResolveMaxCtx(524289); }, "no scaling"), "ResolveMaxCtx: above 524288 throws");
  {
    json small = RealConfig();
    small["max_position_embeddings"] = 4096;
    Check(LlamaConfig::FromJson(small).ResolveMaxCtx(0) == 4096, "ResolveMaxCtx: the default never exceeds max_position_embeddings");
  }
}

void TestRopeSpellings() {
  json t5 = RealConfig();
  t5.erase("rope_theta");
  t5.erase("rope_scaling");
  t5["rope_parameters"] = {{"rope_type", "default"}, {"rope_theta", 5.0e5}};
  Check(LlamaConfig::FromJson(t5).rope_theta == 5.0e5, "rope: transformers 5 rope_parameters {default, theta} is read");
  t5["rope_parameters"]["rope_type"] = "llama3";
  Check(Throws([&] { (void)LlamaConfig::FromJson(t5); }, "rope_type"), "rope: a llama3-scaled rope_parameters is refused");
}

void TestRefusals() {
  auto bad = [&](const char* key, json value, const std::string& needle, const std::string& what) {
    json c = RealConfig();
    c[key] = std::move(value);
    Check(Throws([&] { (void)LlamaConfig::FromJson(c); }, needle), what);
  };
  auto missing = [&](const char* key) {
    json c = RealConfig();
    c.erase(key);
    Check(Throws([&] { (void)LlamaConfig::FromJson(c); }, key), std::string("a missing '") + key + "' names the key");
  };
  bad("model_type", "mistral", "model_type", "refuses another model_type");
  bad("tie_word_embeddings", true, "tie_word_embeddings", "refuses a tied head");
  bad("attention_bias", true, "attention_bias", "refuses attention_bias");
  bad("mlp_bias", true, "mlp_bias", "refuses mlp_bias");
  bad("hidden_act", "gelu", "hidden_act", "refuses an activation other than silu");
  bad("pretraining_tp", 2, "pretraining_tp", "refuses pretraining_tp 2");
  bad("rope_scaling", json{{"rope_type", "linear"}, {"factor", 2.0}}, "rope_scaling", "refuses rope_scaling");
  bad("num_key_value_heads", 9, "multiple", "refuses heads not a multiple of kv heads");
  bad("head_dim", 127, "even", "refuses an odd head_dim");
  {
    json c = RealConfig();
    c.erase("head_dim");
    Check(LlamaConfig::FromJson(c).head_dim == 128, "no head_dim key: hidden / heads = 128");
  }
  for (const char* k : {"hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "intermediate_size",
                        "vocab_size", "max_position_embeddings", "rms_norm_eps", "rope_theta"})
    missing(k);
  Check(Throws([&] { (void)LlamaConfig::FromJson(json::array()); }, "not an object"), "refuses a non-object config");
}

void TestShard() {
  const LlamaConfig g = LlamaConfig::FromJson(RealConfig());
  for (int rank = 0; rank < 2; ++rank) {
    const LlamaConfig r = LlamaConfig::Shard(g, 2, rank);
    Check(r.tp_world == 2 && r.tp_rank == rank && r.num_attention_heads == 28 && r.num_key_value_heads == 4 && r.Gqa() == 7 &&
              r.intermediate_size == 11008 && r.hidden_size == 7168 && r.vocab_size == 166912 && r.VocabShardSize() == 83456 &&
              r.VocabShardBegin() == rank * 83456 && r.IsShard(),
          "Shard(2, " + std::to_string(rank) + "): heads 28 / kv 4 (GQA 7), FFN 11008, vocab slice 83456");
    // The per-rank K the kernels see: attn.o 3584 = 14 x 256, down 11008 = 43 x 256; trellis needs whole 128-blocks too.
    Check(r.QDim() == 3584 && r.KvDim() == 512 && r.QDim() % 256 == 0 && r.intermediate_size % 256 == 0 && r.QDim() % 128 == 0 &&
              r.KvDim() % 128 == 0 && r.intermediate_size % 128 == 0 && r.VocabShardSize() % 128 == 0,
          "Shard(2, " + std::to_string(rank) + "): row-parallel K % 256, every rank shape whole 128-blocks");
    const ModelConfig m = r.ToModelConfig();
    Check(m.tp_world == 2 && m.tp_rank == rank && m.num_attention_heads == 28 && m.intermediate_size == 11008,
          "Shard: ToModelConfig carries the rank");
  }
  {
    // tools/convert_ref/make_tiny_llama.py's config: the real attention geometry (head_dim 128, GQA 7) at hidden 512,
    // which is NOT heads x head_dim (3584), so the explicit head_dim key matters.
    json t = RealConfig();
    t["hidden_size"] = 512;
    t["num_attention_heads"] = 28;
    t["num_key_value_heads"] = 4;
    t["intermediate_size"] = 512;
    t["vocab_size"] = 512;
    t["num_hidden_layers"] = 3;
    t["max_position_embeddings"] = 4096;
    const LlamaConfig tiny = LlamaConfig::FromJson(t);
    const LlamaConfig r = LlamaConfig::Shard(tiny, 2, 1);
    Check(tiny.head_dim == 128 && tiny.QDim() == 3584 && r.num_attention_heads == 14 && r.num_key_value_heads == 2 && r.Gqa() == 7 &&
              r.QDim() == 1792 && r.intermediate_size == 256 && r.VocabShardSize() == 256,
          "tiny fixture: head_dim 128 != hidden / heads, GQA 7, and TP=2-legal (o_proj K 1792 = 7 x 256)");
  }
  const LlamaConfig w1 = LlamaConfig::Shard(g, 1, 0);
  Check(w1.num_attention_heads == 56 && w1.intermediate_size == 22016 && !w1.IsShard(), "Shard(1, 0) is the global shape (and legal)");
  Check(Throws([&] { (void)LlamaConfig::Shard(g, 3, 0); }, "world"), "Shard: world 3 is refused");
  Check(Throws([&] { (void)LlamaConfig::Shard(g, 2, 2); }, "rank"), "Shard: rank 2 of 2 is refused");
  Check(Throws([&] { (void)LlamaConfig::Shard(LlamaConfig::Shard(g, 2, 0), 2, 0); }, "already a rank shard"),
        "Shard: a shard of a shard is refused");
  {
    LlamaConfig bad = g;
    bad.intermediate_size = 22016 + 256;  // /2 = 11136 = 43.5 x 256
    Check(Throws([&] { (void)LlamaConfig::Shard(bad, 2, 0); }, "mlp.down"), "Shard: an FFN width with K % 256 != 0 per rank names the K");
    bad = g;
    bad.num_key_value_heads = 1;
    bad.num_attention_heads = 56;
    Check(Throws([&] { (void)LlamaConfig::Shard(bad, 2, 0); }, "num_key_value_heads"), "Shard: 1 kv head cannot split 2 ways");
    bad = g;
    bad.vocab_size = 166913;
    Check(Throws([&] { (void)LlamaConfig::Shard(bad, 2, 0); }, "vocab_size"), "Shard: a vocab that does not split into 16-row tiles is refused");
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

  Check(std::string(r4dx::model::ArchName(Arch::kLlama)) == "llama", "ArchName(kLlama) == llama");
  Check(DetectArchFromMetadata(json{{"model_arch", "llama"}}) == Arch::kLlama, "metadata: model_arch llama");
  Check(DetectArchFromMetadata(json{{"model_config", RealConfig()}}) == Arch::kLlama, "metadata: model_config.model_type llama, no model_arch key");
  Check(DetectArchFromMetadata(json{{"model_config", {{"architectures", {"LlamaForCausalLM"}}}}}) == Arch::kLlama,
        "metadata: model_config.architectures names LlamaForCausalLM");
  Check(Throws([&] { DetectArchFromMetadata(json{{"model_arch", "mistral"}}); }, "'llama'"),
        "metadata: an unknown model_arch throws, naming llama among the known");
  Check(DetectArchFromMetadata(json{{"model_arch", "qwen3_5"}}) == Arch::kQwen35 &&
            DetectArchFromMetadata(json{{"model_arch", "lfm2_moe"}}) == Arch::kLfm2Moe &&
            DetectArchFromMetadata(json{{"model_arch", "qwen3_5_moe"}}) == Arch::kQwen35Moe &&
            DetectArchFromMetadata(json{{"model_config", {{"model_type", "qwen3_5"}, {"text_config", json::object()}}}}) == Arch::kQwen35 &&
            DetectArchFromMetadata(json::object()) == Arch::kQwen35,
        "metadata: the other arches (and no metadata at all) are unchanged");

  json hdr = {{"__metadata__", {{"model_arch", "llama"}, {"model_config", RealConfig()}}},
              {"text.final_norm", {{"dtype", "U8"}, {"shape", {2}}, {"data_offsets", {0, 2}}}}};
  const std::string f = WriteShell("r4dx_llama_arch.r4dx", hdr);
  Check(DetectArch(f) == Arch::kLlama, "file: llama container header");
  Check(ResolveContainerMaxCtx(f, Arch::kLlama, false, 262144, false) == 32768, "llama: no --max-ctx -> 32768 (not the CLI's 262144)");
  Check(ResolveContainerMaxCtx(f, Arch::kLlama, false, 262144, true) == 32768, "llama: --extended-ctx alone does not change the default");
  Check(ResolveContainerMaxCtx(f, Arch::kLlama, true, 65536, false) == 65536, "llama: an explicit value");
  Check(Throws([&] { ResolveContainerMaxCtx(f, Arch::kLlama, true, 1 << 20, true); }, "no scaling"),
        "llama: an explicit value above max_position_embeddings is refused");
  Check(Throws([&] { ResolveContainerMaxCtx(WriteShell("r4dx_llama_nocfg.r4dx", json::object()), Arch::kLlama, false, 0, false); },
               "model_config"),
        "llama: a container without model_config is refused");
  Check(ResolveContainerMaxCtx("no/such/file.r4dx", Arch::kQwen35, false, 262144, true) == 262144, "qwen: still never reads the file");
}

}  // namespace

int main() {
  TestRealConfig();
  TestRopeSpellings();
  TestRefusals();
  TestShard();
  TestArch();
  std::printf("\n%s: %d failure(s)\n", g_failures == 0 ? "OK" : "FAILED", g_failures);
  return g_failures == 0 ? 0 : 1;
}
