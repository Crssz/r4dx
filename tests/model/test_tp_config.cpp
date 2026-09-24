// tests/model/test_tp_config.cpp -- CPU-only (no GPU, no container). docs/tp.md 3.1 / 10.1.
//
// ModelConfig::Shard, the rank-local config of tensor parallelism:
//   * the real Qwen3.8-27B text_config (the values of C:\AI\models\Qwen3.8-27B\config.json, inlined
//     here so the test is hermetic) at TP=2: every field of docs/tp.md 3.1's table, both ranks;
//   * a synthetic small config (the one test_tp_shard.cpp slices), both ranks, plus world 1;
//   * every validation rule throws std::invalid_argument naming the failing field;
//   * VocabShardSize / VocabShardBegin / IsShard.
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "model_config.h"
#include "nlohmann/json.hpp"

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

// The real checkpoint's text_config, the fields ModelConfig::FromJson reads.
ModelConfig RealConfig() {
  nlohmann::json t;
  t["hidden_size"] = 5120;
  t["num_hidden_layers"] = 64;
  nlohmann::json types = nlohmann::json::array();
  for (int i = 0; i < 64; ++i) types.push_back(i % 4 == 3 ? "full_attention" : "linear_attention");
  t["layer_types"] = types;
  t["num_attention_heads"] = 24;
  t["num_key_value_heads"] = 4;
  t["head_dim"] = 256;
  t["attn_output_gate"] = true;
  t["intermediate_size"] = 17408;
  t["rope_parameters"] = {{"mrope_interleaved", true},
                          {"mrope_section", {11, 11, 10}},
                          {"partial_rotary_factor", 0.25},
                          {"rope_theta", 10000000}};
  t["partial_rotary_factor"] = 0.25;
  t["linear_conv_kernel_dim"] = 4;
  t["linear_key_head_dim"] = 128;
  t["linear_num_key_heads"] = 16;
  t["linear_num_value_heads"] = 48;
  t["linear_value_head_dim"] = 128;
  t["rms_norm_eps"] = 1e-6;
  t["vocab_size"] = 248320;
  t["tie_word_embeddings"] = false;
  t["mtp_num_hidden_layers"] = 1;
  return ModelConfig::FromJson(t);
}

// The synthetic config test_tp_shard.cpp uses: every TP divisibility rule holds with the smallest
// numbers that keep the real model's structure (gqa 2 attention / 4 GDN, 512-multiples of rank K).
ModelConfig SmallConfig() {
  ModelConfig c;
  c.hidden_size = 512;
  c.num_hidden_layers = 4;
  c.layer_types = {"linear_attention", "linear_attention", "linear_attention", "full_attention"};
  c.num_attention_heads = 4;
  c.num_key_value_heads = 2;
  c.head_dim = 256;
  c.intermediate_size = 1024;
  c.linear_num_key_heads = 2;
  c.linear_num_value_heads = 8;
  c.linear_key_head_dim = 128;
  c.linear_value_head_dim = 128;
  c.vocab_size = 1024;
  return c;
}

// Every field that must be copied unchanged from the global config.
bool SameGlobalFields(const ModelConfig& a, const ModelConfig& g) {
  return a.hidden_size == g.hidden_size && a.num_hidden_layers == g.num_hidden_layers &&
         a.layer_types == g.layer_types && a.head_dim == g.head_dim &&
         a.attn_output_gate == g.attn_output_gate &&
         a.partial_rotary_factor == g.partial_rotary_factor && a.rope_theta == g.rope_theta &&
         a.mrope_interleaved == g.mrope_interleaved && a.mrope_section == g.mrope_section &&
         a.linear_key_head_dim == g.linear_key_head_dim &&
         a.linear_value_head_dim == g.linear_value_head_dim &&
         a.linear_conv_kernel_dim == g.linear_conv_kernel_dim && a.rms_norm_eps == g.rms_norm_eps &&
         a.vocab_size == g.vocab_size && a.tie_word_embeddings == g.tie_word_embeddings &&
         a.mtp_num_hidden_layers == g.mtp_num_hidden_layers;
}

void TestRealConfig() {
  const ModelConfig g = RealConfig();
  Check(g.tp_world == 1 && g.tp_rank == 0 && !g.IsShard(), "real: FromJson yields a global config");
  Check(g.VocabShardSize() == 248320 && g.VocabShardBegin() == 0,
        "real: global VocabShardSize/Begin cover the whole vocabulary");
  Check(g.KeyDim() == 2048 && g.ValueDim() == 6144 && g.ConvDim() == 10240,
        "real: global KeyDim/ValueDim/ConvDim 2048/6144/10240");

  for (int r = 0; r < 2; ++r) {
    const ModelConfig s = ModelConfig::Shard(g, 2, r);
    const std::string p = "real rank " + std::to_string(r) + ": ";
    Check(s.tp_world == 2 && s.tp_rank == r && s.IsShard(), p + "tp_world 2, tp_rank r, IsShard");
    Check(s.hidden_size == 5120 && s.num_hidden_layers == 64 && s.layer_types == g.layer_types,
          p + "hidden_size 5120, 64 layers, layer_types global");
    Check(s.num_attention_heads == 12, p + "num_attention_heads 24 -> 12");
    Check(s.num_key_value_heads == 2, p + "num_key_value_heads 4 -> 2");
    Check(s.head_dim == 256, p + "head_dim 256 global");
    Check(s.intermediate_size == 8704, p + "intermediate_size 17408 -> 8704");
    Check(s.linear_num_key_heads == 8, p + "linear_num_key_heads 16 -> 8");
    Check(s.linear_num_value_heads == 24, p + "linear_num_value_heads 48 -> 24");
    Check(s.linear_key_head_dim == 128 && s.linear_value_head_dim == 128 &&
              s.linear_conv_kernel_dim == 4,
          p + "linear head dims 128/128, conv kernel 4 global");
    Check(s.vocab_size == 248320, p + "vocab_size stays GLOBAL (248320)");
    Check(s.VocabShardSize() == 124160 && s.VocabShardBegin() == 124160 * r,
          p + "VocabShardSize 124160, VocabShardBegin r * 124160");
    Check(s.KeyDim() == 1024 && s.ValueDim() == 3072 && s.ConvDim() == 5120,
          p + "derived KeyDim/ValueDim/ConvDim 1024/3072/5120");
    Check(s.AttnGqa() == 6 && s.GqaRepeats() == 3 && s.RotaryDim() == 64,
          p + "AttnGqa 6, GqaRepeats 3, RotaryDim 64 invariant");
    Check(SameGlobalFields(s, g), p + "every other field (rope, mrope, eps, tie, mtp) copied");
    int t = 0, h = 0, w = 0;
    s.MropeSections(&t, &h, &w);
    Check(t == 11 && h == 11 && w == 10, p + "mrope sections still 11/11/10");
  }

  const ModelConfig one = ModelConfig::Shard(g, 1, 0);
  Check(one.tp_world == 1 && one.tp_rank == 0 && !one.IsShard() &&
            one.num_attention_heads == 24 && one.num_key_value_heads == 4 &&
            one.intermediate_size == 17408 && one.linear_num_key_heads == 16 &&
            one.linear_num_value_heads == 48 && SameGlobalFields(one, g) &&
            one.VocabShardSize() == 248320,
        "real: Shard(world 1) is the identity");
}

void TestSmallConfig() {
  const ModelConfig g = SmallConfig();
  for (int r = 0; r < 2; ++r) {
    const ModelConfig s = ModelConfig::Shard(g, 2, r);
    const std::string p = "small rank " + std::to_string(r) + ": ";
    Check(s.num_attention_heads == 2 && s.num_key_value_heads == 1 && s.intermediate_size == 512 &&
              s.linear_num_key_heads == 1 && s.linear_num_value_heads == 4,
          p + "heads 4/2 -> 2/1, GDN heads 2/8 -> 1/4, intermediate 1024 -> 512");
    Check(s.KeyDim() == 128 && s.ValueDim() == 512 && s.ConvDim() == 768,
          p + "KeyDim/ValueDim/ConvDim 128/512/768");
    Check(s.AttnGqa() == 2 && s.GqaRepeats() == 4, p + "AttnGqa 2 and GqaRepeats 4 invariant");
    Check(s.VocabShardSize() == 512 && s.VocabShardBegin() == 512 * r && s.vocab_size == 1024,
          p + "vocab 1024 global, shard 512 at r * 512");
    Check(SameGlobalFields(s, g), p + "every other field copied");
  }
}

// `mutate` breaks exactly one rule of the (otherwise valid) small config; Shard must throw
// std::invalid_argument whose message names `field`.
template <class Mutate>
void ExpectThrowNaming(const char* what, const std::string& field, Mutate mutate, int world = 2,
                       int rank = 0) {
  ModelConfig g = SmallConfig();
  mutate(g);
  std::string msg;
  bool threw = false;
  try {
    ModelConfig::Shard(g, world, rank);
  } catch (const std::invalid_argument& e) {
    threw = true;
    msg = e.what();
  }
  Check(threw && msg.find(field) != std::string::npos,
        std::string("throws naming '") + field + "' when " + what +
            (threw ? " [" + msg + "]" : ""));
}

void TestValidation() {
  // (Every message starts "ModelConfig::Shard(world=.., rank=..): ", so these two look for the
  // rule's own wording.)
  ExpectThrowNaming("world is 3", "world must be", [](ModelConfig&) {}, 3, 0);
  ExpectThrowNaming("world is 0", "world must be", [](ModelConfig&) {}, 0, 0);
  ExpectThrowNaming("rank is 2 at world 2", "rank must be", [](ModelConfig&) {}, 2, 2);
  ExpectThrowNaming("rank is -1", "rank must be", [](ModelConfig&) {}, 2, -1);
  ExpectThrowNaming("the config is already a shard", "tp_world",
                    [](ModelConfig& c) { c = ModelConfig::Shard(c, 2, 0); });
  ExpectThrowNaming("tie_word_embeddings is set", "tie_word_embeddings",
                    [](ModelConfig& c) { c.tie_word_embeddings = true; });
  // Head counts: odd counts, with the other quantities rescaled so nothing else fails first.
  ExpectThrowNaming("num_attention_heads is odd", "num_attention_heads", [](ModelConfig& c) {
    c.num_attention_heads = 3;
    c.num_key_value_heads = 1;
  });
  ExpectThrowNaming("num_key_value_heads is odd", "num_key_value_heads",
                    [](ModelConfig& c) { c.num_key_value_heads = 1; });
  ExpectThrowNaming("linear_num_key_heads is odd", "linear_num_key_heads",
                    [](ModelConfig& c) { c.linear_num_key_heads = 1; });
  ExpectThrowNaming("linear_num_value_heads is odd", "linear_num_value_heads",
                    [](ModelConfig& c) { c.linear_num_value_heads = 7; });
  ExpectThrowNaming("intermediate_size is odd", "intermediate_size",
                    [](ModelConfig& c) { c.intermediate_size = 1023; });
  // Row-parallel rank K % 512.
  ExpectThrowNaming("attn.o's rank K is 256", "attn.o", [](ModelConfig& c) { c.head_dim = 128; });
  ExpectThrowNaming("gdn.out_proj's rank K is 256", "gdn.out_proj",
                    [](ModelConfig& c) { c.linear_value_head_dim = 64; });
  ExpectThrowNaming("mlp.down's rank K is 768 (not a multiple of 512)", "mlp.down",
                    [](ModelConfig& c) { c.intermediate_size = 1536; });
  // Column-parallel rank segments % 16.
  ExpectThrowNaming("the q/k rank segment is 8 rows", "gdn.in_proj_qkv q/k",
                    [](ModelConfig& c) { c.linear_key_head_dim = 8; });
  ExpectThrowNaming("the attn.k/v rank rows are 8", "attn.k/v", [](ModelConfig& c) {
    c.num_attention_heads = 128;  // keeps attn.o's rank K = 64 * 8 = 512
    c.num_key_value_heads = 2;
    c.head_dim = 8;
  });
  // Vocabulary.
  ExpectThrowNaming("vocab_size is not a multiple of 32", "vocab_size",
                    [](ModelConfig& c) { c.vocab_size = 1040; });

  // Shard checks every rule at every world: attn.o's K is 4 * 64 = 256 at world 1.
  bool threw = false;
  try {
    ModelConfig c = SmallConfig();
    c.head_dim = 64;
    ModelConfig::Shard(c, 1, 0);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  Check(threw, "world 1 applies the same shape rules (attn.o K 256 with head_dim 64)");
}

}  // namespace

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  TestRealConfig();
  TestSmallConfig();
  TestValidation();
  if (g_failures > 0) {
    std::fprintf(stderr, "%d check(s) FAILED\n", g_failures);
    return 1;
  }
  std::printf("ALL PASS\n");
  return 0;
}
