// tests/convert/test_lfm2_layout.cpp -- CPU-only (ctest `convert_lfm2_layout`). docs/lfm2-plan.md L1-2.
//
// (a) lfm2_layout.cpp over the REAL-geometry header (40 layers, 64 experts, the 7610 tensor names and shapes of
//     LiquidAI/LFM2.5-24B-A2B generated here from the config) through a recording Kit, no weights: every tensor is
//     consumed, the container names / fused source lists / shapes are the plan's section 4 table (per-expert linears,
//     the squeezed conv weight, the widened expert_bias, the tied lm_head), --layers allow-lists later layers, the
//     coverage audit names a stray tensor, every structural refusal fires. When the real checkpoint is on disk
//     (R4DX_MODELS_ROOT\LiquidAI\LFM2.5-24B-A2B, header bytes only) its tensor set and shapes are compared to the
//     generated header too (SKIPPED when absent).
// (b) r4dx-convert (when built) on a tiny synthetic lfm2_moe checkpoint -- conv + dense, conv + dense, attention + MoE,
//     conv + MoE, the real tensor names, per-expert experts.{e}.{w1,w2,w3}, F32 expert_bias, no lm_head, flat
//     config.json, single-file AND sharded (index) forms: the exact container tensor set, shapes and dtypes, bf16 byte
//     identity (embed, tied lm_head, w1|w3 fusion, squeezed conv weight, router), the fp32 expert_bias, the w4a16
//     layout set, --layers, and every refusal (fused experts, a missing expert, an unconsumed tensor, a wrong shape,
//     --vision / --mtp / --rotate / --trellis-from / --kv-calib, an untied head). tools/convert_ref/make_tiny_lfm2.py
//     writes the same checkpoint for manual runs; this file builds it in C++ so it needs no Python.
// The Qwen / Gemma byte-identity guards are the existing convert_* tests; this file never runs their conversions.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <map>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "lfm2_layout.hpp"
#include "nlohmann/json.hpp"
#include "r4dx/core/dtype.hpp"
#include "r4dx/models_root.h"
#include "r4dx_convert/quant_int8.hpp"

namespace fs = std::filesystem;
namespace lf = r4dx_convert::lfm2;
using nlohmann::json;

namespace {

int g_failures = 0;
int g_skipped = 0;

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

bool EndsWith(const std::string& s, const std::string& suffix) {
  return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::string ReadWhole(const fs::path& p) {
  std::ifstream f(p, std::ios::binary);
  if (!f) throw std::runtime_error("cannot read " + p.string());
  return std::string(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}

void WriteWhole(const fs::path& p, const std::string& bytes) {
  std::ofstream f(p, std::ios::binary | std::ios::trunc);
  f.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

// ---- the layout's inputs: tensor name -> (dtype, shape) -----------------------------------------------------------

struct Geometry {
  int64_t hidden, layers, heads, kv, head_dim, dense_i, moe_i, dense_layers, experts, topk, vocab, conv_l;
};

constexpr Geometry kReal = {2048, 40, 32, 8, 64, 11776, 1536, 2, 64, 4, 128000, 3};
constexpr Geometry kTiny = {256, 4, 4, 2, 64, 256, 128, 2, 4, 2, 64, 3};

bool IsAttn(int64_t i) { return i % 4 == 2; }

json MakeConfig(const Geometry& g) {
  json types = json::array();
  for (int64_t i = 0; i < g.layers; ++i) types.push_back(IsAttn(i) ? "full_attention" : "conv");
  return {{"architectures", {"Lfm2MoeForCausalLM"}},
          {"bos_token_id", 124894},
          {"conv_L_cache", g.conv_l},
          {"conv_bias", false},
          {"dtype", "bfloat16"},
          {"eos_token_id", 124900},
          {"hidden_size", g.hidden},
          {"intermediate_size", g.dense_i},
          {"layer_types", types},
          {"max_position_embeddings", 4096},
          {"model_type", "lfm2_moe"},
          {"moe_intermediate_size", g.moe_i},
          {"norm_eps", 1e-05},
          {"norm_topk_prob", true},
          {"num_attention_heads", g.heads},
          {"num_dense_layers", g.dense_layers},
          {"num_experts", g.experts},
          {"num_experts_per_tok", g.topk},
          {"num_hidden_layers", g.layers},
          {"num_key_value_heads", g.kv},
          {"pad_token_id", 124893},
          {"rope_parameters", {{"rope_theta", 10000000.0}, {"rope_type", "default"}}},
          {"routed_scaling_factor", 1.0},
          {"tie_word_embeddings", true},
          {"use_cache", true},
          {"use_expert_bias", true},
          {"vocab_size", g.vocab}};
}

struct Spec {
  std::string name, dtype;
  std::vector<int64_t> shape;
};

// The checkpoint's tensors of `g` (per-expert style), in header order of layer. `fused`: transformers 5's fused experts.
std::vector<Spec> TensorSpecs(const Geometry& g, bool fused = false) {
  std::vector<Spec> s;
  const int64_t H = g.hidden;
  for (int64_t i = 0; i < g.layers; ++i) {
    const std::string L = "model.layers." + std::to_string(i) + ".";
    s.push_back({L + "operator_norm.weight", "BF16", {H}});
    s.push_back({L + "ffn_norm.weight", "BF16", {H}});
    if (IsAttn(i)) {
      s.push_back({L + "self_attn.q_proj.weight", "BF16", {g.heads * g.head_dim, H}});
      s.push_back({L + "self_attn.k_proj.weight", "BF16", {g.kv * g.head_dim, H}});
      s.push_back({L + "self_attn.v_proj.weight", "BF16", {g.kv * g.head_dim, H}});
      s.push_back({L + "self_attn.out_proj.weight", "BF16", {H, g.heads * g.head_dim}});
      s.push_back({L + "self_attn.q_layernorm.weight", "BF16", {g.head_dim}});
      s.push_back({L + "self_attn.k_layernorm.weight", "BF16", {g.head_dim}});
    } else {
      s.push_back({L + "conv.in_proj.weight", "BF16", {3 * H, H}});
      s.push_back({L + "conv.conv.weight", "BF16", {H, 1, g.conv_l}});
      s.push_back({L + "conv.out_proj.weight", "BF16", {H, H}});
    }
    if (i < g.dense_layers) {
      s.push_back({L + "feed_forward.w1.weight", "BF16", {g.dense_i, H}});
      s.push_back({L + "feed_forward.w2.weight", "BF16", {H, g.dense_i}});
      s.push_back({L + "feed_forward.w3.weight", "BF16", {g.dense_i, H}});
    } else {
      s.push_back({L + "feed_forward.gate.weight", "BF16", {g.experts, H}});
      s.push_back({L + "feed_forward.expert_bias", "F32", {g.experts}});
      if (fused) {
        s.push_back({L + "feed_forward.experts.gate_up_proj", "BF16", {g.experts, 2 * g.moe_i, H}});
        s.push_back({L + "feed_forward.experts.down_proj", "BF16", {g.experts, H, g.moe_i}});
      } else {
        for (int64_t e = 0; e < g.experts; ++e) {
          const std::string E = L + "feed_forward.experts." + std::to_string(e) + ".";
          s.push_back({E + "w1.weight", "BF16", {g.moe_i, H}});
          s.push_back({E + "w2.weight", "BF16", {H, g.moe_i}});
          s.push_back({E + "w3.weight", "BF16", {g.moe_i, H}});
        }
      }
    }
  }
  s.push_back({"model.embed_tokens.weight", "BF16", {g.vocab, H}});
  s.push_back({"model.embedding_norm.weight", "BF16", {H}});
  return s;
}

// ---- (a) the real geometry through a recording kit --------------------------------------------------------------

struct Recorder {
  std::map<std::string, std::vector<int64_t>> shapes;
  std::vector<std::pair<std::string, std::string>> bf16, widen;
  struct Shaped {
    std::string hf, name;
    std::vector<int64_t> shape;
  };
  std::vector<Shaped> shaped;
  struct Linear {
    std::vector<std::string> hf;
    std::string base;
    bool head;
  };
  std::vector<Linear> linear;

  void Load(const std::vector<Spec>& specs) {
    for (const Spec& s : specs) shapes[s.name] = s.shape;
  }
  lf::Kit MakeKit() {
    lf::Kit kit;
    kit.all_names = [this]() {
      std::vector<std::string> n;
      for (const auto& kv : shapes) n.push_back(kv.first);
      return n;
    };
    kit.has = [this](const std::string& n) { return shapes.count(n) != 0; };
    kit.shape = [this](const std::string& n) { return shapes.at(n); };
    kit.add_bf16 = [this](const std::string& a, const std::string& b) { bf16.push_back({a, b}); };
    kit.add_bf16_shaped = [this](const std::string& a, const std::string& b, const std::vector<int64_t>& sh) {
      shaped.push_back({a, b, sh});
    };
    kit.add_fp32_widen = [this](const std::string& a, const std::string& b) { widen.push_back({a, b}); };
    kit.add_linear = [this](const std::vector<std::string>& hf, const std::string& base, bool head) {
      linear.push_back({hf, base, head});
    };
    return kit;
  }
  const Linear* Find(const std::string& base) const {
    for (const auto& l : linear)
      if (l.base == base) return &l;
    return nullptr;
  }
};

void TestRealGeometry() {
  std::printf("---- (a) lfm2_layout.cpp over the real-geometry header ----\n");
  const json cfg = MakeConfig(kReal);
  const std::vector<Spec> specs = TensorSpecs(kReal);
  Check(specs.size() == 7610, "generated header: 7610 tensors (the real checkpoint's count)");
  Recorder rec;
  rec.Load(specs);

  const lf::TextShape shape = lf::ParseConfig(cfg);
  Check(shape.layers_total == 40 && shape.head_dim == 64 && shape.num_dense_layers == 2 && shape.num_experts == 64 &&
            shape.moe_intermediate == 1536 && shape.conv_kernel == 3 && shape.IsAttn(2) && shape.IsAttn(38) &&
            !shape.IsAttn(0) && !shape.IsAttn(39),
        "ParseConfig: the real geometry (head_dim 64 derived from hidden / heads)");

  lf::Kit kit = rec.MakeKit();
  const std::set<std::string> consumed = lf::AddTextStack(kit, shape, 40);
  const std::vector<std::string> all = kit.all_names();
  Check(consumed.size() == 7610 && lf::UnconsumedTensors(all, consumed, 40).empty(),
        "real geometry: all 7610 tensors consumed, nothing left over");
  // 10 attention x 4 + 30 conv x 2 + 2 dense x 2 + 38 MoE x 64 x 2 experts + lm_head
  Check(rec.linear.size() == 40 + 60 + 4 + 38 * 64 * 2 + 1, "4969 linears: 40 attn + 60 conv + 4 dense + 4864 expert + lm_head");
  // norms 80 + qk-norms 20 + embed + final norm + router 38
  Check(rec.bf16.size() == 80 + 20 + 2 + 38, "bf16 passthroughs: norms, q/k norms, embed, final norm, routers");
  Check(rec.shaped.size() == 30 && rec.widen.size() == 38, "30 squeezed conv weights, 38 widened expert_bias");

  // Names and fusion sources.
  const auto* e = rec.Find("text.layers.5.moe.experts.17.gate_up");
  const std::string F5 = "model.layers.5.feed_forward.experts.17.";
  Check(e && e->hf == std::vector<std::string>{F5 + "w1.weight", F5 + "w3.weight"} && !e->head,
        "expert gate_up <- w1 rows (gate) then w3 rows (up)");
  const auto* d = rec.Find("text.layers.5.moe.experts.17.down");
  Check(d && d->hf == std::vector<std::string>{F5 + "w2.weight"}, "expert down <- w2");
  const auto* dg = rec.Find("text.layers.1.mlp.gate_up");
  Check(dg && dg->hf == std::vector<std::string>{"model.layers.1.feed_forward.w1.weight", "model.layers.1.feed_forward.w3.weight"} &&
            rec.Find("text.layers.1.mlp.down")->hf == std::vector<std::string>{"model.layers.1.feed_forward.w2.weight"},
        "dense mlp.gate_up <- w1 | w3, mlp.down <- w2 (layers 0-1 only)");
  Check(rec.Find("text.layers.2.mlp.gate_up") == nullptr && rec.Find("text.layers.1.moe.experts.0.down") == nullptr,
        "dense FFN only below num_dense_layers, experts only from it");
  const auto* q = rec.Find("text.layers.2.attn.q");
  Check(q && q->hf == std::vector<std::string>{"model.layers.2.self_attn.q_proj.weight"} &&
            rec.Find("text.layers.2.attn.o")->hf == std::vector<std::string>{"model.layers.2.self_attn.out_proj.weight"} &&
            rec.Find("text.layers.1.attn.q") == nullptr && rec.Find("text.layers.38.attn.v") != nullptr,
        "attention linears on layers 2, 6, ..., 38 only; o <- out_proj");
  Check(rec.Find("text.layers.0.conv.in_proj") != nullptr && rec.Find("text.layers.2.conv.in_proj") == nullptr &&
            rec.Find("text.layers.39.conv.out_proj") != nullptr,
        "conv linears on the 30 ShortConv layers");
  const auto* head = rec.Find("lm_head");
  Check(head && head->head && head->hf == std::vector<std::string>{"model.embed_tokens.weight"},
        "lm_head <- the tied embedding, the head layout set");
  bool heads_ok = true;
  for (const auto& l : rec.linear) heads_ok = heads_ok && (l.head == (l.base == "lm_head"));
  Check(heads_ok, "only lm_head takes the head layout set");

  std::set<std::string> got_bf16;
  for (const auto& p : rec.bf16) got_bf16.insert(p.second);
  Check(got_bf16.count("text.final_norm") && got_bf16.count("text.embed_tokens") && got_bf16.count("text.layers.0.operator_norm") &&
            got_bf16.count("text.layers.39.ffn_norm") && got_bf16.count("text.layers.38.attn.q_norm") &&
            got_bf16.count("text.layers.3.moe.router") && !got_bf16.count("text.layers.0.moe.router"),
        "bf16 container names (final_norm <- embedding_norm)");
  bool final_ok = false;
  for (const auto& p : rec.bf16) final_ok = final_ok || (p.first == "model.embedding_norm.weight" && p.second == "text.final_norm");
  Check(final_ok, "text.final_norm <- model.embedding_norm.weight");
  bool conv_ok = rec.shaped.size() == 30;
  for (const auto& sh : rec.shaped)
    conv_ok = conv_ok && sh.shape == std::vector<int64_t>({2048, 3}) && EndsWith(sh.name, ".conv.weight") &&
              EndsWith(sh.hf, ".conv.conv.weight");
  Check(conv_ok, "conv.conv.weight [2048,1,3] registered as the squeezed [2048, 3]");
  bool bias_ok = rec.widen.size() == 38;
  for (const auto& p : rec.widen)
    bias_ok = bias_ok && p.first.find("feed_forward.expert_bias") != std::string::npos &&
              p.second.find("moe.expert_bias") != std::string::npos;
  Check(bias_ok, "expert_bias widened to fp32 under moe.expert_bias");

  // --layers: later layers are allow-listed, the head and embedding stay.
  {
    Recorder r2;
    r2.shapes = rec.shapes;
    lf::Kit k2 = r2.MakeKit();
    const std::set<std::string> c2 = lf::AddTextStack(k2, shape, 3);
    Check(lf::UnconsumedTensors(all, c2, 3).empty() && r2.Find("text.layers.2.attn.q") != nullptr &&
              r2.Find("text.layers.3.conv.in_proj") == nullptr && r2.Find("lm_head") != nullptr &&
              r2.Find("text.layers.2.moe.experts.63.down") != nullptr,
          "--layers 3: layers 3+ allow-listed, layers 0-2 + embed/head converted");
    Check(lf::UnconsumedTensors(all, c2, 40).size() > 7000, "audit with all layers expected but 3 converted names the rest");
    // lfm2-l6.r4dx = `--layers 6` (layers 0-5: conv, conv, attention + 4 MoE layers): 10 conv + 4 attn + 4 dense + 512 expert + head.
    Recorder r6;
    r6.shapes = rec.shapes;
    lf::Kit k6 = r6.MakeKit();
    const std::set<std::string> c6 = lf::AddTextStack(k6, shape, 6);
    Check(lf::UnconsumedTensors(all, c6, 6).empty() && r6.linear.size() == 10 + 4 + 4 + 4 * 64 * 2 + 1 && r6.shaped.size() == 5 && r6.widen.size() == 4,
          "--layers 6 (the lfm2-l6 test container): 531 linears, 5 conv weights, 4 expert_bias, nothing left over");
  }
  // The audit.
  {
    Recorder r3;
    r3.shapes = rec.shapes;
    r3.shapes["model.layers.3.conv.mystery.weight"] = {4, 4};
    r3.shapes["lm_head.weight"] = {128000, 2048};
    lf::Kit k3 = r3.MakeKit();
    const std::set<std::string> c3 = lf::AddTextStack(k3, shape, 40);
    Check(lf::UnconsumedTensors(k3.all_names(), c3, 40) ==
              std::vector<std::string>{"lm_head.weight", "model.layers.3.conv.mystery.weight"},
          "audit: an unknown tensor and a stray lm_head.weight are reported by name");
  }
  // Structural refusals, before any job is registered.
  {
    Recorder r4;
    r4.Load(TensorSpecs(kReal, /*fused=*/true));
    lf::Kit k4 = r4.MakeKit();
    Check(Throws([&] { lf::AddTextStack(k4, shape, 40); }, "fused-experts") && r4.linear.size() < 70,
          "fused experts.gate_up_proj is refused by name (and nothing past the first MoE layer was registered)");
    Recorder r5;
    r5.shapes = rec.shapes;
    r5.shapes.erase("model.layers.7.feed_forward.experts.30.w2.weight");
    lf::Kit k5 = r5.MakeKit();
    Check(Throws([&] { lf::AddTextStack(k5, shape, 40); }, "experts.30.w2.weight"), "a missing expert tensor throws, naming it");
    Recorder r6;
    r6.shapes = rec.shapes;
    r6.shapes["model.layers.0.conv.conv.weight"] = {2048, 3};
    lf::Kit k6 = r6.MakeKit();
    Check(Throws([&] { lf::AddTextStack(k6, shape, 40); }, "config says [2048,1,3]"), "a squeezed conv weight in the checkpoint throws, naming both shapes");
    Recorder r7;
    r7.shapes = rec.shapes;
    r7.shapes.erase("model.layers.9.feed_forward.expert_bias");
    lf::Kit k7 = r7.MakeKit();
    Check(Throws([&] { lf::AddTextStack(k7, shape, 40); }, "expert_bias"), "a missing expert_bias throws");
    Recorder r8;
    r8.shapes = rec.shapes;
    r8.shapes["model.layers.2.self_attn.k_proj.weight"] = {2048, 2048};
    lf::Kit k8 = r8.MakeKit();
    Check(Throws([&] { lf::AddTextStack(k8, shape, 40); }, "config says [512,2048]"), "a k_proj with q's shape throws, naming both");
    Check(Throws([&] { lf::AddTextStack(kit, shape, 41); }, "--layers"), "--layers beyond the checkpoint throws");
  }
  // Config refusals.
  {
    json c = MakeConfig(kReal);
    c["tie_word_embeddings"] = false;
    Check(Throws([&] { lf::ParseConfig(c); }, "tie_word_embeddings"), "ParseConfig refuses an untied head");
    c = MakeConfig(kReal);
    c["conv_bias"] = true;
    Check(Throws([&] { lf::ParseConfig(c); }, "conv_bias"), "ParseConfig refuses conv_bias");
    c = MakeConfig(kReal);
    c["use_expert_bias"] = false;
    Check(Throws([&] { lf::ParseConfig(c); }, "use_expert_bias"), "ParseConfig refuses use_expert_bias false");
    c = MakeConfig(kReal);
    c["layer_types"][0] = "linear_attention";
    Check(Throws([&] { lf::ParseConfig(c); }, "linear_attention"), "ParseConfig refuses a foreign layer type");
    c = MakeConfig(kReal);
    c.erase("moe_intermediate_size");
    Check(Throws([&] { lf::ParseConfig(c); }, "moe_intermediate_size"), "ParseConfig names a missing key");
    c = MakeConfig(kReal);
    c["model_type"] = "gemma4_unified";
    Check(Throws([&] { lf::ParseConfig(c); }, "model_type"), "ParseConfig refuses another model_type");
    c = MakeConfig(kReal);
    c["head_dim"] = 128;
    Check(lf::ParseConfig(c).head_dim == 128, "an explicit head_dim key wins over hidden / heads");
  }
}

// The real checkpoint's header bytes (no weights): names and (dtype, shape), every shard of the index.
void TestRealCheckpointHeader() {
  std::printf("---- (a2) the real checkpoint header ----\n");
  const fs::path dir = fs::path(r4dx::ModelsPath("LiquidAI/LFM2.5-24B-A2B"));
  const fs::path idx = dir / "model.safetensors.index.json";
  if (!fs::exists(idx)) {
    std::printf("SKIP: %s not present\n", idx.string().c_str());
    ++g_skipped;
    return;
  }
  const json index = json::parse(ReadWhole(idx));
  std::set<std::string> files;
  for (auto it = index.at("weight_map").begin(); it != index.at("weight_map").end(); ++it) files.insert(it.value().get<std::string>());
  std::map<std::string, std::pair<std::string, std::vector<int64_t>>> real;
  for (const std::string& fn : files) {
    std::ifstream f(dir / fn, std::ios::binary);
    uint64_t hl = 0;
    f.read(reinterpret_cast<char*>(&hl), 8);
    std::string h(static_cast<size_t>(hl), '\0');
    f.read(h.data(), static_cast<std::streamsize>(hl));
    const json hdr = json::parse(h);
    for (auto it = hdr.begin(); it != hdr.end(); ++it) {
      if (it.key() == "__metadata__") continue;
      real[it.key()] = {it.value().at("dtype").get<std::string>(), it.value().at("shape").get<std::vector<int64_t>>()};
    }
  }
  const std::vector<Spec> specs = TensorSpecs(kReal);
  bool same = real.size() == specs.size();
  std::string first_diff;
  for (const Spec& s : specs) {
    const auto it = real.find(s.name);
    if (it == real.end() || it->second.first != s.dtype || it->second.second != s.shape) {
      same = false;
      if (first_diff.empty()) first_diff = s.name;
    }
  }
  Check(same, "real header == the generated real-geometry header (names, dtypes, shapes; first diff: " + first_diff + ")");
  Check(real.at("model.layers.0.conv.conv.weight").second == std::vector<int64_t>({2048, 1, 3}) &&
            real.at("model.layers.2.feed_forward.expert_bias").first == "F32" && real.count("lm_head.weight") == 0,
        "real header: conv.conv.weight [2048,1,3], expert_bias F32, no lm_head");
  Recorder rec;
  for (const auto& kv : real) rec.shapes[kv.first] = kv.second.second;
  lf::Kit kit = rec.MakeKit();
  const std::set<std::string> consumed = lf::AddTextStack(kit, lf::ParseConfig(json::parse(ReadWhole(dir / "config.json"))), 40);
  Check(lf::UnconsumedTensors(kit.all_names(), consumed, 40).empty(), "real header + real config.json: every tensor consumed");
  Recorder rec6;
  rec6.shapes = rec.shapes;
  lf::Kit kit6 = rec6.MakeKit();
  const std::set<std::string> consumed6 = lf::AddTextStack(kit6, lf::ParseConfig(json::parse(ReadWhole(dir / "config.json"))), 6);
  Check(lf::UnconsumedTensors(kit6.all_names(), consumed6, 6).empty() && rec6.linear.size() == 531,
        "real header + real config.json, --layers 6 (lfm2-l6): 531 linears, every other layer allow-listed");
}

// ---- (b) the exe on a tiny synthetic checkpoint -------------------------------------------------------------------

float Bf16Round(float x) { return r4dx::core::Bf16ToFloat(r4dx::core::FloatToBf16(x)); }

struct Tensor {
  std::string name, dtype;
  std::vector<int64_t> shape;
  std::vector<float> v;  // bf16-exact (or the fp32 values for F32)
};

struct Ckpt {
  std::vector<Tensor> t;
  json config;
  int Find(const std::string& name) const {
    for (size_t i = 0; i < t.size(); ++i)
      if (t[i].name == name) return static_cast<int>(i);
    return -1;
  }
  const std::vector<float>& Get(const std::string& name) const { return t.at(static_cast<size_t>(Find(name))).v; }
  void Set(const std::string& name, std::vector<int64_t> shape, const std::string& dtype = "BF16") {
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    std::vector<float> v(static_cast<size_t>(n), 0.5f);
    const int i = Find(name);
    if (i >= 0) {
      t[static_cast<size_t>(i)] = {name, dtype, std::move(shape), std::move(v)};
    } else {
      t.push_back({name, dtype, std::move(shape), std::move(v)});
    }
  }
  void Erase(const std::string& name) {
    const int i = Find(name);
    if (i >= 0) t.erase(t.begin() + i);
  }
};

Ckpt MakeCkpt(bool fused = false) {
  Ckpt c;
  std::mt19937_64 rng(0x1F42ull);
  std::normal_distribution<double> nd(0.0, 1.0);
  for (const Spec& s : TensorSpecs(kTiny, fused)) {
    int64_t n = 1;
    for (int64_t d : s.shape) n *= d;
    std::vector<float> v(static_cast<size_t>(n));
    const bool norm = s.name.find("norm") != std::string::npos || s.name.find("layernorm") != std::string::npos;
    for (auto& x : v) {
      if (s.dtype == "F32") x = static_cast<float>(0.1 * nd(rng));
      else if (norm) x = Bf16Round(static_cast<float>(1.0 + 0.2 * nd(rng)));
      else x = Bf16Round(static_cast<float>(0.05 * nd(rng)));
    }
    c.t.push_back({s.name, s.dtype, s.shape, std::move(v)});
  }
  c.config = MakeConfig(kTiny);
  return c;
}

// Writes `c` as one model.safetensors, or `shards` files + model.safetensors.index.json (like the real checkpoint).
void WriteCkpt(const Ckpt& c, const fs::path& dir, int shards = 1) {
  fs::create_directories(dir);
  auto write_file = [&](const fs::path& path, size_t from, size_t to) {
    json hdr = json::object();
    std::string data;
    for (size_t i = from; i < to; ++i) {
      const Tensor& t = c.t[i];
      const uint64_t b = data.size();
      for (float x : t.v) {
        if (t.dtype == "F32") {
          data.append(reinterpret_cast<const char*>(&x), 4);
        } else {
          const uint16_t h = r4dx::core::FloatToBf16(x);
          data.append(reinterpret_cast<const char*>(&h), 2);
        }
      }
      hdr[t.name] = {{"dtype", t.dtype}, {"shape", t.shape}, {"data_offsets", {b, data.size()}}};
    }
    const std::string h = hdr.dump();
    const uint64_t hl = h.size();
    std::string file(reinterpret_cast<const char*>(&hl), 8);
    file += h;
    file += data;
    WriteWhole(path, file);
  };
  if (shards <= 1) {
    write_file(dir / "model.safetensors", 0, c.t.size());
  } else {
    json wm = json::object();
    const size_t per = (c.t.size() + static_cast<size_t>(shards) - 1) / static_cast<size_t>(shards);
    for (int s = 0; s < shards; ++s) {
      const size_t from = static_cast<size_t>(s) * per, to = std::min(c.t.size(), from + per);
      char fn[64];
      std::snprintf(fn, sizeof fn, "model-%05d-of-%05d.safetensors", s + 1, shards);
      write_file(dir / fn, from, to);
      for (size_t i = from; i < to; ++i) wm[c.t[i].name] = std::string(fn);
    }
    WriteWhole(dir / "model.safetensors.index.json", json({{"metadata", {{"total_size", 0}}}, {"weight_map", wm}}).dump(2));
  }
  WriteWhole(dir / "config.json", c.config.dump(2));
}

struct Container {
  bool ok = false;
  json header;
  std::map<std::string, std::string> tensors;
  json Meta() const { return header.at("__metadata__"); }
  std::vector<int64_t> Shape(const std::string& n) const { return header.at(n).at("shape").get<std::vector<int64_t>>(); }
  std::string Dtype(const std::string& n) const { return header.at(n).at("dtype").get<std::string>(); }
  std::vector<float> Floats(const std::string& n) const {
    const std::string& b = tensors.at(n);
    std::vector<float> v(b.size() / 4);
    std::memcpy(v.data(), b.data(), b.size());
    return v;
  }
  std::vector<std::string> Names() const {
    std::vector<std::string> n;
    for (const auto& kv : tensors) n.push_back(kv.first);
    return n;
  }
};

std::string BytesOf(const std::vector<float>& v) {
  std::string b;
  for (float x : v) {
    const uint16_t h = r4dx::core::FloatToBf16(x);
    b.append(reinterpret_cast<const char*>(&h), 2);
  }
  return b;
}

Container ReadContainer(const fs::path& path) {
  Container c;
  std::string bytes;
  try {
    bytes = ReadWhole(path);
  } catch (const std::exception&) {
    return c;
  }
  if (bytes.size() < 8) return c;
  uint64_t hl = 0;
  std::memcpy(&hl, bytes.data(), 8);
  if (8 + hl > bytes.size()) return c;
  c.header = json::parse(bytes.substr(8, static_cast<size_t>(hl)));
  const std::string data = bytes.substr(static_cast<size_t>(8 + hl));
  for (auto it = c.header.begin(); it != c.header.end(); ++it) {
    if (it.key() == "__metadata__") continue;
    const uint64_t b = it.value()["data_offsets"][0].get<uint64_t>();
    const uint64_t e = it.value()["data_offsets"][1].get<uint64_t>();
    c.tensors[it.key()] = data.substr(static_cast<size_t>(b), static_cast<size_t>(e - b));
  }
  c.ok = true;
  return c;
}

#if defined(R4DX_CONVERT_EXE)

// cmd.exe strips the outermost pair of quotes of a command that starts with one, so wrap it whole.
int Run(const std::string& args, const fs::path& log) {
  const std::string cmd = "\"\"" + std::string(R4DX_CONVERT_EXE) + "\" " + args + " > \"" + log.u8string() + "\" 2>&1\"";
  return std::system(cmd.c_str());
}

fs::path TempDir(const std::string& tag) {
  const fs::path d = fs::temp_directory_path() / ("r4dx_test_lfm2_layout_" + tag);
  std::error_code ec;
  fs::remove_all(d, ec);
  fs::create_directories(d);
  return d;
}

// Every tensor name a bf16-only conversion of the tiny checkpoint (first `layers` layers) must contain (and nothing else).
std::set<std::string> ExpectedNames(int64_t layers) {
  const Geometry& g = kTiny;
  std::set<std::string> n = {"text.embed_tokens", "text.final_norm", "lm_head.bf16.w"};
  for (int64_t i = 0; i < layers; ++i) {
    const std::string b = "text.layers." + std::to_string(i) + ".";
    n.insert(b + "operator_norm");
    n.insert(b + "ffn_norm");
    if (IsAttn(i)) {
      for (const char* x : {"attn.q", "attn.k", "attn.v", "attn.o"}) n.insert(b + x + std::string(".bf16.w"));
      n.insert(b + "attn.q_norm");
      n.insert(b + "attn.k_norm");
    } else {
      n.insert(b + "conv.in_proj.bf16.w");
      n.insert(b + "conv.out_proj.bf16.w");
      n.insert(b + "conv.weight");
    }
    if (i < g.dense_layers) {
      n.insert(b + "mlp.gate_up.bf16.w");
      n.insert(b + "mlp.down.bf16.w");
    } else {
      n.insert(b + "moe.router");
      n.insert(b + "moe.expert_bias");
      for (int64_t e = 0; e < g.experts; ++e) {
        n.insert(b + "moe.experts." + std::to_string(e) + ".gate_up.bf16.w");
        n.insert(b + "moe.experts." + std::to_string(e) + ".down.bf16.w");
      }
    }
  }
  return n;
}

std::string Join(const std::set<std::string>& s) {
  std::string out;
  for (const auto& x : s) out += x + " ";
  return out;
}

void CheckNames(const Container& c, const std::set<std::string>& want, const std::string& what) {
  const std::vector<std::string> names = c.Names();
  const std::set<std::string> got(names.begin(), names.end());
  Check(got == want, what);
  if (got != want) {
    std::set<std::string> extra, missing;
    std::set_difference(got.begin(), got.end(), want.begin(), want.end(), std::inserter(extra, extra.end()));
    std::set_difference(want.begin(), want.end(), got.begin(), got.end(), std::inserter(missing, missing.end()));
    std::fprintf(stderr, "  extra: %s\n  missing: %s\n", Join(extra).c_str(), Join(missing).c_str());
  }
}

void TestExe() {
  std::printf("---- (b) r4dx-convert on a tiny synthetic lfm2_moe checkpoint ----\n");
  const Geometry& g = kTiny;
  const int64_t H = g.hidden;
  const fs::path root = TempDir("exe");
  const fs::path ckpt = root / "ckpt-lfm2-tiny", out = root / "out.r4dx", log = root / "convert.log";
  const Ckpt base = MakeCkpt();
  WriteCkpt(base, ckpt);
  const std::string common = " --threads 2 --layouts bf16 --lm-head bf16 ";
  auto convert = [&](const fs::path& in, const std::string& extra) {
    std::error_code ec;
    fs::remove(out, ec);
    return Run("--input \"" + in.u8string() + "\" --output \"" + out.u8string() + "\"" + common + extra, log);
  };
  auto log_has = [&](const std::string& s) { return ReadWhole(log).find(s) != std::string::npos; };

  // ---- the default bf16 container (single-file checkpoint) ----
  {
    const int rc = convert(ckpt, "");
    Check(rc == 0, "bf16 container: converts a single-file flat-config checkpoint (rc=" + std::to_string(rc) + ")");
    const Container c = ReadContainer(out);
    Check(c.ok, "bf16 container: readable");
    if (c.ok) {
      CheckNames(c, ExpectedNames(4), "bf16 container: exactly the plan's tensor set (per-expert linears, no vision/mtp/descale)");
      const json md = c.Meta();
      Check(md.value("model_arch", "") == "lfm2_moe" && md.value("norm_kind", "") == "plain",
            "metadata: model_arch lfm2_moe, norm_kind plain");
      Check(md.value("model_id", "") == "ckpt-lfm2-tiny", "metadata: model_id is the checkpoint directory name");
      Check(md.at("model_config") == base.config, "metadata: model_config is the verbatim flat config.json");
      Check(md.at("r4dx_convert_run").value("arch", "") == "lfm2_moe" && !md.at("r4dx_convert_run").value("mtp", true) &&
                !md.at("r4dx_convert_run").value("vision", true),
            "metadata: r4dx_convert_run arch, mtp/vision off");
      Check(!md.contains("rotation"), "metadata: no rotation block");

      // Shapes and dtypes.
      Check(c.Shape("text.embed_tokens") == std::vector<int64_t>({g.vocab, H, 2}), "embed_tokens [vocab, hidden] bf16");
      Check(c.Shape("text.final_norm") == std::vector<int64_t>({H, 2}), "final_norm [hidden]");
      Check(c.Shape("text.layers.2.attn.q.bf16.w") == std::vector<int64_t>({g.heads * g.head_dim, H, 2}) &&
                c.Shape("text.layers.2.attn.k.bf16.w") == std::vector<int64_t>({g.kv * g.head_dim, H, 2}) &&
                c.Shape("text.layers.2.attn.o.bf16.w") == std::vector<int64_t>({H, g.heads * g.head_dim, 2}),
            "attention q / k / o shapes");
      Check(c.Shape("text.layers.2.attn.q_norm") == std::vector<int64_t>({g.head_dim, 2}), "q_norm [head_dim]");
      Check(c.Shape("text.layers.0.conv.in_proj.bf16.w") == std::vector<int64_t>({3 * H, H, 2}) &&
                c.Shape("text.layers.0.conv.out_proj.bf16.w") == std::vector<int64_t>({H, H, 2}),
            "conv in_proj [3H, H], out_proj [H, H]");
      Check(c.Shape("text.layers.0.conv.weight") == std::vector<int64_t>({H, g.conv_l, 2}) && c.Dtype("text.layers.0.conv.weight") == "U8",
            "conv.weight squeezed to [hidden, 3] (+ element width 2)");
      Check(c.Shape("text.layers.0.mlp.gate_up.bf16.w") == std::vector<int64_t>({2 * g.dense_i, H, 2}), "dense mlp.gate_up [2I, H]");
      Check(c.Shape("text.layers.3.moe.router") == std::vector<int64_t>({g.experts, H, 2}), "router [E, H] bf16");
      Check(c.Shape("text.layers.3.moe.expert_bias") == std::vector<int64_t>({g.experts, 4}), "expert_bias fp32 [E, 4]");
      Check(c.Shape("text.layers.3.moe.experts.2.gate_up.bf16.w") == std::vector<int64_t>({2 * g.moe_i, H, 2}) &&
                c.Shape("text.layers.3.moe.experts.2.down.bf16.w") == std::vector<int64_t>({H, g.moe_i, 2}),
            "expert gate_up [2I, H], down [H, I]");

      // Bytes.
      Check(c.tensors.at("text.embed_tokens") == BytesOf(base.Get("model.embed_tokens.weight")), "embed_tokens bytes == checkpoint");
      Check(c.tensors.at("lm_head.bf16.w") == BytesOf(base.Get("model.embed_tokens.weight")), "lm_head bf16 == the tied embedding (written untied)");
      Check(c.tensors.at("text.final_norm") == BytesOf(base.Get("model.embedding_norm.weight")), "final_norm bytes == embedding_norm");
      Check(c.tensors.at("text.layers.0.conv.weight") == BytesOf(base.Get("model.layers.0.conv.conv.weight")), "conv.weight bytes == conv.conv.weight (a squeeze, not a copy of values)");
      Check(c.tensors.at("text.layers.2.attn.o.bf16.w") == BytesOf(base.Get("model.layers.2.self_attn.out_proj.weight")), "attn.o bytes == out_proj");
      Check(c.tensors.at("text.layers.2.attn.q_norm") == BytesOf(base.Get("model.layers.2.self_attn.q_layernorm.weight")), "q_norm bytes == q_layernorm");
      Check(c.tensors.at("text.layers.3.moe.router") == BytesOf(base.Get("model.layers.3.feed_forward.gate.weight")), "router bytes == gate.weight");
      Check(c.tensors.at("text.layers.1.mlp.gate_up.bf16.w") ==
                BytesOf(base.Get("model.layers.1.feed_forward.w1.weight")) + BytesOf(base.Get("model.layers.1.feed_forward.w3.weight")),
            "dense gate_up bytes == w1 rows then w3 rows");
      Check(c.tensors.at("text.layers.1.mlp.down.bf16.w") == BytesOf(base.Get("model.layers.1.feed_forward.w2.weight")), "dense down bytes == w2");
      bool experts_ok = true;
      for (int64_t li : {2, 3}) {
        for (int64_t e = 0; e < g.experts; ++e) {
          const std::string hf = "model.layers." + std::to_string(li) + ".feed_forward.experts." + std::to_string(e) + ".";
          const std::string cb = "text.layers." + std::to_string(li) + ".moe.experts." + std::to_string(e) + ".";
          experts_ok = experts_ok &&
                       c.tensors.at(cb + "gate_up.bf16.w") == BytesOf(base.Get(hf + "w1.weight")) + BytesOf(base.Get(hf + "w3.weight")) &&
                       c.tensors.at(cb + "down.bf16.w") == BytesOf(base.Get(hf + "w2.weight"));
        }
      }
      Check(experts_ok, "every expert: gate_up bytes == w1 rows then w3 rows, down bytes == w2 (layers 2 and 3)");
      Check(c.Floats("text.layers.3.moe.expert_bias") == base.Get("model.layers.3.feed_forward.expert_bias"),
            "expert_bias: the F32 values exactly, stored fp32");
    }
  }

  // ---- a sharded checkpoint (10 shards + index in the real one) converts to the same container ----
  {
    const fs::path sh = root / "ckpt-lfm2-sharded";
    WriteCkpt(base, sh, 5);
    const Container ref = ReadContainer(out);
    const int rc = convert(sh, "");
    const Container c = ReadContainer(out);
    Check(rc == 0 && c.ok && c.tensors == ref.tensors, "sharded checkpoint (index.json): the same container bytes as the single file");
  }

  // ---- the w4a16 layout set: only the linears quantize ----
  {
    const int rc = Run("--input \"" + ckpt.u8string() + "\" --output \"" + out.u8string() + "\" --threads 2 --layouts w4a16 --lm-head 4bit", log);
    const Container c = ReadContainer(out);
    Check(rc == 0 && c.ok, "--layouts w4a16: converts (rc=" + std::to_string(rc) + ")");
    if (c.ok) {
      Check(c.tensors.count("text.layers.3.moe.experts.1.gate_up.w4a16.wq") == 1 && c.tensors.count("text.layers.3.moe.experts.1.down.w4a16.wq") == 1 &&
                c.tensors.count("text.layers.0.conv.in_proj.w4a16.wq") == 1 && c.tensors.count("text.layers.2.attn.v.w4a16.wq") == 1 &&
                c.tensors.count("text.layers.1.mlp.down.w4a16.wq") == 1 && c.tensors.count("lm_head.w4a16.wq") == 1,
            "w4a16: expert / conv / attention / dense linears carry a .w4a16.wq");
      Check(c.tensors.count("text.layers.3.moe.router") == 1 && c.tensors.count("text.layers.3.moe.router.w4a16.wq") == 0 &&
                c.tensors.count("text.layers.0.conv.weight") == 1 && c.tensors.count("text.layers.0.conv.weight.w4a16.wq") == 0 &&
                c.tensors.count("text.layers.3.moe.expert_bias") == 1,
            "w4a16: router, conv.weight and expert_bias are never quantized");
    }
  }

  // ---- the w8a16 layout set (docs/w8a16.md 5.2: lfm2_layout.cpp needs no layout code) ----
  {
    const std::string base_args = "--input \"" + ckpt.u8string() + "\" --output \"" + out.u8string() + "\" --threads 2 ";
    auto run = [&](const std::string& extra) {
      std::error_code ec;
      fs::remove(out, ec);
      return Run(base_args + extra, log);
    };
    auto strip = [](const std::string& n) { return n.substr(0, n.size() - 7); };  // ".bf16.w"
    // ExpectedNames(4) with every ".bf16.w" linear for which `is_w8(base)` holds replaced by its w8a16 pair.
    auto expect = [&](const std::function<bool(const std::string&)>& is_w8, bool keep_bf16_too) {
      std::set<std::string> want;
      for (const std::string& n : ExpectedNames(4)) {
        if (!EndsWith(n, ".bf16.w")) {
          want.insert(n);
          continue;
        }
        const std::string b = strip(n);
        if (is_w8(b)) {
          want.insert(b + ".w8a16.wq");
          want.insert(b + ".w8a16.ws");
          if (keep_bf16_too) want.insert(n);
        } else {
          want.insert(n);
        }
      }
      return want;
    };
    auto all_w8 = [](const std::string&) { return true; };
    auto is_expert = [](const std::string& b) { return b.find(".moe.experts.") != std::string::npos; };

    // The fp32 truth the exe quantizes: the checkpoint's bf16 floats, fused like the layout fuses them.
    auto reference = [&](const std::vector<std::string>& hf, int group, int* N, int* K, std::vector<uint8_t>* wq,
                         std::vector<uint8_t>* ws) {
      std::vector<float> w;
      for (const auto& n : hf) {
        const Tensor& t = base.t.at(static_cast<size_t>(base.Find(n)));
        *K = static_cast<int>(t.shape[1]);
        w.insert(w.end(), t.v.begin(), t.v.end());
      }
      *N = static_cast<int>(w.size() / static_cast<size_t>(*K));
      std::vector<uint8_t> q;
      std::vector<uint16_t> s16;
      r4dx_convert::QuantizeInt8Symmetric(w.data(), *N, *K, group, 1, q, s16);
      *wq = r4dx_convert::PackW8Bytes(q, *N, *K, 1);
      const auto s = r4dx_convert::PackW8a16Scales(s16, *N, *K, group);
      ws->assign(reinterpret_cast<const uint8_t*>(s.data()), reinterpret_cast<const uint8_t*>(s.data()) + s.size() * 2);
    };
    auto bytes_eq = [](const std::string& got, const std::vector<uint8_t>& want) {
      return got.size() == want.size() && std::memcmp(got.data(), want.data(), want.size()) == 0;
    };

    // (1) everything w8a16, g128, the head following by default.
    {
      const int rc = run("--layouts w8a16 --no-bf16");
      const Container c = ReadContainer(out);
      Check(rc == 0 && c.ok, "--layouts w8a16 --no-bf16: converts (rc=" + std::to_string(rc) + ")");
      if (rc != 0) std::fprintf(stderr, "%s\n", ReadWhole(log).c_str());
      if (c.ok) {
        CheckNames(c, expect(all_w8, false), "w8a16: every linear (body, every per-expert pair, the tied head by default) is a .w8a16.wq/.ws pair, nothing else");
        const json md = c.Meta();
        const json want_meta = {{"version", 1}, {"group", 128}, {"scale", "f16"}, {"scale_log2_bias", 12},
                                {"code", "int8_sym127_offset_binary_x80"},
                                {"fragment_permutation", "wmma16x16x16_f16_b_lane16_k32block"}};
        Check(md.at("quant").contains("w8a16") && md.at("quant").at("w8a16") == want_meta, "metadata: quant.w8a16 block");
        Check(md.at("quant").contains("w4a16"), "metadata: quant.w4a16 block still present (unchanged base)");
        Check(md.at("r4dx_convert_run").at("lm_head").get<std::string>() == "w8a16+bf16", "run record: the derived head spec");
        // Sizes: wq {N*K}, ws {N*K/128, 2}; and the run record's group totals.
        uint64_t total_groups = 0;
        bool shapes_ok = true;
        for (const auto& kv : c.tensors) {
          if (!EndsWith(kv.first, ".w8a16.wq")) continue;
          const std::string b = kv.first.substr(0, kv.first.size() - 9);
          const std::vector<int64_t> sq = c.Shape(kv.first), ss = c.Shape(b + ".w8a16.ws");
          shapes_ok = shapes_ok && sq.size() == 1 && ss.size() == 2 && ss[0] * 128 == sq[0] && ss[1] == 2 &&
                      kv.second.size() == static_cast<size_t>(sq[0]) && c.tensors.at(b + ".w8a16.ws").size() == static_cast<size_t>(ss[0]) * 2 &&
                      sq[0] % (16 * 128) == 0;
          total_groups += static_cast<uint64_t>(ss[0]);
        }
        Check(shapes_ok, "w8a16: every pair has wq {N*K} bytes and ws {N*K/128, 2}");
        const json run = md.at("r4dx_convert_run").at("w8a16");
        const std::string zg = run.at("zero_groups").get<std::string>(), tg = run.at("groups_total").get<std::string>();
        auto digits20 = [](const std::string& s) {
          return s.size() == 20 && std::all_of(s.begin(), s.end(), [](char ch) { return ch >= '0' && ch <= '9'; });
        };
        Check(run.at("group") == 128 && digits20(zg) && digits20(tg) && std::stoull(tg) == total_groups,
              "run record: w8a16 group 128, groups_total == sum of ws groups (" + std::to_string(total_groups) + "), zero_groups patched (" + zg + ")");
        Check(ReadWhole(log).find("[r4dx-convert] w8a16: " + std::to_string(total_groups) + " groups quantized") != std::string::npos,
              "end-of-run log reports the group and zero-group counts");
        // Bytes of one linear of every class, from the same quantizer on the checkpoint's floats.
        struct Probe {
          std::string cb;
          std::vector<std::string> hf;
        };
        const std::vector<Probe> probes = {
            {"text.layers.3.moe.experts.2.gate_up", {"model.layers.3.feed_forward.experts.2.w1.weight", "model.layers.3.feed_forward.experts.2.w3.weight"}},
            {"text.layers.2.moe.experts.0.down", {"model.layers.2.feed_forward.experts.0.w2.weight"}},
            {"text.layers.0.conv.in_proj", {"model.layers.0.conv.in_proj.weight"}},
            {"text.layers.2.attn.q", {"model.layers.2.self_attn.q_proj.weight"}},
            {"text.layers.1.mlp.gate_up", {"model.layers.1.feed_forward.w1.weight", "model.layers.1.feed_forward.w3.weight"}},
            {"lm_head", {"model.embed_tokens.weight"}}};
        for (const Probe& p : probes) {
          int N = 0, K = 0;
          std::vector<uint8_t> wq, ws;
          reference(p.hf, 128, &N, &K, &wq, &ws);
          Check(bytes_eq(c.tensors.at(p.cb + ".w8a16.wq"), wq) && bytes_eq(c.tensors.at(p.cb + ".w8a16.ws"), ws),
                "w8a16 bytes of " + p.cb + " == the quantizer on the checkpoint weights");
        }
      }
    }
    // (2) --lm-head bf16 is the alternative.
    {
      const int rc = run("--layouts w8a16 --no-bf16 --lm-head bf16");
      const Container c = ReadContainer(out);
      Check(rc == 0 && c.ok, "--lm-head bf16 with w8a16 body: converts");
      if (c.ok)
        CheckNames(c, expect([](const std::string& b) { return b != "lm_head"; }, false),
                   "w8a16 body + --lm-head bf16: the head is lm_head.bf16.w only");
    }
    // (3) the hybrid recipe: experts w8a16, every other linear and the head kept bf16.
    {
      const int rc = run("--layouts w8a16 --no-bf16 --keep-bf16 \"^(?!.*\\.moe\\.experts\\.)\"");
      const Container c = ReadContainer(out);
      Check(rc == 0 && c.ok, "--keep-bf16 '^(?!.*\\.moe\\.experts\\.)': converts (rc=" + std::to_string(rc) + ")");
      if (rc != 0) std::fprintf(stderr, "%s\n", ReadWhole(log).c_str());
      if (c.ok) {
        CheckNames(c, expect(is_expert, false), "hybrid: expert linears w8a16 pairs, attention / conv / dense / head bf16-only");
        const json md = c.Meta();
        Check(md.at("quant").contains("w8a16"), "hybrid: quant.w8a16 present (some linear is w8a16)");
        bool summary = false;
        for (auto it = md.at("quant_summary").begin(); it != md.at("quant_summary").end(); ++it)
          summary = summary || it.value().get<std::string>().find("w4a16|w8a16|bf16") != std::string::npos;
        Check(summary, "hybrid: quant_summary rows read w4a16|w8a16|bf16");
        Check(md.at("r4dx_convert_run").at("keep_bf16_linears").size() > 0, "hybrid: keep_bf16_linears recorded");
      }
    }
    // (4) everything kept: no w8a16 tensors and a header with no w8a16 anywhere.
    {
      const int rc = run("--layouts w8a16 --no-bf16 --keep-bf16 \".*\"");
      const Container c = ReadContainer(out);
      Check(rc == 0 && c.ok, "--keep-bf16 .* with w8a16: converts");
      if (c.ok) {
        bool any = false;
        for (const auto& kv : c.tensors) any = any || kv.first.find(".w8a16.") != std::string::npos;
        const json md = c.Meta();
        Check(!any && !md.at("quant").contains("w8a16") && !md.at("r4dx_convert_run").contains("w8a16"),
              "all kept: no w8a16 tensor, no quant.w8a16 block, no run record");
        bool summary = false;
        for (auto it = md.at("quant_summary").begin(); it != md.at("quant_summary").end(); ++it)
          summary = summary || it.value().get<std::string>().find("w8a16") != std::string::npos;
        Check(!summary, "all kept: quant_summary untouched");
      }
    }
    // (5) g64.
    {
      const int rc = run("--layouts w8a16 --no-bf16 --w8a16-group 64");
      const Container c = ReadContainer(out);
      Check(rc == 0 && c.ok, "--w8a16-group 64: converts");
      if (c.ok) {
        Check(c.Meta().at("quant").at("w8a16").at("group") == 64 &&
                  c.Shape("text.layers.3.moe.experts.1.down.w8a16.ws") == std::vector<int64_t>({kTiny.hidden * kTiny.moe_i / 64, 2}),
              "g64: metadata group 64, ws {N*K/64, 2}");
        int N = 0, K = 0;
        std::vector<uint8_t> wq, ws;
        reference({"model.layers.3.feed_forward.experts.1.w2.weight"}, 64, &N, &K, &wq, &ws);
        Check(bytes_eq(c.tensors.at("text.layers.3.moe.experts.1.down.w8a16.wq"), wq) &&
                  bytes_eq(c.tensors.at("text.layers.3.moe.experts.1.down.w8a16.ws"), ws),
              "g64: bytes == the quantizer at group 64");
      }
    }
    // (6) bf16 left on: a w8a16 + bf16 pair per linear, the head too.
    {
      const int rc = run("--layouts w8a16");
      const Container c = ReadContainer(out);
      Check(rc == 0 && c.ok, "--layouts w8a16 (bf16 default on): converts");
      if (c.ok)
        CheckNames(c, expect(all_w8, true), "w8a16 + bf16: both layouts of every linear (the old-binary-safe pair)");
    }
    // (7) a w4a16 container carries no trace of w8a16 (header untouched).
    {
      const int rc = run("--layouts w4a16 --lm-head 4bit");
      const Container c = ReadContainer(out);
      Check(rc == 0 && c.ok, "control: --layouts w4a16 converts");
      if (c.ok) {
        const json md = c.Meta();
        bool trace = md.at("quant").contains("w8a16") || md.at("r4dx_convert_run").contains("w8a16");
        for (auto it = md.at("quant_summary").begin(); it != md.at("quant_summary").end(); ++it)
          trace = trace || it.value().get<std::string>().find("w8a16") != std::string::npos;
        Check(!trace, "control: a w4a16 container's header has no w8a16 key, run record or summary text");
      }
    }
    // (8) refusals.
    {
      Check(run("--layouts w8a16 --no-bf16 --quant search") != 0 && log_has("--quant search"), "w8a16 + --quant search is refused");
      Check(run("--layouts w8a16 --no-bf16 --hessian-dir nowhere --ldlq .*") != 0 && log_has("not implemented"), "w8a16 + --ldlq is refused (deferred)");
      Check(run("--layouts w8a16 --no-bf16 --w8a16-group 96") != 0 && log_has("64 or 128"), "--w8a16-group 96 is refused");
      Check(run("--layouts w4a16 --w8a16-group 64") != 0 && log_has("needs w8a16"), "--w8a16-group without w8a16 is refused");
      Check(run("--layouts w8a16 --no-bf16 --trellis-from nowhere") != 0 && log_has("--trellis-from"), "w8a16 + --trellis-from is refused");
      Check(Run("--dflash-gguf nowhere.gguf --out \"" + out.u8string() + "\" --layouts w8a16", log) != 0 && log_has("do not apply to --dflash"),
            "w8a16 with --dflash-gguf is refused");
    }
  }

  // ---- --layers: the first 3 layers (conv, conv, attention) + embed / norm / head; the later layers are allow-listed ----
  {
    const int rc = convert(ckpt, "--layers 3");
    const Container c = ReadContainer(out);
    Check(rc == 0 && c.ok, "--layers 3: converts");
    if (c.ok) {
      std::set<std::string> want = ExpectedNames(3);
      CheckNames(c, want, "--layers 3: layers 0-2 (the MoE layer 2 with all its experts) + embed / final norm / head");
      Check(c.Meta().at("r4dx_convert_run").value("layers_converted", 0) == 3, "--layers 3: layers_converted 3 of 4");
    }
  }

  // ---- refusals ----
  {
    Check(convert(ckpt, "--vision on") != 0 && log_has("--vision on"), "--vision on is refused");
    Check(convert(ckpt, "--mtp on") != 0 && log_has("--mtp on"), "--mtp on is refused");
    Check(convert(ckpt, "--rotate q2a") != 0 && log_has("--rotate"), "--rotate q2a is refused");
    Check(convert(ckpt, "--trellis-from nowhere") != 0 && log_has("--trellis-from"), "--trellis-from is refused");
    Check(convert(ckpt, "--kv-calib nowhere.json") != 0 && log_has("--kv-calib"), "--kv-calib is refused");

    const fs::path bad = root / "ckpt-bad";
    {
      const Ckpt fused = MakeCkpt(/*fused=*/true);
      WriteCkpt(fused, bad);
      Check(convert(bad, "") != 0 && log_has("fused-experts"), "a fused-experts checkpoint is refused by name");
    }
    {
      Ckpt c = base;
      c.Erase("model.layers.3.feed_forward.experts.2.w2.weight");
      WriteCkpt(c, bad);
      Check(convert(bad, "") != 0 && log_has("experts.2.w2.weight"), "a missing expert tensor fails the run, naming it");
    }
    {
      Ckpt c = base;
      c.Set("model.layers.0.mystery.weight", {4, 4});
      WriteCkpt(c, bad);
      Check(convert(bad, "") != 0 && log_has("model.layers.0.mystery.weight") && log_has("not converted"),
            "the coverage audit names an unconsumed tensor");
    }
    {
      Ckpt c = base;
      c.Set("lm_head.weight", {g.vocab, H});
      WriteCkpt(c, bad);
      Check(convert(bad, "") != 0 && log_has("lm_head.weight"), "a stray lm_head.weight (untied checkpoint) fails the audit");
    }
    {
      Ckpt c = base;
      c.Set("model.layers.2.self_attn.k_proj.weight", {g.heads * g.head_dim, H});
      WriteCkpt(c, bad);
      Check(convert(bad, "") != 0 && log_has("config says"), "a wrong shape fails the run, naming both shapes");
    }
    {
      Ckpt c = base;
      c.config["tie_word_embeddings"] = false;
      WriteCkpt(c, bad);
      Check(convert(bad, "") != 0 && log_has("tie_word_embeddings"), "an untied config is refused");
    }
    {
      Ckpt c = base;
      c.Erase("model.layers.3.feed_forward.expert_bias");
      WriteCkpt(c, bad);
      Check(convert(bad, "") != 0 && log_has("expert_bias"), "a missing expert_bias fails the run");
    }
  }
  std::error_code ec;
  fs::remove_all(root, ec);
}

#endif  // R4DX_CONVERT_EXE

}  // namespace

int main() {
  TestRealGeometry();
  TestRealCheckpointHeader();
#if defined(R4DX_CONVERT_EXE)
  TestExe();
#else
  std::printf("SKIP: r4dx-convert not built (part b)\n");
  ++g_skipped;
#endif
  std::printf("\n%s: %d failure(s), %d skipped part(s)\n", g_failures == 0 ? "OK" : "FAILED", g_failures, g_skipped);
  return g_failures == 0 ? 0 : 1;
}
