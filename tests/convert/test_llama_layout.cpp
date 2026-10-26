// tests/convert/test_llama_layout.cpp -- CPU-only (ctest `convert_llama_layout`). docs/nanbeige-plan.md N1-4.
//
// (a) llama_layout.cpp over the REAL-geometry header (Nanbeige4.3-33B: 52 layers, the 471 tensor names and shapes
//     generated here from the config) through a recording Kit, no weights: every tensor is consumed, the container
//     names / fused source lists are the plan's section 4 table, --layers allow-lists later layers, the coverage audit
//     names a stray tensor and allow-lists rotary_emb.inv_freq, every structural refusal fires. When the real
//     checkpoint is on disk (R4DX_MODELS_ROOT\Nanbeige\Nanbeige4.3-33B, header bytes only) its tensor set and shapes
//     are compared to the generated header too (SKIPPED when absent).
// (b) r4dx-convert (when built) on a tiny synthetic llama checkpoint (tools/convert_ref/make_tiny_llama.py's geometry:
//     3 layers, hidden 512, 28 q / 4 kv heads of 128, FFN 512, vocab 512, untied head, flat config with no
//     layer_types), single-file AND sharded: the exact container tensor set, shapes, bf16 byte identity (embed, head,
//     gate|up fusion, o_proj, norms), the w8a16 bytes against the quantizer, the w4a16 set, --layers, the header
//     metadata, --kv-calib's descales (N5-2), and every refusal (--vision / --mtp / --rotate, a missing tensor, an
//     unconsumed tensor, a wrong shape, a tied config, biases, rope scaling).
// The other families' byte-identity guards are their own convert_* tests; this file never runs their conversions.
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
#include <tuple>
#include <vector>

#include "llama_layout.hpp"
#include "nlohmann/json.hpp"
#include "r4dx/core/dtype.hpp"
#include "r4dx/models_root.h"
#include "r4dx_convert/quant_int8.hpp"

namespace fs = std::filesystem;
namespace ll = r4dx_convert::llama;
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
  int64_t hidden, layers, heads, kv, head_dim, inter, vocab;
};

constexpr Geometry kReal = {7168, 52, 56, 8, 128, 22016, 166912};
constexpr Geometry kTiny = {512, 3, 28, 4, 128, 512, 512};

// The real config.json's keys (Nanbeige4.3-33B, transformers 4.51.1 spelling), at geometry `g`.
json MakeConfig(const Geometry& g) {
  return {{"architectures", {"LlamaForCausalLM"}},
          {"attention_bias", false},
          {"attention_dropout", 0.0},
          {"bos_token_id", 166100},
          {"eos_token_id", 166101},
          {"head_dim", g.head_dim},
          {"hidden_act", "silu"},
          {"hidden_size", g.hidden},
          {"intermediate_size", g.inter},
          {"max_position_embeddings", 524288},
          {"mlp_bias", false},
          {"model_type", "llama"},
          {"num_attention_heads", g.heads},
          {"num_hidden_layers", g.layers},
          {"num_key_value_heads", g.kv},
          {"pad_token_id", 0},
          {"pretraining_tp", 1},
          {"rms_norm_eps", 1e-05},
          {"rope_scaling", nullptr},
          {"rope_theta", 120000000},
          {"tie_word_embeddings", false},
          {"torch_dtype", "bfloat16"},
          {"use_cache", false},
          {"vocab_size", g.vocab}};
}

struct Spec {
  std::string name, dtype;
  std::vector<int64_t> shape;
};

std::vector<Spec> TensorSpecs(const Geometry& g) {
  std::vector<Spec> s;
  const int64_t H = g.hidden, Q = g.heads * g.head_dim, KV = g.kv * g.head_dim;
  for (int64_t i = 0; i < g.layers; ++i) {
    const std::string L = "model.layers." + std::to_string(i) + ".";
    s.push_back({L + "input_layernorm.weight", "BF16", {H}});
    s.push_back({L + "post_attention_layernorm.weight", "BF16", {H}});
    s.push_back({L + "self_attn.q_proj.weight", "BF16", {Q, H}});
    s.push_back({L + "self_attn.k_proj.weight", "BF16", {KV, H}});
    s.push_back({L + "self_attn.v_proj.weight", "BF16", {KV, H}});
    s.push_back({L + "self_attn.o_proj.weight", "BF16", {H, Q}});
    s.push_back({L + "mlp.gate_proj.weight", "BF16", {g.inter, H}});
    s.push_back({L + "mlp.up_proj.weight", "BF16", {g.inter, H}});
    s.push_back({L + "mlp.down_proj.weight", "BF16", {H, g.inter}});
  }
  s.push_back({"model.embed_tokens.weight", "BF16", {g.vocab, H}});
  s.push_back({"model.norm.weight", "BF16", {H}});
  s.push_back({"lm_head.weight", "BF16", {g.vocab, H}});
  return s;
}

// ---- (a) the real geometry through a recording kit --------------------------------------------------------------

struct Recorder {
  std::map<std::string, std::vector<int64_t>> shapes;
  std::vector<std::pair<std::string, std::string>> bf16;
  struct Linear {
    std::vector<std::string> hf;
    std::string base;
    bool head;
  };
  std::vector<Linear> linear;

  void Load(const std::vector<Spec>& specs) {
    for (const Spec& s : specs) shapes[s.name] = s.shape;
  }
  ll::Kit MakeKit() {
    ll::Kit kit;
    kit.all_names = [this]() {
      std::vector<std::string> n;
      for (const auto& kv : shapes) n.push_back(kv.first);
      return n;
    };
    kit.has = [this](const std::string& n) { return shapes.count(n) != 0; };
    kit.shape = [this](const std::string& n) { return shapes.at(n); };
    kit.add_bf16 = [this](const std::string& a, const std::string& b) { bf16.push_back({a, b}); };
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
  std::printf("---- (a) llama_layout.cpp over the real-geometry header ----\n");
  const json cfg = MakeConfig(kReal);
  const std::vector<Spec> specs = TensorSpecs(kReal);
  Check(specs.size() == 471, "generated header: 471 tensors (the real checkpoint's count)");
  Recorder rec;
  rec.Load(specs);

  const ll::TextShape shape = ll::ParseConfig(cfg);
  Check(shape.hidden == 7168 && shape.layers_total == 52 && shape.heads == 56 && shape.kv_heads == 8 && shape.head_dim == 128 &&
            shape.intermediate == 22016 && shape.vocab == 166912,
        "ParseConfig: the real geometry");

  ll::Kit kit = rec.MakeKit();
  const std::set<std::string> consumed = ll::AddTextStack(kit, shape, 52);
  const std::vector<std::string> all = kit.all_names();
  Check(consumed.size() == 471 && ll::UnconsumedTensors(all, consumed, 52).empty(),
        "real geometry: all 471 tensors consumed, nothing left over");
  Check(rec.linear.size() == 52 * 6 + 1, "313 linears: 52 x (q, k, v, o, gate_up, down) + lm_head");
  Check(rec.bf16.size() == 52 * 2 + 2, "bf16 passthroughs: 104 norms, embed, final norm");

  const auto* gu = rec.Find("text.layers.17.mlp.gate_up");
  Check(gu && gu->hf == std::vector<std::string>{"model.layers.17.mlp.gate_proj.weight", "model.layers.17.mlp.up_proj.weight"} && !gu->head,
        "mlp.gate_up <- gate_proj rows then up_proj rows");
  Check(rec.Find("text.layers.17.mlp.down")->hf == std::vector<std::string>{"model.layers.17.mlp.down_proj.weight"}, "mlp.down <- down_proj");
  Check(rec.Find("text.layers.51.attn.o")->hf == std::vector<std::string>{"model.layers.51.self_attn.o_proj.weight"} &&
            rec.Find("text.layers.0.attn.k")->hf == std::vector<std::string>{"model.layers.0.self_attn.k_proj.weight"},
        "attn.{q,k,v,o} <- self_attn.{q,k,v,o}_proj, separate (no fused qkv)");
  const auto* head = rec.Find("lm_head");
  Check(head && head->head && head->hf == std::vector<std::string>{"lm_head.weight"}, "lm_head <- lm_head.weight (untied), the head layout set");
  bool heads_ok = true;
  for (const auto& l : rec.linear) heads_ok = heads_ok && (l.head == (l.base == "lm_head"));
  Check(heads_ok, "only lm_head takes the head layout set");
  std::map<std::string, std::string> b16(rec.bf16.begin(), rec.bf16.end());
  Check(b16["model.embed_tokens.weight"] == "text.embed_tokens" && b16["model.norm.weight"] == "text.final_norm" &&
            b16["model.layers.0.input_layernorm.weight"] == "text.layers.0.input_norm" &&
            b16["model.layers.51.post_attention_layernorm.weight"] == "text.layers.51.post_attn_norm",
        "bf16 container names: embed_tokens, final_norm, input_norm, post_attn_norm");

  // --layers: later layers are allow-listed, the head and embedding stay.
  {
    Recorder r2;
    r2.shapes = rec.shapes;
    ll::Kit k2 = r2.MakeKit();
    const std::set<std::string> c2 = ll::AddTextStack(k2, shape, 8);
    Check(ll::UnconsumedTensors(all, c2, 8).empty() && r2.linear.size() == 8 * 6 + 1 && r2.Find("text.layers.7.mlp.down") &&
              !r2.Find("text.layers.8.attn.q") && r2.Find("lm_head"),
          "--layers 8 (the nanbeige-l8 test container): layers 0-7 + embed / norm / head, the rest allow-listed");
    Check(ll::UnconsumedTensors(all, c2, 52).size() == 44 * 9, "audit with all layers expected but 8 converted names the other 396");
  }
  // --kv-calib (N5-2): with add_descale set, every layer registers k then v descales of kv_heads entries, and nothing
  // else changes; without it (above) none are registered.
  {
    Recorder r5;
    r5.shapes = rec.shapes;
    ll::Kit k5 = r5.MakeKit();
    std::vector<std::tuple<std::string, int, int, std::string>> d;
    k5.add_descale = [&d](const std::string& name, int n_kv, int layer, const char* kind) {
      d.emplace_back(name, n_kv, layer, kind);
    };
    const std::set<std::string> c5 = ll::AddTextStack(k5, shape, 52);
    bool ok = d.size() == 52 * 2 && c5.size() == 471 && r5.linear.size() == rec.linear.size();
    for (size_t i = 0; ok && i < d.size(); ++i) {
      const int layer = static_cast<int>(i / 2);
      const char* kind = i % 2 ? "v" : "k";
      ok = std::get<0>(d[i]) == "text.layers." + std::to_string(layer) + ".attn." + kind + "_descale" &&
           std::get<1>(d[i]) == 8 && std::get<2>(d[i]) == layer && std::get<3>(d[i]) == kind;
    }
    Check(ok, "--kv-calib: 104 descales (k, v per layer, 8 kv heads each), the tensor set otherwise unchanged");
  }
  // The audit.
  {
    Recorder r3;
    r3.shapes = rec.shapes;
    r3.shapes["model.layers.3.self_attn.mystery.weight"] = {4, 4};
    r3.shapes["model.layers.3.self_attn.rotary_emb.inv_freq"] = {64};
    r3.shapes["model.rotary_emb.inv_freq"] = {64};
    ll::Kit k3 = r3.MakeKit();
    const std::set<std::string> c3 = ll::AddTextStack(k3, shape, 52);
    Check(ll::UnconsumedTensors(k3.all_names(), c3, 52) ==
              std::vector<std::string>{"model.layers.3.self_attn.mystery.weight", "model.rotary_emb.inv_freq"},
          "audit: an unknown tensor is reported; a per-layer rotary_emb.inv_freq is allow-listed, a top-level one is not");
  }
  // Structural refusals, before any job is registered.
  {
    Recorder r4;
    r4.shapes = rec.shapes;
    r4.shapes.erase("model.layers.30.mlp.up_proj.weight");
    ll::Kit k4 = r4.MakeKit();
    Check(Throws([&] { ll::AddTextStack(k4, shape, 52); }, "model.layers.30.mlp.up_proj.weight"), "a missing tensor throws, naming it");
    Recorder r5;
    r5.shapes = rec.shapes;
    r5.shapes["model.layers.2.self_attn.k_proj.weight"] = {7168, 7168};
    ll::Kit k5 = r5.MakeKit();
    Check(Throws([&] { ll::AddTextStack(k5, shape, 52); }, "config says [1024,7168]"), "a k_proj with q's shape throws, naming both");
    Recorder r6;
    r6.shapes = rec.shapes;
    r6.shapes.erase("lm_head.weight");
    ll::Kit k6 = r6.MakeKit();
    Check(Throws([&] { ll::AddTextStack(k6, shape, 52); }, "lm_head.weight"), "a checkpoint without lm_head.weight throws");
    Check(Throws([&] { ll::AddTextStack(kit, shape, 53); }, "--layers"), "--layers beyond the checkpoint throws");
  }
  // Config refusals.
  {
    auto refuses = [&](const char* key, json value, const std::string& needle, const std::string& what) {
      json c = MakeConfig(kReal);
      c[key] = std::move(value);
      Check(Throws([&] { ll::ParseConfig(c); }, needle), what);
    };
    refuses("tie_word_embeddings", true, "tie_word_embeddings", "ParseConfig refuses a tied head");
    refuses("attention_bias", true, "attention_bias", "ParseConfig refuses attention_bias");
    refuses("mlp_bias", true, "mlp_bias", "ParseConfig refuses mlp_bias");
    refuses("rope_scaling", json{{"rope_type", "llama3"}, {"factor", 8.0}}, "rope_scaling", "ParseConfig refuses rope_scaling");
    refuses("hidden_act", "gelu", "hidden_act", "ParseConfig refuses a non-silu activation");
    refuses("pretraining_tp", 2, "pretraining_tp", "ParseConfig refuses pretraining_tp 2");
    refuses("model_type", "mistral", "model_type", "ParseConfig refuses another model_type");
    refuses("num_key_value_heads", 9, "multiple", "ParseConfig refuses heads not a multiple of kv heads");
    json c = MakeConfig(kReal);
    c.erase("intermediate_size");
    Check(Throws([&] { ll::ParseConfig(c); }, "intermediate_size"), "ParseConfig names a missing key");
    c = MakeConfig(kReal);
    c.erase("head_dim");
    Check(ll::ParseConfig(c).head_dim == 128, "no head_dim key: hidden / heads");
  }
}

// The real checkpoint's header bytes (no weights): names and (dtype, shape), every shard of the index.
void TestRealCheckpointHeader() {
  std::printf("---- (a2) the real checkpoint header ----\n");
  const fs::path dir = fs::path(r4dx::ModelsPath("Nanbeige/Nanbeige4.3-33B"));
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
  const json real_cfg = json::parse(ReadWhole(dir / "config.json"));
  Recorder rec;
  for (const auto& kv : real) rec.shapes[kv.first] = kv.second.second;
  ll::Kit kit = rec.MakeKit();
  const std::set<std::string> consumed = ll::AddTextStack(kit, ll::ParseConfig(real_cfg), 52);
  Check(ll::UnconsumedTensors(kit.all_names(), consumed, 52).empty(), "real header + real config.json: every tensor consumed");
}

// ---- (b) the exe on a tiny synthetic checkpoint -------------------------------------------------------------------

float Bf16Round(float x) { return r4dx::core::Bf16ToFloat(r4dx::core::FloatToBf16(x)); }

struct Tensor {
  std::string name, dtype;
  std::vector<int64_t> shape;
  std::vector<float> v;  // bf16-exact
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
  void Set(const std::string& name, std::vector<int64_t> shape) {
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    std::vector<float> v(static_cast<size_t>(n), 0.5f);
    const int i = Find(name);
    if (i >= 0) {
      t[static_cast<size_t>(i)] = {name, "BF16", std::move(shape), std::move(v)};
    } else {
      t.push_back({name, "BF16", std::move(shape), std::move(v)});
    }
  }
  void Erase(const std::string& name) {
    const int i = Find(name);
    if (i >= 0) t.erase(t.begin() + i);
  }
};

Ckpt MakeCkpt() {
  Ckpt c;
  std::mt19937_64 rng(0x11A3Aull);
  std::normal_distribution<double> nd(0.0, 1.0);
  for (const Spec& s : TensorSpecs(kTiny)) {
    int64_t n = 1;
    for (int64_t d : s.shape) n *= d;
    std::vector<float> v(static_cast<size_t>(n));
    const bool norm = s.name.find("norm") != std::string::npos;
    for (auto& x : v) x = Bf16Round(static_cast<float>(norm ? 1.0 + 0.2 * nd(rng) : 0.05 * nd(rng)));
    c.t.push_back({s.name, s.dtype, s.shape, std::move(v)});
  }
  c.config = MakeConfig(kTiny);
  c.config["max_position_embeddings"] = 4096;
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
        const uint16_t h = r4dx::core::FloatToBf16(x);
        data.append(reinterpret_cast<const char*>(&h), 2);
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
  const fs::path d = fs::temp_directory_path() / ("r4dx_test_llama_layout_" + tag);
  std::error_code ec;
  fs::remove_all(d, ec);
  fs::create_directories(d);
  return d;
}

// Every tensor name a conversion of the tiny checkpoint (first `layers` layers) must contain and nothing else; each
// linear base gets `suffixes`.
std::set<std::string> ExpectedNames(int64_t layers, const std::vector<std::string>& suffixes,
                                    const std::vector<std::string>& head_suffixes) {
  std::set<std::string> n = {"text.embed_tokens", "text.final_norm"};
  for (const auto& s : head_suffixes) n.insert("lm_head" + s);
  for (int64_t i = 0; i < layers; ++i) {
    const std::string b = "text.layers." + std::to_string(i) + ".";
    n.insert(b + "input_norm");
    n.insert(b + "post_attn_norm");
    for (const char* x : {"attn.q", "attn.k", "attn.v", "attn.o", "mlp.gate_up", "mlp.down"})
      for (const auto& s : suffixes) n.insert(b + x + s);
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
  std::printf("---- (b) r4dx-convert on a tiny synthetic llama checkpoint ----\n");
  const Geometry& g = kTiny;
  const int64_t H = g.hidden, Q = g.heads * g.head_dim, KV = g.kv * g.head_dim;
  const fs::path root = TempDir("exe");
  const fs::path ckpt = root / "ckpt-llama-tiny", out = root / "out.r4dx", log = root / "convert.log";
  const Ckpt base = MakeCkpt();
  WriteCkpt(base, ckpt);
  auto run = [&](const fs::path& in, const std::string& extra) {
    std::error_code ec;
    fs::remove(out, ec);
    return Run("--input \"" + in.u8string() + "\" --output \"" + out.u8string() + "\" --threads 2 " + extra, log);
  };
  auto log_has = [&](const std::string& s) { return ReadWhole(log).find(s) != std::string::npos; };
  const std::string bf16_args = "--layouts bf16 --lm-head bf16";

  // ---- the bf16 container (single-file checkpoint) ----
  {
    const int rc = run(ckpt, bf16_args);
    Check(rc == 0, "bf16 container: converts a single-file flat-config checkpoint (rc=" + std::to_string(rc) + ")");
    if (rc != 0) std::fprintf(stderr, "%s\n", ReadWhole(log).c_str());
    const Container c = ReadContainer(out);
    Check(c.ok, "bf16 container: readable");
    if (c.ok) {
      CheckNames(c, ExpectedNames(3, {".bf16.w"}, {".bf16.w"}), "bf16 container: exactly the plan's tensor set (no vision / mtp / descale)");
      const json md = c.Meta();
      Check(md.value("model_arch", "") == "llama" && md.value("norm_kind", "") == "plain", "metadata: model_arch llama, norm_kind plain");
      Check(md.value("model_id", "") == "ckpt-llama-tiny", "metadata: model_id is the checkpoint directory name");
      Check(md.at("model_config") == base.config, "metadata: model_config is the verbatim flat config.json");
      Check(md.at("r4dx_convert_run").value("arch", "") == "llama" && !md.at("r4dx_convert_run").value("mtp", true) &&
                !md.at("r4dx_convert_run").value("vision", true),
            "metadata: r4dx_convert_run arch llama, mtp / vision off");
      Check(!md.contains("rotation"), "metadata: no rotation block");

      Check(c.Shape("text.embed_tokens") == std::vector<int64_t>({g.vocab, H, 2}) &&
                c.Shape("text.final_norm") == std::vector<int64_t>({H, 2}) &&
                c.Shape("text.layers.0.input_norm") == std::vector<int64_t>({H, 2}),
            "embed [vocab, hidden], norms [hidden] (bf16)");
      Check(c.Shape("text.layers.1.attn.q.bf16.w") == std::vector<int64_t>({Q, H, 2}) &&
                c.Shape("text.layers.1.attn.k.bf16.w") == std::vector<int64_t>({KV, H, 2}) &&
                c.Shape("text.layers.1.attn.o.bf16.w") == std::vector<int64_t>({H, Q, 2}) &&
                c.Shape("text.layers.1.mlp.gate_up.bf16.w") == std::vector<int64_t>({2 * g.inter, H, 2}) &&
                c.Shape("text.layers.1.mlp.down.bf16.w") == std::vector<int64_t>({H, g.inter, 2}) &&
                c.Shape("lm_head.bf16.w") == std::vector<int64_t>({g.vocab, H, 2}),
            "linear shapes: q [3584, 512] (head_dim key honoured), k [512, 512], o [512, 3584], gate_up [2I, H], head [vocab, H]");

      Check(c.tensors.at("text.embed_tokens") == BytesOf(base.Get("model.embed_tokens.weight")), "embed_tokens bytes == checkpoint");
      Check(c.tensors.at("lm_head.bf16.w") == BytesOf(base.Get("lm_head.weight")) &&
                c.tensors.at("lm_head.bf16.w") != c.tensors.at("text.embed_tokens"),
            "lm_head bytes == lm_head.weight (untied: not the embedding)");
      Check(c.tensors.at("text.final_norm") == BytesOf(base.Get("model.norm.weight")), "final_norm bytes == model.norm");
      Check(c.tensors.at("text.layers.2.post_attn_norm") == BytesOf(base.Get("model.layers.2.post_attention_layernorm.weight")),
            "post_attn_norm bytes == post_attention_layernorm");
      Check(c.tensors.at("text.layers.2.attn.o.bf16.w") == BytesOf(base.Get("model.layers.2.self_attn.o_proj.weight")), "attn.o bytes == o_proj");
      Check(c.tensors.at("text.layers.0.mlp.gate_up.bf16.w") ==
                BytesOf(base.Get("model.layers.0.mlp.gate_proj.weight")) + BytesOf(base.Get("model.layers.0.mlp.up_proj.weight")),
            "gate_up bytes == gate_proj rows then up_proj rows");
      Check(c.tensors.at("text.layers.0.mlp.down.bf16.w") == BytesOf(base.Get("model.layers.0.mlp.down_proj.weight")), "down bytes == down_proj");
    }
  }

  // ---- a sharded checkpoint (14 shards + index in the real one) converts to the same container ----
  {
    const Container ref = ReadContainer(out);
    const fs::path sh = root / "ckpt-llama-sharded";
    WriteCkpt(base, sh, 4);
    const int rc = run(sh, bf16_args);
    const Container c = ReadContainer(out);
    Check(rc == 0 && c.ok && c.tensors == ref.tensors, "sharded checkpoint (index.json): the same container bytes as the single file");
  }

  // ---- w8a16 (the TP=2 shipping format): every linear and the head by default ----
  {
    const int rc = run(ckpt, "--layouts w8a16 --no-bf16");
    const Container c = ReadContainer(out);
    Check(rc == 0 && c.ok, "--layouts w8a16 --no-bf16: converts (rc=" + std::to_string(rc) + ")");
    if (rc != 0) std::fprintf(stderr, "%s\n", ReadWhole(log).c_str());
    if (c.ok) {
      CheckNames(c, ExpectedNames(3, {".w8a16.wq", ".w8a16.ws"}, {".w8a16.wq", ".w8a16.ws"}),
                 "w8a16: every linear and the head (by default) is a .w8a16.wq/.ws pair; norms / embed bf16");
      Check(c.Meta().at("quant").contains("w8a16") && c.Meta().at("quant").at("w8a16").at("group") == 128,
            "metadata: quant.w8a16 group 128");
      Check(c.Meta().at("r4dx_convert_run").at("lm_head").get<std::string>() == "w8a16+bf16", "run record: the derived head spec");
      struct Probe {
        std::string cb;
        std::vector<std::string> hf;
      };
      const std::vector<Probe> probes = {
          {"text.layers.1.attn.q", {"model.layers.1.self_attn.q_proj.weight"}},
          {"text.layers.2.attn.o", {"model.layers.2.self_attn.o_proj.weight"}},
          {"text.layers.0.mlp.gate_up", {"model.layers.0.mlp.gate_proj.weight", "model.layers.0.mlp.up_proj.weight"}},
          {"text.layers.2.mlp.down", {"model.layers.2.mlp.down_proj.weight"}},
          {"lm_head", {"lm_head.weight"}}};
      for (const Probe& p : probes) {
        std::vector<float> w;
        int K = 0;
        for (const auto& n : p.hf) {
          const Tensor& t = base.t.at(static_cast<size_t>(base.Find(n)));
          K = static_cast<int>(t.shape[1]);
          w.insert(w.end(), t.v.begin(), t.v.end());
        }
        const int N = static_cast<int>(w.size() / static_cast<size_t>(K));
        std::vector<uint8_t> q;
        std::vector<uint16_t> s16;
        r4dx_convert::QuantizeInt8Symmetric(w.data(), N, K, 128, 1, q, s16);
        const std::vector<uint8_t> wq = r4dx_convert::PackW8Bytes(q, N, K, 1);
        const auto s = r4dx_convert::PackW8a16Scales(s16, N, K, 128);
        const std::string ws(reinterpret_cast<const char*>(s.data()), s.size() * 2);
        Check(c.tensors.at(p.cb + ".w8a16.wq") == std::string(wq.begin(), wq.end()) && c.tensors.at(p.cb + ".w8a16.ws") == ws,
              "w8a16 bytes of " + p.cb + " == the quantizer on the checkpoint weights");
      }
    }
  }
  // ---- --lm-head bf16 with a w8a16 body ----
  {
    const int rc = run(ckpt, "--layouts w8a16 --no-bf16 --lm-head bf16");
    const Container c = ReadContainer(out);
    Check(rc == 0 && c.ok, "--lm-head bf16 with a w8a16 body: converts");
    if (c.ok)
      CheckNames(c, ExpectedNames(3, {".w8a16.wq", ".w8a16.ws"}, {".bf16.w"}), "w8a16 body + --lm-head bf16: the head is lm_head.bf16.w only");
  }
  // ---- w4a16 ----
  {
    const int rc = run(ckpt, "--layouts w4a16 --lm-head 4bit");
    const Container c = ReadContainer(out);
    Check(rc == 0 && c.ok, "--layouts w4a16: converts (rc=" + std::to_string(rc) + ")");
    if (c.ok)
      Check(c.tensors.count("text.layers.2.attn.v.w4a16.wq") == 1 && c.tensors.count("text.layers.0.mlp.down.w4a16.wq") == 1 &&
                c.tensors.count("lm_head.w4a16.wq") == 1 && c.tensors.count("text.layers.0.input_norm") == 1 &&
                c.tensors.count("text.layers.0.input_norm.w4a16.wq") == 0,
            "w4a16: the linears and the head carry .w4a16.wq, the norms stay bf16");
  }

  // ---- --layers 2 ----
  {
    const int rc = run(ckpt, bf16_args + " --layers 2");
    const Container c = ReadContainer(out);
    Check(rc == 0 && c.ok, "--layers 2: converts");
    if (c.ok) {
      CheckNames(c, ExpectedNames(2, {".bf16.w"}, {".bf16.w"}), "--layers 2: layers 0-1 + embed / final norm / head");
      Check(c.Meta().at("r4dx_convert_run").value("layers_converted", 0) == 2, "--layers 2: layers_converted 2 of 3");
    }
  }

  // ---- refusals ----
  {
    Check(run(ckpt, bf16_args + " --vision on") != 0 && log_has("--vision on"), "--vision on is refused");
    Check(run(ckpt, bf16_args + " --mtp on") != 0 && log_has("--mtp on"), "--mtp on is refused");
    Check(run(ckpt, bf16_args + " --rotate q2a") != 0 && log_has("--rotate"), "--rotate q2a is refused");
  }
  // ---- --kv-calib (N5-2): fp32 [kv_heads] descales = amax / 448 per layer; without it there are none ----
  {
    const fs::path calib = root / "kvcalib.json";
    json j;
    for (int l = 0; l < 3; ++l) {
      std::vector<float> k(4), v(4);
      for (int h = 0; h < 4; ++h) {
        k[h] = 10.0f * (l + 1) + h;
        v[h] = 0.5f * (l + 1) + 0.25f * h;
      }
      j[std::to_string(l)] = {{"k_amax", k}, {"v_amax", v}};
    }
    std::ofstream(calib) << j.dump();
    const int rc = run(ckpt, bf16_args + " --kv-calib \"" + calib.string() + "\"");
    const Container c = ReadContainer(out);
    Check(rc == 0 && c.ok, "--kv-calib: converts");
    if (c.ok) {
      bool ok = true;
      for (int l = 0; l < 3 && ok; ++l) {
        for (const char* kind : {"k", "v"}) {
          const std::string n = "text.layers." + std::to_string(l) + ".attn." + kind + "_descale";
          ok = ok && c.tensors.count(n) == 1 && c.tensors.at(n).size() == 16;
          if (!ok) break;
          const std::vector<float> want = j[std::to_string(l)][std::string(kind) + "_amax"].get<std::vector<float>>();
          for (int h = 0; h < 4 && ok; ++h) {
            float got;
            std::memcpy(&got, c.tensors.at(n).data() + 4 * h, 4);
            ok = got == want[h] / 448.0f;
          }
        }
      }
      Check(ok, "--kv-calib: every layer carries attn.k_descale / v_descale = amax / 448, fp32 [4]");
    }
    const int rc2 = run(ckpt, bf16_args);
    const Container c2 = ReadContainer(out);
    Check(rc2 == 0 && c2.ok && c2.tensors.count("text.layers.0.attn.k_descale") == 0,
          "without --kv-calib: no descale tensors (the container is unchanged)");
  }
  {

    const fs::path bad = root / "ckpt-bad";
    auto refuses = [&](const Ckpt& c, const std::string& needle, const std::string& what) {
      std::error_code ec;
      fs::remove_all(bad, ec);
      WriteCkpt(c, bad);
      Check(run(bad, bf16_args) != 0 && log_has(needle), what);
    };
    {
      Ckpt c = base;
      c.Erase("model.layers.1.mlp.up_proj.weight");
      refuses(c, "model.layers.1.mlp.up_proj.weight", "a missing tensor fails the run, naming it");
    }
    {
      Ckpt c = base;
      c.Set("model.layers.0.self_attn.mystery.weight", {4, 4});
      refuses(c, "model.layers.0.self_attn.mystery.weight", "the coverage audit names an unconsumed tensor");
    }
    {
      Ckpt c = base;
      c.Set("model.layers.0.self_attn.k_proj.weight", {Q, H});
      refuses(c, "config says", "a wrong shape fails the run, naming both shapes");
    }
    {
      Ckpt c = base;
      c.config["tie_word_embeddings"] = true;
      refuses(c, "tie_word_embeddings", "a tied config is refused");
    }
    {
      Ckpt c = base;
      c.config["attention_bias"] = true;
      refuses(c, "attention_bias", "attention_bias is refused");
    }
    {
      Ckpt c = base;
      c.config["rope_scaling"] = {{"rope_type", "llama3"}, {"factor", 8.0}};
      refuses(c, "rope_scaling", "rope_scaling is refused");
    }
    {
      Ckpt c = base;
      c.Set("model.layers.1.self_attn.rotary_emb.inv_freq", {64});
      std::error_code ec;
      fs::remove_all(bad, ec);
      WriteCkpt(c, bad);
      Check(run(bad, bf16_args) == 0, "a per-layer rotary_emb.inv_freq is allow-listed (the runtime computes the rope)");
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
