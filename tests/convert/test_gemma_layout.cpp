// tests/convert/test_gemma_layout.cpp -- CPU-only (ctest `convert_gemma_layout`). docs/gemma4-plan.md M1-6.
//
// (a) gemma_layout.cpp against the REAL Huihui checkpoint header (tools/reference/gemma/tensor_names.json:
//     677 tensors, names and shapes) through a recording Kit, no weights: every tensor is consumed or
//     allow-listed, the container names / fold sites / per-layer descale head counts (8 sliding, 1 full)
//     are the plan's 4.2 table, no v on the 8 full layers, and the coverage audit names a tensor that
//     is not consumed.
// (b) r4dx-convert (when built) on a tiny synthetic gemma4_unified checkpoint -- one sliding + one full
//     layer, hidden 768, the real tensor names including layer_scalar and no v_proj on the full layer, a
//     single model.safetensors without an index: the exact container tensor set, shapes and dtypes,
//     bf16 byte identity (embed, tied lm_head, gate|up fusion), layer_scalar widening, k/v descale from
//     --kv-calib, the vision/audio passthrough, --layers, --rotate q2ab (Hadamard-only o / down,
//     folded in-norms stored as ones, rotation.* tensors and metadata, the folded q and o bytes against
//     an independent fold), and every refusal (an unconsumed tensor, a wrong shape, a missing
//     layer_scalar, v_proj on a k_eq_v layer, --mtp on, --trellis-from, an untied head).
// The Qwen byte-identity guards are the existing convert_* tests; this file never runs a Qwen
// conversion.
#include <algorithm>
#include <cmath>
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
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "gemma_layout.hpp"
#include "nlohmann/json.hpp"
#include "r4dx/core/dtype.hpp"
#include "r4dx_convert/rotation.hpp"

namespace fs = std::filesystem;
namespace gm = r4dx_convert::gemma;
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
    return substr.empty() || std::string(e.what()).find(substr) != std::string::npos;
  }
  return false;
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

// ---- (a) the real checkpoint header through a recording kit -------------------------------------------

json RealTextConfig() {
  json types = json::array();
  for (int i = 0; i < 48; ++i) types.push_back((i + 1) % 6 == 0 ? "full_attention" : "sliding_attention");
  return {{"hidden_size", 3840},          {"num_hidden_layers", 48},   {"layer_types", types},
          {"num_attention_heads", 16},    {"num_key_value_heads", 8},  {"num_global_key_value_heads", 1},
          {"head_dim", 256},              {"global_head_dim", 512},    {"intermediate_size", 15360},
          {"vocab_size", 262144},         {"attention_k_eq_v", true},  {"tie_word_embeddings", true}};
}

struct Recorder {
  std::map<std::string, std::vector<int64_t>> shapes;
  std::vector<std::pair<std::string, std::string>> bf16, folded, widen;
  struct Descale {
    std::string name;
    int n_kv, layer;
    std::string kind;
  };
  std::vector<Descale> descale;
  struct Linear {
    std::vector<std::string> hf;
    std::string base;
    gm::Fold fold;
  };
  std::vector<Linear> linear;

  gm::Kit MakeKit() {
    gm::Kit kit;
    kit.all_names = [this]() {
      std::vector<std::string> n;
      for (const auto& kv : shapes) n.push_back(kv.first);
      return n;
    };
    kit.has = [this](const std::string& n) { return shapes.count(n) != 0; };
    kit.shape = [this](const std::string& n) { return shapes.at(n); };
    kit.add_bf16 = [this](const std::string& a, const std::string& b) { bf16.push_back({a, b}); };
    kit.add_folded_norm = [this](const std::string& a, const std::string& b) { folded.push_back({a, b}); };
    kit.add_fp32_widen = [this](const std::string& a, const std::string& b) { widen.push_back({a, b}); };
    kit.add_descale = [this](const std::string& n, int n_kv, int layer, const char* kind) {
      descale.push_back({n, n_kv, layer, kind});
    };
    kit.add_linear = [this](const std::vector<std::string>& hf, const std::string& base, const gm::Fold& f) {
      linear.push_back({hf, base, f});
    };
    return kit;
  }
  const Linear* Find(const std::string& base) const {
    for (const auto& l : linear)
      if (l.base == base) return &l;
    return nullptr;
  }
};

void TestRealHeader() {
  std::printf("---- (a) gemma_layout.cpp over the real 677-tensor header ----\n");
  const fs::path path = fs::path(R4DX_SOURCE_DIR) / "tools" / "reference" / "gemma" / "tensor_names.json";
  const json j = json::parse(ReadWhole(path));
  Recorder rec;
  for (auto it = j.at("tensors").begin(); it != j.at("tensors").end(); ++it)
    rec.shapes[it.key()] = it.value().at("shape").get<std::vector<int64_t>>();
  Check(rec.shapes.size() == 677, "tensor_names.json: 677 tensors");

  const gm::TextShape shape = gm::ParseTextConfig(RealTextConfig());
  Check(shape.layers_total == 48 && shape.HeadDim(0) == 256 && shape.HeadDim(5) == 512 && shape.KvHeads(0) == 8 &&
            shape.KvHeads(5) == 1,
        "ParseTextConfig: the real geometry");

  gm::Kit kit = rec.MakeKit();
  const std::set<std::string> consumed = gm::AddTextStack(kit, shape, 48, /*vision=*/false, /*audio=*/false);
  const std::vector<std::string> all = kit.all_names();
  Check(gm::UnconsumedTensors(all, consumed, 48, false, false).empty(),
        "real header: every tensor is consumed or allow-listed (vision/audio off)");
  Check(rec.linear.size() == 281, "281 linears: 40 sliding x 6 + 8 full x 5 + lm_head");
  Check(rec.descale.size() == 96 && rec.folded.size() == 96 && rec.widen.size() == 48 && rec.bf16.size() == 194,
        "96 descale + 96 folded norms + 48 layer_scalar + 194 bf16 (post norms, q/k norms, embed, final norm)");
  Check(consumed.size() == 677 - 11, "consumed = 666 text tensors (the 11 vision/audio ones are allow-listed)");

  // Names, fold sites, descale heads.
  const std::string L0 = "model.language_model.layers.0.", L5 = "model.language_model.layers.5.";
  Check(rec.Find("text.layers.5.attn.v") == nullptr && rec.Find("text.layers.0.attn.v") != nullptr,
        "v exists on sliding layers, not on the 8 full ones");
  bool no_v_full = true;
  for (int i = 5; i < 48; i += 6) no_v_full = no_v_full && rec.Find("text.layers." + std::to_string(i) + ".attn.v") == nullptr;
  Check(no_v_full, "no attn.v at 5, 11, ..., 47");
  const auto* q0 = rec.Find("text.layers.0.attn.q");
  Check(q0 && q0->hf == std::vector<std::string>{L0 + "self_attn.q_proj.weight"} &&
            q0->fold.in_norm_hf == L0 + "input_layernorm.weight" && q0->fold.out == gm::OutSite::kNone && !q0->fold.head,
        "attn.q <- q_proj, folds input_layernorm");
  const auto* gu = rec.Find("text.layers.5.mlp.gate_up");
  Check(gu && gu->hf == std::vector<std::string>{L5 + "mlp.gate_proj.weight", L5 + "mlp.up_proj.weight"} &&
            gu->fold.in_norm_hf == L5 + "pre_feedforward_layernorm.weight",
        "mlp.gate_up <- gate rows then up rows, folds pre_feedforward_layernorm");
  Check(rec.Find("text.layers.0.attn.o")->fold.out == gm::OutSite::kOSliding &&
            rec.Find("text.layers.5.attn.o")->fold.out == gm::OutSite::kOFull &&
            rec.Find("text.layers.5.mlp.down")->fold.out == gm::OutSite::kDown,
        "out-projection sites: o_swa on sliding, o_full on full, down");
  const auto* head = rec.Find("lm_head");
  Check(head && head->fold.head && head->hf == std::vector<std::string>{"model.language_model.embed_tokens.weight"},
        "lm_head <- the tied embedding, the head layout set");
  bool kv_ok = true;
  for (const auto& d : rec.descale)
    kv_ok = kv_ok && d.n_kv == ((d.layer % 6 == 5) ? 1 : 8) && (d.kind == "k" || d.kind == "v") &&
            d.name == "text.layers." + std::to_string(d.layer) + ".attn." + d.kind + "_descale";
  Check(kv_ok, "descale: per-layer kv heads, 8 sliding / 1 full, both k and v on every layer");
  bool names_ok = true;
  std::set<std::string> expect_bf16 = {"text.embed_tokens", "text.final_norm"};
  for (int i = 0; i < 48; ++i) {
    const std::string b = "text.layers." + std::to_string(i) + ".";
    for (const char* n : {"post_attention_layernorm", "post_feedforward_layernorm", "attn.q_norm", "attn.k_norm"})
      expect_bf16.insert(b + n);
  }
  std::set<std::string> got_bf16;
  for (const auto& p : rec.bf16) got_bf16.insert(p.second);
  names_ok = got_bf16 == expect_bf16;
  Check(names_ok, "bf16 container names");
  std::set<std::string> got_folded;
  for (const auto& p : rec.folded) got_folded.insert(p.second);
  Check(got_folded.count("text.layers.47.input_layernorm") && got_folded.count("text.layers.0.pre_feedforward_layernorm") &&
            got_folded.size() == 96,
        "folded-norm container names");

  // Vision / audio passthrough.
  {
    Recorder r2;
    r2.shapes = rec.shapes;
    gm::Kit k2 = r2.MakeKit();
    const std::set<std::string> c2 = gm::AddTextStack(k2, shape, 48, true, true);
    Check(c2.size() == 677 && gm::UnconsumedTensors(all, c2, 48, true, true).empty(), "vision+audio on: all 677 consumed");
    std::set<std::string> vn;
    for (const auto& p : r2.bf16)
      if (p.second.rfind("vision.", 0) == 0 || p.second.rfind("audio.", 0) == 0) vn.insert(p.second);
    Check(vn.size() == 11 && vn.count("vision.vision_embedder.patch_dense.weight") &&
              vn.count("vision.embed_vision.embedding_projection.weight") &&
              vn.count("audio.embed_audio.embedding_projection.weight"),
          "passthrough names: vision.vision_embedder.*, vision.embed_vision.*, audio.embed_audio.*");
    Recorder r3;
    r3.shapes = rec.shapes;
    gm::Kit k3 = r3.MakeKit();
    const std::set<std::string> c3 = gm::AddTextStack(k3, shape, 48, true, false);
    Check(gm::UnconsumedTensors(all, c3, 48, true, false).empty() &&
              c3.count("model.embed_audio.embedding_projection.weight") == 0 && c3.size() == 677 - 1,
          "vision on, audio off: the audio tensor is allow-listed (not consumed), nothing else is left over");
    // ... and the allow-list is what hides it: without it the audit names the tensor.
    Check(gm::UnconsumedTensors(all, c3, 48, true, true) ==
              std::vector<std::string>{"model.embed_audio.embedding_projection.weight"},
          "audit with audio expected but not converted names the audio tensor");
  }

  // --layers: later layers are allow-listed; vision/audio still count.
  {
    Recorder r4;
    r4.shapes = rec.shapes;
    gm::Kit k4 = r4.MakeKit();
    const std::set<std::string> c4 = gm::AddTextStack(k4, shape, 6, false, false);
    Check(gm::UnconsumedTensors(all, c4, 6, false, false).empty() && r4.linear.size() == 6 * 6 - 1 + 1 &&
              r4.Find("text.layers.5.attn.q") != nullptr && r4.Find("text.layers.6.attn.q") == nullptr,
          "--layers 6: layers 6+ allow-listed, 5 sliding x 6 + 1 full x 5 + lm_head linears");
  }

  // The audit: a checkpoint tensor nobody consumed is named.
  {
    Recorder r5;
    r5.shapes = rec.shapes;
    r5.shapes["model.language_model.layers.3.self_attn.mystery.weight"] = {4, 4};
    r5.shapes["lm_head.weight"] = {262144, 3840};
    gm::Kit k5 = r5.MakeKit();
    const std::set<std::string> c5 = gm::AddTextStack(k5, shape, 48, false, false);
    const std::vector<std::string> bad = gm::UnconsumedTensors(k5.all_names(), c5, 48, false, false);
    Check(bad == std::vector<std::string>{"lm_head.weight", "model.language_model.layers.3.self_attn.mystery.weight"},
          "audit: an unknown tensor and a stray lm_head.weight are reported by name");
  }

  // Structural refusals, before any job is registered.
  {
    Recorder r6;
    r6.shapes = rec.shapes;
    r6.shapes[L5 + "self_attn.v_proj.weight"] = {512, 3840};
    gm::Kit k6 = r6.MakeKit();
    Check(Throws([&] { gm::AddTextStack(k6, shape, 48, false, false); }, "attention_k_eq_v"),
          "v_proj on a full layer throws");
    Recorder r7;
    r7.shapes = rec.shapes;
    r7.shapes.erase(L0 + "self_attn.v_proj.weight");
    gm::Kit k7 = r7.MakeKit();
    Check(Throws([&] { gm::AddTextStack(k7, shape, 48, false, false); }, "no self_attn.v_proj.weight"),
          "a sliding layer without v_proj throws");
    Recorder r8;
    r8.shapes = rec.shapes;
    r8.shapes[L0 + "self_attn.k_proj.weight"] = {512, 3840};  // a full layer's k rows on a sliding layer
    gm::Kit k8 = r8.MakeKit();
    Check(Throws([&] { gm::AddTextStack(k8, shape, 48, false, false); }, "config says [2048,3840]"),
          "a sliding k_proj with the full layer's shape throws, naming both shapes");
    Recorder r9;
    r9.shapes = rec.shapes;
    r9.shapes.erase(L5 + "layer_scalar");
    gm::Kit k9 = r9.MakeKit();
    Check(Throws([&] { gm::AddTextStack(k9, shape, 48, false, false); }, "layer_scalar"), "a missing layer_scalar throws");
    Check(Throws([&] { gm::AddTextStack(kit, shape, 49, false, false); }, "--layers"), "--layers beyond the checkpoint throws");
  }

  // Config refusals.
  {
    json c = RealTextConfig();
    c["attention_k_eq_v"] = false;
    Check(Throws([&] { gm::ParseTextConfig(c); }, "attention_k_eq_v"), "ParseTextConfig refuses k != v");
    c = RealTextConfig();
    c["tie_word_embeddings"] = false;
    Check(Throws([&] { gm::ParseTextConfig(c); }, "tie_word_embeddings"), "ParseTextConfig refuses an untied head");
    c = RealTextConfig();
    c["layer_types"][0] = "linear_attention";
    Check(Throws([&] { gm::ParseTextConfig(c); }, "linear_attention"), "ParseTextConfig refuses a foreign layer type");
    c = RealTextConfig();
    c.erase("global_head_dim");
    Check(Throws([&] { gm::ParseTextConfig(c); }, "global_head_dim"), "ParseTextConfig names a missing key");
  }
}

// ---- (b) the exe on a tiny synthetic checkpoint -------------------------------------------------------

constexpr int64_t kHidden = 768;   // rotation block 256 x 3
constexpr int64_t kHeads = 4, kKvS = 2, kKvF = 1;
constexpr int64_t kHdS = 64, kHdF = 128;  // q2ab: attn.o's K is a multiple of the 256 Hadamard block
constexpr int64_t kInter = 512;           // q2ab: mlp.down's K is the 512 Hadamard block
constexpr int64_t kVocab = 64;

float Bf16Round(float x) { return r4dx::core::Bf16ToFloat(r4dx::core::FloatToBf16(x)); }

struct Tensor {
  std::string name;
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
      t[static_cast<size_t>(i)].shape = std::move(shape);
      t[static_cast<size_t>(i)].v = std::move(v);
    } else {
      t.push_back({name, std::move(shape), std::move(v)});
    }
  }
  void Erase(const std::string& name) {
    const int i = Find(name);
    if (i >= 0) t.erase(t.begin() + i);
  }
};

Ckpt MakeCkpt() {
  Ckpt c;
  std::mt19937_64 rng(0x6E3A4ull);
  std::normal_distribution<double> nd(0.0, 1.0);
  auto rnd = [&](int64_t n, double sigma) {
    std::vector<float> v(static_cast<size_t>(n));
    for (auto& x : v) x = Bf16Round(static_cast<float>(sigma * nd(rng)));
    return v;
  };
  auto add = [&](const std::string& name, std::vector<int64_t> shape, double sigma) {
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    c.t.push_back({name, std::move(shape), rnd(n, sigma)});
  };
  auto norm_w = [&](int64_t n) {  // plain weights around 1
    std::vector<float> v(static_cast<size_t>(n));
    for (auto& x : v) x = Bf16Round(static_cast<float>(1.0 + 0.2 * nd(rng)));
    return v;
  };
  for (int i = 0; i < 2; ++i) {
    const bool full = i == 1;
    const std::string L = "model.language_model.layers." + std::to_string(i) + ".";
    const int64_t hd = full ? kHdF : kHdS, kv = full ? kKvF : kKvS;
    for (const char* n : {"input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm",
                          "post_feedforward_layernorm"})
      c.t.push_back({L + n + ".weight", {kHidden}, norm_w(kHidden)});
    add(L + "self_attn.q_proj.weight", {kHeads * hd, kHidden}, 0.05);
    add(L + "self_attn.k_proj.weight", {kv * hd, kHidden}, 0.05);
    if (!full) add(L + "self_attn.v_proj.weight", {kv * hd, kHidden}, 0.05);
    add(L + "self_attn.o_proj.weight", {kHidden, kHeads * hd}, 0.05);
    c.t.push_back({L + "self_attn.q_norm.weight", {hd}, norm_w(hd)});
    c.t.push_back({L + "self_attn.k_norm.weight", {hd}, norm_w(hd)});
    add(L + "mlp.gate_proj.weight", {kInter, kHidden}, 0.05);
    add(L + "mlp.up_proj.weight", {kInter, kHidden}, 0.05);
    add(L + "mlp.down_proj.weight", {kHidden, kInter}, 0.05);
    c.t.push_back({L + "layer_scalar", {1}, {Bf16Round(i == 0 ? 0.375f : 0.8125f)}});
  }
  add("model.language_model.embed_tokens.weight", {kVocab, kHidden}, 0.1);
  c.t.push_back({"model.language_model.norm.weight", {kHidden}, norm_w(kHidden)});
  // The non-text tensors (real names, tiny shapes): the converter copies them without looking at shape.
  add("model.vision_embedder.patch_ln1.weight", {16}, 1.0);
  add("model.vision_embedder.patch_dense.weight", {8, 16}, 1.0);
  add("model.vision_embedder.pos_embedding", {4, 2, 8}, 1.0);
  add("model.embed_vision.embedding_projection.weight", {8, 8}, 1.0);
  add("model.embed_audio.embedding_projection.weight", {8, 4}, 1.0);

  c.config = {{"architectures", {"Gemma4UnifiedForConditionalGeneration"}},
              {"model_type", "gemma4_unified"},
              {"eos_token_id", {1, 106}},
              {"text_config",
               {{"model_type", "gemma4_unified_text"},
                {"hidden_size", kHidden},
                {"num_hidden_layers", 2},
                {"layer_types", {"sliding_attention", "full_attention"}},
                {"num_attention_heads", kHeads},
                {"num_key_value_heads", kKvS},
                {"num_global_key_value_heads", kKvF},
                {"head_dim", kHdS},
                {"global_head_dim", kHdF},
                {"intermediate_size", kInter},
                {"vocab_size", kVocab},
                {"attention_k_eq_v", true},
                {"tie_word_embeddings", true}}}};
  return c;
}

void WriteCkpt(const Ckpt& c, const fs::path& dir) {
  fs::create_directories(dir);
  json hdr = json::object();
  std::string data;
  for (const auto& t : c.t) {
    const uint64_t b = data.size();
    for (float x : t.v) {
      const uint16_t h = r4dx::core::FloatToBf16(x);
      data.append(reinterpret_cast<const char*>(&h), 2);
    }
    hdr[t.name] = {{"dtype", "BF16"}, {"shape", t.shape}, {"data_offsets", {b, data.size()}}};
  }
  const std::string h = hdr.dump();
  const uint64_t hl = h.size();
  std::string file(reinterpret_cast<const char*>(&hl), 8);
  file += h;
  file += data;
  WriteWhole(dir / "model.safetensors", file);  // a single file, NO model.safetensors.index.json
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
  std::string BytesOf(const std::vector<float>& v) const {
    std::string b;
    for (float x : v) {
      const uint16_t h = r4dx::core::FloatToBf16(x);
      b.append(reinterpret_cast<const char*>(&h), 2);
    }
    return b;
  }
};

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
  const fs::path d = fs::temp_directory_path() / ("r4dx_test_gemma_layout_" + tag);
  std::error_code ec;
  fs::remove_all(d, ec);
  fs::create_directories(d);
  return d;
}

// Every tensor name a bf16-only conversion of the tiny checkpoint must contain (and nothing else).
std::set<std::string> ExpectedNames(int layers, bool vision, bool audio, bool rotated) {
  std::set<std::string> n = {"text.embed_tokens", "text.final_norm", "lm_head.bf16.w"};
  for (int i = 0; i < layers; ++i) {
    const std::string b = "text.layers." + std::to_string(i) + ".";
    const bool full = i == 1;
    const std::string rot = rotated ? ".rotated" : "";
    for (const char* x : {"post_attention_layernorm", "post_feedforward_layernorm", "attn.q_norm", "attn.k_norm",
                          "attn.k_descale", "attn.v_descale", "layer_scalar"})
      n.insert(b + x);
    n.insert(b + "input_layernorm" + rot);
    n.insert(b + "pre_feedforward_layernorm" + rot);
    for (const char* x : {"attn.q", "attn.k", "attn.o", "mlp.gate_up", "mlp.down"}) n.insert(b + x + std::string(".bf16.w"));
    if (!full) n.insert(b + "attn.v.bf16.w");
  }
  if (vision)
    for (const char* x : {"patch_ln1.weight", "patch_dense.weight", "pos_embedding"})
      n.insert(std::string("vision.vision_embedder.") + x);
  if (vision) n.insert("vision.embed_vision.embedding_projection.weight");
  if (audio) n.insert("audio.embed_audio.embedding_projection.weight");
  if (rotated)
    for (const char* x : {"rotation.signs", "rotation.mix", "rotation.had_down_signs", "rotation.had_o_signs",
                          "rotation.had_o_full_signs"})
      n.insert(x);
  return n;
}

std::string Join(const std::set<std::string>& s) {
  std::string out;
  for (const auto& x : s) out += x + " ";
  return out;
}

void TestExe() {
  std::printf("---- (b) r4dx-convert on a tiny synthetic gemma4_unified checkpoint ----\n");
  const fs::path root = TempDir("exe");
  const fs::path ckpt = root / "ckpt-gemma-tiny", out = root / "out.r4dx", log = root / "convert.log";
  const Ckpt base = MakeCkpt();
  WriteCkpt(base, ckpt);
  const std::string common = " --threads 2 --layouts bf16 --lm-head bf16 ";
  auto convert = [&](const fs::path& in, const std::string& extra) {
    std::error_code ec;
    fs::remove(out, ec);
    return Run("--input \"" + in.u8string() + "\" --output \"" + out.u8string() + "\"" + common + extra, log);
  };
  auto log_has = [&](const std::string& s) { return ReadWhole(log).find(s) != std::string::npos; };

  // ---- the default bf16 container ----
  {
    const int rc = convert(ckpt, "");
    Check(rc == 0, "bf16 container: converts a single-file checkpoint (rc=" + std::to_string(rc) + ")");
    const Container c = ReadContainer(out);
    Check(c.ok, "bf16 container: readable");
    if (c.ok) {
      const std::set<std::string> want = ExpectedNames(2, false, false, false);
      const std::vector<std::string> names = c.Names();
      const std::set<std::string> got(names.begin(), names.end());
      Check(got == want, "bf16 container: exactly the plan's tensor set (vision/audio off by default)");
      if (got != want) {
        std::set<std::string> extra, missing;
        std::set_difference(got.begin(), got.end(), want.begin(), want.end(), std::inserter(extra, extra.end()));
        std::set_difference(want.begin(), want.end(), got.begin(), got.end(), std::inserter(missing, missing.end()));
        std::fprintf(stderr, "  extra: %s\n  missing: %s\n", Join(extra).c_str(), Join(missing).c_str());
      }
      const json md = c.Meta();
      Check(md.value("model_arch", "") == "gemma4_unified" && md.value("norm_kind", "") == "plain",
            "metadata: model_arch gemma4_unified, norm_kind plain");
      Check(md.value("model_id", "") == "ckpt-gemma-tiny", "metadata: model_id is the checkpoint directory name");
      Check(md.at("model_config") == base.config, "metadata: model_config is the verbatim config.json");
      Check(md.at("r4dx_convert_run").value("arch", "") == "gemma4_unified" && !md.at("r4dx_convert_run").value("mtp", true) &&
                !md.at("r4dx_convert_run").value("audio", true) && !md.at("r4dx_convert_run").value("vision", true),
            "metadata: r4dx_convert_run arch, mtp/vision/audio off");
      Check(!md.contains("rotation"), "metadata: no rotation block unrotated");

      // Shapes and dtypes.
      Check(c.Shape("text.embed_tokens") == std::vector<int64_t>({kVocab, kHidden, 2}), "embed_tokens [vocab, hidden] bf16");
      Check(c.Shape("text.layers.0.attn.q.bf16.w") == std::vector<int64_t>({kHeads * kHdS, kHidden, 2}), "sliding attn.q [256, 768]");
      Check(c.Shape("text.layers.1.attn.q.bf16.w") == std::vector<int64_t>({kHeads * kHdF, kHidden, 2}), "full attn.q [512, 768]");
      Check(c.Shape("text.layers.0.attn.k.bf16.w") == std::vector<int64_t>({kKvS * kHdS, kHidden, 2}), "sliding attn.k [128, 768]");
      Check(c.Shape("text.layers.1.attn.k.bf16.w") == std::vector<int64_t>({kKvF * kHdF, kHidden, 2}), "full attn.k [128, 768]");
      Check(c.Shape("text.layers.1.attn.o.bf16.w") == std::vector<int64_t>({kHidden, kHeads * kHdF, 2}), "full attn.o [768, 512]");
      Check(c.Shape("text.layers.0.mlp.gate_up.bf16.w") == std::vector<int64_t>({2 * kInter, kHidden, 2}), "mlp.gate_up [1024, 768]");
      Check(c.Shape("text.layers.0.attn.q_norm") == std::vector<int64_t>({kHdS, 2}) &&
                c.Shape("text.layers.1.attn.k_norm") == std::vector<int64_t>({kHdF, 2}),
            "q_norm/k_norm [head_dim] (64 sliding, 128 full)");
      // The per-layer descale head counts (the plan's n_kv: 8 sliding / 1 full in the real model).
      Check(c.Shape("text.layers.0.attn.k_descale") == std::vector<int64_t>({kKvS, 4}) &&
                c.Shape("text.layers.0.attn.v_descale") == std::vector<int64_t>({kKvS, 4}) &&
                c.Shape("text.layers.1.attn.k_descale") == std::vector<int64_t>({kKvF, 4}) &&
                c.Shape("text.layers.1.attn.v_descale") == std::vector<int64_t>({kKvF, 4}),
            "k/v descale fp32 [kv_heads of the layer]: 2 sliding, 1 full");
      Check(c.Floats("text.layers.0.attn.k_descale") == std::vector<float>({1.0f, 1.0f}) &&
                c.Floats("text.layers.1.attn.v_descale") == std::vector<float>({1.0f}),
            "descale placeholder 1.0 without --kv-calib");
      Check(c.Shape("text.layers.0.layer_scalar") == std::vector<int64_t>({1, 4}) && c.Dtype("text.layers.0.layer_scalar") == "U8" &&
                c.Floats("text.layers.0.layer_scalar") == std::vector<float>({0.375f}) &&
                c.Floats("text.layers.1.layer_scalar") == std::vector<float>({0.8125f}),
            "layer_scalar: the [1] bf16 buffer widened to fp32 [1, 4], exactly");

      // Byte identity of the bf16 passthroughs and layouts.
      const std::string embed = c.BytesOf(base.Get("model.language_model.embed_tokens.weight"));
      Check(c.tensors.at("text.embed_tokens") == embed, "embed_tokens bytes are the checkpoint's");
      Check(c.tensors.at("lm_head.bf16.w") == embed, "lm_head.bf16.w is the tied embedding, byte for byte");
      Check(c.tensors.at("text.final_norm") == c.BytesOf(base.Get("model.language_model.norm.weight")), "final_norm raw");
      Check(c.tensors.at("text.layers.1.input_layernorm") == c.BytesOf(base.Get("model.language_model.layers.1.input_layernorm.weight")) &&
                c.tensors.at("text.layers.0.post_feedforward_layernorm") ==
                    c.BytesOf(base.Get("model.language_model.layers.0.post_feedforward_layernorm.weight")),
            "norms are stored raw (plain weights, never 1 + w)");
      const std::string gate = c.BytesOf(base.Get("model.language_model.layers.0.mlp.gate_proj.weight"));
      const std::string up = c.BytesOf(base.Get("model.language_model.layers.0.mlp.up_proj.weight"));
      Check(c.tensors.at("text.layers.0.mlp.gate_up.bf16.w") == gate + up, "mlp.gate_up is gate rows first, then up rows");
      Check(c.tensors.at("text.layers.1.attn.o.bf16.w") ==
                c.BytesOf(base.Get("model.language_model.layers.1.self_attn.o_proj.weight")),
            "attn.o bf16 bytes are o_proj's, unfolded");
    }
  }

  // ---- w4a16 + bf16 layouts, a spot check ----
  {
    const int rc = Run("--input \"" + ckpt.u8string() + "\" --output \"" + out.u8string() + "\" --threads 2", log);
    const Container c = ReadContainer(out);
    Check(rc == 0 && c.ok && c.tensors.count("text.layers.0.attn.q.w4a16.wq") && c.tensors.count("text.layers.0.attn.q.bf16.w") &&
              c.tensors.count("lm_head.w4a16.wq") && c.tensors.count("lm_head.bf16.w") && !c.tensors.count("text.layers.1.attn.v.w4a16.wq"),
          "default layouts: w4a16 + bf16 for the body, lm_head 4bit+bf16; none for the absent full-layer v");
  }

  // ---- vision + audio passthrough ----
  {
    const int rc = convert(ckpt, "--vision on --audio on");
    const Container c = ReadContainer(out);
    const std::set<std::string> want = ExpectedNames(2, true, true, false);
    const std::vector<std::string> names = c.ok ? c.Names() : std::vector<std::string>();
    Check(rc == 0 && c.ok && std::set<std::string>(names.begin(), names.end()) == want,
          "--vision on --audio on: vision.* and audio.* appear, nothing else changes");
    Check(c.ok && c.Meta().at("r4dx_convert_run").value("audio", false) && c.Meta().at("r4dx_convert_run").value("vision", false),
          "metadata records vision/audio on");
    if (c.ok)
      Check(c.tensors.at("audio.embed_audio.embedding_projection.weight") ==
                c.BytesOf(base.Get("model.embed_audio.embedding_projection.weight")),
            "audio passthrough bytes");
  }

  // ---- --layers 1: the second layer is allow-listed, not an audit failure ----
  {
    const int rc = convert(ckpt, "--layers 1");
    const Container c = ReadContainer(out);
    const std::set<std::string> want = ExpectedNames(1, false, false, false);
    const std::vector<std::string> names = c.ok ? c.Names() : std::vector<std::string>();
    Check(rc == 0 && c.ok && std::set<std::string>(names.begin(), names.end()) == want, "--layers 1 converts layer 0 only");
  }

  // ---- --kv-calib ----
  {
    const fs::path calib = root / "kvcalib.json";
    WriteWhole(calib, json{{"0", {{"k_amax", {448.0, 224.0}}, {"v_amax", {896.0, 448.0}}}},
                           {"1", {{"k_amax", {896.0}}, {"v_amax", {224.0}}}}}.dump());
    const int rc = convert(ckpt, "--kv-calib \"" + calib.u8string() + "\"");
    const Container c = ReadContainer(out);
    Check(rc == 0 && c.ok && c.Floats("text.layers.0.attn.k_descale") == std::vector<float>({1.0f, 0.5f}) &&
              c.Floats("text.layers.0.attn.v_descale") == std::vector<float>({2.0f, 1.0f}) &&
              c.Floats("text.layers.1.attn.k_descale") == std::vector<float>({2.0f}) &&
              c.Floats("text.layers.1.attn.v_descale") == std::vector<float>({0.5f}),
          "--kv-calib: descale = amax / 448 per KV head of each layer (2 sliding, 1 full)");
    // A calibration with the wrong head count for the full layer warns and falls back to 1.0.
    WriteWhole(calib, json{{"0", {{"k_amax", {448.0, 224.0}}, {"v_amax", {896.0, 448.0}}}},
                           {"1", {{"k_amax", {896.0, 896.0}}, {"v_amax", {224.0}}}}}.dump());
    const int rc2 = convert(ckpt, "--kv-calib \"" + calib.u8string() + "\"");
    const Container c2 = ReadContainer(out);
    Check(rc2 == 0 && c2.ok && c2.Floats("text.layers.1.attn.k_descale") == std::vector<float>({1.0f}) && log_has("expected 1"),
          "--kv-calib with 2 heads for the 1-head full layer falls back to 1.0 with a warning");
  }

  // ---- --rotate q2ab (Gemma option A) ----
  {
    const int rc = convert(ckpt, "--rotate q2ab");
    const Container c = ReadContainer(out);
    Check(rc == 0 && c.ok, "--rotate q2ab converts (rc=" + std::to_string(rc) + ")");
    if (c.ok) {
      const std::set<std::string> want = ExpectedNames(2, false, false, true);
      const std::vector<std::string> names = c.Names();
      Check(std::set<std::string>(names.begin(), names.end()) == want,
            "rotated: folded in-norms are <name>.rotated, rotation.signs/mix/had_down/had_o/had_o_full present");
      const json rot = c.Meta().at("rotation");
      Check(rot.value("kind", "") == "q2ab" && rot.value("hidden", 0) == 768 && rot.value("block", 0) == 256 &&
                rot.value("nblk", 0) == 3 && rot.value("out_fold", "") == "had_only" && rot.at("had").value("o_full", 0) == 256 &&
                !rot.at("had").contains("gdn_out"),
            "rotation metadata: block 256 x 3, out_fold had_only, had {down, o, o_full}");
      Check(c.Shape("rotation.signs") == std::vector<int64_t>({kHidden, 4}) && c.Shape("rotation.mix") == std::vector<int64_t>({3, 3, 4}) &&
                c.Shape("rotation.had_down_signs") == std::vector<int64_t>({kInter, 4}) &&
                c.Shape("rotation.had_o_signs") == std::vector<int64_t>({kHeads * kHdS, 4}) &&
                c.Shape("rotation.had_o_full_signs") == std::vector<int64_t>({kHeads * kHdF, 4}),
            "rotation tensor shapes: [768], [3,3], down [512], o [256], o_full [512]");
      // Folded norms are plain ones; the post norms are untouched.
      const std::string ones = c.BytesOf(std::vector<float>(static_cast<size_t>(kHidden), 1.0f));
      Check(c.tensors.at("text.layers.0.input_layernorm.rotated") == ones && c.tensors.at("text.layers.1.pre_feedforward_layernorm.rotated") == ones,
            "folded norms are stored as bf16 ones");
      Check(c.tensors.at("text.layers.0.post_attention_layernorm") ==
                c.BytesOf(base.Get("model.language_model.layers.0.post_attention_layernorm.weight")),
            "post norms are not folded");

      // The fold itself, against an independent computation from the library: q = W diag(w) Q (offset 0),
      // o = W Hb only (no Q^T on the rows), down = W Hb_down.
      r4dx_convert::RotationShape shape;
      shape.hidden = kHidden;
      shape.block = r4dx_convert::ChooseRotationBlock(kHidden);
      shape.k_down = kInter;
      shape.k_o = kHeads * kHdS;
      shape.k_o_full = kHeads * kHdF;
      shape.b_o = shape.b_o_full = r4dx_convert::kHadBlockO;
      const r4dx_convert::RotationSet rs =
          r4dx_convert::GenerateRotationSet(r4dx_convert::RotationKind::kQ2ab, r4dx_convert::kDefaultRotationSeed, shape);
      std::vector<float> q = base.Get("model.language_model.layers.0.self_attn.q_proj.weight");
      r4dx_convert::FoldRowsQ(q, kHeads * kHdS, kHidden, base.Get("model.language_model.layers.0.input_layernorm.weight").data(), rs.q,
                              2, 0.0);
      Check(c.tensors.at("text.layers.0.attn.q.bf16.w") == c.BytesOf(q), "attn.q is W diag(w) Q (plain norm, offset 0)");
      std::vector<float> o = base.Get("model.language_model.layers.0.self_attn.o_proj.weight");
      r4dx_convert::FoldRowsHadamard(o, kHidden, kHeads * kHdS, rs.had_o, 2);
      Check(c.tensors.at("text.layers.0.attn.o.bf16.w") == c.BytesOf(o), "sliding attn.o is W Hb only (output rows keep the original basis)");
      std::vector<float> of = base.Get("model.language_model.layers.1.self_attn.o_proj.weight");
      r4dx_convert::FoldRowsHadamard(of, kHidden, kHeads * kHdF, rs.had_o_full, 2);
      Check(c.tensors.at("text.layers.1.attn.o.bf16.w") == c.BytesOf(of), "full attn.o is W Hb_o_full only");
      std::vector<float> dn = base.Get("model.language_model.layers.1.mlp.down_proj.weight");
      r4dx_convert::FoldRowsHadamard(dn, kHidden, kInter, rs.had_down, 2);
      Check(c.tensors.at("text.layers.1.mlp.down.bf16.w") == c.BytesOf(dn), "mlp.down is W Hb_down only");
      std::vector<float> gu = base.Get("model.language_model.layers.0.mlp.gate_proj.weight");
      const std::vector<float>& upw = base.Get("model.language_model.layers.0.mlp.up_proj.weight");
      gu.insert(gu.end(), upw.begin(), upw.end());
      r4dx_convert::FoldRowsQ(gu, 2 * kInter, kHidden, base.Get("model.language_model.layers.0.pre_feedforward_layernorm.weight").data(),
                              rs.q, 2, 0.0);
      Check(c.tensors.at("text.layers.0.mlp.gate_up.bf16.w") == c.BytesOf(gu), "mlp.gate_up folds pre_feedforward_layernorm");
    }
    // q2a: in-projections fold, out-projections are left alone (the post norm blocks Q^T W).
    const int rca = convert(ckpt, "--rotate q2a");
    const Container ca = ReadContainer(out);
    Check(rca == 0 && ca.ok && ca.tensors.at("text.layers.0.attn.o.bf16.w") ==
                                    ca.BytesOf(base.Get("model.language_model.layers.0.self_attn.o_proj.weight")) &&
              !ca.tensors.count("rotation.had_o_signs") && ca.Meta().at("rotation").value("out_fold", "") == "had_only",
          "--rotate q2a: out-projections unfolded, no Hadamard tensors");
  }

  // ---- refusals ----
  auto fail_case = [&](const std::string& label, const std::string& tag, const std::function<void(Ckpt&)>& mutate,
                       const std::string& extra, const std::string& expect_in_log) {
    Ckpt c = base;
    mutate(c);
    const fs::path dir = root / ("bad-" + tag);
    WriteCkpt(c, dir);
    std::error_code ec;
    fs::remove(out, ec);
    const int rc = convert(dir, extra);
    Check(rc != 0 && log_has(expect_in_log), label + " (rc=" + std::to_string(rc) + ", wants '" + expect_in_log + "')");
    // Refused before the header was written: no container is left behind.
    const bool wrote = fs::exists(out) && fs::file_size(out) > 0;
    Check(!wrote, label + ": no container written");
  };
  fail_case("audit: an unconsumed tensor fails the run and is named", "extra",
            [](Ckpt& c) { c.Set("model.language_model.layers.0.self_attn.mystery.weight", {4, 4}); }, "",
            "layers.0.self_attn.mystery.weight");
  fail_case("audit: a stray lm_head.weight (the config says tied) fails the run", "lmhead",
            [](Ckpt& c) { c.Set("lm_head.weight", {kVocab, kHidden}); }, "", "lm_head.weight");
  fail_case("a wrong q_proj shape fails the run, naming both shapes", "shape",
            [](Ckpt& c) { c.Set("model.language_model.layers.0.self_attn.q_proj.weight", {128, kHidden}); }, "", "config says [256,768]");
  fail_case("a missing layer_scalar fails the run", "scalar", [](Ckpt& c) { c.Erase("model.language_model.layers.1.layer_scalar"); }, "",
            "no tensor 'model.language_model.layers.1.layer_scalar'");
  fail_case("v_proj on the k_eq_v full layer fails the run", "vproj",
            [](Ckpt& c) { c.Set("model.language_model.layers.1.self_attn.v_proj.weight", {kKvF * kHdF, kHidden}); }, "", "attention_k_eq_v");
  fail_case("no v_proj on a sliding layer fails the run", "nov", [](Ckpt& c) { c.Erase("model.language_model.layers.0.self_attn.v_proj.weight"); },
            "", "no self_attn.v_proj.weight");
  fail_case("an untied config fails the run", "untied",
            [](Ckpt& c) { c.config["text_config"]["tie_word_embeddings"] = false; }, "", "tie_word_embeddings");
  fail_case("a vision-on run on a checkpoint without vision tensors fails", "novision",
            [](Ckpt& c) {
              c.Erase("model.vision_embedder.patch_ln1.weight");
              c.Erase("model.vision_embedder.patch_dense.weight");
              c.Erase("model.vision_embedder.pos_embedding");
              c.Erase("model.embed_vision.embedding_projection.weight");
            },
            "--vision on", "no vision tensors");
  fail_case("--mtp on is refused", "mtp", [](Ckpt&) {}, "--mtp on", "no MTP head");
  fail_case("--trellis-from is refused", "trellis", [](Ckpt&) {}, "--trellis-from \"" + (root / "nowhere").u8string() + "\"",
            "not supported for gemma4_unified");
  fail_case("--record-reuse-guard is refused", "guard", [](Ckpt&) {}, "--record-reuse-guard", "not supported for gemma4_unified");

  std::error_code ec;
  fs::remove_all(root, ec);
}

#endif  // R4DX_CONVERT_EXE

}  // namespace

int main() {
  try {
    TestRealHeader();
#if defined(R4DX_CONVERT_EXE)
    TestExe();
#else
    std::printf("SKIP: r4dx-convert is not built, part (b) not run\n");
#endif
  } catch (const std::exception& e) {
    std::fprintf(stderr, "FAIL: uncaught exception: %s\n", e.what());
    ++g_failures;
  }
  if (g_failures != 0) {
    std::fprintf(stderr, "%d check(s) FAILED\n", g_failures);
    return 1;
  }
  std::printf("convert_gemma_layout: all checks passed\n");
  return 0;
}
