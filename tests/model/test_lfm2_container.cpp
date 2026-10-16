// tests/model/test_lfm2_container.cpp -- CPU-only (no GPU, no HIP call). docs/lfm2-plan.md L1-3.
//
// The CPU half of the LFM2 container loader (lfm2_container_info.h, which Lfm2Container::Inspect / Load run first):
// `__metadata__` parsing and the tensor-table checks, on
//   (a) a tiny lfm2_moe checkpoint written here (the real tensor names, per-expert experts, F32 expert_bias, flat config;
//       tools/convert_ref/make_tiny_lfm2.py's geometry: hidden 512, 8 heads x 64 / 2 kv, dense I 512, 8 experts of 512)
//       converted by the real r4dx-convert (when built): the whole table is accepted, every per-layer expert name the
//       loader will read exists, the config and its TP=2 shard parse as the loader needs them, --layers N is honoured;
//   (b) header edits of that container: a missing expert tensor, a wrong expert_bias shape, a stray conv tensor on an
//       attention layer, an expert past num_experts, a missing lm_head, a wrong model_arch / norm_kind, a rotated
//       container -- each names its problem;
//   (c) the real lfm2-l6.r4dx header when one exists (SKIP otherwise);
//   (d) trellis containers' headers (L3-8) when they exist: the tiny fixture (tools/reference/lfm2/tiny_trellis_golden.py)
//       and the real lfm2-trellis45.r4dx -- quant.trellis lists exactly the body linears with the loader's byte sizes.
// The upload half (Lfm2Container::Load) needs a device: see the GPU test list in the lane report.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "lfm2_config.h"
#include "lfm2_container_info.h"
#include "nlohmann/json.hpp"
#include "r4dx/core/dtype.hpp"
#include "r4dx/models_root.h"
#include "r4dx_convert/safetensors_reader.hpp"
#include "trellis_meta.h"

namespace fs = std::filesystem;
using nlohmann::json;
using r4dx::model::Arch;
using r4dx::model::InspectLfm2Container;
using r4dx::model::Lfm2Config;
using r4dx::model::Lfm2ContainerInfo;

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

std::string ReadWhole(const fs::path& p) {
  std::ifstream f(p, std::ios::binary);
  if (!f) throw std::runtime_error("cannot read " + p.string());
  return std::string(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}

void WriteWhole(const fs::path& p, const std::string& bytes) {
  std::ofstream f(p, std::ios::binary | std::ios::trunc);
  f.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

// ---- the tiny checkpoint ----------------------------------------------------------------------------------------------

constexpr int64_t H = 512, HEADS = 8, KV = 2, HD = 64, DENSE_I = 512, MOE_I = 512, EXPERTS = 8, VOCAB = 64, LAYERS = 4;
bool IsAttn(int64_t i) { return i % 4 == 2; }

json Config() {
  json types = json::array();
  for (int64_t i = 0; i < LAYERS; ++i) types.push_back(IsAttn(i) ? "full_attention" : "conv");
  return {{"architectures", {"Lfm2MoeForCausalLM"}}, {"conv_L_cache", 3}, {"conv_bias", false}, {"hidden_size", H},
          {"intermediate_size", DENSE_I}, {"layer_types", types}, {"max_position_embeddings", 4096}, {"model_type", "lfm2_moe"},
          {"moe_intermediate_size", MOE_I}, {"norm_eps", 1e-05}, {"norm_topk_prob", true}, {"num_attention_heads", HEADS},
          {"num_dense_layers", 2}, {"num_experts", EXPERTS}, {"num_experts_per_tok", 2}, {"num_hidden_layers", LAYERS},
          {"num_key_value_heads", KV}, {"rope_parameters", {{"rope_theta", 1e7}, {"rope_type", "default"}}},
          {"routed_scaling_factor", 1.0}, {"tie_word_embeddings", true}, {"use_expert_bias", true}, {"vocab_size", VOCAB}};
}

struct Spec {
  std::string name, dtype;
  std::vector<int64_t> shape;
};

std::vector<Spec> Specs() {
  std::vector<Spec> s;
  for (int64_t i = 0; i < LAYERS; ++i) {
    const std::string L = "model.layers." + std::to_string(i) + ".";
    s.push_back({L + "operator_norm.weight", "BF16", {H}});
    s.push_back({L + "ffn_norm.weight", "BF16", {H}});
    if (IsAttn(i)) {
      s.push_back({L + "self_attn.q_proj.weight", "BF16", {HEADS * HD, H}});
      s.push_back({L + "self_attn.k_proj.weight", "BF16", {KV * HD, H}});
      s.push_back({L + "self_attn.v_proj.weight", "BF16", {KV * HD, H}});
      s.push_back({L + "self_attn.out_proj.weight", "BF16", {H, HEADS * HD}});
      s.push_back({L + "self_attn.q_layernorm.weight", "BF16", {HD}});
      s.push_back({L + "self_attn.k_layernorm.weight", "BF16", {HD}});
    } else {
      s.push_back({L + "conv.in_proj.weight", "BF16", {3 * H, H}});
      s.push_back({L + "conv.conv.weight", "BF16", {H, 1, 3}});
      s.push_back({L + "conv.out_proj.weight", "BF16", {H, H}});
    }
    if (i < 2) {
      s.push_back({L + "feed_forward.w1.weight", "BF16", {DENSE_I, H}});
      s.push_back({L + "feed_forward.w2.weight", "BF16", {H, DENSE_I}});
      s.push_back({L + "feed_forward.w3.weight", "BF16", {DENSE_I, H}});
    } else {
      s.push_back({L + "feed_forward.gate.weight", "BF16", {EXPERTS, H}});
      s.push_back({L + "feed_forward.expert_bias", "F32", {EXPERTS}});
      for (int64_t e = 0; e < EXPERTS; ++e) {
        const std::string E = L + "feed_forward.experts." + std::to_string(e) + ".";
        s.push_back({E + "w1.weight", "BF16", {MOE_I, H}});
        s.push_back({E + "w2.weight", "BF16", {H, MOE_I}});
        s.push_back({E + "w3.weight", "BF16", {MOE_I, H}});
      }
    }
  }
  s.push_back({"model.embed_tokens.weight", "BF16", {VOCAB, H}});
  s.push_back({"model.embedding_norm.weight", "BF16", {H}});
  return s;
}

void WriteCheckpoint(const fs::path& dir) {
  fs::create_directories(dir);
  json hdr = json::object();
  std::string data;
  uint32_t seed = 12345;
  for (const Spec& s : Specs()) {
    int64_t n = 1;
    for (int64_t d : s.shape) n *= d;
    const uint64_t b = data.size();
    for (int64_t i = 0; i < n; ++i) {
      seed = seed * 1664525u + 1013904223u;
      const float v = (static_cast<int>((seed >> 24) % 9) - 4) * 0.02f;
      if (s.dtype == "F32") {
        data.append(reinterpret_cast<const char*>(&v), 4);
      } else {
        const uint16_t h = r4dx::core::FloatToBf16(v);
        data.append(reinterpret_cast<const char*>(&h), 2);
      }
    }
    hdr[s.name] = {{"dtype", s.dtype}, {"shape", s.shape}, {"data_offsets", {b, data.size()}}};
  }
  const std::string h = hdr.dump();
  const uint64_t hl = h.size();
  std::string file(reinterpret_cast<const char*>(&hl), 8);
  file += h;
  file += data;
  WriteWhole(dir / "model.safetensors", file);
  WriteWhole(dir / "config.json", Config().dump(2));
}

// Re-writes container `src` as `dst` with `edit` applied to its parsed header (the data section is carried over untouched:
// offsets are relative to it).
template <class Edit>
void Rewrite(const fs::path& src, const fs::path& dst, Edit&& edit) {
  const std::string bytes = ReadWhole(src);
  uint64_t hl = 0;
  std::memcpy(&hl, bytes.data(), 8);
  json header = json::parse(bytes.substr(8, static_cast<size_t>(hl)));
  edit(header);
  const std::string h = header.dump();
  const uint64_t nl = h.size();
  std::string out(reinterpret_cast<const char*>(&nl), 8);
  out += h;
  out += bytes.substr(static_cast<size_t>(8 + hl));
  WriteWhole(dst, out);
}

#if defined(R4DX_CONVERT_EXE)

int Run(const std::string& args, const fs::path& log) {
  const std::string cmd = "\"\"" + std::string(R4DX_CONVERT_EXE) + "\" " + args + " > \"" + log.u8string() + "\" 2>&1\"";
  return std::system(cmd.c_str());
}

std::vector<std::string> Problems(const std::string& path, int64_t layers, bool bf16_body = true) {
  r4dx_convert::SafetensorsReader reader(r4dx_convert::Utf8ToWide(path));
  const json header = json::parse(reader.HeaderJson());
  const Lfm2ContainerInfo info = r4dx::model::ParseLfm2ContainerMetadata(header.at("__metadata__"), path);
  return r4dx::model::Lfm2TensorTableProblems(reader, info, layers, bf16_body);
}

bool AnyContains(const std::vector<std::string>& v, const std::string& needle) {
  for (const auto& s : v)
    if (s.find(needle) != std::string::npos) return true;
  return false;
}

void TestTiny() {
  std::printf("---- (a) the tiny converted container ----\n");
  const fs::path root = fs::temp_directory_path() / "r4dx_test_lfm2_container";
  std::error_code ec;
  fs::remove_all(root, ec);
  fs::create_directories(root);
  const fs::path ckpt = root / "ckpt-lfm2-tiny", c = root / "tiny.r4dx", log = root / "log.txt";
  WriteCheckpoint(ckpt);
  int rc = Run("--input \"" + ckpt.u8string() + "\" --output \"" + c.u8string() + "\" --threads 2 --layouts bf16 --lm-head bf16", log);
  Check(rc == 0, "r4dx-convert converts the tiny checkpoint (rc=" + std::to_string(rc) + ")");
  if (rc != 0) return;
  const std::string path = c.u8string();

  const Lfm2ContainerInfo info = InspectLfm2Container(path);
  Check(info.model_id == "ckpt-lfm2-tiny" && !info.config_sha256.empty() && !info.has_trellis, "Inspect: model_id, config_sha256, no trellis");
  const Lfm2Config& cfg = info.config;
  Check(cfg.hidden_size == H && cfg.num_hidden_layers == LAYERS && cfg.num_experts == EXPERTS && cfg.moe_intermediate_size == MOE_I &&
            cfg.num_dense_layers == 2 && cfg.head_dim == HD && cfg.NumAttentionLayers() == 1 && cfg.IsAttentionLayer(2) && cfg.tp_world == 1,
        "Inspect: the flat model_config parsed (global config)");
  Check(info.model_config.arch == Arch::kLfm2Moe && info.model_config.num_experts == EXPERTS && info.model_config.moe_intermediate_size == MOE_I,
        "Inspect: the generic ModelConfig carries kLfm2Moe and the MoE geometry (what ShardLoader / RuleFor read)");
  Check(Problems(path, LAYERS, /*bf16_body=*/true).empty() && Problems(path, LAYERS, /*bf16_body=*/false).empty(),
        "the tensor table is accepted (bf16 body required, and any-form)");
  // The TP=2 shard of this config is legal (the loader builds it before uploading anything).
  const Lfm2Config r1 = Lfm2Config::Shard(cfg, 2, 1);
  Check(r1.num_attention_heads == 4 && r1.num_key_value_heads == 1 && r1.moe_intermediate_size == 256 && r1.conv_dim == 256 && r1.intermediate_size == 256,
        "Shard(2, 1) of the tiny config is legal: heads 4 / kv 1, experts 256, conv 256, dense 256");

  // Every expert name the loader reads exists, with the global [N, K] it asks the ShardLoader for.
  r4dx_convert::SafetensorsReader reader(r4dx_convert::Utf8ToWide(path));
  bool names = true;
  for (int64_t li = 0; li < LAYERS; ++li) {
    if (cfg.IsDenseLayer(li)) continue;
    for (int64_t e = 0; e < EXPERTS; ++e) {
      const std::string gu = r4dx::model::Lfm2ExpertBase(li, e, "gate_up") + ".bf16.w", dn = r4dx::model::Lfm2ExpertBase(li, e, "down") + ".bf16.w";
      names = names && reader.Has(gu) && reader.Has(dn) && reader.Meta(gu).shape == std::vector<int64_t>({cfg.ExpertGateUpRows(), H, 2}) &&
              reader.Meta(dn).shape == std::vector<int64_t>({H, cfg.moe_intermediate_size, 2});
    }
  }
  Check(names && !reader.Has(r4dx::model::Lfm2ExpertBase(2, EXPERTS, "gate_up") + ".bf16.w"),
        "every layer's expert table: 8 gate_up + 8 down linears per MoE layer, none past num_experts");
  Check(reader.Meta("text.layers.3.moe.expert_bias").shape == std::vector<int64_t>({EXPERTS, 4}) &&
            reader.Meta("text.layers.0.conv.weight").shape == std::vector<int64_t>({H, 3, 2}),
        "expert_bias fp32 [8, 4] and conv.weight [512, 3, 2] on disk");

  // --layers N.
  {
    const fs::path c3 = root / "tiny3.r4dx";
    rc = Run("--input \"" + ckpt.u8string() + "\" --output \"" + c3.u8string() + "\" --threads 2 --layouts bf16 --lm-head bf16 --layers 3", log);
    Check(rc == 0 && InspectLfm2Container(c3.u8string(), 3).model_id == "ckpt-lfm2-tiny", "a --layers 3 container inspects with layer_limit 3");
    Check(Throws([&] { InspectLfm2Container(c3.u8string()); }, "layers.3."), "... and is refused with the full layer count (layer 3's tensors are missing)");
  }

  // ---- (b) header edits ----
  std::printf("---- (b) header edits of the container ----\n");
  const fs::path e = root / "edited.r4dx";
  auto problems_after = [&](auto&& edit, int64_t layers = LAYERS) {
    Rewrite(c, e, edit);
    return Problems(e.u8string(), layers);
  };
  Check(AnyContains(problems_after([](json& h) { h.erase("text.layers.3.moe.experts.2.down.bf16.w"); }), "missing tensor 'text.layers.3.moe.experts.2.down.bf16.w'"),
        "a missing expert down tensor is named");
  Check(AnyContains(problems_after([](json& h) { h["text.layers.3.moe.expert_bias"]["shape"] = {EXPERTS * 4, 1}; }), "moe.expert_bias"),
        "a wrong expert_bias shape (not [E, 4]) is named");
  Check(AnyContains(problems_after([](json& h) { h["text.layers.2.moe.router"]["shape"] = {H, EXPERTS, 2}; }), "moe.router"), "a wrong router shape is named");
  Check(AnyContains(problems_after([](json& h) { h["text.layers.2.conv.in_proj.bf16.w"] = h["text.layers.0.conv.in_proj.bf16.w"]; }), "layer 2 is an attention layer"),
        "a conv.in_proj on an attention layer is reported");
  Check(AnyContains(problems_after([](json& h) { h["text.layers.0.attn.q.bf16.w"] = h["text.layers.2.attn.q.bf16.w"]; }), "layer 0 is a ShortConv layer"),
        "an attn.q on a ShortConv layer is reported");
  Check(AnyContains(problems_after([](json& h) { h["text.layers.3.moe.experts.8.gate_up.bf16.w"] = h["text.layers.3.moe.experts.7.gate_up.bf16.w"]; }), "only 8 experts"),
        "an expert 8 on an 8-expert layer is reported");
  Check(AnyContains(problems_after([](json& h) { h["text.layers.2.mlp.gate_up.bf16.w"] = h["text.layers.0.mlp.gate_up.bf16.w"]; }), "MoE layer"),
        "a dense mlp on an MoE layer is reported");
  Check(AnyContains(problems_after([](json& h) { h.erase("lm_head.bf16.w"); }), "missing lm_head"), "a missing lm_head is reported");
  Check(AnyContains(problems_after([](json& h) { h["text.layers.1.mlp.down.bf16.w"]["shape"] = {H, DENSE_I * 2}; }), "mlp.down"), "a wrong dense down shape is named");
  Check(AnyContains(problems_after([](json& h) { h["text.layers.0.conv.weight"]["shape"] = {3, H, 2}; }), "conv.weight"), "a transposed conv.weight [3, H] is named");
  Check(problems_after([](json&) {}, 2).empty(), "layers=2 checks only layers 0-1 (the table beyond is not asked for)");
  Check(problems_after([](json& h) { h.erase("text.layers.3.moe.router"); }, 3).empty(),
        "a problem in layer 3 is not asked about when only 3 layers are loaded");

  // Metadata refusals.
  auto inspect_after = [&](auto&& edit) {
    Rewrite(c, e, edit);
    return [&] { (void)InspectLfm2Container(e.u8string()); };
  };
  Check(Throws(inspect_after([](json& h) { h["__metadata__"]["model_arch"] = "gemma4_unified"; }), "not lfm2_moe"), "a gemma4_unified container is refused");
  Check(Throws(inspect_after([](json& h) { h["__metadata__"].erase("model_arch"); }), "not lfm2_moe"), "a container without model_arch is refused");
  Check(Throws(inspect_after([](json& h) { h["__metadata__"]["norm_kind"] = "zero_centered"; }), "norm_kind"), "norm_kind other than plain is refused");
  Check(Throws(inspect_after([](json& h) { h["__metadata__"]["rotation"] = {{"kind", "q2a"}}; }), "rotat"), "a rotated container is refused");
  Check(Throws(inspect_after([](json& h) { h["__metadata__"].erase("model_config"); }), "model_config"), "a container without model_config is refused");
  Check(Throws(inspect_after([](json& h) { h["__metadata__"]["model_config"]["conv_L_cache"] = 4; }), "conv_L_cache"), "a model_config the runtime does not implement is refused");
  Check(Throws(inspect_after([](json& h) { h.erase("__metadata__"); }), "no __metadata__"), "a container without __metadata__ is refused");

  fs::remove_all(root, ec);
}

#endif  // R4DX_CONVERT_EXE

// The real lfm2-l6.r4dx (layers 0-5 of the real checkpoint), when it exists: header only.
void TestRealL6() {
  std::printf("---- (c) the real lfm2-l6 container header ----\n");
  const fs::path p = fs::path(r4dx::ModelsPath("r4dx/lfm2/lfm2-l6.r4dx"));
  if (!fs::exists(p)) {
    std::printf("SKIP: %s not present\n", p.string().c_str());
    ++g_skipped;
    return;
  }
  const Lfm2ContainerInfo info = InspectLfm2Container(p.u8string(), 6);
  Check(info.config.num_hidden_layers == 40 && info.config.num_experts == 64 && info.model_config.arch == Arch::kLfm2Moe,
        "lfm2-l6: the real 40-layer config inside, 6 layers' tensor table accepted");
  Check(Throws([&] { InspectLfm2Container(p.u8string()); }, "missing tensor"), "lfm2-l6 is refused as a 40-layer model (layers 6+ are not in it)");
}

// docs/lfm2-plan.md L3-8: a trellis LFM2 container's header, as Lfm2Container::Load --layout trellis will read it --
// the table Inspect accepts, and __metadata__.quant.trellis.linears lists EXACTLY the body linears (attn q/k/v/o or
// ShortConv in/out, dense gate_up [2 parts] / down, every expert's gate_up [2 parts] / down), each with its three
// tensors at the byte sizes the loader checks and no bf16 copy, while lm_head / router / embedding stay bf16.
void CheckTrellisContainer(const fs::path& p, const std::string& tag) {
  const Lfm2ContainerInfo info = InspectLfm2Container(p.u8string());
  const Lfm2Config& c = info.config;
  Check(info.has_trellis, tag + ": __metadata__.quant.trellis present");
  const std::optional<r4dx::model::TrellisSpec> spec = r4dx::model::ParseTrellisMetadata(info.metadata, p.u8string());
  if (!spec) {
    Check(false, tag + ": ParseTrellisMetadata returns a spec");
    return;
  }
  const r4dx_convert::SafetensorsReader r(r4dx_convert::Utf8ToWide(p.u8string()));
  struct Want {
    std::string base;
    int64_t n, k, part0;  // part0 > 0: a two-part [part0, n - part0] linear
  };
  std::vector<Want> want;
  const int64_t H = c.hidden_size;
  for (int64_t i = 0; i < c.num_hidden_layers; ++i) {
    const std::string b = r4dx::model::Lfm2LayerBase(i);
    if (c.IsAttentionLayer(i)) {
      want.push_back({b + "attn.q", c.QDim(), H, 0});
      want.push_back({b + "attn.k", c.KvDim(), H, 0});
      want.push_back({b + "attn.v", c.KvDim(), H, 0});
      want.push_back({b + "attn.o", H, c.QDim(), 0});
    } else {
      want.push_back({b + "conv.in_proj", c.ConvInProjRows(), H, 0});
      want.push_back({b + "conv.out_proj", H, c.conv_dim, 0});
    }
    if (c.IsDenseLayer(i)) {
      want.push_back({b + "mlp.gate_up", 2 * c.intermediate_size, H, c.intermediate_size});
      want.push_back({b + "mlp.down", H, c.intermediate_size, 0});
    } else {
      for (int64_t e = 0; e < c.num_experts; ++e) {
        want.push_back({r4dx::model::Lfm2ExpertBase(i, e, "gate_up"), c.ExpertGateUpRows(), H, c.moe_intermediate_size});
        want.push_back({r4dx::model::Lfm2ExpertBase(i, e, "down"), H, c.moe_intermediate_size, 0});
      }
    }
  }
  std::set<std::string> listed, wanted;
  for (const auto& kv : spec->linears) listed.insert(kv.first);
  for (const Want& w : want) wanted.insert(w.base);
  Check(listed == wanted, tag + ": quant.trellis.linears lists exactly the " + std::to_string(wanted.size()) +
                              " body linears (" + std::to_string(listed.size()) + " listed)");
  int bad = 0;
  std::string first_bad;
  const auto span = [&](const std::string& n) { return r.Has(n) ? r.Meta(n).end - r.Meta(n).begin : uint64_t{0}; };
  for (const Want& w : want) {
    const r4dx::model::TrellisLinearSpec* t = spec->Find(w.base);
    bool ok = t != nullptr && w.n % 128 == 0 && w.k % 128 == 0 && !r.Has(w.base + ".bf16.w");
    if (ok) {
      const int64_t parts = w.part0 > 0 ? 2 : 1;
      ok = t->Parts() == parts && (parts == 1 || (t->parts.size() == 2 && t->parts[0] == w.part0 && t->parts[1] == w.n - w.part0)) &&
           (t->bits == 4 || t->bits == 5) &&
           span(w.base + ".trellis.w") == static_cast<uint64_t>(w.n * w.k * t->bits / 8) &&
           span(w.base + ".trellis.suh") == static_cast<uint64_t>(parts * w.k * 2) &&
           span(w.base + ".trellis.svh") == static_cast<uint64_t>(w.n * 2);
    }
    if (!ok && bad++ == 0) first_bad = w.base;
  }
  Check(bad == 0, tag + ": every trellis linear has its parts, KB 4|5, the loader's byte sizes and no bf16 copy" +
                      (bad ? " (first bad: " + first_bad + ", " + std::to_string(bad) + " bad)" : std::string()));
  Check(r.Has("lm_head.bf16.w") && !r.Has("lm_head.trellis.w") && r.Has("text.embed_tokens") &&
            !spec->Find("lm_head") && !spec->Find(r4dx::model::Lfm2LayerBase(c.num_dense_layers) + "moe.router"),
        tag + ": lm_head bf16, embedding raw, router not trellis");
}

void TestTrellis() {
  std::printf("---- (d) trellis containers (headers) ----\n");
  bool any = false;
  for (const auto& [rel, tag] : {std::pair<const char*, const char*>{"r4dx/lfm2/tiny_trellis/tiny_trellis.r4dx", "tiny trellis"},
                                 std::pair<const char*, const char*>{"r4dx/lfm2/lfm2-trellis45.r4dx", "lfm2-trellis45"}}) {
    const fs::path p = fs::path(r4dx::ModelsPath(rel));
    if (!fs::exists(p)) {
      std::printf("SKIP: %s not present\n", p.string().c_str());
      continue;
    }
    any = true;
    CheckTrellisContainer(p, tag);
  }
  if (!any) ++g_skipped;
}

}  // namespace

int main() try {
#if defined(R4DX_CONVERT_EXE)
  TestTiny();
#else
  std::printf("SKIP: r4dx-convert not built (parts a, b)\n");
  ++g_skipped;
#endif
  TestRealL6();
  TestTrellis();
  std::printf("\n%s: %d failure(s), %d skipped part(s)\n", g_failures == 0 ? "OK" : "FAILED", g_failures, g_skipped);
  return g_failures == 0 ? 0 : 1;
} catch (const std::exception& e) {
  std::fprintf(stderr, "FAIL: uncaught exception: %s\n", e.what());
  return 2;
}
