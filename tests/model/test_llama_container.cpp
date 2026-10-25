// tests/model/test_llama_container.cpp -- CPU-only (no GPU, no HIP call). docs/nanbeige-plan.md N1-5.
//
// The CPU half of the Llama container loader (llama_container_info.h, which LlamaContainer::Inspect / Load run first)
// and the TP=2 shard table it loads through (tp::RuleFor's Llama rules), on
//   (a) a tiny llama checkpoint written here (tools/convert_ref/make_tiny_llama.py's geometry: hidden 512, 28 q / 4 kv
//       heads of 128, FFN 512, vocab 512, 3 layers, untied head) converted by the real r4dx-convert (when built) as bf16
//       and as w8a16: the table is accepted, the config and its TP=2 shard parse as the loader needs them, --layers N
//       is honoured, and every tensor the loader reads has a shard rule whose per-rank byte plan is legal for its
//       on-disk parts (bf16 / w8a16 wq / ws);
//   (b) header edits of that container: a missing linear, a wrong shape, a missing lm_head, a wrong model_arch /
//       norm_kind, a rotated container, no model_config -- each names its problem;
//   (c) tp::RuleFor's Llama table on the REAL geometry (52 layers): the rule of every tensor, the refusals (an unknown
//       layer tensor, mtp.* / rotation.* / vision.* / dflash.*, a layer past 52);
//   (d) the real Nanbeige w8a16 container header when it exists (SKIP otherwise): accepted, quant.w8a16 group 128,
//       every w8a16 linear's TP=2 byte plan legal.
// The upload half (LlamaContainer::Load) needs a device: a GPU test (N1-9 / N1-11).
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

#include "llama_config.h"
#include "llama_container_info.h"
#include "nlohmann/json.hpp"
#include "r4dx/core/dtype.hpp"
#include "r4dx/models_root.h"
#include "r4dx_convert/safetensors_reader.hpp"
#include "tp/tp_shard.h"

namespace fs = std::filesystem;
namespace tp = r4dx::model::tp;
using nlohmann::json;
using r4dx::model::Arch;
using r4dx::model::InspectLlamaContainer;
using r4dx::model::LlamaConfig;
using r4dx::model::LlamaContainerInfo;

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

bool AnyContains(const std::vector<std::string>& v, const std::string& needle) {
  for (const auto& s : v)
    if (s.find(needle) != std::string::npos) return true;
  return false;
}

// The base names of every linear a container of `layers` layers carries, with their global [N, K].
struct LinearSpec {
  std::string base;
  int64_t N, K;
};
std::vector<LinearSpec> Linears(const LlamaConfig& c, int64_t layers) {
  std::vector<LinearSpec> v;
  const int64_t H = c.hidden_size;
  for (int64_t i = 0; i < layers; ++i) {
    const std::string b = r4dx::model::LlamaLayerBase(i);
    v.push_back({b + "attn.q", c.QDim(), H});
    v.push_back({b + "attn.k", c.KvDim(), H});
    v.push_back({b + "attn.v", c.KvDim(), H});
    v.push_back({b + "attn.o", H, c.QDim()});
    v.push_back({b + "mlp.gate_up", c.GateUpRows(), H});
    v.push_back({b + "mlp.down", H, c.intermediate_size});
  }
  v.push_back({"lm_head", c.vocab_size, H});
  return v;
}

// Every linear's TP=2 rule gives each rank exactly the rows / K that LlamaConfig::Shard says, and the byte plan of each
// on-disk part (`parts`: bf16, w8 wq, w8a16 ws at `group`) is legal. Returns the first problem, or "".
std::string ShardPlanProblem(const LlamaConfig& g, int64_t layers, const std::vector<tp::Part>& parts, int group) {
  const r4dx::model::ModelConfig mc = g.ToModelConfig();
  for (int rank = 0; rank < 2; ++rank) {
    const LlamaConfig r = LlamaConfig::Shard(g, 2, rank);
    for (const LinearSpec& l : Linears(g, layers)) {
      try {
        const tp::ShardRule rule = tp::RuleFor(l.base, mc);
        const bool is_o = l.base.find("attn.o") != std::string::npos, is_down = l.base.find("mlp.down") != std::string::npos;
        if (is_o || is_down) {
          if (rule.split != tp::Split::kCols) return l.base + ": not row-parallel";
          const tp::Range k = tp::RankCols(rule, 2, rank);
          const int64_t want_k = is_o ? r.QDim() : r.intermediate_size;
          if (k.count != want_k || k.begin != rank * want_k) return l.base + ": rank K range != the shard's";
          for (tp::Part p : parts) {
            tp::PartShape sh;
            sh.part = p;
            sh.N = l.N;
            sh.K = l.K;
            sh.group = group;
            (void)tp::PlanCols(sh, k);
          }
        } else {
          if (rule.split != tp::Split::kRows) return l.base + ": not column-parallel";
          const std::vector<tp::Range> rows = tp::RankRows(rule, 2, rank);
          int64_t n = 0;
          for (const tp::Range& rg : rows) n += rg.count;
          const int64_t want_n = l.base == "lm_head" ? r.VocabShardSize()
                                 : l.base.find("attn.q") != std::string::npos ? r.QDim()
                                 : l.base.find("mlp.gate_up") != std::string::npos ? r.GateUpRows()
                                                                                  : r.KvDim();
          if (n != want_n) return l.base + ": rank rows " + std::to_string(n) + " != the shard's " + std::to_string(want_n);
          if (l.base.find("mlp.gate_up") != std::string::npos &&
              (rows.size() != 2 || rows[0].begin != rank * r.intermediate_size ||
               rows[1].begin != g.intermediate_size + rank * r.intermediate_size))
            return l.base + ": gate_up is not a two-segment split (gate half, up half)";
          for (tp::Part p : parts) {
            tp::PartShape sh;
            sh.part = p;
            sh.N = l.N;
            sh.K = l.K;
            sh.group = group;
            (void)tp::PlanRows(sh, rows);
          }
        }
      } catch (const std::exception& e) {
        return l.base + ": " + e.what();
      }
    }
  }
  return "";
}

// ---- the tiny checkpoint ----------------------------------------------------------------------------------------------

constexpr int64_t H = 512, HEADS = 28, KV = 4, HD = 128, FFN = 512, VOCAB = 512, LAYERS = 3;

json Config() {
  return {{"architectures", {"LlamaForCausalLM"}}, {"attention_bias", false}, {"head_dim", HD}, {"hidden_act", "silu"},
          {"hidden_size", H}, {"intermediate_size", FFN}, {"max_position_embeddings", 4096}, {"mlp_bias", false},
          {"model_type", "llama"}, {"num_attention_heads", HEADS}, {"num_hidden_layers", LAYERS},
          {"num_key_value_heads", KV}, {"pretraining_tp", 1}, {"rms_norm_eps", 1e-05}, {"rope_scaling", nullptr},
          {"rope_theta", 120000000}, {"tie_word_embeddings", false}, {"vocab_size", VOCAB}};
}

struct Spec {
  std::string name;
  std::vector<int64_t> shape;
};

std::vector<Spec> Specs() {
  std::vector<Spec> s;
  for (int64_t i = 0; i < LAYERS; ++i) {
    const std::string L = "model.layers." + std::to_string(i) + ".";
    s.push_back({L + "input_layernorm.weight", {H}});
    s.push_back({L + "post_attention_layernorm.weight", {H}});
    s.push_back({L + "self_attn.q_proj.weight", {HEADS * HD, H}});
    s.push_back({L + "self_attn.k_proj.weight", {KV * HD, H}});
    s.push_back({L + "self_attn.v_proj.weight", {KV * HD, H}});
    s.push_back({L + "self_attn.o_proj.weight", {H, HEADS * HD}});
    s.push_back({L + "mlp.gate_proj.weight", {FFN, H}});
    s.push_back({L + "mlp.up_proj.weight", {FFN, H}});
    s.push_back({L + "mlp.down_proj.weight", {H, FFN}});
  }
  s.push_back({"model.embed_tokens.weight", {VOCAB, H}});
  s.push_back({"model.norm.weight", {H}});
  s.push_back({"lm_head.weight", {VOCAB, H}});
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
      const uint16_t h = r4dx::core::FloatToBf16((static_cast<int>((seed >> 24) % 9) - 4) * 0.02f);
      data.append(reinterpret_cast<const char*>(&h), 2);
    }
    hdr[s.name] = {{"dtype", "BF16"}, {"shape", s.shape}, {"data_offsets", {b, data.size()}}};
  }
  const std::string h = hdr.dump();
  const uint64_t hl = h.size();
  std::string file(reinterpret_cast<const char*>(&hl), 8);
  file += h;
  file += data;
  WriteWhole(dir / "model.safetensors", file);
  WriteWhole(dir / "config.json", Config().dump(2));
}

// Re-writes container `src` as `dst` with `edit` applied to its parsed header (the data section is carried over
// untouched: offsets are relative to it).
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

std::vector<std::string> Problems(const std::string& path, int64_t layers, bool bf16_body = true) {
  r4dx_convert::SafetensorsReader reader(r4dx_convert::Utf8ToWide(path));
  const json header = json::parse(reader.HeaderJson());
  const LlamaContainerInfo info = r4dx::model::ParseLlamaContainerMetadata(header.at("__metadata__"), path);
  return r4dx::model::LlamaTensorTableProblems(reader, info, layers, bf16_body);
}

#if defined(R4DX_CONVERT_EXE)

int Run(const std::string& args, const fs::path& log) {
  const std::string cmd = "\"\"" + std::string(R4DX_CONVERT_EXE) + "\" " + args + " > \"" + log.u8string() + "\" 2>&1\"";
  return std::system(cmd.c_str());
}

void TestTiny() {
  std::printf("---- (a) the tiny converted containers ----\n");
  const fs::path root = fs::temp_directory_path() / "r4dx_test_llama_container";
  std::error_code ec;
  fs::remove_all(root, ec);
  fs::create_directories(root);
  const fs::path ckpt = root / "ckpt-llama-tiny", c = root / "tiny.r4dx", c8 = root / "tiny-w8.r4dx", log = root / "log.txt";
  WriteCheckpoint(ckpt);
  const std::string in = "--input \"" + ckpt.u8string() + "\" --threads 2 ";
  int rc = Run(in + "--output \"" + c.u8string() + "\" --layouts bf16 --lm-head bf16", log);
  Check(rc == 0, "r4dx-convert converts the tiny checkpoint to bf16 (rc=" + std::to_string(rc) + ")");
  const int rc8 = Run(in + "--output \"" + c8.u8string() + "\" --layouts w8a16 --no-bf16", log);
  Check(rc8 == 0, "r4dx-convert converts the tiny checkpoint to w8a16 (rc=" + std::to_string(rc8) + ")");
  if (rc != 0 || rc8 != 0) return;
  const std::string path = c.u8string(), path8 = c8.u8string();

  const LlamaContainerInfo info = InspectLlamaContainer(path);
  Check(info.model_id == "ckpt-llama-tiny" && !info.config_sha256.empty() && !info.has_trellis && !info.has_w8a16,
        "Inspect (bf16): model_id, config_sha256, no trellis, no w8a16");
  const LlamaConfig& cfg = info.config;
  Check(cfg.hidden_size == H && cfg.num_hidden_layers == LAYERS && cfg.head_dim == HD && cfg.Gqa() == 7 && cfg.QDim() == HEADS * HD &&
            cfg.intermediate_size == FFN && !cfg.tie_word_embeddings && cfg.tp_world == 1,
        "Inspect: the flat model_config parsed (global config; head_dim 128 != hidden / heads)");
  Check(info.model_config.arch == Arch::kLlama && info.model_config.layer_types.size() == LAYERS,
        "Inspect: the generic ModelConfig carries kLlama (what ShardLoader / RuleFor read)");
  Check(Problems(path, LAYERS, /*bf16_body=*/true).empty() && Problems(path, LAYERS, /*bf16_body=*/false).empty(),
        "bf16: the tensor table is accepted (bf16 body required, and any-form)");

  const LlamaContainerInfo info8 = InspectLlamaContainer(path8);
  Check(info8.has_w8a16 && info8.metadata.at("quant").at("w8a16").at("group") == 128, "Inspect (w8a16): quant.w8a16 group 128");
  Check(Problems(path8, LAYERS, /*bf16_body=*/false).empty(), "w8a16: the tensor table is accepted in any-form mode");
  Check(AnyContains(Problems(path8, LAYERS, /*bf16_body=*/true), "missing tensor 'text.layers.0.attn.q.bf16.w'"),
        "w8a16: a bf16-body requirement names the missing .bf16.w (the w8a16 form is checked by its own loader)");
  {
    r4dx_convert::SafetensorsReader r8(r4dx_convert::Utf8ToWide(path8));
    bool w8 = true;
    for (const LinearSpec& l : Linears(cfg, LAYERS))
      w8 = w8 && r8.Has(l.base + ".w8a16.wq") && r8.Has(l.base + ".w8a16.ws") &&
           r8.Meta(l.base + ".w8a16.wq").shape == std::vector<int64_t>({l.N * l.K}) &&
           r8.Meta(l.base + ".w8a16.ws").shape == std::vector<int64_t>({l.N * l.K / 128, 2});
    Check(w8, "w8a16: every linear the loader reads (19) is a wq {N*K} / ws {N*K/128, 2} pair");
  }

  // The TP=2 shard of this config is legal, and so is every linear's byte plan.
  const LlamaConfig r1 = LlamaConfig::Shard(cfg, 2, 1);
  Check(r1.num_attention_heads == 14 && r1.num_key_value_heads == 2 && r1.intermediate_size == 256 && r1.VocabShardSize() == 256,
        "Shard(2, 1) of the tiny config is legal: heads 14 / kv 2, FFN 256, vocab 256");
  const std::string bf16_plan = ShardPlanProblem(cfg, LAYERS, {tp::Part::kBf16}, 128);
  Check(bf16_plan.empty(), "TP=2 bf16: every linear's rule and byte plan match the shard (" + bf16_plan + ")");
  const std::string w8_plan = ShardPlanProblem(cfg, LAYERS, {tp::Part::kW8Wq, tp::Part::kW8a16Ws}, 128);
  Check(w8_plan.empty(), "TP=2 w8a16: every linear's wq / ws byte plan is legal (" + w8_plan + ")");
  const r4dx::model::ModelConfig mc = cfg.ToModelConfig();
  Check(tp::RuleFor("text.layers.0.input_norm", mc).split == tp::Split::kReplicate &&
            tp::RuleFor("text.layers.2.post_attn_norm", mc).split == tp::Split::kReplicate &&
            tp::RuleFor("text.embed_tokens", mc).split == tp::Split::kReplicate &&
            tp::RuleFor("text.final_norm", mc).split == tp::Split::kReplicate,
        "TP=2: norms, embedding and final norm are replicated");

  // --layers N.
  {
    const fs::path c2 = root / "tiny2.r4dx";
    rc = Run(in + "--output \"" + c2.u8string() + "\" --layouts bf16 --lm-head bf16 --layers 2", log);
    Check(rc == 0 && InspectLlamaContainer(c2.u8string(), 2).model_id == "ckpt-llama-tiny", "a --layers 2 container inspects with layer_limit 2");
    Check(Throws([&] { InspectLlamaContainer(c2.u8string()); }, "layers.2."), "... and is refused with the full layer count (layer 2's tensors are missing)");
  }

  // ---- (b) header edits ----
  std::printf("---- (b) header edits of the container ----\n");
  const fs::path e = root / "edited.r4dx";
  auto problems_after = [&](auto&& edit, int64_t layers = LAYERS) {
    Rewrite(c, e, edit);
    return Problems(e.u8string(), layers);
  };
  Check(AnyContains(problems_after([](json& h) { h.erase("text.layers.1.mlp.down.bf16.w"); }), "missing tensor 'text.layers.1.mlp.down.bf16.w'"),
        "a missing linear is named");
  // (Shape edits keep the byte span: the safetensors reader itself refuses a shape that does not fit its data.)
  Check(AnyContains(problems_after([](json& h) { h["text.layers.0.attn.k.bf16.w"]["shape"] = {KV * HD / 2, H * 2, 2}; }), "attn.k"),
        "a k of the wrong [N, K] (same bytes) is named");
  Check(AnyContains(problems_after([](json& h) { h["text.layers.2.input_norm"]["shape"] = {H / 2, 2, 2}; }), "input_norm"), "a wrong norm shape is named");
  Check(AnyContains(problems_after([](json& h) { h.erase("lm_head.bf16.w"); }), "missing lm_head"), "a missing lm_head is reported");
  Check(AnyContains(problems_after([](json& h) { h.erase("text.embed_tokens"); }), "text.embed_tokens"), "a missing embedding is reported");
  Check(problems_after([](json& h) { h.erase("text.layers.2.mlp.down.bf16.w"); h.erase("text.layers.2.attn.q.bf16.w"); }, 2).empty(),
        "a problem in layer 2 is not asked about when only 2 layers are loaded");

  auto inspect_after = [&](auto&& edit) {
    Rewrite(c, e, edit);
    return [&] { (void)InspectLlamaContainer(e.u8string()); };
  };
  Check(Throws(inspect_after([](json& h) { h["__metadata__"]["model_arch"] = "lfm2_moe"; }), "not llama"), "an lfm2_moe container is refused");
  Check(Throws(inspect_after([](json& h) { h["__metadata__"].erase("model_arch"); }), "not llama"), "a container without model_arch is refused");
  Check(Throws(inspect_after([](json& h) { h["__metadata__"]["norm_kind"] = "zero_centered"; }), "norm_kind"), "norm_kind other than plain is refused");
  Check(Throws(inspect_after([](json& h) { h["__metadata__"]["rotation"] = {{"kind", "q2a"}}; }), "rotat"), "a rotated container is refused");
  Check(Throws(inspect_after([](json& h) { h["__metadata__"].erase("model_config"); }), "model_config"), "a container without model_config is refused");
  Check(Throws(inspect_after([](json& h) { h["__metadata__"]["model_config"]["attention_bias"] = true; }), "attention_bias"),
        "a model_config the runtime does not implement is refused");
  Check(Throws(inspect_after([](json& h) { h["text.layers.0.mlp.gate_up.bf16.w"]["shape"] = {H, 2 * FFN, 2}; }), "mlp.gate_up"),
        "Inspect throws on a bad table, naming the tensor");

  fs::remove_all(root, ec);
}

#endif  // R4DX_CONVERT_EXE

// ---- (c) the shard table on the real geometry --------------------------------------------------------------------

json RealConfig() {
  json c = Config();
  c["hidden_size"] = 7168;
  c["num_attention_heads"] = 56;
  c["num_key_value_heads"] = 8;
  c["intermediate_size"] = 22016;
  c["num_hidden_layers"] = 52;
  c["vocab_size"] = 166912;
  c["max_position_embeddings"] = 524288;
  return c;
}

void TestRealShardTable() {
  std::printf("---- (c) tp::RuleFor's Llama table on the real geometry ----\n");
  const LlamaConfig g = LlamaConfig::FromJson(RealConfig());
  const std::string bf16_plan = ShardPlanProblem(g, 52, {tp::Part::kBf16}, 128);
  Check(bf16_plan.empty(), "real geometry, TP=2 bf16: all 313 linears' rules and byte plans match LlamaConfig::Shard (" + bf16_plan + ")");
  const std::string w8_plan = ShardPlanProblem(g, 52, {tp::Part::kW8Wq, tp::Part::kW8a16Ws}, 128);
  Check(w8_plan.empty(), "real geometry, TP=2 w8a16 g128: every wq / ws byte plan is legal (" + w8_plan + ")");
  const r4dx::model::ModelConfig mc = g.ToModelConfig();
  Check(Throws([&] { tp::RuleFor("text.layers.3.attn.q_norm", mc); }, "unknown tensor"), "RuleFor: a q_norm (LFM2 / Qwen only) is unknown on llama");
  Check(Throws([&] { tp::RuleFor("text.layers.3.gdn.in_proj_qkv", mc); }, "unknown tensor"), "RuleFor: a GDN tensor is unknown on llama");
  Check(Throws([&] { tp::RuleFor("text.layers.52.attn.q", mc); }, "layer 52"), "RuleFor: layer 52 of a 52-layer model is refused");
  for (const char* n : {"mtp.attn.q", "rotation.signs", "vision.patch_embed", "dflash.fc"})
    Check(Throws([&] { tp::RuleFor(n, mc); }, "unknown tensor"), std::string("RuleFor: '") + n + "' is unknown on llama (no MTP / rotation / tower / drafter)");
  Check(Throws([&] { tp::RuleFor("text.layers.0.attn.q.w8a16", mc); }, "unknown tensor"), "RuleFor: a name with its layout suffix is refused");
}

// ---- (d) the real Nanbeige w8a16 container, header only --------------------------------------------------------------

void TestRealContainer() {
  std::printf("---- (d) the real Nanbeige w8a16 container header ----\n");
  const fs::path p = fs::path(r4dx::ModelsPath("r4dx/nanbeige/nanbeige43-33b-w8a16-g128.r4dx"));
  if (!fs::exists(p)) {
    std::printf("SKIP: %s not present\n", p.string().c_str());
    ++g_skipped;
    return;
  }
  const LlamaContainerInfo info = InspectLlamaContainer(p.u8string());
  Check(info.model_id == "Nanbeige4.3-33B" && info.has_w8a16 && !info.has_trellis && info.config.num_hidden_layers == 52 &&
            info.config.vocab_size == 166912 && info.config.Gqa() == 7,
        "real container: Inspect accepts it (model_id, w8a16, 52 layers, vocab 166912, GQA 7)");
  Check(info.metadata.at("quant").at("w8a16").at("group") == 128, "real container: quant.w8a16 group 128");
  r4dx_convert::SafetensorsReader r(r4dx_convert::Utf8ToWide(p.u8string()));
  bool w8 = true, no_bf16 = true;
  for (const LinearSpec& l : Linears(info.config, 52)) {
    w8 = w8 && r.Has(l.base + ".w8a16.wq") && r.Meta(l.base + ".w8a16.wq").shape == std::vector<int64_t>({l.N * l.K});
    no_bf16 = no_bf16 && !r.Has(l.base + ".bf16.w");
  }
  Check(w8 && no_bf16, "real container: all 313 linears (lm_head included) are w8a16 only");
  Check(Problems(p.u8string(), 8, /*bf16_body=*/false).empty(), "real container: a layer_limit 8 table check passes too (the nanbeige-l8 path)");
}

}  // namespace

int main() {
  try {
#if defined(R4DX_CONVERT_EXE)
    TestTiny();
#else
    std::printf("SKIP: r4dx-convert not built (parts a, b)\n");
    ++g_skipped;
#endif
    TestRealShardTable();
    TestRealContainer();
  } catch (const std::exception& e) {
    std::fprintf(stderr, "FAIL: uncaught exception: %s\n", e.what());
    ++g_failures;
  }
  std::printf("\n%s: %d failure(s), %d skipped part(s)\n", g_failures == 0 ? "OK" : "FAILED", g_failures, g_skipped);
  return g_failures == 0 ? 0 : 1;
}
