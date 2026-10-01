// tests/model/test_gemma_container.cpp -- CPU-only (ctest `test_gemma_container`). docs/gemma4-plan.md M1-19.
//
// GemmaContainer's CPU half (Inspect: `__metadata__` + the tensor directory, no HIP call, no weight byte read)
// on
//   (a) a tiny synthetic gemma4_unified checkpoint (6 layers = 5 sliding + 1 full, hidden 768, the real tensor
//       names, one model.safetensors) converted by the REAL r4dx-convert (when it is built): the container
//       parses, the config / layer geometry / EmbedScale are right, the tensor table matches, and every
//       refusal names its cause (a missing tensor, a wrong shape, a stray attn.v on a full layer, a Qwen-style
//       metadata, a non-plain norm_kind, --layer limits);
//   (b) the real D:\models\r4dx\huihui-gemma\bf16.r4dx header (R4DX_GEMMA_BF16_CONTAINER overrides; SKIP when
//       absent): 48 layers, 715 tensors, every table entry, the 8 full layers without attn.v, the 131072
//       default context and the 262144 opt-in;
//   (c) rotation metadata: the real config with a Gemma option-A rotation block (hidden 3840 = 15 x 256) parses
//       to post_norm_rotate / has_o_full, with o_full_elems 8192, and a Qwen-style (no out_fold) block is refused.
// No GPU work: nothing here calls a HIP function.
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
#include <string>
#include <vector>

#include "gemma_container.h"
#include "gemma_container_info.h"
#include "nlohmann/json.hpp"

namespace fs = std::filesystem;
using nlohmann::json;
using namespace r4dx::model;

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
    const bool ok = substr.empty() || std::string(e.what()).find(substr) != std::string::npos;
    if (!ok) std::fprintf(stderr, "  (threw '%s', wanted '%s')\n", e.what(), substr.c_str());
    return ok;
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

// ---- the tiny checkpoint ---------------------------------------------------------------------------------
constexpr int64_t kHidden = 768, kHeads = 4, kKvS = 2, kKvF = 1, kHdS = 64, kHdF = 128, kInter = 512, kVocab = 64;
constexpr int kLayers = 6;

uint16_t Bf16(float f) {
  uint32_t b;
  std::memcpy(&b, &f, 4);
  return static_cast<uint16_t>((b + 0x7FFFu + ((b >> 16) & 1u)) >> 16);
}

json TinyConfig() {
  json types = json::array();
  for (int i = 0; i < kLayers; ++i) types.push_back((i + 1) % 6 == 0 ? "full_attention" : "sliding_attention");
  return {{"architectures", {"Gemma4UnifiedForConditionalGeneration"}},
          {"model_type", "gemma4_unified"},
          {"eos_token_id", {1, 106}},
          {"text_config",
           {{"model_type", "gemma4_unified_text"},
            {"hidden_size", kHidden},
            {"num_hidden_layers", kLayers},
            {"layer_types", types},
            {"num_attention_heads", kHeads},
            {"num_key_value_heads", kKvS},
            {"num_global_key_value_heads", kKvF},
            {"head_dim", kHdS},
            {"global_head_dim", kHdF},
            {"intermediate_size", kInter},
            {"vocab_size", kVocab},
            {"sliding_window", 8},
            {"max_position_embeddings", 256},
            {"rms_norm_eps", 1e-6},
            {"final_logit_softcapping", 30.0},
            {"hidden_activation", "gelu_pytorch_tanh"},
            {"attention_k_eq_v", true},
            {"tie_word_embeddings", true},
            {"rope_parameters",
             {{"sliding_attention", {{"rope_type", "default"}, {"rope_theta", 10000.0}}},
              {"full_attention",
               {{"rope_type", "proportional"}, {"rope_theta", 1000000.0}, {"partial_rotary_factor", 0.25}}}}}}}};
}

void WriteTinyCheckpoint(const fs::path& dir) {
  fs::create_directories(dir);
  std::mt19937_64 rng(7);
  std::normal_distribution<double> nd(0.0, 1.0);
  json hdr = json::object();
  std::string data;
  auto add = [&](const std::string& name, std::vector<int64_t> shape, double mean, double sigma) {
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    const uint64_t b = data.size();
    for (int64_t i = 0; i < n; ++i) {
      const uint16_t h = Bf16(static_cast<float>(mean + sigma * nd(rng)));
      data.append(reinterpret_cast<const char*>(&h), 2);
    }
    hdr[name] = {{"dtype", "BF16"}, {"shape", shape}, {"data_offsets", {b, data.size()}}};
  };
  for (int i = 0; i < kLayers; ++i) {
    const bool full = (i + 1) % 6 == 0;
    const std::string L = "model.language_model.layers." + std::to_string(i) + ".";
    const int64_t hd = full ? kHdF : kHdS, kv = full ? kKvF : kKvS;
    for (const char* n : {"input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm"})
      add(L + n + ".weight", {kHidden}, 1.0, 0.2);
    add(L + "self_attn.q_proj.weight", {kHeads * hd, kHidden}, 0, 0.05);
    add(L + "self_attn.k_proj.weight", {kv * hd, kHidden}, 0, 0.05);
    if (!full) add(L + "self_attn.v_proj.weight", {kv * hd, kHidden}, 0, 0.05);
    add(L + "self_attn.o_proj.weight", {kHidden, kHeads * hd}, 0, 0.05);
    add(L + "self_attn.q_norm.weight", {hd}, 1.0, 0.2);
    add(L + "self_attn.k_norm.weight", {hd}, 1.0, 0.2);
    add(L + "mlp.gate_proj.weight", {kInter, kHidden}, 0, 0.05);
    add(L + "mlp.up_proj.weight", {kInter, kHidden}, 0, 0.05);
    add(L + "mlp.down_proj.weight", {kHidden, kInter}, 0, 0.05);
    add(L + "layer_scalar", {1}, 0.6, 0.0);
  }
  add("model.language_model.embed_tokens.weight", {kVocab, kHidden}, 0, 0.1);
  add("model.language_model.norm.weight", {kHidden}, 1.0, 0.2);
  const std::string h = hdr.dump();
  const uint64_t hl = h.size();
  std::string file(reinterpret_cast<const char*>(&hl), 8);
  file += h;
  file += data;
  WriteWhole(dir / "model.safetensors", file);
  WriteWhole(dir / "config.json", TinyConfig().dump(2));
}

// Rewrites a safetensors container's header through `edit` (offsets are relative to the data region, so the
// data bytes are copied unchanged).
void MutateHeader(const fs::path& src, const fs::path& dst, const std::function<void(json&)>& edit) {
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

std::string Tail(const std::string& s) { return s.size() > 300 ? s.substr(s.size() - 300) : s; }

#if defined(R4DX_CONVERT_EXE)
int Run(const std::string& args, const fs::path& log) {
  const std::string cmd = "\"\"" + std::string(R4DX_CONVERT_EXE) + "\" " + args + " > \"" + log.u8string() + "\" 2>&1\"";
  return std::system(cmd.c_str());
}

void TestTinyConverted() {
  std::printf("---- (a) the tiny fixture through r4dx-convert -> GemmaContainer::Inspect ----\n");
  const fs::path root = fs::temp_directory_path() / "r4dx_test_gemma_container";
  std::error_code ec;
  fs::remove_all(root, ec);
  fs::create_directories(root);
  const fs::path ckpt = root / "ckpt-gemma-tiny6", out = root / "tiny.r4dx", log = root / "convert.log";
  WriteTinyCheckpoint(ckpt);
  const int rc = Run("--input \"" + ckpt.u8string() + "\" --output \"" + out.u8string() +
                         "\" --threads 2 --layouts bf16 --lm-head bf16",
                     log);
  if (rc != 0) {
    Check(false, "r4dx-convert converts the 6-layer tiny checkpoint (rc " + std::to_string(rc) + ": " + Tail(ReadWhole(log)) + ")");
    return;
  }
  Check(true, "r4dx-convert converts the 6-layer tiny checkpoint");

  const GemmaContainerInfo info = GemmaContainer::Inspect(out.u8string());
  const GemmaConfig& c = info.config;
  Check(c.hidden_size == kHidden && c.num_hidden_layers == kLayers && c.IsFullLayer(5) && !c.IsFullLayer(0) &&
            c.NumFullLayers() == 1 && c.sliding_window == 8,
        "Inspect: the config (768 hidden, 6 layers, full at 5, window 8)");
  Check(c.HeadDim(0) == kHdS && c.HeadDim(5) == kHdF && c.NumKvHeads(0) == kKvS && c.NumKvHeads(5) == kKvF &&
            c.QDim(5) == kHeads * kHdF && c.RotaryAngles(5) == 16 && c.RotaryAngles(0) == 32,
        "Inspect: per-layer geometry (head dims 64 / 128, kv heads 2 / 1, rope pairs 32 / 16)");
  Check(c.EmbedScale() == 27.75f, "EmbedScale = bf16(sqrt(768) = 27.7128) = 27.75");
  Check(!info.rotation && !info.has_trellis && info.model_id == "ckpt-gemma-tiny6" &&
            info.model_config.arch == Arch::kGemma4 && info.model_config.hidden_size == kHidden,
        "Inspect: no rotation / trellis, model_id, ModelConfig::arch = Gemma4");
  Check(c.ResolveMaxCtx(0, false) == 256 && Throws([&] { c.ResolveMaxCtx(300, false); }, "opt-in"),
        "ResolveMaxCtx: default is the config's max_position_embeddings; more needs the opt-in");

  {
    r4dx_convert::SafetensorsReader r(r4dx_convert::Utf8ToWide(out.u8string()));
    Check(GemmaTensorTableProblems(r, info, kLayers, /*bf16_body=*/true).empty(), "the tensor table has every name and shape");
    Check(r.Has("text.layers.0.attn.v.bf16.w") && !r.Has("text.layers.5.attn.v.bf16.w"), "v on sliding layers only");
    Check(r.Meta("text.layers.5.attn.k_descale").shape == std::vector<int64_t>({kKvF, 4}) &&
              r.Meta("text.layers.0.attn.k_descale").shape == std::vector<int64_t>({kKvS, 4}),
          "per-layer descale head counts (2 sliding, 1 full)");
  }
  Check(GemmaContainer::Inspect(out.u8string(), 3).config.num_hidden_layers == kLayers,
        "Inspect(layer_limit 3): the config stays global, only the first 3 layers are checked");

  // ---- refusals ----
  auto mutated = [&](const std::string& tag, const std::function<void(json&)>& edit) {
    const fs::path p = root / (tag + ".r4dx");
    MutateHeader(out, p, edit);
    return p.u8string();
  };
  Check(Throws([&] { GemmaContainer::Inspect(mutated("no_q", [](json& h) { h.erase("text.layers.3.attn.q.bf16.w"); })); },
               "missing tensor 'text.layers.3.attn.q.bf16.w'"),
        "a missing tensor is named");
  Check(Throws([&] {
          GemmaContainer::Inspect(mutated("bad_shape", [](json& h) {
            h["text.layers.5.attn.o.bf16.w"]["shape"] = json::array({kHeads * kHdF, kHidden, 2});  // transposed, same bytes
          }));
        }, "tensor 'text.layers.5.attn.o.bf16.w' has shape"),
        "a wrong shape is named (full-layer o_proj K)");
  Check(Throws([&] {
          GemmaContainer::Inspect(mutated("stray_v", [](json& h) { h["text.layers.5.attn.v.bf16.w"] = h["text.layers.5.attn.k.bf16.w"]; }));
        }, "k_eq_v full layer but the container carries attn.v"),
        "an attn.v on a k_eq_v full layer is refused");
  Check(Throws([&] {
          GemmaContainer::Inspect(mutated("no_scalar", [](json& h) { h.erase("text.layers.2.layer_scalar"); }));
        }, "text.layers.2.layer_scalar"),
        "a missing layer_scalar is named");
  Check(Throws([&] { GemmaContainer::Inspect(mutated("qwenmeta", [](json& h) { h["__metadata__"].erase("model_arch"); })); }, "model_arch"),
        "a container without model_arch=gemma4_unified is refused");
  Check(Throws([&] { GemmaContainer::Inspect(mutated("normkind", [](json& h) { h["__metadata__"]["norm_kind"] = "zero_centered"; })); }, "norm_kind"),
        "a non-plain norm_kind is refused");
  Check(Throws([&] {
          GemmaContainer::Inspect(mutated("badpattern", [](json& h) {
            std::string mc = h["__metadata__"]["model_config"].dump();
            json j = json::parse(mc);
            j["text_config"]["layer_types"][2] = "full_attention";
            h["__metadata__"]["model_config"] = j;
          }));
        }, "5 sliding : 1 full"),
        "a layer_types list that is not the 5:1 pattern is refused");
  Check(Throws([&] { GemmaContainer::Inspect((root / "missing.r4dx").u8string()); }), "a missing file throws");
}
#endif  // R4DX_CONVERT_EXE

// ---- (b) the real header ----------------------------------------------------------------------------------
void TestRealContainer() {
  std::printf("---- (b) the real Huihui bf16.r4dx ----\n");
  const char* env = std::getenv("R4DX_GEMMA_BF16_CONTAINER");
  const std::string path = env != nullptr && *env != '\0' ? env : "D:\\models\\r4dx\\huihui-gemma\\bf16.r4dx";
  if (!fs::exists(path)) {
    std::printf("SKIP: %s not present (part (b) not run)\n", path.c_str());
    return;
  }
  const GemmaContainerInfo info = GemmaContainer::Inspect(path);
  const GemmaConfig& c = info.config;
  Check(c.hidden_size == 3840 && c.num_hidden_layers == 48 && c.NumFullLayers() == 8 && c.NumSlidingLayers() == 40 &&
            c.sliding_window == 1024 && c.vocab_size == 262144 && c.final_logit_softcapping == 30.0,
        "real config: 3840 hidden, 48 layers (40 sliding + 8 full), window 1024, vocab 262144, softcap 30");
  Check(c.HeadDim(0) == 256 && c.HeadDim(5) == 512 && c.NumKvHeads(0) == 8 && c.NumKvHeads(5) == 1 && c.QDim(5) == 8192 &&
            c.RotaryAngles(5) == 64 && c.RotaryAngles(0) == 128 && c.EmbedScale() == 62.0f,
        "real geometry: 256 / 512 head dims, 8 / 1 kv heads, rope pairs 128 / 64, EmbedScale 62.0 (bf16)");
  Check(c.ResolveMaxCtx(0, false) == 131072 && c.ResolveMaxCtx(262144, true) == 262144 &&
            Throws([&] { c.ResolveMaxCtx(262144, false); }, "opt-in"),
        "real context: default 131072, 262144 only with the opt-in");
  Check(!info.rotation && !info.has_trellis, "real bf16 container: no rotation, no trellis");
  r4dx_convert::SafetensorsReader r(r4dx_convert::Utf8ToWide(path));
  Check(r.Names().size() == 715, "715 tensors");
  const std::vector<std::string> bad = GemmaTensorTableProblems(r, info, 48, /*bf16_body=*/true);
  for (size_t i = 0; i < bad.size() && i < 5; ++i) std::fprintf(stderr, "  problem: %s\n", bad[i].c_str());
  Check(bad.empty(), "the real tensor table matches the model (every name and shape of 48 layers)");
  bool no_v = true;
  for (int i = 5; i < 48; i += 6) no_v = no_v && !r.Has("text.layers." + std::to_string(i) + ".attn.v.bf16.w");
  Check(no_v && r.Has("text.layers.0.attn.v.bf16.w"), "attn.v on the 40 sliding layers only");
  Check(r.Meta("text.layers.5.layer_scalar").shape == std::vector<int64_t>({1, 4}), "layer_scalar is fp32 [1] (stored [1, 4] bytes)");
}

// ---- (c) rotation metadata --------------------------------------------------------------------------------
void TestRotationMetadata() {
  std::printf("---- (c) option-A rotation metadata on the real config ----\n");
  json types = json::array();
  for (int i = 0; i < 48; ++i) types.push_back((i + 1) % 6 == 0 ? "full_attention" : "sliding_attention");
  json cfg = {{"model_type", "gemma4_unified"},
              {"text_config",
               {{"model_type", "gemma4_unified_text"},
                {"hidden_size", 3840},
                {"num_hidden_layers", 48},
                {"layer_types", types},
                {"num_attention_heads", 16},
                {"num_key_value_heads", 8},
                {"num_global_key_value_heads", 1},
                {"head_dim", 256},
                {"global_head_dim", 512},
                {"intermediate_size", 15360},
                {"vocab_size", 262144},
                {"sliding_window", 1024},
                {"max_position_embeddings", 131072},
                {"rms_norm_eps", 1e-6},
                {"final_logit_softcapping", 30.0},
                {"hidden_activation", "gelu_pytorch_tanh"},
                {"attention_k_eq_v", true},
                {"tie_word_embeddings", true},
                {"rope_parameters",
                 {{"sliding_attention", {{"rope_type", "default"}, {"rope_theta", 10000.0}}},
                  {"full_attention", {{"rope_type", "proportional"}, {"rope_theta", 1000000.0}, {"partial_rotary_factor", 0.25}}}}}}}};
  json md = {{"model_arch", "gemma4_unified"}, {"norm_kind", "plain"}, {"model_config", cfg}, {"model_id", "synthetic"}};
  const GemmaContainerInfo plain = ParseGemmaContainerMetadata(md, "synthetic.r4dx");
  Check(!plain.rotation && plain.full_attn_out_elems == 8192, "unrotated synthetic metadata: no rotation, o_full_elems 8192");
  md["rotation"] = {{"kind", "q2ab"}, {"seed", 1}, {"hidden", 3840}, {"block", 256}, {"nblk", 15},
                    {"out_fold", "had_only"}, {"had", {{"down", 512}, {"o", 256}, {"o_full", 256}}}};
  const GemmaContainerInfo rot = ParseGemmaContainerMetadata(md, "synthetic.r4dx");
  Check(rot.rotation && rot.rotation->post_norm_rotate && rot.rotation->has_o_full && !rot.rotation->has_gdn_out &&
            rot.rotation->nblk == 15 && rot.rotation->block == 256 && rot.rotation->Hadamard(),
        "q2ab option A: post_norm_rotate, has_o_full, no GDN, 15 x 256");
  const auto tensors = RotationTensors(*rot.rotation, rot.model_config, rot.full_attn_out_elems);
  Check(tensors.size() == 5 && tensors[0].elems == 3840 && tensors[1].elems == 225 && tensors[2].elems == 15360 &&
            tensors[3].elems == 4096 && tensors[4].elems == 8192,
        "rotation tensors: signs 3840, mix 15x15, had_down 15360, had_o 4096, had_o_full 8192");
  json qwen_style = md;
  qwen_style["rotation"].erase("out_fold");
  Check(Throws([&] { ParseGemmaContainerMetadata(qwen_style, "synthetic.r4dx"); }), "a Qwen-style rotation block (no out_fold) is refused");
  json q2a = md;
  q2a["rotation"] = {{"kind", "q2a"}, {"seed", 1}, {"hidden", 3840}, {"block", 256}, {"nblk", 15}, {"out_fold", "had_only"}};
  Check(ParseGemmaContainerMetadata(q2a, "synthetic.r4dx").rotation->Hadamard() == false, "q2a option A parses (no Hadamard)");
  json trellis = md;
  trellis.erase("rotation");
  trellis["quant"] = {{"trellis", {{"linears", json::object()}}}};
  Check(ParseGemmaContainerMetadata(trellis, "synthetic.r4dx").has_trellis, "a trellis block is detected");
}

}  // namespace

int main() {
  try {
#if defined(R4DX_CONVERT_EXE)
    TestTinyConverted();
#else
    std::printf("SKIP: r4dx-convert is not built, part (a) not run\n");
#endif
    TestRealContainer();
    TestRotationMetadata();
  } catch (const std::exception& e) {
    std::fprintf(stderr, "FAIL: uncaught exception: %s\n", e.what());
    return 1;
  }
  if (g_failures != 0) {
    std::fprintf(stderr, "%d check(s) FAILED\n", g_failures);
    return 1;
  }
  std::printf("all checks passed\n");
  return 0;
}
