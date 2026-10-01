// tests/convert/test_dflash_hf.cpp -- CPU-only (ctest `convert_dflash_hf`). docs/gemma4-plan.md D-3:
// r4dx_convert::ConvertDflashHf (r4dx-convert --dflash-hf) on a synthetic tiny HF DFlash v1 checkpoint of the
// z-lab/gemma4-12B-it-DFlash shape (3 layers: 2 sliding + 1 full, block 16, softcap 30, mask id 4, tied head,
// target_layer_ids [1, 2]). Everything is generated here: config.json + one model.safetensors.
//
//   * tensor mapping: every linear's `.bf16.w` bytes equal the HF tensor's bytes (no transpose, no float
//     round trip loss), every norm is the widened fp32, the tensor set is exactly the dflash2 set (no stray
//     tensor, none missing);
//   * the synthesized identity conv: conv.base[side][tap0] == 1.0, [side][tap1] == 0.0 for every side of both
//     sublayers of every layer, conv.proj all zero; the selector hidden and both codebooks all zero;
//   * metadata: target_layers = ids + offset (1 by default, 0 with the flag), block_size 16, rope theta 1e6,
//     sliding pattern from layer_types, logit_softcap 30, embed_scale 1.0 (the checkpoint has no
//     input_embedding_scale), variant v1_identity, mask id, vocab, context;
//   * the w4a16 layout converts too, and the zero linears decode to exactly zero (scale 0);
//   * refusals: a missing tensor, a wrong shape, an unconsumed tensor, output_multiplier != 1, a bad offset.
// Writes a few MB under %TEMP% and removes them.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"
#include "r4dx/core/dtype.hpp"
#include "r4dx_convert/dflash2_hf.hpp"
#include "r4dx_convert/safetensors_reader.hpp"

namespace fs = std::filesystem;
using nlohmann::json;
using namespace r4dx_convert;

namespace {

int g_fail = 0, g_checks = 0;
void Check(bool c, const std::string& what) {
  ++g_checks;
  if (!c) {
    ++g_fail;
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
  } else {
    std::printf("PASS: %s\n", what.c_str());
  }
}

constexpr int64_t kHidden = 128, kFfn = 256, kHeads = 4, kKv = 2, kHd = 32, kLayers = 3, kVocab = 512, kBlock = 16;
const std::vector<int64_t> kIds = {1, 2};

struct Tensor {
  std::vector<int64_t> shape;
  std::vector<uint16_t> bf16;
};
using Ckpt = std::map<std::string, Tensor>;

Ckpt MakeCkpt(std::mt19937_64& rng) {
  std::normal_distribution<float> nd(0.0f, 1.0f);
  Ckpt c;
  auto put = [&](const std::string& n, std::vector<int64_t> shape, float sigma, float mu = 0.0f) {
    Tensor t;
    t.shape = shape;
    int64_t cnt = 1;
    for (auto d : shape) cnt *= d;
    t.bf16.resize(static_cast<size_t>(cnt));
    for (auto& v : t.bf16) v = r4dx::core::FloatToBf16(mu + sigma * nd(rng));
    c[n] = std::move(t);
  };
  put("fc.weight", {kHidden, static_cast<int64_t>(kIds.size()) * kHidden}, 0.05f);
  put("hidden_norm.weight", {kHidden}, 0.1f, 1.0f);
  put("norm.weight", {kHidden}, 0.1f, 1.0f);
  for (int i = 0; i < kLayers; ++i) {
    const std::string h = "layers." + std::to_string(i) + ".";
    put(h + "input_layernorm.weight", {kHidden}, 0.1f, 1.0f);
    put(h + "post_attention_layernorm.weight", {kHidden}, 0.1f, 1.0f);
    put(h + "self_attn.q_proj.weight", {kHeads * kHd, kHidden}, 0.05f);
    put(h + "self_attn.k_proj.weight", {kKv * kHd, kHidden}, 0.05f);
    put(h + "self_attn.v_proj.weight", {kKv * kHd, kHidden}, 0.05f);
    put(h + "self_attn.o_proj.weight", {kHidden, kHeads * kHd}, 0.05f);
    put(h + "self_attn.q_norm.weight", {kHd}, 0.1f, 1.0f);
    put(h + "self_attn.k_norm.weight", {kHd}, 0.1f, 1.0f);
    put(h + "mlp.gate_proj.weight", {kFfn, kHidden}, 0.05f);
    put(h + "mlp.up_proj.weight", {kFfn, kHidden}, 0.05f);
    put(h + "mlp.down_proj.weight", {kHidden, kFfn}, 0.05f);
  }
  return c;
}

void WriteSafetensors(const fs::path& p, const Ckpt& c) {
  json header = json::object();
  uint64_t off = 0;
  for (const auto& [name, t] : c) {
    const uint64_t n = t.bf16.size() * 2;
    header[name] = {{"dtype", "BF16"}, {"shape", t.shape}, {"data_offsets", {off, off + n}}};
    off += n;
  }
  header["__metadata__"] = {{"format", "pt"}};
  const std::string h = header.dump();
  std::ofstream f(p, std::ios::binary | std::ios::trunc);
  const uint64_t hl = h.size();
  f.write(reinterpret_cast<const char*>(&hl), 8);
  f.write(h.data(), static_cast<std::streamsize>(h.size()));
  for (const auto& [name, t] : c) f.write(reinterpret_cast<const char*>(t.bf16.data()), static_cast<std::streamsize>(t.bf16.size() * 2));
}

json Config() {
  return json{{"architectures", {"DFlashDraftModel"}},
              {"block_size", kBlock},
              {"dflash_config", {{"mask_token_id", 4}, {"target_layer_ids", kIds}}},
              {"final_logit_softcapping", 30.0},
              {"head_dim", kHd},
              {"hidden_size", kHidden},
              {"intermediate_size", kFfn},
              {"layer_types", {"sliding_attention", "sliding_attention", "full_attention"}},
              {"max_position_embeddings", 262144},
              {"num_attention_heads", kHeads},
              {"num_hidden_layers", kLayers},
              {"num_key_value_heads", kKv},
              {"rms_norm_eps", 1e-6},
              {"rope_parameters", {{"rope_theta", 1000000}, {"rope_type", "default"}}},
              {"sliding_window", 2048},
              {"use_sliding_window", true},
              {"vocab_size", kVocab}};
}

void WriteDir(const fs::path& d, const Ckpt& c, const json& cfg) {
  fs::create_directories(d);
  std::ofstream(d / "config.json") << cfg.dump(2);
  WriteSafetensors(d / "model.safetensors", c);
}

json ReadMeta(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  uint64_t hl = 0;
  f.read(reinterpret_cast<char*>(&hl), 8);
  std::string s(static_cast<size_t>(hl), '\0');
  f.read(s.data(), static_cast<std::streamsize>(hl));
  return json::parse(s).at("__metadata__");
}

bool Throws(const std::function<void()>& fn, const std::string& needle) {
  try {
    fn();
  } catch (const std::exception& e) {
    return std::string(e.what()).find(needle) != std::string::npos;
  }
  return false;
}

bool AllZero(const uint8_t* p, size_t n) {
  for (size_t i = 0; i < n; ++i)
    if (p[i] != 0) return false;
  return true;
}

}  // namespace

int main() try {
  const fs::path root = fs::temp_directory_path() / "r4dx_test_dflash_hf";
  fs::remove_all(root);
  std::mt19937_64 rng(0xD17A5ull);
  const Ckpt ck = MakeCkpt(rng);
  WriteDir(root / "hf", ck, Config());

  // ---- bf16 layout, default offset 1 ------------------------------------------------------------------
  const std::string out_bf = (root / "bf16.r4dx").string();
  {
    DflashHfOptions o;
    o.input_dir = (root / "hf").string();
    o.output_path = out_bf;
    o.layout = "bf16";
    o.quiet = true;
    ConvertDflashHf(o);
  }
  const json md = ReadMeta(out_bf);
  const json& d = md.at("dflash2");
  Check(md.at("container_kind") == "dflash2_draft", "container_kind dflash2_draft");
  Check(d.at("block_size") == kBlock, "block_size 16");
  Check(d.at("target_layers").get<std::vector<int64_t>>() == std::vector<int64_t>({2, 3}), "target_layers = ids + 1");
  Check(d.at("target_layer_offset") == 1, "target_layer_offset recorded");
  Check(d.at("mask_token_id") == 4, "mask_token_id 4");
  Check(d.at("logit_softcap").get<double>() == 30.0, "logit_softcap 30");
  Check(d.at("embed_scale").get<double>() == 1.0, "embed_scale 1.0 (no input_embedding_scale in the config)");
  Check(d.at("variant") == "v1_identity", "variant v1_identity");
  Check(d.at("rope").at("freq_base").get<double>() == 1e6, "rope theta 1e6");
  Check(d.at("rope").at("n_rot") == kHd, "n_rot = head_dim");
  Check(d.at("attention").at("sliding_window_pattern").get<std::vector<bool>>() == std::vector<bool>({true, true, false}),
        "sliding pattern from layer_types");
  Check(d.at("attention").at("causal") == false, "non-causal");
  Check(d.at("vocab_size") == kVocab && d.at("selector_rank") == 256 && d.at("selector_top_k") == 16, "vocab / selector geometry");
  Check(d.at("conv_kernel_size") == 2 && d.at("conv_group_size") == 16, "conv geometry 2 / 16");
  Check(d.at("layout") == "bf16", "layout bf16");
  Check(d.at("context_length") == 262144, "context_length");

  auto r_owner = std::make_unique<SafetensorsReader>(Utf8ToWide(out_bf));
  SafetensorsReader& r = *r_owner;
  // linears: bytes identical to the HF tensors
  auto lin_ok = [&](const std::string& hf, const std::string& base) {
    if (!r.Has(base + ".bf16.w")) return false;
    const Tensor& t = ck.at(hf);
    const auto& m = r.Meta(base + ".bf16.w");
    if (m.shape != std::vector<int64_t>({t.shape[0], t.shape[1], 2})) return false;
    return std::memcmp(r.Data(base + ".bf16.w"), t.bf16.data(), t.bf16.size() * 2) == 0;
  };
  auto norm_ok = [&](const std::string& hf, const std::string& name) {
    if (!r.Has(name)) return false;
    const Tensor& t = ck.at(hf);
    const float* p = reinterpret_cast<const float*>(r.Data(name));
    for (size_t i = 0; i < t.bf16.size(); ++i)
      if (p[i] != r4dx::core::Bf16ToFloat(t.bf16[i])) return false;
    return (r.Meta(name).end - r.Meta(name).begin) == t.bf16.size() * 4;
  };
  Check(lin_ok("fc.weight", "dflash.fc"), "fc.weight -> dflash.fc bytes");
  Check(norm_ok("hidden_norm.weight", "dflash.enc_output_norm"), "hidden_norm -> enc_output_norm (f32)");
  Check(norm_ok("norm.weight", "dflash.output_norm"), "norm -> output_norm (f32)");
  bool all_layers = true;
  for (int i = 0; i < kLayers; ++i) {
    const std::string h = "layers." + std::to_string(i) + ".", b = "dflash.layers." + std::to_string(i) + ".";
    for (const char* m : {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj",
                          "mlp.up_proj", "mlp.down_proj"})
      all_layers &= lin_ok(h + m + ".weight", b + m);
    all_layers &= norm_ok(h + "input_layernorm.weight", b + "input_layernorm");
    all_layers &= norm_ok(h + "post_attention_layernorm.weight", b + "post_attention_layernorm");
    all_layers &= norm_ok(h + "self_attn.q_norm.weight", b + "self_attn.q_norm");
    all_layers &= norm_ok(h + "self_attn.k_norm.weight", b + "self_attn.k_norm");
    // identity conv
    for (const char* sub : {"self_attn", "mlp"}) {
      const std::string cb = b + sub + ".conv.base";
      all_layers &= r.Has(cb) && r.Meta(cb).shape == std::vector<int64_t>({2, 2, kHidden, 2});
      const uint16_t* p = reinterpret_cast<const uint16_t*>(r.Data(cb));
      const uint16_t one = r4dx::core::FloatToBf16(1.0f);
      for (int side = 0; side < 2; ++side)
        for (int64_t c = 0; c < kHidden; ++c) {
          all_layers &= p[(side * 2 + 0) * kHidden + c] == one;
          all_layers &= p[(side * 2 + 1) * kHidden + c] == 0;
        }
      const std::string cp = b + sub + ".conv.proj.bf16.w";
      all_layers &= r.Has(cp) && r.Meta(cp).shape == std::vector<int64_t>({2 * 2 * (kHidden / 16), kHidden, 2});
      all_layers &= AllZero(r.Data(cp), r.Meta(cp).end - r.Meta(cp).begin);
    }
  }
  Check(all_layers, "all layer linears / norms byte-exact, identity conv base, zero conv.proj");
  Check(r.Has("dflash.selector.hidden.bf16.w") && AllZero(r.Data("dflash.selector.hidden.bf16.w"), r.Meta("dflash.selector.hidden.bf16.w").end - r.Meta("dflash.selector.hidden.bf16.w").begin),
        "selector.hidden zero");
  for (const char* cb : {"dflash.selector.predecessor", "dflash.selector.successor"}) {
    const bool ok = r.Has(cb) && r.Meta(cb).shape == std::vector<int64_t>({kVocab, 256, 2}) &&
                    AllZero(r.Data(cb), r.Meta(cb).end - r.Meta(cb).begin);
    Check(ok, std::string(cb) + " zero [vocab, 256]");
  }
  {
    // exact tensor set: 1 fc + 2 norms + 3 selector + per layer (7 linears + 4 norms + 2 conv.base + 2 conv.proj)
    const size_t want = 1 + 2 + 3 + static_cast<size_t>(kLayers) * (7 + 4 + 2 + 2);
    Check(r.Names().size() == want, "tensor count == " + std::to_string(want) + " (got " + std::to_string(r.Names().size()) + ")");
  }

  r_owner.reset();  // unmap before the temp tree is removed

  // ---- offset 0 + explicit embed scale ------------------------------------------------------------------
  {
    DflashHfOptions o;
    o.input_dir = (root / "hf").string();
    o.output_path = (root / "off0.r4dx").string();
    o.target_layer_offset = 0;
    o.embed_scale = 62.0;
    o.quiet = true;
    ConvertDflashHf(o);
    const json m0 = ReadMeta(o.output_path).at("dflash2");
    Check(m0.at("target_layers").get<std::vector<int64_t>>() == kIds, "offset 0: target_layers = ids");
    Check(m0.at("embed_scale").get<double>() == 62.0, "explicit embed scale 62");
  }

  // ---- w4a16 layout ---------------------------------------------------------------------------------------
  {
    DflashHfOptions o;
    o.input_dir = (root / "hf").string();
    o.output_path = (root / "w4.r4dx").string();
    o.layout = "w4a16";
    o.quiet = true;
    ConvertDflashHf(o);
    SafetensorsReader w(Utf8ToWide(o.output_path));
    const std::string cp = "dflash.layers.0.self_attn.conv.proj.w4a16.wsz";
    bool ok = w.Has(cp) && w.Has("dflash.fc.w4a16.wq") && w.Has("dflash.selector.hidden.w4a16.wsz");
    if (ok) {
      const uint32_t* wsz = reinterpret_cast<const uint32_t*>(w.Data(cp));
      const size_t n = (w.Meta(cp).end - w.Meta(cp).begin) / 4;
      for (size_t i = 0; i < n; ++i) ok &= (wsz[i] & 0xFFFFu) == 0;  // fp16 scale 0 -> every weight decodes to 0
    }
    Check(ok, "w4a16 layout converts; zero conv.proj has scale 0");
    Check(w.Has("dflash.selector.predecessor") && !w.Has("dflash.fc.bf16.w"), "w4a16: codebooks stay bf16, no bf16 companion");
  }

  // ---- refusals ---------------------------------------------------------------------------------------------
  auto convert_dir = [&](const fs::path& dir, int64_t offset = 1) {
    DflashHfOptions o;
    o.input_dir = dir.string();
    o.output_path = (root / "refused.r4dx").string();
    o.target_layer_offset = offset;
    o.quiet = true;
    ConvertDflashHf(o);
  };
  {
    Ckpt bad = ck;
    bad.erase("layers.1.mlp.up_proj.weight");
    WriteDir(root / "missing", bad, Config());
    Check(Throws([&] { convert_dir(root / "missing"); }, "layers.1.mlp.up_proj.weight"), "refuses a missing tensor by name");
    Ckpt shp = ck;
    shp["layers.0.self_attn.k_proj.weight"].shape = {kHd, kKv};
    shp["layers.0.self_attn.k_proj.weight"].bf16.resize(static_cast<size_t>(kHd * kKv));
    WriteDir(root / "shape", shp, Config());
    Check(Throws([&] { convert_dir(root / "shape"); }, "has shape"), "refuses a wrong shape");
    Ckpt extra = ck;
    extra["layers.0.surprise.weight"] = Tensor{{4}, std::vector<uint16_t>(4)};
    WriteDir(root / "extra", extra, Config());
    Check(Throws([&] { convert_dir(root / "extra"); }, "not consumed"), "refuses an unconsumed tensor");
    Ckpt tied = ck;  // an embedding / head shipped alongside is tolerated, not converted
    tied["embed_tokens.weight"] = Tensor{{4, 4}, std::vector<uint16_t>(16)};
    WriteDir(root / "tied", tied, Config());
    convert_dir(root / "tied");
    Check(true, "tolerates a shipped embed_tokens (the target's is used)");
    json cfg = Config();
    cfg["output_multiplier"] = 2.0;
    WriteDir(root / "mult", ck, cfg);
    Check(Throws([&] { convert_dir(root / "mult"); }, "output_multiplier"), "refuses output_multiplier != 1");
    cfg = Config();
    cfg["input_embedding_scale"] = 62.0;
    WriteDir(root / "scale", ck, cfg);
    DflashHfOptions o;
    o.input_dir = (root / "scale").string();
    o.output_path = (root / "scale.r4dx").string();
    o.quiet = true;
    ConvertDflashHf(o);
    Check(ReadMeta(o.output_path).at("dflash2").at("embed_scale").get<double>() == 62.0, "config input_embedding_scale is carried");
    Check(Throws([&] { convert_dir(root / "hf", 2); }, "offset"), "refuses a layer offset outside {0, 1}");
  }

  fs::remove_all(root);
  std::printf("%d checks, %d failures\n", g_checks, g_fail);
  return g_fail == 0 ? 0 : 1;
} catch (const std::exception& e) {
  std::fprintf(stderr, "EXCEPTION: %s\n", e.what());
  return 1;
}
