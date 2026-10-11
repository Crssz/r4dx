// tests/convert/test_convert_qwen4exp.cpp -- CPU-only (ctest `convert_qwen4exp`), plan M1-2.
//
// r4dx-convert on qwen4_exp checkpoints, container read back with SafetensorsReader:
//   * awq-mini (tools/reference/qwen4exp/convert_fixtures.py awq-mini): a miniature checkpoint in the AWQ format
//     (pack-quantized quads written by compressed-tensors, bf16 passthrough tensors, vision / mtp / n-gram tensors
//     to skip). --layouts bf16: every tensor name / dtype / shape against the plan 4.11 item 3 list (built below
//     from the config alone), every bf16 tensor's bytes against the source (the quads against awq_dequant.py's
//     expected bf16, gate rows then up rows for gate_up), A_log / dt_bias widened, descale 1.0 or from --kv-calib,
//     the metadata (model_arch, verbatim model_config, ngram_sidecar with the header's sha256, skipped list, awq
//     record); --layouts w4a16 (lm_head g32); --trellis-from (mix of KB 3 and 4, the reconstruction check reading
//     the quads through the virtual source) and the per-(layer, class) KB refusal; every refusal; --plan-only.
//   * the tiny checkpoint (E:/models/r4dx/tiny-q4exp, real geometry, 4 layers, 16 experts, bf16 source): bf16
//     container with EVERY tensor's bytes against the HF tensors, w4a16 names, the trellis oracles
//     (convert_fixtures.py tiny-trellis, forms mix34 / bad).
// Exits 77 (SKIPPED) when the awq-mini fixture or r4dx-convert is missing; the tiny parts SKIP on their own.
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
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "awq_source.hpp"
#include "nlohmann/json.hpp"
#include "r4dx/core/dtype.hpp"
#include "r4dx/models_root.h"
#include "r4dx_convert/safetensors_reader.hpp"

namespace fs = std::filesystem;
using nlohmann::json;
using r4dx_convert::SafetensorsReader;
using r4dx_convert::ShardedModel;

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

std::string ReadWhole(const fs::path& p) {
  std::ifstream f(p, std::ios::binary);
  if (!f) throw std::runtime_error("cannot read " + p.string());
  return std::string(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}

void WriteWhole(const fs::path& p, const std::string& bytes) {
  std::ofstream f(p, std::ios::binary | std::ios::trunc);
  f.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

fs::path FixturesDir() {
  if (const char* e = std::getenv("R4DX_Q4EXP_FIXTURES")) return e;
  return r4dx::ModelsPath("r4dx/q4exp-fixtures/ConvertQ4");
}

fs::path TinyDir() {
  if (const char* e = std::getenv("R4DX_TINY_Q4EXP")) return e;
  return r4dx::ModelsPath("r4dx/tiny-q4exp");
}

#if defined(R4DX_CONVERT_EXE)

struct TempDir {
  fs::path p;
  explicit TempDir(const std::string& tag) {
    p = fs::temp_directory_path() / ("r4dx_q4exp_cv_" + tag);
    std::error_code ec;
    fs::remove_all(p, ec);
    fs::create_directories(p);
  }
  ~TempDir() {
    std::error_code ec;
    fs::remove_all(p, ec);
  }
};

struct Result {
  int code = -1;
  std::string log;
};

// cmd.exe strips the outermost pair of quotes of a command that starts with one, so wrap it whole.
Result Run(const std::string& args, const fs::path& log) {
  const std::string cmd = "\"\"" + std::string(R4DX_CONVERT_EXE) + "\" " + args + " > \"" + log.u8string() + "\" 2>&1\"";
  Result r;
  r.code = std::system(cmd.c_str());
  r.log = ReadWhole(log);
  return r;
}

std::string Q(const fs::path& p) { return "\"" + p.generic_string() + "\""; }

// ---- the plan: what a converted qwen4_exp checkpoint must contain, from the config alone --------------------------

struct Dims {
  int H, L, V, heads, kv, hd, E, moe, shared, hc, lowrank, lkh, lvh, lkd, lvd, conv, ih, ikv, ihd, ple_dim, ple_conv;
  std::vector<std::string> types;
  std::set<int> ple_layers;
  int64_t W() const { return static_cast<int64_t>(hc) * H; }
};

Dims ReadDims(const json& cfg) {
  const json& t = cfg.at("text_config");
  Dims d;
  auto i = [&](const char* k) { return t.at(k).get<int>(); };
  d.H = i("hidden_size"), d.L = i("num_hidden_layers"), d.V = i("vocab_size"), d.heads = i("num_attention_heads");
  d.kv = i("num_key_value_heads"), d.hd = i("head_dim"), d.E = i("num_experts"), d.moe = i("moe_intermediate_size");
  d.shared = i("shared_expert_intermediate_size"), d.hc = i("hc_count"), d.lowrank = i("hc_lowrank");
  d.lkh = i("linear_num_key_heads"), d.lvh = i("linear_num_value_heads"), d.lkd = i("linear_key_head_dim");
  d.lvd = i("linear_value_head_dim"), d.conv = i("linear_conv_kernel_dim"), d.ih = i("indexer_n_heads");
  d.ikv = i("indexer_kv_heads"), d.ihd = i("indexer_head_dim"), d.ple_dim = i("ple_embed_dim"), d.ple_conv = i("ple_conv_kernel_size");
  d.types = t.at("layer_types").get<std::vector<std::string>>();
  for (const auto& id : t.at("ple_layer_ids")) d.ple_layers.insert(id.get<int>() - 1);
  return d;
}

enum class Kind { kBf16, kFp32, kLinear, kDescale };

struct Tensor {
  std::string base;               // container name (linear: the base, without a layout suffix)
  Kind kind;
  std::vector<std::string> hf;    // source tensors (a linear's parts in row order)
  std::vector<int64_t> shape;     // bf16/fp32: the HF shape; linear: {N, K}; descale: {n_kv}
  int layer = -1;                 // descale: the layer index
  const char* which = "";         // descale: "k" | "v"
};

std::vector<Tensor> Plan(const Dims& d) {
  std::vector<Tensor> out;
  const std::string P = "model.language_model.";
  auto bf = [&](std::string hf, std::string name, std::vector<int64_t> shape) { out.push_back({name, Kind::kBf16, {hf}, shape}); };
  auto lin = [&](std::vector<std::string> hf, std::string base, int64_t N, int64_t K) { out.push_back({base, Kind::kLinear, hf, {N, K}}); };
  bf(P + "embed_tokens.weight", "text.embed_tokens", {d.V, d.H});
  bf(P + "hyper_connection_mixer.hc_norm.weight", "text.hc_final.norm", {d.W()});
  bf(P + "hyper_connection_mixer.input_mix_weight_down.weight", "text.hc_final.down", {d.lowrank, d.W()});
  bf(P + "hyper_connection_mixer.input_mix_weight_up.weight", "text.hc_final.up", {d.W(), d.lowrank});
  lin({"lm_head.weight"}, "lm_head", d.V, d.H);
  for (int i = 0; i < d.L; ++i) {
    const std::string h = P + "layers." + std::to_string(i) + ".", b = "text.layers." + std::to_string(i) + ".";
    for (const char* blk : {"attn", "mlp"}) {
      const std::string hh = h + blk + "_hyper_connection.", bb = b + "hc." + blk + ".";
      bf(hh + "hc_norm.weight", bb + "norm", {d.W()});
      bf(hh + "input_mix_weight_down.weight", bb + "down", {d.lowrank, d.W()});
      bf(hh + "input_mix_weight_up.weight", bb + "up", {d.W(), d.lowrank});
      bf(hh + "block_inject_weight.weight", bb + "inject", {d.hc, d.W()});
    }
    if (d.types[static_cast<size_t>(i)] == "full_attention") {
      const int64_t q = static_cast<int64_t>(d.heads) * d.hd, kvr = static_cast<int64_t>(d.kv) * d.hd;
      lin({h + "self_attn.q_proj.weight"}, b + "attn.qg", 2 * q, d.H);
      lin({h + "self_attn.k_proj.weight"}, b + "attn.k", kvr, d.H);
      lin({h + "self_attn.v_proj.weight"}, b + "attn.v", kvr, d.H);
      lin({h + "self_attn.o_proj.weight"}, b + "attn.o", d.H, q);
      bf(h + "self_attn.q_norm.weight", b + "attn.q_norm", {d.hd});
      bf(h + "self_attn.k_norm.weight", b + "attn.k_norm", {d.hd});
      out.push_back({b + "attn.k_descale", Kind::kDescale, {}, {d.kv}, i, "k"});
      out.push_back({b + "attn.v_descale", Kind::kDescale, {}, {d.kv}, i, "v"});
      bf(h + "self_attn.indexer.index_qk_proj.weight", b + "attn.indexer.qk", {static_cast<int64_t>(d.ih + d.ikv) * d.ihd, d.H});
      bf(h + "self_attn.indexer.q_layernorm.weight", b + "attn.indexer.q_norm", {d.ihd});
      bf(h + "self_attn.indexer.k_layernorm.weight", b + "attn.indexer.k_norm", {d.ihd});
    } else {
      const int64_t qkv = 2LL * d.lkh * d.lkd + static_cast<int64_t>(d.lvh) * d.lvd, z = static_cast<int64_t>(d.lvh) * d.lvd;
      lin({h + "linear_attn.in_proj_qkv.weight"}, b + "gdn.in_proj_qkv", qkv, d.H);
      lin({h + "linear_attn.in_proj_z.weight"}, b + "gdn.in_proj_z", z, d.H);
      lin({h + "linear_attn.out_proj.weight"}, b + "gdn.out_proj", d.H, z);
      bf(h + "linear_attn.in_proj_a.weight", b + "gdn.in_proj_a", {d.lvh, d.H});
      bf(h + "linear_attn.in_proj_b.weight", b + "gdn.in_proj_b", {d.lvh, d.H});
      bf(h + "linear_attn.conv1d.weight", b + "gdn.conv1d_weight", {qkv, 1, d.conv});
      out.push_back({b + "gdn.A_log", Kind::kFp32, {h + "linear_attn.A_log"}, {d.lvh}});
      out.push_back({b + "gdn.dt_bias", Kind::kFp32, {h + "linear_attn.dt_bias"}, {d.lvh}});
      bf(h + "linear_attn.norm.weight", b + "gdn.norm_weight", {d.lvd});
    }
    if (d.ple_layers.count(i)) {
      bf(h + "ple.key_proj.weight", b + "ple.key_proj", {d.W(), d.ple_dim});
      bf(h + "ple.value_proj.weight", b + "ple.value_proj", {d.H, d.ple_dim});
      for (const char* n : {"norm_key", "norm_query", "norm_conv"}) bf(h + "ple." + n + ".weight", b + "ple." + n, {d.W()});
      bf(h + "ple.conv1d.weight", b + "ple.conv1d", {d.W(), 1, d.ple_conv});
    }
    bf(h + "mlp.gate.weight", b + "moe.router", {d.E, d.H});
    out.push_back({b + "moe.shared.gate", Kind::kLinear, {h + "mlp.shared_expert.gate_proj.weight"}, {d.shared, d.H}, -2});
    out.push_back({b + "moe.shared.up", Kind::kLinear, {h + "mlp.shared_expert.up_proj.weight"}, {d.shared, d.H}, -2});
    out.push_back({b + "moe.shared.down", Kind::kLinear, {h + "mlp.shared_expert.down_proj.weight"}, {d.H, d.shared}, -2});
    bf(h + "mlp.shared_expert_gate.weight", b + "moe.shared_gate", {1, d.H});
    for (int e = 0; e < d.E; ++e) {
      const std::string he = h + "mlp.experts." + std::to_string(e) + ".", be = b + "moe.experts." + std::to_string(e) + ".";
      lin({he + "gate_proj.weight", he + "up_proj.weight"}, be + "gate_up", 2LL * d.moe, d.H);
      lin({he + "down_proj.weight"}, be + "down", d.H, d.moe);
    }
  }
  return out;
}

struct LayoutSpec {
  std::string body = "bf16";       // bf16 | w4a16 | trellis
  std::string head = "bf16";       // bf16 | w4a16
  int w4a16_group = 64;
  int head_group = 64;
  std::map<std::string, int> trellis_bits;  // base -> bits (body = trellis)
};

// container tensor name -> header shape (dtype is always U8)
std::map<std::string, std::vector<int64_t>> Expected(const std::vector<Tensor>& plan, const LayoutSpec& ls) {
  std::map<std::string, std::vector<int64_t>> m;
  auto w4 = [&](const std::string& base, int64_t N, int64_t K, int g) {
    m[base + ".w4a16.wq"] = {N * K / 2};
    m[base + ".w4a16.wsz" + (g == ls.w4a16_group ? std::string() : ".g" + std::to_string(g))] = {N * K / g, 4};
  };
  for (const Tensor& t : plan) {
    switch (t.kind) {
      case Kind::kBf16: {
        std::vector<int64_t> s = t.shape;
        s.push_back(2);
        m[t.base] = s;
        break;
      }
      case Kind::kFp32: {
        std::vector<int64_t> s = t.shape;
        s.push_back(4);
        m[t.base] = s;
        break;
      }
      case Kind::kDescale:
        m[t.base] = {t.shape[0], 4};
        break;
      case Kind::kLinear: {
        const int64_t N = t.shape[0], K = t.shape[1];
        if (t.base == "lm_head") {
          if (ls.head == "bf16") m[t.base + ".bf16.w"] = {N, K, 2};
          else w4(t.base, N, K, ls.head_group);
        } else if (t.layer == -2) {  // shared experts: bf16 only
          m[t.base + ".bf16.w"] = {N, K, 2};
        } else if (ls.body == "bf16") {
          m[t.base + ".bf16.w"] = {N, K, 2};
        } else if (ls.body == "w4a16") {
          w4(t.base, N, K, ls.w4a16_group);
        } else {
          const int bits = ls.trellis_bits.at(t.base);
          const int64_t P = t.hf.size();
          m[t.base + ".trellis.w"] = {N * K * bits / 8};
          m[t.base + ".trellis.suh"] = {P * K, 2};
          m[t.base + ".trellis.svh"] = {N, 2};
        }
        break;
      }
    }
  }
  return m;
}

json HeaderMetadata(const fs::path& container) {
  SafetensorsReader r(r4dx_convert::Utf8ToWide(container.string()));
  return json::parse(r.HeaderJson()).at("__metadata__");
}

void CheckNamesAndShapes(const fs::path& container, const std::map<std::string, std::vector<int64_t>>& want, const std::string& label) {
  SafetensorsReader r(r4dx_convert::Utf8ToWide(container.string()));
  std::vector<std::string> names = r.Names();
  std::set<std::string> have(names.begin(), names.end());
  std::set<std::string> expect;
  for (const auto& kv : want) expect.insert(kv.first);
  size_t missing = 0, extra = 0, bad_shape = 0, bad_dtype = 0;
  for (const auto& kv : want) {
    if (!have.count(kv.first)) {
      if (missing++ < 5) std::fprintf(stderr, "  missing: %s\n", kv.first.c_str());
      continue;
    }
    const auto& meta = r.Meta(kv.first);
    if (meta.dtype != "U8") ++bad_dtype;
    if (meta.shape != kv.second) {
      if (bad_shape++ < 5) std::fprintf(stderr, "  shape of %s: %s, expected %s\n", kv.first.c_str(), json(meta.shape).dump().c_str(), json(kv.second).dump().c_str());
    }
  }
  for (const auto& n : have)
    if (!expect.count(n) && extra++ < 5) std::fprintf(stderr, "  unexpected: %s\n", n.c_str());
  Check(missing == 0 && extra == 0 && bad_shape == 0 && bad_dtype == 0,
        label + ": " + std::to_string(have.size()) + " tensors, names / U8 dtype / shapes exactly the plan's (" + std::to_string(missing) + " missing, " +
            std::to_string(extra) + " extra, " + std::to_string(bad_shape) + " wrong shape)");
}

// bytes of the source tensors, through the converter's own source reader (checkpoint dir) -- or an override
using Source = std::function<std::vector<uint8_t>(const std::string&)>;

Source ModelSource(ShardedModel& model) {
  return [&model](const std::string& name) {
    const auto& m = model.Meta(name);
    const uint8_t* d = model.Data(name);
    const size_t n = static_cast<size_t>(m.ElemCount()) * static_cast<size_t>(r4dx_convert::SafetensorsDtypeSize(m.dtype));
    return std::vector<uint8_t>(d, d + n);
  };
}

void CheckBf16Contents(const fs::path& container, const std::vector<Tensor>& plan, const Source& src, const LayoutSpec& ls, const std::string& label) {
  SafetensorsReader r(r4dx_convert::Utf8ToWide(container.string()));
  size_t checked = 0, bad = 0;
  uint64_t bytes = 0;
  auto same = [&](const std::string& name, const std::vector<uint8_t>& want) {
    ++checked;
    bytes += want.size();
    const auto& m = r.Meta(name);
    if (m.end - m.begin != want.size() || std::memcmp(r.Data(name), want.data(), want.size()) != 0) {
      if (bad++ < 5) std::fprintf(stderr, "  contents differ: %s\n", name.c_str());
    }
  };
  for (const Tensor& t : plan) {
    if (t.kind == Kind::kBf16) {
      same(t.base, src(t.hf[0]));
    } else if (t.kind == Kind::kFp32) {
      const std::vector<uint8_t> b = src(t.hf[0]);
      std::vector<uint8_t> f(b.size() * 2);
      for (size_t i = 0; i < b.size() / 2; ++i) {
        uint16_t h;
        std::memcpy(&h, b.data() + i * 2, 2);
        const float v = r4dx::core::Bf16ToFloat(h);
        std::memcpy(f.data() + i * 4, &v, 4);
      }
      same(t.base, f);
    } else if (t.kind == Kind::kLinear) {
      const bool bf16 = t.base == "lm_head" ? ls.head == "bf16" : (t.layer == -2 || ls.body == "bf16");
      if (!bf16) continue;
      std::vector<uint8_t> cat;
      for (const std::string& h : t.hf) {
        const std::vector<uint8_t> b = src(h);
        cat.insert(cat.end(), b.begin(), b.end());
      }
      same(t.base + ".bf16.w", cat);
    }
  }
  Check(checked > 0 && bad == 0, label + ": " + std::to_string(checked) + " tensors (" + std::to_string(bytes) + " bytes) byte-identical to the source" +
                                      " (bf16 passthrough, gate rows then up rows, fp32 widening, bf16 linears)");
}

// ---- cases --------------------------------------------------------------------------------------------------------

struct Fixture {
  fs::path hf, sidecar;
  json cfg;
  std::vector<Tensor> plan;
};

void CheckMetadata(const fs::path& container, const Fixture& fx, const std::string& label) {
  const json md = HeaderMetadata(container);
  const json sc = md.at("ngram_sidecar");
  const std::string raw = ReadWhole(fx.sidecar).substr(0, 4096);
  std::string hex;
  for (int i = 0; i < 32; ++i) {
    char b[3];
    std::snprintf(b, sizeof b, "%02x", static_cast<uint8_t>(raw[344 + static_cast<size_t>(i)]));
    hex += b;
  }
  uint64_t rows = 0;
  uint32_t row_bytes = 0;
  std::memcpy(&rows, raw.data() + 12, 8);
  std::memcpy(&row_bytes, raw.data() + 20, 4);
  Check(md.at("model_arch") == "qwen4_exp" && md.at("model_config") == fx.cfg && md.at("model_id") == fx.hf.filename().string(),
        label + ": metadata model_arch qwen4_exp, model_config verbatim, model_id");
  Check(fs::path(sc.at("path").get<std::string>()).is_absolute() && fs::equivalent(sc.at("path").get<std::string>(), fx.sidecar) && sc.at("sha256") == hex &&
            sc.at("rows") == rows && sc.at("row_bytes") == row_bytes,
        label + ": metadata ngram_sidecar = {absolute path, the header's sha256 (" + hex.substr(0, 12) + "...), rows " + std::to_string(rows) + ", row_bytes " +
            std::to_string(row_bytes) + "}");
  const json& run = md.at("r4dx_convert_run");
  bool skipped = run.at("skipped").is_array() && run.at("skipped").size() == 3 && run.at("arch") == "qwen4_exp";
  Check(skipped && md.at("config_sha256").is_string(), label + ": run record: arch, the skipped list (vision, mtp, n-gram tensors)");
}

void TestMini(const fs::path& exe_log_dir) {
  std::printf("---- awq-mini ----\n");
  const fs::path root = FixturesDir() / "awq-mini";
  Fixture fx;
  fx.hf = root / "hf";
  fx.sidecar = root / "sidecar.r4dx-ngram";
  fx.cfg = json::parse(ReadWhole(fx.hf / "config.json"));
  const Dims d = ReadDims(fx.cfg);
  fx.plan = Plan(d);
  TempDir t("mini");
  const std::string common = "--input " + Q(fx.hf) + " --ngram-sidecar " + Q(fx.sidecar);

  // the converter's own source reader: the quads through the overlay, and awq_dequant.py's expected bf16
  ShardedModel model(fx.hf.string());
  r4dx_convert::awq::InstallAwqSource(model, fx.cfg);
  SafetensorsReader expected(r4dx_convert::Utf8ToWide((root / "expected_bf16.safetensors").string()));
  Source src = [&](const std::string& name) {
    if (expected.Has(name)) {
      const auto& m = expected.Meta(name);
      return std::vector<uint8_t>(expected.Data(name), expected.Data(name) + (m.end - m.begin));
    }
    return ModelSource(model)(name);
  };

  // --- bf16
  {
    const fs::path out = t.p / "bf16.r4dx";
    const Result r = Run(common + " --output " + Q(out) + " --layouts bf16 --lm-head bf16", t.p / "bf16.log");
    Check(r.code == 0, "mini --layouts bf16 converts (exit " + std::to_string(r.code) + ")");
    if (r.code != 0) {
      std::fprintf(stderr, "%s\n", r.log.c_str());
      return;
    }
    LayoutSpec ls;
    CheckNamesAndShapes(out, Expected(fx.plan, ls), "mini bf16");
    CheckBf16Contents(out, fx.plan, src, ls, "mini bf16");
    CheckMetadata(out, fx, "mini bf16");
    // descale placeholders 1.0 of the right length
    SafetensorsReader c(r4dx_convert::Utf8ToWide(out.string()));
    float k0 = 0;
    std::memcpy(&k0, c.Data("text.layers.1.attn.k_descale"), 4);
    Check(k0 == 1.0f && c.Meta("text.layers.1.attn.k_descale").end - c.Meta("text.layers.1.attn.k_descale").begin == 4,
          "mini bf16: k_descale = [1.0] x kv_heads without --kv-calib");
    const json md = HeaderMetadata(out);
    Check(md.at("r4dx_convert_run").at("awq_source").at("linears") == 19 && md.at("quant_summary").is_object(),
          "mini bf16: run record says 19 pack-quantized linears; quant_summary is the qwen4_exp one");

    // --plan-only: the header byte count it reports is the real file's
    const Result p = Run(common + " --output " + Q(t.p / "never.r4dx") + " --layouts bf16 --lm-head bf16 --plan-only", t.p / "plan.log");
    const size_t at = p.log.find("header JSON ");
    uint64_t reported = 0;
    if (at != std::string::npos) reported = std::strtoull(p.log.c_str() + at + 12, nullptr, 10);
    std::ifstream f(out, std::ios::binary);
    uint64_t actual = 0;
    f.read(reinterpret_cast<char*>(&actual), 8);
    Check(p.code == 0 && reported == actual && reported > 0 && !fs::exists(t.p / "never.r4dx"),
          "mini --plan-only: exits 0, writes nothing, header JSON " + std::to_string(reported) + " bytes == the real container's " + std::to_string(actual));
  }

  // --- kv-calib on the QSA layer only
  {
    const fs::path calib = t.p / "calib.json";
    WriteWhole(calib, json{{"1", {{"k_amax", {89.6}}, {"v_amax", {44.8}}}}}.dump());
    const fs::path out = t.p / "calib.r4dx";
    const Result r = Run(common + " --output " + Q(out) + " --layouts bf16 --lm-head bf16 --kv-calib " + Q(calib), t.p / "calib.log");
    Check(r.code == 0, "mini --kv-calib converts");
    if (r.code == 0) {
      SafetensorsReader c(r4dx_convert::Utf8ToWide(out.string()));
      float k = 0, v = 0;
      std::memcpy(&k, c.Data("text.layers.1.attn.k_descale"), 4);
      std::memcpy(&v, c.Data("text.layers.1.attn.v_descale"), 4);
      Check(std::fabs(k - 89.6f / 448.0f) < 1e-7f && std::fabs(v - 44.8f / 448.0f) < 1e-7f, "mini --kv-calib: descale = amax / 448 for the QSA layer (layer index 1)");
    }
  }

  // --- w4a16 (lm_head g32 through the recipe flags)
  {
    const fs::path out = t.p / "w4a16.r4dx";
    const Result r = Run(common + " --output " + Q(out) + " --layouts w4a16 --no-bf16 --lm-head w4a16 --w4a16-group-rule \"^lm_head$=32\"", t.p / "w4a16.log");
    Check(r.code == 0, "mini --layouts w4a16 --lm-head w4a16 (lm_head g32) converts");
    if (r.code == 0) {
      const json md = HeaderMetadata(out);
      LayoutSpec ls;
      ls.body = "w4a16";
      ls.head = "w4a16";
      ls.w4a16_group = md.at("quant").at("w4a16").at("group").get<int>();
      ls.head_group = 32;
      CheckNamesAndShapes(out, Expected(fx.plan, ls), "mini w4a16");
      LayoutSpec raw_only = ls;
      CheckBf16Contents(out, fx.plan, src, raw_only, "mini w4a16 (the bf16 passthrough and shared experts)");
      Check(md.at("quant").at("w4a16").at("groups").at("lm_head") == 32, "mini w4a16: metadata quant.w4a16.groups lm_head = 32");
    }
  }

  // --- trellis: mix of KB 3 (experts) and 4 (the rest); the reconstruction check reads the quads through the overlay
  {
    const fs::path oracle = FixturesDir() / "awq-mini-trellis-mix34";
    if (!fs::exists(oracle / "weights_override.json")) {
      std::printf("SKIP: %s (convert_fixtures.py awq-mini-trellis --form mix34)\n", oracle.string().c_str());
    } else {
      const fs::path out = t.p / "trellis.r4dx";
      const Result r = Run(common + " --output " + Q(out) + " --layouts w4a16 --lm-head bf16 --trellis-from " + Q(oracle), t.p / "trellis.log");
      Check(r.code == 0, "mini --trellis-from (KB 3 experts / KB 4 rest) converts, the check reads the AWQ quads (exit " + std::to_string(r.code) + ")");
      if (r.code != 0) std::fprintf(stderr, "%s\n", r.log.c_str());
      if (r.code == 0) {
        const json md = HeaderMetadata(out);
        const json& lin = md.at("quant").at("trellis").at("linears");
        LayoutSpec ls;
        ls.body = "trellis";
        for (auto it = lin.begin(); it != lin.end(); ++it) ls.trellis_bits[it.key()] = it.value().at("bits").get<int>();
        CheckNamesAndShapes(out, Expected(fx.plan, ls), "mini trellis");
        CheckBf16Contents(out, fx.plan, src, ls, "mini trellis (bf16 passthrough, shared experts, lm_head)");
        bool experts_k3 = true, rest_k4 = true, parts_ok = true;
        size_t n_experts = 0;
        for (auto it = lin.begin(); it != lin.end(); ++it) {
          const bool expert = it.key().find(".moe.experts.") != std::string::npos;
          n_experts += expert;
          (expert ? experts_k3 : rest_k4) = (expert ? experts_k3 : rest_k4) && it.value().at("bits") == (expert ? 3 : 4);
          if (it.key().size() > 8 && it.key().substr(it.key().size() - 8) == ".gate_up")
            parts_ok = parts_ok && it.value().at("parts") == json::array({d.moe, d.moe});
        }
        Check(experts_k3 && rest_k4 && n_experts == static_cast<size_t>(d.L * d.E * 2) && parts_ok,
              "mini trellis: quant.trellis.linears has every expert gate_up ([moe, moe] parts) and down at KB 3 and the rest at KB 4");
        const std::string verdict = md.at("r4dx_convert_run").at("trellis").at("verify").at("result").get<std::string>();
        Check(verdict.rfind("pass 19/19", 0) == 0, "mini trellis: reconstruction check '" + verdict.substr(0, 40) + "'");
        CheckMetadata(out, fx, "mini trellis");
      }
    }
    const fs::path bad = FixturesDir() / "awq-mini-trellis-bad";
    if (fs::exists(bad / "weights_override.json")) {
      const Result r = Run(common + " --output " + Q(t.p / "bad.r4dx") + " --layouts w4a16 --lm-head bf16 --trellis-from " + Q(bad), t.p / "bad.log");
      Check(r.code != 0 && r.log.find("disagree on the trellis KB") != std::string::npos && r.log.find("text.layers.1.moe.experts.1.gate_up") != std::string::npos &&
                r.log.find("text.layers.1.moe.experts.1.down") != std::string::npos && !fs::exists(t.p / "bad.r4dx") && !fs::exists(t.p / "bad.r4dx.partial"),
            "mini: experts of one layer and class at different KB are refused, naming them, before any output exists");
    }
  }

  // --- refusals
  {
    auto refuse = [&](const std::string& args, const std::string& want, const std::string& what, const fs::path& input = fs::path()) {
      const std::string in = Q(input.empty() ? fx.hf : input);
      const Result r = Run("--input " + in + " --output " + Q(t.p / "x.r4dx") + " --layouts bf16 --lm-head bf16 " + args, t.p / "refuse.log");
      const bool ok = r.code != 0 && r.log.find(want) != std::string::npos && !fs::exists(t.p / "x.r4dx");
      Check(ok, "refused: " + what);
      if (!ok) std::fprintf(stderr, "  wanted '%s', exit %d, log:\n%s\n", want.c_str(), r.code, r.log.c_str());
    };
    const std::string sc = "--ngram-sidecar " + Q(fx.sidecar);
    refuse("", "needs --ngram-sidecar", "no --ngram-sidecar");
    refuse(sc + " --rotate q2a", "not supported for qwen4_exp", "--rotate");
    refuse(sc + " --vision on", "converts the text model only", "--vision on");
    refuse(sc + " --mtp on", "converts the text model only", "--mtp on");
    refuse(sc + " --draft-vocab-ids x.json", "needs the MTP head", "--draft-vocab-ids");
    refuse(sc + " --record-reuse-guard", "not supported for qwen4_exp", "--record-reuse-guard");
    refuse("--ngram-sidecar " + Q(t.p / "nope"), "cannot stat", "a missing sidecar file");
    {
      std::string raw = ReadWhole(fx.sidecar);
      raw.resize(raw.size() - 32);
      WriteWhole(t.p / "trunc.r4dx-ngram", raw);
      refuse("--ngram-sidecar " + Q(t.p / "trunc.r4dx-ngram"), "truncated or still being written", "a truncated sidecar");
    }
    // a checkpoint copy with a surprise tensor / a missing tensor / another quantization format
    const fs::path copy = t.p / "hf";
    fs::create_directories(copy);
    for (const auto& e : fs::directory_iterator(fx.hf)) fs::copy_file(e.path(), copy / e.path().filename(), fs::copy_options::overwrite_existing);
    json index = json::parse(ReadWhole(copy / "model.safetensors.index.json"));
    json idx2 = index;
    idx2["weight_map"]["model.language_model.layers.1.mlp.experts.9.gate_proj.weight"] = "model-00002-of-00002.safetensors";
    idx2["weight_map"]["model.language_model.surprise"] = "model-00001-of-00002.safetensors";
    WriteWhole(copy / "model.safetensors.index.json", idx2.dump());
    refuse(sc, "2 checkpoint tensor(s) are not converted and not skipped on purpose", "an expert beyond num_experts and an unknown tensor (the coverage audit)", copy);
    {
      const Result r = Run("--input " + Q(copy) + " --output " + Q(t.p / "x.r4dx") + " --layouts bf16 --lm-head bf16 " + sc, t.p / "audit.log");
      Check(r.log.find("model.language_model.surprise") != std::string::npos && r.log.find("experts.9.gate_proj.weight") != std::string::npos,
            "the audit message names both");
    }
    idx2 = index;
    idx2["weight_map"].erase("model.language_model.layers.0.mlp.gate.weight");
    WriteWhole(copy / "model.safetensors.index.json", idx2.dump());
    refuse(sc, "checkpoint has no tensor 'model.language_model.layers.0.mlp.gate.weight'", "a missing router tensor", copy);
    WriteWhole(copy / "model.safetensors.index.json", index.dump());
    json cfg2 = fx.cfg;
    cfg2["quantization_config"]["format"] = "float-quantized";
    WriteWhole(copy / "config.json", cfg2.dump());
    refuse(sc, "quantization_config.format is \"float-quantized\"", "another quantization_config format", copy);
    cfg2 = fx.cfg;
    cfg2["quantization_config"]["config_groups"]["group_0"]["weights"]["group_size"] = 32;
    WriteWhole(copy / "config.json", cfg2.dump());
    refuse(sc, "weights.group_size is 32", "an unsupported pack-quantized group size", copy);
    cfg2 = fx.cfg;
    cfg2["text_config"]["tie_word_embeddings"] = true;
    WriteWhole(copy / "config.json", cfg2.dump());
    refuse(sc, "tie_word_embeddings is true", "a tied head", copy);

    // flags of another arch
    const fs::path other = t.p / "other";
    fs::create_directories(other);
    WriteWhole(other / "config.json", json{{"model_type", "qwen3_5"}, {"text_config", {{"hidden_size", 64}, {"num_hidden_layers", 1}, {"num_key_value_heads", 1}, {"layer_types", {"full_attention"}}}}}.dump());
    refuse(sc, "--ngram-sidecar applies to qwen4_exp checkpoints only", "--ngram-sidecar on another arch", other);
    refuse("--plan-only", "--plan-only applies to qwen4_exp checkpoints only", "--plan-only on another arch", other);
  }
  (void)exe_log_dir;
}

void TestTiny() {
  std::printf("---- tiny-q4exp (real geometry, 4 layers, 16 experts) ----\n");
  const fs::path root = TinyDir();
  Fixture fx;
  fx.hf = root / "hf";
  fx.sidecar = root / "sidecar.r4dx-ngram";
  if (!fs::exists(fx.hf / "model.safetensors.index.json") || !fs::exists(fx.sidecar)) {
    std::printf("SKIP: %s not built (tools/reference/qwen4exp/tiny.py)\n", root.string().c_str());
    return;
  }
  fx.cfg = json::parse(ReadWhole(fx.hf / "config.json"));
  const Dims d = ReadDims(fx.cfg);
  fx.plan = Plan(d);
  TempDir t("tiny");
  const std::string common = "--input " + Q(fx.hf) + " --ngram-sidecar " + Q(fx.sidecar);
  ShardedModel model(fx.hf.string());
  const Source src = ModelSource(model);
  {
    const fs::path out = t.p / "bf16.r4dx";
    const Result r = Run(common + " --output " + Q(out) + " --layouts bf16 --lm-head bf16", t.p / "bf16.log");
    Check(r.code == 0, "tiny --layouts bf16 converts (exit " + std::to_string(r.code) + ")");
    if (r.code == 0) {
      LayoutSpec ls;
      CheckNamesAndShapes(out, Expected(fx.plan, ls), "tiny bf16");
      CheckBf16Contents(out, fx.plan, src, ls, "tiny bf16");
      CheckMetadata(out, fx, "tiny bf16");
      Check(!HeaderMetadata(out).contains("layers_present"), "tiny bf16: a full container carries no layers_present key");
    } else {
      std::fprintf(stderr, "%s\n", r.log.c_str());
    }
  }
  {
    // --layers 2: a PREFIX container (docs/qwen4exp.md): globals + layers 0-1, __metadata__.layers_present 2, the config still 4 layers.
    const fs::path out = t.p / "prefix2.r4dx";
    const Result r = Run(common + " --output " + Q(out) + " --layers 2 --layouts bf16 --lm-head bf16", t.p / "prefix2.log");
    Check(r.code == 0, "tiny --layers 2 converts (exit " + std::to_string(r.code) + ")");
    if (r.code == 0) {
      const json md = HeaderMetadata(out);
      Check(md.value("layers_present", -1) == 2 && md.at("r4dx_convert_run").at("layers_converted") == 2 &&
                md.at("model_config").at("text_config").at("num_hidden_layers") == 4,
            "tiny --layers 2: layers_present 2, layers_converted 2, the recorded config still says 4 layers");
      SafetensorsReader rd(r4dx_convert::Utf8ToWide(out.string()));
      bool l1 = false, l2 = false, globals = rd.Has("text.embed_tokens") && rd.Has("text.hc_final.norm") && rd.Has("lm_head.bf16.w");
      for (const std::string& n : rd.Names()) {
        l1 |= n.rfind("text.layers.1.", 0) == 0;
        l2 |= n.rfind("text.layers.2.", 0) == 0 || n.rfind("text.layers.3.", 0) == 0;
      }
      Check(l1 && !l2 && globals, "tiny --layers 2: layer 1 and the globals are present, layers 2 and 3 are not");
    }
  }
  {
    const fs::path out = t.p / "w4a16.r4dx";
    const Result r = Run(common + " --output " + Q(out) + " --layouts w4a16 --no-bf16 --lm-head w4a16 --w4a16-group-rule \"^lm_head$=32\"", t.p / "w4a16.log");
    Check(r.code == 0, "tiny --layouts w4a16 converts");
    if (r.code == 0) {
      LayoutSpec ls;
      ls.body = "w4a16";
      ls.head = "w4a16";
      ls.w4a16_group = HeaderMetadata(out).at("quant").at("w4a16").at("group").get<int>();
      ls.head_group = 32;
      CheckNamesAndShapes(out, Expected(fx.plan, ls), "tiny w4a16");
      CheckBf16Contents(out, fx.plan, src, ls, "tiny w4a16 (bf16 passthrough and shared experts)");
    }
  }
  const fs::path mix = FixturesDir() / "tiny-trellis-mix34";
  if (!fs::exists(mix / "weights_override.json")) {
    std::printf("SKIP: %s (convert_fixtures.py tiny-trellis --form mix34)\n", mix.string().c_str());
  } else {
    const fs::path out = t.p / "trellis.r4dx";
    const Result r = Run(common + " --output " + Q(out) + " --layouts w4a16 --lm-head bf16 --trellis-from " + Q(mix), t.p / "trellis.log");
    Check(r.code == 0, "tiny --trellis-from (random-word oracle: KB 3 experts of layers 0-1, KB 4 the rest) converts and verifies (exit " + std::to_string(r.code) + ")");
    if (r.code != 0) std::fprintf(stderr, "%s\n", r.log.substr(r.log.size() > 3000 ? r.log.size() - 3000 : 0).c_str());
    if (r.code == 0) {
      const json md = HeaderMetadata(out);
      const json& lin = md.at("quant").at("trellis").at("linears");
      LayoutSpec ls;
      ls.body = "trellis";
      for (auto it = lin.begin(); it != lin.end(); ++it) ls.trellis_bits[it.key()] = it.value().at("bits").get<int>();
      CheckNamesAndShapes(out, Expected(fx.plan, ls), "tiny trellis");
      CheckBf16Contents(out, fx.plan, src, ls, "tiny trellis (bf16 passthrough, shared experts, lm_head)");
      bool uniform = true;
      for (int l = 0; l < d.L; ++l)
        for (const char* cls : {"gate_up", "down"}) {
          int kb = -1;
          for (int e = 0; e < d.E; ++e) {
            const int b = lin.at("text.layers." + std::to_string(l) + ".moe.experts." + std::to_string(e) + "." + cls).at("bits").get<int>();
            if (kb < 0) kb = b;
            uniform = uniform && b == kb && b == (l < 2 ? 3 : 4);
          }
        }
      Check(uniform,
            "tiny trellis: one KB per (layer, expert class) in quant.trellis.linears (3 for layers 0-1, 4 for 2-3)");
      const std::string verdict = md.at("r4dx_convert_run").at("trellis").at("verify").at("result").get<std::string>();
      Check(verdict.rfind("pass ", 0) == 0, "tiny trellis: reconstruction check '" + verdict.substr(0, 48) + "'");
      CheckMetadata(out, fx, "tiny trellis");
    }
  }
  const fs::path bad = FixturesDir() / "tiny-trellis-bad";
  if (fs::exists(bad / "weights_override.json")) {
    const Result r = Run(common + " --output " + Q(t.p / "bad.r4dx") + " --layouts w4a16 --lm-head bf16 --trellis-from " + Q(bad), t.p / "bad.log");
    Check(r.code != 0 && r.log.find("layer 2 gate_up:") != std::string::npos && r.log.find("text.layers.2.moe.experts.3.gate_up") != std::string::npos &&
              r.log.find("KB 5") != std::string::npos,
          "tiny: expert 3 of layer 2 at KB 5 among KB 4 experts is refused, naming layer, class and base");
  }
}

#endif  // R4DX_CONVERT_EXE

}  // namespace

int main() {
  try {
#if defined(R4DX_CONVERT_EXE)
    if (!fs::exists(FixturesDir() / "awq-mini" / "expected_bf16.safetensors")) {
      std::printf("SKIP: %s not generated (python tools/reference/qwen4exp/convert_fixtures.py awq-mini)\n", (FixturesDir() / "awq-mini").string().c_str());
      return 77;
    }
    TestMini(fs::temp_directory_path());
    TestTiny();
#else
    std::printf("SKIP: r4dx-convert is not built\n");
    return 77;
#endif
  } catch (const std::exception& e) {
    std::fprintf(stderr, "FAIL: exception: %s\n", e.what());
    return 1;
  }
  std::printf("%s (%d failure(s))\n", g_failures == 0 ? "ALL PASSED" : "FAILED", g_failures);
  return g_failures == 0 ? 0 : 1;
}
