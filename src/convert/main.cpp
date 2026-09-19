// r4dx-convert -- HF checkpoint (safetensors + config.json) -> r4dx container
// (docs/container-format.md). See docs/build-windows.md for the toolchain and README.md for the
// project. Usage:
//
//   r4dx-convert --input <HF checkpoint dir> --output <container path>
//                [--layouts mxfp4,w4a16,w4a8] [--lm-head 4bit+bf16]
//                [--layers N] [--threads T] [--vision on|off] [--mtp on|off] [--no-bf16]
//                [--kv-calib <tools/reference/kv_calibrate.py JSON>]
//
// --kv-calib fills text.layers.{i}.attn.k_descale/.v_descale (full-attention layers only) from a
// tools/reference/kv_calibrate.py merged JSON: descale[head] = k_amax|v_amax[head] / 448.0 (see
// r4dx_convert::ResolveKvDescale, kv_calib.hpp). A layer absent from the JSON (or with a
// missing/wrong-length amax array) falls back to descale=1.0 with a WARNING on stderr, not a hard
// error -- omitting --kv-calib entirely also yields the 1.0 placeholder, silently (no calibration
// was ever requested, so there is nothing to warn about).
//
//   r4dx-convert --selftest --selftest-input <small .safetensors, one 2D bf16 tensor "w">
//                --selftest-output <container path> [--layouts mxfp4,w4a16,w4a8] [--threads T]
//
// --selftest packs exactly one tensor through every requested layout and writes it as
// "selftest.{layout}.*" -- tools/convert_ref/selftest_compare.py runs the matching Python
// reference on the same input and diffs the two containers byte for byte.
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "nlohmann/json.hpp"
#include "r4d.h"  // r4d_gemm_{w4a16,w4a8,mxfp4a8}_nt_m64_group() -- cross-checked against this
                  // converter's own kInt4Group/kMxfp4Group at startup (ValidateKernelGroupSizes).
#include "r4dx_convert/container_writer.hpp"
#include "r4dx_convert/kv_calib.hpp"
#include "r4dx_convert/linear_layouts.hpp"
#include "r4dx_convert/safetensors_reader.hpp"
#include "r4dx_convert/sha256.hpp"
#include "r4dx_convert/tensor_codec.hpp"

namespace {

using r4dx_convert::ContainerWriter;
using r4dx_convert::LayoutSet;
using r4dx_convert::ShardedModel;

bool StartsWith(const std::string& s, const std::string& prefix) {
  return s.size() >= prefix.size() && s.compare(0, prefix.size(), prefix) == 0;
}

// Review finding (major, quant_int4.hpp): the group sizes this converter packs with
// (r4dx_convert::kInt4Group, kMxfp4Group) are compile-time constants that must match the group
// sizes the GPU kernels were actually built with (r4d_gemm_w4a8_nt_m64.hip's R4D_GEMM_W4A8_GROUP
// default is 256; third_party/CMakeLists.txt:70 overrides it to 128 with a single -D that has no
// other record anywhere). A silent mismatch here produces a container that GEMMs read at the
// wrong stride with no error, ever. r4d_core exports the group each kernel was actually compiled
// with, so assert against it once at startup instead of trusting the -D stayed in sync.
void ValidateKernelGroupSizes() {
  auto check = [](const char* label, int expected, int actual) {
    if (actual != expected) {
      throw std::runtime_error(
          std::string("r4dx-convert: group size mismatch for ") + label + ": this converter packs "
          "with group=" + std::to_string(expected) + " but r4d_core's kernel was built with group=" +
          std::to_string(actual) + " (check third_party/CMakeLists.txt's R4D_EXTRA_* -D flags "
          "against src/convert/include/r4dx_convert/quant_{int4,mxfp4}.hpp's kInt4Group/"
          "kMxfp4Group)");
    }
  };
  check("w4a16", r4dx_convert::kInt4Group, r4d_gemm_w4a16_nt_m64_group());
  check("w4a8", r4dx_convert::kInt4Group, r4d_gemm_w4a8_nt_m64_group());
  check("mxfp4", r4dx_convert::kMxfp4Group, r4d_gemm_mxfp4a8_nt_m64_group());
}

// Builds the __metadata__["quant"] block (review finding, major): the task brief asked for
// per-tensor quant layout/group/permutation in __metadata__ and the original run only had prose
// in quant_summary. This is the authoritative, loader-checkable record -- src/model should assert
// its own kernel's group() matches these before trusting the container.
nlohmann::json BuildQuantMetadata() {
  return {
      {"w4a16",
       {{"group", r4dx_convert::kInt4Group},
        {"zero_mode", "free_0_15"},
        {"nibble_encoding", "offset_binary_xor8"},
        {"fragment_permutation", "wmma16x16x16_lane16_koff_0_8_1_9_2_10_3_11"}}},
      {"w4a8",
       {{"group", r4dx_convert::kInt4Group},
        {"zero_mode", "pinned_8"},
        {"nibble_encoding", "offset_binary_xor8"},
        {"fragment_permutation", "wmma16x16x16_lane16_koff_0_8_1_9_2_10_3_11"}}},
      {"mxfp4",
       {{"group", r4dx_convert::kMxfp4Group},
        {"scale_encoding", "e8m0"},
        {"fragment_permutation", "mxfp4_layout_permute_w"}}},
  };
}

// Parses a comma/plus-separated list of layout tokens ("mxfp4", "w4a16", "w4a8", "bf16", "4bit"
// meaning all three quantized layouts, and "none"/"" meaning bf16 only). Unknown tokens are a
// hard error -- silently ignoring a typo'd --layouts value would produce a container missing a
// layout the caller thinks it asked for.
LayoutSet ParseLayoutList(const std::string& spec, bool bf16_default_on) {
  LayoutSet ls;
  ls.bf16 = bf16_default_on;
  std::string cur;
  std::vector<std::string> tokens;
  for (char c : spec) {
    if (c == ',' || c == '+') {
      if (!cur.empty()) tokens.push_back(cur);
      cur.clear();
    } else if (!std::isspace(static_cast<unsigned char>(c))) {
      cur.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }
  }
  if (!cur.empty()) tokens.push_back(cur);
  for (auto& t : tokens) {
    if (t == "mxfp4") ls.mxfp4 = true;
    else if (t == "w4a16") ls.w4a16 = true;
    else if (t == "w4a8") ls.w4a8 = true;
    else if (t == "bf16") ls.bf16 = true;
    else if (t == "4bit") { ls.mxfp4 = ls.w4a16 = ls.w4a8 = true; }
    else if (t == "none" || t.empty()) { /* no-op */ }
    else throw std::runtime_error("unknown layout token: " + t);
  }
  return ls;
}

struct AppArgs {
  bool selftest = false;
  std::string input, output;
  std::string selftest_input, selftest_output, selftest_name = "w";
  std::string layouts_spec = "mxfp4,w4a16,w4a8";
  std::string lm_head_spec = "4bit+bf16";
  int layers = -1;  // -1 = every text layer
  int threads = 0;  // 0 = hardware_concurrency
  int vision = -1;  // -1 auto (on iff full run), 0 off, 1 on
  int mtp = -1;
  bool no_bf16 = false;  // drop the bf16 layout for body weights even if --layouts/--lm-head asked
  std::string kv_calib;  // path to tools/reference/kv_calibrate.py's merged JSON; empty = no calib
};

// Strict on/off parser for --vision/--mtp (review finding, minor): the previous `(v == "on") ? 1
// : 0` silently treated any typo ("yes", "true", "1", a fat-fingered value) as OFF, which could
// quietly drop the vision tower or MTP head from a multi-hour conversion run with no diagnostic.
int ParseOnOff(const std::string& v, const char* flag) {
  if (v == "on") return 1;
  if (v == "off") return 0;
  throw std::runtime_error(std::string(flag) + " must be 'on' or 'off', got '" + v + "'");
}

AppArgs ParseArgs(int argc, char** argv) {
  AppArgs a;
  auto next = [&](int& i) -> std::string {
    if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + argv[i]);
    return argv[++i];
  };
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--selftest") a.selftest = true;
    else if (arg == "--input") a.input = next(i);
    else if (arg == "--output") a.output = next(i);
    else if (arg == "--selftest-input") a.selftest_input = next(i);
    else if (arg == "--selftest-output") a.selftest_output = next(i);
    else if (arg == "--selftest-name") a.selftest_name = next(i);
    else if (arg == "--layouts") a.layouts_spec = next(i);
    else if (arg == "--lm-head") a.lm_head_spec = next(i);
    else if (arg == "--layers") a.layers = std::stoi(next(i));
    else if (arg == "--threads") a.threads = std::stoi(next(i));
    else if (arg == "--vision") a.vision = ParseOnOff(next(i), "--vision");
    else if (arg == "--mtp") a.mtp = ParseOnOff(next(i), "--mtp");
    else if (arg == "--no-bf16") a.no_bf16 = true;
    else if (arg == "--kv-calib") a.kv_calib = next(i);
    else throw std::runtime_error("unknown argument: " + arg);
  }
  return a;
}

int ResolveThreads(int requested) {
  if (requested > 0) return requested;
  unsigned hw = std::thread::hardware_concurrency();
  return hw > 0 ? static_cast<int>(hw) : 4;
}

std::string ReadFile(const std::string& path) {
  // Wide open (MSVC ifstream(const wchar_t*) extension), not narrow fopen/ifstream(path) --
  // consistent with SafetensorsReader's CreateFileW so a non-ASCII --input path on a non-UTF-8
  // ANSI codepage resolves to the same file both places (review finding, minor).
  std::ifstream f(r4dx_convert::Utf8ToWide(path).c_str(), std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

// ---- normal mode --------------------------------------------------------------------------

int RunConvert(const AppArgs& args) {
  const int threads = ResolveThreads(args.threads);
  const std::string config_text = ReadFile(args.input + "\\config.json");
  const nlohmann::json config = nlohmann::json::parse(config_text);
  const nlohmann::json& text_cfg = config.at("text_config");

  const int hidden = text_cfg.at("hidden_size").get<int>();
  const int num_layers_total = text_cfg.at("num_hidden_layers").get<int>();
  const int kv_heads = text_cfg.at("num_key_value_heads").get<int>();
  const std::vector<std::string> layer_types =
      text_cfg.at("layer_types").get<std::vector<std::string>>();
  const int layers = args.layers > 0 ? std::min(args.layers, num_layers_total) : num_layers_total;
  const bool full_run = (layers == num_layers_total);
  const bool do_vision = args.vision >= 0 ? (args.vision == 1) : full_run;
  const bool do_mtp = args.mtp >= 0 ? (args.mtp == 1) : full_run;

  LayoutSet layouts = ParseLayoutList(args.layouts_spec, /*bf16_default_on=*/true);
  LayoutSet lm_head_layouts = ParseLayoutList(args.lm_head_spec, /*bf16_default_on=*/false);
  if (args.no_bf16) {
    // --layouts/--lm-head have no way to say "and NOT bf16" (bf16 defaults on for body weights,
    // and "none" in a spec is a no-op rather than a subtractive token) -- --no-bf16 is the
    // explicit override so a quantized-only conversion is possible (review finding, minor).
    layouts.bf16 = false;
    lm_head_layouts.bf16 = false;
  }

  std::cout << "[r4dx-convert] input=" << args.input << " output=" << args.output << "\n"
            << "[r4dx-convert] hidden=" << hidden << " layers=" << layers << "/" << num_layers_total
            << " threads=" << threads << " vision=" << (do_vision ? "on" : "off")
            << " mtp=" << (do_mtp ? "on" : "off") << "\n";

  const bool have_kv_calib = !args.kv_calib.empty();
  nlohmann::json kv_calib_json;
  if (have_kv_calib) {
    kv_calib_json = nlohmann::json::parse(ReadFile(args.kv_calib));
    std::cout << "[r4dx-convert] kv-calib=" << args.kv_calib << " (" << kv_calib_json.size()
              << " layer(s) in file)\n";
  }

  ShardedModel model(args.input);
  ContainerWriter writer;

  std::vector<std::function<void()>> plan_jobs, emit_jobs;

  auto add_bf16 = [&](std::string hf_name, std::string container_name) {
    plan_jobs.push_back([&writer, &model, hf_name, container_name]() {
      const auto& meta = model.Meta(hf_name);
      std::vector<int64_t> shape = meta.shape;
      shape.push_back(2);
      writer.Plan(container_name, shape, static_cast<uint64_t>(meta.ElemCount()) * 2);
    });
    emit_jobs.push_back([&writer, &model, hf_name, container_name]() {
      auto bytes = r4dx_convert::CopyRawBf16(model, hf_name);
      writer.WriteTensor(container_name, bytes.data(), bytes.size());
    });
  };
  auto add_fp32_widen = [&](std::string hf_name, std::string container_name) {
    plan_jobs.push_back([&writer, &model, hf_name, container_name]() {
      const auto& meta = model.Meta(hf_name);
      std::vector<int64_t> shape = meta.shape;
      shape.push_back(4);
      writer.Plan(container_name, shape, static_cast<uint64_t>(meta.ElemCount()) * 4);
    });
    emit_jobs.push_back([&writer, &model, hf_name, container_name]() {
      auto w = r4dx_convert::ReadTensorAsFloat(model, hf_name);
      auto bytes = r4dx_convert::EncodeFp32(w);
      writer.WriteTensor(container_name, bytes.data(), bytes.size());
    });
  };
  // `layer_idx` is the checkpoint's text-layer index (matches kv_calibrate.py's --layer);
  // `calib_applicable` should be false for any layer kv_calibrate.py never covers by construction
  // (the MTP layer has no calibration entry -- it isn't one of the 64 text-layer indices) so that
  // case falls back to 1.0 silently instead of emitting a spurious "no entry for layer N" warning
  // that would actually be about an unrelated text layer sharing the same numeric index.
  auto add_descale = [&](std::string container_name, int n_kv, int layer_idx, const char* kind,
                          bool calib_applicable) {
    plan_jobs.push_back([&writer, container_name, n_kv]() {
      writer.Plan(container_name, {n_kv, 4}, static_cast<uint64_t>(n_kv) * 4);
    });
    emit_jobs.push_back([&writer, container_name, n_kv, layer_idx, kind, calib_applicable,
                          have_kv_calib, &kv_calib_json]() {
      auto result = r4dx_convert::ResolveKvDescale(
          kv_calib_json, have_kv_calib && calib_applicable, layer_idx, n_kv, kind);
      if (!result.warning.empty()) std::cerr << "[r4dx-convert] WARNING: " << result.warning << "\n";
      auto bytes = r4dx_convert::EncodeFp32(result.values);
      writer.WriteTensor(container_name, bytes.data(), bytes.size());
    });
  };
  auto add_linear = [&](std::vector<std::string> hf_names, std::string container_base,
                         LayoutSet ls) {
    plan_jobs.push_back([&writer, &model, hf_names, container_base, ls]() {
      int64_t N = 0, K = 0;
      for (auto& n : hf_names) {
        const auto& m = model.Meta(n);
        N += m.shape[0];
        K = m.shape[1];
      }
      r4dx_convert::PlanLinearLayouts(writer, container_base, static_cast<int>(N),
                                       static_cast<int>(K), ls);
    });
    emit_jobs.push_back([&writer, &model, hf_names, container_base, ls, threads]() {
      std::vector<float> w;
      int64_t K = 0;
      for (auto& n : hf_names) {
        auto part = r4dx_convert::ReadTensorAsFloat(model, n);
        K = model.Meta(n).shape[1];
        w.insert(w.end(), part.begin(), part.end());
      }
      const int64_t N = static_cast<int64_t>(w.size()) / K;
      r4dx_convert::EmitLinearLayouts(writer, container_base, w, static_cast<int>(N),
                                       static_cast<int>(K), ls, threads);
    });
  };

  for (int i = 0; i < layers; ++i) {
    const std::string hf = "model.language_model.layers." + std::to_string(i) + ".";
    const std::string base = "text.layers." + std::to_string(i) + ".";
    add_bf16(hf + "input_layernorm.weight", base + "input_layernorm");
    add_bf16(hf + "post_attention_layernorm.weight", base + "post_attention_layernorm");

    if (layer_types.at(i) == "full_attention") {
      // q_proj is already the fused query+output-gate matrix in this checkpoint (HF's
      // Qwen3_5Attention builds it as Linear(hidden, num_heads*head_dim*2)); attn.qg is a direct
      // copy/quantize of it, no fusion needed at convert time.
      add_linear({hf + "self_attn.q_proj.weight"}, base + "attn.qg", layouts);
      // attn.k/v (R1, docs/r9700.md): join the quantized-linear family -- 1024x5120 x16 layers,
      // 0.336 GB/token, previously forced bf16 regardless of --layout. Same LayoutSet as every
      // other body linear, so a run without --layouts w4a16 (say) simply omits that variant here
      // too, exactly like attn.qg/o already do.
      add_linear({hf + "self_attn.k_proj.weight"}, base + "attn.k", layouts);
      add_linear({hf + "self_attn.v_proj.weight"}, base + "attn.v", layouts);
      add_linear({hf + "self_attn.o_proj.weight"}, base + "attn.o", layouts);
      add_bf16(hf + "self_attn.q_norm.weight", base + "attn.q_norm");
      add_bf16(hf + "self_attn.k_norm.weight", base + "attn.k_norm");
      add_descale(base + "attn.k_descale", kv_heads, i, "k", /*calib_applicable=*/true);
      add_descale(base + "attn.v_descale", kv_heads, i, "v", /*calib_applicable=*/true);
    } else {
      add_linear({hf + "linear_attn.in_proj_qkv.weight"}, base + "gdn.in_proj_qkv", layouts);
      // gdn.in_proj_z (R1, docs/r9700.md): joins the quantized-linear family -- 6144x5120 x48
      // layers, 3.02 GB/token, the single largest bf16-only tensor in the model (more bytes/token
      // than the entire quantized GDN weight set combined). in_proj_a/in_proj_b stay bf16 (too
      // small to matter, feed the decay path).
      add_linear({hf + "linear_attn.in_proj_z.weight"}, base + "gdn.in_proj_z", layouts);
      add_bf16(hf + "linear_attn.in_proj_b.weight", base + "gdn.in_proj_b");
      add_bf16(hf + "linear_attn.in_proj_a.weight", base + "gdn.in_proj_a");
      add_bf16(hf + "linear_attn.conv1d.weight", base + "gdn.conv1d_weight");
      add_fp32_widen(hf + "linear_attn.A_log", base + "gdn.A_log");
      add_fp32_widen(hf + "linear_attn.dt_bias", base + "gdn.dt_bias");
      add_bf16(hf + "linear_attn.norm.weight", base + "gdn.norm_weight");
      add_linear({hf + "linear_attn.out_proj.weight"}, base + "gdn.out_proj", layouts);
    }
    add_linear({hf + "mlp.gate_proj.weight", hf + "mlp.up_proj.weight"}, base + "mlp.gate_up",
                layouts);
    add_linear({hf + "mlp.down_proj.weight"}, base + "mlp.down", layouts);
  }

  add_bf16("model.language_model.embed_tokens.weight", "text.embed_tokens");
  add_bf16("model.language_model.norm.weight", "text.final_norm");
  add_linear({"lm_head.weight"}, "lm_head", lm_head_layouts);

  if (do_vision) {
    for (const auto& name : model.AllNames()) {
      if (StartsWith(name, "model.visual.")) {
        add_bf16(name, "vision." + name.substr(std::string("model.visual.").size()));
      }
    }
  }

  if (do_mtp) {
    // Review finding (major): mtp.layers.0 is a COMPLETE full-attention decoder layer -- verified
    // against the real checkpoint's model.safetensors.index.json/shard header, not assumed --
    // self_attn.{q,k,v,o}_proj (q_proj [12288,5120], fused q+gate exactly like a text full-
    // attention layer), q_norm/k_norm, mlp.{gate,up,down}_proj ([17408,5120]x2,[5120,17408]), both
    // layernorms. It is NOT a small bespoke head, so it gets the identical add_linear/add_bf16
    // routing a text full_attention layer gets (mxfp4/w4a16/w4a8/bf16 on qg/o and the MLP, bf16 on
    // k/v/norms) rather than a blanket bf16-only passthrough -- the whole point of the 4-bit
    // layouts is to serve the M=1..64 band MTP self-speculation runs in.
    const int mtp_layers = text_cfg.value("mtp_num_hidden_layers", 1);
    for (int i = 0; i < mtp_layers; ++i) {
      const std::string hf = "mtp.layers." + std::to_string(i) + ".";
      // The real checkpoint has exactly one MTP layer; docs/container-format.md's "mtp.*" section
      // documents that single-layer case with a bare "mtp." prefix (no index), so collapse to that
      // when there is only one layer and fall back to a text-layer-shaped "mtp.layers.{i}."
      // prefix if a future checkpoint ever has more.
      const std::string base = (mtp_layers == 1) ? "mtp." : ("mtp.layers." + std::to_string(i) + ".");
      add_bf16(hf + "input_layernorm.weight", base + "input_layernorm");
      add_bf16(hf + "post_attention_layernorm.weight", base + "post_attention_layernorm");
      add_linear({hf + "self_attn.q_proj.weight"}, base + "attn.qg", layouts);
      add_bf16(hf + "self_attn.k_proj.weight", base + "attn.k");
      add_bf16(hf + "self_attn.v_proj.weight", base + "attn.v");
      add_linear({hf + "self_attn.o_proj.weight"}, base + "attn.o", layouts);
      add_bf16(hf + "self_attn.q_norm.weight", base + "attn.q_norm");
      add_bf16(hf + "self_attn.k_norm.weight", base + "attn.k_norm");
      // MTP's inner layer is not one of the 64 text-layer indices kv_calibrate.py calibrates --
      // calib_applicable=false so this always falls back to descale=1.0, silently.
      add_descale(base + "attn.k_descale", kv_heads, i, "k", /*calib_applicable=*/false);
      add_descale(base + "attn.v_descale", kv_heads, i, "v", /*calib_applicable=*/false);
      add_linear({hf + "mlp.gate_proj.weight", hf + "mlp.up_proj.weight"}, base + "mlp.gate_up",
                  layouts);
      add_linear({hf + "mlp.down_proj.weight"}, base + "mlp.down", layouts);
    }
    // mtp.fc / mtp.norm / mtp.pre_fc_norm_embedding / mtp.pre_fc_norm_hidden have no per-text-
    // layer analogue (the hidden+embedding concat -> fc projection around the MTP layer, per HF's
    // Qwen3_5MTPLayer) -- bf16 passthrough by exact HF name. docs/container-format.md's "mtp.*"
    // section does not mention these four tensors; that is outside this component's owned paths
    // (src/convert/**, tests/convert/**, tools/convert_ref/**) so it is flagged in open_issues
    // rather than edited here.
    add_bf16("mtp.fc.weight", "mtp.fc");
    add_bf16("mtp.norm.weight", "mtp.norm");
    add_bf16("mtp.pre_fc_norm_embedding.weight", "mtp.pre_fc_norm_embedding");
    add_bf16("mtp.pre_fc_norm_hidden.weight", "mtp.pre_fc_norm_hidden");
  }

  const auto t0 = std::chrono::steady_clock::now();

  for (auto& j : plan_jobs) j();

  nlohmann::json metadata;
  metadata["r4dx_format_version"] = "1";
  metadata["model_id"] = "Qwen/Qwen3.8-27B";
  metadata["config_sha256"] = r4dx_convert::Sha256Hex(config_text);
  metadata["produced_by"] = "r4dx-convert (r4dx dev build)";
  metadata["quant_summary"] = {
      // Review finding (minor): this previously said "text.layers.*.attn.qg|k|v|o": "bf16", which
      // is factually wrong for qg/o (they go through add_linear with the full LayoutSet, same as
      // mlp.gate_up/down) -- only k/v are bf16-only. A loader trusting the old string would pick
      // bf16 for attn.o and silently lose the 4-bit A/B the container exists to enable.
      {"text.layers.*.mlp.gate_up|down", "mxfp4|w4a16|w4a8|bf16 (all four present; pick at load time)"},
      {"text.layers.*.attn.qg|o", "mxfp4|w4a16|w4a8|bf16 (all four present)"},
      // R1 (docs/r9700.md): attn.k/v and gdn.in_proj_z now join the quantized-linear family (were
      // "bf16" unconditionally before this pass) -- old containers built before this change still
      // have the bare, unsuffixed bf16-only tensor; src/model/container.cpp's
      // LoadQuantLinearWithFallback handles both on-disk forms.
      {"text.layers.*.attn.k|v", "mxfp4|w4a16|w4a8|bf16 (all four present; pick at load time)"},
      {"text.layers.*.gdn.in_proj_z", "mxfp4|w4a16|w4a8|bf16 (all four present; pick at load time)"},
      {"text.layers.*.gdn.in_proj_a|b", "bf16 (never quantized -- feeds the decay path)"},
      {"mtp.attn.qg|o", "mxfp4|w4a16|w4a8|bf16 (all four present, when --mtp on)"},
      {"mtp.attn.k|v", "bf16 (when --mtp on)"},
      {"mtp.mlp.gate_up|down", "mxfp4|w4a16|w4a8|bf16 (all four present, when --mtp on)"},
      {"mtp.fc|norm|pre_fc_norm_embedding|pre_fc_norm_hidden", "bf16 (when --mtp on)"},
      {"lm_head", "mxfp4|w4a16|w4a8|bf16 (all four present)"},
  };
  metadata["quant"] = BuildQuantMetadata();
  metadata["model_config"] = config;
  metadata["r4dx_convert_run"] = {
      {"layers_converted", layers},
      {"layers_total", num_layers_total},
      {"vision", do_vision},
      {"mtp", do_mtp},
      {"layouts", args.layouts_spec},
      {"lm_head", args.lm_head_spec},
      {"threads", threads},
      {"kv_calib", have_kv_calib ? args.kv_calib : std::string("none (descale placeholder 1.0)")},
  };
  writer.FinalizeHeader(args.output, metadata);
  std::cout << "[r4dx-convert] planned " << writer.PlannedTensorCount() << " tensors, "
            << writer.PlannedDataBytes() << " data bytes\n";

  for (auto& j : emit_jobs) j();
  writer.Finish();

  const auto t1 = std::chrono::steady_clock::now();
  const double secs = std::chrono::duration<double>(t1 - t0).count();
  std::cout << "[r4dx-convert] wrote " << args.output << " in " << secs << " s\n";
  return 0;
}

// ---- --selftest mode -----------------------------------------------------------------------

int RunSelftest(const AppArgs& args) {
  const int threads = ResolveThreads(args.threads);
  const LayoutSet layouts = ParseLayoutList(args.layouts_spec, /*bf16_default_on=*/true);

  r4dx_convert::SafetensorsReader reader(r4dx_convert::Utf8ToWide(args.selftest_input));
  if (!reader.Has(args.selftest_name))
    throw std::runtime_error("--selftest-input has no tensor named " + args.selftest_name);
  const auto& meta = reader.Meta(args.selftest_name);
  if (meta.shape.size() != 2) throw std::runtime_error("--selftest tensor must be 2D [N,K]");
  const int N = static_cast<int>(meta.shape[0]), K = static_cast<int>(meta.shape[1]);
  std::vector<float> w(static_cast<size_t>(N) * K);
  if (meta.dtype == "BF16") {
    const uint16_t* src = reinterpret_cast<const uint16_t*>(reader.Data(args.selftest_name));
    for (size_t i = 0; i < w.size(); ++i) w[i] = r4dx::core::Bf16ToFloat(src[i]);
  } else if (meta.dtype == "F32") {
    std::memcpy(w.data(), reader.Data(args.selftest_name), w.size() * 4);
  } else {
    throw std::runtime_error("--selftest tensor must be BF16 or F32");
  }

  ContainerWriter writer;
  r4dx_convert::PlanLinearLayouts(writer, "selftest", N, K, layouts);
  nlohmann::json metadata;
  metadata["r4dx_format_version"] = "1";
  metadata["model_id"] = "selftest";
  metadata["produced_by"] = "r4dx-convert --selftest";
  metadata["quant"] = BuildQuantMetadata();
  writer.FinalizeHeader(args.selftest_output, metadata);
  r4dx_convert::EmitLinearLayouts(writer, "selftest", w, N, K, layouts, threads);
  writer.Finish();

  std::cout << "[r4dx-convert --selftest] N=" << N << " K=" << K << " -> " << args.selftest_output
            << "\n";
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    ValidateKernelGroupSizes();
    AppArgs args = ParseArgs(argc, argv);
    if (args.selftest) {
      if (args.selftest_input.empty() || args.selftest_output.empty())
        throw std::runtime_error("--selftest requires --selftest-input and --selftest-output");
      return RunSelftest(args);
    }
    if (args.input.empty() || args.output.empty())
      throw std::runtime_error("--input and --output are required (or pass --selftest)");
    return RunConvert(args);
  } catch (const std::exception& e) {
    std::cerr << "r4dx-convert: error: " << e.what() << "\n";
    return 1;
  }
}
