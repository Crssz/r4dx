// r4dx-convert -- HF checkpoint (safetensors + config.json) -> r4dx container
// (docs/container-format.md). See docs/build-windows.md for the toolchain and README.md for the
// project. Usage:
//
//   r4dx-convert --input <HF checkpoint dir> --output <container path>
//                [--layouts mxfp4,w4a16,w4a8] [--lm-head 4bit+bf16]
//                [--layers N] [--threads T] [--vision on|off] [--mtp on|off] [--no-bf16]
//                [--kv-calib <tools/reference/kv_calibrate.py JSON>]
//                [--draft-vocab-ids <tests/model/tool_vocab_calib.cpp output JSON>]
//                [--quant {rtn,search}] [--imatrix <tools/reference/imatrix_capture.py .npz>]
//                [--keep-bf16 <ECMAScript regex over container base names>]
//
// --keep-bf16 (docs/validation.md "Milestone 11 / sensitivity", keep_bf16.hpp) writes every linear
// whose container base name the regex matches as `<base>.bf16.w` ONLY, skipping every quantized
// layout the run asked for; src/model/container.cpp's LoadQuantLinearWithFallback then loads those
// -- and only those -- as bf16. It is the per-tensor-class sensitivity instrument: convert one
// class in bf16, re-run the KL harness, and the KL that disappears is that class's share.
// e.g. --keep-bf16 "attn\.o$", --keep-bf16 "^text\.layers\.[0-7]\.". Matching is regex_SEARCH, so
// anchor when a substring would over-select. A regex that matches nothing warns and converts
// normally (a scripted sweep must not die mid-batch on a class this checkpoint has none of).
//
// --quant selects HOW the 4-bit values are chosen; it does NOT change a single byte of the on-disk
// layout (docs/container-format.md, "How the quantized values are chosen"). `rtn` (the DEFAULT) is
// the historical min/max grid + round-to-nearest and reproduces any pre-existing container byte for
// byte; `search` minimizes the squared reconstruction error per (row, 128-K group) over a small
// candidate grid (src/convert/include/r4dx_convert/quant_search.hpp). --imatrix additionally weights
// that error by each input channel's mean activation energy, from the .npz
// tools/reference/imatrix_capture.py writes (keyed by these same container base names); it requires
// --quant search, and a linear with no entry in the file falls back to unweighted MSE with a
// warning plus an end-of-run coverage line.
//
// `rtn` stays the default deliberately: `--quant search` WITHOUT --imatrix was measured on the real
// checkpoint at mean KL 0.0713 / top-1 87.71% against rtn's 0.0724 / 88.4% -- i.e. no better, and
// marginally worse on top-1 -- while costing ~2x the conversion wall time, and --imatrix cannot be
// a default because it needs a capture file. The pair that IS worth using is
// `--quant search --imatrix <npz>` (w4a16 0.0534 / 89.30%, docs/validation.md "Milestone 10"), and
// it has to be asked for explicitly so that a plain `r4dx-convert --input ... --output ...` keeps
// reproducing every container built before these flags existed.
//
// --draft-vocab-ids (docs/r9700.md R9, "reduced-vocab draft head"): bakes an OPTIONAL smaller
// lm_head (mtp.draft_head.lm_head.{layout} + mtp.draft_head.vocab_ids, docs/container-format.md
// "mtp.*") into the container for MtpHead::Draft's own drafting -- ignored unless --mtp is also on;
// a container built without this flag simply has no draft_head.* tensors (old-container-compatible,
// loader falls back to the full-vocab head automatically, src/model/container.cpp).
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
//
//   r4dx-convert --dflash-gguf <DFlash2 draft .gguf> --out <container path>
//                [--layout {w4a16,w4a8,mxfp4,bf16}] [--threads T]
//
// Converts a DFlash2 speculative-decoding draft model (docs/container-format.md "DFlash2 draft
// container") from its GGUF v3 source into its own r4dx container (container_kind
// "dflash2_draft", separate file from the main text-model container). Produces ONE container per
// invocation carrying exactly the requested `--layout` (bf16 is only included when
// `--layout bf16` is requested; unlike the main HF-checkpoint mode, there is no automatic bf16
// side-by-side -- see DflashLayoutSet).
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "nlohmann/json.hpp"
#include "r4d.h"  // r4d_gemm_{w4a16,w4a8,mxfp4a8}_nt_m64_group() -- cross-checked against this
                  // converter's own kW4A16Group/kW4A8Group/kMxfp4Group at startup
                  // (ValidateKernelGroupSizes).
#include "r4dx_convert/container_writer.hpp"
#include "r4dx_convert/dflash2_container.hpp"
#include "r4dx_convert/gguf_reader.hpp"
#include "r4dx_convert/keep_bf16.hpp"
#include "r4dx_convert/kv_calib.hpp"
#include "r4dx_convert/linear_layouts.hpp"
#include "r4dx_convert/npz_reader.hpp"
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
// (r4dx_convert::kW4A16Group, kW4A8Group, kMxfp4Group) are compile-time constants that must match
// the group sizes the GPU kernels were actually built with (r4d_gemm_w4a8_nt_m64.hip's
// R4D_GEMM_W4A8_GROUP default is 256; third_party/CMakeLists.txt overrides it to 128, and w4a16's
// R4D_GEMM_W4_GROUP is the R4DX_W4A16_GROUP build option). A silent mismatch here produces a
// container that GEMMs read at the wrong stride with no error, ever. r4d_core exports the group
// each kernel was actually compiled with, so assert against it once at startup instead of trusting
// the -D flags stayed in sync. Each layout is checked against ITS OWN kernel export -- w4a16 and
// w4a8 no longer share a constant, precisely so R4DX_W4A16_GROUP can move on its own.
void ValidateKernelGroupSizes() {
  auto check = [](const char* label, int expected, int actual) {
    if (actual != expected) {
      throw std::runtime_error(
          std::string("r4dx-convert: group size mismatch for ") + label + ": this converter packs "
          "with group=" + std::to_string(expected) + " but r4d_core's kernel was built with group=" +
          std::to_string(actual) + " (check third_party/CMakeLists.txt's R4D_EXTRA_* -D flags and "
          "the R4DX_W4A16_GROUP cache variable against "
          "src/convert/include/r4dx_convert/quant_{int4,mxfp4}.hpp's kW4A16Group/kW4A8Group/"
          "kMxfp4Group)");
    }
  };
  check("w4a16", r4dx_convert::kW4A16Group, r4d_gemm_w4a16_nt_m64_group());
  check("w4a8", r4dx_convert::kW4A8Group, r4d_gemm_w4a8_nt_m64_group());
  check("mxfp4", r4dx_convert::kMxfp4Group, r4d_gemm_mxfp4a8_nt_m64_group());
}

// Builds the __metadata__["quant"] block (review finding, major): the task brief asked for
// per-tensor quant layout/group/permutation in __metadata__ and the original run only had prose
// in quant_summary. This is the authoritative, loader-checkable record -- src/model should assert
// its own kernel's group() matches these before trusting the container.
nlohmann::json BuildQuantMetadata() {
  return {
      {"w4a16",
       {{"group", r4dx_convert::kW4A16Group},
        {"zero_mode", "free_0_15"},
        {"nibble_encoding", "offset_binary_xor8"},
        {"fragment_permutation", "wmma16x16x16_lane16_koff_0_8_1_9_2_10_3_11"}}},
      {"w4a8",
       {{"group", r4dx_convert::kW4A8Group},
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
  bool lm_head_spec_explicit = false;  // --lm-head was given: --no-bf16 then leaves it alone
  int layers = -1;  // -1 = every text layer
  int threads = 0;  // 0 = hardware_concurrency
  int vision = -1;  // -1 auto (on iff full run), 0 off, 1 on
  int mtp = -1;
  bool no_bf16 = false;  // drop the bf16 layout for body weights even if --layouts/--lm-head asked
  std::string kv_calib;  // path to tools/reference/kv_calibrate.py's merged JSON; empty = no calib
  // Reduced-vocab MTP draft head (docs/r9700.md R9): path to a JSON file `{"vocab_ids": [...]}`
  // (real vocabulary ids, produced by tests/model/tool_vocab_calib.cpp's calibration pass) --
  // when non-empty AND --mtp is on, slices those rows out of lm_head.weight and emits them as
  // mtp.draft_head.lm_head.{layout} + mtp.draft_head.vocab_ids (docs/container-format.md "mtp.*").
  // Empty (default): no draft_head.* tensors are written -- MtpHead::Draft falls back to the
  // full-vocab head unconditionally, byte-identical to every container built before this flag
  // existed.
  std::string draft_vocab_ids;

  // DFlash2 draft-model mode (docs/dflash2.md): --dflash-gguf <gguf> --out <container> --layout {...}.
  // Mutually exclusive with the HF-checkpoint (--input/--output) and --selftest modes. Uses its
  // own --out/--layout flag names (not --output/--layouts) since this mode packs exactly one
  // requested layout per run (no automatic bf16 companion -- pass --layout bf16 for an
  // exact-precision container), not the HF mode's "every layout side by side" model.
  std::string dflash_gguf;
  std::string dflash_out;
  std::string dflash_layout = "w4a16";  // one of w4a16, w4a8, mxfp4, bf16

  // How the 4-bit quantizers choose their (scale, zero) values -- the on-disk BYTE LAYOUT is
  // identical either way (src/convert/include/r4dx_convert/quant_search.hpp). "rtn" (default) is
  // the historical min/max + round-to-nearest grid and reproduces every container built before this
  // flag existed byte for byte; "search" minimizes the (optionally importance-weighted) squared
  // reconstruction error per (row, group). See this file's header comment for why the default is
  // rtn and not search.
  std::string quant = "rtn";
  // Importance matrix ("imatrix") .npz from tools/reference/imatrix_capture.py, keyed by the
  // converter's own container base names. Weights the search's error term by the mean activation
  // energy of each input channel. Only meaningful with --quant search; a linear with no entry in
  // the file falls back to unweighted MSE (with a count reported at the end of the run).
  std::string imatrix;

  // ECMAScript regex over container base names; matching linears are written as `<base>.bf16.w`
  // only (r4dx_convert::KeepBf16Selector, keep_bf16.hpp). Empty (default) = every linear is
  // quantized exactly as before this flag existed, byte for byte.
  std::string keep_bf16;
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
    else if (arg == "--lm-head") { a.lm_head_spec = next(i); a.lm_head_spec_explicit = true; }
    else if (arg == "--layers") a.layers = std::stoi(next(i));
    else if (arg == "--threads") a.threads = std::stoi(next(i));
    else if (arg == "--vision") a.vision = ParseOnOff(next(i), "--vision");
    else if (arg == "--mtp") a.mtp = ParseOnOff(next(i), "--mtp");
    else if (arg == "--no-bf16") a.no_bf16 = true;
    else if (arg == "--kv-calib") a.kv_calib = next(i);
    else if (arg == "--draft-vocab-ids") a.draft_vocab_ids = next(i);
    else if (arg == "--dflash-gguf") a.dflash_gguf = next(i);
    else if (arg == "--out") a.dflash_out = next(i);
    else if (arg == "--layout") a.dflash_layout = next(i);
    else if (arg == "--quant") a.quant = next(i);
    else if (arg == "--imatrix") a.imatrix = next(i);
    else if (arg == "--keep-bf16") a.keep_bf16 = next(i);
    else throw std::runtime_error("unknown argument: " + arg);
  }
  if (a.quant != "rtn" && a.quant != "search")
    throw std::runtime_error("--quant must be 'rtn' or 'search', got '" + a.quant + "'");
  if (!a.imatrix.empty() && a.quant != "search")
    throw std::runtime_error("--imatrix requires --quant search (got --quant " + a.quant + ")");
  // The imatrix .npz is keyed by the QWEN container's own add_linear base names
  // (tools/reference/imatrix_capture.py); the DFlash2 drafter's tensors share none of them, so an
  // --imatrix passed alongside --dflash-gguf could only ever be a no-op. Reject it instead of
  // silently converting the drafter with unweighted MSE and reporting "imatrix-weighted".
  if (!a.imatrix.empty() && !a.dflash_gguf.empty())
    throw std::runtime_error(
        "--imatrix does not apply to --dflash-gguf (the drafter has no importance matrix; "
        "tools/reference/imatrix_capture.py only captures the main model's linears)");
  // --keep-bf16 is wired into the HF-checkpoint and --selftest paths only. Accepting it silently
  // alongside --dflash-gguf would produce a drafter container in which it did nothing at all -- the
  // exact failure mode the "matched nothing" warning exists to make visible, so reject it outright
  // rather than emit a warning nobody reads in a sweep log. (--layout bf16 is how you get a bf16
  // drafter.)
  if (!a.keep_bf16.empty() && !a.dflash_gguf.empty())
    throw std::runtime_error("--keep-bf16 does not apply to --dflash-gguf (use --layout bf16 for a "
                             "bf16 drafter container)");
  return a;
}

// Loads the imatrix .npz once (or nothing) and hands out per-linear weight vectors by container
// base name. Also tallies coverage so a run against a stale/partial imatrix is visible in the log
// rather than silently degrading to unweighted MSE for half the model.
class ImatrixSource {
 public:
  ImatrixSource(const std::string& path, r4dx_convert::QuantMode mode) : mode_(mode) {
    if (path.empty()) return;
    npz_ = std::make_unique<r4dx_convert::NpzReader>(path);
    std::cout << "[r4dx-convert] imatrix=" << path << " (" << npz_->Count() << " vector(s))\n";
  }

  r4dx_convert::QuantOptions For(const std::string& container_base, int64_t K) {
    r4dx_convert::QuantOptions opts;
    opts.mode = mode_;
    if (!npz_) return opts;
    int64_t len = 0;
    const float* v = npz_->Vector(container_base, &len);
    if (v == nullptr) {
      std::lock_guard<std::mutex> lk(mu_);
      ++missing_;
      if (missing_ <= 8)
        std::cerr << "[r4dx-convert] WARNING: imatrix has no vector for '" << container_base
                  << "' -- that linear is quantized with unweighted MSE\n";
      return opts;
    }
    if (len != K) {
      // A length mismatch means the imatrix was captured against a different checkpoint/shape;
      // using it would weight the wrong channels, which is worse than not weighting at all.
      throw std::runtime_error("r4dx-convert: imatrix vector '" + container_base + "' has length " +
                                std::to_string(len) + " but the linear's K is " +
                                std::to_string(K));
    }
    opts.importance.data = v;
    opts.importance.size = len;
    {
      std::lock_guard<std::mutex> lk(mu_);
      ++hit_;
    }
    return opts;
  }

  // A partial imatrix is a silent quality regression -- the container still converts, still says
  // "imatrix-weighted" at the top of a 250 s log, and just quietly quantizes some fraction of the
  // model with unweighted MSE. So the coverage line goes to stderr and says WARNING when anything
  // fell back, next to the (capped) per-linear warnings, instead of being one more stdout line.
  void ReportCoverage() const {
    if (!npz_) return;
    std::ostream& os = (missing_ > 0) ? std::cerr : std::cout;
    os << "[r4dx-convert] " << (missing_ > 0 ? "WARNING: " : "")
       << "imatrix coverage: " << hit_ << " linear(s) weighted, " << missing_
       << " fell back to unweighted MSE\n";
  }

 private:
  r4dx_convert::QuantMode mode_;
  std::unique_ptr<r4dx_convert::NpzReader> npz_;
  std::mutex mu_;
  int64_t hit_ = 0, missing_ = 0;
};

r4dx_convert::QuantMode ParseQuantMode(const std::string& s) {
  return s == "search" ? r4dx_convert::QuantMode::kSearch : r4dx_convert::QuantMode::kRtn;
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
    // An explicit `--lm-head bf16` (or `4bit+bf16`) is a deliberate request for a bf16 lm_head in
    // an otherwise quantized-only container (rung 4 follow-up: the 4-bit lm_head is where the
    // vocab-tail KL loss sits) -- only the DEFAULT lm_head spec is subject to --no-bf16.
    if (!args.lm_head_spec_explicit) lm_head_layouts.bf16 = false;
  }

  std::cout << "[r4dx-convert] input=" << args.input << " output=" << args.output << "\n"
            << "[r4dx-convert] hidden=" << hidden << " layers=" << layers << "/" << num_layers_total
            << " threads=" << threads << " vision=" << (do_vision ? "on" : "off")
            << " mtp=" << (do_mtp ? "on" : "off") << "\n";

  // The BYTE LAYOUT is identical in both modes (docs/container-format.md "How the quantized values
  // are chosen"); only the (scale, zero) values differ.
  const r4dx_convert::QuantMode quant_mode = ParseQuantMode(args.quant);
  std::cout << "[r4dx-convert] quant=" << args.quant
            << (args.imatrix.empty() ? " (unweighted MSE)" : " (imatrix-weighted)") << "\n";
  ImatrixSource imatrix(args.imatrix, quant_mode);

  // --keep-bf16 (keep_bf16.hpp): constructed here so an invalid regex fails before the first shard
  // is opened, not 250 s into the emit pass.
  r4dx_convert::KeepBf16Selector keep_bf16(args.keep_bf16);
  if (keep_bf16.Enabled())
    std::cout << "[r4dx-convert] keep-bf16=" << keep_bf16.Pattern()
              << " (matching linears written as <base>.bf16.w only)\n";

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
                         LayoutSet requested) {
    // --keep-bf16 decides per container base name, which is known here -- so plan and emit cannot
    // disagree about what this linear is (the two lambdas below capture the SAME resolved `ls`,
    // rather than each re-evaluating the regex).
    const bool kept = keep_bf16.Matches(container_base);
    const LayoutSet ls = kept ? r4dx_convert::KeptBf16LayoutSet() : requested;
    plan_jobs.push_back([&writer, &model, &keep_bf16, hf_names, container_base, ls, requested,
                          kept]() {
      int64_t N = 0, K = 0;
      for (auto& n : hf_names) {
        const auto& m = model.Meta(n);
        N += m.shape[0];
        K = m.shape[1];
      }
      if (kept) {
        keep_bf16.Record(container_base, static_cast<int>(N), static_cast<int>(K), requested,
                         std::cout);
      }
      r4dx_convert::PlanLinearLayouts(writer, container_base, static_cast<int>(N),
                                       static_cast<int>(K), ls);
    });
    emit_jobs.push_back([&writer, &model, &imatrix, hf_names, container_base, ls, threads]() {
      std::vector<float> w;
      int64_t K = 0;
      for (auto& n : hf_names) {
        auto part = r4dx_convert::ReadTensorAsFloat(model, n);
        K = model.Meta(n).shape[1];
        w.insert(w.end(), part.begin(), part.end());
      }
      const int64_t N = static_cast<int64_t>(w.size()) / K;
      // Every add_linear fuses on the OUTPUT axis only (mlp.gate_up is the only fusion and both
      // halves share K), so one length-K importance vector per container base is well defined --
      // tools/reference/imatrix_capture.py asserts the same thing from the other side.
      r4dx_convert::EmitLinearLayouts(writer, container_base, w, static_cast<int>(N),
                                       static_cast<int>(K), ls, threads,
                                       imatrix.For(container_base, K));
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

    // Reduced-vocab draft head (docs/r9700.md R9): OPTIONAL, only when the caller supplied a
    // calibration-chosen vocabulary subset (tests/model/tool_vocab_calib.cpp's own output JSON).
    // `mtp.draft_head.lm_head` is planned/emitted through the SAME PlanLinearLayouts/
    // EmitLinearLayouts helpers every other quantized linear uses (a plain [draft_vocab_size,
    // hidden] slice of lm_head.weight's ROWS, same K as the real lm_head), in the same LayoutSet as
    // every other mtp.* head linear (`layouts`) so Container::Load's mtp_head_layout selection at
    // load time works identically to mtp.attn.qg/o and mtp.mlp.gate_up/down. `vocab_ids` is a plain
    // raw int32 tensor (subset index -> real vocab id), written directly (not through
    // EncodeFp32/EncodeBf16 -- it is already the on-disk bit pattern, no dtype conversion needed).
    if (!args.draft_vocab_ids.empty()) {
      const nlohmann::json ids_json = nlohmann::json::parse(ReadFile(args.draft_vocab_ids));
      const std::vector<int64_t> draft_ids = ids_json.at("vocab_ids").get<std::vector<int64_t>>();
      if (draft_ids.empty()) {
        throw std::runtime_error("--draft-vocab-ids: 'vocab_ids' array is empty in " +
                                  args.draft_vocab_ids);
      }
      const int64_t draft_vocab_size = static_cast<int64_t>(draft_ids.size());
      std::cout << "[r4dx-convert] draft-vocab-ids=" << args.draft_vocab_ids << " ("
                << draft_vocab_size << " ids)\n";

      // --keep-bf16 applies here too, resolved on the same base name the regex sees everywhere
      // else -- so a pattern like "lm_head$" that happens to reach this optional head keeps it in
      // bf16 rather than quantizing it while claiming otherwise.
      const bool draft_head_kept = keep_bf16.Matches("mtp.draft_head.lm_head");
      const LayoutSet draft_head_layouts =
          draft_head_kept ? r4dx_convert::KeptBf16LayoutSet() : layouts;
      plan_jobs.push_back([&writer, &model, &keep_bf16, draft_vocab_size, draft_head_layouts,
                            layouts, draft_head_kept]() {
        const auto& m = model.Meta("lm_head.weight");
        const int64_t hidden_k = m.shape[1];
        if (draft_head_kept) {
          keep_bf16.Record("mtp.draft_head.lm_head", static_cast<int>(draft_vocab_size),
                           static_cast<int>(hidden_k), layouts, std::cout);
        }
        r4dx_convert::PlanLinearLayouts(writer, "mtp.draft_head.lm_head",
                                         static_cast<int>(draft_vocab_size),
                                         static_cast<int>(hidden_k), draft_head_layouts);
        writer.Plan("mtp.draft_head.vocab_ids", {draft_vocab_size, 4},
                    static_cast<uint64_t>(draft_vocab_size) * 4);
      });
      emit_jobs.push_back([&writer, &model, &imatrix, draft_ids, draft_head_layouts, threads]() {
        const auto full = r4dx_convert::ReadTensorAsFloat(model, "lm_head.weight");
        const int64_t hidden_k = model.Meta("lm_head.weight").shape[1];
        const int64_t vocab_full = static_cast<int64_t>(full.size()) / hidden_k;
        std::vector<float> sliced(draft_ids.size() * static_cast<size_t>(hidden_k));
        std::vector<int32_t> ids32(draft_ids.size());
        for (size_t i = 0; i < draft_ids.size(); ++i) {
          const int64_t id = draft_ids[i];
          if (id < 0 || id >= vocab_full) {
            throw std::runtime_error("--draft-vocab-ids: id " + std::to_string(id) +
                                      " out of range [0," + std::to_string(vocab_full) + ")");
          }
          std::copy(full.begin() + id * hidden_k, full.begin() + (id + 1) * hidden_k,
                    sliced.begin() + static_cast<int64_t>(i) * hidden_k);
          ids32[i] = static_cast<int32_t>(id);
        }
        // Its input distribution is the MTP layer's post-`mtp.norm` hidden, NOT the backbone's, so
        // it must never borrow `lm_head`'s vector -- imatrix_capture.py emits its own
        // `mtp.draft_head.lm_head` only under --draft-head, and without that key this falls back to
        // unweighted MSE (with the usual warning) rather than silently mis-weighting.
        r4dx_convert::EmitLinearLayouts(writer, "mtp.draft_head.lm_head", sliced,
                                         static_cast<int>(draft_ids.size()),
                                         static_cast<int>(hidden_k), draft_head_layouts, threads,
                                         imatrix.For("mtp.draft_head.lm_head", hidden_k));
        writer.WriteTensor("mtp.draft_head.vocab_ids", ids32.data(), ids32.size() * 4);
      });
    }
  }

  const auto t0 = std::chrono::steady_clock::now();

  for (auto& j : plan_jobs) j();
  // After planning (every add_linear's shapes are known by now), before the header is written --
  // so the summary/"matched nothing" warning lands ahead of the long emit pass rather than after it.
  keep_bf16.Report(std::cout, std::cerr);

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
      {"mtp.draft_head.lm_head", "mxfp4|w4a16|w4a8|bf16 (all four present, OPTIONAL -- only when "
                                  "--draft-vocab-ids was given, docs/r9700.md R9)"},
      {"mtp.draft_head.vocab_ids", "raw int32[draft_vocab_size] (OPTIONAL, same condition)"},
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
      {"draft_vocab_ids",
       args.draft_vocab_ids.empty() ? std::string("none (no reduced-vocab draft head)")
                                     : args.draft_vocab_ids},
      // How the quantized values were CHOSEN (the byte layout is the same either way) -- so a
      // container on disk says which quantizer produced it without re-deriving it from the bytes.
      {"quant_values", args.quant},
      {"imatrix", args.imatrix.empty() ? std::string("none (unweighted MSE)") : args.imatrix},
      // --keep-bf16 (keep_bf16.hpp): which linears in THIS container are bf16 passthroughs rather
      // than quantized. Recorded as the pattern plus the resolved name list, because the pattern
      // alone cannot be re-evaluated later without the converter's own base-name vocabulary, and a
      // sensitivity table is only readable next to the exact set of tensors each row covers.
      {"keep_bf16", args.keep_bf16.empty() ? std::string("none (every linear quantized)")
                                            : args.keep_bf16},
      {"keep_bf16_linears", keep_bf16.Matched()},
      {"keep_bf16_extra_bytes", keep_bf16.ExtraBytes()},
  };
  writer.FinalizeHeader(args.output, metadata);
  std::cout << "[r4dx-convert] planned " << writer.PlannedTensorCount() << " tensors, "
            << writer.PlannedDataBytes() << " data bytes\n";

  for (auto& j : emit_jobs) j();
  writer.Finish();
  imatrix.ReportCoverage();

  const auto t1 = std::chrono::steady_clock::now();
  const double secs = std::chrono::duration<double>(t1 - t0).count();
  std::cout << "[r4dx-convert] wrote " << args.output << " in " << secs << " s\n";
  return 0;
}

// ---- --selftest mode -----------------------------------------------------------------------

int RunSelftest(const AppArgs& args) {
  const int threads = ResolveThreads(args.threads);
  const LayoutSet requested = ParseLayoutList(args.layouts_spec, /*bf16_default_on=*/true);
  // The selftest's single tensor has container base "selftest", so --keep-bf16 is exercisable end
  // to end here in milliseconds, against tests/convert/fixtures/input.safetensors and without a
  // 27B checkpoint -- the quickest way to see the whole CLI wiring (regex -> LayoutSet -> emitted
  // tensor set -> warning path) by hand. The automated gates are split: tests/convert/
  // test_keep_bf16.cpp owns selection/emission/accounting through the library API (no exe, no
  // checkpoint), and tests/model/test_keep_bf16.cpp runs the real binary on a 4-layer container.
  r4dx_convert::KeepBf16Selector keep_bf16(args.keep_bf16);
  const bool kept = keep_bf16.Matches("selftest");
  const LayoutSet layouts = kept ? r4dx_convert::KeptBf16LayoutSet() : requested;

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

  // The selftest packs one tensor named `--selftest-name`'s container base "selftest", so an
  // --imatrix passed here must carry a "selftest" vector of length K -- that is what
  // tools/convert_ref/selftest_compare.py generates for its weighted pass.
  ImatrixSource imatrix(args.imatrix, ParseQuantMode(args.quant));
  const r4dx_convert::QuantOptions opts = imatrix.For("selftest", K);

  ContainerWriter writer;
  if (kept) keep_bf16.Record("selftest", N, K, requested, std::cout);
  keep_bf16.Report(std::cout, std::cerr);
  r4dx_convert::PlanLinearLayouts(writer, "selftest", N, K, layouts);
  nlohmann::json metadata;
  metadata["r4dx_format_version"] = "1";
  metadata["model_id"] = "selftest";
  metadata["produced_by"] = "r4dx-convert --selftest";
  metadata["quant"] = BuildQuantMetadata();
  writer.FinalizeHeader(args.selftest_output, metadata);
  r4dx_convert::EmitLinearLayouts(writer, "selftest", w, N, K, layouts, threads, opts);
  writer.Finish();

  std::cout << "[r4dx-convert --selftest] N=" << N << " K=" << K << " quant=" << args.quant
            << (opts.importance.empty() ? " (unweighted MSE)" : " (imatrix-weighted)") << " -> "
            << args.selftest_output << "\n";
  imatrix.ReportCoverage();
  return 0;
}

// ---- --dflash-gguf mode (task A1) -----------------------------------------------------------

LayoutSet DflashLayoutSet(const std::string& layout_name) {
  // Exactly the ONE requested layout, no automatic bf16 side-by-side copy (deliberately UNLIKE
  // the main text-model container's "every layout side by side in one file" convention) -- the
  // task spec's own expected container sizes (~1.2 GB each for w4a16/w4a8/mxfp4, ~3.9 GB for
  // bf16) only make sense as single-layout files; `--dflash-gguf ... --layout X` produces ONE
  // container per invocation, matching `r4dx-cli --layout`'s per-run layout selection rather than
  // `r4dx-convert`'s (HF-mode) `--layouts a,b,c` side-by-side list.
  LayoutSet ls;
  ls.bf16 = false;
  if (layout_name == "bf16") ls.bf16 = true;
  else if (layout_name == "w4a16") ls.w4a16 = true;
  else if (layout_name == "w4a8") ls.w4a8 = true;
  else if (layout_name == "mxfp4") ls.mxfp4 = true;
  else throw std::runtime_error("--layout must be one of w4a16, w4a8, mxfp4, bf16 (got '" + layout_name + "')");
  return ls;
}

int RunDflashConvert(const AppArgs& args) {
  using namespace r4dx_convert;
  if (args.dflash_out.empty()) throw std::runtime_error("--dflash-gguf requires --out");
  const int threads = ResolveThreads(args.threads);
  const LayoutSet layouts = DflashLayoutSet(args.dflash_layout);

  GgufReader gguf(Utf8ToWide(args.dflash_gguf));
  Dflash2Metadata meta = ReadDflash2Metadata(gguf);

  // The DFlash2 drafter has no imatrix of its own (imatrix_capture.py's keys are the main model's
  // container bases), so --quant search here is the unweighted-MSE search.
  r4dx_convert::QuantOptions quant_opts;
  quant_opts.mode = ParseQuantMode(args.quant);

  std::cout << "[r4dx-convert --dflash-gguf] input=" << args.dflash_gguf << " out=" << args.dflash_out
            << " layout=" << args.dflash_layout << " threads=" << threads
            << " quant=" << args.quant << "\n"
            << "[r4dx-convert --dflash-gguf] hidden=" << meta.hidden << " block_count=" << meta.block_count
            << " vocab=" << meta.vocab_size << " n_rot=" << meta.n_rot << "\n";

  ContainerWriter writer;
  std::vector<std::function<void()>> plan_jobs, emit_jobs;

  auto add_linear = [&](std::string gguf_name, std::string container_base) {
    plan_jobs.push_back([&writer, &gguf, gguf_name, container_base, layouts]() {
      PlanDflash2Linear(writer, gguf, Dflash2LinearSpec{gguf_name, container_base}, layouts);
    });
    emit_jobs.push_back([&writer, &gguf, gguf_name, container_base, layouts, threads, quant_opts]() {
      EmitDflash2Linear(writer, gguf, Dflash2LinearSpec{gguf_name, container_base}, layouts, threads,
                        quant_opts);
    });
  };
  auto add_f32 = [&](std::string gguf_name, std::string container_name) {
    plan_jobs.push_back([&writer, &gguf, gguf_name, container_name]() {
      PlanDflash2F32(writer, gguf, gguf_name, container_name);
    });
    emit_jobs.push_back([&writer, &gguf, gguf_name, container_name]() {
      EmitDflash2F32(writer, gguf, gguf_name, container_name);
    });
  };
  auto add_bf16 = [&](std::string gguf_name, std::string container_name) {
    plan_jobs.push_back([&writer, &gguf, gguf_name, container_name]() {
      PlanDflash2Bf16(writer, gguf, gguf_name, container_name);
    });
    emit_jobs.push_back([&writer, &gguf, gguf_name, container_name]() {
      EmitDflash2Bf16(writer, gguf, gguf_name, container_name);
    });
  };

  add_linear("fc.weight", "dflash.fc");
  add_f32("enc.output_norm.weight", "dflash.enc_output_norm");
  add_f32("output_norm.weight", "dflash.output_norm");
  add_linear("selector_hidden.weight", "dflash.selector.hidden");
  // Row-gather codebooks (task A1: "NOT quantized to a GEMM layout"), bf16, GGUF's own
  // [vocab][rank] flat order preserved verbatim (see dflash2_container.hpp's file comment).
  add_bf16("selector_predecessor.weight", "dflash.selector.predecessor");
  add_bf16("selector_successor.weight", "dflash.selector.successor");

  for (int64_t i = 0; i < meta.block_count; ++i) {
    const std::string g = "blk." + std::to_string(i) + ".";
    const std::string base = "dflash.layers." + std::to_string(i) + ".";
    add_f32(g + "attn_norm.weight", base + "input_layernorm");
    add_linear(g + "attn_q.weight", base + "self_attn.q_proj");
    add_linear(g + "attn_k.weight", base + "self_attn.k_proj");
    add_linear(g + "attn_v.weight", base + "self_attn.v_proj");
    add_linear(g + "attn_output.weight", base + "self_attn.o_proj");
    add_f32(g + "attn_q_norm.weight", base + "self_attn.q_norm");
    add_f32(g + "attn_k_norm.weight", base + "self_attn.k_norm");
    // conv_base: F32 in the GGUF, ne=[hidden,2,2] = flat [side][tap][hidden] (hidden fastest) --
    // exactly third_party/libr4d/r4d_dflash_conv_body.h's expected per-side [taps,H] slicing, no
    // permutation needed; stored bf16 per the task spec (the libr4d kernel reads bf16 x/base/delta).
    add_bf16(g + "attn_conv_base", base + "self_attn.conv.base");
    add_linear(g + "attn_conv_proj.weight", base + "self_attn.conv.proj");
    add_f32(g + "ffn_norm.weight", base + "post_attention_layernorm");
    add_linear(g + "ffn_gate.weight", base + "mlp.gate_proj");
    add_linear(g + "ffn_up.weight", base + "mlp.up_proj");
    add_linear(g + "ffn_down.weight", base + "mlp.down_proj");
    add_bf16(g + "ffn_conv_base", base + "mlp.conv.base");
    add_linear(g + "ffn_conv_proj.weight", base + "mlp.conv.proj");
  }

  for (auto& j : plan_jobs) j();

  nlohmann::json metadata;
  metadata["r4dx_format_version"] = "1";
  metadata["container_kind"] = "dflash2_draft";
  metadata["model_id"] = "z-lab/Qwen3.8-27B-DFlash2";
  metadata["produced_by"] = "r4dx-convert --dflash-gguf (r4dx dev build)";
  metadata["source_gguf"] = {
      {"filename", args.dflash_gguf.substr(args.dflash_gguf.find_last_of("/\\") + 1)},
      {"sha256_first_1mib", Sha256HexOfFilePrefix(args.dflash_gguf, 1024 * 1024)},
  };
  metadata["dflash2"] = BuildDflash2MetadataJson(meta);
  metadata["dflash2"]["layout"] = args.dflash_layout;
  metadata["quant"] = BuildQuantMetadata();

  const auto t0 = std::chrono::steady_clock::now();
  writer.FinalizeHeader(args.dflash_out, metadata);
  std::cout << "[r4dx-convert --dflash-gguf] planned " << writer.PlannedTensorCount() << " tensors, "
            << writer.PlannedDataBytes() << " data bytes\n";

  for (auto& j : emit_jobs) j();
  writer.Finish();

  const auto t1 = std::chrono::steady_clock::now();
  const double secs = std::chrono::duration<double>(t1 - t0).count();
  std::cout << "[r4dx-convert --dflash-gguf] wrote " << args.dflash_out << " in " << secs << " s\n";
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    ValidateKernelGroupSizes();
    AppArgs args = ParseArgs(argc, argv);
    if (!args.dflash_gguf.empty()) return RunDflashConvert(args);
    if (args.selftest) {
      if (args.selftest_input.empty() || args.selftest_output.empty())
        throw std::runtime_error("--selftest requires --selftest-input and --selftest-output");
      return RunSelftest(args);
    }
    if (args.input.empty() || args.output.empty())
      throw std::runtime_error("--input and --output are required (or pass --selftest or --dflash-gguf)");
    return RunConvert(args);
  } catch (const std::exception& e) {
    std::cerr << "r4dx-convert: error: " << e.what() << "\n";
    return 1;
  }
}
