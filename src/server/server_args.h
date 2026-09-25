// r4dx::server::ServerArgs / ParseArgs -- argument parsing for r4dx-server, split into its own
// header (out of main.cpp) purely so tests/server/test_server_args.cpp can exercise it without
// linking a HIP-dependent executable, mirroring src/cli/cli_args.h's own rationale exactly.
// Header-only: a couple dozen lines of string parsing, not worth a .cpp.
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace r4dx::server {

// See src/cli/cli_args.h's kMaxMtpDraftK for the derivation (Model::VerifyWindow requires
// mtp+1 candidates to fit in a <=64-row chunk) -- duplicated here rather than shared because
// these two headers are already a deliberate, documented duplication of every other --mtp* flag.
inline constexpr int64_t kMaxMtpDraftK = 63;
// ... and 7 under --tp 2: src/cli/cli_args.h's kMaxMtpDraftKTp (docs/tp.md Appendix B N80).
inline constexpr int64_t kMaxMtpDraftKTp = 7;

struct ServerArgs {
  std::string model_path;
  std::string layout = "bf16";
  std::string tokenizer_dir = "C:\\AI\\models\\Qwen3.8-27B";
  std::string host = "127.0.0.1";
  int port = 8080;
  // docs/r9700.md R13 (2026-09-20, measured): raised from 131072 -- the checkpoint's own
  // config.json declares max_position_embeddings=262144 (rope_type "default", no scaling needed),
  // and real hardware measurement shows the full 262144-token KV+GDN allocation still leaves
  // 21-24% of the card's VRAM free (see src/cli/cli_args.h's matching comment and docs/r9700.md's
  // R13/Q17 entries for the full measurement).
  int64_t max_ctx = 262144;
  // -1 (default): load every layer Config().num_hidden_layers declares (the real 64-layer
  // container). >=0: load only the first N layers -- required for a smaller test container that
  // physically carries fewer layers than its (verbatim-copied) config.json declares, e.g.
  // D:\models\r4dx\qwen38-27b-l4-bf16.r4dx (mirrors r4dx-convert's own --layers N and
  // r4dx::model::ModelOptions::layer_limit; see tools/server/smoke.ps1).
  int64_t layers = -1;
  int64_t max_tokens_default = 128;
  int max_queue = 16;
  bool think = false;

  // Sampling defaults applied whenever a request omits the corresponding field.
  float default_temperature = 1.0f;
  float default_top_p = 1.0f;
  int default_top_k = 0;
  float default_min_p = 0.0f;

  std::string log_level = "info";  // one of debug|info|warn|error

  // MTP self-speculative decode (docs/mtp.md), same semantics/default as src/cli/cli_args.h's
  // --mtp: 0 (disabled) unless the loaded --model container was converted with --mtp on. Used by
  // every request once the server is started with --mtp N>0 against an MTP container
  // (Engine::RunRequest's use_mtp = model_->MtpEnabled(), no longer gated on the request's own
  // temperature as of Milestone 6 stage S3, docs/sampling.md section 12/docs/server.md): a
  // request's OWN sampling.temperature<=0 picks Model::DecodeStepMtpGreedy (byte-identical to
  // pre-S3), temperature>0 picks Model::DecodeStepMtpSampled (sample-and-match rejection
  // sampling against the request's own distribution).
  int64_t mtp = 0;
  // MTP head layout (docs/mtp.md "MTP head layout"), same flag/semantics/default as
  // src/cli/cli_args.h's --mtp-head-layout: "layout" (default, measured faster in 23/24 K x layout
  // configurations with no acceptance-rate cost) loads the MTP head's four quantized linears
  // (mtp.attn.qg/o, mtp.mlp.gate_up/down) in the same layout as --layout; "bf16" forces the
  // exact-arithmetic head instead (~0.5 GB extra VRAM). No effect when --mtp is 0 or the container
  // has no mtp.* weights. Was previously CLI-only (docs/status.md Known-gaps: "no server-side
  // flag") -- every server Model used the measured-default layout-matched head unconditionally;
  // added here purely as a passthrough so a server operator can force bf16 too, without changing
  // the measured default.
  std::string mtp_head_layout = "layout";
  // Reduced-vocab draft head (docs/r9700.md R9), same flag/semantics/default as
  // src/cli/cli_args.h's --mtp-draft-head: "reduced" uses the container's OPTIONAL
  // mtp.draft_head.* tensors to speed up drafting when present, "full" (default, flipped
  // 2026-09-20 per Milestone 5 B2 item 8; see src/cli/cli_args.h's own comment for the full
  // measurement -- matched K=3 real-hardware re-measurement: full head 68.20/68.64 tok/s (46.3%
  // acceptance) vs reduced head 53.59/54.17 tok/s (20.9% acceptance), byte-identical output)
  // forces the exact pre-R9 full-vocab draft head.
  // Verification always stays full-vocab regardless of this flag (see model.h's
  // ModelOptions::mtp_draft_reduced_vocab / MtpHead::Draft's own doc comments), so it can only
  // affect drafting speed/acceptance, never generated output. No effect when --mtp is 0, the
  // container has no mtp.* weights, or (silently, "reduced" only) no draft_head.* tensors.
  std::string mtp_draft_head = "full";
  // Device-resident embedding gather (src/model/model.h's ModelOptions::embed_device_resident) --
  // same escape hatch and default as src/cli/cli_args.h's --embed-device-resident (review finding,
  // 2026-09-20: ModelOptions' own doc comment promised this flag before either binary actually had
  // it). "off" forces the host-gather path instead of mirroring text.embed_tokens into VRAM.
  std::string embed_device_resident = "on";
  // DFlash2 self-speculative decode (docs/dflash2.md, stage S3), same flag/semantics/default as
  // src/cli/cli_args.h's --dflash/--dflash-k/--dflash-p-min/--dflash-n-min: empty (default)
  // disables it; mutually exclusive with --mtp > 0 (checked below -- Model::Load re-checks it too).
  std::string dflash;
  int64_t dflash_k = 7;
  float dflash_p_min = 0.0f;
  int64_t dflash_n_min = 0;
  // Vision tower (docs/vision.md "Load policy") -- same three values, same default and the same
  // meaning as src/cli/cli_args.h's --vision: "auto" loads the container's vision.* weights iff it
  // has them, "on" fails the load when it does not, "off" never loads them.
  std::string vision = "auto";
  // Per-image pixel cap (docs/vision.md "Large images") -- same default and meaning as
  // src/cli/cli_args.h's --image-max-pixels: an image above this is DOWNSIZED via the reference's
  // own smart_resize rule rather than rejected. 0 means the checkpoint's own 16777216 ceiling.
  int64_t image_max_pixels = 1048576;

  // ---- tensor parallel (docs/tp.md 9.1) -----------------------------------------------------------
  // The same flags, defaults, ranges and usage errors as src/cli/cli_args.h's --tp* (the two headers
  // are the documented duplication above; tests/server/test_server_args.cpp's TestTpFlags mirrors
  // tests/cli/test_args.cpp's). --tp 1 (default) is today's single-device server and allows no other
  // --tp-* flag; --tp 2 runs r4dx::model::TpModel (real mode: both GPUs, HIP_VISIBLE_DEVICES unset --
  // rank 0 = device 1, rank 1 = device 0).
  int tp = 1;
  std::string tp_mode = "real";  // real|emulate|noop
  std::vector<int> tp_devices;   // --tp-devices a[,b]; empty = auto (docs/tp.md 9.2)
  int tp_rank = 0;               // --tp-rank r: noop only
  int tp_ar_timeout_ms = 500;    // [10, 1500]
  int tp_ar_nb = 4;              // [1, 64]
  int tp_ar_nb_large = 4;        // [1, 64]
  // --tp-submit-layers N / --tp-max-inflight K (docs/tp.md Appendix B N57, N64), both [0, 64];
  // -1 = not given: keep r4dx::model::TpOptions' own default (this header stays HIP-free).
  int tp_submit_layers = -1;
  int tp_max_inflight = -1;
  bool tp_options_given = false;  // any --tp-* flag other than --tp itself (refused at --tp 1)
};

// Thrown for a malformed/incomplete argument list -- ParseArgs never calls std::exit() itself, so
// it is safely unit-testable (tests/server/test_server_args.cpp) as well as callable from main()
// (which catches this and exits(2) with usage text).
struct ServerUsageError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

inline std::string ServerUsageText(const char* argv0) {
  return std::string("usage: ") + argv0 +
         " --model <container.r4dx> --layout {mxfp4|w4a16|w4a8|bf16} "
         "[--tokenizer-dir <dir>] [--host <addr>] [--port N] [--max-ctx N] "
         "[--max-tokens-default N] [--max-queue N] [--think {on|off}] [--layers N] "
         "[--default-temperature F] [--default-top-p F] [--default-top-k N] "
         "[--default-min-p F] [--log-level {debug|info|warn|error}] [--mtp N] "
         "[--mtp-head-layout {bf16|layout}] [--mtp-draft-head {reduced|full}] "
         "[--embed-device-resident {on|off}] [--dflash <draft.r4dx>] [--dflash-k N] "
         "[--dflash-p-min F] [--dflash-n-min N] [--vision {auto|on|off}] "
         "[--image-max-pixels N] "
         "[--tp {1|2}] [--tp-mode {real|emulate|noop}] [--tp-devices a[,b]] [--tp-rank r] "
         "[--tp-ar-timeout-ms N] [--tp-ar-nb N] [--tp-ar-nb-large N] [--tp-submit-layers N] "
         "[--tp-max-inflight K]";
}

inline std::string NextServerArg(int argc, char** argv, int& i, const char* flag) {
  if (i + 1 >= argc) throw ServerUsageError(std::string(flag) + " requires a value");
  return argv[++i];
}

// src/cli/cli_args.h's ParseTpDevices, throwing ServerUsageError: "auto" (or "") -> empty
// (docs/tp.md 9.2 auto); "a" or "a,b" -> the ordinals.
inline std::vector<int> ParseServerTpDevices(const std::string& value) {
  std::vector<int> out;
  if (value.empty() || value == "auto") return out;
  size_t start = 0;
  while (start <= value.size()) {
    const size_t comma = value.find(',', start);
    const std::string item = value.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
    int d = 0;
    try {
      size_t used = 0;
      d = std::stoi(item, &used);
      if (used != item.size()) throw std::invalid_argument(item);
    } catch (const std::exception&) {
      throw ServerUsageError("--tp-devices expects 'auto' or 'a[,b]' (HIP ordinals), got '" + value + "'");
    }
    if (d < 0) throw ServerUsageError("--tp-devices ordinals must be >= 0, got '" + value + "'");
    out.push_back(d);
    if (comma == std::string::npos) break;
    start = comma + 1;
  }
  if (out.size() > 2) throw ServerUsageError("--tp-devices takes at most two ordinals (rank 0, rank 1)");
  return out;
}

// Same rationale as src/cli/cli_args.h's ParseNumber: std::stoll/stof throw std::invalid_argument/
// std::out_of_range, not ServerUsageError, so an unparseable value must be converted here rather
// than escaping uncaught past main()'s ServerUsageError handler.
template <typename T, typename ParseFn>
T ParseServerNumber(const std::string& flag, const std::string& value, ParseFn parse) {
  try {
    return parse(value);
  } catch (const std::invalid_argument&) {
    throw ServerUsageError(flag + " expects a number, got '" + value + "'");
  } catch (const std::out_of_range&) {
    throw ServerUsageError(flag + " value out of range: '" + value + "'");
  }
}

inline int64_t ServerParseI64(const std::string& flag, const std::string& value) {
  return ParseServerNumber<int64_t>(flag, value, [](const std::string& v) { return std::stoll(v); });
}
inline int ServerParseInt(const std::string& flag, const std::string& value) {
  return ParseServerNumber<int>(flag, value, [](const std::string& v) { return std::stoi(v); });
}
inline float ServerParseFloat(const std::string& flag, const std::string& value) {
  return ParseServerNumber<float>(flag, value, [](const std::string& v) { return std::stof(v); });
}

inline ServerArgs ParseServerArgs(int argc, char** argv) {
  ServerArgs a;
  bool tp_rank_given = false;
  bool tp_submit_given = false, tp_inflight_given = false;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--model") a.model_path = NextServerArg(argc, argv, i, "--model");
    else if (arg == "--layout") a.layout = NextServerArg(argc, argv, i, "--layout");
    else if (arg == "--tokenizer-dir") a.tokenizer_dir = NextServerArg(argc, argv, i, "--tokenizer-dir");
    else if (arg == "--host") a.host = NextServerArg(argc, argv, i, "--host");
    else if (arg == "--port") a.port = ServerParseInt("--port", NextServerArg(argc, argv, i, "--port"));
    else if (arg == "--max-ctx") a.max_ctx = ServerParseI64("--max-ctx", NextServerArg(argc, argv, i, "--max-ctx"));
    else if (arg == "--max-tokens-default") a.max_tokens_default = ServerParseI64("--max-tokens-default", NextServerArg(argc, argv, i, "--max-tokens-default"));
    else if (arg == "--max-queue") a.max_queue = ServerParseInt("--max-queue", NextServerArg(argc, argv, i, "--max-queue"));
    else if (arg == "--layers") a.layers = ServerParseI64("--layers", NextServerArg(argc, argv, i, "--layers"));
    else if (arg == "--think") a.think = NextServerArg(argc, argv, i, "--think") == "on";
    else if (arg == "--default-temperature") a.default_temperature = ServerParseFloat("--default-temperature", NextServerArg(argc, argv, i, "--default-temperature"));
    else if (arg == "--default-top-p") a.default_top_p = ServerParseFloat("--default-top-p", NextServerArg(argc, argv, i, "--default-top-p"));
    else if (arg == "--default-top-k") a.default_top_k = ServerParseInt("--default-top-k", NextServerArg(argc, argv, i, "--default-top-k"));
    else if (arg == "--default-min-p") a.default_min_p = ServerParseFloat("--default-min-p", NextServerArg(argc, argv, i, "--default-min-p"));
    else if (arg == "--log-level") a.log_level = NextServerArg(argc, argv, i, "--log-level");
    else if (arg == "--mtp") a.mtp = ServerParseI64("--mtp", NextServerArg(argc, argv, i, "--mtp"));
    else if (arg == "--mtp-head-layout") a.mtp_head_layout = NextServerArg(argc, argv, i, "--mtp-head-layout");
    else if (arg == "--mtp-draft-head") a.mtp_draft_head = NextServerArg(argc, argv, i, "--mtp-draft-head");
    else if (arg == "--embed-device-resident") a.embed_device_resident = NextServerArg(argc, argv, i, "--embed-device-resident");
    else if (arg == "--dflash") a.dflash = NextServerArg(argc, argv, i, "--dflash");
    else if (arg == "--dflash-k") a.dflash_k = ServerParseI64("--dflash-k", NextServerArg(argc, argv, i, "--dflash-k"));
    else if (arg == "--dflash-p-min") a.dflash_p_min = ServerParseFloat("--dflash-p-min", NextServerArg(argc, argv, i, "--dflash-p-min"));
    else if (arg == "--dflash-n-min") a.dflash_n_min = ServerParseI64("--dflash-n-min", NextServerArg(argc, argv, i, "--dflash-n-min"));
    else if (arg == "--vision") a.vision = NextServerArg(argc, argv, i, "--vision");
    else if (arg == "--image-max-pixels") a.image_max_pixels = ServerParseI64("--image-max-pixels", NextServerArg(argc, argv, i, "--image-max-pixels"));
    else if (arg == "--tp") a.tp = ServerParseInt("--tp", NextServerArg(argc, argv, i, "--tp"));
    else if (arg == "--tp-mode") { a.tp_mode = NextServerArg(argc, argv, i, "--tp-mode"); a.tp_options_given = true; }
    else if (arg == "--tp-devices") { a.tp_devices = ParseServerTpDevices(NextServerArg(argc, argv, i, "--tp-devices")); a.tp_options_given = true; }
    else if (arg == "--tp-rank") { a.tp_rank = ServerParseInt("--tp-rank", NextServerArg(argc, argv, i, "--tp-rank")); a.tp_options_given = true; tp_rank_given = true; }
    else if (arg == "--tp-ar-timeout-ms") { a.tp_ar_timeout_ms = ServerParseInt("--tp-ar-timeout-ms", NextServerArg(argc, argv, i, "--tp-ar-timeout-ms")); a.tp_options_given = true; }
    else if (arg == "--tp-ar-nb") { a.tp_ar_nb = ServerParseInt("--tp-ar-nb", NextServerArg(argc, argv, i, "--tp-ar-nb")); a.tp_options_given = true; }
    else if (arg == "--tp-ar-nb-large") { a.tp_ar_nb_large = ServerParseInt("--tp-ar-nb-large", NextServerArg(argc, argv, i, "--tp-ar-nb-large")); a.tp_options_given = true; }
    else if (arg == "--tp-submit-layers") { a.tp_submit_layers = ServerParseInt("--tp-submit-layers", NextServerArg(argc, argv, i, "--tp-submit-layers")); a.tp_options_given = true; tp_submit_given = true; }
    else if (arg == "--tp-max-inflight") { a.tp_max_inflight = ServerParseInt("--tp-max-inflight", NextServerArg(argc, argv, i, "--tp-max-inflight")); a.tp_options_given = true; tp_inflight_given = true; }
    else if (arg == "--help" || arg == "-h") throw ServerUsageError("help requested");
    else throw ServerUsageError("unrecognized argument: " + arg);
  }
  if (a.model_path.empty()) throw ServerUsageError("--model is required");
  if (a.port <= 0 || a.port > 65535) throw ServerUsageError("--port must be in 1..65535");
  if (a.max_ctx <= 0) throw ServerUsageError("--max-ctx must be > 0");
  if (a.max_tokens_default < 0) throw ServerUsageError("--max-tokens-default must be >= 0");
  if (a.max_queue <= 0) throw ServerUsageError("--max-queue must be > 0");
  if (a.layers < -1) throw ServerUsageError("--layers must be >= 0 (or omitted for the full model)");
  if (a.default_top_p < 0.0f || a.default_top_p > 1.0f) throw ServerUsageError("--default-top-p must be in [0, 1]");
  if (a.default_min_p < 0.0f || a.default_min_p > 1.0f) throw ServerUsageError("--default-min-p must be in [0, 1]");
  if (a.default_top_k < 0) throw ServerUsageError("--default-top-k must be >= 0");
  if (a.log_level != "debug" && a.log_level != "info" && a.log_level != "warn" && a.log_level != "error") {
    throw ServerUsageError("--log-level must be one of debug|info|warn|error");
  }
  if (a.mtp < 0 || a.mtp > kMaxMtpDraftK) {
    throw ServerUsageError("--mtp must be in [0, " + std::to_string(kMaxMtpDraftK) +
                            "] (Model::VerifyWindow batches mtp+1 candidates through a <=64-row "
                            "chunk)");
  }
  if (a.mtp_head_layout != "bf16" && a.mtp_head_layout != "layout") {
    throw ServerUsageError("--mtp-head-layout must be 'bf16' or 'layout'");
  }
  if (a.mtp_draft_head != "reduced" && a.mtp_draft_head != "full") {
    throw ServerUsageError("--mtp-draft-head must be 'reduced' or 'full'");
  }
  if (a.embed_device_resident != "on" && a.embed_device_resident != "off") {
    throw ServerUsageError("--embed-device-resident must be 'on' or 'off'");
  }
  if (a.vision != "auto" && a.vision != "on" && a.vision != "off") {
    throw ServerUsageError("--vision must be 'auto', 'on' or 'off'");
  }
  if (a.image_max_pixels != 0 && a.image_max_pixels < 1024) {
    throw ServerUsageError("--image-max-pixels must be 0 (the checkpoint's own ceiling) or >= 1024");
  }
  if (!a.dflash.empty() && a.mtp > 0) {
    throw ServerUsageError("--dflash and --mtp are mutually exclusive (docs/dflash2.md: DFlash2 "
                            "and MTP are separate self-speculation families, not combinable)");
  }
  if (!a.dflash.empty() && (a.dflash_k < 1 || a.dflash_k > 7)) {
    throw ServerUsageError("--dflash-k must be in [1, 7] (DFlash2's block is 8 wide: anchor + up "
                            "to block_size-1 drafted tokens)");
  }
  // ---- tensor parallel (docs/tp.md 9.1): src/cli/cli_args.h's rules, minus --profile* (the server
  // has no profiling flags) ------------------------------------------------------------------------
  if (a.tp != 1 && a.tp != 2) throw ServerUsageError("--tp must be 1 or 2");
  if (a.tp == 1 && a.tp_options_given) {
    throw ServerUsageError("--tp-mode/--tp-devices/--tp-rank/--tp-ar-*/--tp-submit-layers/--tp-max-inflight need --tp 2");
  }
  if (a.tp == 2) {
    if (a.tp_mode != "real" && a.tp_mode != "emulate" && a.tp_mode != "noop") {
      throw ServerUsageError("--tp-mode must be 'real', 'emulate' or 'noop'");
    }
    if (tp_rank_given && a.tp_mode != "noop") throw ServerUsageError("--tp-rank needs --tp-mode noop");
    if (a.tp_rank < 0 || a.tp_rank > 1) throw ServerUsageError("--tp-rank must be 0 or 1");
    if (a.tp_ar_timeout_ms < 10 || a.tp_ar_timeout_ms > 1500) {
      throw ServerUsageError("--tp-ar-timeout-ms must be in [10, 1500] (the Windows TDR limit is 2 s)");
    }
    if (a.tp_ar_nb < 1 || a.tp_ar_nb > 64 || a.tp_ar_nb_large < 1 || a.tp_ar_nb_large > 64) {
      throw ServerUsageError("--tp-ar-nb and --tp-ar-nb-large must be in [1, 64]");
    }
    // -1 is the "not given" marker, so a GIVEN value is checked against [0, 64] including its sign.
    if ((tp_submit_given && (a.tp_submit_layers < 0 || a.tp_submit_layers > 64)) ||
        (tp_inflight_given && (a.tp_max_inflight < 0 || a.tp_max_inflight > 64))) {
      throw ServerUsageError("--tp-submit-layers and --tp-max-inflight must be in [0, 64]");
    }
    if (a.mtp > kMaxMtpDraftKTp) {
      throw ServerUsageError("--mtp must be in [0, " + std::to_string(kMaxMtpDraftKTp) +
                              "] with --tp 2 (a verify window is not split into device-0 submission units, "
                              "so it stays at most 8 rows; docs/tp.md Appendix B N80)");
    }
  }
  return a;
}

}  // namespace r4dx::server
