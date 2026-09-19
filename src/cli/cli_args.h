// r4dx::cli::CliArgs / ParseArgs -- argument parsing for r4dx-cli, split into its own header (out
// of main.cpp) purely so tests/cli/test_args.cpp can exercise it without linking a HIP-dependent
// executable. Header-only: this is a couple dozen lines of string parsing, not worth a .cpp.
#pragma once

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace r4dx::cli {

struct CliArgs {
  std::string model_path;
  std::string layout = "bf16";
  std::string tokenizer_dir = "C:\\AI\\models\\Qwen3.8-27B";
  std::string prompt;
  bool chat = false;
  std::string system_prompt;
  bool thinking = false;
  int64_t max_tokens = 128;
  float temperature = 1.0f;
  int top_k = 0;
  float top_p = 1.0f;
  float min_p = 0.0f;
  uint64_t seed = 0;
  int64_t max_ctx = 131072;
  bool stats = false;
  // tools/profile pass (2026-09-19): profiles exactly ONE decode step (Model::DecodeStepProfiled,
  // hipEvent-timed per op-family) right after the prompt's prefill, prints a table to stderr, then
  // continues generation as normal -- see main.cpp's RunTurn and docs/perf.md's profile table.
  bool profile = false;
  // Milestone 3 profiling pass (docs/r9700.md R5/Q2): which GENERATED decode step (1-indexed) to
  // profile when --profile is set. Default 32 -- Q2's own finding was that profiling the FIRST
  // generated token overstates the step by a near-constant +5.4 +/- 0.5 ms of first-token/warmup
  // effects vs the steady-state throughput-table step; main.cpp's RunTurn now runs
  // (profile_token - 1) plain DecodeStepGreedy warmup steps before calling DecodeStepProfiled on
  // step `profile_token`, so this flag lets a caller pick a different steady-state token if needed
  // (e.g. --profile-token 1 reproduces the OLD, first-token behavior for comparison).
  int64_t profile_token = 32;
  // Milestone 3 profiling pass (docs/r9700.md R5/Q7): profile PREFILL instead of decode --
  // Model::PrefillProfiled runs the whole prompt (chunked at max_chunk, today 64) with per-kernel
  // hipEvent spans, prints the per-op-family table (same "gemm:" GEMM-share split as --profile) to
  // stderr, and returns without generating anything, same "standalone diagnostic" contract as
  // --profile. Combine with a long --prompt (docs/r9700.md Q7 asks for ~1024 tokens) to get a
  // meaningful multi-chunk picture; mutually exclusive with --profile (checked below).
  bool profile_prefill = false;
  // MTP self-speculative decode (docs/mtp.md): draft this many tokens per step via the container's
  // mtp.* head, verify them against the real model in one batched call. Defaults to 0 (disabled --
  // r4dx::model::Model::DecodeStepGreedy/DecodeStep, byte-for-byte unchanged from pre-MTP
  // behavior): most containers on disk (including every one Milestone 1/the perf pass produced)
  // have no mtp.* weights, so a nonzero default would break --model pointed at any of them; the
  // task's own "K default 3" is the recommended value to pass explicitly once a --model container
  // was converted with --mtp on, not this flag's own default. Only used by --temperature 0
  // (greedy) generation -- see main.cpp's RunTurn.
  int64_t mtp = 0;
  // MTP head layout (docs/mtp.md "MTP head layout"): "layout" (default, measured faster in 23/24
  // K x layout configurations with no acceptance-rate cost -- see docs/mtp.md's table) loads the
  // MTP head's four quantized linears (mtp.attn.qg/o, mtp.mlp.gate_up/down) in the same --layout
  // as the body; "bf16" instead forces them from the container's bf16 tensors regardless of
  // --layout (~0.5 GB extra VRAM, the exact-arithmetic form). No effect when --mtp is 0 or the
  // container has no mtp.* weights.
  std::string mtp_head_layout = "layout";
  // Device-resident embedding gather (docs/mtp.md "device-resident draft loop", model.h's
  // ModelOptions::embed_device_resident): mirrors text.embed_tokens into VRAM (~2.37-2.54 GiB
  // depending on vocab/hidden) so decode/draft gathers on-device instead of a host memcpy+H2D per
  // step. Default true (matches ModelOptions' own default). This flag is the escape hatch
  // ModelOptions::embed_device_resident's own doc comment already promised but that no CLI/server
  // flag actually implemented until this fix (review finding, 2026-09-20) -- set to "off" to force
  // the host-gather path instead, e.g. to reclaim that VRAM for KV cache on a constrained run.
  std::string embed_device_resident = "on";
};

// Thrown for a malformed/incomplete argument list (missing required flag, unrecognized flag, a
// value that fails to parse, or --prompt combined with --chat) -- ParseArgs never calls
// std::exit() itself, so it is safely unit-testable (see tests/cli/test_args.cpp) as well as
// callable from main() (which catches this and exits(2) with usage text).
struct CliUsageError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

inline std::string CliUsageText(const char* argv0) {
  return std::string("usage: ") + argv0 +
         " --model <container.r4dx> --layout {mxfp4|w4a16|w4a8|bf16} "
         "(--prompt \"...\" | --chat) [--tokenizer-dir <dir>] [--system \"...\"] "
         "[--think {on|off}] [--max-tokens N] [--temperature F] [--top-k N] [--top-p F] "
         "[--min-p F] [--seed N] [--max-ctx N] [--stats] [--profile] [--profile-token N] "
         "[--profile-prefill] [--mtp N] [--mtp-head-layout {bf16|layout}] "
         "[--embed-device-resident {on|off}]";
}

inline std::string NextCliArg(int argc, char** argv, int& i, const char* flag) {
  if (i + 1 >= argc) throw CliUsageError(std::string(flag) + " requires a value");
  return argv[++i];
}

// std::stoll/stof/stoull throw std::invalid_argument (not a number) / std::out_of_range (doesn't
// fit the target type) -- neither is a CliUsageError, so an unparseable value (e.g. "--max-tokens
// abc") would otherwise fall straight through ParseArgs uncaught, past main()'s CliUsageError
// handler, to the generic top-level catch: exit code 1 with a raw "invalid stoll argument" message
// and no usage text, contradicting this header's own doc comment above (CliUsageError is supposed
// to cover exactly this case). These wrappers convert both into CliUsageError so every malformed
// flag value goes through the same usage-text path.
template <typename T, typename ParseFn>
T ParseNumber(const std::string& flag, const std::string& value, ParseFn parse) {
  try {
    return parse(value);
  } catch (const std::invalid_argument&) {
    throw CliUsageError(flag + " expects a number, got '" + value + "'");
  } catch (const std::out_of_range&) {
    throw CliUsageError(flag + " value out of range: '" + value + "'");
  }
}

inline int64_t ParseI64(const std::string& flag, const std::string& value) {
  return ParseNumber<int64_t>(flag, value, [](const std::string& v) { return std::stoll(v); });
}
inline int ParseInt(const std::string& flag, const std::string& value) {
  return ParseNumber<int>(flag, value, [](const std::string& v) { return std::stoi(v); });
}
inline float ParseFloat(const std::string& flag, const std::string& value) {
  return ParseNumber<float>(flag, value, [](const std::string& v) { return std::stof(v); });
}
inline uint64_t ParseU64(const std::string& flag, const std::string& value) {
  return ParseNumber<uint64_t>(
      flag, value, [](const std::string& v) { return static_cast<uint64_t>(std::stoull(v)); });
}

inline CliArgs ParseArgs(int argc, char** argv) {
  CliArgs a;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--model") a.model_path = NextCliArg(argc, argv, i, "--model");
    else if (arg == "--layout") a.layout = NextCliArg(argc, argv, i, "--layout");
    else if (arg == "--tokenizer-dir") a.tokenizer_dir = NextCliArg(argc, argv, i, "--tokenizer-dir");
    else if (arg == "--prompt") a.prompt = NextCliArg(argc, argv, i, "--prompt");
    else if (arg == "--chat") a.chat = true;
    else if (arg == "--system") a.system_prompt = NextCliArg(argc, argv, i, "--system");
    else if (arg == "--think") a.thinking = NextCliArg(argc, argv, i, "--think") == "on";
    else if (arg == "--max-tokens") a.max_tokens = ParseI64("--max-tokens", NextCliArg(argc, argv, i, "--max-tokens"));
    else if (arg == "--temperature") a.temperature = ParseFloat("--temperature", NextCliArg(argc, argv, i, "--temperature"));
    else if (arg == "--top-k") a.top_k = ParseInt("--top-k", NextCliArg(argc, argv, i, "--top-k"));
    else if (arg == "--top-p") a.top_p = ParseFloat("--top-p", NextCliArg(argc, argv, i, "--top-p"));
    else if (arg == "--min-p") a.min_p = ParseFloat("--min-p", NextCliArg(argc, argv, i, "--min-p"));
    else if (arg == "--seed") a.seed = ParseU64("--seed", NextCliArg(argc, argv, i, "--seed"));
    else if (arg == "--max-ctx") a.max_ctx = ParseI64("--max-ctx", NextCliArg(argc, argv, i, "--max-ctx"));
    else if (arg == "--stats") a.stats = true;
    else if (arg == "--profile") a.profile = true;
    else if (arg == "--profile-token") a.profile_token = ParseI64("--profile-token", NextCliArg(argc, argv, i, "--profile-token"));
    else if (arg == "--profile-prefill") a.profile_prefill = true;
    else if (arg == "--mtp") a.mtp = ParseI64("--mtp", NextCliArg(argc, argv, i, "--mtp"));
    else if (arg == "--mtp-head-layout") a.mtp_head_layout = NextCliArg(argc, argv, i, "--mtp-head-layout");
    else if (arg == "--embed-device-resident") a.embed_device_resident = NextCliArg(argc, argv, i, "--embed-device-resident");
    else if (arg == "--help" || arg == "-h") throw CliUsageError("help requested");
    else throw CliUsageError("unrecognized argument: " + arg);
  }
  if (a.model_path.empty()) throw CliUsageError("--model is required");
  if (a.prompt.empty() && !a.chat) throw CliUsageError("one of --prompt or --chat is required");
  if (!a.prompt.empty() && a.chat) throw CliUsageError("--prompt and --chat are mutually exclusive");
  // Reject nonsensical numeric values outright rather than passing them to the model, where they
  // would surface (if at all) as a confusing HIP/kernel-level failure far from the actual mistake.
  if (a.max_tokens < 0) throw CliUsageError("--max-tokens must be >= 0");
  if (a.top_p < 0.0f || a.top_p > 1.0f) throw CliUsageError("--top-p must be in [0, 1]");
  if (a.min_p < 0.0f || a.min_p > 1.0f) throw CliUsageError("--min-p must be in [0, 1]");
  if (a.max_ctx <= 0) throw CliUsageError("--max-ctx must be > 0");
  if (a.top_k < 0) throw CliUsageError("--top-k must be >= 0");
  if (a.mtp < 0) throw CliUsageError("--mtp must be >= 0");
  if (a.mtp_head_layout != "bf16" && a.mtp_head_layout != "layout") {
    throw CliUsageError("--mtp-head-layout must be 'bf16' or 'layout'");
  }
  if (a.profile_token < 1) throw CliUsageError("--profile-token must be >= 1");
  if (a.embed_device_resident != "on" && a.embed_device_resident != "off") {
    throw CliUsageError("--embed-device-resident must be 'on' or 'off'");
  }
  if (a.profile && a.profile_prefill) {
    throw CliUsageError("--profile and --profile-prefill are mutually exclusive");
  }
  return a;
}

}  // namespace r4dx::cli
