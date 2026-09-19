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
         "[--min-p F] [--seed N] [--max-ctx N] [--stats]";
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
  return a;
}

}  // namespace r4dx::cli
