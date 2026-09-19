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

struct ServerArgs {
  std::string model_path;
  std::string layout = "bf16";
  std::string tokenizer_dir = "C:\\AI\\models\\Qwen3.8-27B";
  std::string host = "127.0.0.1";
  int port = 8080;
  int64_t max_ctx = 131072;
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
         "[--default-min-p F] [--log-level {debug|info|warn|error}]";
}

inline std::string NextServerArg(int argc, char** argv, int& i, const char* flag) {
  if (i + 1 >= argc) throw ServerUsageError(std::string(flag) + " requires a value");
  return argv[++i];
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
  return a;
}

}  // namespace r4dx::server
