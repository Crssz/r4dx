// tests/server/test_server_args.cpp -- pure CPU unit test for r4dx::server::ParseServerArgs
// (src/server/server_args.h). No HIP device needed. Mirrors tests/cli/test_args.cpp's structure.
#include <cstdio>
#include <string>
#include <vector>

#include "server_args.h"

namespace {

int g_failures = 0;

#define CHECK(cond)                                                           \
  do {                                                                        \
    if (!(cond)) {                                                            \
      std::fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, \
                   #cond);                                                    \
      ++g_failures;                                                           \
    }                                                                         \
  } while (0)

std::vector<char*> ToArgv(std::vector<std::string>& storage) {
  std::vector<char*> argv;
  for (auto& s : storage) argv.push_back(s.data());
  return argv;
}

void TestDefaults() {
  std::vector<std::string> storage = {"r4dx-server", "--model", "model.r4dx"};
  auto argv = ToArgv(storage);
  const auto a = r4dx::server::ParseServerArgs(static_cast<int>(argv.size()), argv.data());
  CHECK(a.model_path == "model.r4dx");
  CHECK(a.layout == "bf16");
  CHECK(a.host == "127.0.0.1");
  CHECK(a.port == 8080);
  CHECK(a.max_ctx == 131072);
  CHECK(a.max_tokens_default == 128);
  CHECK(a.max_queue == 16);
  CHECK(a.layers == -1);
  CHECK(!a.think);
  CHECK(a.default_temperature == 1.0f);
  CHECK(a.default_top_p == 1.0f);
  CHECK(a.default_top_k == 0);
  CHECK(a.default_min_p == 0.0f);
  CHECK(a.log_level == "info");
}

void TestAllFlags() {
  std::vector<std::string> storage = {
      "r4dx-server", "--model", "m.r4dx", "--layout", "w4a16", "--tokenizer-dir", "C:\\tok",
      "--host", "0.0.0.0", "--port", "9090", "--max-ctx", "4096", "--max-tokens-default", "64",
      "--max-queue", "4", "--layers", "4", "--think", "on", "--default-temperature", "0.7", "--default-top-p", "0.9",
      "--default-top-k", "40", "--default-min-p", "0.05", "--log-level", "debug"};
  auto argv = ToArgv(storage);
  const auto a = r4dx::server::ParseServerArgs(static_cast<int>(argv.size()), argv.data());
  CHECK(a.layout == "w4a16");
  CHECK(a.tokenizer_dir == "C:\\tok");
  CHECK(a.host == "0.0.0.0");
  CHECK(a.port == 9090);
  CHECK(a.max_ctx == 4096);
  CHECK(a.max_tokens_default == 64);
  CHECK(a.max_queue == 4);
  CHECK(a.layers == 4);
  CHECK(a.think);
  CHECK(a.default_temperature == 0.7f);
  CHECK(a.default_top_p == 0.9f);
  CHECK(a.default_top_k == 40);
  CHECK(a.default_min_p == 0.05f);
  CHECK(a.log_level == "debug");
}

void TestMissingModelThrows() {
  std::vector<std::string> storage = {"r4dx-server", "--port", "8080"};
  auto argv = ToArgv(storage);
  bool threw = false;
  try {
    r4dx::server::ParseServerArgs(static_cast<int>(argv.size()), argv.data());
  } catch (const r4dx::server::ServerUsageError&) {
    threw = true;
  }
  CHECK(threw);
}

void TestUnrecognizedFlagThrows() {
  std::vector<std::string> storage = {"r4dx-server", "--model", "m.r4dx", "--bogus"};
  auto argv = ToArgv(storage);
  bool threw = false;
  try {
    r4dx::server::ParseServerArgs(static_cast<int>(argv.size()), argv.data());
  } catch (const r4dx::server::ServerUsageError&) {
    threw = true;
  }
  CHECK(threw);
}

void TestNonsensicalValuesThrow() {
  const std::vector<std::vector<std::string>> cases = {
      {"r4dx-server", "--model", "m.r4dx", "--port", "0"},
      {"r4dx-server", "--model", "m.r4dx", "--port", "70000"},
      {"r4dx-server", "--model", "m.r4dx", "--max-ctx", "0"},
      {"r4dx-server", "--model", "m.r4dx", "--max-tokens-default", "-1"},
      {"r4dx-server", "--model", "m.r4dx", "--max-queue", "0"},
      {"r4dx-server", "--model", "m.r4dx", "--default-top-p", "1.5"},
      {"r4dx-server", "--model", "m.r4dx", "--default-min-p", "-0.1"},
      {"r4dx-server", "--model", "m.r4dx", "--default-top-k", "-5"},
      {"r4dx-server", "--model", "m.r4dx", "--log-level", "verbose"},
  };
  for (auto storage : cases) {
    auto argv = ToArgv(storage);
    bool threw = false;
    try {
      r4dx::server::ParseServerArgs(static_cast<int>(argv.size()), argv.data());
    } catch (const r4dx::server::ServerUsageError&) {
      threw = true;
    }
    CHECK(threw);
  }
}

void TestUnparseableNumberThrowsServerUsageError() {
  std::vector<std::string> storage = {"r4dx-server", "--model", "m.r4dx", "--port", "abc"};
  auto argv = ToArgv(storage);
  bool threw_usage_error = false;
  try {
    r4dx::server::ParseServerArgs(static_cast<int>(argv.size()), argv.data());
  } catch (const r4dx::server::ServerUsageError&) {
    threw_usage_error = true;
  } catch (...) {
  }
  CHECK(threw_usage_error);
}

// --embed-device-resident (review finding, 2026-09-20): mirrors src/cli/cli_args.h's own flag --
// defaults to "on", accepts "off", rejects anything else.
void TestEmbedDeviceResidentFlag() {
  {
    std::vector<std::string> storage = {"r4dx-server", "--model", "m.r4dx"};
    auto argv = ToArgv(storage);
    const auto a = r4dx::server::ParseServerArgs(static_cast<int>(argv.size()), argv.data());
    CHECK(a.embed_device_resident == "on");
  }
  {
    std::vector<std::string> storage = {"r4dx-server", "--model", "m.r4dx",
                                         "--embed-device-resident", "off"};
    auto argv = ToArgv(storage);
    const auto a = r4dx::server::ParseServerArgs(static_cast<int>(argv.size()), argv.data());
    CHECK(a.embed_device_resident == "off");
  }
  {
    std::vector<std::string> storage = {"r4dx-server", "--model", "m.r4dx",
                                         "--embed-device-resident", "bogus"};
    auto argv = ToArgv(storage);
    bool threw = false;
    try {
      r4dx::server::ParseServerArgs(static_cast<int>(argv.size()), argv.data());
    } catch (const r4dx::server::ServerUsageError&) {
      threw = true;
    }
    CHECK(threw);
  }
}

}  // namespace

int main() {
  TestDefaults();
  TestAllFlags();
  TestMissingModelThrows();
  TestUnrecognizedFlagThrows();
  TestNonsensicalValuesThrow();
  TestUnparseableNumberThrowsServerUsageError();
  TestEmbedDeviceResidentFlag();

  if (g_failures > 0) {
    std::fprintf(stderr, "%d check(s) failed\n", g_failures);
    return 1;
  }
  std::fprintf(stderr, "all server arg-parsing checks passed\n");
  return 0;
}
