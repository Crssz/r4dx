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
  CHECK(a.max_ctx == 262144);  // docs/r9700.md R13: raised from 131072, the model's real ceiling
  CHECK(a.max_tokens_default == 128);
  CHECK(a.max_queue == 16);
  CHECK(a.layers == -1);
  CHECK(!a.think);
  CHECK(a.default_temperature == 1.0f);
  CHECK(a.default_top_p == 1.0f);
  CHECK(a.default_top_k == 0);
  CHECK(a.default_min_p == 0.0f);
  CHECK(a.log_level == "info");
  CHECK(a.mtp_head_layout == "layout");
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

// --mtp-head-layout (docs/mtp.md / docs/status.md Known-gaps item: server-side passthrough for the
// flag src/cli/cli_args.h already had): mirrors that CLI flag exactly -- defaults to "layout",
// accepts "bf16", rejects anything else.
void TestMtpHeadLayoutFlag() {
  {
    std::vector<std::string> storage = {"r4dx-server", "--model", "m.r4dx"};
    auto argv = ToArgv(storage);
    const auto a = r4dx::server::ParseServerArgs(static_cast<int>(argv.size()), argv.data());
    CHECK(a.mtp_head_layout == "layout");
  }
  {
    std::vector<std::string> storage = {"r4dx-server", "--model", "m.r4dx", "--mtp", "3",
                                         "--mtp-head-layout", "bf16"};
    auto argv = ToArgv(storage);
    const auto a = r4dx::server::ParseServerArgs(static_cast<int>(argv.size()), argv.data());
    CHECK(a.mtp_head_layout == "bf16");
  }
  {
    std::vector<std::string> storage = {"r4dx-server", "--model", "m.r4dx",
                                         "--mtp-head-layout", "bogus"};
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

// --mtp-draft-head (docs/r9700.md R9, "reduced-vocab draft head"): mirrors src/cli/cli_args.h's own
// flag -- defaults to "reduced", accepts "full", rejects anything else.
void TestMtpDraftHeadFlag() {
  {
    std::vector<std::string> storage = {"r4dx-server", "--model", "m.r4dx"};
    auto argv = ToArgv(storage);
    const auto a = r4dx::server::ParseServerArgs(static_cast<int>(argv.size()), argv.data());
    CHECK(a.mtp_draft_head == "reduced");
  }
  {
    std::vector<std::string> storage = {"r4dx-server", "--model", "m.r4dx", "--mtp", "3",
                                         "--mtp-draft-head", "full"};
    auto argv = ToArgv(storage);
    const auto a = r4dx::server::ParseServerArgs(static_cast<int>(argv.size()), argv.data());
    CHECK(a.mtp_draft_head == "full");
  }
  {
    std::vector<std::string> storage = {"r4dx-server", "--model", "m.r4dx",
                                         "--mtp-draft-head", "bogus"};
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

// --mtp upper bound (review finding, 2026-09-20): mirrors src/cli/cli_args.h's own kMaxMtpDraftK
// check -- Model::VerifyWindow requires mtp+1 candidates to fit in a <=64-row chunk, so 63 is the
// real ceiling; previously only `>= 0` was checked here.
void TestMtpUpperBound() {
  {
    std::vector<std::string> storage = {"r4dx-server", "--model", "m.r4dx", "--mtp", "63"};
    auto argv = ToArgv(storage);
    const auto a = r4dx::server::ParseServerArgs(static_cast<int>(argv.size()), argv.data());
    CHECK(a.mtp == 63);
  }
  {
    std::vector<std::string> storage = {"r4dx-server", "--model", "m.r4dx", "--mtp", "64"};
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
  TestMtpHeadLayoutFlag();
  TestMtpDraftHeadFlag();
  TestEmbedDeviceResidentFlag();
  TestMtpUpperBound();

  if (g_failures > 0) {
    std::fprintf(stderr, "%d check(s) failed\n", g_failures);
    return 1;
  }
  std::fprintf(stderr, "all server arg-parsing checks passed\n");
  return 0;
}
