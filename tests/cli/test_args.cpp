// tests/cli/test_args.cpp -- pure CPU unit test for r4dx::cli::ParseArgs (src/cli/cli_args.h).
// No HIP device needed (registered as a plain CTest COMMAND, not ENVIRONMENT-gated to a device).
#include <cstdio>
#include <cstring>
#include <vector>

#include "cli_args.h"

namespace {

int g_failures = 0;

#define CHECK(cond)                                                              \
  do {                                                                           \
    if (!(cond)) {                                                               \
      std::fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__,    \
                   #cond);                                                       \
      ++g_failures;                                                              \
    }                                                                            \
  } while (0)

std::vector<char*> ToArgv(std::vector<std::string>& storage) {
  std::vector<char*> argv;
  for (auto& s : storage) argv.push_back(s.data());
  return argv;
}

void TestMinimalPrompt() {
  std::vector<std::string> storage = {"r4dx-cli", "--model", "model.r4dx", "--layout", "mxfp4",
                                       "--prompt", "hello"};
  auto argv = ToArgv(storage);
  const auto a = r4dx::cli::ParseArgs(static_cast<int>(argv.size()), argv.data());
  CHECK(a.model_path == "model.r4dx");
  CHECK(a.layout == "mxfp4");
  CHECK(a.prompt == "hello");
  CHECK(!a.chat);
  CHECK(a.max_tokens == 128);
  CHECK(a.max_ctx == 262144);  // docs/r9700.md R13: raised from 131072, the model's real ceiling
  CHECK(!a.thinking);
}

void TestChatAndOptions() {
  std::vector<std::string> storage = {
      "r4dx-cli", "--model", "m.r4dx", "--layout", "bf16", "--chat", "--system", "be terse",
      "--think", "on", "--max-tokens", "64", "--temperature", "0.7", "--top-k", "40", "--top-p",
      "0.9", "--min-p", "0.05", "--seed", "123", "--max-ctx", "4096", "--stats"};
  auto argv = ToArgv(storage);
  const auto a = r4dx::cli::ParseArgs(static_cast<int>(argv.size()), argv.data());
  CHECK(a.chat);
  CHECK(a.prompt.empty());
  CHECK(a.system_prompt == "be terse");
  CHECK(a.thinking == true);
  CHECK(a.max_tokens == 64);
  CHECK(a.temperature == 0.7f);
  CHECK(a.top_k == 40);
  CHECK(a.top_p == 0.9f);
  CHECK(a.min_p == 0.05f);
  CHECK(a.seed == 123u);
  CHECK(a.max_ctx == 4096);
  CHECK(a.stats);
}

void TestMissingModelThrows() {
  std::vector<std::string> storage = {"r4dx-cli", "--layout", "bf16", "--prompt", "hi"};
  auto argv = ToArgv(storage);
  bool threw = false;
  try {
    r4dx::cli::ParseArgs(static_cast<int>(argv.size()), argv.data());
  } catch (const r4dx::cli::CliUsageError&) {
    threw = true;
  }
  CHECK(threw);
}

void TestPromptAndChatMutuallyExclusive() {
  std::vector<std::string> storage = {"r4dx-cli", "--model", "m.r4dx", "--layout", "bf16",
                                       "--prompt", "hi", "--chat"};
  auto argv = ToArgv(storage);
  bool threw = false;
  try {
    r4dx::cli::ParseArgs(static_cast<int>(argv.size()), argv.data());
  } catch (const r4dx::cli::CliUsageError&) {
    threw = true;
  }
  CHECK(threw);
}

void TestNeitherPromptNorChatThrows() {
  std::vector<std::string> storage = {"r4dx-cli", "--model", "m.r4dx", "--layout", "bf16"};
  auto argv = ToArgv(storage);
  bool threw = false;
  try {
    r4dx::cli::ParseArgs(static_cast<int>(argv.size()), argv.data());
  } catch (const r4dx::cli::CliUsageError&) {
    threw = true;
  }
  CHECK(threw);
}

void TestUnrecognizedFlagThrows() {
  std::vector<std::string> storage = {"r4dx-cli", "--model", "m.r4dx", "--layout", "bf16",
                                       "--prompt", "hi", "--bogus-flag"};
  auto argv = ToArgv(storage);
  bool threw = false;
  try {
    r4dx::cli::ParseArgs(static_cast<int>(argv.size()), argv.data());
  } catch (const r4dx::cli::CliUsageError&) {
    threw = true;
  }
  CHECK(threw);
}

void TestMissingValueThrows() {
  std::vector<std::string> storage = {"r4dx-cli", "--model", "m.r4dx", "--layout", "bf16",
                                       "--prompt", "hi", "--max-tokens"};
  auto argv = ToArgv(storage);
  bool threw = false;
  try {
    r4dx::cli::ParseArgs(static_cast<int>(argv.size()), argv.data());
  } catch (const r4dx::cli::CliUsageError&) {
    threw = true;
  }
  CHECK(threw);
}

// A value that fails to parse (std::stoll/stof/stoull throw std::invalid_argument/
// std::out_of_range, NOT CliUsageError) must still surface as CliUsageError -- ParseArgs's own
// doc comment promises this, and main() only catches CliUsageError (see cli_args.h's ParseNumber
// comment for the bug this guards against).
void TestUnparseableNumberThrowsCliUsageError() {
  const std::vector<std::vector<std::string>> cases = {
      {"r4dx-cli", "--model", "m.r4dx", "--layout", "bf16", "--prompt", "hi", "--max-tokens",
       "abc"},
      {"r4dx-cli", "--model", "m.r4dx", "--layout", "bf16", "--prompt", "hi", "--temperature",
       "hot"},
      {"r4dx-cli", "--model", "m.r4dx", "--layout", "bf16", "--prompt", "hi", "--top-k", "many"},
      {"r4dx-cli", "--model", "m.r4dx", "--layout", "bf16", "--prompt", "hi", "--max-tokens",
       "99999999999999999999"},  // out_of_range for int64_t
  };
  for (auto storage : cases) {
    auto argv = ToArgv(storage);
    bool threw_usage_error = false;
    try {
      r4dx::cli::ParseArgs(static_cast<int>(argv.size()), argv.data());
    } catch (const r4dx::cli::CliUsageError&) {
      threw_usage_error = true;
    } catch (...) {
      // Any other exception type (std::invalid_argument/std::out_of_range escaping uncaught) is
      // exactly the bug this test guards against -- leave threw_usage_error false so CHECK fails.
    }
    CHECK(threw_usage_error);
  }
}

void TestNonsensicalValuesThrow() {
  const std::vector<std::vector<std::string>> cases = {
      {"r4dx-cli", "--model", "m.r4dx", "--layout", "bf16", "--prompt", "hi", "--max-tokens",
       "-1"},
      {"r4dx-cli", "--model", "m.r4dx", "--layout", "bf16", "--prompt", "hi", "--top-p", "1.5"},
      {"r4dx-cli", "--model", "m.r4dx", "--layout", "bf16", "--prompt", "hi", "--min-p", "-0.1"},
      {"r4dx-cli", "--model", "m.r4dx", "--layout", "bf16", "--prompt", "hi", "--max-ctx", "0"},
      {"r4dx-cli", "--model", "m.r4dx", "--layout", "bf16", "--prompt", "hi", "--top-k", "-5"},
  };
  for (auto storage : cases) {
    auto argv = ToArgv(storage);
    bool threw = false;
    try {
      r4dx::cli::ParseArgs(static_cast<int>(argv.size()), argv.data());
    } catch (const r4dx::cli::CliUsageError&) {
      threw = true;
    }
    CHECK(threw);
  }
}

// --embed-device-resident (review finding, 2026-09-20): ModelOptions::embed_device_resident's own
// doc comment promised a CLI escape hatch ("set false to force host-only gather") that no flag
// actually implemented -- this checks the flag now exists, defaults to "on", and rejects anything
// other than "on"/"off".
void TestEmbedDeviceResidentFlag() {
  {
    std::vector<std::string> storage = {"r4dx-cli", "--model", "m.r4dx", "--layout", "bf16",
                                         "--prompt", "hi"};
    auto argv = ToArgv(storage);
    const auto a = r4dx::cli::ParseArgs(static_cast<int>(argv.size()), argv.data());
    CHECK(a.embed_device_resident == "on");  // default
  }
  {
    std::vector<std::string> storage = {"r4dx-cli",  "--model", "m.r4dx", "--layout", "bf16",
                                         "--prompt",  "hi",      "--embed-device-resident", "off"};
    auto argv = ToArgv(storage);
    const auto a = r4dx::cli::ParseArgs(static_cast<int>(argv.size()), argv.data());
    CHECK(a.embed_device_resident == "off");
  }
  {
    std::vector<std::string> storage = {"r4dx-cli",  "--model", "m.r4dx", "--layout", "bf16",
                                         "--prompt",  "hi",      "--embed-device-resident", "bogus"};
    auto argv = ToArgv(storage);
    bool threw = false;
    try {
      r4dx::cli::ParseArgs(static_cast<int>(argv.size()), argv.data());
    } catch (const r4dx::cli::CliUsageError&) {
      threw = true;
    }
    CHECK(threw);
  }
}

// --mtp-draft-head (docs/r9700.md R9, "reduced-vocab draft head"): defaults to "reduced", accepts
// "full", rejects anything else -- mirrors TestMtpHeadLayoutFlag's shape for the sibling flag.
void TestMtpDraftHeadFlag() {
  {
    std::vector<std::string> storage = {"r4dx-cli", "--model", "m.r4dx", "--layout", "bf16",
                                         "--prompt", "hi"};
    auto argv = ToArgv(storage);
    const auto a = r4dx::cli::ParseArgs(static_cast<int>(argv.size()), argv.data());
    CHECK(a.mtp_draft_head == "reduced");  // default
  }
  {
    std::vector<std::string> storage = {"r4dx-cli",  "--model", "m.r4dx", "--layout", "bf16",
                                         "--prompt",  "hi",      "--mtp", "3",
                                         "--mtp-draft-head", "full"};
    auto argv = ToArgv(storage);
    const auto a = r4dx::cli::ParseArgs(static_cast<int>(argv.size()), argv.data());
    CHECK(a.mtp_draft_head == "full");
  }
  {
    std::vector<std::string> storage = {"r4dx-cli",  "--model", "m.r4dx", "--layout", "bf16",
                                         "--prompt",  "hi",      "--mtp-draft-head", "bogus"};
    auto argv = ToArgv(storage);
    bool threw = false;
    try {
      r4dx::cli::ParseArgs(static_cast<int>(argv.size()), argv.data());
    } catch (const r4dx::cli::CliUsageError&) {
      threw = true;
    }
    CHECK(threw);
  }
}

// --mtp upper bound (review finding, 2026-09-20): Model::VerifyWindow requires mtp+1 candidates
// to fit in a <=64-row chunk, so the real ceiling is 63 -- previously only `>= 0` was checked, so
// e.g. --mtp 64 was accepted by the parser and only failed much later, deep inside Model::Load,
// with an uninformative "bad allocation".
void TestMtpUpperBound() {
  {
    std::vector<std::string> storage = {"r4dx-cli", "--model", "m.r4dx", "--layout", "bf16",
                                         "--prompt", "hi",      "--mtp", "63"};
    auto argv = ToArgv(storage);
    const auto a = r4dx::cli::ParseArgs(static_cast<int>(argv.size()), argv.data());
    CHECK(a.mtp == 63);  // the ceiling itself must still be accepted
  }
  {
    std::vector<std::string> storage = {"r4dx-cli", "--model", "m.r4dx", "--layout", "bf16",
                                         "--prompt", "hi",      "--mtp", "64"};
    auto argv = ToArgv(storage);
    bool threw = false;
    try {
      r4dx::cli::ParseArgs(static_cast<int>(argv.size()), argv.data());
    } catch (const r4dx::cli::CliUsageError&) {
      threw = true;
    }
    CHECK(threw);
  }
}

}  // namespace

int main() {
  TestMinimalPrompt();
  TestChatAndOptions();
  TestMissingModelThrows();
  TestPromptAndChatMutuallyExclusive();
  TestNeitherPromptNorChatThrows();
  TestUnrecognizedFlagThrows();
  TestMissingValueThrows();
  TestUnparseableNumberThrowsCliUsageError();
  TestNonsensicalValuesThrow();
  TestEmbedDeviceResidentFlag();
  TestMtpDraftHeadFlag();
  TestMtpUpperBound();

  if (g_failures > 0) {
    std::fprintf(stderr, "%d check(s) failed\n", g_failures);
    return 1;
  }
  std::fprintf(stderr, "all cli arg-parsing checks passed\n");
  return 0;
}
