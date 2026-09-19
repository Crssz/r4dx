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
  CHECK(a.max_ctx == 131072);
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

  if (g_failures > 0) {
    std::fprintf(stderr, "%d check(s) failed\n", g_failures);
    return 1;
  }
  std::fprintf(stderr, "all cli arg-parsing checks passed\n");
  return 0;
}
