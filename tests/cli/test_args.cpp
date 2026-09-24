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

// --mtp-draft-head (docs/r9700.md R9, "reduced-vocab draft head"): defaults to "full" (flipped
// 2026-09-20, Milestone 5 B2 item 8; matched-K=3 real-hardware re-measurement, review fix pass:
// full 68.20/68.64 tok/s vs reduced 53.59/54.17 tok/s, byte-identical output, see
// src/cli/cli_args.h's own comment), accepts "reduced", rejects anything else -- mirrors
// TestMtpHeadLayoutFlag's shape for the sibling flag.
void TestMtpDraftHeadFlag() {
  {
    std::vector<std::string> storage = {"r4dx-cli", "--model", "m.r4dx", "--layout", "bf16",
                                         "--prompt", "hi"};
    auto argv = ToArgv(storage);
    const auto a = r4dx::cli::ParseArgs(static_cast<int>(argv.size()), argv.data());
    CHECK(a.mtp_draft_head == "full");  // default
  }
  {
    std::vector<std::string> storage = {"r4dx-cli",  "--model", "m.r4dx", "--layout", "bf16",
                                         "--prompt",  "hi",      "--mtp", "3",
                                         "--mtp-draft-head", "reduced"};
    auto argv = ToArgv(storage);
    const auto a = r4dx::cli::ParseArgs(static_cast<int>(argv.size()), argv.data());
    CHECK(a.mtp_draft_head == "reduced");
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

// --dflash/--dflash-k/--dflash-p-min/--dflash-n-min (docs/dflash2.md, stage S3 item 1): empty
// --dflash disables DFlash2 entirely (defaults untouched); --dflash with --mtp>0 is an error;
// --dflash-k is bounded to [1,7] (DFlash2's block is 8 wide: anchor + up to block_size-1 drafted).
void TestDflashFlags() {
  {
    std::vector<std::string> storage = {"r4dx-cli", "--model", "m.r4dx", "--layout", "bf16",
                                         "--prompt", "hi"};
    auto argv = ToArgv(storage);
    const auto a = r4dx::cli::ParseArgs(static_cast<int>(argv.size()), argv.data());
    CHECK(a.dflash.empty());
    CHECK(a.dflash_k == 7);
    CHECK(a.dflash_p_min == 0.0f);
    CHECK(a.dflash_n_min == 0);
  }
  {
    std::vector<std::string> storage = {
        "r4dx-cli",     "--model",         "m.r4dx", "--layout", "w4a16",
        "--prompt",     "hi",              "--dflash", "draft.r4dx",
        "--dflash-k",   "5",               "--dflash-p-min", "0.3",
        "--dflash-n-min", "2"};
    auto argv = ToArgv(storage);
    const auto a = r4dx::cli::ParseArgs(static_cast<int>(argv.size()), argv.data());
    CHECK(a.dflash == "draft.r4dx");
    CHECK(a.dflash_k == 5);
    CHECK(a.dflash_p_min > 0.29f && a.dflash_p_min < 0.31f);
    CHECK(a.dflash_n_min == 2);
  }
  {  // --dflash with --mtp > 0 is an error
    std::vector<std::string> storage = {"r4dx-cli", "--model", "m.r4dx", "--layout", "bf16",
                                         "--prompt", "hi",      "--dflash", "draft.r4dx",
                                         "--mtp",    "3"};
    auto argv = ToArgv(storage);
    bool threw = false;
    try {
      r4dx::cli::ParseArgs(static_cast<int>(argv.size()), argv.data());
    } catch (const r4dx::cli::CliUsageError&) {
      threw = true;
    }
    CHECK(threw);
  }
  {  // --dflash-k out of [1,7] is an error, but only when --dflash is actually given
    std::vector<std::string> storage = {"r4dx-cli", "--model", "m.r4dx", "--layout", "bf16",
                                         "--prompt", "hi",      "--dflash", "draft.r4dx",
                                         "--dflash-k", "8"};
    auto argv = ToArgv(storage);
    bool threw = false;
    try {
      r4dx::cli::ParseArgs(static_cast<int>(argv.size()), argv.data());
    } catch (const r4dx::cli::CliUsageError&) {
      threw = true;
    }
    CHECK(threw);
  }
  {  // --dflash-k 0 is also out of range with --dflash set
    std::vector<std::string> storage = {"r4dx-cli", "--model", "m.r4dx", "--layout", "bf16",
                                         "--prompt", "hi",      "--dflash", "draft.r4dx",
                                         "--dflash-k", "0"};
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

// --vision / --image-max-pixels (docs/vision.md "Load policy" / "Large images").
void TestVisionFlags() {
  auto parse = [](std::vector<std::string> extra) {
    std::vector<std::string> storage = {"r4dx-cli", "--model", "m.r4dx", "--layout", "bf16",
                                         "--prompt", "hi"};
    storage.insert(storage.end(), extra.begin(), extra.end());
    auto argv = ToArgv(storage);
    return r4dx::cli::ParseArgs(static_cast<int>(argv.size()), argv.data());
  };
  auto rejects = [](std::vector<std::string> extra) {
    std::vector<std::string> storage = {"r4dx-cli", "--model", "m.r4dx", "--layout", "bf16",
                                         "--prompt", "hi"};
    storage.insert(storage.end(), extra.begin(), extra.end());
    auto argv = ToArgv(storage);
    try {
      r4dx::cli::ParseArgs(static_cast<int>(argv.size()), argv.data());
    } catch (const r4dx::cli::CliUsageError&) {
      return true;
    }
    return false;
  };

  {
    const auto a = parse({});
    CHECK(a.vision == "auto");                 // load iff the container has a tower
    CHECK(a.image_max_pixels == 1048576);      // 1024x1024, the measured default
  }
  CHECK(parse({"--vision", "on"}).vision == "on");
  CHECK(parse({"--vision", "off"}).vision == "off");
  CHECK(rejects({"--vision", "yes"}));
  CHECK(rejects({"--vision", ""}));

  // 0 means "the checkpoint's own ceiling", which is a legal, meaningful value -- not "unset".
  CHECK(parse({"--image-max-pixels", "0"}).image_max_pixels == 0);
  CHECK(parse({"--image-max-pixels", "4194304"}).image_max_pixels == 4194304);
  CHECK(rejects({"--image-max-pixels", "1023"}));  // below one merged token
  CHECK(rejects({"--image-max-pixels", "-1"}));
  CHECK(rejects({"--image-max-pixels", "lots"}));
}

// docs/vision.md stage 5 (user-facing wiring): --image is repeatable, unlike every other flag in
// this file, and attaches to whichever turn main.cpp's run_one_user_turn processes next -- see
// that file's own comment. Purely a parsing test (no container, no decode): main.cpp's own
// --image handling is exercised by tests/vision/tool_vision_chat and tools/server/smoke.ps1
// (server-side), since it needs a real vision-capable container.
void TestImageFlag() {
  {
    std::vector<std::string> storage = {"r4dx-cli", "--model", "m.r4dx", "--layout", "bf16",
                                         "--prompt", "describe this"};
    auto argv = ToArgv(storage);
    const auto a = r4dx::cli::ParseArgs(static_cast<int>(argv.size()), argv.data());
    CHECK(a.image_paths.empty());
  }
  {
    std::vector<std::string> storage = {"r4dx-cli",      "--model", "m.r4dx", "--layout", "bf16",
                                         "--prompt",      "describe these", "--image",
                                         "a.png",         "--image", "b.jpg"};
    auto argv = ToArgv(storage);
    const auto a = r4dx::cli::ParseArgs(static_cast<int>(argv.size()), argv.data());
    CHECK(a.image_paths.size() == 2);
    CHECK(a.image_paths[0] == "a.png");
    CHECK(a.image_paths[1] == "b.jpg");
  }
}

// --tp and --tp-* (docs/tp.md 9.1): defaults are TP=1 and change nothing; every --tp-* flag needs
// --tp 2; the permanent (--profile*) and staged (P2b: real mode, --mtp, --dflash, --vision on,
// --image) rejections are usage errors.
bool TpThrows(std::vector<std::string> extra) {
  std::vector<std::string> storage = {"r4dx-cli", "--model", "m.r4dx", "--layout", "w4a16", "--prompt", "hi"};
  storage.insert(storage.end(), extra.begin(), extra.end());
  auto argv = ToArgv(storage);
  try {
    r4dx::cli::ParseArgs(static_cast<int>(argv.size()), argv.data());
  } catch (const r4dx::cli::CliUsageError&) {
    return true;
  }
  return false;
}

void TestTpFlags() {
  {
    std::vector<std::string> storage = {"r4dx-cli", "--model", "m.r4dx", "--layout", "w4a16", "--prompt", "hi"};
    auto argv = ToArgv(storage);
    const auto a = r4dx::cli::ParseArgs(static_cast<int>(argv.size()), argv.data());
    CHECK(a.tp == 1);
    CHECK(!a.tp_options_given);
    CHECK(a.tp_devices.empty());
  }
  {
    std::vector<std::string> storage = {"r4dx-cli", "--model",   "m.r4dx", "--layout", "w4a16",
                                         "--prompt", "hi",        "--tp",   "2",        "--tp-mode",
                                         "emulate",  "--tp-devices", "0", "--tp-ar-timeout-ms", "700",
                                         "--tp-ar-nb", "8", "--tp-ar-nb-large", "16"};
    auto argv = ToArgv(storage);
    const auto a = r4dx::cli::ParseArgs(static_cast<int>(argv.size()), argv.data());
    CHECK(a.tp == 2);
    CHECK(a.tp_mode == "emulate");
    CHECK(a.tp_devices.size() == 1 && a.tp_devices[0] == 0);
    CHECK(a.tp_ar_timeout_ms == 700);
    CHECK(a.tp_ar_nb == 8);
    CHECK(a.tp_ar_nb_large == 16);
  }
  {
    std::vector<std::string> storage = {"r4dx-cli", "--model", "m.r4dx", "--layout",  "w4a16", "--prompt",
                                         "hi",       "--tp",    "2",      "--tp-mode", "noop",  "--tp-rank",
                                         "1",        "--tp-devices", "1,0"};
    auto argv = ToArgv(storage);
    const auto a = r4dx::cli::ParseArgs(static_cast<int>(argv.size()), argv.data());
    CHECK(a.tp_mode == "noop");
    CHECK(a.tp_rank == 1);
    CHECK(a.tp_devices.size() == 2 && a.tp_devices[0] == 1 && a.tp_devices[1] == 0);
  }
  CHECK(TpThrows({"--tp", "3"}));
  CHECK(TpThrows({"--tp-mode", "emulate"}));  // --tp-* without --tp 2
  CHECK(TpThrows({"--tp", "1", "--tp-ar-nb", "4"}));
  CHECK(TpThrows({"--tp", "2"}));  // default --tp-mode real: docs/tp.md P4
  CHECK(TpThrows({"--tp", "2", "--tp-mode", "real"}));
  CHECK(TpThrows({"--tp", "2", "--tp-mode", "bogus"}));
  CHECK(TpThrows({"--tp", "2", "--tp-mode", "emulate", "--tp-rank", "0"}));  // --tp-rank is noop-only
  CHECK(TpThrows({"--tp", "2", "--tp-mode", "noop", "--tp-rank", "2"}));
  CHECK(TpThrows({"--tp", "2", "--tp-mode", "emulate", "--tp-ar-timeout-ms", "5"}));
  CHECK(TpThrows({"--tp", "2", "--tp-mode", "emulate", "--tp-ar-timeout-ms", "1501"}));
  CHECK(TpThrows({"--tp", "2", "--tp-mode", "emulate", "--tp-ar-nb", "0"}));
  CHECK(TpThrows({"--tp", "2", "--tp-mode", "emulate", "--tp-ar-nb-large", "65"}));
  CHECK(TpThrows({"--tp", "2", "--tp-mode", "emulate", "--tp-devices", "0,1,2"}));
  CHECK(TpThrows({"--tp", "2", "--tp-mode", "emulate", "--tp-devices", "x"}));
  CHECK(TpThrows({"--tp", "2", "--tp-mode", "emulate", "--profile"}));
  CHECK(TpThrows({"--tp", "2", "--tp-mode", "emulate", "--profile-prefill"}));
  CHECK(TpThrows({"--tp", "2", "--tp-mode", "emulate", "--mtp", "3"}));
  CHECK(TpThrows({"--tp", "2", "--tp-mode", "emulate", "--dflash", "d.r4dx"}));
  CHECK(TpThrows({"--tp", "2", "--tp-mode", "emulate", "--vision", "on"}));
  CHECK(TpThrows({"--tp", "2", "--tp-mode", "emulate", "--image", "a.png"}));
  CHECK(!TpThrows({"--tp", "2", "--tp-mode", "emulate", "--vision", "auto"}));
  CHECK(!TpThrows({"--tp", "1"}));
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
  TestDflashFlags();
  TestVisionFlags();
  TestImageFlag();
  TestTpFlags();

  if (g_failures > 0) {
    std::fprintf(stderr, "%d check(s) failed\n", g_failures);
    return 1;
  }
  std::fprintf(stderr, "all cli arg-parsing checks passed\n");
  return 0;
}
