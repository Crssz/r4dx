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
// flag -- defaults to "full" (flipped 2026-09-20, Milestone 5 B2 item 8; matched-K=3 real-hardware
// re-measurement, review fix pass: full 68.20/68.64 tok/s vs reduced 53.59/54.17 tok/s, see
// src/cli/cli_args.h's own comment), accepts "reduced", rejects anything else.
void TestMtpDraftHeadFlag() {
  {
    std::vector<std::string> storage = {"r4dx-server", "--model", "m.r4dx"};
    auto argv = ToArgv(storage);
    const auto a = r4dx::server::ParseServerArgs(static_cast<int>(argv.size()), argv.data());
    CHECK(a.mtp_draft_head == "full");
  }
  {
    std::vector<std::string> storage = {"r4dx-server", "--model", "m.r4dx", "--mtp", "3",
                                         "--mtp-draft-head", "reduced"};
    auto argv = ToArgv(storage);
    const auto a = r4dx::server::ParseServerArgs(static_cast<int>(argv.size()), argv.data());
    CHECK(a.mtp_draft_head == "reduced");
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

// --dflash/--dflash-k/--dflash-p-min/--dflash-n-min (docs/dflash2.md, stage S3 item 1): mirrors
// src/cli/cli_args.h's own TestDflashFlags exactly.
void TestDflashFlags() {
  {
    std::vector<std::string> storage = {"r4dx-server", "--model", "m.r4dx"};
    auto argv = ToArgv(storage);
    const auto a = r4dx::server::ParseServerArgs(static_cast<int>(argv.size()), argv.data());
    CHECK(a.dflash.empty());
    CHECK(a.dflash_k == 7);
    CHECK(a.dflash_p_min == 0.0f);
    CHECK(a.dflash_n_min == 0);
  }
  {
    std::vector<std::string> storage = {"r4dx-server",    "--model",        "m.r4dx",
                                         "--dflash",       "draft.r4dx",     "--dflash-k", "5",
                                         "--dflash-p-min", "0.3",            "--dflash-n-min", "2"};
    auto argv = ToArgv(storage);
    const auto a = r4dx::server::ParseServerArgs(static_cast<int>(argv.size()), argv.data());
    CHECK(a.dflash == "draft.r4dx");
    CHECK(a.dflash_k == 5);
    CHECK(a.dflash_p_min > 0.29f && a.dflash_p_min < 0.31f);
    CHECK(a.dflash_n_min == 2);
  }
  {  // --dflash with --mtp > 0 is an error
    std::vector<std::string> storage = {"r4dx-server", "--model", "m.r4dx", "--dflash",
                                         "draft.r4dx",  "--mtp",   "3"};
    auto argv = ToArgv(storage);
    bool threw = false;
    try {
      r4dx::server::ParseServerArgs(static_cast<int>(argv.size()), argv.data());
    } catch (const r4dx::server::ServerUsageError&) {
      threw = true;
    }
    CHECK(threw);
  }
  {  // --dflash-k out of [1,7] is an error, but only when --dflash is actually given
    std::vector<std::string> storage = {"r4dx-server", "--model", "m.r4dx", "--dflash",
                                         "draft.r4dx",  "--dflash-k", "8"};
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

// --vision / --image-max-pixels (docs/vision.md "Load policy" / "Large images"). Same three values
// and the same default as r4dx-cli's, which tests/cli/test_args.cpp's TestVisionFlags pins on the
// other side -- the two binaries' flags are only "the same flag" if both are asserted.
void TestVisionFlags() {
  auto parse = [](std::vector<std::string> extra) {
    std::vector<std::string> storage = {"r4dx-server", "--model", "m.r4dx"};
    storage.insert(storage.end(), extra.begin(), extra.end());
    auto argv = ToArgv(storage);
    return r4dx::server::ParseServerArgs(static_cast<int>(argv.size()), argv.data());
  };
  auto rejects = [](std::vector<std::string> extra) {
    std::vector<std::string> storage = {"r4dx-server", "--model", "m.r4dx"};
    storage.insert(storage.end(), extra.begin(), extra.end());
    auto argv = ToArgv(storage);
    try {
      r4dx::server::ParseServerArgs(static_cast<int>(argv.size()), argv.data());
    } catch (const r4dx::server::ServerUsageError&) {
      return true;
    }
    return false;
  };

  {
    const auto a = parse({});
    CHECK(a.vision == "auto");
    CHECK(a.image_max_pixels == 1048576);
  }
  CHECK(parse({"--vision", "on"}).vision == "on");
  CHECK(parse({"--vision", "off"}).vision == "off");
  CHECK(rejects({"--vision", "enabled"}));
  CHECK(parse({"--image-max-pixels", "0"}).image_max_pixels == 0);
  CHECK(parse({"--image-max-pixels", "2359296"}).image_max_pixels == 2359296);
  CHECK(rejects({"--image-max-pixels", "512"}));
  CHECK(rejects({"--image-max-pixels", "-4"}));
  CHECK(rejects({"--image-max-pixels", "big"}));
}

// --tp and --tp-* (docs/tp.md 9.1): the same flags, ranges and usage errors as r4dx-cli's, which
// tests/cli/test_args.cpp's TestTpFlags pins on the other side (same reason as TestVisionFlags).
bool TpThrows(std::vector<std::string> extra) {
  std::vector<std::string> storage = {"r4dx-server", "--model", "m.r4dx", "--layout", "w4a16"};
  storage.insert(storage.end(), extra.begin(), extra.end());
  auto argv = ToArgv(storage);
  try {
    r4dx::server::ParseServerArgs(static_cast<int>(argv.size()), argv.data());
  } catch (const r4dx::server::ServerUsageError&) {
    return true;
  }
  return false;
}

void TestTpFlags() {
  auto parse = [](std::vector<std::string> extra) {
    std::vector<std::string> storage = {"r4dx-server", "--model", "m.r4dx", "--layout", "w4a16"};
    storage.insert(storage.end(), extra.begin(), extra.end());
    auto argv = ToArgv(storage);
    return r4dx::server::ParseServerArgs(static_cast<int>(argv.size()), argv.data());
  };
  {
    const auto a = parse({});
    CHECK(a.tp == 1);
    CHECK(!a.tp_options_given);
    CHECK(a.tp_mode == "real");
    CHECK(a.tp_devices.empty());
    CHECK(a.tp_submit_layers == -1 && a.tp_max_inflight == -1);
  }
  {
    const auto a = parse({"--tp", "2", "--tp-mode", "emulate", "--tp-devices", "0", "--tp-ar-timeout-ms", "700",
                          "--tp-ar-nb", "8", "--tp-ar-nb-large", "16"});
    CHECK(a.tp == 2);
    CHECK(a.tp_mode == "emulate");
    CHECK(a.tp_devices.size() == 1 && a.tp_devices[0] == 0);
    CHECK(a.tp_ar_timeout_ms == 700);
    CHECK(a.tp_ar_nb == 8);
    CHECK(a.tp_ar_nb_large == 16);
    CHECK(a.tp_options_given);
  }
  {
    const auto a = parse({"--tp", "2", "--tp-mode", "noop", "--tp-rank", "1", "--tp-devices", "1,0"});
    CHECK(a.tp_mode == "noop");
    CHECK(a.tp_rank == 1);
    CHECK(a.tp_devices.size() == 2 && a.tp_devices[0] == 1 && a.tp_devices[1] == 0);
  }
  {
    const auto a = parse({"--tp", "2"});
    CHECK(a.tp == 2 && a.tp_mode == "real" && !a.tp_options_given);
    const auto b = parse({"--tp", "2", "--tp-submit-layers", "0", "--tp-max-inflight", "3"});
    CHECK(b.tp_submit_layers == 0 && b.tp_max_inflight == 3 && b.tp_options_given);
  }
  CHECK(TpThrows({"--tp", "3"}));
  CHECK(TpThrows({"--tp", "0"}));
  CHECK(TpThrows({"--tp-mode", "emulate"}));  // --tp-* without --tp 2
  CHECK(TpThrows({"--tp", "1", "--tp-ar-nb", "4"}));
  CHECK(TpThrows({"--tp", "1", "--tp-submit-layers", "4"}));
  CHECK(!TpThrows({"--tp", "1"}));
  CHECK(!TpThrows({"--tp", "2", "--tp-mode", "real", "--tp-devices", "1,0"}));
  CHECK(!TpThrows({"--tp", "2", "--tp-devices", "auto"}));
  CHECK(TpThrows({"--tp", "2", "--tp-mode", "bogus"}));
  CHECK(TpThrows({"--tp", "2", "--tp-submit-layers", "-1"}));
  CHECK(TpThrows({"--tp", "2", "--tp-submit-layers", "65"}));
  CHECK(TpThrows({"--tp", "2", "--tp-max-inflight", "-2"}));
  CHECK(TpThrows({"--tp", "2", "--tp-max-inflight", "65"}));
  CHECK(TpThrows({"--tp", "2", "--tp-mode", "emulate", "--tp-rank", "0"}));  // --tp-rank is noop-only
  CHECK(TpThrows({"--tp", "2", "--tp-mode", "noop", "--tp-rank", "2"}));
  CHECK(TpThrows({"--tp", "2", "--tp-ar-timeout-ms", "5"}));
  CHECK(TpThrows({"--tp", "2", "--tp-ar-timeout-ms", "1501"}));
  CHECK(TpThrows({"--tp", "2", "--tp-ar-nb", "0"}));
  CHECK(TpThrows({"--tp", "2", "--tp-ar-nb-large", "65"}));
  CHECK(TpThrows({"--tp", "2", "--tp-devices", "0,1,2"}));
  CHECK(TpThrows({"--tp", "2", "--tp-devices", "x"}));
  CHECK(TpThrows({"--tp", "2", "--tp-devices", "-1"}));
  CHECK(TpThrows({"--tp", "two"}));
  // MTP, DFlash2 and vision run under --tp 2 (docs/tp.md P5) ...
  CHECK(!TpThrows({"--tp", "2", "--mtp", "3"}));
  CHECK(!TpThrows({"--tp", "2", "--dflash", "d.r4dx", "--dflash-k", "7"}));
  CHECK(!TpThrows({"--tp", "2", "--vision", "on"}));
  // ... while the rules that hold at --tp 1 still hold at --tp 2.
  CHECK(TpThrows({"--tp", "2", "--mtp", "3", "--dflash", "d.r4dx"}));
  CHECK(TpThrows({"--tp", "2", "--dflash", "d.r4dx", "--dflash-k", "8"}));
  // --mtp is capped at 7 under --tp 2, as in r4dx-cli (docs/tp.md Appendix B N80).
  CHECK(!TpThrows({"--tp", "2", "--mtp", "7"}));
  CHECK(TpThrows({"--tp", "2", "--mtp", "8"}));
  CHECK(TpThrows({"--tp", "2", "--tp-mode", "emulate", "--mtp", "63"}));
  CHECK(!TpThrows({"--tp", "1", "--mtp", "63"}));
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
  TestDflashFlags();
  TestVisionFlags();
  TestTpFlags();

  if (g_failures > 0) {
    std::fprintf(stderr, "%d check(s) failed\n", g_failures);
    return 1;
  }
  std::fprintf(stderr, "all server arg-parsing checks passed\n");
  return 0;
}
