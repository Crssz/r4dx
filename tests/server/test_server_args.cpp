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
      {"r4dx-server", "--model", "m.r4dx", "--dialect", "llama"},
      {"r4dx-server", "--model", "m.r4dx", "--dialect", "Gemma4"},
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

// --dialect / --extended-ctx and the "was it given" markers main.cpp reads (docs/gemma4-plan.md 5.3 and
// 9.1, task M1-14): the Qwen defaults stay exactly as before.
void TestDialectAndContextFlags() {
  {
    std::vector<std::string> storage = {"r4dx-server", "--model", "m.r4dx"};
    auto argv = ToArgv(storage);
    const auto a = r4dx::server::ParseServerArgs(static_cast<int>(argv.size()), argv.data());
    CHECK(a.dialect == "auto" && !a.tokenizer_dir_given && !a.max_ctx_given && !a.extended_ctx);
    CHECK(a.tokenizer_dir == r4dx::ModelsPath("Huihui-Qwen3.8-27B-abliterated"));
    CHECK(a.max_ctx == 262144);
  }
  {
    std::vector<std::string> storage = {"r4dx-server", "--model", "g.r4dx", "--dialect", "gemma4",
                                        "--tokenizer-dir", "C:\\gtok", "--max-ctx", "65536", "--extended-ctx"};
    auto argv = ToArgv(storage);
    const auto a = r4dx::server::ParseServerArgs(static_cast<int>(argv.size()), argv.data());
    CHECK(a.dialect == "gemma4" && a.tokenizer_dir_given && a.tokenizer_dir == "C:\\gtok");
    CHECK(a.max_ctx_given && a.max_ctx == 65536 && a.extended_ctx);
  }
  for (const char* d : {"auto", "qwen35", "gemma4"}) {
    std::vector<std::string> storage = {"r4dx-server", "--model", "m.r4dx", "--dialect", d};
    auto argv = ToArgv(storage);
    CHECK(r4dx::server::ParseServerArgs(static_cast<int>(argv.size()), argv.data()).dialect == d);
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

// --prompt-checkpoint (docs/server.md "Prefix cache"): defaults to "on", accepts "off", rejects
// anything else.
void TestPromptCheckpointFlag() {
  {
    std::vector<std::string> storage = {"r4dx-server", "--model", "m.r4dx"};
    auto argv = ToArgv(storage);
    const auto a = r4dx::server::ParseServerArgs(static_cast<int>(argv.size()), argv.data());
    CHECK(a.prompt_checkpoint == "on");
  }
  {
    std::vector<std::string> storage = {"r4dx-server", "--model", "m.r4dx", "--prompt-checkpoint", "off"};
    auto argv = ToArgv(storage);
    const auto a = r4dx::server::ParseServerArgs(static_cast<int>(argv.size()), argv.data());
    CHECK(a.prompt_checkpoint == "off");
  }
  {
    std::vector<std::string> storage = {"r4dx-server", "--model", "m.r4dx", "--prompt-checkpoint", "yes"};
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

// --request-log (docs/server.md "Request log"): absent = "" (off, no file is ever named), present =
// the path verbatim (spaces and all), and a missing or empty value is a usage error -- an empty path
// must not silently mean "off".
void TestRequestLogFlag() {
  {
    std::vector<std::string> storage = {"r4dx-server", "--model", "m.r4dx"};
    auto argv = ToArgv(storage);
    const auto a = r4dx::server::ParseServerArgs(static_cast<int>(argv.size()), argv.data());
    CHECK(a.request_log.empty());
  }
  {
    std::vector<std::string> storage = {"r4dx-server", "--model", "m.r4dx", "--request-log",
                                        "D:\\logs dir\\requests.jsonl", "--port", "18080"};
    auto argv = ToArgv(storage);
    const auto a = r4dx::server::ParseServerArgs(static_cast<int>(argv.size()), argv.data());
    CHECK(a.request_log == "D:\\logs dir\\requests.jsonl");
    CHECK(a.port == 18080);  // the flag consumed exactly its one value
  }
  for (const std::vector<std::string>& tail : {std::vector<std::string>{"--request-log"},
                                                std::vector<std::string>{"--request-log", ""}}) {
    std::vector<std::string> storage = {"r4dx-server", "--model", "m.r4dx"};
    storage.insert(storage.end(), tail.begin(), tail.end());
    auto argv = ToArgv(storage);
    bool threw = false;
    try {
      r4dx::server::ParseServerArgs(static_cast<int>(argv.size()), argv.data());
    } catch (const r4dx::server::ServerUsageError&) {
      threw = true;
    }
    CHECK(threw);
  }
  CHECK(r4dx::server::ServerUsageText("r4dx-server").find("--request-log <path>") != std::string::npos);

  // --request-log-tokens: off by default (also with --request-log), a plain switch, and only legal
  // next to --request-log, in either order.
  {
    std::vector<std::string> storage = {"r4dx-server", "--model", "m.r4dx", "--request-log", "r.jsonl"};
    auto argv = ToArgv(storage);
    CHECK(!r4dx::server::ParseServerArgs(static_cast<int>(argv.size()), argv.data()).request_log_tokens);
  }
  for (const std::vector<std::string>& tail :
       {std::vector<std::string>{"--request-log", "r.jsonl", "--request-log-tokens", "--port", "18080"},
        std::vector<std::string>{"--request-log-tokens", "--request-log", "r.jsonl", "--port", "18080"}}) {
    std::vector<std::string> storage = {"r4dx-server", "--model", "m.r4dx"};
    storage.insert(storage.end(), tail.begin(), tail.end());
    auto argv = ToArgv(storage);
    const auto a = r4dx::server::ParseServerArgs(static_cast<int>(argv.size()), argv.data());
    CHECK(a.request_log_tokens && a.request_log == "r.jsonl" && a.port == 18080);  // the switch takes no value
  }
  {
    std::vector<std::string> storage = {"r4dx-server", "--model", "m.r4dx", "--request-log-tokens"};
    auto argv = ToArgv(storage);
    bool threw = false;
    try {
      r4dx::server::ParseServerArgs(static_cast<int>(argv.size()), argv.data());
    } catch (const r4dx::server::ServerUsageError& e) {
      threw = std::string(e.what()).find("--request-log") != std::string::npos;
    }
    CHECK(threw);  // --request-log-tokens without --request-log is a usage error
  }
  CHECK(r4dx::server::ServerUsageText("r4dx-server").find("[--request-log-tokens]") != std::string::npos);
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
  // Gemma 4 (docs/gemma4-plan.md M2): the soft-token budget; 560 / 1120 are valid processor budgets but exceed the
  // 288-row image chunk, so the server refuses them.
  CHECK(parse({}).image_soft_tokens == 280);
  CHECK(parse({"--image-soft-tokens", "70"}).image_soft_tokens == 70);
  CHECK(parse({"--image-soft-tokens", "140"}).image_soft_tokens == 140);
  CHECK(rejects({"--image-soft-tokens", "560"}));
  CHECK(rejects({"--image-soft-tokens", "100"}));
  CHECK(rejects({"--image-soft-tokens", "many"}));
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
  // Pipeline-parallel prefill (docs/pp-prefill.md Phase 2): the same --pp* flags as r4dx-cli.
  {
    const auto a = parse({});
    CHECK(a.pp == -1 && a.pp_split == 0 && a.pp_min_rows == -1 && !a.pp_verify && a.pp_submit_layers == -1 &&
          a.pp_max_inflight == -1 && !a.pp_options_given);
    const auto b = parse({"--pp", "2", "--pp-split", "35", "--pp-min-rows", "512", "--pp-verify", "--pp-submit-layers", "0",
                          "--pp-max-inflight", "2"});
    CHECK(b.pp == 2 && b.pp_split == 35 && b.pp_min_rows == 512 && b.pp_verify && b.pp_submit_layers == 0 &&
          b.pp_max_inflight == 2 && b.pp_options_given);
    CHECK(parse({"--pp", "2", "--pp-split", "auto"}).pp_split == 0);
    CHECK(a.pp_devices.empty() && b.pp_devices.empty());  // auto placement unless asked for
    CHECK(parse({"--pp", "2", "--pp-devices", "1,0"}).pp_devices == std::vector<int>({1, 0}));
    CHECK(parse({"--pp", "2", "--pp-devices", "0,1"}).pp_devices == std::vector<int>({0, 1}));
    CHECK(parse({"--pp", "2", "--pp-devices", "auto"}).pp_devices.empty());
  }
  CHECK(TpThrows({"--pp-devices", "1,0"}));  // needs --pp 2
  CHECK(TpThrows({"--pp", "2", "--pp-devices", "1"}));
  CHECK(TpThrows({"--pp", "2", "--pp-devices", "1,0,2"}));
  CHECK(TpThrows({"--pp", "2", "--pp-devices", "1,1"}));
  CHECK(TpThrows({"--pp", "2", "--pp-devices", "1,x"}));
  CHECK(TpThrows({"--pp", "2", "--pp-devices", "-1,0"}));
  CHECK(!TpThrows({"--pp", "2"}));
  CHECK(!TpThrows({"--pp", "1"}));
  CHECK(!TpThrows({"--pp", "2", "--dflash", "d.r4dx", "--dflash-k", "7"}));
  CHECK(TpThrows({"--pp", "3"}));
  CHECK(TpThrows({"--pp", "0"}));
  CHECK(TpThrows({"--pp-split", "33"}));
  CHECK(TpThrows({"--pp", "1", "--pp-verify"}));
  // --tp 2 --pp 2 is the hybrid serving mode (docs/pp-tp2-hybrid.md): two real GPUs, the stages next to the ranks.
  CHECK(!TpThrows({"--pp", "2", "--tp", "2"}));
  CHECK(!TpThrows({"--pp", "2", "--tp", "2", "--pp-split", "30", "--pp-min-rows", "768", "--pp-devices", "1,0"}));
  CHECK(!TpThrows({"--pp", "2", "--tp", "2", "--hybrid", "off"}));
  CHECK(!TpThrows({"--pp", "2", "--tp", "2", "--hybrid", "on", "--hybrid-ctx", "65536", "--hybrid-reserve-gib", "2.5"}));
  CHECK(!TpThrows({"--pp", "2", "--tp", "2", "--hybrid-ctx", "auto"}));
  CHECK(TpThrows({"--pp", "2", "--tp", "2", "--tp-mode", "emulate"}));
  CHECK(!TpThrows({"--pp", "2", "--tp", "2", "--tp-mode", "emulate", "--hybrid", "off"}));
  // --pp-devices B,A must name the --tp-devices pair or be refused
  CHECK(!TpThrows({"--pp", "2", "--tp", "2", "--tp-devices", "1,0", "--pp-devices", "1,0"}));
  CHECK(TpThrows({"--pp", "2", "--tp", "2", "--tp-devices", "1,0", "--pp-devices", "0,1"}));
  CHECK(!TpThrows({"--pp", "2", "--tp", "2", "--tp-devices", "1,0", "--pp-devices", "0,1", "--hybrid", "off"}));
  CHECK(TpThrows({"--hybrid", "off"}));
  CHECK(TpThrows({"--tp", "2", "--hybrid", "off"}));
  CHECK(TpThrows({"--pp", "2", "--hybrid-ctx", "65536"}));
  CHECK(TpThrows({"--pp", "2", "--tp", "2", "--hybrid", "maybe"}));
  CHECK(TpThrows({"--pp", "2", "--tp", "2", "--hybrid-ctx", "0"}));
  CHECK(TpThrows({"--pp", "2", "--tp", "2", "--hybrid-reserve-gib", "99"}));
  {
    const auto a = parse({"--pp", "2", "--tp", "2", "--hybrid", "off", "--hybrid-ctx", "65536", "--hybrid-reserve-gib", "2.5"});
    CHECK(a.pp == 2 && a.tp == 2 && a.hybrid == 0 && a.hybrid_ctx == 65536 && a.hybrid_reserve_gib == 2.5 && a.hybrid_options_given);
    const auto b = parse({"--pp", "2", "--tp", "2"});
    CHECK(b.hybrid == -1 && b.hybrid_ctx == 0 && b.hybrid_reserve_gib < 0 && !b.hybrid_options_given);
  }
  CHECK(TpThrows({"--pp", "2", "--pp-split", "0"}));
  CHECK(TpThrows({"--pp", "2", "--pp-min-rows", "0"}));
  CHECK(TpThrows({"--pp", "2", "--pp-submit-layers", "65"}));
  CHECK(TpThrows({"--pp", "2", "--pp-max-inflight", "-1"}));
}

}  // namespace

void TestBatchFlags() {
  using r4dx::server::ParseServerArgs;
  using r4dx::server::ServerUsageError;
  const auto parse = [](std::vector<std::string> storage) {
    storage.insert(storage.begin(), "r4dx-server");
    storage.insert(storage.begin() + 1, "--model");
    storage.insert(storage.begin() + 2, "m.r4dx");
    auto argv = ToArgv(storage);
    return ParseServerArgs(static_cast<int>(argv.size()), argv.data());
  };
  const auto throws = [&](std::vector<std::string> storage) {
    try {
      (void)parse(std::move(storage));
    } catch (const ServerUsageError&) {
      return true;
    }
    return false;
  };
  const auto a0 = parse({});
  CHECK(a0.batch == 0 && a0.batch_ctx == 32768 && !a0.batch_ctx_given);  // off by default: the one-request-at-a-time server
  const auto a = parse({"--batch", "4"});
  CHECK(a.batch == 4 && a.batch_ctx == 32768);
  const auto big = parse({"--batch", "16", "--batch-ctx", "65536"});
  CHECK(big.batch == 16 && big.batch_ctx == 65536 && big.batch_ctx_given && big.tp == 1);
  const auto b = parse({"--batch", "8", "--tp", "2"});
  CHECK(b.batch == 8 && b.tp == 2);
  CHECK(throws({"--batch", "9", "--tp", "2"}));  // a TP step stays at most 8 rows
  // the hybrid mode takes it too: decode of the slots runs on the TP ranks after the pipelined prefill
  const auto h = parse({"--tp", "2", "--pp", "2", "--batch", "4", "--batch-ctx", "16384"});
  CHECK(h.batch == 4 && h.tp == 2 && h.pp == 2);
  CHECK(throws({"--batch", "1"}));                        // one slot IS --batch 0
  CHECK(throws({"--batch", "17"}));                       // past the slot bound
  CHECK(throws({"--batch", "-2"}));
  CHECK(throws({"--batch-ctx", "4096"}));                 // without --batch
  CHECK(throws({"--batch", "2", "--batch-ctx", "1000"})); // not a multiple of the 16-token KV block
  CHECK(throws({"--batch", "2", "--batch-ctx", "0"}));
  CHECK(throws({"--batch", "2", "--mtp", "3"}));          // plain decode only
  CHECK(throws({"--batch", "2", "--dflash", "d.r4dx"}));
  CHECK(throws({"--batch", "8", "--max-queue", "4"}));    // the queue has to be able to fill the slots
  CHECK(!throws({"--batch", "8", "--max-queue", "8"}));
  CHECK(throws({"--batch", "abc"}));
}

int main() {
  TestBatchFlags();
  TestDefaults();
  TestAllFlags();
  TestMissingModelThrows();
  TestUnrecognizedFlagThrows();
  TestNonsensicalValuesThrow();
  TestUnparseableNumberThrowsServerUsageError();
  TestDialectAndContextFlags();
  TestMtpHeadLayoutFlag();
  TestMtpDraftHeadFlag();
  TestEmbedDeviceResidentFlag();
  TestPromptCheckpointFlag();
  TestRequestLogFlag();
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
