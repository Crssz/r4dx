// tests/server/test_request_log.cpp -- pure CPU unit test for src/server/request_log.h: the JSON
// Lines field formatting, the ISO-8601 timestamp, and the writer (append, no BOM, one "\n" per line,
// open failure, a failing stream, concurrent writers). No HIP device, no tokenizer.
#define _CRT_SECURE_NO_WARNINGS  // the failing-stream test wants a plain fopen("rb")
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "nlohmann/json.hpp"
#include "request_log.h"

namespace {

using nlohmann::json;
using r4dx::server::FormatIso8601;
using r4dx::server::FormatRequestLogLine;
using r4dx::server::RequestLog;
using r4dx::server::RequestLogRecord;

int g_failures = 0;

#define CHECK(cond, ...)                                                                    \
  do {                                                                                      \
    if (!(cond)) {                                                                          \
      std::fprintf(stderr, "CHECK failed at %s:%d: %s -- ", __FILE__, __LINE__, #cond);     \
      std::fprintf(stderr, "" __VA_ARGS__);                                                 \
      std::fprintf(stderr, "\n");                                                           \
      ++g_failures;                                                                         \
    }                                                                                       \
  } while (0)

// A request that ran to completion on the second turn of a conversation.
RequestLogRecord FullRecord() {
  RequestLogRecord r;
  r.timestamp = "2026-09-30T14:03:22.123+07:00";
  r.request_id = "chatcmpl-abc123";
  r.endpoint = "chat/completions";
  r.stream = true;
  r.http_status = 200;
  r.finish_reason = "stop";
  r.thinking = true;
  r.max_tokens = 4096;
  r.temperature = 0.6;
  r.tools_count = 12;
  r.image_count = 0;
  r.prompt_tokens = 10000;
  r.prompt_n = 400;
  r.cached_tokens = 9600;
  r.completion_tokens = 250;
  r.reasoning_tokens = 180;
  r.queue_wait_ms = 1.25;
  r.prompt_ms = 2000.0;
  r.predicted_ms = 5000.0;
  r.draft_n = 120;
  r.draft_n_accepted = 90;
  r.speculative = "dflash";
  r.full_reset = false;
  r.checkpoint_restore = false;
  return r;
}

// Every key of a line, in order. The documented schema (docs/server.md "Request log"): a reader of
// the file, and this test, depend on it.
const std::vector<std::string> kKeys = {
    "ts",          "request_id",         "endpoint",          "stream",         "http_status",
    "error_status", "finish_reason",     "cancelled",         "thinking",       "max_tokens",
    "temperature", "tools_present",      "tools_count",       "image_count",    "prompt_tokens",
    "prompt_n",    "cached_tokens",      "completion_tokens", "reasoning_tokens", "queue_wait_ms",
    "prompt_ms",   "predicted_ms",       "prompt_per_second", "predicted_per_second", "speculative",
    "draft_n",     "draft_n_accepted",   "full_reset",        "checkpoint_restore", "reset_ms",
    "restore_ms",  "image_n",            "image_ms"};

void TestFormatFullRecord() {
  const std::string line = FormatRequestLogLine(FullRecord());
  CHECK(line.find('\n') == std::string::npos && line.find('\r') == std::string::npos, "a line holds no newline");
  const json j = json::parse(line);
  CHECK(j["ts"] == "2026-09-30T14:03:22.123+07:00");
  CHECK(j["request_id"] == "chatcmpl-abc123");
  CHECK(j["endpoint"] == "chat/completions");
  CHECK(j["stream"] == true);
  CHECK(j["http_status"] == 200);
  CHECK(j["error_status"].is_null());
  CHECK(j["finish_reason"] == "stop");
  CHECK(j["cancelled"] == false);
  CHECK(j["thinking"] == true);
  CHECK(j["max_tokens"] == 4096);
  CHECK(j["temperature"] == 0.6);
  CHECK(j["tools_present"] == true && j["tools_count"] == 12);
  CHECK(j["image_count"] == 0);
  CHECK(j["prompt_tokens"] == 10000 && j["prompt_n"] == 400 && j["cached_tokens"] == 9600);
  CHECK(j["prompt_tokens"].get<int64_t>() == j["prompt_n"].get<int64_t>() + j["cached_tokens"].get<int64_t>());
  CHECK(j["completion_tokens"] == 250 && j["reasoning_tokens"] == 180);
  CHECK(j["queue_wait_ms"] == 1.25 && j["prompt_ms"] == 2000.0 && j["predicted_ms"] == 5000.0);
  CHECK(j["prompt_per_second"] == 200.0);     // 400 tokens / 2 s
  CHECK(j["predicted_per_second"] == 50.0);   // 250 tokens / 5 s
  CHECK(j["speculative"] == "dflash" && j["draft_n"] == 120 && j["draft_n_accepted"] == 90);
  CHECK(j["full_reset"] == false && j["checkpoint_restore"] == false);
  CHECK(j["reset_ms"].is_null() && j["restore_ms"].is_null());
  CHECK(j["image_n"].is_null() && j["image_ms"].is_null());

  // Field order is part of the format: the keys come out exactly as documented, ts first.
  // (nlohmann::json sorts keys on parse; read the order straight from the text instead.)
  size_t pos = 0;
  for (const std::string& k : kKeys) {
    const size_t at = line.find("\"" + k + "\":", pos);
    CHECK(at != std::string::npos, "key %s missing or out of order", k.c_str());
    if (at == std::string::npos) break;
    pos = at;
  }
  CHECK(j.size() == kKeys.size(), "line has %zu keys, schema %zu", j.size(), kKeys.size());
}

void TestFormatUnknownsAreNull() {
  // A request rejected before the engine: only what is known, everything else JSON null -- never 0.
  RequestLogRecord r;
  r.timestamp = "2026-09-30T00:00:00.000+00:00";
  r.endpoint = "completions";
  r.http_status = 429;
  const json j = json::parse(FormatRequestLogLine(r));
  CHECK(j["request_id"].is_null() && j["stream"].is_null());
  CHECK(j["http_status"] == 429);
  for (const char* k : {"prompt_tokens", "prompt_n", "cached_tokens", "completion_tokens", "reasoning_tokens",
                        "prompt_ms", "predicted_ms", "prompt_per_second", "predicted_per_second", "queue_wait_ms",
                        "draft_n", "full_reset", "tools_present", "tools_count", "image_count", "finish_reason",
                        "thinking", "max_tokens", "temperature", "speculative"}) {
    CHECK(j[k].is_null(), "%s should be null", k);
  }
  CHECK(j["speculative"].is_null() && j["cancelled"] == false);
  CHECK(j.size() == kKeys.size());
}

void TestFormatDerivedRates() {
  RequestLogRecord r = FullRecord();
  r.prompt_ms = 0.0;  // the timings object's rule: a rate over zero time is 0, never inf/NaN
  r.predicted_ms = 0.0;
  json j = json::parse(FormatRequestLogLine(r));
  CHECK(j["prompt_per_second"] == 0.0 && j["predicted_per_second"] == 0.0);
  r = FullRecord();
  r.tools_count = 0;
  r.thinking = false;
  r.reasoning_tokens.reset();
  r.full_reset = true;
  r.reset_ms = 23.4567;
  r.cancelled = true;
  r.finish_reason = "cancelled";
  r.temperature = 0.0;
  r.draft_n.reset();
  r.draft_n_accepted.reset();
  j = json::parse(FormatRequestLogLine(r));
  CHECK(j["tools_present"] == false && j["tools_count"] == 0);
  CHECK(j["thinking"] == false && j["reasoning_tokens"].is_null());
  CHECK(j["full_reset"] == true && j["reset_ms"] == 23.457);  // ms rounded to 3 decimals
  CHECK(j["cancelled"] == true && j["finish_reason"] == "cancelled");
  CHECK(j["temperature"] == 0.0 && j["draft_n"].is_null());
}

// Privacy: a record cannot hold caller text, and what a line may carry is fixed. Put recognisable
// "prompt" strings into every string a caller could influence that the record DOES have -- none is
// caller text by design -- and check the line has no key beyond the schema and no value that looks
// like free text other than the documented short enums.
void TestNoFreeTextFields() {
  const json j = json::parse(FormatRequestLogLine(FullRecord()));
  const std::set<std::string> string_keys = {"ts", "request_id", "endpoint", "finish_reason", "speculative"};
  for (auto it = j.begin(); it != j.end(); ++it) {
    if (it.value().is_string()) {
      CHECK(string_keys.count(it.key()) == 1, "unexpected string field %s", it.key().c_str());
    }
  }
  // A record with a hostile request id still produces one valid JSON line.
  RequestLogRecord r = FullRecord();
  r.request_id = "x\"y\n\\z\xff";
  const std::string line = FormatRequestLogLine(r);
  CHECK(line.find('\n') == std::string::npos);
  CHECK(json::parse(line).contains("request_id"));
}

void TestIso8601() {
  std::tm t{};
  t.tm_year = 2026 - 1900;
  t.tm_mon = 8;
  t.tm_mday = 30;
  t.tm_hour = 14;
  t.tm_min = 3;
  t.tm_sec = 22;
  CHECK(FormatIso8601(t, 7 * 60, 123) == "2026-09-30T14:03:22.123+07:00", "%s", FormatIso8601(t, 420, 123).c_str());
  CHECK(FormatIso8601(t, 0, 5) == "2026-09-30T14:03:22.005+00:00");
  CHECK(FormatIso8601(t, -(5 * 60 + 30), 999) == "2026-09-30T14:03:22.999-05:30");

  // The live clock: shape only (local time and offset vary by machine). Also: the offset it writes
  // is self-consistent with the local/UTC breakdown it came from, checked by re-deriving the UTC
  // instant and comparing with the system clock to within a few seconds.
  const std::string now = r4dx::server::LocalIso8601Now();
  CHECK(now.size() == 29, "'%s'", now.c_str());
  auto digit = [&](size_t i) { return i < now.size() && now[i] >= '0' && now[i] <= '9'; };
  bool shape = now.size() == 29 && now[4] == '-' && now[7] == '-' && now[10] == 'T' && now[13] == ':' &&
               now[16] == ':' && now[19] == '.' && (now[23] == '+' || now[23] == '-') && now[26] == ':';
  for (size_t i : {0u, 1u, 2u, 3u, 5u, 6u, 8u, 9u, 11u, 12u, 14u, 15u, 17u, 18u, 20u, 21u, 22u, 24u, 25u, 27u, 28u}) {
    shape = shape && digit(i);
  }
  CHECK(shape, "'%s' is not YYYY-MM-DDTHH:MM:SS.mmm+HH:MM", now.c_str());
  if (shape) {
    std::tm lt{};
    lt.tm_year = std::stoi(now.substr(0, 4)) - 1900;
    lt.tm_mon = std::stoi(now.substr(5, 2)) - 1;
    lt.tm_mday = std::stoi(now.substr(8, 2));
    lt.tm_hour = std::stoi(now.substr(11, 2));
    lt.tm_min = std::stoi(now.substr(14, 2));
    lt.tm_sec = std::stoi(now.substr(17, 2));
    lt.tm_isdst = -1;
    const int off = (std::stoi(now.substr(24, 2)) * 60 + std::stoi(now.substr(27, 2))) * (now[23] == '-' ? -1 : 1);
    // timegm-free: mktime reads `lt` as local time; subtracting the local offset mktime itself
    // implies must give the same instant the stamped offset implies.
    const std::time_t as_local = std::mktime(&lt);
    std::tm again{};
#ifdef _WIN32
    localtime_s(&again, &as_local);
#else
    localtime_r(&as_local, &again);
#endif
    CHECK(again.tm_hour == lt.tm_hour && again.tm_min == lt.tm_min, "stamp is a valid local time");
    std::tm g{};
    const std::time_t sys = std::time(nullptr);
#ifdef _WIN32
    gmtime_s(&g, &sys);
#else
    gmtime_r(&sys, &g);
#endif
    const int utc_min_of_day = g.tm_hour * 60 + g.tm_min;
    int stamped_utc = (lt.tm_hour * 60 + lt.tm_min - off) % (24 * 60);
    if (stamped_utc < 0) stamped_utc += 24 * 60;
    int diff = std::abs(stamped_utc - utc_min_of_day);
    diff = std::min(diff, 24 * 60 - diff);
    CHECK(diff <= 1, "stamped offset %d min puts the instant %d min from the system's UTC clock", off, diff);
  }
}

std::string Slurp(const std::filesystem::path& p);
std::vector<std::string> Lines(const std::string& s);

// A request logged under --request-log-tokens: a 6-token prompt, 5 generated tokens, two DFlash rounds.
RequestLogRecord TokenRecord(std::vector<int32_t> prompt, std::vector<int32_t> generated) {
  RequestLogRecord r = FullRecord();
  r.tokens.emplace();
  r.tokens->prompt_ids = std::move(prompt);
  r.tokens->generated_ids = std::move(generated);
  r.tokens->round_drafted = {7, 5};
  r.tokens->round_accepted = {3, 0};
  r.tokens->draft_k = 7;
  r.tokens->top_p = 0.95;
  r.tokens->top_k = 20;
  r.tokens->min_p = 0.0;
  return r;
}

std::vector<int32_t> Ints(const json& a) {
  std::vector<int32_t> v;
  for (const json& x : a) v.push_back(x.get<int32_t>());
  return v;
}

void TestTokensFormat() {
  // Absent (the default): not one extra key, and no id array anywhere in the line.
  const std::string plain = FormatRequestLogLine(FullRecord());
  CHECK(plain.find("prompt_ids") == std::string::npos && plain.find("generated_ids") == std::string::npos);

  RequestLogRecord r = TokenRecord({10, 11, 12, 13, 14, 15}, {20, 21, 22, 23, 24});
  r.tokens->seed = 1234567890123ULL;
  std::string line = FormatRequestLogLine(r);
  CHECK(line.find('\n') == std::string::npos && line.back() == '}');
  json j = json::parse(line);
  CHECK(j.size() == kKeys.size() + 10, "line has %zu keys", j.size());
  for (const std::string& k : kKeys) CHECK(j.contains(k), "the fixed key %s is kept", k.c_str());
  CHECK(j["prompt_ids"] == json::array({10, 11, 12, 13, 14, 15}) && j["prompt_shared"] == 0);
  CHECK(j["generated_ids"] == json::array({20, 21, 22, 23, 24}));
  CHECK(j["round_drafted"] == json::array({7, 5}) && j["round_accepted"] == json::array({3, 0}));
  CHECK(j["draft_k"] == 7 && j["top_p"] == 0.95 && j["top_k"] == 20 && j["min_p"] == 0.0);
  CHECK(j["seed"] == 1234567890123ULL);
  // The token keys come after every fixed one, in a fixed order.
  size_t pos = line.find("\"image_ms\":");
  for (const char* k : {"draft_k", "top_p", "top_k", "min_p", "seed", "prompt_shared", "prompt_ids", "generated_ids",
                        "round_drafted", "round_accepted"}) {
    const size_t at = line.find(std::string("\"") + k + "\":", pos);
    CHECK(at != std::string::npos, "token key %s missing or out of order", k);
    if (at == std::string::npos) break;
    pos = at;
  }

  // prompt_shared drops that many leading prompt ids from the line (clamped to the prompt's length).
  r.tokens->prompt_shared = 4;
  j = json::parse(FormatRequestLogLine(r));
  CHECK(j["prompt_shared"] == 4 && j["prompt_ids"] == json::array({14, 15}));
  r.tokens->prompt_shared = 99;
  j = json::parse(FormatRequestLogLine(r));
  CHECK(j["prompt_shared"] == 6 && j["prompt_ids"].empty());

  // A request with no speculation and a random seed: empty round arrays, nulls, still valid JSON.
  RequestLogRecord n = TokenRecord({1}, {});
  n.tokens->round_drafted.clear();
  n.tokens->round_accepted.clear();
  n.tokens->draft_k.reset();
  j = json::parse(FormatRequestLogLine(n));
  CHECK(j["generated_ids"].empty() && j["round_accepted"].empty() && j["draft_k"].is_null() && j["seed"].is_null());
}

// The writer measures each token-carrying line against the previous one's prompt + generated ids.
void TestTokensWriterDeltaChain() {
  const std::filesystem::path dir = std::filesystem::temp_directory_path() / "r4dx_request_log_test_tok";
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);
  const std::filesystem::path file = dir / "tok.jsonl";
  std::string err;
  const std::vector<int32_t> p1 = {1, 2, 3, 4, 5, 6};
  const std::vector<int32_t> g1 = {7, 8, 9};
  // 2: prompt 1 + generated 1 + new tokens (the common agent shape); 3: diverges inside prompt 2;
  // 4: unrelated, after a line that carries no tokens; 5: the same prompt again (the shared part is
  // capped by the shorter side, here the prompt), 6: the first line of a restarted server.
  const std::vector<int32_t> p2 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 30, 31};
  const std::vector<int32_t> g2 = {32, 33};
  const std::vector<int32_t> p3 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 30, 99, 98};
  const std::vector<int32_t> p4 = {500, 501};
  {
    auto log = RequestLog::Open(file.string(), &err);
    CHECK(log != nullptr, "%s", err.c_str());
    if (!log) return;
    log->Write(TokenRecord(p1, g1));
    log->Write(TokenRecord(p2, g2));
    log->Write(TokenRecord(p3, {}));
    log->Write(FullRecord());  // a reject / token-less line in between must not disturb the chain
    log->Write(TokenRecord(p4, {1}));
    log->Write(TokenRecord(p4, {1}));
    CHECK(log->lines_written() == 6 && log->write_failures() == 0);
  }
  {
    auto log = RequestLog::Open(file.string(), &err);  // a restart: the first line of the run is whole again
    CHECK(log != nullptr);
    if (!log) return;
    log->Write(TokenRecord(p4, {1}));
  }
  const auto lines = Lines(Slurp(file));
  CHECK(lines.size() == 7, "got %zu lines", lines.size());
  if (lines.size() != 7) return;
  const std::vector<size_t> shared = {0, 9, 10, SIZE_MAX, 0, 2, 0};  // line 3 carries no tokens
  // Replay it the way tools/ngram/sim_ngram.py does.
  std::vector<int32_t> prev;
  const std::vector<std::vector<int32_t>> want_prompt = {p1, p2, p3, {}, p4, p4, p4};
  for (size_t i = 0; i < lines.size(); ++i) {
    const json j = json::parse(lines[i]);
    if (shared[i] == SIZE_MAX) {
      CHECK(!j.contains("prompt_ids"));
      continue;
    }
    const size_t sh = j["prompt_shared"].get<size_t>();
    CHECK(sh == shared[i], "line %zu: prompt_shared %zu", i, sh);
    std::vector<int32_t> prompt(prev.begin(), prev.begin() + static_cast<ptrdiff_t>(sh));
    const std::vector<int32_t> tail = Ints(j["prompt_ids"]);
    prompt.insert(prompt.end(), tail.begin(), tail.end());
    CHECK(prompt == want_prompt[i], "line %zu: rebuilt prompt differs", i);
    prev = prompt;
    const std::vector<int32_t> gen = Ints(j["generated_ids"]);
    prev.insert(prev.end(), gen.begin(), gen.end());
  }
  std::filesystem::remove_all(dir);
}

// A failed write leaves no usable previous line: the next token line must carry its whole prompt.
void TestTokensWriteFailureResetsChain() {
  const std::filesystem::path dir = std::filesystem::temp_directory_path() / "r4dx_request_log_test_tokfail";
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);
  const std::filesystem::path file = dir / "ro.jsonl";
  { std::ofstream(file, std::ios::binary) << "existing\n"; }
  std::FILE* ro = std::fopen(file.string().c_str(), "rb");
  CHECK(ro != nullptr);
  if (!ro) return;
  auto log = RequestLog::FromStream(ro, file.string());
  bool threw = false;
  try {
    for (int i = 0; i < 2; ++i) log->Write(TokenRecord({1, 2, 3}, {4}));
  } catch (...) {
    threw = true;
  }
  CHECK(!threw && log->write_failures() == 2 && log->lines_written() == 0);
  log.reset();
  CHECK(Slurp(file) == "existing\n");
  std::filesystem::remove_all(dir);
}

std::string Slurp(const std::filesystem::path& p) {
  std::ifstream f(p, std::ios::binary);
  return std::string(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}

std::vector<std::string> Lines(const std::string& s) {
  std::vector<std::string> out;
  std::istringstream in(s);
  for (std::string l; std::getline(in, l);) out.push_back(l);
  return out;
}

void TestWriterAppendsWithoutBomAndTruncation() {
  const std::filesystem::path dir = std::filesystem::temp_directory_path() / "r4dx_request_log_test";
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);
  const std::filesystem::path file = dir / "reqs.jsonl";
  CHECK(!std::filesystem::exists(file));

  std::string err;
  {
    auto log = RequestLog::Open(file.string(), &err);
    CHECK(log != nullptr, "open of a missing file must create it: %s", err.c_str());
    if (!log) return;
    CHECK(std::filesystem::exists(file), "Open creates the file");
    RequestLogRecord r = FullRecord();
    r.timestamp.clear();  // the writer stamps it
    log->Write(r);
    // One fflush-ed line per request: readable while the log is still open.
    const std::string now = Slurp(file);
    CHECK(!now.empty() && now.back() == '\n', "the line is on disk before the log closes");
    CHECK(log->lines_written() == 1 && log->write_failures() == 0);
  }
  {
    auto log = RequestLog::Open(file.string(), &err);  // a second server run: append, never truncate
    CHECK(log != nullptr);
    if (!log) return;
    log->Write(FullRecord());
  }
  const std::string bytes = Slurp(file);
  CHECK(bytes.size() > 2 && bytes[0] == '{', "starts with the first JSON byte: no UTF-8 BOM");
  CHECK(bytes.find('\r') == std::string::npos, "lines end in a bare \\n, no CRLF translation");
  const auto lines = Lines(bytes);
  CHECK(lines.size() == 2, "two writes, two lines (got %zu)", lines.size());
  if (lines.size() == 2) {
    const json a = json::parse(lines[0]);
    const json b = json::parse(lines[1]);
    CHECK(a["ts"].is_string() && a["ts"].get<std::string>().size() == 29, "an empty ts is stamped with the clock");
    CHECK(b["ts"] == "2026-09-30T14:03:22.123+07:00");
  }

  // An unwritable path: a clear error, no log object.
  err.clear();
  auto bad = RequestLog::Open((dir / "no_such_dir" / "x.jsonl").string(), &err);
  CHECK(bad == nullptr && !err.empty() && err.find("no_such_dir") != std::string::npos, "err='%s'", err.c_str());
  err.clear();
  auto as_dir = RequestLog::Open(dir.string(), &err);  // a directory is not a file
  CHECK(as_dir == nullptr && !err.empty(), "err='%s'", err.c_str());
  std::filesystem::remove_all(dir);
}

void TestWriteFailureIsCountedNotFatal() {
  const std::filesystem::path dir = std::filesystem::temp_directory_path() / "r4dx_request_log_test_ro";
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);
  const std::filesystem::path file = dir / "ro.jsonl";
  { std::ofstream(file, std::ios::binary) << "existing\n"; }
  // A stream opened read-only: every fwrite fails. Write() must neither throw nor abort, and warns
  // (once, on stderr) -- the failure count is what the test can see.
  std::FILE* ro = std::fopen(file.string().c_str(), "rb");
  CHECK(ro != nullptr);
  if (!ro) return;
  auto log = RequestLog::FromStream(ro, file.string());
  bool threw = false;
  try {
    for (int i = 0; i < 3; ++i) log->Write(FullRecord());
  } catch (...) {
    threw = true;
  }
  CHECK(!threw, "a failing log write must not escape");
  CHECK(log->write_failures() == 3 && log->lines_written() == 0, "failures=%lld", (long long)log->write_failures());
  log.reset();
  CHECK(Slurp(file) == "existing\n", "the failed writes changed nothing");
  std::filesystem::remove_all(dir);
}

void TestConcurrentWritersKeepLinesWhole() {
  const std::filesystem::path dir = std::filesystem::temp_directory_path() / "r4dx_request_log_test_mt";
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);
  const std::filesystem::path file = dir / "mt.jsonl";
  std::string err;
  {
    auto log = RequestLog::Open(file.string(), &err);
    CHECK(log != nullptr, "%s", err.c_str());
    if (!log) return;
    constexpr int kThreads = 6, kPer = 100;
    std::vector<std::thread> threads;
    for (int t = 0; t < kThreads; ++t) {
      threads.emplace_back([&log, t] {
        for (int i = 0; i < kPer; ++i) {
          RequestLogRecord r = FullRecord();
          r.request_id = "t" + std::to_string(t) + "-" + std::to_string(i);
          log->Write(std::move(r));
        }
      });
    }
    for (auto& th : threads) th.join();
    CHECK(log->lines_written() == kThreads * kPer);
  }
  const auto lines = Lines(Slurp(file));
  CHECK(lines.size() == 600, "got %zu lines", lines.size());
  std::set<std::string> ids;
  for (const std::string& l : lines) {
    try {
      ids.insert(json::parse(l)["request_id"].get<std::string>());
    } catch (...) {
      CHECK(false, "a line is not whole JSON: %s", l.c_str());
      break;
    }
  }
  CHECK(ids.size() == 600, "every request id appears once (%zu)", ids.size());
  std::filesystem::remove_all(dir);
}

}  // namespace

int main() {
  TestFormatFullRecord();
  TestFormatUnknownsAreNull();
  TestFormatDerivedRates();
  TestNoFreeTextFields();
  TestIso8601();
  TestWriterAppendsWithoutBomAndTruncation();
  TestWriteFailureIsCountedNotFatal();
  TestConcurrentWritersKeepLinesWhole();
  TestTokensFormat();
  TestTokensWriterDeltaChain();
  TestTokensWriteFailureResetsChain();
  if (g_failures > 0) {
    std::fprintf(stderr, "%d check(s) failed\n", g_failures);
    return 1;
  }
  std::fprintf(stderr, "all request log checks passed\n");
  return 0;
}
