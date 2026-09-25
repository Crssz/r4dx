// tests/server/test_http_server.cpp -- http_server.cpp's routes end to end on CPU: a real Engine and
// HttpServer on a loopback port, driven by a scripted fake of the TP=1 model
// (EngineOptions::model_loader, as in test_engine_recovery.cpp), read back over real HTTP with
// httplib's own client. No GPU work: the fake never calls HIP (the binary links what r4dx-server
// links only because engine.cpp does). Needs the real tokenizer directory (R4DX_TOKENIZER_MODEL_DIR)
// and skips (77) without it -- after checking http_server.h's two content-type constants, which
// need no tokenizer.
//
// The regression it pins (docs/server.md's "Response shapes"): `smoke.ps1 -Model <v6> -Layers -1
// -Mtp 3` failed "thinking+tools: streamed content concatenation == non-streaming message.content"
// (283 vs 286). The server's bytes were identical on both paths. The answer carried a "×" and a "→",
// and the JSON response's bare `Content-Type: application/json` let Windows PowerShell 5.1 (the
// smoke's client) decode it as ISO-8859-1: 2 + 3 bytes became 5 characters instead of 2, while the
// SSE side, read as UTF-8, stayed intact. The fake replays that exact answer, through MTP-shaped
// multi-token rounds and through plain decode, behind the same thinking + tools request, and checks
// per route:
//   1. every JSON response and the SSE stream declare `charset=utf-8`;
//   2. the streamed content concatenation equals the non-streaming `message.content` as that
//      client reads it -- the JSON body decoded by its declared charset, ISO-8859-1 when it names
//      none (DecodeAsDeclared), the SSE body as UTF-8 (what smoke.ps1's Invoke-SseStream does);
//   3. byte for byte, both equal the scripted answer, and the reasoning halves agree -- the stream
//      assembly (ReasoningSplitter, ToolStreamGate) was never the difference, and must not become one.
// 1 and 2 fail with the bare content types; 3 holds either way.
#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "engine.h"
#include "httplib.h"
#include "http_server.h"
#include "model_config.h"
#include "model_types.h"
#include "nlohmann/json.hpp"
#include "text_model.h"
#include "tokenizer.h"

#ifndef R4DX_TOKENIZER_MODEL_DIR
#define R4DX_TOKENIZER_MODEL_DIR "C:/AI/models/Qwen3.8-27B"
#endif

namespace {

int g_failures = 0;

#define CHECK(cond, ...)                                                   \
  do {                                                                     \
    if (!(cond)) {                                                         \
      std::fprintf(stderr, "CHECK failed at %s:%d: %s -- ", __FILE__, __LINE__, #cond); \
      std::fprintf(stderr, __VA_ARGS__);                                   \
      std::fprintf(stderr, "\n");                                          \
      ++g_failures;                                                        \
    }                                                                      \
  } while (0)

using r4dx::model::ImageRows;
using r4dx::model::ImageSpan;
using r4dx::model::StepProfile;
using r4dx::model::VramReport;

constexpr int64_t kVocab = 248320;  // the checkpoint's vocabulary: the engine samples over logits.size()

// The real v6 TP=1 --mtp 3 generation for smoke.ps1's thinking+tools request (captured from
// r4dx-server, 2026-09-25): 58 reasoning tokens, then the 283-character answer with its "×"
// (U+00D7, 2 bytes) and "→" (U+2192, 3 bytes), spelled as UTF-8 escapes so the source encoding
// cannot change them.
constexpr char kReasoning[] =
    "The user is asking a simple math question: 12 plus 30. They want me to show my reasoning. This "
    "is a straightforward arithmetic problem. No tools needed.\n\n12 + 30 = 42.\n\nLet me show the "
    "reasoning clearly.";
constexpr char kAnswer[] =
    "**12 + 30 = 42**\n\nHere's the reasoning:\n\n1. Break 30 into 3 tens: 30 = 3 \xC3\x97 10\n2. Add "
    "the tens: 12 + 30 \xE2\x86\x92 think of 12 as 10 + 2\n3. So: (10 + 2) + 30 = 10 + 30 + 2 = 40 + "
    "2 = **42**\n\nAlternatively, just add the ones and tens separately:\n- Ones: 2 + 0 = 2\n- Tens: 1 "
    "+ 3 = 4\n- Result: 42";

// Replays one fixed token sequence, whatever the prompt: Prefill's logits pick script[0], and every
// later decode call hands out the next ones -- DecodeStepGreedy one at a time, DecodeStepMtpGreedy
// `round_sizes` at a time (cycled, capped at k + 1, like a real round's accepted drafts plus its
// bonus token). Past the end it repeats the script's last token, the EOS.
class ScriptedTextModel final : public r4dx::model::TextModel {
 public:
  // `round_sizes` empty: no MTP head (MtpEnabled() false, the engine's plain greedy loop).
  ScriptedTextModel(std::vector<int32_t> script, std::vector<int64_t> round_sizes)
      : script_(std::move(script)), round_sizes_(std::move(round_sizes)) {}

  const r4dx::model::ModelConfig& Config() const override { return cfg_; }
  const std::string& ModelId() const override { return id_; }
  int64_t ImageTokenId() const override { return 248056; }
  int64_t VisionMergeSize() const override { return 2; }
  bool HasVision() const override { return false; }
  bool MtpEnabled() const override { return !round_sizes_.empty(); }
  bool MtpUsingReducedVocabDraft() const override { return false; }
  bool DflashEnabled() const override { return false; }
  int64_t SampledFallbackRows() const override { return 0; }
  int64_t PositionCount() const override { return static_cast<int64_t>(pos_); }
  int64_t NumLoadedLayers() const override { return 4; }
  int TpWorld() const override { return 1; }
  std::vector<VramReport> Vram() const override { return {}; }
  void SetDflashInjectionEnabled(bool) override {}
  void Reset() override { pos_ = 0; }

  void EncodeImages(const float*, int64_t, const std::vector<r4dx::vision::GridThw>&, ImageRows*,
                    r4dx::vision::VisionEncodeStats*) override {
    throw std::logic_error("ScriptedTextModel: no vision");
  }
  // Every request starts the script over: the engine prefills each one (a full prompt after a
  // Reset(), or a reused prefix's new tail) before its first generated token.
  std::vector<float> Prefill(const std::vector<int32_t>&) override {
    pos_ = 0;
    rounds_ = 0;
    return Row(At(0));
  }
  std::vector<float> PrefillMultimodal(const std::vector<int32_t>& token_ids,
                                       const std::vector<ImageSpan>&) override {
    return Prefill(token_ids);
  }
  std::vector<float> DecodeStep(int32_t) override { return Row(At(++pos_)); }
  int32_t DecodeStepGreedy(int32_t) override { return At(++pos_); }
  int32_t DecodeStepSampled(int32_t, const r4dx::kernels::SampleParams&, std::mt19937_64&) override {
    throw std::logic_error("ScriptedTextModel: greedy requests only");
  }
  std::vector<int32_t> DecodeStepMtpGreedy(int32_t, int64_t k) override {
    if (round_sizes_.empty()) throw std::logic_error("ScriptedTextModel: no MTP head");
    const int64_t n = std::min<int64_t>(round_sizes_[rounds_++ % round_sizes_.size()], k + 1);
    std::vector<int32_t> round;
    for (int64_t i = 0; i < n; ++i) round.push_back(At(++pos_));
    return round;
  }
  std::vector<int32_t> DecodeStepMtpSampled(int32_t, int64_t, const r4dx::kernels::SampleParams&,
                                            std::mt19937_64&) override {
    throw std::logic_error("ScriptedTextModel: greedy requests only");
  }
  std::vector<int32_t> DecodeStepDflashGreedy(int32_t, int64_t, float, int64_t, int64_t*) override {
    throw std::logic_error("ScriptedTextModel: no drafter");
  }
  std::vector<int32_t> DecodeStepDflashSampled(int32_t, int64_t, float, int64_t, const r4dx::kernels::SampleParams&,
                                               std::mt19937_64&, int64_t*) override {
    throw std::logic_error("ScriptedTextModel: no drafter");
  }
  StepProfile DecodeStepProfiled(int32_t) override { throw std::logic_error("ScriptedTextModel: no profiling"); }
  StepProfile PrefillProfiled(const std::vector<int32_t>&) override {
    throw std::logic_error("ScriptedTextModel: no profiling");
  }

 private:
  int32_t At(size_t i) const { return script_[std::min(i, script_.size() - 1)]; }
  static std::vector<float> Row(int32_t tok) {
    std::vector<float> row(static_cast<size_t>(kVocab), 0.0f);
    row[static_cast<size_t>(tok)] = 1.0f;
    return row;
  }

  const std::vector<int32_t> script_;
  const std::vector<int64_t> round_sizes_;
  r4dx::model::ModelConfig cfg_;
  std::string id_ = "fake/scripted";
  size_t pos_ = 0;
  size_t rounds_ = 0;
};

// A real Engine + HttpServer listening on 127.0.0.1, torn down in the destructor.
class LiveServer {
 public:
  LiveServer(const std::string& tokenizer_dir, std::vector<int32_t> script, std::vector<int64_t> round_sizes) {
    r4dx::server::EngineOptions opts;
    opts.tokenizer_dir = tokenizer_dir;
    opts.model_opts.max_ctx = 4096;
    opts.model_opts.mtp_draft_k = round_sizes.empty() ? 0 : 3;  // the engine's round size, as --mtp 3
    opts.log_level = "warn";
    opts.model_loader = [script, round_sizes](const r4dx::model::ModelOptions&, const r4dx::model::TpOptions&) {
      return std::unique_ptr<r4dx::model::TextModel>(std::make_unique<ScriptedTextModel>(script, round_sizes));
    };
    engine_ = std::make_unique<r4dx::server::Engine>(std::move(opts));
    engine_->LoadAndStart();
    http_ = std::make_unique<r4dx::server::HttpServer>(*engine_);
    // A port the OS picks, never a fixed one: two concurrent runs of this test would otherwise share
    // it (HttpServer::BindToAnyPort's own comment).
    port_ = http_->BindToAnyPort("127.0.0.1");
    if (port_ < 0) throw std::runtime_error("HttpServer could not bind a loopback port");
    thread_ = std::thread([this] { http_->ListenAfterBind(); });
    httplib::Client probe("127.0.0.1", port_);
    for (int i = 0; i < 200; ++i) {
      const auto r = probe.Get("/health");
      if (r && r->status == 200) return;
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    // Stop() cannot be relied on to end a ListenAfterBind() that never started serving, so there is
    // no clean way back from here: fail the whole run at once.
    std::fprintf(stderr, "HttpServer on port %d did not answer /health within 10 s\n", port_);
    std::_Exit(1);
  }
  ~LiveServer() {
    http_->Stop();
    thread_.join();
    engine_->Shutdown();
  }
  LiveServer(const LiveServer&) = delete;
  LiveServer& operator=(const LiveServer&) = delete;

  httplib::Client Client() const {
    httplib::Client c("127.0.0.1", port_);
    c.set_read_timeout(120, 0);
    return c;
  }

 private:
  std::unique_ptr<r4dx::server::Engine> engine_;
  std::unique_ptr<r4dx::server::HttpServer> http_;
  std::thread thread_;
  int port_ = -1;
};

std::string Lower(std::string s) {
  for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return s;
}

bool DeclaresUtf8(const std::string& content_type) {
  return Lower(content_type).find("charset=utf-8") != std::string::npos;
}

// What a client that honours the declared charset, and falls back to ISO-8859-1 when none is named,
// makes of `body`: Windows PowerShell 5.1's Invoke-WebRequest / Invoke-RestMethod, smoke.ps1's own
// client (measured: a bare "application/json" body turns "×" into "Ã" + U+0097). Returned
// re-encoded as UTF-8, so it compares directly with a body read as UTF-8.
std::string DecodeAsDeclared(const std::string& body, const std::string& content_type) {
  if (DeclaresUtf8(content_type)) return body;
  std::string out;
  for (const char ch : body) {
    const unsigned char b = static_cast<unsigned char>(ch);
    if (b < 0x80) {
      out += ch;
    } else {  // Latin-1 byte b is code point U+00b: two UTF-8 bytes
      out += static_cast<char>(0xC0 | (b >> 6));
      out += static_cast<char>(0x80 | (b & 0x3F));
    }
  }
  return out;
}

// smoke.ps1's thinking+tools request ($thinkToolStreamBody / $thinkToolNonStreamBody).
nlohmann::json ThinkToolsBody(bool stream) {
  const nlohmann::json tool = {
      {"type", "function"},
      {"function",
       {{"name", "get_current_weather"},
        {"description", "Get the current weather for a location."},
        {"parameters",
         {{"type", "object"},
          {"properties", {{"location", {{"type", "string"}, {"description", "City and state"}}}}},
          {"required", {"location"}}}}}}};
  return {{"messages", {{{"role", "user"}, {"content", "What is 12 plus 30? Show your reasoning."}}}},
          {"tools", {tool}},
          {"chat_template_kwargs", {{"enable_thinking", true}}},
          {"max_tokens", 1024},
          {"temperature", 0},
          {"stream", stream}};
}

struct StreamedChat {
  std::string content, reasoning, finish_reason;
  int content_deltas = 0;
};

// Concatenates every content / reasoning_content delta of an SSE body read as UTF-8.
StreamedChat ParseSse(const std::string& body) {
  StreamedChat out;
  size_t pos = 0;
  while (pos < body.size()) {
    size_t end = body.find("\n\n", pos);
    if (end == std::string::npos) end = body.size();
    const std::string ev = body.substr(pos, end - pos);
    pos = end + 2;
    if (ev.rfind("data: ", 0) != 0) continue;
    const std::string data = ev.substr(6);
    if (data == "[DONE]") continue;
    const nlohmann::json c = nlohmann::json::parse(data);
    if (!c.contains("choices") || c["choices"].empty()) continue;
    const nlohmann::json& choice = c["choices"][0];
    if (choice.contains("finish_reason") && choice["finish_reason"].is_string()) {
      out.finish_reason = choice["finish_reason"].get<std::string>();
    }
    const nlohmann::json& delta = choice["delta"];
    if (delta.contains("content") && delta["content"].is_string()) {
      out.content += delta["content"].get<std::string>();
      ++out.content_deltas;
    }
    if (delta.contains("reasoning_content") && delta["reasoning_content"].is_string()) {
      out.reasoning += delta["reasoning_content"].get<std::string>();
    }
  }
  return out;
}

std::string Trim(const std::string& s) {
  const size_t b = s.find_first_not_of(" \t\r\n");
  if (b == std::string::npos) return "";
  return s.substr(b, s.find_last_not_of(" \t\r\n") - b + 1);
}

void ThinkToolsScenario(const std::string& tokenizer_dir, const std::vector<int32_t>& script,
                        const std::vector<int64_t>& round_sizes, const char* tag) {
  const int failures_before = g_failures;
  LiveServer server(tokenizer_dir, script, round_sizes);
  httplib::Client cli = server.Client();

  const auto s = cli.Post("/v1/chat/completions", ThinkToolsBody(true).dump(), "application/json");
  const auto n = cli.Post("/v1/chat/completions", ThinkToolsBody(false).dump(), "application/json");
  CHECK(s && s->status == 200, "[%s] streaming request failed", tag);
  CHECK(n && n->status == 200, "[%s] non-streaming request failed", tag);
  if (!s || !n || s->status != 200 || n->status != 200) return;

  const std::string sse_type = s->get_header_value("Content-Type");
  const std::string json_type = n->get_header_value("Content-Type");
  CHECK(Lower(sse_type).rfind("text/event-stream", 0) == 0 && DeclaresUtf8(sse_type),
        "[%s] SSE Content-Type '%s', want text/event-stream; charset=utf-8", tag, sse_type.c_str());
  CHECK(Lower(json_type).rfind("application/json", 0) == 0 && DeclaresUtf8(json_type),
        "[%s] JSON Content-Type '%s', want application/json; charset=utf-8", tag, json_type.c_str());

  const StreamedChat streamed = ParseSse(s->body);
  const nlohmann::json raw = nlohmann::json::parse(n->body);
  const nlohmann::json as_read = nlohmann::json::parse(DecodeAsDeclared(n->body, json_type));
  const std::string raw_content = raw["choices"][0]["message"]["content"].get<std::string>();
  const std::string read_content = as_read["choices"][0]["message"]["content"].get<std::string>();

  // 2. smoke.ps1's own check, as its client sees the two responses.
  CHECK(streamed.content == read_content,
        "[%s] streamed content (%zu bytes) != non-streaming message.content as a charset-honouring client "
        "reads it (%zu bytes)",
        tag, streamed.content.size(), read_content.size());
  // 3. The server's bytes: identical on both paths, and the scripted answer exactly.
  CHECK(streamed.content == raw_content && raw_content == kAnswer,
        "[%s] content bytes differ: streamed %zu, non-streaming %zu, scripted %zu", tag, streamed.content.size(),
        raw_content.size(), sizeof(kAnswer) - 1);
  CHECK(streamed.content_deltas > 5, "[%s] only %d content deltas: the answer did not stream live", tag,
        streamed.content_deltas);
  const std::string raw_reasoning = raw["choices"][0]["message"]["reasoning_content"].get<std::string>();
  CHECK(Trim(streamed.reasoning) == raw_reasoning && raw_reasoning == kReasoning,
        "[%s] reasoning differs: streamed '%s', non-streaming '%s'", tag, streamed.reasoning.c_str(),
        raw_reasoning.c_str());
  CHECK(streamed.finish_reason == "stop" && raw["choices"][0]["finish_reason"] == "stop",
        "[%s] finish_reason: streamed '%s', non-streaming '%s'", tag, streamed.finish_reason.c_str(),
        raw["choices"][0]["finish_reason"].dump().c_str());
  if (g_failures == failures_before) {
    std::printf("[%s] thinking+tools: %d content deltas, %zu content bytes, identical streamed and non-streamed, "
                "both declared UTF-8\n",
                tag, streamed.content_deltas, streamed.content.size());
  }
}

// 1. for the routes the scenario above does not reach: every JSON body this server writes, the
// error shape included, and the /v1/completions stream.
void OtherRoutesScenario(const std::string& tokenizer_dir, const std::vector<int32_t>& script) {
  const int failures_before = g_failures;
  LiveServer server(tokenizer_dir, script, {});
  httplib::Client cli = server.Client();
  auto expect = [](const httplib::Result& r, int status, const char* media, const char* what) {
    CHECK(r && r->status == status, "%s: no response or status %d, want %d", what, r ? r->status : -1, status);
    if (!r) return;
    const std::string type = r->get_header_value("Content-Type");
    CHECK(Lower(type).rfind(media, 0) == 0 && DeclaresUtf8(type), "%s: Content-Type '%s', want %s; charset=utf-8",
          what, type.c_str(), media);
  };
  expect(cli.Get("/health"), 200, "application/json", "GET /health");
  expect(cli.Get("/v1/models"), 200, "application/json", "GET /v1/models");
  expect(cli.Get("/v1/models/fake/scripted"), 200, "application/json", "GET /v1/models/{id}");
  expect(cli.Get("/v1/models/no-such-model"), 404, "application/json", "GET /v1/models/{unknown id} (error body)");
  expect(cli.Post("/v1/chat/completions", "{not json", "application/json"), 400, "application/json",
         "POST /v1/chat/completions, malformed body (error body)");
  const nlohmann::json completion = {{"prompt", "12 plus 30 is"}, {"max_tokens", 8}, {"temperature", 0}};
  nlohmann::json completion_stream = completion;
  completion_stream["stream"] = true;
  expect(cli.Post("/v1/completions", completion.dump(), "application/json"), 200, "application/json",
         "POST /v1/completions");
  expect(cli.Post("/v1/completions", completion_stream.dump(), "application/json"), 200, "text/event-stream",
         "POST /v1/completions, stream");
  if (g_failures == failures_before) {
    std::printf("[routes] /health, /v1/models, /v1/models/{id}, both error bodies and /v1/completions (JSON and "
                "SSE) all declare charset=utf-8\n");
  }
}

}  // namespace

int main() {
  // 1. for the two constants themselves, which need no tokenizer: a revert to a bare type fails
  // even where the live-server half below is skipped.
  const std::string json_type = r4dx::server::kJsonContentType;
  const std::string sse_type = r4dx::server::kSseContentType;
  CHECK(Lower(json_type).rfind("application/json", 0) == 0 && DeclaresUtf8(json_type),
        "kJsonContentType '%s', want application/json; charset=utf-8", json_type.c_str());
  CHECK(Lower(sse_type).rfind("text/event-stream", 0) == 0 && DeclaresUtf8(sse_type),
        "kSseContentType '%s', want text/event-stream; charset=utf-8", sse_type.c_str());

  const std::string tokenizer_dir = R4DX_TOKENIZER_MODEL_DIR;
  if (!std::filesystem::exists(std::filesystem::path(tokenizer_dir) / "tokenizer.json")) {
    if (g_failures > 0) {
      std::fprintf(stderr, "%d check(s) failed (the live-server half needs %s/tokenizer.json)\n", g_failures,
                   tokenizer_dir.c_str());
      return 1;
    }
    std::fprintf(stderr, "SKIP: %s/tokenizer.json not found (set -DR4DX_TOKENIZER_MODEL_DIR=...)\n",
                 tokenizer_dir.c_str());
    return 77;
  }
  try {
    r4dx::Tokenizer::Options tok_options;
    tok_options.allow_unimplemented_normalizer = true;  // the same flag Engine::LoadAndStart passes
    const r4dx::Tokenizer tok = r4dx::Tokenizer::from_directory(tokenizer_dir, tok_options);
    // "</think>" is a non-special added token: encode() yields its id and the server's
    // skip_special_tokens stream decoder renders it back, exactly as for a real generation.
    const std::string generated = std::string(kReasoning) + "\n</think>\n\n" + kAnswer;
    std::vector<int32_t> script;
    for (const r4dx::TokenId id : tok.encode(generated)) script.push_back(static_cast<int32_t>(id));
    script.push_back(static_cast<int32_t>(tok.eos_id()));

    // MTP-shaped delivery: rounds of 1-4 tokens (k = 3), the mix a real --mtp 3 run produces
    // (67 rounds for 208 tokens there), so rounds straddle "</think>" and both multi-byte
    // characters' tokens; then plain decode, one token per step.
    ThinkToolsScenario(tokenizer_dir, script, {4, 2, 1, 3, 4, 4, 1, 2, 3}, "mtp k=3");
    ThinkToolsScenario(tokenizer_dir, script, {}, "plain");
    OtherRoutesScenario(tokenizer_dir, script);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "unexpected exception: %s\n", e.what());
    return 1;
  }
  if (g_failures > 0) {
    std::fprintf(stderr, "%d check(s) failed\n", g_failures);
    return 1;
  }
  std::fprintf(stderr, "all http server checks passed\n");
  return 0;
}
