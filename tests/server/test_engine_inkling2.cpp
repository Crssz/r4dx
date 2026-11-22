// tests/server/test_engine_inkling2.cpp -- the Inkling-2 (structured_parts) output path through engine.cpp and http_server.cpp on CPU
// (docs/inkling2-plan.md section 6, lane E2): a real Engine + HttpServer on a loopback port, a scripted CPU fake of the TP=1 model
// that replays fixed token ids (tests/server/test_http_server.cpp's pattern), and the checkpoint's real tokenizer + chat template
// (R4DX_INKLING2_TOKENIZER_DIR; exits 77 = SKIPPED when absent). No GPU work: the fake never calls HIP.
//
// Every scenario's script is the generation as TOKENS (the part markers are real special-token ids, encoded with parse_special),
// ending in the EOS <|content_model_end_sampling|> (200006), and is read back both as one JSON response and as an SSE stream:
//   * thinking + text: reasoning_content / content split, usage.reasoning_tokens counted from the thinking part, streamed at
//     token granularity (plain decode) and in MTP-shaped multi-token rounds;
//   * one tool call, two tool calls (index/id/arguments), finish_reason "tool_calls", content null;
//   * a malformed call (degrades to content, finish "stop"), an undefined tool name (dropped, noted);
//   * max_tokens truncation inside a thinking part, a text part and a tool-call part (finish "length");
//   * reasoning_effort: a level name and the equal number render the same prompt, a number outside [0, 0.99] and an unknown name
//     are 400s, enable_thinking=false renders effort "none".
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <mutex>
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

#ifndef R4DX_INKLING2_TOKENIZER_DIR
#define R4DX_INKLING2_TOKENIZER_DIR ""
#endif

namespace {

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

using json = nlohmann::json;
using r4dx::model::ImageRows;
using r4dx::model::ImageSpan;
using r4dx::model::StepProfile;
using r4dx::model::VramReport;

constexpr int64_t kVocab = 200064;  // the checkpoint's stored lm_head rows: the engine samples over logits.size()
constexpr int32_t kEos = 200006;    // <|content_model_end_sampling|>

// What the fake replays and what it was last prompted with (the engine's worker thread reads/writes, the test thread sets it
// between requests, which are strictly sequential).
struct Shared {
  std::mutex mu;
  std::vector<int32_t> script;
  std::vector<int32_t> last_prompt;
  void SetScript(std::vector<int32_t> s) {
    std::lock_guard<std::mutex> lock(mu);
    script = std::move(s);
  }
  std::vector<int32_t> Script() {
    std::lock_guard<std::mutex> lock(mu);
    return script;
  }
  void SetPrompt(const std::vector<int32_t>& p) {
    std::lock_guard<std::mutex> lock(mu);
    last_prompt = p;
  }
  std::vector<int32_t> Prompt() {
    std::lock_guard<std::mutex> lock(mu);
    return last_prompt;
  }
};

// Replays Shared::script whatever the prompt (see test_http_server.cpp's ScriptedTextModel): Prefill's logits pick script[0], every
// later decode call the next ones -- one per step, or `round_sizes` at a time on the MTP path. Past the end it repeats the last
// token (the EOS).
class ScriptedTextModel final : public r4dx::model::TextModel {
 public:
  ScriptedTextModel(std::shared_ptr<Shared> shared, std::vector<int64_t> round_sizes)
      : shared_(std::move(shared)), round_sizes_(std::move(round_sizes)) {
#ifdef R4DX_TEST_INKLING2_ARCH
    cfg_.arch = R4DX_TEST_INKLING2_ARCH;
#endif
  }

  const r4dx::model::ModelConfig& Config() const override { return cfg_; }
  const std::string& ModelId() const override { return id_; }
  int64_t ImageTokenId() const override { return -1; }
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
  void SaveCheckpoint() override { throw std::logic_error("ScriptedTextModel: no checkpoint"); }
  void RestoreCheckpoint() override { throw std::logic_error("ScriptedTextModel: no checkpoint"); }

  void EncodeImages(const float*, int64_t, const std::vector<r4dx::vision::GridThw>&, ImageRows*,
                    r4dx::vision::VisionEncodeStats*) override {
    throw std::logic_error("ScriptedTextModel: no vision");
  }
  std::vector<float> Prefill(const std::vector<int32_t>& ids) override {
    shared_->SetPrompt(ids);
    script_ = shared_->Script();
    pos_ = 0;
    rounds_ = 0;
    return Row(At(0));
  }
  std::vector<float> PrefillMultimodal(const std::vector<int32_t>& token_ids, const std::vector<ImageSpan>&) override {
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

  std::shared_ptr<Shared> shared_;
  std::vector<int32_t> script_{kEos};
  const std::vector<int64_t> round_sizes_;
  r4dx::model::ModelConfig cfg_;
  std::string id_ = "fake/inkling2";
  size_t pos_ = 0;
  size_t rounds_ = 0;
};

class LiveServer {
 public:
  LiveServer(const std::string& tokenizer_dir, std::shared_ptr<Shared> shared, std::vector<int64_t> round_sizes)
      : shared_(shared) {
    r4dx::server::EngineOptions opts;
    opts.tokenizer_dir = tokenizer_dir;
    opts.dialect = r4dx::server::DialectKind::kInkling2;
    opts.model_opts.max_ctx = 4096;
    opts.model_opts.mtp_draft_k = round_sizes.empty() ? 0 : 3;
    opts.log_level = "warn";
    opts.model_loader = [shared, round_sizes](const r4dx::model::ModelOptions&, const r4dx::model::TpOptions&) {
      return std::unique_ptr<r4dx::model::TextModel>(std::make_unique<ScriptedTextModel>(shared, round_sizes));
    };
    engine_ = std::make_unique<r4dx::server::Engine>(std::move(opts));
    engine_->LoadAndStart();
    http_ = std::make_unique<r4dx::server::HttpServer>(*engine_);
    port_ = http_->BindToAnyPort("127.0.0.1");
    if (port_ < 0) throw std::runtime_error("HttpServer could not bind a loopback port");
    thread_ = std::thread([this] { http_->ListenAfterBind(); });
    httplib::Client probe("127.0.0.1", port_);
    for (int i = 0; i < 200; ++i) {
      const auto r = probe.Get("/health");
      if (r && r->status == 200) return;
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
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
  std::shared_ptr<Shared> shared_;
  std::unique_ptr<r4dx::server::Engine> engine_;
  std::unique_ptr<r4dx::server::HttpServer> http_;
  std::thread thread_;
  int port_ = -1;
};

// ---- the generations (text with part markers, encoded to token ids) ------------------------------------------------------------

constexpr char kMsg[] = "<|message_model|>";
constexpr char kThink[] = "<|content_thinking|>";
constexpr char kText[] = "<|content_text|>";
constexpr char kTool[] = "<|content_invoke_tool_json|>";
constexpr char kEnd[] = "<|end_message|>";

std::string Thinking(const std::string& body) { return std::string(kMsg) + kThink + body + kEnd; }
std::string Answer(const std::string& body) { return std::string(kMsg) + kText + body + kEnd; }
std::string Call(const std::string& body) { return std::string(kMsg) + kTool + body + kEnd; }

const r4dx::Tokenizer* g_tok = nullptr;

std::vector<int32_t> Ids(const std::string& text) {
  std::vector<int32_t> ids;
  for (const r4dx::TokenId id : g_tok->encode(text, /*parse_special=*/true)) ids.push_back(static_cast<int32_t>(id));
  return ids;
}
std::vector<int32_t> WithEos(std::vector<int32_t> ids) {
  ids.push_back(kEos);
  return ids;
}

// ---- requests ----------------------------------------------------------------------------------------------------------------

json ChatBody(bool stream, int64_t max_tokens = 256, const json& tools = nullptr, const json& extra = json::object()) {
  json b = {{"messages", {{{"role", "user"}, {"content", "What is 12 plus 30?"}}}},
            {"max_tokens", max_tokens},
            {"temperature", 0},
            {"stream", stream}};
  if (stream) b["stream_options"] = {{"include_usage", true}};
  if (!tools.is_null()) b["tools"] = tools;
  for (auto it = extra.begin(); it != extra.end(); ++it) b[it.key()] = it.value();
  return b;
}

json Tools(const std::vector<std::string>& names) {
  json tools = json::array();
  for (const std::string& n : names) {
    tools.push_back({{"type", "function"},
                     {"function",
                      {{"name", n},
                       {"description", "a tool"},
                       {"parameters", {{"type", "object"}, {"properties", {{"city", {{"type", "string"}}}}}}}}}});
  }
  return tools;
}

struct Reply {
  int status = 0;
  json message;  // non-streaming: choices[0].message; streaming: assembled from the deltas
  std::string finish_reason;
  json usage;
  // streaming only
  int reasoning_deltas = 0, content_deltas = 0, tool_call_deltas = 0;
  bool marker_in_delta = false;
};

bool HasMarker(const std::string& s) { return s.find("<|") != std::string::npos; }

Reply PostChat(httplib::Client& cli, const json& body) {
  Reply r;
  const auto res = cli.Post("/v1/chat/completions", body.dump(), "application/json");
  if (!res) return r;
  r.status = res->status;
  if (res->status != 200) {
    r.message = json::parse(res->body, nullptr, false);
    return r;
  }
  if (!body.value("stream", false)) {
    const json j = json::parse(res->body);
    r.message = j["choices"][0]["message"];
    r.finish_reason = j["choices"][0]["finish_reason"].get<std::string>();
    r.usage = j["usage"];
    return r;
  }
  r.message = json::object();
  std::string reasoning, content;
  json calls = json::array();
  size_t pos = 0;
  const std::string& sse = res->body;
  while (pos < sse.size()) {
    size_t end = sse.find("\n\n", pos);
    if (end == std::string::npos) end = sse.size();
    const std::string ev = sse.substr(pos, end - pos);
    pos = end + 2;
    if (ev.rfind("data: ", 0) != 0 || ev == "data: [DONE]") continue;
    const json c = json::parse(ev.substr(6));
    if (c.contains("usage") && c["usage"].is_object()) r.usage = c["usage"];
    if (!c.contains("choices") || c["choices"].empty()) continue;
    const json& choice = c["choices"][0];
    if (choice.contains("finish_reason") && choice["finish_reason"].is_string()) {
      r.finish_reason = choice["finish_reason"].get<std::string>();
    }
    const json& d = choice["delta"];
    if (d.contains("reasoning_content") && d["reasoning_content"].is_string()) {
      reasoning += d["reasoning_content"].get<std::string>();
      ++r.reasoning_deltas;
      if (HasMarker(d["reasoning_content"].get<std::string>())) r.marker_in_delta = true;
      if (d.contains("content")) r.marker_in_delta = true;  // never both keys in one delta
    }
    if (d.contains("content") && d["content"].is_string()) {
      content += d["content"].get<std::string>();
      ++r.content_deltas;
      if (HasMarker(d["content"].get<std::string>())) r.marker_in_delta = true;
    }
    if (d.contains("tool_calls")) {
      ++r.tool_call_deltas;
      for (const json& tc : d["tool_calls"]) calls.push_back(tc);
    }
  }
  if (!reasoning.empty()) r.message["reasoning_content"] = reasoning;
  r.message["content"] = content.empty() ? json(nullptr) : json(content);
  if (!calls.empty()) r.message["tool_calls"] = calls;
  return r;
}

std::string Str(const json& j, const char* key) {
  return j.contains(key) && j[key].is_string() ? j[key].get<std::string>() : std::string("<absent>");
}

// ---- scenarios -----------------------------------------------------------------------------------------------------------------

void ThinkingTextScenario(const std::string& dir, const std::vector<int64_t>& rounds, const char* tag) {
  auto shared = std::make_shared<Shared>();
  LiveServer server(dir, shared, rounds);
  httplib::Client cli = server.Client();
  const std::string think = "Let me add twelve and thirty. 12 + 30 = 42.";
  const std::string answer = "The answer is **42**.\n\nA second paragraph.";
  const std::string head = Thinking(think);
  shared->SetScript(WithEos(Ids(head + Answer(answer))));
  // A token is a reasoning token when it opens, sits in, or closes the thinking part: every token of the first part except the
  // leading <|message_model|> (which only opens the header).
  const int64_t want_reasoning_tokens = static_cast<int64_t>(Ids(head).size()) - 1;
  const int64_t want_completion_tokens = static_cast<int64_t>(Ids(head + Answer(answer)).size());

  const Reply n = PostChat(cli, ChatBody(false));
  CHECK(n.status == 200, "[%s] non-streaming status %d", tag, n.status);
  CHECK(Str(n.message, "reasoning_content") == think, "[%s] reasoning_content '%s'", tag, Str(n.message, "reasoning_content").c_str());
  CHECK(Str(n.message, "content") == answer, "[%s] content '%s'", tag, Str(n.message, "content").c_str());
  CHECK(!n.message.contains("tool_calls"), "[%s] stray tool_calls", tag);
  CHECK(n.finish_reason == "stop", "[%s] finish_reason %s", tag, n.finish_reason.c_str());
  CHECK(n.usage.value("completion_tokens", -1) == want_completion_tokens, "[%s] completion_tokens %s", tag, n.usage.dump().c_str());
  CHECK(n.usage.contains("completion_tokens_details") &&
            n.usage["completion_tokens_details"].value("reasoning_tokens", -1) == want_reasoning_tokens,
        "[%s] reasoning_tokens want %lld got %s", tag, static_cast<long long>(want_reasoning_tokens), n.usage.dump().c_str());

  const Reply s = PostChat(cli, ChatBody(true));
  CHECK(s.status == 200, "[%s] streaming status %d", tag, s.status);
  CHECK(Str(s.message, "reasoning_content") == think, "[%s] streamed reasoning '%s'", tag, Str(s.message, "reasoning_content").c_str());
  CHECK(Str(s.message, "content") == answer, "[%s] streamed content '%s'", tag, Str(s.message, "content").c_str());
  CHECK(s.finish_reason == "stop", "[%s] streamed finish_reason %s", tag, s.finish_reason.c_str());
  CHECK(!s.marker_in_delta, "[%s] a part marker leaked into a delta", tag);
  // Token granularity: plain decode forwards each token's text on its own, so both halves arrive in many deltas; an MTP-shaped
  // round carries several tokens per push but still splits at the part boundary.
  CHECK(s.reasoning_deltas >= (rounds.empty() ? 5 : 2), "[%s] only %d reasoning deltas", tag, s.reasoning_deltas);
  CHECK(s.content_deltas >= (rounds.empty() ? 5 : 2), "[%s] only %d content deltas", tag, s.content_deltas);
  CHECK(s.usage.contains("completion_tokens_details") &&
            s.usage["completion_tokens_details"].value("reasoning_tokens", -1) == want_reasoning_tokens,
        "[%s] streamed usage %s", tag, s.usage.dump().c_str());

  // reasoning.exclude: the thought is withheld, content is still the answer alone, the tokens are still reported.
  const Reply x = PostChat(cli, ChatBody(false, 256, nullptr, {{"reasoning", {{"exclude", true}}}}));
  CHECK(x.status == 200 && !x.message.contains("reasoning_content") && Str(x.message, "content") == answer,
        "[%s] reasoning.exclude: %s", tag, x.message.dump().c_str());
  CHECK(x.usage["completion_tokens_details"].value("reasoning_tokens", -1) == want_reasoning_tokens, "[%s] excluded usage", tag);
  std::fprintf(stderr, "[thinking+text %s] ok\n", tag);
}

void ToolCallScenarios(const std::string& dir) {
  auto shared = std::make_shared<Shared>();
  LiveServer server(dir, shared, {});
  httplib::Client cli = server.Client();
  const json tools = Tools({"get_weather", "get_time"});

  // One call after a thinking part.
  shared->SetScript(WithEos(Ids(Thinking("I should call the tool.") + Call(R"({"name":"get_weather","args":{"city":"Paris","unit":"c"}})"))));
  for (const bool stream : {false, true}) {
    const Reply r = PostChat(cli, ChatBody(stream, 256, tools));
    const char* tag = stream ? "one call, stream" : "one call";
    CHECK(r.status == 200 && r.finish_reason == "tool_calls", "[%s] status %d finish %s", tag, r.status, r.finish_reason.c_str());
    CHECK(r.message.contains("tool_calls") && r.message["tool_calls"].size() == 1, "[%s] tool_calls %s", tag, r.message.dump().c_str());
    if (r.message.contains("tool_calls") && r.message["tool_calls"].size() == 1) {
      const json& f = r.message["tool_calls"][0]["function"];
      CHECK(f.value("name", "") == "get_weather", "[%s] name %s", tag, f.dump().c_str());
      // `arguments` is the compact JSON string of the args object, key order kept.
      CHECK(f.value("arguments", "") == R"({"city":"Paris","unit":"c"})", "[%s] arguments %s", tag, f.dump().c_str());
      CHECK(Str(r.message["tool_calls"][0], "id").rfind("call_", 0) == 0, "[%s] id", tag);
    }
    CHECK(r.message["content"].is_null(), "[%s] content should be null: %s", tag, r.message.dump().c_str());
    CHECK(Str(r.message, "reasoning_content") == "I should call the tool.", "[%s] reasoning %s", tag, r.message.dump().c_str());
    CHECK(r.usage["completion_tokens_details"].value("reasoning_tokens", -1) ==
              static_cast<int64_t>(Ids(Thinking("I should call the tool.")).size()) - 1,
          "[%s] reasoning_tokens %s", tag, r.usage.dump().c_str());
  }

  // Two calls, with answer text between and after.
  shared->SetScript(WithEos(Ids(Answer("Checking both.") + Call(R"({"name":"get_weather","args":{"city":"Paris"}})") +
                                Call(R"({"name":"get_time","args":{}})") + Answer("Done."))));
  for (const bool stream : {false, true}) {
    const Reply r = PostChat(cli, ChatBody(stream, 256, tools));
    const char* tag = stream ? "two calls, stream" : "two calls";
    CHECK(r.status == 200 && r.finish_reason == "tool_calls", "[%s] status %d finish %s", tag, r.status, r.finish_reason.c_str());
    CHECK(Str(r.message, "content") == "Checking both.Done.", "[%s] content %s", tag, r.message.dump().c_str());
    const bool two = r.message.contains("tool_calls") && r.message["tool_calls"].size() == 2;
    CHECK(two, "[%s] tool_calls %s", tag, r.message.dump().c_str());
    if (two) {
      const json& a = r.message["tool_calls"][0];
      const json& b = r.message["tool_calls"][1];
      CHECK(a["function"].value("name", "") == "get_weather" && b["function"].value("name", "") == "get_time", "[%s] names", tag);
      CHECK(b["function"].value("arguments", "") == "{}", "[%s] empty args %s", tag, b.dump().c_str());
      CHECK(Str(a, "id") != Str(b, "id"), "[%s] ids must differ", tag);
      if (stream) CHECK(a.value("index", -1) == 0 && b.value("index", -1) == 1, "[%s] indexes %s", tag, r.message.dump().c_str());
    }
  }

  // A call to a tool the request did not offer: dropped and noted as content, no tool_calls, finish "stop".
  shared->SetScript(WithEos(Ids(Call(R"({"name":"launch","args":{"x":1}})"))));
  for (const bool stream : {false, true}) {
    const Reply r = PostChat(cli, ChatBody(stream, 256, tools));
    const char* tag = stream ? "undefined tool, stream" : "undefined tool";
    CHECK(r.status == 200 && r.finish_reason == "stop" && !r.message.contains("tool_calls"), "[%s] %s finish %s", tag,
          r.message.dump().c_str(), r.finish_reason.c_str());
    CHECK(Str(r.message, "content") == R"([dropped call to undefined tool 'launch': {"x":1}])", "[%s] content %s", tag,
          r.message.dump().c_str());
  }

  // A malformed call (not an object with a name/args) degrades to its raw source text as content.
  const std::string bad = Call("not json at all");
  shared->SetScript(WithEos(Ids(bad)));
  for (const bool stream : {false, true}) {
    const Reply r = PostChat(cli, ChatBody(stream, 256, tools));
    const char* tag = stream ? "malformed call, stream" : "malformed call";
    CHECK(r.status == 200 && r.finish_reason == "stop" && !r.message.contains("tool_calls"), "[%s] %s finish %s", tag,
          r.message.dump().c_str(), r.finish_reason.c_str());
    CHECK(Str(r.message, "content") == bad, "[%s] content '%s'", tag, r.message.dump().c_str());
  }
  std::fprintf(stderr, "[tool calls] one, two, undefined, malformed: ok\n");
}

void TruncationScenarios(const std::string& dir) {
  auto shared = std::make_shared<Shared>();
  LiveServer server(dir, shared, {});
  httplib::Client cli = server.Client();
  const json tools = Tools({"get_weather"});

  // max_tokens inside a thinking part: what was generated is reasoning_content, content stays empty, finish "length".
  const std::string long_think = "one two three four five six seven eight nine ten eleven twelve";
  shared->SetScript(WithEos(Ids(Thinking(long_think) + Answer("never reached"))));
  for (const bool stream : {false, true}) {
    const Reply r = PostChat(cli, ChatBody(stream, 8));
    const char* tag = stream ? "cut in thinking, stream" : "cut in thinking";
    CHECK(r.status == 200 && r.finish_reason == "length", "[%s] status %d finish %s", tag, r.status, r.finish_reason.c_str());
    const std::string rc = Str(r.message, "reasoning_content");
    CHECK(rc != "<absent>" && !rc.empty() && long_think.rfind(rc, 0) == 0 && rc.size() < long_think.size(),
          "[%s] reasoning '%s'", tag, rc.c_str());
    CHECK(r.message["content"].is_null() || Str(r.message, "content").empty(), "[%s] content %s", tag, r.message.dump().c_str());
    CHECK(r.usage.value("completion_tokens", -1) == 8, "[%s] completion_tokens %s", tag, r.usage.dump().c_str());
    // 8 tokens generated: <|message_model|> (the header, not reasoning) then <|content_thinking|> + 6 body tokens.
    CHECK(r.usage["completion_tokens_details"].value("reasoning_tokens", -1) == 7, "[%s] 7 of the 8 are reasoning: %s", tag,
          r.usage.dump().c_str());
  }

  // max_tokens inside a text part: the answer so far is content.
  const std::string long_text = "alpha beta gamma delta epsilon zeta eta theta iota kappa";
  shared->SetScript(WithEos(Ids(Answer(long_text))));
  for (const bool stream : {false, true}) {
    const Reply r = PostChat(cli, ChatBody(stream, 9));
    const char* tag = stream ? "cut in text, stream" : "cut in text";
    const std::string c = Str(r.message, "content");
    CHECK(r.status == 200 && r.finish_reason == "length" && c != "<absent>" && !c.empty() && long_text.rfind(c, 0) == 0 &&
              c.size() < long_text.size(),
          "[%s] finish %s content '%s'", tag, r.finish_reason.c_str(), c.c_str());
    CHECK(!r.message.contains("reasoning_content") || Str(r.message, "reasoning_content").empty(), "[%s] stray reasoning", tag);
  }

  // max_tokens inside a tool-call part: the call never completed, so it degrades to its raw source text as content and no
  // tool_calls / finish "tool_calls" appears.
  const std::string call = Call(R"({"name":"get_weather","args":{"city":"Paris","note":"a rather long argument value"}})");
  shared->SetScript(WithEos(Ids(call)));
  for (const bool stream : {false, true}) {
    const Reply r = PostChat(cli, ChatBody(stream, 14, tools));
    const char* tag = stream ? "cut in call, stream" : "cut in call";
    const std::string c = Str(r.message, "content");
    CHECK(r.status == 200 && r.finish_reason == "length" && !r.message.contains("tool_calls"), "[%s] %s finish %s", tag,
          r.message.dump().c_str(), r.finish_reason.c_str());
    CHECK(c.rfind(std::string(kMsg) + kTool, 0) == 0 && call.rfind(c, 0) == 0 && c.size() < call.size(), "[%s] content '%s'", tag,
          c.c_str());
  }
  std::fprintf(stderr, "[truncation] inside thinking / text / tool call: ok\n");
}

// The prompt the fake last saw, decoded with the special tokens kept.
std::string LastPrompt(Shared& shared) {
  std::vector<r4dx::TokenId> ids;
  for (const int32_t id : shared.Prompt()) ids.push_back(static_cast<r4dx::TokenId>(id));
  return g_tok->decode(ids, /*skip_special_tokens=*/false);
}

void EffortScenarios(const std::string& dir) {
  auto shared = std::make_shared<Shared>();
  LiveServer server(dir, shared, {});
  httplib::Client cli = server.Client();
  shared->SetScript(WithEos(Ids(Answer("ok"))));
  const auto prompt_for = [&](const json& extra, int expect_status) {
    const Reply r = PostChat(cli, ChatBody(false, 16, nullptr, extra));
    CHECK(r.status == expect_status, "request %s: status %d, want %d (%s)", extra.dump().c_str(), r.status, expect_status,
          r.message.dump().c_str());
    return r.status == 200 ? shared->Prompt() : std::vector<int32_t>{};
  };

  const auto base = prompt_for(json::object(), 200);  // no control: the template's default 0.9
  CHECK(LastPrompt(*shared).find("Thinking effort level:") != std::string::npos, "no effort message in the prompt");
  // "high" is 0.9 in the template's own table; the number 0.9 and the default render the same message.
  CHECK(prompt_for({{"reasoning_effort", "high"}}, 200) == base, "reasoning_effort high differs from the default (0.9)");
  CHECK(prompt_for({{"reasoning_effort", 0.9}}, 200) == base, "numeric reasoning_effort 0.9 differs from the default");
  CHECK(prompt_for({{"reasoning", {{"effort", "HIGH"}}}}, 200) == base, "reasoning.effort HIGH differs from the default");
  const auto low = prompt_for({{"reasoning_effort", "low"}}, 200);
  CHECK(low != base && prompt_for({{"reasoning_effort", 0.2}}, 200) == low, "low (0.2) vs 0.2 vs default");
  const auto none = prompt_for({{"reasoning_effort", "none"}}, 200);
  CHECK(none != base, "effort none renders like the default");
  CHECK(prompt_for({{"reasoning_effort", 0}}, 200) == none, "numeric 0 differs from none");
  CHECK(prompt_for({{"enable_thinking", false}}, 200) == none, "enable_thinking=false differs from effort none");
  CHECK(prompt_for({{"chat_template_kwargs", {{"enable_thinking", false}}}}, 200) == none, "ctk enable_thinking=false");
  // An explicit template variable wins over the request field.
  CHECK(prompt_for({{"chat_template_kwargs", {{"reasoning_effort", "low"}}}, {"reasoning_effort", "high"}}, 200) == low,
        "chat_template_kwargs.reasoning_effort must win");
  // Out-of-range / unknown / wrong-typed efforts are 400s.
  prompt_for({{"reasoning_effort", 1.5}}, 400);
  prompt_for({{"reasoning_effort", -0.1}}, 400);
  prompt_for({{"reasoning_effort", "bogus"}}, 400);
  prompt_for({{"reasoning_effort", true}}, 400);
  std::fprintf(stderr, "[reasoning effort] levels, numbers, none / enable_thinking, 400s: ok\n");
}

}  // namespace

int main() {
  const std::string tokenizer_dir = R4DX_INKLING2_TOKENIZER_DIR;
  if (tokenizer_dir.empty() || !std::filesystem::exists(std::filesystem::path(tokenizer_dir) / "tokenizer.json") ||
      !std::filesystem::exists(std::filesystem::path(tokenizer_dir) / "chat_template.jinja")) {
    std::fprintf(stderr, "SKIP: Inkling-2 tokenizer directory '%s' has no tokenizer.json / chat_template.jinja\n",
                 tokenizer_dir.c_str());
    return 77;
  }
  try {
    r4dx::Tokenizer::Options tok_options;
    tok_options.allow_unimplemented_normalizer = true;
    tok_options.keep_special_on_decode = {kMsg, kThink, kText, kTool, kEnd};
    const r4dx::Tokenizer tok = r4dx::Tokenizer::from_directory(tokenizer_dir, tok_options);
    g_tok = &tok;
    ThinkingTextScenario(tokenizer_dir, {}, "plain");
    ThinkingTextScenario(tokenizer_dir, {4, 2, 1, 3, 4, 4, 1, 2, 3}, "mtp k=3");
    ToolCallScenarios(tokenizer_dir);
    TruncationScenarios(tokenizer_dir);
    EffortScenarios(tokenizer_dir);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "unexpected exception: %s\n", e.what());
    return 1;
  }
  if (g_failures > 0) {
    std::fprintf(stderr, "%d check(s) failed\n", g_failures);
    return 1;
  }
  std::fprintf(stderr, "all inkling2 engine checks passed\n");
  return 0;
}
