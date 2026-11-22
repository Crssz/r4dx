// tests/server/test_inkling2_dialect.cpp -- the Inkling-2 server dialect (docs/inkling2-plan.md section 6): dialect table,
// auto-detection, arch cross-check, the typed-part stream parser (token-level replay vectors, every chunk boundary), the
// request-side normalisation, and -- when the checkpoint's tokenizer files exist -- the same parser driven by real token ids
// through the real tokenizer + chat template. CPU only.
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "chat_template.h"
#include "dialect.h"
#include "inkling2_stream_parser.h"
#include "nlohmann/json.hpp"
#include "tokenizer.h"
#include "tool_call_parser.h"

namespace {

int g_failures = 0;

#define CHECK(cond)                                                                   \
  do {                                                                                \
    if (!(cond)) {                                                                    \
      std::fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      ++g_failures;                                                                   \
    }                                                                                 \
  } while (0)

#define CHECK_EQ(a, b)                                                                                              \
  do {                                                                                                              \
    const auto& va = (a);                                                                                           \
    const auto& vb = (b);                                                                                           \
    if (!(va == vb)) {                                                                                              \
      std::fprintf(stderr, "CHECK_EQ failed at %s:%d: %s == %s\n  left : %s\n  right: %s\n", __FILE__, __LINE__, #a, #b, \
                   Show(va).c_str(), Show(vb).c_str());                                                             \
      ++g_failures;                                                                                                 \
    }                                                                                                               \
  } while (0)

std::string Show(const std::string& s) { return "\"" + s + "\""; }
std::string Show(bool b) { return b ? "true" : "false"; }
std::string Show(size_t n) { return std::to_string(n); }
std::string Show(const r4dx::server::DialectKind k) { return std::to_string(static_cast<int>(k)); }

using namespace r4dx::server;
using Event = Inkling2StreamParser::Event;

const std::string M = "<|message_model|>", T = "<|content_thinking|>", X = "<|content_text|>",
                  J = "<|content_invoke_tool_json|>", E = "<|end_message|>";

std::string Join(const std::vector<std::string>& tokens) {
  std::string s;
  for (const auto& t : tokens) s += t;
  return s;
}

struct Result {
  std::string reasoning, content;
  std::vector<std::pair<std::string, std::string>> calls;  // name, arguments_json
  bool malformed = false;
  size_t call_count = 0;
  bool seen = false;
  std::string event_kinds;  // "R", "C", "T" per event, in order
  bool operator==(const Result& o) const {
    return reasoning == o.reasoning && content == o.content && calls == o.calls && malformed == o.malformed &&
           call_count == o.call_count && seen == o.seen;
  }
};

Result RunChunks(const std::string& text, const std::vector<size_t>& cuts /* chunk boundaries, ascending */) {
  Inkling2StreamParser p;
  Result r;
  auto take = [&](const std::vector<Event>& events) {
    for (const auto& e : events) {
      switch (e.kind) {
        case Event::Kind::kReasoning: r.reasoning += e.text; r.event_kinds += 'R'; break;
        case Event::Kind::kContent: r.content += e.text; r.event_kinds += 'C'; break;
        case Event::Kind::kToolCall: r.calls.emplace_back(e.call.name, e.call.arguments_json); r.event_kinds += 'T'; break;
      }
    }
  };
  size_t prev = 0;
  for (const size_t c : cuts) {
    take(p.Push(std::string_view(text).substr(prev, c - prev)));
    prev = c;
  }
  take(p.Push(std::string_view(text).substr(prev)));
  take(p.Finish());
  r.malformed = p.had_malformed();
  r.call_count = p.tool_call_count();
  r.seen = p.tool_call_seen();
  return r;
}

// Runs `text` whole, then split at every byte boundary into two pieces, then in 1-byte and 3-byte pieces; every run must
// agree with the whole-text run on reasoning, content, calls and flags. Returns the whole-text result.
Result RunAllChunkings(const std::string& text, const char* label) {
  const Result whole = RunChunks(text, {});
  for (size_t i = 1; i < text.size(); ++i) {
    const Result r = RunChunks(text, {i});
    if (!(r == whole)) {
      std::fprintf(stderr, "chunking mismatch (%s) at split %zu\n", label, i);
      ++g_failures;
      return whole;
    }
  }
  for (const size_t step : {1u, 3u, 7u}) {
    std::vector<size_t> cuts;
    for (size_t i = step; i < text.size(); i += step) cuts.push_back(i);
    if (!(RunChunks(text, cuts) == whole)) {
      std::fprintf(stderr, "chunking mismatch (%s) at step %zu\n", label, static_cast<size_t>(step));
      ++g_failures;
    }
  }
  return whole;
}

void TestDialectTable() {
  const ModelDialect& d = Inkling2Dialect();
  CHECK(d.kind == DialectKind::kInkling2);
  CHECK_EQ(std::string(d.name), std::string("inkling2"));
  CHECK(d.structured_parts && !d.chat_template_polyfills && !d.bos_on_raw_prompt && d.has_reasoning_effort && !d.forces_thinking);
  CHECK(d.sampling.temperature.has_value() && *d.sampling.temperature == 1.0f);
  CHECK(!d.sampling.top_k && !d.sampling.top_p && !d.sampling.min_p && !d.sampling.repetition_penalty);
  CHECK((d.eos_ids == std::vector<int32_t>{200006}));
  CHECK_EQ(d.keep_special_on_decode.size(), static_cast<size_t>(5));
  CHECK(d.default_tokenizer_dir.find("Inkling-2-Mini-Uncensored") != std::string::npos);
  CHECK(&DialectFor(DialectKind::kInkling2) == &d);
  CHECK(!Qwen35Dialect().structured_parts && !Gemma4Dialect().structured_parts && !Lfm2Dialect().structured_parts);
  DialectKind k;
  CHECK(ParseDialectName("inkling2", &k) && k == DialectKind::kInkling2);
  CHECK(!ParseDialectName("Inkling2", &k));
  CHECK(ArchNameIsInkling2("inkling2_loop") && !ArchNameIsInkling2("inkling2") && !ArchNameIsInkling2("qwen3_5"));
}

void TestDetectResolveArch() {
  CHECK(DetectDialect("{\"tokenizer_class\":\"PreTrainedTokenizerFast\"}", false, false, true) == DialectKind::kInkling2);
  CHECK(DetectDialect("{\"tokenizer_class\":\"PreTrainedTokenizerFast\"}", false, false, false) == DialectKind::kQwen35);
  CHECK(DetectDialect("{}", true, false, true) == DialectKind::kGemma4);  // Gemma still wins
  CHECK(DetectDialect("{\"bos_token\":\"<|startoftext|>\"}", false, true, true) == DialectKind::kLfm2);  // so does LFM2

  namespace fs = std::filesystem;
  const fs::path root = fs::temp_directory_path() / "r4dx_test_inkling2_dialect";
  fs::remove_all(root);
  auto make_dir = [&](const std::string& name, const std::string& tokenizer_json) {
    const fs::path d = root / name;
    fs::create_directories(d);
    std::ofstream(d / "tokenizer_config.json") << "{\"tokenizer_class\":\"PreTrainedTokenizerFast\"}";
    std::ofstream(d / "tokenizer.json") << tokenizer_json;
    return d.string();
  };
  const std::string ink = make_dir("ink",
      "{\"added_tokens\":[{\"id\":200001,\"content\":\"<|message_model|>\"},{\"id\":200006,\"content\":\"<|content_model_end_sampling|>\"}]}");
  const std::string half = make_dir("half", "{\"added_tokens\":[{\"id\":200001,\"content\":\"<|message_model|>\"}]}");
  CHECK(ResolveDialectKindForArch(std::nullopt, ink, "") == DialectKind::kInkling2);
  CHECK(ResolveDialectKindForArch(std::nullopt, ink, "inkling2_loop") == DialectKind::kInkling2);
  CHECK(ResolveDialectKindForArch(std::nullopt, half, "") == DialectKind::kQwen35);
  CHECK(ResolveDialectKindForArch(std::nullopt, "", "inkling2_loop") == DialectKind::kInkling2);
  CHECK(ResolveDialectKindForArch(DialectKind::kQwen35, ink, "inkling2_loop") == DialectKind::kQwen35);  // explicit wins
  CHECK(ResolveDialectKind(std::nullopt, ink, false) == DialectKind::kInkling2);
  fs::remove_all(root);

  const ModelDialect& d = Inkling2Dialect();
  CHECK(CheckDialectAgainstArch(d, "inkling2_loop").empty());
  CHECK(CheckDialectAgainstArch(d, "").empty());
  CHECK(!CheckDialectAgainstArch(d, "qwen3_5").empty());
  CHECK(!CheckDialectAgainstArch(d, "llama").empty());
  CHECK(!CheckDialectAgainstArch(d, "gemma4_unified").empty());
  CHECK(!CheckDialectAgainstArch(Qwen35Dialect(), "inkling2_loop").empty());
  CHECK(!CheckDialectAgainstArch(Gemma4Dialect(), "inkling2_loop").empty());
  CHECK(!CheckDialectAgainstArch(Lfm2Dialect(), "inkling2_loop").empty());
  CHECK(!CheckDialectAgainstArch(NanbeigeDialect(), "inkling2_loop").empty());
  CHECK(CheckDialectAgainstArch(Qwen35Dialect(), "qwen3_5").empty());  // unchanged
}

void TestPromptState() {
  const ModelDialect& d = Inkling2Dialect();
  const std::string prompt = "<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|>"
                             "<|message_user|><|content_text|>hi<|end_message|>";
  for (const bool thinking : {true, false}) {
    CHECK(d.CheckpointSuffix(thinking, prompt).empty());  // no generation prompt: nothing to hold back, ckpt_back 0
    CHECK(!d.ReasoningOpenInPrompt(thinking, prompt));
  }
  CHECK(d.CheckpointSuffix(true, "").empty());
}

void TestParserVectors() {
  // thinking + text
  {
    const std::string s = Join({M, T, "Let me think", " about it.", E, M, X, "Hello", " world", E});
    const Result r = RunAllChunkings(s, "think+text");
    CHECK_EQ(r.reasoning, std::string("Let me think about it."));
    CHECK_EQ(r.content, std::string("Hello world"));
    CHECK(r.calls.empty() && !r.malformed && !r.seen);
  }
  // thinking + text + tool call (name in the header, args re-serialised compactly, key order kept)
  {
    const std::string s = Join({M, T, "plan", E, M, X, "Calling", E, M, "get_weather", J,
                                "{\"name\":\"get_weather\",\"args\":{\"city\":\"Bangkok\",\"units\":\"c\",\"n\":[1,2.5,null,true]}}", E});
    const Result r = RunAllChunkings(s, "think+text+tool");
    CHECK_EQ(r.reasoning, std::string("plan"));
    CHECK_EQ(r.content, std::string("Calling"));
    CHECK_EQ(r.calls.size(), static_cast<size_t>(1));
    if (r.calls.size() == 1) {
      CHECK_EQ(r.calls[0].first, std::string("get_weather"));
      CHECK_EQ(r.calls[0].second, std::string("{\"city\":\"Bangkok\",\"units\":\"c\",\"n\":[1,2.5,null,true]}"));
    }
    CHECK(r.seen && !r.malformed && r.call_count == 1);
    CHECK_EQ(RunChunks(s, {}).event_kinds, std::string("RCT"));
  }
  // multiple calls, empty args, nested args
  {
    const std::string s = Join({M, "a", J, "{\"name\":\"a\",\"args\":{}}", E, M, "b", J,
                                "{\"name\":\"b\",\"args\":{\"x\":[1,2,{\"y\":null}],\"z\":{\"k\":\"caf\xC3\xA9 \xF0\x9F\x98\x80\"}}}", E});
    const Result r = RunAllChunkings(s, "multi-call");
    CHECK_EQ(r.calls.size(), static_cast<size_t>(2));
    if (r.calls.size() == 2) {
      CHECK_EQ(r.calls[0].second, std::string("{}"));
      CHECK_EQ(r.calls[1].first, std::string("b"));
      CHECK_EQ(r.calls[1].second, std::string("{\"x\":[1,2,{\"y\":null}],\"z\":{\"k\":\"caf\xC3\xA9 \xF0\x9F\x98\x80\"}}"));
    }
    CHECK(r.content.empty() && r.reasoning.empty() && r.seen && r.call_count == 2);
  }
  // header name missing / body name missing / args missing / args double-encoded
  {
    const Result a = RunAllChunkings(Join({M, J, "{\"name\":\"f\",\"args\":{\"q\":1}}", E}), "no header name");
    CHECK(a.calls.size() == 1 && a.calls[0].first == "f" && a.calls[0].second == "{\"q\":1}");
    const Result b = RunAllChunkings(Join({M, "g", J, "{\"args\":{\"k\":\"v\"}}", E}), "no body name");
    CHECK(b.calls.size() == 1 && b.calls[0].first == "g" && b.calls[0].second == "{\"k\":\"v\"}");
    const Result c = RunAllChunkings(Join({M, J, "{\"name\":\"h\"}", E}), "no args");
    CHECK(c.calls.size() == 1 && c.calls[0].first == "h" && c.calls[0].second == "{}");
    const Result d = RunAllChunkings(Join({M, "h", J, "{\"name\":\"h\",\"args\":\"{\\\"a\\\":1}\"}", E}), "double-encoded");
    CHECK(d.calls.size() == 1 && d.calls[0].second == "{\"a\":1}");
    const Result e = RunAllChunkings(Join({M, "old", J, "{\"name\":\"new\",\"args\":{}}", E}), "body name wins");
    CHECK(e.calls.size() == 1 && e.calls[0].first == "new");
  }
  // malformed tool calls degrade to their raw source text
  {
    const std::string bad_json = Join({M, "f", J, "{\"name\":\"f\",\"args\":{", E});
    const Result a = RunAllChunkings(bad_json, "bad json");
    CHECK(a.calls.empty() && a.malformed && !a.seen);
    CHECK_EQ(a.content, bad_json);
    const std::string args_array = Join({M, "f", J, "{\"name\":\"f\",\"args\":[1]}", E});
    const Result b = RunAllChunkings(args_array, "args array");
    CHECK(b.calls.empty() && b.malformed);
    CHECK_EQ(b.content, args_array);
    const std::string not_object = Join({M, J, "[1,2]", E});
    CHECK_EQ(RunAllChunkings(not_object, "not object").content, not_object);
    const std::string empty_name = Join({M, J, "{\"args\":{}}", E});  // no name anywhere
    const Result c = RunAllChunkings(empty_name, "no name at all");
    CHECK(c.calls.empty() && c.malformed);
    CHECK_EQ(c.content, empty_name);
    // a good call after a bad one still parses
    const Result d = RunAllChunkings(bad_json + Join({M, "ok", J, "{\"name\":\"ok\",\"args\":{}}", E}), "bad then good");
    CHECK(d.calls.size() == 1 && d.malformed && d.content == bad_json);
  }
  // truncation (generation ended mid-part)
  {
    const std::string trunc_tool = Join({M, "f", J, "{\"name\":\"f\",\"args\":{\"a\":"});
    const Result a = RunAllChunkings(trunc_tool, "truncated tool");
    CHECK(a.calls.empty() && a.malformed);
    CHECK_EQ(a.content, trunc_tool);
    const Result b = RunAllChunkings(Join({M, T, "half a thought"}), "truncated thinking");
    CHECK_EQ(b.reasoning, std::string("half a thought"));
    CHECK(b.content.empty() && !b.malformed);
    const Result c = RunAllChunkings(Join({M, X, "half an answer"}), "truncated text");
    CHECK_EQ(c.content, std::string("half an answer"));
    const Result d = RunAllChunkings(Join({M, "partial_na"}), "truncated header");
    CHECK_EQ(d.content, Join({M, "partial_na"}));
    CHECK(d.malformed);
    const Result e = RunAllChunkings(Join({M, X, "x<|end_mes"}), "partial marker at end");
    CHECK_EQ(e.content, std::string("x<|end_mes"));
    CHECK(RunAllChunkings("", "empty").content.empty());
  }
  // named parts, stray markers, text outside parts
  {
    const Result a = RunAllChunkings(Join({M, "assistant_x", X, "hi", E}), "named text part");
    CHECK_EQ(a.content, std::string("hi"));
    const Result b = RunAllChunkings("plain text with no header", "no header");
    CHECK_EQ(b.content, std::string("plain text with no header"));
    const Result c = RunAllChunkings(Join({X, "no message header", E}), "kind without header");
    CHECK_EQ(c.content, std::string("no message header"));
    const Result d = RunAllChunkings(Join({M, X, "a <| b <|content_te c < | <", E}), "lookalikes");
    CHECK_EQ(d.content, std::string("a <| b <|content_te c < | <"));
    const Result e = RunAllChunkings(Join({M, X, "a", T, "b", E}), "kind marker inside a part");
    CHECK_EQ(e.content, Join({"a", T, "b"}));
    const Result f = RunAllChunkings(Join({M, M, X, "z", E}), "double header");
    CHECK_EQ(f.content, M + "z");
    const Result g = RunAllChunkings(Join({M, T, "t1", E, M, T, "t2", E, M, X, "a", E, M, X, "b", E}), "repeated parts");
    CHECK_EQ(g.reasoning, std::string("t1t2"));
    CHECK_EQ(g.content, std::string("ab"));
    const Result h = RunAllChunkings(Join({M, T, "t", M, X, "no end marker before the next header", E}), "missing end");
    CHECK_EQ(h.reasoning, std::string("t"));
    CHECK_EQ(h.content, std::string("no end marker before the next header"));
    const Result i = RunAllChunkings(Join({E, M, X, "x", E, E}), "stray end markers");
    CHECK_EQ(i.content, std::string("x"));
    const Result j = RunAllChunkings(Join({M, T, "\xE2\x80\x9C" "thinking" "\xE2\x80\x9D \xF0\x9F\x98\x80", E, M, X, "\xE0\xB8\xAA\xE0\xB8\xA7\xE0\xB8\xB1\xE0\xB8\xAA\xE0\xB8\x94\xE0\xB8\xB5", E}), "utf8");
    CHECK_EQ(j.content, std::string("\xE0\xB8\xAA\xE0\xB8\xA7\xE0\xB8\xB1\xE0\xB8\xAA\xE0\xB8\x94\xE0\xB8\xB5"));
    // a name longer than the header limit is not a header: degrade, then the stream continues
    const Result k = RunAllChunkings(Join({M, std::string(2000, 'n'), X, "after", E}), "overlong header");
    CHECK(k.malformed && k.content.find("after") != std::string::npos);
  }
  // whole-text API and ParseToolCalls dispatch
  {
    const std::string s = Join({M, T, "think", E, M, X, "answer", E, M, "f", J, "{\"name\":\"f\",\"args\":{\"a\":1}}", E});
    const Inkling2Parsed p = ParseInkling2Generation(s);
    CHECK_EQ(p.reasoning, std::string("think"));
    CHECK_EQ(p.content, std::string("answer"));
    CHECK(p.tool_calls.size() == 1 && p.tool_calls[0].name == "f" && p.tool_calls[0].arguments_json == "{\"a\":1}");
    const ToolCallParseResult r = ParseToolCalls(s, Inkling2Dialect());
    CHECK_EQ(r.content, std::string("answer"));
    CHECK(r.tool_calls.size() == 1 && !r.had_malformed_call);
    // other dialects are untouched
    CHECK_EQ(ParseToolCalls("plain", Qwen35Dialect()).content, std::string("plain"));
  }
}

void TestRequestNormalisation() {
  using nlohmann::json;
  using nlohmann::ordered_json;
  {
    ordered_json m = ordered_json::parse(R"([
      {"role":"user","content":"hi"},
      {"role":"assistant","content":"","tool_calls":[
        {"id":"c1","type":"function","function":{"name":"a","arguments":"{\"b\":1,\"a\":{\"z\":[1,2]}}"}},
        {"id":"c2","type":"function","function":{"name":"b","arguments":""}},
        {"id":"c3","type":"function","function":{"name":"c","arguments":{"k":"v"}}},
        {"id":"c4","type":"function","function":{"name":"d"}}]}])");
    CanonicalizeInkling2Messages(m);
    const auto& calls = m[1]["tool_calls"];
    CHECK(calls[0]["function"]["arguments"].is_object());
    CHECK_EQ(calls[0]["function"]["arguments"].dump(), std::string("{\"b\":1,\"a\":{\"z\":[1,2]}}"));  // key order kept
    CHECK(calls[1]["function"]["arguments"].is_object() && calls[1]["function"]["arguments"].empty());
    CHECK_EQ(calls[2]["function"]["arguments"].dump(), std::string("{\"k\":\"v\"}"));
    CHECK(!calls[3]["function"].contains("arguments"));
    ordered_json bad = ordered_json::parse(R"([{"role":"assistant","tool_calls":[{"function":{"name":"a","arguments":"not json"}}]}])");
    bool threw = false;
    try { CanonicalizeInkling2Messages(bad); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);
    ordered_json arr = ordered_json::parse(R"([{"role":"assistant","tool_calls":[{"function":{"name":"a","arguments":"[1]"}}]}])");
    threw = false;
    try { CanonicalizeInkling2Messages(arr); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);
  }
  auto effort = [](const json& ctk, std::optional<bool> think, const json& req) -> std::optional<std::string> {
    const auto r = ResolveInkling2ReasoningEffort(ctk, think, req);
    return r ? std::optional<std::string>(r->dump()) : std::nullopt;
  };
  auto throws = [](const json& ctk, std::optional<bool> think, const json& req) {
    try { ResolveInkling2ReasoningEffort(ctk, think, req); } catch (const std::invalid_argument&) { return true; }
    return false;
  };
  CHECK(!effort(nullptr, std::nullopt, nullptr).has_value());
  CHECK(!effort(json::object(), true, nullptr).has_value());
  CHECK(effort(nullptr, false, nullptr) == std::optional<std::string>("\"none\""));   // enable_thinking=false -> none
  CHECK(effort(nullptr, false, "high") == std::optional<std::string>("\"none\""));    // ... whatever effort was named
  CHECK(effort(nullptr, false, 0.5) == std::optional<std::string>("\"none\""));
  CHECK(effort(nullptr, true, "  HIGH ") == std::optional<std::string>("\"high\""));
  for (const char* level : {"none", "minimal", "low", "medium", "high", "xhigh", "max"}) {
    CHECK(effort(nullptr, std::nullopt, level) == std::optional<std::string>(std::string("\"") + level + "\""));
  }
  CHECK(effort(nullptr, std::nullopt, 0.5) == std::optional<std::string>("0.5"));
  CHECK(effort(nullptr, std::nullopt, 0) == std::optional<std::string>("0.0"));
  CHECK(effort(nullptr, std::nullopt, 0.99) == std::optional<std::string>("0.99"));
  CHECK(effort(nullptr, std::nullopt, 1e-3) == std::optional<std::string>("0.001"));
  CHECK(throws(nullptr, std::nullopt, 1.0));
  CHECK(throws(nullptr, std::nullopt, 1));
  CHECK(throws(nullptr, std::nullopt, -0.1));
  CHECK(throws(nullptr, std::nullopt, 0.991));
  CHECK(throws(nullptr, std::nullopt, "bogus"));
  CHECK(throws(nullptr, std::nullopt, "0.5"));
  CHECK(throws(nullptr, std::nullopt, true));
  CHECK(throws(nullptr, std::nullopt, json::array()));
  // chat_template_kwargs.reasoning_effort wins and is passed through untouched (the template validates it)
  CHECK(effort(json{{"reasoning_effort", "low"}}, false, "high") == std::optional<std::string>("\"low\""));
  CHECK(effort(json{{"reasoning_effort", 0.25}}, std::nullopt, nullptr) == std::optional<std::string>("0.25"));
  CHECK(effort(json{{"reasoning_effort", nullptr}}, std::nullopt, "max") == std::optional<std::string>("\"max\""));
  // request body field precedence: reasoning.effort over the top-level field
  CHECK(Inkling2RequestEffort(json::parse(R"({"reasoning_effort":"low"})")) == json("low"));
  CHECK(Inkling2RequestEffort(json::parse(R"({"reasoning_effort":"low","reasoning":{"effort":"high"}})")) == json("high"));
  CHECK(Inkling2RequestEffort(json::parse(R"({"reasoning_effort":0.3})")) == json(0.3));
  CHECK(Inkling2RequestEffort(json::parse(R"({"reasoning":{"enabled":true}})")).is_null());
  CHECK(Inkling2RequestEffort(json::parse("[]")).is_null());
}

// Real token ids through the real tokenizer and chat template (skipped when the checkpoint's tokenizer files are absent).
void TestWithTokenizer(const std::string& dir) {
  using nlohmann::ordered_json;
  if (!std::ifstream(dir + "/tokenizer.json", std::ios::binary)) {
    std::printf("test_inkling2_dialect: tokenizer part SKIPPED (%s/tokenizer.json not found)\n", dir.c_str());
    return;
  }
  r4dx::Tokenizer::Options opts;
  opts.keep_special_on_decode = Inkling2Dialect().keep_special_on_decode;
  r4dx::Tokenizer tok;
  r4dx::ChatTemplate tmpl;
  try {
    tok = r4dx::Tokenizer::from_directory(dir, opts);
    tmpl = r4dx::ChatTemplate::from_directory(dir, r4dx::ChatTemplateOptions{false});
  } catch (const std::exception& e) {
    std::fprintf(stderr, "failed to load %s: %s\n", dir.c_str(), e.what());
    ++g_failures;
    return;
  }
  CHECK(tok.eos_id() == 200006);

  // token-level replay: the generation as token ids, streamed through the StreamDecoder one token at a time
  const std::string gen = Join({M, T, "I should check the weather \xE0\xB8\xAA\xE0\xB8\xA7\xE0\xB8\xB1\xE0\xB8\xAA\xE0\xB8\x94\xE0\xB8\xB5 \xF0\x9F\x98\x80.", E, M, X, "Checking.", E, M, "get_weather", J,
                                "{\"name\":\"get_weather\",\"args\":{\"city\":\"Bangkok\",\"units\":\"celsius\"}}", E, "<|content_model_end_sampling|>"});
  const std::vector<r4dx::TokenId> ids = tok.encode(gen, /*parse_special=*/true);
  CHECK(!ids.empty() && ids.front() == 200001 && ids.back() == 200006);
  Inkling2StreamParser p;
  auto dec = tok.make_stream_decoder(/*skip_special_tokens=*/true);
  Result r;
  auto take = [&](const std::vector<Event>& events) {
    for (const auto& e : events) {
      if (e.kind == Event::Kind::kReasoning) r.reasoning += e.text;
      else if (e.kind == Event::Kind::kContent) r.content += e.text;
      else r.calls.emplace_back(e.call.name, e.call.arguments_json);
    }
  };
  for (const r4dx::TokenId id : ids) take(p.Push(dec.push(id)));  // the EOS id decodes to nothing (special, not kept)
  take(p.Push(dec.flush()));
  take(p.Finish());
  CHECK_EQ(r.reasoning, std::string("I should check the weather \xE0\xB8\xAA\xE0\xB8\xA7\xE0\xB8\xB1\xE0\xB8\xAA\xE0\xB8\x94\xE0\xB8\xB5 \xF0\x9F\x98\x80."));
  CHECK_EQ(r.content, std::string("Checking."));
  CHECK(r.calls.size() == 1 && r.calls[0].first == "get_weather" &&
        r.calls[0].second == "{\"city\":\"Bangkok\",\"units\":\"celsius\"}");

  // template round trip: what the template renders for an assistant turn parses back to the same calls (arguments sorted)
  ordered_json messages = ordered_json::parse(R"([
    {"role":"user","content":"weather?"},
    {"role":"assistant","content":"Let me look.","reasoning_content":"need a tool","tool_calls":[
      {"id":"c1","type":"function","function":{"name":"get_weather","arguments":"{\"units\":\"c\",\"city\":\"Chiang Mai\",\"deep\":{\"z\":[1,{\"a\":null}],\"B\":true}}"}}]}])");
  CanonicalizeInkling2Messages(messages);
  const std::string rendered = tmpl.render(messages, true);
  const size_t first = rendered.find(M);
  CHECK(first != std::string::npos);
  // (the template appends the EOS text after an assistant turn; a real generation stops on it and never decodes it)
  const std::string eos_text = "<|content_model_end_sampling|>";
  std::string turn = rendered.substr(first);
  CHECK(turn.size() > eos_text.size() && turn.compare(turn.size() - eos_text.size(), eos_text.size(), eos_text) == 0);
  turn.resize(turn.size() - eos_text.size());
  const Inkling2Parsed parsed = ParseInkling2Generation(turn);
  CHECK_EQ(parsed.reasoning, std::string("need a tool"));
  CHECK_EQ(parsed.content, std::string("Let me look."));
  CHECK(parsed.tool_calls.size() == 1 && parsed.tool_calls[0].name == "get_weather");
  if (parsed.tool_calls.size() == 1) {
    // the template sorts the argument keys (case-sensitive) at the top level only
    CHECK_EQ(parsed.tool_calls[0].arguments_json,
             std::string("{\"city\":\"Chiang Mai\",\"deep\":{\"z\":[1,{\"a\":null}],\"B\":true},\"units\":\"c\"}"));
  }

  // request effort through the template: enable_thinking=false -> "Thinking effort level: 0"
  const ordered_json user = ordered_json::parse(R"([{"role":"user","content":"hi"}])");
  auto render_with = [&](std::optional<ordered_json> eff) {
    ordered_json extra = ordered_json::object();
    if (eff) extra["reasoning_effort"] = *eff;
    return tmpl.render(user, true, ordered_json::array(), extra);
  };
  CHECK(render_with(ResolveInkling2ReasoningEffort(nullptr, false, nullptr)).find("Thinking effort level: 0<|end_message|>") != std::string::npos);
  CHECK(render_with(std::nullopt).find("Thinking effort level: 0.9<|end_message|>") != std::string::npos);
  CHECK(render_with(ResolveInkling2ReasoningEffort(nullptr, std::nullopt, "xhigh")).find("Thinking effort level: 0.99<|end_message|>") != std::string::npos);
  CHECK(render_with(ResolveInkling2ReasoningEffort(nullptr, std::nullopt, 0.35)).find("Thinking effort level: 0.35<|end_message|>") != std::string::npos);
  // the prompt ends with the user's end_message: no generation prompt
  const std::string prompt = render_with(std::nullopt);
  CHECK(prompt.size() >= E.size() && prompt.compare(prompt.size() - E.size(), E.size(), E) == 0);
}

}  // namespace

int main(int argc, char** argv) {
  TestDialectTable();
  TestDetectResolveArch();
  TestPromptState();
  TestParserVectors();
  TestRequestNormalisation();
  std::string dir = R4DX_INKLING2_TOKENIZER_DIR;
  if (argc > 1) dir = argv[1];
  TestWithTokenizer(dir);
  if (g_failures != 0) {
    std::fprintf(stderr, "test_inkling2_dialect: %d failure(s)\n", g_failures);
    return 1;
  }
  std::printf("test_inkling2_dialect: OK\n");
  return 0;
}
