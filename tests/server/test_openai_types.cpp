// tests/server/test_openai_types.cpp -- pure CPU unit test for src/server/openai_types.h: chat/
// completion request parsing+validation and OpenAI-shaped response JSON building.
#include <cstdio>
#include <functional>
#include <string>

#include "nlohmann/json.hpp"
#include "openai_types.h"

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

using nlohmann::json;
using namespace r4dx::server;

bool ThrowsApiError(const std::function<void()>& fn, int expect_status = -1) {
  try {
    fn();
  } catch (const ApiError& e) {
    return expect_status < 0 || e.http_status == expect_status;
  }
  return false;
}

// ---- ParseChatCompletionRequest ---------------------------------------------------------------

void TestChatMinimal() {
  json body = {{"messages", json::array({{{"role", "user"}, {"content", "hi"}}})}};
  const auto req = ParseChatCompletionRequest(body);
  CHECK(req.messages.size() == 1);
  CHECK(req.messages[0].role == "user");
  CHECK(req.messages[0].content == "hi");
  CHECK(!req.stream);
  CHECK(!req.max_tokens.has_value());
  CHECK(req.stop.empty());
  CHECK(req.sampling.temperature == 1.0f);
  CHECK(req.sampling.top_p == 1.0f);
  CHECK(req.sampling.top_k == 0);
  CHECK(req.sampling.min_p == 0.0f);
  CHECK(!req.sampling.has_seed);
}

void TestChatSamplingDefaultsApplyOnlyWhenOmitted() {
  SamplingParams defaults;
  defaults.temperature = 0.5f;
  defaults.top_p = 0.8f;
  defaults.top_k = 20;
  defaults.min_p = 0.02f;

  json body = {{"messages", json::array({{{"role", "user"}, {"content", "hi"}}})},
               {"top_k", 40}};  // client overrides only top_k
  const auto req = ParseChatCompletionRequest(body, defaults);
  CHECK(req.sampling.temperature == 0.5f);  // from server defaults
  CHECK(req.sampling.top_p == 0.8f);        // from server defaults
  CHECK(req.sampling.top_k == 40);          // client override wins
  CHECK(req.sampling.min_p == 0.02f);       // from server defaults
}

void TestChatFullFields() {
  json body = {
      {"model", "qwen"},
      {"messages", json::array({{{"role", "system"}, {"content", "be terse"}},
                                 {{"role", "user"}, {"content", "hi"}}})},
      {"temperature", 0.7}, {"top_p", 0.9}, {"top_k", 40}, {"min_p", 0.05},
      {"seed", 123}, {"max_tokens", 64}, {"stop", json::array({"</s>", "STOP"})},
      {"stream", true}, {"chat_template_kwargs", {{"enable_thinking", false}}},
      {"tools", json::array()}};
  const auto req = ParseChatCompletionRequest(body);
  CHECK(req.model == "qwen");
  CHECK(req.messages.size() == 2);
  CHECK(req.sampling.temperature == 0.7f);
  CHECK(req.sampling.top_p == 0.9f);
  CHECK(req.sampling.top_k == 40);
  CHECK(req.sampling.min_p == 0.05f);
  CHECK(req.sampling.has_seed && req.sampling.seed == 123);
  CHECK(req.max_tokens.has_value() && *req.max_tokens == 64);
  CHECK(req.stop.size() == 2 && req.stop[0] == "</s>" && req.stop[1] == "STOP");
  CHECK(req.stream);
  CHECK(req.chat_template_kwargs.at("enable_thinking") == false);
}

void TestChatContentAsTextParts() {
  json body = {{"messages", json::array({{{"role", "user"},
                                           {"content", json::array({{{"type", "text"}, {"text", "a"}},
                                                                     {{"type", "text"}, {"text", "b"}}})}}})}};
  const auto req = ParseChatCompletionRequest(body);
  CHECK(req.messages[0].content == "ab");
}

void TestChatImagePartRejected() {
  json body = {{"messages",
                json::array({{{"role", "user"},
                              {"content", json::array({{{"type", "image_url"},
                                                         {"image_url", {{"url", "http://x"}}}}})}}})}};
  CHECK(ThrowsApiError([&] { ParseChatCompletionRequest(body); }, 400));
}

void TestChatMissingMessagesThrows() {
  json body = json::object();
  CHECK(ThrowsApiError([&] { ParseChatCompletionRequest(body); }, 400));
}

void TestChatEmptyMessagesThrows() {
  json body = {{"messages", json::array()}};
  CHECK(ThrowsApiError([&] { ParseChatCompletionRequest(body); }, 400));
}

void TestChatUnsupportedRoleThrows() {
  json body = {{"messages", json::array({{{"role", "carrier_pigeon"}, {"content", "x"}}})}};
  CHECK(ThrowsApiError([&] { ParseChatCompletionRequest(body); }, 400));
}

// ---- tool/function roles + tool_calls (task point 4: accepting tool results) ------------------

void TestChatToolRoleRequiresToolCallId() {
  json body = {{"messages", json::array({{{"role", "tool"}, {"content", "72F"}}})}};
  CHECK(ThrowsApiError([&] { ParseChatCompletionRequest(body); }, 400));
}

void TestChatToolRoleAccepted() {
  json body = {{"messages", json::array({{{"role", "user"}, {"content", "weather?"}},
                                          {{"role", "tool"}, {"content", "72F, sunny"},
                                           {"tool_call_id", "call_abc"}}})}};
  const auto req = ParseChatCompletionRequest(body);
  CHECK(req.messages.size() == 2);
  CHECK(req.messages[1].role == "tool");
  CHECK(req.messages[1].content == "72F, sunny");
  CHECK(req.messages[1].tool_call_id.has_value() && *req.messages[1].tool_call_id == "call_abc");
}

void TestChatFunctionRoleRequiresName() {
  json body = {{"messages", json::array({{{"role", "function"}, {"content", "72F"}}})}};
  CHECK(ThrowsApiError([&] { ParseChatCompletionRequest(body); }, 400));
}

void TestChatFunctionRoleAccepted() {
  json body = {{"messages", json::array({{{"role", "function"}, {"content", "72F"},
                                           {"name", "get_current_weather"}}})}};
  const auto req = ParseChatCompletionRequest(body);
  CHECK(req.messages[0].name.has_value() && *req.messages[0].name == "get_current_weather");
}

void TestChatAssistantToolCallsRoundTrip() {
  json body = {{"messages",
                json::array({{{"role", "user"}, {"content", "weather?"}},
                              {{"role", "assistant"},
                               {"content", nullptr},
                               {"tool_calls",
                                json::array({{{"id", "call_1"},
                                              {"type", "function"},
                                              {"function",
                                               {{"name", "get_current_weather"},
                                                {"arguments", "{\"location\":\"Boston, MA\"}"}}}}})}},
                              {{"role", "tool"}, {"tool_call_id", "call_1"}, {"content", "72F"}}})}};
  const auto req = ParseChatCompletionRequest(body);
  CHECK(req.messages.size() == 3);
  CHECK(!req.messages[1].content.has_value());
  CHECK(req.messages[1].tool_calls.size() == 1);
  CHECK(req.messages[1].tool_calls[0].id == "call_1");
  CHECK(req.messages[1].tool_calls[0].name == "get_current_weather");
  CHECK(nlohmann::json::parse(req.messages[1].tool_calls[0].arguments_json).at("location") == "Boston, MA");
}

void TestChatAssistantToolCallsGeneratesIdWhenMissing() {
  json body = {{"messages",
                json::array({{{"role", "assistant"},
                               {"content", nullptr},
                               {"tool_calls",
                                json::array({{{"type", "function"},
                                              {"function", {{"name", "f"}, {"arguments", "{}"}}}}})}}})}};
  const auto req = ParseChatCompletionRequest(body);
  CHECK(!req.messages[0].tool_calls[0].id.empty());
}

void TestChatAssistantMissingContentAndToolCallsThrows() {
  json body = {{"messages", json::array({{{"role", "assistant"}}})}};
  CHECK(ThrowsApiError([&] { ParseChatCompletionRequest(body); }, 400));
}

void TestChatToolCallsArgumentsMustBeJsonObjectString() {
  json body = {{"messages",
                json::array({{{"role", "assistant"},
                               {"content", nullptr},
                               {"tool_calls",
                                json::array({{{"type", "function"},
                                              {"function", {{"name", "f"}, {"arguments", "not json"}}}}})}}})}};
  CHECK(ThrowsApiError([&] { ParseChatCompletionRequest(body); }, 400));

  json body2 = {{"messages",
                 json::array({{{"role", "assistant"},
                                {"content", nullptr},
                                {"tool_calls",
                                 json::array({{{"type", "function"},
                                               {"function", {{"name", "f"}, {"arguments", "[1,2]"}}}}})}}})}};
  CHECK(ThrowsApiError([&] { ParseChatCompletionRequest(body2); }, 400));
}

// ---- tool_choice (task point 5) ----------------------------------------------------------------

json OneToolBody(json extra) {
  json body = {
      {"messages", json::array({{{"role", "user"}, {"content", "hi"}}})},
      {"tools", json::array({{{"type", "function"}, {"function", {{"name", "get_current_weather"},
                                                                    {"parameters", json::object()}}}},
                               {{"type", "function"}, {"function", {{"name", "get_time"},
                                                                    {"parameters", json::object()}}}}})}};
  body.update(extra);
  return body;
}

void TestToolChoiceNoneClearsTools() {
  const auto req = ParseChatCompletionRequest(OneToolBody({{"tool_choice", "none"}}));
  CHECK(req.tools.empty());
}

void TestToolChoiceAutoLeavesToolsUntouched() {
  const auto req = ParseChatCompletionRequest(OneToolBody({{"tool_choice", "auto"}}));
  CHECK(req.tools.size() == 2);
}

void TestToolChoiceDefaultIsAutoLeavesToolsUntouched() {
  const auto req = ParseChatCompletionRequest(OneToolBody(json::object()));
  CHECK(req.tools.size() == 2);
}

void TestToolChoiceRequiredNeedsNonEmptyTools() {
  json body = {{"messages", json::array({{{"role", "user"}, {"content", "hi"}}})},
               {"tool_choice", "required"}};
  CHECK(ThrowsApiError([&] { ParseChatCompletionRequest(body); }, 400));

  const auto req = ParseChatCompletionRequest(OneToolBody({{"tool_choice", "required"}}));
  CHECK(req.tools.size() == 2);  // required does not filter, just needs >=1 tool present
}

void TestToolChoiceNamedFiltersToJustThatTool() {
  const auto req = ParseChatCompletionRequest(
      OneToolBody({{"tool_choice", {{"type", "function"}, {"function", {{"name", "get_time"}}}}}}));
  CHECK(req.tools.size() == 1);
  CHECK(req.tools[0].at("function").at("name") == "get_time");
}

void TestToolChoiceNamedUnknownFunctionThrows() {
  CHECK(ThrowsApiError(
      [&] {
        ParseChatCompletionRequest(
            OneToolBody({{"tool_choice", {{"type", "function"}, {"function", {{"name", "nope"}}}}}}));
      },
      400));
}

void TestToolChoiceUnrecognizedStringThrows() {
  CHECK(ThrowsApiError([&] { ParseChatCompletionRequest(OneToolBody({{"tool_choice", "sometimes"}})); }, 400));
}

void TestToolChoiceMalformedObjectThrows() {
  CHECK(ThrowsApiError([&] { ParseChatCompletionRequest(OneToolBody({{"tool_choice", json::object()}})); }, 400));
  CHECK(ThrowsApiError([&] { ParseChatCompletionRequest(OneToolBody({{"tool_choice", 5}})); }, 400));
}

// ---- tools[] shape validation (review finding, 2026-09-20) --------------------------------------
// Previously only `tools` itself being an array was checked -- an individual malformed entry
// reached chat_template.jinja unvalidated (raising there, an uninformative 500) AND silently
// disabled DropUnknownToolCalls's undefined-tool filter (engine.cpp's `known_names` comes out
// empty when `t.at("function").at("name")` doesn't exist, and an empty known_names list is
// DropUnknownToolCalls's own "no filter configured" no-op case).
void TestToolsArrayEntryNotObjectThrows() {
  json body = {{"messages", json::array({{{"role", "user"}, {"content", "hi"}}})},
               {"tools", json::array({1, 2})}};
  CHECK(ThrowsApiError([&] { ParseChatCompletionRequest(body); }, 400));
}

void TestToolsArrayEntryMissingFunctionThrows() {
  json body = {{"messages", json::array({{{"role", "user"}, {"content", "hi"}}})},
               {"tools", json::array({{{"type", "function"}}})}};
  CHECK(ThrowsApiError([&] { ParseChatCompletionRequest(body); }, 400));
}

void TestToolsArrayEntryFunctionMissingNameThrows() {
  json body = {{"messages", json::array({{{"role", "user"}, {"content", "hi"}}})},
               {"tools", json::array({{{"type", "function"}, {"function", json::object()}}})}};
  CHECK(ThrowsApiError([&] { ParseChatCompletionRequest(body); }, 400));
}

void TestToolsArrayEntryWrongTypeThrows() {
  json body = {
      {"messages", json::array({{{"role", "user"}, {"content", "hi"}}})},
      {"tools", json::array({{{"type", "not_function"}, {"function", {{"name", "f"}}}}})}};
  CHECK(ThrowsApiError([&] { ParseChatCompletionRequest(body); }, 400));
}

void TestToolsArrayWellFormedEntriesAccepted() {
  // Regression guard: the new validation must not reject the shape every other tools[] test in
  // this file already relies on.
  const auto req = ParseChatCompletionRequest(OneToolBody(json::object()));
  CHECK(req.tools.size() == 2);
}

void TestChatMissingContentThrows() {
  json body = {{"messages", json::array({{{"role", "user"}}})}};
  CHECK(ThrowsApiError([&] { ParseChatCompletionRequest(body); }, 400));
}

void TestChatBadSamplingRangesThrow() {
  auto with = [](json extra) {
    json body = {{"messages", json::array({{{"role", "user"}, {"content", "hi"}}})}};
    body.update(extra);
    return body;
  };
  CHECK(ThrowsApiError([&] { ParseChatCompletionRequest(with({{"top_p", 1.5}})); }, 400));
  CHECK(ThrowsApiError([&] { ParseChatCompletionRequest(with({{"min_p", -0.1}})); }, 400));
  CHECK(ThrowsApiError([&] { ParseChatCompletionRequest(with({{"top_k", -1}})); }, 400));
  CHECK(ThrowsApiError([&] { ParseChatCompletionRequest(with({{"temperature", "hot"}})); }, 400));
  CHECK(ThrowsApiError([&] { ParseChatCompletionRequest(with({{"stream", "yes"}})); }, 400));
  CHECK(ThrowsApiError([&] { ParseChatCompletionRequest(with({{"stop", 5}})); }, 400));
  CHECK(ThrowsApiError([&] { ParseChatCompletionRequest(with({{"max_tokens", -1}})); }, 400));
  CHECK(ThrowsApiError([&] { ParseChatCompletionRequest(with({{"chat_template_kwargs", 5}})); }, 400));
}

void TestChatNotAnObjectThrows() {
  CHECK(ThrowsApiError([&] { ParseChatCompletionRequest(json::array()); }, 400));
}

void TestChatStopAsSingleString() {
  json body = {{"messages", json::array({{{"role", "user"}, {"content", "hi"}}})}, {"stop", "END"}};
  const auto req = ParseChatCompletionRequest(body);
  CHECK(req.stop.size() == 1 && req.stop[0] == "END");
}

// ---- ParseCompletionRequest -------------------------------------------------------------------

void TestCompletionMinimal() {
  json body = {{"prompt", "once upon a time"}};
  const auto req = ParseCompletionRequest(body);
  CHECK(req.prompt == "once upon a time");
  CHECK(!req.stream);
}

void TestCompletionMissingPromptThrows() {
  json body = json::object();
  CHECK(ThrowsApiError([&] { ParseCompletionRequest(body); }, 400));
}

void TestCompletionPromptWrongTypeThrows() {
  json body = {{"prompt", 5}};
  CHECK(ThrowsApiError([&] { ParseCompletionRequest(body); }, 400));
}

// ---- response builders --------------------------------------------------------------------

void TestBuildModelsResponse() {
  const json resp = BuildModelsResponse("my-model", 1000);
  CHECK(resp.at("object") == "list");
  CHECK(resp.at("data").size() == 1);
  CHECK(resp.at("data")[0].at("id") == "my-model");
  CHECK(resp.at("data")[0].at("object") == "model");
}

void TestBuildChatCompletionResponse() {
  UsageStats usage{7, 3};
  TimingStats timings;
  timings.prompt_n = 7;
  timings.prompt_ms = 100.0;
  timings.predicted_n = 3;
  timings.predicted_ms = 60.0;
  const json resp = BuildChatCompletionResponse("id1", "m", 42, "hello", "stop", usage, timings);
  CHECK(resp.at("id") == "id1");
  CHECK(resp.at("object") == "chat.completion");
  CHECK(resp.at("model") == "m");
  CHECK(resp.at("choices")[0].at("message").at("role") == "assistant");
  CHECK(resp.at("choices")[0].at("message").at("content") == "hello");
  CHECK(resp.at("choices")[0].at("finish_reason") == "stop");
  CHECK(resp.at("usage").at("prompt_tokens") == 7);
  CHECK(resp.at("usage").at("completion_tokens") == 3);
  CHECK(resp.at("usage").at("total_tokens") == 10);
  CHECK(resp.at("timings").at("prompt_n") == 7);
  CHECK(resp.at("timings").at("predicted_n") == 3);
}

// ---- timings (task point 1) --------------------------------------------------------------------

void TestBuildTimingsJsonArithmetic() {
  TimingStats t;
  t.prompt_n = 100;
  t.prompt_ms = 200.0;  // 500 tok/s
  t.predicted_n = 50;
  t.predicted_ms = 250.0;  // 200 tok/s
  const json j = BuildTimingsJson(t);
  CHECK(j.at("prompt_n") == 100);
  CHECK(j.at("prompt_ms") == 200.0);
  CHECK(j.at("prompt_per_second") == 500.0);
  CHECK(j.at("predicted_n") == 50);
  CHECK(j.at("predicted_ms") == 250.0);
  CHECK(j.at("predicted_per_second") == 200.0);
  CHECK(!j.contains("draft_n"));
  CHECK(!j.contains("draft_n_accepted"));
}

void TestBuildTimingsJsonZeroMsAvoidsDivideByZero() {
  TimingStats t;  // every field defaults to 0
  const json j = BuildTimingsJson(t);
  CHECK(j.at("prompt_per_second") == 0.0);
  CHECK(j.at("predicted_per_second") == 0.0);
  CHECK(!j.at("prompt_per_second").is_null());
  CHECK(!j.at("predicted_per_second").is_null());
  // Round-trip through dump()/parse() -- a NaN/inf would fail nlohmann's own JSON dump (it throws
  // type_error.406 rather than emit invalid JSON), so a successful dump()+re-parse is itself proof
  // there is no NaN/inf hiding in there.
  const std::string dumped = j.dump();
  CHECK(nlohmann::json::parse(dumped).at("prompt_per_second") == 0.0);
}

void TestBuildTimingsJsonDraftFieldsPresentWhenSet() {
  TimingStats t;
  t.prompt_n = 10;
  t.prompt_ms = 10.0;
  t.predicted_n = 20;
  t.predicted_ms = 20.0;
  t.draft_n = 32;
  t.draft_n_accepted = 24;
  const json j = BuildTimingsJson(t);
  CHECK(j.contains("draft_n"));
  CHECK(j.at("draft_n") == 32);
  CHECK(j.contains("draft_n_accepted"));
  CHECK(j.at("draft_n_accepted") == 24);
}

void TestBuildTimingsJsonDraftFieldsAbsentWhenUnset() {
  TimingStats t;
  t.prompt_n = 10;
  t.prompt_ms = 10.0;
  // draft_n/draft_n_accepted left unset -- the "no speculative path ran" case.
  const json j = BuildTimingsJson(t);
  CHECK(!j.contains("draft_n"));
  CHECK(!j.contains("draft_n_accepted"));
}

void TestBuildCompletionResponseAttachesTimings() {
  UsageStats usage{4, 2};
  TimingStats timings;
  timings.prompt_n = 4;
  timings.prompt_ms = 8.0;
  timings.predicted_n = 2;
  timings.predicted_ms = 4.0;
  const json resp = BuildCompletionResponse("id2", "m", 1, "text out", "length", usage, timings);
  CHECK(resp.at("timings").at("prompt_n") == 4);
  CHECK(resp.at("timings").at("predicted_n") == 2);
}

void TestBuildChatCompletionChunkTimingsOnlyWhenProvided() {
  const json noTimings = BuildChatCompletionChunk("id1", "m", 42, json::object(), std::string("stop"));
  CHECK(!noTimings.contains("timings"));
  CHECK(!noTimings.contains("usage"));

  TimingStats t;
  t.prompt_n = 5;
  const json withTimings =
      BuildChatCompletionChunk("id1", "m", 42, json::object(), std::string("stop"), false, t);
  CHECK(withTimings.at("timings").at("prompt_n") == 5);
  CHECK(!withTimings.contains("usage"));

  const json withUsageNull =
      BuildChatCompletionChunk("id1", "m", 42, {{"content", "hi"}}, std::nullopt, true);
  CHECK(withUsageNull.at("usage").is_null());
  CHECK(!withUsageNull.contains("timings"));
}

void TestBuildChatCompletionUsageChunkShape() {
  UsageStats usage{7, 3};
  TimingStats timings;
  timings.prompt_n = 7;
  const json j = BuildChatCompletionUsageChunk("id1", "m", 42, usage, timings);
  CHECK(j.at("object") == "chat.completion.chunk");
  CHECK(j.at("choices").is_array() && j.at("choices").empty());
  CHECK(j.at("usage").at("completion_tokens") == 3);
  CHECK(j.at("timings").at("prompt_n") == 7);
}

void TestBuildCompletionUsageChunkShape() {
  UsageStats usage{4, 2};
  TimingStats timings;
  const json j = BuildCompletionUsageChunk("id2", "m", 1, usage, timings);
  CHECK(j.at("object") == "text_completion");
  CHECK(j.at("choices").is_array() && j.at("choices").empty());
  CHECK(j.at("usage").at("prompt_tokens") == 4);
}

// ---- stream_options (task point 2) --------------------------------------------------------------

void TestStreamOptionsAbsentDefaultsFalse() {
  json body = {{"messages", json::array({{{"role", "user"}, {"content", "hi"}}})}};
  const auto req = ParseChatCompletionRequest(body);
  CHECK(!req.stream_options_include_usage);
}

void TestStreamOptionsIncludeUsageTrueParsed() {
  json body = {{"messages", json::array({{{"role", "user"}, {"content", "hi"}}})},
               {"stream", true},
               {"stream_options", {{"include_usage", true}}}};
  const auto req = ParseChatCompletionRequest(body);
  CHECK(req.stream_options_include_usage);
}

void TestStreamOptionsOnNonStreamingRequestAcceptedAndIgnored() {
  json body = {{"messages", json::array({{{"role", "user"}, {"content", "hi"}}})},
               {"stream", false},
               {"stream_options", {{"include_usage", true}}}};
  const auto req = ParseChatCompletionRequest(body);
  CHECK(!req.stream);
  CHECK(req.stream_options_include_usage);  // parsed regardless -- just unused by a BufferingSink
}

void TestStreamOptionsNotAnObjectThrows() {
  json body = {{"messages", json::array({{{"role", "user"}, {"content", "hi"}}})},
               {"stream_options", "yes"}};
  CHECK(ThrowsApiError([&] { ParseChatCompletionRequest(body); }, 400));
}

void TestStreamOptionsIncludeUsageNotBooleanThrows() {
  json body = {{"messages", json::array({{{"role", "user"}, {"content", "hi"}}})},
               {"stream_options", {{"include_usage", "yes"}}}};
  CHECK(ThrowsApiError([&] { ParseChatCompletionRequest(body); }, 400));
}

void TestCompletionStreamOptionsParsedTheSameWay() {
  json body = {{"prompt", "hi"}, {"stream_options", {{"include_usage", true}}}};
  const auto req = ParseCompletionRequest(body);
  CHECK(req.stream_options_include_usage);

  json bad = {{"prompt", "hi"}, {"stream_options", 5}};
  CHECK(ThrowsApiError([&] { ParseCompletionRequest(bad); }, 400));
}

void TestBuildToolCallsJson() {
  const json arr = BuildToolCallsJson({{"call_1", "get_current_weather", "{\"location\":\"Boston, MA\"}"}});
  CHECK(arr.size() == 1);
  CHECK(arr[0].at("index") == 0);
  CHECK(arr[0].at("id") == "call_1");
  CHECK(arr[0].at("type") == "function");
  CHECK(arr[0].at("function").at("name") == "get_current_weather");
  // arguments is a STRING, not an object -- the field this task flagged as "trips people up".
  CHECK(arr[0].at("function").at("arguments").is_string());
  CHECK(nlohmann::json::parse(arr[0].at("function").at("arguments").get<std::string>()).at("location") ==
        "Boston, MA");
}

void TestBuildChatCompletionResponseWithToolCalls() {
  UsageStats usage{10, 5};
  TimingStats timings;
  const json resp = BuildChatCompletionResponse("id1", "m", 42, std::nullopt,
                                                 {{"call_1", "f", "{}"}}, "tool_calls", usage, timings);
  const json& msg = resp.at("choices")[0].at("message");
  CHECK(msg.at("content").is_null());
  CHECK(msg.at("tool_calls").size() == 1);
  CHECK(msg.at("tool_calls")[0].at("function").at("name") == "f");
  CHECK(resp.at("choices")[0].at("finish_reason") == "tool_calls");
  CHECK(resp.contains("timings"));
}

void TestBuildChatCompletionResponseWithProseAndToolCalls() {
  UsageStats usage{10, 5};
  TimingStats timings;
  const json resp = BuildChatCompletionResponse("id1", "m", 42, std::string("Let me check."),
                                                 {{"call_1", "f", "{}"}}, "tool_calls", usage, timings);
  CHECK(resp.at("choices")[0].at("message").at("content") == "Let me check.");
  CHECK(resp.at("choices")[0].at("message").at("tool_calls").size() == 1);
}

void TestBuildChatCompletionResponseNoToolCallsOmitsField() {
  UsageStats usage{1, 1};
  TimingStats timings;
  const json resp =
      BuildChatCompletionResponse("id1", "m", 42, std::string("hi"), {}, "stop", usage, timings);
  CHECK(!resp.at("choices")[0].at("message").contains("tool_calls"));
}

void TestBuildChatCompletionChunk() {
  const json chunk = BuildChatCompletionChunk("id1", "m", 42, {{"content", "hi"}}, std::nullopt);
  CHECK(chunk.at("object") == "chat.completion.chunk");
  CHECK(chunk.at("choices")[0].at("delta").at("content") == "hi");
  CHECK(chunk.at("choices")[0].at("finish_reason").is_null());

  const json last = BuildChatCompletionChunk("id1", "m", 42, json::object(), std::string("stop"));
  CHECK(last.at("choices")[0].at("finish_reason") == "stop");
}

void TestBuildCompletionResponseAndChunk() {
  UsageStats usage{4, 2};
  TimingStats timings;
  const json resp = BuildCompletionResponse("id2", "m", 1, "text out", "length", usage, timings);
  CHECK(resp.at("object") == "text_completion");
  CHECK(resp.at("choices")[0].at("text") == "text out");
  CHECK(resp.at("choices")[0].at("finish_reason") == "length");

  const json chunk = BuildCompletionChunk("id2", "m", 1, "delta", std::nullopt);
  CHECK(chunk.at("choices")[0].at("text") == "delta");
  CHECK(chunk.at("choices")[0].at("finish_reason").is_null());
}

void TestGenerateRequestIdPrefixAndUniqueness() {
  const std::string a = GenerateRequestId("chatcmpl-");
  const std::string b = GenerateRequestId("chatcmpl-");
  CHECK(a.rfind("chatcmpl-", 0) == 0);
  CHECK(a != b);
}

void TestErrorBody() {
  const json body = ErrorBody(ApiError{400, "invalid_request_error", "bad"});
  CHECK(body.at("error").at("message") == "bad");
  CHECK(body.at("error").at("type") == "invalid_request_error");
}

}  // namespace

int main() {
  TestChatMinimal();
  TestChatSamplingDefaultsApplyOnlyWhenOmitted();
  TestChatFullFields();
  TestChatContentAsTextParts();
  TestChatImagePartRejected();
  TestChatMissingMessagesThrows();
  TestChatEmptyMessagesThrows();
  TestChatUnsupportedRoleThrows();
  TestChatToolRoleRequiresToolCallId();
  TestChatToolRoleAccepted();
  TestChatFunctionRoleRequiresName();
  TestChatFunctionRoleAccepted();
  TestChatAssistantToolCallsRoundTrip();
  TestChatAssistantToolCallsGeneratesIdWhenMissing();
  TestChatAssistantMissingContentAndToolCallsThrows();
  TestChatToolCallsArgumentsMustBeJsonObjectString();
  TestToolChoiceNoneClearsTools();
  TestToolChoiceAutoLeavesToolsUntouched();
  TestToolChoiceDefaultIsAutoLeavesToolsUntouched();
  TestToolChoiceRequiredNeedsNonEmptyTools();
  TestToolChoiceNamedFiltersToJustThatTool();
  TestToolChoiceNamedUnknownFunctionThrows();
  TestToolChoiceUnrecognizedStringThrows();
  TestToolChoiceMalformedObjectThrows();
  TestToolsArrayEntryNotObjectThrows();
  TestToolsArrayEntryMissingFunctionThrows();
  TestToolsArrayEntryFunctionMissingNameThrows();
  TestToolsArrayEntryWrongTypeThrows();
  TestToolsArrayWellFormedEntriesAccepted();
  TestChatMissingContentThrows();
  TestChatBadSamplingRangesThrow();
  TestChatNotAnObjectThrows();
  TestChatStopAsSingleString();
  TestCompletionMinimal();
  TestCompletionMissingPromptThrows();
  TestCompletionPromptWrongTypeThrows();
  TestBuildModelsResponse();
  TestBuildChatCompletionResponse();
  TestBuildToolCallsJson();
  TestBuildChatCompletionResponseWithToolCalls();
  TestBuildChatCompletionResponseWithProseAndToolCalls();
  TestBuildChatCompletionResponseNoToolCallsOmitsField();
  TestBuildChatCompletionChunk();
  TestBuildCompletionResponseAndChunk();
  TestGenerateRequestIdPrefixAndUniqueness();
  TestErrorBody();
  TestBuildTimingsJsonArithmetic();
  TestBuildTimingsJsonZeroMsAvoidsDivideByZero();
  TestBuildTimingsJsonDraftFieldsPresentWhenSet();
  TestBuildTimingsJsonDraftFieldsAbsentWhenUnset();
  TestBuildCompletionResponseAttachesTimings();
  TestBuildChatCompletionChunkTimingsOnlyWhenProvided();
  TestBuildChatCompletionUsageChunkShape();
  TestBuildCompletionUsageChunkShape();
  TestStreamOptionsAbsentDefaultsFalse();
  TestStreamOptionsIncludeUsageTrueParsed();
  TestStreamOptionsOnNonStreamingRequestAcceptedAndIgnored();
  TestStreamOptionsNotAnObjectThrows();
  TestStreamOptionsIncludeUsageNotBooleanThrows();
  TestCompletionStreamOptionsParsedTheSameWay();

  if (g_failures > 0) {
    std::fprintf(stderr, "%d check(s) failed\n", g_failures);
    return 1;
  }
  std::fprintf(stderr, "all openai_types checks passed\n");
  return 0;
}
