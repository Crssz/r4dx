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

// ---- reasoning_content on messages (task item 5e) -----------------------------------------------

void TestChatAssistantReasoningContentAccepted() {
  json body = {{"messages",
                json::array({{{"role", "user"}, {"content", "hi"}},
                              {{"role", "assistant"},
                               {"content", "the answer"},
                               {"reasoning_content", "the reasoning"}}})}};
  const auto req = ParseChatCompletionRequest(body);
  CHECK(req.messages[1].reasoning_content.has_value());
  CHECK(*req.messages[1].reasoning_content == "the reasoning");
}

void TestChatReasoningContentAbsentByDefault() {
  json body = {{"messages", json::array({{{"role", "user"}, {"content", "hi"}}})}};
  const auto req = ParseChatCompletionRequest(body);
  CHECK(!req.messages[0].reasoning_content.has_value());
}

void TestChatReasoningContentNonStringThrows() {
  json body = {{"messages",
                json::array({{{"role", "assistant"}, {"content", "x"}, {"reasoning_content", 5}}})}};
  CHECK(ThrowsApiError([&] { ParseChatCompletionRequest(body); }, 400));
}

void TestChatReasoningContentNullIsSameAsAbsent() {
  json body = {{"messages",
                json::array({{{"role", "assistant"}, {"content", "x"}, {"reasoning_content", nullptr}}})}};
  const auto req = ParseChatCompletionRequest(body);
  CHECK(!req.messages[0].reasoning_content.has_value());
}

// ---- ResolveEnableThinking (task item 5) --------------------------------------------------------

void TestResolveEnableThinkingUsesKwargsWhenPresent() {
  CHECK(ResolveEnableThinking(json{{"enable_thinking", true}}, false) == true);
  CHECK(ResolveEnableThinking(json{{"enable_thinking", false}}, true) == false);
}

void TestResolveEnableThinkingFallsBackToDefaultWhenAbsent() {
  CHECK(ResolveEnableThinking(json::object(), true) == true);
  CHECK(ResolveEnableThinking(json::object(), false) == false);
}

void TestResolveEnableThinkingFallsBackToDefaultWhenNonBoolean() {
  CHECK(ResolveEnableThinking(json{{"enable_thinking", "yes"}}, false) == false);
  CHECK(ResolveEnableThinking(json{{"enable_thinking", "yes"}}, true) == true);
}

void TestResolveEnableThinkingFallsBackToDefaultWhenNull() {
  CHECK(ResolveEnableThinking(json{{"enable_thinking", nullptr}}, true) == true);
}

// ---- thinking controls (ParseThinkingControls / ResolveEnableThinking(ThinkingControls, bool)) --
//
// Every wire spelling an OpenAI-compatible client uses to turn thinking on/off, its precedence,
// and the effort mapping onto the three levels chat_template.jinja itself accepts.

json ThinkBody(json extra) {
  json body = {{"messages", json::array({{{"role", "user"}, {"content", "hi"}}})}};
  body.update(extra);
  return body;
}

// Parses a full chat request and returns its reduced controls -- exercises the real path
// (ParseChatCompletionRequest calls ParseThinkingControls), not just the helper in isolation.
ThinkingControls Controls(json extra) {
  return ParseChatCompletionRequest(ThinkBody(std::move(extra))).thinking;
}

void TestThinkingAbsentLeavesEnabledUnsetAndFollowsServerDefault() {
  const ThinkingControls c = Controls(json::object());
  CHECK(!c.enabled.has_value());
  CHECK(!c.template_effort.has_value());
  CHECK(c.include_reasoning);
  CHECK(ResolveEnableThinking(c, true) == true);
  CHECK(ResolveEnableThinking(c, false) == false);
}

void TestThinkingChatTemplateKwargsStillWins() {
  const ThinkingControls on = Controls({{"chat_template_kwargs", {{"enable_thinking", true}}}});
  CHECK(ResolveEnableThinking(on, false) == true);
  const ThinkingControls off = Controls({{"chat_template_kwargs", {{"enable_thinking", false}}}});
  CHECK(ResolveEnableThinking(off, true) == false);
  // ... over every other spelling, including ones that disagree with it.
  const ThinkingControls conflict =
      Controls({{"chat_template_kwargs", {{"enable_thinking", false}}},
                 {"reasoning", {{"enabled", true}}},
                 {"thinking", {{"type", "enabled"}}},
                 {"enable_thinking", true},
                 {"reasoning_effort", "high"}});
  CHECK(ResolveEnableThinking(conflict, true) == false);
  // A non-boolean there is still tolerated (opaque template passthrough) and falls through.
  const ThinkingControls non_bool =
      Controls({{"chat_template_kwargs", {{"enable_thinking", "yes"}}}, {"enable_thinking", true}});
  CHECK(ResolveEnableThinking(non_bool, false) == true);
}

void TestThinkingTopLevelEnableThinking() {
  CHECK(ResolveEnableThinking(Controls({{"enable_thinking", true}}), false) == true);
  CHECK(ResolveEnableThinking(Controls({{"enable_thinking", false}}), true) == false);
  CHECK(ThrowsApiError([] { Controls({{"enable_thinking", "yes"}}); }, 400));
  // null is "absent", not an error.
  CHECK(!Controls({{"enable_thinking", nullptr}}).enabled.has_value());
}

void TestThinkingReasoningEffortString() {
  CHECK(ResolveEnableThinking(Controls({{"reasoning_effort", "none"}}), true) == false);
  for (const char* level : {"minimal", "low", "medium", "high", "xhigh", "max", "invented"}) {
    CHECK(ResolveEnableThinking(Controls({{"reasoning_effort", level}}), false) == true);
  }
  CHECK(ThrowsApiError([] { Controls({{"reasoning_effort", 3}}); }, 400));
  CHECK(ThrowsApiError([] { Controls({{"reasoning_effort", "   "}}); }, 400));
}

void TestThinkingEffortMapsOntoTemplateLevels() {
  // The template accepts exactly xhigh/medium/low; everything else must be mapped or dropped, or
  // chat_template.jinja's own raise_exception turns an ordinary request into a 400.
  CHECK(Controls({{"reasoning_effort", "minimal"}}).template_effort == std::string("low"));
  CHECK(Controls({{"reasoning_effort", "low"}}).template_effort == std::string("low"));
  CHECK(Controls({{"reasoning_effort", "medium"}}).template_effort == std::string("medium"));
  CHECK(Controls({{"reasoning_effort", "high"}}).template_effort == std::string("xhigh"));
  CHECK(Controls({{"reasoning_effort", "xhigh"}}).template_effort == std::string("xhigh"));
  CHECK(Controls({{"reasoning_effort", "max"}}).template_effort == std::string("xhigh"));
  // Case/whitespace insensitive, like every real client spells it.
  CHECK(Controls({{"reasoning_effort", "  HIGH "}}).template_effort == std::string("xhigh"));
  // "none" is an off-switch, not a level; an unrecognized level means "on, template default".
  CHECK(!Controls({{"reasoning_effort", "none"}}).template_effort.has_value());
  CHECK(!Controls({{"reasoning_effort", "invented"}}).template_effort.has_value());
  // The object's own effort wins over the top-level one.
  CHECK(Controls({{"reasoning", {{"effort", "low"}}}, {"reasoning_effort", "high"}}).template_effort ==
        std::string("low"));
}

void TestThinkingOpenRouterReasoningObject() {
  CHECK(ResolveEnableThinking(Controls({{"reasoning", {{"enabled", true}}}}), false) == true);
  CHECK(ResolveEnableThinking(Controls({{"reasoning", {{"enabled", false}}}}), true) == false);
  CHECK(ResolveEnableThinking(Controls({{"reasoning", {{"effort", "high"}}}}), false) == true);
  CHECK(ResolveEnableThinking(Controls({{"reasoning", {{"effort", "none"}}}}), true) == false);
  // `enabled` wins over `effort` inside the object.
  CHECK(ResolveEnableThinking(Controls({{"reasoning", {{"enabled", false}, {"effort", "high"}}}}),
                               true) == false);
  // max_tokens is validated (so a malformed body is a clean 400) but not acted on.
  CHECK(ResolveEnableThinking(
            Controls({{"reasoning", {{"enabled", true}, {"max_tokens", 2048}}}}), false) == true);
  CHECK(ThrowsApiError([] { Controls({{"reasoning", "high"}}); }, 400));
  CHECK(ThrowsApiError([] { Controls({{"reasoning", {{"enabled", "yes"}}}}); }, 400));
  CHECK(ThrowsApiError([] { Controls({{"reasoning", {{"effort", 1}}}}); }, 400));
  CHECK(ThrowsApiError([] { Controls({{"reasoning", {{"exclude", "yes"}}}}); }, 400));
  CHECK(ThrowsApiError([] { Controls({{"reasoning", {{"max_tokens", -1}}}}); }, 400));
  CHECK(ThrowsApiError([] { Controls({{"reasoning", {{"max_tokens", 1.5}}}}); }, 400));
}

void TestThinkingAnthropicThinkingObject() {
  CHECK(ResolveEnableThinking(Controls({{"thinking", {{"type", "enabled"}}}}), false) == true);
  CHECK(ResolveEnableThinking(Controls({{"thinking", {{"type", "disabled"}}}}), true) == false);
  CHECK(ResolveEnableThinking(
            Controls({{"thinking", {{"type", "enabled"}, {"budget_tokens", 1024}}}}), false) == true);
  CHECK(ThrowsApiError([] { Controls({{"thinking", "enabled"}}); }, 400));
  CHECK(ThrowsApiError([] { Controls({{"thinking", {{"type", "on"}}}}); }, 400));
  CHECK(ThrowsApiError([] { Controls({{"thinking", {{"type", true}}}}); }, 400));
  CHECK(ThrowsApiError([] { Controls({{"thinking", {{"budget_tokens", -5}}}}); }, 400));
}

void TestThinkingPrecedenceOrderBelowChatTemplateKwargs() {
  // reasoning > thinking > enable_thinking > reasoning_effort, each proven by making the
  // higher-precedence field disagree with every lower one.
  CHECK(ResolveEnableThinking(Controls({{"reasoning", {{"enabled", false}}},
                                         {"thinking", {{"type", "enabled"}}},
                                         {"enable_thinking", true},
                                         {"reasoning_effort", "high"}}),
                               true) == false);
  CHECK(ResolveEnableThinking(
            Controls({{"thinking", {{"type", "disabled"}}}, {"enable_thinking", true},
                       {"reasoning_effort", "high"}}),
            true) == false);
  CHECK(ResolveEnableThinking(
            Controls({{"enable_thinking", false}, {"reasoning_effort", "high"}}), true) == false);
}

void TestThinkingIncludeReasoningAndExclude() {
  CHECK(Controls(json::object()).include_reasoning);
  CHECK(!Controls({{"reasoning", {{"enabled", true}, {"exclude", true}}}}).include_reasoning);
  CHECK(!Controls({{"include_reasoning", false}}).include_reasoning);
  CHECK(Controls({{"include_reasoning", true}}).include_reasoning);
  // Either one asking for exclusion wins over the other asking for inclusion.
  CHECK(!Controls({{"reasoning", {{"exclude", true}}}, {"include_reasoning", true}}).include_reasoning);
  // Excluding the TEXT says nothing about whether thinking happens.
  CHECK(ResolveEnableThinking(Controls({{"include_reasoning", false}}), true) == true);
  CHECK(!Controls({{"include_reasoning", false}}).enabled.has_value());
  CHECK(ThrowsApiError([] { Controls({{"include_reasoning", "no"}}); }, 400));
}

void TestThinkingControlsOnCompletionsEndpointAreIgnored() {
  // /v1/completions has no chat template and therefore no thinking concept at all -- a body
  // carrying these fields is still parsed (they simply have nowhere to go), not rejected.
  json body = {{"prompt", "hello"}, {"reasoning_effort", "high"}, {"enable_thinking", true}};
  const auto req = ParseCompletionRequest(body);
  CHECK(req.prompt == "hello");
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
  const json resp = BuildModelsResponse("my-model", 1000, 262144);
  CHECK(resp.at("object") == "list");
  CHECK(resp.at("data").size() == 1);
  CHECK(resp.at("data")[0].at("id") == "my-model");
  CHECK(resp.at("data")[0].at("object") == "model");
}

// ---- model metadata (task item 1) --------------------------------------------------------------

void TestBuildModelEntryJsonCoreFieldsUnchanged() {
  const json m = BuildModelEntryJson("my-model", 1000, 65536);
  CHECK(m.at("id") == "my-model");
  CHECK(m.at("object") == "model");
  CHECK(m.at("created") == 1000);
  CHECK(m.at("owned_by") == "r4dx");
}

void TestBuildModelEntryJsonContextLengthFields() {
  const json m = BuildModelEntryJson("my-model", 1000, 65536);
  CHECK(m.at("context_length") == 65536);
  CHECK(m.at("max_model_len") == 65536);
  CHECK(m.at("max_completion_tokens") == 65536);
  CHECK(m.at("meta").at("n_ctx") == 65536);
  CHECK(m.at("meta").at("n_ctx_train") == kModelNativeContextLength);
  CHECK(kModelNativeContextLength == 262144);
}

void TestBuildModelEntryJsonCapabilitiesAndArchitecture() {
  const json m = BuildModelEntryJson("my-model", 1000, 65536);
  const json& caps = m.at("capabilities");
  CHECK(caps.is_array());
  auto has = [&](const char* v) {
    for (const auto& c : caps) if (c == v) return true;
    return false;
  };
  CHECK(has("completion"));
  CHECK(has("chat"));
  CHECK(has("tool_use"));
  CHECK(has("reasoning"));
  CHECK(m.at("architecture").at("input_modalities")[0] == "text");
  CHECK(m.at("architecture").at("output_modalities")[0] == "text");
  // The modality list has exactly one writer, so the vision milestone flips it in one place.
  CHECK(m.at("architecture").at("input_modalities") == ModelInputModalities());
  CHECK(ModelInputModalities().size() == 1);
}

// ---- OpenRouter-shaped capability block (docs/server.md's "Model metadata") ---------------------

void TestBuildModelEntryJsonTopProviderBlock() {
  const json m = BuildModelEntryJson("my-model", 1000, 65536);
  CHECK(m.at("top_provider").at("context_length") == 65536);
  CHECK(m.at("top_provider").at("max_completion_tokens") == 65536);
  CHECK(m.at("top_provider").at("is_moderated") == false);
}

void TestBuildModelEntryJsonReasoningBlockFollowsThinkFlag() {
  const json off = BuildModelEntryJson("my-model", 1000, 65536, /*default_thinking=*/false);
  CHECK(off.at("reasoning").at("default_enabled") == false);
  CHECK(off.at("reasoning").at("mandatory") == false);
  CHECK(off.at("reasoning").at("supported_efforts") == ModelSupportedReasoningEfforts());

  const json on = BuildModelEntryJson("my-model", 1000, 65536, /*default_thinking=*/true);
  CHECK(on.at("reasoning").at("default_enabled") == true);

  // Every advertised effort must be one ParseThinkingControls really accepts.
  for (const auto& effort : ModelSupportedReasoningEfforts()) {
    json body = {{"messages", json::array({{{"role", "user"}, {"content", "hi"}}})},
                  {"reasoning_effort", effort}};
    const ThinkingControls c = ParseChatCompletionRequest(body).thinking;
    CHECK(c.enabled.has_value());
    CHECK(*c.enabled == (effort != "none"));
  }
}

void TestBuildModelsResponsePassesDefaultThinkingThrough() {
  const json resp = BuildModelsResponse("my-model", 1000, 65536, /*default_thinking=*/true);
  CHECK(resp.at("data")[0].at("reasoning").at("default_enabled") == true);
}

void TestBuildModelEntryJsonSupportedParametersOnlyListsParsedFields() {
  const json m = BuildModelEntryJson("my-model", 1000, 65536);
  const json& sp = m.at("supported_parameters");
  CHECK(sp.is_array());
  auto has = [&](const char* v) {
    for (const auto& c : sp) if (c == v) return true;
    return false;
  };
  for (const char* expected : {"temperature", "top_p", "top_k", "min_p", "seed", "max_tokens",
                               "max_completion_tokens", "stop", "stream", "stream_options", "tools",
                               "tool_choice", "chat_template_kwargs", "reasoning",
                               "reasoning_effort", "include_reasoning", "enable_thinking",
                               "thinking"}) {
    CHECK(has(expected));
  }
  // Fields this server does not parse at all must not be falsely advertised.
  CHECK(!has("logprobs"));
  CHECK(!has("presence_penalty"));
  CHECK(!has("frequency_penalty"));
  CHECK(!has("n"));
}

void TestBuildModelsResponseWrapsSingleEntry() {
  const json resp = BuildModelsResponse("my-model", 1000, 65536);
  CHECK(resp.at("data")[0].at("context_length") == 65536);
  CHECK(resp.at("data")[0].at("meta").at("n_ctx_train") == kModelNativeContextLength);
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

// ---- reasoning_content / completion_tokens_details (task item 5) --------------------------------

void TestBuildChatCompletionResponsePlainOverloadOmitsReasoningContentByDefault() {
  UsageStats usage{7, 3};
  TimingStats timings;
  const json resp = BuildChatCompletionResponse("id1", "m", 42, "hello", "stop", usage, timings);
  CHECK(!resp.at("choices")[0].at("message").contains("reasoning_content"));
  CHECK(!resp.at("usage").contains("completion_tokens_details"));
}

void TestBuildChatCompletionResponsePlainOverloadWithReasoningContent() {
  UsageStats usage{7, 3};
  usage.reasoning_tokens = 2;
  TimingStats timings;
  const json resp = BuildChatCompletionResponse("id1", "m", 42, "hello", "stop", usage, timings,
                                                 std::string("my reasoning"));
  CHECK(resp.at("choices")[0].at("message").at("reasoning_content") == "my reasoning");
  CHECK(resp.at("choices")[0].at("message").at("content") == "hello");
  CHECK(resp.at("usage").at("completion_tokens_details").at("reasoning_tokens") == 2);
}

void TestBuildChatCompletionResponseToolCallsOverloadWithReasoningContent() {
  UsageStats usage{10, 5};
  usage.reasoning_tokens = 3;
  TimingStats timings;
  const json resp = BuildChatCompletionResponse("id1", "m", 42, std::string("Let me check."),
                                                 {{"call_1", "f", "{}"}}, "tool_calls", usage, timings,
                                                 std::string("thinking about the tool"));
  CHECK(resp.at("choices")[0].at("message").at("reasoning_content") == "thinking about the tool");
  CHECK(resp.at("choices")[0].at("message").at("tool_calls").size() == 1);
  CHECK(resp.at("usage").at("completion_tokens_details").at("reasoning_tokens") == 3);
}

void TestUsageJsonOmitsCompletionTokensDetailsWhenReasoningTokensUnset() {
  UsageStats usage{4, 2};  // reasoning_tokens left unset -- the thinking-off / completions path
  TimingStats timings;
  const json resp = BuildCompletionResponse("id2", "m", 1, "text out", "length", usage, timings);
  CHECK(!resp.at("usage").contains("completion_tokens_details"));
}

void TestChatCompletionUsageChunkCarriesCompletionTokensDetailsWhenSet() {
  UsageStats usage{7, 3};
  usage.reasoning_tokens = 1;
  TimingStats timings;
  const json j = BuildChatCompletionUsageChunk("id1", "m", 42, usage, timings);
  CHECK(j.at("usage").at("completion_tokens_details").at("reasoning_tokens") == 1);
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
  TestChatAssistantReasoningContentAccepted();
  TestChatReasoningContentAbsentByDefault();
  TestChatReasoningContentNonStringThrows();
  TestChatReasoningContentNullIsSameAsAbsent();
  TestResolveEnableThinkingUsesKwargsWhenPresent();
  TestResolveEnableThinkingFallsBackToDefaultWhenAbsent();
  TestResolveEnableThinkingFallsBackToDefaultWhenNonBoolean();
  TestResolveEnableThinkingFallsBackToDefaultWhenNull();
  TestThinkingAbsentLeavesEnabledUnsetAndFollowsServerDefault();
  TestThinkingChatTemplateKwargsStillWins();
  TestThinkingTopLevelEnableThinking();
  TestThinkingReasoningEffortString();
  TestThinkingEffortMapsOntoTemplateLevels();
  TestThinkingOpenRouterReasoningObject();
  TestThinkingAnthropicThinkingObject();
  TestThinkingPrecedenceOrderBelowChatTemplateKwargs();
  TestThinkingIncludeReasoningAndExclude();
  TestThinkingControlsOnCompletionsEndpointAreIgnored();
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
  TestBuildModelEntryJsonCoreFieldsUnchanged();
  TestBuildModelEntryJsonContextLengthFields();
  TestBuildModelEntryJsonCapabilitiesAndArchitecture();
  TestBuildModelEntryJsonSupportedParametersOnlyListsParsedFields();
  TestBuildModelEntryJsonTopProviderBlock();
  TestBuildModelEntryJsonReasoningBlockFollowsThinkFlag();
  TestBuildModelsResponsePassesDefaultThinkingThrough();
  TestBuildModelsResponseWrapsSingleEntry();
  TestBuildChatCompletionResponse();
  TestBuildToolCallsJson();
  TestBuildChatCompletionResponseWithToolCalls();
  TestBuildChatCompletionResponseWithProseAndToolCalls();
  TestBuildChatCompletionResponseNoToolCallsOmitsField();
  TestBuildChatCompletionResponsePlainOverloadOmitsReasoningContentByDefault();
  TestBuildChatCompletionResponsePlainOverloadWithReasoningContent();
  TestBuildChatCompletionResponseToolCallsOverloadWithReasoningContent();
  TestUsageJsonOmitsCompletionTokensDetailsWhenReasoningTokensUnset();
  TestChatCompletionUsageChunkCarriesCompletionTokensDetailsWhenSet();
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
