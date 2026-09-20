// tests/server/test_tool_call_parser.cpp -- pure CPU unit test for src/server/tool_call_parser.h.
//
// The "real capture" fixtures below are byte-for-byte what the real 64-layer container
// (D:\models\r4dx\qwen38-27b-v3.r4dx, w4a16, HIP device 1) actually emitted for a
// get_current_weather tool definition, greedy decode, captured via a live r4dx-server request
// during this task's own investigation step (docs/server.md's "Tool calls" section has the full
// request/response transcripts) -- not hand-invented text. The malformed-input cases are
// synthetic (by construction, since a real model rarely misbehaves on demand), covering this
// task's own explicit robustness list.
#include <cstdio>
#include <string>

#include "nlohmann/json.hpp"
#include "tool_call_parser.h"

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

using namespace r4dx::server;

// ---- real captures -----------------------------------------------------------------------

void TestRealSingleCallOneParam() {
  // Verbatim, non-streaming /v1/chat/completions response content for
  // "What is the weather like in Boston, MA right now? Use the tool.", temperature 0.
  const std::string text =
      "<tool_call>\n<function=get_current_weather>\n<parameter=location>\nBoston, MA\n"
      "</parameter>\n<parameter=unit>\nfahrenheit\n</parameter>\n</function>\n</tool_call>";
  const ToolCallParseResult r = ParseToolCalls(text);
  CHECK(!r.had_malformed_call);
  CHECK(r.content.empty());
  CHECK(r.tool_calls.size() == 1);
  CHECK(r.tool_calls[0].name == "get_current_weather");
  const nlohmann::json args = nlohmann::json::parse(r.tool_calls[0].arguments_json);
  CHECK(args.at("location") == "Boston, MA");
  CHECK(args.at("unit") == "fahrenheit");
}

void TestRealTwoCallsInOneTurn() {
  // Verbatim capture for "...weather like in Boston, MA and in Tokyo right now? Call the tool
  // once per city." -- two back-to-back <tool_call> blocks, no separator beyond the model's own
  // literal newline (confirmed real behavior, not assumed).
  const std::string text =
      "<tool_call>\n<function=get_current_weather>\n<parameter=location>\nBoston, MA\n"
      "</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=get_current_weather>\n"
      "<parameter=location>\nTokyo, JP\n</parameter>\n</function>\n</tool_call>";
  const ToolCallParseResult r = ParseToolCalls(text);
  CHECK(!r.had_malformed_call);
  CHECK(r.tool_calls.size() == 2);
  CHECK(r.tool_calls[0].name == "get_current_weather");
  CHECK(nlohmann::json::parse(r.tool_calls[0].arguments_json).at("location") == "Boston, MA");
  CHECK(r.tool_calls[1].name == "get_current_weather");
  CHECK(nlohmann::json::parse(r.tool_calls[1].arguments_json).at("location") == "Tokyo, JP");
  // The one literal "\n" the model emitted between the two blocks is prose outside both spans.
  CHECK(r.content == "\n");
}

void TestRealThinkingBlockPrecedesCall() {
  // Verbatim capture with chat_template_kwargs.enable_thinking=true: the model's own generated
  // text does NOT include the opening "<think>\n" (that's part of the prompt's own generation
  // preamble, never generated), but DOES include the closing "</think>\n\n".
  const std::string text =
      "The user is asking about the current weather in Boston, Massachusetts, and is requesting "
      "to use a tool. I should call the get_current_weather tool with location \"Boston, MA\".\n"
      "</think>\n\n<tool_call>\n<function=get_current_weather>\n<parameter=location>\nBoston, MA\n"
      "</parameter>\n</function>\n</tool_call>";
  const ToolCallParseResult r = ParseToolCalls(text);
  CHECK(!r.had_malformed_call);
  CHECK(r.tool_calls.size() == 1);
  CHECK(r.tool_calls[0].name == "get_current_weather");
  // The reasoning block is untouched, literal prose (no reasoning_content extraction -- out of
  // scope, see docs/server.md).
  CHECK(r.content.find("</think>") != std::string::npos);
  CHECK(r.content.find("<tool_call>") == std::string::npos);
}

// ---- plain text (the overwhelmingly common case) ------------------------------------------

void TestNoToolCallPassesThroughUnchanged() {
  const std::string text = "Silicon threads weave,\nParallel light in the dark.";
  const ToolCallParseResult r = ParseToolCalls(text);
  CHECK(!r.had_malformed_call);
  CHECK(r.tool_calls.empty());
  CHECK(r.content == text);
}

void TestEmptyString() {
  const ToolCallParseResult r = ParseToolCalls("");
  CHECK(!r.had_malformed_call);
  CHECK(r.tool_calls.empty());
  CHECK(r.content.empty());
}

// ---- argument type inference ---------------------------------------------------------------

void TestArgumentTypeInference() {
  const std::string text =
      "<tool_call>\n<function=f>\n"
      "<parameter=n>\n42\n</parameter>\n"
      "<parameter=flag>\ntrue\n</parameter>\n"
      "<parameter=arr>\n[1, 2, 3]\n</parameter>\n"
      "<parameter=obj>\n{\"a\": 1}\n</parameter>\n"
      "<parameter=nul>\nnull\n</parameter>\n"
      "<parameter=s>\nBoston, MA\n</parameter>\n"
      "<parameter=numeric_string_lookalike>\n007\n</parameter>\n"  // NOT valid JSON (leading 0)
      "</function>\n</tool_call>";
  const ToolCallParseResult r = ParseToolCalls(text);
  CHECK(!r.had_malformed_call);
  CHECK(r.tool_calls.size() == 1);
  const nlohmann::json args = nlohmann::json::parse(r.tool_calls[0].arguments_json);
  CHECK(args.at("n").is_number_integer() && args.at("n") == 42);
  CHECK(args.at("flag").is_boolean() && args.at("flag") == true);
  CHECK(args.at("arr").is_array() && args.at("arr").size() == 3);
  CHECK(args.at("obj").is_object() && args.at("obj").at("a") == 1);
  CHECK(args.at("nul").is_null());
  CHECK(args.at("s").is_string() && args.at("s") == "Boston, MA");
  CHECK(args.at("numeric_string_lookalike").is_string() && args.at("numeric_string_lookalike") == "007");
}

void TestOverflowingNumericParameterValueDegradesToString() {
  // Regression test (review finding, 2026-09-20): "1e999"/"-1e999"/"1e309" are syntactically
  // valid JSON numbers but overflow `double`, so nlohmann::json::parse throws
  // `json::out_of_range` (406) -- a SIBLING of `json::parse_error`, not a subclass. A narrower
  // `catch (const nlohmann::json::parse_error&)` in ParseParameters let this escape uncaught,
  // violating ParseToolCalls' own documented "never throws" contract and reaching
  // Engine::RunRequest's catch(std::exception&), which invalidates the prefix-reuse cache and
  // returns a 500 for what should have been a normal response.
  const std::string text =
      "<tool_call>\n<function=f>\n"
      "<parameter=huge>\n1e999\n</parameter>\n"
      "<parameter=neg_huge>\n-1e999\n</parameter>\n"
      "<parameter=also_huge>\n1e309\n</parameter>\n"
      "<parameter=normal>\n42\n</parameter>\n"
      "</function>\n</tool_call>";
  const ToolCallParseResult r = ParseToolCalls(text);  // must not throw
  CHECK(!r.had_malformed_call);
  CHECK(r.tool_calls.size() == 1);
  const nlohmann::json args = nlohmann::json::parse(r.tool_calls[0].arguments_json);
  CHECK(args.at("huge").is_string() && args.at("huge") == "1e999");
  CHECK(args.at("neg_huge").is_string() && args.at("neg_huge") == "-1e999");
  CHECK(args.at("also_huge").is_string() && args.at("also_huge") == "1e309");
  CHECK(args.at("normal").is_number_integer() && args.at("normal") == 42);
}

void TestMultilineParameterValuePreserved() {
  const std::string text =
      "<tool_call>\n<function=f>\n<parameter=p>\nline one\nline two\nline three\n</parameter>\n"
      "</function>\n</tool_call>";
  const ToolCallParseResult r = ParseToolCalls(text);
  CHECK(!r.had_malformed_call);
  const nlohmann::json args = nlohmann::json::parse(r.tool_calls[0].arguments_json);
  CHECK(args.at("p") == "line one\nline two\nline three");
}

void TestCallWithNoParameters() {
  const std::string text = "<tool_call>\n<function=ping>\n</function>\n</tool_call>";
  const ToolCallParseResult r = ParseToolCalls(text);
  CHECK(!r.had_malformed_call);
  CHECK(r.tool_calls.size() == 1);
  CHECK(r.tool_calls[0].name == "ping");
  CHECK(r.tool_calls[0].arguments_json == "{}");
}

void TestProseBeforeAndAfterCallPreserved() {
  const std::string text = "Let me check that.\n<tool_call>\n<function=f>\n</function>\n</tool_call>\nDone.";
  const ToolCallParseResult r = ParseToolCalls(text);
  CHECK(!r.had_malformed_call);
  CHECK(r.tool_calls.size() == 1);
  CHECK(r.content == "Let me check that.\n\nDone.");
}

// ---- malformed input: every case must degrade, never throw/crash --------------------------

void TestUnclosedToolCallTag() {
  const std::string text = "Sure, calling it now.\n<tool_call>\n<function=f>\n<parameter=p>\nv\n";
  const ToolCallParseResult r = ParseToolCalls(text);
  CHECK(r.had_malformed_call);
  CHECK(r.tool_calls.empty());
  CHECK(r.content == text);  // nothing lost, just not parsed as a call
}

void TestMissingFunctionTag() {
  const std::string text = "<tool_call>\nI want to call something but forgot the tag\n</tool_call>";
  const ToolCallParseResult r = ParseToolCalls(text);
  CHECK(r.had_malformed_call);
  CHECK(r.tool_calls.empty());
  CHECK(r.content == text);
}

void TestUnclosedFunctionNameTag() {
  const std::string text = "<tool_call>\n<function=f\n<parameter=p>\nv\n</parameter>\n</function>\n</tool_call>";
  const ToolCallParseResult r = ParseToolCalls(text);
  CHECK(r.had_malformed_call);
  CHECK(r.tool_calls.empty());
}

void TestMissingFunctionCloseTag() {
  const std::string text = "<tool_call>\n<function=f>\n<parameter=p>\nv\n</parameter>\n</tool_call>";
  const ToolCallParseResult r = ParseToolCalls(text);
  CHECK(r.had_malformed_call);
  CHECK(r.tool_calls.empty());
}

void TestMissingParameterCloseTag() {
  const std::string text = "<tool_call>\n<function=f>\n<parameter=p>\nv\n</function>\n</tool_call>";
  const ToolCallParseResult r = ParseToolCalls(text);
  CHECK(r.had_malformed_call);
  CHECK(r.tool_calls.empty());
}

void TestUnclosedParameterNameTag() {
  const std::string text = "<tool_call>\n<function=f>\n<parameter=p\nv\n</parameter>\n</function>\n</tool_call>";
  const ToolCallParseResult r = ParseToolCalls(text);
  CHECK(r.had_malformed_call);
  CHECK(r.tool_calls.empty());
}

void TestProseMixedInsideCallBeforeFunctionTag() {
  const std::string text =
      "<tool_call>\nOops I meant to say\n<function=f>\n</function>\n</tool_call>";
  const ToolCallParseResult r = ParseToolCalls(text);
  CHECK(r.had_malformed_call);
  CHECK(r.tool_calls.empty());
  CHECK(r.content == text);
}

void TestProseMixedInsideCallAfterFunctionClose() {
  const std::string text =
      "<tool_call>\n<function=f>\n</function>\nand also some trailing prose\n</tool_call>";
  const ToolCallParseResult r = ParseToolCalls(text);
  CHECK(r.had_malformed_call);
  CHECK(r.tool_calls.empty());
}

void TestProseMixedBetweenParameters() {
  const std::string text =
      "<tool_call>\n<function=f>\n<parameter=a>\n1\n</parameter>\nstray text\n"
      "<parameter=b>\n2\n</parameter>\n</function>\n</tool_call>";
  const ToolCallParseResult r = ParseToolCalls(text);
  CHECK(r.had_malformed_call);
  CHECK(r.tool_calls.empty());
}

void TestMalformedCallDoesNotAffectOtherCallsInSameTurn() {
  const std::string text =
      "<tool_call>\n<function=good_one>\n<parameter=x>\n1\n</parameter>\n</function>\n</tool_call>\n"
      "<tool_call>\nbroken\n</tool_call>\n"
      "<tool_call>\n<function=also_good>\n</function>\n</tool_call>";
  const ToolCallParseResult r = ParseToolCalls(text);
  CHECK(r.had_malformed_call);
  CHECK(r.tool_calls.size() == 2);
  CHECK(r.tool_calls[0].name == "good_one");
  CHECK(r.tool_calls[1].name == "also_good");
  CHECK(r.content.find("<tool_call>\nbroken\n</tool_call>") != std::string::npos);
}

void TestEmptyFunctionName() {
  const std::string text = "<tool_call>\n<function=>\n</function>\n</tool_call>";
  const ToolCallParseResult r = ParseToolCalls(text);
  CHECK(r.had_malformed_call);
  CHECK(r.tool_calls.empty());
}

// ---- DropUnknownToolCalls -------------------------------------------------------------------

void TestDropUnknownToolCall() {
  const std::string text =
      "<tool_call>\n<function=defined_fn>\n</function>\n</tool_call>\n"
      "<tool_call>\n<function=made_up_fn>\n<parameter=x>\n1\n</parameter>\n</function>\n</tool_call>";
  ToolCallParseResult r = ParseToolCalls(text);
  CHECK(r.tool_calls.size() == 2);
  const int dropped = DropUnknownToolCalls(r, {"defined_fn"});
  CHECK(dropped == 1);
  CHECK(r.tool_calls.size() == 1);
  CHECK(r.tool_calls[0].name == "defined_fn");
  CHECK(r.content.find("made_up_fn") != std::string::npos);
}

void TestDropUnknownToolCallNoOpWhenKnownNamesEmpty() {
  const std::string text = "<tool_call>\n<function=anything>\n</function>\n</tool_call>";
  ToolCallParseResult r = ParseToolCalls(text);
  const int dropped = DropUnknownToolCalls(r, {});
  CHECK(dropped == 0);
  CHECK(r.tool_calls.size() == 1);
}

}  // namespace

int main() {
  TestRealSingleCallOneParam();
  TestRealTwoCallsInOneTurn();
  TestRealThinkingBlockPrecedesCall();
  TestNoToolCallPassesThroughUnchanged();
  TestEmptyString();
  TestArgumentTypeInference();
  TestOverflowingNumericParameterValueDegradesToString();
  TestMultilineParameterValuePreserved();
  TestCallWithNoParameters();
  TestProseBeforeAndAfterCallPreserved();
  TestUnclosedToolCallTag();
  TestMissingFunctionTag();
  TestUnclosedFunctionNameTag();
  TestMissingFunctionCloseTag();
  TestMissingParameterCloseTag();
  TestUnclosedParameterNameTag();
  TestProseMixedInsideCallBeforeFunctionTag();
  TestProseMixedInsideCallAfterFunctionClose();
  TestProseMixedBetweenParameters();
  TestMalformedCallDoesNotAffectOtherCallsInSameTurn();
  TestEmptyFunctionName();
  TestDropUnknownToolCall();
  TestDropUnknownToolCallNoOpWhenKnownNamesEmpty();

  if (g_failures > 0) {
    std::fprintf(stderr, "%d check(s) failed\n", g_failures);
    return 1;
  }
  std::fprintf(stderr, "all tool_call_parser checks passed\n");
  return 0;
}
