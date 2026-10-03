// tests/server/test_gemma_tool_call_parser.cpp -- Gemma 4 tool-call syntax (gemma_tool_call_parser.cpp).
// Vectors are hand-written from google/gemma-4-12B-it's chat_template.jinja render direction; replay
// the HF `tok.parse_response` vectors from M1-9 here once generated.
#include <cstdio>
#include <string>

#include "nlohmann/json.hpp"
#include "dialect.h"
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

using r4dx::server::ParseToolCalls;
using r4dx::server::ToolCallParseResult;
using Json = nlohmann::json;

const r4dx::server::ModelDialect& D() { return r4dx::server::Gemma4Dialect(); }

#define Q "<|\"|>"

void TestSimpleCall() {
  auto r = ParseToolCalls("<|tool_call>call:get_weather{city:" Q "Paris" Q ",days:3}<tool_call|>", D());
  CHECK(r.tool_calls.size() == 1);
  CHECK(r.content.empty());
  CHECK(!r.had_malformed_call);
  CHECK(r.tool_calls[0].name == "get_weather");
  Json a = Json::parse(r.tool_calls[0].arguments_json);
  CHECK(a["city"] == "Paris");
  CHECK(a["days"] == 3);
  // Encounter order is preserved.
  CHECK(r.tool_calls[0].arguments_json == "{\"city\":\"Paris\",\"days\":3}");
}

void TestNoArguments() {
  auto r = ParseToolCalls("<|tool_call>call:ping{}<tool_call|>", D());
  CHECK(r.tool_calls.size() == 1 && r.tool_calls[0].arguments_json == "{}");
  auto r2 = ParseToolCalls("<|tool_call>call:ping<tool_call|>", D());
  CHECK(r2.tool_calls.size() == 1 && r2.tool_calls[0].arguments_json == "{}");
}

void TestScalarsAndNesting() {
  auto r = ParseToolCalls(
      "<|tool_call>call:f{b:true,c:false,n:null,x:-1.5e2,i:42,o:{k:" Q "v" Q ",in:{z:1}},"
      "l:[1," Q "two" Q ",[3],{a:4}],e:[],eo:{}}<tool_call|>",
      D());
  CHECK(r.tool_calls.size() == 1);
  Json a = Json::parse(r.tool_calls[0].arguments_json);
  CHECK(a["b"] == true && a["c"] == false && a["n"].is_null());
  CHECK(a["x"] == -150.0);
  CHECK(a["i"].is_number_integer() && a["i"] == 42);
  CHECK(a["o"]["k"] == "v" && a["o"]["in"]["z"] == 1);
  CHECK(a["l"].size() == 4 && a["l"][1] == "two" && a["l"][2][0] == 3 && a["l"][3]["a"] == 4);
  CHECK(a["e"].is_array() && a["e"].empty());
  CHECK(a["eo"].is_object() && a["eo"].empty());
}

void TestStringsAreDelimiterAware() {
  // Delimiters inside strings: commas, braces, the close marker, newlines, quotes.
  auto r = ParseToolCalls(
      "<|tool_call>call:f{s:" Q "a,b}{c]<tool_call|>\n\"q\"" Q ",t:" Q Q "}<tool_call|>tail", D());
  CHECK(r.tool_calls.size() == 1);
  Json a = Json::parse(r.tool_calls[0].arguments_json);
  CHECK(a["s"] == "a,b}{c]<tool_call|>\n\"q\"");
  CHECK(a["t"] == "");
  CHECK(r.content == "tail");
}

void TestQuotedKeysAndBareWords() {
  auto r = ParseToolCalls("<|tool_call>call:f{" Q "my key" Q ":1,mode:fast,id:007}<tool_call|>", D());
  CHECK(r.tool_calls.size() == 1);
  Json a = Json::parse(r.tool_calls[0].arguments_json);
  CHECK(a["my key"] == 1);
  CHECK(a["mode"] == "fast");  // bare word -> string
  CHECK(a["id"] == "007");     // not valid JSON number -> string
}

void TestUnicodeStrings() {
  auto r = ParseToolCalls("<|tool_call>call:f{t:" Q "สวัสดี 世界 \xF0\x9F\x98\x80" Q "}<tool_call|>", D());
  CHECK(r.tool_calls.size() == 1);
  Json a = Json::parse(r.tool_calls[0].arguments_json);
  CHECK(a["t"] == "สวัสดี 世界 \xF0\x9F\x98\x80");
}

void TestMultipleCallsAndSurroundingContent() {
  auto r = ParseToolCalls("Intro. <|tool_call>call:a{x:1}<tool_call|>\n<|tool_call>call:b{y:" Q "z" Q
                          "}<tool_call|> outro",
                          D());
  CHECK(r.tool_calls.size() == 2);
  CHECK(r.tool_calls[0].name == "a" && r.tool_calls[1].name == "b");
  CHECK(r.content == "Intro. \n outro");
  CHECK(!r.had_malformed_call);
}

void TestNoCallIsUnchanged() {
  auto r = ParseToolCalls("plain <|channel> text", D());
  CHECK(r.tool_calls.empty() && r.content == "plain <|channel> text" && !r.had_malformed_call);
  auto e = ParseToolCalls("", D());
  CHECK(e.tool_calls.empty() && e.content.empty());
}

void TestMalformedDegradesToContent() {
  const std::string bad[] = {
      "<|tool_call>call:f{a:1<tool_call|>",            // missing }
      "<|tool_call>call:f{a:" Q "unterminated}<tool_call|>",
      "<|tool_call>notcall:f{}<tool_call|>",
      "<|tool_call>call:{a:1}<tool_call|>",            // empty name
      "<|tool_call>call:f{a:}<tool_call|>",            // empty value
      "<|tool_call>call:f{a:1}garbage<tool_call|>",    // junk before close
      "<|tool_call>call:f{:1}<tool_call|>",            // empty key
  };
  for (const auto& b : bad) {
    auto r = ParseToolCalls("pre " + b + " post", D());
    CHECK(r.tool_calls.empty());
    CHECK(r.had_malformed_call);
    CHECK(r.content == "pre " + b + " post");
  }
}

void TestUnclosedSpan() {
  auto r = ParseToolCalls("pre <|tool_call>call:f{a:1}", D());
  CHECK(r.tool_calls.empty() && r.had_malformed_call);
  CHECK(r.content == "pre <|tool_call>call:f{a:1}");
}

void TestMalformedThenGoodRecovers() {
  auto r = ParseToolCalls("<|tool_call>call:f{a:<tool_call|><|tool_call>call:g{b:2}<tool_call|>", D());
  CHECK(r.tool_calls.size() == 1 && r.tool_calls[0].name == "g");
  CHECK(r.had_malformed_call);
  CHECK(r.content == "<|tool_call>call:f{a:<tool_call|>");
}

void TestDeepNestingDoesNotCrash() {
  std::string deep = "<|tool_call>call:f{a:";
  for (int i = 0; i < 5000; ++i) deep += "[";
  for (int i = 0; i < 5000; ++i) deep += "]";
  deep += "}<tool_call|>";
  auto r = ParseToolCalls(deep, D());
  CHECK(r.tool_calls.empty() && r.had_malformed_call);
}

void TestQwenDispatchUnchanged() {
  const std::string q = "<tool_call>\n<function=f>\n<parameter=a>\n1\n</parameter>\n</function>\n</tool_call>";
  auto a = ParseToolCalls(q, r4dx::server::Qwen35Dialect());
  auto b = ParseToolCalls(q);
  CHECK(a.tool_calls.size() == 1 && b.tool_calls.size() == 1);
  CHECK(a.tool_calls[0].arguments_json == b.tool_calls[0].arguments_json);
  CHECK(a.content == b.content);
}

}  // namespace

int main() {
  TestSimpleCall();
  TestNoArguments();
  TestScalarsAndNesting();
  TestStringsAreDelimiterAware();
  TestQuotedKeysAndBareWords();
  TestUnicodeStrings();
  TestMultipleCallsAndSurroundingContent();
  TestNoCallIsUnchanged();
  TestMalformedDegradesToContent();
  TestUnclosedSpan();
  TestMalformedThenGoodRecovers();
  TestDeepNestingDoesNotCrash();
  TestQwenDispatchUnchanged();
  if (g_failures != 0) {
    std::fprintf(stderr, "%d failure(s)\n", g_failures);
    return 1;
  }
  std::puts("test_gemma_tool_call_parser: OK");
  return 0;
}
