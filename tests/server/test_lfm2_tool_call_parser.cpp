// tests/server/test_lfm2_tool_call_parser.cpp -- LFM2.5's Pythonic tool-call surface syntax
// (lfm2_tool_call_parser.cpp, docs/lfm2-plan.md 5.3) and its streaming pipeline, CPU only:
//   * hand-written vectors: value types, quoting and escapes, JSON-literal arguments from `tojson`,
//     nested containers, several calls in one list, several spans, content around spans;
//   * the HF round trip: every assistant `tool_calls` message of tests/tokenizer/golden_lfm2.json
//     (rendered by HF's jinja through the real chat template) is cut out of the rendered prompt and parsed
//     back; the arguments must equal the original mapping;
//   * malformed input: positional args, bare words, unbalanced brackets, a missing close marker, an empty
//     list, a JSON tool-call list -> plain-text fallback, had_malformed_call, nothing thrown;
//   * EVERY byte split: each prefix of every reply (generation truncated by max_tokens) parses without
//     throwing into either a complete call or verbatim content, and the live pipeline (ReasoningSplitter ->
//     ToolStreamGate -> end-of-generation parse) gives the same content/calls for every 1-, 2- and 3-piece
//     chunking as for the whole text, with the invariant streamed + remainder == parsed.content.
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "dialect.h"
#include "nlohmann/json.hpp"
#include "reasoning_splitter.h"
#include "tool_call_parser.h"
#include "tool_stream_gate.h"

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
using r4dx::server::ToolStreamGate;
using r4dx::server::ToolStreamRemainder;
using Json = nlohmann::ordered_json;

const r4dx::server::ModelDialect& D() { return r4dx::server::Lfm2Dialect(); }

#define OPEN "<|tool_call_start|>"
#define CLOSE "<|tool_call_end|>"

Json Args(const ToolCallParseResult& r, size_t i) { return Json::parse(r.tool_calls.at(i).arguments_json); }

// ---- hand-written vectors ----------------------------------------------------------------------------
void TestSimple() {
  auto r = ParseToolCalls(OPEN "[get_weather(city='Bangkok', units=\"celsius\")]" CLOSE, D());
  CHECK(r.tool_calls.size() == 1 && r.content.empty() && !r.had_malformed_call);
  CHECK(r.tool_calls[0].name == "get_weather");
  CHECK(r.tool_calls[0].arguments_json == "{\"city\":\"Bangkok\",\"units\":\"celsius\"}");  // order kept
}

void TestNoArgsAndWhitespace() {
  auto r = ParseToolCalls(OPEN "[ping()]" CLOSE, D());
  CHECK(r.tool_calls.size() == 1 && r.tool_calls[0].arguments_json == "{}");
  auto r2 = ParseToolCalls(OPEN "\n[ a ( x = 1 ,\n y = 'z' ) ,\n b() ]\n" CLOSE, D());
  CHECK(r2.tool_calls.size() == 2 && !r2.had_malformed_call);
  CHECK(Args(r2, 0)["x"] == 1 && Args(r2, 0)["y"] == "z" && r2.tool_calls[1].name == "b");
}

void TestScalars() {
  auto r = ParseToolCalls(
      OPEN "[f(i=7, neg=-3, f1=1.5, fe=1e-07, fe2=2.5E+10, f3=3.0, t=True, fa=False, n=None, "
           "jt=true, jf=false, jn=null, big=12345678901234567890, dot=.5, tr=5., plus=+4)]" CLOSE, D());
  CHECK(r.tool_calls.size() == 1 && !r.had_malformed_call);
  Json a = Args(r, 0);
  CHECK(a["i"].is_number_integer() && a["i"] == 7 && a["neg"] == -3);
  CHECK(a["f1"] == 1.5 && a["fe"] == 1e-7 && a["fe2"] == 2.5e10 && a["f3"].is_number_float() && a["f3"] == 3.0);
  CHECK(a["t"] == true && a["fa"] == false && a["n"].is_null());
  CHECK(a["jt"] == true && a["jf"] == false && a["jn"].is_null());
  CHECK(a["big"].is_number() && a["dot"] == 0.5 && a["tr"] == 5.0 && a["plus"] == 4);
}

void TestStringEscapes() {
  // exactly what the template renders: backslash, quote, newline and CR escaped; tab and others raw
  auto r = ParseToolCalls(OPEN "[f(a='plain', b='it\\'s \"q\"', c='a\\\\b', d='l1\\nl2\\r\tx', e=\"dq 'x'\", "
                               "f='caf\xC3\xA9 \xE0\xB9\x84\xE0\xB8\x97\xE0\xB8\xA2', g='', h=\"\\\"\")]" CLOSE, D());
  CHECK(r.tool_calls.size() == 1 && !r.had_malformed_call);
  Json a = Args(r, 0);
  CHECK(a["a"] == "plain");
  CHECK(a["b"] == "it's \"q\"");
  CHECK(a["c"] == "a\\b");
  CHECK(a["d"] == "l1\nl2\r\tx");
  CHECK(a["e"] == "dq 'x'");
  CHECK(a["f"] == "caf\xC3\xA9 \xE0\xB9\x84\xE0\xB8\x97\xE0\xB8\xA2");
  CHECK(a["g"] == "" && a["h"] == "\"");
  // JSON escapes inside tojson-rendered nested strings, and Python \x / unknown escapes
  auto r2 = ParseToolCalls(OPEN "[f(l=[\"\\u0001\\n\\t\\\"\\\\\\/\", \"\\ud83d\\ude00 \\u00e9\"], x='\\x41\\q')]" CLOSE, D());
  CHECK(r2.tool_calls.size() == 1);
  Json b = Args(r2, 0);
  CHECK(b["l"][0] == std::string("\x01\n\t\"\\/", 6));
  CHECK(b["l"][1] == "\xF0\x9F\x98\x80 \xC3\xA9");
  CHECK(b["x"] == "A\\q");
}

void TestNesting() {
  auto r = ParseToolCalls(OPEN "[f(lst=[1, \"two\", 3.0, true, null, [4, 5], {\"k\": \"v\"}], empty_l=[], "
                               "dct={\"a\": 1, \"b\": {\"c\": [1, 2]}}, empty_d={}, dq={\"k\": \"it's\"}, "
                               "py={'a': True, 'b': None})]" CLOSE, D());
  CHECK(r.tool_calls.size() == 1 && !r.had_malformed_call);
  Json a = Args(r, 0);
  CHECK(a["lst"].size() == 7 && a["lst"][1] == "two" && a["lst"][5][1] == 5 && a["lst"][6]["k"] == "v");
  CHECK(a["empty_l"].is_array() && a["empty_l"].empty() && a["empty_d"].is_object() && a["empty_d"].empty());
  CHECK(a["dct"]["b"]["c"][1] == 2 && a["dq"]["k"] == "it's");
  CHECK(a["py"]["a"] == true && a["py"]["b"].is_null());
  CHECK(r.tool_calls[0].arguments_json.find("\"lst\":[1,\"two\"") != std::string::npos);
  // trailing commas (Python allows them)
  auto r2 = ParseToolCalls(OPEN "[f(a=[1, 2,], b={\"k\": 1,}, c=3,),]" CLOSE, D());
  CHECK(r2.tool_calls.size() == 1 && Args(r2, 0)["a"].size() == 2 && Args(r2, 0)["c"] == 3);
}

void TestMultipleCallsAndSpans() {
  auto r = ParseToolCalls(OPEN "[a(), b(x=1), c(y='z')]" CLOSE, D());
  CHECK(r.tool_calls.size() == 3 && r.tool_calls[0].name == "a" && r.tool_calls[2].name == "c");
  CHECK(r.tool_calls[1].arguments_json == "{\"x\":1}");
  auto r2 = ParseToolCalls("Before. " OPEN "[a(x=1)]" CLOSE " mid " OPEN "[b(y=2), c()]" CLOSE " after", D());
  CHECK(r2.tool_calls.size() == 3 && r2.content == "Before.  mid  after" && !r2.had_malformed_call);
  auto r3 = ParseToolCalls("no call here", D());
  CHECK(r3.tool_calls.empty() && r3.content == "no call here" && !r3.had_malformed_call);
  CHECK(ParseToolCalls("", D()).content.empty());
}

void TestNames() {
  auto r = ParseToolCalls(OPEN "[mcp__srv__tool-1(a-b='x', c.d=2), ns.fn()]" CLOSE, D());
  CHECK(r.tool_calls.size() == 2 && r.tool_calls[0].name == "mcp__srv__tool-1" && r.tool_calls[1].name == "ns.fn");
  CHECK(Args(r, 0)["a-b"] == "x" && Args(r, 0)["c.d"] == 2);
}

void TestStringsAreDelimiterAware() {
  // the close marker, brackets, parentheses and commas inside a string do not end the span or the call
  auto r = ParseToolCalls(OPEN "[f(s='a,b)](c" CLOSE "\"q\"', t='')]" CLOSE "tail", D());
  CHECK(r.tool_calls.size() == 1 && !r.had_malformed_call);
  CHECK(Args(r, 0)["s"] == "a,b)](c" CLOSE "\"q\"" && Args(r, 0)["t"] == "");
  CHECK(r.content == "tail");
}

void TestMalformed() {
  const char* const bad[] = {
      OPEN "[f('positional')]" CLOSE,                 // positional string
      OPEN "[f(1)]" CLOSE,                            // positional number
      OPEN "[f(city=Bangkok)]" CLOSE,                 // bare identifier value
      OPEN "[f(a=1]" CLOSE,                           // unbalanced paren
      OPEN "[f(a=1)" CLOSE,                           // unbalanced bracket
      OPEN "f(a=1)" CLOSE,                            // no list
      OPEN "[]" CLOSE,                                // empty list
      OPEN "[f(a=)]" CLOSE,                           // missing value
      OPEN "[f(a='unterminated)]" CLOSE,              // unterminated string
      OPEN "[f(a=[1, 2)]" CLOSE,                      // unbalanced list
      OPEN "[f(a={'k'})]" CLOSE,                      // dict without value
      OPEN "[f(a={1: 2})]" CLOSE,                     // non-string key
      OPEN "[f(a=1e999)]" CLOSE,                      // not representable
      OPEN "[f(a=007)]" CLOSE,                        // leading zeros
      OPEN "[{\"name\": \"f\", \"arguments\": {}}]" CLOSE,  // the JSON tool-call dialect (not parsed in M1)
      OPEN "[f(a=1)]",                                // truncated: no close marker
      OPEN "[f(a=1), g(b=",                           // truncated mid-call
      OPEN,                                           // bare opener
      OPEN CLOSE,                                     // empty span
      OPEN "[f(a=1)] trailing" CLOSE,                 // junk between ']' and the close marker
      OPEN "[f(a=1)(b=2)]" CLOSE,                     // double call
  };
  for (const char* b : bad) {
    const std::string text = std::string("pre ") + b + " post";
    auto r = ParseToolCalls(text, D());
    CHECK(r.tool_calls.empty());
    CHECK(r.had_malformed_call);
    CHECK(r.content == text);  // verbatim, markers included
  }
  // one bad span does not poison a good one
  auto r = ParseToolCalls(OPEN "[f('x')]" CLOSE " then " OPEN "[g(a=1)]" CLOSE, D());
  CHECK(r.tool_calls.size() == 1 && r.tool_calls[0].name == "g" && r.had_malformed_call);
  CHECK(r.content == OPEN "[f('x')]" CLOSE " then ");
  // deep nesting does not blow the stack
  std::string deep = OPEN "[f(a=";
  for (int i = 0; i < 5000; ++i) deep += "[";
  for (int i = 0; i < 5000; ++i) deep += "]";
  deep += ")]" CLOSE;
  auto r2 = ParseToolCalls(deep, D());
  CHECK(r2.tool_calls.empty() && r2.had_malformed_call && r2.content == deep);
  // invalid UTF-8 in a string argument is replaced, never thrown
  auto r3 = ParseToolCalls(std::string(OPEN "[f(a='\xFF\xFE')]" CLOSE), D());
  CHECK(r3.tool_calls.size() == 1);
}

// ---- HF-rendered round trip ----------------------------------------------------------------------------
void TestGoldenRoundTrip() {
#ifdef R4DX_LFM2_GOLDEN
  std::ifstream f(R4DX_LFM2_GOLDEN, std::ios::binary);
  if (!f) {
    std::printf("test_lfm2_tool_call_parser: golden_lfm2.json not found, skipping the HF round trip\n");
    return;
  }
  std::ostringstream ss;
  ss << f.rdbuf();
  const Json doc = Json::parse(ss.str());
  size_t calls_checked = 0, cases = 0;
  for (const Json& c : doc.at("cases")) {
    if (c.at("kind") != "chat") continue;
    // The expected calls, in order, from the request's messages.
    std::vector<std::pair<std::string, Json>> expected;
    for (const Json& m : c.at("messages")) {
      if (!m.contains("tool_calls") || m.at("tool_calls").is_null()) continue;
      for (const Json& tc : m.at("tool_calls")) {
        const Json& fn = tc.contains("function") ? tc.at("function") : tc;
        // a JSON-string "{}" (the only string the template accepts) renders as no arguments
        expected.emplace_back(fn.at("name").get<std::string>(),
                              fn.at("arguments").is_string() ? Json::object() : fn.at("arguments"));
      }
    }
    // The spans in the rendered prompt, in order.
    const std::string prompt = c.at("prompt").get<std::string>();
    std::vector<std::pair<std::string, Json>> got;
    size_t pos = 0;
    while (true) {
      const size_t a = prompt.find(OPEN, pos);
      if (a == std::string::npos) break;
      const size_t b = prompt.find(CLOSE, a);
      CHECK(b != std::string::npos);
      if (b == std::string::npos) break;
      const std::string span = prompt.substr(a, b + std::string(CLOSE).size() - a);
      auto r = ParseToolCalls(span, D());
      CHECK(!r.had_malformed_call && r.content.empty());
      for (const auto& tc : r.tool_calls) got.emplace_back(tc.name, Json::parse(tc.arguments_json));
      pos = b + 1;
    }
    if (expected.empty() && got.empty()) continue;
    ++cases;
    CHECK(got.size() == expected.size());
    for (size_t i = 0; i < std::min(got.size(), expected.size()); ++i) {
      ++calls_checked;
      const bool same = got[i].first == expected[i].first && got[i].second == expected[i].second;
      if (!same) {
        std::fprintf(stderr, "round trip mismatch in case %s call %zu:\n  expected %s\n  got      %s\n",
                     c.at("name").get<std::string>().c_str(), i, expected[i].second.dump().c_str(),
                     got[i].second.dump().c_str());
      }
      CHECK(same);
    }
  }
  CHECK(calls_checked >= 8);
  std::printf("test_lfm2_tool_call_parser: HF round trip, %zu rendered cases, %zu calls\n", cases, calls_checked);
#endif
}

// ---- every byte split ------------------------------------------------------------------------------------
const char* const kReplies[] = {
    // reasoning, answer, one call
    "Let me think.\nThe user wants weather.</think>Sure!" OPEN "[get_weather(city='Bangkok')]" CLOSE,
    // no reasoning close at all (the model stopped inside the span), plain text
    "thinking only",
    // answer prose only
    "I need no tool.</think>\n\nHere is the answer, with <b>tags</b> and a < sign.",
    // several calls, nested args, unicode, an escaped quote
    "hmm</think>" OPEN "[a(x=1, l=[1, \"two\", {\"k\": null}]), b(s='it\\'s caf\xC3\xA9 \xE0\xB8\x81')]" CLOSE,
    // text around two spans
    "t</think>pre " OPEN "[a()]" CLOSE " mid " OPEN "[b(y=2)]" CLOSE " post",
    // a malformed span among prose
    "t</think>pre " OPEN "[f('x')]" CLOSE " post",
    // generation cut inside the span
    "t</think>go " OPEN "[f(a='abc",
    // tag-like prose that is not the opener
    "t</think><|tool_call|> <|tool_call_start| <|tool_call_start|x",
    // opener split candidates at the very start of the answer
    "</think>" OPEN "[x(a=1)]" CLOSE,
};

// The live pipeline over `pieces`; returns what a streaming client would see (content, reasoning) plus the
// end-of-generation parse of the answer text.
struct PipelineResult {
  std::string streamed_content;
  std::string reasoning;
  std::string answer;  // the ANSWER half, as the gate saw it
  ToolCallParseResult parsed;
  std::string content_total;  // streamed + remainder
  bool gate_closed = false;
};

PipelineResult RunPipeline(const std::vector<std::string>& pieces) {
  using r4dx::server::ReasoningSplitter;
  PipelineResult out;
  ReasoningSplitter splitter(D().reasoning, ReasoningSplitter::StartState::kInReasoning);
  ToolStreamGate gate(D().tool_open);
  auto feed = [&](const std::vector<ReasoningSplitter::Event>& events) {
    for (const auto& ev : events) {
      if (ev.is_reasoning) {
        out.reasoning += ev.text;
      } else {
        out.answer += ev.text;
        out.streamed_content += gate.Push(ev.text);
      }
    }
  };
  for (const auto& p : pieces) feed(splitter.Push(p));
  feed(splitter.Finish());
  out.streamed_content += gate.Finish();
  out.gate_closed = gate.closed();
  out.parsed = ParseToolCalls(out.answer, D());
  out.content_total = out.streamed_content + ToolStreamRemainder(out.streamed_content, out.parsed.content);
  return out;
}

void ExpectSamePipeline(const PipelineResult& whole, const PipelineResult& got, const std::string& label) {
  const bool ok = got.reasoning == whole.reasoning && got.answer == whole.answer &&
                  got.parsed.content == whole.parsed.content &&
                  got.parsed.tool_calls.size() == whole.parsed.tool_calls.size() &&
                  got.parsed.had_malformed_call == whole.parsed.had_malformed_call &&
                  got.content_total == got.parsed.content && got.gate_closed == whole.gate_closed &&
                  got.streamed_content.size() <= got.parsed.content.size() &&
                  got.parsed.content.compare(0, got.streamed_content.size(), got.streamed_content) == 0;
  if (!ok) std::fprintf(stderr, "pipeline mismatch: %s\n", label.c_str());
  CHECK(ok);
  for (size_t i = 0; i < got.parsed.tool_calls.size() && i < whole.parsed.tool_calls.size(); ++i) {
    CHECK(got.parsed.tool_calls[i].name == whole.parsed.tool_calls[i].name);
    CHECK(got.parsed.tool_calls[i].arguments_json == whole.parsed.tool_calls[i].arguments_json);
  }
}

void TestEveryByteSplit() {
  for (const char* reply_c : kReplies) {
    const std::string text = reply_c;
    const PipelineResult whole = RunPipeline({text});
    // The opener never leaks to the client: a closed gate streamed nothing from the opener on.
    if (whole.gate_closed) CHECK(whole.streamed_content.find("<|tool_call_start|>") == std::string::npos);
    // Byte by byte.
    {
      std::vector<std::string> bytes;
      for (char ch : text) bytes.emplace_back(1, ch);
      ExpectSamePipeline(whole, RunPipeline(bytes), "bytes of: " + text);
    }
    // Every 2-piece split, every 3-piece split.
    for (size_t i = 0; i <= text.size(); ++i) {
      ExpectSamePipeline(whole, RunPipeline({text.substr(0, i), text.substr(i)}), "2-split " + std::to_string(i));
    }
    if (text.size() <= 140) {
      for (size_t i = 0; i <= text.size(); ++i) {
        for (size_t j = i; j <= text.size(); ++j) {
          ExpectSamePipeline(whole, RunPipeline({text.substr(0, i), text.substr(i, j - i), text.substr(j)}),
                             "3-split " + std::to_string(i) + "," + std::to_string(j));
        }
      }
    }
    // The gate never streams a partial opener while the generation is still running.
    for (size_t i = 0; i <= text.size(); ++i) {
      ToolStreamGate gate(D().tool_open);
      const std::string a = gate.Push(text.substr(0, i));
      CHECK(a.find("<|tool_call_start|>") == std::string::npos);
      for (size_t k = 1; k < 19 && k <= a.size(); ++k) {
        CHECK(a.substr(a.size() - k) != std::string("<|tool_call_start|>").substr(0, k));  // no dangling prefix
      }
    }
  }
}

void TestEveryPrefixParses() {
  // A generation cut at any byte (max_tokens): never throws, a call only when its span is complete, the
  // rest verbatim content.
  const std::string full = std::string("go ") + OPEN "[a(x=1, s='q\\'z'), b(l=[1, 2])]" CLOSE " tail";
  for (size_t i = 0; i <= full.size(); ++i) {
    const std::string p = full.substr(0, i);
    const ToolCallParseResult r = ParseToolCalls(p, D());
    const size_t span_end = full.find(CLOSE) + std::string(CLOSE).size();
    if (i >= span_end) {
      CHECK(r.tool_calls.size() == 2 && !r.had_malformed_call);
      CHECK(r.content == "go " + p.substr(span_end));
    } else if (i >= 3 + std::string(OPEN).size()) {
      CHECK(r.tool_calls.empty() && r.had_malformed_call && r.content == p);
    } else {
      CHECK(r.tool_calls.empty() && r.content == p);  // before the whole opener is there: plain text
    }
  }
}

}  // namespace

int main() {
  TestSimple();
  TestNoArgsAndWhitespace();
  TestScalars();
  TestStringEscapes();
  TestNesting();
  TestMultipleCallsAndSpans();
  TestNames();
  TestStringsAreDelimiterAware();
  TestMalformed();
  TestGoldenRoundTrip();
  TestEveryByteSplit();
  TestEveryPrefixParses();
  if (g_failures != 0) {
    std::fprintf(stderr, "%d failure(s)\n", g_failures);
    return 1;
  }
  std::puts("test_lfm2_tool_call_parser: OK");
  return 0;
}
