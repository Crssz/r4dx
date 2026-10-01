// tests/server/test_gemma_tool_stream_gate.cpp -- ToolStreamGate with the Gemma 4 opener
// "<|tool_call>" (holdback 11), at every byte split, plus the streamed-vs-parsed invariant against
// ParseToolCalls(text, gemma dialect). The Qwen gate has its own untouched test.
#include <cstdio>
#include <string>
#include <vector>

#include "dialect.h"
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

using r4dx::server::Gemma4Dialect;
using r4dx::server::ToolStreamGate;

struct Result {
  std::string streamed;
  bool closed;
  size_t streamed_bytes;
  size_t max_hold;
};

Result Run(const std::vector<std::string>& pieces) {
  ToolStreamGate g(Gemma4Dialect().tool_open);
  Result r{};
  size_t fed = 0;
  for (const auto& p : pieces) {
    r.streamed += g.Push(p);
    fed += p.size();
    if (!g.closed()) {
      const size_t held = fed - r.streamed.size();
      if (held > r.max_hold) r.max_hold = held;
    }
  }
  r.streamed += g.Finish();
  r.closed = g.closed();
  r.streamed_bytes = g.streamed_bytes();
  return r;
}

void CheckAllSplits(const std::string& text, const std::string& want_streamed, bool want_closed) {
  const Result whole = Run({text});
  CHECK(whole.streamed == want_streamed);
  CHECK(whole.closed == want_closed);
  CHECK(whole.streamed_bytes == want_streamed.size());
  for (size_t i = 0; i <= text.size(); ++i) {
    const Result r = Run({text.substr(0, i), text.substr(i)});
    if (r.streamed != want_streamed || r.closed != want_closed || r.streamed_bytes != want_streamed.size()) {
      std::fprintf(stderr, "split %zu of [%s] differs: got [%s]\n", i, text.c_str(), r.streamed.c_str());
      ++g_failures;
    }
  }
  std::vector<std::string> bytes;
  for (char c : text) bytes.emplace_back(1, c);
  const Result b = Run(bytes);
  CHECK(b.streamed == want_streamed);
  CHECK(b.closed == want_closed);
  CHECK(b.max_hold <= 11);  // strlen("<|tool_call>") - 1
}

void TestPlainProseStreamsEntirely() {
  CheckAllSplits("Just prose, no call.", "Just prose, no call.", false);
}

void TestStopsAtOpener() {
  CheckAllSplits("Sure.<|tool_call>call:f{a:1}<tool_call|>", "Sure.", true);
  CheckAllSplits("<|tool_call>call:f{}<tool_call|>", "", true);
}

void TestNearMissOpenersAreReleased() {
  CheckAllSplits("a <|tool_cal b", "a <|tool_cal b", false);
  CheckAllSplits("x<|tool_call", "x<|tool_call", false);     // ends one byte short of the opener
  CheckAllSplits("<|channel>", "<|channel>", false);         // not a tool opener
  CheckAllSplits("<tool_call>call:f{}", "<tool_call>call:f{}", false);  // Qwen opener is plain text
  CheckAllSplits("<<|tool_call>call:f{}<tool_call|>", "<", true);
}

void TestStreamedPlusRemainderEqualsParsedContent() {
  const auto& d = Gemma4Dialect();
  const std::vector<std::string> texts = {
      "Let me check.<|tool_call>call:get{city:<|\"|>Paris<|\"|>}<tool_call|>",
      "Pre <|tool_call>call:broken{<tool_call|> post",
      "Only prose <|tool_c",
  };
  for (const auto& text : texts) {
    for (size_t i = 0; i <= text.size(); ++i) {
      ToolStreamGate g(d.tool_open);
      std::string streamed = g.Push(text.substr(0, i));
      streamed += g.Push(text.substr(i));
      streamed += g.Finish();
      const auto parsed = r4dx::server::ParseToolCalls(text, d);
      CHECK(streamed + r4dx::server::ToolStreamRemainder(streamed, parsed.content) == parsed.content);
    }
  }
}

}  // namespace

int main() {
  TestPlainProseStreamsEntirely();
  TestStopsAtOpener();
  TestNearMissOpenersAreReleased();
  TestStreamedPlusRemainderEqualsParsedContent();
  if (g_failures != 0) {
    std::fprintf(stderr, "%d failure(s)\n", g_failures);
    return 1;
  }
  std::puts("test_gemma_tool_stream_gate: OK");
  return 0;
}
