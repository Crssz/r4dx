// tests/server/test_gemma_reasoning_splitter.cpp -- pure CPU test of ReasoningSplitter's Gemma 4
// syntax and kExpectOpener start state, plus ReasoningLocator. Every case is checked at EVERY byte
// split (all 2-way cuts, plus byte-by-byte) against the whole-buffer result. The Qwen behavior has
// its own untouched test_reasoning_splitter.cpp.
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

#include "dialect.h"
#include "reasoning_splitter.h"

namespace {

int g_failures = 0;

#define CHECK(cond)                                                                      \
  do {                                                                                   \
    if (!(cond)) {                                                                       \
      std::fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #cond);    \
      ++g_failures;                                                                      \
    }                                                                                    \
  } while (0)

#define CHECK_EQ(a, b)                                                                        \
  do {                                                                                        \
    const std::string _a = (a), _b = (b);                                                     \
    if (_a != _b) {                                                                           \
      std::fprintf(stderr, "CHECK_EQ failed at %s:%d: [%s] != [%s]\n", __FILE__, __LINE__,    \
                   _a.c_str(), _b.c_str());                                                   \
      ++g_failures;                                                                           \
    }                                                                                         \
  } while (0)

using r4dx::server::Gemma4Dialect;
using r4dx::server::ReasoningLocator;
using r4dx::server::ReasoningSplitter;
using Start = r4dx::server::ReasoningSplitter::StartState;

struct Out {
  std::string reasoning, answer;
};

Out Run(const std::vector<std::string>& pieces, Start start) {
  ReasoningSplitter s(Gemma4Dialect().reasoning, start);
  Out o;
  auto collect = [&](const std::vector<ReasoningSplitter::Event>& evs) {
    for (const auto& e : evs) (e.is_reasoning ? o.reasoning : o.answer) += e.text;
  };
  for (const auto& p : pieces) collect(s.Push(p));
  collect(s.Finish());
  return o;
}

// Whole-buffer result must equal every 2-way split and the byte-by-byte split.
void CheckAllSplits(const std::string& text, Start start, const std::string& want_reasoning,
                    const std::string& want_answer) {
  Out whole = Run({text}, start);
  CHECK_EQ(whole.reasoning, want_reasoning);
  CHECK_EQ(whole.answer, want_answer);
  for (size_t i = 0; i <= text.size(); ++i) {
    Out o = Run({text.substr(0, i), text.substr(i)}, start);
    if (o.reasoning != want_reasoning || o.answer != want_answer) {
      std::fprintf(stderr, "split at %zu of [%s] differs\n", i, text.c_str());
      ++g_failures;
    }
  }
  std::vector<std::string> bytes;
  for (char c : text) bytes.emplace_back(1, c);
  Out o = Run(bytes, start);
  CHECK_EQ(o.reasoning, want_reasoning);
  CHECK_EQ(o.answer, want_answer);
  // All 3-way splits.
  if (text.size() <= 80) {
    for (size_t i = 0; i <= text.size(); ++i) {
      for (size_t j = i; j <= text.size(); ++j) {
        Out o3 = Run({text.substr(0, i), text.substr(i, j - i), text.substr(j)}, start);
        if (o3.reasoning != want_reasoning || o3.answer != want_answer) {
          std::fprintf(stderr, "3-split %zu,%zu of [%s] differs\n", i, j, text.c_str());
          ++g_failures;
        }
      }
    }
  }
}

void TestExpectOpenerThinkingThenAnswer() {
  CheckAllSplits("<|channel>thought\nlet me think\n<channel|>The answer.", Start::kExpectOpener,
                 "let me think\n", "The answer.");
}

void TestExpectOpenerBlankLinesAfterCloseDropped() {
  CheckAllSplits("<|channel>thought\nhm<channel|>\n\nAnswer", Start::kExpectOpener, "hm", "Answer");
}

void TestExpectOpenerEmptyThought() {
  // Thinking off prompts end with an empty span; a model that re-emits one still splits cleanly.
  CheckAllSplits("<|channel>thought\n<channel|>hello", Start::kExpectOpener, "", "hello");
}

void TestExpectOpenerNoReasoningPlainAnswer() {
  CheckAllSplits("Just an answer.", Start::kExpectOpener, "", "Just an answer.");
  // No blank-line skipping when there was no close tag.
  CheckAllSplits("\n\nIndented answer", Start::kExpectOpener, "", "\n\nIndented answer");
}

void TestExpectOpenerDivergenceReleasesHeldBytes() {
  // Longest proper prefix of the opener, then divergence: held bytes come back as ANSWER.
  CheckAllSplits("<|channel>thoughX rest", Start::kExpectOpener, "", "<|channel>thoughX rest");
  CheckAllSplits("<|channel>", Start::kExpectOpener, "", "<|channel>");  // ends mid-opener
  CheckAllSplits("<|channel>thought", Start::kExpectOpener, "", "<|channel>thought");
  CheckAllSplits("<", Start::kExpectOpener, "", "<");
  CheckAllSplits("<|tool_call>call:f{}<tool_call|>", Start::kExpectOpener, "",
                 "<|tool_call>call:f{}<tool_call|>");
}

void TestExpectOpenerHoldbackBound() {
  ReasoningSplitter s(Gemma4Dialect().reasoning, Start::kExpectOpener);
  const std::string open = Gemma4Dialect().reasoning.open;
  CHECK(open.size() == 18);  // "<|channel>thought\n" (the plan's "<= 17 held" = size - 1)
  for (size_t n = 1; n < open.size(); ++n) {
    ReasoningSplitter t(Gemma4Dialect().reasoning, Start::kExpectOpener);
    CHECK(t.Push(open.substr(0, n)).empty());
    CHECK(t.mode() == ReasoningSplitter::Mode::kExpectOpener);
  }
  auto ev = s.Push(open);
  CHECK(ev.empty());
  CHECK(s.mode() == ReasoningSplitter::Mode::kReasoning);
}

void TestInReasoningStart() {
  // After a tool response with thinking on the prompt already ends with the opener.
  CheckAllSplits("thinking about the result<channel|>It is 5.", Start::kInReasoning,
                 "thinking about the result", "It is 5.");
  CheckAllSplits("never closed <chan", Start::kInReasoning, "never closed <chan", "");
}

void TestAnswerStart() {
  CheckAllSplits("<channel|> plain", Start::kAnswer, "", "<channel|> plain");
}

void TestReasoningThenToolCall() {
  CheckAllSplits("<|channel>thought\nneed weather<channel|><|tool_call>call:w{c:<|\"|>Paris<|\"|>}<tool_call|>",
                 Start::kExpectOpener, "need weather",
                 "<|tool_call>call:w{c:<|\"|>Paris<|\"|>}<tool_call|>");
}

void TestUtf8AtCharBoundaries() {
  const std::string text = "<|channel>thought\nคิด 思考 \xF0\x9F\x98\x80<channel|>\n\nสวัสดี";
  Out whole = Run({text}, Start::kExpectOpener);
  CHECK_EQ(whole.reasoning, "คิด 思考 \xF0\x9F\x98\x80");
  CHECK_EQ(whole.answer, "สวัสดี");
  // Split only at UTF-8 character starts (the StreamDecoder contract).
  for (size_t i = 0; i <= text.size(); ++i) {
    if (i < text.size() && (static_cast<unsigned char>(text[i]) & 0xC0) == 0x80) continue;
    Out o = Run({text.substr(0, i), text.substr(i)}, Start::kExpectOpener);
    CHECK_EQ(o.reasoning, whole.reasoning);
    CHECK_EQ(o.answer, whole.answer);
  }
}

// ---- ReasoningLocator --------------------------------------------------------------------------------

void TestLocatorGemmaPendingThenReasoningThenAnswer() {
  ReasoningLocator loc(Gemma4Dialect().reasoning, Start::kExpectOpener);
  std::string acc;
  loc.Update(acc);
  CHECK(loc.state() == ReasoningLocator::State::kPending);
  acc = "<|channel>thou";
  loc.Update(acc);
  CHECK(loc.state() == ReasoningLocator::State::kPending);
  CHECK(loc.reasoning_open());
  acc = "<|channel>thought\nabc";
  loc.Update(acc);
  CHECK(loc.state() == ReasoningLocator::State::kReasoning);
  CHECK(loc.reasoning_end() == std::string::npos);
  acc += "<channel";
  loc.Update(acc);
  CHECK(loc.reasoning_open());
  acc += "|>\n\n";
  loc.Update(acc);
  CHECK(loc.state() == ReasoningLocator::State::kAnswer);
  CHECK(loc.reasoning_end() == acc.size() - 2);
  CHECK(loc.AnswerBegin(acc, acc.size()) == std::string::npos);  // only blank lines so far
  acc += "Hi";
  loc.Update(acc);
  CHECK(loc.AnswerBegin(acc, acc.size()) == acc.size() - 2);
  CHECK(loc.AnswerBegin(acc, acc.size() - 1) == acc.size() - 2);
  CHECK(loc.AnswerBegin(acc, acc.size() - 2) == std::string::npos);
}

void TestLocatorGemmaDiverges() {
  ReasoningLocator loc(Gemma4Dialect().reasoning, Start::kExpectOpener);
  std::string acc = "<|tool_call>call:f{}";
  loc.Update(acc);
  CHECK(loc.state() == ReasoningLocator::State::kAnswer);
  CHECK(!loc.reasoning_open());
  CHECK(loc.reasoning_end() == 0);
  CHECK(loc.AnswerBegin(acc, acc.size()) == 0);
  // A later close tag in the answer is not special.
  acc += "<channel|>";
  loc.Update(acc);
  CHECK(loc.reasoning_end() == 0);
}

void TestLocatorQwenMatchesLegacyBookkeeping() {
  const auto& q = r4dx::server::Qwen35Dialect();
  ReasoningLocator on(q.reasoning, Start::kInReasoning);
  std::string acc = "thinking</think>\n\nanswer";
  on.Update(acc);
  CHECK(on.reasoning_end() == acc.find("</think>") + 8);
  CHECK(on.AnswerBegin(acc, acc.size()) == acc.find("answer"));
  ReasoningLocator off(q.reasoning, Start::kAnswer);
  off.Update("anything</think>x");
  CHECK(off.reasoning_end() == 0);
  CHECK(off.AnswerBegin("anything</think>x", 17) == 0);
  ReasoningLocator never(q.reasoning, Start::kInReasoning);
  never.Update("still thinking </thi");
  CHECK(never.reasoning_open());
  CHECK(never.reasoning_end() == std::string::npos);
}

}  // namespace

int main() {
  TestExpectOpenerThinkingThenAnswer();
  TestExpectOpenerBlankLinesAfterCloseDropped();
  TestExpectOpenerEmptyThought();
  TestExpectOpenerNoReasoningPlainAnswer();
  TestExpectOpenerDivergenceReleasesHeldBytes();
  TestExpectOpenerHoldbackBound();
  TestInReasoningStart();
  TestAnswerStart();
  TestReasoningThenToolCall();
  TestUtf8AtCharBoundaries();
  TestLocatorGemmaPendingThenReasoningThenAnswer();
  TestLocatorGemmaDiverges();
  TestLocatorQwenMatchesLegacyBookkeeping();
  if (g_failures != 0) {
    std::fprintf(stderr, "%d failure(s)\n", g_failures);
    return 1;
  }
  std::puts("test_gemma_reasoning_splitter: OK");
  return 0;
}
