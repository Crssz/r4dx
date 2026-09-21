// tests/server/test_reasoning_splitter.cpp -- pure CPU unit test for
// src/server/reasoning_splitter.h's ReasoningSplitter (and its TrimReasoningWhitespace helper):
// the "</think>" close-tag split docs/server.md's "reasoning_content" section documents.
#include <cstdio>
#include <string>
#include <vector>

#include "reasoning_splitter.h"

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

using r4dx::server::ReasoningSplitter;
using r4dx::server::TrimReasoningWhitespace;

// Concatenates every kReasoning/kAnswer event's text, in order, so a test can assert on the whole
// reconstructed reasoning/answer strings without caring how many events they arrived as.
struct Collected {
  std::string reasoning;
  std::string answer;
  int reasoning_event_count = 0;
  int answer_event_count = 0;
};

void Collect(Collected& out, const std::vector<ReasoningSplitter::Event>& events) {
  for (const auto& ev : events) {
    if (ev.is_reasoning) {
      out.reasoning += ev.text;
      ++out.reasoning_event_count;
    } else {
      out.answer += ev.text;
      ++out.answer_event_count;
    }
  }
}

// ---- whole-buffer (one-shot, engine.cpp tool_mode's own use pattern) ---------------------------

void TestWholeBufferBasicSplit() {
  ReasoningSplitter s;
  Collected c;
  Collect(c, s.Push("let me think</think>\n\nthe answer"));
  Collect(c, s.Finish());
  CHECK(c.reasoning == "let me think");
  CHECK(c.answer == "the answer");
}

void TestZeroLengthReasoningBeforeTag() {
  ReasoningSplitter s;
  Collected c;
  Collect(c, s.Push("</think>\n\nanswer"));
  Collect(c, s.Finish());
  CHECK(c.reasoning.empty());
  CHECK(c.reasoning_event_count == 0);  // no empty reasoning event emitted
  CHECK(c.answer == "answer");
}

void TestBlankLinesAfterTagNeverReachAnswer() {
  ReasoningSplitter s;
  Collected c;
  Collect(c, s.Push("thinking</think>\n\n\n\nanswer text"));
  Collect(c, s.Finish());
  CHECK(c.reasoning == "thinking");
  CHECK(c.answer == "answer text");
  CHECK(c.answer.find('\n') == std::string::npos);
}

void TestNoBlankLinesAfterTagPassesThroughImmediately() {
  ReasoningSplitter s;
  Collected c;
  Collect(c, s.Push("thinking</think>immediate answer, no blank line"));
  Collect(c, s.Finish());
  CHECK(c.answer == "immediate answer, no blank line");
}

// ---- never-closed (task item 5a) ----------------------------------------------------------------

void TestNeverClosedEverythingIsReasoning() {
  ReasoningSplitter s;
  Collected c;
  Collect(c, s.Push("the model just keeps thinking and thinking and never stops"));
  Collect(c, s.Finish());
  CHECK(c.reasoning == "the model just keeps thinking and thinking and never stops");
  CHECK(c.answer.empty());
}

void TestNeverClosedPartialTagLikeSuffixFlushedOnFinish() {
  // Ends with a byte sequence that looks like it COULD start "</think>" but generation simply
  // stops there -- Finish() must flush it as reasoning, not silently drop it.
  ReasoningSplitter s;
  Collected c;
  Collect(c, s.Push("some thoughts</thi"));
  CHECK(c.reasoning == "some thoughts");  // "</thi" held back, not yet flushed
  Collect(c, s.Finish());
  CHECK(c.reasoning == "some thoughts</thi");
  CHECK(c.answer.empty());
}

// ---- piece-by-piece streaming, every split position of "</think>" -----------------------------

void TestEveryTagSplitPositionAcrossTwoPieces() {
  const std::string kTag = "</think>";
  const std::string before = "reasoning body";
  const std::string after = "\n\nthe final answer";
  // Split "</think>" itself at every possible byte boundary (0..8) between two Push() calls --
  // covers the tag-holdback logic at every internal state it can be caught in.
  for (size_t split = 0; split <= kTag.size(); ++split) {
    ReasoningSplitter s;
    Collected c;
    Collect(c, s.Push(before + kTag.substr(0, split)));
    Collect(c, s.Push(kTag.substr(split) + after));
    Collect(c, s.Finish());
    CHECK(c.reasoning == before);
    CHECK(c.answer == "the final answer");
  }
}

void TestTagSplitByteByByte() {
  // The most extreme fragmentation: one byte per Push() call, including across the tag itself and
  // the blank lines following it.
  const std::string whole = "abc</think>\n\ndef";
  ReasoningSplitter s;
  Collected c;
  for (char ch : whole) Collect(c, s.Push(std::string(1, ch)));
  Collect(c, s.Finish());
  CHECK(c.reasoning == "abc");
  CHECK(c.answer == "def");
}

// ---- UTF-8 multi-byte content on both sides ----------------------------------------------------

void TestUtf8MultiByteContentBothSides() {
  // U+00E9 (e-acute, 2 bytes), U+4E2D (CJK "middle", 3 bytes), U+1F600 (grinning face, 4 bytes) --
  // deliberately placed adjacent to the tag boundary on both sides.
  const std::string reasoning_text = "caf\xc3\xa9 thinking \xe4\xb8\xad";  // "café thinking 中"
  const std::string answer_text = "\xf0\x9f\x98\x80 done";                // "(emoji) done"
  ReasoningSplitter s;
  Collected c;
  Collect(c, s.Push(reasoning_text + "</think>\n\n" + answer_text));
  Collect(c, s.Finish());
  CHECK(c.reasoning == reasoning_text);
  CHECK(c.answer == answer_text);
}

void TestUtf8MultiByteSplitAcrossPieceBoundaryNearTag() {
  // The multi-byte character sits RIGHT before the tag, and the piece boundary falls between the
  // character's own bytes and the tag -- the tag-holdback logic must never mistake a UTF-8
  // continuation byte (always >= 0x80) for part of "</think>" (all ASCII).
  const std::string cjk = "\xe4\xb8\xad";  // U+4E2D, 3 bytes
  ReasoningSplitter s;
  Collected c;
  Collect(c, s.Push("thoughts " + cjk));
  Collect(c, s.Push("</think>\n\nanswer"));
  Collect(c, s.Finish());
  CHECK(c.reasoning == "thoughts " + cjk);
  CHECK(c.answer == "answer");
}

// ---- once in answer mode, a literal "</think>" in real content is not special-cased -----------

void TestLiteralTagInAnswerPassesThroughUnscanned() {
  ReasoningSplitter s;
  Collected c;
  Collect(c, s.Push("thinking</think>\n\nthe answer mentions </think> literally"));
  Collect(c, s.Finish());
  CHECK(c.reasoning == "thinking");
  CHECK(c.answer == "the answer mentions </think> literally");
}

// ---- composition with a stop sequence: reasoning/answer events stay independently addressable,
// so a caller (Engine::RunRequest) that only applies stop-string matching to kAnswer events can
// never have a stop string inside the model's own chain-of-thought terminate generation early. ---

void TestComposesWithStopSequenceAppliedToAnswerEventsOnly() {
  // A plausible stop string ("\n\n") appears BOTH inside the reasoning span and inside the answer
  // -- the splitter itself must not touch it either way (that is Engine::RunRequest's own job,
  // scoped to accumulated text at/after the answer's start position); this test demonstrates the
  // event tagging gives a caller everything it needs to scope a stop search to the answer alone.
  ReasoningSplitter s;
  Collected c;
  Collect(c, s.Push("first\n\nsecond thought</think>\n\nkeep this\n\nstop after this"));
  Collect(c, s.Finish());
  CHECK(c.reasoning == "first\n\nsecond thought");  // the splitter left the reasoning-side
                                                     // "\n\n" completely untouched
  CHECK(c.answer == "keep this\n\nstop after this");

  // Emulate "search for the first stop string, but only within kAnswer text" (Engine::RunRequest's
  // own `stop_search_floor` composition, see engine.h's EmitToken doc comment) using ONLY the
  // answer-side text the splitter produced -- confirms the match lands in the answer, never in the
  // reasoning span the splitter routed separately.
  const size_t stop_pos = c.answer.find("\n\n");
  CHECK(stop_pos != std::string::npos);
  CHECK(c.answer.substr(0, stop_pos) == "keep this");
}

// ---- TrimReasoningWhitespace --------------------------------------------------------------------

void TestTrimReasoningWhitespaceBothEnds() {
  CHECK(TrimReasoningWhitespace("  \n\t hello world \n ") == "hello world");
}

void TestTrimReasoningWhitespacePreservesInternal() {
  CHECK(TrimReasoningWhitespace("line one\n\nline two") == "line one\n\nline two");
}

void TestTrimReasoningWhitespaceAllWhitespaceBecomesEmpty() {
  CHECK(TrimReasoningWhitespace("   \n\t\r\n  ").empty());
}

void TestTrimReasoningWhitespaceEmptyStaysEmpty() {
  CHECK(TrimReasoningWhitespace("").empty());
}

void TestTrimReasoningWhitespaceNoWhitespaceUnchanged() {
  CHECK(TrimReasoningWhitespace("no whitespace to trim") == "no whitespace to trim");
}

}  // namespace

int main() {
  TestWholeBufferBasicSplit();
  TestZeroLengthReasoningBeforeTag();
  TestBlankLinesAfterTagNeverReachAnswer();
  TestNoBlankLinesAfterTagPassesThroughImmediately();
  TestNeverClosedEverythingIsReasoning();
  TestNeverClosedPartialTagLikeSuffixFlushedOnFinish();
  TestEveryTagSplitPositionAcrossTwoPieces();
  TestTagSplitByteByByte();
  TestUtf8MultiByteContentBothSides();
  TestUtf8MultiByteSplitAcrossPieceBoundaryNearTag();
  TestLiteralTagInAnswerPassesThroughUnscanned();
  TestComposesWithStopSequenceAppliedToAnswerEventsOnly();
  TestTrimReasoningWhitespaceBothEnds();
  TestTrimReasoningWhitespacePreservesInternal();
  TestTrimReasoningWhitespaceAllWhitespaceBecomesEmpty();
  TestTrimReasoningWhitespaceEmptyStaysEmpty();
  TestTrimReasoningWhitespaceNoWhitespaceUnchanged();

  if (g_failures > 0) {
    std::fprintf(stderr, "%d check(s) failed\n", g_failures);
    return 1;
  }
  std::fprintf(stderr, "all reasoning-splitter checks passed\n");
  return 0;
}
