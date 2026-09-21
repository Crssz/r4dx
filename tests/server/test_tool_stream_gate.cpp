// tests/server/test_tool_stream_gate.cpp -- pure CPU unit test for src/server/tool_stream_gate.h's
// ToolStreamGate and its ToolStreamRemainder helper: the live "stream prose, shut off at the first
// <tool_call>" gate docs/server.md's "Tool calls" section documents.
//
// The heart of this file is TestPropertyEveryChunkingOfEveryGeneration below: for a set of
// representative generations, and for EVERY way of splitting each one into pieces (byte-by-byte,
// then every single 2-piece split point), it asserts the invariant the whole design rests on --
//
//   streamed_concat + ToolStreamRemainder(streamed_concat, ParseToolCalls(whole).content)
//       == ParseToolCalls(whole).content
//
// -- i.e. what a STREAMING client receives as content deltas is byte-identical to the
// `message.content` the non-streaming path returns for the same generation, no matter how the
// tokenizer happened to chop the text up. It also asserts no streamed piece ever leaks a whole or
// partial "<tool_call>" opener when the span turns out to be a well-formed call.
#include <cstdio>
#include <string>
#include <vector>

#include "tool_call_parser.h"
#include "tool_stream_gate.h"

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

using r4dx::server::ParseToolCalls;
using r4dx::server::ToolCallParseResult;
using r4dx::server::ToolStreamGate;
using r4dx::server::ToolStreamRemainder;

// A verbatim-shaped call in this checkpoint's real surface syntax (tool_call_parser.h's file
// comment) -- the same shape tests/server/test_tool_call_parser.cpp's real captures use.
const char* const kCall =
    "<tool_call>\n<function=get_current_weather>\n<parameter=location>\nBoston, MA\n</parameter>\n"
    "</function>\n</tool_call>";

// Runs `pieces` through one gate, returning everything it streamed (Push events then Finish).
// `streamed_pieces`, when given, collects only the PUSH-time events: Finish()'s flush is by
// definition the "generation ended inside a would-be opener" case, where the dangling bytes are
// literal content (no call can exist there at all) and are supposed to go out verbatim -- the
// "never leak an opener prefix" property below is about what goes out WHILE generation is still
// running, which is the only point at which a leak could turn out to be premature.
std::string RunGate(const std::vector<std::string>& pieces, ToolStreamGate& gate,
                    std::vector<std::string>* streamed_pieces = nullptr) {
  std::string out;
  for (const auto& p : pieces) {
    const std::string s = gate.Push(p);
    out += s;
    if (streamed_pieces && !s.empty()) streamed_pieces->push_back(s);
  }
  out += gate.Finish();
  return out;
}

// ---- basic behavior ---------------------------------------------------------------------------

void TestPureProseStreamsWholeAndNeverCloses() {
  ToolStreamGate gate;
  std::vector<std::string> pieces = {"The sea is ", "vast", " and blue."};
  CHECK(RunGate(pieces, gate) == "The sea is vast and blue.");
  CHECK(!gate.closed());
  CHECK(gate.streamed_bytes() == std::string("The sea is vast and blue.").size());
}

void TestClosesAtOpenerAndStreamsOnlyTheProseBefore() {
  ToolStreamGate gate;
  const std::string whole = std::string("Sure, let me check.\n\n") + kCall;
  CHECK(RunGate({whole}, gate) == "Sure, let me check.\n\n");
  CHECK(gate.closed());
}

void TestCallWithNoLeadingProseStreamsNothing() {
  ToolStreamGate gate;
  CHECK(RunGate({kCall}, gate).empty());
  CHECK(gate.closed());
  CHECK(gate.streamed_bytes() == 0);
}

void TestStaysClosedAcrossLaterPieces() {
  // Everything after the first opener belongs to the end-of-generation parse, including prose
  // BETWEEN two calls -- the gate must never re-open and stream it live.
  ToolStreamGate gate;
  RunGate({std::string("prose ") + kCall, "\nand now more prose\n", kCall}, gate);
  CHECK(gate.closed());
  CHECK(gate.streamed_bytes() == std::string("prose ").size());
}

void TestWhitespaceBeforeOpenerIsStreamed() {
  // ParseToolCalls keeps the blank lines before a span in `content` verbatim, so the gate must too
  // -- holding them back would make the streamed total SHORTER than message.content
  // (tool_stream_gate.h's "NOT trimmed here" note).
  ToolStreamGate gate;
  const std::string streamed = RunGate({std::string("Checking.\n\n  \t\n") + kCall}, gate);
  CHECK(streamed == "Checking.\n\n  \t\n");
  CHECK(streamed == ParseToolCalls(std::string("Checking.\n\n  \t\n") + kCall).content);
}

// ---- holdback: lookalikes and partial openers ---------------------------------------------------

void TestLookalikeTagsAreEventuallyStreamed() {
  for (const std::string& text : {std::string("a < b and c < d"), std::string("see <tool_box> here"),
                                   std::string("<tool_calls> is not the opener"),
                                   std::string("<tool_cal")}) {
    ToolStreamGate gate;
    CHECK(RunGate({text}, gate) == text);
    CHECK(!gate.closed());
  }
}

void TestPartialOpenerHeldBackThenReleasedWhenDisambiguated() {
  ToolStreamGate gate;
  CHECK(gate.Push("done <tool_") == "done ");  // "<tool_" could still become the opener
  CHECK(gate.Push("box>") == "<tool_box>");    // ... it could not, so it is released whole
  CHECK(!gate.closed());
  CHECK(gate.Finish().empty());
}

void TestUnterminatedPartialOpenerFlushedOnFinish() {
  // Generation simply stopped inside a would-be opener: that span can never be a well-formed call,
  // so ParseToolCalls keeps the bytes as literal content and Finish() must release them.
  ToolStreamGate gate;
  CHECK(gate.Push("the answer is <tool_ca") == "the answer is ");
  CHECK(gate.Finish() == "<tool_ca");
  CHECK(!gate.closed());
  CHECK(gate.streamed_bytes() == std::string("the answer is <tool_ca").size());
}

void TestOpenerSplitAtEveryByteBoundaryAcrossTwoPieces() {
  const std::string opener = "<tool_call>";
  for (size_t split = 0; split <= opener.size(); ++split) {
    ToolStreamGate gate;
    std::vector<std::string> streamed_pieces;
    const std::string streamed =
        RunGate({"prose " + opener.substr(0, split), opener.substr(split) + "\n<function=f>\n"},
                gate, &streamed_pieces);
    CHECK(streamed == "prose ");
    CHECK(gate.closed());
    for (const auto& p : streamed_pieces) CHECK(p.find('<') == std::string::npos);
  }
}

// ---- ToolStreamRemainder -------------------------------------------------------------------------

void TestRemainderIsTheUnstreamedTail() {
  CHECK(ToolStreamRemainder("abc", "abcdef") == "def");
  CHECK(ToolStreamRemainder("abcdef", "abcdef").empty());
  CHECK(ToolStreamRemainder("", "abc") == "abc");
  CHECK(ToolStreamRemainder("", "").empty());
}

void TestRemainderFallsBackToTheWholeContentOnAMismatch() {
  // Defensive path (see ToolStreamRemainder's doc comment): duplicate rather than lose.
  CHECK(ToolStreamRemainder("xyz", "abcdef") == "abcdef");
  CHECK(ToolStreamRemainder("abcdef", "abc") == "abc");
}

// ---- the property test ---------------------------------------------------------------------------

// One generation's worth of ANSWER text (the gate never sees a thinking span -- Engine::RunRequest
// strips that first), run through the gate as `pieces` and checked against what ParseToolCalls
// makes of the whole thing.
void CheckOneChunking(const std::string& whole, const std::vector<std::string>& pieces,
                      const std::string& label) {
  const ToolCallParseResult parsed = ParseToolCalls(whole);

  ToolStreamGate gate;
  std::vector<std::string> streamed_pieces;
  const std::string streamed = RunGate(pieces, gate, &streamed_pieces);

  // 1. streamed + remainder == the non-streaming content, byte for byte.
  const std::string remainder = ToolStreamRemainder(streamed, parsed.content);
  if (streamed + remainder != parsed.content) {
    std::fprintf(stderr, "[%s] content mismatch: streamed='%s' remainder='%s' parsed='%s'\n",
                 label.c_str(), streamed.c_str(), remainder.c_str(), parsed.content.c_str());
    ++g_failures;
  }

  // 2. The gate closes if and only if an opener was really there -- so a generation that never
  // called anything streams in full (the whole point of this class), and one that did stops.
  const bool has_opener = whole.find("<tool_call>") != std::string::npos;
  if (gate.closed() != has_opener) {
    std::fprintf(stderr, "[%s] closed()=%d but has_opener=%d\n", label.c_str(),
                 static_cast<int>(gate.closed()), static_cast<int>(has_opener));
    ++g_failures;
  }

  // 3. No streamed piece may contain any part of a WELL-FORMED call's opener. When the parse found
  // no malformed span, every "<tool_call>" in `whole` was well-formed, so no streamed piece may
  // carry even a proper prefix of one (checked by re-running the gate's own rule: a piece that is
  // safe to stream can neither contain the opener nor end in a prefix of it).
  if (!parsed.had_malformed_call) {
    for (const auto& p : streamed_pieces) {
      if (p.find("<tool_call>") != std::string::npos) {
        std::fprintf(stderr, "[%s] streamed piece contains a whole opener: '%s'\n", label.c_str(),
                     p.c_str());
        ++g_failures;
      }
      for (size_t k = 1; k < 11; ++k) {  // every proper prefix of "<tool_call>"
        const std::string prefix = std::string("<tool_call>").substr(0, k);
        if (p.size() >= k && p.compare(p.size() - k, k, prefix) == 0) {
          std::fprintf(stderr, "[%s] streamed piece ends in opener prefix '%s': '%s'\n",
                       label.c_str(), prefix.c_str(), p.c_str());
          ++g_failures;
        }
      }
    }
  }
}

void TestPropertyEveryChunkingOfEveryGeneration() {
  const std::string call2 =
      "<tool_call>\n<function=get_current_weather>\n<parameter=location>\nNew York, NY\n"
      "</parameter>\n</function>\n</tool_call>";
  const std::vector<std::string> generations = {
      // pure prose (the case this whole change exists for)
      "The sea is vast and blue. It covers most of the planet.",
      // prose + one call
      std::string("Let me look that up.\n\n") + kCall,
      // two calls back to back, nothing else
      std::string(kCall) + "\n" + call2,
      // a call with no leading prose at all
      std::string(kCall),
      // prose AFTER a call, which ParseToolCalls keeps in content
      std::string(kCall) + "\nThat should do it.",
      // malformed: unclosed <tool_call> -- degrades to literal content, must still reach the client
      "Here goes: <tool_call>\n<function=broken>\n",
      // malformed: a closed span whose body is not this checkpoint's syntax
      std::string("before <tool_call>just prose, no function tag</tool_call> after"),
      // "<" and "<tool" lookalikes
      "if a < b and <tool_box> is not <tool_calls> then fine",
      // trailing whitespace right before the opener
      std::string("Checking now.\n\n   ") + kCall,
      // unicode on both sides of the boundary (2-, 3- and 4-byte characters)
      std::string("caf\xc3\xa9 \xe4\xb8\xad \xf0\x9f\x98\x80\n\n") + kCall + "\nfertig \xc3\xa9",
      // an opener that never completes
      "almost there <tool_cal",
      // empty generation
      "",
  };

  for (size_t g = 0; g < generations.size(); ++g) {
    const std::string& whole = generations[g];
    const std::string tag = "gen" + std::to_string(g);

    // (a) one whole-buffer Push -- how a non-streaming caller would use it.
    CheckOneChunking(whole, {whole}, tag + "/whole");

    // (b) one byte per Push -- the most extreme fragmentation a StreamDecoder could produce.
    std::vector<std::string> bytes;
    bytes.reserve(whole.size());
    for (char ch : whole) bytes.push_back(std::string(1, ch));
    CheckOneChunking(whole, bytes, tag + "/bytes");

    // (c) every 2-piece split point, including the degenerate empty-first/empty-last ones.
    for (size_t split = 0; split <= whole.size(); ++split) {
      CheckOneChunking(whole, {whole.substr(0, split), whole.substr(split)},
                       tag + "/split" + std::to_string(split));
    }
  }
}

}  // namespace

int main() {
  TestPureProseStreamsWholeAndNeverCloses();
  TestClosesAtOpenerAndStreamsOnlyTheProseBefore();
  TestCallWithNoLeadingProseStreamsNothing();
  TestStaysClosedAcrossLaterPieces();
  TestWhitespaceBeforeOpenerIsStreamed();
  TestLookalikeTagsAreEventuallyStreamed();
  TestPartialOpenerHeldBackThenReleasedWhenDisambiguated();
  TestUnterminatedPartialOpenerFlushedOnFinish();
  TestOpenerSplitAtEveryByteBoundaryAcrossTwoPieces();
  TestRemainderIsTheUnstreamedTail();
  TestRemainderFallsBackToTheWholeContentOnAMismatch();
  TestPropertyEveryChunkingOfEveryGeneration();

  if (g_failures > 0) {
    std::fprintf(stderr, "%d check(s) failed\n", g_failures);
    return 1;
  }
  std::fprintf(stderr, "all tool-stream-gate checks passed\n");
  return 0;
}
