// tests/server/test_response_sink.cpp -- pure CPU unit test for src/server/response_sink.h's
// BufferingSink and StreamingSink.
#include <chrono>
#include <cstdio>
#include <string>
#include <thread>

#include "nlohmann/json.hpp"
#include "response_sink.h"

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

using r4dx::server::BufferingSink;
using r4dx::server::StreamingSink;
using r4dx::server::TimingStats;

void TestBufferingSinkHappyPath() {
  BufferingSink sink;
  sink.OnStart(10);
  sink.OnToken("hello ");
  sink.OnToken("world");
  sink.OnDone("stop", 2);
  sink.Wait();  // already done -- must not block
  CHECK(sink.text == "hello world");
  CHECK(sink.finish_reason == "stop");
  CHECK(sink.prompt_tokens == 10);
  CHECK(sink.completion_tokens == 2);
  CHECK(!sink.errored);
}

void TestBufferingSinkError() {
  BufferingSink sink;
  sink.OnStart(5);
  sink.OnError(400, "bad request");
  sink.Wait();
  CHECK(sink.errored);
  CHECK(sink.error_status == 400);
  CHECK(sink.error_message == "bad request");
}

void TestBufferingSinkWaitBlocksUntilDone() {
  BufferingSink sink;
  std::thread producer([&sink] {
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    sink.OnDone("length", 1);
  });
  sink.Wait();  // must not return before OnDone runs
  CHECK(sink.finish_reason == "length");
  producer.join();
}

void TestStreamingSinkChatShapes() {
  StreamingSink sink(StreamingSink::Kind::kChat, "chatcmpl-x", "test-model", 1234);
  sink.OnStart(3);
  sink.OnToken("hi");
  sink.OnDone("stop", 1);

  std::string event;
  CHECK(sink.Next(event));  // role preamble chunk
  CHECK(event.find("\"role\":\"assistant\"") != std::string::npos);
  CHECK(event.find("\"object\":\"chat.completion.chunk\"") != std::string::npos);

  CHECK(sink.Next(event));  // content delta
  CHECK(event.find("\"content\":\"hi\"") != std::string::npos);

  CHECK(sink.Next(event));  // final chunk with finish_reason
  CHECK(event.find("\"finish_reason\":\"stop\"") != std::string::npos);

  CHECK(sink.Next(event));  // [DONE] sentinel
  CHECK(event == "data: [DONE]\n\n");

  CHECK(!sink.Next(event));  // stream closed
}

void TestStreamingSinkCompletionShapesSkipRolePreamble() {
  StreamingSink sink(StreamingSink::Kind::kCompletion, "cmpl-x", "test-model", 1234);
  sink.OnStart(3);  // no-op for kCompletion -- no role preamble chunk
  sink.OnToken("hi");
  sink.OnDone("length", 1);

  std::string event;
  CHECK(sink.Next(event));  // text delta directly, no role chunk first
  CHECK(event.find("\"text\":\"hi\"") != std::string::npos);
  CHECK(event.find("\"object\":\"text_completion\"") != std::string::npos);

  CHECK(sink.Next(event));  // final chunk
  CHECK(event.find("\"finish_reason\":\"length\"") != std::string::npos);

  CHECK(sink.Next(event));  // [DONE]
  CHECK(event == "data: [DONE]\n\n");
  CHECK(!sink.Next(event));
}

void TestStreamingSinkEmptyTokenIsNoop() {
  StreamingSink sink(StreamingSink::Kind::kCompletion, "cmpl-y", "m", 1);
  sink.OnToken("");  // must not enqueue an event
  sink.OnDone("stop", 0);
  std::string event;
  CHECK(sink.Next(event));  // straight to the finish_reason chunk
  CHECK(event.find("\"finish_reason\":\"stop\"") != std::string::npos);
}

void TestStreamingSinkCancelStopsIteration() {
  StreamingSink sink(StreamingSink::Kind::kChat, "chatcmpl-z", "m", 1);
  CHECK(!sink.IsCancelled());
  sink.Cancel();
  CHECK(sink.IsCancelled());
  std::string event;
  CHECK(!sink.Next(event));  // queue closed, nothing was ever pushed
}

void TestBufferingSinkStoresToolCalls() {
  BufferingSink sink;
  sink.OnStart(5);
  sink.OnToken("Let me check.");
  sink.OnToolCalls({{"call_1", "get_current_weather", "{\"location\":\"Boston, MA\"}"}});
  sink.OnDone("tool_calls", 10);
  CHECK(sink.tool_calls.size() == 1);
  CHECK(sink.tool_calls[0].id == "call_1");
  CHECK(sink.finish_reason == "tool_calls");
}

void TestStreamingSinkToolCallsEmitsOneCompleteDelta() {
  StreamingSink sink(StreamingSink::Kind::kChat, "chatcmpl-t", "m", 1);
  sink.OnStart(3);
  sink.OnToolCalls({{"call_1", "f", "{}"}});
  sink.OnDone("tool_calls", 1);

  std::string event;
  CHECK(sink.Next(event));  // role preamble
  CHECK(sink.Next(event));  // the whole tool_calls batch, one chunk
  CHECK(event.find("\"tool_calls\"") != std::string::npos);
  CHECK(event.find("\"name\":\"f\"") != std::string::npos);
  CHECK(event.find("\"index\":0") != std::string::npos);

  CHECK(sink.Next(event));  // finish_reason chunk
  CHECK(event.find("\"finish_reason\":\"tool_calls\"") != std::string::npos);
  CHECK(sink.Next(event));  // [DONE]
  CHECK(!sink.Next(event));
}

void TestStreamingSinkOnToolCallsNoopWhenEmpty() {
  StreamingSink sink(StreamingSink::Kind::kChat, "chatcmpl-u", "m", 1);
  sink.OnStart(3);
  sink.OnToolCalls({});  // no calls parsed -- must not enqueue an event
  sink.OnDone("stop", 0);
  std::string event;
  CHECK(sink.Next(event));  // role preamble
  CHECK(sink.Next(event));  // straight to finish_reason, no tool_calls chunk in between
  CHECK(event.find("\"finish_reason\":\"stop\"") != std::string::npos);
}

void TestStreamingSinkOnToolCallsNoopForCompletionKind() {
  StreamingSink sink(StreamingSink::Kind::kCompletion, "cmpl-t", "m", 1);
  sink.OnToolCalls({{"call_1", "f", "{}"}});  // /v1/completions has no tool_calls concept
  sink.OnDone("stop", 0);
  std::string event;
  CHECK(sink.Next(event));  // straight to finish_reason
  CHECK(event.find("\"finish_reason\":\"stop\"") != std::string::npos);
}

// ---- timings (task point 1) --------------------------------------------------------------------

void TestBufferingSinkStoresTimings() {
  BufferingSink sink;
  sink.OnStart(10);
  sink.OnToken("hi");
  TimingStats timings;
  timings.prompt_n = 10;
  timings.prompt_ms = 5.0;
  timings.predicted_n = 1;
  timings.predicted_ms = 2.0;
  sink.OnDone("stop", 1, timings);
  CHECK(sink.timings.prompt_n == 10);
  CHECK(sink.timings.predicted_n == 1);
}

void TestBufferingSinkOnDoneDefaultTimingsIsZero() {
  // Regression guard: OnDone's `timings` parameter defaults to {} so existing call sites that
  // don't care (like every other test above) keep compiling unchanged.
  BufferingSink sink;
  sink.OnStart(1);
  sink.OnDone("stop", 0);
  CHECK(sink.timings.prompt_n == 0);
  CHECK(sink.timings.predicted_n == 0);
}

void TestStreamingSinkFinishChunkCarriesTimings() {
  StreamingSink sink(StreamingSink::Kind::kChat, "chatcmpl-x", "m", 1);
  sink.OnStart(5);
  sink.OnToken("hi");
  TimingStats timings;
  timings.prompt_n = 5;
  timings.prompt_ms = 4.0;
  timings.predicted_n = 1;
  timings.predicted_ms = 2.0;
  sink.OnDone("stop", 1, timings);

  std::string event;
  CHECK(sink.Next(event));  // role preamble
  CHECK(sink.Next(event));  // content delta
  CHECK(sink.Next(event));  // finish_reason chunk -- must carry "timings"
  CHECK(event.find("\"finish_reason\":\"stop\"") != std::string::npos);
  CHECK(event.find("\"timings\"") != std::string::npos);
  CHECK(event.find("\"prompt_n\":5") != std::string::npos);
  CHECK(sink.Next(event));  // [DONE]
  CHECK(event == "data: [DONE]\n\n");
  CHECK(!sink.Next(event));
}

// ---- stream_options.include_usage (task point 2) ------------------------------------------------

void TestStreamingSinkIncludeUsageAddsNullUsageToNormalChunksAndFinalUsageChunk() {
  StreamingSink sink(StreamingSink::Kind::kChat, "chatcmpl-y", "m", 1, /*include_usage=*/true);
  sink.OnStart(7);
  sink.OnToken("hi");
  TimingStats timings;
  timings.prompt_n = 7;
  sink.OnDone("stop", 3, timings);

  std::string event;
  CHECK(sink.Next(event));  // role preamble -- must carry "usage":null
  CHECK(event.find("\"usage\":null") != std::string::npos);

  CHECK(sink.Next(event));  // content delta -- must also carry "usage":null
  CHECK(event.find("\"usage\":null") != std::string::npos);

  CHECK(sink.Next(event));  // finish_reason chunk -- "usage":null AND "timings"
  CHECK(event.find("\"finish_reason\":\"stop\"") != std::string::npos);
  CHECK(event.find("\"usage\":null") != std::string::npos);
  CHECK(event.find("\"timings\"") != std::string::npos);

  CHECK(sink.Next(event));  // the dedicated usage chunk: empty choices, real usage, timings
  CHECK(event.find("\"choices\":[]") != std::string::npos);
  CHECK(event.find("\"completion_tokens\":3") != std::string::npos);
  CHECK(event.find("\"prompt_tokens\":7") != std::string::npos);
  CHECK(event.find("\"timings\"") != std::string::npos);

  CHECK(sink.Next(event));  // [DONE] -- still exactly one, after the usage chunk
  CHECK(event == "data: [DONE]\n\n");
  CHECK(!sink.Next(event));
}

void TestStreamingSinkWithoutIncludeUsageHasNoUsageKeyAnywhere() {
  // Regression guard for the "byte-for-byte unchanged except the finish_reason chunk's `timings`"
  // requirement -- no chunk should ever gain a "usage" key when include_usage was never requested
  // (the default -- every StreamingSink constructed without the 5th argument, and every existing
  // test above, exercises exactly this path).
  StreamingSink sink(StreamingSink::Kind::kChat, "chatcmpl-z2", "m", 1);
  sink.OnStart(4);
  sink.OnToken("hi");
  sink.OnDone("stop", 1);

  std::string event;
  while (sink.Next(event)) {
    CHECK(event.find("\"usage\"") == std::string::npos);
  }
}

void TestStreamingSinkCompletionIncludeUsageUsageChunkShape() {
  StreamingSink sink(StreamingSink::Kind::kCompletion, "cmpl-y", "m", 1, /*include_usage=*/true);
  sink.OnStart(2);
  sink.OnToken("hi");
  sink.OnDone("length", 1);

  std::string event;
  CHECK(sink.Next(event));  // text delta -- "usage":null
  CHECK(event.find("\"usage\":null") != std::string::npos);
  CHECK(sink.Next(event));  // finish_reason chunk -- "usage":null + timings
  CHECK(event.find("\"usage\":null") != std::string::npos);
  CHECK(event.find("\"timings\"") != std::string::npos);
  CHECK(sink.Next(event));  // dedicated usage chunk
  CHECK(event.find("\"object\":\"text_completion\"") != std::string::npos);
  CHECK(event.find("\"choices\":[]") != std::string::npos);
  CHECK(event.find("\"completion_tokens\":1") != std::string::npos);
  CHECK(sink.Next(event));  // [DONE]
  CHECK(event == "data: [DONE]\n\n");
  CHECK(!sink.Next(event));
}

// ---- reasoning_content (docs/server.md's "reasoning_content" section, task item 5) --------------

void TestBufferingSinkThinkingOffLiteralTagPassesThroughByteIdentical() {
  // Task item 5c: with thinking off (the default -- every existing BufferingSink test above
  // exercises exactly this path), a literal "</think>" in normal output must never be scanned for
  // or stripped.
  BufferingSink sink;
  sink.OnStart(5);
  sink.OnToken("normal answer with </think> literally in it");
  sink.OnDone("stop", 8);
  CHECK(sink.text == "normal answer with </think> literally in it");
  CHECK(sink.reasoning_text.empty());
}

void TestBufferingSinkSplitsReasoningAndContent() {
  BufferingSink sink(/*enable_thinking=*/true);
  sink.OnStart(5);
  sink.OnToken("some reasoning");
  sink.OnToken("</think>\n\nthe final answer");
  sink.OnDone("stop", 10, TimingStats{}, /*reasoning_tokens=*/6);
  CHECK(sink.reasoning_text == "some reasoning");
  CHECK(sink.text == "the final answer");
  CHECK(sink.reasoning_tokens == 6);
}

void TestBufferingSinkNeverClosedReasoningLeavesContentEmpty() {
  BufferingSink sink(/*enable_thinking=*/true);
  sink.OnStart(5);
  sink.OnToken("the model just keeps thinking and never emits a close tag");
  sink.OnDone("length", 12);
  CHECK(sink.reasoning_text == "the model just keeps thinking and never emits a close tag");
  CHECK(sink.text.empty());
}

void TestBufferingSinkReasoningTrimmedLeadingAndTrailingWhitespace() {
  BufferingSink sink(/*enable_thinking=*/true);
  sink.OnStart(5);
  sink.OnToken("  \n reasoning with padding \n  </think>\n\nanswer");
  sink.OnDone("stop", 10);
  CHECK(sink.reasoning_text == "reasoning with padding");
  CHECK(sink.text == "answer");
}

void TestBufferingSinkOnReasoningContentToolModeBypassesSplitter() {
  // Engine::RunRequest's tool_mode block delivers an ALREADY-TRIMMED reasoning span via
  // OnReasoningContent, then calls OnToken once more with `parsed.content` (tool-call tags already
  // stripped, no thinking tag left in it either) -- that OnToken call must be treated as plain
  // content, not re-run through the splitter (which would otherwise misclassify it, since the
  // splitter's own internal state never saw the tag).
  BufferingSink sink(/*enable_thinking=*/true);
  sink.OnStart(5);
  sink.OnReasoningContent("already trimmed reasoning");
  sink.OnToken("Let me check the weather.");
  sink.OnDone("tool_calls", 20);
  CHECK(sink.reasoning_text == "already trimmed reasoning");
  CHECK(sink.text == "Let me check the weather.");
}

// ---- emit_reasoning = false (OpenRouter's `reasoning.exclude` / `include_reasoning: false`) -----
//
// "Think, but do not return the thought": the split must still happen (so `content`/the content
// deltas are the ANSWER alone, with no `</think>` leaking through) while the reasoning text is
// withheld entirely.

void TestBufferingSinkExcludeReasoningKeepsAnswerAndDropsThought() {
  BufferingSink sink(/*enable_thinking=*/true, /*emit_reasoning=*/false);
  sink.OnStart(5);
  sink.OnToken("some reasoning");
  sink.OnToken("</think>\n\nthe final answer");
  sink.OnDone("stop", 10, TimingStats{}, /*reasoning_tokens=*/6);
  CHECK(sink.reasoning_text.empty());
  CHECK(sink.text == "the final answer");
  // The token COUNT is still reported: the tokens were really spent.
  CHECK(sink.reasoning_tokens == 6);
}

void TestBufferingSinkExcludeReasoningAlsoDropsOneShotToolModeSpan() {
  BufferingSink sink(/*enable_thinking=*/true, /*emit_reasoning=*/false);
  sink.OnStart(5);
  sink.OnReasoningContent("already trimmed reasoning");
  sink.OnToken("Let me check the weather.");
  sink.OnDone("tool_calls", 20);
  CHECK(sink.reasoning_text.empty());
  CHECK(sink.text == "Let me check the weather.");
}

void TestStreamingSinkExcludeReasoningDropsReasoningDeltasOnly() {
  StreamingSink sink(StreamingSink::Kind::kChat, "chatcmpl-x1", "m", 1, /*include_usage=*/false,
                     /*enable_thinking=*/true, /*emit_reasoning=*/false);
  sink.OnStart(3);
  sink.OnToken("some reasoning");
  sink.OnToken("</think>\n\nthe answer");
  sink.OnDone("stop", 6, TimingStats{}, /*reasoning_tokens=*/4);

  std::string event;
  CHECK(sink.Next(event));  // role preamble
  CHECK(event.find("\"role\":\"assistant\"") != std::string::npos);
  CHECK(sink.Next(event));  // straight to the content delta -- no reasoning delta at all
  CHECK(event.find("\"content\":\"the answer\"") != std::string::npos);
  CHECK(event.find("</think>") == std::string::npos);
  CHECK(sink.Next(event));  // finish_reason chunk
  CHECK(event.find("\"finish_reason\":\"stop\"") != std::string::npos);
  CHECK(sink.Next(event));  // [DONE]
  CHECK(!sink.Next(event));
}

void TestStreamingSinkExcludeReasoningNoReasoningKeyAnywhere() {
  for (bool one_shot : {false, true}) {
    StreamingSink sink(StreamingSink::Kind::kChat, "chatcmpl-x2", "m", 1, /*include_usage=*/true,
                       /*enable_thinking=*/true, /*emit_reasoning=*/false);
    sink.OnStart(3);
    if (one_shot) {
      sink.OnReasoningContent("already trimmed");
      sink.OnToken("visible");
    } else {
      // Never-closed span: OnDone's Finish() flush must be suppressed too.
      sink.OnToken("thinking forever, no close tag");
    }
    sink.OnDone("stop", 5, TimingStats{}, /*reasoning_tokens=*/2);
    std::string event;
    while (sink.Next(event)) {
      CHECK(event.find("reasoning_content") == std::string::npos);
    }
  }
}

void TestStreamingSinkThinkingOffNoReasoningContentKeyAnywhere() {
  StreamingSink sink(StreamingSink::Kind::kChat, "chatcmpl-r0", "m", 1);  // enable_thinking defaults false
  sink.OnStart(3);
  sink.OnToken("answer with </think> literally");
  sink.OnDone("stop", 5);
  std::string event;
  while (sink.Next(event)) {
    CHECK(event.find("reasoning_content") == std::string::npos);
  }
}

void TestStreamingSinkSplitsIntoReasoningThenContentDeltas() {
  StreamingSink sink(StreamingSink::Kind::kChat, "chatcmpl-r1", "m", 1, /*include_usage=*/false,
                     /*enable_thinking=*/true);
  sink.OnStart(3);
  sink.OnToken("some reasoning");
  sink.OnToken("</think>\n\nthe answer");
  sink.OnDone("stop", 6, TimingStats{}, /*reasoning_tokens=*/4);

  std::string event;
  CHECK(sink.Next(event));  // role preamble
  CHECK(event.find("\"role\":\"assistant\"") != std::string::npos);

  CHECK(sink.Next(event));  // reasoning_content delta
  CHECK(event.find("\"reasoning_content\":\"some reasoning\"") != std::string::npos);
  CHECK(event.find("\"content\"") == std::string::npos);  // never both keys in one delta

  CHECK(sink.Next(event));  // content delta
  CHECK(event.find("\"content\":\"the answer\"") != std::string::npos);
  CHECK(event.find("\"reasoning_content\"") == std::string::npos);
  CHECK(event.find("</think>") == std::string::npos);

  CHECK(sink.Next(event));  // finish_reason chunk
  CHECK(event.find("\"finish_reason\":\"stop\"") != std::string::npos);
  CHECK(sink.Next(event));  // [DONE]
  CHECK(!sink.Next(event));
}

void TestStreamingSinkNeverClosedFlushesReasoningOnDone() {
  StreamingSink sink(StreamingSink::Kind::kChat, "chatcmpl-r2", "m", 1, /*include_usage=*/false,
                     /*enable_thinking=*/true);
  sink.OnStart(3);
  sink.OnToken("thinking forever, no close tag");
  sink.OnDone("length", 6);

  std::string event;
  CHECK(sink.Next(event));  // role preamble
  CHECK(sink.Next(event));  // the ORIGINAL piece's own reasoning delta (tag-holdback safe prefix)
  CHECK(event.find("\"reasoning_content\"") != std::string::npos);
  // Whatever OnToken already streamed plus whatever OnDone's Finish() flush adds must reconstruct
  // to the whole never-closed text with nothing lost or duplicated; check via the finish chunk's
  // absence of "content" (never-closed means content stays empty, task item 5a).
  bool sawContentKey = false;
  std::string finishEvent;
  while (sink.Next(finishEvent)) {
    if (finishEvent.find("\"finish_reason\":\"length\"") != std::string::npos) break;
    if (finishEvent.find("\"content\"") != std::string::npos) sawContentKey = true;
  }
  CHECK(!sawContentKey);
}

void TestStreamingSinkOnReasoningContentToolModeOneShot() {
  StreamingSink sink(StreamingSink::Kind::kChat, "chatcmpl-r3", "m", 1, /*include_usage=*/false,
                     /*enable_thinking=*/true);
  sink.OnStart(3);
  sink.OnReasoningContent("already trimmed");
  sink.OnToken("the visible answer");
  sink.OnToolCalls({{"call_1", "f", "{}"}});
  sink.OnDone("tool_calls", 10);

  std::string event;
  CHECK(sink.Next(event));  // role preamble
  CHECK(sink.Next(event));  // one-shot reasoning_content delta
  CHECK(event.find("\"reasoning_content\":\"already trimmed\"") != std::string::npos);
  CHECK(sink.Next(event));  // content delta -- plain, not re-split
  CHECK(event.find("\"content\":\"the visible answer\"") != std::string::npos);
  CHECK(sink.Next(event));  // tool_calls delta
  CHECK(event.find("\"tool_calls\"") != std::string::npos);
  CHECK(sink.Next(event));  // finish_reason chunk
  CHECK(sink.Next(event));  // [DONE]
  CHECK(!sink.Next(event));
}

void TestStreamingSinkIncludeUsageCarriesReasoningTokens() {
  StreamingSink sink(StreamingSink::Kind::kChat, "chatcmpl-r4", "m", 1, /*include_usage=*/true,
                     /*enable_thinking=*/true);
  sink.OnStart(3);
  sink.OnToken("reason</think>\n\nans");
  sink.OnDone("stop", 5, TimingStats{}, /*reasoning_tokens=*/2);

  std::string event, lastDataEvent;
  while (sink.Next(event)) {
    if (event != "data: [DONE]\n\n") lastDataEvent = event;
  }
  CHECK(lastDataEvent.find("\"reasoning_tokens\":2") != std::string::npos);
}

void TestStreamingSinkCompletionKindNeverSplitsEvenIfEnableThinkingPassed() {
  // Task item 5g: /v1/completions is never split -- defensive check that a (hypothetical)
  // true `enable_thinking` on a kCompletion sink still behaves exactly like today.
  StreamingSink sink(StreamingSink::Kind::kCompletion, "cmpl-r5", "m", 1, /*include_usage=*/false,
                     /*enable_thinking=*/true);
  sink.OnStart(3);
  sink.OnToken("text</think>more text");
  sink.OnDone("stop", 4);
  std::string event;
  CHECK(sink.Next(event));  // text delta, unsplit
  CHECK(event.find("\"text\":\"text</think>more text\"") != std::string::npos);
}

void TestStreamingSinkErrorEmitsAndCloses() {
  StreamingSink sink(StreamingSink::Kind::kChat, "chatcmpl-e", "m", 1);
  sink.OnError(500, "boom");
  std::string event;
  CHECK(sink.Next(event));
  CHECK(event.find("\"message\":\"boom\"") != std::string::npos);
  CHECK(sink.Next(event));
  CHECK(event == "data: [DONE]\n\n");
  CHECK(!sink.Next(event));
}

}  // namespace

int main() {
  TestBufferingSinkHappyPath();
  TestBufferingSinkError();
  TestBufferingSinkWaitBlocksUntilDone();
  TestBufferingSinkStoresToolCalls();
  TestStreamingSinkToolCallsEmitsOneCompleteDelta();
  TestStreamingSinkOnToolCallsNoopWhenEmpty();
  TestStreamingSinkOnToolCallsNoopForCompletionKind();
  TestStreamingSinkChatShapes();
  TestStreamingSinkCompletionShapesSkipRolePreamble();
  TestStreamingSinkEmptyTokenIsNoop();
  TestStreamingSinkCancelStopsIteration();
  TestStreamingSinkErrorEmitsAndCloses();
  TestBufferingSinkStoresTimings();
  TestBufferingSinkOnDoneDefaultTimingsIsZero();
  TestStreamingSinkFinishChunkCarriesTimings();
  TestStreamingSinkIncludeUsageAddsNullUsageToNormalChunksAndFinalUsageChunk();
  TestStreamingSinkWithoutIncludeUsageHasNoUsageKeyAnywhere();
  TestStreamingSinkCompletionIncludeUsageUsageChunkShape();
  TestBufferingSinkThinkingOffLiteralTagPassesThroughByteIdentical();
  TestBufferingSinkSplitsReasoningAndContent();
  TestBufferingSinkNeverClosedReasoningLeavesContentEmpty();
  TestBufferingSinkReasoningTrimmedLeadingAndTrailingWhitespace();
  TestBufferingSinkOnReasoningContentToolModeBypassesSplitter();
  TestBufferingSinkExcludeReasoningKeepsAnswerAndDropsThought();
  TestBufferingSinkExcludeReasoningAlsoDropsOneShotToolModeSpan();
  TestStreamingSinkExcludeReasoningDropsReasoningDeltasOnly();
  TestStreamingSinkExcludeReasoningNoReasoningKeyAnywhere();
  TestStreamingSinkThinkingOffNoReasoningContentKeyAnywhere();
  TestStreamingSinkSplitsIntoReasoningThenContentDeltas();
  TestStreamingSinkNeverClosedFlushesReasoningOnDone();
  TestStreamingSinkOnReasoningContentToolModeOneShot();
  TestStreamingSinkIncludeUsageCarriesReasoningTokens();
  TestStreamingSinkCompletionKindNeverSplitsEvenIfEnableThinkingPassed();

  if (g_failures > 0) {
    std::fprintf(stderr, "%d check(s) failed\n", g_failures);
    return 1;
  }
  std::fprintf(stderr, "all response-sink checks passed\n");
  return 0;
}
