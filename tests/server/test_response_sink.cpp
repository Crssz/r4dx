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

  if (g_failures > 0) {
    std::fprintf(stderr, "%d check(s) failed\n", g_failures);
    return 1;
  }
  std::fprintf(stderr, "all response-sink checks passed\n");
  return 0;
}
