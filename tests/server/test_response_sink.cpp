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
  TestStreamingSinkChatShapes();
  TestStreamingSinkCompletionShapesSkipRolePreamble();
  TestStreamingSinkEmptyTokenIsNoop();
  TestStreamingSinkCancelStopsIteration();
  TestStreamingSinkErrorEmitsAndCloses();

  if (g_failures > 0) {
    std::fprintf(stderr, "%d check(s) failed\n", g_failures);
    return 1;
  }
  std::fprintf(stderr, "all response-sink checks passed\n");
  return 0;
}
