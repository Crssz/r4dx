// r4dx::server::ResponseSink -- what the Engine's worker thread (engine.cpp, HIP-dependent, not
// built by this header) drives per request, decoupled from whether the HTTP layer is serving it
// as one-shot JSON or an SSE stream. Both concrete sinks below are pure CPU/string-formatting
// code (openai_types.h + sse.h + request_queue.h), so this header and its .cpp are unit-testable
// without a GPU (tests/server/test_response_sink.cpp).
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>

#include "nlohmann/json.hpp"
#include "request_queue.h"

namespace r4dx::server {

class ResponseSink {
 public:
  virtual ~ResponseSink() = default;

  // Called once, before the first generated token, with the full prompt token count (for the
  // response's `usage.prompt_tokens`).
  virtual void OnStart(int64_t prompt_tokens) = 0;

  // Called once per newly-decoded piece of text (Tokenizer::StreamDecoder::push's return value --
  // may span more or less than one token, see tokenizer.h).
  virtual void OnToken(const std::string& piece) = 0;

  // Called exactly once, after either the last OnToken or immediately (zero generated tokens).
  // `finish_reason` is one of "stop" (eos or a --stop string matched), "length" (max_tokens
  // reached), or "cancelled" (client disconnected mid-stream, see IsCancelled below).
  virtual void OnDone(const std::string& finish_reason, int64_t completion_tokens) = 0;

  // Called instead of OnDone if something failed before or during generation (a bad request that
  // slipped past validation, a KV-cache-capacity overrun, ...).
  virtual void OnError(int http_status, const std::string& message) = 0;

  // The worker polls this between decode steps; true means stop generating now. Only
  // StreamingSink ever returns true (set by Cancel() when the HTTP layer detects the client is
  // gone); BufferingSink's non-streaming request has no analogous "give up early" signal.
  virtual bool IsCancelled() const { return false; }
};

// Non-streaming (/v1/chat/completions and /v1/completions with "stream" false or absent): buffers
// the whole reply, then wakes the HTTP handler thread blocked in Wait() once OnDone/OnError runs.
class BufferingSink : public ResponseSink {
 public:
  void OnStart(int64_t prompt_tokens) override;
  void OnToken(const std::string& piece) override;
  void OnDone(const std::string& finish_reason, int64_t completion_tokens) override;
  void OnError(int http_status, const std::string& message) override;

  // Blocks the calling thread until OnDone or OnError has been called.
  void Wait();

  std::string text;
  std::string finish_reason;
  int64_t prompt_tokens = 0;
  int64_t completion_tokens = 0;
  bool errored = false;
  int error_status = 500;
  std::string error_message;

 private:
  std::mutex mu_;
  std::condition_variable cv_;
  bool done_ = false;
};

// Streaming: formats each event as an OpenAI SSE chunk (chat.completion.chunk or text_completion,
// per `kind`) and hands the formatted "data: ...\n\n" string to httplib's chunked content
// provider (http_server.cpp) via a small blocking queue.
class StreamingSink : public ResponseSink {
 public:
  enum class Kind { kChat, kCompletion };

  StreamingSink(Kind kind, std::string id, std::string model_id, int64_t created_unix);

  void OnStart(int64_t prompt_tokens) override;
  void OnToken(const std::string& piece) override;
  void OnDone(const std::string& finish_reason, int64_t completion_tokens) override;
  void OnError(int http_status, const std::string& message) override;
  bool IsCancelled() const override { return cancelled_.load(std::memory_order_relaxed); }

  // Called by the HTTP handler thread (httplib's chunked-content-provider callback) to pull the
  // next formatted SSE event; blocks until one is available. Returns false once the stream is
  // finished (queue closed and drained) -- the caller should stop calling and end the response.
  bool Next(std::string& out);

  // Called by the HTTP handler thread when the client disconnects (DataSink::write/is_writable
  // fails) -- makes IsCancelled() true so the worker's generation loop stops at its next check,
  // and unblocks any Next()/Push() waiter. Idempotent.
  void Cancel();

 private:
  Kind kind_;
  std::string id_, model_id_;
  int64_t created_unix_;
  std::atomic<bool> cancelled_{false};
  BoundedQueue<std::string> queue_;  // formatted SSE events; small bounded capacity gives natural
                                      // backpressure against a slow/stalled client.
};

}  // namespace r4dx::server
