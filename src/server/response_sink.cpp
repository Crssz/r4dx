#include "response_sink.h"

#include "openai_types.h"
#include "sse.h"

namespace r4dx::server {

// ---- BufferingSink --------------------------------------------------------------------------

void BufferingSink::OnStart(int64_t prompt_tokens_in) {
  std::lock_guard<std::mutex> lock(mu_);
  prompt_tokens = prompt_tokens_in;
}

void BufferingSink::OnToken(const std::string& piece) {
  std::lock_guard<std::mutex> lock(mu_);
  text += piece;
}

void BufferingSink::OnDone(const std::string& finish_reason_in, int64_t completion_tokens_in,
                            const TimingStats& timings_in) {
  std::lock_guard<std::mutex> lock(mu_);
  finish_reason = finish_reason_in;
  completion_tokens = completion_tokens_in;
  timings = timings_in;
  done_ = true;
  cv_.notify_all();
}

void BufferingSink::OnError(int http_status, const std::string& message) {
  std::lock_guard<std::mutex> lock(mu_);
  errored = true;
  error_status = http_status;
  error_message = message;
  done_ = true;
  cv_.notify_all();
}

void BufferingSink::Wait() {
  std::unique_lock<std::mutex> lock(mu_);
  cv_.wait(lock, [&] { return done_; });
}

void BufferingSink::OnToolCalls(const std::vector<ToolCallOut>& calls) {
  std::lock_guard<std::mutex> lock(mu_);
  tool_calls = calls;
}

// ---- StreamingSink ---------------------------------------------------------------------------

StreamingSink::StreamingSink(Kind kind, std::string id, std::string model_id, int64_t created_unix,
                             bool include_usage)
    : kind_(kind), id_(std::move(id)), model_id_(std::move(model_id)), created_unix_(created_unix),
      include_usage_(include_usage), queue_(/*max_size=*/256) {}

void StreamingSink::OnStart(int64_t prompt_tokens) {
  prompt_tokens_ = prompt_tokens;
  if (kind_ != Kind::kChat) return;  // /v1/completions has no role-preamble chunk
  // OpenAI's first chat-completion chunk carries only {"role": "assistant"} with an empty delta
  // otherwise -- lets clients render the message bubble before any text arrives.
  nlohmann::json delta = {{"role", "assistant"}};
  queue_.Push(FormatSseEvent(
      BuildChatCompletionChunk(id_, model_id_, created_unix_, delta, std::nullopt, include_usage_)));
}

void StreamingSink::OnToken(const std::string& piece) {
  if (piece.empty()) return;
  if (kind_ == Kind::kChat) {
    nlohmann::json delta = {{"content", piece}};
    queue_.Push(FormatSseEvent(
        BuildChatCompletionChunk(id_, model_id_, created_unix_, delta, std::nullopt, include_usage_)));
  } else {
    queue_.Push(FormatSseEvent(
        BuildCompletionChunk(id_, model_id_, created_unix_, piece, std::nullopt, include_usage_)));
  }
}

void StreamingSink::OnToolCalls(const std::vector<ToolCallOut>& calls) {
  if (calls.empty() || kind_ != Kind::kChat) return;  // /v1/completions has no tool_calls concept
  // The single "emit tool calls whole" delta -- see ResponseSink::OnToolCalls's own doc comment
  // and docs/server.md's "Tool calls" streaming section for why this is one complete chunk
  // (index-tagged, so a client that DOES expect real per-delta streaming still assembles it
  // correctly, it just receives the whole thing in one delta instead of many) rather than
  // OpenAI's own byte-by-byte argument streaming.
  nlohmann::json delta = {{"tool_calls", BuildToolCallsJson(calls)}};
  queue_.Push(FormatSseEvent(
      BuildChatCompletionChunk(id_, model_id_, created_unix_, delta, std::nullopt, include_usage_)));
}

void StreamingSink::OnDone(const std::string& finish_reason, int64_t completion_tokens,
                           const TimingStats& timings_in) {
  if (kind_ == Kind::kChat) {
    queue_.Push(FormatSseEvent(BuildChatCompletionChunk(id_, model_id_, created_unix_,
                                                        nlohmann::json::object(), finish_reason,
                                                        include_usage_, timings_in)));
  } else {
    queue_.Push(FormatSseEvent(BuildCompletionChunk(id_, model_id_, created_unix_, "", finish_reason,
                                                    include_usage_, timings_in)));
  }
  // stream_options.include_usage (task point 2): one extra chunk after the finish_reason chunk,
  // before [DONE] -- empty `choices`, the real `usage` (prompt_tokens_ from OnStart +
  // completion_tokens handed to us here) and the same `timings` object.
  if (include_usage_) {
    const UsageStats usage{prompt_tokens_, completion_tokens};
    if (kind_ == Kind::kChat) {
      queue_.Push(FormatSseEvent(BuildChatCompletionUsageChunk(id_, model_id_, created_unix_, usage,
                                                                timings_in)));
    } else {
      queue_.Push(FormatSseEvent(BuildCompletionUsageChunk(id_, model_id_, created_unix_, usage,
                                                            timings_in)));
    }
  }
  queue_.Push(FormatSseDone());
  queue_.Close();
}

void StreamingSink::OnError(int /*http_status*/, const std::string& message) {
  // The HTTP status line/headers are already committed once a chunked stream starts, so a
  // mid-stream error can only be surfaced in-band -- emit it as an SSE "error" field on an
  // otherwise-normal chunk shape rather than silently dropping the connection.
  nlohmann::json payload = {{"error", {{"message", message}, {"type", "server_error"}}}};
  queue_.Push(FormatSseEvent(payload));
  queue_.Push(FormatSseDone());
  queue_.Close();
}

bool StreamingSink::Next(std::string& out) {
  std::optional<std::string> item = queue_.Pop();
  if (!item) return false;
  out = std::move(*item);
  return true;
}

void StreamingSink::Cancel() {
  cancelled_.store(true, std::memory_order_relaxed);
  queue_.Close();
}

}  // namespace r4dx::server
