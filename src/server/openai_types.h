// r4dx::server OpenAI-compatible request/response JSON (src/server/openai_types.cpp). Pure
// nlohmann::json in and out -- no HIP, no r4dx::model dependency -- so this half of the server is
// unit-testable on CPU alone (tests/server/test_openai_types.cpp) exactly like src/cli/cli_args.h
// is for the CLI's argument parsing.
//
// Deferred (documented in docs/server.md, not implemented here): tool-call parsing/emission
// (request "tools"/"tool_choice" are accepted and passed through to the chat template so the
// template can still render them into the prompt text, but a model-emitted tool call in the
// generated text is not parsed back into a structured `tool_calls` response field -- it comes
// back as plain `content` text); image message-content parts (rejected with a 400, see
// ParseChatCompletionRequest below) until the vision tower milestone lands.
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"

namespace r4dx::server {

struct SamplingParams {
  float temperature = 1.0f;
  float top_p = 1.0f;
  int top_k = 0;
  float min_p = 0.0f;
  bool has_seed = false;
  uint64_t seed = 0;
};

// Thrown by every Parse* function below on any invalid input; the HTTP layer (http_server.cpp)
// catches this, maps http_status to the response code, and serializes ErrorBody(*this) as the
// body -- mirroring OpenAI's `{"error": {"message", "type", ...}}` shape.
struct ApiError {
  int http_status = 400;
  std::string type;  // e.g. "invalid_request_error"
  std::string message;
};

nlohmann::json ErrorBody(const ApiError& err);

struct ChatMessage {
  std::string role;     // "system" | "user" | "assistant" (see ParseChatCompletionRequest)
  std::string content;
};

struct ChatCompletionRequest {
  std::string model;
  std::vector<ChatMessage> messages;
  SamplingParams sampling;
  std::optional<int64_t> max_tokens;  // unset -> caller applies the server's --max-tokens-default
  std::vector<std::string> stop;
  bool stream = false;
  // Passed straight through to r4dx::ChatTemplate::render()'s extra_context / tools arguments --
  // chat_template.h documents enable_thinking/reasoning_effort/preserve_thinking/add_vision_id as
  // the fields Qwen3.8-27B's own template understands.
  nlohmann::json chat_template_kwargs = nlohmann::json::object();
  nlohmann::json tools = nlohmann::json::array();
};

struct CompletionRequest {
  std::string model;
  std::string prompt;
  SamplingParams sampling;
  std::optional<int64_t> max_tokens;
  std::vector<std::string> stop;
  bool stream = false;
};

// Throws ApiError on any structurally or semantically invalid request body (missing/mistyped
// field, an unsupported role, an image content part, an out-of-range sampling value, ...).
//
// `sampling_defaults` seeds every field this request's body does not itself set (the server's
// --default-temperature/--default-top-p/--default-top-k/--default-min-p flags) -- defaulted to
// the library's own {1, 1, 0, 0} for callers (tests) that don't care.
ChatCompletionRequest ParseChatCompletionRequest(const nlohmann::json& body,
                                                  const SamplingParams& sampling_defaults = {});
CompletionRequest ParseCompletionRequest(const nlohmann::json& body,
                                          const SamplingParams& sampling_defaults = {});

// A short random "chatcmpl-"/"cmpl-"-prefixed id, used both as the OpenAI response `id` field and
// as this request's correlation id in the one-line-per-request log (engine.cpp).
std::string GenerateRequestId(const char* prefix);

nlohmann::json BuildModelsResponse(const std::string& model_id, int64_t created_unix);

struct UsageStats {
  int64_t prompt_tokens = 0;
  int64_t completion_tokens = 0;
  int64_t TotalTokens() const { return prompt_tokens + completion_tokens; }
};

nlohmann::json BuildChatCompletionResponse(const std::string& id, const std::string& model_id,
                                            int64_t created_unix, const std::string& content,
                                            const std::string& finish_reason,
                                            const UsageStats& usage);

// `delta` is the caller-built `{"role": "assistant"}` / `{"content": "..."}` / `{}` object for
// this chunk (response_sink.cpp assembles these); `finish_reason` is unset for every chunk but
// the last.
nlohmann::json BuildChatCompletionChunk(const std::string& id, const std::string& model_id,
                                         int64_t created_unix, const nlohmann::json& delta,
                                         const std::optional<std::string>& finish_reason);

nlohmann::json BuildCompletionResponse(const std::string& id, const std::string& model_id,
                                        int64_t created_unix, const std::string& text,
                                        const std::string& finish_reason, const UsageStats& usage);

nlohmann::json BuildCompletionChunk(const std::string& id, const std::string& model_id,
                                     int64_t created_unix, const std::string& text_delta,
                                     const std::optional<std::string>& finish_reason);

}  // namespace r4dx::server
