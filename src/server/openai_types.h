// r4dx::server OpenAI-compatible request/response JSON (src/server/openai_types.cpp). Pure
// nlohmann::json in and out -- no HIP, no r4dx::model dependency -- so this half of the server is
// unit-testable on CPU alone (tests/server/test_openai_types.cpp) exactly like src/cli/cli_args.h
// is for the CLI's argument parsing.
//
// Tool calls (docs/server.md's "Tool calls" section has the full design/surface-syntax writeup):
// `tools`/`tool_choice` are parsed here and rendered into the prompt by the chat template
// (src/server/engine.cpp); a model-emitted tool call is parsed back into a structured
// `message.tool_calls` field by src/server/tool_call_parser.h, not here (this header only carries
// the OpenAI wire shapes both directions -- request-side `ChatMessage::tool_calls`/`tool_call_id`
// for a multi-turn tool round trip, response-side `ToolCallOut` for what the server emits back).
//
// Still deferred: image message-content parts (rejected with a 400, see
// ParseChatCompletionRequest below) until the vision tower milestone lands.
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"

namespace r4dx::server {

// One tool call, request or response direction. `arguments_json` is always a JSON-ENCODED STRING
// (OpenAI's own `function.arguments` wire shape -- NOT a JSON object; this is the field the task
// itself flagged as "trips people up"). Request-side (an earlier assistant turn's tool_calls
// being replayed back in `messages`), `id` is whatever the caller supplied. Response-side (a call
// this server just parsed out of a real generation), `id` is server-generated
// (GenerateRequestId("call_")) since the model's own surface syntax carries no id at all.
struct ToolCallOut {
  std::string id;
  std::string name;
  std::string arguments_json;
};

// tool_choice, parsed once at request-parse time and already applied to `ChatCompletionRequest::
// tools` by the time a caller sees it (see ParseChatCompletionRequest): "none" clears `tools`
// entirely (the template then never mentions tools, so the model has no way to call one); a named
// choice filters `tools` down to just that one entry (the only enforceable lever this engine has
// -- there is no constrained decoding here); "auto"/"required" leave `tools` untouched.
// `kRequired` is accepted but NOT enforced at generation time (no constrained decoding) -- a
// `required` request whose generation happens not to contain a tool call is not itself an error,
// exactly documented in docs/server.md rather than silently pretended-supported.
enum class ToolChoiceKind { kAuto, kNone, kRequired, kNamed };
struct ToolChoice {
  ToolChoiceKind kind = ToolChoiceKind::kAuto;
  std::string name;  // only meaningful when kind == kNamed
};

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
  std::string role;  // "system" | "user" | "assistant" | "tool" | "function"
  // Absent (std::nullopt) only ever allowed for role=="assistant" with a non-empty `tool_calls`
  // (an assistant turn that was purely a tool call, no accompanying prose) -- every other role
  // requires a real (possibly empty-string) content, enforced by ParseChatCompletionRequest.
  std::optional<std::string> content;
  std::vector<ToolCallOut> tool_calls;      // role=="assistant" only; empty otherwise
  std::optional<std::string> tool_call_id;  // role=="tool" (required there); unused otherwise
  std::optional<std::string> name;          // role=="function" (legacy, required there): the
                                             // function name this message is a result of
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
  // Already filtered/cleared per `tool_choice` by the time ParseChatCompletionRequest returns --
  // see ToolChoice's own doc comment.
  nlohmann::json tools = nlohmann::json::array();
  ToolChoice tool_choice;
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

// Tool-call-carrying overload: `content` is `nullopt` when the message was a pure tool call with
// no accompanying prose (OpenAI convention: `message.content` is JSON null in that case, not "");
// `tool_calls` is rendered as `message.tool_calls` when non-empty (omitted from the JSON object
// entirely when empty, matching a plain-text response's existing shape exactly).
nlohmann::json BuildChatCompletionResponse(const std::string& id, const std::string& model_id,
                                            int64_t created_unix,
                                            const std::optional<std::string>& content,
                                            const std::vector<ToolCallOut>& tool_calls,
                                            const std::string& finish_reason,
                                            const UsageStats& usage);

// `[{"index":N,"id":...,"type":"function","function":{"name":...,"arguments":...}}, ...]` --
// the `message.tool_calls` array shape (response) / the streaming `delta.tool_calls` array shape
// (each entry additionally requires an `index` when streamed, harmless when reused verbatim in
// the non-streaming response too, since OpenAI's own response `tool_calls` entries don't forbid
// an extra field client SDKs don't look at).
nlohmann::json BuildToolCallsJson(const std::vector<ToolCallOut>& tool_calls);

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
