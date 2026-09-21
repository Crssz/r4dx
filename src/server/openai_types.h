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

// Every thinking/reasoning control a chat request may carry, normalized into one struct by
// ParseThinkingControls below (docs/server.md's "Thinking controls" section has the full
// precedence table and the per-field wire shapes). Different OpenAI-compatible clients spell the
// same intent five different ways, so all five are accepted and reduced here rather than each
// consumer re-deriving its own answer.
struct ThinkingControls {
  // std::nullopt means the request named NO thinking control at all -- the caller then applies the
  // server's own `--think` default, exactly as before these fields existed.
  std::optional<bool> enabled;
  // The requested effort, ALREADY MAPPED onto the three levels this checkpoint's own
  // `chat_template.jinja` accepts ("low"/"medium"/"xhigh"; anything else makes the template's own
  // `raise_exception('Unexpected reasoning effort ...')` fire). Unset when the request named no
  // effort, named "none" (which means thinking off, not an effort), or named one that carries no
  // meaning here -- in which case the template applies its own default ("xhigh").
  std::optional<std::string> template_effort;
  // OpenRouter's `reasoning.exclude: true` / the legacy `include_reasoning: false`: think, but do
  // not return the reasoning TEXT to the client. The generation is still split (so `content` stays
  // the answer only) and `usage.completion_tokens_details.reasoning_tokens` is still reported --
  // only the `reasoning_content` field/deltas are suppressed.
  bool include_reasoning = true;
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
  // Multi-turn reasoning_content replay (docs/server.md's "reasoning_content" section, task item
  // 5e): a client resending an earlier assistant turn this server generated may include the
  // `reasoning_content` this server returned for it. Accepted on any role (validated as a string
  // when present -- a non-string is a 400, ParseChatCompletionRequest), attached verbatim to the
  // rendered message JSON (Engine::RunRequest) -- the chat template itself already reads
  // `message.reasoning_content` on assistant turns and decides whether to keep it based on
  // `chat_template_kwargs.preserve_thinking` (C:\AI\models\Qwen3.8-27B\chat_template.jinja), so no
  // extra server-side gating logic is needed here: this field is passed through and the template
  // decides.
  std::optional<std::string> reasoning_content;
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
  // Every thinking control this request carried, already reduced to one answer -- see
  // ThinkingControls and ParseThinkingControls. `chat_template_kwargs.enable_thinking` is still
  // carried verbatim in `chat_template_kwargs` above (the template reads it directly); this field
  // is what decides the RESOLVED behavior for every consumer.
  ThinkingControls thinking;
  // `stream_options: {"include_usage": bool}` (OpenAI's own streaming-usage opt-in). Parsed and
  // validated (must be an object; `include_usage` must be a boolean when present) regardless of
  // `stream` -- a non-streaming request carrying `stream_options` is accepted and this field is
  // simply ignored (OpenAI itself rejects that combination; being tolerant here is friendlier and
  // costs nothing, since the field only ever matters to the streaming sink). Only meaningful when
  // `stream` is also true: every SSE chunk then carries `"usage": null` except one extra
  // `"choices": []` chunk sent after the finish_reason chunk with the real `usage` (and
  // `timings`), right before `[DONE]` -- see response_sink.cpp's `StreamingSink`.
  bool stream_options_include_usage = false;
};

struct CompletionRequest {
  std::string model;
  std::string prompt;
  SamplingParams sampling;
  std::optional<int64_t> max_tokens;
  std::vector<std::string> stop;
  bool stream = false;
  // `stream_options: {"include_usage": bool}` -- see ChatCompletionRequest's own field for the
  // full contract (identical here; OpenAI defines it once and both endpoints honor it the same
  // way).
  bool stream_options_include_usage = false;
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

// Resolves the effective `enable_thinking` value for a chat request exactly the way
// Engine::RunRequest does (`chat_template_kwargs.enable_thinking` if present and a JSON boolean,
// else `default_thinking` -- the server's `--think` flag): a non-boolean value falls back to
// `default_thinking` rather than throwing, since chat_template_kwargs is otherwise an opaque
// passthrough to the Jinja template (ChatTemplate::render() itself tolerates any JSON type there
// via Jinja truthiness). Shared by http_server.cpp (which must decide a request's reasoning_content
// behavior -- ResponseSink construction -- before the request even reaches the worker thread) and
// engine.cpp (which decides the actual generation-time splitting), so the two independently-made
// calls can never drift apart.
bool ResolveEnableThinking(const nlohmann::json& chat_template_kwargs, bool default_thinking);

// Parses -- and validates -- every thinking control a chat request may carry, reducing them to one
// ThinkingControls (docs/server.md's "Thinking controls"). Accepted spellings, in the precedence
// order applied here (FIRST one present wins; an explicit request field always beats `--think`):
//
//   1. `chat_template_kwargs.enable_thinking` (bool)  -- the field that addresses THIS checkpoint's
//      own template directly, and the only one this server accepted before; kept highest so no
//      existing caller's behavior moves. A non-boolean here is tolerated, not rejected, exactly as
//      ResolveEnableThinking(json, bool) already tolerated it.
//   2. `reasoning: {...}` (OpenRouter) -- `enabled` (bool) wins over `effort` (string) inside it.
//   3. `thinking: {"type": "enabled"|"disabled"}` (Anthropic-style).
//   4. `enable_thinking` (bool, top level).
//   5. `reasoning_effort` (string, top level).
//   6. nothing -> `enabled` stays unset and the caller applies `--think`.
//
// An effort of "none" means thinking OFF; every other non-empty effort means ON. Independently of
// the on/off answer, `reasoning.exclude: true` or `include_reasoning: false` suppresses the
// reasoning TEXT in the response (either one asking for exclusion wins).
//
// Throws ApiError (400, the standard error body) on a malformed shape: a non-object `reasoning`/
// `thinking`, a non-boolean `enabled`/`exclude`/`enable_thinking`/`include_reasoning`, a
// non-string or empty `effort`/`reasoning_effort`, a `thinking.type` that is neither "enabled" nor
// "disabled", or a negative/non-integer `reasoning.max_tokens`/`thinking.budget_tokens`.
ThinkingControls ParseThinkingControls(const nlohmann::json& body);

// The resolved `enable_thinking` for a request: whatever ParseThinkingControls decided, else the
// server's `--think` default. Shared by http_server.cpp (which must pick the ResponseSink's
// splitting behavior before the request reaches the worker thread) and engine.cpp (which decides
// the generation-time splitting and what the chat template is rendered with), so the two
// independently-made calls can never drift apart.
bool ResolveEnableThinking(const ThinkingControls& thinking, bool default_thinking);

// `architecture.input_modalities` for `/v1/models` -- the ONE place that list is written. Today
// `["text"]`; the vision-tower milestone flips it to `["text","image"]` here and nowhere else
// (docs/server.md's "Deferred / known gaps").
nlohmann::json ModelInputModalities();

// The `reasoning.supported_efforts` list `/v1/models` advertises, and the exact set of effort
// strings ParseThinkingControls maps onto a template level. See ThinkingControls::template_effort
// for what each one actually does on this checkpoint.
nlohmann::json ModelSupportedReasoningEfforts();

// The model's own native context length (Qwen3.8-27B's config.json `max_position_embeddings`,
// server_args.h's `--max-ctx` default derivation) -- NOT the same as a server's own configured
// `--max-ctx`, which may be set lower. r4dx::model::ModelConfig does not carry this field (nothing
// in the layer graph needs it), so it is a named constant here rather than read from the loaded
// container, per BuildModelsResponse's `meta.n_ctx_train` (llama.cpp's own field name for exactly
// this "the checkpoint's trained/native limit" concept).
inline constexpr int64_t kModelNativeContextLength = 262144;

// One `/v1/models` list entry / the `/v1/models/{id}` single-object response body -- OpenAI's
// {id, object, created, owned_by} plus the r4dx extension fields BuildModelsResponse's own doc
// comment enumerates (docs/server.md's "Model metadata" section has the full field-by-field
// writeup). `max_ctx` is the server's own `--max-ctx` (Engine::MaxCtx()), NOT
// kModelNativeContextLength.
// `default_thinking` is the server's own `--think` flag, reported as `reasoning.default_enabled`
// so a client can render a thinking toggle pre-set the way this server will actually behave.
nlohmann::json BuildModelEntryJson(const std::string& model_id, int64_t created_unix,
                                    int64_t max_ctx, bool default_thinking = false);

// `{"object": "list", "data": [BuildModelEntryJson(...)]}` -- GET /v1/models's shape. This server
// ever loads exactly one model, so `data` always has exactly one entry (task design point 2).
nlohmann::json BuildModelsResponse(const std::string& model_id, int64_t created_unix,
                                    int64_t max_ctx, bool default_thinking = false);

struct UsageStats {
  int64_t prompt_tokens = 0;
  int64_t completion_tokens = 0;
  int64_t TotalTokens() const { return prompt_tokens + completion_tokens; }
  // `completion_tokens_details.reasoning_tokens` (DeepSeek/vLLM convention, docs/server.md's
  // "reasoning_content" section): the number of generated tokens up to and including the
  // "</think>" close tag. unset (not merely 0) whenever this request had thinking off -- every
  // usage-object builder below adds the `completion_tokens_details` object iff this is set, so a
  // thinking-off response's `usage` stays byte-identical to before this field existed.
  std::optional<int64_t> reasoning_tokens;
};

// `timings` -- an r4dx extension beyond the OpenAI spec, attached as a top-level sibling of
// `usage` (llama.cpp-compatible field names, so existing bench tooling written against llama.cpp's
// own `/completion` `timings` object keeps working unmodified against this server). Unlike
// `usage.prompt_tokens` (the full conversation-so-far token count), `prompt_n` is only the tokens
// actually fed to Model::Prefill THIS request -- Engine::RunRequest's `new_tokens_i32.size()` --
// which is smaller than `usage.prompt_tokens` whenever prefix reuse (`PrefixState::Extend`) skips
// re-prefilling a conversation's already-cached prefix. `draft_n`/`draft_n_accepted` are only set
// (and only serialized by BuildTimingsJson) when a speculative decode path -- MTP or DFlash2 --
// actually ran at least one round this request; a plain-decode or sampled (`temperature>0`)
// request leaves them unset.
struct TimingStats {
  int64_t prompt_n = 0;
  double prompt_ms = 0.0;
  int64_t predicted_n = 0;
  double predicted_ms = 0.0;
  std::optional<int64_t> draft_n;
  std::optional<int64_t> draft_n_accepted;
};

// {"prompt_n", "prompt_ms", "prompt_per_second", "predicted_n", "predicted_ms",
// "predicted_per_second"} plus "draft_n"/"draft_n_accepted" iff `timings.draft_n` is set. A
// `*_per_second` value is 0.0 (never NaN/inf) when the corresponding `*_ms` is 0.
nlohmann::json BuildTimingsJson(const TimingStats& timings);

// `reasoning_content`: unset (the default) omits the key entirely -- a thinking-off response is
// byte-identical to before this field existed (docs/server.md's "reasoning_content" section, task
// item 5c). When set, `message.reasoning_content` is added alongside `message.content` (which by
// this point is the ANSWER text only, already split out by the caller -- see response_sink.cpp's
// BufferingSink, which owns the actual splitting).
nlohmann::json BuildChatCompletionResponse(const std::string& id, const std::string& model_id,
                                            int64_t created_unix, const std::string& content,
                                            const std::string& finish_reason,
                                            const UsageStats& usage, const TimingStats& timings,
                                            const std::optional<std::string>& reasoning_content = std::nullopt);

// Tool-call-carrying overload: `content` is `nullopt` when the message was a pure tool call with
// no accompanying prose (OpenAI convention: `message.content` is JSON null in that case, not "");
// `tool_calls` is rendered as `message.tool_calls` when non-empty (omitted from the JSON object
// entirely when empty, matching a plain-text response's existing shape exactly). `reasoning_content`
// -- see the plain-text overload's own doc comment -- is the thinking span stripped out BEFORE
// tool-call parsing even ran (docs/server.md's "reasoning_content" section, task item 5d).
nlohmann::json BuildChatCompletionResponse(const std::string& id, const std::string& model_id,
                                            int64_t created_unix,
                                            const std::optional<std::string>& content,
                                            const std::vector<ToolCallOut>& tool_calls,
                                            const std::string& finish_reason,
                                            const UsageStats& usage, const TimingStats& timings,
                                            const std::optional<std::string>& reasoning_content = std::nullopt);

// `[{"index":N,"id":...,"type":"function","function":{"name":...,"arguments":...}}, ...]` --
// the `message.tool_calls` array shape (response) / the streaming `delta.tool_calls` array shape
// (each entry additionally requires an `index` when streamed, harmless when reused verbatim in
// the non-streaming response too, since OpenAI's own response `tool_calls` entries don't forbid
// an extra field client SDKs don't look at).
nlohmann::json BuildToolCallsJson(const std::vector<ToolCallOut>& tool_calls);

// `delta` is the caller-built `{"role": "assistant"}` / `{"content": "..."}` / `{}` object for
// this chunk (response_sink.cpp assembles these); `finish_reason` is unset for every chunk but
// the last. `include_usage_null` adds a top-level `"usage": null` (OpenAI's `stream_options.
// include_usage` contract: every chunk but the dedicated usage-carrying one gets a null usage
// field). `timings`, when set, adds a top-level `"timings"` object -- only ever passed for the
// final finish_reason chunk (StreamingSink::OnDone), never the earlier role-preamble/content/
// tool_calls chunks.
nlohmann::json BuildChatCompletionChunk(const std::string& id, const std::string& model_id,
                                         int64_t created_unix, const nlohmann::json& delta,
                                         const std::optional<std::string>& finish_reason,
                                         bool include_usage_null = false,
                                         const std::optional<TimingStats>& timings = std::nullopt);

// The dedicated `stream_options.include_usage` usage chunk: empty `choices`, the real `usage` and
// `timings` objects, sent after the finish_reason chunk and before `[DONE]`.
nlohmann::json BuildChatCompletionUsageChunk(const std::string& id, const std::string& model_id,
                                              int64_t created_unix, const UsageStats& usage,
                                              const TimingStats& timings);

nlohmann::json BuildCompletionResponse(const std::string& id, const std::string& model_id,
                                        int64_t created_unix, const std::string& text,
                                        const std::string& finish_reason, const UsageStats& usage,
                                        const TimingStats& timings);

nlohmann::json BuildCompletionChunk(const std::string& id, const std::string& model_id,
                                     int64_t created_unix, const std::string& text_delta,
                                     const std::optional<std::string>& finish_reason,
                                     bool include_usage_null = false,
                                     const std::optional<TimingStats>& timings = std::nullopt);

// See BuildChatCompletionUsageChunk's own comment -- identical role for `/v1/completions`.
nlohmann::json BuildCompletionUsageChunk(const std::string& id, const std::string& model_id,
                                          int64_t created_unix, const UsageStats& usage,
                                          const TimingStats& timings);

}  // namespace r4dx::server
