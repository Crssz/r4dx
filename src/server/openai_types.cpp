#include "openai_types.h"

#include <cstdio>
#include <mutex>
#include <random>

namespace r4dx::server {

namespace {

const nlohmann::json& RequireField(const nlohmann::json& obj, const char* key,
                                    const char* context) {
  if (!obj.contains(key) || obj.at(key).is_null()) {
    throw ApiError{400, "invalid_request_error", std::string(context) + ": missing '" + key + "'"};
  }
  return obj.at(key);
}

std::string RequireString(const nlohmann::json& obj, const char* key, const char* context) {
  const nlohmann::json& v = RequireField(obj, key, context);
  if (!v.is_string()) {
    throw ApiError{400, "invalid_request_error", std::string(context) + ": '" + key + "' must be a string"};
  }
  return v.get<std::string>();
}

// Parses a message's `content` field, which OpenAI allows as either a plain string or an array of
// typed parts (`[{"type":"text","text":"..."}, ...]`). Any non-"text" part (image_url, input_
// audio, ...) is rejected outright -- vision/audio input isn't implemented yet (task point 1's
// "image parts rejected with a clear 400 until vision lands").
std::string ParseMessageContent(const nlohmann::json& content, const std::string& role) {
  if (content.is_string()) return content.get<std::string>();
  if (content.is_array()) {
    std::string out;
    for (const auto& part : content) {
      if (!part.is_object() || !part.contains("type") || !part.at("type").is_string()) {
        throw ApiError{400, "invalid_request_error",
                        "messages[].content: each part must be an object with a string 'type'"};
      }
      const std::string type = part.at("type").get<std::string>();
      if (type == "text") {
        if (!part.contains("text") || !part.at("text").is_string()) {
          throw ApiError{400, "invalid_request_error", "messages[].content: text part missing 'text'"};
        }
        out += part.at("text").get<std::string>();
      } else {
        throw ApiError{400, "invalid_request_error",
                        "messages[].content: part type '" + type + "' is not supported yet "
                        "(only 'text' -- image/audio input is not implemented; see docs/server.md)"};
      }
    }
    return out;
  }
  throw ApiError{400, "invalid_request_error",
                  "messages[" + role + "].content must be a string or an array of content parts"};
}

// Parses an assistant message's `tool_calls` array (the request-side replay of an earlier turn's
// tool call, for a multi-turn tool round trip -- docs/server.md's "Tool calls"). Each entry must
// be `{"id":..., "type":"function", "function":{"name":..., "arguments": "<json string>"}}`;
// `arguments` MUST decode to a JSON *object* (not a scalar/array) because the chat template
// iterates it with Jinja's `|items` filter (chat_template.jinja) -- a caller sending a
// non-object-decoding `arguments` string gets a clear 400 here rather than a confusing template
// render failure deep inside ChatTemplate::render().
std::vector<ToolCallOut> ParseToolCallsField(const nlohmann::json& arr) {
  std::vector<ToolCallOut> out;
  out.reserve(arr.size());
  for (const auto& entry : arr) {
    if (!entry.is_object()) {
      throw ApiError{400, "invalid_request_error", "messages[].tool_calls[] entries must be objects"};
    }
    if (entry.contains("type") && !entry.at("type").is_null() && entry.at("type") != "function") {
      throw ApiError{400, "invalid_request_error",
                      "messages[].tool_calls[].type must be \"function\" (the only kind this "
                      "checkpoint's template supports)"};
    }
    const nlohmann::json& fn = RequireField(entry, "function", "messages[].tool_calls[]");
    if (!fn.is_object()) {
      throw ApiError{400, "invalid_request_error", "messages[].tool_calls[].function must be an object"};
    }
    ToolCallOut tc;
    tc.id = entry.contains("id") && entry.at("id").is_string() ? entry.at("id").get<std::string>()
                                                                 : GenerateRequestId("call_");
    tc.name = RequireString(fn, "name", "messages[].tool_calls[].function");
    const std::string args_str =
        fn.contains("arguments") && !fn.at("arguments").is_null()
            ? RequireString(fn, "arguments", "messages[].tool_calls[].function")
            : "{}";
    nlohmann::json parsed_args;
    try {
      parsed_args = nlohmann::json::parse(args_str);
    } catch (const nlohmann::json::parse_error&) {
      throw ApiError{400, "invalid_request_error",
                      "messages[].tool_calls[].function.arguments must be a JSON-encoded string "
                      "(it is a STRING field whose contents are JSON, not a JSON object itself)"};
    }
    if (!parsed_args.is_object()) {
      throw ApiError{400, "invalid_request_error",
                      "messages[].tool_calls[].function.arguments must decode to a JSON object"};
    }
    tc.arguments_json = args_str;
    out.push_back(std::move(tc));
  }
  return out;
}

// tool_choice (task item 5): "none" clears `tools` outright (rendered), a named choice filters
// `tools` down to that one entry, "auto"/"required" leave `tools` untouched. Any other shape is a
// clear 400 rather than a silently-ignored field (task's own "silently ignoring tool_choice is
// worse than a clean error").
ToolChoice ParseToolChoice(const nlohmann::json& body, nlohmann::json& tools) {
  ToolChoice choice;
  if (!body.contains("tool_choice") || body.at("tool_choice").is_null()) return choice;
  const nlohmann::json& tc = body.at("tool_choice");

  if (tc.is_string()) {
    const std::string s = tc.get<std::string>();
    if (s == "none") {
      choice.kind = ToolChoiceKind::kNone;
      tools = nlohmann::json::array();
    } else if (s == "auto") {
      choice.kind = ToolChoiceKind::kAuto;
    } else if (s == "required") {
      if (tools.empty()) {
        throw ApiError{400, "invalid_request_error", "'tool_choice: required' needs a non-empty 'tools'"};
      }
      choice.kind = ToolChoiceKind::kRequired;
    } else {
      throw ApiError{400, "invalid_request_error",
                      "'tool_choice' string must be one of \"none\", \"auto\", \"required\""};
    }
    return choice;
  }

  if (tc.is_object()) {
    if (!tc.contains("type") || tc.at("type") != "function" || !tc.contains("function") ||
        !tc.at("function").is_object() || !tc.at("function").contains("name") ||
        !tc.at("function").at("name").is_string()) {
      throw ApiError{400, "invalid_request_error",
                      "'tool_choice' object must be {\"type\":\"function\",\"function\":"
                      "{\"name\":...}}"};
    }
    const std::string name = tc.at("function").at("name").get<std::string>();
    if (tools.empty()) {
      throw ApiError{400, "invalid_request_error", "'tool_choice' names a function but 'tools' is empty"};
    }
    nlohmann::json filtered = nlohmann::json::array();
    for (const auto& t : tools) {
      if (t.is_object() && t.contains("function") && t.at("function").is_object() &&
          t.at("function").contains("name") && t.at("function").at("name") == name) {
        filtered.push_back(t);
      }
    }
    if (filtered.empty()) {
      throw ApiError{400, "invalid_request_error",
                      "'tool_choice' names a function ('" + name + "') not present in 'tools'"};
    }
    choice.kind = ToolChoiceKind::kNamed;
    choice.name = name;
    tools = filtered;
    return choice;
  }

  throw ApiError{400, "invalid_request_error", "'tool_choice' must be a string or an object"};
}

SamplingParams ParseSampling(const nlohmann::json& body, const SamplingParams& defaults) {
  SamplingParams sp = defaults;
  if (body.contains("temperature") && !body.at("temperature").is_null()) {
    if (!body.at("temperature").is_number()) {
      throw ApiError{400, "invalid_request_error", "'temperature' must be a number"};
    }
    sp.temperature = body.at("temperature").get<float>();
    // Unlike top_p/top_k/min_p above, a negative temperature was previously accepted and silently
    // became greedy (r4dx::kernels::Sample treats <=0 as Argmax) -- reject it explicitly instead
    // (review finding, 2026-09-19), matching this function's own validate-every-sampling-field
    // pattern.
    if (sp.temperature < 0.0f) {
      throw ApiError{400, "invalid_request_error", "'temperature' must be >= 0"};
    }
  }
  if (body.contains("top_p") && !body.at("top_p").is_null()) {
    if (!body.at("top_p").is_number()) throw ApiError{400, "invalid_request_error", "'top_p' must be a number"};
    sp.top_p = body.at("top_p").get<float>();
    if (sp.top_p < 0.0f || sp.top_p > 1.0f) {
      throw ApiError{400, "invalid_request_error", "'top_p' must be in [0, 1]"};
    }
  }
  if (body.contains("top_k") && !body.at("top_k").is_null()) {
    if (!body.at("top_k").is_number_integer()) {
      throw ApiError{400, "invalid_request_error", "'top_k' must be an integer"};
    }
    sp.top_k = body.at("top_k").get<int>();
    if (sp.top_k < 0) throw ApiError{400, "invalid_request_error", "'top_k' must be >= 0"};
  }
  if (body.contains("min_p") && !body.at("min_p").is_null()) {
    if (!body.at("min_p").is_number()) throw ApiError{400, "invalid_request_error", "'min_p' must be a number"};
    sp.min_p = body.at("min_p").get<float>();
    if (sp.min_p < 0.0f || sp.min_p > 1.0f) {
      throw ApiError{400, "invalid_request_error", "'min_p' must be in [0, 1]"};
    }
  }
  if (body.contains("seed") && !body.at("seed").is_null()) {
    if (!body.at("seed").is_number_integer()) {
      throw ApiError{400, "invalid_request_error", "'seed' must be an integer"};
    }
    sp.has_seed = true;
    sp.seed = static_cast<uint64_t>(body.at("seed").get<int64_t>());
  }
  return sp;
}

std::optional<int64_t> ParseMaxTokens(const nlohmann::json& body) {
  // OpenAI has deprecated "max_tokens" in favor of "max_completion_tokens"; accept either, the
  // newer name taking precedence if both are present.
  for (const char* key : {"max_completion_tokens", "max_tokens"}) {
    if (body.contains(key) && !body.at(key).is_null()) {
      if (!body.at(key).is_number_integer()) {
        throw ApiError{400, "invalid_request_error", std::string("'") + key + "' must be an integer"};
      }
      const int64_t v = body.at(key).get<int64_t>();
      if (v < 0) throw ApiError{400, "invalid_request_error", std::string("'") + key + "' must be >= 0"};
      return v;
    }
  }
  return std::nullopt;
}

std::vector<std::string> ParseStop(const nlohmann::json& body) {
  std::vector<std::string> out;
  if (!body.contains("stop") || body.at("stop").is_null()) return out;
  const auto& s = body.at("stop");
  if (s.is_string()) {
    out.push_back(s.get<std::string>());
  } else if (s.is_array()) {
    for (const auto& e : s) {
      if (!e.is_string()) throw ApiError{400, "invalid_request_error", "'stop' array entries must be strings"};
      out.push_back(e.get<std::string>());
    }
  } else {
    throw ApiError{400, "invalid_request_error", "'stop' must be a string or an array of strings"};
  }
  return out;
}

bool ParseStream(const nlohmann::json& body) {
  if (!body.contains("stream") || body.at("stream").is_null()) return false;
  if (!body.at("stream").is_boolean()) throw ApiError{400, "invalid_request_error", "'stream' must be a boolean"};
  return body.at("stream").get<bool>();
}

// `stream_options: {"include_usage": bool}` -- parsed and validated unconditionally (independent
// of `stream`): a non-streaming request carrying this field is accepted and the result simply goes
// unused (see ChatCompletionRequest::stream_options_include_usage's own doc comment for why
// tolerating it is the friendlier choice over OpenAI's own stricter rejection).
bool ParseStreamOptions(const nlohmann::json& body) {
  if (!body.contains("stream_options") || body.at("stream_options").is_null()) return false;
  const nlohmann::json& so = body.at("stream_options");
  if (!so.is_object()) {
    throw ApiError{400, "invalid_request_error", "'stream_options' must be an object"};
  }
  bool include_usage = false;
  if (so.contains("include_usage") && !so.at("include_usage").is_null()) {
    if (!so.at("include_usage").is_boolean()) {
      throw ApiError{400, "invalid_request_error", "'stream_options.include_usage' must be a boolean"};
    }
    include_usage = so.at("include_usage").get<bool>();
  }
  return include_usage;
}

}  // namespace

nlohmann::json ErrorBody(const ApiError& err) {
  return {{"error", {{"message", err.message}, {"type", err.type}, {"param", nullptr}, {"code", nullptr}}}};
}

bool ResolveEnableThinking(const nlohmann::json& chat_template_kwargs, bool default_thinking) {
  if (!chat_template_kwargs.contains("enable_thinking") ||
      chat_template_kwargs.at("enable_thinking").is_null()) {
    return default_thinking;
  }
  const nlohmann::json& v = chat_template_kwargs.at("enable_thinking");
  return v.is_boolean() ? v.get<bool>() : default_thinking;
}

ChatCompletionRequest ParseChatCompletionRequest(const nlohmann::json& body,
                                                  const SamplingParams& sampling_defaults) {
  if (!body.is_object()) throw ApiError{400, "invalid_request_error", "request body must be a JSON object"};

  ChatCompletionRequest req;
  if (body.contains("model") && !body.at("model").is_null()) {
    req.model = RequireString(body, "model", "chat completion request");
  }

  const nlohmann::json& messages = RequireField(body, "messages", "chat completion request");
  if (!messages.is_array() || messages.empty()) {
    throw ApiError{400, "invalid_request_error", "'messages' must be a non-empty array"};
  }
  for (const auto& m : messages) {
    if (!m.is_object()) throw ApiError{400, "invalid_request_error", "messages[] entries must be objects"};
    const std::string role = RequireString(m, "role", "messages[]");
    if (role != "system" && role != "user" && role != "assistant" && role != "tool" && role != "function") {
      throw ApiError{400, "invalid_request_error",
                      "messages[].role '" + role + "' is not supported (must be one of "
                      "system/user/assistant/tool/function)"};
    }

    ChatMessage cm;
    cm.role = role;

    std::vector<ToolCallOut> tool_calls;
    if (role == "assistant" && m.contains("tool_calls") && !m.at("tool_calls").is_null()) {
      if (!m.at("tool_calls").is_array()) {
        throw ApiError{400, "invalid_request_error", "messages[].tool_calls must be an array"};
      }
      tool_calls = ParseToolCallsField(m.at("tool_calls"));
    }

    // content: required for every role except an assistant message that is a pure tool call
    // (tool_calls non-empty, content omitted/null -- OpenAI's own shape for that turn).
    const bool content_optional = role == "assistant" && !tool_calls.empty();
    if (!m.contains("content") || m.at("content").is_null()) {
      if (!content_optional) {
        throw ApiError{400, "invalid_request_error", "messages[]: missing 'content'"};
      }
      cm.content = std::nullopt;
    } else {
      cm.content = ParseMessageContent(m.at("content"), role);
    }
    cm.tool_calls = std::move(tool_calls);

    // Multi-turn reasoning_content replay (task item 5e): accepted on any role (the chat template
    // only ever reads it on an assistant turn, chat_template.jinja's own `elif message.role ==
    // "assistant"` branch -- attaching it to another role's message is harmless, just never read),
    // validated as a string when present so a caller sending e.g. an object gets a clear 400 rather
    // than a confusing template render failure.
    if (m.contains("reasoning_content") && !m.at("reasoning_content").is_null()) {
      cm.reasoning_content = RequireString(m, "reasoning_content", "messages[]");
    }

    if (role == "tool") {
      cm.tool_call_id = RequireString(m, "tool_call_id", "messages[] (role \"tool\")");
    } else if (role == "function") {
      // Legacy pre-"tools" OpenAI function-calling shape: identifies the call by function name
      // rather than an id (there was no tool_call_id concept yet). Accepted for API completeness.
      // CORRECTION (review finding, 2026-09-20): chat_template.jinja has NO "function" branch at
      // all -- only system/user/assistant/tool -- so this role is remapped to "tool" by
      // Engine::RunRequest (engine.cpp) before rendering, rather than being passed through as-is
      // (the previous claim here, that the template "looks at `content` for either `tool`/
      // `function` role", was never true: an unmapped "function" role hit the template's own
      // `raise_exception('Unexpected message role.')` and 500'd). `name` still carries no template
      // effect either way -- kept purely so a caller round-tripping the legacy shape doesn't 400.
      cm.name = RequireString(m, "name", "messages[] (role \"function\")");
    }

    req.messages.push_back(std::move(cm));
  }

  req.sampling = ParseSampling(body, sampling_defaults);
  req.max_tokens = ParseMaxTokens(body);
  req.stop = ParseStop(body);
  req.stream = ParseStream(body);
  req.stream_options_include_usage = ParseStreamOptions(body);

  if (body.contains("chat_template_kwargs") && !body.at("chat_template_kwargs").is_null()) {
    if (!body.at("chat_template_kwargs").is_object()) {
      throw ApiError{400, "invalid_request_error", "'chat_template_kwargs' must be an object"};
    }
    req.chat_template_kwargs = body.at("chat_template_kwargs");
  }
  if (body.contains("tools") && !body.at("tools").is_null()) {
    if (!body.at("tools").is_array()) throw ApiError{400, "invalid_request_error", "'tools' must be an array"};
    // Shape-validate each entry (review finding, 2026-09-20): previously only `is_array()` was
    // checked, unlike tool_choice's own strict validation below. Consequences of leaving a
    // malformed entry unvalidated: (a) it would reach chat_template.jinja's tool-rendering loop
    // and raise there, surfacing as an uninformative 500 instead of a clear 400; (b) worse, it
    // still sets Engine::RunRequest's tool_mode (any non-empty `tools`), while `known_names`
    // (built from `t.at("function").at("name")`, engine.cpp) silently comes out empty for entries
    // missing that shape -- and DropUnknownToolCalls (tool_call_parser.cpp) treats an empty
    // known_names list as "no filter configured" and returns early, passing through ANY
    // model-emitted call name unfiltered, the exact opposite of the intended "drop calls to
    // undefined tools" behavior.
    for (const auto& t : body.at("tools")) {
      if (!t.is_object() || !t.contains("type") || t.at("type") != "function" ||
          !t.contains("function") || !t.at("function").is_object() ||
          !t.at("function").contains("name") || !t.at("function").at("name").is_string()) {
        throw ApiError{400, "invalid_request_error",
                        "each 'tools[]' entry must be {\"type\":\"function\",\"function\":"
                        "{\"name\":...}}"};
      }
    }
    req.tools = body.at("tools");
  }
  req.tool_choice = ParseToolChoice(body, req.tools);

  return req;
}

CompletionRequest ParseCompletionRequest(const nlohmann::json& body,
                                          const SamplingParams& sampling_defaults) {
  if (!body.is_object()) throw ApiError{400, "invalid_request_error", "request body must be a JSON object"};

  CompletionRequest req;
  if (body.contains("model") && !body.at("model").is_null()) {
    req.model = RequireString(body, "model", "completion request");
  }
  req.prompt = RequireString(body, "prompt", "completion request");

  req.sampling = ParseSampling(body, sampling_defaults);
  req.max_tokens = ParseMaxTokens(body);
  req.stop = ParseStop(body);
  req.stream = ParseStream(body);
  req.stream_options_include_usage = ParseStreamOptions(body);
  return req;
}

std::string GenerateRequestId(const char* prefix) {
  static std::mutex rng_mu;
  static std::mt19937_64 rng(std::random_device{}());
  uint64_t r;
  {
    std::lock_guard<std::mutex> lock(rng_mu);
    r = rng();
  }
  char buf[24];
  std::snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(r));
  return std::string(prefix) + buf;
}

namespace {

// `usage.completion_tokens_details.reasoning_tokens` -- added iff `usage.reasoning_tokens` is set,
// so every builder below shares this exact "omit when not applicable" rule instead of repeating it
// (task item 5f: a thinking-off response's usage object stays byte-identical to before this field
// existed).
nlohmann::json BuildUsageJson(const UsageStats& usage) {
  nlohmann::json out = {{"prompt_tokens", usage.prompt_tokens},
                        {"completion_tokens", usage.completion_tokens},
                        {"total_tokens", usage.TotalTokens()}};
  if (usage.reasoning_tokens) {
    out["completion_tokens_details"] = {{"reasoning_tokens", *usage.reasoning_tokens}};
  }
  return out;
}

}  // namespace

// Model metadata (docs/server.md's "Model metadata" section, task item 1): every field beyond
// {id, object, created, owned_by} is an r4dx extension -- different clients read different ones, so
// this emits the common set several inference servers already use:
//   context_length / max_model_len / max_completion_tokens -- this server's own `--max-ctx`
//     (`max_ctx`), under three different names because different client libraries look for
//     different ones (vLLM's OpenAI-compatible server uses `max_model_len`; some clients probe
//     `context_length`; `max_completion_tokens` mirrors the request-side field name).
//   meta.n_ctx / meta.n_ctx_train -- llama.cpp `/v1/models` shape: `n_ctx` is this server's
//     configured `--max-ctx`, `n_ctx_train` is the checkpoint's own native limit
//     (kModelNativeContextLength -- see that constant's own doc comment for why it is not read
//     from the loaded container).
//   capabilities -- a fixed list reflecting what this server actually does: plain completion, chat
//     (the chat template), tool_use (`tools`/`tool_choice`, docs/server.md's "Tool calls"), and
//     reasoning (`chat_template_kwargs.enable_thinking`, docs/server.md's "reasoning_content").
//   supported_parameters -- exactly the request fields ParseChatCompletionRequest/ParseSampling/
//     etc. actually parse (openai_types.cpp): sampling (`temperature`/`top_p`/`top_k`/`min_p`/
//     `seed`), generation length (`max_tokens`/`max_completion_tokens`), `stop`, `stream`/
//     `stream_options`, tool calling (`tools`/`tool_choice`), and `chat_template_kwargs`. Anything
//     NOT in this list (e.g. `logprobs`, `presence_penalty`) is silently ignored by this server
//     today, so it is deliberately left out rather than falsely advertised.
//   architecture -- text in, text out (no vision tower yet, docs/server.md's "Deferred" section).
nlohmann::json BuildModelEntryJson(const std::string& model_id, int64_t created_unix,
                                    int64_t max_ctx) {
  return {{"id", model_id},
          {"object", "model"},
          {"created", created_unix},
          {"owned_by", "r4dx"},
          {"context_length", max_ctx},
          {"max_model_len", max_ctx},
          {"max_completion_tokens", max_ctx},
          {"meta", {{"n_ctx", max_ctx}, {"n_ctx_train", kModelNativeContextLength}}},
          {"capabilities", nlohmann::json::array({"completion", "chat", "tool_use", "reasoning"})},
          {"supported_parameters",
           nlohmann::json::array({"temperature", "top_p", "top_k", "min_p", "seed", "max_tokens",
                                  "max_completion_tokens", "stop", "stream", "stream_options",
                                  "tools", "tool_choice", "chat_template_kwargs"})},
          {"architecture", {{"input_modalities", nlohmann::json::array({"text"})},
                            {"output_modalities", nlohmann::json::array({"text"})}}}};
}

nlohmann::json BuildModelsResponse(const std::string& model_id, int64_t created_unix, int64_t max_ctx) {
  return {{"object", "list"},
          {"data", nlohmann::json::array({BuildModelEntryJson(model_id, created_unix, max_ctx)})}};
}

nlohmann::json BuildTimingsJson(const TimingStats& timings) {
  const double prompt_per_second =
      timings.prompt_ms > 0.0 ? timings.prompt_n / (timings.prompt_ms / 1000.0) : 0.0;
  const double predicted_per_second =
      timings.predicted_ms > 0.0 ? timings.predicted_n / (timings.predicted_ms / 1000.0) : 0.0;
  nlohmann::json out = {{"prompt_n", timings.prompt_n},
                        {"prompt_ms", timings.prompt_ms},
                        {"prompt_per_second", prompt_per_second},
                        {"predicted_n", timings.predicted_n},
                        {"predicted_ms", timings.predicted_ms},
                        {"predicted_per_second", predicted_per_second}};
  if (timings.draft_n) {
    out["draft_n"] = *timings.draft_n;
    out["draft_n_accepted"] = timings.draft_n_accepted.value_or(0);
  }
  return out;
}

nlohmann::json BuildChatCompletionResponse(const std::string& id, const std::string& model_id,
                                            int64_t created_unix, const std::string& content,
                                            const std::string& finish_reason,
                                            const UsageStats& usage, const TimingStats& timings,
                                            const std::optional<std::string>& reasoning_content) {
  nlohmann::json message = {{"role", "assistant"}, {"content", content}};
  if (reasoning_content) message["reasoning_content"] = *reasoning_content;
  return {{"id", id},
          {"object", "chat.completion"},
          {"created", created_unix},
          {"model", model_id},
          {"choices", nlohmann::json::array({{{"index", 0},
                                               {"message", message},
                                               {"finish_reason", finish_reason}}})},
          {"usage", BuildUsageJson(usage)},
          {"timings", BuildTimingsJson(timings)}};
}

nlohmann::json BuildToolCallsJson(const std::vector<ToolCallOut>& tool_calls) {
  nlohmann::json arr = nlohmann::json::array();
  for (size_t i = 0; i < tool_calls.size(); ++i) {
    arr.push_back({{"index", static_cast<int64_t>(i)},
                    {"id", tool_calls[i].id},
                    {"type", "function"},
                    {"function", {{"name", tool_calls[i].name}, {"arguments", tool_calls[i].arguments_json}}}});
  }
  return arr;
}

nlohmann::json BuildChatCompletionResponse(const std::string& id, const std::string& model_id,
                                            int64_t created_unix,
                                            const std::optional<std::string>& content,
                                            const std::vector<ToolCallOut>& tool_calls,
                                            const std::string& finish_reason,
                                            const UsageStats& usage, const TimingStats& timings,
                                            const std::optional<std::string>& reasoning_content) {
  nlohmann::json message = {{"role", "assistant"}, {"content", content ? nlohmann::json(*content) : nlohmann::json(nullptr)}};
  if (!tool_calls.empty()) message["tool_calls"] = BuildToolCallsJson(tool_calls);
  if (reasoning_content) message["reasoning_content"] = *reasoning_content;
  return {{"id", id},
          {"object", "chat.completion"},
          {"created", created_unix},
          {"model", model_id},
          {"choices", nlohmann::json::array({{{"index", 0},
                                               {"message", message},
                                               {"finish_reason", finish_reason}}})},
          {"usage", BuildUsageJson(usage)},
          {"timings", BuildTimingsJson(timings)}};
}

nlohmann::json BuildChatCompletionChunk(const std::string& id, const std::string& model_id,
                                         int64_t created_unix, const nlohmann::json& delta,
                                         const std::optional<std::string>& finish_reason,
                                         bool include_usage_null,
                                         const std::optional<TimingStats>& timings) {
  nlohmann::json out = {{"id", id},
          {"object", "chat.completion.chunk"},
          {"created", created_unix},
          {"model", model_id},
          {"choices", nlohmann::json::array({{{"index", 0},
                                               {"delta", delta},
                                               {"finish_reason", finish_reason ? nlohmann::json(*finish_reason)
                                                                                : nlohmann::json(nullptr)}}})}};
  if (include_usage_null) out["usage"] = nullptr;
  if (timings) out["timings"] = BuildTimingsJson(*timings);
  return out;
}

nlohmann::json BuildChatCompletionUsageChunk(const std::string& id, const std::string& model_id,
                                              int64_t created_unix, const UsageStats& usage,
                                              const TimingStats& timings) {
  return {{"id", id},
          {"object", "chat.completion.chunk"},
          {"created", created_unix},
          {"model", model_id},
          {"choices", nlohmann::json::array()},
          {"usage", BuildUsageJson(usage)},
          {"timings", BuildTimingsJson(timings)}};
}

nlohmann::json BuildCompletionResponse(const std::string& id, const std::string& model_id,
                                        int64_t created_unix, const std::string& text,
                                        const std::string& finish_reason, const UsageStats& usage,
                                        const TimingStats& timings) {
  return {{"id", id},
          {"object", "text_completion"},
          {"created", created_unix},
          {"model", model_id},
          {"choices", nlohmann::json::array({{{"index", 0},
                                               {"text", text},
                                               {"logprobs", nullptr},
                                               {"finish_reason", finish_reason}}})},
          {"usage", BuildUsageJson(usage)},
          {"timings", BuildTimingsJson(timings)}};
}

nlohmann::json BuildCompletionChunk(const std::string& id, const std::string& model_id,
                                     int64_t created_unix, const std::string& text_delta,
                                     const std::optional<std::string>& finish_reason,
                                     bool include_usage_null,
                                     const std::optional<TimingStats>& timings) {
  nlohmann::json out = {{"id", id},
          {"object", "text_completion"},
          {"created", created_unix},
          {"model", model_id},
          {"choices", nlohmann::json::array({{{"index", 0},
                                               {"text", text_delta},
                                               {"logprobs", nullptr},
                                               {"finish_reason", finish_reason ? nlohmann::json(*finish_reason)
                                                                                : nlohmann::json(nullptr)}}})}};
  if (include_usage_null) out["usage"] = nullptr;
  if (timings) out["timings"] = BuildTimingsJson(*timings);
  return out;
}

nlohmann::json BuildCompletionUsageChunk(const std::string& id, const std::string& model_id,
                                          int64_t created_unix, const UsageStats& usage,
                                          const TimingStats& timings) {
  return {{"id", id},
          {"object", "text_completion"},
          {"created", created_unix},
          {"model", model_id},
          {"choices", nlohmann::json::array()},
          {"usage", BuildUsageJson(usage)},
          {"timings", BuildTimingsJson(timings)}};
}

}  // namespace r4dx::server
