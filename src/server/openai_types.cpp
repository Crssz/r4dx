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

}  // namespace

nlohmann::json ErrorBody(const ApiError& err) {
  return {{"error", {{"message", err.message}, {"type", err.type}, {"param", nullptr}, {"code", nullptr}}}};
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
    if (role != "system" && role != "user" && role != "assistant") {
      throw ApiError{400, "invalid_request_error",
                      "messages[].role '" + role + "' is not supported (only system/user/assistant; "
                      "'tool'/'function' roles need tool-call parsing, not implemented yet -- see docs/server.md)"};
    }
    const nlohmann::json& content = RequireField(m, "content", "messages[]");
    req.messages.push_back(ChatMessage{role, ParseMessageContent(content, role)});
  }

  req.sampling = ParseSampling(body, sampling_defaults);
  req.max_tokens = ParseMaxTokens(body);
  req.stop = ParseStop(body);
  req.stream = ParseStream(body);

  if (body.contains("chat_template_kwargs") && !body.at("chat_template_kwargs").is_null()) {
    if (!body.at("chat_template_kwargs").is_object()) {
      throw ApiError{400, "invalid_request_error", "'chat_template_kwargs' must be an object"};
    }
    req.chat_template_kwargs = body.at("chat_template_kwargs");
  }
  if (body.contains("tools") && !body.at("tools").is_null()) {
    if (!body.at("tools").is_array()) throw ApiError{400, "invalid_request_error", "'tools' must be an array"};
    req.tools = body.at("tools");
  }

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

nlohmann::json BuildModelsResponse(const std::string& model_id, int64_t created_unix) {
  return {{"object", "list"},
          {"data", nlohmann::json::array({{{"id", model_id},
                                            {"object", "model"},
                                            {"created", created_unix},
                                            {"owned_by", "r4dx"}}})}};
}

nlohmann::json BuildChatCompletionResponse(const std::string& id, const std::string& model_id,
                                            int64_t created_unix, const std::string& content,
                                            const std::string& finish_reason,
                                            const UsageStats& usage) {
  return {{"id", id},
          {"object", "chat.completion"},
          {"created", created_unix},
          {"model", model_id},
          {"choices", nlohmann::json::array({{{"index", 0},
                                               {"message", {{"role", "assistant"}, {"content", content}}},
                                               {"finish_reason", finish_reason}}})},
          {"usage",
           {{"prompt_tokens", usage.prompt_tokens},
            {"completion_tokens", usage.completion_tokens},
            {"total_tokens", usage.TotalTokens()}}}};
}

nlohmann::json BuildChatCompletionChunk(const std::string& id, const std::string& model_id,
                                         int64_t created_unix, const nlohmann::json& delta,
                                         const std::optional<std::string>& finish_reason) {
  return {{"id", id},
          {"object", "chat.completion.chunk"},
          {"created", created_unix},
          {"model", model_id},
          {"choices", nlohmann::json::array({{{"index", 0},
                                               {"delta", delta},
                                               {"finish_reason", finish_reason ? nlohmann::json(*finish_reason)
                                                                                : nlohmann::json(nullptr)}}})}};
}

nlohmann::json BuildCompletionResponse(const std::string& id, const std::string& model_id,
                                        int64_t created_unix, const std::string& text,
                                        const std::string& finish_reason, const UsageStats& usage) {
  return {{"id", id},
          {"object", "text_completion"},
          {"created", created_unix},
          {"model", model_id},
          {"choices", nlohmann::json::array({{{"index", 0},
                                               {"text", text},
                                               {"logprobs", nullptr},
                                               {"finish_reason", finish_reason}}})},
          {"usage",
           {{"prompt_tokens", usage.prompt_tokens},
            {"completion_tokens", usage.completion_tokens},
            {"total_tokens", usage.TotalTokens()}}}};
}

nlohmann::json BuildCompletionChunk(const std::string& id, const std::string& model_id,
                                     int64_t created_unix, const std::string& text_delta,
                                     const std::optional<std::string>& finish_reason) {
  return {{"id", id},
          {"object", "text_completion"},
          {"created", created_unix},
          {"model", model_id},
          {"choices", nlohmann::json::array({{{"index", 0},
                                               {"text", text_delta},
                                               {"logprobs", nullptr},
                                               {"finish_reason", finish_reason ? nlohmann::json(*finish_reason)
                                                                                : nlohmann::json(nullptr)}}})}};
}

}  // namespace r4dx::server
