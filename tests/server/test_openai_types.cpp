// tests/server/test_openai_types.cpp -- pure CPU unit test for src/server/openai_types.h: chat/
// completion request parsing+validation and OpenAI-shaped response JSON building.
#include <cstdio>
#include <functional>
#include <string>

#include "nlohmann/json.hpp"
#include "openai_types.h"

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

using nlohmann::json;
using namespace r4dx::server;

bool ThrowsApiError(const std::function<void()>& fn, int expect_status = -1) {
  try {
    fn();
  } catch (const ApiError& e) {
    return expect_status < 0 || e.http_status == expect_status;
  }
  return false;
}

// ---- ParseChatCompletionRequest ---------------------------------------------------------------

void TestChatMinimal() {
  json body = {{"messages", json::array({{{"role", "user"}, {"content", "hi"}}})}};
  const auto req = ParseChatCompletionRequest(body);
  CHECK(req.messages.size() == 1);
  CHECK(req.messages[0].role == "user");
  CHECK(req.messages[0].content == "hi");
  CHECK(!req.stream);
  CHECK(!req.max_tokens.has_value());
  CHECK(req.stop.empty());
  CHECK(req.sampling.temperature == 1.0f);
  CHECK(req.sampling.top_p == 1.0f);
  CHECK(req.sampling.top_k == 0);
  CHECK(req.sampling.min_p == 0.0f);
  CHECK(!req.sampling.has_seed);
}

void TestChatSamplingDefaultsApplyOnlyWhenOmitted() {
  SamplingParams defaults;
  defaults.temperature = 0.5f;
  defaults.top_p = 0.8f;
  defaults.top_k = 20;
  defaults.min_p = 0.02f;

  json body = {{"messages", json::array({{{"role", "user"}, {"content", "hi"}}})},
               {"top_k", 40}};  // client overrides only top_k
  const auto req = ParseChatCompletionRequest(body, defaults);
  CHECK(req.sampling.temperature == 0.5f);  // from server defaults
  CHECK(req.sampling.top_p == 0.8f);        // from server defaults
  CHECK(req.sampling.top_k == 40);          // client override wins
  CHECK(req.sampling.min_p == 0.02f);       // from server defaults
}

void TestChatFullFields() {
  json body = {
      {"model", "qwen"},
      {"messages", json::array({{{"role", "system"}, {"content", "be terse"}},
                                 {{"role", "user"}, {"content", "hi"}}})},
      {"temperature", 0.7}, {"top_p", 0.9}, {"top_k", 40}, {"min_p", 0.05},
      {"seed", 123}, {"max_tokens", 64}, {"stop", json::array({"</s>", "STOP"})},
      {"stream", true}, {"chat_template_kwargs", {{"enable_thinking", false}}},
      {"tools", json::array()}};
  const auto req = ParseChatCompletionRequest(body);
  CHECK(req.model == "qwen");
  CHECK(req.messages.size() == 2);
  CHECK(req.sampling.temperature == 0.7f);
  CHECK(req.sampling.top_p == 0.9f);
  CHECK(req.sampling.top_k == 40);
  CHECK(req.sampling.min_p == 0.05f);
  CHECK(req.sampling.has_seed && req.sampling.seed == 123);
  CHECK(req.max_tokens.has_value() && *req.max_tokens == 64);
  CHECK(req.stop.size() == 2 && req.stop[0] == "</s>" && req.stop[1] == "STOP");
  CHECK(req.stream);
  CHECK(req.chat_template_kwargs.at("enable_thinking") == false);
}

void TestChatContentAsTextParts() {
  json body = {{"messages", json::array({{{"role", "user"},
                                           {"content", json::array({{{"type", "text"}, {"text", "a"}},
                                                                     {{"type", "text"}, {"text", "b"}}})}}})}};
  const auto req = ParseChatCompletionRequest(body);
  CHECK(req.messages[0].content == "ab");
}

void TestChatImagePartRejected() {
  json body = {{"messages",
                json::array({{{"role", "user"},
                              {"content", json::array({{{"type", "image_url"},
                                                         {"image_url", {{"url", "http://x"}}}}})}}})}};
  CHECK(ThrowsApiError([&] { ParseChatCompletionRequest(body); }, 400));
}

void TestChatMissingMessagesThrows() {
  json body = json::object();
  CHECK(ThrowsApiError([&] { ParseChatCompletionRequest(body); }, 400));
}

void TestChatEmptyMessagesThrows() {
  json body = {{"messages", json::array()}};
  CHECK(ThrowsApiError([&] { ParseChatCompletionRequest(body); }, 400));
}

void TestChatUnsupportedRoleThrows() {
  json body = {{"messages", json::array({{{"role", "tool"}, {"content", "x"}}})}};
  CHECK(ThrowsApiError([&] { ParseChatCompletionRequest(body); }, 400));
}

void TestChatMissingContentThrows() {
  json body = {{"messages", json::array({{{"role", "user"}}})}};
  CHECK(ThrowsApiError([&] { ParseChatCompletionRequest(body); }, 400));
}

void TestChatBadSamplingRangesThrow() {
  auto with = [](json extra) {
    json body = {{"messages", json::array({{{"role", "user"}, {"content", "hi"}}})}};
    body.update(extra);
    return body;
  };
  CHECK(ThrowsApiError([&] { ParseChatCompletionRequest(with({{"top_p", 1.5}})); }, 400));
  CHECK(ThrowsApiError([&] { ParseChatCompletionRequest(with({{"min_p", -0.1}})); }, 400));
  CHECK(ThrowsApiError([&] { ParseChatCompletionRequest(with({{"top_k", -1}})); }, 400));
  CHECK(ThrowsApiError([&] { ParseChatCompletionRequest(with({{"temperature", "hot"}})); }, 400));
  CHECK(ThrowsApiError([&] { ParseChatCompletionRequest(with({{"stream", "yes"}})); }, 400));
  CHECK(ThrowsApiError([&] { ParseChatCompletionRequest(with({{"stop", 5}})); }, 400));
  CHECK(ThrowsApiError([&] { ParseChatCompletionRequest(with({{"max_tokens", -1}})); }, 400));
  CHECK(ThrowsApiError([&] { ParseChatCompletionRequest(with({{"chat_template_kwargs", 5}})); }, 400));
}

void TestChatNotAnObjectThrows() {
  CHECK(ThrowsApiError([&] { ParseChatCompletionRequest(json::array()); }, 400));
}

void TestChatStopAsSingleString() {
  json body = {{"messages", json::array({{{"role", "user"}, {"content", "hi"}}})}, {"stop", "END"}};
  const auto req = ParseChatCompletionRequest(body);
  CHECK(req.stop.size() == 1 && req.stop[0] == "END");
}

// ---- ParseCompletionRequest -------------------------------------------------------------------

void TestCompletionMinimal() {
  json body = {{"prompt", "once upon a time"}};
  const auto req = ParseCompletionRequest(body);
  CHECK(req.prompt == "once upon a time");
  CHECK(!req.stream);
}

void TestCompletionMissingPromptThrows() {
  json body = json::object();
  CHECK(ThrowsApiError([&] { ParseCompletionRequest(body); }, 400));
}

void TestCompletionPromptWrongTypeThrows() {
  json body = {{"prompt", 5}};
  CHECK(ThrowsApiError([&] { ParseCompletionRequest(body); }, 400));
}

// ---- response builders --------------------------------------------------------------------

void TestBuildModelsResponse() {
  const json resp = BuildModelsResponse("my-model", 1000);
  CHECK(resp.at("object") == "list");
  CHECK(resp.at("data").size() == 1);
  CHECK(resp.at("data")[0].at("id") == "my-model");
  CHECK(resp.at("data")[0].at("object") == "model");
}

void TestBuildChatCompletionResponse() {
  UsageStats usage{7, 3};
  const json resp = BuildChatCompletionResponse("id1", "m", 42, "hello", "stop", usage);
  CHECK(resp.at("id") == "id1");
  CHECK(resp.at("object") == "chat.completion");
  CHECK(resp.at("model") == "m");
  CHECK(resp.at("choices")[0].at("message").at("role") == "assistant");
  CHECK(resp.at("choices")[0].at("message").at("content") == "hello");
  CHECK(resp.at("choices")[0].at("finish_reason") == "stop");
  CHECK(resp.at("usage").at("prompt_tokens") == 7);
  CHECK(resp.at("usage").at("completion_tokens") == 3);
  CHECK(resp.at("usage").at("total_tokens") == 10);
}

void TestBuildChatCompletionChunk() {
  const json chunk = BuildChatCompletionChunk("id1", "m", 42, {{"content", "hi"}}, std::nullopt);
  CHECK(chunk.at("object") == "chat.completion.chunk");
  CHECK(chunk.at("choices")[0].at("delta").at("content") == "hi");
  CHECK(chunk.at("choices")[0].at("finish_reason").is_null());

  const json last = BuildChatCompletionChunk("id1", "m", 42, json::object(), std::string("stop"));
  CHECK(last.at("choices")[0].at("finish_reason") == "stop");
}

void TestBuildCompletionResponseAndChunk() {
  UsageStats usage{4, 2};
  const json resp = BuildCompletionResponse("id2", "m", 1, "text out", "length", usage);
  CHECK(resp.at("object") == "text_completion");
  CHECK(resp.at("choices")[0].at("text") == "text out");
  CHECK(resp.at("choices")[0].at("finish_reason") == "length");

  const json chunk = BuildCompletionChunk("id2", "m", 1, "delta", std::nullopt);
  CHECK(chunk.at("choices")[0].at("text") == "delta");
  CHECK(chunk.at("choices")[0].at("finish_reason").is_null());
}

void TestGenerateRequestIdPrefixAndUniqueness() {
  const std::string a = GenerateRequestId("chatcmpl-");
  const std::string b = GenerateRequestId("chatcmpl-");
  CHECK(a.rfind("chatcmpl-", 0) == 0);
  CHECK(a != b);
}

void TestErrorBody() {
  const json body = ErrorBody(ApiError{400, "invalid_request_error", "bad"});
  CHECK(body.at("error").at("message") == "bad");
  CHECK(body.at("error").at("type") == "invalid_request_error");
}

}  // namespace

int main() {
  TestChatMinimal();
  TestChatSamplingDefaultsApplyOnlyWhenOmitted();
  TestChatFullFields();
  TestChatContentAsTextParts();
  TestChatImagePartRejected();
  TestChatMissingMessagesThrows();
  TestChatEmptyMessagesThrows();
  TestChatUnsupportedRoleThrows();
  TestChatMissingContentThrows();
  TestChatBadSamplingRangesThrow();
  TestChatNotAnObjectThrows();
  TestChatStopAsSingleString();
  TestCompletionMinimal();
  TestCompletionMissingPromptThrows();
  TestCompletionPromptWrongTypeThrows();
  TestBuildModelsResponse();
  TestBuildChatCompletionResponse();
  TestBuildChatCompletionChunk();
  TestBuildCompletionResponseAndChunk();
  TestGenerateRequestIdPrefixAndUniqueness();
  TestErrorBody();

  if (g_failures > 0) {
    std::fprintf(stderr, "%d check(s) failed\n", g_failures);
    return 1;
  }
  std::fprintf(stderr, "all openai_types checks passed\n");
  return 0;
}
