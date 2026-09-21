#include "http_server.h"

#include <chrono>
#include <cstdio>

#include "httplib.h"
#include "openai_types.h"
#include "response_sink.h"
#include "sse.h"

namespace r4dx::server {

namespace {

int64_t NowUnix() {
  return static_cast<int64_t>(std::chrono::duration_cast<std::chrono::seconds>(
                                   std::chrono::system_clock::now().time_since_epoch())
                                   .count());
}

void RespondError(httplib::Response& res, const ApiError& err) {
  res.status = err.http_status;
  res.set_content(ErrorBody(err).dump(), "application/json");
}

void RespondError(httplib::Response& res, int status, const std::string& type,
                   const std::string& message) {
  RespondError(res, ApiError{status, type, message});
}

// Engine::RunRequest reports every failure (a genuine caller-error 400, e.g. "prompt rendered to
// zero tokens" or "exceeds --max-ctx", equally with a real internal fault) through the same
// sink->error_status/error_message pair, with no `type` of its own (unlike ApiError above) --
// review finding, 2026-09-19: the two BufferingSink call sites below previously hardcoded
// "server_error" even for those genuine 400s. Map by status instead, matching ApiError's own
// convention (<500 is the caller's fault, >=500 is ours).
const char* ErrorTypeForStatus(int status) {
  return status < 500 ? "invalid_request_error" : "server_error";
}

// Parses the request body as JSON, mapping a malformed body to the same ApiError shape
// ParseChatCompletionRequest/ParseCompletionRequest throw for a semantically invalid one, so the
// caller only needs one catch site.
nlohmann::json ParseJsonBody(const httplib::Request& req) {
  try {
    return nlohmann::json::parse(req.body);
  } catch (const nlohmann::json::parse_error& e) {
    throw ApiError{400, "invalid_request_error", std::string("malformed JSON body: ") + e.what()};
  }
}

// Wires httplib's chunked-content-provider pull loop to a StreamingSink's push queue. `sink` is
// captured by shared_ptr so it outlives both the (already-returned) route handler and any
// in-flight Engine::RunRequest call still writing to it.
void ServeStream(httplib::Response& res, std::shared_ptr<StreamingSink> sink) {
  res.status = 200;
  res.set_header("Cache-Control", "no-cache");
  res.set_header("X-Accel-Buffering", "no");  // hint reverse proxies not to buffer the stream
  res.set_chunked_content_provider(
      "text/event-stream",
      [sink](size_t /*offset*/, httplib::DataSink& ds) -> bool {
        // cpp-httplib's chunked-provider contract (detail::write_content_chunked): returning
        // false from this callback is treated as Error::Canceled -- the response is aborted
        // WITHOUT the terminating "0\r\n\r\n" chunk, which every HTTP/1.1 client (curl, browsers)
        // correctly reports as a truncated/incomplete transfer. A clean end-of-stream must
        // instead call ds.done() (writes the terminator) and return true.
        std::string chunk;
        if (!sink->Next(chunk)) {
          ds.done();
          return true;
        }
        if (!ds.write(chunk.data(), chunk.size())) {
          sink->Cancel();
          return false;
        }
        return true;
      },
      [sink](bool /*success*/) { sink->Cancel(); });
}

}  // namespace

struct HttpServer::Impl {
  explicit Impl(Engine& engine_in) : engine(engine_in) {}

  Engine& engine;
  httplib::Server svr;
};

HttpServer::HttpServer(Engine& engine) : impl_(std::make_unique<Impl>(engine)) {
  httplib::Server& svr = impl_->svr;
  Engine& engine_ref = impl_->engine;

  svr.Get("/health", [&engine_ref](const httplib::Request&, httplib::Response& res) {
    nlohmann::json body = {{"status", "ok"}, {"model", engine_ref.ModelId()}};
    res.set_content(body.dump(), "application/json");
  });

  svr.Get("/v1/models", [&engine_ref](const httplib::Request&, httplib::Response& res) {
    res.set_content(BuildModelsResponse(engine_ref.ModelId(), NowUnix()).dump(), "application/json");
  });

  svr.Post("/v1/chat/completions", [&engine_ref](const httplib::Request& httpreq,
                                                   httplib::Response& res) {
    try {
      const nlohmann::json body = ParseJsonBody(httpreq);
      ChatCompletionRequest req = ParseChatCompletionRequest(body, engine_ref.SamplingDefaults());
      const std::string model_id = req.model.empty() ? engine_ref.ModelId() : req.model;
      const int64_t max_tokens = req.max_tokens.value_or(engine_ref.MaxTokensDefault());
      const std::string id = GenerateRequestId("chatcmpl-");
      const int64_t created = NowUnix();

      auto pending = std::make_shared<PendingRequest>();
      pending->kind = RequestKind::kChat;
      pending->request_id = id;
      pending->messages = req.messages;
      pending->chat_template_kwargs = req.chat_template_kwargs;
      pending->tools = req.tools;
      pending->sampling = req.sampling;
      pending->max_tokens = max_tokens;
      pending->stop = req.stop;

      if (req.stream) {
        auto sink = std::make_shared<StreamingSink>(StreamingSink::Kind::kChat, id, model_id, created,
                                                     req.stream_options_include_usage);
        pending->sink = sink;
        if (!engine_ref.Submit(pending)) {
          RespondError(res, 429, "rate_limit_error", "server request queue is full, try again shortly");
          return;
        }
        ServeStream(res, sink);
      } else {
        auto sink = std::make_shared<BufferingSink>();
        pending->sink = sink;
        if (!engine_ref.Submit(pending)) {
          RespondError(res, 429, "rate_limit_error", "server request queue is full, try again shortly");
          return;
        }
        sink->Wait();
        if (sink->errored) {
          RespondError(res, sink->error_status, ErrorTypeForStatus(sink->error_status),
                       sink->error_message);
          return;
        }
        UsageStats usage{sink->prompt_tokens, sink->completion_tokens};
        // `content` is JSON null (not "") only when the whole turn was a pure tool call with no
        // accompanying prose -- OpenAI's own convention (openai_types.h's tool_calls-carrying
        // BuildChatCompletionResponse overload doc comment).
        const std::optional<std::string> content =
            (!sink->tool_calls.empty() && sink->text.empty()) ? std::nullopt
                                                                : std::optional<std::string>(sink->text);
        res.set_content(BuildChatCompletionResponse(id, model_id, created, content, sink->tool_calls,
                                                     sink->finish_reason, usage, sink->timings)
                             .dump(),
                         "application/json");
      }
    } catch (const ApiError& err) {
      RespondError(res, err);
    } catch (const std::exception& e) {
      RespondError(res, 500, "server_error", e.what());
    }
  });

  svr.Post("/v1/completions", [&engine_ref](const httplib::Request& httpreq, httplib::Response& res) {
    try {
      const nlohmann::json body = ParseJsonBody(httpreq);
      CompletionRequest req = ParseCompletionRequest(body, engine_ref.SamplingDefaults());
      const std::string model_id = req.model.empty() ? engine_ref.ModelId() : req.model;
      const int64_t max_tokens = req.max_tokens.value_or(engine_ref.MaxTokensDefault());
      const std::string id = GenerateRequestId("cmpl-");
      const int64_t created = NowUnix();

      auto pending = std::make_shared<PendingRequest>();
      pending->kind = RequestKind::kCompletion;
      pending->request_id = id;
      pending->raw_prompt = req.prompt;
      pending->sampling = req.sampling;
      pending->max_tokens = max_tokens;
      pending->stop = req.stop;

      if (req.stream) {
        auto sink = std::make_shared<StreamingSink>(StreamingSink::Kind::kCompletion, id, model_id,
                                                     created, req.stream_options_include_usage);
        pending->sink = sink;
        if (!engine_ref.Submit(pending)) {
          RespondError(res, 429, "rate_limit_error", "server request queue is full, try again shortly");
          return;
        }
        ServeStream(res, sink);
      } else {
        auto sink = std::make_shared<BufferingSink>();
        pending->sink = sink;
        if (!engine_ref.Submit(pending)) {
          RespondError(res, 429, "rate_limit_error", "server request queue is full, try again shortly");
          return;
        }
        sink->Wait();
        if (sink->errored) {
          RespondError(res, sink->error_status, ErrorTypeForStatus(sink->error_status),
                       sink->error_message);
          return;
        }
        UsageStats usage{sink->prompt_tokens, sink->completion_tokens};
        res.set_content(
            BuildCompletionResponse(id, model_id, created, sink->text, sink->finish_reason, usage,
                                    sink->timings)
                .dump(),
            "application/json");
      }
    } catch (const ApiError& err) {
      RespondError(res, err);
    } catch (const std::exception& e) {
      RespondError(res, 500, "server_error", e.what());
    }
  });
}

HttpServer::~HttpServer() = default;

bool HttpServer::Listen(const std::string& host, int port) { return impl_->svr.listen(host, port); }

void HttpServer::Stop() { impl_->svr.stop(); }

}  // namespace r4dx::server
