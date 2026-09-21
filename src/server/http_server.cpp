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
    res.set_content(BuildModelsResponse(engine_ref.ModelId(), NowUnix(), engine_ref.MaxCtx(),
                                         engine_ref.DefaultThinking())
                         .dump(),
                     "application/json");
  });

  // GET /v1/models/{id} (task item 1): this server ever loads exactly one model, so any id other
  // than the loaded container's own ModelId() is a clean 404 with the standard error JSON, same
  // shape ApiError/ErrorBody already produce for every other error path.
  //
  // A REGEX pattern (`(.+)`), not httplib's `:id` path-param shorthand, on purpose: `:id` compiles
  // to a single-path-segment matcher (`PathParamsMatcher`, no slashes allowed), but this
  // checkpoint's own `model_id` IS `"Qwen/Qwen3.8-27B"` (`__metadata__.model_id`,
  // docs/container-format.md) -- a real HuggingFace-style "org/repo" id containing a literal "/".
  // `:id` would 404 on the container's own real id (caught end to end by `tools/server/smoke.ps1`,
  // not by any CPU-only unit test, since none of them drive real httplib routing). `(.+)` greedily
  // captures everything after the prefix, slashes included.
  svr.Get(R"(/v1/models/(.+))", [&engine_ref](const httplib::Request& httpreq, httplib::Response& res) {
    const std::string requested_id = httpreq.matches[1];
    if (requested_id != engine_ref.ModelId()) {
      RespondError(res, 404, "invalid_request_error",
                   "model '" + requested_id + "' not found (this server has loaded '" +
                       engine_ref.ModelId() + "')");
      return;
    }
    res.set_content(BuildModelEntryJson(engine_ref.ModelId(), NowUnix(), engine_ref.MaxCtx(),
                                         engine_ref.DefaultThinking())
                         .dump(),
                     "application/json");
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
      // The RESOLVED enable_thinking value (task item 5, "reasoning_content"), computed here --
      // before the request even reaches the worker thread -- with the exact same formula
      // Engine::RunRequest uses (ResolveEnableThinking, openai_types.h) so the two independently-
      // made calls can never drift apart. Decides whether the sink runs its reasoning/content
      // splitter at all (task item 5c: a thinking-off request's sink behavior must stay byte-for-
      // byte identical to before this feature existed). `emit_reasoning` is the separate
      // "think, but do not return the thought" question (reasoning.exclude / include_reasoning,
      // docs/server.md's "Thinking controls").
      const bool enable_thinking = ResolveEnableThinking(req.thinking, engine_ref.DefaultThinking());
      const bool emit_reasoning = enable_thinking && req.thinking.include_reasoning;

      auto pending = std::make_shared<PendingRequest>();
      pending->kind = RequestKind::kChat;
      pending->request_id = id;
      pending->messages = req.messages;
      pending->chat_template_kwargs = req.chat_template_kwargs;
      pending->thinking = req.thinking;
      pending->tools = req.tools;
      pending->sampling = req.sampling;
      pending->max_tokens = max_tokens;
      pending->stop = req.stop;
      pending->stream = req.stream;

      if (req.stream) {
        auto sink = std::make_shared<StreamingSink>(StreamingSink::Kind::kChat, id, model_id, created,
                                                     req.stream_options_include_usage, enable_thinking,
                                                     emit_reasoning);
        pending->sink = sink;
        if (!engine_ref.Submit(pending)) {
          RespondError(res, 429, "rate_limit_error", "server request queue is full, try again shortly");
          return;
        }
        ServeStream(res, sink);
      } else {
        auto sink = std::make_shared<BufferingSink>(enable_thinking, emit_reasoning);
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
        // Reported even when the TEXT was excluded: the tokens were really spent, and a client
        // that asked not to see the thought still bills/accounts for it.
        if (enable_thinking) usage.reasoning_tokens = sink->reasoning_tokens;
        // `content` is JSON null (not "") only when the whole turn was a pure tool call with no
        // accompanying prose -- OpenAI's own convention (openai_types.h's tool_calls-carrying
        // BuildChatCompletionResponse overload doc comment).
        const std::optional<std::string> content =
            (!sink->tool_calls.empty() && sink->text.empty()) ? std::nullopt
                                                                : std::optional<std::string>(sink->text);
        const std::optional<std::string> reasoning_content =
            emit_reasoning ? std::optional<std::string>(sink->reasoning_text) : std::nullopt;
        res.set_content(BuildChatCompletionResponse(id, model_id, created, content, sink->tool_calls,
                                                     sink->finish_reason, usage, sink->timings,
                                                     reasoning_content)
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
      pending->stream = req.stream;

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
