// r4dx::server::HttpServer -- wires cpp-httplib routes (/health, /v1/models,
// /v1/chat/completions, /v1/completions) to an Engine. HIP-dependent only transitively (through
// engine.h); this file itself only touches httplib + openai_types + response_sink.
#pragma once

#include <memory>
#include <string>

#include "engine.h"

namespace r4dx::server {

// The Content-Type of every JSON body (errors included) and of every SSE stream HttpServer writes.
// Every body is UTF-8 (nlohmann's dump() emits raw UTF-8, never \u escapes for non-ASCII), and both
// types say so explicitly. RFC 8259 defines no charset parameter for application/json and the SSE
// spec fixes text/event-stream to UTF-8, so a conforming client decodes UTF-8 either way -- but
// Windows PowerShell 5.1's Invoke-WebRequest/Invoke-RestMethod decode a body whose Content-Type
// names no charset as ISO-8859-1. Without the parameter, every non-ASCII character in a
// non-streaming `message.content` reached such a client as mojibake ("×" as "Ã" + U+0097), while
// the same text streamed and read as UTF-8 came through intact (docs/server.md's "Response
// shapes"). Declared here, not in http_server.cpp, so tests/server/test_http_server.cpp pins them
// even on a machine where it skips its live-server half for want of the tokenizer.
inline constexpr char kJsonContentType[] = "application/json; charset=utf-8";
inline constexpr char kSseContentType[] = "text/event-stream; charset=utf-8";

class HttpServer {
 public:
  explicit HttpServer(Engine& engine);
  ~HttpServer();

  HttpServer(const HttpServer&) = delete;
  HttpServer& operator=(const HttpServer&) = delete;

  // Blocks serving requests until Stop() is called (e.g. from a signal handler) or the listen
  // socket fails. Returns false on a listen failure (bad --host/--port).
  bool Listen(const std::string& host, int port);

  // Listen() in two halves, on a port the OS picks: BindToAnyPort returns it (-1 on failure),
  // then ListenAfterBind serves on it, blocking exactly like Listen. For a caller that cannot
  // safely pick a fixed port -- tests/server/test_http_server.cpp, where two concurrent runs would
  // share one (httplib sets SO_REUSEADDR on Windows, so the second bind can succeed and steal
  // requests).
  int BindToAnyPort(const std::string& host);
  bool ListenAfterBind();

  void Stop();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace r4dx::server
