// r4dx::server::HttpServer -- wires cpp-httplib routes (/health, /v1/models,
// /v1/chat/completions, /v1/completions) to an Engine. HIP-dependent only transitively (through
// engine.h); this file itself only touches httplib + openai_types + response_sink.
#pragma once

#include <memory>
#include <string>

#include "engine.h"

namespace r4dx::server {

class HttpServer {
 public:
  explicit HttpServer(Engine& engine);
  ~HttpServer();

  HttpServer(const HttpServer&) = delete;
  HttpServer& operator=(const HttpServer&) = delete;

  // Blocks serving requests until Stop() is called (e.g. from a signal handler) or the listen
  // socket fails. Returns false on a listen failure (bad --host/--port).
  bool Listen(const std::string& host, int port);

  void Stop();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace r4dx::server
