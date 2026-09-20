// r4dx-server: OpenAI-compatible chat/completions HTTP API (docs/architecture.md "src/server/").
// Single model, single GPU (HIP device 1, project rule -- this binary does not call hipSetDevice
// itself, matching r4dx-cli), one request processed at a time by Engine's worker thread; HTTP
// handler threads only ever enqueue work and read back a ResponseSink (see engine.h).
#include <csignal>
#include <cstdio>

#include "engine.h"
#include "http_server.h"
#include "model.h"
#include "server_args.h"

namespace {

r4dx::server::HttpServer* g_server = nullptr;

void HandleSignal(int) {
  // Async-signal-safety note: httplib::Server::stop() just closes the listening socket and sets a
  // flag checked by the accept loop -- documented-safe to call from a signal handler in the same
  // way every other httplib-based server relies on.
  if (g_server != nullptr) g_server->Stop();
}

}  // namespace

int main(int argc, char** argv) {
  r4dx::server::ServerArgs args;
  try {
    args = r4dx::server::ParseServerArgs(argc, argv);
  } catch (const r4dx::server::ServerUsageError& e) {
    std::fprintf(stderr, "%s\n%s\n", e.what(), r4dx::server::ServerUsageText(argv[0]).c_str());
    return 2;
  }

  r4dx::server::EngineOptions opts;
  opts.model_opts.container_path = args.model_path;
  try {
    opts.model_opts.layout = r4dx::model::LayoutFromName(args.layout);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 2;
  }
  opts.model_opts.max_ctx = args.max_ctx;
  opts.model_opts.layer_limit = args.layers;
  opts.model_opts.mtp_draft_k = args.mtp;
  opts.model_opts.mtp_head_layout = (args.mtp_head_layout == "bf16")
                                         ? std::optional<r4dx::model::Layout>(r4dx::model::Layout::kBf16)
                                         : std::nullopt;
  opts.model_opts.embed_device_resident = (args.embed_device_resident != "off");
  opts.model_opts.mtp_draft_reduced_vocab = (args.mtp_draft_head != "full");
  opts.tokenizer_dir = args.tokenizer_dir;
  opts.max_tokens_default = args.max_tokens_default;
  opts.max_queue = args.max_queue;
  opts.default_thinking = args.think;
  opts.log_level = args.log_level;
  opts.sampling_defaults.temperature = args.default_temperature;
  opts.sampling_defaults.top_p = args.default_top_p;
  opts.sampling_defaults.top_k = args.default_top_k;
  opts.sampling_defaults.min_p = args.default_min_p;

  r4dx::server::Engine engine(std::move(opts));
  try {
    std::fprintf(stderr, "[r4dx-server] loading model %s (layout=%s) ...\n", args.model_path.c_str(),
                 args.layout.c_str());
    engine.LoadAndStart();
    std::fprintf(stderr, "[r4dx-server] model loaded: %s\n", engine.ModelId().c_str());
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: failed to load model: %s\n", e.what());
    return 1;
  }

  r4dx::server::HttpServer server(engine);
  g_server = &server;
  std::signal(SIGINT, HandleSignal);
  std::signal(SIGTERM, HandleSignal);

  std::fprintf(stderr, "[r4dx-server] listening on %s:%d\n", args.host.c_str(), args.port);
  const bool ok = server.Listen(args.host, args.port);
  g_server = nullptr;
  if (!ok) {
    std::fprintf(stderr, "error: failed to listen on %s:%d\n", args.host.c_str(), args.port);
    return 1;
  }
  return 0;
}
