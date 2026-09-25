// r4dx-server: OpenAI-compatible chat/completions HTTP API (docs/architecture.md "src/server/").
// Single model, single GPU (HIP device 1, project rule -- this binary does not call hipSetDevice
// itself, matching r4dx-cli), one request processed at a time by Engine's worker thread; HTTP
// handler threads only ever enqueue work and read back a ResponseSink (see engine.h). With
// `--tp 2` (docs/tp.md 9) the model is tensor parallel across two ranks (r4dx::model::TpModel):
// real mode puts one rank on each GPU and needs HIP_VISIBLE_DEVICES unset.
#include <csignal>
#include <cstdio>

#include "engine.h"
#include "http_server.h"
#include "model.h"
#include "server_args.h"
#include "text_model.h"  // r4dx::model::TpOptions (docs/tp.md 9.1)

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
  // docs/vision.md "Load policy" -- same three-way policy as r4dx-cli's --vision.
  opts.model_opts.vision = args.vision == "on"    ? r4dx::model::ModelOptions::VisionMode::kOn
                            : args.vision == "off" ? r4dx::model::ModelOptions::VisionMode::kOff
                                                    : r4dx::model::ModelOptions::VisionMode::kAuto;
  opts.model_opts.dflash_container = args.dflash;
  opts.model_opts.dflash_draft_k = args.dflash.empty() ? 0 : args.dflash_k;
  opts.dflash_p_min = args.dflash_p_min;
  opts.dflash_n_min = args.dflash_n_min;
  opts.image_max_pixels = args.image_max_pixels;
  opts.tokenizer_dir = args.tokenizer_dir;
  opts.max_tokens_default = args.max_tokens_default;
  opts.max_queue = args.max_queue;
  opts.default_thinking = args.think;
  opts.log_level = args.log_level;
  opts.sampling_defaults.temperature = args.default_temperature;
  opts.sampling_defaults.top_p = args.default_top_p;
  opts.sampling_defaults.top_k = args.default_top_k;
  opts.sampling_defaults.min_p = args.default_min_p;
  // Tensor parallel (docs/tp.md 9.1): filled exactly as src/cli/main.cpp fills its own TpOptions;
  // server_args.h already refused every --tp-* flag at --tp 1.
  opts.tp.world = args.tp;
  if (args.tp == 2) {
    opts.tp.mode = args.tp_mode == "emulate" ? r4dx::model::TpOptions::Mode::kEmulate
                   : args.tp_mode == "noop"  ? r4dx::model::TpOptions::Mode::kNoop
                                             : r4dx::model::TpOptions::Mode::kReal;
    opts.tp.devices = args.tp_devices;
    opts.tp.noop_rank = args.tp_rank;
    opts.tp.ar_timeout_ms = args.tp_ar_timeout_ms;
    opts.tp.ar_nb_small = args.tp_ar_nb;
    opts.tp.ar_nb_large = args.tp_ar_nb_large;
    if (args.tp_submit_layers >= 0) opts.tp.submit_layers = args.tp_submit_layers;  // -1: TpOptions' default
    if (args.tp_max_inflight >= 0) opts.tp.max_inflight_units = args.tp_max_inflight;
  }

  r4dx::server::Engine engine(std::move(opts));
  try {
    if (args.tp == 1) {
      std::fprintf(stderr, "[r4dx-server] loading model %s (layout=%s) ...\n", args.model_path.c_str(),
                   args.layout.c_str());
    } else {
      std::fprintf(stderr, "[r4dx-server] loading model %s (layout=%s, --tp 2 --tp-mode %s) ...\n",
                   args.model_path.c_str(), args.layout.c_str(), args.tp_mode.c_str());
    }
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
