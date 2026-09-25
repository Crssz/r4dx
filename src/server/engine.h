// r4dx::server::Engine -- owns the single model/Tokenizer/ChatTemplate instance and the one worker
// thread that ever calls into it (task point 2: "single-model, single-GPU, one request at a time: a
// request queue with a worker thread that owns the Model"). HTTP handler threads (http_server.cpp)
// only ever call Submit()/ModelId()/etc -- never touch the model directly, so there is exactly one
// call path into the GPU.
//
// The model is an r4dx::model::TextModel (docs/tp.md 2.8): at `--tp 1` a LocalTextModel -- one
// Model on HIP device 1, every call a one-line forward, byte for byte the pre-TP server -- and at
// `--tp 2` a TpModel, whose rank threads own one device each (docs/tp.md 2.1). Under TP neither the
// worker thread nor any HTTP thread makes a HIP call; a failed request leaves the group in
// kNeedsRecovery and the next request's Reset() recovers it (docs/tp.md 2.4, 8.4).
//
// This header (unlike request_queue.h/openai_types.h/response_sink.h) pulls in r4dx::model::Model
// and is therefore HIP-dependent -- it is not linked into tests/server's CPU-only unit tests, only
// into the r4dx-server executable (see src/server/CMakeLists.txt) and exercised end to end by
// tools/server/smoke.ps1 instead. The one exception is tests/server/test_engine_recovery.cpp, which
// links what r4dx-server links but runs RunRequest against a CPU fake model (EngineOptions::
// model_loader), with no GPU work.
#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "chat_template.h"
#include "model.h"  // ModelOptions
#include "openai_types.h"
#include "preprocess.h"  // src/vision: ImageProcessorConfig (docs/vision.md "Large images")
#include "prefix_state.h"
#include "request_queue.h"
#include "response_sink.h"
#include "text_model.h"  // r4dx::model::TextModel / TpOptions (docs/tp.md 2.8)
#include "tokenizer.h"

namespace r4dx::model {
class TpModel;
}

namespace r4dx::server {

struct EngineOptions {
  r4dx::model::ModelOptions model_opts;
  // The --tp* flags (docs/tp.md 9.1), filled by main.cpp exactly as r4dx-cli fills its own:
  // world 1 (the default) is the single-device server; world 2 loads a TpModel.
  r4dx::model::TpOptions tp;
  // How LoadAndStart builds the model. Empty (always, in r4dx-server) = r4dx::model::LoadTextModel.
  // tests/server/test_engine_recovery.cpp substitutes a CPU fake to drive RunRequest's error path.
  std::function<std::unique_ptr<r4dx::model::TextModel>(const r4dx::model::ModelOptions&,
                                                        const r4dx::model::TpOptions&)>
      model_loader;
  std::string tokenizer_dir;
  int64_t max_tokens_default = 128;
  int max_queue = 16;
  SamplingParams sampling_defaults;
  bool default_thinking = false;
  std::string log_level = "info";
  // DFlash2 self-speculative decode (docs/dflash2.md, stage S3): model_opts.dflash_draft_k already
  // carries the per-round `k` (same dual sizing/per-call-cap role as model_opts.mtp_draft_k has for
  // MTP -- see src/cli/cli_args.h's own comment on --mtp/--dflash-k). p_min/n_min are NOT sizing
  // knobs (Model::Load/ModelOptions doesn't need them), only per-round DraftRound arguments, so they
  // live here instead, mirroring how sampling_defaults above carries per-request-but-not-per-Load
  // values.
  float dflash_p_min = 0.0f;
  int64_t dflash_n_min = 0;
  // Per-image pixel cap (docs/vision.md "Large images", --image-max-pixels). Like p_min/n_min
  // above this is not a Model::Load sizing knob -- it is a PREPROCESSING parameter, applied when a
  // request's image is decoded, so it lives here rather than in model_opts. 0 means the
  // checkpoint's own preprocessor_config.json ceiling; r4dx::vision::MakeImageProcessorConfig
  // turns it into the ImageProcessorConfig the decode path uses.
  int64_t image_max_pixels = 1048576;
};

enum class RequestKind { kChat, kCompletion };

struct PendingRequest {
  RequestKind kind = RequestKind::kChat;
  std::string request_id;

  // kChat
  std::vector<ChatMessage> messages;
  nlohmann::json chat_template_kwargs = nlohmann::json::object();
  // Set by http_server.cpp from ParseChatCompletionRequest -- the reduced answer to "did this
  // request ask for thinking, at what effort, and does it want the thought back" (openai_types.h's
  // ThinkingControls). Unused for kCompletion (no chat template, nothing to split).
  ThinkingControls thinking;
  nlohmann::json tools = nlohmann::json::array();

  // kCompletion
  std::string raw_prompt;

  SamplingParams sampling;
  int64_t max_tokens = 0;
  std::vector<std::string> stop;
  // Set by http_server.cpp from the request body's own `stream` field (openai_types.h). RunRequest
  // does not otherwise know which ResponseSink subclass `sink` is, and the per-request stderr log
  // line (stage S3, docs/server.md) reports it.
  bool stream = false;

  std::shared_ptr<ResponseSink> sink;
};

class Engine {
 public:
  explicit Engine(EngineOptions opts);
  ~Engine();

  Engine(const Engine&) = delete;
  Engine& operator=(const Engine&) = delete;

  // Loads the tokenizer/chat-template/model and starts the worker thread. `--tp 1`: HIP device 1,
  // per the project's GPU rule -- this process does not call hipSetDevice itself, same as r4dx-cli;
  // the caller's environment must already have HIP_VISIBLE_DEVICES=1 set. `--tp 2`: the TpModel's
  // rank threads pick their devices (docs/tp.md 9.2; real mode needs HIP_VISIBLE_DEVICES unset).
  // Throws on failure -- call before the HTTP server starts listening, so a bad --model/
  // --tokenizer-dir fails fast instead of accepting connections it can never serve.
  void LoadAndStart();

  const std::string& ModelId() const { return model_id_; }
  int64_t MaxCtx() const { return opts_.model_opts.max_ctx; }
  int64_t MaxTokensDefault() const { return opts_.max_tokens_default; }
  const SamplingParams& SamplingDefaults() const { return opts_.sampling_defaults; }
  bool DefaultThinking() const { return opts_.default_thinking; }
  // The image preprocessing policy every image content part is decoded with (docs/vision.md
  // "Large images", --image-max-pixels) -- read by http_server.cpp so ParseChatCompletionRequest
  // can preprocess an image at PARSE time (openai_types.cpp), before the request even reaches this
  // Engine's worker thread.
  const r4dx::vision::ImageProcessorConfig& ImagePreprocessing() const { return image_preproc_; }
  // True iff the loaded container's vision tower is actually resident (docs/vision.md "Load
  // policy") -- read by http_server.cpp for /v1/models' `architecture.input_modalities`/
  // `capabilities` and by RunRequest itself for the "this model/container has no vision tower"
  // 400. Safe to call from any thread once LoadAndStart() has returned: model_ is never
  // reassigned to a different model after that (Reset() reuses the same object in place), and
  // TextModel::HasVision is a value cached at load under TP (docs/tp.md 2.4's host-only table).
  bool HasVision() const { return model_ && model_->HasVision(); }
  // True once a `--tp 2` request has left the TP group kFatal (its recovery failed, or a rank got
  // stuck inside a HIP call): every later request answers 500 until the process is restarted, so
  // http_server.cpp's /health reports it (503) instead of "ok" (docs/tp.md 2.4, R13; Appendix B N80).
  // Set only by the worker thread; any thread may read it. Always false at `--tp 1`.
  bool TpFatal() const { return tp_fatal_.load(std::memory_order_acquire); }

  // Enqueues `req` for the worker thread. Returns false (queue already at --max-queue) if the
  // caller should answer 429 instead.
  bool Submit(std::shared_ptr<PendingRequest> req);

  // Stops accepting new work, closes the queue, and joins the worker thread. Safe to call more
  // than once.
  void Shutdown();

 private:
  void WorkerLoop();

  // Runs one request end to end against tok_/tmpl_/model_, feeding req->sink. Never throws --
  // every failure path reports through req->sink->OnError instead, since this runs on the single
  // worker thread and an uncaught exception here would take the whole server down.
  void RunRequest(PendingRequest& req);

  // Shared stop-string-aware token emission (engine.cpp) used by both the plain-decode and MTP
  // generation loops in RunRequest -- see that function's definition for the full contract.
  // `forward`: where the surviving (stop-trimmed) text goes. RunRequest builds exactly one such
  // router per request -- a straight `sink->OnToken` passthrough for a request with no `tools`, the
  // ToolStreamGate-controlled one (tool_stream_gate.h, docs/server.md's "Tool calls" streaming
  // decision) for a streaming request that offers them, and a drop for a NON-streaming request that
  // offers them (nothing is delivered live there at all; the whole generation is re-derived from
  // `accumulated` after the loop). `accumulated`/the stop-string check itself are unaffected by
  // which router is in play, so every decode path stays byte-for-byte identical. `stop_match_pos`
  // (review finding, 2026-09-20): when a --stop string matches, the position within `accumulated`
  // where it begins is written here (if non-null) -- `accumulated` itself still gets the FULL
  // decoded piece appended (including the stop text), only what's *streamed* to the client is
  // trimmed, so a caller that re-derives its response from `accumulated` after the loop ends (the
  // tool_mode block) must re-trim at this boundary itself or it echoes the stop text back.
  // `min_stop_search_from` (task item 5, "reasoning_content" -- "stop sequences apply to the
  // ANSWER part only"): the stop-string search below never looks before this position in
  // `accumulated`. RunRequest passes `std::string::npos` for as long as a thinking-enabled
  // request's "</think>" close tag has not yet been seen (npos always exceeds
  // `accumulated.size()`, so the search collapses to nothing this call -- see EmitToken's own
  // definition), then the tag's own end position once it has, so a stop string that happens to
  // appear inside the model's own chain-of-thought can never truncate generation before the answer
  // even starts. Defaults to 0 (no restriction, today's behavior) so a thinking-off request is
  // byte-for-byte unaffected.
  static bool EmitToken(PendingRequest& req, r4dx::Tokenizer::StreamDecoder& decoder,
                         std::string& accumulated, int32_t tok,
                         const std::function<void(const std::string&)>& forward,
                         size_t* stop_match_pos = nullptr, size_t min_stop_search_from = 0);

  EngineOptions opts_;
  std::string model_id_;

  std::unique_ptr<r4dx::Tokenizer> tok_;
  std::unique_ptr<r4dx::ChatTemplate> tmpl_;
  std::unique_ptr<r4dx::model::TextModel> model_;
  // model_ itself when it is a TpModel (`--tp 2`), else null: the TP diagnostics (the group state
  // around a recovery, the `--log-level debug` stats line) are not part of TextModel (docs/tp.md
  // 2.8), the same reason r4dx-cli reaches the facade through a dynamic_cast.
  r4dx::model::TpModel* tp_model_ = nullptr;

  // Image preprocessing policy, built once in LoadAndStart from EngineOptions::image_max_pixels
  // (docs/vision.md "Large images"). Read back out through ImagePreprocessing() above, which is
  // how http_server.cpp's chat route decodes an `image_url` content part at PARSE time.
  r4dx::vision::ImageProcessorConfig image_preproc_;

  // Prefix-reuse bookkeeping: every token already committed to model_'s KV/GDN state, in the
  // order fed. Mirrors src/cli/main.cpp's --chat loop's `fed_tokens` exactly (same mechanism, per
  // task point 2's "prefix reuse: if the new request's token prefix equals the previous session's
  // fed tokens, continue from the cache ... otherwise reset and re-prefill") -- this server has no
  // separate notion of "session" beyond "the token sequence the last request left the model in",
  // which is exactly what a client resending a growing `messages` array (the normal OpenAI chat
  // client pattern: resend the whole conversation each turn) produces. See prefix_state.h for the
  // MTP-aware Commit() contract (docs/mtp.md's "mid-round" gap, closed here).
  PrefixState prefix_;

  BoundedQueue<std::shared_ptr<PendingRequest>> queue_;
  std::thread worker_;
  std::atomic<bool> stop_{false};
  // TpFatal(): the worker thread's copy of `tp_model_->GetState() == kFatal`, taken after every
  // failed request -- TpModel's own state is facade-thread-only, so the HTTP threads read this.
  std::atomic<bool> tp_fatal_{false};
};

}  // namespace r4dx::server
