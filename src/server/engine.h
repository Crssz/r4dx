// r4dx::server::Engine -- owns the single Model/Tokenizer/ChatTemplate instance on HIP device 1
// and the one worker thread that ever calls into it (task point 2: "single-model, single-GPU, one
// request at a time: a request queue with a worker thread that owns the Model"). HTTP handler
// threads (http_server.cpp) only ever call Submit()/ModelId()/etc -- never touch r4dx::model::
// Model directly, so there is exactly one call path into the GPU.
//
// This header (unlike request_queue.h/openai_types.h/response_sink.h) pulls in r4dx::model::Model
// and is therefore HIP-dependent -- it is not linked into tests/server's CPU-only unit tests, only
// into the r4dx-server executable (see src/server/CMakeLists.txt) and exercised end to end by
// tools/server/smoke.ps1 instead.
#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "chat_template.h"
#include "model.h"
#include "openai_types.h"
#include "prefix_state.h"
#include "request_queue.h"
#include "response_sink.h"
#include "tokenizer.h"

namespace r4dx::server {

struct EngineOptions {
  r4dx::model::ModelOptions model_opts;
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
};

enum class RequestKind { kChat, kCompletion };

struct PendingRequest {
  RequestKind kind = RequestKind::kChat;
  std::string request_id;

  // kChat
  std::vector<ChatMessage> messages;
  nlohmann::json chat_template_kwargs = nlohmann::json::object();
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

  // Loads the tokenizer/chat-template/model (HIP device 1, per the project's GPU rule -- this
  // process does not call hipSetDevice itself, same as r4dx-cli; the caller's environment must
  // already have HIP_VISIBLE_DEVICES=1 set) and starts the worker thread. Throws on failure --
  // call before the HTTP server starts listening, so a bad --model/--tokenizer-dir fails fast
  // instead of accepting connections it can never serve.
  void LoadAndStart();

  const std::string& ModelId() const { return model_id_; }
  int64_t MaxCtx() const { return opts_.model_opts.max_ctx; }
  int64_t MaxTokensDefault() const { return opts_.max_tokens_default; }
  const SamplingParams& SamplingDefaults() const { return opts_.sampling_defaults; }
  bool DefaultThinking() const { return opts_.default_thinking; }

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
  // `stream_to_client`: false when this request has `tools` (docs/server.md's "Tool calls"
  // streaming decision -- generation is buffered whole, not streamed live, whenever a tool call
  // could plausibly appear, so a client never sees a half-formed "<tool_call>" tag as content).
  // `accumulated`/the stop-string check itself are unaffected either way. `stop_match_pos`
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
                         std::string& accumulated, int32_t tok, bool stream_to_client = true,
                         size_t* stop_match_pos = nullptr, size_t min_stop_search_from = 0);

  EngineOptions opts_;
  std::string model_id_;

  std::unique_ptr<r4dx::Tokenizer> tok_;
  std::unique_ptr<r4dx::ChatTemplate> tmpl_;
  std::unique_ptr<r4dx::model::Model> model_;

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
};

}  // namespace r4dx::server
