#include "engine.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <optional>
#include <random>
#include <stdexcept>

#include "mtp_round.hpp"
#include "r4dx/kernels/sampler.hpp"

namespace r4dx::server {

namespace {

using Clock = std::chrono::steady_clock;
double Seconds(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double>(b - a).count();
}

void LogLine(const std::string& log_level, const char* level, const std::string& msg) {
  // Simple ordinal gate: debug < info < warn < error. Every request's required one-line summary
  // is logged at "info" so it always shows unless log_level=="error"/"warn" asks for quiet.
  auto rank = [](const std::string& l) -> int {
    if (l == "debug") return 0;
    if (l == "info") return 1;
    if (l == "warn") return 2;
    return 3;  // "error"
  };
  auto msg_rank = [](const char* l) -> int {
    if (std::string(l) == "debug") return 0;
    if (std::string(l) == "info") return 1;
    if (std::string(l) == "warn") return 2;
    return 3;
  };
  if (msg_rank(level) < rank(log_level)) return;
  std::fprintf(stderr, "[r4dx-server][%s] %s\n", level, msg.c_str());
}

// Finds the earliest position at or after `search_from` in `text` where any of `stops` occurs.
// Returns npos if none match. Used to trim a --stop string's own text out of the emitted output
// (OpenAI/vLLM semantics: the stop string terminates generation but is not itself echoed back).
size_t FindEarliestStop(const std::string& text, const std::vector<std::string>& stops,
                        size_t search_from) {
  size_t best = std::string::npos;
  for (const auto& s : stops) {
    if (s.empty()) continue;
    const size_t pos = text.find(s, search_from);
    if (pos != std::string::npos && (best == std::string::npos || pos < best)) best = pos;
  }
  return best;
}

}  // namespace

// Decodes `tok` into text, applies --stop trimming (identical semantics/lookback window to
// src/cli/main.cpp's own inline copy of this logic -- see docs/server.md's "--stop trimming
// across token boundaries"), and pushes whatever survives to req.sink->OnToken (so every accepted
// token -- plain-decode or MTP-round -- streams to the client as soon as it is committed, per this
// stage's task point 2). Returns true iff a --stop string matched (the caller should stop
// generating and report finish_reason="stop").
bool Engine::EmitToken(PendingRequest& req, r4dx::Tokenizer::StreamDecoder& decoder,
                       std::string& accumulated, int32_t tok) {
  const std::string piece = decoder.push(tok);
  if (piece.empty()) return false;
  const size_t already_emitted = accumulated.size();
  accumulated += piece;
  const size_t lookback = already_emitted > 0 ? std::min<size_t>(already_emitted, 63) : 0;
  const size_t match = FindEarliestStop(accumulated, req.stop, already_emitted - lookback);
  if (match != std::string::npos) {
    const size_t emit_len = match > already_emitted ? match - already_emitted : 0;
    const std::string trimmed = piece.substr(0, emit_len);
    if (!trimmed.empty()) req.sink->OnToken(trimmed);
    return true;
  }
  req.sink->OnToken(piece);
  return false;
}

Engine::Engine(EngineOptions opts) : opts_(std::move(opts)), queue_(static_cast<size_t>(opts_.max_queue)) {}

Engine::~Engine() { Shutdown(); }

void Engine::LoadAndStart() {
  r4dx::Tokenizer::Options tok_options;
  // Same acknowledged gap as src/cli/main.cpp: Qwen3.8-27B's tokenizer.json declares an NFC
  // normalizer this tokenizer doesn't implement (tokenizer.h's KNOWN GAP comment) -- without this
  // flag, from_directory() refuses to load this exact checkpoint's tokenizer.json at all.
  tok_options.allow_unimplemented_normalizer = true;
  tok_ = std::make_unique<r4dx::Tokenizer>(
      r4dx::Tokenizer::from_directory(opts_.tokenizer_dir, tok_options));
  tmpl_ = std::make_unique<r4dx::ChatTemplate>(r4dx::ChatTemplate::from_directory(opts_.tokenizer_dir));

  model_ = std::make_unique<r4dx::model::Model>(r4dx::model::Model::Load(opts_.model_opts));
  model_id_ = model_->GetContainer().ModelId();

  worker_ = std::thread(&Engine::WorkerLoop, this);
}

bool Engine::Submit(std::shared_ptr<PendingRequest> req) { return queue_.TryPush(std::move(req)); }

void Engine::Shutdown() {
  const bool already_stopping = stop_.exchange(true);
  if (already_stopping) {
    if (worker_.joinable()) worker_.join();
    return;
  }
  queue_.Close();
  if (worker_.joinable()) worker_.join();
}

void Engine::WorkerLoop() {
  while (true) {
    std::optional<std::shared_ptr<PendingRequest>> item = queue_.Pop();
    if (!item) break;  // queue closed and drained -- Shutdown() was called
    RunRequest(**item);
  }
}

void Engine::RunRequest(PendingRequest& req) {
  try {
    std::vector<r4dx::TokenId> full_tokens;
    if (req.kind == RequestKind::kChat) {
      r4dx::ChatJson messages = r4dx::ChatJson::array();
      for (const auto& m : req.messages) {
        messages.push_back({{"role", m.role}, {"content", m.content}});
      }
      r4dx::ChatJson extra_context = req.chat_template_kwargs;
      if (!extra_context.contains("enable_thinking")) {
        extra_context["enable_thinking"] = opts_.default_thinking;
      }
      const std::string rendered = tmpl_->render(messages, /*add_generation_prompt=*/true,
                                                   req.tools, extra_context);
      // parse_special=true: required so the template's own <|im_start|>/<|im_end|> control
      // sequences become their token ids -- see tokenizer.h's encode() CAUTION note (message
      // bodies are spliced in verbatim, same caveat this server inherits from the chat template
      // and does not sandbox, exactly like src/cli/main.cpp).
      full_tokens = tok_->encode(rendered, /*parse_special=*/true);
    } else {
      // Raw prompt: never trust literal special-token surface forms in caller-supplied text.
      full_tokens = tok_->encode(req.raw_prompt, /*parse_special=*/false);
    }

    if (full_tokens.empty()) {
      req.sink->OnError(400, "prompt rendered to zero tokens");
      return;
    }
    if (static_cast<int64_t>(full_tokens.size()) > MaxCtx()) {
      req.sink->OnError(400, "prompt (" + std::to_string(full_tokens.size()) +
                                  " tokens) exceeds --max-ctx (" + std::to_string(MaxCtx()) + ")");
      return;
    }

    // Prefix reuse (task point 2): continue from the existing KV/GDN state if `full_tokens`
    // extends what's already fed; otherwise Model::Reset() (re-zero GDN/KV/MTP state in place,
    // milliseconds -- NOT a full Model::Load(), see model.h's Reset() doc comment and this
    // stage's own measurement below) and re-prefill from scratch.
    std::vector<int32_t> new_tokens_i32;
    double reset_ms = -1.0;  // -1 == no reset happened this request (prefix extended)
    std::optional<std::vector<int32_t>> tail = prefix_.Extend(full_tokens);
    if (tail) {
      new_tokens_i32 = std::move(*tail);
    } else {
      const auto r0 = Clock::now();
      model_->Reset();
      const auto r1 = Clock::now();
      reset_ms = Seconds(r0, r1) * 1000.0;
      prefix_.Clear();
      new_tokens_i32.assign(full_tokens.begin(), full_tokens.end());
    }

    int64_t max_tokens = req.max_tokens;
    const int64_t ctx_budget = MaxCtx() - static_cast<int64_t>(full_tokens.size());
    if (max_tokens > ctx_budget) max_tokens = std::max<int64_t>(0, ctx_budget);

    req.sink->OnStart(static_cast<int64_t>(full_tokens.size()));

    const auto t0 = Clock::now();
    std::vector<float> logits = model_->Prefill(new_tokens_i32);
    const auto t1 = Clock::now();
    const double prefill_seconds = Seconds(t0, t1);

    r4dx::kernels::SampleParams sp;
    sp.temperature = req.sampling.temperature;
    sp.top_k = req.sampling.top_k;
    sp.top_p = req.sampling.top_p;
    sp.min_p = req.sampling.min_p;
    const uint64_t seed_used =
        req.sampling.has_seed ? req.sampling.seed : std::random_device{}();
    std::mt19937_64 rng = r4dx::kernels::MakeRng(seed_used);

    auto decoder = tok_->make_stream_decoder(/*skip_special_tokens=*/true);
    const auto& eos_ids = tok_->eos_ids();
    auto is_eos = [&](int32_t id) {
      for (int32_t e : eos_ids) {
        if (e == id) return true;
      }
      return false;
    };

    // generated_tokens: tokens actually shown to the client (usage.completion_tokens).
    // committed_tokens: tokens actually fed into model_'s real KV/GDN state this turn -- see
    // prefix_state.h's file comment for why these two can differ (MTP round stopping mid-vector).
    std::vector<int32_t> generated_tokens;
    std::vector<int32_t> committed_tokens;
    std::string accumulated;
    std::string finish_reason = "length";

    // Greedy MTP (task point 2): only a temperature<=0 request on a Model actually Load()'d with
    // mtp_draft_k>0 takes the speculative path -- everything else (non-greedy, or MTP disabled at
    // startup) is plain decode, byte-for-byte the pre-existing loop below. Mirrors src/cli/
    // main.cpp's RunTurn `greedy && args.mtp > 0` gate exactly (MTP is greedy-only, docs/mtp.md --
    // "probabilistic acceptance later" is still future work, per this stage's task).
    const bool use_mtp = model_->MtpEnabled() && req.sampling.temperature <= 0.0f;
    int64_t mtp_rounds = 0, mtp_drafted = 0, mtp_accepted = 0;

    const auto d0 = Clock::now();
    if (use_mtp) {
      const int64_t draft_k = opts_.model_opts.mtp_draft_k;
      int32_t next = r4dx::kernels::Argmax(logits.data(), static_cast<int64_t>(logits.size()));
      bool stopped = false;
      if (is_eos(next)) {
        finish_reason = "stop";
        stopped = true;
      } else {
        generated_tokens.push_back(next);
        if (EmitToken(req, decoder, accumulated, next)) {
          finish_reason = "stop";
          stopped = true;
        }
      }
      while (!stopped && static_cast<int64_t>(generated_tokens.size()) < max_tokens) {
        if (req.sink->IsCancelled()) {
          finish_reason = "cancelled";
          break;
        }
        const std::vector<int32_t> round = model_->DecodeStepMtpGreedy(next, draft_k);
        ++mtp_rounds;
        mtp_drafted += draft_k;
        mtp_accepted += static_cast<int64_t>(round.size()) - 1;  // last token is never a draft
        // `next` (this call's own token_id argument) is now committed -- see prefix_state.h/
        // model.h's Reset() comment. ProcessMtpRound (src/model/mtp_round.hpp) computes the rest
        // of `round` that is ALSO unconditionally committed by the atomic call above, independent
        // of how far the display/EmitToken loop below gets (max-tokens, an EOS candidate that
        // isn't round's own last element, or a --stop string match mid-round all end that loop
        // early) -- see that header's file comment for why this must be computed up front rather
        // than incrementally inside a loop that can `break` (docs/mtp.md's "mid-round" gap).
        const r4dx::model::MtpRoundResult outcome = r4dx::model::ProcessMtpRound(
            round, is_eos, max_tokens - static_cast<int64_t>(generated_tokens.size()));
        committed_tokens.push_back(next);
        committed_tokens.insert(committed_tokens.end(), outcome.committed.begin(),
                                 outcome.committed.end());
        for (int32_t tok : outcome.displayed) {
          generated_tokens.push_back(tok);
          next = tok;
          if (EmitToken(req, decoder, accumulated, tok)) {
            finish_reason = "stop";
            stopped = true;
            break;
          }
        }
        if (!stopped) {
          if (outcome.hit_eos) {
            finish_reason = "stop";
            stopped = true;
          } else if (outcome.hit_max_tokens) {
            stopped = true;
          }
        }
      }
    } else {
      for (int64_t step = 0; step < max_tokens; ++step) {
        if (req.sink->IsCancelled()) {
          finish_reason = "cancelled";
          break;
        }
        const int32_t next =
            r4dx::kernels::Sample(logits.data(), static_cast<int64_t>(logits.size()), sp, rng);
        if (is_eos(next)) {
          finish_reason = "stop";
          break;
        }
        generated_tokens.push_back(next);
        const bool stop_hit = EmitToken(req, decoder, accumulated, next);
        // Feed `next` into the model regardless of stop_hit, so committed_tokens below accurately
        // reflects what the KV/GDN state actually holds (see prefix_state.h's file comment) -- the
        // token is real generated content, only its stop-marker tail text is withheld from the
        // client.
        logits = model_->DecodeStep(next);
        committed_tokens.push_back(next);
        if (stop_hit) {
          finish_reason = "stop";
          break;
        }
      }
    }
    const std::string tail_text = decoder.flush();
    if (!tail_text.empty()) req.sink->OnToken(tail_text);
    const auto d1 = Clock::now();
    const double decode_seconds = Seconds(d0, d1);

    prefix_.Commit(full_tokens, committed_tokens);

    req.sink->OnDone(finish_reason, static_cast<int64_t>(generated_tokens.size()));

    const double prefill_tps = prefill_seconds > 0 ? new_tokens_i32.size() / prefill_seconds : 0.0;
    const double decode_tps =
        decode_seconds > 0 ? static_cast<double>(generated_tokens.size()) / decode_seconds : 0.0;
    char buf[448];
    int n = std::snprintf(
        buf, sizeof(buf),
        "request %s: prompt=%lld new=%lld generated=%lld finish=%s prefill=%.2f tok/s "
        "decode=%.2f tok/s",
        req.request_id.c_str(), static_cast<long long>(full_tokens.size()),
        static_cast<long long>(new_tokens_i32.size()),
        static_cast<long long>(generated_tokens.size()), finish_reason.c_str(), prefill_tps,
        decode_tps);
    if (reset_ms >= 0.0 && n > 0 && n < static_cast<int>(sizeof(buf))) {
      n += std::snprintf(buf + n, sizeof(buf) - static_cast<size_t>(n), " reset=%.2fms", reset_ms);
    }
    if (use_mtp && mtp_rounds > 0 && n > 0 && n < static_cast<int>(sizeof(buf))) {
      const double accept_rate =
          mtp_drafted > 0 ? 100.0 * static_cast<double>(mtp_accepted) / static_cast<double>(mtp_drafted)
                          : 0.0;
      std::snprintf(buf + n, sizeof(buf) - static_cast<size_t>(n),
                    " mtp: draft_k=%lld rounds=%lld drafted=%lld accepted=%lld (%.1f%% accept, "
                    "%.2f tok/round)",
                    static_cast<long long>(opts_.model_opts.mtp_draft_k),
                    static_cast<long long>(mtp_rounds), static_cast<long long>(mtp_drafted),
                    static_cast<long long>(mtp_accepted), accept_rate,
                    static_cast<double>(generated_tokens.size()) / static_cast<double>(mtp_rounds));
    }
    LogLine(opts_.log_level, "info", buf);
  } catch (const std::exception& e) {
    // Invalidate the prefix-reuse fast path (review finding, 2026-09-19; unchanged by this stage):
    // if Model::Prefill/DecodeStep*/Reset threw partway through, the model may have already
    // consumed some tokens that prefix_ (only Commit()'d on the success path above) does not
    // record. Leaving it stale would let the NEXT request's Extend() take the "matches, feed only
    // the tail" branch against a model whose real state has silently diverged from what prefix_
    // claims -- Invalidate() forces the next request down the full-reset path instead.
    prefix_.Invalidate();
    LogLine(opts_.log_level, "error",
            "request " + req.request_id + ": " + std::string(e.what()));
    req.sink->OnError(500, e.what());
  }
}

}  // namespace r4dx::server
