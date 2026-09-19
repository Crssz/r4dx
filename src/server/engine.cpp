#include "engine.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <random>
#include <stdexcept>

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
    // extends what's already fed; otherwise reset (a fresh Model::Load, exactly like src/cli/
    // main.cpp's chat-prefix-mismatch fallback) and re-prefill from scratch.
    bool matches_prefix = full_tokens.size() >= fed_tokens_.size() &&
                           std::equal(fed_tokens_.begin(), fed_tokens_.end(), full_tokens.begin());
    std::vector<int32_t> new_tokens;
    if (matches_prefix) {
      new_tokens.assign(full_tokens.begin() + static_cast<ptrdiff_t>(fed_tokens_.size()),
                         full_tokens.end());
      if (new_tokens.empty()) {
        // Nothing new to feed (e.g. a byte-identical repeated request) -- Model::Prefill throws
        // on an empty vector, so degrade to a full re-prefill rather than special-casing "zero
        // new tokens" as its own code path.
        matches_prefix = false;
      }
    }
    if (!matches_prefix) {
      // Drop the OLD Model before constructing the replacement (review finding, 2026-09-19):
      // `model_ = make_unique<Model>(Model::Load(...))` would otherwise fully construct the new
      // Model (every weight DeviceBuffer) BEFORE the assignment destroys the old one, holding TWO
      // complete containers' worth of VRAM at once (2x peak for however long Load() takes -- e.g.
      // ~2x15.75 GiB for w4a16). This still pays a full container reload's latency (measured
      // ~18.6s) on every non-extending request; see docs/server.md's "Prefix reuse" section.
      model_.reset();
      model_ = std::make_unique<r4dx::model::Model>(r4dx::model::Model::Load(opts_.model_opts));
      fed_tokens_.clear();
      new_tokens.assign(full_tokens.begin(), full_tokens.end());
    }

    int64_t max_tokens = req.max_tokens;
    const int64_t ctx_budget = MaxCtx() - static_cast<int64_t>(full_tokens.size());
    if (max_tokens > ctx_budget) max_tokens = std::max<int64_t>(0, ctx_budget);

    req.sink->OnStart(static_cast<int64_t>(full_tokens.size()));

    const auto t0 = Clock::now();
    std::vector<float> logits = model_->Prefill(new_tokens);
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

    std::vector<int32_t> generated_tokens;
    std::string accumulated;
    std::string finish_reason = "length";

    const auto d0 = Clock::now();
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
      const std::string piece = decoder.push(next);
      bool stop_hit = false;
      if (!piece.empty()) {
        const size_t already_emitted = accumulated.size();
        accumulated += piece;
        const size_t lookback = already_emitted > 0 ? std::min<size_t>(already_emitted, 63) : 0;
        const size_t match = FindEarliestStop(accumulated, req.stop, already_emitted - lookback);
        if (match != std::string::npos) {
          const size_t emit_len = match > already_emitted ? match - already_emitted : 0;
          const std::string trimmed = piece.substr(0, emit_len);
          if (!trimmed.empty()) req.sink->OnToken(trimmed);
          stop_hit = true;
        } else {
          req.sink->OnToken(piece);
        }
      }
      // Feed `next` into the model regardless of stop_hit, so fed_tokens_ below accurately
      // reflects what the KV/GDN state actually holds (see PendingRequest/Engine's fed_tokens_
      // doc comment) -- the token is real generated content, only its stop-marker tail text is
      // withheld from the client.
      logits = model_->DecodeStep(next);
      if (stop_hit) {
        finish_reason = "stop";
        break;
      }
    }
    const std::string tail = decoder.flush();
    if (!tail.empty()) req.sink->OnToken(tail);
    const auto d1 = Clock::now();
    const double decode_seconds = Seconds(d0, d1);

    fed_tokens_ = full_tokens;
    fed_tokens_.insert(fed_tokens_.end(), generated_tokens.begin(), generated_tokens.end());

    req.sink->OnDone(finish_reason, static_cast<int64_t>(generated_tokens.size()));

    const double prefill_tps = prefill_seconds > 0 ? new_tokens.size() / prefill_seconds : 0.0;
    const double decode_tps =
        decode_seconds > 0 ? static_cast<double>(generated_tokens.size()) / decode_seconds : 0.0;
    char buf[320];
    std::snprintf(buf, sizeof(buf),
                  "request %s: prompt=%lld new=%lld generated=%lld finish=%s prefill=%.2f tok/s "
                  "decode=%.2f tok/s",
                  req.request_id.c_str(), static_cast<long long>(full_tokens.size()),
                  static_cast<long long>(new_tokens.size()),
                  static_cast<long long>(generated_tokens.size()), finish_reason.c_str(),
                  prefill_tps, decode_tps);
    LogLine(opts_.log_level, "info", buf);
  } catch (const std::exception& e) {
    // Invalidate the prefix-reuse fast path (review finding, 2026-09-19): if Model::Prefill or
    // DecodeStep threw partway through, the model may have already consumed some tokens that
    // fed_tokens_ (only updated on the success path above) does not record. Leaving fed_tokens_
    // stale would let the NEXT request's matches_prefix check take the "extend" branch and feed
    // only a tail onto a model whose real state has silently diverged from what fed_tokens_ claims
    // -- clearing it forces the next request down the full-reset (fresh Model::Load) path instead.
    fed_tokens_.clear();
    LogLine(opts_.log_level, "error",
            "request " + req.request_id + ": " + std::string(e.what()));
    req.sink->OnError(500, e.what());
  }
}

}  // namespace r4dx::server
