#include "engine.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <optional>
#include <random>
#include <stdexcept>

#include "mtp_round.hpp"
#include "r4dx/kernels/sampler.hpp"
#include "reasoning_splitter.h"
#include "tool_call_parser.h"

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
                       std::string& accumulated, int32_t tok, bool stream_to_client,
                       size_t* stop_match_pos, size_t min_stop_search_from) {
  const std::string piece = decoder.push(tok);
  if (piece.empty()) return false;
  const size_t already_emitted = accumulated.size();
  accumulated += piece;
  const size_t lookback = already_emitted > 0 ? std::min<size_t>(already_emitted, 63) : 0;
  size_t search_from = already_emitted - lookback;
  if (search_from < min_stop_search_from) search_from = min_stop_search_from;
  const size_t match = search_from > accumulated.size()
                            ? std::string::npos
                            : FindEarliestStop(accumulated, req.stop, search_from);
  if (match != std::string::npos) {
    const size_t emit_len = match > already_emitted ? match - already_emitted : 0;
    const std::string trimmed = piece.substr(0, emit_len);
    if (stream_to_client && !trimmed.empty()) req.sink->OnToken(trimmed);
    if (stop_match_pos) *stop_match_pos = match;
    return true;
  }
  if (stream_to_client) req.sink->OnToken(piece);
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
    // Resolved `enable_thinking` (docs/server.md's "reasoning_content" section, task item 5):
    // false for /v1/completions unconditionally (task item 5g -- no chat template, nothing to
    // split) and the same ResolveEnableThinking formula http_server.cpp already used to decide the
    // sink's own splitting behavior, so the two independently-made calls can never drift apart.
    const bool enable_thinking = req.kind == RequestKind::kChat
                                      ? ResolveEnableThinking(req.chat_template_kwargs, opts_.default_thinking)
                                      : false;
    if (req.kind == RequestKind::kChat) {
      r4dx::ChatJson messages = r4dx::ChatJson::array();
      for (const auto& m : req.messages) {
        r4dx::ChatJson entry = r4dx::ChatJson::object();
        // Map the legacy "function" role onto "tool" before rendering: chat_template.jinja
        // (C:\AI\models\Qwen3.8-27B\chat_template.jinja) has no "function" branch at all -- only
        // system/user/assistant/tool -- so a "function"-role message previously fell through to
        // the template's own `{% else %}{{ raise_exception('Unexpected message role.') }}`,
        // contradicting openai_types.cpp's own comment claiming the template treats "tool"/
        // "function" identically (review finding, 2026-09-20). The template's "tool" branch reads
        // only `content` (never `tool_call_id`), so this remap is exact -- `m.name` (the legacy
        // shape's own way of identifying which function answered) carries no template effect
        // either way, matching the pre-existing "tool" role's own behavior.
        entry["role"] = (m.role == "function") ? "tool" : m.role;
        entry["content"] = m.content ? r4dx::ChatJson(*m.content) : r4dx::ChatJson(nullptr);
        if (!m.tool_calls.empty()) {
          // Rebuild the OpenAI-wire tool_calls array into the shape chat_template.jinja's own
          // "render an earlier turn's tool call back into the prompt" branch expects: `arguments`
          // must be a real JSON object here (the template does `tool_call.arguments|items`), NOT
          // the JSON-encoded STRING the wire format itself carries -- ParseToolCallsField
          // (openai_types.cpp) already validated `arguments_json` decodes to an object, so this
          // parse cannot fail (task point 4: multi-turn tool round trip composes with the
          // existing chat template, no separate tool_call_id-aware rendering needed since this
          // checkpoint's own template never looks at tool_call_id -- see chat_template.jinja's
          // `elif message.role == "tool"` branch, which only reads `content`).
          r4dx::ChatJson tool_calls = r4dx::ChatJson::array();
          for (const auto& tc : m.tool_calls) {
            r4dx::ChatJson call = r4dx::ChatJson::object();
            call["id"] = tc.id;
            call["type"] = "function";
            call["function"] = {{"name", tc.name}, {"arguments", r4dx::ChatJson::parse(tc.arguments_json)}};
            tool_calls.push_back(call);
          }
          entry["tool_calls"] = tool_calls;
        }
        if (m.tool_call_id) entry["tool_call_id"] = *m.tool_call_id;
        if (m.name) entry["name"] = *m.name;
        // Multi-turn reasoning_content replay (task item 5e): passed through verbatim -- the chat
        // template's own `elif message.role == "assistant"` branch already reads
        // `message.reasoning_content` and decides whether to keep it based on
        // `chat_template_kwargs.preserve_thinking` (see openai_types.h's ChatMessage::
        // reasoning_content doc comment), so no extra gating is needed here.
        if (m.reasoning_content) entry["reasoning_content"] = *m.reasoning_content;
        messages.push_back(entry);
      }
      r4dx::ChatJson extra_context = req.chat_template_kwargs;
      if (!extra_context.contains("enable_thinking")) {
        extra_context["enable_thinking"] = opts_.default_thinking;
      }
      std::string rendered;
      try {
        rendered = tmpl_->render(messages, /*add_generation_prompt=*/true, req.tools, extra_context);
      } catch (const std::exception& e) {
        // Any ChatTemplate::render() failure here is a caller-shape problem, not an engine bug:
        // the template itself already parsed successfully at load time (ChatTemplate::from_
        // directory), so a render-time failure can only come from one of chat_template.jinja's own
        // `raise_exception(...)` validation calls over THIS request's messages/tools/extra_context
        // (e.g. "No user query found in messages." when a conversation is only role:"tool"/
        // "function" turns with no user turn anywhere, or "Unexpected message role."). Report it
        // as a 400 rather than falling through to the generic catch below, which would invalidate
        // the prefix-reuse cache and return an uninformative 500 for what is really a bad request
        // (review finding, 2026-09-20).
        req.sink->OnError(400, std::string("messages could not be rendered by this checkpoint's "
                                            "chat template: ") + e.what());
        return;
      }
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

    // DFlash2 self-speculative decode (docs/dflash2.md, stage S3): identical greedy-only gate as
    // MTP below, mutually exclusive with it (--dflash/--mtp are rejected together at the arg-parse
    // layer, so at most one of use_mtp/use_dflash is ever true). model_->DflashEnabled() is private
    // to r4dx::model::Model in the CLI's own header, so this uses the same "was the drafter loaded"
    // signal DecodeStepDflashGreedy itself throws on -- opts_.model_opts.dflash_draft_k > 0 is set
    // if and only if Model::Load was given a non-empty dflash_container (cli_args.h/server_args.h's
    // own mutual-exclusion + range checks already guarantee this pairing holds).
    //
    // Computed HERE, before this request's prefill, because it also drives the drafter-injection
    // toggle immediately below -- the decode loop further down just reads it again.
    const bool use_dflash =
        !opts_.model_opts.dflash_container.empty() && req.sampling.temperature <= 0.0f;
    // Per-request drafter-injection toggle (docs/server.md's "Sampled traffic pays nothing",
    // model.h's SetDflashInjectionEnabled). A sampled request will never call
    // DecodeStepDflashGreedy, so it should not pay the drafter's per-chunk feature capture +
    // encoder GEMM + 5-layer KV injection either. Set before Prefill so the whole request --
    // prefill chunks and plain decode steps alike -- runs with the right policy, and set on BOTH
    // the prefix-reuse and the Reset()+reprefill path (this line precedes both). A greedy request
    // arriving after sampled ones simply resumes injection at the current position, which
    // DflashDraft turns into a cold-ring gap rather than a throw.
    if (!opts_.model_opts.dflash_container.empty()) {
      model_->SetDflashInjectionEnabled(use_dflash);
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
    // Set by EmitToken (via its stop_match_pos out-param) the moment a --stop string matches;
    // std::string::npos means no stop matched. `accumulated` itself keeps growing past this point
    // (EmitToken only trims what it STREAMS, see its own comment) -- the tool_mode block below
    // must re-trim at this boundary before re-deriving its response from `accumulated`, or it
    // echoes the stop text (and anything decoded in the same token after it) back to the client
    // (review finding, 2026-09-20).
    size_t stop_match_pos = std::string::npos;

    // Reasoning-span bookkeeping (task item 5, "reasoning_content"): tracks, in `accumulated`'s
    // own character-offset space, whether/where the "</think>" close tag has appeared -- purely so
    // (a) the stop-string search inside EmitToken can be confined to the ANSWER part only
    // (`stop_search_floor`, see EmitToken's own doc comment for why) and (b)
    // usage.completion_tokens_details.reasoning_tokens can be computed in TOKEN space (the
    // sink-level ReasoningSplitter, response_sink.h, only ever sees decoded TEXT, so it cannot
    // count tokens itself). `stop_search_floor` starts at `npos` (skip stop matching entirely) for
    // a thinking-enabled request and 0 (no restriction) otherwise, byte-for-byte preserving
    // today's behavior when thinking is off.
    bool reasoning_open = enable_thinking;
    int64_t reasoning_tokens = 0;
    size_t stop_search_floor = enable_thinking ? std::string::npos : 0;
    auto NoteReasoningProgress = [&]() {
      if (!reasoning_open) return;
      const size_t p = accumulated.find("</think>");
      if (p == std::string::npos) return;
      reasoning_open = false;
      reasoning_tokens = static_cast<int64_t>(generated_tokens.size());
      stop_search_floor = p + 8;  // strlen("</think>")
    };

    // Tool calls (docs/server.md's "Tool calls" streaming decision): whenever this request could
    // plausibly produce a "<tool_call>" span (i.e. it offered any tool definitions), buffer the
    // ENTIRE generation instead of streaming it token-by-token -- EmitToken below still tracks
    // `accumulated`/applies --stop trimming exactly as always, it just skips the sink->OnToken()
    // forward. This guarantees a client can never see a half-formed tag as a content delta (the
    // CRITICAL failure mode this design avoids), at the honest cost of "fake" (all-at-once)
    // streaming for any request that offers tools, whether or not a call actually happens. A
    // request with no `tools` is completely unaffected (real per-token streaming, unchanged).
    const bool tool_mode = req.kind == RequestKind::kChat && !req.tools.empty();

    // Greedy MTP (task point 2): only a temperature<=0 request on a Model actually Load()'d with
    // mtp_draft_k>0 takes the speculative path -- everything else (non-greedy, or MTP disabled at
    // startup) is plain decode, byte-for-byte the pre-existing loop below. Mirrors src/cli/
    // main.cpp's RunTurn `greedy && args.mtp > 0` gate exactly (MTP is greedy-only, docs/mtp.md --
    // "probabilistic acceptance later" is still future work, per this stage's task).
    const bool use_mtp = model_->MtpEnabled() && req.sampling.temperature <= 0.0f;
    int64_t mtp_rounds = 0, mtp_drafted = 0, mtp_accepted = 0;
    // `use_dflash` is computed above, before the prefill, because it also gates this request's
    // drafter injection (see its own comment there).
    int64_t dflash_rounds = 0, dflash_drafted = 0, dflash_accepted = 0;

    const auto d0 = Clock::now();
    if (use_dflash) {
      const int64_t draft_k = opts_.model_opts.dflash_draft_k;
      int32_t next = r4dx::kernels::Argmax(logits.data(), static_cast<int64_t>(logits.size()));
      bool stopped = false;
      if (is_eos(next)) {
        finish_reason = "stop";
        stopped = true;
      } else {
        generated_tokens.push_back(next);
        if (EmitToken(req, decoder, accumulated, next, /*stream_to_client=*/!tool_mode,
                      &stop_match_pos, stop_search_floor)) {
          finish_reason = "stop";
          stopped = true;
        }
        NoteReasoningProgress();
      }
      while (!stopped && static_cast<int64_t>(generated_tokens.size()) < max_tokens) {
        if (req.sink->IsCancelled()) {
          finish_reason = "cancelled";
          break;
        }
        int64_t walk_len = 0;
        const std::vector<int32_t> round = model_->DecodeStepDflashGreedy(
            next, draft_k, opts_.dflash_p_min, opts_.dflash_n_min, &walk_len);
        ++dflash_rounds;
        dflash_drafted += walk_len;
        dflash_accepted += static_cast<int64_t>(round.size()) - 1;  // last token is never a draft
        // Same "compute the commit decision up front, atomically" contract as the MTP branch below
        // -- ProcessMtpRound is speculation-family-agnostic (mtp_round.hpp's own file comment).
        const r4dx::model::MtpRoundResult outcome = r4dx::model::ProcessMtpRound(
            round, is_eos, max_tokens - static_cast<int64_t>(generated_tokens.size()));
        committed_tokens.push_back(next);
        committed_tokens.insert(committed_tokens.end(), outcome.committed.begin(),
                                 outcome.committed.end());
        for (int32_t tok : outcome.displayed) {
          generated_tokens.push_back(tok);
          next = tok;
          if (EmitToken(req, decoder, accumulated, tok, /*stream_to_client=*/!tool_mode,
                        &stop_match_pos, stop_search_floor)) {
            finish_reason = "stop";
            stopped = true;
            NoteReasoningProgress();
            break;
          }
          NoteReasoningProgress();
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
    } else if (use_mtp) {
      const int64_t draft_k = opts_.model_opts.mtp_draft_k;
      int32_t next = r4dx::kernels::Argmax(logits.data(), static_cast<int64_t>(logits.size()));
      bool stopped = false;
      if (is_eos(next)) {
        finish_reason = "stop";
        stopped = true;
      } else {
        generated_tokens.push_back(next);
        if (EmitToken(req, decoder, accumulated, next, /*stream_to_client=*/!tool_mode,
                      &stop_match_pos, stop_search_floor)) {
          finish_reason = "stop";
          stopped = true;
        }
        NoteReasoningProgress();
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
          if (EmitToken(req, decoder, accumulated, tok, /*stream_to_client=*/!tool_mode,
                        &stop_match_pos, stop_search_floor)) {
            finish_reason = "stop";
            stopped = true;
            NoteReasoningProgress();
            break;
          }
          NoteReasoningProgress();
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
        const bool stop_hit = EmitToken(req, decoder, accumulated, next, /*stream_to_client=*/!tool_mode,
                                        &stop_match_pos, stop_search_floor);
        NoteReasoningProgress();
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
    if (!tool_mode && !tail_text.empty()) req.sink->OnToken(tail_text);
    accumulated += tail_text;  // harmless when empty; needed below so tool-call parsing sees the
                                // full text even in the rare case generation stopped mid-codepoint
    NoteReasoningProgress();  // covers the rare case where the close tag only completes via `flush`
    const auto d1 = Clock::now();
    const double decode_seconds = Seconds(d0, d1);

    prefix_.Commit(full_tokens, committed_tokens);

    if (tool_mode) {
      // Parse this checkpoint's real tool-call surface syntax (src/server/tool_call_parser.h,
      // docs/server.md's "Tool calls") out of the fully-buffered generation, then deliver it as
      // one content piece (the leading/trailing prose, if any -- empty when the whole turn was a
      // call) followed by one complete tool_calls batch, matching the "buffer whole" streaming
      // decision documented above. Robustness (task item 7): ParseToolCalls never throws --
      // malformed JSON in a parameter value degrades that value to a string, and a malformed
      // <tool_call> span degrades to literal content -- so this always produces a sane response,
      // never a crashed worker thread or a desynced stream, no matter how the model misbehaves.
      // Re-trim at the same boundary EmitToken used to decide what to STREAM (review finding,
      // 2026-09-20): `accumulated` itself always carries the full decoded text including a
      // matched --stop string (and anything decoded in the same token after it) -- parsing the
      // untrimmed string here would echo that stop text back via `parsed.content`, breaking
      // OpenAI's "the stop string terminates generation but is never itself returned" semantics
      // for any request combining `tools` and `stop`.
      std::string tool_parse_source =
          stop_match_pos != std::string::npos && stop_match_pos < accumulated.size()
              ? accumulated.substr(0, stop_match_pos)
              : accumulated;

      // reasoning_content in tool-call mode (task item 5d): strip the thinking span out of the
      // buffered generation BEFORE tool-call parsing even runs, using the same ReasoningSplitter
      // class the sinks use for the per-piece streaming split -- fed the whole buffered string in
      // one Push() call rather than piece by piece (ReasoningSplitter's own file comment on why one
      // state machine serves both). `sink->OnReasoningContent` is the one-shot equivalent of
      // OnToken's per-piece split, since tool_mode never streams per-piece deltas at all.
      if (enable_thinking) {
        ReasoningSplitter splitter;
        std::string reasoning_raw;
        std::string rest;
        auto collect = [&](const std::vector<ReasoningSplitter::Event>& events) {
          for (const auto& ev : events) (ev.is_reasoning ? reasoning_raw : rest) += ev.text;
        };
        collect(splitter.Push(tool_parse_source));
        collect(splitter.Finish());
        req.sink->OnReasoningContent(TrimReasoningWhitespace(reasoning_raw));
        tool_parse_source = std::move(rest);
      }

      r4dx::server::ToolCallParseResult parsed = r4dx::server::ParseToolCalls(tool_parse_source);
      std::vector<std::string> known_names;
      for (const auto& t : req.tools) {
        if (t.is_object() && t.contains("function") && t.at("function").is_object() &&
            t.at("function").contains("name") && t.at("function").at("name").is_string()) {
          known_names.push_back(t.at("function").at("name").get<std::string>());
        }
      }
      r4dx::server::DropUnknownToolCalls(parsed, known_names);

      if (!parsed.content.empty()) req.sink->OnToken(parsed.content);
      if (!parsed.tool_calls.empty()) {
        std::vector<ToolCallOut> tool_calls_out;
        tool_calls_out.reserve(parsed.tool_calls.size());
        for (auto& c : parsed.tool_calls) {
          tool_calls_out.push_back(ToolCallOut{GenerateRequestId("call_"), c.name, c.arguments_json});
        }
        req.sink->OnToolCalls(tool_calls_out);
        // A client mid-stream disconnect ("cancelled") is reported as-is regardless of whether a
        // well-formed call happens to have already been fully buffered -- the generation as a
        // whole did not complete normally, so claiming finish_reason "tool_calls" here would be
        // misleading (and for a non-streaming BufferingSink, "cancelled" never actually occurs --
        // see response_sink.h's IsCancelled() doc comment -- so this guard only ever matters for
        // the streaming path).
        if (finish_reason != "cancelled") finish_reason = "tool_calls";
      }
    }

    // Never-closed reasoning span (task item 5a): if thinking was on but generation ended (EOS,
    // --stop, cancellation, or max_tokens) before "</think>" ever appeared, every generated token
    // was reasoning -- matches the sinks' own Finish()-flush contract (ReasoningSplitter's own doc
    // comment), which puts all of it in reasoning_content and leaves content empty.
    if (enable_thinking && reasoning_open) reasoning_tokens = static_cast<int64_t>(generated_tokens.size());

    // `timings` (task point 1, llama.cpp-compatible field names): `prompt_n` is the tokens
    // actually fed to THIS request's Prefill (`new_tokens_i32`, excludes whatever prefix reuse
    // skipped) -- deliberately NOT `full_tokens.size()` (the whole conversation-so-far, which is
    // `usage.prompt_tokens`). `draft_n`/`draft_n_accepted` are set only when a speculative path
    // actually ran at least one round this request (`*_rounds > 0`), mirroring the stderr log
    // line's own "only mention mtp:/dflash: when rounds happened" gate just below.
    TimingStats timings;
    timings.prompt_n = static_cast<int64_t>(new_tokens_i32.size());
    timings.prompt_ms = prefill_seconds * 1000.0;
    timings.predicted_n = static_cast<int64_t>(generated_tokens.size());
    timings.predicted_ms = decode_seconds * 1000.0;
    if (use_mtp && mtp_rounds > 0) {
      timings.draft_n = mtp_drafted;
      timings.draft_n_accepted = mtp_accepted;
    } else if (use_dflash && dflash_rounds > 0) {
      timings.draft_n = dflash_drafted;
      timings.draft_n_accepted = dflash_accepted;
    }

    req.sink->OnDone(finish_reason, static_cast<int64_t>(generated_tokens.size()), timings,
                     reasoning_tokens);

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
    if (use_dflash && dflash_rounds > 0 && n > 0 && n < static_cast<int>(sizeof(buf))) {
      const double accept_rate = dflash_drafted > 0
          ? 100.0 * static_cast<double>(dflash_accepted) / static_cast<double>(dflash_drafted)
          : 0.0;
      std::snprintf(buf + n, sizeof(buf) - static_cast<size_t>(n),
                    " dflash: k=%lld rounds=%lld drafted=%lld accepted=%lld (%.1f%% accept, "
                    "%.2f tok/round)",
                    static_cast<long long>(opts_.model_opts.dflash_draft_k),
                    static_cast<long long>(dflash_rounds), static_cast<long long>(dflash_drafted),
                    static_cast<long long>(dflash_accepted), accept_rate,
                    static_cast<double>(generated_tokens.size()) / static_cast<double>(dflash_rounds));
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
