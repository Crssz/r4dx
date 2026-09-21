#include "engine.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <optional>
#include <random>
#include <stdexcept>

#include "image_prompt.h"  // src/vision: ExpandImagePlaceholders (docs/vision.md)
#include "mtp_round.hpp"
#include "r4dx/kernels/sampler.hpp"
#include "reasoning_splitter.h"
#include "tool_call_parser.h"
#include "tool_stream_gate.h"

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
// across token boundaries"), and hands whatever survives to `forward` (so every accepted token --
// plain-decode or MTP-round -- reaches the client as soon as it is committed, per this stage's task
// point 2). Returns true iff a --stop string matched (the caller should stop generating and report
// finish_reason="stop").
bool Engine::EmitToken(PendingRequest& req, r4dx::Tokenizer::StreamDecoder& decoder,
                       std::string& accumulated, int32_t tok,
                       const std::function<void(const std::string&)>& forward,
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
    if (!trimmed.empty()) forward(trimmed);
    if (stop_match_pos) *stop_match_pos = match;
    return true;
  }
  forward(piece);
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

  // The preprocessing config every image attachment will be decoded with (docs/vision.md "Large
  // images"). Built once here, not per request: `--image-max-pixels` is a server-lifetime policy,
  // and building it at load time is also what makes a bad value fail at startup rather than on the
  // first request with an image.
  image_preproc_ = r4dx::vision::MakeImageProcessorConfig(opts_.image_max_pixels);
  if (model_->HasVision()) {
    std::fprintf(stderr,
                 "[r4dx-server] vision tower ready (image_max_pixels=%lld, an image above that is "
                 "downsized by smart_resize, not rejected)\n",
                 static_cast<long long>(image_preproc_.max_pixels));
  }

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
    // The (possibly image-EXPANDED) prompt token sequence -- see the vision block below for why
    // this is int32 rather than r4dx::TokenId from the start (ExpandImagePlaceholders and every
    // downstream Model call want int32_t). `image_spans`/`image_keys` are filled only when this
    // request's messages actually carried an image content part; both stay empty for every
    // text-only request, which is what keeps that path byte-identical to before this stage.
    std::vector<int32_t> full_tokens_i32;
    std::vector<r4dx::model::Model::ImageSpan> image_spans;  // offsets relative to new_tokens_i32
    std::vector<r4dx::server::ImageKey> image_keys;          // this request's own fingerprints
    std::vector<r4dx::core::DeviceBuffer<uint16_t>> image_embeds_owned;  // keeps EncodeImages'
                                                                          // rows alive until
                                                                          // PrefillMultimodal runs
    int64_t image_encode_count = 0;
    double image_encode_ms_total = 0.0;
    // Every image span this request's (possibly expanded) prompt carries, offset relative to the
    // WHOLE conversation (`full_tokens_i32`) -- filled by the vision block below, consumed once
    // the prefix-reuse decision (further down) reveals which of them are actually NEW.
    std::vector<r4dx::vision::ImagePlaceholderSpan> pending_image_spans;
    std::vector<const ImagePart*> pending_image_ptrs;  // parallel to pending_image_spans
    // Resolved `enable_thinking` (docs/server.md's "reasoning_content" section, task item 5):
    // false for /v1/completions unconditionally (task item 5g -- no chat template, nothing to
    // split) and the same ResolveEnableThinking formula http_server.cpp already used to decide the
    // sink's own splitting behavior, so the two independently-made calls can never drift apart.
    const bool enable_thinking = req.kind == RequestKind::kChat
                                      ? ResolveEnableThinking(req.thinking, opts_.default_thinking)
                                      : false;
    if (req.kind == RequestKind::kChat) {
      // Vision (docs/vision.md, docs/server.md "Images"): every image content part across the
      // WHOLE conversation, in the order the client's `messages` array carries them -- a client
      // resends the full history on every turn (the normal OpenAI chat client pattern), so an
      // earlier turn's image is decoded again here too, even though its rows will turn out not to
      // need re-encoding below (only the pixel-decode/preprocess already happened once, at PARSE
      // time in openai_types.cpp -- this loop only re-derives grids/hashes from that, no image
      // bytes are re-decoded). `image_ptrs` keeps each image's own ImagePart (pixel_values,
      // content_hash) alongside `placeholders_in` (grid only) at the SAME index, since
      // ExpandImagePlaceholders preserves input order 1:1 into its returned spans.
      std::vector<r4dx::vision::ImagePlaceholderSpan> placeholders_in;
      std::vector<const ImagePart*> image_ptrs;
      for (const auto& m : req.messages) {
        for (const auto& part : m.content_parts) {
          if (!part.is_image) continue;
          r4dx::vision::ImagePlaceholderSpan sp;
          sp.grid = part.image.grid;
          placeholders_in.push_back(sp);
          image_ptrs.push_back(&part.image);
        }
      }
      if (!placeholders_in.empty() && !model_->HasVision()) {
        req.sink->OnError(400, "this model/container has no vision tower (loaded without "
                                "vision.* tensors, or started with --vision off)");
        return;
      }

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
        // Vision: a message with at least one image content part renders as a real content ARRAY
        // ({"type":"image"}/{"type":"text",...} entries, order preserved) so the chat template's
        // own image handling (vision_start/image_pad/vision_end, docs/vision.md) fires -- every
        // other message (content_parts empty, the overwhelming common case) renders exactly as
        // before this stage, a plain string or JSON null.
        if (!m.content_parts.empty()) {
          r4dx::ChatJson content = r4dx::ChatJson::array();
          for (const auto& part : m.content_parts) {
            content.push_back(part.is_image ? r4dx::ChatJson{{"type", "image"}}
                                             : r4dx::ChatJson{{"type", "text"}, {"text", part.text}});
          }
          entry["content"] = content;
        } else {
          entry["content"] = m.content ? r4dx::ChatJson(*m.content) : r4dx::ChatJson(nullptr);
        }
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
      // `chat_template_kwargs.enable_thinking`, when the caller sent one, is left exactly as it
      // arrived (an opaque passthrough the template reads with Jinja truthiness -- unchanged, and
      // the highest-precedence source ResolveEnableThinking already read). Only when the caller
      // sent none does the RESOLVED value go in: that is `--think` for every request that names no
      // thinking control at all (byte-identical to before this stage) and the request's own answer
      // for one that used `reasoning`/`thinking`/`enable_thinking`/`reasoning_effort` instead.
      if (!extra_context.contains("enable_thinking")) {
        extra_context["enable_thinking"] = enable_thinking;
      }
      // A requested effort, already mapped onto the three levels this template accepts
      // (ThinkingControls::template_effort). An explicit `chat_template_kwargs.reasoning_effort`
      // still wins, same passthrough rule as above; absent both, the template applies its own
      // default ("xhigh").
      if (req.thinking.template_effort && !extra_context.contains("reasoning_effort")) {
        extra_context["reasoning_effort"] = *req.thinking.template_effort;
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
      const std::vector<r4dx::TokenId> raw_tokens = tok_->encode(rendered, /*parse_special=*/true);
      full_tokens_i32.assign(raw_tokens.begin(), raw_tokens.end());

      // Vision: expand every `<|image_pad|>` placeholder the template just emitted (one per image
      // content part, docs/vision.md) into that image's real merged-token-count run, and record
      // where each run landed -- `full_tokens_i32` becomes the EXPANDED sequence from here on, and
      // `image_keys`/`pending_image_spans` feed the prefix-reuse decision just below. Untouched
      // (an empty vector, a no-op) for every text-only request, exactly the pre-vision behavior.
      if (!placeholders_in.empty()) {
        const int32_t image_token_id = static_cast<int32_t>(model_->GetContainer().ImageTokenId());
        const int merge_size =
            static_cast<int>(model_->GetContainer().Vision().config.spatial_merge_size);
        r4dx::vision::ExpandedImagePrompt expanded;
        try {
          expanded = r4dx::vision::ExpandImagePlaceholders(full_tokens_i32, image_token_id,
                                                            placeholders_in, merge_size);
        } catch (const std::exception& e) {
          req.sink->OnError(400,
                             std::string("image content parts do not match the rendered prompt's "
                                         "own placeholders: ") + e.what());
          return;
        }
        full_tokens_i32 = std::move(expanded.tokens);
        pending_image_spans = expanded.spans;
        pending_image_ptrs = image_ptrs;
        image_keys.reserve(expanded.spans.size());
        for (size_t i = 0; i < expanded.spans.size(); ++i) {
          r4dx::server::ImageKey key;
          key.content_hash = image_ptrs[i]->content_hash;
          key.grid_t = expanded.spans[i].grid.t;
          key.grid_h = expanded.spans[i].grid.h;
          key.grid_w = expanded.spans[i].grid.w;
          key.token_offset = expanded.spans[i].offset;
          image_keys.push_back(key);
        }
      }
    } else {
      // Raw prompt: never trust literal special-token surface forms in caller-supplied text.
      const std::vector<r4dx::TokenId> raw_tokens = tok_->encode(req.raw_prompt, /*parse_special=*/false);
      full_tokens_i32.assign(raw_tokens.begin(), raw_tokens.end());
    }

    if (full_tokens_i32.empty()) {
      req.sink->OnError(400, "prompt rendered to zero tokens");
      return;
    }
    if (static_cast<int64_t>(full_tokens_i32.size()) > MaxCtx()) {
      req.sink->OnError(400, "prompt (" + std::to_string(full_tokens_i32.size()) +
                                  " tokens, including any spliced image tokens) exceeds --max-ctx ("
                                  + std::to_string(MaxCtx()) + ")");
      return;
    }

    // DFlash2 self-speculative decode (docs/dflash2.md, docs/sampling.md section 9/10, Milestone 6
    // stage S3): runs at ANY temperature now -- mutually exclusive with MTP below (--dflash/--mtp
    // are rejected together at the arg-parse layer, so at most one of use_mtp/use_dflash is ever
    // true). model_->DflashEnabled() is private to r4dx::model::Model in the CLI's own header, so
    // this uses the same "was the drafter loaded" signal DecodeStepDflashGreedy/Sampled themselves
    // throw on -- opts_.model_opts.dflash_draft_k > 0 is set if and only if Model::Load was given a
    // non-empty dflash_container (cli_args.h/server_args.h's own mutual-exclusion + range checks
    // already guarantee this pairing holds).
    //
    // Computed HERE, before this request's prefill, because it also drives the drafter-injection
    // toggle immediately below -- the decode loop further down just reads it again.
    const bool use_dflash = !opts_.model_opts.dflash_container.empty();
    // Per-request drafter-injection toggle (model.h's SetDflashInjectionEnabled). Every request that
    // has a drafter loaded now uses it -- greedy via DecodeStepDflashGreedy, sampled via
    // DecodeStepDflashSampled (docs/sampling.md section 9.2's lossless sample-and-match) -- so
    // injection is simply always enabled whenever a drafter is configured; this call and the ring
    // gap tolerance mechanism it feeds (model.cpp's DflashDraft cold-ring handling) are otherwise
    // unchanged from before this stage. Set before Prefill so the whole request -- prefill chunks
    // and plain decode steps alike -- runs with the right policy, and set on BOTH the prefix-reuse
    // and the Reset()+reprefill path (this line precedes both).
    if (!opts_.model_opts.dflash_container.empty()) {
      model_->SetDflashInjectionEnabled(use_dflash);
    }

    // Prefix reuse (task point 2): continue from the existing KV/GDN state if `full_tokens_i32`
    // extends what's already fed; otherwise Model::Reset() (re-zero GDN/KV/MTP state in place,
    // milliseconds -- NOT a full Model::Load(), see model.h's Reset() doc comment and this
    // stage's own measurement below) and re-prefill from scratch. Image-aware (docs/vision.md
    // "Prefix reuse across a turn that contained an image"): `image_keys` is this request's own
    // per-image fingerprint list, so two requests carrying two DIFFERENT pictures at the same
    // (identical, since every placeholder is the same token id) position never reuse each other's
    // KV state -- PrefixState::Extend refuses and this falls to the Reset()+reprefill branch below.
    std::vector<int32_t> new_tokens_i32;
    double reset_ms = -1.0;  // -1 == no reset happened this request (prefix extended)
    int64_t skip = 0;        // tokens of full_tokens_i32 NOT re-fed this request (the fed prefix)
    std::optional<std::vector<int32_t>> tail = prefix_.Extend(full_tokens_i32, image_keys);
    if (tail) {
      skip = static_cast<int64_t>(full_tokens_i32.size() - tail->size());
      new_tokens_i32 = std::move(*tail);
    } else {
      const auto r0 = Clock::now();
      model_->Reset();
      const auto r1 = Clock::now();
      reset_ms = Seconds(r0, r1) * 1000.0;
      prefix_.Clear();
      new_tokens_i32.assign(full_tokens_i32.begin(), full_tokens_i32.end());
      skip = 0;
    }

    // Vision: only a span AT OR PAST the already-fed prefix boundary needs its rows spliced THIS
    // request -- an older image's rows are already resident in the model's real KV/GDN state from
    // the turn that first fed them, so re-encoding it here would be pure waste (the deliverable's
    // own "does NOT re-encode the image" requirement) as well as wrong (there is no position in
    // `new_tokens_i32` for it to land on). EncodeImages runs once per NEW image (simplicity over
    // batching -- a request rarely carries more than a couple), timed together for `timings.
    // image_n`/`image_ms`.
    for (size_t i = 0; i < pending_image_spans.size(); ++i) {
      const auto& sp = pending_image_spans[i];
      if (sp.offset < skip) continue;  // already fed on an earlier turn -- no re-encode
      const ImagePart& img = *pending_image_ptrs[i];
      r4dx::core::DeviceBuffer<uint16_t> embeds;
      r4dx::vision::VisionEncodeStats stats;
      const auto e0 = Clock::now();
      model_->EncodeImages(img.pixel_values.data(), sp.grid.PatchCount(), {sp.grid}, &embeds, &stats);
      const auto e1 = Clock::now();
      image_encode_count += 1;
      image_encode_ms_total += Seconds(e0, e1) * 1000.0;
      r4dx::model::Model::ImageSpan ms;
      ms.offset = sp.offset - skip;
      ms.tokens = sp.tokens;
      ms.grid = sp.grid;
      ms.embeds = embeds.data();
      image_spans.push_back(ms);
      image_embeds_owned.push_back(std::move(embeds));
    }

    int64_t max_tokens = req.max_tokens;
    const int64_t ctx_budget = MaxCtx() - static_cast<int64_t>(full_tokens_i32.size());
    if (max_tokens > ctx_budget) max_tokens = std::max<int64_t>(0, ctx_budget);

    req.sink->OnStart(static_cast<int64_t>(full_tokens_i32.size()));

    const auto t0 = Clock::now();
    // PrefillMultimodal with an EMPTY `image_spans` is byte-identical to Prefill() (that method's
    // own doc comment: the exact pre-vision code path, no extra upload, no extra kernel, the
    // single-row rope entry point) -- so this call-site unification carries no text-only-behavior
    // regression risk.
    std::vector<float> logits = image_spans.empty()
                                     ? model_->Prefill(new_tokens_i32)
                                     : model_->PrefillMultimodal(new_tokens_i32, image_spans);
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
    // One seeded rng per request (stage S3): the request's own seed when given, otherwise today's
    // behavior (a fresh std::random_device seed each time, above) -- unchanged either way. Every
    // sampling path below (plain, MTP, DFlash2) draws from this SAME generator, exactly one draw per
    // emitted token (docs/sampling.md design point A), which is what keeps a seeded sampled request
    // reproducible run to run at a fixed speculation setting (docs/sampling.md section 11).
    const bool greedy = req.sampling.temperature <= 0.0f;

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
    // The offset in `accumulated` of the first byte AFTER "</think>" -- the same position
    // `stop_search_floor` ends up holding, tracked separately because the two answer very different
    // questions (one bounds the stop-string search, the other tells the tool-call stream gate below
    // where the ANSWER text it may gate actually starts) and nothing should silently couple them.
    // `0` from the start when thinking is off: the whole generation is answer text then.
    size_t think_end = enable_thinking ? std::string::npos : 0;
    auto NoteReasoningProgress = [&]() {
      if (!reasoning_open) return;
      const size_t p = accumulated.find("</think>");
      if (p == std::string::npos) return;
      reasoning_open = false;
      reasoning_tokens = static_cast<int64_t>(generated_tokens.size());
      stop_search_floor = p + 8;  // strlen("</think>")
      think_end = p + 8;
    };

    // Tool calls (docs/server.md's "Tool calls" streaming decision): a request that offers tool
    // definitions gets its generation parsed for "<tool_call>" spans once, at the end, out of the
    // fully-accumulated text (the `tool_mode` block far below) -- that part is unchanged. What the
    // CLIENT sees while that generation is still running now depends on whether it asked to stream:
    //
    //   * no `tools` at all            -- straight per-token streaming, completely unaffected.
    //   * `tools` + "stream": false    -- nothing is delivered live (there is no live anything on a
    //                                     non-streaming request); the whole response is re-derived
    //                                     from `accumulated` afterwards, byte-for-byte as before.
    //   * `tools` + "stream": true     -- LIVE per-token content deltas, gated by ToolStreamGate
    //                                     (tool_stream_gate.h) so the stream shuts off the instant a
    //                                     real "<tool_call>" opener appears and no client ever sees
    //                                     even a partial "<tool_" prefix of one. Previously this
    //                                     case buffered the whole generation too, which made every
    //                                     turn of a client that always sends `tools` (Unsloth
    //                                     Studio does) arrive in one lump after the fact.
    const bool tool_mode = req.kind == RequestKind::kChat && !req.tools.empty();
    const bool live_tool_stream = tool_mode && req.stream;

    // Raw-offset bookkeeping for `live_tool_stream`. The sink is fed RAW generated bytes exactly as
    // a no-tools request would feed it -- thinking span, "</think>" tag and all -- because
    // BufferingSink/StreamingSink run their own ReasoningSplitter over that same byte stream
    // (response_sink.h) and must see it intact to split reasoning_content from content. Only the
    // ANSWER half runs through the gate, and since the gate can only ever hold back a SUFFIX of the
    // stream, "how much raw text is safe to forward" collapses to a single monotonically advancing
    // watermark over `accumulated`.
    size_t forwarded = 0;    // bytes of `accumulated` already handed to req.sink->OnToken
    size_t streamable = 0;   // bytes of `accumulated` eligible to be streamed at all (stop-trimmed)
    // Offset of the first byte the gate is fed -- i.e. where the string the end-of-generation
    // ParseToolCalls call will see begins inside `accumulated`. With thinking off that is simply 0
    // (no tag, no blank-line skip, the whole generation is answer text and the sinks append it
    // verbatim); with thinking on it is only known once the tag AND the first byte that survives
    // ReasoningSplitter's leading-blank-line skip have both arrived.
    size_t answer_begin = enable_thinking ? std::string::npos : 0;
    size_t gate_fed = 0;     // answer bytes already pushed into `gate`
    ToolStreamGate gate;
    auto FlushGated = [&](bool finishing) {
      // Keep `think_end` current before using it: a single decoded piece can carry the close tag
      // AND answer text (even an entire "<tool_call>" opener) at once, so waiting for the call
      // site's own NoteReasoningProgress() below would forward answer bytes the gate never saw.
      // Idempotent -- it returns immediately once the tag has been found.
      NoteReasoningProgress();
      size_t safe = streamable;  // still inside the thinking span: no answer text exists yet
      if (think_end != std::string::npos) {
        if (answer_begin == std::string::npos) {
          // Skip the blank lines right after the tag exactly as ReasoningSplitter does, so the gate
          // is fed the same `rest` string the end-of-generation split produces. The skip is only
          // final once a non-newline byte shows up; until then every byte so far is one the sink's
          // own splitter will drop, so forwarding it raw is free.
          size_t i = think_end;
          while (i < streamable && (accumulated[i] == '\n' || accumulated[i] == '\r')) ++i;
          if (i < streamable) answer_begin = i;
        }
        if (answer_begin != std::string::npos) {
          if (streamable > answer_begin + gate_fed) {
            gate.Push(accumulated.substr(answer_begin + gate_fed,
                                          streamable - (answer_begin + gate_fed)));
            gate_fed = streamable - answer_begin;
          }
          if (finishing) gate.Finish();
          safe = answer_begin + gate.streamed_bytes();
        }
      }
      if (safe > forwarded) {
        req.sink->OnToken(accumulated.substr(forwarded, safe - forwarded));
        forwarded = safe;
      }
    };
    // The one per-request router every EmitToken call below hands its surviving text to.
    auto forward = [&](const std::string& text) {
      if (!tool_mode) {
        req.sink->OnToken(text);
        return;
      }
      if (!live_tool_stream) return;  // non-streaming + tools: buffered whole, delivered at the end
      streamable += text.size();
      // Mirror the `tool_parse_source` re-trim the tool_mode block does, so nothing at or past a
      // matched --stop string is streamed once the match is known -- in practice this is the
      // decoder's own `flush()` tail, the only text forwarded after EmitToken has written
      // `stop_match_pos`. It cannot un-send a --stop string that only completed ACROSS a token
      // boundary (EmitToken's 63-byte lookback can match at a position the client already has):
      // that is the same inherent streaming limitation a request with no `tools` has, and the
      // `min()` guard where `still_owed` is computed below keeps it from turning into duplicated
      // content. The non-streaming path re-derives everything from `accumulated` and is exact.
      if (stop_match_pos != std::string::npos && streamable > stop_match_pos) {
        streamable = stop_match_pos;
      }
      FlushGated(/*finishing=*/false);
    };

    // MTP (docs/mtp.md, docs/sampling.md section 9/10, stage S3): any request on a Model actually
    // Load()'d with mtp_draft_k>0 takes the speculative path now, greedy or sampled -- everything
    // else (MTP disabled at startup, or DFlash2 active instead) is plain decode, byte-for-byte the
    // pre-existing loop below for a greedy request. Mirrors src/cli/main.cpp's RunTurn `args.mtp > 0`
    // gate exactly.
    const bool use_mtp = model_->MtpEnabled();
    int64_t mtp_rounds = 0, mtp_drafted = 0, mtp_accepted = 0;
    // `use_dflash` is computed above, before the prefill, because it also gates this request's
    // drafter injection (see its own comment there).
    int64_t dflash_rounds = 0, dflash_drafted = 0, dflash_accepted = 0;

    const auto d0 = Clock::now();
    if (use_dflash) {
      const int64_t draft_k = opts_.model_opts.dflash_draft_k;
      int32_t next = greedy
                          ? r4dx::kernels::Argmax(logits.data(), static_cast<int64_t>(logits.size()))
                          : r4dx::kernels::Sample(logits.data(), static_cast<int64_t>(logits.size()),
                                                   sp, rng);
      bool stopped = false;
      if (is_eos(next)) {
        finish_reason = "stop";
        stopped = true;
      } else {
        generated_tokens.push_back(next);
        if (EmitToken(req, decoder, accumulated, next, forward, &stop_match_pos, stop_search_floor)) {
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
        // Greedy: DecodeStepDflashGreedy, byte-identical to before this stage. Sampled: sample-and-
        // match rejection sampling (DecodeStepDflashSampled, docs/sampling.md section 9.2) -- for a
        // fixed seed this emits the same sequence as the plain sampled branch below.
        const std::vector<int32_t> round =
            greedy ? model_->DecodeStepDflashGreedy(next, draft_k, opts_.dflash_p_min,
                                                     opts_.dflash_n_min, &walk_len)
                   : model_->DecodeStepDflashSampled(next, draft_k, opts_.dflash_p_min,
                                                      opts_.dflash_n_min, sp, rng, &walk_len);
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
          if (EmitToken(req, decoder, accumulated, tok, forward, &stop_match_pos, stop_search_floor)) {
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
      int32_t next = greedy
                          ? r4dx::kernels::Argmax(logits.data(), static_cast<int64_t>(logits.size()))
                          : r4dx::kernels::Sample(logits.data(), static_cast<int64_t>(logits.size()),
                                                   sp, rng);
      bool stopped = false;
      if (is_eos(next)) {
        finish_reason = "stop";
        stopped = true;
      } else {
        generated_tokens.push_back(next);
        if (EmitToken(req, decoder, accumulated, next, forward, &stop_match_pos, stop_search_floor)) {
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
        // Greedy: DecodeStepMtpGreedy, byte-identical to before this stage. Sampled: sample-and-match
        // rejection sampling (DecodeStepMtpSampled, docs/sampling.md section 9).
        const std::vector<int32_t> round = greedy ? model_->DecodeStepMtpGreedy(next, draft_k)
                                                   : model_->DecodeStepMtpSampled(next, draft_k, sp, rng);
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
          if (EmitToken(req, decoder, accumulated, tok, forward, &stop_match_pos, stop_search_floor)) {
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
    } else if (greedy) {
      // Plain greedy decode (temperature<=0): untouched by this stage.
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
        const bool stop_hit = EmitToken(req, decoder, accumulated, next, forward, &stop_match_pos, stop_search_floor);
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
    } else {
      // Plain sampled decode (docs/sampling.md section 8, stage S3): the device-row-summary fast
      // path (Model::DecodeStepSampled) instead of a full-vocab logits D2H every step -- this
      // milestone's whole point (docs/sampling.md's measured cost table). The FIRST token is sampled
      // from Prefill's already-returned full logits (unavoidable -- that vector is what Prefill
      // gives us, one draw); every subsequent token comes from DecodeStepSampled, also exactly one
      // draw each -- so a seeded request consumes exactly one draw per emitted token throughout,
      // same as the plain greedy loop above consumes none.
      int32_t next = r4dx::kernels::Sample(logits.data(), static_cast<int64_t>(logits.size()), sp, rng);
      for (int64_t step = 0; step < max_tokens; ++step) {
        if (req.sink->IsCancelled()) {
          finish_reason = "cancelled";
          break;
        }
        const int32_t tok = next;
        if (is_eos(tok)) {
          finish_reason = "stop";
          break;
        }
        generated_tokens.push_back(tok);
        const bool stop_hit = EmitToken(req, decoder, accumulated, tok, forward, &stop_match_pos, stop_search_floor);
        NoteReasoningProgress();
        // Same "feed regardless of stop_hit" reasoning as the greedy branch above.
        next = model_->DecodeStepSampled(tok, sp, rng);
        committed_tokens.push_back(tok);
        if (stop_hit) {
          finish_reason = "stop";
          break;
        }
      }
    }
    const std::string tail_text = decoder.flush();
    accumulated += tail_text;  // BEFORE forwarding it: the live-tool-stream router reads its
                                // watermark out of `accumulated` itself. Harmless when empty;
                                // needed below so tool-call parsing sees the full text even in the
                                // rare case generation stopped mid-codepoint.
    if (!tail_text.empty()) forward(tail_text);
    NoteReasoningProgress();  // covers the rare case where the close tag only completes via `flush`
    // Release whatever the gate is still holding back (a proper prefix of "<tool_call>" that
    // generation ended before completing -- see ToolStreamGate::Finish), so `gate.streamed_bytes()`
    // below is final before the remainder is computed against it.
    if (live_tool_stream) FlushGated(/*finishing=*/true);
    const auto d1 = Clock::now();
    const double decode_seconds = Seconds(d0, d1);

    prefix_.Commit(full_tokens_i32, committed_tokens, image_keys);

    if (tool_mode) {
      // Parse this checkpoint's real tool-call surface syntax (src/server/tool_call_parser.h,
      // docs/server.md's "Tool calls") out of the fully-accumulated generation, then deliver
      // whatever content has not already gone out live followed by one complete tool_calls batch.
      // Robustness (task item 7): ParseToolCalls never throws --
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
      // accumulated generation BEFORE tool-call parsing even runs, using the same ReasoningSplitter
      // class the sinks use for the per-piece streaming split -- fed the whole string in one Push()
      // call rather than piece by piece (ReasoningSplitter's own file comment on why one state
      // machine serves both). Two things come out of this: `rest` (the ANSWER text, which is
      // exactly what the live gate above was fed, byte for byte -- both skip the tag and the blank
      // lines right after it at the same offsets) and, for a NON-streaming request only, the
      // trimmed reasoning span. A live-streaming request has already delivered its reasoning as
      // per-piece `reasoning_content` deltas through the sink's own splitter (which saw the raw
      // "</think>" bytes the router forwarded), so calling the one-shot OnReasoningContent here too
      // would duplicate it.
      if (enable_thinking) {
        ReasoningSplitter splitter;
        std::string reasoning_raw;
        std::string rest;
        auto collect = [&](const std::vector<ReasoningSplitter::Event>& events) {
          for (const auto& ev : events) (ev.is_reasoning ? reasoning_raw : rest) += ev.text;
        };
        collect(splitter.Push(tool_parse_source));
        collect(splitter.Finish());
        if (!live_tool_stream) req.sink->OnReasoningContent(TrimReasoningWhitespace(reasoning_raw));
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

      // What is left to say. Non-streaming: all of it, exactly as before. Live stream: only the
      // tail the gate held back -- a malformed span the parser degraded to literal content, prose
      // that followed a call, a DropUnknownToolCalls note -- so that the concatenation of every
      // content delta the client received equals `parsed.content` byte for byte, which is precisely
      // what the non-streaming `message.content` would have been (tool_stream_gate.h's
      // ToolStreamRemainder, the same function tests/server/test_tool_stream_gate.cpp asserts that
      // invariant with).
      const std::string still_owed =
          live_tool_stream
              ? ToolStreamRemainder(
                    tool_parse_source.substr(0, std::min(gate.streamed_bytes(),
                                                          tool_parse_source.size())),
                    parsed.content)
              : parsed.content;
      if (!still_owed.empty()) req.sink->OnToken(still_owed);
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
    // skipped) -- deliberately NOT `full_tokens_i32.size()` (the whole conversation-so-far, which is
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
    // `image_n`/`image_ms` (docs/vision.md, docs/server.md's "Images"): only THIS request's own
    // real EncodeImages calls -- an image whose rows were reused from an earlier turn's prefix
    // costs nothing here and is not counted, so this measures actual per-request GPU cost.
    if (image_encode_count > 0) {
      timings.image_n = image_encode_count;
      timings.image_ms = image_encode_ms_total;
    }

    req.sink->OnDone(finish_reason, static_cast<int64_t>(generated_tokens.size()), timings,
                     reasoning_tokens);

    const double prefill_tps = prefill_seconds > 0 ? new_tokens_i32.size() / prefill_seconds : 0.0;
    const double decode_tps =
        decode_seconds > 0 ? static_cast<double>(generated_tokens.size()) / decode_seconds : 0.0;
    char buf[480];
    int n = std::snprintf(
        buf, sizeof(buf),
        "request %s: prompt=%lld new=%lld generated=%lld finish=%s prefill=%.2f tok/s "
        "decode=%.2f tok/s",
        req.request_id.c_str(), static_cast<long long>(full_tokens_i32.size()),
        static_cast<long long>(new_tokens_i32.size()),
        static_cast<long long>(generated_tokens.size()), finish_reason.c_str(), prefill_tps,
        decode_tps);
    // Stage S3 (docs/server.md): speculation now runs at any temperature, so the log line reports
    // the settings that decide which decode path this request actually took -- temperature (greedy
    // vs sampled), whether it streamed, and whether thinking was on. `tools` is the number of tool
    // definitions the request OFFERED (not the number the model called, which `finish=tool_calls`
    // already implies): together with `stream` it is exactly what decides whether this request took
    // the live-gated tool stream, so a "why did my client see nothing until the end" report can be
    // diagnosed from the log alone.
    if (n > 0 && n < static_cast<int>(sizeof(buf))) {
      n += std::snprintf(buf + n, sizeof(buf) - static_cast<size_t>(n),
                         " temperature=%.3g stream=%s thinking=%s tools=%d",
                         static_cast<double>(req.sampling.temperature), req.stream ? "yes" : "no",
                         enable_thinking ? "yes" : "no", static_cast<int>(req.tools.size()));
    }
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
