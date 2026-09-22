// r4dx-cli: single-process text-generation entry point (docs/architecture.md "src/cli/").
//
// Two modes:
//   --prompt "..."   one-shot: render the chat template around a single user turn, generate, exit.
//   --chat           interactive: read one user line at a time from stdin, print the streamed
//                     assistant reply, keep the conversation (and the model's KV/GDN state) alive
//                     across turns.
// Runs on whichever HIP device HIP_VISIBLE_DEVICES selects (project rule: device 1, never 0 --
// this binary does not call hipSetDevice itself, matching every other r4dx entry point).
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>

#include "chat_template.h"
#include "cli_args.h"
#include "image_decode.h"  // src/vision: DecodeImageFile (docs/vision.md, --image)
#include "image_prompt.h"  // src/vision: ExpandImagePlaceholders (docs/vision.md, --image)
#include "model.h"
#include "mtp_round.hpp"
#include "preprocess.h"  // src/vision: ImageProcessorConfig (docs/vision.md "Large images")
#include "r4dx/core/device_buffer.hpp"
#include "r4dx/kernels/sampler.hpp"
#include "tokenizer.h"

namespace {

using r4dx::cli::CliArgs;
using r4dx::cli::CliUsageError;
using r4dx::cli::CliUsageText;
using r4dx::cli::ParseArgs;

using Clock = std::chrono::steady_clock;
double Seconds(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double>(b - a).count();
}

double VramUsedGiB() {
  size_t free_b = 0, total_b = 0;
  if (hipMemGetInfo(&free_b, &total_b) != hipSuccess) return -1.0;
  return static_cast<double>(total_b - free_b) / (1024.0 * 1024.0 * 1024.0);
}

// Runs one assistant turn: `new_tokens` are the tokens not yet fed to `model` (the chat template's
// re-render of the whole conversation, tail-sliced against what was already fed -- see main()).
// Streams decoded text to stdout as it is generated; returns the token ids actually generated
// (including a trailing eos id if one was hit) plus prefill/decode wall-clock seconds.
struct TurnResult {
  std::vector<int32_t> generated_tokens;
  // Tokens actually committed into the model's real KV/GDN state this turn -- NOT always the same
  // set as generated_tokens (docs/mtp.md's "mid-round" gap): an MTP round
  // (Model::DecodeStepMtpGreedy) commits every candidate up to (not including) its own returned
  // vector's LAST element atomically, in one call, regardless of whether main()'s per-token loop
  // below then stops mid-vector (--max-tokens reached, or an EOS candidate that isn't the last
  // element). `fed_tokens` (main(), below) must track committed_tokens, not generated_tokens, or
  // a LATER --chat turn's prefix match can silently desync from the model's real position. Equal
  // to generated_tokens on every non-MTP path (see RunTurn's plain-greedy/sampling loops, which
  // feed each token in the very same iteration it is generated, same as this field would compute).
  std::vector<int32_t> committed_tokens;
  std::string generated_text;
  double prefill_seconds = 0.0;
  double decode_seconds = 0.0;
  int64_t prefill_tokens = 0;
  int64_t decode_tokens = 0;
  bool hit_eos = false;
  // MTP acceptance stats (docs/mtp.md), zero/unset when --mtp 0. mtp_rounds: number of
  // DecodeStepMtpGreedy calls. mtp_drafted: sum of args.mtp across those calls (tokens offered).
  // mtp_accepted: sum of confirmed drafts (excludes the trailing correction/bonus token each round
  // always contributes) -- mtp_accepted/mtp_drafted is the acceptance rate docs/mtp.md reports.
  int64_t mtp_rounds = 0;
  int64_t mtp_drafted = 0;
  int64_t mtp_accepted = 0;
  // DFlash2 acceptance stats (docs/dflash2.md), zero/unset when --dflash is not given. Same
  // convention as the mtp_* fields above: dflash_drafted sums each round's OWN walk_len (not
  // args.dflash_k -- DFlash2's walk can stop early via p_min/n_min, unlike MTP which always offers
  // exactly args.mtp candidates), dflash_accepted excludes the trailing correction/bonus token.
  int64_t dflash_rounds = 0;
  int64_t dflash_drafted = 0;
  int64_t dflash_accepted = 0;
  // Set when this turn's plain-sampled branch was forced onto the pre-Milestone-6 full-vocab path
  // by R4DX_DEBUG_FULL_VOCAB_SAMPLER (see RunTurn) -- SampledFallbackRows() is never incremented on
  // that path (SampleFromSummary is never called), so the "fallback rate" --stats line below is not
  // applicable and must be suppressed rather than misreported as 0%.
  bool used_full_vocab_debug_sampler = false;
};

// Milestone 3 profiling pass (docs/r9700.md R5/Q7): prints a StepProfile's per-op-family table plus
// the "gemm:"-prefix GEMM-vs-non-GEMM split (profile_span.h's naming convention) shared by both
// --profile (decode) and --profile-prefill. `divisor` lets a caller show "per chunk" or "per layer"
// averages without changing what was actually measured (PrefillProfiled sums every chunk's every
// layer into one bucket per name -- see that method's own doc comment, model.h).
void PrintProfileTable(const r4dx::model::Model::StepProfile& prof, double divisor,
                        const char* divisor_label) {
  std::fprintf(stderr, "  %-55s %10s %8s %10s %12s\n", "GPU op family (hipEvent, overlaps enqueue)",
               "ms", "calls", "% gpu_sum", divisor_label);
  double gemm_ms = 0.0, non_gemm_ms = 0.0;
  for (const auto& e : prof.entries) {
    std::fprintf(stderr, "  %-55s %10.4f %8d %9.1f%% %12.4f\n", e.name.c_str(), e.ms, e.count,
                 100.0 * e.ms / prof.gpu_sum_ms, e.ms / divisor);
    if (e.name.rfind("gemm:", 0) == 0) gemm_ms += e.ms; else non_gemm_ms += e.ms;
  }
  std::fprintf(stderr, "  %-55s %10.4f\n", "(gpu_sum, NOT additive with wall/finish_wait)",
               prof.gpu_sum_ms);
  std::fprintf(stderr, "\n  %-55s %10.4f %9.1f%%\n", "GEMM share (names prefixed \"gemm:\")", gemm_ms,
               100.0 * gemm_ms / prof.gpu_sum_ms);
  std::fprintf(stderr, "  %-55s %10.4f %9.1f%%\n", "non-GEMM share (norm/rope/state/elementwise/attn-core)",
               non_gemm_ms, 100.0 * non_gemm_ms / prof.gpu_sum_ms);
  std::fprintf(stderr, "\n  %-55s %10lld\n",
               "r4dx-owned kernel launches (docs/r9700.md P2/R3)",
               static_cast<long long>(prof.r4dx_kernel_launches));
}

TurnResult RunTurn(r4dx::model::Model& model, const r4dx::Tokenizer& tok,
                    const std::vector<int32_t>& new_tokens, const CliArgs& args,
                    const std::vector<r4dx::model::Model::ImageSpan>& image_spans = {}) {
  TurnResult result;
  result.prefill_tokens = static_cast<int64_t>(new_tokens.size());

  // Milestone 3 profiling pass (docs/r9700.md R5/Q7): --profile-prefill profiles the WHOLE prefill
  // (chunked, per-kernel hipEvent spans -- Model::PrefillProfiled, model.cpp) instead of calling the
  // normal (uninstrumented) Model::Prefill -- prints the table to stderr and returns immediately,
  // same "standalone diagnostic, no generated text" contract --profile has (see below).
  if (args.profile_prefill) {
    const auto t0 = Clock::now();
    const r4dx::model::Model::StepProfile prof = model.PrefillProfiled(new_tokens);
    const auto t1 = Clock::now();
    const int64_t num_chunks = (static_cast<int64_t>(new_tokens.size()) + 63) / 64;  // max_chunk_=64
    std::fprintf(stderr,
                 "[profile-prefill] %zu prompt tokens, %lld chunks (T<=64 each), layout=%s:\n",
                 new_tokens.size(), static_cast<long long>(num_chunks), args.layout.c_str());
    PrintProfileTable(prof, static_cast<double>(num_chunks), "ms/chunk");
    std::fprintf(stderr, "  %-55s %10.4f\n", "wall (host, whole prefill)", prof.wall_ms);
    std::fprintf(stderr, "  %-55s %10.2f\n", "prefill tok/s (wall)",
                 static_cast<double>(new_tokens.size()) / Seconds(t0, t1));
    result.prefill_seconds = Seconds(t0, t1);
    return result;
  }

  const auto t0 = Clock::now();
  // Vision (docs/vision.md "Text-side splicing"): PrefillMultimodal with an EMPTY `image_spans` is
  // byte-identical to Prefill() (that method's own doc comment) -- it takes the exact pre-vision
  // code path, no extra upload, no extra kernel, the single-row rope entry point -- so this is not
  // a text-only-behavior regression risk, only a call-site unification. Every non-empty-span call
  // comes from run_one_user_turn(), which only ever builds one when --image / "/image ..." queued
  // a real image for this turn.
  std::vector<float> logits =
      image_spans.empty() ? model.Prefill(new_tokens) : model.PrefillMultimodal(new_tokens, image_spans);
  const auto t1 = Clock::now();
  result.prefill_seconds = Seconds(t0, t1);

  // Milestone 3 profiling pass (docs/r9700.md R5/Q2): --profile now profiles a STEADY-STATE decode
  // step (args.profile_token, default the 32nd generated token) instead of the first one -- Q2's own
  // finding was that the first generated token's DecodeStepProfiled call runs ~5.4 +/- 0.5 ms hotter
  // than the throughput table's steady-state step in every layout, a near-constant offset diagnosed
  // as first-token/instrumentation-warmup effects, not a per-layout effect (docs/perf.md's
  // "Milestone 3 profiling truth" section has the full before/after numbers). The
  // (args.profile_token - 1) DecodeStepGreedy calls below are plain, uninstrumented decode steps
  // that warm the step up to steady state before the one profiled call; DecodeStepProfiled still
  // commits its own token to real KV/GDN state and advances pos_ exactly like a real DecodeStep, so
  // falling through into the ordinary sample-then-DecodeStep loop afterward would double-commit that
  // position -- --profile remains a standalone diagnostic invocation, not meant to be combined with
  // getting correct generated text back.
  if (args.profile) {
    int32_t cur_tok = r4dx::kernels::Argmax(logits.data(), static_cast<int64_t>(logits.size()));
    for (int64_t i = 1; i < args.profile_token; ++i) {
      cur_tok = model.DecodeStepGreedy(cur_tok);
    }
    const r4dx::model::Model::StepProfile prof = model.DecodeStepProfiled(cur_tok);
    std::fprintf(stderr,
                 "[profile] decode step #%lld (1-indexed generated token, T=1), layout=%s:\n",
                 static_cast<long long>(args.profile_token), args.layout.c_str());
    PrintProfileTable(prof, 1.0, "ms");
    std::fprintf(stderr, "\n  %-55s %10s %9s\n", "host-clock breakdown", "ms", "% of wall");
    std::fprintf(stderr, "  %-55s %10.4f %9.1f%%\n", "host_enqueue (CPU work + async launch issue)",
                 prof.host_enqueue_ms, 100.0 * prof.host_enqueue_ms / prof.wall_ms);
    std::fprintf(stderr, "  %-55s %10.4f %9.1f%%\n",
                 "finish_wait (sync GPU-blocked wait + 4B d2h readback)", prof.finish_wait_ms,
                 100.0 * prof.finish_wait_ms / prof.wall_ms);
    std::fprintf(stderr, "  %-55s %10.4f\n", "(wall = host_enqueue + finish_wait)", prof.wall_ms);
    return result;
  }

  r4dx::kernels::SampleParams sp;
  sp.temperature = args.temperature;
  sp.top_k = args.top_k;
  sp.top_p = args.top_p;
  sp.min_p = args.min_p;
  sp.seed = args.seed;
  std::mt19937_64 rng = r4dx::kernels::MakeRng(args.seed);

  auto decoder = tok.make_stream_decoder(/*skip_special_tokens=*/true);
  const auto& eos_ids = tok.eos_ids();
  auto is_eos = [&](int32_t id) {
    for (int32_t e : eos_ids) if (e == id) return true;
    return false;
  };

  // Greedy (--temperature 0, r4dx::kernels::Sample's own "temperature<=0 => Argmax" rule) skips
  // the per-token vocab-sized logits D2H entirely: Model::DecodeStepGreedy argmaxes ON DEVICE and
  // reads back a single int32 (host-overhead pass, 2026-09-19) instead of the ~1MB fp32 logits
  // vector this loop would otherwise copy back every decode step just to re-scan it here on the
  // CPU for the same answer. The non-greedy path (temperature>0, or any top-k/top-p/min-p
  // sampling) used to need the full distribution on the host every step; as of Milestone 6 stage
  // S1/S2 it instead goes through Model::DecodeStepSampled's device row summary (docs/sampling.md
  // section 8) and only falls back to a full-vocab D2H+host pass on rows the summary cannot prove
  // (Model::SampledFallbackRows(), reported below via --stats).
  const bool greedy = args.temperature <= 0.0f;

  const auto d0 = Clock::now();
  if (!args.dflash.empty()) {
    // DFlash2 self-speculative decode (docs/dflash2.md, docs/sampling.md section 9/10, Milestone 6
    // stage S3): now runs at ANY temperature, not just greedy. `next` (the very first generated
    // token, from Prefill's own logits) is an Argmax for a greedy run and a canonical Sample
    // (r4dx::kernels::Sample, exactly one rng draw) for a sampled one -- same "must push/emit it
    // here or the prompt's first generated token is silently dropped" reasoning either way, since
    // Model::DecodeStepDflash{Greedy,Sampled}'s `token_id` parameter is "the last ALREADY-ACCEPTED
    // token" and does not itself re-emit it. The round loop below picks DecodeStepDflashGreedy
    // (argmax verify, byte-identical to before this stage) or DecodeStepDflashSampled
    // (sample-and-match rejection sampling, docs/sampling.md section 9.2 -- lossless: for a fixed
    // seed this emits the same sequence as the plain-sampled branch below) per round, both sharing
    // the identical round-vector contract and ProcessMtpRound reuse.
    int32_t next = greedy ? r4dx::kernels::Argmax(logits.data(), static_cast<int64_t>(logits.size()))
                          : r4dx::kernels::Sample(logits.data(), static_cast<int64_t>(logits.size()),
                                                   sp, rng);
    bool stopped = false;
    if (is_eos(next)) {
      result.hit_eos = true;
      stopped = true;
    } else {
      result.generated_tokens.push_back(next);
      const std::string piece0 = decoder.push(next);
      if (!piece0.empty()) {
        std::cout << piece0 << std::flush;
        result.generated_text += piece0;
      }
    }
    while (!stopped && static_cast<int64_t>(result.generated_tokens.size()) < args.max_tokens) {
      int64_t walk_len = 0;
      const std::vector<int32_t> round =
          greedy ? model.DecodeStepDflashGreedy(next, args.dflash_k, args.dflash_p_min,
                                                 args.dflash_n_min, &walk_len)
                 : model.DecodeStepDflashSampled(next, args.dflash_k, args.dflash_p_min,
                                                  args.dflash_n_min, sp, rng, &walk_len);
      result.dflash_rounds += 1;
      result.dflash_drafted += walk_len;
      result.dflash_accepted += static_cast<int64_t>(round.size()) - 1;  // last token is never a draft
      const r4dx::model::MtpRoundResult outcome = r4dx::model::ProcessMtpRound(
          round, is_eos, args.max_tokens - static_cast<int64_t>(result.generated_tokens.size()));
      result.committed_tokens.push_back(next);
      result.committed_tokens.insert(result.committed_tokens.end(), outcome.committed.begin(),
                                      outcome.committed.end());
      for (int32_t tok : outcome.displayed) {
        result.generated_tokens.push_back(tok);
        const std::string piece = decoder.push(tok);
        if (!piece.empty()) {
          std::cout << piece << std::flush;
          result.generated_text += piece;
        }
        next = tok;
      }
      if (outcome.hit_eos) { result.hit_eos = true; stopped = true; }
      if (outcome.hit_max_tokens) stopped = true;
    }
  } else if (args.mtp > 0) {
    // MTP self-speculative decode (docs/mtp.md, docs/sampling.md section 9/10, stage S3): now runs
    // at any temperature too -- same greedy/sampled split as the DFlash2 branch above
    // (DecodeStepMtpGreedy / DecodeStepMtpSampled), same "push the Prefill-derived first token here"
    // reasoning (review finding, 2026-09-19: "Silicon" came out as "icon" when this was missed).
    int32_t next = greedy ? r4dx::kernels::Argmax(logits.data(), static_cast<int64_t>(logits.size()))
                          : r4dx::kernels::Sample(logits.data(), static_cast<int64_t>(logits.size()),
                                                   sp, rng);
    bool stopped = false;
    if (is_eos(next)) {
      result.hit_eos = true;
      stopped = true;
    } else {
      result.generated_tokens.push_back(next);
      const std::string piece0 = decoder.push(next);
      if (!piece0.empty()) {
        std::cout << piece0 << std::flush;
        result.generated_text += piece0;
      }
    }
    while (!stopped && static_cast<int64_t>(result.generated_tokens.size()) < args.max_tokens) {
      const std::vector<int32_t> round = greedy ? model.DecodeStepMtpGreedy(next, args.mtp)
                                                 : model.DecodeStepMtpSampled(next, args.mtp, sp, rng);
      result.mtp_rounds += 1;
      result.mtp_drafted += args.mtp;
      result.mtp_accepted += static_cast<int64_t>(round.size()) - 1;  // last token is never a draft
      // `next` (this call's own token_id argument) is now committed by the call above; ProcessMtpRound
      // (src/model/mtp_round.hpp) computes the rest of `round` that is ALSO unconditionally
      // committed atomically by that same call, independent of how far the display loop below gets
      // -- see that header's file comment for why this must be computed up front rather than
      // incrementally inside a loop that can `break` early (docs/mtp.md's "mid-round" gap).
      const r4dx::model::MtpRoundResult outcome = r4dx::model::ProcessMtpRound(
          round, is_eos, args.max_tokens - static_cast<int64_t>(result.generated_tokens.size()));
      result.committed_tokens.push_back(next);
      result.committed_tokens.insert(result.committed_tokens.end(), outcome.committed.begin(),
                                      outcome.committed.end());
      for (int32_t tok : outcome.displayed) {
        result.generated_tokens.push_back(tok);
        const std::string piece = decoder.push(tok);
        if (!piece.empty()) {
          std::cout << piece << std::flush;
          result.generated_text += piece;
        }
        next = tok;
      }
      if (outcome.hit_eos) { result.hit_eos = true; stopped = true; }
      if (outcome.hit_max_tokens) stopped = true;
    }
  } else if (greedy) {
    int32_t next = r4dx::kernels::Argmax(logits.data(), static_cast<int64_t>(logits.size()));
    for (int64_t step = 0; step < args.max_tokens; ++step) {
      if (is_eos(next)) { result.hit_eos = true; break; }
      result.generated_tokens.push_back(next);
      const std::string piece = decoder.push(next);
      if (!piece.empty()) {
        std::cout << piece << std::flush;
        result.generated_text += piece;
      }
      result.committed_tokens.push_back(next);  // fed by the DecodeStepGreedy call just above
      next = model.DecodeStepGreedy(next);
    }
  } else if ([]() {
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif
    const char* v = std::getenv("R4DX_DEBUG_FULL_VOCAB_SAMPLER");
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
    return v != nullptr;
  }()) {
    // Documented debug flag (docs/sampling.md section 12, docs/perf.md's stage S3 measurement
    // section): forces the pre-Milestone-6 plain-sampled path -- a full ~993 KB logits D2H every
    // step plus the full-vocab host `r4dx::kernels::Sample` -- instead of `DecodeStepSampled`'s
    // device-row-summary fast path, SOLELY so a "before" number can be measured against an "after"
    // number on the exact same binary/container/request, without touching another checkout (the
    // task's own "measure the old path" ask). Not wired to any CLI flag on purpose: this is a
    // measurement tool, not a user-facing knob, and leaving it env-gated means it can never be hit
    // by accident. Byte-for-byte the loop this branch replaced before this stage.
    result.used_full_vocab_debug_sampler = true;
    for (int64_t step = 0; step < args.max_tokens; ++step) {
      const int32_t next = r4dx::kernels::Sample(
          logits.data(), static_cast<int64_t>(logits.size()), sp, rng);
      if (is_eos(next)) { result.hit_eos = true; break; }
      result.generated_tokens.push_back(next);
      const std::string piece = decoder.push(next);
      if (!piece.empty()) {
        std::cout << piece << std::flush;
        result.generated_text += piece;
      }
      logits = model.DecodeStep(next);
      result.committed_tokens.push_back(next);
    }
  } else {
    // Plain sampled decode (docs/sampling.md section 8): same first-token-then-loop shape as the
    // greedy branch above -- Argmax -> Sample, DecodeStepGreedy -> DecodeStepSampled -- so exactly
    // one rng draw is consumed per emitted token (the initial Sample call for token 0, one
    // DecodeStepSampled call per token after that). This is the reference the DFlash2/MTP sampled
    // branches above are proven lossless against (docs/sampling.md section 9.3, tests/model/
    // test_mtp.cpp's CheckSampledRoundsMatchPlain, tests/model/test_dflash_e2e.cpp).
    int32_t next = r4dx::kernels::Sample(logits.data(), static_cast<int64_t>(logits.size()), sp, rng);
    for (int64_t step = 0; step < args.max_tokens; ++step) {
      if (is_eos(next)) { result.hit_eos = true; break; }
      result.generated_tokens.push_back(next);
      const std::string piece = decoder.push(next);
      if (!piece.empty()) {
        std::cout << piece << std::flush;
        result.generated_text += piece;
      }
      result.committed_tokens.push_back(next);  // fed by the DecodeStepSampled call just below
      next = model.DecodeStepSampled(next, sp, rng);
    }
  }
  const std::string tail = decoder.flush();
  if (!tail.empty()) { std::cout << tail << std::flush; result.generated_text += tail; }
  const auto d1 = Clock::now();
  result.decode_seconds = Seconds(d0, d1);
  result.decode_tokens = static_cast<int64_t>(result.generated_tokens.size());
  return result;
}

}  // namespace

int RunMain(int argc, char** argv) {
  CliArgs args;
  try {
    args = ParseArgs(argc, argv);
  } catch (const CliUsageError& e) {
    std::fprintf(stderr, "%s\n%s\n", e.what(), CliUsageText(argv[0]).c_str());
    return 2;
  }

  r4dx::model::ModelOptions opts;
  opts.container_path = args.model_path;
  try {
    opts.layout = r4dx::model::LayoutFromName(args.layout);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 2;
  }
  opts.max_ctx = args.max_ctx;
  opts.layer_limit = args.layers;  // -1 (default) == the container's own num_hidden_layers
  // --mtp K and --dflash now both run at ANY temperature (docs/sampling.md section 9/10, Milestone 6
  // stage S3: DecodeStepMtpSampled / DecodeStepDflashSampled implement lossless sample-and-match
  // rejection sampling for a non-greedy request). RunTurn picks the greedy or sampled method per
  // round based on `greedy` there; no arg-time disabling is needed any more for either flag.
  opts.mtp_draft_k = args.mtp;
  opts.dflash_container = args.dflash;
  opts.dflash_draft_k = args.dflash.empty() ? 0 : args.dflash_k;
  // nullopt (== "layout") tracks opts.layout -- measured default, docs/mtp.md "MTP head layout".
  opts.mtp_head_layout = (args.mtp_head_layout == "bf16")
                              ? std::make_optional(r4dx::model::Layout::kBf16)
                              : std::nullopt;
  opts.embed_device_resident = (args.embed_device_resident != "off");
  opts.mtp_draft_reduced_vocab = (args.mtp_draft_head != "full");
  // docs/vision.md "Load policy": auto (the default) loads the container's vision.* weights iff it
  // has them, so a text-only container costs exactly what it did before this milestone.
  opts.vision = args.vision == "on"    ? r4dx::model::ModelOptions::VisionMode::kOn
                : args.vision == "off" ? r4dx::model::ModelOptions::VisionMode::kOff
                                        : r4dx::model::ModelOptions::VisionMode::kAuto;

  // allow_unimplemented_normalizer=true: Qwen3.8-27B's tokenizer.json declares normalizer.type=
  // NFC, which r4dx's tokenizer does not implement (tokenizer.h's file comment KNOWN GAP) --
  // without this, from_directory() refuses to load this exact checkpoint's tokenizer.json at all.
  // Same acknowledged, already-accounted-for gap tests/tokenizer/golden_test.cpp's own
  // default_options sets (its golden corpus is NFC-normalized by construction to route around it).
  r4dx::Tokenizer::Options tok_options;
  tok_options.allow_unimplemented_normalizer = true;
  r4dx::Tokenizer tok = r4dx::Tokenizer::from_directory(args.tokenizer_dir, tok_options);
  r4dx::ChatTemplate tmpl = r4dx::ChatTemplate::from_directory(args.tokenizer_dir);

  const double vram_before = VramUsedGiB();
  const auto load_t0 = Clock::now();
  r4dx::model::Model model = r4dx::model::Model::Load(opts);
  const auto load_t1 = Clock::now();
  const double vram_after = VramUsedGiB();

  if (args.stats) {
    std::fprintf(stderr, "[stats] container load: %.2fs, VRAM used: %.2f GiB (delta %.2f GiB)\n",
                 Seconds(load_t0, load_t1), vram_after, vram_after - vram_before);
  }

  // Image preprocessing policy (docs/vision.md "Large images"). Validated and reported at startup
  // rather than at the first image, so `--image-max-pixels 0.5` fails here instead of mid-session.
  // `--image` (and `--chat`'s own `/image <path>` lines) feed it below.
  const r4dx::vision::ImageProcessorConfig image_preproc =
      r4dx::vision::MakeImageProcessorConfig(args.image_max_pixels);
  if (model.HasVision()) {
    std::fprintf(stderr,
                 "[r4dx-cli] vision tower ready (image_max_pixels=%lld, an image above that is "
                 "downsized by smart_resize, not rejected)\n",
                 static_cast<long long>(image_preproc.max_pixels));
  }

  r4dx::ChatJson messages = r4dx::ChatJson::array();
  if (!args.system_prompt.empty()) {
    messages.push_back({{"role", "system"}, {"content", args.system_prompt}});
  }
  r4dx::ChatJson extra_context = r4dx::ChatJson::object();
  extra_context["enable_thinking"] = args.thinking;

  std::vector<int32_t> fed_tokens;  // everything already committed to model's KV/GDN state

  // Vision (docs/vision.md "Text-side splicing", --image): one entry per TURN that attached at
  // least one image, kept alive for the whole process -- a later turn's chat-template re-render
  // still carries every earlier turn's own "image" content part (messages is append-only), so its
  // placeholder has to be re-expanded every turn even though that image's rows were already
  // spliced into the model's real KV/GDN state and must NOT be re-encoded (docs/vision.md "Prefix
  // reuse across a turn that contained an image"). Grouped by turn (not by image) so a
  // multi-image turn's rows stay in the ONE device buffer EncodeImages produced for it -- no
  // device-to-device copy needed to split them apart.
  struct ImageBatch {
    std::vector<r4dx::vision::GridThw> grids;
    r4dx::core::DeviceBuffer<uint16_t> embeds;
  };
  std::vector<ImageBatch> session_image_batches;
  const int32_t image_token_id = static_cast<int32_t>(model.GetContainer().ImageTokenId());
  const int merge_size = model.HasVision()
                              ? static_cast<int>(model.GetContainer().Vision().config.spatial_merge_size)
                              : 2;

  auto run_one_user_turn = [&](const std::string& user_text,
                                const std::vector<std::string>& image_paths_for_turn) {
    int64_t turn_image_n = 0;
    double turn_image_ms = 0.0;
    if (!image_paths_for_turn.empty()) {
      if (!model.HasVision()) {
        throw std::runtime_error(
            "--image / \"/image\" needs a vision-capable container (this container has no "
            "vision.* tensors, or --vision off was given)");
      }
      std::vector<r4dx::vision::DecodedImage> decoded;
      decoded.reserve(image_paths_for_turn.size());
      for (const std::string& p : image_paths_for_turn) {
        decoded.push_back(r4dx::vision::DecodeImageFile(p));
      }
      const r4dx::vision::PreprocessedImages pre =
          r4dx::vision::PreprocessImages(decoded, image_preproc);
      ImageBatch batch;
      batch.grids = pre.grid_thw;
      r4dx::vision::VisionEncodeStats stats;
      model.EncodeImages(pre.pixel_values.data(), pre.TotalPatches(), pre.grid_thw, &batch.embeds,
                         &stats);
      turn_image_n = static_cast<int64_t>(pre.grid_thw.size());
      turn_image_ms = stats.encode_ms;
      session_image_batches.push_back(std::move(batch));
    }

    // Content: this turn's images (if any) followed by its text, matching the surface syntax
    // tests/vision/tool_vision_chat.cpp already validated against the real chat template. A
    // turn with no image keeps the exact plain-string content shape used before --image existed.
    if (image_paths_for_turn.empty()) {
      messages.push_back({{"role", "user"}, {"content", user_text}});
    } else {
      r4dx::ChatJson content = r4dx::ChatJson::array();
      for (size_t i = 0; i < image_paths_for_turn.size(); ++i) content.push_back({{"type", "image"}});
      content.push_back({{"type", "text"}, {"text", user_text}});
      messages.push_back({{"role", "user"}, {"content", content}});
    }

    const std::string rendered = tmpl.render(messages, /*add_generation_prompt=*/true,
                                              r4dx::ChatJson::array(), extra_context);
    const std::vector<r4dx::TokenId> raw_tokens = tok.encode(rendered, /*parse_special=*/true);
    std::vector<int32_t> full_tokens(raw_tokens.begin(), raw_tokens.end());

    // Re-expand EVERY image placeholder the re-rendered whole conversation carries -- not just
    // this turn's -- using every image ever attached this session, in order. This is what keeps
    // `full_tokens` (the EXPANDED sequence) comparable against `fed_tokens` (also expanded, since
    // it was built the very same way on an earlier turn). session_image_batches is empty for
    // every text-only session, in which case this is a no-op and `full_tokens` is untouched --
    // the same tokens tok.encode() produced, exactly as before --image existed.
    std::vector<r4dx::vision::ImagePlaceholderSpan> expanded_spans;
    if (!session_image_batches.empty()) {
      std::vector<r4dx::vision::ImagePlaceholderSpan> spans_in;
      for (const auto& batch : session_image_batches) {
        int64_t row = 0;
        for (const auto& g : batch.grids) {
          r4dx::vision::ImagePlaceholderSpan sp;
          sp.grid = g;
          sp.embeds = batch.embeds.data() + row * model.Config().hidden_size;
          spans_in.push_back(sp);
          row += g.MergedTokenCount(merge_size);
        }
      }
      auto expanded =
          r4dx::vision::ExpandImagePlaceholders(full_tokens, image_token_id, spans_in, merge_size);
      full_tokens = std::move(expanded.tokens);
      expanded_spans = std::move(expanded.spans);
    }

    std::vector<int32_t> new_tokens;
    int64_t skip = 0;
    if (full_tokens.size() < fed_tokens.size() ||
        !std::equal(fed_tokens.begin(), fed_tokens.end(), full_tokens.begin())) {
      // The re-rendered conversation did not extend the previously-fed token prefix -- e.g. the
      // stream decoder's skip_special_tokens=true dropped a special token or a thinking block
      // from generated_text (messages' assistant content), so re-tokenizing the re-rendered
      // template no longer matches what was actually fed via raw token ids. This should not
      // happen for Qwen's append-only chat template in the common case, but a model whose KV/GDN
      // state assumes strict prefix continuation cannot just resume from a divergent history --
      // degrade to dropping all state and re-prefilling the whole conversation from scratch
      // rather than killing the session (std::exit(1)).
      std::fprintf(stderr, "warning: chat template re-render did not extend the previous token "
                            "prefix; dropping state and re-prefilling the whole conversation\n");
      model = r4dx::model::Model::Load(opts);
      new_tokens.assign(full_tokens.begin(), full_tokens.end());
      skip = 0;
    } else {
      skip = static_cast<int64_t>(fed_tokens.size());
      new_tokens.assign(full_tokens.begin() + static_cast<ptrdiff_t>(fed_tokens.size()),
                         full_tokens.end());
    }

    // Only a span at or past the already-fed prefix boundary belongs to THIS Prefill/
    // PrefillMultimodal call -- an older image's rows are already resident in the model's real
    // KV/GDN state from the turn that first fed them (docs/vision.md "Prefix reuse across a turn
    // that contained an image"); re-including it here would try to splice into a position
    // `new_tokens` does not cover. Offsets are shifted from "index into the whole conversation"
    // to "index into new_tokens", matching Model::ImageSpan::offset's own contract.
    std::vector<r4dx::model::Model::ImageSpan> image_spans;
    for (const auto& sp : expanded_spans) {
      if (sp.offset < skip) continue;
      r4dx::model::Model::ImageSpan ms;
      ms.offset = sp.offset - skip;
      ms.tokens = sp.tokens;
      ms.grid = sp.grid;
      ms.embeds = sp.embeds;
      image_spans.push_back(ms);
    }

    // Sampled-fallback rate (docs/sampling.md section 4/12, docs/perf.md's stage S3 measurement
    // ask): Model::SampledFallbackRows() is a cumulative-since-Load() counter, so bracket this
    // turn's own decode with before/after reads to get just this turn's rows -- meaningful only for
    // a temperature>0 turn (a greedy one never calls SampleFromSummary at all, so the delta is
    // always 0 there).
    const int64_t fallback_rows_before = model.SampledFallbackRows();
    const TurnResult r = RunTurn(model, tok, new_tokens, args, image_spans);
    const int64_t fallback_rows_this_turn = model.SampledFallbackRows() - fallback_rows_before;
    std::cout << std::endl;

    // committed_tokens (not generated_tokens) is what's actually in the model's KV/GDN state --
    // see TurnResult::committed_tokens's own comment / docs/mtp.md's "mid-round" gap (fixed here).
    fed_tokens = full_tokens;
    fed_tokens.insert(fed_tokens.end(), r.committed_tokens.begin(), r.committed_tokens.end());
    messages.push_back({{"role", "assistant"}, {"content", r.generated_text}});

    // --dump-token-ids (docs/validation.md "Rung 4 tooling", cli_args.h's own comment): the exact
    // ids now resident in the model's KV/GDN state, in the Rung 4 tokens.json shared format, so
    // tests/model/tool_teacher_forced_logprobs can score the SAME sequence this run generated.
    // Rewritten in full after every turn. Hand-written JSON rather than the nlohmann object this TU
    // already has: one integer array does not need a DOM, and keeping the writer trivial keeps the
    // "compact, no spaces" token_ids serialization that the shared sha256_of_token_ids_json field
    // is defined over visible right here.
    if (!args.dump_token_ids.empty()) {
      std::FILE* f = std::fopen(args.dump_token_ids.c_str(), "wb");
      if (f == nullptr) {
        std::fprintf(stderr, "warning: cannot write --dump-token-ids %s\n",
                     args.dump_token_ids.c_str());
      } else {
        std::string tok_dir_json = r4dx::ChatJson(args.tokenizer_dir).dump();
        std::fprintf(f, "{\n  \"tokenizer\": %s,\n  \"segments\": [\n    {\n", tok_dir_json.c_str());
        std::fprintf(f, "      \"name\": \"cli\",\n      \"token_ids\": [");
        for (size_t i = 0; i < fed_tokens.size(); ++i) {
          std::fprintf(f, "%s%d", i ? "," : "", fed_tokens[i]);
        }
        std::fprintf(f, "]\n    }\n  ]\n}\n");
        std::fclose(f);
        std::fprintf(stderr, "[r4dx-cli] wrote %zu token ids to %s\n", fed_tokens.size(),
                     args.dump_token_ids.c_str());
      }
    }

    if (args.stats) {
      if (turn_image_n > 0) {
        std::fprintf(stderr, "[stats] image: %lld image(s) encoded in %.1f ms\n",
                     static_cast<long long>(turn_image_n), turn_image_ms);
      }
      if (!image_spans.empty()) {
        int64_t image_tokens = 0;
        for (const auto& sp : image_spans) image_tokens += sp.tokens;
        std::fprintf(stderr, "[stats] image: %lld image token(s) spliced into this prefill\n",
                     static_cast<long long>(image_tokens));
      }
      const double pfx_tps = r.prefill_seconds > 0 ? r.prefill_tokens / r.prefill_seconds : 0.0;
      const double dec_tps = r.decode_seconds > 0 ? r.decode_tokens / r.decode_seconds : 0.0;
      std::fprintf(stderr,
                   "[stats] prefill: %lld tok in %.3fs (%.2f tok/s) | decode: %lld tok in "
                   "%.3fs (%.2f tok/s) | eos=%s | VRAM: %.2f GiB\n",
                   static_cast<long long>(r.prefill_tokens), r.prefill_seconds, pfx_tps,
                   static_cast<long long>(r.decode_tokens), r.decode_seconds, dec_tps,
                   r.hit_eos ? "yes" : "no (max-tokens)", VramUsedGiB());
      // Denominator is committed_tokens (rows actually emitted through the sampler this turn --
      // Model::SampledFallbackRows()'s own unit), NOT decode_tokens/generated_tokens (DISPLAYED
      // tokens): a speculative round that stops mid-round commits more rows than it displays
      // (docs/mtp.md's "mid-round" gap), so dividing by decode_tokens could read over 100%.
      // Suppressed entirely when R4DX_DEBUG_FULL_VOCAB_SAMPLER forced the old full-vocab path,
      // where SampleFromSummary is never consulted and "0% fallback" would misreport "N/A".
      if (args.temperature > 0.0f && !r.used_full_vocab_debug_sampler &&
          !r.committed_tokens.empty()) {
        std::fprintf(stderr,
                     "[stats] sampled: fallback_rows=%lld/%lld (%.1f%% of emitted tokens took the "
                     "full-vocab path)\n",
                     static_cast<long long>(fallback_rows_this_turn),
                     static_cast<long long>(r.committed_tokens.size()),
                     100.0 * static_cast<double>(fallback_rows_this_turn) /
                         static_cast<double>(r.committed_tokens.size()));
      }
      if (args.mtp > 0) {
        const double accept_rate =
            r.mtp_drafted > 0 ? 100.0 * static_cast<double>(r.mtp_accepted) / static_cast<double>(r.mtp_drafted) : 0.0;
        std::fprintf(stderr,
                     "[stats] mtp: draft_k=%lld rounds=%lld drafted=%lld accepted=%lld "
                     "(%.1f%% acceptance, %.2f tok/round avg) draft_head=%s\n",
                     static_cast<long long>(args.mtp), static_cast<long long>(r.mtp_rounds),
                     static_cast<long long>(r.mtp_drafted), static_cast<long long>(r.mtp_accepted),
                     accept_rate,
                     r.mtp_rounds > 0
                         ? static_cast<double>(r.decode_tokens) / static_cast<double>(r.mtp_rounds)
                         : 0.0,
                     // docs/r9700.md R9: report which draft head this run actually used --
                     // "reduced" only if the container has one AND --mtp-draft-head didn't force
                     // "full" (Model::MtpUsingReducedVocabDraft's own two-condition check).
                     model.MtpUsingReducedVocabDraft() ? "reduced" : "full");
      }
      if (!args.dflash.empty()) {
        const double accept_rate = r.dflash_drafted > 0
            ? 100.0 * static_cast<double>(r.dflash_accepted) / static_cast<double>(r.dflash_drafted)
            : 0.0;
        std::fprintf(stderr,
                     "[stats] dflash: k=%lld p_min=%.3g n_min=%lld rounds=%lld drafted=%lld "
                     "accepted=%lld (%.1f%% acceptance, %.2f tok/round avg)\n",
                     static_cast<long long>(args.dflash_k), static_cast<double>(args.dflash_p_min),
                     static_cast<long long>(args.dflash_n_min),
                     static_cast<long long>(r.dflash_rounds), static_cast<long long>(r.dflash_drafted),
                     static_cast<long long>(r.dflash_accepted), accept_rate,
                     r.dflash_rounds > 0
                         ? static_cast<double>(r.decode_tokens) / static_cast<double>(r.dflash_rounds)
                         : 0.0);
      }
    }
  };

  if (!args.prompt.empty()) {
    run_one_user_turn(args.prompt, args.image_paths);
    return 0;
  }

  // --chat: "/image <path>" queues one image for the NEXT real input line (repeatable -- several
  // "/image" lines in a row attach several images to that one turn), matching --prompt's own
  // --image flag one level up. --image on the command line (if given) attaches to the FIRST typed
  // line, the natural "first turn" reading of a flag that has no other notion of "which turn" in
  // an interactive session.
  std::vector<std::string> queued_images = args.image_paths;
  std::string line;
  std::cout << "> " << std::flush;
  bool first_line = true;
  while (std::getline(std::cin, line)) {
    // A redirected/piped stdin (a test harness, `foo.txt | r4dx-cli --chat`, some terminal/locale
    // combinations) can prepend a UTF-8 BOM to the very first line and/or leave a trailing '\r' on
    // every line (text vs binary stdin mode) -- neither survives an interactive human typing into a
    // real console, but both would otherwise make "/image " command detection silently miss on
    // exactly the line most likely to be it (the very first thing piped in).
    if (first_line && line.size() >= 3 && static_cast<unsigned char>(line[0]) == 0xEF &&
        static_cast<unsigned char>(line[1]) == 0xBB && static_cast<unsigned char>(line[2]) == 0xBF) {
      line.erase(0, 3);
    }
    first_line = false;
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.rfind("/image ", 0) == 0) {
      std::string path = line.substr(7);
      while (!path.empty() && (path.front() == ' ' || path.front() == '"')) path.erase(path.begin());
      while (!path.empty() && (path.back() == ' ' || path.back() == '"')) path.pop_back();
      if (path.empty()) {
        std::fprintf(stderr, "usage: /image <path>\n");
      } else {
        queued_images.push_back(path);
        std::fprintf(stderr, "[r4dx-cli] queued image '%s' for the next turn (%zu queued)\n",
                     path.c_str(), queued_images.size());
      }
      std::cout << "> " << std::flush;
      continue;
    }
    if (!line.empty()) {
      run_one_user_turn(line, queued_images);
      queued_images.clear();
    }
    std::cout << "> " << std::flush;
  }
  return 0;
}

int main(int argc, char** argv) {
  // Top-level guard: any exception RunMain doesn't already handle itself (a bad --model path, a
  // malformed container, a HIP allocation failure, ...) prints a clean message and a non-zero
  // exit code instead of propagating past main() into std::terminate()/abort() -- which on this
  // MSVC/clang-cl toolchain surfaces to the shell as an opaque STATUS_STACK_BUFFER_OVERRUN-style
  // fail-fast (0xC0000409), indistinguishable from a real memory-safety bug.
  try {
    return RunMain(argc, argv);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
