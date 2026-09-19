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
#include <cstring>
#include <iostream>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>

#include "chat_template.h"
#include "cli_args.h"
#include "model.h"
#include "mtp_round.hpp"
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
                    const std::vector<int32_t>& new_tokens, const CliArgs& args) {
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
  std::vector<float> logits = model.Prefill(new_tokens);
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
  // CPU for the same answer. Only the non-greedy path (temperature>0, or any top-k/top-p/min-p
  // sampling) needs the full distribution on the host.
  const bool greedy = args.temperature <= 0.0f;

  const auto d0 = Clock::now();
  if (greedy && args.mtp > 0) {
    // MTP self-speculative decode (docs/mtp.md): each DecodeStepMtpGreedy call drafts up to
    // args.mtp tokens and returns however many the real model actually confirmed (1..args.mtp+1,
    // always at least the corrected/bonus token) -- emit them one at a time, exactly like the
    // plain-greedy loop below, so stop-on-EOS and --max-tokens truncation behave identically
    // regardless of how many tokens one round happened to produce.
    //
    // Model::DecodeStepMtpGreedy's `token_id` parameter is "the last ALREADY-ACCEPTED token" (same
    // convention as DecodeStep/DecodeStepGreedy) -- it does not itself re-emit that token, only
    // whatever comes after it. `next` below (Prefill's own argmax'd result) is the FIRST generated
    // token and has not been emitted by anything yet (unlike every later round's seed, which was
    // already pushed into result.generated_tokens by the round that produced it) -- push it here,
    // exactly like the plain-greedy loop below does before its own first DecodeStepGreedy call, or
    // the prompt's first generated token is silently dropped from the output (review finding: this
    // is exactly what happened before this fix -- "Silicon" came out as "icon").
    int32_t next = r4dx::kernels::Argmax(logits.data(), static_cast<int64_t>(logits.size()));
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
      const std::vector<int32_t> round = model.DecodeStepMtpGreedy(next, args.mtp);
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
  } else {
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
      result.committed_tokens.push_back(next);  // fed by the DecodeStep call just above
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
  // --mtp K combined with --temperature > 0 previously did nothing useful: RunTurn's
  // `greedy && args.mtp > 0` branch (below) is skipped whenever temperature>0, generation falls
  // through to plain Model::DecodeStep, and no warning was printed -- while the Model was still
  // loaded with mtp_draft_k=K, needlessly allocating MtpHead's own KV cache and widening every GDN
  // layer's window bank (review finding, 2026-09-19; cli_args.h documents the restriction in a
  // comment, but the runtime was silent about it). Warn and drop mtp_draft_k to 0 instead -- only
  // greedy sampling is implemented for MTP (docs/mtp.md).
  if (args.mtp > 0 && args.temperature > 0.0f) {
    std::fprintf(stderr,
                 "r4dx-cli: --mtp %lld has no effect with --temperature %.3g > 0 (MTP is "
                 "greedy-only, docs/mtp.md) -- disabling MTP for this run\n",
                 static_cast<long long>(args.mtp), static_cast<double>(args.temperature));
    args.mtp = 0;
  }
  opts.mtp_draft_k = args.mtp;
  // nullopt (== "layout") tracks opts.layout -- measured default, docs/mtp.md "MTP head layout".
  opts.mtp_head_layout = (args.mtp_head_layout == "bf16")
                              ? std::make_optional(r4dx::model::Layout::kBf16)
                              : std::nullopt;
  opts.embed_device_resident = (args.embed_device_resident != "off");

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

  r4dx::ChatJson messages = r4dx::ChatJson::array();
  if (!args.system_prompt.empty()) {
    messages.push_back({{"role", "system"}, {"content", args.system_prompt}});
  }
  r4dx::ChatJson extra_context = r4dx::ChatJson::object();
  extra_context["enable_thinking"] = args.thinking;

  std::vector<int32_t> fed_tokens;  // everything already committed to model's KV/GDN state

  auto run_one_user_turn = [&](const std::string& user_text) {
    messages.push_back({{"role", "user"}, {"content", user_text}});
    const std::string rendered = tmpl.render(messages, /*add_generation_prompt=*/true,
                                              r4dx::ChatJson::array(), extra_context);
    const std::vector<r4dx::TokenId> full_tokens = tok.encode(rendered, /*parse_special=*/true);
    std::vector<int32_t> new_tokens;
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
    } else {
      new_tokens.assign(full_tokens.begin() + static_cast<ptrdiff_t>(fed_tokens.size()),
                         full_tokens.end());
    }

    const TurnResult r = RunTurn(model, tok, new_tokens, args);
    std::cout << std::endl;

    // committed_tokens (not generated_tokens) is what's actually in the model's KV/GDN state --
    // see TurnResult::committed_tokens's own comment / docs/mtp.md's "mid-round" gap (fixed here).
    fed_tokens = full_tokens;
    fed_tokens.insert(fed_tokens.end(), r.committed_tokens.begin(), r.committed_tokens.end());
    messages.push_back({{"role", "assistant"}, {"content", r.generated_text}});

    if (args.stats) {
      const double pfx_tps = r.prefill_seconds > 0 ? r.prefill_tokens / r.prefill_seconds : 0.0;
      const double dec_tps = r.decode_seconds > 0 ? r.decode_tokens / r.decode_seconds : 0.0;
      std::fprintf(stderr,
                   "[stats] prefill: %lld tok in %.3fs (%.2f tok/s) | decode: %lld tok in "
                   "%.3fs (%.2f tok/s) | eos=%s | VRAM: %.2f GiB\n",
                   static_cast<long long>(r.prefill_tokens), r.prefill_seconds, pfx_tps,
                   static_cast<long long>(r.decode_tokens), r.decode_seconds, dec_tps,
                   r.hit_eos ? "yes" : "no (max-tokens)", VramUsedGiB());
      if (args.mtp > 0) {
        const double accept_rate =
            r.mtp_drafted > 0 ? 100.0 * static_cast<double>(r.mtp_accepted) / static_cast<double>(r.mtp_drafted) : 0.0;
        std::fprintf(stderr,
                     "[stats] mtp: draft_k=%lld rounds=%lld drafted=%lld accepted=%lld "
                     "(%.1f%% acceptance, %.2f tok/round avg)\n",
                     static_cast<long long>(args.mtp), static_cast<long long>(r.mtp_rounds),
                     static_cast<long long>(r.mtp_drafted), static_cast<long long>(r.mtp_accepted),
                     accept_rate,
                     r.mtp_rounds > 0
                         ? static_cast<double>(r.decode_tokens) / static_cast<double>(r.mtp_rounds)
                         : 0.0);
      }
    }
  };

  if (!args.prompt.empty()) {
    run_one_user_turn(args.prompt);
    return 0;
  }

  std::string line;
  std::cout << "> " << std::flush;
  while (std::getline(std::cin, line)) {
    if (!line.empty()) run_one_user_turn(line);
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
