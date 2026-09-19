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
  std::string generated_text;
  double prefill_seconds = 0.0;
  double decode_seconds = 0.0;
  int64_t prefill_tokens = 0;
  int64_t decode_tokens = 0;
  bool hit_eos = false;
};

TurnResult RunTurn(r4dx::model::Model& model, const r4dx::Tokenizer& tok,
                    const std::vector<int32_t>& new_tokens, const CliArgs& args) {
  TurnResult result;
  result.prefill_tokens = static_cast<int64_t>(new_tokens.size());

  const auto t0 = Clock::now();
  std::vector<float> logits = model.Prefill(new_tokens);
  const auto t1 = Clock::now();
  result.prefill_seconds = Seconds(t0, t1);

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

  const auto d0 = Clock::now();
  for (int64_t step = 0; step < args.max_tokens; ++step) {
    const int32_t next = r4dx::kernels::Sample(logits.data(), static_cast<int64_t>(logits.size()),
                                                sp, rng);
    if (is_eos(next)) { result.hit_eos = true; break; }
    result.generated_tokens.push_back(next);
    const std::string piece = decoder.push(next);
    if (!piece.empty()) {
      std::cout << piece << std::flush;
      result.generated_text += piece;
    }
    logits = model.DecodeStep(next);
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

    fed_tokens = full_tokens;
    fed_tokens.insert(fed_tokens.end(), r.generated_tokens.begin(), r.generated_tokens.end());
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
