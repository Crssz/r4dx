// r4dx::cli::CliArgs / ParseArgs -- argument parsing for r4dx-cli, split into its own header (out
// of main.cpp) purely so tests/cli/test_args.cpp can exercise it without linking a HIP-dependent
// executable. Header-only: this is a couple dozen lines of string parsing, not worth a .cpp.
#pragma once

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace r4dx::cli {

// Real ceiling for --mtp: Model::VerifyWindow (src/model/model.cpp) requires `candidates.size()
// <= mtp_draft_k_+1`, and every candidate batch goes through the same <=max_chunk_(64)-row chunk
// path RunChunk uses -- so mtp_draft_k_+1 <= 64, i.e. mtp <= 63. Duplicated in
// src/server/server_args.h (kept in sync manually -- these two headers are already a deliberate,
// documented duplication of every other --mtp* flag's parsing/validation, not a new pattern).
inline constexpr int64_t kMaxMtpDraftK = 63;

struct CliArgs {
  std::string model_path;
  std::string layout = "bf16";
  std::string tokenizer_dir = "C:\\AI\\models\\Qwen3.8-27B";
  std::string prompt;
  bool chat = false;
  std::string system_prompt;
  bool thinking = false;
  int64_t max_tokens = 128;
  float temperature = 1.0f;
  int top_k = 0;
  float top_p = 1.0f;
  float min_p = 0.0f;
  uint64_t seed = 0;
  // docs/r9700.md R13 (2026-09-20, measured): the checkpoint's own config.json declares
  // max_position_embeddings=262144 with rope_type "default" (no scaling trick needed) -- 262144
  // positions are natively in-distribution for this model. The old 131072 default was a
  // self-imposed cap at half the model's real capability, inherited from an early design decision
  // and never revisited. Real hardware measurement (this same pass) shows the full 262144-token
  // paged KV cache + GDN state costs 8.34 GiB (w4a16/w4a8) / 8.28 GiB (mxfp4) against a 31.86 GiB
  // card that already holds ~15.5 GiB of weights (embedding mirror included) -- total 23.9-24.1
  // GiB used, ~7.75 GiB (24%) still free at `--mtp 0`, ~6.83 GiB (21%) free at `--mtp 3`. Real
  // end-to-end generation at 262144 tokens of actual prefilled context (haiku prompt + a needle
  // retrieval fact, both padded with real corpus text) produced coherent, correct output --
  // confirmed on real hardware, not capacity arithmetic. See docs/r9700.md's R13/Q17 entries and
  // docs/perf.md's "Long-context validation" section for the full measurement.
  int64_t max_ctx = 262144;
  bool stats = false;
  // tools/profile pass (2026-09-19): profiles exactly ONE decode step (Model::DecodeStepProfiled,
  // hipEvent-timed per op-family) right after the prompt's prefill, prints a table to stderr, then
  // continues generation as normal -- see main.cpp's RunTurn and docs/perf.md's profile table.
  bool profile = false;
  // Milestone 3 profiling pass (docs/r9700.md R5/Q2): which GENERATED decode step (1-indexed) to
  // profile when --profile is set. Default 32 -- Q2's own finding was that profiling the FIRST
  // generated token overstates the step by a near-constant +5.4 +/- 0.5 ms of first-token/warmup
  // effects vs the steady-state throughput-table step; main.cpp's RunTurn now runs
  // (profile_token - 1) plain DecodeStepGreedy warmup steps before calling DecodeStepProfiled on
  // step `profile_token`, so this flag lets a caller pick a different steady-state token if needed
  // (e.g. --profile-token 1 reproduces the OLD, first-token behavior for comparison).
  int64_t profile_token = 32;
  // Milestone 3 profiling pass (docs/r9700.md R5/Q7): profile PREFILL instead of decode --
  // Model::PrefillProfiled runs the whole prompt (chunked at max_chunk, today 64) with per-kernel
  // hipEvent spans, prints the per-op-family table (same "gemm:" GEMM-share split as --profile) to
  // stderr, and returns without generating anything, same "standalone diagnostic" contract as
  // --profile. Combine with a long --prompt (docs/r9700.md Q7 asks for ~1024 tokens) to get a
  // meaningful multi-chunk picture; mutually exclusive with --profile (checked below).
  bool profile_prefill = false;
  // MTP self-speculative decode (docs/mtp.md): draft this many tokens per step via the container's
  // mtp.* head, verify them against the real model in one batched call. Defaults to 0 (disabled --
  // r4dx::model::Model::DecodeStepGreedy/DecodeStep, byte-for-byte unchanged from pre-MTP
  // behavior): most containers on disk (including every one Milestone 1/the perf pass produced)
  // have no mtp.* weights, so a nonzero default would break --model pointed at any of them; the
  // task's own "K default 3" is the recommended value to pass explicitly once a --model container
  // was converted with --mtp on, not this flag's own default. Only used by --temperature 0
  // (greedy) generation -- see main.cpp's RunTurn. Ceiling: 63 (kMaxMtpDraftK below) -- Model's
  // VerifyWindow batches `mtp+1` candidates through the same <=64-row chunk path RunChunk uses
  // (model.h's max_chunk_), so mtp+1 <= 64. VRAM cost is linear in K: the GDN window bank's
  // rolling depth is `conv_width-2+(1+mtp)`, so widening K costs real, undocumented-until-now
  // per-K device memory (measured ~2.3 GiB extra at K=16 against the real 64-layer container,
  // review finding 2026-09-20) on top of whatever K=0's own fixed state already reserves.
  int64_t mtp = 0;
  // MTP head layout (docs/mtp.md "MTP head layout"): "layout" (default, measured faster in 23/24
  // K x layout configurations with no acceptance-rate cost -- see docs/mtp.md's table) loads the
  // MTP head's four quantized linears (mtp.attn.qg/o, mtp.mlp.gate_up/down) in the same --layout
  // as the body; "bf16" instead forces them from the container's bf16 tensors regardless of
  // --layout (~0.5 GB extra VRAM, the exact-arithmetic form). No effect when --mtp is 0 or the
  // container has no mtp.* weights.
  std::string mtp_head_layout = "layout";
  // Reduced-vocab draft head (docs/r9700.md R9, model.h's ModelOptions::mtp_draft_reduced_vocab):
  // "reduced" uses the container's OPTIONAL mtp.draft_head.* tensors (a smaller lm_head over a
  // subset of the real vocabulary) to speed up drafting -- verification always stays full-vocab
  // regardless, so this cannot change generated output, only speed/acceptance at wide K.
  // "full" (default, flipped 2026-09-20 per Milestone 5 B2 item 8) forces the exact pre-R9
  // full-vocab draft head: M4's own K-sweep measured "reduced" SLOWER at its own best K than "full"
  // at ITS own best K (53.35 vs 65.02 tok/s, different K values each -- docs/mtp.md's K-sweep
  // table), and a later matched-K=3 re-measurement (review finding, 2026-09-20 -- the K-sweep
  // comparison above was NOT apples-to-apples) confirms the same conclusion at FIXED K: real
  // hardware, D:/models/r4dx/qwen38-27b-v3-draftvocab.r4dx, w4a16, docs/perf.md's standard
  // prompt/flags, `--mtp 3`, two runs each -- full head 68.20/68.64 tok/s (46.3% acceptance, 2.31
  // tok/round) vs reduced head 53.59/54.17 tok/s (20.9% acceptance, 1.63 tok/round), generated text
  // byte-identical either way (confirms the flip changes only speed, never correctness). "reduced"
  // is therefore not safe to default to until a larger/more diverse vocab-coverage calibration
  // corpus is measured (docs/mtp.md "Known gaps"). No effect when --mtp is 0, the container has no
  // mtp.* weights, or (silently) no draft_head.* tensors -- "reduced" degrades to the full-vocab
  // behavior automatically in that last case.
  std::string mtp_draft_head = "full";
  // Device-resident embedding gather (docs/mtp.md "device-resident draft loop", model.h's
  // ModelOptions::embed_device_resident): mirrors text.embed_tokens into VRAM (~2.37-2.54 GiB
  // depending on vocab/hidden) so decode/draft gathers on-device instead of a host memcpy+H2D per
  // step. Default true (matches ModelOptions' own default). This flag is the escape hatch
  // ModelOptions::embed_device_resident's own doc comment already promised but that no CLI/server
  // flag actually implemented until this fix (review finding, 2026-09-20) -- set to "off" to force
  // the host-gather path instead, e.g. to reclaim that VRAM for KV cache on a constrained run.
  std::string embed_device_resident = "on";
  // DFlash2 self-speculative decode (docs/dflash2.md, stage S3 -- model.h's
  // ModelOptions::dflash_container/dflash_draft_k): the block-diffusion self-speculation family,
  // mutually exclusive with --mtp (checked below -- Model::Load re-checks it too, so a caller
  // constructing ModelOptions directly cannot bypass either). Empty (default) disables DFlash2
  // entirely, byte-identical to before this flag existed. --dflash-k mirrors --mtp's own dual role
  // (both a Load()-time sizing knob via ModelOptions::dflash_draft_k and the per-round cap passed to
  // every Model::DecodeStepDflashGreedy call, exactly like args.mtp is passed to every
  // DecodeStepMtpGreedy call) -- see main.cpp's RunTurn.
  std::string dflash;
  // Ceiling 7: DFlash2's block is 8 wide with the anchor at position 0 (docs/dflash2.md), so the
  // walk can produce at most block_size-1 = 7 tokens regardless of what a caller asks for --
  // Model::Load also re-validates this against the ACTUAL container's own block_size (which every
  // shipped DFlash2 container sets to 8, but Model::Load never assumes that from this constant
  // alone).
  int64_t dflash_k = 7;
  // Selector-walk early-stop probability gate (docs/dflash2.md section 4.3): <=0 (default) disables
  // it, so the walk always runs to `dflash_k` (or fewer if `dflash_n_min` discards it).
  float dflash_p_min = 0.0f;
  // Selector-walk minimum-accepted-length discard gate (docs/dflash2.md section 4.3): <=0 (default)
  // disables it -- the whole draft is discarded (not just truncated) when the walk produced fewer
  // than this many tokens.
  int64_t dflash_n_min = 0;
};

// Thrown for a malformed/incomplete argument list (missing required flag, unrecognized flag, a
// value that fails to parse, or --prompt combined with --chat) -- ParseArgs never calls
// std::exit() itself, so it is safely unit-testable (see tests/cli/test_args.cpp) as well as
// callable from main() (which catches this and exits(2) with usage text).
struct CliUsageError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

inline std::string CliUsageText(const char* argv0) {
  return std::string("usage: ") + argv0 +
         " --model <container.r4dx> --layout {mxfp4|w4a16|w4a8|bf16} "
         "(--prompt \"...\" | --chat) [--tokenizer-dir <dir>] [--system \"...\"] "
         "[--think {on|off}] [--max-tokens N] [--temperature F] [--top-k N] [--top-p F] "
         "[--min-p F] [--seed N] [--max-ctx N] [--stats] [--profile] [--profile-token N] "
         "[--profile-prefill] [--mtp N] [--mtp-head-layout {bf16|layout}] "
         "[--mtp-draft-head {reduced|full}] [--embed-device-resident {on|off}] "
         "[--dflash <draft.r4dx>] [--dflash-k N] [--dflash-p-min F] [--dflash-n-min N]";
}

inline std::string NextCliArg(int argc, char** argv, int& i, const char* flag) {
  if (i + 1 >= argc) throw CliUsageError(std::string(flag) + " requires a value");
  return argv[++i];
}

// std::stoll/stof/stoull throw std::invalid_argument (not a number) / std::out_of_range (doesn't
// fit the target type) -- neither is a CliUsageError, so an unparseable value (e.g. "--max-tokens
// abc") would otherwise fall straight through ParseArgs uncaught, past main()'s CliUsageError
// handler, to the generic top-level catch: exit code 1 with a raw "invalid stoll argument" message
// and no usage text, contradicting this header's own doc comment above (CliUsageError is supposed
// to cover exactly this case). These wrappers convert both into CliUsageError so every malformed
// flag value goes through the same usage-text path.
template <typename T, typename ParseFn>
T ParseNumber(const std::string& flag, const std::string& value, ParseFn parse) {
  try {
    return parse(value);
  } catch (const std::invalid_argument&) {
    throw CliUsageError(flag + " expects a number, got '" + value + "'");
  } catch (const std::out_of_range&) {
    throw CliUsageError(flag + " value out of range: '" + value + "'");
  }
}

inline int64_t ParseI64(const std::string& flag, const std::string& value) {
  return ParseNumber<int64_t>(flag, value, [](const std::string& v) { return std::stoll(v); });
}
inline int ParseInt(const std::string& flag, const std::string& value) {
  return ParseNumber<int>(flag, value, [](const std::string& v) { return std::stoi(v); });
}
inline float ParseFloat(const std::string& flag, const std::string& value) {
  return ParseNumber<float>(flag, value, [](const std::string& v) { return std::stof(v); });
}
inline uint64_t ParseU64(const std::string& flag, const std::string& value) {
  return ParseNumber<uint64_t>(
      flag, value, [](const std::string& v) { return static_cast<uint64_t>(std::stoull(v)); });
}

inline CliArgs ParseArgs(int argc, char** argv) {
  CliArgs a;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--model") a.model_path = NextCliArg(argc, argv, i, "--model");
    else if (arg == "--layout") a.layout = NextCliArg(argc, argv, i, "--layout");
    else if (arg == "--tokenizer-dir") a.tokenizer_dir = NextCliArg(argc, argv, i, "--tokenizer-dir");
    else if (arg == "--prompt") a.prompt = NextCliArg(argc, argv, i, "--prompt");
    else if (arg == "--chat") a.chat = true;
    else if (arg == "--system") a.system_prompt = NextCliArg(argc, argv, i, "--system");
    else if (arg == "--think") a.thinking = NextCliArg(argc, argv, i, "--think") == "on";
    else if (arg == "--max-tokens") a.max_tokens = ParseI64("--max-tokens", NextCliArg(argc, argv, i, "--max-tokens"));
    else if (arg == "--temperature") a.temperature = ParseFloat("--temperature", NextCliArg(argc, argv, i, "--temperature"));
    else if (arg == "--top-k") a.top_k = ParseInt("--top-k", NextCliArg(argc, argv, i, "--top-k"));
    else if (arg == "--top-p") a.top_p = ParseFloat("--top-p", NextCliArg(argc, argv, i, "--top-p"));
    else if (arg == "--min-p") a.min_p = ParseFloat("--min-p", NextCliArg(argc, argv, i, "--min-p"));
    else if (arg == "--seed") a.seed = ParseU64("--seed", NextCliArg(argc, argv, i, "--seed"));
    else if (arg == "--max-ctx") a.max_ctx = ParseI64("--max-ctx", NextCliArg(argc, argv, i, "--max-ctx"));
    else if (arg == "--stats") a.stats = true;
    else if (arg == "--profile") a.profile = true;
    else if (arg == "--profile-token") a.profile_token = ParseI64("--profile-token", NextCliArg(argc, argv, i, "--profile-token"));
    else if (arg == "--profile-prefill") a.profile_prefill = true;
    else if (arg == "--mtp") a.mtp = ParseI64("--mtp", NextCliArg(argc, argv, i, "--mtp"));
    else if (arg == "--mtp-head-layout") a.mtp_head_layout = NextCliArg(argc, argv, i, "--mtp-head-layout");
    else if (arg == "--mtp-draft-head") a.mtp_draft_head = NextCliArg(argc, argv, i, "--mtp-draft-head");
    else if (arg == "--embed-device-resident") a.embed_device_resident = NextCliArg(argc, argv, i, "--embed-device-resident");
    else if (arg == "--dflash") a.dflash = NextCliArg(argc, argv, i, "--dflash");
    else if (arg == "--dflash-k") a.dflash_k = ParseI64("--dflash-k", NextCliArg(argc, argv, i, "--dflash-k"));
    else if (arg == "--dflash-p-min") a.dflash_p_min = ParseFloat("--dflash-p-min", NextCliArg(argc, argv, i, "--dflash-p-min"));
    else if (arg == "--dflash-n-min") a.dflash_n_min = ParseI64("--dflash-n-min", NextCliArg(argc, argv, i, "--dflash-n-min"));
    else if (arg == "--help" || arg == "-h") throw CliUsageError("help requested");
    else throw CliUsageError("unrecognized argument: " + arg);
  }
  if (a.model_path.empty()) throw CliUsageError("--model is required");
  if (a.prompt.empty() && !a.chat) throw CliUsageError("one of --prompt or --chat is required");
  if (!a.prompt.empty() && a.chat) throw CliUsageError("--prompt and --chat are mutually exclusive");
  // Reject nonsensical numeric values outright rather than passing them to the model, where they
  // would surface (if at all) as a confusing HIP/kernel-level failure far from the actual mistake.
  if (a.max_tokens < 0) throw CliUsageError("--max-tokens must be >= 0");
  if (a.top_p < 0.0f || a.top_p > 1.0f) throw CliUsageError("--top-p must be in [0, 1]");
  if (a.min_p < 0.0f || a.min_p > 1.0f) throw CliUsageError("--min-p must be in [0, 1]");
  if (a.max_ctx <= 0) throw CliUsageError("--max-ctx must be > 0");
  if (a.top_k < 0) throw CliUsageError("--top-k must be >= 0");
  if (a.mtp < 0 || a.mtp > kMaxMtpDraftK) {
    throw CliUsageError("--mtp must be in [0, " + std::to_string(kMaxMtpDraftK) +
                         "] (Model::VerifyWindow batches mtp+1 candidates through a <=64-row "
                         "chunk)");
  }
  if (a.mtp_head_layout != "bf16" && a.mtp_head_layout != "layout") {
    throw CliUsageError("--mtp-head-layout must be 'bf16' or 'layout'");
  }
  if (a.mtp_draft_head != "reduced" && a.mtp_draft_head != "full") {
    throw CliUsageError("--mtp-draft-head must be 'reduced' or 'full'");
  }
  if (a.profile_token < 1) throw CliUsageError("--profile-token must be >= 1");
  if (a.embed_device_resident != "on" && a.embed_device_resident != "off") {
    throw CliUsageError("--embed-device-resident must be 'on' or 'off'");
  }
  if (a.profile && a.profile_prefill) {
    throw CliUsageError("--profile and --profile-prefill are mutually exclusive");
  }
  if (!a.dflash.empty() && a.mtp > 0) {
    throw CliUsageError("--dflash and --mtp are mutually exclusive (docs/dflash2.md: DFlash2 and "
                         "MTP are separate self-speculation families, not combinable)");
  }
  if (!a.dflash.empty() && (a.dflash_k < 1 || a.dflash_k > 7)) {
    throw CliUsageError("--dflash-k must be in [1, 7] (DFlash2's block is 8 wide: anchor + up to "
                         "block_size-1 drafted tokens)");
  }
  // Review finding (2026-09-21): DecodeStepProfiled/PrefillProfiled do not call
  // CaptureDflashLayerInput/DflashDraft::InjectFeatures, but both still advance Model::pos_ --
  // with --dflash attached this desyncs pos_ from DflashDraft::InjectedCount(), and the very next
  // real decode step throws "start_pos != InjectedCount()". Cheaper to reject the combination at
  // parse time (matches --profile's existing "standalone diagnostic" framing) than to thread
  // capture+inject into both profiled loops for a diagnostic that already doesn't measure the
  // --mtp path's own drafter cost either.
  if (!a.dflash.empty() && (a.profile || a.profile_prefill)) {
    throw CliUsageError("--profile/--profile-prefill and --dflash are mutually exclusive "
                         "(the profiled step/chunk loops do not feed the DFlash2 drafter, which "
                         "would desync Model::pos_ from the drafter's own injected-row count)");
  }
  return a;
}

}  // namespace r4dx::cli

