// tests/model/tool_tp_soak.cpp -- the tensor-parallel soak (docs/tp.md 10.5, gate G8).
//
// ONE TpModel (--tp-mode real by default: both GPUs) for the whole run. Each iteration: pick a random
// slice (--min-prompt..--max-prompt tokens, default 16..2048) of the kl_corpus token pool (every
// segment of --tokens, concatenated, wrapping), Reset(), Prefill, then a random 32..512 decode
// tokens in a random mode -- greedy (DecodeStepGreedy), seeded sampled (DecodeStepSampled; T 0.7,
// top_k 20, top_p 0.8) or full-row (DecodeStep + host argmax: the vocab gather). Every
// --canary-every-th iteration (and the first) re-runs a fixed canary: the pool's first 512 tokens,
// then 128 greedy tokens, which must equal the first canary run exactly. With --dflash <drafter>
// (docs/tp.md 10.5, P5) two more modes join the mix: DFlash2 greedy rounds and seeded sampled rounds
// (DecodeStepDflash{Greedy,Sampled}, k = --dflash-k, p_min / n_min off), each until at least the
// iteration's decode length has been emitted; the canary stays plain greedy. Without --dflash the
// random sequence of iterations is the pre-P5 one.
//
// One JSON object per line is appended to --json (a "start" line, a "loaded" line, one "iter" line
// per iteration, an "error" line on failure, a "summary" line at the end and a "teardown" line once
// the TpModel is destroyed), each flushed to disk (_commit) before the next iteration starts, so a
// crash -- a TDR can take the display driver down with it -- still leaves every finished iteration
// on disk.
//
// On ANY TpModel error the soak stops at once: it logs the error and exits 1 without calling Reset()
// (a recovery would hide the failure the soak exists to find). Pass (exit 0): every canary equal,
// 0 aborts, per-rank buffer drift <= 64 MiB from right after load to the end, and
// core::g_tp_collective_allocs == 0 (docs/tp.md 2.7). The drift is this process's live DeviceBuffer
// bytes on each rank's device (core::DeviceBufferBytes), not hipMemGetInfo's device-wide used bytes,
// which on HIP device 0 move with the desktop; those are logged next to it (Appendix B N64). Exit 2:
// a canary mismatch; exit 3: another pass criterion failed. The TDR watch (WER LiveKernelEvent 141 /
// System 4101, polled while the soak runs) is tools/tp/soak.ps1's, which wraps this tool.
//
// Pre-flight (docs/tp.md 9.2): before loading, a helper thread (the main thread is the TpModel
// facade and makes no HIP call, 2.1) requires --need-gib free on every device the ranks will use.
//
// HYBRID SOAK (--pp 2; docs/pp-tp2-hybrid.md 9 P4, tools/hybrid/soak.ps1): the same loop on the hybrid serving mode,
// `TpModel::Load(opts, tp, pp)` = --tp 2 --pp 2 (real mode only). The prompts are long enough to engage the pipelined prefill
// (defaults under --pp: --max-ctx 16384, --max-prompt 8192, so about 90 % of the prompts are >= --pp-min-rows = 1024 rows), the mode
// mix gains "turns" (and with --dflash "turns_dflash"): a conversation of 2-3 prompts, each followed by a decode, whose later tails are
// 1..4096 rows (clipped to --max-ctx) -- the WARM calls: the TP-master gather, the speculative-window collapse, the DFlash tail on top of
// a decoded state. There is no TP=1 reference in the process (the two stage Models and the ranks do not leave room for a third Model),
// so the comparisons are:
//   * canaries: the TP canary above (512 rows: the TP path), plus a HYBRID canary -- a 2-turn conversation (1536 rows, 48 greedy tokens,
//     a 1100-row tail, 48 more) that must engage the pipeline twice (cold + warm) and, with --dflash, a second one decoded in DFlash2
//     greedy rounds. Each is repeated every --canary-every iterations among all the other work and must equal its first run token for
//     token (stale stage mirror, leaked state, a wrong gather after unrelated traffic all show up here);
//   * --tp2-ref-every N (default 0 = off): every N-th greedy / turns iteration is also generated with the pipelining switched off
//     (TpModel::SetHybridMinRows(huge): the ordinary TP=2 prefill) and the two token streams are compared. They are NOT equal in general
//     (different prefill numerics, the KL floor of docs/tp.md; greedy streams diverge at the first flip and stay apart), so this is a
//     gross-corruption detector: over >= 4 compared iterations the soak fails when more than half differ at the very first token, or
//     the mean agreeing prefix is below 10 % of the compared length;
//   * a hybrid soak must have pipelined (>= 1 pipelined call; the summary carries the whole `hybrid:` counter set) and the hybrid must
//     have stayed engaged (no "hybrid mode OFF" fallback at load).
// --pp-verify has no meaning here (the hybrid has no peer mirror to digest, tp_model_hybrid.cpp ignores it): the canaries and the TP=2
// reference are the hybrid's G-H4 comparison. The extra iteration-line fields (hy_*) and the summary's "hybrid" object are the counters.
//
// Built, never add_test()'d. Usage (HIP_VISIBLE_DEVICES unset, production server stopped):
//   tool_tp_soak --model <R4DX_MODELS_ROOT>\r4dx\huihui-qwen38-27b-abl-trellis-mix45m.r4dx --layout trellis --minutes 60
//                --max-ctx 8192 --json build\logs\tp_soak.jsonl [--seed 1] [--iterations N]
//                [--tokens tools\reference\kl_corpus\tokens.json] [--canary-every 10]
//                [--min-prompt 16] [--max-prompt 2048] [--min-decode 32] [--max-decode 512]
//                [--tp-mode real|emulate] [--tp-devices a,b] [--tp-ar-timeout-ms 500]
//                [--tp-submit-layers N] [--tp-max-inflight K] [--need-gib 14] [--layers N]
//                [--dflash <drafter.r4dx> [--dflash-k 7]]
//                [--pp 2 [--pp-split N] [--pp-min-rows N] [--hybrid-ctx N] [--hybrid-reserve-gib X] [--tp2-ref-every N]]
#include <hip/hip_runtime.h>
#include <io.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <fstream>

#include "model.h"  // also nlohmann/json.hpp, via model_config.h
#include "r4dx/core/tp_alloc_guard.hpp"
#include "r4dx/kernels/sampler.hpp"
#include "test_common.h"
#include "tp_model.h"

namespace {

using Clock = std::chrono::steady_clock;
using r4dx::model::ModelOptions;
using r4dx::model::TpModel;
using r4dx::model::TpOptions;
namespace core = r4dx::core;
namespace kernels = r4dx::kernels;

constexpr int kCanaryPrompt = 512;
constexpr int kCanaryDecode = 128;
// The hybrid canary (--pp 2): turn 1 = the pool's first 1536 tokens, turn 2 = the next 1100; both are above the hybrid's 1024-row threshold, so
// both calls are pipelined (the second one warm), with kHybridCanaryDecode greedy tokens after each.
constexpr int kHybridCanaryTurn1 = 1536;
constexpr int kHybridCanaryTurn2 = 1100;
constexpr int kHybridCanaryRows = kHybridCanaryTurn1 + kHybridCanaryTurn2;
constexpr int kHybridCanaryDecode = 48;
constexpr double kMaxVramDriftMiB = 64.0;

struct Args {
  std::string model = r4dx_test::ProductionTargetPath();
  std::string layout = r4dx_test::ProductionLayoutName();
  std::string tokens = "tools/reference/kl_corpus/tokens.json";
  std::string json;
  double minutes = 60.0;
  int64_t iterations = 0;  // 0 = until --minutes
  int64_t max_ctx = 8192;
  int64_t layers = -1;
  uint64_t seed = 1;
  int canary_every = 10;
  int min_prompt = 16, max_prompt = 2048, min_decode = 32, max_decode = 512;
  std::string tp_mode = "real";
  std::vector<int> tp_devices;
  int tp_ar_timeout_ms = 500;
  int tp_submit_layers = -1, tp_max_inflight = -1;  // -1: TpOptions' default
  double need_gib = 14.0;
  std::string dflash;   // empty: no drafter, the pre-P5 mode mix
  int64_t dflash_k = 7;
  // ---- the hybrid soak (--pp 2, see the header) ----
  int pp = 1;                      // 2: TpModel::Load(opts, tp, pp) with --tp 2 --pp 2
  int pp_split = 0;                // --pp-split (0 = auto: 29, 30 with a drafter)
  int pp_min_rows = -1;            // --pp-min-rows (-1 = the hybrid's 1024)
  int64_t hybrid_ctx = 0;          // --hybrid-ctx (0 = auto)
  double hybrid_reserve_gib = -1;  // --hybrid-reserve-gib (< 0 = built-in)
  int tp2_ref_every = 0;           // every N-th greedy / turns iteration is also run with the pipelining off (0 = never)
  bool ctx_given = false, max_prompt_given = false, need_given = false;
};

[[noreturn]] void Usage(const std::string& why) {
  std::fprintf(stderr,
               "tool_tp_soak: %s\nusage: tool_tp_soak [--model <c.r4dx>] [--layout w4a16] [--minutes 60] "
               "[--iterations N] [--max-ctx 8192] [--seed 1] [--json <log.jsonl>] [--tokens <tokens.json>] "
               "[--canary-every 10] [--min-prompt 16] [--max-prompt 2048] [--min-decode 32] [--max-decode 512] "
               "[--tp-mode real|emulate] [--tp-devices a,b] [--tp-ar-timeout-ms N] [--tp-submit-layers N] "
               "[--tp-max-inflight K] [--need-gib X] [--layers N] [--dflash <drafter.r4dx> [--dflash-k 7]] "
               "[--pp 2 [--pp-split N] [--pp-min-rows N] [--hybrid-ctx N] [--hybrid-reserve-gib X] [--tp2-ref-every N]]\n",
               why.c_str());
  std::exit(1);
}

Args Parse(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    const std::string s = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) Usage(s + " needs a value");
      return argv[++i];
    };
    try {
      if (s == "--model") a.model = next();
      else if (s == "--layout") a.layout = next();
      else if (s == "--tokens") a.tokens = next();
      else if (s == "--json") a.json = next();
      else if (s == "--minutes") a.minutes = std::stod(next());
      else if (s == "--iterations") a.iterations = std::stoll(next());
      else if (s == "--max-ctx") { a.max_ctx = std::stoll(next()); a.ctx_given = true; }
      else if (s == "--layers") a.layers = std::stoll(next());
      else if (s == "--seed") a.seed = std::stoull(next());
      else if (s == "--canary-every") a.canary_every = std::stoi(next());
      else if (s == "--min-prompt") a.min_prompt = std::stoi(next());
      else if (s == "--max-prompt") { a.max_prompt = std::stoi(next()); a.max_prompt_given = true; }
      else if (s == "--min-decode") a.min_decode = std::stoi(next());
      else if (s == "--max-decode") a.max_decode = std::stoi(next());
      else if (s == "--tp-mode") a.tp_mode = next();
      else if (s == "--tp-devices") {
        std::stringstream ss(next());
        std::string item;
        while (std::getline(ss, item, ',')) a.tp_devices.push_back(std::stoi(item));
      } else if (s == "--tp-ar-timeout-ms") a.tp_ar_timeout_ms = std::stoi(next());
      else if (s == "--tp-submit-layers") a.tp_submit_layers = std::stoi(next());
      else if (s == "--tp-max-inflight") a.tp_max_inflight = std::stoi(next());
      else if (s == "--need-gib") { a.need_gib = std::stod(next()); a.need_given = true; }
      else if (s == "--dflash") a.dflash = next();
      else if (s == "--dflash-k") a.dflash_k = std::stoll(next());
      else if (s == "--pp") a.pp = std::stoi(next());
      else if (s == "--pp-split") a.pp_split = std::stoi(next());
      else if (s == "--pp-min-rows") a.pp_min_rows = std::stoi(next());
      else if (s == "--hybrid-ctx") a.hybrid_ctx = std::stoll(next());
      else if (s == "--hybrid-reserve-gib") a.hybrid_reserve_gib = std::stod(next());
      else if (s == "--tp2-ref-every") a.tp2_ref_every = std::stoi(next());
      else Usage("unknown argument " + s);
    } catch (const std::exception&) {
      Usage("bad value for " + s);
    }
  }
  if (a.json.empty()) Usage("--json <log.jsonl> is required (the per-iteration log)");
  if (a.tp_mode != "real" && a.tp_mode != "emulate") Usage("--tp-mode must be real or emulate");
  if (a.pp != 1 && a.pp != 2) Usage("--pp must be 1 or 2 (2 = the hybrid serving mode, --tp 2 --pp 2)");
  if (a.pp == 1 && (a.pp_split != 0 || a.pp_min_rows >= 0 || a.hybrid_ctx != 0 || a.hybrid_reserve_gib >= 0 || a.tp2_ref_every != 0)) {
    Usage("--pp-split, --pp-min-rows, --hybrid-ctx, --hybrid-reserve-gib and --tp2-ref-every need --pp 2");
  }
  if (a.pp == 2) {
    if (a.tp_mode != "real") Usage("--pp 2 (the hybrid) needs --tp-mode real");
    if (a.pp_split < 0 || a.hybrid_ctx < 0 || a.hybrid_reserve_gib > 24.0 || a.tp2_ref_every < 0) {
      Usage("--pp-split and --hybrid-ctx must be >= 0, --hybrid-reserve-gib <= 24, --tp2-ref-every >= 0");
    }
    // The hybrid's stage-KV capacity S must be >= 16384 tokens (docs/pp-tp2-hybrid.md 2), and its prompts must be long enough to engage it.
    if (!a.ctx_given) a.max_ctx = 16384;
    if (!a.max_prompt_given) a.max_prompt = 8192;
    if (!a.need_given) a.need_gib = 20.0;  // two rank Models + two stage Models per card: far above the plain 14
  }
  if (!(a.minutes > 0) && a.iterations <= 0) Usage("--minutes must be > 0 (or give --iterations)");
  if (a.canary_every < 1) Usage("--canary-every must be >= 1");
  if (a.min_prompt < 1 || a.max_prompt < a.min_prompt || a.min_decode < 1 || a.max_decode < a.min_decode) {
    Usage("need 1 <= --min-prompt <= --max-prompt and 1 <= --min-decode <= --max-decode");
  }
  // A DFlash2 generation may run past its decode length by up to one round (k tokens) and verifies
  // a k+1-row window beyond that.
  const int64_t spec_slack = a.dflash.empty() ? 0 : 2 * (a.dflash_k + 1);
  if (a.max_prompt + a.max_decode + spec_slack > a.max_ctx || kCanaryPrompt + kCanaryDecode > a.max_ctx) {
    Usage("--max-prompt + --max-decode (+ 2 * (--dflash-k + 1) with --dflash, and the 640-position canary) must "
          "fit in --max-ctx");
  }
  if (!a.dflash.empty() && (a.dflash_k < 1 || a.dflash_k > 7)) Usage("--dflash-k must be in [1, 7]");
  // The hybrid's "turns" modes feed up to three prompts with a decode after each, and its canaries two prompts (1536 + 1100 rows).
  if (a.pp == 2 && (a.max_prompt + 3 * (a.max_decode + spec_slack) + 16 > a.max_ctx || kHybridCanaryRows + 2 * kHybridCanaryDecode + 2 * (spec_slack + 8) > a.max_ctx)) {
    Usage("--pp 2: --max-prompt + 3 * (--max-decode + slack) (a 3-turn conversation) and the 2636-row hybrid canary must fit in --max-ctx");
  }
  if (a.tp_submit_layers < -1 || a.tp_submit_layers > 64 || a.tp_max_inflight < -1 || a.tp_max_inflight > 64) {
    Usage("--tp-submit-layers and --tp-max-inflight must be in [0, 64]");
  }
  return a;
}

// ---- the log ----------------------------------------------------------------------------------

class JsonLog {
 public:
  explicit JsonLog(const std::string& path) {
    if (fopen_s(&f_, path.c_str(), "ab") != 0) f_ = nullptr;
    if (f_ == nullptr) throw std::runtime_error("cannot open " + path + " for appending");
  }
  ~JsonLog() {
    if (f_ != nullptr) std::fclose(f_);
  }
  // One line, flushed through the CRT and the OS cache before returning.
  void Line(const std::string& obj) {
    std::fwrite(obj.data(), 1, obj.size(), f_);
    std::fputc('\n', f_);
    std::fflush(f_);
    (void)_commit(_fileno(f_));
  }

 private:
  FILE* f_ = nullptr;
};

std::string Str(const std::string& s) {
  std::string o = "\"";
  for (unsigned char c : s) {
    if (c == '"') o += "\\\"";
    else if (c == '\\') o += "\\\\";
    else if (c == '\n') o += "\\n";
    else if (c < 0x20) {
      char b[8];
      std::snprintf(b, sizeof b, "\\u%04x", c);
      o += b;
    } else o += static_cast<char>(c);
  }
  return o + "\"";
}
std::string Num(double v) {
  if (!std::isfinite(v)) return "null";
  char b[64];
  std::snprintf(b, sizeof b, "%.6g", v);
  return b;
}
template <class T>
std::string Arr(const std::vector<T>& v) {
  std::string o = "[";
  for (size_t i = 0; i < v.size(); ++i) o += (i ? "," : "") + Num(static_cast<double>(v[i]));
  return o + "]";
}

// ---- pre-flight (docs/tp.md 9.2), on a helper thread ------------------------------------------

// Returns "" when every device has `need_gib` free, else the message to print.
std::string Preflight(const std::vector<int>& devices_in, bool real, double need_gib) {
  std::string msg;
  std::thread t([&] {
    int n = 0;
    if (hipGetDeviceCount(&n) != hipSuccess) n = 0;
    std::vector<int> devices = devices_in;
    if (devices.empty()) {
      if (real && n >= 2) devices = {n - 1, n - 2};
      else if (n >= 1) devices = {n - 1};
    }
    if (devices.empty()) msg = "no visible HIP device";
    for (int d : devices) {
      size_t free_b = 0, total_b = 0;
      if (d < 0 || d >= n || hipSetDevice(d) != hipSuccess || hipMemGetInfo(&free_b, &total_b) != hipSuccess) {
        msg = "cannot query HIP device " + std::to_string(d);
        break;
      }
      const double free_gib = static_cast<double>(free_b) / (1024.0 * 1024.0 * 1024.0);
      if (free_gib < need_gib) {
        char b[256];
        std::snprintf(b, sizeof b, "need %.1f GiB free on HIP device %d, have %.2f GiB -- is the production server running?",
                      need_gib, d, free_gib);
        msg = b;
        break;
      }
    }
    (void)hipGetLastError();
  });
  t.join();
  return msg;
}

// ---- one generation -----------------------------------------------------------------------------

// The first three modes are the pre-P5 mix (drawn from [0, 2]); the DFlash2 ones join it only with
// --dflash (drawn from [0, 4]); --pp 2 appends kTurns (and with --dflash kTurnsDflash): a conversation
// of 2-3 prompts with a greedy / DFlash2 greedy decode after each (the hybrid's WARM calls).
enum class Mode { kGreedy, kSampled, kFullRow, kDflashGreedy, kDflashSampled, kTurns, kTurnsDflash };
const char* ModeName(Mode m) {
  switch (m) {
    case Mode::kGreedy: return "greedy";
    case Mode::kSampled: return "sampled";
    case Mode::kFullRow: return "full_row";
    case Mode::kDflashGreedy: return "dflash_greedy";
    case Mode::kDflashSampled: return "dflash_sampled";
    case Mode::kTurns: return "turns";
    case Mode::kTurnsDflash: return "turns_dflash";
  }
  return "?";
}
bool IsDflashMode(Mode m) { return m == Mode::kDflashGreedy || m == Mode::kDflashSampled || m == Mode::kTurnsDflash; }
bool IsSampledMode(Mode m) { return m == Mode::kSampled || m == Mode::kDflashSampled; }
// Modes whose output the TP=2 reference (--tp2-ref-every) can be compared with token by token: greedy ones.
bool IsGreedyMode(Mode m) { return m == Mode::kGreedy || m == Mode::kDflashGreedy || m == Mode::kTurns || m == Mode::kTurnsDflash; }

struct Gen {
  std::vector<int32_t> tokens;
  double prefill_s = 0, decode_s = 0;  // summed over the turns
  int64_t rounds = 0;  // DFlash2 modes: speculative rounds run
  int64_t emitted = 0;  // tokens past the one each turn's prefill produced (== the decode length except for DFlash2 rounds)
  std::vector<size_t> turn_ends;  // tokens.size() after each turn
};

// One conversation: Reset(), then per turn a Prefill of `parts[i]` (the first is a cold prompt, the rest are warm tails continuing the
// state the previous turn's decode left) and `decode` tokens. A single part is exactly the pre-hybrid generation.
Gen Generate(TpModel& m, const std::vector<std::vector<int32_t>>& parts, int decode, Mode mode, uint64_t sample_seed,
             int64_t dflash_k = 0) {
  Gen g;
  const int64_t V = m.Config().vocab_size;
  kernels::SampleParams sp;
  sp.temperature = 0.7f;
  sp.top_k = 20;
  sp.top_p = 0.8f;
  sp.seed = sample_seed;
  std::mt19937_64 rng = kernels::MakeRng(sample_seed);
  m.Reset();
  const bool dflash = IsDflashMode(mode);
  const bool sampled = IsSampledMode(mode);
  for (const std::vector<int32_t>& prompt : parts) {
    const size_t base = g.tokens.size();
    const auto t0 = Clock::now();
    const std::vector<float> logits = m.Prefill(prompt);
    int32_t next = sampled ? kernels::Sample(logits.data(), V, sp, rng) : kernels::Argmax(logits.data(), V);
    const auto t1 = Clock::now();
    if (dflash) {
      // Whole rounds until at least `decode` tokens past the first one were emitted; every round's
      // last token is the next round's (not yet committed) anchor, like `next` above.
      g.tokens.push_back(next);
      while (static_cast<int>(g.tokens.size() - base) <= decode) {
        const std::vector<int32_t> r = !sampled ? m.DecodeStepDflashGreedy(next, dflash_k, 0.0f, 0)
                                                : m.DecodeStepDflashSampled(next, dflash_k, 0.0f, 0, sp, rng);
        g.tokens.insert(g.tokens.end(), r.begin(), r.end());
        next = r.back();
        ++g.rounds;
      }
    } else {
      for (int i = 0; i < decode; ++i) {
        g.tokens.push_back(next);
        if (mode == Mode::kGreedy || mode == Mode::kTurns) {
          next = m.DecodeStepGreedy(next);
        } else if (mode == Mode::kSampled) {
          next = m.DecodeStepSampled(next, sp, rng);
        } else {
          const std::vector<float> row = m.DecodeStep(next);
          next = kernels::Argmax(row.data(), V);
        }
      }
      g.tokens.push_back(next);
    }
    g.prefill_s += std::chrono::duration<double>(t1 - t0).count();
    g.decode_s += std::chrono::duration<double>(Clock::now() - t1).count();
    g.turn_ends.push_back(g.tokens.size());
  }
  g.emitted = static_cast<int64_t>(g.tokens.size()) - static_cast<int64_t>(parts.size());
  return g;
}
Gen Generate(TpModel& m, const std::vector<int32_t>& prompt, int decode, Mode mode, uint64_t sample_seed,
             int64_t dflash_k = 0) {
  return Generate(m, std::vector<std::vector<int32_t>>{prompt}, decode, mode, sample_seed, dflash_k);
}

struct Snapshot {
  // Per rank: this process's live DeviceBuffer bytes on the rank's device (the drift gate), and
  // hipMemGetInfo's device-wide used bytes -- which on HIP device 0 include the desktop, so they are
  // logged for information only (docs/tp.md Appendix B N64).
  std::vector<double> buffers_gib, vram_used_gib;
  std::vector<core::TpCommStats> comm;
  std::vector<r4dx::model::tp::SubmitBounder::Stats> submit;
};
Snapshot Snap(TpModel& m) {
  Snapshot s;
  for (const r4dx::model::VramReport& v : m.Vram()) {
    s.buffers_gib.push_back(v.buffers_gib);
    s.vram_used_gib.push_back(v.used_gib);
  }
  s.comm = m.CommStats();
  s.submit = m.SubmitStats();
  return s;
}

// Per rank, in MiB: `now - first` of the buffer bytes (buffers = true) or of the device-wide used.
std::vector<double> DriftMiB(const Snapshot& now, const Snapshot& first, bool buffers) {
  const std::vector<double>& a = buffers ? now.buffers_gib : now.vram_used_gib;
  const std::vector<double>& b = buffers ? first.buffers_gib : first.vram_used_gib;
  std::vector<double> d;
  for (size_t r = 0; r < a.size() && r < b.size(); ++r) d.push_back((a[r] - b[r]) * 1024.0);
  return d;
}

std::string CommJson(const Snapshot& s) {
  std::vector<double> ch0, ch1, hx, wait, aborts, units, cap_waits, cap_wait_max;
  for (const core::TpCommStats& c : s.comm) {
    ch0.push_back(static_cast<double>(c.ar_calls[0]));
    ch1.push_back(static_cast<double>(c.ar_calls[1]));
    hx.push_back(static_cast<double>(c.host_exchanges));
    wait.push_back(c.host_exchange_wait_us_max);
    aborts.push_back(static_cast<double>(c.aborts));
  }
  for (const auto& b : s.submit) {
    units.push_back(static_cast<double>(b.units));
    cap_waits.push_back(static_cast<double>(b.waits));
    cap_wait_max.push_back(b.wait_us_max);
  }
  return "\"buffers_gib\":" + Arr(s.buffers_gib) + ",\"vram_used_gib\":" + Arr(s.vram_used_gib) +
         ",\"ar_calls_ch0\":" + Arr(ch0) + ",\"ar_calls_ch1\":" + Arr(ch1) +
         ",\"host_exchanges\":" + Arr(hx) + ",\"max_exchange_wait_us\":" + Arr(wait) + ",\"aborts\":" + Arr(aborts) +
         ",\"submit_units\":" + Arr(units) + ",\"cap_waits\":" + Arr(cap_waits) + ",\"max_cap_wait_us\":" +
         Arr(cap_wait_max) + ",\"collective_allocs\":" + std::to_string(core::g_tp_collective_allocs.load());
}

// The hybrid's counters (TpModel::GetHybridStats; zeros when it is not engaged) as the members of a JSON object: the iteration lines
// carry the cumulative call counts and the latest call's time, the summary the whole set.
std::string HybridJson(TpModel& m, bool full) {
  const TpModel::HybridStats s = m.GetHybridStats();
  std::string o = "\"engaged\":" + std::string(m.HybridEngaged() ? "true" : "false") + ",\"pipelined\":" +
                  std::to_string(s.pipelined_calls) + ",\"tp_prefill\":" + std::to_string(s.tp_prefill_calls) +
                  ",\"tail_rows\":" + std::to_string(s.tail_rows) + ",\"last_rows\":" + std::to_string(s.last_rows) +
                  ",\"last_p0\":" + std::to_string(s.last_p0) + ",\"last_total_ms\":" + Num(s.last_total_ms);
  if (!full) return o;
  std::vector<int64_t> declined(s.declined, s.declined + 6);
  const auto mib = [](uint64_t b) { return static_cast<double>(b) / (1024.0 * 1024.0); };
  return o + ",\"split\":" + std::to_string(m.HybridSplit()) + ",\"stage_ctx\":" + std::to_string(m.HybridStageCtx()) +
         ",\"min_rows\":" + std::to_string(m.HybridMinRows()) + ",\"declined\":" + Arr(declined) +
         ",\"gather_mib\":" + Arr(std::vector<double>{mib(s.gather_bytes[0]), mib(s.gather_bytes[1])}) +
         ",\"reshard_mib\":" + Arr(std::vector<double>{mib(s.reshard_bytes[0]), mib(s.reshard_bytes[1])}) +
         ",\"same_card_mib\":" + Num(mib(s.local_bytes)) + ",\"stats_line\":" + Str(m.HybridStatsLine());
}

// First-difference comparison of two token streams: the agreeing prefix length and the compared length.
struct Agree {
  size_t prefix = 0, of = 0;
};
Agree CompareTokens(const std::vector<int32_t>& x, const std::vector<int32_t>& y) {
  Agree r;
  r.of = std::min(x.size(), y.size());
  while (r.prefix < r.of && x[r.prefix] == y[r.prefix]) ++r.prefix;
  return r;
}

}  // namespace

int main(int argc, char** argv) {
  const Args a = Parse(argc, argv);
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  if (!r4dx_test::FileExists(a.model)) {
    std::fprintf(stderr, "tool_tp_soak: container not found: %s\n", a.model.c_str());
    return 1;
  }
  const bool real = a.tp_mode == "real";
  const std::string pre = Preflight(a.tp_devices, real, a.need_gib);
  if (!pre.empty()) {
    std::fprintf(stderr, "tool_tp_soak: %s\n", pre.c_str());
    return 1;
  }

  std::unique_ptr<JsonLog> log;
  try {
    log = std::make_unique<JsonLog>(a.json);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "tool_tp_soak: %s\n", e.what());
    return 1;
  }
  const auto wall0 = Clock::now();
  const auto since = [&] { return std::chrono::duration<double>(Clock::now() - wall0).count(); };
  const auto fail = [&](const char* what, const std::string& why, int64_t iter) {
    std::fprintf(stderr, "tool_tp_soak: %s at iteration %lld: %s\n", what, static_cast<long long>(iter), why.c_str());
    log->Line("{\"type\":\"error\",\"t_s\":" + Num(since()) + ",\"iter\":" + std::to_string(iter) + ",\"what\":" +
              Str(what) + ",\"error\":" + Str(why) + "}");
    return 1;
  };

  // The token pool: every kl_corpus segment's token_ids (tokens.json, the Rung 4 shared format --
  // tests/model/teacher_forced.h), concatenated.
  std::vector<int32_t> pool;
  try {
    std::ifstream f(a.tokens, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + a.tokens);
    nlohmann::json j;
    f >> j;
    for (const auto& s : j.at("segments")) {
      for (const auto& t : s.at("token_ids")) pool.push_back(t.get<int32_t>());
    }
  } catch (const std::exception& e) {
    return fail("tokens", e.what(), -1);
  }
  const bool hybrid = a.pp == 2;
  if (pool.size() < static_cast<size_t>(std::max(std::max(kCanaryPrompt, a.max_prompt), hybrid ? kHybridCanaryRows : 0))) {
    return fail("tokens", "the token pool has fewer tokens than --max-prompt / the canary", -1);
  }

  ModelOptions opts;
  opts.container_path = a.model;
  opts.layout = r4dx::model::LayoutFromName(a.layout);
  opts.max_ctx = a.max_ctx;
  opts.layer_limit = a.layers;
  opts.vision = ModelOptions::VisionMode::kOff;
  const bool dflash = !a.dflash.empty();
  if (dflash) {
    opts.dflash_container = a.dflash;
    opts.dflash_draft_k = a.dflash_k;
  }
  if (hybrid) opts.pp = 2;  // --pp 2 (ModelOptions::pp): with tp.world 2 this is the hybrid serving mode, TpModel::Load(opts, tp, pp)
  r4dx::model::PpOptions ppo;
  ppo.split = a.pp_split;
  if (a.pp_min_rows > 0) {
    ppo.min_rows = a.pp_min_rows;
    ppo.min_rows_given = true;
  }
  ppo.hybrid = 1;  // on (the kill switch R4DX_HYBRID is not consulted: a soak that silently ran plain --tp 2 would prove nothing)
  ppo.hybrid_ctx = a.hybrid_ctx;
  ppo.hybrid_reserve_gib = a.hybrid_reserve_gib;
  TpOptions tpo;
  tpo.world = 2;
  tpo.mode = real ? TpOptions::Mode::kReal : TpOptions::Mode::kEmulate;
  tpo.devices = a.tp_devices;
  tpo.ar_timeout_ms = a.tp_ar_timeout_ms;
  if (a.tp_submit_layers >= 0) tpo.submit_layers = a.tp_submit_layers;
  if (a.tp_max_inflight >= 0) tpo.max_inflight_units = a.tp_max_inflight;

  log->Line("{\"type\":\"start\",\"model\":" + Str(a.model) + ",\"layout\":" + Str(a.layout) + ",\"tp_mode\":" +
            Str(a.tp_mode) + ",\"minutes\":" + Num(a.minutes) + ",\"iterations\":" + std::to_string(a.iterations) +
            ",\"max_ctx\":" + std::to_string(a.max_ctx) + ",\"seed\":" + std::to_string(a.seed) +
            ",\"submit_layers\":" + std::to_string(tpo.submit_layers) + ",\"max_inflight_units\":" +
            std::to_string(tpo.max_inflight_units) + ",\"ar_timeout_ms\":" + std::to_string(tpo.ar_timeout_ms) +
            ",\"pool_tokens\":" + std::to_string(pool.size()) + ",\"dflash\":" + Str(a.dflash) + ",\"dflash_k\":" +
            std::to_string(dflash ? a.dflash_k : 0) +
            (hybrid ? ",\"pp\":2,\"pp_split\":" + std::to_string(a.pp_split) + ",\"pp_min_rows\":" + std::to_string(a.pp_min_rows) +
                          ",\"hybrid_ctx\":" + std::to_string(a.hybrid_ctx) + ",\"tp2_ref_every\":" + std::to_string(a.tp2_ref_every)
                    : std::string()) +
            "}");

  std::unique_ptr<TpModel> m;
  const auto load0 = Clock::now();
  try {
    m = hybrid ? TpModel::Load(opts, tpo, ppo) : TpModel::Load(opts, tpo);
  } catch (const std::exception& e) {
    return fail("load", e.what(), -1);
  }
  const double load_s = std::chrono::duration<double>(Clock::now() - load0).count();
  std::printf("[soak] loaded in %.1f s (--tp-mode %s, submit %d/%d)\n", load_s, a.tp_mode.c_str(), tpo.submit_layers,
              tpo.max_inflight_units);
  if (hybrid) {
    // A budget that leaves no room for the stages makes TpModel::Load fall back to plain --tp 2 with one log line: not a soak of the hybrid.
    if (!m->HybridEngaged()) return fail("load", "the hybrid mode is not engaged (plain --tp 2): " + m->HybridRefusal(), -1);
    std::printf("[soak] hybrid engaged: split k=%lld, S=%lld tokens, min rows %lld\n", static_cast<long long>(m->HybridSplit()),
                static_cast<long long>(m->HybridStageCtx()), static_cast<long long>(m->HybridMinRows()));
  }
  // The drift baseline: right after Load (warm-up included), before iteration 0 (docs/tp.md 10.5
  // "VRAM at start").
  Snapshot first;
  try {
    first = Snap(*m);
  } catch (const std::exception& e) {
    return fail("first snapshot", e.what(), -1);
  }
  log->Line("{\"type\":\"loaded\",\"t_s\":" + Num(since()) + ",\"load_s\":" + Num(load_s) + "," + CommJson(first) + "}");

  std::mt19937_64 rng(a.seed);
  const std::vector<int32_t> canary_prompt(pool.begin(), pool.begin() + kCanaryPrompt);
  std::vector<int32_t> canary_ref;
  // The hybrid canaries (--pp 2): a 2-turn conversation (1536 + 1100 rows) in greedy rounds and, with --dflash, in DFlash2 greedy rounds.
  const std::vector<std::vector<int32_t>> canary_parts = {
      std::vector<int32_t>(pool.begin(), pool.begin() + kHybridCanaryTurn1),
      std::vector<int32_t>(pool.begin() + kHybridCanaryTurn1, pool.begin() + kHybridCanaryRows)};
  std::vector<int32_t> canary_h_ref, canary_hd_ref;
  // The mode mix: the pre-hybrid draws are the first 3 (or 5 with --dflash) entries, so without --pp the random sequence is unchanged.
  std::vector<Mode> modes = {Mode::kGreedy, Mode::kSampled, Mode::kFullRow};
  if (dflash) {
    modes.push_back(Mode::kDflashGreedy);
    modes.push_back(Mode::kDflashSampled);
  }
  if (hybrid) {
    modes.push_back(Mode::kTurns);
    if (dflash) modes.push_back(Mode::kTurnsDflash);
  }
  const int64_t spec_slack = dflash ? 2 * (a.dflash_k + 1) : 0;
  int64_t iter = 0, canaries = 0, tokens_total = 0;
  bool canary_fail = false;
  int64_t ref_compared = 0, ref_first_diff = 0;  // --tp2-ref-every: iterations compared / whose very first token differed
  double ref_agree_sum = 0;                      // sum of agreeing-prefix / compared-length
  std::vector<std::string> soak_failures;        // criteria other than the canaries (exit code 3)
  try {
    for (;; ++iter) {
      if (a.iterations > 0 ? iter >= a.iterations : since() >= a.minutes * 60.0) break;
      const bool canary = iter % a.canary_every == 0;
      std::uniform_int_distribution<int> plen(a.min_prompt, a.max_prompt), dlen(a.min_decode, a.max_decode),
          mode_d(0, static_cast<int>(modes.size()) - 1);
      std::uniform_int_distribution<size_t> start_d(0, pool.size() - 1);
      const int P = plen(rng), D = dlen(rng);
      const Mode mode = modes[static_cast<size_t>(mode_d(rng))];
      const size_t start = start_d(rng);
      const uint64_t sample_seed = rng();
      std::vector<std::vector<int32_t>> parts(1);
      parts[0].resize(static_cast<size_t>(P));
      for (int i = 0; i < P; ++i) parts[0][static_cast<size_t>(i)] = pool[(start + static_cast<size_t>(i)) % pool.size()];
      if (mode == Mode::kTurns || mode == Mode::kTurnsDflash) {
        // 1-2 more prompts (tails of 1..4096 rows, about three quarters of them above the 1024-row threshold), clipped so the whole
        // conversation (every prompt, every decode and its speculative slack) fits --max-ctx (Parse guarantees room for 3 turns).
        std::uniform_int_distribution<int> turns_d(2, 3), tail_d(1, 4096);
        const int turns = turns_d(rng);
        const int64_t room = a.max_ctx - P - static_cast<int64_t>(turns) * (D + spec_slack) - 16;
        size_t at = start + static_cast<size_t>(P);
        for (int t = 1; t < turns; ++t) {
          const int tail = static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(tail_d(rng), room / (turns - 1))));
          std::vector<int32_t> tail_ids(static_cast<size_t>(tail));
          for (int i = 0; i < tail; ++i) tail_ids[static_cast<size_t>(i)] = pool[(at + static_cast<size_t>(i)) % pool.size()];
          at += static_cast<size_t>(tail);
          parts.push_back(std::move(tail_ids));
        }
      }
      int64_t rows = 0;
      for (const auto& p : parts) rows += static_cast<int64_t>(p.size());

      const Gen g = Generate(*m, parts, D, mode, sample_seed, a.dflash_k);
      const int64_t emitted = g.emitted;  // == D (per turn) except for DFlash2 rounds
      tokens_total += rows + emitted;

      // --tp2-ref-every: the same conversation with the pipelining off (the ordinary TP=2 prefill). Not equal in general; a
      // gross-corruption detector (see the header). DFlash2 streams are cut at round boundaries that depend on the acceptance, so
      // only the first turn is comparable there.
      std::string ref_state = "null";
      if (hybrid && a.tp2_ref_every > 0 && iter % a.tp2_ref_every == 0 && IsGreedyMode(mode)) {
        const int64_t keep_min_rows = m->HybridMinRows();
        m->SetHybridMinRows(int64_t{1} << 40);
        const Gen r = Generate(*m, parts, D, mode, sample_seed, a.dflash_k);
        m->SetHybridMinRows(keep_min_rows);
        tokens_total += rows + r.emitted;
        const size_t cut = IsDflashMode(mode) ? std::min(g.turn_ends.front(), r.turn_ends.front()) : std::min(g.tokens.size(), r.tokens.size());
        const Agree ag = CompareTokens(std::vector<int32_t>(g.tokens.begin(), g.tokens.begin() + static_cast<ptrdiff_t>(cut)),
                                       std::vector<int32_t>(r.tokens.begin(), r.tokens.begin() + static_cast<ptrdiff_t>(cut)));
        ++ref_compared;
        ref_agree_sum += ag.of ? static_cast<double>(ag.prefix) / static_cast<double>(ag.of) : 1.0;
        if (ag.of > 0 && ag.prefix == 0) ++ref_first_diff;
        ref_state = "{\"agree\":" + std::to_string(ag.prefix) + ",\"of\":" + std::to_string(ag.of) + "}";
      }

      std::string canary_state = "null", canary_h_state = "null", canary_hd_state = "null";
      double canary_s = 0;
      if (canary) {
        const auto c0 = Clock::now();
        const Gen c = Generate(*m, canary_prompt, kCanaryDecode, Mode::kGreedy, 0);
        ++canaries;
        tokens_total += kCanaryPrompt + kCanaryDecode;
        if (canary_ref.empty()) {
          canary_ref = c.tokens;
          canary_state = "\"reference\"";
        } else if (c.tokens == canary_ref) {
          canary_state = "\"equal\"";
        } else {
          size_t k = 0;
          while (k < c.tokens.size() && k < canary_ref.size() && c.tokens[k] == canary_ref[k]) ++k;
          canary_state = "\"DIFF at token " + std::to_string(k) + "\"";
          canary_fail = true;
        }
        // The hybrid canaries: each must be PIPELINED twice (cold + warm call) the first time and equal its first run afterwards.
        const auto hybrid_canary = [&](Mode cm, std::vector<int32_t>& ref, const char* name) -> std::string {
          const int64_t before = m->GetHybridStats().pipelined_calls;
          const Gen h = Generate(*m, canary_parts, kHybridCanaryDecode, cm, 0, a.dflash_k);
          const int64_t pipelined = m->GetHybridStats().pipelined_calls - before;
          tokens_total += kHybridCanaryRows + h.emitted;
          if (pipelined < 2) {
            soak_failures.push_back(std::string("the ") + name + " canary (2 prompts of " + std::to_string(kHybridCanaryTurn1) + " and " +
                                    std::to_string(kHybridCanaryTurn2) + " rows) ran " + std::to_string(pipelined) +
                                    " pipelined call(s), not 2: " + m->HybridStatsLine());
          }
          if (ref.empty()) {
            ref = h.tokens;
            return "\"reference\"";
          }
          if (h.tokens == ref) return "\"equal\"";
          size_t k = 0;
          while (k < h.tokens.size() && k < ref.size() && h.tokens[k] == ref[k]) ++k;
          canary_fail = true;
          return "\"DIFF at token " + std::to_string(k) + "\"";
        };
        if (hybrid) {
          canary_h_state = hybrid_canary(Mode::kTurns, canary_h_ref, "hybrid");
          if (dflash) canary_hd_state = hybrid_canary(Mode::kTurnsDflash, canary_hd_ref, "hybrid DFlash2");
        }
        canary_s = std::chrono::duration<double>(Clock::now() - c0).count();
      }
      const Snapshot s = Snap(*m);
      log->Line("{\"type\":\"iter\",\"iter\":" + std::to_string(iter) + ",\"t_s\":" + Num(since()) + ",\"mode\":\"" +
                ModeName(mode) + "\",\"prompt_tokens\":" + std::to_string(hybrid ? rows : P) + ",\"decode_tokens\":" +
                std::to_string(emitted) + ",\"rounds\":" + std::to_string(g.rounds) + ",\"start\":" +
                std::to_string(start) + ",\"prefill_tok_s\":" + Num(static_cast<double>(rows) / g.prefill_s) + ",\"decode_tok_s\":" +
                Num(static_cast<double>(emitted) / g.decode_s) + ",\"canary\":" + canary_state +
                (hybrid ? ",\"turns\":" + std::to_string(parts.size()) + ",\"canary_h\":" + canary_h_state + ",\"canary_hd\":" +
                              canary_hd_state + ",\"tp2_ref\":" + ref_state + ",\"hybrid\":{" + HybridJson(*m, false) + "}"
                        : std::string()) +
                ",\"canary_s\":" + Num(canary_s) + ",\"buffer_drift_mib\":" + Arr(DriftMiB(s, first, true)) +
                ",\"vram_used_drift_mib\":" + Arr(DriftMiB(s, first, false)) + "," + CommJson(s) +
                ",\"fallback_rows\":" + std::to_string(m->SampledFallbackRows()) + "}");
      std::printf("[soak] iter %lld  %.0f s  %-14s prompt %5lld%s  decode %3lld  prefill %.0f tok/s  decode %.1f tok/s%s%s\n",
                  static_cast<long long>(iter), since(), ModeName(mode), static_cast<long long>(rows),
                  parts.size() > 1 ? (" in " + std::to_string(parts.size()) + " turns").c_str() : "", static_cast<long long>(emitted),
                  static_cast<double>(rows) / g.prefill_s, static_cast<double>(emitted) / g.decode_s,
                  canary ? (std::string("  canary ") + canary_state +
                            (hybrid ? " hybrid-canary " + canary_h_state + (dflash ? " " + canary_hd_state : std::string()) : std::string()))
                               .c_str()
                         : "",
                  hybrid ? ("  [" + std::to_string(m->GetHybridStats().pipelined_calls) + " pipelined / " +
                            std::to_string(m->GetHybridStats().tp_prefill_calls) + " TP prefill]")
                               .c_str()
                         : "");
      if (canary_fail || !soak_failures.empty()) break;
    }
  } catch (const std::exception& e) {
    // No Reset(), no retry (docs/tp.md 10.5): the first error ends the soak.
    return fail("TpModel error", e.what(), iter);
  }

  // Pass criteria (docs/tp.md 10.5); the TDR check is soak.ps1's.
  Snapshot last;
  try {
    last = Snap(*m);
  } catch (const std::exception& e) {
    return fail("final snapshot", e.what(), iter);
  }
  const std::vector<double> drift = DriftMiB(last, first, true), used_drift = DriftMiB(last, first, false);
  double worst_drift = 0;
  for (double d : drift) worst_drift = std::max(worst_drift, std::fabs(d));
  uint64_t aborts = 0;
  for (const core::TpCommStats& c : last.comm) aborts += c.aborts;
  const uint64_t allocs = core::g_tp_collective_allocs.load();
  std::vector<std::string> failures;
  if (canary_fail) failures.push_back("canary mismatch");
  if (aborts != 0) failures.push_back(std::to_string(aborts) + " abort(s)");
  if (worst_drift > kMaxVramDriftMiB) failures.push_back("buffer drift " + Num(worst_drift) + " MiB > 64 MiB");
  if (allocs != 0) failures.push_back(std::to_string(allocs) + " device allocation(s) inside collective commands");
  if (iter == 0) failures.push_back("no iteration ran");
  for (const std::string& f : soak_failures) failures.push_back(f);
  std::string hybrid_json;
  if (hybrid) {
    // The hybrid must have pipelined, still be engaged, and (with --tp2-ref-every) agree with the TP=2 prefill's tokens to a gross degree.
    const TpModel::HybridStats hs = m->GetHybridStats();
    if (!m->HybridEngaged()) failures.push_back("the hybrid mode was torn down during the soak (plain --tp 2 from then on): " + m->HybridRefusal());
    if (hs.pipelined_calls == 0) failures.push_back("no pipelined call in the whole soak: " + m->HybridStatsLine());
    const double mean_agree = ref_compared > 0 ? ref_agree_sum / static_cast<double>(ref_compared) : 1.0;
    // Greedy streams that start from different prefill numerics (KL floor ~0.001) diverge at the first flipped token and never
    // re-converge, so a healthy mean agreeing prefix is only ~0.35-0.55 of a 32..512-token run: it cannot be gated at 0.5. The
    // very first token (one argmax of the prefill's logits) flips for ~1 % of iterations when healthy and for most of them when the
    // gather or the stage state is corrupt, so that is the signal; the mean prefix is a floor for outright garbage only.
    if (ref_compared >= 4 && (ref_first_diff * 2 > ref_compared || mean_agree < 0.1)) {
      failures.push_back("the hybrid and the TP=2 prefill agree on a mean " + Num(mean_agree * 100.0) + " % of the compared prefix over " +
                         std::to_string(ref_compared) + " iterations (" + std::to_string(ref_first_diff) + " differ at the very first token; gate: "
                         "more than half differing at the first token, or a mean below 10 %)");
    }
    hybrid_json = ",\"hybrid\":{" + HybridJson(*m, true) + ",\"tp2_ref\":{\"compared\":" + std::to_string(ref_compared) + ",\"first_token_diff\":" +
                  std::to_string(ref_first_diff) + ",\"mean_agree\":" + Num(mean_agree) + "}}";
  }
  const int code = failures.empty() ? 0 : canary_fail ? 2 : 3;
  std::string fl = "[";
  for (size_t i = 0; i < failures.size(); ++i) fl += (i ? "," : "") + Str(failures[i]);
  fl += "]";
  log->Line("{\"type\":\"summary\",\"t_s\":" + Num(since()) + ",\"load_s\":" + Num(load_s) + ",\"iterations\":" +
            std::to_string(iter) + ",\"canaries\":" + std::to_string(canaries) + ",\"tokens\":" +
            std::to_string(tokens_total) + ",\"buffer_drift_mib\":" + Arr(drift) + ",\"vram_used_drift_mib\":" +
            Arr(used_drift) + "," + CommJson(last) + hybrid_json + ",\"exit_code\":" + std::to_string(code) + ",\"failures\":" + fl +
            "}");
  std::printf("[soak] %s: %lld iterations, %lld canaries, %lld tokens in %.0f s; buffer drift %s MiB (device-wide "
              "used %s MiB); %s\n",
              code == 0 ? "PASS" : "FAIL", static_cast<long long>(iter), static_cast<long long>(canaries),
              static_cast<long long>(tokens_total), since(), Arr(drift).c_str(), Arr(used_drift).c_str(),
              m->StatsLine().c_str());
  if (hybrid) std::printf("[soak] [stats] %s\n", m->HybridStatsLine().c_str());
  for (const std::string& f : failures) std::printf("[soak]   %s\n", f.c_str());
  // Tear the group down explicitly and say so, on stdout and as the log's last line: an exit crash
  // is then placed on one side of it (docs/tp.md Appendix B N63 records one inside amdhip64_7.dll
  // after this point), and tools/tp/soak.ps1 requires the "teardown" line for G8 (N64).
  m.reset();
  log->Line("{\"type\":\"teardown\",\"t_s\":" + Num(since()) + ",\"exit_code\":" + std::to_string(code) + "}");
  std::printf("[soak] TpModel torn down; exiting with code %d\n", code);
  return code;
}
