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
// Built, never add_test()'d. Usage (HIP_VISIBLE_DEVICES unset, production server stopped):
//   tool_tp_soak --model D:\models\r4dx\qwen38-27b-v6.r4dx --layout w4a16 --minutes 60
//                --max-ctx 8192 --json build\logs\tp_soak.jsonl [--seed 1] [--iterations N]
//                [--tokens tools\reference\kl_corpus\tokens.json] [--canary-every 10]
//                [--min-prompt 16] [--max-prompt 2048] [--min-decode 32] [--max-decode 512]
//                [--tp-mode real|emulate] [--tp-devices a,b] [--tp-ar-timeout-ms 500]
//                [--tp-submit-layers N] [--tp-max-inflight K] [--need-gib 14] [--layers N]
//                [--dflash <drafter.r4dx> [--dflash-k 7]]
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
constexpr double kMaxVramDriftMiB = 64.0;

struct Args {
  std::string model = r4dx_test::ProductionTargetPath();
  std::string layout = "w4a16";
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
};

[[noreturn]] void Usage(const std::string& why) {
  std::fprintf(stderr,
               "tool_tp_soak: %s\nusage: tool_tp_soak [--model <c.r4dx>] [--layout w4a16] [--minutes 60] "
               "[--iterations N] [--max-ctx 8192] [--seed 1] [--json <log.jsonl>] [--tokens <tokens.json>] "
               "[--canary-every 10] [--min-prompt 16] [--max-prompt 2048] [--min-decode 32] [--max-decode 512] "
               "[--tp-mode real|emulate] [--tp-devices a,b] [--tp-ar-timeout-ms N] [--tp-submit-layers N] "
               "[--tp-max-inflight K] [--need-gib X] [--layers N] [--dflash <drafter.r4dx> [--dflash-k 7]]\n",
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
      else if (s == "--max-ctx") a.max_ctx = std::stoll(next());
      else if (s == "--layers") a.layers = std::stoll(next());
      else if (s == "--seed") a.seed = std::stoull(next());
      else if (s == "--canary-every") a.canary_every = std::stoi(next());
      else if (s == "--min-prompt") a.min_prompt = std::stoi(next());
      else if (s == "--max-prompt") a.max_prompt = std::stoi(next());
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
      else if (s == "--need-gib") a.need_gib = std::stod(next());
      else if (s == "--dflash") a.dflash = next();
      else if (s == "--dflash-k") a.dflash_k = std::stoll(next());
      else Usage("unknown argument " + s);
    } catch (const std::exception&) {
      Usage("bad value for " + s);
    }
  }
  if (a.json.empty()) Usage("--json <log.jsonl> is required (the per-iteration log)");
  if (a.tp_mode != "real" && a.tp_mode != "emulate") Usage("--tp-mode must be real or emulate");
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
// --dflash (drawn from [0, 4]).
enum class Mode { kGreedy, kSampled, kFullRow, kDflashGreedy, kDflashSampled };
const char* ModeName(Mode m) {
  switch (m) {
    case Mode::kGreedy: return "greedy";
    case Mode::kSampled: return "sampled";
    case Mode::kFullRow: return "full_row";
    case Mode::kDflashGreedy: return "dflash_greedy";
    case Mode::kDflashSampled: return "dflash_sampled";
  }
  return "?";
}

struct Gen {
  std::vector<int32_t> tokens;
  double prefill_s = 0, decode_s = 0;
  int64_t rounds = 0;  // DFlash2 modes: speculative rounds run
};

Gen Generate(TpModel& m, const std::vector<int32_t>& prompt, int decode, Mode mode, uint64_t sample_seed,
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
  const auto t0 = Clock::now();
  const std::vector<float> logits = m.Prefill(prompt);
  const bool sampled = mode == Mode::kSampled || mode == Mode::kDflashSampled;
  int32_t next = sampled ? kernels::Sample(logits.data(), V, sp, rng) : kernels::Argmax(logits.data(), V);
  const auto t1 = Clock::now();
  if (mode == Mode::kDflashGreedy || mode == Mode::kDflashSampled) {
    // Whole rounds until at least `decode` tokens past the first one were emitted; every round's
    // last token is the next round's (not yet committed) anchor, like `next` above.
    g.tokens.push_back(next);
    while (static_cast<int>(g.tokens.size()) <= decode) {
      const std::vector<int32_t> r = mode == Mode::kDflashGreedy
                                         ? m.DecodeStepDflashGreedy(next, dflash_k, 0.0f, 0)
                                         : m.DecodeStepDflashSampled(next, dflash_k, 0.0f, 0, sp, rng);
      g.tokens.insert(g.tokens.end(), r.begin(), r.end());
      next = r.back();
      ++g.rounds;
    }
    g.prefill_s = std::chrono::duration<double>(t1 - t0).count();
    g.decode_s = std::chrono::duration<double>(Clock::now() - t1).count();
    return g;
  }
  for (int i = 0; i < decode; ++i) {
    g.tokens.push_back(next);
    if (mode == Mode::kGreedy) {
      next = m.DecodeStepGreedy(next);
    } else if (mode == Mode::kSampled) {
      next = m.DecodeStepSampled(next, sp, rng);
    } else {
      const std::vector<float> row = m.DecodeStep(next);
      next = kernels::Argmax(row.data(), V);
    }
  }
  g.tokens.push_back(next);
  g.prefill_s = std::chrono::duration<double>(t1 - t0).count();
  g.decode_s = std::chrono::duration<double>(Clock::now() - t1).count();
  return g;
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
  if (pool.size() < static_cast<size_t>(std::max(kCanaryPrompt, a.max_prompt))) {
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
            std::to_string(dflash ? a.dflash_k : 0) + "}");

  std::unique_ptr<TpModel> m;
  const auto load0 = Clock::now();
  try {
    m = TpModel::Load(opts, tpo);
  } catch (const std::exception& e) {
    return fail("load", e.what(), -1);
  }
  const double load_s = std::chrono::duration<double>(Clock::now() - load0).count();
  std::printf("[soak] loaded in %.1f s (--tp-mode %s, submit %d/%d)\n", load_s, a.tp_mode.c_str(), tpo.submit_layers,
              tpo.max_inflight_units);
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
  int64_t iter = 0, canaries = 0, tokens_total = 0;
  bool canary_fail = false;
  try {
    for (;; ++iter) {
      if (a.iterations > 0 ? iter >= a.iterations : since() >= a.minutes * 60.0) break;
      const bool canary = iter % a.canary_every == 0;
      std::uniform_int_distribution<int> plen(a.min_prompt, a.max_prompt), dlen(a.min_decode, a.max_decode),
          mode_d(0, dflash ? 4 : 2);
      std::uniform_int_distribution<size_t> start_d(0, pool.size() - 1);
      const int P = plen(rng), D = dlen(rng);
      const Mode mode = static_cast<Mode>(mode_d(rng));
      const size_t start = start_d(rng);
      const uint64_t sample_seed = rng();
      std::vector<int32_t> prompt(static_cast<size_t>(P));
      for (int i = 0; i < P; ++i) prompt[static_cast<size_t>(i)] = pool[(start + static_cast<size_t>(i)) % pool.size()];

      const Gen g = Generate(*m, prompt, D, mode, sample_seed, a.dflash_k);
      const int emitted = static_cast<int>(g.tokens.size()) - 1;  // == D except for DFlash2 rounds
      tokens_total += P + emitted;
      std::string canary_state = "null";
      double canary_s = 0;
      if (canary) {
        const auto c0 = Clock::now();
        const Gen c = Generate(*m, canary_prompt, kCanaryDecode, Mode::kGreedy, 0);
        canary_s = std::chrono::duration<double>(Clock::now() - c0).count();
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
      }
      const Snapshot s = Snap(*m);
      log->Line("{\"type\":\"iter\",\"iter\":" + std::to_string(iter) + ",\"t_s\":" + Num(since()) + ",\"mode\":\"" +
                ModeName(mode) + "\",\"prompt_tokens\":" + std::to_string(P) + ",\"decode_tokens\":" +
                std::to_string(emitted) + ",\"rounds\":" + std::to_string(g.rounds) + ",\"start\":" +
                std::to_string(start) + ",\"prefill_tok_s\":" + Num(P / g.prefill_s) + ",\"decode_tok_s\":" +
                Num(emitted / g.decode_s) + ",\"canary\":" + canary_state +
                ",\"canary_s\":" + Num(canary_s) + ",\"buffer_drift_mib\":" + Arr(DriftMiB(s, first, true)) +
                ",\"vram_used_drift_mib\":" + Arr(DriftMiB(s, first, false)) + "," + CommJson(s) +
                ",\"fallback_rows\":" + std::to_string(m->SampledFallbackRows()) + "}");
      std::printf("[soak] iter %lld  %.0f s  %-8s prompt %4d  decode %3d  prefill %.0f tok/s  decode %.1f tok/s%s\n",
                  static_cast<long long>(iter), since(), ModeName(mode), P, emitted, P / g.prefill_s,
                  emitted / g.decode_s, canary ? (std::string("  canary ") + canary_state).c_str() : "");
      if (canary_fail) break;
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
  const int code = failures.empty() ? 0 : canary_fail ? 2 : 3;
  std::string fl = "[";
  for (size_t i = 0; i < failures.size(); ++i) fl += (i ? "," : "") + Str(failures[i]);
  fl += "]";
  log->Line("{\"type\":\"summary\",\"t_s\":" + Num(since()) + ",\"load_s\":" + Num(load_s) + ",\"iterations\":" +
            std::to_string(iter) + ",\"canaries\":" + std::to_string(canaries) + ",\"tokens\":" +
            std::to_string(tokens_total) + ",\"buffer_drift_mib\":" + Arr(drift) + ",\"vram_used_drift_mib\":" +
            Arr(used_drift) + "," + CommJson(last) + ",\"exit_code\":" + std::to_string(code) + ",\"failures\":" + fl +
            "}");
  std::printf("[soak] %s: %lld iterations, %lld canaries, %lld tokens in %.0f s; buffer drift %s MiB (device-wide "
              "used %s MiB); %s\n",
              code == 0 ? "PASS" : "FAIL", static_cast<long long>(iter), static_cast<long long>(canaries),
              static_cast<long long>(tokens_total), since(), Arr(drift).c_str(), Arr(used_drift).c_str(),
              m->StatsLine().c_str());
  for (const std::string& f : failures) std::printf("[soak]   %s\n", f.c_str());
  // Tear the group down explicitly and say so, on stdout and as the log's last line: an exit crash
  // is then placed on one side of it (docs/tp.md Appendix B N63 records one inside amdhip64_7.dll
  // after this point), and tools/tp/soak.ps1 requires the "teardown" line for G8 (N64).
  m.reset();
  log->Line("{\"type\":\"teardown\",\"t_s\":" + Num(since()) + ",\"exit_code\":" + std::to_string(code) + "}");
  std::printf("[soak] TpModel torn down; exiting with code %d\n", code);
  return code;
}
