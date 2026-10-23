// tool_q35moe_tp_soak -- the --tp 2 soak of Ornith-1.5-122B (docs/qwen35moe-tp-design.md P5, gate G8'): tests/model/tool_tp_soak.cpp's loop on a
// Qwen35MoeTpModel, without the hybrid and DFlash2 parts.
//
// ONE Qwen35MoeTpModel (real mode: both GPUs) for the whole run. Each iteration: pick a random slice (--min-prompt..--max-prompt tokens) of the token pool
// (every segment of --tokens concatenated, wrapping), Reset(), Prefill, then a random --min-decode..--max-decode decode tokens in a random mode -- greedy
// (DecodeStepGreedy), seeded sampled (DecodeStepSampled; T 0.7, top_k 20, top_p 0.8) or full-row (DecodeStep + host argmax: the vocab gather). Every
// --canary-every-th iteration (and the first) re-runs a fixed canary: the pool's first 512 tokens, then 128 greedy tokens, which must equal the first
// canary run exactly (a corrupted state, a leaked buffer or a wrong merge after unrelated traffic shows up here).
//
// One JSON object per line is appended to --json (start, loaded, one iter line per iteration, error, summary, teardown), each flushed to disk before the
// next iteration starts, so a crash -- a TDR can take the display driver down with it -- still leaves every finished iteration. On ANY error the soak stops
// at once without Reset(). Pass (exit 0): every canary equal, 0 aborts, per-rank buffer drift <= 64 MiB from right after load, and
// core::g_tp_collective_allocs == 0. Exit 2: a canary mismatch; 3: another criterion. The TDR watch is tools/tp/q35moe_soak.ps1's.
//
// Built, never add_test()'d. Usage (HIP_VISIBLE_DEVICES unset, no r4dx-server):
//   tool_q35moe_tp_soak --model E:\models\r4dx\ornith-1.5-122b-trellis-mix-kv.r4dx --tokens E:\models\r4dx\q35moe\kl\tokens_gate.json --minutes 5 --max-ctx 8192
//                       --json build\logs\q35moe_soak.jsonl [--seed 1] [--iterations N] [--canary-every 10] [--min-prompt 16] [--max-prompt 2048]
//                       [--min-decode 32] [--max-decode 512] [--tp-ar-timeout-ms 500] [--tp-submit-layers N] [--tp-max-inflight K] [--need-gib 24]
#include <hip/hip_runtime.h>
#include <io.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "model.h"
#include "qwen35moe_tp_model.h"
#include "r4dx/core/tp_alloc_guard.hpp"
#include "r4dx/kernels/sampler.hpp"

namespace {

using Clock = std::chrono::steady_clock;
using r4dx::model::ModelOptions;
using r4dx::model::Qwen35MoeTpModel;
using r4dx::model::TpOptions;
namespace core = r4dx::core;
namespace kernels = r4dx::kernels;

constexpr int kCanaryPrompt = 512;
constexpr int kCanaryDecode = 128;
constexpr double kMaxVramDriftMiB = 64.0;

struct Args {
  std::string model = "E:\\models\\r4dx\\ornith-1.5-122b-trellis-mix-kv.r4dx";
  std::string tokens = "E:\\models\\r4dx\\q35moe\\kl\\tokens_gate.json";
  std::string json;
  double minutes = 5.0;
  int64_t iterations = 0;
  int64_t max_ctx = 8192;
  uint64_t seed = 1;
  int canary_every = 10;
  int min_prompt = 16, max_prompt = 2048, min_decode = 32, max_decode = 512;
  int tp_ar_timeout_ms = 500;
  int tp_submit_layers = -1, tp_max_inflight = -1;
  double need_gib = 24.0;
  std::string dflash;  // --dflash <drafter.r4dx>: DFlash2 greedy rounds join the mode mix (the canary stays plain greedy)
  int64_t dflash_k = 11;
};

[[noreturn]] void Usage(const std::string& why) {
  std::fprintf(stderr, "tool_q35moe_tp_soak: %s\nusage: tool_q35moe_tp_soak [--model c.r4dx] [--tokens t.json] --json log.jsonl [--minutes 5] [--iterations N] [--max-ctx 8192] "
                       "[--seed 1] [--canary-every 10] [--min-prompt 16] [--max-prompt 2048] [--min-decode 32] [--max-decode 512] [--tp-ar-timeout-ms N] "
                       "[--tp-submit-layers N] [--tp-max-inflight K] [--need-gib X]\n", why.c_str());
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
      else if (s == "--tokens") a.tokens = next();
      else if (s == "--json") a.json = next();
      else if (s == "--minutes") a.minutes = std::stod(next());
      else if (s == "--iterations") a.iterations = std::stoll(next());
      else if (s == "--max-ctx") a.max_ctx = std::stoll(next());
      else if (s == "--seed") a.seed = std::stoull(next());
      else if (s == "--canary-every") a.canary_every = std::stoi(next());
      else if (s == "--min-prompt") a.min_prompt = std::stoi(next());
      else if (s == "--max-prompt") a.max_prompt = std::stoi(next());
      else if (s == "--min-decode") a.min_decode = std::stoi(next());
      else if (s == "--max-decode") a.max_decode = std::stoi(next());
      else if (s == "--tp-ar-timeout-ms") a.tp_ar_timeout_ms = std::stoi(next());
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
  if (a.json.empty()) Usage("--json <log.jsonl> is required");
  if (!(a.minutes > 0) && a.iterations <= 0) Usage("--minutes must be > 0 (or give --iterations)");
  if (a.canary_every < 1) Usage("--canary-every must be >= 1");
  if (a.min_prompt < 1 || a.max_prompt < a.min_prompt || a.min_decode < 1 || a.max_decode < a.min_decode) Usage("need 1 <= --min-prompt <= --max-prompt and 1 <= --min-decode <= --max-decode");
  const int64_t spec_slack = a.dflash.empty() ? 0 : 2 * (a.dflash_k + 1);
  if (a.max_prompt + a.max_decode + spec_slack > a.max_ctx || kCanaryPrompt + kCanaryDecode > a.max_ctx) Usage("--max-prompt + --max-decode and the 640-position canary must fit in --max-ctx");
  if (a.tp_submit_layers < -1 || a.tp_submit_layers > 64 || a.tp_max_inflight < -1 || a.tp_max_inflight > 64) Usage("--tp-submit-layers and --tp-max-inflight must be in [0, 64]");
  return a;
}

class JsonLog {
 public:
  explicit JsonLog(const std::string& path) {
    if (fopen_s(&f_, path.c_str(), "ab") != 0) f_ = nullptr;
    if (f_ == nullptr) throw std::runtime_error("cannot open " + path + " for appending");
  }
  ~JsonLog() {
    if (f_ != nullptr) std::fclose(f_);
  }
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

// Returns "" when every device has `need_gib` free, else the message to print (a helper thread: the main thread is the facade and makes no HIP call).
std::string Preflight(double need_gib) {
  std::string msg;
  std::thread t([&] {
    int n = 0;
    if (hipGetDeviceCount(&n) != hipSuccess || n < 2) {
      msg = "--tp 2 real needs two visible HIP devices";
      return;
    }
    for (int d : {n - 1, n - 2}) {
      size_t free_b = 0, total_b = 0;
      if (hipSetDevice(d) != hipSuccess || hipMemGetInfo(&free_b, &total_b) != hipSuccess) {
        msg = "cannot query HIP device " + std::to_string(d);
        return;
      }
      const double free_gib = static_cast<double>(free_b) / (1024.0 * 1024.0 * 1024.0);
      if (free_gib < need_gib) {
        char b[256];
        std::snprintf(b, sizeof b, "need %.1f GiB free on HIP device %d, have %.2f GiB -- is another job or the server running?", need_gib, d, free_gib);
        msg = b;
        return;
      }
    }
    (void)hipGetLastError();
  });
  t.join();
  return msg;
}

enum class Mode { kGreedy, kSampled, kFullRow, kDflashGreedy };
const char* ModeName(Mode m) {
  return m == Mode::kGreedy ? "greedy" : m == Mode::kSampled ? "sampled" : m == Mode::kFullRow ? "full_row" : "dflash_greedy";
}

struct Gen {
  std::vector<int32_t> tokens;
  double prefill_s = 0, decode_s = 0;
};

Gen Generate(Qwen35MoeTpModel& m, const std::vector<int32_t>& prompt, int decode, Mode mode, uint64_t sample_seed, int64_t dflash_k = 0) {
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
  int32_t next = mode == Mode::kSampled ? kernels::Sample(logits.data(), V, sp, rng) : kernels::Argmax(logits.data(), V);
  const auto t1 = Clock::now();
  if (mode == Mode::kDflashGreedy) {
    // whole rounds until at least `decode` tokens past the first one were emitted; every round's last token is the next round's (uncommitted) anchor
    g.tokens.push_back(next);
    while (static_cast<int>(g.tokens.size()) <= decode) {
      const std::vector<int32_t> r = m.DecodeStepDflashGreedy(next, dflash_k, 0.8f, 0);
      g.tokens.insert(g.tokens.end(), r.begin(), r.end());
      next = r.back();
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
  std::vector<double> buffers_gib, vram_used_gib;
  std::vector<core::TpCommStats> comm;
  std::vector<r4dx::model::tp::SubmitBounder::Stats> submit;
};
Snapshot Snap(Qwen35MoeTpModel& m) {
  Snapshot s;
  for (const r4dx::model::VramReport& v : m.Vram()) {
    s.buffers_gib.push_back(v.buffers_gib);
    s.vram_used_gib.push_back(v.used_gib);
  }
  s.comm = m.CommStats();
  s.submit = m.SubmitStats();
  return s;
}
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
  return "\"buffers_gib\":" + Arr(s.buffers_gib) + ",\"vram_used_gib\":" + Arr(s.vram_used_gib) + ",\"ar_calls_ch0\":" + Arr(ch0) + ",\"ar_calls_ch1\":" + Arr(ch1) +
         ",\"host_exchanges\":" + Arr(hx) + ",\"max_exchange_wait_us\":" + Arr(wait) + ",\"aborts\":" + Arr(aborts) + ",\"submit_units\":" + Arr(units) +
         ",\"cap_waits\":" + Arr(cap_waits) + ",\"max_cap_wait_us\":" + Arr(cap_wait_max) + ",\"collective_allocs\":" + std::to_string(core::g_tp_collective_allocs.load());
}

}  // namespace

int main(int argc, char** argv) {
  const Args a = Parse(argc, argv);
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  {
    std::ifstream probe(a.model, std::ios::binary);
    if (!probe) {
      std::fprintf(stderr, "tool_q35moe_tp_soak: container not found: %s\n", a.model.c_str());
      return 1;
    }
  }
  const std::string pre = Preflight(a.need_gib);
  if (!pre.empty()) {
    std::fprintf(stderr, "tool_q35moe_tp_soak: %s\n", pre.c_str());
    return 1;
  }
  std::unique_ptr<JsonLog> log;
  try {
    log = std::make_unique<JsonLog>(a.json);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "tool_q35moe_tp_soak: %s\n", e.what());
    return 1;
  }
  const auto wall0 = Clock::now();
  const auto since = [&] { return std::chrono::duration<double>(Clock::now() - wall0).count(); };
  const auto fail = [&](const char* what, const std::string& why, int64_t iter) {
    std::fprintf(stderr, "tool_q35moe_tp_soak: %s at iteration %lld: %s\n", what, static_cast<long long>(iter), why.c_str());
    log->Line("{\"type\":\"error\",\"t_s\":" + Num(since()) + ",\"iter\":" + std::to_string(iter) + ",\"what\":" + Str(what) + ",\"error\":" + Str(why) + "}");
    return 1;
  };

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
  if (pool.size() < static_cast<size_t>(std::max(kCanaryPrompt, a.max_prompt))) return fail("tokens", "the token pool has fewer tokens than --max-prompt / the canary", -1);

  ModelOptions opts;
  opts.container_path = a.model;
  opts.layout = r4dx::model::Layout::kTrellis;
  opts.max_ctx = a.max_ctx;
  opts.vision = ModelOptions::VisionMode::kOff;
  if (!a.dflash.empty()) {
    opts.dflash_container = a.dflash;
    opts.dflash_draft_k = a.dflash_k;
  }
  TpOptions tpo;
  tpo.world = 2;
  tpo.mode = TpOptions::Mode::kReal;
  tpo.ar_timeout_ms = a.tp_ar_timeout_ms;
  if (a.tp_submit_layers >= 0) tpo.submit_layers = a.tp_submit_layers;
  if (a.tp_max_inflight >= 0) tpo.max_inflight_units = a.tp_max_inflight;

  log->Line("{\"type\":\"start\",\"model\":" + Str(a.model) + ",\"minutes\":" + Num(a.minutes) + ",\"iterations\":" + std::to_string(a.iterations) + ",\"max_ctx\":" +
            std::to_string(a.max_ctx) + ",\"seed\":" + std::to_string(a.seed) + ",\"submit_layers\":" + std::to_string(tpo.submit_layers) + ",\"max_inflight_units\":" +
            std::to_string(tpo.max_inflight_units) + ",\"ar_timeout_ms\":" + std::to_string(tpo.ar_timeout_ms) + ",\"pool_tokens\":" + std::to_string(pool.size()) + "}");

  std::unique_ptr<Qwen35MoeTpModel> m;
  const auto load0 = Clock::now();
  try {
    m = Qwen35MoeTpModel::Load(opts, tpo);
  } catch (const std::exception& e) {
    return fail("load", e.what(), -1);
  }
  const double load_s = std::chrono::duration<double>(Clock::now() - load0).count();
  std::printf("[soak] loaded in %.1f s (--tp-mode real, submit %d/%d)\n", load_s, tpo.submit_layers, tpo.max_inflight_units);
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
  std::vector<Mode> modes = {Mode::kGreedy, Mode::kSampled, Mode::kFullRow};
  if (!a.dflash.empty()) modes.push_back(Mode::kDflashGreedy);
  int64_t iter = 0, canaries = 0, tokens_total = 0;
  bool canary_fail = false;
  try {
    for (;; ++iter) {
      if (a.iterations > 0 ? iter >= a.iterations : since() >= a.minutes * 60.0) break;
      const bool canary = iter % a.canary_every == 0;
      std::uniform_int_distribution<int> plen(a.min_prompt, a.max_prompt), dlen(a.min_decode, a.max_decode), mode_d(0, static_cast<int>(modes.size()) - 1);
      std::uniform_int_distribution<size_t> start_d(0, pool.size() - 1);
      const int P = plen(rng), D = dlen(rng);
      const Mode mode = modes[static_cast<size_t>(mode_d(rng))];
      const size_t start = start_d(rng);
      const uint64_t sample_seed = rng();
      std::vector<int32_t> prompt(static_cast<size_t>(P));
      for (int i = 0; i < P; ++i) prompt[static_cast<size_t>(i)] = pool[(start + static_cast<size_t>(i)) % pool.size()];

      const Gen g = Generate(*m, prompt, D, mode, sample_seed, a.dflash_k);
      tokens_total += P + static_cast<int64_t>(g.tokens.size());
      std::string canary_state = "null";
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
        canary_s = std::chrono::duration<double>(Clock::now() - c0).count();
      }
      const Snapshot s = Snap(*m);
      log->Line("{\"type\":\"iter\",\"iter\":" + std::to_string(iter) + ",\"t_s\":" + Num(since()) + ",\"mode\":\"" + ModeName(mode) + "\",\"prompt_tokens\":" + std::to_string(P) +
                ",\"decode_tokens\":" + std::to_string(g.tokens.size()) + ",\"start\":" + std::to_string(start) + ",\"prefill_tok_s\":" + Num(static_cast<double>(P) / g.prefill_s) +
                ",\"decode_tok_s\":" + Num(static_cast<double>(g.tokens.size()) / g.decode_s) + ",\"canary\":" + canary_state + ",\"canary_s\":" + Num(canary_s) + ",\"buffer_drift_mib\":" +
                Arr(DriftMiB(s, first, true)) + ",\"vram_used_drift_mib\":" + Arr(DriftMiB(s, first, false)) + "," + CommJson(s) + ",\"fallback_rows\":" +
                std::to_string(m->SampledFallbackRows()) + "}");
      std::printf("[soak] iter %lld  %.0f s  %-13s prompt %5d  decode %3d  prefill %.0f tok/s  decode %.1f tok/s%s\n", static_cast<long long>(iter), since(), ModeName(mode), P,
                  static_cast<int>(g.tokens.size()), static_cast<double>(P) / g.prefill_s, static_cast<double>(g.tokens.size()) / g.decode_s, canary ? (std::string("  canary ") + canary_state).c_str() : "");
      if (canary_fail) break;
    }
  } catch (const std::exception& e) {
    return fail("Qwen35MoeTpModel error", e.what(), iter);  // no Reset(), no retry
  }

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
  log->Line("{\"type\":\"summary\",\"t_s\":" + Num(since()) + ",\"load_s\":" + Num(load_s) + ",\"iterations\":" + std::to_string(iter) + ",\"canaries\":" + std::to_string(canaries) +
            ",\"tokens\":" + std::to_string(tokens_total) + ",\"buffer_drift_mib\":" + Arr(drift) + ",\"vram_used_drift_mib\":" + Arr(used_drift) + "," + CommJson(last) +
            ",\"exit_code\":" + std::to_string(code) + ",\"failures\":" + fl + "}");
  std::printf("[soak] %s: %lld iterations, %lld canaries, %lld tokens in %.0f s; buffer drift %s MiB (device-wide used %s MiB); %s\n", code == 0 ? "PASS" : "FAIL",
              static_cast<long long>(iter), static_cast<long long>(canaries), static_cast<long long>(tokens_total), since(), Arr(drift).c_str(), Arr(used_drift).c_str(),
              m->StatsLine().c_str());
  for (const std::string& f : failures) std::printf("[soak]   %s\n", f.c_str());
  m.reset();
  log->Line("{\"type\":\"teardown\",\"t_s\":" + Num(since()) + ",\"exit_code\":" + std::to_string(code) + "}");
  std::printf("[soak] Qwen35MoeTpModel torn down; exiting with code %d\n", code);
  return code;
}
